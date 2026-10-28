"""
GraspCVAE baseline for DexGYS (language-guided dexterous grasping).

CVAE structure follows GraspTTA (Jiang et al., ICCV 2021):
    network/CVAE.py, network/affordanceNet_obman_mano_vertex.py
Encoders / losses follow the DexGYS reproduction protocol (paper A.3.3):
    - object: PointNet++ (same as IDGC), language: CLIP ViT-L/14 (+adapter)
    - condition c = [f_obj ; f_lan], decoder input = [z ; c]
    - task losses = Eq.5 (para, hand chamfer, obj penetration, self penetration, cmap), plus KL

Shapes (default config):
    hand pc (B, 1024, 3) -> PointNet++ -> (B, 512, 16) -> max-pool -> f_hand (B, 512)
    obj pc  (B, 4096, 3) -> PointNet++ -> (B, 512, 64) -> max-pool -> f_obj  (B, 512)
    text -> CLIP + adapter -> EOS token                              -> f_lan  (B, 512)
    encoder: [f_hand ; c] -> MLP -> mu, log_var (B, 64)
    decoder: [z ; c]      -> MLP -> normalized pose (B, 28) = [trans(3), qpos(22), euler(3)]
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .backbone import build_backbone
from .loss import GraspLossPose


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes: List[int], latent_size: int, decoder_layer_sizes: List[int],
                 condition_size: int):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size, condition_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, condition_size)

    def forward(self, x, c):
        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, c):
        z = torch.randn([c.size(0), self.latent_size], device=c.device)
        recon_x = self.decoder(z, c)
        return recon_x, z


class Encoder(nn.Module):

    def __init__(self, layer_sizes: List[int], latent_size: int, condition_size: int):
        super().__init__()
        # copy: GraspTTA mutates layer_sizes[0] in place, which would corrupt the config list
        layer_sizes = list(layer_sizes)
        layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes: List[int], latent_size: int, condition_size: int):
        super().__init__()
        input_size = latent_size + condition_size

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x


class GraspCVAE(nn.Module):
    def __init__(self, cfg) -> None:
        super(GraspCVAE, self).__init__()

        self.encoder_input = cfg.encoder_input  # "hand_pc" | "pose"
        self.d_x = cfg.d_x

        self.obj_encoder = build_backbone(cfg.backbone)
        # named `language_encoder` so that train.py's frozen_clip branch can find it
        self.language_encoder = build_backbone(cfg.language_encoder)
        if self.encoder_input == "hand_pc":
            self.hand_encoder = build_backbone(cfg.hand_backbone)
            x_dim = cfg.hand_backbone.layer3.mlp_list[-1]
        elif self.encoder_input == "pose":
            x_dim = self.d_x
        else:
            raise ValueError(f"Unsupported encoder_input: {self.encoder_input}")

        condition_size = cfg.backbone.layer3.mlp_list[-1] + cfg.language_encoder.dim_out
        self.cvae = VAE(
            encoder_layer_sizes=[x_dim] + list(cfg.cvae.encoder_hidden_sizes),
            latent_size=cfg.cvae.latent_size,
            decoder_layer_sizes=list(cfg.cvae.decoder_hidden_sizes) + [self.d_x],
            condition_size=condition_size,
        )

        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights

    @property
    def device(self):
        return next(self.parameters()).device

    def condition_obj(self, data: Dict) -> torch.Tensor:
        obj_pc = data["obj_pc"].to(torch.float32)
        _, obj_feat = self.obj_encoder(obj_pc)  # (B, C, K)
        return obj_feat.max(dim=-1)[0]  # (B, C)

    def condition_language(self, data: Dict) -> torch.Tensor:
        txt_feat, tokens = self.language_encoder(data["guidance"], data["clip_data"] if "clip_data" in data else None)
        # EOS token (highest token id) carries the sentence embedding in CLIP
        return txt_feat[torch.arange(txt_feat.shape[0]), tokens.argmax(dim=-1)].float()  # (B, C)

    def condition(self, data: Dict) -> torch.Tensor:
        return torch.cat([self.condition_obj(data), self.condition_language(data)], dim=-1)

    def encode_hand(self, data: Dict) -> torch.Tensor:
        """CVAE encoder input x built from the ground-truth grasp."""
        if self.encoder_input == "pose":
            return data["norm_pose"]
        with torch.no_grad():
            hand_pc = self.criterion.hand_model(data["hand_model_pose"], with_surface_points=True)["surface_points"]
        _, hand_feat = self.hand_encoder(hand_pc.contiguous())  # (B, C, K)
        return hand_feat.max(dim=-1)[0]  # (B, C)

    def forward(self, data: Dict):
        if self.training:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def _weighted_sum(self, loss_dict: Dict) -> torch.Tensor:
        loss = 0
        for k, v in loss_dict.items():
            if k in self.loss_weights:
                loss += v * self.loss_weights[k]
        return loss

    def forward_train(self, data: Dict):
        cond = self.condition(data)
        x = self.encode_hand(data)
        recon, means, log_var, _ = self.cvae(x, cond)

        pred_dict = {
            "pred_pose_norm": recon,
            "means": means,
            "log_var": log_var,
        }
        loss_dict = self.criterion(pred_dict, data)
        return self._weighted_sum(loss_dict), loss_dict, None

    @torch.no_grad()
    def sample(self, data: Dict, k: int = 1) -> torch.Tensor:
        """Sample poses from the prior z ~ N(0, I). Returns normalized poses <B, k, d_x>."""
        cond = self.condition(data)
        ksamples = [self.cvae.inference(cond)[0] for _ in range(k)]
        return torch.stack(ksamples, dim=1)

    def _pred_dict(self, pred: torch.Tensor) -> Dict:
        # same layout as DDPM.forward_test so GraspLossPose / test.py work unchanged
        return {
            "pred_pose_norm": pred,
            "qpos_norm": pred[..., 3:25],
            "translation_norm": pred[..., :3],
            "rotation": pred[..., 25:],
            "pred_noise": torch.tensor([1.0]),
            "noise": torch.tensor([1.0]),
        }

    def forward_test(self, data: Dict):
        pred = self.sample(data)[:, 0]
        loss_dict, predictions = self.criterion(self._pred_dict(pred), data)
        return self._weighted_sum(loss_dict), loss_dict, predictions

    def forward_infer(self, data: Dict, k: int = 4):
        pred = self.sample(data, k=k)
        preds_hand, targets_hand = self.criterion.infer_norm_process_dict(self._pred_dict(pred), data)
        return preds_hand, targets_hand
