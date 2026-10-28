"""
ContactNet of GraspTTA (Jiang et al., ICCV 2021) for the Shadow hand on DexGYSNet.

Architecture follows the released code, GraspTTA network/cmapnet_objhand.py (pointnet_reg) and
network/pointnet.py (PointNetEncoder, global_feat=False, no STN), with the point counts adapted:
    object: 3000 points sampled on the mesh surface (as GraspTTA resample_obj_xyz)
    hand:   1024 Shadow hand surface points (HandModel) instead of the 778 MANO vertices
    convfuse: Conv1d(n_obj + n_hand -> n_obj) mixes along the point axis, as the original 3778 -> 3000.

Training (GraspTTA paper Sec 3.2): inputs are ground-truth hand + object, target is the pseudo contact
map of the object computed from the ground-truth hand, loss = L2.
The target uses all hand points, the same definition GraspTTA's TTA uses for the hand contact map
(gen_diverse_grasp_ho3d.py: get_pseudo_cmap(get_NN(obj, hand_verts))), so the TTA consistency
loss compares like with like.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from .utils.hand_model import HandModel


def get_pseudo_cmap(nn_dists: torch.Tensor) -> torch.Tensor:
    """GraspTTA utils.get_pseudo_cmap: squared distance (m^2) -> contact score in [0, 1], 0~3cm -> 1~0."""
    nn_dists = 100.0 * torch.sqrt(nn_dists)  # center-meter
    return 1.0 - 2 * (torch.sigmoid(nn_dists * 2) - 0.5)


def object_contact_map(obj_xyz: torch.Tensor, hand_xyz: torch.Tensor) -> torch.Tensor:
    """obj_xyz (B, N, 3), hand_xyz (B, M, 3) -> pseudo contact map (B, N) from NN squared distances."""
    nn_dists = knn_points(obj_xyz, hand_xyz, K=1).dists[..., 0]
    return get_pseudo_cmap(nn_dists)


class PointNetEncoder(nn.Module):
    """GraspTTA network/pointnet.py PointNetEncoder (STN disabled there as well)."""

    def __init__(self, global_feat=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # global feature: [B, 1024]
        if self.global_feat:
            return x
        x = x.view(-1, 1024, 1).repeat(1, 1, N)  # [B, 1024, N]
        return torch.cat([x, pointfeat], 1)  # [B, 1088, N]


class PointNetReg(nn.Module):
    """GraspTTA network/cmapnet_objhand.py pointnet_reg (with_rgb=False, num_class=1)."""

    def __init__(self, n_obj_points: int, n_hand_points: int):
        super(PointNetReg, self).__init__()
        self.feat_o = PointNetEncoder(global_feat=False, channel=3)
        self.feat_h = PointNetEncoder(global_feat=False, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.convfuse = torch.nn.Conv1d(n_obj_points + n_hand_points, n_obj_points, 1)
        self.bnfuse = nn.BatchNorm1d(n_obj_points)

    def forward(self, x, hand):
        """
        :param x: obj pc [B, 3, N]
        :param hand: hand pc [B, 3, M]
        :return: regressed cmap [B, N]
        """
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat_o(x)  # [B, 1088, N]
        hand = self.feat_h(hand)  # [B, 1088, M]
        x = torch.cat((x, hand), dim=2).permute(0, 2, 1).contiguous()  # [B, N+M, 1088]
        x = F.relu(self.bnfuse(self.convfuse(x)))  # [B, N, 1088]
        x = x.permute(0, 2, 1).contiguous()  # [B, 1088, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N]
        x = self.conv4(x)  # [B, 1, N]
        x = torch.sigmoid(x)
        return x.view(batchsize, n_pts)


class ContactNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(ContactNet, self).__init__()
        self.n_obj_points = cfg.n_obj_points
        self.n_hand_points = cfg.hand_model.n_surface_points
        self.net = PointNetReg(self.n_obj_points, self.n_hand_points)
        self.hand_model = HandModel(cfg.hand_model, cfg.device)
        # HandModel samples its surface points randomly at construction, but convfuse is tied to the
        # point index (like MANO vertex ids in the original). Keep the exact point set in the
        # checkpoint and restore it on load, so TTA sees the same hand points as training.
        self.register_buffer("hand_surface_points", torch.cat(
            [self.hand_model.mesh[k]["surface_points"] for k in self.hand_model.mesh], dim=0).detach().clone())

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        start = 0
        for k in self.hand_model.mesh:
            n = self.hand_model.mesh[k]["surface_points"].shape[0]
            self.hand_model.mesh[k]["surface_points"] = self.hand_surface_points[start:start + n].to(self.hand_model.device)
            start += n
        assert start == self.hand_surface_points.shape[0]
        return result

    def forward_cmap(self, obj_xyz: torch.Tensor, hand_xyz: torch.Tensor) -> torch.Tensor:
        """obj_xyz (B, N, 3), hand_xyz (B, M, 3) -> predicted contact map (B, N). Used by TTA."""
        return self.net(obj_xyz.permute(0, 2, 1).contiguous(), hand_xyz.permute(0, 2, 1).contiguous())

    def forward(self, data: Dict):
        obj_xyz = data["obj_pc"].to(torch.float32)
        assert obj_xyz.shape[1] == self.n_obj_points, (obj_xyz.shape, self.n_obj_points)
        with torch.no_grad():
            hand_xyz = self.hand_model(data["hand_model_pose"], with_surface_points=True)["surface_points"]
            gt_cmap = object_contact_map(obj_xyz, hand_xyz)

        pred_cmap = self.forward_cmap(obj_xyz, hand_xyz)
        cmap_loss = F.mse_loss(pred_cmap, gt_cmap)
        loss_dict = {
            "cmap": cmap_loss,
            # contact-region error, more readable than the MSE dominated by far (~0) points
            "cmap_l1_contact": (pred_cmap - gt_cmap).abs()[gt_cmap > 0.5].mean()
                               if (gt_cmap > 0.5).any() else pred_cmap.new_zeros(()),
        }
        return cmap_loss, loss_dict, {"pred_cmap": pred_cmap, "gt_cmap": gt_cmap}
