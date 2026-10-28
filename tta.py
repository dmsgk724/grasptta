"""
GraspTTA test-time adaptation on DexGYSNet (Shadow hand).

Port of the TTA loop in GraspTTA's released gen_diverse_grasp_ho3d.py + utils/loss.py (TTT_loss):
the generated grasp parameters are optimized directly (network weights frozen) with SGD, minimizing
    w_contact * L_contact + w_consistency * L_consistency + w_penetration * L_penetration
  L_consistency = s * ||Omega_hand - Omega_c||^2       Omega_hand: pseudo contact map of the object from the
                                                       current hand, Omega_c: frozen ContactNet prediction,
                                                       max-normalized and detached (as the code does)
  L_contact     = s * mean_{obj pts within T of hand} min_j ||p - v_prior_j||^2   (hand-centric contact)
  L_penetration = s * sum_{obj pts inside hand} d^2
MANO -> Shadow hand replacements:
  hand vertices      -> the 1024 HandModel surface points ContactNet was trained with (restored from its ckpt)
  prior contact verts-> HandModel contact candidates (data/mjcf/contact_points.json, 140 points)
  normal-based inside test -> HandModel signed distance (penetration > 0 inside, squared distance)
  + joint angles clamped to the Shadow hand limits after every step
Losses are summed per sample (no 1/B), so a batch behaves like the original one-sample loop.

Initial grasps come from the GraspCVAE decoder (z ~ N(0, I)), exactly as test.py samples them; the output
files have test.py's format, so kubeflow/run_eval_metrics.sh scores them unchanged.

    python tta.py --train_cfg ./config/graspcvae.yaml --test_cfg ./config/infer_graspcvae_test.yaml \
        --tta_cfg ./config/tta.yaml --override model.checkpoint_path "'<graspcvae ckpt>'" save_root "'<out>'"
"""
import argparse
import json
import logging
import os.path as osp
from ast import literal_eval
from typing import Any, Dict, Optional

import pytorch3d.transforms as T
import numpy as np
import torch
from pytorch3d.ops import knn_points

from datasets import TaskDataset_Pose
from model import build_model
from model.contactnet import get_pseudo_cmap
from test import Tester
from utils.config_utils import EasyConfig


def nn_sq_dist(src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    """(B, N, 3), (B, M, 3) -> squared distance from each src point to its nearest trg point, (B, N)."""
    return knn_points(src, trg, K=1).dists[..., 0]


class TTATester(Tester):
    def __init__(self, cfg: EasyConfig) -> None:
        super().__init__(cfg)
        self.tta = cfg.tta
        cn_cfg = EasyConfig()
        cn_cfg.load(self.tta.contactnet_cfg)
        cn_cfg.model.device = cfg.device
        self.contactnet = build_model(cn_cfg.model).to(self.device).eval()
        self.contactnet.load_state_dict(torch.load(self.tta.contactnet_ckpt, map_location=self.device)["model_state_dict"])
        for p in self.contactnet.parameters():
            p.requires_grad_(False)
        assert self.contactnet.n_obj_points == self.tta.n_obj_points
        # the same hand model (and surface point set) ContactNet was trained with
        self.hand_model = self.contactnet.hand_model
        crit = self.model.criterion
        self.norm_fn = getattr(TaskDataset_Pose, f"norm_by_{crit.cfg.norm_type}")
        self.norm_factor = crit.norm_factor
        self.log_rows = []
        self.traj = {"iters": [], "poses": [], "max_pen_cm": []}  # first batch only, for visualization
        logging.info(f"TTA: {dict(self.tta)}")

    # ---- pose conversions (inverse of GraspLossPose.get_hand_model_pose_test) ----
    def norm_to_hand_pose(self, pose_norm: torch.Tensor) -> torch.Tensor:
        out, _ = self.model.criterion.get_hand_model_pose_test({"pred_pose_norm": pose_norm.clone()}, {})
        return out["hand_model_pose"][:, 0]

    def hand_pose_to_norm(self, hand_pose: torch.Tensor) -> torch.Tensor:
        f = self.norm_factor
        euler = T.matrix_to_euler_angles(T.axis_angle_to_matrix(torch.flip(hand_pose[:, 3:6], dims=[-1])), "ZYX")
        return torch.cat([self.norm_fn(hand_pose[:, :3], f[:3]), self.norm_fn(hand_pose[:, 6:], f[6:]),
                          self.norm_fn(euler, f[3:6])], dim=-1)

    # ---- the TTA objective, per sample ----
    def tta_terms(self, pose: torch.Tensor, obj: torch.Tensor) -> Dict[str, torch.Tensor]:
        hand = self.hand_model(pose, obj, with_surface_points=True, with_penetration=True, with_contact_candidates=True)
        hand_pts = hand["surface_points"]
        prior_pts = hand["contact_candidates_dis"]  # HandModel stores the candidate POINTS under this key
        d_all = nn_sq_dist(obj, hand_pts)
        cmap_hand = get_pseudo_cmap(d_all)
        with torch.no_grad():
            cmap_net = self.contactnet.forward_cmap(obj, hand_pts)
            cmap_net = cmap_net / cmap_net.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
        s = self.tta.scales
        consistency = s.consistency * ((cmap_hand - cmap_net) ** 2).sum(dim=1)
        near = (d_all < self.tta.contact_thresh ** 2).float()
        contact = s.contact * (nn_sq_dist(obj, prior_pts) * near).sum(dim=1) / near.sum(dim=1).clamp(min=1.0)
        pen_sq = hand["penetration"].clamp(min=0)
        penetration = s.penetration * pen_sq.sum(dim=1)
        return {"contact": contact, "consistency": consistency, "penetration": penetration,
                "max_pen_cm": pen_sq.max(dim=1)[0].detach().sqrt() * 100}

    def run_tta(self, pose0: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        w = self.tta.weights
        pose = pose0.detach().clone().requires_grad_(True)
        opt = torch.optim.SGD([pose], lr=self.tta.lr, momentum=self.tta.momentum)
        lower, upper = self.hand_model.joints_lower, self.hand_model.joints_upper
        for it in range(self.tta.iters + 1):
            terms = self.tta_terms(pose, obj)
            if it % self.tta.log_every == 0 or it == self.tta.iters:
                if self.batch_idx == 0 and self.tta.get("save_traj", False):
                    self.traj["iters"].append(it)
                    self.traj["poses"].append(pose.detach().cpu().numpy())
                    self.traj["max_pen_cm"].append(terms["max_pen_cm"].cpu().numpy())
                self.log_rows.append({"batch": self.batch_idx, "iter": it,
                                      **{k: v.mean().item() for k, v in terms.items()},
                                      "delta_trans_cm": (pose - pose0)[:, :3].norm(dim=1).mean().item() * 100,
                                      "delta_qpos_deg": (pose - pose0)[:, 6:].abs().mean().item() * 57.2958})
            if it == self.tta.iters:
                break
            loss = (w.contact * terms["contact"] + w.consistency * terms["consistency"]
                    + w.penetration * terms["penetration"]).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if self.tta.clamp_joints:
                with torch.no_grad():
                    pose[:, 6:] = torch.max(torch.min(pose[:, 6:], upper), lower)
        return pose.detach()

    @torch.no_grad()
    def test_one_iter(self, data: Dict):
        crit = self.model.criterion
        pose_norm0 = self.model.sample(data)[:, 0]  # (B, 28), z ~ N(0, I)
        pose0 = self.norm_to_hand_pose(pose_norm0)
        if self.batch_idx == 0:  # the conversion back must reproduce the decoder's pose
            back = self.norm_to_hand_pose(self.hand_pose_to_norm(pose0))
            rot = (T.axis_angle_to_matrix(back[:, 3:6]) - T.axis_angle_to_matrix(pose0[:, 3:6])).abs().max().item()
            logging.info(f"pose round trip: trans {(back - pose0)[:, :3].abs().max().item():.2e}, "
                         f"qpos {(back - pose0)[:, 6:].abs().max().item():.2e}, rotmat {rot:.2e}")
        obj = data["obj_pc"][:, :self.tta.n_obj_points].float()  # trimesh samples at random -> random subset
        if self.batch_idx == 0:
            for k in ["obj_id", "cate_id", "intend_id", "guidance"]:
                self.traj[k] = [str(x) for x in data[k]]
        start = len(self.log_rows)
        with torch.enable_grad():
            pose = self.run_tta(pose0, obj)
        first, last = self.log_rows[start], self.log_rows[-1]
        logging.info(f"batch {self.batch_idx}: max pen {first['max_pen_cm']:.2f} -> {last['max_pen_cm']:.2f} cm, "
                     f"d_trans {last['delta_trans_cm']:.2f} cm, d_qpos {last['delta_qpos_deg']:.2f} deg")

        # match the refined grasps to the GT set exactly as GraspLossPose.forward_test does
        pose_norm = self.hand_pose_to_norm(pose).unsqueeze(1)
        outputs = {"pred_pose_norm": pose_norm, "hand_model_pose": pose.unsqueeze(1),
                   "translation_norm": pose_norm[..., :3], "qpos_norm": pose_norm[..., 3:25], "rotation": pose_norm[..., 25:]}
        assignments = crit.matcher(outputs, data)
        matched_preds, matched_targets = crit.get_matched_by_assignment(outputs, data, assignments)
        raw_results, matched_results = [], []
        for i in range(pose.shape[0]):
            raw_results.append({"obj_id": data["obj_id"][i], "cate_id": data["cate_id"][i], "action_id": data["intend_id"][i],
                                "guidance": data["guidance"][i], "predictions": outputs["hand_model_pose"][i].cpu().numpy().tolist()})
            matched_results.append({"cate_id": data["cate_id"][i], "obj_id": data["obj_id"][i], "action_id": data["intend_id"][i],
                                    "guidance": data["guidance"][i],
                                    "predictions": matched_preds["hand_model_pose"][i].cpu().numpy().tolist(),
                                    "targets": matched_targets["hand_model_pose"][i].cpu().numpy().tolist()})
        self.batch_idx += 1
        return raw_results, matched_results

    def test(self) -> float:
        self.batch_idx = 0
        super().test()
        path = osp.join(self.save_root, "tta_log.json")
        json.dump(self.log_rows, open(path, "w"), indent=1)
        if self.traj["poses"]:
            np.savez(osp.join(self.save_root, "tta_traj.npz"), iters=np.array(self.traj["iters"]),
                     poses=np.stack(self.traj["poses"]), max_pen_cm=np.stack(self.traj["max_pen_cm"]),
                     **{k: np.array(self.traj[k]) for k in ["obj_id", "cate_id", "intend_id", "guidance"]})
        first = [r for r in self.log_rows if r["iter"] == 0]
        last = [r for r in self.log_rows if r["iter"] == self.tta.iters]
        for name in ["contact", "consistency", "penetration", "max_pen_cm", "delta_trans_cm", "delta_qpos_deg"]:
            a = sum(r[name] for r in first) / len(first)
            b = sum(r[name] for r in last) / len(last)
            logging.info(f"TTA {name:<15s} {a:10.4f} -> {b:10.4f}")
        logging.info(f"TTA log saved to {path}")


def load_cfg(args: Optional[argparse.Namespace] = None) -> EasyConfig:
    """train_cfg < test_cfg < tta_cfg < --override"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_cfg", type=str, required=True)
    parser.add_argument("-o", "--test_cfg", type=str, required=True)
    parser.add_argument("--tta_cfg", type=str, default="./config/tta.yaml")
    parser.add_argument("--override", nargs="*", default=[], metavar="model.checkpoint_path '<path>'")
    args = parser.parse_args()
    cfg = EasyConfig()
    cfg.load(args.train_cfg)
    for path in [args.test_cfg, args.tta_cfg]:
        new_cfg = EasyConfig()
        new_cfg.load(path)
        cfg.update(new_cfg.dict())

    def _load_argparse_cfg(args_dict: Dict[str, Any]):
        output = {}
        for key, arg in args_dict.items():
            arg = literal_eval(arg)
            current = output
            keys = key.split(".")
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = arg
        return output

    assert len(args.override) % 2 == 0
    cfg.update(_load_argparse_cfg(dict(zip(args.override[::2], args.override[1::2]))))
    return cfg


if __name__ == "__main__":
    TTATester(load_cfg()).test()
