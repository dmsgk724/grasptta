# GraspTTA on DexGYSNet

Reproduction of the GraspTTA baseline of Table 1 in *Grasp as You Say* (DexGYS, NeurIPS 2024, arXiv 2405.19291)
on the DexGYSNet dataset with the Shadow hand. GraspTTA = GraspCVAE + ContactNet + test-time adaptation (TTA);
this repo contains all three (GraspCVAE alone is also in the `graspcvae` repo).

## Results

Scored like our DexGYSGrasp reproduction (dexter_temp DexGYS benchmark; diversity = 8 seeded samplings per
input condition, each refined by TTA). Con. and Q1 are raw × 10 (the paper's ×10⁻¹ unit).

| Method | P-FID ↓ | CD ↓ | Con. ↓ | SR sim-only ↑ | SR E_pen² < 0.001 ↑ | SR max_pen < 5mm ↑ | Q1 ↑ | Pen. (cm) ↓ | δt (cm) | δr (deg) | δq (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GraspTTA (paper) | 33.15 | 12.19 | 1.11 | 43.46 * | | | 0.71 | 0.19 | 2.11 | 6.15 | 3.87 |
| **GraspTTA (ours)** | 28.75 | 5.66 | 1.03 | 52.75 | 45.40 | 22.68 | 1.02 | 0.48 | 0.73 | 3.82 | 0.60 |

\* the paper reports one Success rate without saying which penetration filter it uses.

## Checkpoints

GraspTTA has no weights of its own: it uses the GraspCVAE and ContactNet checkpoints below with `config/tta.yaml`.
Both are private Hugging Face repos (log in with a token that has access); paths mirror `Experiments/`, so the
commands below and `config/tta.yaml` work unchanged after:

```bash
hf auth login   # once
hf download EunhaPark/graspcvae --local-dir ./Experiments
hf download EunhaPark/grasptta  --local-dir ./Experiments
```

| What | Checkpoint | Config | Notes |
|---|---|---|---|
| **GraspCVAE** | [`EunhaPark/graspcvae`](https://huggingface.co/EunhaPark/graspcvae): `graspcvae_kl10/epoch200_minus_loss_-5.0587_latest.pth` (1.7 GB) | `config/graspcvae.yaml` + `model.criterion.loss_weights.kl 10.0` | β = 10, epoch 200; initial grasps |
| **ContactNet** | [`EunhaPark/grasptta`](https://huggingface.co/EunhaPark/grasptta): `contactnet/epoch100_minus_loss_-0.0032_latest.pth` (150 MB) | `config/contactnet.yaml` | already set in `config/tta.yaml`; stores its hand point set |

## Environment

The Grasp-as-You-Say image (`Dockerfile.gys` in Grasp-as-You-Say; `postech-a.kr-central-2.kcr.dev/dmsgk724/grasp-as-you-say:latest`)
for training / test / TTA, and the dexter image (`.../dmsgk724/dexter:87c1e1d`) for the benchmark metrics.
`data/` must point to Grasp-as-You-Say's `data/` (DexGYSNet json, OakInk shapes, Shadow hand MJCF); the committed
symlink expects Grasp-as-You-Say two levels up (`../../Grasp-as-You-Say/data`). Run everything from the repo root.

## Evaluation commands

### 1. Generate grasps on the test set

```bash
# GraspCVAE samples (z ~ N(0, I), seed 3407), then 300 TTA steps (ContactNet ckpt is taken from config/tta.yaml)
python tta.py --train_cfg ./config/graspcvae.yaml --test_cfg ./config/infer_graspcvae_test.yaml --tta_cfg ./config/tta.yaml \
  --override model.checkpoint_path "'./Experiments/graspcvae_kl10/epoch200_minus_loss_-5.0587_latest.pth'" \
             save_root "'./Experiments/my_eval/grasptta'" seed 3407
```
Writes `<save_root>/test_results/epoch_200_test/{raw_results,matched_results}.json` and the TTA log
(~85 min on one GPU). `save_root` must not exist yet, otherwise the script asks before deleting it.

### 2. Convert to the dexter benchmark format

```bash
python scripts/to_dexter_predictions.py \
  --matched ./Experiments/my_eval/grasptta/test_results/epoch_200_test/matched_results.json \
  --out-dir ./Experiments/my_eval/grasptta_pred
# -> grasptta_pred/dexter309/predictions.json   (same rule as our DexGYSGrasp reproduction)
```

### 3. Metrics: P-FID, CD, Con., Q1, Pen. (dexter image)

```bash
source /python/.venv/bin/activate
cd /workspace/users/dmsgk724/dexter_temp && uv pip install -e .
PRED=<this repo>/Experiments/my_eval/grasptta_pred/dexter309/predictions.json
DATA=/workspace/datasets/dexterp/dexgys_final
python -m benchmark.dexgys.chamfer --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
python -m benchmark.dexgys.q1      --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
python -m benchmark.dexgys.fid     --pred-path $PRED --data-path $DATA
# -> metrics.json next to predictions.json (cmap and valid_q1 raw; mean_pen in m)
```
`kubeflow/run_eval_metrics.sh` does steps 2–3 for a whole run (`SAVE_ROOT`, `NAME`, `VARIANTS` env vars).

### 4. Success rate (Isaac Gym, dexter_temp `.venv-isaacgym`)

```bash
bash benchmark/dexgys/setup_isaacgym.sh && source .venv-isaacgym/bin/activate   # once, in dexter_temp
python benchmark/dexgys/success_rate.py --pred-path $PRED --data-path $DATA --no_force --gpu 0 --parallel 8
```
Prediction file of the reported run (cholab PVC):
`/workspace/users/dmsgk724/dexgys_baseline/graspcvae/Experiments/eval/grasptta_latest/dexter309/predictions.json`

### 5. Diversity (δt, δr, δq)

```bash
# each of the 8 samplings (seed 3407..3414) is refined by TTA; split seeds over GPUs, then merge
NAME=grasptta_latest CVAE_CKPT=./Experiments/graspcvae_kl10/epoch200_minus_loss_-5.0587_latest.pth \
  MODE=sample SEEDS="0 1 2 3 4 5 6 7" bash kubeflow/run_tta_diversity.sh
NAME=grasptta_latest MODE=merge bash kubeflow/run_tta_diversity.sh
# -> Experiments/eval/grasptta_latest/diversity/dexter309/diversity_with_std.json  (mean_object = reported)
```

## Training

```bash
# GraspCVAE, beta = 10 (the config default kl is 0.01, so override it)
python train.py -t ./config/graspcvae.yaml --override save_root "'./Experiments/graspcvae_kl10'" model.criterion.loss_weights.kl 10.0
# ContactNet (GraspTTA paper Sec 3.2 / 4.1: GT hand + object -> pseudo contact map, L2)
python train.py -t ./config/contactnet.yaml
```
200 epochs of GraspCVAE take ~13 h on one GPU.

Kubeflow manifests of the reported runs (job names end with `-eunha`; they run from the cholab PVC working copy
`dexgys_baseline/graspcvae`): `kubeflow/graspcvae/` (GraspCVAE train + eval), `kubeflow/grasptta/trainjob-contactnet.yaml`,
`kubeflow/grasptta/evaljob-grasptta-latest-*.yaml` (TTA samplings g1–g4, merge, metrics).

## How it was built

Ported from GraspTTA's released code to the Shadow hand, on top of our GraspCVAE (see the `graspcvae` repo for
how GraspCVAE was built).
- **ContactNet** (`model/contactnet.py`): `cmapnet_objhand.py` architecture (two PointNets, `convfuse` mixing along
  the point axis) with 3,000 object points and 1,024 Shadow hand points (convfuse 4024 → 3000). Trained on GT hand +
  object against the pseudo contact map `1 − 2(σ(2d) − 0.5)` (d in cm), L2, Adam 1e-4, 100 epochs, lr halved at
  30/60/80/90. The hand point set is saved in the checkpoint because convfuse is tied to point order.
- **TTA** (`tta.py`, `config/tta.yaml`): as the released code (not the paper's decoder fine-tuning), the networks are
  frozen and the 28-d pose is optimized with SGD (lr 6.25e-6, momentum 0.8, 300 steps) on consistency with
  ContactNet (weight 1), hand-centric contact (1) and penetration (7). MANO parts replaced: hand vertices →
  ContactNet's surface points, prior contact vertices → 140 Shadow contact candidates, normal-based inside test →
  HandModel signed distance; joint limits clamped every step.
- **Initial grasps**: GraspCVAE β = 10, epoch 200 (the reported GraspCVAE checkpoint).
- Known gap: penetration rises during TTA instead of falling as in the paper.
