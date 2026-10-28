#!/usr/bin/env bash
# Diversity of the GraspCVAE baseline, same protocol as our DexGYS reproduction
# (Grasp-as-You-Say/kubeflow/run_diversity.sh) minus the QGC refinement step:
#   "We perform eight samples ... under the same input conditions" -> std of translation (cm),
#   rotation and joints (Euler, deg), averaged over input conditions.
#
#   per checkpoint <V> in $VARIANTS (best, latest):
#     test-set sampling with seed 3407+s, s = 0..N-1 -> Experiments/eval/<NAME>_<V>/diversity/s<s>
#     merge the N raw_results per input condition (obj_id, cate_id, action_id, guidance)
#     -> scripts/diversity/calc_diversity.py on {all, dexter309}
#   seed 3407 (s=0) is the config seed, so sample 0 approximately reproduces $SAVE_ROOT/test_<V>
#   (GPU nondeterminism -> not bit-exact).
#
# Output: Experiments/eval/<NAME>_<V>/diversity/{all,dexter309}/diversity_with_std.json
# Usage:  VARIANTS="best latest" SAVE_ROOT=./Experiments/graspcvae NAME=graspcvae bash kubeflow/run_diversity.sh
#         (finished samplings are skipped via .done markers)
set -euo pipefail
cd "$(dirname "$0")/.."

N=${N_SAMPLES:-8}
SAVE_ROOT=${SAVE_ROOT:-./Experiments/graspcvae}
NAME=${NAME:-graspcvae}
TRAIN_CFG=${TRAIN_CFG:-./config/graspcvae.yaml}
VARIANTS=${VARIANTS:-"best latest"}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}   # extra "key value" config overrides for test.py, e.g. "model.encoder_input 'pose'"
EV=./Experiments/eval
trap 'chown -R 1000:100 Experiments/eval 2>/dev/null || true' EXIT

# pick_ckpt <train save_root> <best|latest>  (same rule as trainjob-graspcvae-full.yaml)
pick_ckpt() {
  python - "$1" "$2" <<'PY'
import glob, re, sys
root, which = sys.argv[1:]
if which == "latest":
    (c,) = glob.glob(f"{root}/epoch*_minus_loss_*_latest.pth")
    print(c)
else:
    cands = [(float(m.group(1)), p) for p in glob.glob(f"{root}/epoch*_minus_loss_*_.pth")
             if (m := re.search(r"_minus_loss_(-?[\d.]+)_\.pth$", p))]
    print(max(cands)[1])
PY
}

for V in $VARIANTS; do
  D="$EV/${NAME}_$V/diversity"
  mkdir -p "$D"
  CKPT=$(pick_ckpt "$SAVE_ROOT" "$V")
  echo "##### ${NAME}_$V   ckpt: $CKPT"
  for s in $(seq 0 $((N - 1))); do
    [ -f "$D/s$s.done" ] && continue
    echo "=================== ${NAME}_$V sample $s / seed $((3407 + s)) ($(date '+%F %T')) ==================="
    rm -rf "$D/s$s"   # test.py would input()-prompt otherwise
    python ./test.py --train_cfg "$TRAIN_CFG" --test_cfg ./config/infer_graspcvae_test.yaml \
      --override model.checkpoint_path "'$CKPT'" save_root "'$D/s$s'" seed "$((3407 + s))" $EXTRA_OVERRIDES < /dev/null
    touch "$D/s$s.done"
  done

  echo "=================== ${NAME}_$V: merge + diversity ($(date '+%F %T')) ==================="
  python - "$D" "$N" "$SAVE_ROOT/test_$V" <<'PY'
import json, glob, os, sys
import numpy as np
D, N, REF = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dexter = {x["obj_id"] for x in json.load(open(
    "/workspace/users/dmsgk724/dexter_temp/test_output/dexter_rot_bs16x4_ls1_final_test/predictions.json"))}
merged, n_per_run = {}, []
for s in range(N):
    (path,) = glob.glob(f"{D}/s{s}/test_results/*/raw_results.json")
    raw = json.load(open(path))
    n_per_run.append(sum(len(r["predictions"]) for r in raw))
    for r in raw:
        k = (r["obj_id"], r["cate_id"], r["action_id"], r["guidance"])
        merged.setdefault(k, dict(r, predictions=[]))["predictions"] += r["predictions"]
assert len(set(n_per_run)) == 1, n_per_run
# sanity: seed 3407 (s=0) vs the full job's test result for this checkpoint (close, not bit-exact)
ref = glob.glob(f"{REF}/test_results/*/raw_results.json")
if ref:
    (s0,) = glob.glob(f"{D}/s0/test_results/*/raw_results.json")
    a = np.array([p for r in json.load(open(ref[0])) for p in r["predictions"]]).reshape(-1, 28)
    b = np.array([p for r in json.load(open(s0)) for p in r["predictions"]]).reshape(-1, 28)
    print(f"seed-0 vs full-job test result: median trans diff {np.median(np.linalg.norm(a[:, :3] - b[:, :3], axis=1)) * 100:.2f} cm")
groups = list(merged.values())
print(f"{len(groups)} input conditions, predictions per condition: {sorted({len(g['predictions']) for g in groups})}")
for name, sel in [("all", groups), ("dexter309", [g for g in groups if g["obj_id"] in dexter])]:
    os.makedirs(f"{D}/{name}", exist_ok=True)
    json.dump(sel, open(f"{D}/{name}/raw_results_x{N}.json", "w"))
PY
  for SET in all dexter309; do
    rm -f "$D/$SET/diversity_with_std.json"
    python scripts/diversity/calc_diversity.py -r "$D/$SET/raw_results_x$N.json" < /dev/null
    python - "$D/$SET/diversity_with_std.json" "${NAME}_$V/$SET" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["diversity"]["mean_object"]
print(f"[{sys.argv[2]}] δt {m['mean_std_trans']:.3f} cm | "
      f"δr {m['mean_std_rotation']:.3f} deg (wrap-aware {m['mean_real_std_rotation']:.3f}) | "
      f"δq {m['mean_std_joints']:.3f} deg (wrap-aware {m['mean_real_std_joint_angle']:.3f})"
      "   [mean over input conditions of per-condition std]")
PY
  done
done
echo "DIVERSITY DONE"
