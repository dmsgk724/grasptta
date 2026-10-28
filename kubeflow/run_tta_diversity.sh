#!/usr/bin/env bash
# Diversity of GraspTTA, same protocol as kubeflow/run_diversity.sh: the test set is sampled with seed 3407+s,
# s = 0..N-1, each sampling is refined by TTA (tta.py, config/tta.yaml), and the N refined sets are merged per
# input condition (obj_id, cate_id, action_id, guidance) and scored with scripts/diversity/calc_diversity.py.
#
#   MODE=sample SEEDS="1 2"  -> TTA samplings s1, s2 into $D/s<s>   (several jobs can split the seeds)
#   MODE=merge               -> merge s0..s{N-1} + calc_diversity on {all, dexter309}
# s0 reuses the full TTA run in $REF (seed 3407) instead of running it again.
# Output: Experiments/eval/<NAME>/diversity/{all,dexter309}/diversity_with_std.json
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${MODE:-sample}
N=${N_SAMPLES:-8}
SEEDS=${SEEDS:-"$(seq -s ' ' 0 $((N - 1)))"}
NAME=${NAME:-grasptta_best}
CVAE_CKPT=${CVAE_CKPT:-./Experiments/graspcvae_kl10/epoch35_minus_loss_-4.6387_.pth}
REF=${REF:-./Experiments/grasptta/test_best}
D=./Experiments/eval/$NAME/diversity
mkdir -p "$D"
trap 'chown -R 1000:100 Experiments/eval 2>/dev/null || true' EXIT

if [ "$MODE" = "sample" ]; then
  for s in $SEEDS; do
    [ -f "$D/s$s.done" ] && continue
    if [ "$s" = "0" ] && ls "$REF"/test_results/*/raw_results.json >/dev/null 2>&1; then
      rm -rf "$D/s0" && mkdir -p "$D/s0" && cp -r "$REF/test_results" "$D/s0/"
      echo "s0: reused $REF (seed 3407)"; touch "$D/s0.done"; continue
    fi
    echo "=================== $NAME sample $s / seed $((3407 + s)) ($(date '+%F %T')) ==================="
    rm -rf "$D/s$s"   # tta.py (Tester) would input()-prompt otherwise
    python tta.py --train_cfg ./config/graspcvae.yaml --test_cfg ./config/infer_graspcvae_test.yaml --tta_cfg ./config/tta.yaml \
      --override model.checkpoint_path "'$CVAE_CKPT'" save_root "'$D/s$s'" seed "$((3407 + s))" \
                 data.test.num_workers 2 tta.save_traj False < /dev/null 2>&1 | grep -E --line-buffered "TTA [a-z_]+ |Traceback|Error"
    touch "$D/s$s.done"
  done
  echo "SAMPLING DONE: $SEEDS"
  exit 0
fi

for s in $(seq 0 $((N - 1))); do
  [ -f "$D/s$s.done" ] || { echo "ERROR: sample $s missing"; exit 1; }
done
echo "=================== $NAME: merge + diversity ($(date '+%F %T')) ==================="
python - "$D" "$N" <<'PY'
import json, glob, os, sys
D, N = sys.argv[1], int(sys.argv[2])
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
groups = list(merged.values())
print(f"{len(groups)} input conditions, predictions per condition: {sorted({len(g['predictions']) for g in groups})}")
for name, sel in [("all", groups), ("dexter309", [g for g in groups if g["obj_id"] in dexter])]:
    os.makedirs(f"{D}/{name}", exist_ok=True)
    json.dump(sel, open(f"{D}/{name}/raw_results_x{N}.json", "w"))
PY
for SET in all dexter309; do
  rm -f "$D/$SET/diversity_with_std.json"
  python scripts/diversity/calc_diversity.py -r "$D/$SET/raw_results_x$N.json" < /dev/null
  python - "$D/$SET/diversity_with_std.json" "$NAME/$SET" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["diversity"]["mean_object"]
print(f"[{sys.argv[2]}] δt {m['mean_std_trans']:.3f} cm | δr {m['mean_std_rotation']:.3f} deg | δq {m['mean_std_joints']:.3f} deg")
PY
done
echo "DIVERSITY DONE"
