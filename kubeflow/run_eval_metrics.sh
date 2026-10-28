#!/usr/bin/env bash
# Computational metrics of the GraspCVAE baseline, scored with dexter_temp's DexGYS benchmark exactly
# like our DexGYS reproduction (Grasp-as-You-Say/kubeflow/run_dexter_metrics.sh):
#   chamfer -> hand_chamfer (CD), cmap (contact map)
#   q1      -> valid_q1 (Q1, zeroed if pen > 5mm), mean_pen (max pen depth, m) + q1.csv
#   fid     -> fid (P-FID; point-e pointnet from dexter_temp/point_e_model_cache)
# Runs inside the dexter image (source /python/.venv/bin/activate first).
#
#   per checkpoint <V> in $VARIANTS (best, latest):
#     $SAVE_ROOT/test_<V>/test_results/*/matched_results.json
#       -> scripts/to_dexter_predictions.py -> Experiments/eval/<NAME>_<V>/{all,dexter309}/predictions.json
#       -> chamfer / q1 / fid -> metrics.json next to each predictions.json
# Usage: VARIANTS="best latest" SAVE_ROOT=./Experiments/graspcvae NAME=graspcvae bash kubeflow/run_eval_metrics.sh
set -eo pipefail
ROOT=/workspace/users/dmsgk724/dexgys_baseline/graspcvae
SAVE_ROOT=${SAVE_ROOT:-./Experiments/graspcvae}
NAME=${NAME:-graspcvae}
VARIANTS=${VARIANTS:-"best latest"}
DATA=/workspace/datasets/dexterp/dexgys_final
trap 'chown -R 1000:100 $ROOT/Experiments/eval 2>/dev/null || true' EXIT

cd $ROOT
for V in $VARIANTS; do
  MATCHED=$(ls $SAVE_ROOT/test_$V/test_results/*/matched_results.json)
  python scripts/to_dexter_predictions.py --matched "$MATCHED" --out-dir "$ROOT/Experiments/eval/${NAME}_$V"
done

cd /workspace/users/dmsgk724/dexter_temp
git config --global --add safe.directory '*'
git log -1 --oneline
nvidia-smi --query-gpu=name --format=csv,noheader
uv pip install -e .

for V in $VARIANTS; do
  E=$ROOT/Experiments/eval/${NAME}_$V
  for SET in dexter309 all; do
    PRED=$E/$SET/predictions.json
    echo "=================== ${NAME}_$V / $SET ($(date '+%F %T')) ==================="
    rm -f "$E/$SET/metrics.json"
    python -m benchmark.dexgys.chamfer --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
    python -m benchmark.dexgys.q1      --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
    python -c "import csv,sys; p=[float(r['pen']) for r in csv.DictReader(open(sys.argv[1]))]; print(f'Q1 max<5mm {sum(x<0.005 for x in p)/len(p)*100:.2f}% (n={len(p)})')" $E/$SET/q1.csv
    python -m benchmark.dexgys.fid     --pred-path $PRED --data-path $DATA
    echo "--- ${NAME}_$V / $SET metrics.json"; cat $E/$SET/metrics.json; echo
  done
done
echo "EVAL DONE"
