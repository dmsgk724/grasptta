"""
Convert test.py's matched_results.json into the dexter_temp DexGYS benchmark schema
(benchmark/dexgys/{chamfer,q1,fid}.py read predictions.json):

    [{"obj_id", "guidance", "predictions": hand_model_pose(28), "targets": GT dex_grasp(28)}, ...]

Same rule as the predictions.json of our DexGYS reproduction (verified 9971/9971 against
Grasp-as-You-Say/Experiments/results/idgc_best__qgc_best/.../predictions.json):
  - predictions: matched_results[i]["predictions"], in matched_results order
  - targets:     the k-th GT dex_grasp of the same (obj_id, guidance) in test.json, where k is the
                 k-th occurrence of that (obj_id, guidance) in matched_results
  - dexter309:   entries whose obj_id is among the 309 objects of dexter's test predictions

Uses only the standard library, so it runs in either image.

    python scripts/to_dexter_predictions.py --matched <.../matched_results.json> --out-dir Experiments/eval/<variant>
"""
import argparse
import json
import os
from collections import defaultdict

DEXTER_PRED = "/workspace/users/dmsgk724/dexter_temp/test_output/dexter_rot_bs16x4_ls1_final_test/predictions.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", required=True, help="test.py matched_results.json")
    parser.add_argument("--test-json", default="./data/dexgys/test.json")
    parser.add_argument("--dexter-pred", default=DEXTER_PRED, help="defines the dexter309 object subset")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    matched = json.load(open(args.matched))
    gt = defaultdict(list)
    for e in json.load(open(args.test_json)):
        gt[(e["obj_id"], e["guidance"])].append(e["dex_grasp"])

    seen = defaultdict(int)
    preds = []
    for r in matched:
        k = (r["obj_id"], r["guidance"])
        preds.append({
            "obj_id": r["obj_id"],
            "guidance": r["guidance"],
            "predictions": r["predictions"],
            "targets": gt[k][seen[k]],
        })
        seen[k] += 1

    dexter_objs = {x["obj_id"] for x in json.load(open(args.dexter_pred))}
    for name, sel in [("all", preds), ("dexter309", [p for p in preds if p["obj_id"] in dexter_objs])]:
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)
        path = os.path.join(args.out_dir, name, "predictions.json")
        json.dump(sel, open(path, "w"))
        print(f"{path}: {len(sel)} grasps, {len({p['obj_id'] for p in sel})} objects")


if __name__ == "__main__":
    main()
