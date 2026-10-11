"""Train the fusion head + calibrate thresholds from collected features.

Usage:
  python train_head.py --features results/features.json \
      --out results/fusion_head.joblib --t1_out results/thresholds.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "code"))

from umis.fusion import FusionHead, choose_thresholds  # noqa: E402


def run(args):
    with open(args.features) as f:
        data = json.load(f)
    train = [(k, v) for k, v in data.items()
             if "error" not in v and v["split"] == "train"]
    val = [(k, v) for k, v in data.items()
           if "error" not in v and v["split"] == "val"]
    print(f"train: {len(train)}, val: {len(val)}")
    if not train or not val:
        raise ValueError("Nonempty train and validation splits are required")
    head = FusionHead().fit([v["features"] for _, v in train],
                            [v["label"] for _, v in train])
    head.calibrate([v["features"] for _, v in val], [v["label"] for _, v in val])
    head.save(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.t1_out)), exist_ok=True)
    t1, t2 = choose_thresholds(
        head.predict_calibrated([v["features"] for _, v in val]),
        [v["label"] for _, v in val], args.t1, args.t2, args.fpr_target)
    with open(args.t1_out, "w") as f:
        json.dump({"t1": t1, "t2": t2}, f)
    print(f"head -> {args.out}; t1={t1:.3f} t2={t2:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t1_out", required=True)
    ap.add_argument("--t1", type=float, default=0.3)
    ap.add_argument("--t2", type=float, default=0.75)
    ap.add_argument("--fpr_target", type=float, default=0.05)
    args = ap.parse_args()
    run(args)
