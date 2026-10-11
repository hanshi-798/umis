"""Phase G: generate paper figures from stored eval results.

Usage:
  python harness/make_figures.py --results results --manifest \
      bench/out/manifest.json --out_dir results/figs
"""
import argparse
import json
import os

import numpy as np


def load(out_dir, name):
    with open(os.path.join(out_dir, name)) as f:
        return json.load(f)


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    man = load(args.manifest_dir, "manifest.json") if args.manifest_dir else None
    ev = load(args.results, "umis_eval.json")
    eid = {e["id"]: e for e in man["entries"]}

    # ---- 1. ASR-FPR threshold curve: sweep t2 on stored risk scores ----
    test_atk = [(k, v) for k, v in ev.items()
                if "error" not in v and eid[k]["split"] == "test"
                and eid[k]["kind"] == "attack"]
    test_ben = [(k, v) for k, v in ev.items()
                if "error" not in v and eid[k]["split"] == "test"
                and eid[k]["kind"] == "benign"]
    live_fam = {"typographic", "typo_obfuscated", "mixed_cross",
                "tts_injection", "piggyback", "semantic_framing",
                "echo_embed"}
    live = [(k, v) for k, v in test_atk if eid[k]["family"] in live_fam]
    ts = np.linspace(0.1, 1.0, 19)
    allowed, blocked = [], []
    for t2 in ts:
        al = sum(1 for _, v in live
                 if (v.get("risk_score") or 0) < t2)
        bl = sum(1 for _, v in test_ben
                 if (v.get("risk_score") or 0) >= t2)
        allowed.append(al / max(len(live), 1))
        blocked.append(bl / max(len(test_ben), 1))
    plt.figure(figsize=(4, 2.8))
    plt.plot(blocked, allowed, "o-", markersize=3)
    plt.xlabel("benign blocked (L1)")
    plt.ylabel("live attacks allowed (L1)")
    plt.title("ASR-FPR trade-off (test, L1 scores)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_curve.pdf"))

    # ---- 2. per-probe bars (from summary) ----
    summ = load(args.results, "umis_eval.json.summary.json")
    pp = summ["per_probe"]
    probes = ["p_file", "p_content", "p_intent"]
    prec = [pp[p]["precision"] for p in probes]
    rec = [pp[p]["recall"] for p in probes]
    x = np.arange(3)
    plt.figure(figsize=(4, 2.8))
    plt.bar(x - 0.18, prec, 0.34, label="precision")
    plt.bar(x + 0.18, rec, 0.34, label="recall")
    plt.xticks(x, ["P-file", "P-content", "P-intent"])
    plt.ylim(0, 1.05)
    plt.ylabel("rate")
    plt.title("Per-probe precision/recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "per_probe.pdf"))

    # ---- 3. per-placement bars ----
    pd_ = summ["per_placement"]
    keys = ["container_metadata", "content", "signal",
            "semantic_only", "mixed"]
    vals = [pd_.get(k, {}).get("rate", 0.0) for k in keys]
    plt.figure(figsize=(4, 2.8))
    plt.bar(keys, vals)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("BLOCK rate")
    plt.title("Per-placement detection")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "per_placement.pdf"))

    # ---- 4. ablation/baseline bars ----
    names, allow, block = [], [], []
    rows = [("UMIS-F", "umis_eval.json.summary.json"),
            ("naive", "naive_eval.json.summary.json"),
            ("image-only", "image_only_eval.json.summary.json"),
            ("audio-only", "audio_only_eval.json.summary.json")]
    for nm, fn in rows:
        p = os.path.join(args.results, fn)
        if not os.path.exists(p):
            continue
        s = load(args.results, fn)
        names.append(nm)
        allow.append(s["attack_allowed_rate"])
        block.append(s["benign_block_rate"])
    for p in ("p_file", "p_content", "p_intent"):
        fn = f"ablate_{p}.json.summary.json"
        if not os.path.exists(os.path.join(args.results, fn)):
            continue
        s = load(args.results, fn)
        names.append(f"--ablate {p}")
        allow.append(s["attack_allowed_rate"])
        block.append(s["benign_block_rate"])
    x = np.arange(len(names))
    plt.figure(figsize=(5.2, 2.8))
    plt.bar(x - 0.18, allow, 0.34, label="attack allowed")
    plt.bar(x + 0.18, block, 0.34, label="benign blocked")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Ablations and baselines")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "ablations.pdf"))
    print("figures ->", args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
