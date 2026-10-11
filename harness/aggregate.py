"""Phase G aggregation: build tables + figures from raw result files.

Figures: ROC/PR curves, ASR-FPR trade-off, per-probe bars, per-placement
bars, latency distribution, L2 escalation rate, ablation bars, adaptive
curves. All tables emitted as CSV/JSON for the paper.
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import metrics  # noqa: E402

STYLE = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
         "legend.fontsize": 8, "figure.dpi": 150}


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def fig_roc_pr(umis_eval, features, out_dir):
    """ROC + PR curves from UMIS risk scores vs labels."""
    scores, labels = [], []
    for k, v in umis_eval.items():
        if "error" in v or v.get("risk_score") is None:
            continue
        if k not in features:
            continue
        scores.append(v["risk_score"])
        labels.append(int(features[k]["label"]))
    r = metrics.roc_pr_auc(scores, labels)
    if np.isnan(r["roc_auc"]):
        return r
    plt.style.use("default")
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(r["fpr"], r["tpr"], "b-", lw=1.5)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC (AUC={r['roc_auc']:.3f})")
    axes[1].plot(r["rec"], r["prec"], "b-", lw=1.5)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR (AUC={r['pr_auc']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_pr.pdf"), bbox_inches="tight")
    plt.close(fig)
    return r


def fig_bars(d, key, title, out_dir, fname, color="#4C72B0"):
    if not d:
        return
    names = sorted(d)
    vals = [d[k][key] for k in names]
    plt.style.use("default")
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5, 2.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=color, height=0.6)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(key)
    ax.set_title(title)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)


def fig_latency(umis_eval, out_dir):
    lats = [v["latency_s"] for v in umis_eval.values()
            if "error" not in v and v.get("latency_s") is not None]
    if not lats:
        return
    lats = np.asarray(lats)
    plt.style.use("default")
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.hist(lats, bins=40, color="#55A868", edgecolor="white")
    ax.axvline(np.median(lats), color="r", lw=1, label=f"median {np.median(lats):.2f}s")
    ax.axvline(np.percentile(lats, 95), color="k", ls="--", lw=1,
               label=f"p95 {np.percentile(lats, 95):.2f}s")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_adaptive(r1, r2, out_dir):
    """ASR degradation curve across adaptive rounds (if available)."""
    plt.style.use("default")
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(4, 2.6))
    xs, ys, labels = [], [], []
    if r1 is not None:
        xs.append(0); ys.append(r1.get("asr", 0)); labels.append("no-defense")
        xs.append(1); ys.append(r1.get("defended_asr", 0))
        labels.append("after Round 1")
    if r2 is not None:
        xs.append(2); ys.append(r2.get("defended_asr", 0))
        labels.append("after Round 2")
    if xs:
        ax.plot(xs, ys, "o-", color="#C44E52")
        ax.set_xticks(xs, labels)
        ax.set_ylabel("Residual ASR")
        ax.set_ylim(0, max(1.0, max(ys) * 1.1))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "adaptive_curve.pdf"),
                    bbox_inches="tight")
    plt.close(fig)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = load_json(args.manifest)
    entries = manifest["entries"] if manifest else []
    features = load_json(args.features, {})
    out = {}

    # per-probe / per-placement / ablations from eval files
    for name, path in (args.evals or {}).items():
        ev = load_json(path, {})
        if ev is None:
            continue
        summ = {}
        verdicts = {k: v.get("verdict", "") for k, v in ev.items()
                    if "error" not in v}
        by_id = {e["id"]: e for e in entries}
        atk = [k for k in verdicts if by_id.get(k, {}).get("kind")
               == "attack"]
        ben = [k for k in verdicts if by_id.get(k, {}).get("kind")
               == "benign"]
        summ["attack_allowed_rate"] = sum(
            1 for k in atk if verdicts[k] != "BLOCK") / max(len(atk), 1)
        summ["benign_block_rate"] = sum(
            1 for k in ben if verdicts[k] != "ALLOW+AUDIT") / \
            max(len(ben), 1)
        summ["verdict_counts"] = {v: sum(1 for x in verdicts.values()
                                         if x == v)
                                  for v in set(verdicts.values())}
        summ["l2_escalation"] = sum(
            1 for k, v in ev.items() if v.get("escalation_used")
            and "error" not in v) / max(len(ev), 1)
        out[name] = summ
        if name.startswith("umis"):
            roc = fig_roc_pr(ev, features, args.out_dir)
            summ["roc_auc"] = roc["roc_auc"]
            summ["pr_auc"] = roc["pr_auc"]
            fig_latency(ev, args.out_dir)

    # adaptive curve
    fig_adaptive(load_json(args.adaptive_r1), load_json(args.adaptive_r2),
                 args.out_dir)

    with open(os.path.join(args.out_dir, "aggregate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--evals", default=None, help="name=path,name=path")
    ap.add_argument("--adaptive_r1", default=None)
    ap.add_argument("--adaptive_r2", default=None)
    args = ap.parse_args()
    evals = {}
    if args.evals:
        for kv in args.evals.split(","):
            k, v = kv.split("=", 1)
            evals[k] = v
    args.evals = evals
    main(args)
