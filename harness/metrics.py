"""Shared metric computation for UMIS experiments.

Every metric in the paper must come from this module (consistency
invariant). Metrics:
  - ASR (attack success rate, with/without defense)
  - FPR / FNR / F1 / ROC-AUC / PR-AUC / ECE
  - per-probe precision/recall against probe_ground_truth
  - per-placement detection rate
  - localization: image bbox IoU; audio segment F1 + timestamp MAE
  - cost: latency, L2 escalation rate
  - statistical tests: McNemar
"""
import json
import math
import os

import numpy as np
from scipy import stats


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def f1(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1v = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1v


def roc_pr_auc(scores, labels):
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, \
        roc_curve
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "fpr": [], "tpr": [], "prec": [], "rec": []}
    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(rec, prec)
    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "prec": prec.tolist(), "rec": rec.tolist()}


def ece(scores, labels, n_bins=10):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(scores, bins[1:-1])
    ece_v = 0.0
    total = len(scores)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        acc = labels[m].mean()
        conf = scores[m].mean()
        ece_v += (len(scores[m]) / total) * abs(acc - conf)
    return float(ece_v)


def per_probe_metrics(entries, probe_outputs):
    """Precision/recall/F1 per probe against probe_ground_truth.

    probe_outputs: {input_id: {"p_file": bool, "p_content": bool,
                               "p_intent": bool}}  (triggered or not)
    """
    out = {}
    for probe in ("p_file", "p_content", "p_intent"):
        y_true, y_pred = [], []
        for e in entries:
            if e["kind"] != "attack" and e["kind"] != "benign":
                continue
            gt = e["probe_ground_truth"].get(probe)
            if gt is None:
                continue
            y_true.append(bool(gt))
            y_pred.append(bool(probe_outputs.get(e["id"], {}).get(
                probe, False)))
        tp, fp, fn, tn = confusion(y_true, y_pred)
        prec, rec, f1v = f1(tp, fp, fn)
        out[probe] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": prec, "recall": rec, "f1": f1v}
    return out


def placement_detection(entries, verdicts, block_verdicts=("BLOCK",)):
    """Detection rate per placement layer (attack entries blocked)."""
    out = {}
    for e in entries:
        if e["kind"] != "attack":
            continue
        pl = e["placement"]
        v = verdicts.get(e["id"], "")
        out.setdefault(pl, {"n": 0, "detected": 0})
        out[pl]["n"] += 1
        out[pl]["detected"] += int(v in block_verdicts)
    for pl in out:
        out[pl]["rate"] = out[pl]["detected"] / max(out[pl]["n"], 1)
    return out


def bbox_iou(a, b):
    if a is None or b is None:
        return None
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def audio_segment_f1(pred_ts, gt_ts, tol=0.3):
    """F1 of predicted vs ground-truth speech intervals (with tolerance)."""
    if not pred_ts or not gt_ts:
        return 0.0
    if isinstance(gt_ts, list) and gt_ts and isinstance(gt_ts[0], list):
        gt_segs = gt_ts
    else:
        gt_segs = [gt_ts]
    hit = 0.0
    for p0, p1 in pred_ts:
        for g0, g1 in gt_segs:
            inter = max(0, min(p1, g1 + tol) - max(p0, g0 - tol))
            if inter > 0:
                hit += inter / max(p1 - p0, 1e-6)
                break
    prec = hit / max(len(pred_ts), 1)
    rec = min(1.0, hit / max(len(gt_segs), 1))
    f1v = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # timestamp MAE: distance of closest pred start to gt start
    mae = None
    if pred_ts and gt_segs:
        mae = float(np.mean([min(abs(p0 - g0) for p0, _ in pred_ts)
                             for g0, _ in gt_segs]))
    return f1v, mae


def summary_from_eval(eval_json_path, manifest_path, defended=False):
    """Aggregate an eval result file against the manifest.

    Returns dict with headline metrics + per-family/per-layer tables.
    """
    with open(eval_json_path) as f:
        ev = json.load(f)
    with open(manifest_path) as f:
        man = json.load(f)
    entries = {e["id"]: e for e in man["entries"]}

    atk_ids = [i for i, r in ev.items() if entries.get(i, {}).get("kind")
               == "attack" and "error" not in r]
    ben_ids = [i for i, r in ev.items() if entries.get(i, {}).get("kind")
               == "benign" and "error" not in r]

    asr = np.mean([ev[i]["success"] for i in atk_ids]) if atk_ids else None
    # benign "positive" = contains any attack trigger
    fpr = np.mean([ev[i]["success"] for i in ben_ids]) if ben_ids else None

    per_family = {}
    for i, r in ev.items():
        e = entries.get(i)
        if not e or "error" in r:
            continue
        fam = e["family"]
        per_family.setdefault(fam, {"n": 0, "pos": 0, "kind": e["kind"]})
        per_family[fam]["n"] += 1
        per_family[fam]["pos"] += int(r["success"])

    per_layer = {}
    for i in atk_ids:
        e = entries[i]
        per_layer.setdefault(e["layer"], {"n": 0, "detected": 0})
        per_layer[e["layer"]]["n"] += 1
        if defended:
            v = ev[i].get("verdict")
            per_layer[e["layer"]]["detected"] += int(v == "BLOCK")
        else:
            per_layer[e["layer"]]["detected"] += int(not ev[i]["success"])

    return {
        "n_attack": len(atk_ids), "n_benign": len(ben_ids),
        "asr": float(asr) if asr is not None else None,
        "fpr": float(fpr) if fpr is not None else None,
        "per_family": {k: {**v, "rate": v["pos"] / max(v["n"], 1)}
                       for k, v in per_family.items()},
        "per_layer": {k: {**v, "rate": v["detected"] / max(v["n"], 1)}
                      for k, v in per_layer.items()},
    }


def mcnemar(a_success, b_success):
    """Paired McNemar test on two defenses over the same sample set."""
    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)
    b_ = (~a) & b
    c_ = a & (~b)
    b_cnt, c_cnt = int(b_.sum()), int(c_.sum())
    if b_cnt + c_cnt == 0:
        return 1.0
    p = stats.binomtest(b_cnt, b_cnt + c_cnt, 0.5).pvalue
    return float(p)
