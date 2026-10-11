"""Compute the benign-train embedding centroid (ZEDD-style drift reference).

Usage:
  python compute_centroid.py --manifest bench/manifest.json \
      --out results/centroid.npz --cache_dir results/cache \
      --encoder_mode omni [--limit 100]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "code"))

from umis.probes.p_intent import SharedEncoder  # noqa: E402


def run(args):
    from umis.io import load_manifest
    man = load_manifest(args.manifest)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    entries = [e for e in man["entries"]
               if e["split"] == "train" and e["kind"] == "benign" and e["modality"] == "image"]
    if args.limit:
        entries = entries[:args.limit]
    encoder = SharedEncoder(mode=args.encoder_mode,
                            model_path=args.encoder_path,
                            device=args.device,
                            cache_dir=os.path.join(args.cache_dir, "emb"))
    embs = []
    for e in entries:
        p = e["source_files"][0]
        try:
            if e["modality"] == "audio":
                embs.append(encoder.embed_audio(p))
            else:
                embs.append(encoder.embed_image(p))
        except Exception as ex:
            print(f"  skip {e['id']}: {ex}")
    if len(embs) < 2:
        raise ValueError("At least two benign training images are required for image drift")
    X = np.stack(embs)
    mean = X.mean(axis=0)
    try:
        inv_cov = np.linalg.inv(np.cov(X.T) +
                                np.eye(X.shape[1]) * 1e-3)
    except np.linalg.LinAlgError:
        inv_cov = None
    arrays = {"mean": mean, "n": len(embs), "encoder_mode": args.encoder_mode}
    if inv_cov is not None:
        arrays["inv_cov"] = inv_cov
    np.savez(args.out, **arrays)
    print(f"centroid from {len(embs)} benign embeddings -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encoder_mode", default="clip_clap", choices=["omni", "clip_clap"])
    ap.add_argument("--encoder_path", default=None)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    run(args)
