"""Pre-compute UMIS features for train/val samples (calibration data).

For each entry: runs Stage 0/1/2/3 feature extraction (without decision),
saves {input_id: features_dict} + labels, so the fusion head can be trained
and thresholds calibrated. Everything cached by file hash.

Usage:
  python collect_features.py --manifest bench/manifest.json \
      --out results/features.json --cache_dir results/cache \
      --device cuda [--limit 200]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "code"))

from umis.fusion import build_feature_vector  # noqa: E402
from umis.probes import p_content, p_file  # noqa: E402
from umis.stages.asr import ASR  # noqa: E402
from umis.stages.ocr import OCR  # noqa: E402



def run(args):
    from umis.io import load_manifest
    man = load_manifest(args.manifest)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    entries = [e for e in man["entries"]
               if e["split"] in args.splits.split(",")]
    if args.limit:
        entries = entries[:args.limit]

    ocr = OCR(cache_dir=os.path.join(args.cache_dir, "ocr"))
    # ASR runs on CUDA via ctranslate2. ctranslate2 is built against
    # CUDA 12 (needs libcublas.so.12/libcudart.so.12) while the box has
    # CUDA 13 + torch cu130 - the cu12 libs are pip-installed
    # (nvidia-cublas-cu12 / nvidia-cuda-runtime-cu12) and the scripts
    # prepend them to LD_LIBRARY_PATH. GPU int8 = ~1.4 s/clip vs 47 s/clip
    # on CPU float32 (fixed 2026-08-15).
    asr = ASR(cache_dir=os.path.join(args.cache_dir, "asr"),
              device=args.asr_device)
    if args.synth_weights:
        from umis.stages.synth_speech import SynthSpeechDetector
        if not os.path.isfile(args.synth_weights):
            raise FileNotFoundError(args.synth_weights)
    synth = SynthSpeechDetector(args.synth_weights) \
        if args.synth_weights else None
    encoder = None
    if not args.skip_encoder:
        from umis.probes.p_intent import SharedEncoder
        encoder = SharedEncoder(mode=args.encoder_mode,
                                model_path=args.encoder_path,
                                device=args.device,
                                cache_dir=os.path.join(args.cache_dir,
                                                       "emb"))

    benign_centroid = benign_inv_cov = None
    if args.centroid:
        import numpy as np
        with np.load(args.centroid) as centroid:
            benign_centroid = centroid["mean"]
            benign_inv_cov = centroid.get("inv_cov")
    feats = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            feats = json.load(f)

    for i, e in enumerate(entries):
        if e["id"] in feats and "error" not in feats[e["id"]]:
            continue
        image_path = audio_path = None
        if e["modality"] in ("image", "mixed"):
            image_path = e["source_files"][0]
        if e["modality"] in ("audio", "mixed"):
            audio_path = e["source_files"][-1]
        user_text = e.get("user_text") or "Describe the content."
        try:
            from umis.features import extract_features
            fv = extract_features(ocr, asr, encoder, image_path, audio_path,
                                  user_text, synth, benign_centroid, benign_inv_cov)
            feats[e["id"]] = {"features": fv, "label": int(
                e["kind"] == "attack"), "family": e["family"],
                "split": e["split"], "placement": e["placement"],
                "modality": e["modality"]}
        except Exception as ex:
            feats[e["id"]] = {"error": str(ex)}
        if i % 20 == 0:
            with open(args.out, "w") as f:
                json.dump(feats, f, indent=1)
            print(f"  {i + 1}/{len(entries)}")
    with open(args.out, "w") as f:
        json.dump(feats, f, indent=1)
    print(f"saved {len(feats)} entries -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--asr_device", default="cuda")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--synth_weights", default=None)
    ap.add_argument("--encoder_mode", default="clip_clap", choices=["omni", "clip_clap"])
    ap.add_argument("--encoder_path", default=None)
    ap.add_argument("--skip_encoder", action="store_true")
    ap.add_argument("--centroid", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(args)
