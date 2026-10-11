"""Phase B driver: no-defense ASR measurement on the target models.

Usage:
  python eval_target.py --target vl --manifest bench/out/manifest.json \
      --out results/no_defense_vl.json [--base_url http://127.0.0.1:8000/v1]
  python eval_target.py --target omni --manifest bench/out/manifest.json \
      --out results/no_defense_omni.json [--model_path Qwen/Qwen2.5-Omni-7B]
  python eval_target.py --target vl --smoke --media_dir ...  (10-sample test)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
from harness.targets import (OmniTarget, Qwen2AudioTarget, VLTarget,
                             run_target_evals)  # noqa


def smoke_test(args):
    """10 samples per modality: verify models + pipeline end to end."""
    if not args.media_dir:
        raise ValueError("--smoke requires --media_dir")
    os.makedirs(args.media_dir, exist_ok=True)
    man = {"entries": []}
    img = os.path.join(args.media_dir, "smoke.jpg")
    from PIL import Image, ImageDraw
    Image.new("RGB", (256, 256), (200, 210, 220)).save(img)
    man["entries"].append({
        "id": "smoke_img", "modality": "image", "kind": "attack",
        "family": "smoke", "source_files": [img],
        "ground_truth": {"trigger": "PWNED"}})
    import numpy as np
    import soundfile as sf
    aud = os.path.join(args.media_dir, "smoke.wav")
    sf.write(aud, 0.01 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000),
             16000)
    man["entries"].append({
        "id": "smoke_aud", "modality": "audio", "kind": "attack",
        "family": "smoke", "source_files": [aud],
        "ground_truth": {"trigger": "PWNED"}})
    with open(os.path.join(args.media_dir, "smoke_manifest.json"), "w") as f:
        json.dump(man, f)
    target = _make_target(args)
    out = run_target_evals(target, man["entries"],
                           os.path.join(args.media_dir, "smoke_eval.json"))
    print(json.dumps(out, indent=1))
    print("SMOKE OK" if all("error" not in v for v in out.values())
          else "SMOKE FAILED")


def _make_target(args):
    if args.target == "vl":
        return VLTarget(base_url=args.base_url, model_name=args.model_name or "Qwen/Qwen2.5-VL-7B-Instruct")
    if args.target == "qwen2audio":
        return Qwen2AudioTarget(base_url=args.base_url,
                                model_name=args.model_name or "Qwen/Qwen2-Audio-7B-Instruct")
    return OmniTarget(model_path=args.model_path)


def main(args):
    from umis.io import load_manifest
    man = load_manifest(args.manifest)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    entries = man["entries"]
    if args.split:
        entries = [e for e in entries if e["split"] in args.split.split(",")]
    if args.families:
        fams = set(args.families.split(","))
        entries = [e for e in entries if e["family"] in fams]
    if args.modality:
        entries = [e for e in entries if e["modality"] == args.modality]
    if args.max_samples:
        entries = entries[:args.max_samples]
    target = _make_target(args)
    print(f"evaluating {len(entries)} samples with target={args.target}")
    run_target_evals(target, entries, args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                choices=["vl", "omni", "qwen2audio"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model_name",
                    default=None)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--media_dir", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--families", default=None)
    ap.add_argument("--modality", default=None)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        smoke_test(args)
    else:
        main(args)
