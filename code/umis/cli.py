"""Local inspection and single-input detection entry point."""
import argparse
import json
import sys
from pathlib import Path
from . import __version__

def main(argv=None):
    parser = argparse.ArgumentParser(prog="umis", description="UMIS multimodal prompt-injection inspection")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="Inspect file and signal features without model downloads")
    probe.add_argument("file", type=Path)
    detect = sub.add_parser("detect", help="Run OCR/ASR + UMIS on one image/audio input")
    detect.add_argument("--image", type=Path)
    detect.add_argument("--audio", type=Path)
    detect.add_argument("--user-text", default="Describe the content.")
    detect.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    detect.add_argument("--encoder", choices=["none", "clip_clap", "omni"], default="none")
    detect.add_argument("--head", type=Path)
    detect.add_argument("--thresholds", type=Path)
    detect.add_argument("--centroid", type=Path)
    detect.add_argument("--judge-url")
    detect.add_argument("--judge-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    detect.add_argument("--cache-dir", type=Path, default=Path("results/cache"))
    detect.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            from dataclasses import asdict
            from .probes.p_file import probe_file
            from .probes.p_content import probe_content
            report = {"file": str(args.file), "p_file": asdict(probe_file(str(args.file))),
                      "p_content": probe_content(str(args.file))}
        else:
            if not args.image and not args.audio:
                parser.error("detect requires --image or --audio")
            from .pipeline import UMIS
            from .fusion import FusionHead
            from .stages.ocr import OCR
            from .stages.asr import ASR
            ocr = OCR(cache_dir=str(args.cache_dir / "ocr")) if args.image else None
            asr = ASR(device=args.device, cache_dir=str(args.cache_dir / "asr")) if args.audio else None
            encoder = None
            if args.encoder != "none":
                from .probes.p_intent import SharedEncoder
                encoder = SharedEncoder(mode=args.encoder, device=args.device,
                                        cache_dir=str(args.cache_dir / "emb"))
            judge = None
            if args.judge_url:
                from .judge import TextJudge
                judge = TextJudge(args.judge_url, args.judge_model)
            thresholds = json.loads(args.thresholds.read_text(encoding="utf-8")) if args.thresholds else {"t1": 0.3, "t2": 0.75}
            mean = inv_cov = None
            if args.centroid:
                import numpy as np
                with np.load(args.centroid) as data:
                    mean, inv_cov = data["mean"], data.get("inv_cov")
            detector = UMIS(ocr, asr, encoder=encoder,
                            fusion=FusionHead.load(args.head) if args.head else None,
                            t1=thresholds["t1"], t2=thresholds["t2"], judge=judge,
                            benign_centroid=mean, benign_inv_cov=inv_cov)
            report = detector.process("cli", image_path=str(args.image) if args.image else None,
                                      audio_path=str(args.audio) if args.audio else None,
                                      user_text=args.user_text)
            report["configuration"] = {"encoder": args.encoder, "calibrated_head": bool(args.head)}
            if not args.head:
                print("Using heuristic scores; train a head for calibrated evaluation.", file=sys.stderr)
        result = json.dumps(report, indent=2, ensure_ascii=False)
        if getattr(args, "out", None):
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(result + "\n", encoding="utf-8")
        print(result)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"umis: {exc}", file=sys.stderr)
        return 2
