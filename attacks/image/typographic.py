"""Typographic injection generators (FigStep-style and AgentTypo-style).

Renders injection instructions as visible text on top of natural images.
Two families:
  - typographic:   clean readable text in a solid box (content placement)
  - typo_obf:      leetspeak/spacing obfuscation of the same text (content placement)
The same pipeline with benign text produces the nuisance-matched benign twins.
"""
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import INSTRUCTIONS, BENIGN_TEXTS, find_font  # noqa: E402

LEET = {
    "a": ["4", "@"], "b": ["8"], "e": ["3"], "g": ["6", "9"], "i": ["1", "!"],
    "l": ["1"], "o": ["0"], "s": ["5", "$"], "t": ["7"], "z": ["2"],
}


def obfuscate(text, level=0.35, rng=None):
    """Leetspeak-style obfuscation. level = fraction of substitutable chars."""
    rng = rng or random
    out = []
    for ch in text.lower():
        if ch in LEET and rng.random() < level:
            out.append(rng.choice(LEET[ch]))
        elif ch == " " and rng.random() < 0.15:
            out.append("  ")
        else:
            out.append(ch)
    return "".join(out)


def render_text_block(image, text, box_xyxy, font_path, font_size,
                      text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    """Render wrapped text inside a solid box on `image` IN PLACE.

    Modifies `image` (which must be an RGB PIL.Image). Returns bbox of the
    rendered region for ground-truth localization.
    """
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box_xyxy
    draw.rectangle([x0, y0, x1, y1], fill=bg_color)
    font = ImageFont.truetype(font_path, font_size)
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        w_ = draw.textlength(trial, font=font)
        if w_ > (x1 - x0 - 20) and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    line_h = int(font_size * 1.35)
    y = y0 + 10
    for ln in lines:
        draw.text((x0 + 10, y), ln, fill=text_color, font=font)
        y += line_h
    # drawn text bbox = the box (conservative, well-defined ground truth)
    return [x0, y0, x1, min(y1, y + 10)]


def generate(base_image_paths, out_dir, family, n_samples, benign=False,
             seed=0, obfuscation=0.0, box_ratio=(0.06, 0.3, 0.94, 0.85),
             font_size=26):
    """Generate n_samples typographic samples.

    Returns list of manifest entries.
    box_ratio = (x0r, y0r, x1r, y1r) relative box location.
    """
    rng = random.Random(seed)
    font_path = find_font()
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i in range(n_samples):
        img_path = base_image_paths[i % len(base_image_paths)]
        base = Image.open(img_path).convert("RGB")
        W, H = base.size
        bx0, by0 = int(W * box_ratio[0]), int(H * box_ratio[1])
        bx1, by1 = int(W * box_ratio[2]), int(H * box_ratio[3])
        if benign:
            text = rng.choice(BENIGN_TEXTS)
        else:
            text = INSTRUCTIONS[i % len(INSTRUCTIONS)]["text"]
            if obfuscation > 0:
                text = obfuscate(text, level=obfuscation, rng=rng)
        bbox = render_text_block(base, text, (bx0, by0, bx1, by1),
                                 font_path, font_size)
        out_path = os.path.join(
            out_dir, f"{family}_{i:04d}.png")
        base.save(out_path)
        entries.append({
            "id": f"{family}_{i:04d}",
            "modality": "image",
            "family": family,
            "layer": "content",
            "placement": "content",
            "kind": "benign" if benign else "attack",
            "probe_ground_truth": {"p_file": False, "p_content": True,
                                   "p_intent": False if benign else True},
            "ground_truth": {
                "instruction_text": text,
                "bbox": bbox,
                "trigger": None if benign else
                INSTRUCTIONS[i % len(INSTRUCTIONS)]["trigger"],
            },
            "source_files": [out_path],
            "metadata": {"base_image": img_path, "obfuscation": obfuscation},
        })
    return entries


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="directory with base images")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benign", action="store_true")
    ap.add_argument("--family", default="typographic")
    ap.add_argument("--obfuscation", type=float, default=0.0)
    args = ap.parse_args()
    bases = sorted(
        os.path.join(args.base_dir, f) for f in os.listdir(args.base_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")))
    entries = generate(bases, args.out_dir, args.family, args.n,
                       benign=args.benign, seed=args.seed,
                       obfuscation=args.obfuscation)
    import json
    with open(os.path.join(args.out_dir, f"{args.family}.json"), "w") as f:
        json.dump(entries, f, indent=1)
    print(f"generated {len(entries)} samples in {args.out_dir}")
