"""UMI-Bench construction: orchestrates attack generators + benign data,
produces the benchmark manifest with train/val/test splits per family.

Usage:
  python build_bench.py --coco_dir ... --ls_dir ... --out_dir ... \
      --n 100 --seed 0
"""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.common import read_json, write_json  # noqa: E402
from attacks.image import signal as img_signal  # noqa: E402
from attacks.image import structural as img_struct  # noqa: E402
from attacks.image import typographic  # noqa: E402
from attacks.audio import audio_attacks  # noqa: E402


def load_coco_subset(coco_dir, out_dir, n, seed=0):
    """Extract n COCO val images to a working dir (natural images)."""
    rng = random.Random(seed)
    files = sorted(f for f in os.listdir(coco_dir)
                   if f.lower().endswith(".jpg"))
    chosen = rng.sample(files, min(n, len(files)))
    dst = os.path.join(out_dir, "coco_subset")
    os.makedirs(dst, exist_ok=True)
    for f in chosen:
        shutil.copy(os.path.join(coco_dir, f), os.path.join(dst, f))
    return sorted(os.path.join(dst, f) for f in chosen)


def load_librispeech(ls_dir, out_dir, n, seed=0):
    """Extract n LibriSpeech test-clean flacs to a working dir."""
    rng = random.Random(seed)
    files = []
    for root, _, fs in os.walk(ls_dir):
        for f in fs:
            if f.endswith(".flac"):
                files.append(os.path.join(root, f))
    files.sort()
    chosen = rng.sample(files, min(n, len(files)))
    dst = os.path.join(out_dir, "ls_subset")
    os.makedirs(dst, exist_ok=True)
    for i, f in enumerate(chosen):
        shutil.copy(f, os.path.join(dst, f"ls_{i:04d}.flac"))
    return sorted(os.path.join(dst, f) for f in os.listdir(dst))


def split_entries(entries, seed=0, ratios=(0.5, 0.2, 0.3)):
    """Per-family/kind sample split; base media can be shared across splits."""
    rng = random.Random(seed)
    by_family = {}
    for e in entries:
        by_family.setdefault((e["family"], e["kind"]), []).append(e)
    for key in by_family:
        rng.shuffle(by_family[key])
        fam = by_family[key]
        n1 = int(len(fam) * ratios[0])
        n2 = int(len(fam) * ratios[1])
        for i, e in enumerate(fam):
            e["split"] = "train" if i < n1 else \
                ("val" if i < n1 + n2 else "test")
    return entries


def build(args):
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    media = os.path.join(out, "media")
    n = args.n
    if n < 1:
        raise ValueError("--n must be positive")
    if not any(Path(args.coco_dir).glob("*.jpg")):
        raise ValueError("COCO directory must contain JPG files")
    if not any(Path(args.ls_dir).rglob("*.flac")):
        raise ValueError("LibriSpeech directory must contain FLAC files")

    # ---- benign base media -------------------------------------------------
    coco = load_coco_subset(args.coco_dir, out, n + 20, args.seed)
    ls = load_librispeech(args.ls_dir, out, n + 20, args.seed)
    coco_dir = os.path.join(out, "coco_subset")
    ls_dir = os.path.join(out, "ls_subset")

    all_entries = []

    def add(entries):
        all_entries.extend(entries)
        return len(entries)

    # ---- image attacks ----------------------------------------------------
    img_dir = os.path.join(media, "image")
    print("== typographic (FigStep-style) ==")
    add(typographic.generate(coco, os.path.join(img_dir, "typographic"),
                             "typographic", n, seed=args.seed))
    print("== typo obfuscated (AgentTypo-style) ==")
    add(typographic.generate(coco, os.path.join(img_dir, "typo_obf"),
                             "typo_obfuscated", n, seed=args.seed,
                             obfuscation=0.4))
    print("== typographic benign twins ==")
    add(typographic.generate(coco, os.path.join(img_dir, "typographic_benign"),
                             "typographic_benign", n, benign=True,
                             seed=args.seed))
    print("== qr injection ==")
    add(img_signal.gen_qr(coco_dir, os.path.join(img_dir, "qr"), n,
                          seed=args.seed))
    add(img_signal.gen_qr(coco_dir, os.path.join(img_dir, "qr_benign"), n,
                          benign=True, seed=args.seed))
    print("== hidden low-contrast text ==")
    add(img_signal.gen_white_hidden(coco_dir, os.path.join(img_dir, "hidden"),
                                    n, seed=args.seed))
    add(img_signal.gen_white_hidden(coco_dir, os.path.join(img_dir, "hidden_benign"),
                                    n, benign=True, seed=args.seed))
    print("== exif payload ==")
    add(img_struct.gen_exif(coco_dir, os.path.join(img_dir, "exif"), n,
                            seed=args.seed))
    add(img_struct.gen_exif(coco_dir, os.path.join(img_dir, "exif_benign"), n,
                            benign=True, seed=args.seed))
    print("== png text chunk ==")
    add(img_struct.gen_png_chunk(coco_dir, os.path.join(img_dir, "pngchunk"), n,
                                 seed=args.seed))
    add(img_struct.gen_png_chunk(coco_dir, os.path.join(img_dir, "pngchunk_benign"),
                                 n, benign=True, seed=args.seed))
    print("== noncanonical chunks ==")
    add(img_struct.gen_noncanonical(coco_dir, os.path.join(img_dir, "noncanon"),
                                    n, seed=args.seed))
    print("== appended bytes ==")
    add(img_struct.gen_appended(coco_dir, os.path.join(img_dir, "append"), n,
                                seed=args.seed))
    add(img_struct.gen_appended(coco_dir, os.path.join(img_dir, "append_benign"),
                                n, benign=True, seed=args.seed))
    print("== lsb stego ==")
    add(img_signal.gen_lsb_stego(coco_dir, os.path.join(img_dir, "lsb"), n,
                                 seed=args.seed))
    add(img_signal.gen_lsb_stego(coco_dir, os.path.join(img_dir, "lsb_benign"), n,
                                 benign=True, seed=args.seed))
    print("== dct embed ==")
    add(img_signal.gen_dct_embed(coco_dir, os.path.join(img_dir, "dct"), n,
                                 seed=args.seed))
    add(img_signal.gen_dct_embed(coco_dir, os.path.join(img_dir, "dct_benign"), n,
                                 benign=True, seed=args.seed))
    if not args.skip_adv:
        print("== adversarial semantic-only ==")
        add(img_signal.gen_adversarial(coco_dir, os.path.join(img_dir, "adv"), n,
                                       seed=args.seed))

    # ---- audio attacks ----------------------------------------------------
    aud_dir = os.path.join(media, "audio")
    print("== tts injection ==")
    add(audio_attacks.gen_tts(ls_dir, os.path.join(aud_dir, "tts"), n,
                              seed=args.seed))
    add(audio_attacks.gen_tts(ls_dir, os.path.join(aud_dir, "tts_benign"), n,
                              benign=True, seed=args.seed))
    print("== piggyback ==")
    add(audio_attacks.gen_piggyback(ls_dir, os.path.join(aud_dir, "piggy"), n,
                                    seed=args.seed))
    add(audio_attacks.gen_piggyback(ls_dir, os.path.join(aud_dir, "piggy_benign"),
                                    n, benign=True, seed=args.seed))
    print("== lsb stego (audio) ==")
    add(audio_attacks.gen_lsb_stego(ls_dir, os.path.join(aud_dir, "lsb"), n,
                                    seed=args.seed))
    add(audio_attacks.gen_lsb_stego(ls_dir, os.path.join(aud_dir, "lsb_benign"),
                                    n, benign=True, seed=args.seed))
    print("== id3 payload ==")
    add(audio_attacks.gen_id3(ls_dir, os.path.join(aud_dir, "id3"), n,
                              seed=args.seed))
    add(audio_attacks.gen_id3(ls_dir, os.path.join(aud_dir, "id3_benign"), n,
                              benign=True, seed=args.seed))
    print("== echo embed ==")
    add(audio_attacks.gen_echo(ls_dir, os.path.join(aud_dir, "echo"), n,
                               seed=args.seed))
    add(audio_attacks.gen_echo(ls_dir, os.path.join(aud_dir, "echo_benign"), n,
                               benign=True, seed=args.seed))
    print("== semantic framing ==")
    add(audio_attacks.gen_framing(ls_dir, os.path.join(aud_dir, "framing"), n,
                                  seed=args.seed))
    add(audio_attacks.gen_framing(ls_dir, os.path.join(aud_dir, "framing_benign"),
                                  n, benign=True, seed=args.seed))

    # ---- mixed (CrossInject-style): image + audio in one turn -------------
    print("== mixed image+audio ==")
    mix_dir = os.path.join(media, "mixed")
    typo_entries = typographic.generate(
        coco, os.path.join(mix_dir, "img_part"), "mixed_cross", n,
        seed=args.seed + 1)
    tts_entries = audio_attacks.gen_tts(ls_dir, os.path.join(mix_dir, "aud_part"),
                                        n, seed=args.seed + 2)
    for te, ae in zip(typo_entries, tts_entries):
        e = dict(te)
        e["id"] = f"mixed_{te['id'].split('_')[-1]}"
        e["family"] = "mixed_cross"
        e["modality"] = "mixed"
        e["layer"] = "mixed"
        e["placement"] = "mixed"
        e["source_files"] = te["source_files"] + ae["source_files"]
        e["ground_truth"] = {
            "instruction_text": te["ground_truth"]["instruction_text"],
            "trigger": te["ground_truth"]["trigger"],
            "image_bbox": te["ground_truth"]["bbox"],
            "audio_timestamps": ae["ground_truth"]["timestamps"],
        }
        e["metadata"] = {"image_part": te["source_files"][0],
                         "audio_part": ae["source_files"][0]}
        add([e])

    # ---- benign natural media (no injection at all) -----------------------
    print("== benign natural media ==")
    bn_img = os.path.join(media, "benign_natural_image")
    os.makedirs(bn_img, exist_ok=True)
    for i, p in enumerate(coco[:n]):
        dst = os.path.join(bn_img, f"natural_{i:04d}.jpg")
        shutil.copy(p, dst)
        add([{
            "id": f"natural_img_{i:04d}", "modality": "image",
            "family": "natural_image", "layer": "benign",
            "placement": "benign", "kind": "benign",
            "probe_ground_truth": {"p_file": False, "p_content": False,
                                   "p_intent": False},
            "ground_truth": {"instruction_text": "", "bbox": None,
                             "trigger": None},
            "source_files": [dst], "metadata": {},
        }])
    bn_aud = os.path.join(media, "benign_natural_audio")
    os.makedirs(bn_aud, exist_ok=True)
    for i, p in enumerate(ls[:n]):
        dst = os.path.join(bn_aud, f"natural_{i:04d}.flac")
        shutil.copy(p, dst)
        add([{
            "id": f"natural_aud_{i:04d}", "modality": "audio",
            "family": "natural_audio", "layer": "benign",
            "placement": "benign", "kind": "benign",
            "probe_ground_truth": {"p_file": False, "p_content": False,
                                   "p_intent": False},
            "ground_truth": {"instruction_text": "", "timestamps": None,
                             "trigger": None},
            "source_files": [dst], "metadata": {},
        }])

    # ---- splits + manifest -------------------------------------------------
    # Global id uniqueness guard: run_target_evals deduplicates results by
    # id, so a duplicate id would silently drop one of the two entries
    # (fixed 2026-08-15 after attack/benign twins collided on ids and 1200
    # entries were skipped in Phase B).
    from collections import Counter
    id_counts = Counter(e["id"] for e in all_entries)
    dups = [i for i, c in id_counts.items() if c > 1]
    if dups:
        raise ValueError(f"{len(dups)} duplicate entry ids, e.g. {dups[:10]}")
    all_entries = split_entries(all_entries, seed=args.seed)
    for e in all_entries:
        e["user_text"] = e.get("user_text") or "Describe the content."
        e["source_files"] = [Path(os.path.relpath(os.path.abspath(p), os.path.abspath(out))).as_posix() for p in e["source_files"]]
        for key in ("image_part", "audio_part"):
            if key in e.get("metadata", {}):
                e["metadata"][key] = Path(os.path.relpath(os.path.abspath(e["metadata"][key]), os.path.abspath(out))).as_posix()

    manifest = {
        "name": "UMI-Bench",
        "version": "1.0",
        "n_samples": len(all_entries),
        "splits": {"train": {}, "val": {}, "test": {}},
        "entries": all_entries,
    }
    for split in ("train", "val", "test"):
        for e in all_entries:
            if e["split"] == split:
                key = f"{e['family']}"
                manifest["splits"][split].setdefault(key, 0)
                manifest["splits"][split][key] += 1
    write_json(os.path.join(out, "manifest.json"), manifest)
    print(f"benchmark built: {len(all_entries)} samples -> {out}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", required=True)
    ap.add_argument("--ls_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_adv", action="store_true",
                    help="skip adversarial (GPU) family")
    args = ap.parse_args()
    build(args)
