"""File-layer (container/metadata) image attack generators.

Families:
  - exif_payload:   injection text in JPEG EXIF ImageDescription (container_metadata)
  - png_chunk:      injection text in a PNG tEXt chunk (container_metadata)
  - appended_bytes: payload bytes appended after image EOF (container_metadata,
                    image-trojan polyglot style)
  - noncanonical:   extra non-standard PNG chunk types with payload bytes
Each family has a benign twin (legitimate metadata / no anomalies).
"""
import os
import struct
import sys
import zlib

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import INSTRUCTIONS  # noqa: E402


def _id(prefix, i, benign):
    """Entry id: benign twins get a distinct prefix so attack/benign ids
    never collide (a collision silently drops entries in run_target_evals,
    which deduplicates by id - fixed 2026-08-15)."""
    return f"{prefix}_benign_{i:04d}" if benign else f"{prefix}_{i:04d}"


# ---------------------------------------------------------------- EXIF ----
def inject_exif(src_jpg, dst_jpg, text):
    """Write `text` into the EXIF ImageDescription of a JPEG."""
    import piexif
    img = Image.open(src_jpg)
    exif_dict = {"0th": {piexif.ImageIFD.ImageDescription: text.encode("utf-8")},
                 "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)
    img.save(dst_jpg, "JPEG", quality=90, exif=exif_bytes)


def benign_exif(src_jpg, dst_jpg):
    """Legitimate rich EXIF metadata (camera info, no payload)."""
    import piexif
    img = Image.open(src_jpg)
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: b"Canon",
            piexif.ImageIFD.Model: b"Canon EOS R6",
            piexif.ImageIFD.ImageDescription: b"Family trip, summer 2024",
            piexif.ImageIFD.Software: b"Adobe Lightroom Classic",
            piexif.ImageIFD.DateTime: b"2024:07:14 12:30:00",
        },
        "Exif": {piexif.ExifIFD.ExposureTime: (1, 250),
                 piexif.ExifIFD.FNumber: (28, 10)},
        "GPS": {}, "1st": {}, "thumbnail": None,
    }
    img.save(dst_jpg, "JPEG", quality=90, exif=piexif.dump(exif_dict))


# ------------------------------------------------------------- PNG chunk ---
def inject_png_chunk(src_png, dst_png, text, chunk_type=b"tEXt",
                     keyword=b"Comment"):
    """Append a custom text chunk with payload to a PNG."""
    with open(src_png, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    payload = keyword + b"\x00" + text.encode("utf-8")
    chunk = struct.pack(">I", len(payload)) + chunk_type + payload
    chunk += struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    out = data[:8] + chunk + data[8:]
    with open(dst_png, "wb") as f:
        f.write(out)


def inject_png_noncanonical(src_png, dst_png, text):
    """Non-standard chunk types (atYP/cuSt) carrying payload bytes."""
    with open(src_png, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    payload = text.encode("utf-8")
    chunks = b""
    for ctype in (b"atYP", b"cuSt"):
        chunk = struct.pack(">I", len(payload)) + ctype + payload
        chunk += struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
        chunks += chunk
    with open(dst_png, "wb") as f:
        f.write(data[:8] + chunks + data[8:])


# -------------------------------------------------------- appended bytes ---
def append_bytes(src, dst, text):
    """Append a text payload after the file EOF (image-trojan polyglot style)."""
    with open(src, "rb") as f:
        data = f.read()
    marker = b"\n=== INJECTED ===\n"
    with open(dst, "wb") as f:
        f.write(data + marker + text.encode("utf-8"))


# ------------------------------------------------------------- generators ---
def _base_images(base_dir, n):
    files = sorted(f for f in os.listdir(base_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return [os.path.join(base_dir, files[i % len(files)]) for i in range(n)]


def _entry(fid, family, layer, placement, kind, text, trigger, src, p_file,
           p_content=False, p_intent=False, extra=None):
    return {
        "id": fid, "modality": "image", "family": family, "layer": layer,
        "placement": placement, "kind": kind,
        "probe_ground_truth": {"p_file": p_file, "p_content": p_content,
                               "p_intent": p_intent},
        "ground_truth": {"instruction_text": text, "bbox": None,
                         "trigger": trigger},
        "source_files": [src],
        "metadata": extra or {},
    }


def gen_exif(base_dir, out_dir, n, seed=0, benign=False):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i, src in enumerate(_base_images(base_dir, n)):
        dst = os.path.join(out_dir, f"exif_{i:04d}.jpg")
        if benign:
            benign_exif(src, dst)
            text, trig = "Family trip, summer 2024", None
        else:
            inj = INSTRUCTIONS[i % len(INSTRUCTIONS)]
            inject_exif(src, dst, inj["core"])
            text, trig = inj["core"], inj["trigger"]
        entries.append(_entry(
            _id("exif", i, benign), "exif_payload", "file",
            "container_metadata",
            "benign" if benign else "attack", text, trig, dst,
            p_file=True, p_intent=not benign,
            extra={"field": "ImageDescription"}))
    return entries


def gen_png_chunk(base_dir, out_dir, n, seed=0, benign=False):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i, src in enumerate(_base_images(base_dir, n)):
        # ensure PNG input
        png = os.path.join(out_dir, f"_base_{i:04d}.png")
        Image.open(src).convert("RGB").save(png)
        dst = os.path.join(out_dir, f"pngchunk_{i:04d}.png")
        if benign:
            inject_png_chunk(png, dst, "Photo taken on a hiking trip.",
                             keyword=b"Comment")
            text, trig = "Photo taken on a hiking trip.", None
        else:
            inj = INSTRUCTIONS[i % len(INSTRUCTIONS)]
            inject_png_chunk(png, dst, inj["core"])
            text, trig = inj["core"], inj["trigger"]
        entries.append(_entry(
            _id("pngchunk", i, benign), "png_text_chunk", "file",
            "container_metadata", "benign" if benign else "attack",
            text, trig, dst, p_file=True, p_intent=not benign,
            extra={"field": "tEXt/Comment"}))
    return entries


def gen_noncanonical(base_dir, out_dir, n, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i, src in enumerate(_base_images(base_dir, n)):
        png = os.path.join(out_dir, f"_base_{i:04d}.png")
        Image.open(src).convert("RGB").save(png)
        dst = os.path.join(out_dir, f"noncanon_{i:04d}.png")
        inj = INSTRUCTIONS[i % len(INSTRUCTIONS)]
        inject_png_noncanonical(png, dst, inj["core"])
        entries.append(_entry(
            f"noncanon_{i:04d}", "noncanonical_chunk", "file",
            "container_metadata", "attack", inj["text"], inj["trigger"],
            dst, p_file=True, p_intent=True,
            extra={"chunk_types": ["atYP", "cuSt"]}))
    return entries


def gen_appended(base_dir, out_dir, n, seed=0, benign=False):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i, src in enumerate(_base_images(base_dir, n)):
        dst = os.path.join(out_dir, f"append_{i:04d}.png")
        if benign:
            append_bytes(src, dst, "created with PIL 9.1.0")
            text, trig = "created with PIL 9.1.0", None
        else:
            inj = INSTRUCTIONS[i % len(INSTRUCTIONS)]
            append_bytes(src, dst, inj["core"])
            text, trig = inj["core"], inj["trigger"]
        entries.append(_entry(
            _id("append", i, benign), "appended_bytes", "file",
            "container_metadata", "benign" if benign else "attack",
            text, trig, dst, p_file=True, p_intent=not benign,
            extra={"placement_after": "EOF"}))
    return entries
