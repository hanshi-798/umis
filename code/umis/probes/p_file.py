"""Probe P-file (Stage 1): unified container/metadata parser.

Mirrors the file/container-parsing link of the attack chain. Walks the
chunk/box/marker structure of PNG/JPEG/WAV/MP3 (+ RIFF-family), parses
metadata tags (EXIF/XMP/ID3/APE/tEXt), scans trailing/appended bytes after
EOF, and emits structural anomaly features + evidence entries.

Pure Python + stdlib: no heavy dependencies.
"""
import math
import os
import re
import struct
import zlib
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    hist = [0] * 256
    for b in data:
        hist[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in hist if c)


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    n_print = sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13))
    return n_print / len(data)


def _as_text(data: bytes) -> str:
    """Best-effort decode of arbitrary bytes to text."""
    for enc in ("utf-8", "latin-1"):
        try:
            s = data.decode(enc)
            if s.isprintable() or any(32 <= ord(c) < 127 for c in s):
                return s
        except (UnicodeDecodeError, ValueError):
            continue
    return ""


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CANONICAL_PNG = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"cHRM",
                 b"gAMA", b"iCCP", b"sBIT", b"sRGB", b"tEXt", b"zTXt",
                 b"iTXt", b"bKGD", b"hIST", b"pHYs", b"sPLT", b"tIME",
                 b"eXIf", b"vpAg"}


def parse_png(data: bytes):
    ev = {"format": "png", "magic_valid": data[:8] == PNG_MAGIC}
    if not ev["magic_valid"]:
        return ev, "bad PNG magic"
    pos = 8
    n_critical = 0
    noncanonical = []
    text_payloads = []
    crc_mismatch = False
    structure_ok = True
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        if pos + 8 + length + 4 > len(data):
            structure_ok = False
            break
        chunk = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:
                                          pos + 12 + length])
        if zlib.crc32(ctype + chunk) & 0xFFFFFFFF != crc:
            crc_mismatch = True
        if ctype not in CANONICAL_PNG:
            noncanonical.append(ctype.decode("latin-1"))
        if ctype == b"IEND":
            trailing = data[pos + 12 + length:]
            ev.update(trailing_bytes=len(trailing),
                      trailing_text=_as_text(trailing[:4096]),
                      trailing_entropy=_entropy(trailing[:65536]))
            break
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            if ctype == b"zTXt":
                try:
                    chunk = zlib.decompress(chunk.split(b"\x00", 1)[1])
                except Exception:
                    chunk = b""
            text_payloads.append(_as_text(chunk))
        pos += 12 + length
    else:
        structure_ok = False  # no IEND found
    ev.update(structure_valid=structure_ok, crc_mismatch=crc_mismatch,
              noncanonical_chunks=noncanonical,
              text_chunk_payloads=text_payloads)
    return ev, None


# ---------------------------------------------------------------------------
# JPEG
# ---------------------------------------------------------------------------


def parse_jpeg(data: bytes):
    ev = {"format": "jpeg", "magic_valid": data[:2] == b"\xff\xd8"}
    if not ev["magic_valid"]:
        return ev, "bad JPEG magic"
    pos = 2
    exif_texts = []
    xmp_texts = []
    n_app = 0
    structure_ok = True  # baseline JPEG streams end in entropy-coded data;
    # that is NORMAL, not a structural failure
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            break  # entropy-coded scan data (normal for baseline JPEG)
        marker = data[pos + 1]
        while marker == 0xFF:  # fill bytes
            pos += 1
            if pos + 2 > len(data):
                break
            marker = data[pos + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            if marker == 0xD9:
                structure_ok = True
                break
            pos += 2
            continue
        if pos + 4 > len(data):
            break
        (seg_len,) = struct.unpack(">H", data[pos + 2:pos + 4])
        seg = data[pos + 4:pos + 2 + seg_len]
        if marker == 0xE1:  # APP1
            if seg.startswith(b"Exif\x00\x00"):
                exif_texts.append(_as_text(seg))
            elif seg.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                xmp_texts.append(_as_text(seg))
        elif marker == 0xE0:
            n_app += 1
        pos += 2 + seg_len
    # trailing bytes = everything after the LAST EOI marker (robust against
    # entropy-region 0xFF bytes)
    eoi = data.rfind(b"\xff\xd9")
    if eoi != -1:
        trailing = data[eoi + 2:]
        ev.update(trailing_bytes=len(trailing),
                  trailing_text=_as_text(trailing[:4096]),
                  trailing_entropy=_entropy(trailing[:65536]))
    ev.update(structure_valid=structure_ok, exif_blobs=exif_texts,
              xmp_blobs=xmp_texts)
    return ev, None


# ---------------------------------------------------------------------------
# WAV (RIFF family) + ID3-in-WAV
# ---------------------------------------------------------------------------

CANONICAL_RIFF = {b"RIFF", b"RIFX", b"LIST", b"fmt ", b"data", b"fact",
                  b"cue ", b"smpl", b"bext", b"JUNK", b"INFO", b"iXML",
                  b"id3 ", b"ID3 ", b"wavl", b"slnt", b"minf", b"elmo"}


def parse_wav(data: bytes):
    ev = {"format": "wav", "magic_valid": data[:4] in (b"RIFF", b"RIFX")}
    if not ev["magic_valid"]:
        return ev, "bad RIFF magic"
    be = data[:4] == b"RIFX"
    endian = ">" if be else "<"
    pos = 12
    noncanonical = []
    metadata = []
    structure_ok = False
    while pos + 8 <= len(data):
        ctype = data[pos:pos + 4]
        (size,) = struct.unpack(endian + "I", data[pos + 4:pos + 8])
        if ctype not in CANONICAL_RIFF:
            noncanonical.append(ctype.decode("latin-1"))
        body = data[pos + 8:pos + 8 + size]
        if ctype in (b"LIST", b"INFO"):
            metadata.append(_as_text(body))
        if ctype == b"data":
            # data chunk is (typically) last; trailing scan after it
            end = pos + 8 + size + (size & 1)
            trailing = data[end:]
            ev.update(trailing_bytes=len(trailing),
                      trailing_text=_as_text(trailing[:4096]),
                      trailing_entropy=_entropy(trailing[:65536]))
            structure_ok = True
            break
        pos += 8 + size + (size & 1)
    ev.update(structure_valid=structure_ok,
              noncanonical_chunks=noncanonical,
              metadata_blobs=metadata)
    return ev, None


# ---------------------------------------------------------------------------
# MP3: ID3v2 header + tag frames
# ---------------------------------------------------------------------------

ID3_FRAME_KINDS = {
    b"TIT2": "title", b"TPE1": "artist", b"TALB": "album", b"COMM": "comment",
    b"USLT": "lyrics", b"TXXX": "user_text", b"APIC": "attached_picture",
    b"PRIV": "private", b"TSSE": "encoder", b"WOAR": "url",
}


def parse_mp3(data: bytes):
    ev = {"format": "mp3", "magic_valid": data[:3] == b"ID3"}
    if not ev["magic_valid"]:
        return ev, "no ID3v2 header"
    try:
        ver = data[3]
        flags = data[5]
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | \
               ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        tag_end = 10 + size
        unsync = bool(flags & 0x80)
        if unsync:
            pass  # we do not fully de-unsync; payloads still captured below
        frames = []
        pos = 10
        while pos + 10 <= min(tag_end, len(data)):
            fid = data[pos:pos + 4]
            if fid == b"\x00\x00\x00\x00":
                break
            (fsize,) = struct.unpack(">I", data[pos + 4:pos + 8])
            body = data[pos + 10:pos + 10 + fsize]
            kind = ID3_FRAME_KINDS.get(fid, fid.decode("latin-1", "replace"))
            frames.append({"id": kind, "size": fsize,
                           "text": _as_text(body[:8192])})
            pos += 10 + fsize
        ev.update(tag_size=size, frames=frames,
                  duplicate_tags=_dup_count([f["id"] for f in frames]))
    except Exception:
        pass
    return ev, None


def _dup_count(items):
    seen = set()
    dups = 0
    for it in items:
        if it in seen:
            dups += 1
        seen.add(it)
    return dups


# ---------------------------------------------------------------------------
# unified probe
# ---------------------------------------------------------------------------


@dataclass
class PFileResult:
    format: str = "unknown"
    magic_valid: bool = False
    structure_valid: bool = False
    trailing_bytes: int = 0
    trailing_text: str = ""
    trailing_entropy: float = 0.0
    metadata_bytes: int = 0
    metadata_entropy: float = 0.0
    metadata_texts: list = field(default_factory=list)
    noncanonical_chunks: list = field(default_factory=list)
    text_chunk_payloads: list = field(default_factory=list)
    duplicate_tags: int = 0
    crc_mismatch: bool = False
    score: float = 0.0
    evidence: list = field(default_factory=list)
    error: str = ""


def probe_file(path: str) -> PFileResult:
    """Run P-file on a media file. Returns scored structural analysis."""
    res = PFileResult()
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 16:
        res.error = "file too small"
        return res

    parser = None
    for magic_check, fn in (
        (lambda d: d[:8] == PNG_MAGIC, parse_png),
        (lambda d: d[:2] == b"\xff\xd8", parse_jpeg),
        (lambda d: d[:4] in (b"RIFF", b"RIFX"), parse_wav),
        (lambda d: d[:3] == b"ID3", parse_mp3),
    ):
        if magic_check(data):
            parser = fn
            break
    if parser is None:
        res.error = "unsupported format"
        res.magic_valid = False
        res.score = 0.2  # unknown format: mildly anomalous
        res.evidence.append({
            "probe": "p_file", "trigger": True, "evidence_type":
            "unknown_format", "localization": "file",
            "extracted_text": ""})
        return res

    ev, err = parser(data)
    res.format = ev.get("format", "unknown")
    res.magic_valid = ev.get("magic_valid", False)
    res.structure_valid = ev.get("structure_valid", False)
    res.trailing_bytes = ev.get("trailing_bytes", 0)
    res.trailing_text = ev.get("trailing_text", "")
    res.trailing_entropy = ev.get("trailing_entropy", 0.0)
    res.noncanonical_chunks = ev.get("noncanonical_chunks", [])
    res.text_chunk_payloads = ev.get("text_chunk_payloads", [])
    res.duplicate_tags = ev.get("duplicate_tags", 0)
    res.crc_mismatch = ev.get("crc_mismatch", False)

    meta_blobs = (ev.get("exif_blobs", []) + ev.get("xmp_blobs", []) +
                  ev.get("metadata_blobs", []) +
                  ev.get("text_chunk_payloads", []) +
                  [f["text"] for f in ev.get("frames", [])])
    meta_blobs = [b for b in meta_blobs if b]
    res.metadata_texts = list(meta_blobs)
    if meta_blobs:
        meta_bytes = "".join(meta_blobs).encode("utf-8", "replace")
        res.metadata_bytes = len(meta_bytes)
        res.metadata_entropy = _entropy(meta_bytes)

    # ------------------------------------------------------- rule scoring --
    score = 0.0
    if not res.magic_valid:
        score += 0.4
    if not res.structure_valid:
        score += 0.3
    if res.trailing_bytes > 32:
        score += 0.35
        res.evidence.append({
            "probe": "p_file", "trigger": True,
            "evidence_type": "trailing_data",
            "localization": "after-EOF",
            "extracted_text": res.trailing_text[:200],
            "detail": f"{res.trailing_bytes} bytes after EOF"})
    if res.trailing_bytes > 0 and res.trailing_text.strip():
        # trailing data that decodes to text is highly suspicious
        score += 0.25
    if res.noncanonical_chunks:
        score += 0.3
        res.evidence.append({
            "probe": "p_file", "trigger": True,
            "evidence_type": "noncanonical_block",
            "localization": ",".join(res.noncanonical_chunks[:8]),
            "extracted_text": ""})
    if res.crc_mismatch:
        score += 0.2
        res.evidence.append({"probe": "p_file", "trigger": True,
                             "evidence_type": "crc_mismatch",
                             "localization": "file", "extracted_text": ""})
    if res.duplicate_tags:
        score += 0.15
    if res.metadata_bytes > 500 and res.metadata_entropy < 4.0:
        # large low-entropy (text-like) metadata blob
        score += 0.25
        res.evidence.append({
            "probe": "p_file", "trigger": True,
            "evidence_type": "large_metadata",
            "localization": "metadata-tag",
            "extracted_text": "".join(meta_blobs)[:200],
            "detail": f"{res.metadata_bytes} bytes"})
    res.score = min(score, 1.0)
    return res


# ---------------------------------------------------------------------------
# standalone CLI (used by the harness for pre-computation/caching)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    for p in sys.argv[1:]:
        r = probe_file(p)
        print(json.dumps({
            "file": os.path.basename(p), "format": r.format,
            "magic_valid": r.magic_valid,
            "structure_valid": r.structure_valid,
            "trailing_bytes": r.trailing_bytes,
            "noncanonical": r.noncanonical_chunks[:8],
            "metadata_bytes": r.metadata_bytes,
            "duplicate_tags": r.duplicate_tags,
            "score": round(r.score, 3),
            "evidence": r.evidence,
        }, ensure_ascii=False))
