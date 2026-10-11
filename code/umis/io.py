"""Manifest paths are relative to the manifest directory in v1.3."""
import json
from pathlib import Path

def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    ids = [e["id"] for e in data["entries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest entry IDs must be unique")
    for entry in data["entries"]:
        entry["source_files"] = [str((path.parent / p).resolve())
                                 for p in entry["source_files"]]
    return data
