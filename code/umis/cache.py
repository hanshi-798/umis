"""Content-based cache keys; replacing a file invalidates cached extraction."""
import hashlib
from pathlib import Path

def media_key(path, namespace):
    h = hashlib.sha256(namespace.encode("utf-8"))
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
