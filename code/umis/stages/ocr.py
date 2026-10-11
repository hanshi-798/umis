"""Stage 0 image front-end: OCR + layout (PaddleOCR PP-OCRv4).

Extracts text regions with bounding boxes. Runs on CPU by default (PaddleOCR
CPU inference is fast enough and keeps VRAM free for the target/encoder).
"""
import os
import threading


class OCR:
    def __init__(self, use_gpu=False, cache_dir=None):
        import paddleocr
        from paddleocr import PaddleOCR
        self._lock = threading.Lock()
        # version-adaptive init (fixed 2026-08-15): PaddleOCR 3.x removed
        # `use_gpu`/`use_angle_cls` (now `device`/`use_textline_orientation`)
        # and dropped the `cls=` kwarg of `.ocr()`.
        major = int(paddleocr.__version__.split(".")[0])
        if major >= 3:
            # enable_mkldnn=False is REQUIRED on this stack: PaddlePaddle
            # 3.3.1 PIR+oneDNN CPU inference crashes the det predictor
            # (ConvertPirAttribute2RuntimeAttribute NotImplementedError).
            # The orientation/unwarping preprocessing models (doc_ori,
            # UVDoc, textline_ori) are DISABLED: they add ~45 s/image on
            # CPU and are not needed for synthetic/typographic media
            # (fixed 2026-08-15 after measuring 53 s/image with them on).
            kwargs = dict(lang="en", device="gpu" if use_gpu else "cpu",
                          use_textline_orientation=False,
                          use_doc_orientation_classify=False,
                          use_doc_unwarping=False,
                          ocr_version="PP-OCRv4", enable_mkldnn=False)
        else:
            kwargs = dict(use_angle_cls=True, lang="en", use_gpu=use_gpu,
                          show_log=False)
        self._ocr = PaddleOCR(**kwargs)
        self._major = major
        self.cache = {}
        self.cache_dir = cache_dir

    def _cache_path(self, img_path):
        if not self.cache_dir:
            return None
        import hashlib
        from ..cache import media_key
        h = media_key(img_path, f"ocr:{self._major}:en:PP-OCRv4")
        return os.path.join(self.cache_dir, h + ".json")

    def extract(self, img_path):
        """Return list of {text, confidence, bbox:[x0,y0,x1,y1]}."""
        cp = self._cache_path(img_path)
        if cp and os.path.exists(cp):
            import json
            with open(cp) as f:
                return json.load(f)
        with self._lock:
            try:
                if self._major >= 3:
                    result = self._ocr.ocr(img_path)
                else:
                    result = self._ocr.ocr(img_path, cls=True)
            except Exception as ex:
                # PaddleOCR's reader rejects some structurally unusual
                # PNGs (e.g. files with injected tEXt chunks); fall back
                # to a clean PIL re-encode (fixed 2026-08-15).
                if "Image read Error" not in str(ex):
                    raise
                from PIL import Image
                import tempfile
                img = Image.open(img_path).convert("RGB")
                tmp = tempfile.NamedTemporaryFile(suffix=".png",
                                                  delete=False)
                img.save(tmp.name)
                tmp.close()
                try:
                    if self._major >= 3:
                        result = self._ocr.ocr(tmp.name)
                    else:
                        result = self._ocr.ocr(tmp.name, cls=True)
                finally:
                    os.remove(tmp.name)
        regions = []
        if self._major >= 3:
            # PaddleOCR 3.x predict() returns one dict per page with
            # rec_texts / rec_scores / rec_polys (not the 2.x
            # [box, (text, conf)] list format - fixed 2026-08-15)
            for item in result or []:
                if not isinstance(item, dict):
                    continue
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                polys = item.get("rec_polys") or []
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    bbox = None
                    if i < len(polys) and polys[i] is not None:
                        poly = polys[i]
                        if hasattr(poly, "tolist"):
                            poly = poly.tolist()
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    conf = float(scores[i]) if i < len(scores) else 1.0
                    regions.append({"text": text,
                                    "confidence": conf,
                                    "bbox": bbox})
        elif result and result[0]:
            for line in result[0]:
                box, (text, conf) = line[0], line[1]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                regions.append({"text": text,
                                "confidence": float(conf),
                                "bbox": [min(xs), min(ys), max(xs), max(ys)]})
        if cp:
            import json
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, "w") as f:
                json.dump(regions, f)
        return regions

    def full_text(self, img_path):
        return " ".join(r["text"] for r in self.extract(img_path))
