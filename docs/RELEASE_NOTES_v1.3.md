# UMIS v1.3.0

UMIS is now packaged as an installable Python research tool with a `umis` CLI,
MIT license, English/Chinese documentation, and offline regression coverage.

This release removes private server assumptions and fixes ASR extraction,
media-content caches, audio edge cases, held-out fusion calibration, feature
consistency, and conservative L2 response handling. It supports configurable
image/audio inspection, optional semantic encoders, and external judge endpoints.

Historical results are retained separately for reference. Dataset media and
trained model artifacts are not bundled. These results are not new v1.3
measurements; users should rerun calibration and evaluation after the fixes.

Quick start:

```bash
python -m pip install -e .
umis --version
umis probe image.png
```

See the README for optional model dependencies and complete experiment commands.
