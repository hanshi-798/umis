# UMIS — Unified Multimodal Injection Sentinel

**Version 1.3.0 · MIT** · [中文说明](README.zh-CN.md)

UMIS inspects image- and audio-carried prompt injection before media reaches a
multimodal assistant. File structure, decoded signal statistics, and optional
semantic features feed a three-way decision: `BLOCK`, `QUARANTINE+CONFIRM`, or
`ALLOW+AUDIT`. Reports include localized evidence and extraction text.

This is a research implementation. It returns a decision; your application must
enforce blocking, quarantine, confirmation, and audit storage. Text-only inputs
are outside its scope. A low score is not a security guarantee.

## Install

Python 3.10+ is required. The lightweight probes run on CPU without model downloads:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
umis --version
umis probe path/to/image.png
```

Install only the model stages you need:

```bash
python -m pip install -e ".[ocr,audio,encoder,judge,bench,plots]"
```

OCR/ASR/encoders download models on first use. The full research stack is intended
for Linux with a compatible PyTorch/CUDA setup. `ffmpeg` must be on `PATH` for
MP3 conversion and benchmark audio generation. GPU serving with vLLM is installed
separately; its CUDA/PyTorch requirements should determine your server environment.
The release does not include pretrained fusion weights or benchmark media.

## Use

```bash
# Image-only heuristic detection; requires the ocr extra
umis detect --image path/to/image.png --user-text "Describe this image." --out results/report.json

# Audio-only heuristic detection; requires the audio extra
umis detect --audio path/to/audio.wav --user-text "Transcribe this audio."

# Calibrated pipeline; requires matching locally trained artifacts and encoder extra
umis detect --image path/to/image.png --encoder clip_clap --device cuda \
  --head results/fusion_head.joblib --thresholds results/thresholds.json \
  --centroid results/centroid.npz
```

`probe` reports file/signal features only, not the full security verdict. `detect`
defaults to no semantic encoder and heuristic scores. CLIP+CLAP mode requires
`CLAP_CKPT` pointing to a locally downloaded LAION-CLAP checkpoint. Image and audio
embeddings occupy separate spaces in this mode; cross-modal similarity and
OCR-versus-VLM description disagreement are unavailable. Omni mode is retained as
an experimental backend and needs a separate large-model environment.

For an optional L2 judge, add `--judge-url http://127.0.0.1:8000/v1` and
`--judge-model MODEL_ID`. Unrecognized responses and request failures retain
quarantine. `UMIS_API_KEY` configures the judge client. Extracted media text is
sent to the configured endpoint; reports can contain sensitive content.

The Python API accepts injectable OCR/ASR/encoder objects:

```python
from umis import UMIS
from umis.stages.ocr import OCR

detector = UMIS(ocr=OCR(), asr=None, l2_enabled=False)
report = detector.process("sample", image_path="image.png", user_text="Describe it.")
print(report["verdict"])
```

## Benchmark and evaluation

Commands below run from the repository root. Obtain COCO JPG images and
LibriSpeech FLAC audio separately. TTS generators contact the Edge-TTS service.
The benchmark builder uses family/kind stratified sample splits; shared source
carriers can occur across splits. This is **not a carrier-disjoint evaluation**.

```bash
python bench/build_bench.py --coco_dir data/coco/val2017 \
  --ls_dir data/LibriSpeech/test-clean --out_dir bench/out --n 100 --seed 0 --skip_adv

python harness/compute_centroid.py --manifest bench/out/manifest.json \
  --out results/centroid.npz --cache_dir results/cache --encoder_mode clip_clap

python harness/collect_features.py --manifest bench/out/manifest.json \
  --out results/features.json --cache_dir results/cache --encoder_mode clip_clap \
  --centroid results/centroid.npz

python harness/train_head.py --features results/features.json \
  --out results/fusion_head.joblib --t1_out results/thresholds.json

bash run_all.sh evaluate
```

The runner reads thresholds from the saved JSON. It evaluates the test split and
disables L2 for a first pass; optional second-pass judging is described in
[reproducibility](docs/REPRODUCIBILITY.md). No server is started automatically.
Model artifacts must match the feature configuration used at training time.
Only load `.joblib` and model weights from trusted sources.

Historical experiment JSONs are under `results/reference/`; they lack the full
raw reports, benchmark media, trained weights, and model snapshots. They are not
new v1.3 measurements and cannot independently reproduce the earlier tables.
Changes to calibration and feature extraction require rerunning experiments.
Detector pass-through is not target-model attack success rate; compute target
ASR only from target responses and ground-truth triggers.

## Repository

| Path | Purpose |
| --- | --- |
| `code/umis/` | Installable detector, CLI, probes, extraction stages |
| `attacks/` | Image/audio benchmark sample generators |
| `bench/` | Benchmark builder |
| `harness/` | Calibration, evaluation, metrics and aggregation |
| `tests/` | Offline regression tests with synthetic fixtures |
| `docs/` | Release and reproducibility notes |
| `results/reference/` | Labeled historical results |

```bash
python -m pip install -e ".[dev,bench,judge]"
python -m pytest -q
python -m build
```

See [CHANGELOG](CHANGELOG.md), [release instructions](docs/RELEASE.md), and
[third-party notices](docs/THIRD_PARTY.md). MIT covers project code; external
models and data retain their own terms. Paper drafts and private infrastructure
logs are excluded from this tool release.
