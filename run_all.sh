#!/usr/bin/env bash
# Repository-relative phase runner; no model server is started automatically.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/code:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
DATA="${UMIS_DATA_DIR:-$ROOT/data}"
BENCH="${UMIS_BENCH_DIR:-$ROOT/bench/out}"
RESULTS="${UMIS_RESULTS_DIR:-$ROOT/results}"
CACHE="$RESULTS/cache"
ENCODER="${UMIS_ENCODER:-clip_clap}"
DEVICE="${UMIS_DEVICE:-cuda}"
mkdir -p "$RESULTS" "$CACHE"
case "${1:-}" in
  setup)
    "$PYTHON" -m pip install -e '.[ocr,audio,encoder,judge,bench,plots]'
    ;;
  bench)
    "$PYTHON" bench/build_bench.py --coco_dir "${COCO_DIR:-$DATA/coco/val2017}" \
      --ls_dir "${LIBRISPEECH_DIR:-$DATA/LibriSpeech/test-clean}" \
      --out_dir "$BENCH" --n "${UMIS_N:-100}" --seed 0 --skip_adv
    ;;
  train)
    "$PYTHON" harness/compute_centroid.py --manifest "$BENCH/manifest.json" \
      --out "$RESULTS/centroid.npz" --cache_dir "$CACHE" --encoder_mode "$ENCODER" --device "$DEVICE"
    "$PYTHON" harness/collect_features.py --manifest "$BENCH/manifest.json" \
      --out "$RESULTS/features.json" --cache_dir "$CACHE" --encoder_mode "$ENCODER" \
      --device "$DEVICE" --asr_device "$DEVICE" --centroid "$RESULTS/centroid.npz"
    "$PYTHON" harness/train_head.py --features "$RESULTS/features.json" \
      --out "$RESULTS/fusion_head.joblib" --t1_out "$RESULTS/thresholds.json"
    ;;
  evaluate)
    T1="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["t1"])' "$RESULTS/thresholds.json")"
    T2="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["t2"])' "$RESULTS/thresholds.json")"
    "$PYTHON" harness/eval_umis.py --manifest "$BENCH/manifest.json" \
      --out "$RESULTS/umis_eval.json" --cache_dir "$CACHE" --head "$RESULTS/fusion_head.joblib" \
      --centroid "$RESULTS/centroid.npz" --encoder_mode "$ENCODER" --device "$DEVICE" \
      --asr_device "$DEVICE" --t1 "$T1" --t2 "$T2" --split test --no_l2
    ;;
  metrics)
    "$PYTHON" harness/eval_umis.py --manifest "$BENCH/manifest.json" \
      --out "$RESULTS/umis_eval.json" --cache_dir "$CACHE" --metrics_only --split test
    ;;
  *)
    echo 'usage: bash run_all.sh <setup|bench|train|evaluate|metrics>' >&2
    exit 2
    ;;
esac
