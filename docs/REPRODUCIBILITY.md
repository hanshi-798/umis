# Reproducibility and limitations

The source snapshot contains implementations and summary JSONs, not a complete
experiment bundle. Benchmark media, fusion weights, synthetic-speech weights,
raw detector reports, and downloaded model snapshots are absent.

## Current workflow

1. Install the extras required for your experiment and FFmpeg. Record the Python,
   dependency, CUDA, GPU, and model revision information in your experiment.
2. Acquire COCO and LibriSpeech data independently; do not commit their media.
3. Build the benchmark with `bench/build_bench.py`. It generates relative media
   paths, stable IDs, and train/val/test labels. TTS uses an external service and
   therefore cannot promise byte-identical regeneration. Existing output folders
   should not be reused for a different benchmark configuration.
4. Compute the image-only benign training centroid before collecting features.
5. Collect train/val features with the same centroid and encoder configuration
   later used for evaluation. Avoid reusing cached feature JSONs after a config
   change: those output JSONs are resumable, not configuration-versioned caches.
6. Train logistic regression on train data; calibrate isotonic regression and
   thresholds on validation data. Never use test labels to fit either component.
7. Run `bash run_all.sh evaluate` for the test split. Outputs include extracted
   text, so treat the reports as potentially sensitive.
8. Optionally start your own OpenAI-compatible judge endpoint and run:

```bash
python harness/eval_umis.py --manifest bench/out/manifest.json \
  --out results/umis_eval.json --cache_dir results/cache --only_l2 \
  --judge_base_url http://127.0.0.1:8000/v1 \
  --judge_model YOUR_MODEL --t2 YOUR_SAVED_T2
```

The second pass operates on stored reports, preserving extraction and intent;
it then recalculates the summary. Legacy reports without these fields use stage
text and the manifest intent as a fallback, which can lose extraction detail.

The v1.3 baseline uses manifest user tasks. The historical baseline instead
explicitly asked targets to follow media instructions, so its ASR is not comparable
to the corrected workflow. For no-defense targets, start a compatible target server, then:

```bash
python harness/eval_target.py --target vl --manifest bench/out/manifest.json \
  --out results/no_defense_vl.json --split test \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct

python harness/eval_target.py --target qwen2audio --modality audio \
  --manifest bench/out/manifest.json --out results/no_defense_audio.json \
  --split test --model_name Qwen/Qwen2-Audio-7B-Instruct
```

The experimental Omni backend is retained for research compatibility; its
generation path was reported as unreliable in the historical environment and
is not part of offline release validation.

## Interpretation

- Historical splits are per family/kind, not grouped by source media. Shared
  carriers can leak across partitions. A carrier-disjoint study needs a new
  manifest and a fresh run, not relabeling the old results.
- CLIP+CLAP has separate image/text and audio/text spaces. Cross-modal similarity
  and VLM description disagreement are disabled; the centroid models images only.
- P-content is strongest on PNG and 16-bit PCM WAV. JPEG features are a decoded
  block-DCT proxy. Other containers can yield unsupported-format/error features;
  missing signal features must not be interpreted as proof of benign content.
- Heuristic L0 keyword rules can block harmless quotations. Fusion scores are
  heuristic until a matched head is trained and calibrated.
- Quarantine requires application intervention. The legacy `attack_allowed_rate`
  field counts every non-BLOCK verdict, including quarantine; it is not actual
  execution or target ASR. `benign_block_rate` similarly includes quarantine.
- L2 judges remain susceptible to prompt injection. Strict label parsing and
  quarantine fallback do not establish robust security.
- Optional dependency ranges are installation constraints, not a frozen historical
  GPU environment. Validate and freeze your own full environment after installation.
- Local release checks cover CPU probes, mocked extraction/judging, calibration,
  metrics and packaging; they do not establish GPU or model quality.
