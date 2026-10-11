# Changelog

## 1.3.0 — 2026-10-11

This release packages the supplied research snapshot for public GitHub distribution.
The snapshot did not contain a verifiable v1.1/v1.2 release history.

- Add installable Python package, versioned CLI, MIT license, bilingual README,
  optional dependency groups, source manifest, GitHub CI, and regression tests.
- Replace remote server paths with repository-relative and configurable paths.
- Save portable benchmark manifests and consistently resolve media paths.
- Use held-out validation data for isotonic calibration; reject missing splits.
- Reuse pipeline feature extraction for training, including both mixed-input
  file probes and optional image drift reference.
- Preserve complete extracted content and typed intent in two-pass L2 judging;
  unknown judge responses remain quarantined.
- Fix ASR generator consumption, CPU compute type, content-based caches, short
  PCM return shape, non-power-of-two echo windows, and MP3 temporary cleanup.
- Use only image embeddings for the image drift centroid and avoid object arrays.
- Disable unavailable CLIP+CLAP description disagreement; fix Omni audio/text
  alignment and require an explicit CLAP checkpoint path.
- Retry failed evaluation entries; filter metric reports to selected manifest IDs;
  evaluate test split by default and avoid duplicate OCR/ASR initialization.
- Evaluate no-defense targets with the same manifest user task rather than explicit
  instructions to obey media content; infer the correct audio target model and MIME type.
- Label historical results and remove private infrastructure logs and paper drafts.

No new GPU benchmark claims are made. Old numbers must be rerun after these changes.
