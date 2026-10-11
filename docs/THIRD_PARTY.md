# Third-party models and data

MIT applies to project code. External datasets, images, audio, model weights,
fonts and services are not included or relicensed by this repository.

Model identifiers referenced by the implementation include:

- Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2-Audio-7B-Instruct,
  Qwen/Qwen2.5-Omni-7B
- openai/clip-vit-base-patch32 and LAION-CLAP checkpoints
- faster-whisper large-v3 and PaddleOCR detection/recognition models

Review the exact model card and revision terms when downloading. Dataset users
must review COCO image licensing per image and LibriSpeech terms. Edge-TTS is an
external service; check its service and generated-media terms before redistribution.
Generated benchmark samples derived from third-party media are not automatically
covered by this repository's MIT license.

The generators are simplified research implementations inspired by multimodal
injection families. They do not establish faithful reproductions of every
referenced attack or certification of defensive coverage.
