"""Stage 0 audio front-end: ASR (faster-whisper large-v3) with word
timestamps + VAD (silero-vad)."""
import hashlib
import json
import os

import numpy as np


class ASR:
    def __init__(self, model_size="large-v3", device="cuda",
                 compute_type=None, cache_dir=None, cpu_threads=16):
        from faster_whisper import WhisperModel
        if compute_type is None:
            # CPU uses integer quantization; CUDA supports mixed int8/float16.
            compute_type = "int8_float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type,
                                  cpu_threads=cpu_threads)
        self.cache_dir = cache_dir
        self.cache_namespace = f"asr:{model_size}:{device}:{compute_type}"

    def _cache_path(self, audio_path):
        if not self.cache_dir:
            return None
        from ..cache import media_key
        h = media_key(audio_path, self.cache_namespace)
        return os.path.join(self.cache_dir, h + ".json")

    def transcribe(self, audio_path):
        """Return {text, segments:[{start,end,text,words:[{word,start,end}]}]}"""
        cp = self._cache_path(audio_path)
        if cp and os.path.exists(cp):
            with open(cp) as f:
                return json.load(f)
        segments, info = self.model.transcribe(
            audio_path, beam_size=1, word_timestamps=True,
            vad_filter=True)
        segments = list(segments)  # faster-whisper returns a one-shot generator
        out = {"text": " ".join(s.text for s in segments).strip(),
               "language": info.language,
               "segments": [
                   {"start": float(s.start), "end": float(s.end),
                    "text": s.text,
                    "words": [{"word": w.word, "start": float(w.start),
                               "end": float(w.end)}
                              for w in (s.words or [])]}
                   for s in segments]}
        if cp:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, "w") as f:
                json.dump(out, f)
        return out


class VAD:
    def __init__(self, device="cpu"):
        import torch
        self.device = device
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            trust_repo=True)
        self.model.to(device).eval()

    def speech_timestamps(self, audio_path, sr=16000):
        import torch
        import soundfile as sf
        x, _ = sf.read(audio_path)
        if len(x.shape) > 1:
            x = x.mean(axis=1)
        with torch.no_grad():
            frames = torch.from_numpy(
                (np.clip(x, -1, 1) * 32767).astype(np.int16)).float()
            # chunked VAD on 512-sample windows
            timestamps = []
            frame_ms = 512 / sr * 1000
            window = torch.tensor([], dtype=torch.float32)
            idx = 0
            batch = []
            while idx < len(frames):
                batch = frames[idx:idx + 512]
                if len(batch) < 512:
                    break
                prob = self.model(batch.to(self.device), sr).item()
                if prob > 0.5:
                    t0 = idx / sr
                    timestamps.append([t0, (idx + 512) / sr])
                idx += 512
            # merge adjacent
            merged = []
            for ts in timestamps:
                if merged and ts[0] - merged[-1][1] < 0.3:
                    merged[-1][1] = ts[1]
                else:
                    merged.append(list(ts))
            return merged
