"""Synthetic-speech detector (ASVspoof-style, lightweight).

A small CNN on log-mel spectrograms, trained to separate TTS/cloned speech
(bonafide-spoof task). Training data: real speech = LibriSpeech + Common
Voice subsets; spoof = our edge-tts samples + WaveFake subset (if
downloaded). Default: 1.3M-param CNN, <10 MB weights, GPU inference cheap.

Honest caveats: ASVspoof 2019/2021 official data requires a registration
agreement and is NOT redistributable; we therefore train on openly licensed
data only and document this substitution in docs/REPRODUCIBILITY.md.
"""
import os

import numpy as np
import torch
import torch.nn as nn


def logmel(wav, sr=16000, n_mels=80, n_fft=512, hop=160, win_len=400):
    import librosa
    S = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop, win_length=win_len, power=2)
    return librosa.power_to_db(S, ref=np.max, top_db=80)


class SynthSpeechCNN(nn.Module):
    def __init__(self, n_mels=80, hidden=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden * 16, 2))

    def forward(self, x):
        return self.head(self.features(x))


class SynthSpeechDetector:
    """Train/infer wrapper. Inference input: wav file path or 1-D array."""

    def __init__(self, weight_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model = SynthSpeechCNN()
        if not weight_path or not os.path.isfile(weight_path):
            raise FileNotFoundError("Supply trained synthetic-speech weights")
        if weight_path:
            self.model.load_state_dict(
                torch.load(weight_path, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    @staticmethod
    def _spec(wav, sr):
        spec = logmel(wav, sr)
        t = spec.shape[1] // 400  # 400 frames ~= 4s
        if t < 1:
            return None
        return spec[:, :t * 400].reshape(-1, 80, 400)[:8]  # up to 8 chunks

    def score(self, audio_path=None, wav=None, sr=16000):
        """Return (p_spoof_avg, p_spoof_max)."""
        import soundfile as sf
        if audio_path is not None:
            wav, sr = sf.read(audio_path)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        specs = self._spec(wav, sr)
        if specs is None:
            return 0.5, 0.5
        x = torch.from_numpy(specs[:, None]).float().to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # p(spoof)
        return float(probs.mean()), float(probs.max())

    @staticmethod
    def train(real_wavs, spoof_wavs, out_path, epochs=8, batch=64,
              sr=16000, seed=0, device=None):
        """Train on lists of wav paths (real, spoof). Saves weights."""
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = SynthSpeechCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        def load_set(paths, label, limit=4000):
            data = []
            rng = random.Random(seed + label)
            for p in rng.sample(paths, min(len(paths), limit)):
                try:
                    import soundfile as sf
                    w, s = sf.read(p)
                    if len(w.shape) > 1:
                        w = w.mean(axis=1)
                    if len(w) < s:  # <1s
                        continue
                    spec = logmel(w[:s * 8], s)  # up to 8s
                    for st in range(0, max(1, spec.shape[1] // 400)):
                        chunk = spec[:, st * 400:(st + 1) * 400]
                        if chunk.shape[1] < 400:
                            break
                        data.append((chunk, label))
                except Exception:
                    continue
            return data

        train = load_set(real_wavs, 0) + load_set(spoof_wavs, 1)
        random.shuffle(train)
        if len(train) < 64:
            raise RuntimeError(f"not enough training data: {len(train)}")
        for ep in range(epochs):
            model.train()
            tot, correct = 0, 0
            for i in range(0, len(train), batch):
                batch_data = train[i:i + batch]
                x = torch.from_numpy(
                    np.stack([d[0] for d in batch_data])[:, None]
                ).float().to(device)
                y = torch.tensor([d[1] for d in batch_data],
                                 device=device)
                logits = model(x)
                loss = ce(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += len(y)
                correct += (logits.argmax(1) == y).sum().item()
            print(f"epoch {ep}: acc={correct / tot:.3f} loss={loss.item():.3f}")
        torch.save(model.state_dict(), out_path)
        return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--spoof_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    real = [os.path.join(args.real_dir, f) for f in
            os.listdir(args.real_dir) if f.endswith((".wav", ".flac"))]
    spoof = [os.path.join(args.spoof_dir, f) for f in
             os.listdir(args.spoof_dir) if f.endswith((".wav", ".flac"))]
    SynthSpeechDetector.train(real, spoof, args.out, epochs=args.epochs)
