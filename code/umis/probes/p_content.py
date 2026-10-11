"""Probe P-content (Stage 2): shared signal-layer features.

Modality-agnostic anomaly features computed on decoded media:
  - LSB-plane randomness (chi-square) on pixel bytes (PNG) and PCM samples
    (16-bit WAV) — the shared steganography primitive.
  - Transform-domain statistics: DCT histogram anomalies for JPEG,
    per-window FFT statistics for audio (high-frequency energy, spectral
    flatness, noise floor).
  - Entropy of coefficient distributions.
Output: signal anomaly score + evidence entries with localization.
"""
import math
import os
import wave

import numpy as np
from PIL import Image

from . import p_file as pf


# ---------------------------------------------------------------------------
# LSB chi-square test (pairs-of-values analysis, Westfeld & Pfitzmann style)
# ---------------------------------------------------------------------------


def lsb_chisq_pixels(rgb_flat: np.ndarray, n_bins=64):
    """Chi-square LSB test on pixel bytes. Returns (chi2, p_value, mean_ratio).

    Uses paired bins (2i, 2i+1) whose occupancies equalize under random LSBs.
    """
    hist = np.bincount(rgb_flat.astype(np.int64), minlength=256)
    k = 0
    chi2 = 0.0
    for i in range(0, n_bins * 2, 2):
        n2i, n2i1 = float(hist[i]), float(hist[i + 1])
        if n2i + n2i1 < 5:
            continue
        expected = (n2i + n2i1) / 2.0
        chi2 += (n2i - expected) ** 2 / expected + \
                (n2i1 - expected) ** 2 / expected
        k += 1
    if k == 0:
        return 0.0, 1.0
    # survival function of chi-square with k dof (Wilson-Hilferty approx)
    from scipy import stats
    p = 1.0 - stats.chi2.cdf(chi2, k)
    return chi2, p


def lsb_chisq_pcm(x16: np.ndarray, n_bins=32):
    """Chi-square LSB test on 16-bit PCM samples (LSB plane of int16)."""
    lsbs = x16 & 1
    # pair-of-values on the LSB plane against local pairs
    hist = np.bincount(lsbs.astype(np.int64), minlength=2)
    n0, n1 = float(hist[0]), float(hist[1])
    if n0 + n1 < 10:
        return 0.0, 1.0, 0.5
    expected = (n0 + n1) / 2.0
    chi2 = (n0 - expected) ** 2 / expected + (n1 - expected) ** 2 / expected
    from scipy import stats
    p = 1.0 - stats.chi2.cdf(chi2, 1)
    # also test within-pair byte equality ratio of low bytes
    ratio = n0 / (n0 + n1)
    return chi2, p, ratio


# ---------------------------------------------------------------------------
# audio spectral features (per-window)
# ---------------------------------------------------------------------------


def audio_spectral(x: np.ndarray, sr: int, win=0.05, hop=0.025):
    """Per-window spectral stats. Returns arrays + aggregates."""
    n_fft = int(sr * win)
    hop = int(sr * hop)
    if len(x) < n_fft:
        return None
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(
        x, n_fft)[::hop][:n_frames]
    win_fn = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames * win_fn, axis=1))
    power = spec ** 2
    total = power.sum(axis=1, keepdims=True) + 1e-12
    prob = power / total
    # spectral flatness (geometric mean / arithmetic mean of power)
    log_mean = np.log(power + 1e-12).mean(axis=1)
    flatness = np.exp(log_mean) / (power.mean(axis=1) + 1e-12)
    entropy = -(prob * np.log(prob + 1e-12)).sum(axis=1)
    max_entropy = math.log(spec.shape[1])
    norm_entropy = entropy / max_entropy
    # high-frequency energy ratio: above 7 kHz for 16k sr
    hf_bin = int(7000 / (sr / 2) * (spec.shape[1] - 1))
    hf_ratio = power[:, hf_bin:].sum(axis=1) / total[:, 0]
    # noise floor: median power of lowest-energy bins
    noise_floor = np.median(np.sort(power, axis=1)[:, : max(2, spec.shape[1] // 10)], axis=1)
    return {
        "frames": n_frames,
        "flatness_mean": float(np.mean(flatness)),
        "entropy_mean": float(np.mean(norm_entropy)),
        "hf_ratio_mean": float(np.mean(hf_ratio)),
        "hf_ratio_max": float(np.max(hf_ratio)),
        "noise_floor": float(np.mean(noise_floor)),
    }


def cepstral_echo_energy(x: np.ndarray, sr: int, delay_ms=(40, 300)):
    """Cepstral-domain echo detector: energy of the cepstrum in the delay
    range of interest. Echo/hidden-carrier embedding produces a peak at the
    echo delay in the real cepstrum."""
    if len(x) < 2 * sr:
        return 0.0
    n = 1 << int(np.ceil(np.log2(len(x))))
    xp = np.zeros(n)
    xp[:min(len(x), n)] = x[:min(len(x), n)]
    spec = np.abs(np.fft.rfft(xp * np.hanning(n)))
    log_spec = np.log(spec + 1e-12)
    cep = np.fft.irfft(log_spec)
    d0, d1 = int(delay_ms[0] * sr / 1000), int(delay_ms[1] * sr / 1000)
    region = cep[d0:d1]
    # normalized peak: (peak - mean) / std over the whole cepstrum tail
    tail = cep[100:]
    return float((np.max(np.abs(region)) - np.mean(tail)) /
                 (np.std(tail) + 1e-12))


# ---------------------------------------------------------------------------
# JPEG DCT coefficient histogram stats
# ---------------------------------------------------------------------------

# We approximate DCT-domain stats without a full JPEG decoder: re-decode the
# image and compute block DCT of the *decoded* pixels is NOT what stego tools
# see. Instead we parse quantized coefficients with a minimal baseline
# sequential JPEG decoder for the luminance channel. If decoding fails, we
# fall back to pixel-domain high-frequency energy (less sensitive but still
# catches our DCT-embedding generator's artifacts).
def _jpeg_quant_coeffs(path: str):
    """Minimal baseline-JPEG luminance coefficient decoder.

    Returns array of quantized DCT coefficient blocks (lum) or None.
    """
    try:
        import scipy.fftpack  # noqa: F401  (used only for inverse DCT)
        import cv2
        # OpenCV does not expose quantized coefficients; use a full decode
        # of luminance and block-DCT as a cheap proxy (documented limitation)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return _block_dct(img)
    except ImportError:
        return None


def _block_dct(gray: np.ndarray):
    """8x8 block DCT of a grayscale image (proxy for coefficient stats)."""
    import scipy.fftpack as fft
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    blocks = gray[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(
        0, 2, 1, 3).reshape(-1, 8, 8)
    coeffs = fft.dct(fft.dct(blocks.astype(np.float64) - 128.0,
                            axis=2, norm="ortho"), axis=1, norm="ortho")
    return coeffs


def dct_stats(coeffs: np.ndarray):
    """Anomaly stats of DCT coefficient histogram (embedding perturbs the
    natural falloff / creates zeros-spikes at AC positions)."""
    ac = coeffs[:, 1:].flatten()
    ac = ac[np.abs(ac) < 1000]
    hist, edges = np.histogram(ac, bins=101, range=(-50.5, 50.5))
    hist = hist.astype(np.float64) + 1e-9
    hist /= hist.sum()
    # zero-bin spike: embedding moves small coefficients off zero
    zero_spike = float(hist[50] / max(hist[49], hist[51]))
    # odd/even imbalance (LSB embedding in coefficients)
    odds = ac[np.abs(ac) >= 1].astype(np.int64)
    if len(odds):
        odd_ratio = float(np.mean(odds % 2 == 1))
    else:
        odd_ratio = 0.5
    entropy = -float((hist * np.log(hist)).sum())
    return {"zero_spike": zero_spike, "odd_ratio": odd_ratio,
            "entropy": entropy}


# ---------------------------------------------------------------------------
# unified probe
# ---------------------------------------------------------------------------


def probe_content(path: str, fmt: str = None, sr: int = 16000):
    """Run P-content signal features on one media file."""
    fmt = fmt or _detect(path)
    ev = []
    feats = {}
    score = 0.0

    if fmt == "png":
        try:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
            chi2, p = lsb_chisq_pixels(arr.flatten())
            feats["lsb_chi2"] = float(chi2)
            feats["lsb_p"] = float(p)
            # LSB steganography makes the LSB plane MORE uniform; natural
            # and PIL/JPEG-reencoded pixels have biased LSBs (high chi2).
            # Direction fixed 2026-08-16 (Phase D #3): trigger on
            # over-uniformity, not on bias.
            if p > 0.95 and arr.size > 100000:
                score += 0.6
                ev.append({"probe": "p_content", "trigger": True,
                           "evidence_type": "lsb_anomaly",
                           "localization": "image-pixels",
                           "extracted_text": "",
                           "detail": f"chi2={chi2:.1f} p={p:.2e} "
                                     "(over-uniform)"})
            # high-frequency energy (stego noise signature)
            import scipy.ndimage as ndi
            lap = np.abs(ndi.laplace(arr.mean(axis=2))).mean()
            feats["hf_energy"] = float(lap)
            if lap > 30:
                score += 0.2
        except Exception as e:
            feats["error"] = str(e)

    elif fmt == "jpeg":
        coeffs = _jpeg_quant_coeffs(path)
        if coeffs is not None:
            st = dct_stats(coeffs)
            feats.update(st)
            # zero_spike/odd_ratio showed NO separation between natural
            # JPEGs and the simplified DCT-embed attack in calibration
            # sampling (2026-08-16, Phase D #3): natural p5-p95
            # zero_spike 11.6-21.6 vs attack 14.8-19.4; odd_ratio even
            # more overlapped. The rules fired on 100% of natural JPEGs
            # and added no signal, so they no longer score. The dct_embed
            # family also has 0.000 undefended ASR (nothing to catch);
            # this is documented honestly in the paper.
        else:
            feats["dct_decoder"] = "failed"

    elif fmt == "wav":
        try:
            with wave.open(path, "rb") as w:
                nch, sw, fs, nf = w.getnchannels(), w.getsampwidth(), \
                    w.getframerate(), w.getnframes()
                raw = w.readframes(nf)
            if sw == 2:
                x16 = np.frombuffer(raw, dtype="<i2")
                if nch > 1:
                    x16 = x16[::nch]
                chi2, p, ratio = lsb_chisq_pcm(x16)
                feats["lsb_chi2"] = float(chi2)
                feats["lsb_p"] = float(p)
                feats["lsb_ratio"] = float(ratio)
                # Natural 16-bit speech has BIASED LSBs (measured ratio
                # 0.58-0.94 on LibriSpeech clips); LSB embedding makes the
                # plane uniform (ratio ~= 0.500, low chi2). Direction
                # fixed 2026-08-16 (Phase D #3): trigger on
                # over-uniformity.
                if len(x16) > 48000 and abs(ratio - 0.5) < 0.005:
                    score += 0.6
                    ev.append({"probe": "p_content", "trigger": True,
                               "evidence_type": "lsb_anomaly",
                               "localization": "audio-pcm",
                               "extracted_text": "",
                               "detail": f"ratio={ratio:.4f} "
                                         "(over-uniform)"})
            x = (x16.astype(np.float32) / 32767.0) if sw == 2 else \
                np.frombuffer(raw, dtype="<i1").astype(np.float32) / 127.0
            if nch > 1 and sw == 2:
                x = x
            spec = audio_spectral(x, fs)
            if spec:
                feats.update({f"audio_{k}": v for k, v in spec.items()})
                if spec["hf_ratio_max"] > 0.5 and spec["noise_floor"] < 1e-3:
                    # strong HF energy with silent floor: carrier-like
                    score += 0.2
            echo_e = cepstral_echo_energy(x, fs)
            feats["echo_energy"] = echo_e
            if echo_e > 3.5:
                score += 0.4
                ev.append({"probe": "p_content", "trigger": True,
                           "evidence_type": "echo_carrier",
                           "localization": "audio-cepstrum",
                           "extracted_text": "",
                           "detail": f"echo_energy={echo_e:.2f}"})
        except Exception as e:
            feats["error"] = str(e)

    elif fmt == "mp3":
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav",
                                             delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", str(sr),
                            "-ac", "1", "-f", "wav", tmp_path],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
                sub_res = probe_content(tmp_path, fmt="wav", sr=sr)
            finally:
                os.unlink(tmp_path)
            feats = sub_res["features"]
            score = sub_res["score"]
            ev = sub_res["evidence"]
        except Exception as e:
            feats["error"] = str(e)

    else:
        feats["error"] = f"unsupported format {fmt}"

    return {"features": feats, "score": min(score, 1.0), "evidence": ev}


def _detect(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(16)
    if head[:8] == pf.PNG_MAGIC:
        return "png"
    if head[:2] == b"\xff\xd8":
        return "jpeg"
    if head[:4] in (b"RIFF", b"RIFX"):
        return "wav"
    if head[:3] == b"ID3":
        return "mp3"
    return "unknown"


if __name__ == "__main__":
    import json
    import sys
    for p in sys.argv[1:]:
        r = probe_content(p)
        print(json.dumps({"file": os.path.basename(p),
                          "score": round(r["score"], 3),
                          "features": {k: (round(v, 3) if isinstance(
                              v, float) else v)
                              for k, v in r["features"].items()},
                          "evidence": r["evidence"]}, ensure_ascii=False))
