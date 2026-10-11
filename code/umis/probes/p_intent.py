"""Probe P-intent (Stage 3): shared semantic layer.

Encodes image / audio / text into a shared embedding space with the
Qwen2.5-Omni Thinker encoder (fallback CLIP+CLAP), then computes:
  - three-way pairwise alignment among image-derived instruction, audio-
    derived instruction, and the user's typed intent (conflict = signal)
  - OCR-reading vs VLM semantic description disagreement (4th channel)
  - embedding drift from the benign-train distribution (ZEDD-style)
  - dangerous-action taxonomy hits (task-execution link)
Output: features, score, and evidence entries with localized conflict pairs.
"""
import hashlib
import json
import os
import re

import numpy as np
import torch


# ---------------------------------------------------------------------------
# dangerous-action taxonomy (aligned with the task-execution link)
# ---------------------------------------------------------------------------

from ..taxonomy import DANGER_PATTERNS, danger_taxonomy


# ---------------------------------------------------------------------------
# shared encoder
# ---------------------------------------------------------------------------


class SharedEncoder:
    """Qwen2.5-Omni Thinker encoder (frozen). Fallback: CLIP + CLAP."""

    def __init__(self, mode="omni", model_path=None, device=None,
                 cache_dir=None):
        self.mode = mode
        self.cache_namespace = f"encoder:{mode}:{model_path}:{os.environ.get('CLAP_CKPT', '')}"
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.cache_dir = cache_dir
        if mode == "omni":
            from transformers import (AutoProcessor,
                                      Qwen2_5OmniForConditionalGeneration)
            path = model_path or "Qwen/Qwen2.5-Omni-7B"
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="eager")
            self.processor = AutoProcessor.from_pretrained(path)
            self.model.eval()
            # freeze
            for p in self.model.parameters():
                p.requires_grad_(False)
        elif mode == "clip_clap":
            from transformers import CLIPModel, CLIPProcessor
            self.clip = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(self.device).eval()
            self.clip_proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")
            import laion_clap
            self.clap = laion_clap.CLAP_Module(enable_fusion=False,
                                               device=self.device)
            # laion_clap (lukewys) API: load_ckpt(ckpt=local_path). Its
            # built-in download goes to huggingface.co directly, which is
            # unreachable from this datacenter - the checkpoint must be
            # pre-downloaded via the hf-mirror and passed via CLAP_CKPT.
            # The tokenizer (roberta-base) is resolved from the HF hub
            # cache at init; keep HF_HOME/HF_ENDPOINT exported.
            ckpt = os.environ.get("CLAP_CKPT")
            if not ckpt or not os.path.isfile(ckpt):
                raise FileNotFoundError("Set CLAP_CKPT to a local LAION-CLAP checkpoint")
            self.clap.load_ckpt(ckpt=ckpt, model_id=-1, verbose=False)
        else:
            raise ValueError(mode)

    def _cache_path(self, key, kind):
        if not self.cache_dir:
            return None
        from ..cache import media_key
        ns = f"{self.cache_namespace}:{kind}"
        h = (media_key(key, ns) if kind in ("img", "aud") else
             hashlib.sha256((ns + key).encode()).hexdigest())
        return os.path.join(self.cache_dir, f"{kind}_{h}.npy")

    def _load_cache(self, cp):
        if cp and os.path.exists(cp):
            return np.load(cp)
        return None

    def _save_cache(self, cp, arr):
        if cp:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            np.save(cp, arr)

    @torch.no_grad()
    def embed_image(self, image_path):
        cp = self._cache_path(image_path, "img")
        cached = self._load_cache(cp)
        if cached is not None:
            return cached
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        if self.mode == "omni":
            with torch.autocast(self.device, torch.bfloat16):
                inputs = self.processor(
                    text=["Describe this image."], images=[img],
                    padding=True, return_tensors="pt").to(self.device)
                out = self.model.thinker(**inputs, output_hidden_states=True,
                                         return_dict=True)
                hs = out.hidden_states[-1]
                emb = hs.mean(dim=1)[0].float().cpu().numpy()
        else:
            pi = self.clip_proc(images=img, return_tensors="pt").to(
                self.device)
            emb = self.clip.get_image_features(**pi)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output  # transformers >= 5 returns a model output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb[0].float().cpu().numpy()
        self._save_cache(cp, emb)
        return emb

    @torch.no_grad()
    def embed_audio(self, audio_path):
        cp = self._cache_path(audio_path, "aud")
        cached = self._load_cache(cp)
        if cached is not None:
            return cached
        import soundfile as sf
        x, sr = sf.read(audio_path)
        if len(x.shape) > 1:
            x = x.mean(axis=1)
        if self.mode == "omni":
            with torch.autocast(self.device, torch.bfloat16):
                inputs = self.processor(
                    text=["Transcribe this audio."],
                    audios=[x], sampling_rate=sr, padding=True,
                    return_tensors="pt").to(self.device)
                out = self.model.thinker(**inputs, output_hidden_states=True,
                                         return_dict=True)
                hs = out.hidden_states[-1]
                emb = hs.mean(dim=1)[0].float().cpu().numpy()
        else:
            # laion_clap expects a BATCH of waveforms (N, T) at the model's
            # native sample rate (48 kHz); fixed 2026-08-15 (the installed
            # laion_clap version requires a batch and assumes 48k input).
            if sr != 48000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=48000)
            emb = np.asarray(self.clap.get_audio_embedding_from_data(
                x=x[None, :], use_tensor=False))[0]
        self._save_cache(cp, emb)
        return emb

    @torch.no_grad()
    def embed_text(self, text):
        cp = self._cache_path("txt:" + text, "txt")
        cached = self._load_cache(cp)
        if cached is not None:
            return cached
        if self.mode == "omni":
            with torch.autocast(self.device, torch.bfloat16):
                inputs = self.processor(
                    text=[text], padding=True,
                    return_tensors="pt").to(self.device)
                out = self.model.thinker(**inputs, output_hidden_states=True,
                                         return_dict=True)
                hs = out.hidden_states[-1]
                emb = hs.mean(dim=1)[0].float().cpu().numpy()
        else:
            pi = self.clip_proc(text=[text], return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
            emb = self.clip.get_text_features(**pi)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output  # transformers >= 5 returns a model output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb[0].float().cpu().numpy()
        self._save_cache(cp, emb)
        return emb

    @torch.no_grad()
    def embed_text_clap(self, text):
        """CLAP text-tower embedding (audio-side counterpart of
        embed_text). CLIP and CLAP live in DIFFERENT embedding spaces, so
        the audio-vs-user alignment must use the CLAP text tower
        (fixed 2026-08-15; using CLIP text embeddings there produced
        meaningless cosine values). Omni mode does not need this (single
        shared space)."""
        if self.mode != "clip_clap":
            return self.embed_text(text)
        cp = self._cache_path("claptxt:" + text, "claptxt")
        cached = self._load_cache(cp)
        if cached is not None:
            return cached
        emb = np.asarray(self.clap.get_text_embedding([text],
                                                      use_tensor=False))[0]
        self._save_cache(cp, emb)
        return emb

    @torch.no_grad()
    def describe_image(self, image_path, max_new=64):
        """VLM semantic description of an image (4th evidence channel)."""
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        if self.mode != "omni":
            return ""
        inputs = self.processor(
            text=["Briefly describe the visual content of this image, "
                  "including any text you can read."],
            images=[img], padding=True, return_tensors="pt").to(self.device)
        with torch.autocast(self.device, torch.bfloat16):
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                use_audio_in_video=False, return_dict_in_generate=False)
        resp_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(resp_ids, skip_special_tokens=True)

    @torch.no_grad()
    def judge(self, instruction_text, user_intent, probe_report):
        """L2 heavy judge: final verdict on borderline inputs."""
        if self.mode != "omni":
            return None
        return None  # Supply TextJudge separately for L2 arbitration.



def _cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.5
    return float((a @ b) / (na * nb))


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def probe_intent(encoder, image_path=None, audio_path=None,
                 user_text="", ocr_text=None, asr_text=None,
                 benign_centroid=None, benign_inv_cov=None,
                 drift_scale=1.0):
    """Compute P-intent features + score + evidence.

    benign_centroid / benign_inv_cov: ZEDD-style benign-train distribution
    (mean and inverse covariance of benign multimodal embeddings). If None,
    drift is not computed.
    """
    emb_img = encoder.embed_image(image_path) if image_path else None
    emb_aud = encoder.embed_audio(audio_path) if audio_path else None
    emb_usr = encoder.embed_text(user_text or "no user instruction")
    # audio-side alignment uses the CLAP text tower (CLIP/CLAP spaces are
    # disjoint in clip_clap mode; documented limitation of the encoder
    # swap, which is exactly what the encoder-swap ablation measures)
    emb_usr_aud = (encoder.embed_text_clap(user_text or
                                           "no user instruction")
                   if emb_aud is not None else None)

    feats = {}
    evidence = []

    # three-way pairwise alignment. cross-modal (img-aud) alignment only
    # exists in a single shared space (omni mode); in clip_clap mode it is
    # None (CLIP and CLAP spaces are disjoint).
    align_img_user = _cos(emb_img, emb_usr) if emb_img is not None else None
    align_aud_user = (_cos(emb_aud, emb_usr_aud)
                      if emb_aud is not None and emb_usr_aud is not None
                      else None)
    align_img_aud = None
    if encoder.mode != "clip_clap" and emb_img is not None and \
            emb_aud is not None:
        align_img_aud = _cos(emb_img, emb_aud)
    feats["align_img_user"] = align_img_user
    feats["align_aud_user"] = align_aud_user
    feats["align_img_aud"] = align_img_aud

    # cross-modal conflict evidence
    if align_img_aud is not None and align_img_aud < 0.15:
        evidence.append({
            "probe": "p_intent", "trigger": True,
            "evidence_type": "cross_modal_conflict",
            "localization": "image-vs-audio",
            "extracted_text": f"img={ocr_text or ''} | aud={asr_text or ''}",
            "detail": f"alignment={align_img_aud:.3f}"})
    if align_img_user is not None and align_img_user < 0.15:
        evidence.append({
            "probe": "p_intent", "trigger": True,
            "evidence_type": "intent_conflict",
            "localization": "image-vs-user",
            "extracted_text": f"img={ocr_text or ''} | usr={user_text}",
            "detail": f"alignment={align_img_user:.3f}"})
    if align_aud_user is not None and align_aud_user < 0.15:
        evidence.append({
            "probe": "p_intent", "trigger": True,
            "evidence_type": "intent_conflict",
            "localization": "audio-vs-user",
            "extracted_text": f"aud={asr_text or ''} | usr={user_text}",
            "detail": f"alignment={align_aud_user:.3f}"})

    # embedding drift (ZEDD-style)
    drift = None
    if benign_centroid is not None and emb_img is not None:
        d = emb_img - benign_centroid
        if benign_inv_cov is not None:
            drift = float(np.sqrt(max(d @ benign_inv_cov @ d, 0.0)))
        else:
            drift = float(np.linalg.norm(d))
        feats["drift_img"] = drift * drift_scale

    # OCR-vs-VLM disagreement (4th channel): only when an image exists
    vlm_desc = None
    if image_path and ocr_text and encoder.mode == "omni":
        vlm_desc = encoder.describe_image(image_path)
        ocr_emb = encoder.embed_text(ocr_text)
        desc_emb = encoder.embed_text(vlm_desc)
        disagree = 1.0 - _cos(ocr_emb, desc_emb)
        feats["ocr_vlm_disagreement"] = disagree
        if disagree > 0.55:
            evidence.append({
                "probe": "p_intent", "trigger": True,
                "evidence_type": "ocr_vlm_disagreement",
                "localization": "image",
                "extracted_text": f"ocr={ocr_text[:100]} | "
                                  f"vlm={vlm_desc[:100]}",
                "detail": f"disagreement={disagree:.3f}"})
    else:
        feats["ocr_vlm_disagreement"] = None

    # dangerous-action taxonomy on extracted content
    extracted = " ".join(t for t in (ocr_text or "", asr_text or "") if t)
    hits = danger_taxonomy(extracted)
    feats["danger_hits"] = len(hits)
    feats["danger_cats"] = [h[0] for h in hits]
    for cat, matched in hits:
        evidence.append({
            "probe": "p_intent", "trigger": True,
            "evidence_type": "dangerous_action",
            "localization": "extracted-text",
            "extracted_text": matched, "detail": cat})

    # heuristic score: conflict-driven + taxonomy
    score = 0.0
    if hits:
        score += min(0.35 + 0.15 * (len(hits) - 1), 0.6)
    n_conflicts = sum(1 for e in evidence if e["evidence_type"] in
                      ("cross_modal_conflict", "intent_conflict"))
    score += 0.25 * n_conflicts
    if feats.get("ocr_vlm_disagreement") is not None and \
            feats["ocr_vlm_disagreement"] > 0.55:
        score += 0.2
    feats["score_pre_fusion"] = min(score, 1.0)

    return {"features": feats, "score": min(score, 1.0),
            "evidence": evidence, "vlm_desc": vlm_desc}
