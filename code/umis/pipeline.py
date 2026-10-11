"""UMIS end-to-end pipeline: Stage 0 front-ends -> P-file -> P-content ->
P-intent -> cost-aware cascade L0/L1/L2 -> verdict + probe report."""
import json
import os
import time

from .fusion import (FusionHead, build_feature_vector, choose_thresholds,
                     to_vector)
from .probes import p_content, p_file
from .taxonomy import danger_taxonomy

# L0 safety-critical keyword heuristics (internal plumbing, NOT a research
# contribution; based on the dangerous-action taxonomy)
L0_BLOCK_PATTERNS = [
    "send all", "upload all", "delete all", "erase all", "wipe all",
    "remove all files", "execute command", "run shell", "reveal your system",
    "ignore previous", "ignore all previous", "you are now dan",
    "no restrictions", "developer mode", "exfiltrate", "transfer 10000",
    "password", "secret key",
]


class UMIS:
    def __init__(self, ocr, asr, synth_det=None, encoder=None,
                 fusion=None, t1=0.25, t2=0.75, l2_enabled=True,
                 randomization=0.0, rng=None, benign_centroid=None,
                 benign_inv_cov=None, judge=None):
        if not (0 <= t1 < t2 <= 1):
            raise ValueError("Require 0 <= t1 < t2 <= 1")
        self.ocr = ocr
        self.asr = asr
        self.synth_det = synth_det
        self.encoder = encoder
        self.judge = judge
        self.fusion = fusion
        self.t1, self.t2 = t1, t2
        self.l2_enabled = l2_enabled
        self.randomization = randomization
        self.rng = rng
        self.benign_centroid = benign_centroid
        self.benign_inv_cov = benign_inv_cov
        self.stats = {"l0": 0, "l1": 0, "l2": 0, "total": 0}

    # ---------------------------------------------------------------- run --
    def process(self, input_id, image_path=None, audio_path=None,
                user_text="", ground_truth=None, ablate=None, skip_l0=False):
        if not image_path and not audio_path:
            raise ValueError("Provide an image or audio file; text-only inputs are unsupported")
        t_start = time.time()
        modality = []
        if image_path:
            modality.append("image")
        if audio_path:
            modality.append("audio")
        modality = "+".join(modality) or "text"
        stages = []
        escalation_used = 0

        # ---------------- Stage 0: extraction (OCR / ASR) ----------------
        ocr_text, asr_text = "", ""
        if image_path:
            ocr_text = self.ocr.full_text(image_path)
            for r in self.ocr.extract(image_path):
                if r["text"].strip():
                    stages.append({
                        "probe": "ocr", "trigger": True,
                        "evidence_type": "text_region",
                        "localization": [round(x) for x in r["bbox"]],
                        "extracted_text": r["text"][:200]})
        if audio_path:
            tr = self.asr.transcribe(audio_path)
            asr_text = tr["text"]
            for seg in tr["segments"]:
                stages.append({
                    "probe": "asr", "trigger": True,
                    "evidence_type": "speech_segment",
                    "localization": [round(seg["start"], 2),
                                     round(seg["end"], 2)],
                    "extracted_text": seg["text"][:200]})
        extracted = " ".join(t for t in (ocr_text, asr_text) if t)

        # ---------------- Stage 1: P-file --------------------------------
        pf_res = None
        pf_entries = []
        file_texts = []
        for media_path in (image_path, audio_path):
            if not media_path:
                continue
            r = p_file.probe_file(media_path)
            if pf_res is None or r.score > pf_res.score:
                pf_res = r
            for e in r.evidence:
                stages.append(e)
            pf_entries.extend(r.evidence)
            # extracted payload bytes (metadata tags, trailing data) join the
            # extraction pool so L0 heuristics and the P-intent taxonomy see
            # file-layer payloads too (per the method spec: "extracted
            # payload bytes if any" as P-file output)
            if r.trailing_text.strip():
                file_texts.append(r.trailing_text)
            for tp in r.metadata_texts:
                if tp.strip():
                    file_texts.append(tp)
        if file_texts:
            extracted = " ".join(
                [t for t in (extracted, " ".join(file_texts)) if t])

        # ---------------- L0 cheap rules (every input) -------------------
        block_now = False
        block_reasons = []
        if pf_res is not None:
            # safety-critical container evidence -> immediate BLOCK.
            # Modified 2026-08-16 (Phase D #1): a structural anomaly alone
            # (e.g. a legitimate "created with PIL" trailing comment, which
            # benign twins carry by design) must NOT insta-block; only
            # payload text that matches the dangerous-action taxonomy does.
            # Plain structural payloads still feed the L1 scoring.
            for e in pf_entries:
                if e["evidence_type"] not in ("trailing_data",
                                              "noncanonical_block"):
                    continue
                txt = e.get("extracted_text", "").strip()
                if not txt:
                    continue
                low = txt.lower()
                if any(pat in low for pat in L0_BLOCK_PATTERNS) or \
                        danger_taxonomy(txt):
                    block_now = True
                    block_reasons.append(f"p_file:{e['evidence_type']}")
        low = extracted.lower()
        for pat in L0_BLOCK_PATTERNS:
            if pat in low:
                block_now = True
                block_reasons.append(f"l0_keyword:{pat}")
                break

        risk_score = 1.0 if block_now else None
        verdict = None
        if block_now and not skip_l0:
            verdict = "BLOCK"
            self.stats["l0"] += 1
            self.stats["total"] += 1
            return self._report(input_id, modality, stages, 1.0, verdict,
                                escalation_used, time.time() - t_start,
                                block_reasons, user_text, extracted)

        # ---------------- Stage 2: P-content -----------------------------
        pc_res = None
        synth_score = None
        if image_path:
            pc_res = p_content.probe_content(image_path)
            for e in pc_res["evidence"]:
                stages.append(e)
        if audio_path:
            pc_aud = p_content.probe_content(audio_path)
            if pc_res is None:
                pc_res = pc_aud
            else:
                pc_res["score"] = max(pc_res["score"], pc_aud["score"])
                pc_res["features"].update(
                    {f"audio_{k}": v for k, v in pc_aud["features"].items()})
                pc_res["evidence"] += pc_aud["evidence"]
            for e in pc_aud["evidence"]:
                stages.append(e)
            if self.synth_det is not None:
                try:
                    _, synth_score = self.synth_det.score(audio_path)
                except Exception:
                    synth_score = 0.5
                if synth_score > 0.7:
                    stages.append({
                        "probe": "synth_speech", "trigger": True,
                        "evidence_type": "synthetic_speech",
                        "localization": "audio",
                        "extracted_text": "",
                        "detail": f"p_spoof={synth_score:.3f}"})

        # ---------------- Stage 3: P-intent ------------------------------
        pi_res = None
        if self.encoder is not None:
            from .probes import p_intent
            pi_res = p_intent.probe_intent(
                self.encoder, image_path=image_path, audio_path=audio_path,
                user_text=user_text, ocr_text=" ".join([ocr_text] + file_texts), asr_text=asr_text,
                benign_centroid=self.benign_centroid,
                benign_inv_cov=self.benign_inv_cov)
            for e in pi_res["evidence"]:
                stages.append(e)

        # ---------------- L1: fused scoring ------------------------------
        self.stats["l1"] += 1
        self.stats["total"] += 1
        probe_results = {"p_file": pf_res, "p_content": pc_res,
                         "p_intent": pi_res}
        feats = build_feature_vector(probe_results, extracted,
                                     synth_score=synth_score)
        # ablations: neutralize the ablated probe's contribution
        if ablate:
            for ab in ablate:
                if ab == "p_file":
                    feats["p_file"] = 0.0
                elif ab == "p_content":
                    feats["p_content"] = 0.0
                    feats["synth_speech"] = 0.5
                elif ab == "p_intent":
                    feats["p_intent"] = 0.0
                    for k in ("align_img_user", "align_aud_user",
                              "align_img_aud", "drift_img",
                              "ocr_vlm_disagreement", "danger_hits"):
                        feats[k] = 0.5 if k != "danger_hits" else 0.0
                elif ab == "synth_speech":
                    # synthetic-speech detector on/off (audio channel)
                    feats["synth_speech"] = 0.5
                elif ab == "drift":
                    # embedding-drift feature on/off (P-intent)
                    feats["drift_img"] = 0.5
                elif ab == "consistency":
                    # three-way consistency -> single alignment: keep
                    # only image-vs-user, neutralize audio-vs-user and
                    # image-vs-audio (2026-08-16 addition)
                    for k in ("align_aud_user", "align_img_aud"):
                        feats[k] = 0.5
        if self.fusion is not None and self.fusion.fitted:
            risk_score = float(self.fusion.predict_calibrated([feats])[0])
        else:
            # fallback: mean of probe scores (pre-calibration heuristic)
            risk_score = float(np_mean([
                feats["p_file"], feats["p_content"], feats["p_intent"]]))

        # randomization (ablation): jitter thresholds / feature dropout
        if self.randomization > 0 and self.rng is not None:
            risk_score += self.rng.uniform(-self.randomization,
                                           self.randomization)
            risk_score = max(0.0, min(1.0, risk_score))

        # ---------------- L2: heavy judge (borderline only) ---------------
        if (risk_score < self.t1) and not (
                pi_res is not None and
                (pi_res["features"].get("danger_hits") or 0) > 0):
            verdict = "ALLOW+AUDIT"
        elif risk_score >= self.t2:
            verdict = "BLOCK"
        else:
            # borderline: uncertainty-based escalation to the L2 judge
            escalation_used = 1
            if self.l2_enabled and self.judge is not None:
                self.stats["l2"] += 1
                judge_score = self.judge.judge(extracted, user_text,
                                               self._stages_summary(stages))
                if judge_score is not None and judge_score >= 0.5:
                    risk_score = max(risk_score, self.t2)
                    verdict = "BLOCK"
                elif judge_score is not None:
                    verdict = "ALLOW+AUDIT"
                else:
                    verdict = "QUARANTINE+CONFIRM"
            else:
                verdict = "QUARANTINE+CONFIRM"

        return self._report(input_id, modality, stages, risk_score, verdict,
                            escalation_used, time.time() - t_start, [], user_text, extracted)

    # ------------------------------------------------------------ report --
    def _report(self, input_id, modality, stages, risk_score, verdict,
                escalation_used, latency, block_reasons, user_text="", extracted=""):
        conflict_pairs = [s for s in stages
                          if s.get("evidence_type") in
                          ("cross_modal_conflict", "intent_conflict",
                           "ocr_vlm_disagreement")]
        return {
            "user_text": user_text,
            "extracted_text": extracted,
            "input_id": input_id,
            "modality": modality,
            "stages": stages,
            "conflict_pairs": conflict_pairs,
            "risk_score": round(float(risk_score), 4),
            "verdict": verdict,
            "escalation_used": escalation_used,
            "latency_s": round(latency, 3),
            "block_reasons": block_reasons,
        }

    @staticmethod
    def _stages_summary(stages):
        return [{"probe": s["probe"], "trigger": s["trigger"],
                 "evidence_type": s["evidence_type"],
                 "localization": s["localization"],
                 "extracted_text": s.get("extracted_text", "")[:120]}
                for s in stages]


def np_mean(xs):
    return sum(xs) / max(len(xs), 1)


# ---------------------------------------------------------------------------
# calibration entry point (used by harness before evaluation)
# ---------------------------------------------------------------------------


def calibrate(feature_dicts, labels, val_idx, out_path,
              t1=0.25, t2=0.75, fpr_target=0.05):
    """Fit fusion head on train split, calibrate on validation split,
    choose thresholds. Saves the head + thresholds."""
    train_feats = [feature_dicts[i] for i in range(len(feature_dicts))
                   if i not in val_idx]
    train_y = [labels[i] for i in range(len(labels)) if i not in val_idx]
    val_feats = [feature_dicts[i] for i in val_idx]
    val_y = [labels[i] for i in val_idx]
    head = FusionHead().fit(train_feats, train_y, calibrate=False)
    head.calibrate(val_feats, val_y)
    head.save(out_path)
    cal_scores = head.predict_calibrated(val_feats)
    t1, t2 = choose_thresholds(cal_scores, val_y, t1, t2, fpr_target)
    return head, t1, t2
