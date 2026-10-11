"""Fusion head: calibrated lightweight classifier over the three probe
scores + auxiliary features. Logistic regression with isotonic/Platt
calibration on the validation split."""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "p_file",           # probe scores
    "p_content",
    "p_intent",
    "synth_speech",     # synthetic-speech detector score
    "align_img_user",   # three-way consistency
    "align_aud_user",
    "align_img_aud",
    "drift_img",        # ZEDD-style embedding drift
    "ocr_vlm_disagreement",  # 4th channel
    "danger_hits",      # taxonomy
    "text_len",         # extracted text length (controls)
]


def build_feature_vector(probe_results, extracted, synth_score=None):
    """Assemble the fused feature vector from probe outputs."""
    feats = {}
    p_file = probe_results.get("p_file")
    p_content = probe_results.get("p_content")
    p_intent = probe_results.get("p_intent")
    feats["p_file"] = float(p_file.score if p_file else 0.0)
    feats["p_content"] = float(p_content["score"] if p_content else 0.0)
    feats["p_intent"] = float(p_intent["score"] if p_intent else 0.0)
    feats["synth_speech"] = float(synth_score if synth_score is not None
                                  else 0.5)
    if p_intent:
        f = p_intent["features"]
        for k in ("align_img_user", "align_aud_user", "align_img_aud",
                  "drift_img", "ocr_vlm_disagreement", "danger_hits"):
            v = f.get(k)
            feats[k] = float(v) if v is not None else 0.5
    else:
        for k in ("align_img_user", "align_aud_user", "align_img_aud",
                  "drift_img", "ocr_vlm_disagreement", "danger_hits"):
            feats[k] = 0.5
    text = extracted or ""
    feats["text_len"] = float(len(text))
    return feats


def to_vector(feats, names=FEATURE_NAMES):
    v = np.zeros(len(names))
    for i, n in enumerate(names):
        x = feats.get(n)
        v[i] = float(x) if x is not None else 0.5
    return v


class FusionHead:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.calibrator = None
        self.fitted = False
        self.names = FEATURE_NAMES

    def fit(self, X_dicts, y, calibrate=False):
        if not X_dicts or len(X_dicts) != len(y) or len(set(y)) != 2:
            raise ValueError("Training data must contain both labels and matching features")
        if calibrate:
            raise ValueError("Pass held-out data to head.calibrate separately")
        X = np.stack([to_vector(d, self.names) for d in X_dicts])
        y = np.asarray(y)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        if calibrate:
            raise ValueError("Use head.calibrate(validation_features, validation_labels)")
        self.calibrator = None
        self.fitted = True
        return self

    def calibrate(self, X_dicts, y):
        if not self.fitted:
            raise ValueError("Fit the classifier before calibration")
        if not X_dicts or len(X_dicts) != len(y) or len(set(y)) != 2:
            raise ValueError("Validation data must contain both labels and matching features")
        scores = self.predict_scores(X_dicts)
        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(scores, y)
        return self

    def predict_scores(self, X_dicts):
        X = np.stack([to_vector(d, self.names) for d in X_dicts])
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]

    def predict_calibrated(self, X_dicts):
        s = self.predict_scores(X_dicts)
        if self.calibrator is not None:
            s = self.calibrator.predict(s)
        return s

    def decision_boundary_uncertainty(self, X_dicts, t1, t2):
        """Uncertainty criterion for L2 escalation: distance of the raw
        score from the nearest decision threshold, normalized."""
        s = self.predict_scores(X_dicts)
        d = np.minimum(np.abs(s - t1), np.abs(s - t2))
        span = max(t2 - t1, 1e-6)
        return 1.0 - np.clip(d / span, 0, 1)

    def save(self, path):
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "clf": self.clf,
                     "calibrator": self.calibrator,
                     "names": self.names}, path)

    @staticmethod
    def load(path):
        import joblib
        d = joblib.load(path)
        h = FusionHead()
        h.scaler = d["scaler"]
        h.clf = d["clf"]
        h.calibrator = d["calibrator"]
        h.names = d["names"]
        h.fitted = True
        return h


def choose_thresholds(cal_scores, cal_y, t1=0.25, t2=0.75, fpr_target=0.05):
    """Pick t1/t2 so that (score < t1) has near-zero FNR on attacks and
    (score > t2) has near-zero FPR on benign, leaving the middle to L2."""
    if len(cal_scores) == 0:
        return t1, t2
    atk = [s for s, y in zip(cal_scores, cal_y) if y == 1]
    ben = [s for s, y in zip(cal_scores, cal_y) if y == 0]
    if atk and ben:
        t1 = min(t1, float(np.percentile(atk, 5)))   # 95% attacks >= t1
        t2 = max(t2, float(np.percentile(ben, 100 * (1 - fpr_target))))
    return float(t1), float(t2)
