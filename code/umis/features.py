"""Extract exactly the features used by UMIS L1, with L0 bypassed for training."""
from .pipeline import UMIS
from .fusion import build_feature_vector

class _Capture:
    fitted = True
    def predict_calibrated(self, features):
        self.features = features[0]
        return [0.0]

def extract_features(ocr, asr, encoder, image_path=None, audio_path=None,
                     user_text="", synth_det=None, benign_centroid=None,
                     benign_inv_cov=None):
    capture = _Capture()
    detector = UMIS(ocr, asr, synth_det=synth_det, encoder=encoder,
                    fusion=capture, l2_enabled=False,
                    benign_centroid=benign_centroid, benign_inv_cov=benign_inv_cov)
    detector.process("feature", image_path=image_path, audio_path=audio_path,
                     user_text=user_text, skip_l0=True)
    return capture.features
