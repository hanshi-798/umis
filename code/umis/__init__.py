"""UMIS: Unified Multimodal Injection Sentinel."""
__version__ = "1.3.0"

def __getattr__(name):
    if name == "UMIS":
        from .pipeline import UMIS
        return UMIS
    raise AttributeError(name)
