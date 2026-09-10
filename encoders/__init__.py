from .eeg_encoder import EEGEncoderWrapper, EEGProjection
from .fmri_encoder import (
    NeuroSTORMEncoder,
    BrainLMEncoder,
    create_fmri_encoder,
)
from .meg_encoder import MEGEncoderWrapper, MEGProjection
from .hub_fusion import HubTokenFusion, CrossModalAdapter, SlowManifoldProjector, LatentHRFBridge

def __getattr__(name):
    if name == "EEGEncoderWrapperV2":
        try:
            from .eeg_encoder_v2 import EEGEncoderWrapperV2
        except ImportError as exc:
            raise AttributeError(
                "EEGEncoderWrapperV2 requires the Magi v2 dependencies"
            ) from exc
        return EEGEncoderWrapperV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EEGEncoderWrapper",
    "EEGProjection",
    "EEGEncoderWrapperV2",
    "NeuroSTORMEncoder",
    "BrainLMEncoder",
    "create_fmri_encoder",
    "MEGEncoderWrapper",
    "MEGProjection",
    "HubTokenFusion",
    "CrossModalAdapter",
    "SlowManifoldProjector",
    "LatentHRFBridge",
]