"""
Magi EEG Foundation Model — migrated into brain_moe_pinn.

Self-contained package with no external dependency on /home/yanlu/Documents/magi.
"""

from .encoder import EEGFoundationModel

try:
    from .encoder_v2 import EEGFoundationModelV2
    from .magi_v2 import MagiV2EEGEncoder, create_magi_v2_from_v1
    __all__ = [
        "EEGFoundationModel",
        "EEGFoundationModelV2",
        "MagiV2EEGEncoder",
        "create_magi_v2_from_v1",
    ]
except ImportError:
    __all__ = ["EEGFoundationModel"]
