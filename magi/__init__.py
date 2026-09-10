"""
Magi EEG Foundation Model — migrated into brain_moe_pinn.

Self-contained package with no external dependency on /home/yanlu/Documents/magi.
"""

from .momentum import MomentumEncoder

__all__ = ["MomentumEncoder"]

try:
    from .encoder import EEGFoundationModel
except ImportError:
    EEGFoundationModel = None
else:
    __all__.append("EEGFoundationModel")

try:
    from .encoder_v2 import EEGFoundationModelV2
except ImportError:
    EEGFoundationModelV2 = None
else:
    __all__.append("EEGFoundationModelV2")

try:
    from .magi_v2 import MagiV2EEGEncoder, create_magi_v2_from_v1
except ImportError:
    MagiV2EEGEncoder = None
    create_magi_v2_from_v1 = None
else:
    __all__.extend(["MagiV2EEGEncoder", "create_magi_v2_from_v1"])
