"""Training phases and loss definitions."""

from .training_phases import (
    TrainingStage, TrainingPhase, LossWeights, FreezeConfig,
    get_all_phases, get_phase_neg_1, get_stage_1_p1,
)
from .losses import (
    ReconstructionLoss, VelocitySmoothnessLoss, GenerickeConstraintLoss,
    MoELoadBalancingLoss, GrassmannianRegularization, JacobiRegularization,
    HebbianRegularization, NSPLoss, CrossModalAlignmentLoss,
    CrossSoftContrastiveLoss, InterventionResponseLoss, WeakSIGRegLoss,
    TotalLoss,
)

__all__ = [
    "TrainingStage", "TrainingPhase", "LossWeights", "FreezeConfig",
    "get_all_phases", "get_phase_neg_1", "get_stage_1_p1",
    "ReconstructionLoss", "VelocitySmoothnessLoss",
    "GenerickeConstraintLoss", "MoELoadBalancingLoss",
    "GrassmannianRegularization", "JacobiRegularization",
    "HebbianRegularization", "NSPLoss", "CrossModalAlignmentLoss",
    "CrossSoftContrastiveLoss", "InterventionResponseLoss",
    "WeakSIGRegLoss", "TotalLoss",
]
