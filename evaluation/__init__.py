"""Evaluation package for Brain MoE-PINN."""

from .resting_state import evaluate_resting_state, compute_psd_slope, compute_avalanche_stats
from .task_switching import evaluate_basin_transfer, evaluate_expert_reorganization
from .memory_tests import (
    evaluate_pattern_completion,
    evaluate_replay_fidelity,
    evaluate_forgetting,
    evaluate_energy_landscape,
)

__all__ = [
    "evaluate_resting_state",
    "compute_psd_slope",
    "compute_avalanche_stats",
    "evaluate_basin_transfer",
    "evaluate_expert_reorganization",
    "evaluate_pattern_completion",
    "evaluate_replay_fidelity",
    "evaluate_forgetting",
    "evaluate_energy_landscape",
]
