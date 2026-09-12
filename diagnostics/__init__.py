"""Offline diagnostics for controlled dynamics and generated trajectories."""

from .causal_dynamics import (
    rollout_controlled, intervention_response_metrics,
    forward_reverse_prediction_gap,
)
from .free_run_metrics import (
    corr_matrix_mse, variance_ratio, autocorr_profile_mse,
    state_path_stats, chattering_index, run_free_run_suite,
)
from .species_parameters import (
    TauEstimate, estimate_tau, estimate_ar_order_gain, fit_coupling_matrix,
    coupling_split, zero_lag_network_r2, estimate_ladder_parameters,
)
from .scaling_probe import (
    information_horizon, spectral_floor, variance_maintenance,
    run_scale_ablation,
)

__all__ = [
    "rollout_controlled", "intervention_response_metrics",
    "forward_reverse_prediction_gap", "corr_matrix_mse", "variance_ratio",
    "autocorr_profile_mse", "state_path_stats", "chattering_index",
    "run_free_run_suite", "information_horizon", "spectral_floor",
    "variance_maintenance", "run_scale_ablation",
    "TauEstimate", "estimate_tau", "estimate_ar_order_gain",
    "fit_coupling_matrix", "coupling_split", "zero_lag_network_r2",
    "estimate_ladder_parameters",
]
