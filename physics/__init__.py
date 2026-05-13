"""Physics analysis package for Brain MoE-PINN."""

from .energy_landscape import compute_waddington_landscape, compute_casimir_invariants
from .thermodynamics import compute_epr_trajectory, compute_jarzynski_work, compute_landauer_bound, BergsonMonitor
from .statistics import compute_tsallis_entropy, compute_1f_exponent, compute_lyapunov_exponents

__all__ = [
    "compute_waddington_landscape",
    "compute_casimir_invariants",
    "compute_epr_trajectory",
    "compute_jarzynski_work",
    "compute_landauer_bound",
    "BergsonMonitor",
    "compute_tsallis_entropy",
    "compute_1f_exponent",
    "compute_lyapunov_exponents",
]
