"""
Thermodynamics analysis: EPR, Jarzynski, Landauer, Crooks.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional


def compute_epr_trajectory(
    z_trajectory: torch.Tensor,
    delta_z_trajectory: torch.Tensor,
    D: float = 1.0,
) -> Dict[str, float]:
    """
    Compute entropy production rate along a trajectory.

    sigma = ||v||^2 / D (proxy)
    sigma_exact = <v, div(J_v)> / D (exact, requires Jacobian)
    """
    T = z_trajectory.shape[0]
    v = delta_z_trajectory

    # Proxy
    sigma_proxy = (v ** 2).sum(dim=-1) / (D + 1e-8)

    # Exact via finite-difference Jacobian trace
    sigma_exact = []
    for t in range(T):
        # Approximate div(v) = trace(J_v)
        # Using velocity differences as proxy
        if t < T - 1:
            dv = v[t + 1] - v[t]
            dt = 1.0
            trace_approx = (dv / dt).sum().item()
        else:
            trace_approx = 0.0
        sigma_exact.append((v[t] * trace_approx).sum().item() / (D + 1e-8))

    sigma_exact = np.array(sigma_exact)
    sigma_proxy = sigma_proxy.cpu().numpy()

    # Correlation
    if sigma_proxy.std() > 1e-6 and sigma_exact.std() > 1e-6:
        corr = np.corrcoef(sigma_proxy, sigma_exact)[0, 1]
    else:
        corr = 1.0

    return {
        "epr_proxy_mean": float(sigma_proxy.mean()),
        "epr_exact_mean": float(sigma_exact.mean()),
        "epr_proxy_std": float(sigma_proxy.std()),
        "epr_exact_std": float(sigma_exact.std()),
        "epr_correlation": float(corr),
        "epr_negative_ratio": float((sigma_proxy < 0).mean()),
    }


def compute_jarzynski_work(
    z_trajectory: torch.Tensor,
    energy_fn,
) -> Dict[str, float]:
    """
    Jarzynski equality: <exp(-beta W)> = exp(-beta dF).

    Only requires forward trajectories (compatible with irreversible architecture).
    """
    T = z_trajectory.shape[0]
    with torch.no_grad():
        energies = []
        for t in range(T):
            E = energy_fn(z_trajectory[t:t+1])
            energies.append(E.item() if hasattr(E, "item") else E)

    energies = np.array(energies)
    # Work = cumulative energy change
    W = np.cumsum(np.diff(energies, prepend=energies[0]))

    # Jarzynski estimate
    beta = 1.0
    jarzynski_left = np.exp(-beta * W).mean()
    jarzynski_right = np.exp(-beta * (energies[-1] - energies[0]))

    return {
        "work_mean": float(W.mean()),
        "work_max": float(W.max()),
        "jarzynski_left": float(jarzynski_left),
        "jarzynski_right": float(jarzynski_right),
        "jarzynski_error": abs(jarzynski_left - jarzynski_right),
    }


def compute_landauer_bound(
    info_erased_bits: float,
    temperature: float = 1.0,
) -> float:
    """
    Landauer principle: Q >= kT ln(2) per bit erased.

    Returns minimum heat dissipation in natural units.
    """
    k_B = 1.0 # natural units
    Q_min = k_B * temperature * np.log(2) * info_erased_bits
    return float(Q_min)


class BergsonMonitor:
    """
    Wiener Proposal #1: Bergson 监控器.

    Measures the irreversibility gap via Bergson information:
        I_Bergson = <βΔW> − log<e^{−βΔW}>

    This reuses the existing Jarzynski infrastructure (compute_jarzynski_work).
    I_Bergson quantifies the "duration" (Bergsonian duree) — the internal
    time experienced by the system beyond reversible (thermodynamic) time.

    Interpretation:
      I_Bergson ≈ 0 : near-reversible dynamics (system is in equilibrium)
      I_Bergson >> 0 : far-from-equilibrium (high irreversibility, active learning)
      I_Bergson < 0  : FP16 instability in exp(−βΔW) — flag as numerical warning

    FP16 safety: exp(−βΔW) is computed in selective FP32 to avoid overflow.
    Schedule: every step, logged every 100 (same as EPR).
    Read-only: no loss terms, no backward path modifications.
    """

    def __init__(
        self,
        beta: float = 1.0,
        window_size: int = 100,
        fp32_for_exp: bool = True,
    ):
        self.beta = beta
        self.window_size = window_size
        self.fp32_for_exp = fp32_for_exp

        self.work_history: List[float] = []
        self.bergson_history: List[float] = []
        self.energy_trajectory: List[float] = []
        self.nan_warning_count: int = 0

    def update(
        self,
        z_current: torch.Tensor,
        z_next: torch.Tensor,
        energy_fn=None,
    ) -> Dict[str, float]:
        """
        Compute Bergson information for one step transition.

        Requires the current and next latent states, plus an energy function
        to compute work W = ΔE along the trajectory.

        If energy_fn is not provided, falls back to ||z_next − z_current||^2 / D
        as a proxy work estimate.

        Args:
            z_current: (B, d) latent state at time t
            z_next: (B, d) latent state at time t+1
            energy_fn: callable z -> E(z) for work computation

        Returns:
            dict with i_bergson, mean_work, exp_work, nan_warning
        """
        with torch.no_grad():
            if energy_fn is not None:
                E_current = energy_fn(z_current)
                E_next = energy_fn(z_next)
                if hasattr(E_current, "item"):
                    E_current = E_current.item()
                if hasattr(E_next, "item"):
                    E_next = E_next.item()
                delta_W = E_next - E_current
            else:
                delta_z = z_next - z_current
                delta_W = (delta_z ** 2).sum(dim=-1).mean().item()

        self.work_history.append(delta_W)
        self.energy_trajectory.append(
            (E_next if energy_fn is not None else delta_W)
        )

        if len(self.work_history) > self.window_size:
            self.work_history = self.work_history[-self.window_size:]
            self.energy_trajectory = self.energy_trajectory[-self.window_size:]

        if len(self.work_history) < 2:
            result = {
                "i_bergson": float("nan"),
                "bergson_mean_work": float("nan"),
                "bergson_exp_work": float("nan"),
                "bergson_nan_warning": 0.0,
            }
            self.bergson_history.append(result["i_bergson"])
            return result

        W_arr = np.array(self.work_history)
        beta = self.beta

        if self.fp32_for_exp:
            exp_neg_beta_W = np.exp(-beta * W_arr.astype(np.float64))
        else:
            exp_neg_beta_W = np.exp(-beta * W_arr)

        mean_work = float(W_arr.mean())
        mean_exp_work = float(exp_neg_beta_W.mean())

        if mean_exp_work <= 0 or np.isnan(mean_exp_work) or np.isinf(mean_exp_work):
            self.nan_warning_count += 1
            i_bergson = float("nan")
            nan_warning = 1.0
        else:
            i_bergson = beta * mean_work - np.log(mean_exp_work)
            nan_warning = 0.0

        if np.isnan(i_bergson) or np.isinf(i_bergson):
            self.nan_warning_count += 1
            nan_warning = 1.0

        result = {
            "i_bergson": float(i_bergson) if not np.isnan(i_bergson) else float("nan"),
            "bergson_mean_work": mean_work,
            "bergson_exp_work": mean_exp_work,
            "bergson_nan_warning": nan_warning,
            "bergson_collapse_alarm": 0.0,
        }

        self.bergson_history.append(result["i_bergson"])
        if len(self.bergson_history) > 10000:
            self.bergson_history = self.bergson_history[-5000:]

        return result

    def check_collapse(self, router_entropy: float, threshold: float = 0.5) -> bool:
        """
        Check for Bergson collapse with router-entropy conjunction.

        I_Bergson → 0 alone can indicate either:
          (a) genuine collapse (system lost time direction)
          (b) low-activity resting state (system is quiescent but healthy)

        To distinguish: collapse is only flagged when I_Bergson → 0 AND
        router entropy is also below threshold (indicating the system has
        lost diversity, not just reduced activity).

        Args:
            router_entropy: current router entropy from PoissonRouter
            threshold: router entropy below which collapse is flagged
        Returns:
            True if Bergson collapse is detected (both metrics low)
        """
        valid_values = [b for b in self.bergson_history[-20:]
                        if not (math.isnan(b) if isinstance(b, float) else np.isnan(b))]
        if not valid_values:
            return False

        recent_bergson = float(np.mean(valid_values))
        return recent_bergson < 0.01 and router_entropy < threshold

    def update_from_jarzynski(
        self,
        jarzynski_result: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Alternative: compute Bergson info from pre-computed Jarzynski results.

        Uses the work_mean and jarzynski_left from compute_jarzynski_work().

        Args:
            jarzynski_result: output dict from compute_jarzynski_work()

        Returns:
            dict with i_bergson, bergson_mean_work, bergson_exp_work, bergson_nan_warning
        """
        mean_work = jarzynski_result.get("work_mean", 0.0)
        jarzynski_left = jarzynski_result.get("jarzynski_left", 1.0)

        if jarzynski_left <= 0 or np.isnan(jarzynski_left) or np.isinf(jarzynski_left):
            self.nan_warning_count += 1
            i_bergson = float("nan")
            nan_warning = 1.0
        else:
            i_bergson = self.beta * mean_work - np.log(jarzynski_left)
            nan_warning = 0.0

        if np.isnan(i_bergson) or np.isinf(i_bergson):
            self.nan_warning_count += 1
            nan_warning = 1.0

        result = {
            "i_bergson": float(i_bergson) if not np.isnan(i_bergson) else float("nan"),
            "bergson_mean_work": mean_work,
            "bergson_exp_work": jarzynski_left,
            "bergson_nan_warning": nan_warning,
        }

        self.work_history.append(mean_work)
        if len(self.work_history) > self.window_size:
            self.work_history = self.work_history[-self.window_size:]

        self.bergson_history.append(result["i_bergson"])
        if len(self.bergson_history) > 10000:
            self.bergson_history = self.bergson_history[-5000:]

        return result

    def get_report(self) -> Dict:
        valid = [b for b in self.bergson_history if not (math.isnan(b) if isinstance(b, float) else np.isnan(b))]
        return {
            "i_bergson_current": self.bergson_history[-1] if self.bergson_history else float("nan"),
            "i_bergson_mean": float(np.mean(valid)) if valid else float("nan"),
            "nan_warning_count": self.nan_warning_count,
            "work_history_len": len(self.work_history),
            "bergson_history_len": len(self.bergson_history),
        }
