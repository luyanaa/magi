"""
Thermodynamics-related diagnostics and monitoring utilities.

Trajectory diagnostics are intentionally labelled as proxies unless a
complete stochastic-process definition makes a thermodynamic quantity
identifiable.
"""

import torch
import numpy as np
from typing import Dict, List


def compute_trajectory_diagnostics(
    z_trajectory: torch.Tensor,
    delta_z_trajectory: torch.Tensor,
    D: float = 1.0,
    div_drift_fn=None,
) -> Dict[str, float]:
    """Compute trajectory diagnostics without claiming thermodynamic EPR.

    The observed trajectory alone does not identify stochastic entropy
    production.  This function therefore reports only explicit diagnostics:

      velocity_proxy = ||delta_z||^2 / D
      signed_work_proxy = <delta_z, d(delta_z)/dt> / D
      drift_divergence = div f(z), when a divergence function is supplied

    ``signed_work_proxy`` is useful for detecting local acceleration or
    deceleration, but it is not entropy production.  ``div_drift_fn`` is
    reported directly and is not combined with velocity into an EPR estimate.
    A pathwise entropy-production calculation requires an explicitly defined
    stochastic process, diffusion tensor, time-step convention, and
    forward/reverse path-probability ratio.
    """
    if z_trajectory.shape[0] != delta_z_trajectory.shape[0]:
        raise ValueError("z_trajectory and delta_z_trajectory must share time length")
    v = delta_z_trajectory
    T = v.shape[0]
    if T < 3:
        raise ValueError("compute_trajectory_diagnostics needs >= 3 time steps")

    denom = float(D) + 1e-8
    velocity_proxy = (v ** 2).sum(dim=-1) / denom

    # Central differences estimate local change in the observed increment.
    dv = torch.zeros_like(v)
    dv[1:-1] = (v[2:] - v[:-2]) / 2.0
    dv[0] = v[1] - v[0]
    dv[-1] = v[-1] - v[-2]
    signed_work_proxy = (v * dv).sum(dim=-1) / denom

    def _red(x):
        x = x.detach().float().cpu()
        return float(x.mean()), float(x.std(unbiased=False))

    velocity_mean, velocity_std = _red(velocity_proxy)
    work_mean, work_std = _red(signed_work_proxy)
    corr = 1.0
    if velocity_std > 1e-6 and work_std > 1e-6:
        vp = velocity_proxy.detach().float().cpu().numpy().reshape(-1)
        wp = signed_work_proxy.detach().float().cpu().numpy().reshape(-1)
        if vp.size >= 2 and wp.size >= 2:
            corr = float(np.corrcoef(vp, wp)[0, 1])

    out = {
        "velocity_proxy_mean": velocity_mean,
        "velocity_proxy_std": velocity_std,
        "signed_work_proxy_mean": work_mean,
        "signed_work_proxy_std": work_std,
        "negative_work_ratio": float((signed_work_proxy < 0).float().mean()),
        "velocity_work_correlation": corr,
    }
    if div_drift_fn is not None:
        with torch.no_grad():
            div = div_drift_fn(z_trajectory)
        div = torch.as_tensor(
            div, dtype=v.dtype, device=v.device).reshape(-1)
        if div.shape[0] != T:
            raise ValueError("div_drift_fn must return one value per time step")
        div_mean, div_std = _red(div)
        out["drift_divergence_mean"] = div_mean
        out["drift_divergence_std"] = div_std
        out["negative_divergence_ratio"] = float((div < 0).float().mean())
    return out


def compute_energy_change_diagnostic(
    z_trajectory: torch.Tensor,
    energy_fn,
    beta: float = 1.0,
) -> Dict[str, float]:
    """Summarize energy changes without claiming work or Jarzynski validity.

    A Jarzynski estimate requires a specified driven protocol, an initial
    equilibrium ensemble, inverse temperature, and work measurements.  A
    latent trajectory plus an arbitrary scalar field supplies none of those
    guarantees.  This helper therefore reports only energy-change statistics
    and a cumulant-like numerical dispersion proxy.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    with torch.no_grad():
        energies = torch.stack([
            torch.as_tensor(energy_fn(z_trajectory[t:t + 1]))
            .detach().float().mean()
            for t in range(z_trajectory.shape[0])
        ]).cpu().numpy()
    delta_e = np.diff(energies)
    exp_neg_beta_delta = np.exp(-beta * delta_e.astype(np.float64))
    mean_exp = float(exp_neg_beta_delta.mean()) if delta_e.size else float("nan")
    proxy = (
        beta * float(delta_e.mean()) - np.log(mean_exp)
        if delta_e.size and np.isfinite(mean_exp) and mean_exp > 0
        else float("nan")
    )
    return {
        "energy_change_mean": float(delta_e.mean()) if delta_e.size else 0.0,
        "energy_change_std": float(delta_e.std()) if delta_e.size else 0.0,
        "energy_change_max": float(delta_e.max()) if delta_e.size else 0.0,
        "exp_neg_beta_change_mean": mean_exp,
        "energy_change_dispersion_proxy": float(proxy),
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


class EnergyChangeMonitor:
    """Heuristic monitor for latent energy-change and routing proxies.

    The monitor is deliberately not named or interpreted as a Bergson
    information, Jarzynski, work, irreversibility, or entropy-production
    estimator.  Its collapse flag is a model-health heuristic that requires
    both low proxy activity and low router entropy.
    """

    def __init__(
        self,
        beta: float = 1.0,
        window_size: int = 100,
        fp32_for_exp: bool = True,
    ):
        if beta <= 0:
            raise ValueError("beta must be positive")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.beta = beta
        self.window_size = window_size
        self.fp32_for_exp = fp32_for_exp
        self.change_history: List[float] = []
        self.proxy_history: List[float] = []
        self.nan_warning_count = 0

    def update(
        self,
        z_current: torch.Tensor,
        z_next: torch.Tensor,
        energy_fn=None,
    ) -> Dict[str, float]:
        """Record one scalar energy-change or latent-step proxy."""
        with torch.no_grad():
            if energy_fn is not None:
                current = torch.as_tensor(energy_fn(z_current)).float().mean().item()
                next_value = torch.as_tensor(energy_fn(z_next)).float().mean().item()
                change = next_value - current
            else:
                change = (z_next - z_current).pow(2).sum(dim=-1).mean().item()
        self.change_history.append(float(change))
        self.change_history = self.change_history[-self.window_size:]

        if len(self.change_history) < 2:
            return {
                "energy_change_proxy": float("nan"),
                "mean_energy_change": float("nan"),
                "exp_neg_beta_change": float("nan"),
                "proxy_nan_warning": 0.0,
            }

        values = np.asarray(self.change_history)
        dtype = np.float64 if self.fp32_for_exp else values.dtype
        exp_neg_beta = np.exp(-self.beta * values.astype(dtype))
        mean_exp = float(exp_neg_beta.mean())
        if not np.isfinite(mean_exp) or mean_exp <= 0:
            self.nan_warning_count += 1
            proxy = float("nan")
            warning = 1.0
        else:
            proxy = self.beta * float(values.mean()) - np.log(mean_exp)
            warning = 0.0
        self.proxy_history.append(proxy)
        self.proxy_history = self.proxy_history[-10000:]
        return {
            "energy_change_proxy": proxy,
            "mean_energy_change": float(values.mean()),
            "exp_neg_beta_change": mean_exp,
            "proxy_nan_warning": warning,
        }

    def check_collapse(self, router_entropy: float, threshold: float = 0.5) -> bool:
        """Return a heuristic low-activity/low-routing-diversity flag."""
        valid = [x for x in self.proxy_history if np.isfinite(x)]
        if not valid:
            return False
        return float(np.mean(valid[-20:])) < 0.01 and router_entropy < threshold

    def get_report(self) -> Dict:
        valid = [x for x in self.proxy_history if np.isfinite(x)]
        return {
            "energy_change_proxy_current": (
                self.proxy_history[-1] if self.proxy_history else float("nan")),
            "energy_change_proxy_mean": (
                float(np.mean(valid)) if valid else float("nan")),
            "nan_warning_count": self.nan_warning_count,
            "change_history_len": len(self.change_history),
            "proxy_history_len": len(self.proxy_history),
        }
