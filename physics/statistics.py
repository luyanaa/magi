"""
Statistical physics analysis: Tsallis entropy, 1/f noise, avalanches.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_tsallis_entropy(
    z: torch.Tensor,
    q: float = 2.0,
    num_bins: int = 100,
) -> Dict[str, float]:
    """
    Compute Tsallis entropy of latent state distribution.

    H_q = 1/(q-1) * (1 - sum_i p_i^q)
    """
    z_flat = z.reshape(-1).cpu().numpy()

    # Histogram
    hist, bin_edges = np.histogram(z_flat, bins=num_bins, density=True)
    p = hist / (hist.sum() + 1e-10)

    if q == 1.0:
        H = -np.sum(p * np.log(p + 1e-10))
    else:
        H = (1.0 - np.sum(p ** q)) / (q - 1.0)

    return {
        "tsallis_entropy": float(H),
        "q": q,
        "num_bins": num_bins,
    }


def compute_1f_exponent(
    signal: torch.Tensor,
    sample_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Compute 1/f noise exponent (alpha) via log-log PSD slope.

    Target: alpha in [0.8, 1.2] for critical brain dynamics.
    """
    x = signal.cpu().numpy()
    if x.ndim > 1:
        x = x.mean(axis=-1)

    fft_vals = np.fft.rfft(x)
    psd = np.abs(fft_vals) ** 2 + 1e-10
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)

    valid = freqs > 0
    log_psd = np.log(psd[valid])
    log_freq = np.log(freqs[valid])

    # Fit slope
    if len(log_freq) > 1:
        slope, intercept = np.polyfit(log_freq, log_psd, 1)
    else:
        slope, intercept = 0.0, 0.0

    return {
        "psd_slope": float(slope),
        "psd_intercept": float(intercept),
        "is_1f": 0.8 <= abs(slope) <= 1.2,
    }


def compute_velocity_growth_proxy(
    z_trajectory: torch.Tensor,
    delta_z_trajectory: torch.Tensor,
    k: int = 5,
) -> Dict[str, float]:
    """Summarize velocity-magnitude growth without calling it Lyapunov data.

    A Lyapunov exponent requires perturbation trajectories or tangent-space
    dynamics with a stated time normalization.  This function only fits the
    log slope of the supplied velocity magnitudes and reports it as an
    observational proxy.
    """
    del z_trajectory, k
    if delta_z_trajectory.shape[0] < 2:
        return {"velocity_log_slope": 0.0, "max_v_norm": 0.0, "mean_v_norm": 0.0}

    v_norms = torch.norm(delta_z_trajectory, p=2, dim=-1).detach().cpu().numpy()
    t_vals = np.arange(len(v_norms))
    if v_norms.std() > 1e-6:
        slope = np.polyfit(t_vals, np.log(v_norms + 1e-10), 1)[0]
    else:
        slope = 0.0

    return {
        "velocity_log_slope": float(slope),
        "max_v_norm": float(v_norms.max()),
        "mean_v_norm": float(v_norms.mean()),
    }


def estimate_local_transition_growth(
    fn,
    x: torch.Tensor,
    dt: float = 1.0,
    iters: int = 15,
    target_growth: float = 1.0,
) -> Dict[str, float]:
    """Estimate local growth of the explicit transition map.

    ``fn`` is interpreted as a continuous-time velocity field:

        dz/dt = fn(z)

    The monitored map is the explicit-Euler transition

        Phi_dt(z) = z + dt * fn(z).

    The returned quantity is the asymptotic JVP growth estimate of
    ``D Phi_dt``.  It is a local diagnostic, not a proof of global stability:
    non-normal Jacobians can have transient amplification even when their
    asymptotic growth is below one.  Use the reported value together with
    finite-horizon rollout diagnostics.

    ``fn`` must act independently on each batch row.  ``dt`` is required
    explicitly because a threshold on the velocity Jacobian alone has no
    time-unit meaning.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if iters < 1:
        raise ValueError("iters must be positive")
    if target_growth <= 0:
        raise ValueError("target_growth must be positive")

    x = x.detach()

    def transition(y):
        return y + float(dt) * fn(y)

    v = torch.randn_like(x)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    growth_prev = None
    converged = False
    iters_used = 0
    growth = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(iters):
        iters_used = i + 1
        with torch.enable_grad():
            _, jv = torch.autograd.functional.jvp(
                transition, (x,), (v,), create_graph=False)
        jv = jv.detach()
        growth = jv.norm(dim=-1)
        if growth_prev is not None and (growth - growth_prev).abs().max() < 1e-3:
            converged = True
            break
        if growth.max() > 1e6:
            break
        growth_prev = growth
        v = jv / growth[:, None].clamp_min(1e-12)

    growth_mean = float(growth.mean().item())
    growth_max = float(growth.max().item())
    return {
        "transition_growth_mean": growth_mean,
        "transition_growth_max": growth_max,
        "converged": bool(converged),
        "locally_bounded": bool(growth_max <= target_growth),
        "dt": float(dt),
        "target_growth": float(target_growth),
        "iters_used": iters_used,
    }
