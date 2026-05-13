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


def compute_lyapunov_exponents(
    z_trajectory: torch.Tensor,
    delta_z_trajectory: torch.Tensor,
    k: int = 5,
) -> Dict[str, float]:
    """
    Estimate largest Lyapunov exponents from trajectory.

    lambda_i = lim_{t->inf} (1/t) log(||delta_z(t)|| / ||delta_z(0)||)
    """
    T = z_trajectory.shape[0]
    if T < 2:
        return {"lambda_1": 0.0, "lambda_k": 0.0}

    # Use velocity magnitude as proxy for local expansion
    v_norms = torch.norm(delta_z_trajectory, p=2, dim=-1).cpu().numpy()

    # Fit exponential growth
    t_vals = np.arange(len(v_norms))
    if v_norms.std() > 1e-6:
        log_v = np.log(v_norms + 1e-10)
        slope = np.polyfit(t_vals, log_v, 1)[0]
    else:
        slope = 0.0

    return {
        "lambda_1": float(slope),
        "max_v_norm": float(v_norms.max()),
        "mean_v_norm": float(v_norms.mean()),
    }
