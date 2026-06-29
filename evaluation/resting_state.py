"""
Resting-state evaluation metrics.

Assesses whether the model produces biologically plausible dynamics
when run in imagination mode without task cues.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


def compute_psd_slope(z_sequence: torch.Tensor, sample_rate: float = 1.0) -> float:
    """
    Compute 1/f PSD slope of latent trajectory.

    Args:
        z_sequence: (T, d) latent state sequence
        sample_rate: sampling rate in Hz
    Returns:
        slope: estimated power-law exponent (target 0.8-1.2)
    """
    T, d = z_sequence.shape
    slopes = []
    for i in range(min(d, 32)):
        x = z_sequence[:, i].cpu().numpy()
        fft_vals = np.fft.rfft(x)
        psd = np.abs(fft_vals) ** 2 + 1e-10
        freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate)
        valid = freqs > 0
        if valid.sum() < 2:
            continue
        log_psd = np.log(psd[valid])
        log_freq = np.log(freqs[valid])
        slope = np.polyfit(log_freq, log_psd, 1)[0]
        slopes.append(slope)
    return float(np.mean(slopes)) if slopes else 0.0


def compute_avalanche_stats(z_sequence: torch.Tensor, threshold_factor: float = 2.0) -> Dict[str, float]:
    """
    Detect neuronal avalanches in latent velocity norm.

    Returns:
        dict with tau (size-duration exponent), size_distribution
    """
    T = z_sequence.shape[0]
    v_norm = torch.norm(z_sequence[1:] - z_sequence[:-1], p=2, dim=-1).cpu().numpy()
    threshold = np.median(v_norm) * threshold_factor

    in_avalanche = False
    avalanches = []
    current = {"start": 0, "size": 0, "duration": 0, "sum": 0.0}

    for t in range(len(v_norm)):
        if v_norm[t] > threshold:
            if not in_avalanche:
                current = {"start": t, "size": 0, "duration": 0, "sum": 0.0}
                in_avalanche = True
            current["duration"] += 1
            current["sum"] += v_norm[t]
        else:
            if in_avalanche:
                current["size"] = current["sum"]
                avalanches.append(current)
                in_avalanche = False

    if len(avalanches) < 3:
        return {"tau": 0.0, "num_avalanches": 0}

    sizes = [a["size"] for a in avalanches]
    durations = [a["duration"] for a in avalanches]

    # Fit power law: log(size) ~ -tau * log(duration)
    if len(set(durations)) > 1:
        tau = -np.polyfit(np.log(durations + 1e-8), np.log(sizes + 1e-8), 1)[0]
    else:
        tau = 0.0

    return {
        "tau": float(tau),
        "num_avalanches": len(avalanches),
        "mean_size": float(np.mean(sizes)),
        "mean_duration": float(np.mean(durations)),
    }


def compute_functional_connectivity(z_sequence: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise correlation (functional connectivity) from latent states.

    Args:
        z_sequence: (T, d) latent state sequence
    Returns:
        fc: (d, d) correlation matrix
    """
    z_np = z_sequence.cpu().numpy()
    fc = np.corrcoef(z_np.T)
    return torch.from_numpy(fc).float()


def evaluate_resting_state(
    model: nn.Module,
    num_steps: int = 1024,
    latent_dim: int = 1024,
) -> Dict[str, float]:
    """
    Run model in imagination mode and evaluate resting-state metrics.

    Returns:
        dict with psd_slope, avalanche_tau, fc_mean_corr
    """
    model.eval()
    device = next(model.parameters()).device

    z = torch.randn(1, latent_dim, device=device) * 0.1
    z_sequence = []

    with torch.no_grad():
        for _ in range(num_steps):
            dummy_eeg = torch.zeros(1, 19, 2560, device=device)
            dummy_fmri = torch.zeros(1, 400, 100, device=device)
            out = model(dummy_eeg, dummy_fmri, mode="imagination")
            z = out["z_next"]
            z_sequence.append(z.squeeze(0))

    z_seq = torch.stack(z_sequence)

    psd_slope = compute_psd_slope(z_seq)
    avalanche = compute_avalanche_stats(z_seq)
    fc = compute_functional_connectivity(z_seq)

    # Compare FC to identity (self-correlation)
    fc_off_diag = fc - torch.eye(latent_dim, device=fc.device)
    fc_mean_corr = fc_off_diag.abs().mean().item()

    return {
        "psd_slope": psd_slope,
        "avalanche_tau": avalanche["tau"],
        "num_avalanches": avalanche["num_avalanches"],
        "fc_mean_corr": fc_mean_corr,
    }
