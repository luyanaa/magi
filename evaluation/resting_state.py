"""
Resting-state evaluation metrics.

Assesses whether the model produces biologically plausible dynamics
when run in imagination mode without task cues.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_psd_slope(z_sequence: torch.Tensor, sample_rate: float = 1.0) -> float:
    """
    Signed log-log PSD slope of a latent trajectory.

    Args:
        z_sequence: (T, d) latent state sequence
        sample_rate: sampling rate in Hz
    Returns:
        slope: estimated power-law exponent (negative for 1/f-like decay)

    This is a diagnostic of the latent trajectory's own spectrum.  It is *not*
    a 1/f health check: the 1/f property is an observable signature of neural
    signals, not a guaranteed property of latent dynamical trajectories, and
    the aperiodic exponent in real recordings varies with subject, region, and
    state rather than sitting in a fixed window.
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


def compute_avalanche_stats(
    z_sequence: torch.Tensor,
    threshold_factor: float = 2.0,
    min_avalanches: int = 20,
) -> Dict[str, float]:
    """Avalanche statistics for a latent population trajectory.

    Neuronal avalanches are defined over a *population*: at each time bin the
    units whose activity exceeds their own threshold are counted, consecutive
    active bins form one avalanche, and the avalanche size is the total
    activation count across those bins.  Criticality is summarised by the
    exponent of the size distribution ``P(s) ~ s^-tau``, estimated by maximum
    likelihood (Clauset et al., 2009).

    The previous implementation regressed ``log(size)`` on ``log(duration)``
    and reported that slope as ``tau``.  Size-versus-duration scaling is a
    *different* exponent (close to 2 under mean-field criticality) and is not
    the avalanche size exponent, so the reported number was mislabelled.

    Args:
        z_sequence: (T, d) latent trajectory; the d components act as the
            population and first differences define activity.
        threshold_factor: per-unit threshold in standard deviations above the
            unit's own mean activity.
        min_avalanches: minimum number of avalanches before an exponent is
            reported; below this the estimate is returned as NaN rather than
            fitted on noise.
    """
    if z_sequence.dim() != 2:
        raise ValueError("z_sequence must be (T, d)")
    time_len, width = z_sequence.shape
    if time_len < 3 or width < 1:
        return {"tau": float("nan"), "num_avalanches": 0,
                "size_exponent_definition": "P(size) ~ size^-tau (MLE)",
                "insufficient_data": True}

    activity = (z_sequence[1:] - z_sequence[:-1]).abs().cpu().numpy()
    mean = activity.mean(axis=0, keepdims=True)
    std = activity.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1e-12
    active = activity > (mean + threshold_factor * std)
    counts = active.sum(axis=1)

    sizes: List[float] = []
    durations: List[int] = []
    run_size = 0
    run_len = 0
    for count in counts.tolist():
        if count > 0:
            run_size += count
            run_len += 1
        elif run_len:
            sizes.append(float(run_size))
            durations.append(run_len)
            run_size, run_len = 0, 0
    if run_len:
        sizes.append(float(run_size))
        durations.append(run_len)

    num_avalanches = len(sizes)
    base = {
        "num_avalanches": num_avalanches,
        "size_exponent_definition": "P(size) ~ size^-tau (MLE)",
        "threshold_factor": float(threshold_factor),
        "mean_size": float(np.mean(sizes)) if sizes else 0.0,
        "mean_duration": float(np.mean(durations)) if durations else 0.0,
        "max_size": float(np.max(sizes)) if sizes else 0.0,
    }
    if num_avalanches < int(min_avalanches):
        return {"tau": float("nan"), "insufficient_data": True, **base}

    sizes_arr = np.asarray(sizes, dtype=float)
    size_min = max(1.0, float(sizes_arr.min()))
    log_terms = np.log(sizes_arr / size_min)
    denominator = float(log_terms.sum())
    if denominator <= 1e-12:
        # Every avalanche has the same size: no power law is identifiable.
        return {"tau": float("nan"), "insufficient_data": True, **base}
    tau = 1.0 + num_avalanches / denominator

    # First-order branching estimate: mean ratio of successive active counts
    # inside an avalanche.  Reported as a crude criticality summary only.
    ratios = [
        counts[k + 1] / counts[k]
        for k in range(len(counts) - 1)
        if counts[k] > 0
    ]
    branching = float(np.mean(ratios)) if ratios else float("nan")

    return {
        "tau": float(tau),
        "insufficient_data": False,
        "size_min": size_min,
        "branching_estimate": branching,
        **base,
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
