"""
GPU-accelerated quality assessment for EEG: bad channel detection,
bad span detection, and quality scoring.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QualityReport:
    bad_channels: torch.Tensor
    bad_spans: torch.Tensor
    channel_mask: torch.Tensor
    span_mask: torch.Tensor
    quality_score: float
    channel_correlations: Optional[torch.Tensor]
    channel_variances: Optional[torch.Tensor]
    artifact_ratio: float
    details: Dict


def detect_bad_channels(
    data: torch.Tensor,
    corr_threshold: float = 0.1,
    var_min: float = 1e-6,
    var_max_factor: float = 100.0,
    amplitude_max: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Detect bad channels in EEG data on GPU.

    Criteria:
    1. Low mean correlation with other channels (disconnected electrode)
    2. Near-zero variance (flat/dead channel)
    3. Excessively high variance (noisy channel)
    4. Excessive amplitude (optional)

    Args:
        data: (C, T) or (B, C, T) EEG data
        corr_threshold: minimum mean correlation (default 0.1)
        var_min: minimum variance threshold (default 1e-6)
        var_max_factor: max variance as factor of median (default 100×)
        amplitude_max: optional max amplitude threshold (e.g., 200e-6 for 200µV)

    Returns:
        bad_channel_mask: (C,) bool — True = bad channel
        details: dict with diagnostic info
    """
    if data.dim() == 3:
        data = data[0]

    C, T = data.shape

    centered = data - data.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1) + 1e-10

    if C > 1:
        corr_matrix = torch.corrcoef(data)
        mean_corr = corr_matrix.fill_diagonal_(0).abs().mean(dim=1)
        bad_corr = mean_corr < corr_threshold
    else:
        mean_corr = torch.ones(C, device=data.device)
        bad_corr = torch.zeros(C, dtype=torch.bool, device=data.device)

    channel_var = data.var(dim=-1)
    median_var = channel_var.median()
    bad_flat = channel_var < var_min
    bad_noisy = channel_var > (var_max_factor * median_var + 1e-12)

    bad_amp = torch.zeros(C, dtype=torch.bool, device=data.device)
    if amplitude_max is not None:
        bad_amp = data.abs().max(dim=-1).values > amplitude_max

    bad_mask = bad_corr | bad_flat | bad_noisy | bad_amp

    details = {
        "mean_correlations": mean_corr,
        "channel_variances": channel_var,
        "bad_by_correlation": bad_corr,
        "bad_by_flat": bad_flat,
        "bad_by_noisy": bad_noisy,
        "bad_by_amplitude": bad_amp,
    }

    return bad_mask, details


def detect_bad_spans(
    data: torch.Tensor,
    z_threshold: float = 5.0,
    window_sec: float = 1.0,
    sfreq: float = 256.0,
    min_span_sec: float = 0.1,
) -> Tuple[torch.Tensor, Dict]:
    """Detect bad time spans (artifacts) in EEG data on GPU.

    Uses sliding window z-score thresholding. Spans exceeding
    the threshold are marked as artifacts.

    Args:
        data: (C, T) or (B, C, T) EEG data
        z_threshold: z-score threshold for artifact detection (default 5.0)
        window_sec: detection window length in seconds (default 1.0)
        sfreq: sampling rate (Hz)
        min_span_sec: minimum artifact span duration (default 0.1s)

    Returns:
        span_mask: (T,) bool — True = good span, False = artifact
        details: dict with diagnostic info
    """
    if data.dim() == 3:
        data = data[0]

    C, T = data.shape
    window_samples = max(int(window_sec * sfreq), 1)
    min_span_samples = max(int(min_span_sec * sfreq), 1)

    channel_mean = data.mean(dim=-1, keepdim=True)
    channel_std = data.std(dim=-1, keepdim=True) + 1e-10
    z_scores = (data - channel_mean) / channel_std

    max_z_per_window = []
    for start in range(0, T - window_samples + 1, window_samples):
        window = z_scores[:, start:start + window_samples]
        max_z_per_window.append(window.abs().max())

    if not max_z_per_window:
        return torch.ones(T, dtype=torch.bool, device=data.device), {"artifact_ratio": 0.0}

    window_z = torch.stack(max_z_per_window)
    n_windows = len(max_z_per_window)

    bad_windows = window_z > z_threshold
    bad_sample_mask = torch.zeros(T, dtype=torch.bool, device=data.device)
    for i, start in enumerate(range(0, T - window_samples + 1, window_samples)):
        if bad_windows[i]:
            end = min(start + window_samples, T)
            bad_sample_mask[start:end] = True

    artifact_ratio = bad_sample_mask.float().mean().item()

    details = {
        "window_max_zscores": window_z,
        "bad_windows": bad_windows,
        "artifact_ratio": artifact_ratio,
        "n_bad_windows": bad_windows.sum().item(),
        "n_total_windows": n_windows,
    }

    span_mask = ~bad_sample_mask

    return span_mask, details


def compute_quality_score(
    bad_channel_mask: torch.Tensor,
    span_mask: torch.Tensor,
    total_channels: int,
    total_samples: int,
) -> float:
    """Compute overall quality score (0-1) for a recording.

    Score = (1 - bad_channel_ratio) × (1 - artifact_span_ratio)
    """
    ch_ratio = bad_channel_mask.float().mean().item() if total_channels > 0 else 0.0
    span_ratio = 1.0 - (span_mask.float().mean().item() if total_samples > 0 else 1.0)
    score = (1.0 - ch_ratio) * (1.0 - span_ratio)
    return max(0.0, min(1.0, score))


def full_quality_assessment(
    data: torch.Tensor,
    sfreq: float = 256.0,
    corr_threshold: float = 0.1,
    var_min: float = 1e-6,
    var_max_factor: float = 100.0,
    amplitude_max: Optional[float] = None,
    z_threshold: float = 5.0,
    window_sec: float = 1.0,
) -> QualityReport:
    """Run full quality assessment pipeline.

    Args:
        data: (C, T) or (B, C, T) EEG data on GPU
        sfreq: sampling rate
        corr_threshold: bad channel correlation threshold
        var_min: minimum variance threshold
        var_max_factor: max variance factor
        amplitude_max: max amplitude threshold (None = skip)
        z_threshold: artifact z-score threshold
        window_sec: detection window length

    Returns:
        QualityReport with all assessment results
    """
    if data.dim() == 3:
        data_2d = data[0]
    else:
        data_2d = data

    C, T = data_2d.shape

    bad_channels, ch_details = detect_bad_channels(
        data_2d,
        corr_threshold=corr_threshold,
        var_min=var_min,
        var_max_factor=var_max_factor,
        amplitude_max=amplitude_max,
    )

    span_mask, span_details = detect_bad_spans(
        data_2d,
        z_threshold=z_threshold,
        window_sec=window_sec,
        sfreq=sfreq,
    )

    channel_mask = ~bad_channels
    quality_score = compute_quality_score(bad_channels, span_mask, C, T)

    return QualityReport(
        bad_channels=bad_channels,
        bad_spans=~span_mask,
        channel_mask=channel_mask,
        span_mask=span_mask,
        quality_score=quality_score,
        channel_correlations=ch_details.get("mean_correlations"),
        channel_variances=ch_details.get("channel_variances"),
        artifact_ratio=span_details["artifact_ratio"],
        details={**ch_details, **span_details},
    )
