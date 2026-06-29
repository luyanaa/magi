"""
GPU-accelerated resampling for EEG and fMRI signals.

Uses torchaudio.transforms.Resample for 1D (EEG) and
torch.nn.functional.interpolate for ND (fMRI).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union

try:
    import torchaudio.transforms as tat
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


def resample_1d(
    data: torch.Tensor,
    orig_freq: float,
    new_freq: float,
    method: str = "torchaudio",
) -> torch.Tensor:
    """Resample 1D signal (EEG time series) on GPU.

    Args:
        data: (B, C, T) or (C, T) signal tensor
        orig_freq: Original sampling rate (Hz)
        new_freq: Target sampling rate (Hz)
        method: "torchaudio" (polyphase sinc) or "interpolate" (torch)

    Returns:
        Resampled data with T' = round(T * new_freq / orig_freq)
    """
    if orig_freq == new_freq:
        return data

    squeeze = False
    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze = True

    if method == "torchaudio" and TORCHAUDIO_AVAILABLE:
        result = _resample_torchaudio(data, orig_freq, new_freq)
    else:
        result = _resample_interpolate(data, orig_freq, new_freq)

    if squeeze:
        result = result.squeeze(0)
    return result


def resample_3d(
    volume: torch.Tensor,
    target_shape: tuple,
    mode: str = "trilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Resample 3D volume (fMRI spatial) on GPU.

    Args:
        volume: (B, C, D, H, W) or (D, H, W) tensor
        target_shape: (D', H', W') target spatial dimensions
        mode: interpolation mode ("trilinear", "nearest")

    Returns:
        Resampled volume
    """
    squeeze_batch = False
    squeeze_ch = False
    if volume.dim() == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        squeeze_batch = True
        squeeze_ch = True
    elif volume.dim() == 4:
        volume = volume.unsqueeze(0)  # (1, C, D, H, W)
        squeeze_batch = True

    result = F.interpolate(
        volume,
        size=target_shape,
        mode=mode,
        align_corners=align_corners if mode != "nearest" else None,
    )

    if squeeze_ch:
        result = result.squeeze(1)
    if squeeze_batch:
        result = result.squeeze(0)
    return result


def resample_fmri_4d(
    bold: torch.Tensor,
    target_spatial: Optional[tuple] = None,
    target_tr: Optional[float] = None,
    orig_tr: Optional[float] = None,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Resample 4D fMRI BOLD data (spatial and/or temporal).

    Args:
        bold: (X, Y, Z, T) or (B, X, Y, Z, T) 4D fMRI data
        target_spatial: (X', Y', Z') target spatial dims (None = no spatial resample)
        target_tr: Target TR in seconds (None = no temporal resample)
        orig_tr: Original TR in seconds (required if target_tr specified)
        mode: spatial interpolation mode

    Returns:
        Resampled BOLD data
    """
    if target_spatial is None and target_tr is None:
        return bold

    result = bold
    if target_spatial is not None:
        if result.dim() == 4:
            result = result.unsqueeze(0)  # (1, X, Y, Z, T)
            squeeze = True
        else:
            squeeze = False

        B, X, Y, Z, T = result.shape
        result_spatial = result.permute(0, 4, 1, 2, 3).reshape(B * T, 1, X, Y, Z)
        result_spatial = F.interpolate(
            result_spatial,
            size=target_spatial,
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        )
        X2, Y2, Z2 = target_spatial
        result = result_spatial.reshape(B, T, X2, Y2, Z2).permute(0, 2, 3, 4, 1)

        if squeeze:
            result = result.squeeze(0)

    if target_tr is not None and orig_tr is not None:
        if result.dim() == 4:
            orig_freq = 1.0 / orig_tr
            new_freq = 1.0 / target_tr
            result = result.permute(3, 0, 1, 2)  # (T, X, Y, Z)
            X, Y, Z = result.shape[1:]
            result_flat = result.reshape(T, -1).unsqueeze(0).unsqueeze(0)  # (1, 1, T, V)
            result_flat = _resample_interpolate_1d_features(result_flat, orig_freq, new_freq)
            T2 = result_flat.shape[-1]
            result = result_flat.squeeze(0).squeeze(0).reshape(T2, X, Y, Z).permute(1, 2, 3, 0)

    return result


def _resample_torchaudio(data: torch.Tensor, orig_freq: float, new_freq: float) -> torch.Tensor:
    B, C, T = data.shape
    resampler = tat.Resample(orig_freq=orig_freq, new_freq=new_freq).to(data.device).to(data.dtype)
    result = torch.empty(B, C, resampler.forward(data[0:1, 0:1]).shape[-1],
                         device=data.device, dtype=data.dtype)
    for b in range(B):
        for c in range(C):
            result[b, c] = resampler(data[b:b+1, c:c+1]).squeeze(0).squeeze(0)
    return result


def _resample_interpolate(data: torch.Tensor, orig_freq: float, new_freq: float) -> torch.Tensor:
    B, C, T = data.shape
    scale_factor = new_freq / orig_freq
    new_T = int(round(T * scale_factor))
    result = torch.empty(B, C, new_T, device=data.device, dtype=data.dtype)
    for b in range(B):
        result[b] = F.interpolate(
            data[b:b+1], size=new_T, mode="linear", align_corners=False
        ).squeeze(0)
    return result


def _resample_interpolate_1d_features(data: torch.Tensor, orig_freq: float, new_freq: float) -> torch.Tensor:
    scale_factor = new_freq / orig_freq
    new_T = int(round(data.shape[-1] * scale_factor))
    return F.interpolate(data, size=new_T, mode="linear", align_corners=False)
