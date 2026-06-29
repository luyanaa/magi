"""
GPU-accelerated FIR/IIR filtering for EEG and fMRI signals.

Uses torch.nn.functional.conv1d for FIR filtering and
torchaudio.functional.lfilter for IIR filtering.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import warnings

try:
    import torchaudio.functional as taf
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from scipy.signal import firwin, iirnotch, iirfilter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def design_fir_bandpass(
    low_freq: float,
    high_freq: float,
    sfreq: float,
    numtaps: int = 101,
) -> torch.Tensor:
    """Design FIR bandpass filter kernel using windowed-sinc method.

    Args:
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        sfreq: Sampling rate (Hz)
        numtaps: Filter length (odd recommended for zero-phase)

    Returns:
        kernel: (numtaps,) float32 filter kernel on CPU
    """
    if not SCIPY_AVAILABLE:
        return _fir_bandpass_torch(low_freq, high_freq, sfreq, numtaps)
    nyq = sfreq / 2.0
    low = max(low_freq / nyq, 1e-6)
    high = min(high_freq / nyq, 1.0 - 1e-6)
    if low >= high:
        raise ValueError(f"low_freq ({low_freq}) >= high_freq ({high_freq}) after Nyquist normalization")
    kernel = firwin(numtaps, [low, high], pass_zero=False)
    return torch.tensor(kernel, dtype=torch.float32)


def design_fir_lowpass(
    cutoff: float,
    sfreq: float,
    numtaps: int = 101,
) -> torch.Tensor:
    nyq = sfreq / 2.0
    cutoff_norm = min(cutoff / nyq, 1.0 - 1e-6)
    if not SCIPY_AVAILABLE:
        return _fir_lowpass_torch(cutoff_norm, numtaps)
    kernel = firwin(numtaps, cutoff_norm)
    return torch.tensor(kernel, dtype=torch.float32)


def design_fir_highpass(
    cutoff: float,
    sfreq: float,
    numtaps: int = 101,
) -> torch.Tensor:
    nyq = sfreq / 2.0
    cutoff_norm = max(cutoff / nyq, 1e-6)
    if not SCIPY_AVAILABLE:
        return _fir_highpass_torch(cutoff_norm, numtaps)
    kernel = firwin(numtaps, cutoff_norm, pass_zero=False)
    return torch.tensor(kernel, dtype=torch.float32)


def design_fir_notch(
    freq: float,
    sfreq: float,
    quality_factor: float = 30.0,
    numtaps: int = 101,
) -> torch.Tensor:
    """Design FIR notch (band-stop) filter kernel.

    Uses a band-stop approach: stop band is [freq - bandwidth/2, freq + bandwidth/2]
    where bandwidth = freq / quality_factor.
    """
    nyq = sfreq / 2.0
    bandwidth = freq / quality_factor
    low = max((freq - bandwidth / 2) / nyq, 1e-6)
    high = min((freq + bandwidth / 2) / nyq, 1.0 - 1e-6)
    if not SCIPY_AVAILABLE:
        return _fir_notch_torch(low, high, numtaps)
    kernel = firwin(numtaps, [low, high], pass_zero=True)
    return torch.tensor(kernel, dtype=torch.float32)


def design_notch_kernel_set(
    freqs: List[float],
    sfreq: float,
    quality_factor: float = 30.0,
    numtaps: int = 101,
) -> List[torch.Tensor]:
    """Design multiple notch filter kernels for line noise removal."""
    kernels = []
    for f in freqs:
        if f < sfreq / 2.0:
            kernels.append(design_fir_notch(f, sfreq, quality_factor, numtaps))
    return kernels


def apply_fir_filter(
    data: torch.Tensor,
    kernel: torch.Tensor,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Apply FIR filter via 1D convolution on GPU.

    Args:
        data: (B, C, T) or (C, T) signal tensor
        kernel: (K,) FIR filter kernel
        pad_mode: padding mode for edge handling

    Returns:
        Filtered data with same shape as input
    """
    squeeze = False
    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze = True

    B, C, T = data.shape
    pad_len = kernel.shape[0] // 2
    padded = F.pad(data, (pad_len, pad_len), mode=pad_mode)

    k = kernel.to(data.device).to(data.dtype).view(1, 1, -1)
    flat_input = padded.reshape(B * C, 1, T + 2 * pad_len)
    filtered = F.conv1d(flat_input, k, groups=1)
    filtered = filtered.reshape(B, C, T)

    if squeeze:
        filtered = filtered.squeeze(0)
    return filtered


def apply_fir_filter_batched(
    data: torch.Tensor,
    kernel: torch.Tensor,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Apply FIR filter to batched (B, C, T) data.

    Same as apply_fir_filter but explicitly expects 3D input.
    Uses depthwise convolution for per-channel filtering.
    """
    return apply_fir_filter(data, kernel, pad_mode)


def apply_iir_filter(
    data: torch.Tensor,
    b_coeffs: torch.Tensor,
    a_coeffs: torch.Tensor,
) -> torch.Tensor:
    """Apply IIR filter using torchaudio.functional.lfilter.

    Args:
        data: (B, C, T) or (C, T) signal tensor
        b_coeffs: numerator (feedforward) coefficients
        a_coeffs: denominator (feedback) coefficients

    Returns:
        Filtered data
    """
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio required for IIR filtering. Install with: pip install torchaudio")

    squeeze = False
    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze = True

    B, C, T = data.shape
    b = b_coeffs.to(data.device).to(data.dtype)
    a = a_coeffs.to(data.device).to(data.dtype)

    result = torch.empty_like(data)
    for b_idx in range(B):
        for c_idx in range(C):
            result[b_idx, c_idx] = taf.lfilter(data[b_idx, c_idx], a, b)

    if squeeze:
        result = result.squeeze(0)
    return result


def apply_iir_filtfilt(
    data: torch.Tensor,
    b_coeffs: torch.Tensor,
    a_coeffs: torch.Tensor,
) -> torch.Tensor:
    """Zero-phase IIR filtering (forward-backward)."""
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio required for filtfilt. Install with: pip install torchaudio")

    forward = apply_iir_filter(data, b_coeffs, a_coeffs)
    reversed_data = torch.flip(forward, dims=[-1])
    backward = apply_iir_filter(reversed_data, b_coeffs, a_coeffs)
    return torch.flip(backward, dims=[-1])


def design_iir_bandpass(
    low_freq: float,
    high_freq: float,
    sfreq: float,
    order: int = 4,
    btype: str = "band",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design IIR Butterworth bandpass filter coefficients.

    Returns (b_coeffs, a_coeffs) as torch tensors.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy required for IIR filter design")
    nyq = sfreq / 2.0
    low = max(low_freq / nyq, 1e-6)
    high = min(high_freq / nyq, 1.0 - 1e-6)
    b, a = iirfilter(order, [low, high], btype=btype, output="ba")
    return torch.tensor(b, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)


def _fir_bandpass_torch(low_norm: float, high_norm: float, sfreq: float, numtaps: int) -> torch.Tensor:
    """Fallback FIR bandpass design without scipy."""
    n = torch.arange(numtaps) - (numtaps - 1) / 2.0
    low_w = 2 * torch.pi * low_norm
    high_w = 2 * torch.pi * high_norm
    with torch.no_grad():
        kernel = (torch.sin(high_w * n) / (torch.pi * n + 1e-12) -
                  torch.sin(low_w * n) / (torch.pi * n + 1e-12))
        kernel[(numtaps - 1) // 2] = high_norm - low_norm
        window = torch.hamming_window(numtaps)
        kernel = kernel * window
    return kernel.float()


def _fir_lowpass_torch(cutoff_norm: float, numtaps: int) -> torch.Tensor:
    n = torch.arange(numtaps) - (numtaps - 1) / 2.0
    w = 2 * torch.pi * cutoff_norm
    with torch.no_grad():
        kernel = torch.sin(w * n) / (torch.pi * n + 1e-12)
        kernel[(numtaps - 1) // 2] = cutoff_norm
        window = torch.hamming_window(numtaps)
        kernel = kernel * window
    return kernel.float()


def _fir_highpass_torch(cutoff_norm: float, numtaps: int) -> torch.Tensor:
    n = torch.arange(numtaps) - (numtaps - 1) / 2.0
    w = 2 * torch.pi * cutoff_norm
    with torch.no_grad():
        kernel = -torch.sin(w * n) / (torch.pi * n + 1e-12)
        kernel[(numtaps - 1) // 2] = 1.0 - cutoff_norm
        window = torch.hamming_window(numtaps)
        kernel = kernel * window
    return kernel.float()


def _fir_notch_torch(low_norm: float, high_norm: float, numtaps: int) -> torch.Tensor:
    n = torch.arange(numtaps) - (numtaps - 1) / 2.0
    low_w = 2 * torch.pi * low_norm
    high_w = 2 * torch.pi * high_norm
    with torch.no_grad():
        kernel = (torch.sin(torch.pi * n) / (torch.pi * n + 1e-12) -
                  (torch.sin(high_w * n) / (torch.pi * n + 1e-12) -
                   torch.sin(low_w * n) / (torch.pi * n + 1e-12)))
        kernel[(numtaps - 1) // 2] = 1.0 - (high_norm - low_norm)
        window = torch.hamming_window(numtaps)
        kernel = kernel * window
    return kernel.float()
