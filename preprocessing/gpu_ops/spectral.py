"""
GPU-accelerated spectral analysis: STFT, PSD, bandpower.

Uses torch.stft and torch.fft for GPU-native spectral computation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

BAND_DEFINITIONS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def compute_stft(
    data: torch.Tensor,
    n_fft: int = 256,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    normalized: bool = False,
) -> torch.Tensor:
    """Compute Short-Time Fourier Transform on GPU.

    Args:
        data: (B, C, T) or (C, T) signal tensor
        n_fft: FFT size
        hop_length: hop between successive frames (default: n_fft // 4)
        win_length: window length (default: n_fft)
        window: window tensor (default: Hann)
        center: whether to pad input
        normalized: normalize by sqrt(n_fft)

    Returns:
        Complex STFT: (B, C, n_fft//2+1, T') or (C, n_fft//2+1, T')
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, device=data.device, dtype=data.dtype)

    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, C, T = data.shape
    n_freqs = n_fft // 2 + 1

    results = []
    for b in range(B):
        channel_results = []
        for c in range(C):
            stft_out = torch.stft(
                data[b, c],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                normalized=normalized,
                return_complex=True,
            )
            channel_results.append(stft_out)
        results.append(torch.stack(channel_results, dim=0))

    result = torch.stack(results, dim=0)

    if squeeze:
        result = result.squeeze(0)
    return result


def compute_psd(
    data: torch.Tensor,
    sfreq: float,
    n_fft: int = 256,
    hop_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    dB: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Power Spectral Density via Welch's method on GPU.

    Args:
        data: (B, C, T) or (C, T) signal tensor
        sfreq: Sampling rate (Hz)
        n_fft: FFT size
        hop_length: hop length (default: n_fft // 2 for 50% overlap)
        window: window tensor
        dB: return in dB

    Returns:
        freqs: (n_fft//2+1,) frequency bins in Hz
        psd: (B, C, n_fft//2+1) or (C, n_fft//2+1) PSD values
    """
    if hop_length is None:
        hop_length = n_fft // 2

    stft = compute_stft(data, n_fft=n_fft, hop_length=hop_length, window=window)

    psd = (stft.abs() ** 2).mean(dim=-1)

    freqs = torch.linspace(0, sfreq / 2, psd.shape[-1], device=data.device)

    if dB:
        psd = 10.0 * torch.log10(psd + 1e-12)

    return freqs, psd


def compute_bandpower(
    data: torch.Tensor,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    n_fft: int = 256,
) -> Dict[str, torch.Tensor]:
    """Compute bandpower for standard EEG frequency bands on GPU.

    Args:
        data: (B, C, T) or (C, T) signal tensor
        sfreq: Sampling rate (Hz)
        bands: dict of {band_name: (low_hz, high_hz)}. Default: delta/theta/alpha/beta/gamma
        n_fft: FFT size

    Returns:
        dict of {band_name: (B, C) or (C,) bandpower per channel}
    """
    if bands is None:
        bands = BAND_DEFINITIONS

    freqs, psd = compute_psd(data, sfreq=sfreq, n_fft=n_fft)

    if psd.dim() == 2:
        squeeze = True
        psd = psd.unsqueeze(0)
    else:
        squeeze = False

    B, C, F = psd.shape
    df = freqs[1] - freqs[0] if freqs.numel() > 1 else torch.tensor(1.0, device=data.device)

    result = {}
    for band_name, (low, high) in bands.items():
        freq_mask = (freqs >= low) & (freqs <= high)
        if freq_mask.any():
            band_psd = psd[:, :, freq_mask]
            power = band_psd.sum(dim=-1) * df
        else:
            power = torch.zeros(B, C, device=data.device, dtype=data.dtype)
        if squeeze:
            power = power.squeeze(0)
        result[band_name] = power

    return result


def compute_spectral_slope(
    data: torch.Tensor,
    sfreq: float,
    freq_range: Tuple[float, float] = (1.0, 100.0),
    n_fft: int = 512,
) -> torch.Tensor:
    """Compute 1/f spectral slope via linear regression on log-log PSD.

    Args:
        data: (B, C, T) or (C, T) signal tensor
        sfreq: Sampling rate (Hz)
        freq_range: frequency range for slope fitting
        n_fft: FFT size

    Returns:
        slope: (B, C) or (C,) spectral slope
    """
    freqs, psd = compute_psd(data, sfreq=sfreq, n_fft=n_fft)

    if psd.dim() == 2:
        psd = psd.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, C, F = psd.shape
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    log_freqs = torch.log10(freqs[freq_mask] + 1e-12)
    log_psd = torch.log10(psd[:, :, freq_mask] + 1e-12)

    x = log_freqs - log_freqs.mean()
    y = log_psd - log_psd.mean(dim=-1, keepdim=True)

    x_var = (x ** 2).sum()
    slopes = (y * x.unsqueeze(0).unsqueeze(0)).sum(dim=-1) / (x_var + 1e-12)

    if squeeze:
        slopes = slopes.squeeze(0)
    return slopes


def compute_cross_frequency_coupling(
    data: torch.Tensor,
    sfreq: float,
    phase_band: Tuple[float, float] = (4.0, 8.0),
    amplitude_band: Tuple[float, float] = (30.0, 100.0),
    n_fft: int = 256,
) -> torch.Tensor:
    """Compute phase-amplitude coupling (modulation index) on GPU.

    Args:
        data: (B, C, T) or (C, T)
        sfreq: Sampling rate
        phase_band: frequency band for phase (default theta)
        amplitude_band: frequency band for amplitude (default gamma)
        n_fft: FFT size

    Returns:
        mi: (B, C) or (C,) modulation index per channel
    """
    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, C, T = data.shape

    phase_signal = _bandpass_filter_simple(data, phase_band[0], phase_band[1], sfreq)
    amp_signal = _bandpass_filter_simple(data, amplitude_band[0], amplitude_band[1], sfreq)

    phase = torch.angle(torch.signal.hilbert(phase_signal, dim=-1)) if hasattr(torch.signal, 'hilbert') else _angle_via_analytic(phase_signal)
    amplitude = torch.abs(torch.signal.hilbert(amp_signal, dim=-1)) if hasattr(torch.signal, 'hilbert') else _envelope_via_analytic(amp_signal)

    n_bins = 18
    bin_edges = torch.linspace(-torch.pi, torch.pi, n_bins + 1, device=data.device)
    mi = torch.zeros(B, C, device=data.device, dtype=data.dtype)

    for b_idx in range(B):
        for c_idx in range(C):
            mean_amp = torch.zeros(n_bins, device=data.device, dtype=data.dtype)
            for k in range(n_bins):
                mask = (phase[b_idx, c_idx] >= bin_edges[k]) & (phase[b_idx, c_idx] < bin_edges[k + 1])
                if mask.any():
                    mean_amp[k] = amplitude[b_idx, c_idx, mask].mean()
            p = mean_amp / (mean_amp.sum() + 1e-12)
            p = p + 1e-12
            uniform = torch.ones(n_bins, device=data.device, dtype=data.dtype) / n_bins
            kl = (p * torch.log(p / uniform)).sum()
            mi[b_idx, c_idx] = kl / torch.log(torch.tensor(n_bins, dtype=data.dtype, device=data.device))

    if squeeze:
        mi = mi.squeeze(0)
    return mi


def _bandpass_filter_simple(data: torch.Tensor, low: float, high: float, sfreq: float) -> torch.Tensor:
    n_fft = data.shape[-1]
    spectrum = torch.fft.rfft(data, dim=-1)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq, device=data.device)
    mask = (freqs >= low) & (freqs <= high)
    spectrum = spectrum * mask.float().unsqueeze(0).unsqueeze(0)
    filtered = torch.fft.irfft(spectrum, n=n_fft, dim=-1)
    return filtered


def _angle_via_analytic(signal: torch.Tensor) -> torch.Tensor:
    N = signal.shape[-1]
    spectrum = torch.fft.fft(signal, dim=-1)
    h = torch.zeros(N, device=signal.device, dtype=signal.dtype)
    h[0] = 1
    if N % 2 == 0:
        h[1:N // 2] = 2
        h[N // 2] = 1
    else:
        h[1:(N + 1) // 2] = 2
    analytic = torch.fft.ifft(spectrum * h, dim=-1)
    return torch.angle(analytic)


def _envelope_via_analytic(signal: torch.Tensor) -> torch.Tensor:
    N = signal.shape[-1]
    spectrum = torch.fft.fft(signal, dim=-1)
    h = torch.zeros(N, device=signal.device, dtype=signal.dtype)
    h[0] = 1
    if N % 2 == 0:
        h[1:N // 2] = 2
        h[N // 2] = 1
    else:
        h[1:(N + 1) // 2] = 2
    analytic = torch.fft.ifft(spectrum * h, dim=-1)
    return torch.abs(analytic)
