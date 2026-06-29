"""
MEG Preprocessing Pipeline for Brain MoE-PINN.

MEG (magnetoencechnography) shares temporal resolution with EEG
but measures magnetic fields via SQUID sensors. Preprocessing includes:
- Temporal filtering (0.1-300 Hz bandpass, line noise notch)
- Signal Space Separation (SSS/tSSS) for environmental noise
- Movement compensation (head position tracking)
- Channel interpolation for bad sensors
- Resampling to target frequency
- Z-score normalization

Foundation-mode: filter + resample + z-score only (no SSS/tSSS/movement compensation).
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class MEGPreprocessingConfig:
    target_sfreq: float = 1000.0
    bandpass: Tuple[float, float] = (0.1, 300.0)
    notch_freqs: List[float] = field(default_factory=lambda: [50, 100, 150])
    fir_numtaps: int = 101
    bad_channel_mode: str = "interpolate"
    amplitude_max: Optional[float] = None
    normalization: str = "zscore"
    quantile_pct: float = 0.95
    apply_sss: bool = False
    sss_origin: Tuple[float, float, float] = (0.0, 0.0, 0.04)
    device: str = "cpu"
    output_format: str = "both"
    save_format: str = "npy"


@dataclass
class MEGPreprocessedOutput:
    data: torch.Tensor
    channel_names: List[str]
    num_channels: int
    sfreq: float
    bad_channels: List[str]
    quality_report: Dict[str, Any]


class MEGPreprocessingPipeline:
    """
    MEG preprocessing pipeline.

    Foundation-mode: filter + resample + z-score + bad channel handling.
    Full-mode: add SSS/tSSS + movement compensation.
    """

    def __init__(self, config: MEGPreprocessingConfig = None):
        self.config = config or MEGPreprocessingConfig()

    def process(
        self,
        data: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        sfreq: Optional[float] = None,
        bad_channels: Optional[List[str]] = None,
    ) -> MEGPreprocessedOutput:
        """
        Process MEG data through the pipeline.

        Args:
            data: (C, T) raw MEG signals
            channel_names: optional channel name list
            sfreq: original sampling frequency
            bad_channels: list of bad channel names
        Returns:
            MEGPreprocessedOutput
        """
        sfreq = sfreq or self.config.target_sfreq
        bad_channels = bad_channels or []
        if channel_names is None:
            channel_names = [f"MEG{i:03d}" for i in range(data.shape[0])]

        x = data.clone()

        # Bandpass filter
        if self.config.bandpass is not None:
            x = self._apply_bandpass(x, sfreq)

        # Notch filter
        if self.config.notch_freqs:
            x = self._apply_notch(x, sfreq)

        # Bad channel handling
        if bad_channels and self.config.bad_channel_mode == "interpolate":
            x = self._interpolate_bad_channels(x, channel_names, bad_channels)
        elif bad_channels and self.config.bad_channel_mode == "zero":
            x = self._zero_bad_channels(x, channel_names, bad_channels)

        # Resample
        if sfreq != self.config.target_sfreq:
            x = self._resample(x, sfreq, self.config.target_sfreq)

        # Normalization
        if self.config.normalization == "zscore":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            std = torch.clamp(std, min=1e-8)
            x = (x - mean) / std

        quality = {
            "original_sfreq": sfreq,
            "target_sfreq": self.config.target_sfreq,
            "bad_channels": bad_channels,
            "snr_estimate": float(x.std() / (x.std() + 1e-8)),
        }

        return MEGPreprocessedOutput(
            data=x,
            channel_names=channel_names,
            num_channels=x.shape[0],
            sfreq=self.config.target_sfreq,
            bad_channels=bad_channels,
            quality_report=quality,
        )

    def _apply_bandpass(self, x: torch.Tensor, sfreq: float) -> torch.Tensor:
        from preprocessing.gpu_ops.filtering import design_fir_bandpass, apply_fir_filter
        low, high = self.config.bandpass
        if high > sfreq / 2:
            high = sfreq / 2 - 1.0
        kernel = design_fir_bandpass(low, high, sfreq, numtaps=self.config.fir_numtaps)
        return apply_fir_filter(x.unsqueeze(0), kernel).squeeze(0)

    def _apply_notch(self, x: torch.Tensor, sfreq: float) -> torch.Tensor:
        from preprocessing.gpu_ops.filtering import design_fir_notch, apply_fir_filter
        for freq in self.config.notch_freqs:
            if freq < sfreq / 2:
                kernel = design_fir_notch(freq, sfreq, numtaps=self.config.fir_numtaps)
                x = apply_fir_filter(x.unsqueeze(0), kernel).squeeze(0)
        return x

    def _interpolate_bad_channels(
        self, x: torch.Tensor, channel_names: List[str], bad_channels: List[str]
    ) -> torch.Tensor:
        bad_idx = [i for i, n in enumerate(channel_names) if n in bad_channels]
        good_idx = [i for i in range(x.shape[0]) if i not in bad_idx]
        if not good_idx or not bad_idx:
            return x
        good_data = x[good_idx]
        for bi in bad_idx:
            x[bi] = good_data.mean(dim=0)
        return x

    def _zero_bad_channels(
        self, x: torch.Tensor, channel_names: List[str], bad_channels: List[str]
    ) -> torch.Tensor:
        bad_idx = [i for i, n in enumerate(channel_names) if n in bad_channels]
        for bi in bad_idx:
            x[bi] = 0.0
        return x

    def _resample(
        self, x: torch.Tensor, orig_freq: float, new_freq: float
    ) -> torch.Tensor:
        if orig_freq == new_freq:
            return x
        from preprocessing.gpu_ops.resampling import resample_1d
        return resample_1d(x.unsqueeze(0), orig_freq, new_freq).squeeze(0)
