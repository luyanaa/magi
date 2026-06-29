"""
EEG Preprocessing Pipeline — 6-stage GPU-accelerated pipeline.

Stages:
  0. I/O & Validation (CPU)
  1. Signal Conditioning (GPU: resample, bandpass, notch, re-reference)
  2. Quality Assessment (GPU: bad channel/span detection)
  3. Artifact Handling (GPU: mask or repair with ICA/ASR)
  4. Normalization (GPU: z-score, quantile, or robust)
  5. Montage Standardization (GPU: virtual montage, channel alignment)
  6. Output (GPU tensor + optional disk save)

Two modes:
  - "foundation" (default): mask bad data, no repair — matches field standard
  - "clinical": ICA/ASR artifact removal + bad channel interpolation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import numpy as np

from .gpu_ops.filtering import (
    design_fir_bandpass, design_fir_notch, design_notch_kernel_set,
    apply_fir_filter,
)
from .gpu_ops.resampling import resample_1d
from .gpu_ops.quality import (
    full_quality_assessment, QualityReport,
    detect_bad_channels, detect_bad_spans, compute_quality_score,
)
from .gpu_ops.ica import GPUFastICA, classify_ica_components
from .gpu_ops.asr import GPUASR
from .io.eeg_io import (
    load_edf, load_nwb, load_bids, load_numpy,
    eeg_raw_to_tensor, EEGRawData,
)

STANDARD_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

CHANNEL_3D_POSITIONS = {
    "Fp1": (-0.291, 0.831, -0.025), "Fp2": (0.291, 0.831, -0.025),
    "F7": (-0.711, 0.553, -0.025), "F3": (-0.424, 0.642, 0.426),
    "Fz": (0.0, 0.734, 0.537), "F4": (0.424, 0.642, 0.426),
    "F8": (0.711, 0.553, -0.025), "T3": (-0.831, 0.0, -0.025),
    "C3": (-0.664, 0.0, 0.426), "Cz": (0.0, 0.0, 0.734),
    "C4": (0.664, 0.0, 0.426), "T4": (0.831, 0.0, -0.025),
    "T5": (-0.711, -0.553, -0.025), "P3": (-0.424, -0.642, 0.426),
    "Pz": (0.0, -0.734, 0.537), "P4": (0.424, -0.642, 0.426),
    "T6": (0.711, -0.553, -0.025), "O1": (-0.291, -0.831, -0.025),
    "O2": (0.291, -0.831, -0.025),
}


@dataclass
class EEGPreprocessingConfig:
    target_sfreq: float = 256.0
    bandpass: Tuple[float, float] = (0.1, 100.0)
    notch_freqs: List[float] = field(default_factory=lambda: [50, 60])
    fir_numtaps: int = 101
    reference: str = "average"
    artifact_mode: str = "mask"
    artifact_z_threshold: float = 5.0
    bad_channel_corr_threshold: float = 0.1
    bad_channel_var_min: float = 1e-6
    bad_channel_var_max_factor: float = 100.0
    amplitude_max: Optional[float] = None
    ica_components: int = 20
    ica_max_iter: int = 200
    asr_cutoff: float = -3.5
    normalization: str = "zscore"
    quantile_pct: float = 0.95
    target_montage: List[str] = field(default_factory=lambda: STANDARD_10_20)
    virtual_montage: bool = True
    idw_k: int = 4
    min_good_channels: int = 4
    device: str = "cpu"
    output_format: str = "both"
    save_format: str = "npy"


@dataclass
class EEGPreprocessedOutput:
    data: torch.Tensor
    channel_names: List[str]
    channel_types: torch.Tensor
    channel_mask: torch.Tensor
    span_mask: torch.Tensor
    quality_score: float
    sfreq: float
    montage_3d: Optional[torch.Tensor]
    artifact_labels: Dict[str, Any]
    metadata: Dict[str, Any]


class EEGPreprocessingPipeline:
    """6-stage GPU-accelerated EEG preprocessing pipeline.

    Usage:
        pipeline = EEGPreprocessingPipeline.from_config("eeg_foundation.yaml")
        output = pipeline(raw_eeg_data)
    """

    def __init__(self, config: Optional[EEGPreprocessingConfig] = None):
        self.config = config or EEGPreprocessingConfig()
        self.device = torch.device(self.config.device)

    @classmethod
    def from_config(cls, config_path: str) -> "EEGPreprocessingPipeline":
        """Load pipeline from YAML config file."""
        config = _load_yaml_config(config_path)
        return cls(config)

    def __call__(
        self,
        raw_input: Union[EEGRawData, torch.Tensor, np.ndarray],
        sfreq: Optional[float] = None,
        ch_names: Optional[List[str]] = None,
        ch_types: Optional[List[str]] = None,
        montage_3d: Optional[torch.Tensor] = None,
    ) -> EEGPreprocessedOutput:
        """Run full preprocessing pipeline.

        Args:
            raw_input: raw EEG data (EEGRawData, tensor, or numpy array)
            sfreq: sampling rate (required if raw_input is tensor/numpy)
            ch_names: channel names
            ch_types: channel types
            montage_3d: 3D electrode positions

        Returns:
            EEGPreprocessedOutput with preprocessed data and masks
        """
        data, meta = self._stage0_load(raw_input, sfreq, ch_names, ch_types, montage_3d)
        data = self._stage1_condition(data, meta)
        quality = self._stage2_quality(data, meta)
        data, artifact_labels = self._stage3_artifact(data, quality, meta)
        data = self._stage4_normalize(data, quality, meta)
        data, meta = self._stage5_montage(data, quality, meta)
        return self._stage6_output(data, quality, meta, artifact_labels)

    def _stage0_load(
        self,
        raw_input: Union[EEGRawData, torch.Tensor, np.ndarray],
        sfreq: Optional[float],
        ch_names: Optional[List[str]],
        ch_types: Optional[List[str]],
        montage_3d: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """Stage 0: I/O & Validation."""
        if isinstance(raw_input, EEGRawData):
            data, meta = eeg_raw_to_tensor(raw_input, device=self.device)
            if ch_names is None:
                ch_names = raw_input.ch_names
        elif isinstance(raw_input, np.ndarray):
            data = torch.from_numpy(raw_input).float().to(self.device)
            if data.dim() == 1:
                data = data.unsqueeze(0)
            meta = {"sfreq": sfreq or self.config.target_sfreq, "ch_names": ch_names or []}
        elif isinstance(raw_input, torch.Tensor):
            data = raw_input.float().to(self.device)
            if data.dim() == 1:
                data = data.unsqueeze(0)
            meta = {"sfreq": sfreq or self.config.target_sfreq, "ch_names": ch_names or []}
        else:
            raise TypeError(f"Unsupported input type: {type(raw_input)}")

        if data.dim() == 3:
            data = data[0]

        C, T = data.shape
        meta["sfreq"] = meta.get("sfreq", sfreq or self.config.target_sfreq)
        meta["ch_names"] = meta.get("ch_names", ch_names or [f"Ch{i}" for i in range(C)])
        meta["n_channels"] = C
        meta["n_samples"] = T

        if ch_types is not None:
            ch_type_map = {"eeg": 0, "ecog": 1, "seeg": 2, "eog": 3, "ecg": 4, "emg": 5, "misc": 6}
            meta["ch_types"] = torch.tensor(
                [ch_type_map.get(t.lower(), 6) for t in ch_types], dtype=torch.long
            )
        else:
            meta["ch_types"] = torch.zeros(C, dtype=torch.long)

        meta["montage_3d"] = montage_3d
        return data, meta

    def _stage1_condition(self, data: torch.Tensor, meta: Dict) -> torch.Tensor:
        """Stage 1: Signal Conditioning (GPU)."""
        sfreq = meta["sfreq"]
        target_sfreq = self.config.target_sfreq

        if sfreq != target_sfreq:
            data = resample_1d(data, orig_freq=sfreq, new_freq=target_sfreq)
            meta["sfreq"] = target_sfreq

        if self.config.bandpass[0] > 0 or self.config.bandpass[1] < target_sfreq / 2:
            kernel = design_fir_bandpass(
                self.config.bandpass[0], self.config.bandpass[1],
                target_sfreq, self.config.fir_numtaps,
            )
            data = apply_fir_filter(data.unsqueeze(0), kernel).squeeze(0)
            meta["bandpass"] = self.config.bandpass

        if self.config.notch_freqs:
            valid_notch = [f for f in self.config.notch_freqs if f < target_sfreq / 2]
            for notch_f in valid_notch:
                kernel = design_fir_notch(notch_f, target_sfreq, numtaps=self.config.fir_numtaps)
                data = apply_fir_filter(data.unsqueeze(0), kernel).squeeze(0)
            meta["notch_freqs"] = valid_notch

        if self.config.reference == "average":
            data = data - data.mean(dim=0, keepdim=True)
        elif self.config.reference == "laplacian":
            data = self._laplacian_reference(data, meta)

        return data

    def _stage2_quality(self, data: torch.Tensor, meta: Dict) -> QualityReport:
        """Stage 2: Quality Assessment (GPU)."""
        return full_quality_assessment(
            data.unsqueeze(0),
            sfreq=meta["sfreq"],
            corr_threshold=self.config.bad_channel_corr_threshold,
            var_min=self.config.bad_channel_var_min,
            var_max_factor=self.config.bad_channel_var_max_factor,
            amplitude_max=self.config.amplitude_max,
            z_threshold=self.config.artifact_z_threshold,
        )

    def _stage3_artifact(
        self,
        data: torch.Tensor,
        quality: QualityReport,
        meta: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Stage 3: Artifact Handling (GPU)."""
        artifact_labels = {}

        if self.config.artifact_mode == "mask":
            artifact_labels["mode"] = "mask"
            artifact_labels["bad_channels"] = quality.bad_channels.nonzero(as_tuple=True)[0].tolist()
            artifact_labels["artifact_spans"] = quality.artifact_ratio

            # Foundation-mode: actually zero out bad channels and artifact spans
            if quality.bad_channels.any():
                data = data.clone()
                data[~quality.channel_mask, :] = 0.0
            if hasattr(quality, "span_mask") and quality.span_mask is not None and not quality.span_mask.all():
                data = data.clone()
                data[:, ~quality.span_mask] = 0.0

        elif self.config.artifact_mode == "repair":
            artifact_labels["mode"] = "repair"

            if quality.bad_channels.any():
                data = self._interpolate_bad_channels(data, quality.bad_channels, meta)
                quality.channel_mask = ~quality.bad_channels
                artifact_labels["interpolated_channels"] = quality.bad_channels.nonzero(as_tuple=True)[0].tolist()

            if self.config.ica_components > 0:
                try:
                    ica = GPUFastICA(
                        n_components=self.config.ica_components,
                        max_iter=self.config.ica_max_iter,
                    )
                    ica.fit(data)
                    sources = ica.transform(data)
                    classification = classify_ica_components(sources, data, meta["sfreq"])
                    exclude = classification.get("eog", []) + classification.get("ecg", [])
                    if exclude:
                        data = ica.inverse_transform(sources, exclude=exclude)
                    artifact_labels["ica_excluded"] = exclude
                    artifact_labels["ica_classification"] = classification
                except Exception as e:
                    warnings.warn(f"ICA failed: {e}. Falling back to mask mode.")
                    artifact_labels["ica_error"] = str(e)

            if quality.bad_spans.any():
                try:
                    clean_end = min(int(30 * meta["sfreq"]), data.shape[1])
                    if clean_end > meta["sfreq"]:
                        asr = GPUASR(sfreq=meta["sfreq"], cutoff=self.config.asr_cutoff)
                        data = asr.fit_transform(data, clean_start=0, clean_end=clean_end)
                        artifact_labels["asr_applied"] = True
                except Exception as e:
                    warnings.warn(f"ASR failed: {e}.")
                    artifact_labels["asr_error"] = str(e)

        return data, artifact_labels

    def _stage4_normalize(
        self,
        data: torch.Tensor,
        quality: QualityReport,
        meta: Dict,
    ) -> torch.Tensor:
        """Stage 4: Normalization (GPU)."""
        if self.config.normalization == "zscore":
            good_mask = quality.channel_mask
            if good_mask.any():
                # Per-channel zscore using only good channels as reference,
                # but applied to all channels (bad channels remain zero from mask)
                means = data[good_mask].mean(dim=-1, keepdim=True)
                stds = data[good_mask].std(dim=-1, keepdim=True) + 1e-10
                data = (data - means.mean()) / stds.mean()
            else:
                means = data.mean()
                stds = data.std() + 1e-10
                data = (data - means) / stds

        elif self.config.normalization == "quantile":
            pct = self.config.quantile_pct
            scale = torch.quantile(data.abs(), pct) + 1e-10
            data = data / scale

        elif self.config.normalization == "robust":
            median = data.median()
            q1 = torch.quantile(data, 0.25)
            q3 = torch.quantile(data, 0.75)
            iqr = q3 - q1 + 1e-10
            data = (data - median) / iqr

        elif self.config.normalization == "per_channel_zscore":
            means = data.mean(dim=1, keepdim=True)
            stds = data.std(dim=1, keepdim=True) + 1e-10
            data = (data - means) / stds

        meta["normalization"] = self.config.normalization
        return data

    def _stage5_montage(
        self,
        data: torch.Tensor,
        quality: QualityReport,
        meta: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Stage 5: Montage Standardization (GPU)."""
        target_montage = self.config.target_montage
        current_ch_names = meta.get("ch_names", [])

        if not target_montage or not self.config.virtual_montage:
            meta["montage_aligned"] = False
            return data, meta

        if current_ch_names == target_montage:
            meta["montage_aligned"] = True
            return data, meta

        C_current = data.shape[0]
        C_target = len(target_montage)

        name_to_idx = {name.upper(): i for i, name in enumerate(current_ch_names)}
        mapping = []
        missing = []

        for t_name in target_montage:
            if t_name.upper() in name_to_idx:
                mapping.append(name_to_idx[t_name.upper()])
            else:
                mapping.append(-1)
                missing.append(t_name)

        new_data = torch.zeros(C_target, data.shape[1], device=data.device, dtype=data.dtype)

        for t_idx, src_idx in enumerate(mapping):
            if src_idx >= 0:
                new_data[t_idx] = data[src_idx]

        if missing and self.config.virtual_montage and C_current >= self.config.min_good_channels:
            new_data = self._idw_interpolate_missing(
                new_data, mapping, current_ch_names, target_montage, meta
            )

        new_ch_types = torch.zeros(C_target, dtype=torch.long, device=data.device)
        for t_idx, src_idx in enumerate(mapping):
            if src_idx >= 0 and meta.get("ch_types") is not None:
                new_ch_types[t_idx] = meta["ch_types"][src_idx]

        new_ch_mask = torch.zeros(C_target, dtype=torch.bool, device=data.device)
        for t_idx, src_idx in enumerate(mapping):
            new_ch_mask[t_idx] = (src_idx >= 0 and quality.channel_mask[src_idx])

        meta["ch_names"] = target_montage
        meta["ch_types"] = new_ch_types
        meta["channel_mask"] = new_ch_mask
        meta["n_channels"] = C_target
        meta["montage_aligned"] = True
        meta["missing_channels_interpolated"] = missing

        data = new_data
        quality.channel_mask = new_ch_mask
        return data, meta

    def _stage6_output(
        self,
        data: torch.Tensor,
        quality: QualityReport,
        meta: Dict,
        artifact_labels: Dict,
    ) -> EEGPreprocessedOutput:
        """Stage 6: Output packaging."""
        montage_3d = meta.get("montage_3d")
        if montage_3d is not None and not isinstance(montage_3d, torch.Tensor):
            montage_3d = torch.from_numpy(montage_3d).float().to(self.device)

        if self.config.normalization in ("zscore", "per_channel_zscore"):
            span_mask = quality.span_mask
        else:
            span_mask = quality.span_mask

        return EEGPreprocessedOutput(
            data=data,
            channel_names=meta.get("ch_names", []),
            channel_types=meta.get("ch_types", torch.zeros(data.shape[0], dtype=torch.long)),
            channel_mask=quality.channel_mask,
            span_mask=span_mask,
            quality_score=quality.quality_score,
            sfreq=meta.get("sfreq", self.config.target_sfreq),
            montage_3d=montage_3d,
            artifact_labels=artifact_labels,
            metadata=meta,
        )

    def _interpolate_bad_channels(
        self,
        data: torch.Tensor,
        bad_mask: torch.Tensor,
        meta: Dict,
    ) -> torch.Tensor:
        """Interpolate bad channels using IDW from good neighbors."""
        if not bad_mask.any():
            return data

        C, T = data.shape
        good_mask = ~bad_mask
        if good_mask.sum() < 2:
            return data

        good_indices = good_mask.nonzero(as_tuple=True)[0]
        bad_indices = bad_mask.nonzero(as_tuple=True)[0]

        for bad_idx in bad_indices:
            distances = (good_indices.float() - bad_idx.float()).abs()
            k = min(self.config.idw_k, len(good_indices))
            _, nearest_k = distances.topk(k, largest=False)
            nearest_good = good_indices[nearest_k]

            weights = 1.0 / (distances[nearest_k] + 1e-6)
            weights = weights / weights.sum()

            data[bad_idx] = (data[nearest_good] * weights.unsqueeze(1)).sum(dim=0)

        return data

    def _idw_interpolate_missing(
        self,
        data: torch.Tensor,
        mapping: List[int],
        current_ch_names: List[str],
        target_montage: List[str],
        meta: Dict,
    ) -> torch.Tensor:
        """IDW interpolation for missing channels using 3D positions."""
        available_positions = {}
        for i, name in enumerate(current_ch_names):
            if name.upper() in {k.upper(): k for k in CHANNEL_3D_POSITIONS}:
                key = next(k for k in CHANNEL_3D_POSITIONS if k.upper() == name.upper())
                available_positions[i] = torch.tensor(CHANNEL_3D_POSITIONS[key], device=data.device)

        if not available_positions:
            return data

        for t_idx, src_idx in enumerate(mapping):
            if src_idx >= 0:
                continue

            target_name = target_montage[t_idx]
            if target_name not in CHANNEL_3D_POSITIONS:
                continue

            target_pos = torch.tensor(CHANNEL_3D_POSITIONS[target_name], device=data.device)

            distances = []
            indices = []
            for avail_idx, avail_pos in available_positions.items():
                if mapping.count(avail_idx) == 0 or True:
                    dist = (target_pos - avail_pos).norm()
                    distances.append(dist)
                    indices.append(avail_idx)

            if not distances:
                continue

            dist_tensor = torch.tensor(distances, device=data.device)
            k = min(self.config.idw_k, len(distances))
            _, topk_idx = dist_tensor.topk(k, largest=False)

            weights = 1.0 / (dist_tensor[topk_idx] ** 2 + 1e-6)
            weights = weights / weights.sum()

            nearest = [indices[i] for i in topk_idx.tolist()]
            data[t_idx] = (data[nearest] * weights.unsqueeze(1)).sum(dim=0)

        return data

    @staticmethod
    def _laplacian_reference(data: torch.Tensor, meta: Dict) -> torch.Tensor:
        """Compute small Laplacian (Hjorth) reference."""
        C, T = data.shape
        result = data.clone()
        for i in range(C):
            neighbors = [i - 1, i + 1]
            neighbor_vals = []
            for n in neighbors:
                if 0 <= n < C:
                    neighbor_vals.append(data[n])
            if neighbor_vals:
                avg_neighbor = torch.stack(neighbor_vals).mean(dim=0)
                result[i] = data[i] - avg_neighbor
        return result


def _load_yaml_config(config_path: str) -> EEGPreprocessingConfig:
    """Load config from YAML file (minimal parser, no pyyaml dependency)."""
    config = EEGPreprocessingConfig()
    try:
        with open(config_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if hasattr(config, key):
                    attr = getattr(config, key)
                    if isinstance(attr, bool):
                        setattr(config, key, val.lower() in ("true", "yes", "1"))
                    elif isinstance(attr, int):
                        setattr(config, key, int(val))
                    elif isinstance(attr, float):
                        setattr(config, key, float(val))
                    elif isinstance(attr, str):
                        setattr(config, key, val)
    except Exception:
        pass
    return config
