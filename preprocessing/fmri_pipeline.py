"""
fMRI Preprocessing Pipeline — 4-stage GPU-accelerated pipeline.

Stages:
  0. I/O & Validation (CPU)
  1. Spatial Preprocessing (CPU: fMRIPrep for registration; GPU: slice timing, optional VoxelMorph)
  2. Temporal Preprocessing (GPU: confound regression, bandpass, smoothing)
  3. Parcellation (GPU: ROI extraction, z-score)

Design: fMRIPrep handles spatial preprocessing (gold standard, CPU, run once).
GPU accelerates temporal steps that run per-session or per-batch.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import numpy as np

from .gpu_ops.spatial import smooth_4d_fmri, slice_timing_correction
from .gpu_ops.parcellation import (
    GPUParcellation, confound_regression, temporal_filter,
    ATLAS_REGISTRY,
)
from .gpu_ops.registration import GPURegistration
from .io.fmri_io import (
    load_nifti, load_bids_fmri, load_fmriprep_derivatives,
    fmri_raw_to_tensor, fMRIRawData,
)


@dataclass
class fMRIPreprocessingConfig:
    atlas: str = "schaefer_400"
    tr: Optional[float] = None
    n_dummy_scans: int = 5
    slice_timing_correction: bool = True
    confound_types: List[str] = field(default_factory=lambda: [
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
    ])
    temporal_bandpass: Tuple[float, float] = (0.008, 0.1)
    smoothing_fwhm: float = 4.0
    voxel_size: float = 2.0
    normalization: str = "zscore"
    use_voxelmorph: bool = False
    voxelmorph_model_path: Optional[str] = None
    template_path: Optional[str] = None
    device: str = "cpu"
    output_format: str = "both"
    save_format: str = "npy"


@dataclass
class fMRIPreprocessedOutput:
    roi_ts: torch.Tensor
    roi_names: List[str]
    n_rois: int
    tr: float
    motion_params: Optional[torch.Tensor]
    confounds: Optional[torch.Tensor]
    brain_mask: Optional[torch.Tensor]
    quality_report: Dict[str, Any]
    metadata: Dict[str, Any]


class fMRIPreprocessingPipeline:
    """4-stage GPU-accelerated fMRI preprocessing pipeline.

    Two usage patterns:
    1. From fMRIPrep derivatives (recommended): spatial preprocessing already done
    2. From raw NIfTI: optional VoxelMorph for fast registration

    Usage:
        pipeline = fMRIPreprocessingPipeline(config)
        output = pipeline(fmriprep_derivatives_dir, subject="01")
    """

    def __init__(self, config: Optional[fMRIPreprocessingConfig] = None):
        self.config = config or fMRIPreprocessingConfig()
        self.device = torch.device(self.config.device)
        self.parcellation = GPUParcellation(
            atlas_name=self.config.atlas,
            device=self.device,
        )
        self.registration = None
        if self.config.use_voxelmorph:
            self.registration = GPURegistration(
                template_path=self.config.template_path,
                model_path=self.config.voxelmorph_model_path,
                device=self.device,
            )

    @classmethod
    def from_config(cls, config_path: str) -> "fMRIPreprocessingPipeline":
        config = _load_fmri_yaml_config(config_path)
        return cls(config)

    def __call__(
        self,
        raw_input: Union[fMRIRawData, torch.Tensor, np.ndarray, str],
        subject: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        atlas_volume: Optional[torch.Tensor] = None,
        brain_mask: Optional[torch.Tensor] = None,
        confounds: Optional[torch.Tensor] = None,
    ) -> fMRIPreprocessedOutput:
        """Run full fMRI preprocessing pipeline."""
        bold, meta = self._stage0_load(raw_input, subject, session, task, run)
        bold, brain_mask = self._stage1_spatial(bold, meta, brain_mask)
        roi_ts, confounds_clean = self._stage2_temporal(bold, meta, confounds, brain_mask)
        roi_ts, roi_names = self._stage3_parcellation(roi_ts, meta, atlas_volume, brain_mask)
        return self._stage4_output(roi_ts, roi_names, meta, confounds_clean, brain_mask)

    def _stage0_load(
        self,
        raw_input: Union[fMRIRawData, torch.Tensor, np.ndarray, str],
        subject: Optional[str],
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
    ) -> Tuple[torch.Tensor, Dict]:
        """Stage 0: I/O & Validation."""
        if isinstance(raw_input, fMRIRawData):
            bold, meta = fmri_raw_to_tensor(raw_input, device=self.device)
        elif isinstance(raw_input, str):
            path = Path(raw_input)
            if path.is_dir():
                raw_data = load_fmriprep_derivatives(
                    raw_input, subject=subject or "01",
                    session=session, task=task, run=run,
                )
            elif path.suffix in (".nii", ".gz"):
                raw_data = load_nifti(str(path))
            else:
                raw_data = load_nifti(str(path))
            bold, meta = fmri_raw_to_tensor(raw_data, device=self.device)
        elif isinstance(raw_input, np.ndarray):
            bold = torch.from_numpy(raw_input).float().to(self.device)
            meta = {"tr": self.config.tr or 2.0}
        elif isinstance(raw_input, torch.Tensor):
            bold = raw_input.float().to(self.device)
            meta = {"tr": self.config.tr or 2.0}
        else:
            raise TypeError(f"Unsupported input type: {type(raw_input)}")

        if bold.dim() == 3:
            bold = bold.unsqueeze(-1)  # (X, Y, Z, 1)

        meta["tr"] = meta.get("tr", self.config.tr or 2.0)
        meta["shape"] = bold.shape
        return bold, meta

    def _stage1_spatial(
        self,
        bold: torch.Tensor,
        meta: Dict,
        brain_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Stage 1: Spatial Preprocessing."""
        if self.config.n_dummy_scans > 0 and bold.dim() == 4:
            n = self.config.n_dummy_scans
            if bold.shape[-1] > n:
                bold = bold[..., n:]
                meta["dummy_scans_removed"] = n

        if self.config.slice_timing_correction and bold.dim() == 4:
            slice_times = meta.get("slice_times")
            if slice_times is not None:
                tr = meta["tr"]
                bold = slice_timing_correction(bold, slice_times, tr)
                meta["slice_timing_corrected"] = True

        if brain_mask is None:
            if bold.dim() == 4:
                mean_bold = bold.mean(dim=-1)
                threshold = mean_bold.mean() + 0.5 * mean_bold.std()
                brain_mask = mean_bold > threshold
            else:
                brain_mask = None

        return bold, brain_mask

    def _stage2_temporal(
        self,
        bold: torch.Tensor,
        meta: Dict,
        confounds: Optional[torch.Tensor],
        brain_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Stage 2: Temporal Preprocessing (GPU)."""
        tr = meta["tr"]
        bold_clean = bold

        if bold.dim() == 4 and brain_mask is not None:
            X, Y, Z, T = bold.shape
            masked_voxels = bold[brain_mask]  # (V, T)

            if confounds is not None:
                confounds = confounds.to(self.device).float()
                if confounds.dim() == 1:
                    confounds = confounds.unsqueeze(0)
                masked_voxels = confound_regression(
                    masked_voxels.unsqueeze(0), confounds.unsqueeze(0)
                ).squeeze(0)

            if self.config.temporal_bandpass is not None:
                masked_voxels = temporal_filter(
                    masked_voxels.unsqueeze(0), tr,
                    low_freq=self.config.temporal_bandpass[0],
                    high_freq=self.config.temporal_bandpass[1],
                ).squeeze(0)

            bold_clean = torch.zeros_like(bold)
            bold_clean[brain_mask] = masked_voxels
        elif bold.dim() <= 3:
            # ROI time series or 2D input: apply filtering directly
            if self.config.temporal_bandpass is not None:
                bold_clean = temporal_filter(
                    bold_clean.unsqueeze(0), tr,
                    low_freq=self.config.temporal_bandpass[0],
                    high_freq=self.config.temporal_bandpass[1],
                ).squeeze(0)
            if confounds is not None:
                confounds = confounds.to(self.device).float()
                if confounds.dim() == 1:
                    confounds = confounds.unsqueeze(0)
                bold_clean = confound_regression(
                    bold_clean.unsqueeze(0), confounds.unsqueeze(0)
                ).squeeze(0)

        if self.config.smoothing_fwhm > 0 and bold_clean.dim() == 4:
            bold_clean = smooth_4d_fmri(
                bold_clean, self.config.smoothing_fwhm, self.config.voxel_size
            )
            meta["smoothing_fwhm"] = self.config.smoothing_fwhm

        return bold_clean, confounds

    def _stage3_parcellation(
        self,
        bold: torch.Tensor,
        meta: Dict,
        atlas_volume: Optional[torch.Tensor],
        brain_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[str]]:
        """Stage 3: Parcellation (GPU)."""
        if atlas_volume is not None:
            self.parcellation.load_atlas(
                atlas_volume.to(self.device),
                brain_mask=brain_mask.to(self.device) if brain_mask is not None else None,
            )

        if self.parcellation.weight_matrix is None:
            C, T = bold.shape[0], bold.shape[-1] if bold.dim() == 4 else bold.shape[-1]
            n_rois = self.parcellation.n_rois
            if brain_mask is not None and bold.dim() == 4:
                roi_ts = bold[brain_mask]  # (V, T)
                if roi_ts.shape[0] > n_rois:
                    step = roi_ts.shape[0] // n_rois
                    indices = torch.arange(0, n_rois * step, step, device=roi_ts.device)
                    roi_ts = roi_ts[indices]
                else:
                    roi_ts = roi_ts
            else:
                roi_ts = bold.reshape(-1, bold.shape[-1])[:n_rois]

            roi_names = [f"ROI_{i}" for i in range(roi_ts.shape[0])]
        else:
            if bold.dim() == 4:
                roi_ts = self.parcellation.extract(bold, brain_mask)
            else:
                roi_ts = bold
            roi_names = self.parcellation.roi_names or [f"ROI_{i}" for i in range(roi_ts.shape[0])]

        if self.config.normalization == "zscore":
            means = roi_ts.mean(dim=1, keepdim=True)
            stds = roi_ts.std(dim=1, keepdim=True) + 1e-10
            roi_ts = (roi_ts - means) / stds
        elif self.config.normalization == "robust":
            for r in range(roi_ts.shape[0]):
                median = roi_ts[r].median()
                q1 = torch.quantile(roi_ts[r], 0.25)
                q3 = torch.quantile(roi_ts[r], 0.75)
                iqr = q3 - q1 + 1e-10
                roi_ts[r] = (roi_ts[r] - median) / iqr

        meta["n_rois"] = roi_ts.shape[0]
        meta["atlas"] = self.config.atlas
        meta["normalization"] = self.config.normalization

        return roi_ts, roi_names

    def _stage4_output(
        self,
        roi_ts: torch.Tensor,
        roi_names: List[str],
        meta: Dict,
        confounds: Optional[torch.Tensor],
        brain_mask: Optional[torch.Tensor],
    ) -> fMRIPreprocessedOutput:
        """Stage 4: Output packaging."""
        quality_report = {
            "n_rois": roi_ts.shape[0],
            "n_timepoints": roi_ts.shape[1],
            "tr": meta.get("tr", self.config.tr),
            "atlas": self.config.atlas,
            "smoothing_fwhm": self.config.smoothing_fwhm,
            "normalization": self.config.normalization,
        }

        return fMRIPreprocessedOutput(
            roi_ts=roi_ts,
            roi_names=roi_names,
            n_rois=roi_ts.shape[0],
            tr=meta.get("tr", self.config.tr or 2.0),
            motion_params=None,
            confounds=confounds,
            brain_mask=brain_mask,
            quality_report=quality_report,
            metadata=meta,
        )


def _load_fmri_yaml_config(config_path: str) -> fMRIPreprocessingConfig:
    config = fMRIPreprocessingConfig()
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
