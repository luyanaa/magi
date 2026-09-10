"""
GPU-accelerated fMRI parcellation and ROI time series extraction.

Uses sparse matrix operations on GPU for atlas-based parcellation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from ...runtime.device_utils import get_device

ATLAS_REGISTRY = {
    "schaefer_100": {"n_rois": 100, "description": "Schaefer 100 ROI"},
    "schaefer_200": {"n_rois": 200, "description": "Schaefer 200 ROI"},
    "schaefer_400": {"n_rois": 400, "description": "Schaefer 400 ROI (NeuroSTORM default)"},
    "schaefer_1000": {"n_rois": 1000, "description": "Schaefer 1000 ROI"},
    "glasser_360": {"n_rois": 360, "description": "Glasser HCP-MMP 360 ROI"},
    "a424": {"n_rois": 424, "description": "A424 atlas (BrainLM default)"},
    "difumo_64": {"n_rois": 64, "description": "DiFuMo 64"},
    "difumo_128": {"n_rois": 128, "description": "DiFuMo 128"},
    "difumo_512": {"n_rois": 512, "description": "DiFuMo 512"},
    "difumo_1024": {"n_rois": 1024, "description": "DiFuMo 1024"},
}


class GPUParcellation:
    """GPU-accelerated ROI time series extraction from fMRI volumes.

    Uses pre-computed atlas weight matrices for efficient GPU extraction.

    Usage:
        parcel = GPUParcellation("schaefer_400")
        parcel.load_atlas(atlas_volume, roi_names)
        roi_ts = parcel.extract(bold_4d, brain_mask)
    """

    def __init__(
        self,
        atlas_name: str = "schaefer_400",
        device: Optional[torch.device] = None,
    ):
        self.atlas_name = atlas_name
        self.device = get_device(device)
        self.n_rois = ATLAS_REGISTRY.get(atlas_name, {}).get("n_rois", 400)

        self.weight_matrix: Optional[torch.Tensor] = None
        self.roi_names: Optional[List[str]] = None
        self.atlas_volume: Optional[torch.Tensor] = None

    def load_atlas(
        self,
        atlas_volume: torch.Tensor,
        roi_names: Optional[List[str]] = None,
        brain_mask: Optional[torch.Tensor] = None,
    ) -> "GPUParcellation":
        """Load atlas and pre-compute weight matrix for GPU extraction.

        Args:
            atlas_volume: (X, Y, Z) integer label volume (0 = background)
            roi_names: optional list of ROI names
            brain_mask: (X, Y, Z) bool brain mask (None = atlas > 0)

        Returns:
            self
        """
        atlas_volume = atlas_volume.to(self.device).long()
        self.atlas_volume = atlas_volume

        if brain_mask is None:
            brain_mask = atlas_volume > 0
        else:
            brain_mask = brain_mask.to(self.device).bool()

        labels_in_mask = atlas_volume[brain_mask]
        unique_labels = torch.unique(labels_in_mask)
        unique_labels = unique_labels[unique_labels > 0]

        n_voxels = brain_mask.sum().item()
        n_labels = len(unique_labels)

        self.weight_matrix = torch.zeros(n_labels, n_voxels,
                                          device=self.device, dtype=torch.float32)

        for i, label in enumerate(unique_labels):
            voxel_mask = (labels_in_mask == label)
            n_voxels_in_roi = voxel_mask.sum().item()
            if n_voxels_in_roi > 0:
                self.weight_matrix[i, voxel_mask] = 1.0 / n_voxels_in_roi

        self.n_rois = n_labels

        if roi_names is not None:
            self.roi_names = roi_names
        else:
            self.roi_names = [f"ROI_{label.item()}" for label in unique_labels]

        return self

    def load_atlas_from_nifti(
        self,
        atlas_path: str,
        mask_path: Optional[str] = None,
    ) -> "GPUParcellation":
        """Load atlas from NIfTI file.

        Args:
            atlas_path: path to atlas NIfTI file
            mask_path: optional path to brain mask NIfTI
        """
        try:
            import nibabel as nib
        except ImportError:
            raise RuntimeError("nibabel required for NIfTI loading")

        atlas_img = nib.load(atlas_path)
        atlas_data = torch.from_numpy(atlas_img.get_fdata()).long()

        mask_data = None
        if mask_path is not None:
            mask_img = nib.load(mask_path)
            mask_data = torch.from_numpy(mask_img.get_fdata()).bool()

        return self.load_atlas(atlas_data, brain_mask=mask_data)

    def extract(
        self,
        bold_4d: torch.Tensor,
        brain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract ROI time series from 4D BOLD data on GPU.

        Args:
            bold_4d: (X, Y, Z, T) BOLD time series on GPU
            brain_mask: (X, Y, Z) bool brain mask (must match atlas mask)

        Returns:
            roi_ts: (R, T) ROI time series
        """
        if self.weight_matrix is None:
            raise RuntimeError("Atlas not loaded. Call load_atlas() first.")

        bold_4d = bold_4d.to(self.device).float()

        if brain_mask is None:
            brain_mask = self.atlas_volume > 0
        else:
            brain_mask = brain_mask.to(self.device).bool()

        X, Y, Z, T = bold_4d.shape
        masked_data = bold_4d[brain_mask]  # (V, T)

        roi_ts = self.weight_matrix @ masked_data  # (R, T)

        return roi_ts

    def extract_batch(
        self,
        bold_batch: torch.Tensor,
        brain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract ROI time series from batched 4D BOLD data.

        Args:
            bold_batch: (B, X, Y, Z, T) batched BOLD data
            brain_mask: (X, Y, Z) shared brain mask

        Returns:
            roi_ts_batch: (B, R, T) ROI time series
        """
        B = bold_batch.shape[0]
        results = []
        for b in range(B):
            roi_ts = self.extract(bold_batch[b], brain_mask)
            results.append(roi_ts)
        return torch.stack(results, dim=0)


def confound_regression(
    roi_ts: torch.Tensor,
    confounds: torch.Tensor,
) -> torch.Tensor:
    """Regress out confounds from ROI time series on GPU.

    Args:
        roi_ts: (R, T) or (B, R, T) ROI time series
        confounds: (K, T) or (B, K, T) confound time series

    Returns:
        residual: (R, T) or (B, R, T) residual after regression
    """
    if roi_ts.dim() == 2:
        roi_ts = roi_ts.unsqueeze(0)
        confounds = confounds.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, R, T = roi_ts.shape
    K = confounds.shape[1]

    residual = torch.empty_like(roi_ts)
    for b in range(B):
        C = confounds[b].T  # (T, K)
        Y = roi_ts[b].T  # (T, R)

        ones = torch.ones(T, 1, device=roi_ts.device, dtype=roi_ts.dtype)
        C_with_intercept = torch.cat([ones, C], dim=1)  # (T, K+1)

        beta = torch.linalg.lstsq(C_with_intercept, Y).solution  # (K+1, R)
        predicted = C_with_intercept @ beta  # (T, R)
        residual[b] = (Y - predicted).T  # (R, T)

    if squeeze:
        residual = residual.squeeze(0)
    return residual


def temporal_filter(
    roi_ts: torch.Tensor,
    tr: float,
    low_freq: float = 0.008,
    high_freq: float = 0.1,
) -> torch.Tensor:
    """Apply temporal bandpass filter to ROI time series on GPU.

    Args:
        roi_ts: (R, T) or (B, R, T) ROI time series
        tr: Repetition time (seconds)
        low_freq: highpass cutoff (Hz, default 0.008)
        high_freq: lowpass cutoff (Hz, default 0.1)

    Returns:
        filtered: temporally filtered ROI time series
    """
    if roi_ts.dim() == 2:
        roi_ts = roi_ts.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, R, T = roi_ts.shape
    sfreq = 1.0 / tr

    for b in range(B):
        for r in range(R):
            signal = roi_ts[b, r]
            spectrum = torch.fft.rfft(signal)
            freqs = torch.fft.rfftfreq(T, d=tr, device=roi_ts.device)

            mask = (freqs >= low_freq) & (freqs <= high_freq)
            mask[0] = False
            spectrum = spectrum * mask.float()

            roi_ts[b, r] = torch.fft.irfft(spectrum, n=T)

    if squeeze:
        roi_ts = roi_ts.squeeze(0)
    return roi_ts
