"""
fMRI file I/O: Load raw fMRI from NIfTI/BIDS formats.

All I/O is CPU-based. Data is transferred to GPU after loading.
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from ...utils.device_utils import get_device
from dataclasses import dataclass
import warnings
import json

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


@dataclass
class fMRIRawData:
    """Container for raw fMRI data and metadata."""
    bold: np.ndarray
    affine: np.ndarray
    header: Any
    tr: float
    shape: Tuple[int, ...]
    slice_times: Optional[np.ndarray]
    info: Dict[str, Any]
    file_path: str


def load_nifti(
    file_path: str,
    dtype: np.dtype = np.float32,
) -> fMRIRawData:
    """Load fMRI from NIfTI file.

    Args:
        file_path: path to .nii or .nii.gz file
        dtype: target numpy dtype

    Returns:
        fMRIRawData container
    """
    if not NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel required for NIfTI loading. Install with: pip install nibabel")

    img = nib.load(file_path)
    data = img.get_fdata(dtype=dtype)
    affine = img.affine
    header = img.header

    pixdim = header.get_zooms()
    tr = pixdim[3] if len(pixdim) > 3 else 2.0

    slice_times = None
    if hasattr(header, "get_slice_times"):
        try:
            slice_times = np.array(header.get_slice_times())
        except Exception:
            pass

    return fMRIRawData(
        bold=data,
        affine=affine,
        header=header,
        tr=tr,
        shape=data.shape,
        slice_times=slice_times,
        info={"nifti_path": file_path, "pixdim": pixdim},
        file_path=file_path,
    )


def load_bids_fmri(
    bids_root: str,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    space: Optional[str] = None,
) -> fMRIRawData:
    """Load fMRI from BIDS dataset structure.

    Args:
        bids_root: root of BIDS dataset
        subject: subject ID
        session: session ID
        task: task name
        run: run ID
        space: space label (e.g., "MNI152NLin2009cAsym")

    Returns:
        fMRIRawData container
    """
    if not NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel required for BIDS loading")

    bids_root = Path(bids_root)
    filename_parts = [f"sub-{subject}"]
    if session:
        filename_parts.append(f"ses-{session}")
    task_part = f"task-{task}" if task else None
    run_part = f"run-{run}" if run else None
    space_part = f"space-{space}" if space else None

    search_dir = bids_root / f"sub-{subject}"
    if session:
        search_dir = search_dir / f"ses-{session}"
    search_dir = search_dir / "func"

    candidates = list(bids_root.rglob(f"sub-{subject}*bold.nii.gz"))
    if not candidates:
        candidates = list(bids_root.rglob(f"sub-{subject}*bold.nii"))

    if not candidates:
        raise FileNotFoundError(f"No fMRI BOLD file found for sub-{subject} in {bids_root}")

    file_path = str(candidates[0])

    json_path = file_path.replace(".nii.gz", ".json").replace(".nii", ".json")
    tr = 2.0
    slice_times = None
    if Path(json_path).exists():
        with open(json_path, "r") as f:
            sidecar = json.load(f)
            tr = sidecar.get("RepetitionTime", 2.0)
            if "SliceTiming" in sidecar:
                slice_times = np.array(sidecar["SliceTiming"])

    result = load_nifti(file_path)
    result.tr = tr
    result.slice_times = slice_times
    result.info["bids_sidecar"] = json_path if Path(json_path).exists() else None

    return result


def load_fmriprep_derivatives(
    bids_root: str,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    space: str = "MNI152NLin2009cAsym",
    desc: str = "preproc",
) -> fMRIRawData:
    """Load fMRIPrep preprocessed derivatives.

    Args:
        bids_root: root of fMRIPrep derivatives directory
        subject: subject ID
        session: session ID
        task: task name
        run: run ID
        space: template space
        desc: preprocessing description (default "preproc")

    Returns:
        fMRIRawData container
    """
    if not NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel required")

    bids_root = Path(bids_root)
    pattern = f"sub-{subject}"
    if session:
        pattern += f"_ses-{session}"
    if task:
        pattern += f"_task-{task}"
    if run:
        pattern += f"_run-{run}"
    pattern += f"_space-{space}_desc-{desc}_bold.nii.gz"

    candidates = list(bids_root.rglob(pattern))
    if not candidates:
        pattern_no_gz = pattern.replace(".nii.gz", ".nii")
        candidates = list(bids_root.rglob(pattern_no_gz))

    if not candidates:
        raise FileNotFoundError(f"No fMRIPrep derivative found: {pattern}")

    return load_nifti(str(candidates[0]))


def fmri_raw_to_tensor(
    raw_data: fMRIRawData,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict]:
    """Convert fMRIRawData to GPU-ready tensor.

    Returns:
        bold: (X, Y, Z, T) tensor on device
        metadata: dict with tr, affine, slice_times, etc.
    """
    device = get_device(device)
    bold = torch.from_numpy(raw_data.bold).to(dtype=dtype, device=device)

    metadata = {
        "tr": raw_data.tr,
        "affine": torch.from_numpy(raw_data.affine).to(dtype=torch.float64),
        "shape": raw_data.shape,
        "slice_times": torch.from_numpy(raw_data.slice_times).to(dtype=torch.float32, device=device)
                       if raw_data.slice_times is not None else None,
        "file_path": raw_data.file_path,
        "info": raw_data.info,
    }

    return bold, metadata
