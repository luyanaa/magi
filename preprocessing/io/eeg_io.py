"""
EEG file I/O: Load raw EEG from EDF/BDF/BIDS/NWB formats.

All I/O is CPU-based. Data is transferred to GPU after loading.
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from ...utils.device_utils import get_device
from dataclasses import dataclass
import warnings

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from pynwb import NWBHDF5IO
    NWB_AVAILABLE = True
except ImportError:
    NWB_AVAILABLE = False


@dataclass
class EEGRawData:
    """Container for raw EEG data and metadata."""
    data: np.ndarray
    sfreq: float
    ch_names: List[str]
    ch_types: List[str]
    montage_3d: Optional[np.ndarray]
    info: Dict[str, Any]
    file_path: str


def load_edf(
    file_path: str,
    preload: bool = True,
    verbose: bool = False,
) -> EEGRawData:
    """Load EEG from EDF/BDF file via MNE.

    Args:
        file_path: path to .edf or .bdf file
        preload: whether to load data into memory
        verbose: print loading info

    Returns:
        EEGRawData container
    """
    if not MNE_AVAILABLE:
        raise RuntimeError("mne-python required for EDF loading. Install with: pip install mne")

    raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=verbose)

    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    ch_types = raw.get_channel_types()

    montage_3d = None
    if raw.info.get("dig") is not None and len(raw.info["dig"]) > 0:
        montage_3d = _extract_mne_montage(raw.info)

    return EEGRawData(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        ch_types=ch_types,
        montage_3d=montage_3d,
        info={"mne_info": raw.info, "n_times": raw.n_times},
        file_path=file_path,
    )


def load_nwb(
    file_path: str,
    verbose: bool = False,
) -> EEGRawData:
    """Load EEG from NWB file (DANDI format).

    Args:
        file_path: path to .nwb file
        verbose: print loading info

    Returns:
        EEGRawData container
    """
    if not NWB_AVAILABLE:
        raise RuntimeError("pynwb required for NWB loading. Install with: pip install pynwb")

    with NWBHDF5IO(file_path, "r") as io:
        nwbfile = io.read()

        ecephys = nwbfile.processing.get("ecephys", None)
        if ecephys is None:
            raise ValueError(f"No 'ecephys' processing module in {file_path}")

        lfp = ecephys.data_interfaces.get("LFP", None)
        if lfp is None:
            raise ValueError(f"No LFP data in {file_path}")

        es = list(lfp.electrical_series.values())[0]
        data = np.array(es.data[:]).T  # (C, T)
        sfreq = es.rate
        electrodes = es.electrodes

        ch_names = [str(e.label) if hasattr(e, "label") else f"Ch{i}"
                     for i, e in enumerate(electrodes[:])] if electrodes is not None else \
                    [f"Ch{i}" for i in range(data.shape[0])]

        ch_types = ["eeg"] * data.shape[0]

        montage_3d = None
        if electrodes is not None and all(c in electrodes.colnames for c in ["x", "y", "z"]):
            coords = []
            for e in electrodes:
                coords.append([e["x"], e["y"], e["z"]])
            montage_3d = np.array(coords)

    return EEGRawData(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        ch_types=ch_types,
        montage_3d=montage_3d,
        info={"nwb_path": file_path},
        file_path=file_path,
    )


def load_bids(
    bids_root: str,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    suffix: str = "eeg",
    extension: str = ".edf",
) -> EEGRawData:
    """Load EEG from BIDS dataset structure.

    Args:
        bids_root: root of BIDS dataset
        subject: subject ID (e.g., "01")
        session: session ID (e.g., "01")
        task: task name (e.g., "rest")
        run: run ID
        suffix: BIDS suffix
        extension: file extension

    Returns:
        EEGRawData container
    """
    bids_root = Path(bids_root)
    filename_parts = [f"sub-{subject}"]
    if session:
        filename_parts.append(f"ses-{session}")
    filename_parts.append(f"task-{task}" if task else suffix)
    if run:
        filename_parts.append(f"run-{run}")
    if task:
        filename_parts.append(suffix)

    filename = "_".join(filename_parts) + extension

    search_dir = bids_root / f"sub-{subject}"
    if session:
        search_dir = search_dir / f"ses-{session}"
    search_dir = search_dir / "eeg"

    file_path = search_dir / filename
    if not file_path.exists():
        candidates = list(bids_root.rglob(f"sub-{subject}*{extension}"))
        if candidates:
            file_path = candidates[0]
        else:
            raise FileNotFoundError(f"No EEG file found for sub-{subject} in {bids_root}")

    if extension in (".edf", ".bdf"):
        return load_edf(str(file_path))
    elif extension == ".nwb":
        return load_nwb(str(file_path))
    else:
        raise ValueError(f"Unsupported extension: {extension}")


def load_numpy(
    file_path: str,
    sfreq: float = 256.0,
    ch_names: Optional[List[str]] = None,
) -> EEGRawData:
    """Load pre-saved EEG from .npy/.npz file.

    Args:
        file_path: path to .npy or .npz file
        sfreq: sampling rate (must be provided)
        ch_names: channel names (default: auto-generated)

    Returns:
        EEGRawData container
    """
    if file_path.endswith(".npy"):
        data = np.load(file_path)
    elif file_path.endswith(".npz"):
        npz = np.load(file_path)
        data = npz.get("data", npz.get("eeg", npz[list(npz.keys())[0]]))
    else:
        raise ValueError(f"Unsupported format: {file_path}")

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if data.shape[0] > data.shape[1]:
        data = data.T

    n_channels = data.shape[0]
    if ch_names is None:
        ch_names = [f"Ch{i}" for i in range(n_channels)]

    return EEGRawData(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        ch_types=["eeg"] * n_channels,
        montage_3d=None,
        info={"numpy_path": file_path},
        file_path=file_path,
    )


def _extract_mne_montage(info) -> Optional[np.ndarray]:
    """Extract 3D electrode positions from MNE Info object."""
    ch_names = info["ch_names"]
    n_ch = len(ch_names)
    coords = np.zeros((n_ch, 3))

    for i in range(n_ch):
        loc = info["chs"][i]["loc"][:3]
        if np.any(loc != 0):
            coords[i] = loc * 1000

    if np.all(coords == 0):
        return None
    return coords


def eeg_raw_to_tensor(
    raw_data: EEGRawData,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict]:
    """Convert EEGRawData to GPU-ready tensor.

    Returns:
        data: (C, T) tensor on device
        metadata: dict with sfreq, ch_names, ch_types, montage_3d, etc.
    """
    device = get_device(device)
    data = torch.from_numpy(raw_data.data).to(dtype=dtype, device=device)

    ch_type_map = {"eeg": 0, "ecog": 1, "seeg": 2, "eog": 3, "ecg": 4, "emg": 5, "misc": 6}
    ch_types_int = [ch_type_map.get(t.lower(), 6) for t in raw_data.ch_types]

    metadata = {
        "sfreq": raw_data.sfreq,
        "ch_names": raw_data.ch_names,
        "ch_types": torch.tensor(ch_types_int, dtype=torch.long),
        "montage_3d": torch.from_numpy(raw_data.montage_3d).to(dtype=torch.float32, device=device)
                     if raw_data.montage_3d is not None else None,
        "file_path": raw_data.file_path,
    }

    return data, metadata
