"""
NWB file I/O for DANDI datasets.

Reuses and extends the ecog_preprocessing.py NWB loading logic.
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from ...runtime.device_utils import get_device
import warnings

try:
    from pynwb import NWBHDF5IO
    NWB_AVAILABLE = True
except ImportError:
    NWB_AVAILABLE = False

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


@dataclass
class NWBData:
    """Container for NWB-loaded data."""
    data: np.ndarray
    sfreq: float
    ch_names: List[str]
    ch_types: List[str]
    coords: Optional[np.ndarray]
    timestamps: Optional[np.ndarray]
    metadata: Dict[str, Any]
    file_path: str


def load_dandi_nwb(
    nwb_path: str,
    processing_module: str = "ecephys",
    data_interface: str = "LFP",
    series_name: Optional[str] = None,
) -> NWBData:
    """Load neural data from DANDI NWB file.

    Args:
        nwb_path: path to .nwb file
        processing_module: NWB processing module name
        data_interface: data interface name within module
        series_name: specific electrical series name (None = first)

    Returns:
        NWBData container
    """
    if not NWB_AVAILABLE:
        raise RuntimeError("pynwb required for NWB loading. Install with: pip install pynwb")

    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()

        module = nwbfile.processing.get(processing_module, None)
        if module is None:
            available = list(nwbfile.processing.keys())
            raise ValueError(f"No '{processing_module}' module. Available: {available}")

        interface = module.data_interfaces.get(data_interface, None)
        if interface is None:
            available = list(module.data_interfaces.keys())
            raise ValueError(f"No '{data_interface}' interface. Available: {available}")

        if hasattr(interface, "electrical_series"):
            es_dict = interface.electrical_series
            if series_name is not None:
                es = es_dict[series_name]
            else:
                es = list(es_dict.values())[0]
        else:
            es = interface

        data = np.array(es.data[:]).T  # (C, T)
        sfreq = es.rate if hasattr(es, "rate") else 1.0

        timestamps = None
        if hasattr(es, "timestamps") and es.timestamps is not None:
            timestamps = np.array(es.timestamps[:])

        ch_names = [f"Ch{i}" for i in range(data.shape[0])]
        ch_types = ["eeg"] * data.shape[0]
        coords = None

        if hasattr(es, "electrodes") and es.electrodes is not None:
            electrodes = es.electrodes
            n_elec = len(electrodes)
            ch_names = []
            coords_list = []

            for i in range(n_elec):
                e = electrodes[i]
                if hasattr(e, "label"):
                    ch_names.append(str(e.label))
                else:
                    ch_names.append(f"Ch{i}")

                if all(c in electrodes.colnames for c in ["x", "y", "z"]):
                    try:
                        coords_list.append([e["x"], e["y"], e["z"]])
                    except Exception:
                        pass

                if "location" in electrodes.colnames:
                    try:
                        loc = e["location"]
                        if "ecog" in str(loc).lower():
                            ch_types[i] = "ecog"
                        elif "seeg" in str(loc).lower() or "depth" in str(loc).lower():
                            ch_types[i] = "seeg"
                    except Exception:
                        pass

            if coords_list:
                coords = np.array(coords_list)

        return NWBData(
            data=data,
            sfreq=sfreq,
            ch_names=ch_names,
            ch_types=ch_types,
            coords=coords,
            timestamps=timestamps,
            metadata={"nwb_path": nwb_path, "processing_module": processing_module},
            file_path=nwb_path,
        )


def nwb_to_tensor(
    nwb_data: NWBData,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict]:
    """Convert NWBData to GPU-ready tensor."""
    device = get_device(device)
    data = torch.from_numpy(nwb_data.data).to(dtype=dtype, device=device)

    ch_type_map = {"eeg": 0, "ecog": 1, "seeg": 2, "eog": 3, "ecg": 4, "emg": 5, "misc": 6}
    ch_types_int = [ch_type_map.get(t.lower(), 6) for t in nwb_data.ch_types]

    metadata = {
        "sfreq": nwb_data.sfreq,
        "ch_names": nwb_data.ch_names,
        "ch_types": torch.tensor(ch_types_int, dtype=torch.long),
        "coords": torch.from_numpy(nwb_data.coords).to(dtype=torch.float32, device=device)
                  if nwb_data.coords is not None else None,
        "timestamps": torch.from_numpy(nwb_data.timestamps).to(dtype=torch.float32, device=device)
                      if nwb_data.timestamps is not None else None,
        "file_path": nwb_data.file_path,
    }

    return data, metadata
