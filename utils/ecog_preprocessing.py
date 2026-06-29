"""
ECoG / iEEG Preprocessing Pipeline

Converts raw ECoG/iEEG recordings (NWB/EDF/BIDS) into Magi-compatible format.
Handles coordinate extraction, amplitude normalization, and artifact rejection.

Usage:
    from brain_moe_pinn.utils.ecog_preprocessing import preprocess_ecog, load_dandi_nwb
    raw, coords = load_dandi_nwb("dandi://000019/sub-01.nwb")
    raw, coords = preprocess_ecog(raw, target_srate=512)
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

import torch

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("mne-python not installed. ECoG preprocessing will not work.")

try:
    from pynwb import NWBHDF5IO
    NWB_AVAILABLE = True
except ImportError:
    NWB_AVAILABLE = False


# Standard MNI coordinate reference for ECoG electrode registration
MNI_X_RANGE = (-90, 90)
MNI_Y_RANGE = (-126, 90)
MNI_Z_RANGE = (-72, 108)

# Channel type constants for ChannelTypeEmbedding
CHANNEL_TYPES = {
    0: "scalp_EEG",
    1: "ecog_grid", 
    2: "seeg_depth",
    3: "unknown"
}

# Channel type mapping for ChannelTypeEmbedding
def map_channel_type(ch_type: str) -> int:
    """
    Map MNE channel type string to integer for ChannelTypeEmbedding.

    Args:
        ch_type: MNE channel type (e.g., 'ecog', 'seeg', 'eeg', 'dbs')
    Returns:
        int: 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
    """
    ch_type = ch_type.lower()
    if ch_type in ('eeg',):
        return 0  # scalp_EEG
    elif ch_type in ('ecog', 'grid', 'strip'):
        return 1  # ecog_grid
    elif ch_type in ('seeg', 'dbs', 'depth'):
        return 2  # seeg_depth
    else:
        return 3  # unknown


def extract_mni_coords(info) -> np.ndarray:
    """
    Extract MNI coordinates from MNE Info object.

    Priority:
    1. info['dig'] digitization points (most accurate)
    2. info['chs'][i]['loc'][:3] (channel-localized positions)
    3. Standard 10-20 lookup as fallback

    Args:
        info: mne.Info object
    Returns:
        coords: (C, 3) array of MNI (x, y, z) coordinates
    """
    ch_names = info['ch_names']
    n_ch = len(ch_names)
    coords = np.zeros((n_ch, 3))

    # Try digitization points first
    if info['dig'] is not None and len(info['dig']) > 0:
        dig_coords = {d['ident']: d['r'] for d in info['dig'] if d['kind'] == 3}  # 3 = extra points
        for i, ch in enumerate(ch_names):
            if i + 1 in dig_coords:
                coords[i] = dig_coords[i + 1] * 1000  # m → mm
            else:
                coords[i] = info['chs'][i]['loc'][:3] * 1000
    else:
        # Fallback: use channel-localized positions
        for i in range(n_ch):
            coords[i] = info['chs'][i]['loc'][:3] * 1000  # m → mm

    # Validate MNI ranges
    if np.any(coords == 0):
        warnings.warn(
            f"Some coordinates are zero (missing localization). "
            f"Channels: {[ch_names[i] for i in range(n_ch) if np.all(coords[i] == 0)]}"
        )

    return coords


def preprocess_ecog(
    raw,
    target_srate: int = 512,
    l_freq: float = 0.1,
    h_freq: float = 200.0,
    notch_freqs: List[float] = None,
    reference: str = "average",
    zscore: bool = True,
    bad_channel_threshold: float = 5.0,
    verbose: bool = False,
) -> Tuple:
    """
    Preprocess ECoG/iEEG recording for Magi input.

    Pipeline:
        1. Resample to target_srate
        2. Bandpass filter (l_freq-h_freq Hz)
        3. Notch filter line noise
        4. Re-reference (average / common / bipolar)
        5. Z-score normalization per channel
        6. Bad channel detection (amplitude threshold)
        7. Extract MNI coordinates

    Args:
        raw: mne.io.Raw object (already loaded)
        target_srate: target sampling rate (default 512 for ECoG HFB)
        l_freq: high-pass cutoff (default 0.1 Hz)
        h_freq: low-pass cutoff (default 200 Hz for HFB)
        notch_freqs: list of notch frequencies (default [50, 100, 150])
        reference: re-reference method ('average', 'bipolar', or None)
        zscore: whether to z-score normalize per channel
        bad_channel_threshold: z-score threshold for bad channel detection
        verbose: print progress

    Returns:
        raw: mne.io.Raw — preprocessed raw object
        coords: np.ndarray (C, 3) — MNI coordinates in mm
        channel_types: np.ndarray (C,) — int array: 0=scalp, 1=ecog, 2=seeg, 3=unknown
    """
    if not MNE_AVAILABLE:
        raise RuntimeError("mne-python required for ECoG preprocessing")

    if verbose:
        print(f"[ECoG Preprocess] Initial: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels")

    # 1. Resample
    if raw.info['sfreq'] != target_srate:
        raw = raw.resample(target_srate, n_jobs=1)
        if verbose:
            print(f"[ECoG Preprocess] Resampled: {target_srate} Hz")

    # 2. Bandpass filter
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    if verbose:
        print(f"[ECoG Preprocess] Bandpass: {l_freq}-{h_freq} Hz")

    # 3. Notch filter
    if notch_freqs is None:
        # Auto-detect based on sampling rate region
        # Europe/Asia: 50Hz; Americas: 60Hz
        # We use both to be safe
        notch_freqs = [50, 60, 100, 120, 150, 180]
    valid_notch = [f for f in notch_freqs if f < raw.info['sfreq'] / 2]
    if valid_notch:
        raw = raw.notch_filter(valid_notch, method='fir', verbose=False)
        if verbose:
            print(f"[ECoG Preprocess] Notch: {valid_notch}")

    # 4. Re-reference
    if reference == "average":
        raw = raw.set_eeg_reference(ref_channels="average", verbose=False)
        if verbose:
            print("[ECoG Preprocess] Re-reference: common average")
    elif reference == "bipolar":
        raw = mne.set_bipolar_reference(raw, verbose=False)
        if verbose:
            print("[ECoG Preprocess] Re-reference: bipolar")

    # 5. Z-score normalization per channel
    if zscore:
        data = raw.get_data()
        means = data.mean(axis=1, keepdims=True)
        stds = data.std(axis=1, keepdims=True)
        # Avoid division by zero
        stds[stds == 0] = 1e-6
        data = (data - means) / stds
        # Update raw data in-place
        raw._data = data
        if verbose:
            print("[ECoG Preprocess] Z-score normalized per channel")

    # 6. Bad channel detection
    if bad_channel_threshold > 0:
        data = raw.get_data()
        max_z = np.abs(data).max(axis=1)
        bad_idx = np.where(max_z > bad_channel_threshold)[0]
        if len(bad_idx) > 0:
            bad_names = [raw.ch_names[i] for i in bad_idx]
            warnings.warn(
                f"Detected {len(bad_idx)} bad channels (z>{bad_channel_threshold}): {bad_names}"
            )
            raw.info['bads'] = bad_names

    # 7. Extract coordinates
    coords = extract_mni_coords(raw.info)

    # 8. Extract channel types
    channel_types = np.array([
        map_channel_type(raw.get_channel_types()[i])
        for i in range(len(raw.ch_names))
    ], dtype=np.int64)

    if verbose:
        print(f"[ECoG Preprocess] Final: {len(raw.ch_names)} channels, {raw.n_times} samples")
        print(f"[ECoG Preprocess] Coords range: x=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}], "
              f"y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}], "
              f"z=[{coords[:,2].min():.1f}, {coords[:,2].max():.1f}]")
        type_counts = {0: "scalp", 1: "ecog", 2: "seeg", 3: "unknown"}
        for t, name in type_counts.items():
            count = (channel_types == t).sum()
            if count > 0:
                print(f"[ECoG Preprocess]   {name}: {count} channels")

    return raw, coords, channel_types


def load_dandi_nwb(nwb_path: str) -> Tuple:
    """
    Load ECoG/iEEG data from DANDI NWB file.

    Args:
        nwb_path: path to .nwb file or dandi:// URI
    Returns:
        raw: mne.io.Raw
        coords: np.ndarray (C, 3)
        channel_types: np.ndarray (C,)
    """
    if not NWB_AVAILABLE:
        raise RuntimeError("pynwb required for NWB loading")

    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        # Extract electrical series
        ecephys_module = nwbfile.processing.get('ecephys', None)
        if ecephys_module is None:
            raise ValueError(f"No 'ecephys' processing module found in {nwb_path}")

        # Get the LFP or raw electrical series
        lfp = ecephys_module.data_interfaces.get('LFP', None)
        if lfp is None:
            raise ValueError(f"No LFP data found in {nwb_path}")

        # Extract data
        electrical_series = lfp.electrical_series['ElectricalSeries']
        data = electrical_series.data[:]  # (time, channels)
        timestamps = electrical_series.timestamps[:]
        electrodes = electrical_series.electrodes[:]

        # Sampling rate
        srate = electrical_series.rate

        # Build MNE Info
        ch_names = [str(e['label']) for e in electrodes]
        ch_types = ['ecog'] * len(ch_names)  # Default; refine if metadata available

        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)

        # Set coordinates if available
        if 'x' in electrodes.colnames and 'y' in electrodes.colnames and 'z' in electrodes.colnames:
            for i, e in enumerate(electrodes):
                info['chs'][i]['loc'][:3] = [e['x'], e['y'], e['z']]

        # Create Raw object
        raw = mne.io.RawArray(data.T, info)  # (channels, time)

        coords = extract_mni_coords(info)
        channel_types = np.array([map_channel_type(t) for t in ch_types])

    return raw, coords, channel_types


def load_edf_ecog(edf_path: str) -> Tuple:
    """
    Load ECoG/iEEG data from EDF/BDF file (IEEG.org format).

    Args:
        edf_path: path to .edf or .bdf file
    Returns:
        raw: mne.io.Raw
        coords: np.ndarray (C, 3) — may be zeros if not embedded
        channel_types: np.ndarray (C,)
    """
    if not MNE_AVAILABLE:
        raise RuntimeError("mne-python required for EDF loading")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Try to extract coordinates from header (IEEG.org embeds coords in channel names)
    coords = extract_mni_coords(raw.info)
    channel_types = np.array([
        map_channel_type(raw.get_channel_types()[i])
        for i in range(len(raw.ch_names))
    ])

    return raw, coords, channel_types


if __name__ == "__main__":
    print("ECoG Preprocessing Pipeline")
    print("=" * 50)
    print(f"MNE available: {MNE_AVAILABLE}")
    print(f"PyNWB available: {NWB_AVAILABLE}")
    print()
    print("Channel type mapping:")
    for name, idx in [("scalp_EEG", 0), ("ecog_grid", 1), ("seeg_depth", 2), ("unknown", 3)]:
        print(f"  {idx}: {name}")
    print()
    print("Usage:")
    print("  from brain_moe_pinn.utils.ecog_preprocessing import preprocess_ecog")
    print("  raw, coords, types = preprocess_ecog(raw_mne_object)")
