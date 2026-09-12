"""Source adapters for cross-species and human neuroimaging data ladders.

The repository stores one canonical representation regardless of source:

``<root>/manifest.csv`` plus ``<root>/<modality>/<sample_id>.npy`` arrays
with shape ``(channels, time)``.  This module keeps source-specific details at
the ingestion boundary and leaves ``SpeciesSignalDataset`` responsible only
for windowing, masks, roles, and subject-preserving splits.

Supported source records:

* NWB/DANDI ``TimeSeries`` objects (including ROI traces and explicitly
  reduced image volumes),
* NIfTI/BIDS BOLD runs (voxel or atlas-parcel traces),
* local EEG/MEG recordings through lazy MNE readers (EDF/BDF/FIF,
  BrainVision, EEGLAB, and CTF), and
* extracted Dryad/Figshare arrays in NPY/NPZ/MAT/CSV/TSV/JSON form.

The source manifest is deliberately explicit.  No network download is hidden
inside training or ingestion, and ambiguous matrix orientation is rejected for
square arrays instead of being guessed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .readers import emit_sample, read_manifest


_ARRAY_SUFFIXES = {".npy", ".npz", ".mat", ".csv", ".tsv", ".txt", ".json"}
_MNE_SUFFIXES = {
    ".edf", ".bdf", ".fif", ".fif.gz", ".vhdr", ".set", ".ds",
}


def _source_suffix(path: Path) -> str:
    """Return a compound source suffix such as ``.fif.gz``."""
    name = path.name.lower()
    for suffix in (".nii.gz", ".fif.gz", ".edf", ".bdf", ".vhdr",
                   ".set", ".ds"):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _sidecar_path(path: Path) -> Path:
    suffix = _source_suffix(path)
    if suffix in {".nii.gz", ".fif.gz"}:
        return path.with_name(path.name[:-len(suffix)] + ".json")
    return path.with_suffix(".json")


def _load_mne_recording(
    spec: Mapping[str, Any],
    base: Path,
    modality: Optional[str],
) -> LoadedModality:
    """Load an EEG/MEG recording through MNE without changing identities."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MNE sources require mne-python: pip install mne") from exc

    path = _resolve_path(spec["path"], base)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = _source_suffix(path)
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".fif": mne.io.read_raw_fif,
        ".fif.gz": mne.io.read_raw_fif,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
        ".ds": mne.io.read_raw_ctf,
    }
    reader = readers.get(suffix)
    if reader is None:
        raise ValueError(
            f"unsupported MNE source {path}; expected EDF/BDF/FIF/"
            "BrainVision/EEGLAB/CTF")
    raw = reader(
        str(path),
        preload=bool(spec.get("preload", True)),
        verbose="ERROR",
    )

    explicit_picks = spec.get("picks")
    channel_types = spec.get("channel_types")
    if isinstance(channel_types, str):
        channel_types = [channel_types]
    channel_types = [
        str(kind).lower() for kind in (channel_types or [])
    ]
    if explicit_picks is not None:
        if isinstance(explicit_picks, str):
            explicit_picks = [explicit_picks]
        include = [str(name) for name in explicit_picks]
        missing = sorted(set(include) - set(raw.ch_names))
        if missing:
            raise ValueError(
                f"MNE source {path} requested missing channels: {missing}")
        picks = list(mne.pick_channels(
            raw.ch_names, include=include, ordered=True))
    else:
        requested_kind = str(
            spec.get("modality") or modality or "").lower()
        if not channel_types:
            if requested_kind == "eeg":
                channel_types = ["eeg"]
            elif requested_kind == "meg":
                channel_types = ["meg"]
            else:
                raise ValueError(
                    "MNE source needs modality='eeg'/'meg', channel_types, "
                    "or explicit picks")
        picks = []
        for kind in channel_types:
            kwargs: Dict[str, Any] = {"exclude": []}
            if kind == "meg":
                kwargs.update(meg=True, ref_meg=False)
            elif kind in {"mag", "grad"}:
                kwargs.update(meg=kind, ref_meg=False)
            elif kind in {"eeg", "eog", "ecg", "emg", "misc", "stim"}:
                kwargs[kind] = True
            else:
                raise ValueError(
                    f"unsupported MNE channel type {kind!r}; expected "
                    "eeg, meg, mag, grad, eog, ecg, emg, misc, or stim")
            picks.extend(mne.pick_types(raw.info, **kwargs).tolist())
        picks = sorted(set(picks))

    excluded = {
        str(name) for name in (spec.get("exclude_channels") or [])
    }
    picks = [index for index in picks if raw.ch_names[index] not in excluded]
    if not picks:
        raise ValueError(f"MNE source {path} contains no selected channels")
    if spec.get("min_channels") is not None:
        minimum = int(spec["min_channels"])
        if len(picks) < minimum:
            raise ValueError(
                f"MNE source {path} has {len(picks)} selected channels; "
                f"minimum is {minimum}")
    if spec.get("max_channels") is not None:
        maximum = int(spec["max_channels"])
        if len(picks) > maximum:
            raise ValueError(
                f"MNE source {path} has {len(picks)} selected channels, "
                f"exceeding max_channels={maximum}; pass explicit picks or "
                "channel_types instead of silently dropping channels")
    raw.pick(picks)

    l_freq = spec.get("l_freq")
    h_freq = spec.get("h_freq")
    nyquist = float(raw.info["sfreq"]) / 2.0
    if h_freq is not None and float(h_freq) >= nyquist:
        raise ValueError(
            f"h_freq={h_freq} must be below the source Nyquist frequency "
            f"{nyquist:g} Hz for {path}")
    if l_freq is not None and float(l_freq) < 0:
        raise ValueError("l_freq must be non-negative")
    if l_freq is not None or h_freq is not None:
        raw.filter(
            l_freq=None if l_freq is None else float(l_freq),
            h_freq=None if h_freq is None else float(h_freq),
            verbose="ERROR",
        )
    notch = spec.get("notch_freqs", spec.get("notch"))
    if notch is not None:
        if isinstance(notch, (int, float)):
            notch = [notch]
        notch = [float(freq) for freq in notch]
        invalid = [freq for freq in notch if freq <= 0 or freq >= nyquist]
        if invalid:
            raise ValueError(
                f"notch frequencies must lie in (0, {nyquist:g}) Hz; "
                f"invalid values: {invalid}")
        if notch:
            raw.notch_filter(freqs=notch, verbose="ERROR")

    target_rate = spec.get("target_rate_hz", spec.get("resample_hz"))
    if target_rate is not None:
        target_rate = float(target_rate)
        if target_rate <= 0:
            raise ValueError("target_rate_hz must be positive")
        if not np.isclose(float(raw.info["sfreq"]), target_rate):
            raw.resample(target_rate, npad="auto", verbose="ERROR")

    all_types = raw.get_channel_types()
    data, finite = _clean_signal(raw.get_data())
    ids = [str(name) for name in raw.ch_names]
    rate_hz = float(raw.info["sfreq"])
    metadata: Dict[str, str] = {
        "source_format": "mne",
        "source_file": str(path),
        "mne_suffix": suffix,
        "channel_types": "|".join(
            str(kind) for kind in all_types),
        "duration_s": f"{data.shape[1] / rate_hz:.9g}",
    }
    sidecar = _sidecar_path(path)
    if sidecar.exists() and sidecar.is_file():
        try:
            with sidecar.open() as handle:
                sidecar_values = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"could not parse BIDS sidecar {sidecar}") from exc
        if not isinstance(sidecar_values, Mapping):
            raise ValueError(f"BIDS sidecar must contain a JSON object: {sidecar}")
        for key in (
                "TaskName", "SamplingFrequency", "PowerLineFrequency",
                "RecordingType", "Manufacturer", "DeviceSerialNumber"):
            if sidecar_values.get(key) is not None:
                metadata[f"bids_{key.lower()}"] = str(sidecar_values[key])
        metadata["bids_sidecar"] = str(sidecar)
    return LoadedModality(
        data=data,
        mask=finite,
        ids=ids,
        rate_hz=rate_hz,
        dt_s=1.0 / rate_hz,
        metadata=metadata,
    )


@dataclass
class LoadedModality:
    """One source record converted to a clean ``(C, T)`` array."""

    data: np.ndarray
    mask: np.ndarray
    ids: Optional[List[str]] = None
    rate_hz: Optional[float] = None
    dt_s: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)


def _nifti_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _numeric_candidates(values: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    candidates: Dict[str, np.ndarray] = {}
    for key, value in values.items():
        if str(key).startswith("_"):
            continue
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if array.ndim >= 1 and np.issubdtype(array.dtype, np.number):
            candidates[str(key)] = array
    return candidates


def _read_mat(path: Path, key: Optional[str]) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError:
        loadmat = None

    if loadmat is not None:
        try:
            values = loadmat(path)
        except (NotImplementedError, ValueError):
            values = None
        if values is not None:
            candidates = _numeric_candidates(values)
            if key is not None:
                if key not in candidates:
                    raise KeyError(
                        f"MAT key {key!r} not found in {path}; available: "
                        f"{sorted(candidates)}")
                return np.asarray(candidates[key])
            if len(candidates) == 1:
                return next(iter(candidates.values()))
            raise ValueError(
                f"MAT file {path} has multiple numeric arrays "
                f"{sorted(candidates)}; pass an explicit 'key'")

    # MATLAB v7.3 files are HDF5 containers.  Keep this fallback lazy so
    # ordinary array manifests do not require h5py.
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            f"cannot read {path}: install scipy for MAT files, or h5py for "
            "MAT v7.3 files") from exc
    with h5py.File(path, "r") as handle:
        candidates = {}

        def visit(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset) and node.ndim >= 1:
                try:
                    array = np.asarray(node[()])
                except Exception:
                    return
                if np.issubdtype(array.dtype, np.number):
                    candidates[name] = array

        handle.visititems(visit)
        if key is not None:
            if key not in candidates:
                raise KeyError(
                    f"MAT v7.3 key {key!r} not found in {path}; available: "
                    f"{sorted(candidates)}")
            return candidates[key]
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        raise ValueError(
            f"MAT v7.3 file {path} has multiple numeric arrays "
            f"{sorted(candidates)}; pass an explicit 'key'")


def _read_array_file(
    path: Path,
    *,
    key: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> np.ndarray:
    """Read one non-NWB, non-NIfTI numeric array without changing axes."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            chosen = key
            if chosen is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ file {path} has keys {keys}; pass an explicit "
                        "'key'")
                chosen = keys[0]
            if chosen not in archive:
                raise KeyError(
                    f"NPZ key {chosen!r} not found in {path}; available: {keys}")
            return np.asarray(archive[chosen])
    if suffix == ".mat":
        return _read_mat(path, key)
    if suffix in {".csv", ".tsv", ".txt"}:
        if delimiter is None:
            delimiter = "\t" if suffix == ".tsv" else ","
        return np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    if suffix == ".json":
        with path.open() as handle:
            return np.asarray(json.load(handle))
    raise ValueError(
        f"unsupported array source {path}; expected one of "
        f"{sorted(_ARRAY_SUFFIXES)}")


def _as_channels_time(array: np.ndarray, orientation: str = "auto") -> np.ndarray:
    """Convert a 1-D/2-D source array to the canonical ``(C, T)`` layout."""

    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            f"source array must be 1-D or 2-D before orientation; got "
            f"shape {array.shape}")
    orientation = str(orientation).lower()
    if orientation in {"ct", "channels_time", "channels-first"}:
        out = array
    elif orientation in {"tc", "time_channels", "time-first"}:
        out = array.T
    elif orientation == "auto":
        if array.shape[0] == array.shape[1] and array.shape[0] > 1:
            raise ValueError(
                "square source array has ambiguous orientation; set "
                "orientation to 'ct' or 'tc'")
        # Most extracted recordings are time-major.  The heuristic is only a
        # convenience for non-square arrays; source manifests should state the
        # orientation whenever the source contract is known.
        out = array.T if array.shape[0] > array.shape[1] else array
    else:
        raise ValueError("orientation must be 'ct', 'tc', or 'auto'")
    if out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError(f"source array has an empty dimension: {out.shape}")
    return np.ascontiguousarray(out, dtype=np.float32)


def _clean_signal(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    mask = np.isfinite(array)
    if not mask.any():
        raise ValueError("source array contains no finite samples")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), mask

def encode_optogenetic_control(
    waveform: np.ndarray,
    target_ids: Sequence[str],
    target_vocab: Sequence[str],
    *,
    gate_channel: int = 0,
    active_threshold: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """Append target-gated optogenetic features to a control waveform.

    The returned channels are ``[waveform, drive * target_one_hot]``.  Target
    identity therefore cannot drive the SDE when the light drive is zero:
    ``u_t = 0`` remains an exact no-op.  ``target_vocab`` is intentionally
    explicit and fixed for a species so batches retain one control dimension.
    """
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("optogenetic waveform must have shape (U, T)")
    if not target_ids:
        raise ValueError("optogenetic control needs at least one target_id")
    vocab = [str(item) for item in target_vocab]
    if not vocab or len(set(vocab)) != len(vocab):
        raise ValueError("target_vocab must be a non-empty unique sequence")
    targets = [str(item) for item in target_ids]
    unknown = sorted(set(targets) - set(vocab))
    if unknown:
        raise ValueError(
            f"optogenetic targets are absent from target_vocab: {unknown}")
    if gate_channel < 0 or gate_channel >= values.shape[0]:
        raise ValueError(
            f"gate_channel {gate_channel} is outside waveform channels "
            f"{values.shape[0]}")
    if active_threshold < 0:
        raise ValueError("active_threshold must be non-negative")
    drive = np.maximum(
        np.abs(values[gate_channel]) - float(active_threshold), 0.0)
    target_features = np.zeros(
        (len(vocab), values.shape[1]), dtype=np.float32)
    positions = {target: index for index, target in enumerate(vocab)}
    for target in targets:
        target_features[positions[target]] += drive
    features = np.concatenate((values, target_features), axis=0)
    feature_ids = [
        f"waveform_{index:03d}" for index in range(values.shape[0])
    ] + [f"target:{target}" for target in vocab]
    return np.ascontiguousarray(features), feature_ids


def _read_ids(value: Any, base: Path) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    path = _resolve_path(value, base)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            values = json.load(handle)
        if not isinstance(values, list):
            raise ValueError(f"channel id JSON must contain a list: {path}")
        return [str(item) for item in values]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def _augment_optogenetic_control(
    spec: Mapping[str, Any],
    record: LoadedModality,
    base: Path,
) -> None:
    target_value = spec.get("target_ids")
    if target_value is None and spec.get("target_id") is not None:
        target_value = [spec["target_id"]]
    if isinstance(target_value, str):
        target_ids = [target_value]
    else:
        target_ids = _read_ids(target_value, base)
    target_vocab = _read_ids(spec.get("target_vocab"), base)
    if target_ids is None or target_vocab is None:
        raise ValueError(
            "optogenetic control requires explicit target_ids and target_vocab")
    waveform_channels = record.data.shape[0]
    encoded, feature_ids = encode_optogenetic_control(
        record.data,
        target_ids,
        target_vocab,
        gate_channel=int(spec.get("gate_channel", 0)),
        active_threshold=float(spec.get("active_threshold", 0.0)),
    )
    target_mask = np.ones(
        (len(target_vocab), record.data.shape[1]), dtype=bool)
    record.data = encoded
    record.mask = np.concatenate((record.mask, target_mask), axis=0)
    # These are control features, not observation channels.  Do not expose
    # them to union alignment as if target codes were signal identities.
    record.ids = None
    record.metadata.update({
        "control_kind": "optogenetic",
        "target_scope": str(spec.get("target_scope", "neuron")),
        "target_ids": "|".join(str(item) for item in target_ids),
        "target_vocab_size": str(len(target_vocab)),
        "target_feature_ids": "|".join(feature_ids[waveform_channels:]),
    })


def _read_mask(
    value: Any,
    base: Path,
    shape: Tuple[int, int],
    orientation: str,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        raw = np.asarray(value)
    else:
        raw = _read_array_file(_resolve_path(value, base))
    # Masks are normally already stored in canonical (C,T) order.  Check the
    # exact shape before applying an orientation heuristic; otherwise a
    # high-channel, short-recording mask (e.g. 400 x 20 fMRI) is transposed.
    if raw.shape == shape:
        mask = raw
    elif raw.shape == (shape[1], shape[0]):
        mask = raw.T
    else:
        mask = _as_channels_time(raw, orientation)
    if mask.shape != shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match signal shape {shape}")
    return mask.astype(bool, copy=False)


def _rate_from_timestamps(timestamps: Optional[np.ndarray]) -> Optional[float]:
    if timestamps is None:
        return None
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return None
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs)) if diffs.size else None


def _resolve_nwb_series(nwb: Any, name: str) -> Any:
    if name in nwb.acquisition:
        return nwb.acquisition[name]
    for module in nwb.processing.values():
        if name in module.data_interfaces:
            return module.data_interfaces[name]
    # Some callers use a processing-module-qualified name.
    if "/" in name:
        module_name, local_name = name.split("/", 1)
        if module_name in nwb.processing:
            module = nwb.processing[module_name]
            if local_name in module.data_interfaces:
                return module.data_interfaces[local_name]
    available = list(nwb.acquisition.keys())
    for module in nwb.processing.values():
        available.extend(f"{module.name}/{key}" for key in module.data_interfaces)
    raise KeyError(f"NWB series {name!r} not found; available: {sorted(available)}")


def _reduce_nwb_volume(
    raw: np.ndarray,
    *,
    reduce: Optional[str],
    max_channels: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Reduce a time-first NWB volume and return ``(C,T), mask, ids``."""

    if raw.ndim <= 2:
        raise ValueError("volume reduction is only needed for arrays with >2 dimensions")
    time_major = raw.reshape(raw.shape[0], -1)
    finite = np.isfinite(time_major)
    cleaned = np.nan_to_num(time_major, nan=0.0, posinf=0.0, neginf=0.0)
    mode = (reduce or "").lower()
    if mode == "mean":
        counts = finite.sum(axis=1)
        sums = cleaned.sum(axis=1)
        values = np.divide(sums, np.maximum(counts, 1), dtype=np.float32)
        mask = (counts > 0).reshape(1, -1)
        return values.reshape(1, -1), mask, None
    if mode not in {"flatten", "variance_topk"}:
        raise ValueError(
            "NWB arrays with >2 dimensions require reduce='mean', "
            "'flatten', or 'variance_topk'")
    values = cleaned.T
    mask = finite.T
    ids: Optional[List[str]] = None
    if mode == "variance_topk" and max_channels is not None and values.shape[0] > max_channels:
        variance = np.nanvar(np.where(mask, values, np.nan), axis=1)
        variance = np.nan_to_num(variance, nan=0.0)
        order = np.argsort(-variance, kind="stable")[: int(max_channels)]
        order.sort()
        values, mask = values[order], mask[order]
        ids = [f"volume_{int(index):06d}" for index in order]
    elif mode == "flatten":
        if max_channels is not None and values.shape[0] > max_channels:
            variance = np.nanvar(np.where(mask, values, np.nan), axis=1)
            variance = np.nan_to_num(variance, nan=0.0)
            order = np.argsort(-variance, kind="stable")[: int(max_channels)]
            order.sort()
            values, mask = values[order], mask[order]
            ids = [f"volume_{int(index):06d}" for index in order]
    return values.astype(np.float32), mask, ids


def _load_nwb_series(spec: Mapping[str, Any], base: Path) -> LoadedModality:
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise ImportError(
            "NWB sources require pynwb: pip install pynwb") from exc
    path = _resolve_path(spec["path"], base)
    series_name = spec.get("series") or spec.get("key")
    if not series_name:
        raise ValueError(f"NWB source {path} needs a 'series' name")
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        series = _resolve_nwb_series(nwb, str(series_name))
        raw = np.asarray(series.data[:])
        timestamps = None
        if getattr(series, "timestamps", None) is not None:
            timestamps = np.asarray(series.timestamps[:])
        rate = getattr(series, "rate", None)
        rate_hz = float(rate) if rate is not None and float(rate) > 0 else _rate_from_timestamps(timestamps)
        if raw.ndim > 2:
            data, mask, generated_ids = _reduce_nwb_volume(
                raw, reduce=spec.get("reduce"),
                max_channels=spec.get("max_channels"))
        else:
            oriented = _as_channels_time(raw, str(spec.get("orientation", "tc")))
            data, mask = _clean_signal(oriented)
            generated_ids = None
        explicit_ids = _read_ids(spec.get("ids"), base)
        ids = explicit_ids if explicit_ids is not None else generated_ids
        if ids is not None and len(ids) != data.shape[0]:
            raise ValueError(
                f"NWB ids count {len(ids)} does not match {data.shape[0]} channels")
        metadata = {
            "source_format": "nwb",
            "source_file": str(path),
            "source_series": str(series_name),
        }
        if getattr(series, "description", None):
            metadata["series_description"] = str(series.description)
    return LoadedModality(data, mask, ids, rate_hz, None, metadata)


def _atlas_matrix(
    data: np.ndarray,
    atlas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if data.ndim != 4:
        raise ValueError(f"NIfTI BOLD data must be 4-D, got {data.shape}")
    if atlas.shape != data.shape[:3]:
        raise ValueError(
            f"atlas shape {atlas.shape} does not match BOLD spatial shape "
            f"{data.shape[:3]}")
    flat_data = data.reshape(-1, data.shape[-1])
    flat_atlas = np.asarray(atlas).reshape(-1)
    labels = sorted({int(x) for x in flat_atlas if np.isfinite(x) and int(x) > 0})
    if not labels:
        raise ValueError("atlas contains no positive parcel labels")
    traces: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    ids: List[str] = []
    for label in labels:
        voxels = flat_data[flat_atlas == label]
        finite = np.isfinite(voxels)
        counts = finite.sum(axis=0)
        clean = np.nan_to_num(voxels, nan=0.0, posinf=0.0, neginf=0.0)
        trace = np.divide(
            clean.sum(axis=0), np.maximum(counts, 1), dtype=np.float32)
        traces.append(trace)
        masks.append(counts > 0)
        ids.append(f"region_{label:04d}")
    return np.stack(traces), np.stack(masks), ids


def _load_nifti(spec: Mapping[str, Any], base: Path) -> LoadedModality:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "NIfTI/BIDS sources require nibabel: pip install nibabel") from exc
    path = _resolve_path(spec["path"], base)
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim != 4:
        raise ValueError(f"NIfTI BOLD data must be 4-D, got {data.shape}")
    atlas_value = spec.get("atlas") or spec.get("atlas_path")
    if atlas_value:
        atlas_path = _resolve_path(atlas_value, base)
        atlas = np.asarray(nib.load(str(atlas_path)).get_fdata())
        matrix, mask, ids = _atlas_matrix(data, atlas)
        parcellation = str(atlas_path)
    else:
        time_major = data.reshape(-1, data.shape[-1])
        finite = np.isfinite(time_major)
        matrix = np.nan_to_num(time_major, nan=0.0, posinf=0.0, neginf=0.0)
        mask = finite
        max_channels = spec.get("max_channels")
        if max_channels is not None and matrix.shape[0] > int(max_channels):
            variance = np.nanvar(np.where(mask, matrix, np.nan), axis=1)
            variance = np.nan_to_num(variance, nan=0.0)
            order = np.argsort(-variance, kind="stable")[: int(max_channels)]
            order.sort()
            matrix, mask = matrix[order], mask[order]
            ids = [f"voxel_{int(index):08d}" for index in order]
        else:
            ids = [f"voxel_{index:08d}" for index in range(matrix.shape[0])]
        parcellation = "variance_topk" if max_channels is not None else "voxel"
    rate_hz = None
    tr = spec.get("tr_s") or spec.get("tr")
    if tr is None:
        zooms = image.header.get_zooms()
        tr = zooms[3] if len(zooms) > 3 else None
    if tr is not None and float(tr) > 0:
        rate_hz = 1.0 / float(tr)
    metadata = {
        "source_format": "nifti",
        "source_file": str(path),
        "parcellation": parcellation,
    }
    return LoadedModality(matrix.astype(np.float32), mask, ids, rate_hz, None, metadata)


def load_modality(
    spec: Mapping[str, Any] | str,
    base: Path,
    *,
    modality: Optional[str] = None,
) -> LoadedModality:
    """Load one source-manifest modality specification.

    ``format='mne'`` handles local EEG/MEG recordings.  It deliberately
    requires the modality, channel types, or explicit picks so a raw recording
    cannot silently mix EEG, reference, stimulus, or auxiliary channels.
    """
    if isinstance(spec, str):
        spec = {"path": spec}
    spec = dict(spec)
    if "values" in spec:
        raw = np.asarray(spec["values"])
        data = _as_channels_time(raw, str(spec.get("orientation", "auto")))
        data, finite = _clean_signal(data)
        metadata = {"source_format": "inline"}
        generated = LoadedModality(data, finite, metadata=metadata)
    else:
        if "path" not in spec:
            raise ValueError("modality source needs 'path' or inline 'values'")
        path = _resolve_path(spec["path"], base)
        fmt = str(spec.get("format", "")).lower()
        if not fmt:
            suffix = _source_suffix(path)
            fmt = (
                "nifti" if suffix in {".nii", ".nii.gz"} else
                "nwb" if suffix == ".nwb" else
                "mne" if suffix in _MNE_SUFFIXES else "array"
            )
        if fmt == "nwb":
            generated = _load_nwb_series(spec, base)
        elif fmt in {"nifti", "nii", "bids_fmri"}:
            generated = _load_nifti(spec, base)
        elif fmt in {
                "mne", "edf", "bdf", "fif", "brainvision", "eeglab",
                "ctf", "bids_eeg", "bids_meg"}:
            requested_modality = str(
                spec.get("modality") or modality or "").lower()
            if requested_modality in {"eeg", "meg"}:
                mne_modality = requested_modality
            elif fmt in {"bids_eeg", "edf", "bdf", "brainvision", "eeglab"}:
                mne_modality = "eeg"
            elif fmt in {"bids_meg", "fif", "ctf"}:
                mne_modality = "meg"
            else:
                mne_modality = None
            generated = _load_mne_recording(
                {**spec, "modality": mne_modality},
                base,
                str(mne_modality) if mne_modality else None,
            )
        else:
            raw = _read_array_file(
                path, key=spec.get("key"), delimiter=spec.get("delimiter"))
            data = _as_channels_time(raw, str(spec.get("orientation", "auto")))
            data, finite = _clean_signal(data)
            generated = LoadedModality(
                data=data,
                mask=finite,
                metadata={"source_format": fmt, "source_file": str(path)},
            )
    mask_override = _read_mask(
        spec.get("mask"), base, generated.data.shape,
        str(spec.get("orientation", "auto")))
    if mask_override is not None:
        generated.mask &= mask_override
        generated.data = np.where(generated.mask, generated.data, 0.0)
    explicit_ids = _read_ids(spec.get("ids"), base)
    if explicit_ids is not None:
        generated.ids = explicit_ids
    if generated.ids is not None and len(generated.ids) != generated.data.shape[0]:
        raise ValueError(
            f"channel ids count {len(generated.ids)} does not match "
            f"{generated.data.shape[0]} channels")
    if spec.get("rate_hz") is not None:
        generated.rate_hz = float(spec["rate_hz"])
        if generated.rate_hz <= 0:
            raise ValueError("modality rate_hz must be positive")
        if spec.get("dt_s") is None:
            generated.dt_s = 1.0 / generated.rate_hz
    if spec.get("dt_s") is not None:
        generated.dt_s = float(spec["dt_s"])
        generated.rate_hz = 1.0 / generated.dt_s
    if generated.rate_hz is not None and generated.rate_hz <= 0:
        raise ValueError("modality rate_hz must be positive")
    if generated.dt_s is not None and generated.dt_s <= 0:
        raise ValueError("modality dt_s must be positive")
    return generated


def _emit_loaded_sample(
    out_root: str | Path,
    sample_id: str,
    loaded: Mapping[str, LoadedModality],
    *,
    species: str,
    sample: Mapping[str, Any],
    source_name: str,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Emit loaded modalities while preserving per-modality clocks."""
    if not loaded:
        raise ValueError(f"sample {sample_id!r} has no loaded modalities")
    rates = [record.rate_hz for record in loaded.values()
             if record.rate_hz is not None]
    sample_rate = sample.get("rate_hz")
    dt_s = sample.get("dt_s")
    if sample_rate is None or dt_s is None:
        for row in read_manifest(Path(out_root) / "manifest.csv"):
            if row.get("sample_id") != sample_id:
                continue
            if sample_rate is None and row.get("rate_hz"):
                sample_rate = row["rate_hz"]
            if dt_s is None and row.get("dt_s"):
                dt_s = row["dt_s"]
            break
    if sample_rate is None and rates:
        sample_rate = rates[0]
    if dt_s is None and sample_rate is not None:
        dt_s = 1.0 / float(sample_rate)
    if sample_rate is not None and float(sample_rate) <= 0:
        raise ValueError(f"sample {sample_id!r} has non-positive rate_hz")
    if dt_s is not None and float(dt_s) <= 0:
        raise ValueError(f"sample {sample_id!r} has non-positive dt_s")
    extra = _manifest_extra(species, loaded, sample)
    for key in (
            "cross_modal_label", "split_hint", "alignment_quality",
            "time_offset_s", "window_start_s"):
        if sample.get(key) is not None:
            extra[key] = str(sample[key])
    return emit_sample(
        Path(out_root),
        sample_id,
        {modality: record.data for modality, record in loaded.items()},
        subject=str(sample.get("subject") or sample_id),
        session=str(sample.get("session") or "1"),
        origin=str(sample.get("origin") or source_name),
        condition=str(sample.get("condition") or ""),
        rate_hz=float(sample_rate) if sample_rate is not None else None,
        dt_s=float(dt_s) if dt_s is not None else None,
        masks={
            modality: record.mask
            for modality, record in loaded.items()
            if not record.mask.all()
        } or None,
        ids={
            modality: record.ids
            for modality, record in loaded.items()
            if record.ids is not None
        } or None,
        overwrite=overwrite,
        extra=extra,
    )


def _manifest_extra(
    species: str,
    loaded: Mapping[str, LoadedModality],
    sample: Mapping[str, Any],
) -> Dict[str, str]:
    extra: Dict[str, str] = {"species": species}
    for key in ("dataset", "task", "anesthesia", "notes"):
        if sample.get(key) is not None:
            extra[key] = str(sample[key])
    for modality, record in loaded.items():
        if record.rate_hz is not None:
            extra[f"{modality}_rate_hz"] = f"{record.rate_hz:.9g}"
        if record.dt_s is not None:
            extra[f"{modality}_dt_s"] = f"{record.dt_s:.9g}"
        for key, value in record.metadata.items():
            extra[f"{modality}_{key}"] = str(value)
    return extra


def ingest_source_manifest(
    source_manifest: str | Path,
    out_root: str | Path,
    *,
    species: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Ingest a source manifest into the canonical species ladder layout.

    Source-manifest example::

        {
          "species": "mouse",
          "samples": [{
            "sample_id": "ds004402__sub-01__ses-01",
            "subject": "sub-01",
            "session": "ses-01",
            "origin": "OpenNeuro:ds004402",
            "condition": "odor_discrimination",
            "modalities": {
              "fmri": {"format": "nifti", "path": "...bold.nii.gz", "tr_s": 1.0}
            }
          }]
        }

    Relative source paths are resolved against the source-manifest directory.
    """

    source_manifest = Path(source_manifest)
    with source_manifest.open() as handle:
        document = json.load(handle)
    manifest_base = source_manifest.parent
    declared_species = document.get("species")
    species = species or declared_species
    if not species:
        raise ValueError("source manifest needs a 'species' field")
    if declared_species and declared_species != species:
        raise ValueError(
            f"source manifest species {declared_species!r} disagrees with "
            f"requested {species!r}")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("source manifest needs a non-empty 'samples' list")
    defaults = dict(document.get("defaults") or {})
    out_root = Path(out_root)
    written = 0
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"sample {index} must be an object")
        sample = {**defaults, **dict(raw_sample)}
        sample_id = str(sample.get("sample_id") or f"sample_{index:05d}")
        modalities = sample.get("modalities") or sample.get("signals")
        if not isinstance(modalities, Mapping) or not modalities:
            raise ValueError(f"sample {sample_id!r} needs a modalities mapping")
        loaded: Dict[str, LoadedModality] = {}
        for modality_name, source_spec in modalities.items():
            modality = str(modality_name)
            spec = ({"path": source_spec}
                    if isinstance(source_spec, str)
                    else dict(source_spec))
            record = load_modality(
                spec, manifest_base, modality=modality)
            kind = str(spec.get("control_kind", "")).lower()
            if modality == "opto" or kind == "optogenetic":
                _augment_optogenetic_control(spec, record, manifest_base)
            loaded[modality] = record
        _emit_loaded_sample(
            out_root,
            sample_id,
            loaded,
            species=species,
            sample=sample,
            source_name=str(
                sample.get("origin") or document.get("origin") or
                source_manifest.stem),
            overwrite=overwrite,
        )
        written += 1
    print(f"[corpus] ingested {written} samples -> {out_root}")
    return out_root / "manifest.csv"


def _find_bids_bold(
    bids_root: str | Path,
    subject: str,
    *,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    space: Optional[str] = None,
) -> Path:
    root = Path(bids_root)
    candidates = []
    for path in root.rglob("*.nii*"):
        name = path.name
        if not (name.endswith("_bold.nii") or name.endswith("_bold.nii.gz")):
            continue
        required = [f"sub-{subject}"]
        if session:
            required.append(f"ses-{session}")
        if task:
            required.append(f"task-{task}")
        if run:
            required.append(f"run-{run}")
        if space:
            required.append(f"space-{space}")
        if all(token in name for token in required):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"no BOLD NIfTI found under {root} for sub-{subject} "
            f"session={session!r}, task={task!r}, run={run!r}, space={space!r}")
    candidates = sorted(candidates)
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in candidates[:5])
        raise FileExistsError(
            "multiple BIDS BOLD recordings match; specify session/task/run/"
            f"space or an explicit path: {choices}")
    return candidates[0]


def ingest_bids_fmri_session(
    bids_root: str | Path,
    out_root: str | Path,
    *,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    space: Optional[str] = None,
    bold_path: Optional[str | Path] = None,
    atlas: Optional[str | Path] = None,
    tr_s: Optional[float] = None,
    max_channels: Optional[int] = 400,
    sample_id: Optional[str] = None,
    origin: str = "OpenNeuro",
    condition: Optional[str] = None,
    cross_modal_label: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Ingest one BIDS BOLD run into the canonical ``fmri/`` ladder."""
    bids_root = Path(bids_root)
    if bold_path is None:
        bold = _find_bids_bold(
            bids_root, subject, session=session, task=task, run=run,
            space=space)
    else:
        bold = _resolve_path(bold_path, bids_root)
    sidecar = _sidecar_path(bold)
    if tr_s is None and sidecar.exists():
        with sidecar.open() as handle:
            tr_s = json.load(handle).get("RepetitionTime")
    spec: Dict[str, Any] = {
        "format": "nifti",
        "path": str(bold),
        "tr_s": tr_s,
        "max_channels": max_channels,
        "modality": "fmri",
    }
    if atlas is not None:
        spec["atlas"] = str(atlas)
    loaded = load_modality(spec, Path("."), modality="fmri")
    sample_id = sample_id or _bids_sample_id(
        origin, subject, session=session, task=task, run=run)
    sample: Dict[str, Any] = {
        "dataset": origin,
        "task": task or "rest",
        "subject": _bids_entity(subject, "sub-"),
        "session": _bids_entity(session, "ses-") if session else "1",
        "condition": condition or task or "rest",
    }
    if cross_modal_label is not None:
        sample["cross_modal_label"] = cross_modal_label
    _emit_loaded_sample(
        out_root,
        sample_id,
        {"fmri": loaded},
        species="human",
        sample=sample,
        source_name=origin,
        overwrite=overwrite,
    )
    return Path(out_root) / "manifest.csv"
def _bids_entity(value: str, prefix: str) -> str:
    text = str(value)
    return text if text.startswith(prefix) else f"{prefix}{text}"


def _bids_sample_id(
    origin: str,
    subject: str,
    *,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
) -> str:
    prefix = str(origin).split(":", 1)[0].strip().lower()
    prefix = "".join(
        char if char.isalnum() else "_" for char in prefix).strip("_")
    parts = [prefix or "bids", _bids_entity(subject, "sub-")]
    if session:
        parts.append(_bids_entity(session, "ses-"))
    if task:
        parts.append(_bids_entity(task, "task-"))
    if run:
        parts.append(_bids_entity(run, "run-"))
    return "__".join(parts)


def _find_bids_recording(
    bids_root: str | Path,
    modality: str,
    subject: str,
    *,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
) -> Path:
    """Find one BIDS EEG or MEG recording without downloading data."""
    modality = str(modality).lower()
    if modality not in {"eeg", "meg"}:
        raise ValueError("BIDS recording modality must be 'eeg' or 'meg'")
    root = Path(bids_root)
    allowed = (
        {".edf", ".bdf", ".fif", ".fif.gz", ".vhdr", ".set"}
        if modality == "eeg" else
        {".fif", ".fif.gz", ".ds"}
    )
    required = {
        _bids_entity(subject, "sub-"),
    }
    if session:
        required.add(_bids_entity(session, "ses-"))
    if task:
        required.add(_bids_entity(task, "task-"))
    if run:
        required.add(_bids_entity(run, "run-"))
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() and not (
                modality == "meg" and path.is_dir()):
            continue
        suffix = _source_suffix(path)
        if suffix not in allowed:
            continue
        stem = path.name[:-len(suffix)]
        tokens = set(stem.split("_"))
        if f"_{modality}" not in stem:
            continue
        if required.issubset(tokens):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"no BIDS {modality.upper()} recording found under {root} for "
            f"sub-{str(subject).removeprefix('sub-')}; "
            f"session={session!r}, task={task!r}, run={run!r}")
    candidates = sorted(candidates)
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in candidates[:5])
        raise FileExistsError(
            f"multiple BIDS {modality.upper()} recordings match; specify "
            f"session/task/run or an explicit path: {choices}")
    return candidates[0]


def _resolve_bids_source(
    bids_root: Path,
    path: Optional[str | Path],
    modality: str,
    subject: str,
    *,
    session: Optional[str],
    task: Optional[str],
    run: Optional[str],
) -> Path:
    if path is None:
        return _find_bids_recording(
            bids_root, modality, subject, session=session, task=task,
            run=run)
    return _resolve_path(path, bids_root)


def _ingest_bids_electrical_session(
    bids_root: str | Path,
    out_root: str | Path,
    *,
    modality: str,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    recording_path: Optional[str | Path] = None,
    target_rate_hz: Optional[float] = None,
    channel_types: Optional[Sequence[str]] = None,
    picks: Optional[Sequence[str]] = None,
    exclude_channels: Optional[Sequence[str]] = None,
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freqs: Optional[Sequence[float]] = None,
    min_channels: Optional[int] = None,
    max_channels: Optional[int] = None,
    sample_id: Optional[str] = None,
    origin: str = "BIDS",
    condition: Optional[str] = None,
    cross_modal_label: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Ingest one local BIDS EEG/MEG recording into ``(C,T)`` arrays."""
    modality = str(modality).lower()
    bids_root = Path(bids_root)
    recording = _resolve_bids_source(
        bids_root, recording_path, modality, subject, session=session,
        task=task, run=run)
    spec: Dict[str, Any] = {
        "format": "mne",
        "path": str(recording),
        "modality": modality,
    }
    if target_rate_hz is not None:
        spec["target_rate_hz"] = target_rate_hz
    if channel_types is not None:
        spec["channel_types"] = list(channel_types)
    if picks is not None:
        spec["picks"] = list(picks)
    if exclude_channels is not None:
        spec["exclude_channels"] = list(exclude_channels)
    if l_freq is not None:
        spec["l_freq"] = l_freq
    if h_freq is not None:
        spec["h_freq"] = h_freq
    if notch_freqs is not None:
        spec["notch_freqs"] = list(notch_freqs)
    if min_channels is not None:
        spec["min_channels"] = min_channels
    if max_channels is not None:
        spec["max_channels"] = max_channels
    loaded = load_modality(spec, Path("."), modality=modality)
    sample_id = sample_id or _bids_sample_id(
        origin, subject, session=session, task=task, run=run)
    sample: Dict[str, Any] = {
        "dataset": origin,
        "task": condition or task or "rest",
        "subject": _bids_entity(subject, "sub-"),
        "session": _bids_entity(session, "ses-") if session else "1",
        "condition": condition or task or "rest",
    }
    if cross_modal_label is not None:
        sample["cross_modal_label"] = cross_modal_label
    _emit_loaded_sample(
        out_root,
        sample_id,
        {modality: loaded},
        species="human",
        sample=sample,
        source_name=origin,
        overwrite=overwrite,
    )
    return Path(out_root) / "manifest.csv"


def ingest_bids_eeg_session(
    bids_root: str | Path,
    out_root: str | Path,
    *,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    recording_path: Optional[str | Path] = None,
    target_rate_hz: Optional[float] = None,
    channel_types: Optional[Sequence[str]] = None,
    picks: Optional[Sequence[str]] = None,
    exclude_channels: Optional[Sequence[str]] = None,
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freqs: Optional[Sequence[float]] = None,
    min_channels: Optional[int] = None,
    max_channels: Optional[int] = None,
    sample_id: Optional[str] = None,
    origin: str = "BIDS",
    condition: Optional[str] = None,
    cross_modal_label: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Ingest one local BIDS EEG run; MNE is loaded lazily."""
    return _ingest_bids_electrical_session(
        bids_root, out_root, modality="eeg", subject=subject,
        session=session, task=task, run=run, recording_path=recording_path,
        target_rate_hz=target_rate_hz, channel_types=channel_types,
        picks=picks, exclude_channels=exclude_channels, l_freq=l_freq,
        h_freq=h_freq, notch_freqs=notch_freqs, min_channels=min_channels,
        max_channels=max_channels, sample_id=sample_id, origin=origin,
        condition=condition, cross_modal_label=cross_modal_label,
        overwrite=overwrite)


def ingest_bids_meg_session(
    bids_root: str | Path,
    out_root: str | Path,
    *,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    recording_path: Optional[str | Path] = None,
    target_rate_hz: Optional[float] = None,
    channel_types: Optional[Sequence[str]] = None,
    picks: Optional[Sequence[str]] = None,
    exclude_channels: Optional[Sequence[str]] = None,
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freqs: Optional[Sequence[float]] = None,
    min_channels: Optional[int] = None,
    max_channels: Optional[int] = None,
    sample_id: Optional[str] = None,
    origin: str = "BIDS",
    condition: Optional[str] = None,
    cross_modal_label: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Ingest one local BIDS MEG run; MNE is loaded lazily."""
    return _ingest_bids_electrical_session(
        bids_root, out_root, modality="meg", subject=subject,
        session=session, task=task, run=run, recording_path=recording_path,
        target_rate_hz=target_rate_hz, channel_types=channel_types,
        picks=picks, exclude_channels=exclude_channels, l_freq=l_freq,
        h_freq=h_freq, notch_freqs=notch_freqs, min_channels=min_channels,
        max_channels=max_channels, sample_id=sample_id, origin=origin,
        condition=condition, cross_modal_label=cross_modal_label,
        overwrite=overwrite)


def ingest_bids_session(
    bids_root: str | Path,
    out_root: str | Path,
    *,
    subject: str,
    modalities: Sequence[str] = ("eeg", "meg", "fmri"),
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    source_paths: Optional[Mapping[str, str | Path]] = None,
    atlas: Optional[str | Path] = None,
    tr_s: Optional[float] = None,
    space: Optional[str] = None,
    max_fmri_channels: Optional[int] = 400,
    eeg_target_rate_hz: Optional[float] = None,
    meg_target_rate_hz: Optional[float] = None,
    eeg_channel_types: Optional[Sequence[str]] = None,
    meg_channel_types: Optional[Sequence[str]] = None,
    eeg_picks: Optional[Sequence[str]] = None,
    meg_picks: Optional[Sequence[str]] = None,
    condition: Optional[str] = None,
    sample_id: Optional[str] = None,
    origin: str = "BIDS",
    cross_modal_label: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """Ingest selected BIDS EEG, MEG, and/or fMRI runs as one sample row.

    The same ``sample_id`` is used for every selected modality, while each
    modality keeps its own sampling rate.  Synchronous status is never
    inferred from shared BIDS entities; pass ``cross_modal_label=1`` only for
    a verified synchronized recording and ``0`` for an intentionally
    asynchronous pair.
    """
    requested = tuple(dict.fromkeys(str(item).lower() for item in modalities))
    if not requested or set(requested) - {"eeg", "meg", "fmri"}:
        raise ValueError("modalities must contain eeg, meg, and/or fmri")
    source_paths = dict(source_paths or {})
    common_id = sample_id or _bids_sample_id(
        origin, subject, session=session, task=task, run=run)
    if "eeg" in requested:
        ingest_bids_eeg_session(
            bids_root, out_root, subject=subject, session=session, task=task,
            run=run, recording_path=source_paths.get("eeg"),
            target_rate_hz=eeg_target_rate_hz,
            channel_types=eeg_channel_types, picks=eeg_picks,
            sample_id=common_id, origin=origin, condition=condition,
            cross_modal_label=cross_modal_label, overwrite=overwrite)
    if "meg" in requested:
        ingest_bids_meg_session(
            bids_root, out_root, subject=subject, session=session, task=task,
            run=run, recording_path=source_paths.get("meg"),
            target_rate_hz=meg_target_rate_hz,
            channel_types=meg_channel_types, picks=meg_picks,
            sample_id=common_id, origin=origin, condition=condition,
            cross_modal_label=cross_modal_label, overwrite=overwrite)
    if "fmri" in requested:
        ingest_bids_fmri_session(
            bids_root, out_root, subject=subject, session=session, task=task,
            run=run, space=space, bold_path=source_paths.get("fmri"),
            atlas=atlas, tr_s=tr_s, max_channels=max_fmri_channels,
            sample_id=common_id, origin=origin, condition=condition,
            cross_modal_label=cross_modal_label, overwrite=overwrite)
    return Path(out_root) / "manifest.csv"




def validate_ladder(
    root: str | Path,
    *,
    modalities: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate canonical files and return a compact machine-readable report."""

    root = Path(root)
    rows = read_manifest(root / "manifest.csv")
    if not rows:
        raise ValueError(f"no rows in {root / 'manifest.csv'}")
    seen = set()
    report: Dict[str, Any] = {
        "root": str(root), "samples": len(rows), "modalities": {},
    }
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in seen:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        seen.add(sample_id)
        row_modalities = list(modalities or [])
        if not row_modalities:
            row_modalities = [
                path.name for path in root.iterdir()
                if path.is_dir() and "_" not in path.name
            ]
        for modality in row_modalities:
            path = root / modality / f"{sample_id}.npy"
            if not path.exists():
                # Federated rows legitimately omit a modality.
                continue
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
                raise ValueError(f"{path} is not a non-empty (C,T) array")
            rate = row.get(f"{modality}_rate_hz") or row.get("rate_hz")
            if rate and float(rate) <= 0:
                raise ValueError(f"{sample_id}/{modality} has non-positive rate")
            ids_path = root / f"{modality}_ids" / f"{sample_id}.txt"
            if ids_path.exists():
                ids = [line for line in ids_path.read_text().splitlines() if line]
                if len(ids) != array.shape[0]:
                    raise ValueError(
                        f"{ids_path} has {len(ids)} ids for {array.shape[0]} channels")
            stats = report["modalities"].setdefault(
                modality, {"samples": 0, "channels": [], "timepoints": 0})
            stats["samples"] += 1
            stats["channels"].append(int(array.shape[0]))
            stats["timepoints"] += int(array.shape[1])
    report["sample_ids"] = sorted(seen)
    return report


def _cli(species: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Ingest and validate {species} cross-repository corpora")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="source manifest -> canonical ladder")
    ingest.add_argument("--source-manifest", required=True)
    ingest.add_argument("--out", required=True)
    ingest.add_argument("--overwrite", action="store_true")
    validate = sub.add_parser("validate", help="validate canonical ladder files")
    validate.add_argument("--root", required=True)
    validate.add_argument("--modalities", nargs="*")
    args = parser.parse_args()
    if args.command == "ingest":
        ingest_source_manifest(
            args.source_manifest, args.out, species=species,
            overwrite=args.overwrite)
    else:
        print(json.dumps(validate_ladder(
            args.root, modalities=args.modalities or None), indent=2))


__all__ = [
    "LoadedModality",
    "encode_optogenetic_control",
    "ingest_bids_eeg_session",
    "ingest_bids_fmri_session",
    "ingest_bids_meg_session",
    "ingest_bids_session",
    "ingest_source_manifest",
    "load_modality",
    "validate_ladder",
]
