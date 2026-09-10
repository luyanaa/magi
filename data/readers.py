"""Source readers: heterogeneous corpus formats -> canonical ladder layout.

Canonical layout (see ``species_dataset.py``)::

    <root>/manifest.csv  +  <root>/<modality>/<sample>.npy (C, T)
    optional: <modality>_ids/<sample>.txt, <modality>_mask/<sample>.npy,
              <modality>_trials/<sample>.json, graphs/<sample>.npy

Each reader below is a thin adapter for one source family (NWB/DANDI, HDF5,
HF parquet, ALF-style npy). Optional dependencies (pynwb, h5py, pandas)
are imported lazily and produce actionable errors when missing.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .species_dataset import read_manifest


def _write_rows(manifest_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    existing = read_manifest(manifest_path)
    merged = {r.get("sample_id", ""): r for r in existing}
    for row in rows:
        merged[row.get("sample_id", "")] = row
    ordered = [merged[k] for k in sorted(merged) if k]
    with open(manifest_path, "w", newline="") as f:
        if not ordered:
            return
        writer = csv.DictWriter(f, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)


def emit_sample(
    out_root: Path,
    sample_id: str,
    signals: Dict[str, np.ndarray],
    *,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    origin: Optional[str] = None,
    condition: Optional[str] = None,
    rate_hz: Optional[float] = None,
    dt_s: Optional[float] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    ids: Optional[Dict[str, Sequence[str]]] = None,
    trials: Optional[Dict[str, Sequence[Tuple[int, int]]]] = None,
    graph: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Write one sample into the canonical layout and update the manifest."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for modality, array in signals.items():
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"{modality} for {sample_id} must be (C, T)")
        d = out_root / modality
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{sample_id}.npy"
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists (set overwrite=True)")
        np.save(target, np.ascontiguousarray(arr, dtype=np.float32))
    for modality, rows in (ids or {}).items():
        d = out_root / f"{modality}_ids"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sample_id}.txt").write_text(
            "\n".join(str(x) for x in rows) + "\n")
    for modality, arr in (masks or {}).items():
        d = out_root / f"{modality}_mask"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{sample_id}.npy",
                np.ascontiguousarray(np.asarray(arr, dtype=bool)))
    for modality, windows in (trials or {}).items():
        d = out_root / f"{modality}_trials"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sample_id}.json").write_text(json.dumps(
            [[int(a), int(b)] for a, b in windows]))
    if graph is not None:
        g = out_root / "graphs"
        g.mkdir(parents=True, exist_ok=True)
        np.save(g / f"{sample_id}.npy",
                np.ascontiguousarray(np.asarray(graph, dtype=np.float32)))
    row = {
        "sample_id": sample_id,
        "subject": subject or sample_id,
        "session": session or "1",
        "origin": origin or "",
        "condition": condition or "",
        "rate_hz": "" if rate_hz is None else str(rate_hz),
        "dt_s": "" if dt_s is None else str(dt_s),
    }
    _write_rows(out_root / "manifest.csv", [row])
    return row


def ingest_h5_traces(
    h5_path: Path,
    out_root: Path,
    sample_id: str,
    trace_keys: Dict[str, str],
    *,
    subject: Optional[str] = None,
    origin: Optional[str] = None,
    condition: Optional[str] = None,
    rate_hz: Optional[float] = None,
    name_key: Optional[str] = None,
) -> Dict[str, str]:
    """Import HDF5 traces (e.g. Portugues-lab deposits) to the ladder.

    ``trace_keys`` maps modality -> HDF5 dataset path of a (T, N) or
    (N, T) array; ``name_key`` optionally points to a dataset of neuron
    names. Time-major arrays are transposed.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError("ingest_h5_traces requires h5py: pip install h5py") from exc
    signals: Dict[str, np.ndarray] = {}
    names: Optional[List[str]] = None
    with h5py.File(h5_path, "r") as f:
        if name_key is not None:
            raw = f[name_key][()]
            names = [str(x) for x in np.asarray(raw).tolist()]
        for modality, key in trace_keys.items():
            arr = np.asarray(f[key][()])
            if arr.ndim != 2:
                raise ValueError(f"{key} is not 2-D")
            # time-major (T, N) -> (N, T) ladder convention when longer dim
            # is first and channel metadata (if any) matches the second.
            signals[modality] = arr.T if arr.shape[0] > arr.shape[1] else arr
    return emit_sample(
        out_root, sample_id, signals, subject=subject, origin=origin,
        condition=condition, rate_hz=rate_hz,
        ids={m: names for m in signals} if names else None)


def ingest_nwb_session(
    nwb_path: Path,
    out_root: Path,
    sample_id: str,
    series_name: str = "TwoPhotonSeries",
    *,
    subject: Optional[str] = None,
    origin: Optional[str] = None,
    condition: Optional[str] = None,
    rate_hz: Optional[float] = None,
    roi_key: Optional[str] = None,
) -> Dict[str, str]:
    """Import an NWB imaging session (DANDI-style) to the ladder.

    Pulls the imaging ``time_series`` as (C, T) after reshape/transpose of
    the raw (X, Y, T)-style volume via mean over pixels per plane, or raw
    ROI traces when ``roi_key`` points to a (T, N) ROI-response dataset.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "ingest_nwb_session requires pynwb: pip install pynwb") from exc
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        arrays: Dict[str, np.ndarray] = {}
        if roi_key is not None:
            arr = np.asarray(nwb.processing.get(roi_key, [])[()])
        else:
            ts = nwb.acquisition[series_name]
            arr = np.asarray(ts.data[()])
        if arr.ndim == 3:  # (T, X, Y) light-sheet-like: flatten pixels
            arr = arr.reshape(arr.shape[0], -1).T
        elif arr.ndim > 3:  # (T, X, Y, Z): mean over depth
            arr = arr.mean(axis=-1).reshape(arr.shape[0], -1).T
        arrays["calcium"] = arr
        return emit_sample(
            out_root, sample_id, arrays, subject=subject, origin=origin,
            condition=condition, rate_hz=rate_hz)


def summarize_ladder(root: Path) -> Dict[str, dict]:
    """Report per-modality sample counts and total timepoints."""
    root = Path(root)
    report: Dict[str, dict] = {}
    for modality_dir in sorted(root.iterdir()):
        if not modality_dir.is_dir() or "_" in modality_dir.name:
            continue
        files = list(modality_dir.glob("*.npy"))
        timepoints = 0
        for f in files:
            try:
                timepoints += int(np.load(f, mmap_mode="r").shape[1])
            except Exception:
                pass
        report[modality_dir.name] = {
            "samples": len(files),
            "total_timepoints": timepoints,
        }
    return report
