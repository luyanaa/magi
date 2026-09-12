"""ZAPBench (larval zebrafish whole-brain activity) -> canonical ladder.

Source: Lueckmann et al., "ZAPBench: A Benchmark for Whole-Brain Activity
Prediction in Zebrafish" (arXiv:2503.02618, ICLR 2025). Data: CC-BY 4.0
(Google Research); companion repo Apache-2.0.

Facts this adapter encodes (all from the paper / release constants):

============================  ==================================================
acquisition                   4D light-sheet (LSFM), 2048x1328x72 voxels
                              x 7879 volumes at 406 nm x 406 nm x 4 um
                              x **914 ms** -> ``dt = 0.914 s`` (~1.09 vol/s)
reporter                      nuclear-localised GCaMP7f,
                              ``Tg(elavl3:H2B-GCaMP7f)`` -- chosen by the
                              authors *because* a ~1 Hz volume rate cannot
                              follow cytoplasmic GCaMP transients
activity traces               ``gs://zapbench-release/volumes/20240930/traces``
                              zarr3, ``(t=7879, f=71721)``, normalised to
                              ``[-0.25, 1.5]`` (labels ``['t', 'f']``)
control / covariates          ``.../stimuli_features`` zarr, ``(7879, 26)``
segmentation                  ``.../segmentation/dataframe.json`` (soma ids +
                              positions, used here for region pooling)
conditions                    9 stimulus protocols with offsets
                              ``(0,649,2422,3078,3735,5047,5638,6623,7279,7879)``
                              named gain/dots/flash/taxis/turning/position/
                              open loop/rotation/dark; ``taxis`` is the
                              benchmark holdout condition
============================  ==================================================

Ladder mapping: **one sample per condition** (``subject`` = animal, ``session`` =
condition), the 26-d stimulus-feature series as a control-role ``stimulus``
modality, the condition span (optionally tiled by ``--trial-seconds``) as
``<modality>_trials``, and per-sample rate/provenance in the manifest. Channel
reduction is required before this is trainable: 71,721 cells exceed every
channel cap in the ladder, so pass ``--region-bins`` (mean pooling over soma
position bins) or ``--max-cells`` (highest-variance cells).

Usage::

    python -m brain_moe_pinn.data.zapbench inspect \\
        --source /data/zapbench/traces
    python -m brain_moe_pinn.data.zapbench ingest \\
        --source /data/zapbench --out ../data_ladder/zebrafish_zapbench \\
        --rate-hz 1.0938 --conditions train --region-bins 8 6 3
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .readers import emit_sample

# Release constants (mirrored from zapbench/constants.py so the adapter works
# without the benchmark package installed; see ``_constants`` for the override).
CONDITION_OFFSETS: Tuple[int, ...] = (
    0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)
CONDITION_NAMES: Tuple[str, ...] = (
    "gain", "dots", "flash", "taxis", "turning", "position", "open loop",
    "rotation", "dark")
CONDITIONS_TRAIN: Tuple[int, ...] = (0, 1, 2, 4, 5, 6, 7, 8)
CONDITIONS_HOLDOUT: Tuple[int, ...] = (3,)
CONDITION_PADDING = 1
TRACES_URL = "gs://zapbench-release/volumes/20240930/traces"
STIMULI_URL = "gs://zapbench-release/volumes/20240930/stimuli_features"
SEGMENTATION_URL = (
    "gs://zapbench-release/volumes/20240930/segmentation/dataframe.json")
VOLUME_INTERVAL_S = 0.914
DEFAULT_RATE_HZ = 1.0 / VOLUME_INTERVAL_S
ZEBRAFISH_SUBJECT = "zapbench-20240930"


def _constants() -> Dict[str, object]:
    """Prefer the installed benchmark's constants when available."""
    try:  # pragma: no cover - optional dependency
        from zapbench import constants as zc  # type: ignore
        return {
            "offsets": tuple(zc.CONDITION_OFFSETS),
            "names": tuple(zc.CONDITION_NAMES),
            "train": tuple(zc.CONDITIONS_TRAIN),
            "holdout": tuple(zc.CONDITIONS_HOLDOUT),
            "padding": int(zc.CONDITION_PADDING),
        }
    except Exception:
        return {
            "offsets": CONDITION_OFFSETS,
            "names": CONDITION_NAMES,
            "train": CONDITIONS_TRAIN,
            "holdout": CONDITIONS_HOLDOUT,
            "padding": CONDITION_PADDING,
        }


@dataclass(frozen=True)
class ConditionSpan:
    index: int
    name: str
    start: int
    stop: int

    @property
    def n_steps(self) -> int:
        return self.stop - self.start


def condition_spans(padding: Optional[int] = None,
                    offsets: Optional[Sequence[int]] = None,
                    names: Optional[Sequence[str]] = None) -> List[ConditionSpan]:
    """Condition segment table, with the benchmark's padding applied."""
    const = _constants()
    offsets = tuple(offsets if offsets is not None else const["offsets"])
    names = tuple(names if names is not None else const["names"])
    pad = int(const["padding"] if padding is None else padding)
    if len(offsets) != len(names) + 1:
        raise ValueError(
            f"{len(offsets)} offsets describe {len(offsets) - 1} conditions, "
            f"but {len(names)} names were given")
    spans = []
    for index, name in enumerate(names):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        spans.append(ConditionSpan(index=index, name=str(name),
                                   start=lo + pad, stop=hi - pad))
    return spans


# --------------------------------------------------------------------- sources
class ArraySource:
    """Minimal ``(T, F)`` reader over zarr / npz / npy (local or gs://)."""

    def __init__(self, path, key: Optional[str] = None):
        self.path = str(path)
        self.key = key
        self._array = None
        self._kind = None

    def _open(self):
        if self._array is not None:
            return self._array
        p = self.path
        if p.endswith((".npz", ".npy")):
            loaded = np.load(p, mmap_mode="r")
            if hasattr(loaded, "files"):
                keys = list(loaded.files)
                key = self.key or ("traces" if "traces" in keys else keys[0])
                self._array = loaded[key]
            else:
                self._array = loaded
            self._kind = "numpy"
            return self._array
        try:
            import zarr  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                f"reading {p!r} needs zarr: pip install zarr "
                f"(add gcsfs for gs:// paths, or export to .npz with "
                f"`np.savez('traces.npz', traces=..., stimuli=...)`)") from exc
        try:
            store = zarr.open(p, mode="r")
        except Exception as exc:  # pragma: no cover - backend dependent
            raise RuntimeError(
                f"could not open {p!r} as zarr ({exc}); for gs:// paths install "
                f"gcsfs/tensorstore") from exc
        if isinstance(store, dict) or hasattr(store, "keys"):
            keys = list(store.keys()) if hasattr(store, "keys") else []
            key = self.key or ("traces" if "traces" in keys else
                               (keys[0] if keys else None))
            if key is None:
                raise ValueError(f"{p!r} is a zarr group without arrays")
            store = store[key]
        self._array = store
        self._kind = "zarr"
        return self._array

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._open().shape)

    def read(self, start: int = 0, stop: Optional[int] = None,
             columns: Optional[Sequence[int]] = None) -> np.ndarray:
        arr = self._open()
        stop = arr.shape[0] if stop is None else stop
        block = arr[start:stop]
        if columns is not None:
            block = block[:, list(columns)]
        return np.asarray(block, dtype=np.float32)


def read_traces(source, *, key: Optional[str] = None) -> ArraySource:
    return ArraySource(source, key=key)


def read_stimuli(source) -> Optional[ArraySource]:
    """Stimulus-feature series for ``source`` (``stimuli`` key or sibling dir).

    ``source`` may be the traces array, an ``.npz`` holding both, or a
    directory containing a ``stimuli_features`` sibling.
    """
    source = str(source)
    if source.endswith((".npz", ".npy")):
        if source.endswith(".npz"):
            with np.load(source, mmap_mode="r") as handle:
                if "stimuli" in handle.files:
                    return ArraySource(source, key="stimuli")
        return None
    path = Path(source)
    for candidate in (path.parent / "stimuli_features", path / "stimuli_features",
                      STIMULI_URL):
        try:
            src = ArraySource(str(candidate))
            if len(src.shape) == 2:
                return src
        except Exception:
            continue
    return None


def read_positions(path) -> Optional[Dict[str, np.ndarray]]:
    """Parse a segmentation dataframe defensively; ``None`` if unusable."""
    if not path:
        return None
    p = Path(str(path))
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(payload, dict):
        records = payload.get("data") or payload.get("records") or []
    elif isinstance(payload, list):
        records = payload
    else:
        return None
    if not records:
        return None
    keys = {str(k).lower(): k for k in records[0]}
    def pick(*names):
        for name in names:
            if name in keys:
                return keys[name]
        return None
    kx, ky, kz = pick("x", "cx", "pos_x"), pick("y", "cy", "pos_y"), pick("z", "cz", "pos_z")
    kid = pick("id", "soma_id", "cell_id", "label", "f")
    if kx is None or ky is None or kz is None:
        return None
    out = {
        "x": np.array([float(r[kx]) for r in records], dtype=np.float64),
        "y": np.array([float(r[ky]) for r in records], dtype=np.float64),
        "z": np.array([float(r[kz]) for r in records], dtype=np.float64),
    }
    if kid is not None:
        out["id"] = np.array([str(r[kid]) for r in records], dtype=object)
    return out


# ----------------------------------------------------------------------- modes
def select_channels(traces: np.ndarray, *, max_cells: Optional[int]) -> np.ndarray:
    """Indices of the ``max_cells`` highest-variance cells (deterministic)."""
    if max_cells is None or traces.shape[1] <= max_cells:
        return np.arange(traces.shape[1])
    variance = np.nanvar(traces, axis=0)
    order = np.argsort(-variance, kind="stable")
    return np.sort(order[: int(max_cells)])


def region_bins(positions: Dict[str, np.ndarray], bins: Sequence[int],
                n_cells: int) -> Tuple[np.ndarray, List[str]]:
    """Assign cells to ``(z, y, x)`` bins; returns labels and bin names."""
    bz, by, bx = (int(bins[0]), int(bins[1]), int(bins[2]))
    for name, count in (("z", bz), ("y", by), ("x", bx)):
        if count <= 0:
            raise ValueError(f"region bin count for {name} must be positive")
    def edges(values: np.ndarray, count: int) -> np.ndarray:
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi <= lo:
            hi = lo + 1e-6
        return np.linspace(lo, hi, count + 1)
    ex, ey, ez = (edges(positions["x"], bx), edges(positions["y"], by),
                  edges(positions["z"], bz))
    labels = [f"z{z}_y{y}_x{x}"
              for z in range(bz) for y in range(by) for x in range(bx)]
    index = {name: i for i, name in enumerate(labels)}
    ix = np.clip(np.digitize(positions["x"], ex) - 1, 0, bx - 1)
    iy = np.clip(np.digitize(positions["y"], ey) - 1, 0, by - 1)
    iz = np.clip(np.digitize(positions["z"], ez) - 1, 0, bz - 1)
    assignment = np.array(
        [index[f"z{z}_y{y}_x{x}"] for z, y, x in zip(iz, iy, ix)], dtype=int)
    if assignment.shape[0] < n_cells:
        assignment = np.concatenate(
            [assignment, np.zeros(n_cells - assignment.shape[0], dtype=int)])
    return assignment[:n_cells], labels


def pool_by_bin(block: np.ndarray, assignment: np.ndarray,
                n_bins: int) -> np.ndarray:
    """Mean-pool ``(T, F)`` traces into ``(n_bins, T)`` region traces."""
    out = np.zeros((n_bins, block.shape[0]), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.float64)
    for cell in range(block.shape[1]):
        out[assignment[cell]] += block[:, cell]
        counts[assignment[cell]] += 1.0
    counts[counts == 0] = 1.0
    return out / counts[:, None]


# --------------------------------------------------------------------- inspect
def inspect(source, *, stimuli_source=None, positions=None) -> None:
    traces = read_traces(source)
    print(f"[zapbench] traces {source!r}: shape {traces.shape}")
    control = (read_stimuli(stimuli_source or source))
    if control is not None:
        print(f"[zapbench] control features: shape {control.shape}")
    const = _constants()
    print(f"[zapbench] dt = {VOLUME_INTERVAL_S} s "
          f"(~{DEFAULT_RATE_HZ:.3f} volumes/s), nuclear GCaMP7f, LSFM")
    print("[zapbench] conditions:")
    for span in condition_spans():
        kind = ("holdout" if span.index in tuple(const["holdout"])
                else "train")
        print(f"   {span.index} {span.name:<10} [{span.start}, {span.stop}) "
              f"{span.n_steps} steps ({span.n_steps * VOLUME_INTERVAL_S:.0f} s) "
              f"{kind}")
    pos = read_positions(positions)
    print(f"[zapbench] positions: {'loaded' if pos else 'not available'}")


# ---------------------------------------------------------------------- ingest
def ingest(source, out_root: Path, *, rate_hz: Optional[float] = None,
           conditions: str = "train", stimuli_source: Optional[str] = None,
           positions: Optional[str] = None,
           region_bins_spec: Optional[Sequence[int]] = None,
           max_cells: Optional[int] = None, max_timesteps: Optional[int] = None,
           trial_seconds: Optional[float] = None,
           subject: str = ZEBRAFISH_SUBJECT,
           condition_offsets: Optional[Sequence[int]] = None,
           condition_names: Optional[Sequence[str]] = None,
           condition_padding: Optional[int] = None,
           overwrite: bool = False) -> Path:
    """Convert ZAPBench (traces, control features) into the canonical ladder.

    ``condition_offsets``/``condition_names``/``condition_padding`` default to
    the 2024-09-30 release layout; override them for other releases or for
    reduced fixtures (the mapping logic is identical).
    """
    if rate_hz is None:
        raise ValueError(
            "pass --rate-hz explicitly: ZAPBench's volume interval is 914 ms "
            f"({DEFAULT_RATE_HZ:.4f} Hz, arXiv:2503.02618), and the ladder must "
            "never assume a sampling rate silently")
    rate = float(rate_hz)
    dt_s = 1.0 / rate
    const = _constants()
    spans = condition_spans(padding=condition_padding,
                            offsets=condition_offsets,
                            names=condition_names)
    wanted = _select_conditions(spans, conditions, const)
    traces = read_traces(source)
    control = read_stimuli(stimuli_source or source)
    pos = read_positions(positions)
    if region_bins_spec is not None and pos is None:
        raise ValueError(
            "--region-bins needs soma positions: pass --positions "
            f"<segmentation dataframe> (release default: {SEGMENTATION_URL})")
    total_steps = traces.shape[0]
    for span in wanted:
        if span.stop > total_steps:
            raise ValueError(
                f"condition {span.name!r} spans [{span.start}, {span.stop}) but "
                f"{source!r} has only {total_steps} timesteps")

    out_root = Path(out_root)
    written = []
    for span in wanted:
        stop = span.stop
        if max_timesteps:
            stop = min(stop, span.start + int(max_timesteps))
        block = traces.read(span.start, stop)
        n_cells = block.shape[1]
        if region_bins_spec is not None:
            assignment, names = region_bins(pos, region_bins_spec, n_cells)
            signal = pool_by_bin(block, assignment, len(names))
            ids = names
        else:
            keep = select_channels(block, max_cells=max_cells)
            signal = np.ascontiguousarray(block[:, keep].T, dtype=np.float32)
            if pos is not None and "id" in pos and len(pos["id"]) >= n_cells:
                ids = [str(pos["id"][i]) for i in keep]
            else:
                ids = [f"cell{i}" for i in keep]
        signals = {"calcium": signal}
        if control is not None:
            control_block = control.read(span.start, stop)
            signals["stimulus"] = np.ascontiguousarray(
                control_block.T, dtype=np.float32)
        trials = {}
        frames = signal.shape[1]
        if trial_seconds and trial_seconds > 0:
            step = max(1, int(round(float(trial_seconds) * rate)))
            trials["calcium"] = [(lo, min(frames, lo + step))
                                 for lo in range(0, frames, step)]
        emit_sample(
            out_root, f"{span.index:02d}_{span.name.replace(' ', '_')}",
            signals,
            subject=subject,
            session=span.name,
            origin="zapbench:20240930",
            condition=span.name,
            rate_hz=rate,
            dt_s=dt_s,
            ids={"calcium": ids},
            trials=trials or None,
            overwrite=overwrite,
            extra={
                "animal": subject,
                "species": "zebrafish",
                "imaging": "lsfm_spim",
                "reporter": "h2b_gcamp7f",
                "readout": "dff",
                "source_steps": f"{span.start}:{stop}",
                "source_url": TRACES_URL,
                "benchmark_split": ("holdout" if span.index in
                                    tuple(const["holdout"]) else "train"),
            })
        written.append(span)
        print(f"[zapbench] {span.name:<10} -> {signal.shape[0]} channels x "
              f"{frames} steps ({frames * dt_s:.0f} s)")

    print(f"[zapbench] wrote {len(written)} sessions to {out_root} "
          f"(subject={subject}, rate={rate:g} Hz)")
    return out_root / "manifest.csv"


def _select_conditions(spans: Sequence[ConditionSpan], selection: str,
                       const: Dict[str, object]) -> List[ConditionSpan]:
    if selection == "all":
        return list(spans)
    if selection == "train":
        wanted = tuple(const["train"])
        return [s for s in spans if s.index in wanted]
    if selection == "holdout":
        wanted = tuple(const["holdout"])
        return [s for s in spans if s.index in wanted]
    names = [token.strip().lower() for token in selection.split(",") if token.strip()]
    by_name = {s.name.lower(): s for s in spans}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(
            f"unknown condition(s) {unknown}; known: {[s.name for s in spans]}")
    return [by_name[n] for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    insp = sub.add_parser("inspect", help="print shapes, dt and condition table")
    insp.add_argument("--source", required=True)
    insp.add_argument("--stimuli-source", default=None)
    insp.add_argument("--positions", default=None)
    ing = sub.add_parser("ingest", help="traces + control -> canonical ladder")
    ing.add_argument("--source", required=True,
                     help="zarr/npz traces path or gs:// URL")
    ing.add_argument("--out", required=True)
    ing.add_argument("--rate-hz", type=float, default=None,
                     help=f"volume rate in Hz (ZAPBench: {DEFAULT_RATE_HZ:.4f})")
    ing.add_argument("--conditions", default="train",
                     help="train | holdout | all | comma-separated names")
    ing.add_argument("--stimuli-source", default=None)
    ing.add_argument("--positions", default=None,
                     help="segmentation dataframe for region pooling")
    ing.add_argument("--region-bins", type=int, nargs=3, default=None,
                     metavar=("Z", "Y", "X"),
                     help="mean-pool cells into a ZxYxX region grid")
    ing.add_argument("--max-cells", type=int, default=None,
                     help="keep the N highest-variance cells")
    ing.add_argument("--max-timesteps", type=int, default=None)
    ing.add_argument("--trial-seconds", type=float, default=None,
                     help="tile the condition span into trials of this length")
    ing.add_argument("--subject", default=ZEBRAFISH_SUBJECT)
    ing.add_argument("--condition-offsets", type=int, nargs="+", default=None,
                     help="override the release's condition offsets")
    ing.add_argument("--condition-names", type=str, nargs="+", default=None)
    ing.add_argument("--condition-padding", type=int, default=None)
    ing.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        inspect(args.source, stimuli_source=args.stimuli_source,
                positions=args.positions)
    else:
        ingest(args.source, Path(args.out), rate_hz=args.rate_hz,
               conditions=args.conditions, stimuli_source=args.stimuli_source,
               positions=args.positions, region_bins_spec=args.region_bins,
               max_cells=args.max_cells, max_timesteps=args.max_timesteps,
               trial_seconds=args.trial_seconds, subject=args.subject,
               condition_offsets=args.condition_offsets,
               condition_names=args.condition_names,
               condition_padding=args.condition_padding,
               overwrite=args.overwrite)


if __name__ == "__main__":
    main()
