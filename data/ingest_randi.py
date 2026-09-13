"""Ingest Randi et al. (2023) PumpProbe exports into the species ladder.

The downloaded export is a set of recording-local text files::

    <id>_gcamp.txt          # time-major fluorescence, (T, N)
    <id>_t.txt              # timestamps, T values in seconds
    <id>_stim_neurons.txt   # recording-local target column indices
    <id>_stim_volume_i.txt  # zero-based pulse-start frame indices
    <id>_labels.txt         # optional, one label per source ROI
    <id>_ds_name.txt        # optional source recording path

The source target indices are treated as zero-based because the corpus contains
valid target 0 values and target N-1 values, while no target N values.  They are
*recording-local positions*, not a cross-recording neuron identity map.  The
importer therefore emits a fixed target-position vocabulary and preserves the
original labels separately instead of pretending that partially populated or
duplicated labels are canonical IDs.

The pulse waveform follows the supplied TSMixer-Ext notebook: three consecutive
levels (3.0, 1.5, 0.75) for ten source frames each.  At the observed 0.5-second
frame interval this is a 15-second pulse.  Fluorescence values outside the
explicit [0, 200] range, including NaN/Inf, are masked frame-by-frame rather
than deleting an entire channel; this retains target positions while preventing
extreme values from entering the model.  Valid values are per-channel min-max
scaled to [-3, 3] by default, matching the notebook's model input scale.

Usage::

    python -m brain_moe_pinn.data.ingest_randi \\
        --input ../exported_data \\
        --output ../data_ladder/c_elegans_randi

The output contains ``calcium`` and target-gated ``opto`` arrays, per-frame
calcium masks, recording-local calcium IDs, event JSON, source labels, a fixed
target vocabulary, and a provenance-rich ``manifest.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .readers import emit_sample

DEFAULT_PULSE_LEVELS = (3.0, 1.5, 0.75)
DEFAULT_PULSE_WIDTHS = (10, 10, 10)
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 200.0
DEFAULT_OUTPUT_MIN = -3.0
DEFAULT_OUTPUT_MAX = 3.0


def _numeric_sort_key(path: Path) -> Tuple[int, int | str]:
    """Sort numeric recording IDs numerically, then other names lexically."""
    stem = path.name.removesuffix("_gcamp.txt")
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _recording_refs(input_dir: Path) -> List[Tuple[str, Path]]:
    refs = [
        (path.name.removesuffix("_gcamp.txt"), path)
        for path in input_dir.glob("*_gcamp.txt")
    ]
    refs.sort(key=lambda item: _numeric_sort_key(item[1]))
    if not refs:
        raise FileNotFoundError(f"no *_gcamp.txt files under {input_dir}")
    return refs


def _read_matrix(path: Path) -> np.ndarray:
    array = np.loadtxt(path, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D matrix")
    return np.asarray(array, dtype=np.float64)


def _read_integer_vector(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64, ndmin=1)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{path} must contain finite integer values")
    return values.astype(np.int64)


def _read_time_vector(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64, ndmin=1)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{path} must contain at least two finite timestamps")
    diffs = np.diff(values)
    if np.any(diffs <= 0):
        raise ValueError(f"{path} timestamps must be strictly increasing")
    dt = float(np.median(diffs))
    if not np.allclose(diffs, dt, rtol=1e-3, atol=1e-6):
        raise ValueError(
            f"{path} timestamps are not uniformly sampled; "
            f"median dt={dt:.9g}s")
    return values


def _read_labels(path: Path) -> List[str]:
    if not path.exists():
        return []
    # Keep blank physical lines: source label position is part of the file
    # contract even when a label is unavailable.
    return [line.strip() for line in path.read_text().splitlines()]


def _pulse_template(
    levels: Sequence[float], widths: Sequence[int],
) -> np.ndarray:
    if len(levels) == 0 or len(levels) != len(widths):
        raise ValueError("pulse levels and widths must be non-empty and equal length")
    if any(float(level) < 0 for level in levels):
        raise ValueError("pulse levels must be non-negative")
    if any(int(width) <= 0 for width in widths):
        raise ValueError("pulse widths must be positive frame counts")
    return np.repeat(
        np.asarray(levels, dtype=np.float32),
        np.asarray(widths, dtype=np.int64),
    )


def _clean_calcium(
    raw: np.ndarray,
    *,
    min_value: Optional[float],
    max_value: Optional[float],
    scale: str,
    output_min: float = DEFAULT_OUTPUT_MIN,
    output_max: float = DEFAULT_OUTPUT_MAX,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Mask invalid fluorescence and optionally scale each channel."""
    if scale not in {"none", "minmax"}:
        raise ValueError("scale must be 'none' or 'minmax'")
    if min_value is not None and max_value is not None and min_value >= max_value:
        raise ValueError("min_value must be less than max_value")

    finite = np.isfinite(raw)
    valid = finite.copy()
    below = np.zeros(raw.shape, dtype=bool)
    above = np.zeros(raw.shape, dtype=bool)
    if min_value is not None:
        below = finite & (raw < float(min_value))
        valid &= raw >= float(min_value)
    if max_value is not None:
        above = finite & (raw > float(max_value))
        valid &= raw <= float(max_value)

    cleaned = np.zeros(raw.shape, dtype=np.float64)
    if scale == "none":
        cleaned[valid] = raw[valid]
    else:
        # Fit each channel only on valid source values.  Invalid frames remain
        # zero and are excluded by the emitted calcium mask.
        for channel in range(raw.shape[1]):
            channel_valid = valid[:, channel]
            if not channel_valid.any():
                continue
            values = raw[channel_valid, channel]
            lo = float(values.min())
            hi = float(values.max())
            if hi <= lo:
                continue
            cleaned[channel_valid, channel] = (
                float(output_min)
                + (values - lo) * (float(output_max) - float(output_min))
                / (hi - lo)
            )

    stats = {
        "raw_nonfinite_frames": int((~finite).sum()),
        "raw_below_min_frames": int(below.sum()),
        "raw_above_max_frames": int(above.sum()),
        "raw_invalid_frames": int((~valid).sum()),
        "raw_all_invalid_channels": int((~valid.any(axis=0)).sum()),
    }
    return (
        np.ascontiguousarray(cleaned.T, dtype=np.float32),
        np.ascontiguousarray(valid.T, dtype=bool),
        stats,
    )


def _scan_target_vocab(
    refs: Iterable[Tuple[str, Path]],
    *,
    explicit_size: Optional[int],
) -> Tuple[int, int, int]:
    max_target = -1
    negative_events = 0
    total_events = 0
    for sample_id, gcamp_path in refs:
        target_path = gcamp_path.with_name(f"{sample_id}_stim_neurons.txt")
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        targets = _read_integer_vector(target_path)
        total_events += int(targets.size)
        negative_events += int((targets < 0).sum())
        nonnegative = targets[targets >= 0]
        if nonnegative.size:
            max_target = max(max_target, int(nonnegative.max()))
    target_vocab_size = (
        int(explicit_size) if explicit_size is not None else max_target + 1
    )
    if target_vocab_size <= 0:
        raise ValueError(
            "could not infer a target vocabulary; pass --target-vocab-size")
    if max_target >= target_vocab_size:
        raise ValueError(
            f"target index {max_target} exceeds target vocabulary size "
            f"{target_vocab_size}")
    return target_vocab_size, total_events, negative_events


def _build_opto(
    n_frames: int,
    targets: np.ndarray,
    starts: np.ndarray,
    labels: Sequence[str],
    *,
    target_vocab_size: int,
    source_channel_count: int,
    pulse: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Dict[str, object]], int]:
    """Build waveform + recording-local target-gated control features."""
    if targets.size != starts.size:
        raise ValueError(
            "stim_neurons and stim_volume_i must have the same event count")
    control = np.zeros((1 + target_vocab_size, n_frames), dtype=np.float32)
    windows: List[Tuple[int, int]] = []
    events: List[Dict[str, object]] = []
    invalid_count = 0
    for event_index, (raw_target, raw_start) in enumerate(zip(targets, starts)):
        target = int(raw_target)
        start = int(raw_start)
        event: Dict[str, object] = {
            "event_index": int(event_index),
            "target_index": target,
            "start_frame": start,
            "target_label": (
                labels[target].strip() if 0 <= target < len(labels) else ""
            ),
        }
        reason: Optional[str] = None
        if target < 0:
            reason = "negative_target_sentinel"
        elif target >= source_channel_count:
            reason = "target_out_of_recording"
        elif target >= target_vocab_size:
            reason = "target_out_of_vocab"
        elif start < 0 or start >= n_frames:
            reason = "start_frame_out_of_range"
        if reason is not None:
            invalid_count += 1
            event.update({"valid": False, "reason": reason})
            events.append(event)
            continue

        end = min(n_frames, start + int(pulse.size))
        if end <= start:
            invalid_count += 1
            event.update({"valid": False, "reason": "empty_clipped_pulse"})
            events.append(event)
            continue
        segment = pulse[: end - start]
        # max is deterministic for the source's non-overlapping events and
        # preserves both drives if a future export contains overlapping events.
        control[0, start:end] = np.maximum(control[0, start:end], segment)
        control[1 + target, start:end] = np.maximum(
            control[1 + target, start:end], segment
        )
        windows.append((start, end))
        event.update({
            "valid": True,
            "end_frame": int(end),
            "duration_frames": int(end - start),
        })
        events.append(event)
    return control, windows, events, invalid_count


def _write_companion(
    path: Path,
    content: str,
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (set overwrite=True)")
    path.write_text(content)


def _write_target_vocab(
    output_dir: Path,
    target_vocab_size: int,
    *,
    overwrite: bool,
) -> List[str]:
    vocab = [f"cell_{index:03d}" for index in range(target_vocab_size)]
    path = output_dir / "opto_target_vocab.txt"
    text = "\n".join(vocab) + "\n"
    if path.exists() and not overwrite and path.read_text() != text:
        raise ValueError(
            f"existing target vocabulary at {path} disagrees with source; "
            "use a new output directory")
    if overwrite or not path.exists():
        _write_companion(path, text, overwrite=overwrite)
    return vocab


def convert_randi_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    limit: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
    pulse_levels: Sequence[float] = DEFAULT_PULSE_LEVELS,
    pulse_widths: Sequence[int] = DEFAULT_PULSE_WIDTHS,
    min_value: Optional[float] = DEFAULT_MIN_VALUE,
    max_value: Optional[float] = DEFAULT_MAX_VALUE,
    scale: str = "minmax",
    overwrite: bool = False,
) -> Path:
    """Convert Randi text recordings into canonical ``calcium`` + ``opto``.

    ``target_vocab_size`` is inferred from all source recordings before an
    optional ``limit`` is applied, so smoke conversions retain the same control
    width as a full-corpus conversion.  Every source timestamp file is checked
    for a uniform clock; its measured rate is written to the row and the
    importer refuses a mismatched or non-monotonic clock.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    refs = _recording_refs(input_dir)
    target_vocab_size, total_events, negative_events = _scan_target_vocab(
        refs, explicit_size=target_vocab_size)
    vocab = _write_target_vocab(
        output_dir, target_vocab_size, overwrite=overwrite)
    selected = refs if limit is None else refs[: max(1, int(limit))]
    pulse = _pulse_template(pulse_levels, pulse_widths)
    output_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "samples": 0,
        "calcium_channels": 0,
        "valid_events": 0,
        "invalid_events": 0,
        "masked_frames": 0,
    }
    for sample_id, gcamp_path in selected:
        time_path = gcamp_path.with_name(f"{sample_id}_t.txt")
        target_path = gcamp_path.with_name(f"{sample_id}_stim_neurons.txt")
        start_path = gcamp_path.with_name(f"{sample_id}_stim_volume_i.txt")
        labels_path = gcamp_path.with_name(f"{sample_id}_labels.txt")
        source_path = gcamp_path.with_name(f"{sample_id}_ds_name.txt")
        for required in (time_path, target_path, start_path):
            if not required.exists():
                raise FileNotFoundError(required)

        raw = _read_matrix(gcamp_path)
        timestamps = _read_time_vector(time_path)
        if timestamps.size != raw.shape[0]:
            raise ValueError(
                f"{sample_id}: gcamp has {raw.shape[0]} frames but t has "
                f"{timestamps.size} timestamps")
        targets = _read_integer_vector(target_path)
        starts = _read_integer_vector(start_path)
        if targets.size != starts.size:
            raise ValueError(
                f"{sample_id}: stim_neurons has {targets.size} values but "
                f"stim_volume_i has {starts.size}")

        labels = _read_labels(labels_path)
        calcium, calcium_mask, clean_stats = _clean_calcium(
            raw,
            min_value=min_value,
            max_value=max_value,
            scale=scale,
        )
        opto, windows, events, invalid_events = _build_opto(
            raw.shape[0], targets, starts, labels,
            target_vocab_size=target_vocab_size,
            source_channel_count=raw.shape[1],
            pulse=pulse,
        )
        dt_s = float(np.median(np.diff(timestamps)))
        rate_hz = 1.0 / dt_s
        labels_nonempty = [label for label in labels if label]
        duplicate_labels = len(labels_nonempty) - len(set(labels_nonempty))
        source_name = source_path.read_text().strip() if source_path.exists() else ""
        extra = {
            "dataset": "Randi et al. (2023) PumpProbe",
            "source_recording_id": sample_id,
            "source_path": source_name,
            "target_index_base": "0",
            "frame_index_base": "0",
            "target_scope": "recording_local_cell",
            "target_vocab_size": str(target_vocab_size),
            "opto_feature_count": str(1 + target_vocab_size),
            "pulse_levels": "|".join(f"{float(x):g}" for x in pulse_levels),
            "pulse_widths_frames": "|".join(str(int(x)) for x in pulse_widths),
            "pulse_duration_s": f"{pulse.size * dt_s:.9g}",
            "stim_event_count": str(int(targets.size)),
            "stim_valid_event_count": str(int(len(windows))),
            "stim_invalid_event_count": str(int(invalid_events)),
            "stim_invalid_target_values": "|".join(
                str(int(x)) for x in sorted(set(targets[targets < 0].tolist()))
            ),
            "label_line_count": str(len(labels)),
            "label_channel_count": str(raw.shape[1]),
            "label_nonempty_count": str(len(labels_nonempty)),
            "label_duplicate_count": str(max(0, duplicate_labels)),
            "label_length_delta": str(len(labels) - raw.shape[1]),
            "preprocess_scale": scale,
            "preprocess_range": (
                f"{min_value if min_value is not None else '-inf'}:"
                f"{max_value if max_value is not None else 'inf'}"
            ),
            "timestamp_start_s": f"{timestamps[0]:.9g}",
            "timestamp_end_s": f"{timestamps[-1]:.9g}",
            "frame_count": str(int(raw.shape[0])),
            "channel_count": str(int(raw.shape[1])),
            **{key: str(value) for key, value in clean_stats.items()},
        }
        calcium_ids = [
            f"randi_{sample_id}__cell_{index:03d}"
            for index in range(raw.shape[1])
        ]
        emit_sample(
            output_dir,
            sample_id,
            {"calcium": calcium, "opto": opto},
            subject=f"randi_recording_{sample_id}",
            session="1",
            origin="Randi et al. (2023) PumpProbe",
            condition="optogenetic_pumpprobe",
            rate_hz=rate_hz,
            dt_s=dt_s,
            masks={"calcium": calcium_mask} if not calcium_mask.all() else None,
            ids={"calcium": calcium_ids},
            trials={"opto": windows} if windows else None,
            overwrite=overwrite,
            extra=extra,
        )
        _write_companion(
            output_dir / "randi_labels" / f"{sample_id}.txt",
            labels_path.read_text() if labels_path.exists() else "",
            overwrite=overwrite,
        )
        _write_companion(
            output_dir / "opto_events" / f"{sample_id}.json",
            json.dumps({
                "sample_id": sample_id,
                "target_index_base": 0,
                "frame_index_base": 0,
                "target_vocab": vocab,
                "events": events,
            }, indent=2) + "\n",
            overwrite=overwrite,
        )
        totals["samples"] += 1
        totals["calcium_channels"] += int(raw.shape[1])
        totals["valid_events"] += len(windows)
        totals["invalid_events"] += int(invalid_events)
        totals["masked_frames"] += int((~calcium_mask).sum())

    print(
        f"[randi] converted {totals['samples']} recordings -> {output_dir} "
        f"(target_vocab={target_vocab_size}, features={1 + target_vocab_size})"
    )
    print(
        f"[randi] source events={total_events}; negative sentinels="
        f"{negative_events}; selected valid={totals['valid_events']}; "
        f"invalid={totals['invalid_events']}"
    )
    print(
        f"[randi] calcium channels={totals['calcium_channels']}; "
        f"masked frames={totals['masked_frames']}"
    )
    return output_dir / "manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="directory containing <id>_gcamp.txt exports")
    parser.add_argument("--output", required=True,
                        help="canonical ladder root")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert at most N recordings after full-vocab scan")
    parser.add_argument("--target-vocab-size", type=int, default=None,
                        help="fixed width; otherwise infer max non-negative target + 1")
    parser.add_argument("--pulse-levels", type=float, nargs="+",
                        default=list(DEFAULT_PULSE_LEVELS),
                        help="piecewise pulse levels, matching the notebook")
    parser.add_argument("--pulse-widths", type=int, nargs="+",
                        default=list(DEFAULT_PULSE_WIDTHS),
                        help="pulse segment widths in source frames")
    parser.add_argument("--min-value", type=float, default=DEFAULT_MIN_VALUE,
                        help="inclusive valid fluorescence lower bound")
    parser.add_argument("--max-value", type=float, default=DEFAULT_MAX_VALUE,
                        help="inclusive valid fluorescence upper bound")
    parser.add_argument("--scale", choices=("none", "minmax"), default="minmax",
                        help="valid calcium output scaling")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing sample arrays and companions")
    args = parser.parse_args()
    convert_randi_directory(
        args.input,
        args.output,
        limit=args.limit,
        target_vocab_size=args.target_vocab_size,
        pulse_levels=args.pulse_levels,
        pulse_widths=args.pulse_widths,
        min_value=args.min_value,
        max_value=args.max_value,
        scale=args.scale,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
