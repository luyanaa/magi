#!/usr/bin/env python3
"""Summarise saved free-run arrays into per-sample and per-channel distributions.

``tools/remote_free_run_eval.py --save-arrays`` writes ``NNNN_real.npy`` and
``NNNN_generated.npy`` pairs as ``(T, C)`` matrices restricted to the channels
that evaluation actually scored. This tool turns those pairs into the two
distribution views the C. elegans ladder needs:

* per sample: channel-averaged scale ratio, dispersion of that ratio across
  channels, and the lag-1 autocorrelation of both signals;
* per channel: the real and generated standard deviation, mean, and lag-1, so
  the report can say whether a bad aggregate comes from a few channels or from
  a systematic shift.

The JSON keeps quantiles and per-sample rows; the optional ``--npz`` keeps the
full per-channel matrices (padded to the widest recording).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _lag1(values: np.ndarray) -> np.ndarray:
    """Per-column lag-1 autocorrelation of a ``(T, C)`` matrix."""
    a, b = values[1:], values[:-1]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a * b).sum(axis=0) / denominator
    return np.where(np.isfinite(out), out, np.nan)


def _quantiles(values: np.ndarray) -> Dict[str, Optional[float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"q{int(q * 100):02d}": None for q in QUANTILES} | {
            "mean": None, "min": None, "max": None}
    return {f"q{int(q * 100):02d}": float(np.quantile(finite, q))
            for q in QUANTILES} | {
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max())}


def _pairs(arrays_dir: Path) -> List[tuple]:
    indices = sorted(
        int(path.name.split("_", 1)[0])
        for path in arrays_dir.glob("*_real.npy"))
    out = []
    for index in indices:
        real_path = arrays_dir / f"{index:04d}_real.npy"
        generated_path = arrays_dir / f"{index:04d}_generated.npy"
        if real_path.exists() and generated_path.exists():
            out.append((index, real_path, generated_path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays-dir", action="append", required=True,
                        type=Path,
                        help="directory of FFFF_real.npy / FFFF_generated.npy")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--npz", type=Path, default=None)
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {}
    for arrays_dir in args.arrays_dir:
        if not arrays_dir.is_dir():
            raise FileNotFoundError(f"no such arrays directory: {arrays_dir}")
        per_sample = []
        std_real_rows: List[np.ndarray] = []
        std_generated_rows: List[np.ndarray] = []
        for index, real_path, generated_path in _pairs(arrays_dir):
            real = np.load(real_path).astype("float64")
            generated = np.load(generated_path).astype("float64")
            if real.shape != generated.shape:
                raise ValueError(
                    f"{real_path.name} {real.shape} != {generated_path.name} "
                    f"{generated.shape}")
            std_real = real.std(axis=0)
            std_generated = generated.std(axis=0)
            # Channels with a near-constant real trace make the ratio
            # meaningless (measured pooled means of 1e8+), so the ratio is
            # computed on channels that carry at least 0.1% of the sample's
            # median real standard deviation.
            floor = max(1e-12, 1e-3 * float(np.median(std_real)))
            active = std_real > floor
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_all = std_generated / np.maximum(std_real, 1e-12)
            ratio = ratio_all[active]
            lag1_real = _lag1(real)
            lag1_generated = _lag1(generated)
            per_channel = {
                "std_real": std_real,
                "std_generated": std_generated,
                "mean_real": real.mean(axis=0),
                "mean_generated": generated.mean(axis=0),
                "lag1_real": lag1_real,
                "lag1_generated": lag1_generated,
            }
            std_real_rows.append(std_real[active])
            std_generated_rows.append(std_generated[active])
            positive = ratio[ratio > 0]
            per_sample.append({
                "index": index,
                "channels": int(real.shape[1]),
                "channels_used": int(active.sum()),
                "frames": int(real.shape[0]),
                "std_ratio_mean": float(np.nanmean(ratio)),
                "std_ratio_median": float(np.nanmedian(ratio)),
                "std_ratio_log_mean": (float(np.exp(np.log(positive).mean()))
                                       if positive.size else None),
                "std_ratio_channel_std": float(np.nanstd(ratio)),
                "std_ratio_q05": float(np.nanquantile(ratio, 0.05)),
                "std_ratio_q95": float(np.nanquantile(ratio, 0.95)),
                "lag1_real_mean": float(np.nanmean(lag1_real)),
                "lag1_generated_mean": float(np.nanmean(lag1_generated)),
            })
            if args.npz is not None:
                for name, values in per_channel.items():
                    key = f"{arrays_dir.name}__{index:04d}__{name}"
                    arrays[key] = values.astype("float32")
        if not per_sample:
            raise RuntimeError(f"{arrays_dir} contains no real/generated pairs")
        stacked_real = np.concatenate([row for row in std_real_rows]) \
            if std_real_rows else np.array([])
        stacked_generated = np.concatenate(
            [row for row in std_generated_rows]) if std_generated_rows else np.array([])
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled_ratio = stacked_generated / np.maximum(stacked_real, 1e-12)
        report[arrays_dir.name] = {
            "path": str(arrays_dir),
            "n_samples": len(per_sample),
            "channels_total": int(stacked_real.size),
            "channels_active_total": int(stacked_real.size),
            "channels_scored_total": int(sum(
                row["channels"] for row in per_sample)),
            "channels_per_sample": _quantiles(np.array(
                [row["channels"] for row in per_sample], dtype=float)),
            "channels_used_per_sample": _quantiles(np.array(
                [row["channels_used"] for row in per_sample], dtype=float)),
            "std_real_channel_quantiles": _quantiles(stacked_real),
            "std_generated_channel_quantiles": _quantiles(stacked_generated),
            "std_ratio_channel_quantiles": _quantiles(pooled_ratio),
            "std_ratio_sample_quantiles": _quantiles(np.array(
                [row["std_ratio_mean"] for row in per_sample])),
            "lag1_sample_quantiles_real": _quantiles(np.array(
                [row["lag1_real_mean"] for row in per_sample])),
            "lag1_sample_quantiles_generated": _quantiles(np.array(
                [row["lag1_generated_mean"] for row in per_sample])),
            "samples": per_sample,
        }
        if args.npz is not None:
            arrays[f"{arrays_dir.name}__std_real"] = stacked_real.astype("float32")
            arrays[f"{arrays_dir.name}__std_generated"] = \
                stacked_generated.astype("float32")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.npz is not None:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz, **arrays)
    print(json.dumps({
        name: {
            "n_samples": block["n_samples"],
            "channels_total": block["channels_total"],
            "std_ratio_mean": block["std_ratio_channel_quantiles"]["mean"],
            "std_ratio_q05": block["std_ratio_channel_quantiles"]["q05"],
            "std_ratio_q95": block["std_ratio_channel_quantiles"]["q95"],
        }
        for name, block in report.items()}, indent=2))


if __name__ == "__main__":
    main()
