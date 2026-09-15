#!/usr/bin/env python3
"""Compare generated C. elegans free runs in a fixed TDE-RICA basis.

The fixed-basis mode is specifically for the Toyoshima/``cleandata_smoothened2``
cohort. It consumes a reviewed MAT export containing ``coeffEmbed2`` and
``strNamesOrdered``; it never fits TDE-RICA on the evaluation window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from brain_moe_pinn.diagnostics.free_run_metrics import (
    tderica_biological_report,
)


def _load_array(path: Path, key: Optional[str] = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        values = np.load(path)
        if key is None:
            names = list(values.files)
            if len(names) != 1:
                raise ValueError(f"{path} contains multiple arrays; pass --key")
            key = names[0]
        return values[key]
    raise ValueError(f"unsupported array format {path.suffix}; use .npy/.npz")


def _orient(array: np.ndarray, orientation: str, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got {array.shape}")
    if orientation == "tn":
        return array
    if orientation == "nt":
        return array.T
    raise ValueError("orientation must be tn or nt")


def _name_text(value: object) -> str:
    """Convert MAT/NPZ string cells to stable canonical names."""
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"channel name cell must contain one value; got {array.shape}")
    value = array[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _load_fixed_basis(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load and validate a fixed basis without changing channel ordering."""
    if path.suffix.lower() == ".npz":
        values = np.load(path, allow_pickle=False)
        required = {"coeffEmbed2", "strNamesOrdered"}
        missing = required.difference(values.files)
        if missing:
            raise ValueError(
                f"{path} is missing fixed-basis arrays: {sorted(missing)}")
        coeff = np.asarray(values["coeffEmbed2"], dtype=float)
        names = [_name_text(value) for value in values["strNamesOrdered"]]
    elif path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("fixed MAT mode requires scipy") from exc
        values = loadmat(path, variable_names=["coeffEmbed2", "strNamesOrdered"])
        missing = {"coeffEmbed2", "strNamesOrdered"}.difference(values)
        if missing:
            raise ValueError(
                f"{path} is missing fixed-basis arrays: {sorted(missing)}")
        coeff = np.asarray(values["coeffEmbed2"], dtype=float)
        raw_names = np.asarray(values["strNamesOrdered"], dtype=object).reshape(-1)
        names = [_name_text(value) for value in raw_names]
    else:
        raise ValueError("basis must be .mat or .npz")
    if coeff.ndim != 3:
        raise ValueError(f"coeffEmbed2 must be (K,D,N); got {coeff.shape}")
    if min(coeff.shape) < 1 or not np.isfinite(coeff).all():
        raise ValueError("coeffEmbed2 must be non-empty and finite")
    if len(names) != coeff.shape[2]:
        raise ValueError(
            f"basis names ({len(names)}) do not match basis channels ({coeff.shape[2]})")
    if any(not name for name in names):
        raise ValueError("fixed-basis channel names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("fixed-basis channel names must be unique")
    return coeff, names


def _finite_mean(value: object) -> Optional[float]:
    """Return the finite mean of a scalar/vector metric, if available."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else None


def _nested_value(report: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _metric_summary(report: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """Extract comparable scalar summaries from one projected report."""
    paths = {
        "native_corr_matrix_mse": ("native", "corr_matrix_mse"),
        "native_autocorr_mse": ("native", "autocorr", "mse"),
        "native_variance_log_rmse": ("native", "variance_ratio", "log_rmse"),
        "tderica_w1": ("tderica", "distribution", "wasserstein_global"),
        "tderica_w1_component_mean": (
            "tderica", "distribution", "wasserstein_per_component"),
        "tderica_kl_mean": ("tderica", "distribution", "kl_divergence"),
        "tderica_cosine_diagonal_mean": (
            "tderica", "time_alignment", "cosine_diagonal_mean"),
        "tderica_cosine_global_mean": (
            "tderica", "time_alignment", "cosine_global_mean"),
        "tderica_kernel_transition": (
            "tderica", "dynamics", "kernel_transition"),
    }
    return {
        name: _finite_mean(_nested_value(report, path))
        for name, path in paths.items()
    }


def _compare_metric_reports(
    model_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare reports produced after projection through one common basis."""
    model = _metric_summary(model_report)
    baseline = _metric_summary(baseline_report)
    delta = {
        key: (model[key] - baseline[key])
        if model[key] is not None and baseline[key] is not None else None
        for key in model
    }
    return {
        "projection_contract": "same_fixed_basis",
        "model": model,
        "baseline": baseline,
        "model_minus_baseline": delta,
    }


def _basis_metadata(
    path: Path,
    coeff: np.ndarray,
    names: list[str],
    valid_channels: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    metadata = {
        "path": str(path),
        "channels": len(names),
        "components": int(coeff.shape[0]),
        "embed_width": int(coeff.shape[1]),
        "names_sha256": __import__("hashlib").sha256(
            "\n".join(names).encode()).hexdigest(),
        "fit_on_evaluation_window": False,
        "projection_channel_mask_policy": "shared_finite_delay_embedded_channels",
    }
    if valid_channels is not None:
        valid = np.asarray(valid_channels, dtype=bool)
        metadata["valid_channels"] = int(valid.sum())
        metadata["valid_channel_names"] = [
            name for name, keep in zip(names, valid) if keep
        ]
    return metadata
    
    


def _fixed_valid_channels(
    signal: np.ndarray, coeff: np.ndarray, toolbox_path: Path
) -> np.ndarray:
    """Return channels finite across one signal's full delay embedding."""
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import delayembed
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D; got {signal.shape}")
    if signal.shape[1] != coeff.shape[2]:
        raise ValueError(
            f"signal has {signal.shape[1]} channels; fixed basis requires "
            f"{coeff.shape[2]} Toyoshima channels")
    if signal.shape[0] < coeff.shape[1]:
        raise ValueError("signal is shorter than the fixed embedding width")
    embedded = delayembed(signal, coeff.shape[1])
    return np.isfinite(embedded).all(axis=(0, 1))


def _fixed_components(
    signal: np.ndarray,
    coeff: np.ndarray,
    toolbox_path: Path,
    valid_channels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project one signal through a fixed basis and shared valid-channel mask."""
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import delayembed, project
    if valid_channels is None:
        valid_channels = _fixed_valid_channels(signal, coeff, toolbox_path)
    valid_channels = np.asarray(valid_channels, dtype=bool)
    if valid_channels.shape != (coeff.shape[2],):
        raise ValueError(
            f"valid_channels must have shape ({coeff.shape[2]},); "
            f"got {valid_channels.shape}")
    embedded = delayembed(np.asarray(signal), coeff.shape[1])
    signal_valid = np.isfinite(embedded).all(axis=(0, 1))
    if np.any(valid_channels & ~signal_valid):
        raise ValueError(
            "shared fixed-basis channel mask includes nonfinite signal data")
    if not valid_channels.any():
        raise ValueError("no channels are finite across the fixed embedding")
    return project(
        embedded[:, :, valid_channels], coeff[:, :, valid_channels])




def select_pareto_checkpoint(
    reports: list[dict],
    *,
    metric_keys: tuple[str, ...] = (
        "raw_std_error", "w1", "kl_mean", "kernel_transition",
        "occurrence_lag1_gap", "native_corr_matrix_mse",
    ),
) -> dict:
    """Return a Pareto archive and deterministic compromise checkpoint."""
    from brain_moe_pinn.diagnostics.free_run_metrics import pareto_front
    candidates = []
    for report in reports:
        metrics = report.get("metrics", report)
        if all(key in metrics and np.isfinite(metrics[key]) for key in metric_keys):
            candidates.append({
                "metrics": {key: float(metrics[key]) for key in metric_keys},
                "checkpoint": report.get("checkpoint"),
            })
    if not candidates:
        raise ValueError("no checkpoint report has all finite Pareto metrics")
    front = pareto_front(candidates, {key: "min" for key in metric_keys})
    scales = {
        key: max(max(abs(item["metrics"][key]) for item in candidates), 1e-12)
        for key in metric_keys
    }
    def score(index):
        return sum(candidates[index]["metrics"][key] / scales[key]
                   for key in metric_keys)
    compromise = min(front, key=score)
    return {
        "pareto_indices": front,
        "pareto_checkpoints": [candidates[index]["checkpoint"] for index in front],
        "compromise_index": compromise,
        "compromise_checkpoint": candidates[compromise]["checkpoint"],
        "metric_keys": list(metric_keys),
        "normalized_score": float(score(compromise)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--persistence", type=Path,
                        help="optional persistence baseline in the same channel space")
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--dt-s", required=True, type=float)
    parser.add_argument("--orientation", choices=("tn", "nt"), default="tn")
    parser.add_argument("--tderica", type=Path, default=ROOT.parent / "TDE-RICA")
    parser.add_argument("--no-d3", action="store_true")
    parser.add_argument("--full-d3", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    real = _orient(_load_array(args.real), args.orientation, "real")
    generated = _orient(_load_array(args.generated), args.orientation, "generated")
    persistence = (
        None if args.persistence is None
        else _orient(_load_array(args.persistence), args.orientation, "persistence")
    )
    if real.shape != generated.shape:
        raise ValueError(f"real/generated shapes differ: {real.shape} vs {generated.shape}")
    if persistence is not None and persistence.shape != real.shape:
        raise ValueError(
            f"real/persistence shapes differ: {real.shape} vs {persistence.shape}")
    coeff, names = _load_fixed_basis(args.basis)
    valid_channels = _fixed_valid_channels(real, coeff, args.tderica)
    valid_channels &= _fixed_valid_channels(generated, coeff, args.tderica)
    if persistence is not None:
        valid_channels &= _fixed_valid_channels(
            persistence, coeff, args.tderica)
    if not valid_channels.any():
        raise ValueError(
            "real, generated, and persistence have no shared finite "
            "fixed-basis channels")
    comp_real = _fixed_components(
        real, coeff, args.tderica, valid_channels=valid_channels)
    comp_generated = _fixed_components(
        generated, coeff, args.tderica, valid_channels=valid_channels)
    report = tderica_biological_report(
        comp_real, comp_generated, dt_s=args.dt_s,
        include_d3=not args.no_d3, d3_fast=not args.full_d3,
        tderica_path=str(args.tderica))
    report["basis"] = _basis_metadata(
        args.basis, coeff, names, valid_channels=valid_channels)
    if persistence is not None:
        comp_persistence = _fixed_components(
            persistence, coeff, args.tderica, valid_channels=valid_channels)
        persistence_report = tderica_biological_report(
            comp_real, comp_persistence, dt_s=args.dt_s,
            include_d3=not args.no_d3, d3_fast=not args.full_d3,
            tderica_path=str(args.tderica))
        report["baseline_reports"] = {"persistence": persistence_report}
        report["baseline_comparison"] = _compare_metric_reports(
            report, persistence_report)
    text = json.dumps(report, indent=2, default=lambda x: x.tolist())
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
