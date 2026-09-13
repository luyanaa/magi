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
from typing import Optional

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


def _load_fixed_basis(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load a fixed basis from MAT/NPZ without silently changing ordering."""
    if path.suffix.lower() == ".npz":
        values = np.load(path, allow_pickle=False)
        coeff = np.asarray(values["coeffEmbed2"], dtype=float)
        names = [str(x) for x in values["strNamesOrdered"].tolist()]
    elif path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("fixed MAT mode requires scipy") from exc
        values = loadmat(path, variable_names=["coeffEmbed2", "strNamesOrdered"])
        coeff = np.asarray(values["coeffEmbed2"], dtype=float)
        raw_names = np.asarray(values["strNamesOrdered"], dtype=object).reshape(-1)
        names = []
        for value in raw_names:
            array = np.asarray(value).reshape(-1)
            names.append(str(array[0]))
    else:
        raise ValueError("basis must be .mat or .npz")
    if coeff.ndim != 3:
        raise ValueError(f"coeffEmbed2 must be (K,D,N); got {coeff.shape}")
    if len(names) != coeff.shape[2]:
        raise ValueError(
            f"basis names ({len(names)}) do not match basis channels ({coeff.shape[2]})")
    return coeff, names


def _fixed_components(signal: np.ndarray, coeff: np.ndarray, toolbox_path: Path) -> np.ndarray:
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import delayembed, project
    if signal.shape[1] != coeff.shape[2]:
        raise ValueError(
            f"signal has {signal.shape[1]} channels; fixed basis requires "
            f"{coeff.shape[2]} Toyoshima channels")
    if signal.shape[0] < coeff.shape[1]:
        raise ValueError("signal is shorter than the fixed embedding width")
    return project(delayembed(signal, coeff.shape[1]), coeff)


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
    if real.shape != generated.shape:
        raise ValueError(f"real/generated shapes differ: {real.shape} vs {generated.shape}")
    coeff, names = _load_fixed_basis(args.basis)
    comp_real = _fixed_components(real, coeff, args.tderica)
    comp_generated = _fixed_components(generated, coeff, args.tderica)
    report = tderica_biological_report(
        comp_real, comp_generated, dt_s=args.dt_s,
        include_d3=not args.no_d3, d3_fast=not args.full_d3,
        tderica_path=str(args.tderica))
    report["basis"] = {
        "path": str(args.basis), "channels": len(names),
        "components": int(coeff.shape[0]), "embed_width": int(coeff.shape[1]),
        "names_sha256": __import__("hashlib").sha256(
            "\n".join(names).encode()).hexdigest(),
        "fit_on_evaluation_window": False,
    }
    text = json.dumps(report, indent=2, default=lambda x: x.tolist())
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
