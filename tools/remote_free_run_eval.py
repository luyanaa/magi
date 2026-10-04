#!/usr/bin/env python3
"""Evaluate deterministic ODE or sampled SDE free-run forecasts.

The evaluator uses the production C. elegans loader and compares held-out
future windows with autonomous rollouts. ODE reports preserve the existing
single-path metrics. SDE reports keep the recursive transition-mean path
separate from N sampled realizations; primary TDE-RICA W1/KL compare real
features with the pooled projections of all sampled trajectories, never the
transition-mean path. TDE-RICA bases are fit on each context window or loaded
from a recording-local fixed basis, avoiding cross-source neuron identity
assumptions.

Each sample is labelled with its ladder source and carries per-horizon
trajectory metrics, so one run yields the source x horizon breakdown without
re-running the loader per source.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from brain_moe_pinn import BrainMoEPINNConfig
from brain_moe_pinn.diagnostics.free_run_metrics import (
    run_free_run_suite, sde_transition_diagnostics, tderica_biological_report,
)
from brain_moe_pinn.training.training_loop import partition_generic_batch


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device,
                     config_path: Optional[Path] = None) -> Dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    model_state = state.get("model_state", state.get("state_dict"))
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint has no model_state/state_dict: {path}")
    info: Dict[str, Any] = {
        "checkpoint_step": state.get("step"),
        "checkpoint_phase": state.get("phase"),
        "world_size": state.get("world_size"),
        "config_sha256": state.get("config_sha256"),
        "config_path": state.get("config_path"),
    }
    if config_path is not None:
        # A checkpoint and a species profile are chosen independently; the
        # stored fingerprint makes the pairing checkable instead of assumed.
        # Report it before load_state_dict so a shape error is not the first
        # (and least informative) symptom of the wrong profile.
        import hashlib
        candidate = Path(config_path).resolve()
        info["evaluated_config_path"] = str(candidate)
        info["evaluated_config_sha256"] = hashlib.sha256(
            candidate.read_bytes()).hexdigest()
        stored = info.get("config_sha256")
        info["config_match"] = (
            None if not stored
            else bool(stored == info["evaluated_config_sha256"]))
        if info["config_match"] is False:
            print("[config] WARNING: checkpoint was produced by a different "
                  f"profile ({info.get('config_path')}); evaluating "
                  f"{candidate} anyway", flush=True)
    incompatible = model.load_state_dict(model_state, strict=False)
    info["missing_keys"] = list(incompatible.missing_keys)
    info["unexpected_keys"] = list(incompatible.unexpected_keys)
    return info


def _as_per_sample_dt(value: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        dt = value.detach().flatten().to(device=device, dtype=torch.float32)
    elif isinstance(value, (list, tuple)):
        dt = torch.tensor([float(v) for v in value], device=device)
    else:
        dt = torch.tensor([float(value)], device=device)
    if dt.numel() == 1:
        dt = dt.expand(batch_size)
    if dt.numel() != batch_size or not torch.isfinite(dt).all() or bool((dt <= 0).any()):
        raise ValueError(f"batch dt must contain one positive finite value per sample; got {dt}")
    return dt


def _concat_horizons(value: torch.Tensor) -> np.ndarray:
    """Convert ``(K,C,T)`` to ``(K*T,C)`` in horizon order."""
    return value.detach().float().cpu().numpy().transpose(0, 2, 1).reshape(
        value.shape[0] * value.shape[2], value.shape[1])


def _concat_masks(value: torch.Tensor) -> np.ndarray:
    """Convert ``(K,C,T)`` masks to ``(K*T,C)`` in horizon order."""
    return value.detach().bool().cpu().numpy().transpose(0, 2, 1).reshape(
        value.shape[0] * value.shape[2], value.shape[1])


def _aggregate_numeric_mean(reports: Iterable[Dict[str, Any]], path: Tuple[str, ...]):
    """Aggregate scalar or vector-valued numeric fields per report."""
    values = []
    for report in reports:
        current: Any = report
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is None:
            continue
        try:
            array = np.asarray(current, dtype=float)
        except (TypeError, ValueError):
            continue
        finite = array[np.isfinite(array)]
        if finite.size:
            values.append(float(np.mean(finite)))
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
    }


def _mean_numeric_reports(reports: List[Any]) -> Any:
    """Average matching numeric leaves without collapsing trajectories first."""
    if not reports:
        return {}
    if all(isinstance(report, dict) for report in reports):
        shared_keys = set.intersection(*(set(report) for report in reports))
        return {
            key: value
            for key in sorted(shared_keys)
            if (value := _mean_numeric_reports(
                [report[key] for report in reports])) is not None
        }
    try:
        arrays = [np.asarray(value, dtype=float) for value in reports]
    except (TypeError, ValueError):
        return None
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        return None
    stacked = np.stack(arrays)
    finite = np.isfinite(stacked)
    if not finite.any():
        return None
    mean = np.divide(
        np.where(finite, stacked, 0.0).sum(axis=0),
        finite.sum(axis=0),
        out=np.full(stacked.shape[1:], np.nan),
        where=finite.sum(axis=0) > 0,
    )
    if not np.isfinite(mean).all():
        return None
    return float(mean) if mean.ndim == 0 else mean.tolist()




_SOURCE_PREFIXES = {
    "toyoshima_salt": "Toyoshima",
    "randi_pumpprobe": "Randi",
    "hf_activity": "Simeon",
}


def _source_label(sample_id: str, origin: str) -> str:
    """Recover the ladder source of a merged row.

    ``data/merge_c_elegans.py`` writes ``<source_prefix>__<source_id>`` as the
    unified sample id, so the prefix is the authoritative source label; the
    origin string is only a fallback for rows assembled by other tooling.
    """
    prefix = str(sample_id).split("__", 1)[0]
    if prefix in _SOURCE_PREFIXES:
        return prefix
    text = f"{sample_id} {origin}"
    for label, marker in _SOURCE_PREFIXES.items():
        if marker.lower() in text.lower():
            return label
    return "unknown"



def _sampled_horizon_reports(
    real_all: np.ndarray,
    generated_samples_all: np.ndarray,
    transition_mean_all: np.ndarray,
    rollouts: int,
) -> List[Dict[str, Any]]:
    """Report horizon-local path statistics and native metrics."""
    frames_total = real_all.shape[0]
    if rollouts < 1 or frames_total % rollouts:
        raise ValueError(
            f"{frames_total} frames do not divide into {rollouts} horizons")
    if generated_samples_all.ndim != 3:
        raise ValueError("sample paths must have shape (S,T,C)")
    if generated_samples_all.shape[1:] != real_all.shape:
        raise ValueError("sample paths must match the real trajectory shape")
    if transition_mean_all.shape != real_all.shape:
        raise ValueError("transition mean must match the real trajectory shape")
    horizon_frames = frames_total // rollouts
    entries = []
    for horizon in range(rollouts):
        start = horizon * horizon_frames
        stop = start + horizon_frames
        real = real_all[start:stop]
        mean_path = transition_mean_all[start:stop]
        samples = generated_samples_all[:, start:stop]
        if real.shape[0] < 3:
            continue
        native = [
            run_free_run_suite(real, generated)
            for generated in samples
        ]
        real_std = np.std(real, axis=0)
        sample_ratios = (
            np.std(samples, axis=1)
            / np.maximum(real_std[None, :], 1e-12))
        mean_std = np.std(mean_path, axis=0)
        mean_path_std_ratio = float(
            mean_std.mean() / max(real_std.mean(), 1e-12))
        entries.append({
            "horizon": horizon + 1,
            "frames": int(real.shape[0]),
            "n_realizations": int(samples.shape[0]),
            "r_sigma": float(np.mean(sample_ratios)),
            "raw_std_ratio": float(np.mean(sample_ratios)),
            "mean_path_raw_std_ratio": mean_path_std_ratio,
            "raw_std_ratio_median": float(np.median(sample_ratios)),
            **_path_ensemble_diagnostics(real, mean_path, samples),
            "native": _mean_numeric_reports(native),
            "mean_path_native": run_free_run_suite(real, mean_path),
            "native_per_realization": native,
        })
    return entries


def _group_aggregate(reports: Iterable[Dict[str, Any]],
                     paths: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    return {name: _aggregate_numeric_mean(reports, path)
            for name, path in paths.items()}


def _horizon_aggregate(reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate path, distribution, and dynamics metrics per horizon."""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for report in reports:
        for entry in report.get("horizons", ()) or ():
            buckets.setdefault(int(entry["horizon"]), []).append(entry)
    paths = {
        "r_sigma": ("r_sigma",),
        "raw_std_ratio_median": ("raw_std_ratio_median",),
        "between_sample_variance": ("between_sample_variance",),
        "within_path_temporal_variance": ("within_path_temporal_variance",),
        "real_temporal_variance": ("real_temporal_variance",),
        "mean_path_raw_std_ratio": ("mean_path_raw_std_ratio",),
        "realization_lag1_mean": ("realization_lag1", "mean"),
        "mean_path_lag1": ("realization_lag1", "mean_path"),
        "mean_path_w1": (
            "tderica", "mean_path_distribution", "wasserstein_global"),
        "mean_path_kl": (
            "tderica", "mean_path_distribution", "kl_divergence"),
        "pooled_sample_w1": (
            "tderica", "pooled_sample_distribution", "wasserstein_global"),
        "pooled_sample_kl": (
            "tderica", "pooled_sample_distribution", "kl_divergence"),
        "mean_path_kernel_transition": (
            "tderica", "mean_path_kernel_transition"),
        "sample_kernel_transition_mean": (
            "tderica", "sample_kernel_transition_mean"),
        "native_corr_matrix_mse": ("native", "corr_matrix_mse"),
        "mean_path_native_corr_matrix_mse": (
            "mean_path_native", "corr_matrix_mse"),
        "mean_path_native_autocorr_mse": (
            "mean_path_native", "autocorr", "mse"),
        "native_autocorr_mse": ("native", "autocorr", "mse"),
        "native_variance_log_rmse": ("native", "variance_ratio", "log_rmse"),
    }
    return {
        str(horizon): {"n": len(entries),
                       "metrics": _group_aggregate(entries, paths)}
        for horizon, entries in sorted(buckets.items())
    }



def _first(value: Any, default: Any) -> Any:
    """First element of a batched metadata field (species batches are lists)."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _lag1(values: np.ndarray) -> Optional[float]:
    if values.shape[0] < 3:
        return None
    out = []
    for k in range(values.shape[1]):
        a, b = values[1:, k], values[:-1, k]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(out)) if out else None

def _path_ensemble_diagnostics(
    real: np.ndarray,
    transition_mean: np.ndarray,
    samples: np.ndarray,
) -> Dict[str, Any]:
    """Separate cross-realization spread from temporal path variation."""
    if real.ndim != 2 or transition_mean.shape != real.shape:
        raise ValueError("real and transition-mean paths must share shape (T,C)")
    if samples.ndim != 3 or samples.shape[1:] != real.shape:
        raise ValueError("sample paths must have shape (S,T,C)")
    realization_lag1 = [_lag1(path) for path in samples]
    finite_lag1 = [
        value for value in realization_lag1
        if value is not None and np.isfinite(value)
    ]
    return {
        "between_sample_variance": float(np.var(samples, axis=0).mean()),
        "within_path_temporal_variance": float(np.var(samples, axis=1).mean()),
        "real_temporal_variance": float(np.var(real, axis=0).mean()),
        "realization_lag1": {
            "real": _lag1(real),
            "mean_path": _lag1(transition_mean),
            "per_realization": realization_lag1,
            "mean": (
                float(np.mean(finite_lag1)) if finite_lag1 else None),
        },
    }




def _fit_tderica_context_projections(
    context: np.ndarray,
    trajectories: List[np.ndarray],
    *,
    dim_embed: int,
    n_components: int,
    toolbox_path: Path,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Fit one context-only motif basis and project each held-out trajectory."""
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import decompose, delayembed, project

    if context.shape[0] <= dim_embed + 2:
        raise ValueError(
            f"context has {context.shape[0]} frames; dim_embed={dim_embed} "
            "leaves too few TDE-RICA samples")
    _, motifs, _ = decompose(
        context[:, :, None], dim_embed=dim_embed,
        n_components=n_components, whiten=False, lambda_=0.001,
        max_iter=200)
    components = [
        project(delayembed(trajectory, dim_embed), motifs)
        for trajectory in trajectories
    ]
    if any(not np.isfinite(value).all() for value in components):
        raise FloatingPointError("context-fitted TDE-RICA projection is non-finite")
    return components, motifs


def _pooled_tderica_distribution(
    real_components: np.ndarray,
    generated_components: List[np.ndarray],
    *,
    toolbox_path: Path,
) -> Dict[str, Any]:
    """Score one real feature cloud against all sampled projected paths."""
    if not generated_components:
        raise ValueError("pooled SDE scoring requires sampled trajectories")
    pooled = np.concatenate(generated_components, axis=0)
    if (real_components.ndim != 2 or pooled.ndim != 2
            or real_components.shape[1] != pooled.shape[1]
            or not np.isfinite(real_components).all()
            or not np.isfinite(pooled).all()):
        raise ValueError("TDE-RICA distribution inputs must be finite (T,K) arrays")
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import kl_divergence_1d, wasserstein_distance

    w1_per_component = wasserstein_distance(
        real_components, pooled, per_component=True)
    kl_per_component = kl_divergence_1d(real_components, pooled)
    return {
        "wasserstein_global": float(
            wasserstein_distance(real_components, pooled)),
        "wasserstein_per_component": np.asarray(
            w1_per_component, dtype=float).tolist(),
        "kl_divergence": float(np.mean(kl_per_component)),
        "kl_divergence_per_component": np.asarray(
            kl_per_component, dtype=float).tolist(),
        "n_real_features": int(real_components.shape[0]),
        "n_sample_features": int(pooled.shape[0]),
        "n_realizations": int(len(generated_components)),
    }



def _tderica_horizon_reports(
    real_all: np.ndarray,
    transition_mean_all: np.ndarray,
    sampled_paths_all: np.ndarray,
    rollouts: int,
    *,
    projector,
    toolbox_path: Path,
    include_d3: bool,
    minimum_input_frames: int = 3,
) -> List[Dict[str, Any]]:
    """Score each raw horizon independently through the same fitted basis."""
    frames_total = real_all.shape[0]
    if rollouts < 1 or frames_total % rollouts:
        raise ValueError(
            f"{frames_total} frames do not divide into {rollouts} horizons")
    if (real_all.ndim != 2
            or transition_mean_all.shape != real_all.shape
            or sampled_paths_all.ndim != 3
            or sampled_paths_all.shape[1:] != real_all.shape):
        raise ValueError("TDE-RICA paths must share the (S,T,C) contract")
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import kernel_transition_comparison

    horizon_frames = frames_total // rollouts
    minimum_input_frames = int(minimum_input_frames)
    if minimum_input_frames < 3:
        raise ValueError("minimum_input_frames must be at least three")
    if horizon_frames < minimum_input_frames:
        return [
            {
                "horizon": horizon + 1,
                "frames": int(horizon_frames),
                "tderica": None,
                "skipped_reason": (
                    f"requires at least {minimum_input_frames} input frames "
                    "for three projected frames"),
            }
            for horizon in range(rollouts)
        ]
    reports = []
    for horizon in range(rollouts):
        start = horizon * horizon_frames
        stop = start + horizon_frames
        real = projector(real_all[start:stop])
        mean_path = projector(transition_mean_all[start:stop])
        sample_paths = [
            projector(sample[start:stop]) for sample in sampled_paths_all
        ]
        if min(
            real.shape[0], mean_path.shape[0],
            *(sample.shape[0] for sample in sample_paths),
        ) < 3:
            reports.append({
                "horizon": horizon + 1,
                "frames": int(stop - start),
                "tderica": None,
                "skipped_reason": "fewer than three projected frames",
            })
            continue
        mean_distribution = _pooled_tderica_distribution(
            real, [mean_path], toolbox_path=toolbox_path)
        pooled_distribution = _pooled_tderica_distribution(
            real, sample_paths, toolbox_path=toolbox_path)
        mean_kernel = (
            float(kernel_transition_comparison(real, mean_path))
            if include_d3 else None)
        sample_kernels = (
            [float(kernel_transition_comparison(real, sample))
             for sample in sample_paths]
            if include_d3 else [None] * len(sample_paths))
        finite_sample_kernels = [
            value for value in sample_kernels if value is not None
        ]
        reports.append({
            "horizon": horizon + 1,
            "frames": int(stop - start),
            "tderica": {
                "mean_path_distribution": mean_distribution,
                "pooled_sample_distribution": pooled_distribution,
                "mean_path_kernel_transition": mean_kernel,
                "sample_kernel_transition_per_realization": sample_kernels,
                "sample_kernel_transition_mean": (
                    float(np.mean(finite_sample_kernels))
                    if finite_sample_kernels else None),
            },
        })
    return reports


def _sde_tderica_reports(
    real_components: np.ndarray,
    mean_components: np.ndarray,
    sample_components: List[np.ndarray],
    *,
    toolbox_path: Path,
    mean_similarity: Optional[Dict[str, Any]] = None,
    include_d3: bool = False,
    d3_fast: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Return mean-path alignment, per-realization reports, pooled W1/KL."""
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import similarity_report

    if mean_similarity is None:
        mean_similarity = similarity_report(
            real_components, mean_components,
            include_d3=include_d3, d3_fast=d3_fast)
    sample_similarities = [
        similarity_report(
            real_components, sample, include_d3=False, d3_fast=True)
        for sample in sample_components
    ]
    distribution = _pooled_tderica_distribution(
        real_components, sample_components, toolbox_path=toolbox_path)
    return mean_similarity, sample_similarities, distribution
def _load_recording_basis(
        basis_bank: Path,
        sample_id: str,
) -> Tuple[Path, np.ndarray, np.ndarray]:
    """Load one recording-local basis from a Randi basis bank."""
    recording_id = str(sample_id).rsplit("__", 1)[-1]
    candidates = (
        basis_bank / f"randi_{recording_id}.npz",
        basis_bank / f"{sample_id}.npz",
    )
    basis_path = next((path for path in candidates if path.exists()), None)
    if basis_path is None:
        raise FileNotFoundError(
            f"no recording-local TDE-RICA basis for sample {sample_id!r}; "
            f"checked {[str(path) for path in candidates]}"
        )
    with np.load(basis_path, allow_pickle=False) as values:
        required = {"coeffEmbed2", "channel_indices"}
        missing = required.difference(values.files)
        if missing:
            raise ValueError(
                f"{basis_path} is missing basis arrays {sorted(missing)}"
            )
        coeff = np.asarray(values["coeffEmbed2"], dtype=float)
        indices = np.asarray(values["channel_indices"], dtype=np.int64).reshape(-1)
    if coeff.ndim != 3 or coeff.shape[0] < 1 or coeff.shape[1] < 2:
        raise ValueError(f"{basis_path} has invalid coefficient shape {coeff.shape}")
    if coeff.shape[2] != indices.size or indices.size < 2:
        raise ValueError(
            f"{basis_path} has incompatible basis channels: "
            f"{coeff.shape} and {indices.shape}"
        )
    if indices.min() < 0 or not np.isfinite(coeff).all():
        raise ValueError(f"{basis_path} contains invalid indices or coefficients")
    return basis_path, coeff, indices


def _project_recording_basis(
        signal: np.ndarray,
        coeff: np.ndarray,
        channel_indices: np.ndarray,
        valid_channels: np.ndarray,
        toolbox_path: Path,
) -> Tuple[np.ndarray, int]:
    """Project one signal through a recording-local basis and valid mask."""
    if signal.ndim != 2:
        raise ValueError(f"signal must be (T, C); got {signal.shape}")
    valid_channels = np.asarray(valid_channels, dtype=bool).reshape(-1)
    if channel_indices.max() >= valid_channels.size:
        raise ValueError(
            f"basis channel index {int(channel_indices.max())} exceeds "
            f"signal channel count {valid_channels.size}"
        )
    keep = valid_channels[channel_indices]
    if int(keep.sum()) < 2:
        raise ValueError("fewer than two fully observed basis channels")
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import delayembed, project
    selected = signal[:, channel_indices[keep]]
    selected_coeff = coeff[:, :, keep]
    embedded = delayembed(selected, int(coeff.shape[1]))
    projected = project(embedded, selected_coeff)
    if not np.isfinite(projected).all():
        raise FloatingPointError("recording-local TDE-RICA projection is non-finite")
    return projected, int(keep.sum())






def _forward_free_run_paths(
    model,
    signals: Dict[str, torch.Tensor],
    *,
    masks: Dict[str, torch.Tensor],
    rollout_steps: int,
    dt: torch.Tensor,
    frame_dt: torch.Tensor,
    sample_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Keep the conditional-mean path distinct from SDE realizations."""
    common = {
        "masks": masks,
        "perturbation": None,
        "num_steps": rollout_steps,
        "return_all": True,
        "return_sequences": True,
        "reconstruct": True,
        "recon_max_channels": 2048,
        "dt": dt,
        "frame_dt": frame_dt,
        "step_dt": dt,
    }
    if getattr(model, "transition_mode", "ode") != "sde":
        return {
            "transition_mean_path": model.forward_modalities(
                signals, **common),
            "sampled_paths": None,
            "sample_count": 0,
        }
    count = int(
        getattr(model, "stochastic_samples", 0)
        if sample_count is None else sample_count)
    if count < 2:
        raise ValueError("SDE free-run evaluation requires at least two samples")

    def reset_runtime():
        reset = getattr(model, "reset_runtime_state", None)
        if callable(reset):
            reset(batch_size=next(iter(signals.values())).shape[0])

    reset_runtime()
    mean_path = model.forward_modalities(signals, **common)
    reset_runtime()
    sampled_paths = model.forward_modalities(
        signals, **common, sample_transition=True, num_samples=count)
    return {
        "transition_mean_path": mean_path,
        "sampled_paths": sampled_paths,
        "sample_count": count,
    }
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--rollout", type=int, default=3)
    parser.add_argument(
        "--sde-samples", type=int, default=None,
        help="SDE realizations; defaults to the model profile's sample count",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda"))
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed model initialization and sampled transition paths",
    )
    parser.add_argument(
        "--tderica-basis-bank", type=Path, default=None,
        help="recording-local fixed basis bank; bypasses per-window fitting",
    )
    parser.add_argument("--tderica", type=Path, default=ROOT.parent / "TDE-RICA")
    parser.add_argument("--tderica-dim-embed", type=int, default=12)
    parser.add_argument("--tderica-components", type=int, default=5)
    parser.add_argument("--no-tderica-fit", action="store_true")
    parser.add_argument("--no-d3", action="store_true")
    parser.add_argument("--full-d3", action="store_true")
    parser.add_argument("--save-arrays", type=Path, default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.max_samples < 1 or args.rollout < 1:
        raise ValueError("max-samples and rollout must be positive")
    if args.tderica_dim_embed < 2 or args.tderica_components < 1:
        raise ValueError("TDE-RICA dimensions must be positive")
    if args.tderica_basis_bank is not None and not args.tderica_basis_bank.is_dir():
        raise ValueError(
            f"TDE-RICA basis bank is not a directory: {args.tderica_basis_bank}"
        )

    device = _device(args.device)
    experiment_config = BrainMoEPINNConfig.from_file(str(args.config))
    model = experiment_config.to_model().to(device).eval()
    checkpoint_info = _load_checkpoint(
        model, args.checkpoint, device, config_path=args.config)

    is_sde = getattr(model, "transition_mode", "ode") == "sde"
    if args.sde_samples is not None and args.sde_samples < 2:
        raise ValueError("--sde-samples must be at least 2")
    if args.sde_samples is not None and not is_sde:
        raise ValueError("--sde-samples requires an SDE model")
    if is_sde and args.no_tderica_fit and args.tderica_basis_bank is None:
        raise ValueError(
            "SDE pooled TDE-RICA metrics require context fitting or a "
            "recording-local fixed basis")
    sde_sample_count = (
        int(args.sde_samples or model.stochastic_samples)
        if is_sde else 0)
    from train import build_data_loaders
    loaders = build_data_loaders(args.data)
    loader = loaders[1] if args.split == "val" else (loaders[2] if len(loaders) > 2 else None)
    if loader is None:
        raise ValueError(f"requested split {args.split!r} is unavailable")

    reports: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    if args.save_arrays is not None:
        args.save_arrays.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            if sample_index >= args.max_samples:
                break
            if "calcium" not in batch or "calcium_future" not in batch:
                continue
            signals, targets, _ = partition_generic_batch(
                batch, signal_modalities=("calcium",), recon_modalities=("calcium",),
                device=device, control_modalities=(), rollout_steps=args.rollout,
                require_future_targets=True, require_multi_horizon=True)
            current = signals["calcium"]
            future = targets["calcium"]
            if current.shape[0] != 1 or future.shape[0] != 1:
                raise ValueError("free-run evaluator requires batch_size=1")
            if future.shape[1] != args.rollout:
                raise ValueError(f"future target has {future.shape[1]} horizons, requested {args.rollout}")
            context_mask = batch.get("calcium_mask")
            if context_mask is None:
                context_mask = torch.ones_like(current, dtype=torch.bool)
            future_mask = batch.get("calcium_future_mask")
            if future_mask is None:
                future_mask = torch.ones_like(future, dtype=torch.bool)
            context_mask = context_mask.to(device)
            future_mask = future_mask.to(device)
            dt_frame = _as_per_sample_dt(batch.get("dt"), 1, device)
            window_dt = dt_frame * current.shape[-1]
            path_outputs = _forward_free_run_paths(
                model,
                {"calcium": current},
                masks={"calcium": context_mask},
                rollout_steps=args.rollout,
                dt=window_dt,
                frame_dt=dt_frame,
                sample_count=sde_sample_count if is_sde else None,
            )
            mean_output = path_outputs["transition_mean_path"]
            generated_seq = mean_output.get("calcium_recon_sequence")
            if generated_seq is None:
                raise RuntimeError(
                    "model did not emit calcium_recon_sequence")
            real_all = _concat_horizons(future[0])
            generated_all = _concat_horizons(generated_seq[0])
            sampled_all_full = None
            if is_sde:
                sampled_recon = path_outputs["sampled_paths"].get(
                    "calcium_recon_samples")
                if not isinstance(sampled_recon, torch.Tensor):
                    raise RuntimeError(
                        "SDE model did not emit calcium_recon_samples")
                sampled_all_full = np.stack([
                    _concat_horizons(sampled_recon[index, 0])
                    for index in range(path_outputs["sample_count"])
                ])
            target_mask = _concat_masks(future_mask[0])
            valid_channels = target_mask.all(axis=0)
            if int(valid_channels.sum()) < 2:
                # A window whose future frames are masked out cannot be scored.
                # Large heterogeneous audits contain such rows, so they are
                # recorded as skips (and counted in the report) instead of
                # aborting the whole run; the model itself is never excused.
                skipped.append({
                    "index": sample_index,
                    "sample_id": str(_first(batch.get("sample_id"), sample_index)),
                    "reason": f"{int(valid_channels.sum())} fully observed "
                              "future channels",
                })
                if len(skipped) == 1:
                    print(f"[skip] sample {sample_index}: "
                          f"{skipped[-1]['reason']}", flush=True)
                continue
            context_all = current[0].detach().float().cpu().numpy().T
            context_mask_np = context_mask[0].detach().bool().cpu().numpy().T
            context_valid = context_mask_np[:, valid_channels].all(axis=0)
            fit_valid_channels = valid_channels.copy()
            fit_valid_channels[fit_valid_channels] = context_valid
            real = real_all[:, fit_valid_channels]
            transition_mean = generated_all[:, fit_valid_channels]
            generated = transition_mean
            context = context_all[:, fit_valid_channels]
            persistence = np.tile(context, (args.rollout, 1))
            sampled = (
                sampled_all_full[:, :, fit_valid_channels]
                if is_sde else None)
            sampled_paths_raw = (
                sampled if is_sde else generated[None, :, :])
            if (not np.isfinite(real).all()
                    or not np.isfinite(transition_mean).all()
                    or (is_sde and not np.isfinite(sampled).all())):
                raise FloatingPointError(
                    f"sample {sample_index} produced non-finite free-run output")
            real_std = np.std(real, axis=0)
            sample_std = np.std(sampled_paths_raw, axis=1)
            sample_ratios = (
                sample_std / np.maximum(real_std[None, :], 1e-12))
            std_ratio = float(np.mean(sample_ratios))
            std_ratio_median = float(np.median(sample_ratios))
            tail_frames = max(1, real.shape[0] // 5)
            generated_tail_ratio = float(np.mean([
                np.std(path[-tail_frames:], axis=0).mean()
                / max(np.std(path[:tail_frames], axis=0).mean(), 1e-12)
                for path in sampled_paths_raw
            ]))
            sampled_native_reports = [
                run_free_run_suite(real, path) for path in sampled_paths_raw
            ]
            frame_dt_s = float(dt_frame.item())
            sample_id = _first(batch.get("sample_id"), str(sample_index))
            origin = _first(batch.get("origin"), "")
            subject = _first(batch.get("subject"), "")
            basis_metadata = None
            mean_similarity = None
            sample_similarities = []
            pooled_distribution = None
            mean_tderica_details = None
            mean_distribution = None
            comp_real = None
            comp_generated = None
            projected_samples = []
            tderica_projector = None
            horizon_min_input_frames = 3
            horizon_real = real
            horizon_mean = transition_mean
            horizon_samples = sampled_paths_raw
            if args.tderica_basis_bank is not None:
                basis_path, basis_coeff, basis_indices = _load_recording_basis(
                    args.tderica_basis_bank, str(sample_id)
                )
                comp_real, basis_used = _project_recording_basis(
                    real_all, basis_coeff, basis_indices, fit_valid_channels,
                    args.tderica,
                )
                comp_generated, generated_basis_used = _project_recording_basis(
                    generated_all, basis_coeff, basis_indices, fit_valid_channels,
                    args.tderica,
                )
                comp_persistence, persistence_basis_used = _project_recording_basis(
                    np.tile(context_all, (args.rollout, 1)), basis_coeff,
                    basis_indices, fit_valid_channels, args.tderica,
                )
                if len({basis_used, generated_basis_used, persistence_basis_used}) != 1:
                    raise ValueError(
                        f"basis channel masks diverged for sample {sample_id}: "
                        f"{basis_used}, {generated_basis_used}, "
                        f"{persistence_basis_used}"
                    )
                projected_samples = []
                sampled_full_paths = (
                    sampled_all_full
                    if is_sde else generated_all[None, :, :])
                for index, sample_path in enumerate(sampled_full_paths):
                    comp_sample, sample_basis_used = _project_recording_basis(
                        sample_path, basis_coeff, basis_indices,
                        fit_valid_channels, args.tderica)
                    if sample_basis_used != basis_used:
                        raise ValueError(
                            f"sample {index} used {sample_basis_used} basis "
                            f"channels; expected {basis_used}")
                    projected_samples.append(comp_sample)
                projected_model = tderica_biological_report(
                    comp_real, comp_generated, dt_s=frame_dt_s,
                    include_d3=not args.no_d3, d3_fast=not args.full_d3,
                    tderica_path=str(args.tderica),
                )
                mean_similarity, sample_similarities, pooled_distribution = (
                    _sde_tderica_reports(
                        comp_real, comp_generated, projected_samples,
                        toolbox_path=args.tderica,
                        mean_similarity=projected_model.get("tderica"),
                        include_d3=not args.no_d3,
                        d3_fast=not args.full_d3))
                mean_tderica_details = projected_model
                def tderica_projector(signal):
                    return _project_recording_basis(
                        signal, basis_coeff, basis_indices, fit_valid_channels,
                        args.tderica)[0]
                horizon_real = real_all
                horizon_mean = generated_all
                horizon_samples = sampled_full_paths
                tderica_report = {
                    **projected_model,
                    "native": run_free_run_suite(real, generated),
                }
                projected_persistence = tderica_biological_report(
                    comp_real, comp_persistence, dt_s=frame_dt_s,
                    include_d3=False, tderica_path=str(args.tderica),
                )
                persistence_report = {
                    **projected_persistence,
                    "native": run_free_run_suite(real, persistence),
                }
                basis_metadata = {
                    "path": str(basis_path),
                    "components": int(basis_coeff.shape[0]),
                    "embed_width": int(basis_coeff.shape[1]),
                    "basis_channels": int(basis_coeff.shape[2]),
                    "basis_channels_used": int(basis_used),
                }
                horizon_min_input_frames = int(basis_coeff.shape[1]) + 2
            elif args.no_tderica_fit:
                comp_real = real
                comp_generated = generated
                projected_samples = [path for path in sampled_paths_raw]
                tderica_report = tderica_biological_report(
                    real, generated, dt_s=frame_dt_s, include_d3=not args.no_d3,
                    d3_fast=not args.full_d3, tderica_path=str(args.tderica))
                mean_similarity = tderica_report.get("tderica")
                pooled_distribution = _pooled_tderica_distribution(
                    comp_real, projected_samples, toolbox_path=args.tderica)
                def tderica_projector(signal):
                    return signal
                persistence_report = tderica_biological_report(
                    real, persistence, dt_s=frame_dt_s, include_d3=False,
                    tderica_path=str(args.tderica))
            else:
                fit_dim = min(
                    args.tderica_dim_embed, max(2, context.shape[0] - 3))
                horizon_min_input_frames = int(fit_dim) + 2
                fit_points = context.shape[0] - fit_dim + 1
                fit_components = min(
                    args.tderica_components, fit_points,
                    fit_dim * context.shape[1])
                fit_components = max(1, fit_components)
                projected, motifs = _fit_tderica_context_projections(
                    context,
                    [real, generated, *list(sampled_paths_raw)],
                    dim_embed=fit_dim,
                    n_components=fit_components,
                    toolbox_path=args.tderica,
                )
                comp_real, comp_generated, *projected_samples = projected
                mean_similarity, sample_similarities, pooled_distribution = (
                    _sde_tderica_reports(
                        comp_real, comp_generated, projected_samples,
                        toolbox_path=args.tderica,
                        include_d3=not args.no_d3,
                        d3_fast=not args.full_d3))
                roughness = []
                spatial = []
                for motif in motifs:
                    if motif.shape[0] >= 3:
                        roughness.append(float(np.mean(np.var(
                            np.diff(motif, n=2, axis=0), axis=0))))
                    if motif.shape[1] >= 2:
                        corr = np.corrcoef(motif.T)
                        upper = corr[np.triu_indices(corr.shape[0], k=1)]
                        upper = upper[np.isfinite(upper)]
                        if upper.size:
                            spatial.append(float(np.mean(upper)))
                mean_tderica_details = {
                    "dim_embed": int(fit_dim),
                    "n_components": int(motifs.shape[0]),
                    "component_shape": [
                        int(comp_real.shape[0]), int(comp_real.shape[1])],
                    "similarity": mean_similarity,
                    "real_occurrence_lag1": _lag1(comp_real),
                    "generated_occurrence_lag1": _lag1(comp_generated),
                    "motif_roughness_mean": (
                        float(np.mean(roughness)) if roughness else None),
                    "motif_spatial_coherence_mean": (
                        float(np.mean(spatial)) if spatial else None),
                }
                tderica_report = {
                    "native": run_free_run_suite(real, generated),
                    **mean_tderica_details,
                }
                if str(args.tderica) not in sys.path:
                    sys.path.insert(0, str(args.tderica))
                from tderica import delayembed, project

                def tderica_projector(signal):
                    return project(delayembed(signal, fit_dim), motifs)

                basis_metadata = {
                    "mode": "context_fitted",
                    "dim_embed": int(fit_dim),
                    "components": int(motifs.shape[0]),
                }
                persistence_report = tderica_biological_report(
                    real, persistence, dt_s=frame_dt_s, include_d3=False,
                    tderica_path=str(args.tderica))
            if (comp_real is None or comp_generated is None
                    or not projected_samples
                    or not callable(tderica_projector)):
                raise RuntimeError("TDE-RICA projection setup is incomplete")
            mean_distribution = _pooled_tderica_distribution(
                comp_real, [comp_generated], toolbox_path=args.tderica)
            tderica_horizons = _tderica_horizon_reports(
                horizon_real, horizon_mean, horizon_samples, args.rollout,
                projector=tderica_projector,
                toolbox_path=args.tderica,
                include_d3=not args.no_d3,
                minimum_input_frames=horizon_min_input_frames,
            )
            mean_kernel_transition = None
            sample_kernel_transitions = [None] * len(projected_samples)
            if not args.no_d3:
                if str(args.tderica) not in sys.path:
                    sys.path.insert(0, str(args.tderica))
                from tderica import kernel_transition_comparison

                mean_kernel_transition = float(kernel_transition_comparison(
                    comp_real, comp_generated))
                sample_kernel_transitions = [
                    float(kernel_transition_comparison(comp_real, sample))
                    for sample in projected_samples
                ]
            path_diagnostics = _path_ensemble_diagnostics(
                real, transition_mean, sampled_paths_raw)
            if is_sde:
                if pooled_distribution is None or not sample_similarities:
                    raise RuntimeError(
                        "SDE evaluation did not produce pooled TDE-RICA metrics")
                tderica_report = {
                    "native": _mean_numeric_reports(sampled_native_reports),
                    "trajectory_similarity": _mean_numeric_reports(
                        sample_similarities),
                    "pooled_distribution": pooled_distribution,
                    "distribution_source": "pooled_sampled_realizations",
                    "n_realizations": path_outputs["sample_count"],
                    "realizations": [
                        {
                            "index": index,
                            "native": sampled_native_reports[index],
                            "trajectory_similarity": sample_similarities[index],
                            "kernel_transition": sample_kernel_transitions[index],
                        }
                        for index in range(path_outputs["sample_count"])
                    ],
                }
            else:
                tderica_report["pooled_distribution"] = pooled_distribution
                tderica_report["distribution_source"] = (
                    "single_deterministic_path")
                tderica_report["n_realizations"] = 1
            finite_sample_kernels = [
                value for value in sample_kernel_transitions
                if value is not None
            ]
            tderica_report["sample_kernel_transition_per_realization"] = (
                sample_kernel_transitions)
            tderica_report["sample_kernel_transition_mean"] = (
                float(np.mean(finite_sample_kernels))
                if finite_sample_kernels else None)
            transition_mean_report = {
                "native": run_free_run_suite(real, transition_mean),
                "similarity": mean_similarity,
                "distribution": mean_distribution,
                "kernel_transition": mean_kernel_transition,
                "horizons": tderica_horizons,
                "tderica_report": mean_tderica_details,
            }
            if is_sde:
                sampled_output = path_outputs["sampled_paths"]
                transition_diagnostics = sde_transition_diagnostics(
                    sampled_output["z_global"].detach().float().cpu().numpy(),
                    sampled_output[
                        "z_next_sequence_samples"
                    ].detach().float().cpu().numpy(),
                    sampled_output[
                        "transition_mean_sequence_samples"
                    ].detach().float().cpu().numpy(),
                    sampled_output[
                        "transition_factor_sequence_samples"
                    ].detach().float().cpu().numpy(),
                    sampled_output[
                        "transition_diag_std_sequence_samples"
                    ].detach().float().cpu().numpy(),
                    model.transition_diffusion.factor_basis
                    .detach().float().cpu().numpy(),
                    dt=window_dt.detach().float().cpu().numpy(),
                )
            horizon_reports = _sampled_horizon_reports(
                real, sampled_paths_raw, transition_mean, args.rollout)
            tderica_by_horizon = {
                entry["horizon"]: entry["tderica"]
                for entry in tderica_horizons
            }
            for entry in horizon_reports:
                if entry["horizon"] in tderica_by_horizon:
                    entry["tderica"] = tderica_by_horizon[entry["horizon"]]
            report = {
                "index": sample_index, "sample_id": str(sample_id),
                "origin": str(origin), "subject": str(subject),
                "source": _source_label(str(sample_id), str(origin)),
                "transition_mode": "sde" if is_sde else "ode",
                "channels_scored": int(real.shape[1]),
                "channels_available": int(real_all.shape[1]),
                "frames": int(real.shape[0]), "frame_dt_s": frame_dt_s,
                "target_valid_channel_fraction": float(valid_channels.mean()),
                "r_sigma": std_ratio,
                "raw_std_ratio": std_ratio,
                "raw_std_ratio_median": std_ratio_median,
                "generated_tail_std_ratio": generated_tail_ratio,
                "path_diagnostics": path_diagnostics,
                "horizons": horizon_reports,
                "model": tderica_report, "persistence": persistence_report,
                "tderica_basis": basis_metadata,
                "n_realizations": len(sampled_paths_raw),
                "transition_mean_path": transition_mean_report,
            }
            if is_sde:
                report["transition_diagnostics"] = transition_diagnostics
            reports.append(report)
            if args.save_arrays is not None:
                np.save(
                    args.save_arrays / f"{sample_index:04d}_real.npy",
                    real.astype("float32"))
                if is_sde:
                    np.save(
                        args.save_arrays
                        / f"{sample_index:04d}_transition_mean_path.npy",
                        transition_mean.astype("float32"))
                    np.save(
                        args.save_arrays
                        / f"{sample_index:04d}_sampled_realizations.npy",
                        sampled.astype("float32"))
                else:
                    np.save(
                        args.save_arrays / f"{sample_index:04d}_generated.npy",
                        generated.astype("float32"))

    if not reports:
        raise RuntimeError("no evaluable calcium future samples found")
    aggregate_paths = {
        "native_corr_matrix_mse": ("model", "native", "corr_matrix_mse"),
        "native_autocorr_mse": ("model", "native", "autocorr", "mse"),
        "native_variance_log_rmse": ("model", "native", "variance_ratio", "log_rmse"),
        "r_sigma": ("r_sigma",),
        "raw_std_ratio": ("raw_std_ratio",),
        "raw_std_ratio_median": ("raw_std_ratio_median",),
        "generated_tail_std_ratio": ("generated_tail_std_ratio",),
        "between_sample_variance": (
            "path_diagnostics", "between_sample_variance"),
        "within_path_temporal_variance": (
            "path_diagnostics", "within_path_temporal_variance"),
        "real_temporal_variance": (
            "path_diagnostics", "real_temporal_variance"),
        "realization_lag1_mean": (
            "path_diagnostics", "realization_lag1", "mean"),
        "mean_path_lag1": (
            "path_diagnostics", "realization_lag1", "mean_path"),
        "mean_path_w1": (
            "transition_mean_path", "distribution", "wasserstein_global"),
        "mean_path_kl": (
            "transition_mean_path", "distribution", "kl_divergence"),
        "mean_path_kernel_transition": (
            "transition_mean_path", "kernel_transition"),
        "pooled_sample_w1": (
            "model", "pooled_distribution", "wasserstein_global"),
        "pooled_sample_kl": (
            "model", "pooled_distribution", "kl_divergence"),
        "sample_kernel_transition_mean": (
            "model", "sample_kernel_transition_mean"),
        "persistence_native_corr_matrix_mse": (
            "persistence", "native", "corr_matrix_mse"),
    }
    if is_sde:
        aggregate_paths.update({
            "sde_drift_increment_rms": (
                "transition_diagnostics", "sde_drift_increment_rms"),
            "sde_diffusion_rms": (
                "transition_diagnostics", "sde_diffusion_rms"),
            "sde_noise_increment_rms": (
                "transition_diagnostics", "sde_noise_increment_rms"),
            "sde_noise_drift_ratio": (
                "transition_diagnostics", "sde_noise_drift_ratio"),
            "sde_trace_q": ("transition_diagnostics", "sde_trace_q"),
            "sde_standardized_innovation_rms": (
                "transition_diagnostics",
                "sde_standardized_innovation_rms"),
            "sde_innovation_lag_autocorr_rms_1": (
                "transition_diagnostics",
                "sde_innovation_lag_autocorr_rms_1"),
            "sde_innovation_lag_autocorr_rms_2": (
                "transition_diagnostics",
                "sde_innovation_lag_autocorr_rms_2"),
            "sde_innovation_lag_autocorr_rms_4": (
                "transition_diagnostics",
                "sde_innovation_lag_autocorr_rms_4"),
            "tderica_dtw": (
                "model", "trajectory_similarity", "time_alignment",
                "dtw_distance"),
            "tderica_frechet": (
                "model", "trajectory_similarity", "time_alignment",
                "frechet_distance"),
            "tderica_w1": (
                "model", "pooled_distribution", "wasserstein_global"),
            "tderica_w1_component_mean": (
                "model", "pooled_distribution", "wasserstein_per_component"),
            "tderica_kl_mean": (
                "model", "pooled_distribution", "kl_divergence"),
            "tderica_cosine_diagonal_mean": (
                "model", "trajectory_similarity", "time_alignment",
                "cosine_diagonal_mean"),
            "tderica_cosine_global_mean": (
                "model", "trajectory_similarity", "time_alignment",
                "cosine_global_mean"),
            "transition_mean_native_corr_matrix_mse": (
                "transition_mean_path", "native", "corr_matrix_mse"),
            "transition_mean_native_autocorr_mse": (
                "transition_mean_path", "native", "autocorr", "mse"),
            "transition_mean_native_variance_log_rmse": (
                "transition_mean_path", "native", "variance_ratio",
                "log_rmse"),
            "transition_mean_tderica_dtw": (
                "transition_mean_path", "similarity", "time_alignment",
                "dtw_distance"),
            "transition_mean_tderica_frechet": (
                "transition_mean_path", "similarity", "time_alignment",
                "frechet_distance"),
            "transition_mean_tderica_w1": (
                "transition_mean_path", "similarity", "distribution",
                "wasserstein_global"),
            "transition_mean_tderica_kl": (
                "transition_mean_path", "similarity", "distribution",
                "kl_divergence"),
            "transition_mean_tderica_kernel_transition": (
                "transition_mean_path", "similarity", "dynamics",
                "kernel_transition"),
            "transition_mean_tderica_transfer_entropy_a_to_b": (
                "transition_mean_path", "similarity", "dynamics",
                "transfer_entropy_a_to_b"),
            "transition_mean_tderica_transfer_entropy_b_to_a": (
                "transition_mean_path", "similarity", "dynamics",
                "transfer_entropy_b_to_a"),
        })
    elif args.no_tderica_fit or args.tderica_basis_bank is not None:
        aggregate_paths.update({
            "tderica_dtw": ("model", "tderica", "time_alignment", "dtw_distance"),
            "tderica_frechet": ("model", "tderica", "time_alignment", "frechet_distance"),
            "tderica_w1": ("model", "tderica", "distribution", "wasserstein_global"),
            "tderica_w1_component_mean": (
                "model", "tderica", "distribution", "wasserstein_per_component"),
            "tderica_kl_mean": ("model", "tderica", "distribution", "kl_divergence"),
            "tderica_cosine_diagonal_mean": (
                "model", "tderica", "time_alignment", "cosine_diagonal_mean"),
            "tderica_cosine_global_mean": (
                "model", "tderica", "time_alignment", "cosine_global_mean"),
            "tderica_kernel_transition": (
                "model", "tderica", "dynamics", "kernel_transition"),
            "tderica_transfer_entropy_a_to_b": (
                "model", "tderica", "dynamics", "transfer_entropy_a_to_b"),
            "tderica_transfer_entropy_b_to_a": (
                "model", "tderica", "dynamics", "transfer_entropy_b_to_a"),
            "tderica_jacobian_distance": (
                "model", "tderica", "dynamics", "local_jacobian", "jacobian_distance"),
            "tderica_expansion_diff": (
                "model", "tderica", "dynamics", "local_jacobian", "expansion_diff"),
            "tderica_rotation_diff": (
                "model", "tderica", "dynamics", "local_jacobian", "rotation_diff"),
        })
    else:
        aggregate_paths.update({
            "tderica_dtw": ("model", "similarity", "time_alignment", "dtw_distance"),
            "tderica_frechet": ("model", "similarity", "time_alignment", "frechet_distance"),
            "tderica_w1": ("model", "similarity", "distribution", "wasserstein_global"),
            "tderica_w1_component_mean": (
                "model", "similarity", "distribution", "wasserstein_per_component"),
            "tderica_kl_mean": ("model", "similarity", "distribution", "kl_divergence"),
            "tderica_cosine_diagonal_mean": (
                "model", "similarity", "time_alignment", "cosine_diagonal_mean"),
            "tderica_cosine_global_mean": (
                "model", "similarity", "time_alignment", "cosine_global_mean"),
            "tderica_kernel_transition": (
                "model", "similarity", "dynamics", "kernel_transition"),
            "tderica_transfer_entropy_a_to_b": (
                "model", "similarity", "dynamics", "transfer_entropy_a_to_b"),
            "tderica_transfer_entropy_b_to_a": (
                "model", "similarity", "dynamics", "transfer_entropy_b_to_a"),
            "tderica_jacobian_distance": (
                "model", "similarity", "dynamics", "local_jacobian", "jacobian_distance"),
            "tderica_expansion_diff": (
                "model", "similarity", "dynamics", "local_jacobian", "expansion_diff"),
            "tderica_rotation_diff": (
                "model", "similarity", "dynamics", "local_jacobian", "rotation_diff"),
            "real_occurrence_lag1": ("model", "real_occurrence_lag1"),
            "generated_occurrence_lag1": ("model", "generated_occurrence_lag1"),
            "motif_roughness_mean": ("model", "motif_roughness_mean"),
            "motif_spatial_coherence_mean": ("model", "motif_spatial_coherence_mean"),
        })
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for report in reports:
        by_source.setdefault(str(report.get("source", "unknown")), []).append(report)
    output = {
        "checkpoint": str(args.checkpoint), "split": args.split,
        "device": str(device), "rollout_windows": args.rollout,
        "transition_mode": "sde" if is_sde else "ode",
        "sde_samples": sde_sample_count if is_sde else None,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "tderica": str(args.tderica),
        "tderica_dim_embed": args.tderica_dim_embed,
        "tderica_components": args.tderica_components,
        "transition_mean_semantics": (
            "recursive conditional transition means; not the exact "
            "multi-step expectation of the nonlinear SDE"
            if is_sde else None),
        "sde_distribution_metric_source": (
            "pooled sampled trajectories projected through one TDE-RICA basis"
            if is_sde else None),
        "tderica_context_fit": (
            not args.no_tderica_fit and args.tderica_basis_bank is None
        ),
        "tderica_basis_mode": (
            "recording_local_fixed"
            if args.tderica_basis_bank is not None else None
        ),
        "tderica_basis_bank": (
            str(args.tderica_basis_bank)
            if args.tderica_basis_bank is not None else None
        ),
        "checkpoint_info": checkpoint_info,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "aggregate": _group_aggregate(reports, aggregate_paths),
        "aggregate_by_source": {
            source: {"n_samples": len(group),
                     "metrics": _group_aggregate(group, aggregate_paths)}
            for source, group in sorted(by_source.items())
        },
        "aggregate_by_horizon": _horizon_aggregate(reports),
        "samples": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=_json_default) + "\n")
    print(json.dumps({"output": str(args.output), "n_samples": len(reports),
                      "n_skipped": len(skipped), "device": str(device),
                      "aggregate": output["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
