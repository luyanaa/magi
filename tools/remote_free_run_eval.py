#!/usr/bin/env python3
"""Run deterministic biological free-run evaluation on a GPU checkpoint.

The evaluator uses the production C. elegans loader, rolls the latent system
forward without controls or observation reinjection, and compares the result
with held-out future windows. It reports native statistics plus a TDE-RICA
reference representation fitted only on each sample's context window. The
per-sample context fit avoids asserting cross-source neuron identity in the
unified federated ladder; for a fixed pre-fit motif basis, use
``tools/tderica_free_run.py`` with that basis externally.
"""
from __future__ import annotations

import argparse
import json
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
    run_free_run_suite, tderica_biological_report,
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


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> Dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    model_state = state.get("model_state", state.get("state_dict"))
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint has no model_state/state_dict: {path}")
    incompatible = model.load_state_dict(model_state, strict=False)
    return {
        "checkpoint_step": state.get("step"),
        "checkpoint_phase": state.get("phase"),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "world_size": state.get("world_size"),
    }


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


def _aggregate_scalar(reports: Iterable[Dict[str, Any]], path: Tuple[str, ...]):
    values = []
    for report in reports:
        current: Any = report
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)) and np.isfinite(current):
            values.append(float(current))
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
    }


def _aggregate_dict_mean(reports: Iterable[Dict[str, Any]], path: Tuple[str, ...]):
    values = []
    for report in reports:
        current: Any = report
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, dict):
            finite = [float(v) for v in current.values()
                      if isinstance(v, (int, float)) and np.isfinite(v)]
            if finite:
                values.append(float(statistics.fmean(finite)))
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
    }


def _lag1(values: np.ndarray) -> Optional[float]:
    if values.shape[0] < 3:
        return None
    out = []
    for k in range(values.shape[1]):
        a, b = values[1:, k], values[:-1, k]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(out)) if out else None


def _fit_tderica_context(context, real, generated, *, dim_embed,
                         n_components, include_d3, d3_fast, toolbox_path):
    """Fit TDE-RICA on context only, then compare held-out trajectories."""
    path = str(toolbox_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from tderica import decompose, delayembed, project, similarity_report

    if context.shape[0] <= dim_embed + 2:
        raise ValueError(
            f"context has {context.shape[0]} frames; dim_embed={dim_embed} leaves "
            "too few TDE-RICA samples")
    occurrences, motifs, _ = decompose(
        context[:, :, None], dim_embed=dim_embed, n_components=n_components,
        whiten=False, lambda_=0.001, max_iter=200)
    comp_real = project(delayembed(real, dim_embed), motifs)
    comp_generated = project(delayembed(generated, dim_embed), motifs)
    report = similarity_report(
        comp_real, comp_generated, include_d3=include_d3, d3_fast=d3_fast)
    roughness = []
    spatial = []
    for motif in motifs:
        if motif.shape[0] >= 3:
            roughness.append(float(np.mean(np.var(np.diff(motif, n=2, axis=0), axis=0))))
        if motif.shape[1] >= 2:
            corr = np.corrcoef(motif.T)
            upper = corr[np.triu_indices(corr.shape[0], k=1)]
            upper = upper[np.isfinite(upper)]
            if upper.size:
                spatial.append(float(np.mean(upper)))
    return {
        "native": run_free_run_suite(real, generated),
        "similarity": report,
        "dim_embed": int(dim_embed),
        "n_components": int(n_components),
        "component_shape": [int(comp_real.shape[0]), int(comp_real.shape[1])],
        "real_occurrence_lag1": _lag1(comp_real),
        "generated_occurrence_lag1": _lag1(comp_generated),
        "motif_roughness_mean": float(np.mean(roughness)) if roughness else None,
        "motif_spatial_coherence_mean": float(np.mean(spatial)) if spatial else None,
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
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--tderica", type=Path, default=ROOT.parent / "TDE-RICA")
    parser.add_argument("--tderica-dim-embed", type=int, default=12)
    parser.add_argument("--tderica-components", type=int, default=5)
    parser.add_argument("--no-tderica-fit", action="store_true")
    parser.add_argument("--no-d3", action="store_true")
    parser.add_argument("--full-d3", action="store_true")
    parser.add_argument("--save-arrays", type=Path, default=None)
    args = parser.parse_args()
    if args.max_samples < 1 or args.rollout < 1:
        raise ValueError("max-samples and rollout must be positive")
    if args.tderica_dim_embed < 2 or args.tderica_components < 1:
        raise ValueError("TDE-RICA dimensions must be positive")

    device = _device(args.device)
    experiment_config = BrainMoEPINNConfig.from_file(str(args.config))
    model = experiment_config.to_model().to(device).eval()
    checkpoint_info = _load_checkpoint(model, args.checkpoint, device)

    from train import build_data_loaders
    loaders = build_data_loaders(args.data)
    loader = loaders[1] if args.split == "val" else (loaders[2] if len(loaders) > 2 else None)
    if loader is None:
        raise ValueError(f"requested split {args.split!r} is unavailable")

    reports: List[Dict[str, Any]] = []
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
            out = model.forward_modalities(
                {"calcium": current}, masks={"calcium": context_mask},
                perturbation=None, num_steps=args.rollout, return_all=True,
                return_sequences=True, reconstruct=True, recon_max_channels=2048,
                dt=window_dt, frame_dt=dt_frame, step_dt=window_dt)
            generated_seq = out.get("calcium_recon_sequence")
            if generated_seq is None:
                raise RuntimeError("model did not emit calcium_recon_sequence")
            real_all = _concat_horizons(future[0])
            generated_all = _concat_horizons(generated_seq[0])
            target_mask = _concat_masks(future_mask[0])
            valid_channels = target_mask.all(axis=0)
            if int(valid_channels.sum()) < 2:
                raise ValueError(f"sample {sample_index} has fewer than two fully observed future channels")
            context_all = current[0].detach().float().cpu().numpy().T
            context_mask_np = context_mask[0].detach().bool().cpu().numpy().T
            context_valid = context_mask_np[:, valid_channels].all(axis=0)
            fit_valid_channels = valid_channels.copy()
            fit_valid_channels[fit_valid_channels] = context_valid
            real = real_all[:, fit_valid_channels]
            generated = generated_all[:, fit_valid_channels]
            context = context_all[:, fit_valid_channels]
            persistence = np.tile(context, (args.rollout, 1))
            if not np.isfinite(real).all() or not np.isfinite(generated).all():
                raise FloatingPointError(f"sample {sample_index} produced non-finite free-run output")
            real_std = np.std(real, axis=0)
            generated_std = np.std(generated, axis=0)
            std_ratio = float(generated_std.mean() / max(real_std.mean(), 1e-12))
            std_ratio_median = float(np.median(
                generated_std / np.maximum(real_std, 1e-12)))
            tail_frames = max(1, real.shape[0] // 5)
            generated_tail_ratio = float(
                np.std(generated[-tail_frames:], axis=0).mean()
                / max(np.std(generated[:tail_frames], axis=0).mean(), 1e-12))
            frame_dt_s = float(dt_frame.item())
            if args.no_tderica_fit:
                tderica_report = tderica_biological_report(
                    real, generated, dt_s=frame_dt_s, include_d3=not args.no_d3,
                    d3_fast=not args.full_d3, tderica_path=str(args.tderica))
            else:
                fit_dim = min(args.tderica_dim_embed, max(2, context.shape[0] - 3))
                fit_points = context.shape[0] - fit_dim + 1
                fit_components = min(args.tderica_components, fit_points,
                                     fit_dim * context.shape[1])
                tderica_report = _fit_tderica_context(
                    context, real, generated, dim_embed=fit_dim,
                    n_components=max(1, fit_components), include_d3=not args.no_d3,
                    d3_fast=not args.full_d3, toolbox_path=args.tderica)
            persistence_report = tderica_biological_report(
                real, persistence, dt_s=frame_dt_s, include_d3=False,
                tderica_path=str(args.tderica))
            sample_id = batch.get("sample_id", [str(sample_index)])
            if isinstance(sample_id, (list, tuple)):
                sample_id = sample_id[0]
            reports.append({
                "index": sample_index, "sample_id": str(sample_id),
                "channels_scored": int(real.shape[1]),
                "channels_available": int(real_all.shape[1]),
                "frames": int(real.shape[0]), "frame_dt_s": frame_dt_s,
                "target_valid_channel_fraction": float(valid_channels.mean()),
                "raw_std_ratio": std_ratio,
                "raw_std_ratio_median": std_ratio_median,
                "generated_tail_std_ratio": generated_tail_ratio,
                "model": tderica_report, "persistence": persistence_report,
            })
            if args.save_arrays is not None:
                np.save(args.save_arrays / f"{sample_index:04d}_real.npy", real.astype("float32"))
                np.save(args.save_arrays / f"{sample_index:04d}_generated.npy", generated.astype("float32"))

    if not reports:
        raise RuntimeError("no evaluable calcium future samples found")
    aggregate_paths = {
        "native_corr_matrix_mse": ("model", "native", "corr_matrix_mse"),
        "native_autocorr_mse": ("model", "native", "autocorr", "mse"),
        "native_variance_log_rmse": ("model", "native", "variance_ratio", "log_rmse"),
        "raw_std_ratio": ("raw_std_ratio",),
        "raw_std_ratio_median": ("raw_std_ratio_median",),
        "generated_tail_std_ratio": ("generated_tail_std_ratio",),
        "persistence_native_corr_matrix_mse": ("persistence", "native", "corr_matrix_mse"),
    }
    if args.no_tderica_fit:
        aggregate_paths.update({
            "tderica_dtw": ("model", "tderica", "time_alignment", "dtw_distance"),
            "tderica_frechet": ("model", "tderica", "time_alignment", "frechet_distance"),
        })
    else:
        aggregate_paths.update({
            "tderica_dtw": ("model", "similarity", "time_alignment", "dtw_distance"),
            "tderica_frechet": ("model", "similarity", "time_alignment", "frechet_distance"),
            "real_occurrence_lag1": ("model", "real_occurrence_lag1"),
            "generated_occurrence_lag1": ("model", "generated_occurrence_lag1"),
        })
    output = {
        "checkpoint": str(args.checkpoint), "split": args.split,
        "device": str(device), "rollout_windows": args.rollout,
        "n_samples": len(reports), "tderica_context_fit": not args.no_tderica_fit,
        "checkpoint_info": checkpoint_info,
        "aggregate": {name: _aggregate_scalar(reports, path) for name, path in aggregate_paths.items()},
        "samples": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=_json_default) + "\n")
    print(json.dumps({"output": str(args.output), "n_samples": len(reports),
                      "device": str(device), "aggregate": output["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
