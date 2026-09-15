#!/usr/bin/env python3
"""Measure whether a checkpoint's control conditioning moves the free run.

Free-run evaluation answers "does the model reproduce the recording". This
tool answers the intervention question that follows: given the *same* context,
does the recorded drive move the model the way the recorded future moved?

For every row it rolls the latent system out three times from one context:

1. with the recorded control drive,
2. with an explicit all-zero drive,
3. with no drive argument at all (``perturbation=None``).

The recorded-versus-zero difference is the model's intervention effect. The
real effect is the held-out future minus the context's own persistence
baseline, so "the drive pushed the signal away from where it already was" is
compared against "the model's drive pushed the latent the same way". Reported
per sample and aggregated per ladder source:

* ``direction_cosine`` - sign agreement between the model and real effects,
* ``gain`` - magnitude ratio of the model effect to the real effect,
* ``channel_pattern_corr`` - whether the same channels respond,
* ``targeted_ratio`` - Randi rows only, mean ``|effect|`` on channels whose
  target gate is active over channels whose gate is not,
* ``zero_drive_max_abs_diff`` - the documented "zero stimulus is an exact
  no-op" invariant, measured as the difference between rollout 2 and 3.

Rows without a control track (the HuggingFace arm) carry no intervention
contrast; they are counted and used for the zero-drive invariant only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from brain_moe_pinn import BrainMoEPINNConfig
from brain_moe_pinn.training.training_loop import (
    pad_control_features, partition_generic_batch,
)

# Shared checkpoint/device/metric plumbing lives in the free-run evaluator;
# importing it keeps one definition of those contracts.
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from brain_moe_pinn.tools.remote_free_run_eval import (  # noqa: E402
    _as_per_sample_dt, _device, _first, _load_checkpoint, _source_label,
)


def _rollout(model, current, mask, perturbation, *, steps, dt, frame_dt):
    """One deterministic free-run rollout; returns ``(K, C, T)``."""
    out = model.forward_modalities(
        {"calcium": current}, masks={"calcium": mask},
        perturbation=perturbation, num_steps=steps, return_all=True,
        return_sequences=True, reconstruct=True, recon_max_channels=2048,
        dt=dt, frame_dt=frame_dt, step_dt=dt)
    sequence = out.get("calcium_recon_sequence")
    if sequence is None:
        raise RuntimeError("model did not emit calcium_recon_sequence")
    return sequence[0]


def _persistence(current: torch.Tensor, steps: int) -> torch.Tensor:
    """Repeat the context's last frame over every horizon (K, C, T)."""
    last = current[0][:, -1:]
    return last.expand(-1, current.shape[-1] * steps).reshape(
        current.shape[1], steps, current.shape[-1]).permute(1, 0, 2)


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def _targeted_split(perturbation: torch.Tensor, channels: int):
    """Channel mask of the drive's target gates.

    The unified control contract stores feature 0 as the scalar drive and
    features 1..N as recording-local target gates, so gate ``k`` addresses
    calcium channel ``k - 1``. Returns ``None`` for a scalar-only drive.
    """
    if perturbation.shape[-1] <= 1:
        return None
    gates = (perturbation[0, :, 1:].detach().cpu().numpy() != 0).any(axis=0)
    width = min(channels, gates.size)
    if width == 0 or not gates[:width].any():
        return None
    targeted = np.zeros(channels, dtype=bool)
    targeted[:width] = gates[:width]
    return targeted


def _sample_metrics(model, batch, *, input_modalities, control_modalities,
                    control_reduction, rollout_steps, device) -> Dict[str, Any]:
    drive_width = getattr(getattr(model, "velocity_brain", None),
                          "perturbation_dim", None)
    signals, targets, perturbation = partition_generic_batch(
        batch, signal_modalities=input_modalities,
        recon_modalities=("calcium",), device=device,
        control_modalities=control_modalities,
        control_reduction=control_reduction, rollout_steps=rollout_steps,
        require_future_targets=True, require_multi_horizon=True,
        control_dim=drive_width)
    current = signals["calcium"]
    future = targets["calcium"]
    if current.shape[0] != 1 or future.shape[0] != 1:
        raise ValueError("intervention evaluator requires batch_size=1")
    if future.shape[1] != rollout_steps:
        raise ValueError(
            f"future target has {future.shape[1]} horizons, "
            f"requested {rollout_steps}")
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

    # partition_generic_batch already padded the drive to the model width;
    # guard the contract here instead of silently rolling out a short drive.
    if perturbation is not None and drive_width is not None and \
            perturbation.shape[-1] != int(drive_width):
        perturbation = pad_control_features(perturbation, drive_width)
    roll_kwargs = dict(steps=rollout_steps, dt=window_dt, frame_dt=dt_frame)
    recorded = _rollout(model, current, context_mask, perturbation, **roll_kwargs)
    zero_drive = None if perturbation is None else torch.zeros_like(perturbation)
    without = _rollout(model, current, context_mask, None, **roll_kwargs)
    effect_zeroed = _rollout(model, current, context_mask, zero_drive, **roll_kwargs)

    report: Dict[str, Any] = {
        "sample_id": str(_first(batch.get("sample_id"), "?")),
        "origin": str(_first(batch.get("origin"), "")),
        "source": _source_label(str(_first(batch.get("sample_id"), "?")),
                                str(_first(batch.get("origin"), ""))),
        "channels": int(current.shape[1]),
        "frames_per_horizon": int(current.shape[-1]),
        "frame_dt_s": float(dt_frame.item()),
        "drive_available": perturbation is not None,
        "zero_drive_max_abs_diff": float(
            (without - effect_zeroed).abs().max().item()),
    }
    if perturbation is None:
        report["drive_norm"] = 0.0
        report["drive_active"] = False
        return report

    drive_np = perturbation.detach().float().cpu().numpy()
    report["drive_norm"] = float(np.linalg.norm(drive_np))
    report["drive_active"] = bool(np.any(drive_np != 0))
    if not report["drive_active"]:
        return report

    valid = future_mask[0].all(dim=0).all(dim=-1)
    if int(valid.sum()) < 2:
        report["skipped"] = True
        return report
    valid_np = valid.detach().cpu().numpy()

    model_effect = (recorded - effect_zeroed)[:, valid, :].detach().float().cpu().numpy()
    real_effect = (
        future[0] - _persistence(current, rollout_steps)
    )[:, valid, :].detach().float().cpu().numpy()
    if model_effect.shape != real_effect.shape:
        raise RuntimeError(
            f"effect shapes differ: {model_effect.shape} vs {real_effect.shape}")
    report["model_effect_norm"] = float(np.linalg.norm(model_effect))
    report["real_effect_norm"] = float(np.linalg.norm(real_effect))
    report["direction_cosine"] = _cosine(model_effect, real_effect)
    report["gain"] = (
        float(np.linalg.norm(model_effect) / np.linalg.norm(real_effect))
        if float(np.linalg.norm(real_effect)) > 1e-12 else None)
    report["channel_pattern_corr"] = _pearson(
        np.abs(model_effect).mean(axis=(0, 2)),
        np.abs(real_effect).mean(axis=(0, 2)))

    horizon_cosine, horizon_gain = [], []
    for k in range(rollout_steps):
        model_k = model_effect[k]
        real_k = real_effect[k]
        horizon_cosine.append(_cosine(model_k, real_k))
        denominator = float(np.linalg.norm(real_k))
        horizon_gain.append(
            float(np.linalg.norm(model_k) / denominator)
            if denominator > 1e-12 else None)
    report["horizon_direction_cosine"] = horizon_cosine
    report["horizon_gain"] = horizon_gain

    # The gate-to-channel map lives in the recording's own channel space, so
    # build the mask there and then compact it with the validity mask.
    targeted = _targeted_split(perturbation, int(current.shape[1]))
    if targeted is not None:
        targeted_mask = targeted[valid_np]
        non_targeted = ~targeted_mask
        model_channels = np.abs(model_effect).mean(axis=(0, 2))
        real_channels = np.abs(real_effect).mean(axis=(0, 2))
        report["targeted_channels"] = int(targeted_mask.sum())
        if targeted_mask.any() and non_targeted.any():
            model_targeted = float(model_channels[targeted_mask].mean())
            model_other = float(model_channels[non_targeted].mean())
            real_targeted = float(real_channels[targeted_mask].mean())
            real_other = float(real_channels[non_targeted].mean())
            report["targeted_effect_mean"] = model_targeted
            report["nontarget_effect_mean"] = model_other
            report["targeted_ratio"] = (
                model_targeted / model_other if model_other > 1e-12 else None)
            report["real_targeted_effect_mean"] = real_targeted
            report["real_nontarget_effect_mean"] = real_other
            report["real_targeted_ratio"] = (
                real_targeted / real_other if real_other > 1e-12 else None)
    return report


SCALAR_FIELDS = (
    "drive_norm", "model_effect_norm", "real_effect_norm", "direction_cosine",
    "gain", "channel_pattern_corr", "targeted_ratio", "real_targeted_ratio",
    "targeted_effect_mean", "nontarget_effect_mean", "real_targeted_effect_mean",
    "real_nontarget_effect_mean", "zero_drive_max_abs_diff",
)


def _summarise(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    summary: Dict[str, Any] = {
        "n": len(rows),
        "n_drive_available": sum(1 for r in rows if r.get("drive_available")),
        "n_drive_active": sum(1 for r in rows if r.get("drive_active")),
        "n_contrast": sum(1 for r in rows if "direction_cosine" in r),
        "n_targeted": sum(1 for r in rows if "targeted_ratio" in r),
    }
    for field in SCALAR_FIELDS:
        values = [float(r[field]) for r in rows
                  if isinstance(r.get(field), (int, float))
                  and math.isfinite(float(r[field]))]
        summary[field] = {
            "n": len(values),
            "mean": float(statistics.fmean(values)) if values else None,
            "median": float(statistics.median(values)) if values else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--rollout", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--save-arrays", type=Path, default=None)
    args = parser.parse_args()
    if args.max_samples < 1 or args.rollout < 1:
        raise ValueError("max-samples and rollout must be positive")

    device = _device(args.device)
    experiment = BrainMoEPINNConfig.from_file(str(args.config))
    model = experiment.to_model().to(device).eval()
    checkpoint_info = _load_checkpoint(
        model, args.checkpoint, device, config_path=args.config)

    from train import build_data_loaders
    loaders = build_data_loaders(args.data)
    loader = loaders[1] if args.split == "val" else (
        loaders[2] if len(loaders) > 2 else None)
    if loader is None:
        raise ValueError(f"requested split {args.split!r} is unavailable")

    profile = json.loads(Path(args.data).read_text())
    control_modalities = tuple(
        profile.get("control_modalities")
        or experiment.data.control_specs
        or ("stimulus",))
    control_reduction = str(profile.get("control_reduction", "resample"))

    rows: List[Dict[str, Any]] = []
    if args.save_arrays is not None:
        args.save_arrays.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.max_samples:
                break
            if "calcium" not in batch or "calcium_future" not in batch:
                continue
            report = _sample_metrics(
                model, batch, input_modalities=("calcium",),
                control_modalities=control_modalities,
                control_reduction=control_reduction,
                rollout_steps=args.rollout, device=device)
            report["index"] = index
            if report.get("skipped"):
                print(f"[skip] sample {index}: <2 valid future channels")
                continue
            rows.append(report)
    if not rows:
        raise RuntimeError("no evaluable samples found")

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for report in rows:
        by_source.setdefault(str(report["source"]), []).append(report)
    output = {
        "checkpoint": str(args.checkpoint),
        "data_profile": str(args.data),
        "split": args.split,
        "rollout_windows": args.rollout,
        "device": str(device),
        "checkpoint_info": checkpoint_info,
        "zero_drive_invariant_max_abs_diff": max(
            float(r.get("zero_drive_max_abs_diff", 0.0)) for r in rows),
        "aggregate": _summarise(rows),
        "aggregate_by_source": {
            source: _summarise(group)
            for source, group in sorted(by_source.items())
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, default=_json_default) + "\n")
    print(json.dumps({
        "output": str(args.output), "n_samples": len(rows),
        "zero_drive_invariant_max_abs_diff": output[
            "zero_drive_invariant_max_abs_diff"],
        "aggregate": output["aggregate"]}, indent=2))


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    main()
