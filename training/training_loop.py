"""
Training loop for Brain MoE-PINN with ZeRO Stage 2.

Supports:
- Multi-stage training (Phase -1 through Stage 3)
- DeepSpeed ZeRO Stage 2 for gradient sharding
- Mixed precision (FP16 with selective FP32)
- Gradient accumulation
- Logging and checkpointing
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Mapping
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..runtime.device_utils import (
    get_device, get_device_type, move_to_device,
    autocast_context, grad_scaler, set_device as set_accelerator_device,
)

try:
    import deepspeed
    from deepspeed import DeepSpeedConfig
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

try:
    from brain_moe_pinn.physics.thermodynamics import EnergyChangeMonitor
    ENERGY_MONITOR_AVAILABLE = True
except ImportError:
    ENERGY_MONITOR_AVAILABLE = False
    EnergyChangeMonitor = None

try:
    from brain_moe_pinn.core.velocity_brain import WienerHomeostat
    HOMEOSTAT_AVAILABLE = True
except ImportError:
    HOMEOSTAT_AVAILABLE = False
    WienerHomeostat = None

from .stability import StabilityController, AutoRollback, QFactorMonitor, AtaxiaCatalepsyMonitor
from .losses import TotalLoss
from .training_phases import LossWeights, evaluate_transition_gate

try:
    from brain_moe_pinn.config import SUPPORTED_MODALITIES, SPECIES_PROFILES
except ImportError:  # pragma: no cover - config is always importable in-repo
    SUPPORTED_MODALITIES = ()
    SPECIES_PROFILES = {}


def make_dummy_generic_signals(
    modalities,
    batch_size: int,
    channels: int = 64,
    time_len: int = 256,
    device=None,
) -> Dict[str, torch.Tensor]:
    """Random signals ``{modality: (B, C, T)}`` for scaffold species batches.

    Channel count and time length are scaffold defaults; real species
    dataloaders replace this method's output (same dict contract).
    """
    if not modalities:
        raise ValueError("at least one modality is required")
    signals = {}
    for modality in modalities:
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"unsupported modality {modality!r}")
        signals[modality] = torch.randn(
            batch_size, channels, time_len, device=device)
    return signals


# torch_autocast dtype per --precision choice.  Only the dtype differs, so
# both precisions are served by one config plus a parameter.
_PRECISION_DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def apply_precision(config: Dict, precision: Optional[str]) -> Dict:
    """Select the DeepSpeed autocast dtype.

    fp16 and bf16 differ in exactly one field, so the choice is a parameter
    rather than three more near-identical config files that would drift apart.

    Native ``fp16``/``bf16`` blocks are rejected rather than translated: they
    cast module weights to half *and* disable autocast inside
    ``engine.forward()``, which this model cannot use -- its physics
    differentiates with ``create_graph=True`` and holds fp32 structural
    constants.  Only ``torch_autocast`` keeps weights in fp32.

    "bf16" is the right choice wherever fp16's 65504 ceiling or its 5-bit
    exponent is a problem, and on TPU/XLA, which has no fp16 compute path.

    Args:
        config: the loaded ds_config JSON (not mutated)
        precision: None to keep the config's own dtype, else "fp16"/"bf16"
    Returns:
        a copy with the requested precision applied
    """
    cfg = dict(config)
    if precision is None:
        return cfg
    if precision not in _PRECISION_DTYPES:
        raise ValueError(
            f"unknown precision {precision!r}; expected one of "
            f"{sorted(_PRECISION_DTYPES)}")

    autocast = cfg.get("torch_autocast")
    if not isinstance(autocast, dict):
        raise ValueError(
            f"precision {precision!r} requires a 'torch_autocast' block in the "
            "DeepSpeed config; native fp16/bf16 casting is incompatible with "
            "this model (it disables autocast inside engine.forward() and "
            "casts weights to half, breaking the fp32 create_graph path)")

    cfg["torch_autocast"] = dict(autocast, enabled=True,
                                 dtype=_PRECISION_DTYPES[precision])
    # A native block alongside torch_autocast would still cast the weights,
    # so drop them rather than leave a contradictory pair of settings.
    cfg.pop("fp16", None)
    cfg.pop("bf16", None)
    return cfg


def rectify_deepspeed_config(
    config: Dict,
    micro_batch: Optional[int] = None,
    gradient_accumulation: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Dict:
    """Make the DeepSpeed batch arithmetic self-consistent.

    DeepSpeed asserts

        train_batch_size == train_micro_batch_size_per_gpu
                            * gradient_accumulation_steps * world_size

    and the shipped configs satisfied it for none of the launch topologies:
    ``ds_config_zero2.json`` (micro 1, accum 16, train_batch_size 16) is only
    valid at world_size 1, and the two multinode configs (micro 1, accum 4,
    train_batch_size 64) are only valid at exactly 16 GPUs — while both
    launchers default to ``NNODES=1, GPUS_PER_NODE=4`` (world 4).  DeepSpeed
    raises at ``initialize`` in that case.

    Independently, every phase declares ``batch_size: 16`` for the DataLoader
    while the config claimed a micro batch of 1.  DeepSpeed never inspects the
    tensor, so that mismatch is silent rather than fatal: it mis-scales the
    effective batch and silently ignores the phase's own accumulation, because
    the DeepSpeed branch calls ``model.backward()`` every step and lets the
    engine accumulate.

    This rebuilds those three fields from values that are actually known.

    Args:
        config: the loaded ds_config JSON (not mutated)
        micro_batch: batch size the DataLoader actually yields
        gradient_accumulation: accumulation from the phase config
        world_size: number of data-parallel ranks
    Returns:
        a corrected copy of the config
    """
    cfg = dict(config)
    # Not DeepSpeed engine keys; they only produce unknown-key warnings.
    cfg.pop("checkpoint", None)
    cfg.pop("sparse_checkpoint", None)
    # Let the derived value be authoritative instead of the stale literal.
    cfg.pop("train_batch_size", None)

    if micro_batch is not None:
        cfg["train_micro_batch_size_per_gpu"] = int(micro_batch)
    if gradient_accumulation is not None:
        cfg["gradient_accumulation_steps"] = int(gradient_accumulation)

    micro = int(cfg.get("train_micro_batch_size_per_gpu", 0) or 0)
    if micro <= 0:
        raise ValueError(
            "train_micro_batch_size_per_gpu must be positive; pass the "
            "DataLoader batch size to rectify_deepspeed_config")
    accum = int(cfg.get("gradient_accumulation_steps", 1) or 1)
    cfg["train_batch_size"] = micro * accum * max(1, int(world_size or 1))
    return cfg


def reduce_control(
    control: torch.Tensor,
    reduction: str = "resample",
    rollout_steps: int = 1,
) -> torch.Tensor:
    """Reduce windowed control ``(B, U, T)`` to per-step perturbation.

    - ``resample`` (default): ``rollout_steps`` segment means -> ``(B, K, U)``
      for per-latent-step conditioning.  With ``rollout_steps == 1`` this is
      a single window mean, identical to ``mean``.
    - ``peak``/``max``: segment maxima -> ``(B, K, U)``.  Use this for sparse
      optogenetic pulses when a segment mean would dilute the intervention.
    - ``mean``: one control vector per window, collapsing all within-window
      timing.  A salt step, a grating onset, or an event time is reduced to
      the window average, so the model can no longer see when the stimulus
      happened — keep this only for genuinely stationary controls.
    - ``last``: last frame in the window.
    """
    if control.dim() == 2:
        return control
    if reduction == "last":
        return control[..., -1]
    if reduction in {"peak", "max"}:
        steps = max(1, int(rollout_steps))
        pooled = torch.nn.functional.adaptive_max_pool1d(control, steps)
        return pooled.permute(0, 2, 1).contiguous()  # (B, K, U)
    if reduction == "resample":
        steps = max(1, int(rollout_steps))
        pooled = torch.nn.functional.adaptive_avg_pool1d(control, steps)
        return pooled.permute(0, 2, 1).contiguous()  # (B, K, U)
    return control.mean(dim=-1)


def _resize_temporal(
    value: torch.Tensor, length: int, *, is_mask: bool = False
) -> torch.Tensor:
    """Match a ``(B,C,T)`` target to a decoder time grid."""
    if value.dim() != 3:
        raise ValueError("temporal targets must have shape (B, C, T)")
    if value.shape[-1] == length:
        return value
    if value.shape[-1] > length:
        resized = torch.nn.functional.adaptive_avg_pool1d(
            value.float(), length)
        return resized.bool() if is_mask else resized
    padded = torch.nn.functional.pad(value, (0, length - value.shape[-1]))
    return padded.bool() if is_mask else padded


def masked_channel_correlation(
    prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    """Mean per-channel Pearson correlation over valid frames.

    Same estimator the correlation reconstruction loss uses, so a trivial
    baseline and a model score are directly comparable.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    if prediction.dim() == 3:
        pred, tgt = prediction, target
    elif prediction.dim() == 2:
        pred, tgt = prediction.unsqueeze(0), target.unsqueeze(0)
    else:
        raise ValueError("expected (B, C, T) or (C, T) tensors")
    valid = (torch.ones_like(pred, dtype=torch.bool) if mask is None
             else mask.bool())
    if valid.dim() == 2:
        valid = valid.unsqueeze(0)
    valid = valid.reshape(pred.shape)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    pred_c = pred - (pred * valid).sum(dim=-1, keepdim=True) / count
    tgt_c = tgt - (tgt * valid).sum(dim=-1, keepdim=True) / count
    num = (pred_c * tgt_c * valid).sum(dim=-1)
    den = (((pred_c ** 2) * valid).sum(dim=-1)
           * ((tgt_c ** 2) * valid).sum(dim=-1)).clamp_min(1e-12).sqrt()
    rho = (num / den)[count.squeeze(-1) >= 2]
    return float(rho.detach().mean()) if rho.numel() else 0.0


def partition_generic_batch(
    batch,
    signal_modalities,
    recon_modalities,
    device,
    control_modalities=(),
    control_reduction: str = "resample",
    rollout_steps: int = 1,
    require_future_targets: bool = False,
    require_multi_horizon: bool = False,
):
    """Split a loader batch into current signals, future targets, and control."""
    moved = {
        key: move_to_device(value, device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    control_modalities = tuple(control_modalities)
    signals = {
        modality: moved[modality]
        for modality in signal_modalities
        if modality in moved and modality not in control_modalities
    }
    targets = {}
    expected_horizons = max(1, int(rollout_steps))
    for modality in recon_modalities:
        if modality not in signals:
            continue
        future_key = f"{modality}_future"
        if future_key in moved:
            future = moved[future_key]
            if future.dim() != 4:
                raise ValueError(
                    f"{future_key} must have shape (B,K,C,T)")
            if future.shape[1] != max(1, int(rollout_steps)):
                raise ValueError(
                    f"{future_key} provides {future.shape[1]} horizons but "
                    f"rollout_steps={rollout_steps}")
            targets[modality] = future
            future_mask_key = f"{modality}_future_mask"
            if future_mask_key in moved:
                targets[f"{modality}_mask"] = moved[future_mask_key]
            continue
        horizon_keys = [
            f"{modality}_next_{horizon}"
            for horizon in range(1, max(1, int(rollout_steps)) + 1)
        ]
        if len(horizon_keys) > 1 and all(key in moved for key in horizon_keys):
            targets[modality] = torch.stack(
                [moved[key] for key in horizon_keys], dim=1)
            masks = [
                moved.get(f"{key}_mask") for key in horizon_keys]
            if all(mask is not None for mask in masks):
                targets[f"{modality}_mask"] = torch.stack(masks, dim=1)
            continue
        future_key = f"{modality}_next"
        alternate_future_key = f"next_{modality}"
        if future_key in moved:
            targets[modality] = moved[future_key]
            future_mask_key = f"{modality}_next_mask"
            alternate_mask_key = f"next_{modality}_mask"
            if future_mask_key in moved:
                targets[f"{modality}_mask"] = moved[future_mask_key]
            elif alternate_mask_key in moved:
                targets[f"{modality}_mask"] = moved[alternate_mask_key]
        elif alternate_future_key in moved:
            targets[modality] = moved[alternate_future_key]
            mask_key = f"{alternate_future_key}_mask"
            if mask_key in moved:
                targets[f"{modality}_mask"] = moved[mask_key]
        elif require_future_targets:
            raise ValueError(
                f"missing future target for modality {modality!r}; "
                "enable the dataset next-step target contract")
        else:
            targets[modality] = signals[modality]
            mask_key = f"{modality}_mask"
            if mask_key in moved:
                targets[mask_key] = moved[mask_key]
        if require_multi_horizon and expected_horizons > 1:
            target = targets.get(modality)
            if not isinstance(target, torch.Tensor) or target.dim() != 4:
                raise ValueError(
                    f"multi-horizon rollout requires {modality}_future or "
                    f"{expected_horizons} explicit horizon targets")
    for label_key in ("cross_modal_labels", "cross_modal_label"):
        if label_key in moved:
            targets["cross_modal_labels"] = moved[label_key]
            break
    for key in (
            "intervention_target",
            "intervention_mask",
            "action_utility_target",
            "replay_target"):
        if key in moved:
            targets[key] = moved[key]
    perturbation = None
    control_parts = [moved[m] for m in control_modalities if m in moved]
    if control_parts:
        concatenated = torch.cat(control_parts, dim=1)
        perturbation = reduce_control(
            concatenated, control_reduction, rollout_steps)
    return signals, targets, perturbation


def control_diagnostics(
    control: torch.Tensor, reduction: str = "resample", rollout_steps: int = 1
) -> Dict[str, float]:
    """Health metrics for a windowed control track ``(B, U, T)``.

    Two failure modes are common enough to be measured rather than assumed:

    * ``control_active_frac`` -- windows with no stimulus at all (a recording's
      pre-stimulus baseline is a legitimate sample, but the model cannot tell
      "no stimulus" from "a stimulus whose window mean is zero").
    * ``control_collapse_ratio`` -- ``|window mean| / within-window std``. With
      ``rollout_steps == 1`` the reduction is the window mean, so a *zero-mean
      alternating* drive (a salt cycle: +30 s / -30 s) collapses to ~0 and the
      model receives no control information at all. Near 1 means the surviving
      mean carries the signal; near 0 means the timing was discarded.
    """
    if not isinstance(control, torch.Tensor) or control.dim() != 3:
        return {}
    with torch.no_grad():
        per_sample_active = (control.abs().sum(dim=1) > 0).float().mean()
        std = control.std(dim=-1).clamp_min(1e-12)
        # What the model actually receives: the reduction's output.
        delivered = reduce_control(control, reduction, rollout_steps)
        # Magnitude of what each sample actually delivers, per segment: the
        # segments must NOT be averaged together, or an alternating drive whose
        # signs cancel would look identical to a genuinely lost drive.
        delivered = delivered.reshape(control.shape[0], -1)
        per_sample = delivered.abs().mean(dim=-1)
        collapse = per_sample / std.mean(dim=-1)
        return {
            "control_active_frac": float(per_sample_active),
            "control_collapse_ratio": float(collapse.mean()),
            "control_abs_mean": float(per_sample.mean()),
            "control_within_window_std": float(std.mean()),
        }


def augment_phase_loss_weights(
    weights,
    recon_modalities,
    recon_loss_types: Optional[Dict[str, str]] = None,
):
    """Enable supported generic terms without hiding unsupported objectives.

    Generic species models do not emit standard EEG/fMRI hubs or the latent
    HRF bridge, so cross-modal objectives are explicitly disabled rather than
    silently skipped.  Policy terms remain user-controlled and are validated
    per batch by ``TotalLoss``.
    """
    from dataclasses import replace
    extras = dict(getattr(weights, "recon_extra", {}) or {})
    types = dict(getattr(weights, "recon_loss_types", {}) or {})
    changed = False
    dedicated = {"eeg", "fmri", "meg"}
    for modality in recon_modalities:
        if modality in dedicated:
            continue
        if modality not in extras:
            extras[modality] = 1.0
            changed = True
    for modality, criterion in (recon_loss_types or {}).items():
        if (modality in recon_modalities and modality not in dedicated
                and modality not in types):
            types[modality] = criterion
            changed = True
    structural = {
        field: getattr(weights, field, 0.0)
        for field in ("cross_modal", "cross", "cross_soft")
    }
    if any(value > 0 for value in structural.values()):
        changed = True
        weights = replace(
            weights, cross_modal=0.0, cross=0.0, cross_soft=0.0)
    if not changed:
        return weights
    return replace(weights, recon_extra=extras, recon_loss_types=types)


class MetricsLogger:
    """Logs metrics to console and optionally to file."""

    def __init__(self, log_dir: str, log_interval: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.metrics_history = []

    def log(self, step: int, metrics: Dict[str, float], phase: str = ""):
        """Log metrics at given step."""
        entry = {
            "step": step,
            "phase": phase,
            "timestamp": time.time(),
            **metrics
        }
        self.metrics_history.append(entry)

        if step % self.log_interval == 0:
            loss_str = f"Step {step} [{phase}]"
            for k, v in metrics.items():
                if isinstance(v, float):
                    loss_str += f" | {k}: {v:.4f}"
                else:
                    loss_str += f" | {k}: {v}"
            print(loss_str)

    def save(self, filename: str = "metrics.json"):
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)


class EarlyStopping:
    """Early stopping based on validation metrics."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        metric: str = "total_loss",
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Check if should stop based on metric value."""
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def get_learning_rate_schedule(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
    schedule: str = "cosine",
) -> float:
    """
    Learning rate schedule.

    Args:
        step: current training step
        base_lr: peak learning rate after warmup
        warmup_steps: number of warmup steps
        total_steps: total steps for this phase
        min_lr: minimum learning rate
        schedule: "cosine" (decay from base_lr to min_lr) or "flat" (constant base_lr after warmup)
    Returns:
        current learning rate
    """
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)

    if schedule == "flat":
        return base_lr

    # Cosine decay
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return max(min_lr, min_lr + (base_lr - min_lr) * cosine_decay)


def setup_distributed():
    """Initialize training-process and rank metadata."""
    if not dist.is_initialized():
        device_type = get_device_type()
        if device_type == "cuda":
            backend = "nccl"
        elif device_type == "xpu":
            backend = "ccl"
        else:
            backend = "gloo"
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = int(os.environ.get("MASTER_PORT", "29500"))
            init_method = f"tcp://{master_addr}:{master_port}"
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            rank = 0
            world_size = 1
        return local_rank, rank, world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank, rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_global_rank_zero():
    """Check if this is the global rank 0 process."""
    return dist.is_initialized() and dist.get_rank() == 0


def global_barrier():
    """Synchronize all processes across all nodes."""
    if dist.is_initialized():
        dist.barrier()


class BrainMoETrainer:
    """
    Trainer for Brain MoE-PINN with multi-stage support.
    Supports DeepSpeed ZeRO Stage 2 and DDP fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        deepspeed_config: Optional[Dict] = None,
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        test_dataloader: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        # Opening steps produce the largest gradients of the run; PyTorch's
        # default 2**16 scale overflows fp16 there and burns skipped optimizer
        # steps.  See grad_scaler() for the reasoning.
        self.scaler_init_scale = config.get("scaler_init_scale")
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.local_rank, self.rank, self.world_size = setup_distributed()
        self.is_main_process = self.rank == 0

        self.device = get_device(index=self.local_rank)
        self.model = move_to_device(self.model, self.device)

        # Generic (non-EEG) species route: detect before DDP wrapping so the
        # flag survives module wrappers. Experiment data (modalities) comes
        # from train.py; profile defaults fill gaps.
        model_ref = model.module if hasattr(model, "module") else model
        self.generic_model = bool(
            getattr(model_ref, "generic_observation_only", False))
        species = (config.get("model_config") or {}).get("species") or "human"
        profile = SPECIES_PROFILES.get(species) if SPECIES_PROFILES else None
        experiment = config.get("experiment_data") or {}
        profile_modalities = profile.modalities if profile else ()
        modalities = tuple(
            experiment.get("modalities") or profile_modalities
        ) or ("calcium", "voltage")
        self.requested_modalities = tuple(modalities)
        self.species_modalities = tuple(
            m for m in self.requested_modalities
            if m in SUPPORTED_MODALITIES)
        declared_roles = (
            experiment.get("roles")
            or config.get("modality_roles")
            or {})
        invalid_roles = set(declared_roles.values()) - {
            "signal", "control", "aux", "graph"}
        if invalid_roles:
            raise ValueError(
                f"unknown modality roles: {sorted(invalid_roles)}")
        unknown_roles = set(declared_roles) - set(self.requested_modalities)
        if unknown_roles:
            raise ValueError(
                f"roles reference modalities not requested: "
                f"{sorted(unknown_roles)}")
        self.modality_roles = dict(declared_roles)
        unsupported_signals = {
            modality for modality in self.requested_modalities
            if modality not in SUPPORTED_MODALITIES
            and self.modality_roles.get(modality, "signal") == "signal"
        }
        if unsupported_signals:
            raise ValueError(
                f"unsupported signal modalities: {sorted(unsupported_signals)}")
        self.signal_modalities = tuple(
            m for m in self.species_modalities
            if self.modality_roles.get(m, "signal") == "signal")
        # 'behavior' is not auto-reconstructed (its contract may be
        # categorical); controls/auxiliary/graph modalities are not neural
        # reconstruction targets.
        self.recon_modalities = tuple(
            m for m in self.signal_modalities if m != "behavior")
        self.recon_loss_types = dict(
            (profile.recon_loss_types if profile else {}) or {})
        if config.get("recon_loss_types"):
            self.recon_loss_types.update(config["recon_loss_types"])
        self.require_future_targets = bool(
            experiment.get(
                "paired_next_step_targets",
                experiment.get("require_next_step_targets",
                            config.get("require_next_step_targets", False))))
        self._forecast_enabled = False
        self.future_steps = max(1, int(
            experiment.get("future_steps", config.get("future_steps", 1))))
        self.generic_channels = int(config.get("generic_channels", 64))
        self.generic_time = int(config.get("generic_time", 256))
        self.recon_max_channels = int(config.get("recon_max_channels", 2048))
        control_modalities = config.get("control_modalities") or ()
        if not control_modalities:
            control_modalities = tuple(
                m for m in self.species_modalities
                if self.modality_roles.get(m) == "control")
        if not control_modalities and "stimulus" in self.species_modalities:
            control_modalities = ("stimulus",)  # canonical control name
        self.control_modalities = tuple(control_modalities)
        self.control_reduction = str(
            config.get("control_reduction", "resample"))
        # A species/data profile may pin the rollout length (e.g. two 30 s
        # substeps inside a 60 s salt window); the phase table is the default.
        override = config.get("rollout_steps")
        self._rollout_steps_override = int(override) if override else None
        self._perturbation_warned = False
        self.model_use_meg = bool(getattr(model_ref, "use_meg", False))
        self._meg_recon_enabled = False
        if self.generic_model and self.is_main_process:
            print(f"[GenericSpecies] modalities={self.species_modalities} "
                  f"recon={self.recon_modalities} "
                  f"criteria={self.recon_loss_types}")

        self.ds_engine = None
        self.magi_pretraining = None
        if config.get("magi_pretraining_enabled", False):
            target_model = (
                self.model.module
                if hasattr(self.model, "module") else self.model)
            eeg_wrapper = getattr(target_model, "eeg_encoder", None)
            base_encoder = getattr(eeg_wrapper, "encoder", None)
            from ..magi.pretraining import MagiPretrainingObjective
            self.magi_pretraining = MagiPretrainingObjective(
                base_encoder,
                mask_ratio=float(config.get("magi_mask_ratio", 0.75)),
                momentum=float(config.get("magi_momentum", 0.999)),
                projection_dim=int(config.get("magi_projection_dim", 256)),
                temperature=float(config.get("magi_temperature", 0.2)),
            ).to(self.device)
            # Register objective heads/EMA state on the canonical model before
            # DDP/DeepSpeed initialization so their parameters are optimized
            # and saved with the production encoder checkpoint.
            target_model.magi_pretraining_objective = self.magi_pretraining

        self.ds_config = None
        self.optimizer = None
        self.scaler = None

        if deepspeed_config and DEEPSPEED_AVAILABLE:
            self.ds_config = deepspeed_config
            # DeepSpeed handles optimizer, scaler, and DDP internally
            self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=deepspeed_config,
            )
            self.scaler = None  # DeepSpeed manages its own mixed precision
        else:
            self.ds_config = None
            if self.world_size > 1 and dist.is_initialized():
                self.model = DDP(
                    self.model, device_ids=[self.local_rank],
                    find_unused_parameters=bool(
                        config.get("magi_pretraining_enabled", False)))
            self.optimizer = None
            self.scaler = grad_scaler(
                self.device, enabled=True,
                init_scale=self.scaler_init_scale)

        self.logger = MetricsLogger(log_dir) if self.is_main_process else None
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self._train_sampler = None
        self._val_sampler = None
        self._test_sampler = None
        if self.world_size > 1:
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler
            from torch.utils.data import RandomSampler

            def shard_loader(loader, shuffle):
                if loader is None:
                    return None, None
                if isinstance(loader.sampler, DistributedSampler):
                    return loader, loader.sampler
                if not isinstance(loader, DataLoader):
                    raise TypeError(
                        "distributed training requires torch DataLoader inputs")
                sampler = DistributedSampler(
                    loader.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=shuffle,
                )
                kwargs = {
                    "batch_size": loader.batch_size,
                    "sampler": sampler,
                    "drop_last": loader.drop_last,
                    "num_workers": loader.num_workers,
                    "collate_fn": loader.collate_fn,
                    "pin_memory": loader.pin_memory,
                    "persistent_workers": loader.persistent_workers,
                }
                if loader.num_workers > 0 and loader.prefetch_factor is not None:
                    kwargs["prefetch_factor"] = loader.prefetch_factor
                if getattr(loader, "pin_memory_device", ""):
                    kwargs["pin_memory_device"] = loader.pin_memory_device
                return DataLoader(loader.dataset, **kwargs), sampler

            train_shuffle = isinstance(
                getattr(train_dataloader, "sampler", None), RandomSampler)
            self.train_dataloader, self._train_sampler = shard_loader(
                train_dataloader, train_shuffle)
            self.val_dataloader, self._val_sampler = shard_loader(
                val_dataloader, False)
            self.test_dataloader, self._test_sampler = shard_loader(
                test_dataloader, False)
        self._epoch = 0
        self._dataloader_iter = None
        self._standard_targets: Dict[str, torch.Tensor] = {}
        self._standard_target_masks: Dict[str, torch.Tensor] = {}
        self._standard_signal_masks: Dict[str, torch.Tensor] = {}
        self._standard_contract_targets: Dict[str, torch.Tensor] = {}
        self._generic_batch_tensors: Dict[str, torch.Tensor] = {}
        self._generic_signal_masks: Dict[str, torch.Tensor] = {}

        self.stability = StabilityController(hidden_dim=1024, device=str(self.device))
        self.rollback = AutoRollback(checkpoint_dir=str(self.checkpoint_dir))
        self.rollback_trigger_count = 0

        self.q_factor_monitor = QFactorMonitor(window_size=512)
        self.ataxia_catalepsy_monitor = AtaxiaCatalepsyMonitor()
        if ENERGY_MONITOR_AVAILABLE and EnergyChangeMonitor is not None:
            self.energy_change_monitor = EnergyChangeMonitor(beta=1.0, window_size=100)
        else:
            self.energy_change_monitor = None

        if HOMEOSTAT_AVAILABLE and WienerHomeostat is not None:
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            inferred_dim = getattr(model_ref, "latent_dim", None)
            if inferred_dim is None and hasattr(model_ref, "velocity_brain"):
                inferred_dim = getattr(model_ref.velocity_brain, "hidden_dim", None)
            use_per_component = bool(config.get("homeostat_per_component", True))
            homeostat_dim = int(inferred_dim) if use_per_component and inferred_dim else None
            self.wiener_homeostat = WienerHomeostat(
                D_0=1.0, target_dissipative_proxy=1.0, target_entropy=1.0,
                dim=homeostat_dim,
            )
        else:
            self.wiener_homeostat = None

        # TotalLoss with physics constraints (lazily updated per phase)
        self.total_loss = TotalLoss(loss_weights=LossWeights())
        self._loss_weights_cache = None

        # Auto-create validation dataloader if data dirs provided
        if val_dataloader is None and train_dataloader is not None:
            val_ratio = config.get("validation_ratio", 0.05)
            if hasattr(train_dataloader.dataset, "split"):
                train_dataloader.dataset.split(val_ratio)
            self.val_dataloader = val_dataloader

        # Wire DataMixer for progressive task mixing (opt out with
        # config["use_mixer"] = False when real loaders are supplied).
        if (self.train_dataloader is not None
                and not hasattr(self.train_dataloader, "mixer")
                and config.get("use_mixer", True)):
            from ..data.data_loader import DataMixer
            try:
                self.train_dataloader.mixer = DataMixer(
                    resting_loader=self.train_dataloader,
                )
            except Exception:
                pass

        # Early stopping for validation
        self.early_stopper = None

    def broadcast_config(self, config: Dict) -> Dict:
        """Broadcast config dict from rank 0 to all nodes."""
        if self.world_size == 1:
            return config
        config_bytes = json.dumps(config).encode()
        if self.is_main_process:
            broadcast_tensor = torch.tensor([len(config_bytes)], dtype=torch.long, device=self.device)
        else:
            broadcast_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
        dist.broadcast(broadcast_tensor, src=0)
        config_len = broadcast_tensor.item()
        if self.is_main_process:
            config_tensor = torch.from_buffer(bytearray(config_bytes), dtype=torch.uint8).to(self.device)
        else:
            config_tensor = torch.zeros(config_len, dtype=torch.uint8, device=self.device)
        dist.broadcast(config_tensor, src=0)
        return json.loads(config_tensor.cpu().tobytes().decode())

    def global_barrier(self):
        """Synchronize all processes across all nodes."""
        if dist.is_initialized():
            dist.barrier()

    def create_optimizer(self, phase_config: Dict):
        """Create optimizer for current phase with proper parameter grouping."""
        lr = phase_config.get("learning_rate", 3e-5)
        optimizer_type = phase_config.get("optimizer", "adamw")
        weight_decay = phase_config.get("weight_decay", 0.01)
        grad_clip = phase_config.get("grad_clip", 1.0)

        # --- Parameter grouping ---
        # Adafactor factorization is unstable for 1D params (bias, LayerNorm,
        # embeddings), which should use standard AdamW even under Adafactor.
        # Also: encoder params get reduced LR (×0.1) to protect pretrained weights.
        adafactor_params = []
        no_factor_params = []
        encoder_adafactor = []
        encoder_no_factor = []
        adam_other = []
        adam_encoder = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_1d = param.ndim <= 1
            is_embed = "embed" in name or "token" in name
            is_norm = "norm" in name or "ln" in name or "layernorm" in name.lower()
            is_bias = "bias" in name
            no_factor = is_1d or is_embed or is_norm or is_bias

            is_encoder = "encoder" in name

            if optimizer_type.lower() == "adafactor":
                if no_factor:
                    (encoder_no_factor if is_encoder else no_factor_params).append(param)
                else:
                    (encoder_adafactor if is_encoder else adafactor_params).append(param)
            else:
                (adam_encoder if is_encoder else adam_other).append(param)

        param_groups = []
        if optimizer_type.lower() == "adafactor":
            if adafactor_params:
                param_groups.append({
                    "params": adafactor_params,
                    "lr": lr, "weight_decay": weight_decay,
                    "factorize_second_moments": True,
                })
            if no_factor_params:
                param_groups.append({
                    "params": no_factor_params,
                    "lr": lr, "weight_decay": weight_decay,
                    "factorize_second_moments": False,
                })
            if encoder_adafactor:
                param_groups.append({
                    "params": encoder_adafactor,
                    "lr": lr * 0.1, "weight_decay": weight_decay * 0.1,
                    "factorize_second_moments": True,
                })
            if encoder_no_factor:
                param_groups.append({
                    "params": encoder_no_factor,
                    "lr": lr * 0.1, "weight_decay": weight_decay * 0.1,
                    "factorize_second_moments": False,
                })
        else:
            if adam_other:
                param_groups.append({"params": adam_other, "lr": lr, "weight_decay": weight_decay})
            if adam_encoder:
                param_groups.append({"params": adam_encoder, "lr": lr * 0.1, "weight_decay": weight_decay * 0.1})

        if optimizer_type.lower() == "adafactor":
            try:
                from transformers.optimization import Adafactor
                self.optimizer = Adafactor(
                    param_groups,
                    lr=lr,
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False,
                )
                print("[Optimizer] Adafactor created (scale=False, rel_step=False)")
                for pg in self.optimizer.param_groups:
                    n = sum(p.numel() for p in pg["params"])
                    factor = pg.get("factorize_second_moments", True)
                    print(f"  group: {n/1e6:.1f}M params, factorize={factor}, lr={pg['lr']:.2e}")
            except ImportError:
                print("[Warning] Adafactor not available, using AdamW")
                self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            fused_available = hasattr(torch.optim.AdamW, "__init__") and hasattr(torch.optim.AdamW, "__doc__")
            try:
                self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, fused=True)
            except (TypeError, RuntimeError):
                self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        self.scaler = grad_scaler(
            self.device, enabled=True, init_scale=self.scaler_init_scale)
        self._grad_clip = grad_clip

    def transition_optimizer(
        self,
        new_optimizer_type: str = "adafactor",
        new_lr: float = 5e-4,
        freeze_warmup_steps: int = 200,
        weight_decay: float = 0.01,
        gradual_transition_steps: int = 1000,
    ):
        """
        Transition from current optimizer to a new one (e.g., AdamW -> Adafactor).

        During freeze-warmup, all existing parameters are frozen for
        `freeze_warmup_steps`. Only parameters tagged with
        `requires_grad=True` and `'adapter' in name` are initially unfrozen.

        Gradual LR transition: linearly interpolate LR from old to new over
        `gradual_transition_steps` to smooth the optimizer change. The old
        optimizer's momentum states are discarded; the new optimizer starts
        from scratch with cold slot variables.

        Args:
            new_optimizer_type: Target optimizer type ('adafactor', 'adamw')
            new_lr: Learning rate for new optimizer
            freeze_warmup_steps: Number of steps to freeze existing params
            weight_decay: Weight decay for new optimizer
            gradual_transition_steps: Steps for linear LR ramp from old_lr to new_lr
        """
        old_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else self.config.get("learning_rate", 3e-5)
        print(f"[OPTIMIZER] Transitioning {self.config.get('optimizer','adamw')} -> {new_optimizer_type}")
        print(f"[OPTIMIZER] LR: {old_lr} -> {new_lr} over {gradual_transition_steps} steps")
        print(f"[OPTIMIZER] Params frozen for {freeze_warmup_steps} steps")

        # Freeze all params initially
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Unfreeze adapter/new params immediately
        for name, param in self.model.named_parameters():
            if "adapter" in name or "projection" in name or "upscaling" in name:
                param.requires_grad = True

        # Create new optimizer with old LR (interpolate during training)
        phase_config = {
            "learning_rate": old_lr,
            "optimizer": new_optimizer_type,
            "weight_decay": weight_decay,
        }
        self.create_optimizer(phase_config)

        # Store transition state for gradual LR ramp
        self._transition_start_step = getattr(self, "_current_step", 0)
        self._transition_end_step = self._transition_start_step + gradual_transition_steps
        self._transition_new_lr = new_lr
        self._transition_old_lr = old_lr

        # Schedule: thaw all remaining params after warmup
        self._freeze_warmup_end_step = getattr(self, "_current_step", 0) + freeze_warmup_steps
        self._pending_thaw = True
        print(f"[OPTIMIZER] LR ramp {old_lr} -> {new_lr} over steps {self._transition_start_step} to {self._transition_end_step}")

    def _apply_pending_thaw(self, step: int):
        """Thaw frozen parameters after freeze-warmup period ends."""
        if getattr(self, "_pending_thaw", False) and step >= getattr(self, "_freeze_warmup_end_step", float("inf")):
            print(f"[OPTIMIZER] Thawing all params at step {step}")
            for param in self.model.parameters():
                param.requires_grad = True
            self._pending_thaw = False
            self._freeze_warmup_end_step = float("inf")

    def _convert_selective_fp32(self, state_dict: Dict) -> Dict:
        """Convert precision-sensitive modules to FP32 before saving."""
        fp32_modules = {"poisson_router", "kimi_delta", "kda", "router", "hebbian"}
        converted = {}
        for k, v in state_dict.items():
            key_lower = k.lower()
            if any(m in key_lower for m in fp32_modules):
                converted[k] = v.float()
            else:
                converted[k] = v
        return converted

    def save_checkpoint(self, step: int, phase: str, filename: Optional[str] = None):
        """Save world-size-aware checkpoint for elastic recovery."""
        if not self.is_main_process:
            return

        if filename is None:
            filename = f"checkpoint_{phase}_step{step}.pt"

        path = self.checkpoint_dir / filename

        if self.ds_engine is not None:
            state_dict = self._convert_selective_fp32(
                self.ds_engine.module.state_dict()
            )
        else:
            state_dict = self._convert_selective_fp32(
                self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
            )

        state = {
            "step": step,
            "phase": phase,
            "model_state": state_dict,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "global_rank": self.rank,
        }

        if self.optimizer is not None and self.ds_engine is None:
            state["optimizer_state"] = self.optimizer.state_dict()

        torch.save(state, path)
        if self.is_main_process:
            print(f"Checkpoint saved: {path} (world_size={self.world_size}, step={step})")

        # DeepSpeed engine checkpoint for full optimizer+lr scheduler state
        if self.ds_engine is not None:
            ds_path = self.checkpoint_dir / f"{filename.replace('.pt', '')}_ds"
            try:
                self.ds_engine.save_checkpoint(str(ds_path))
                if self.is_main_process:
                    print(f"DeepSpeed checkpoint saved: {ds_path}")
            except Exception as e:
                if self.is_main_process:
                    print(f"DeepSpeed checkpoint save failed (non-fatal): {e}")

    def load_checkpoint(self, path: str) -> Tuple[int, str]:
        """Load model checkpoint with world-size-aware handling."""
        checkpoint = torch.load(path, map_location=str(self.device))

        saved_world_size = checkpoint.get("world_size", self.world_size)
        if saved_world_size != self.world_size:
            if self.is_main_process:
                print(f"[Elastic] World size changed: {saved_world_size} → {self.world_size}. "
                      f"Checkpoint was saved on {saved_world_size} ranks, now running on {self.world_size}.")

        model = self.ds_engine if self.ds_engine is not None else self.model
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            ckpt = checkpoint["model_state"]
            model_dict = model.state_dict()
            converted = {}
            skipped = 0
            for k, v in ckpt.items():
                if k in model_dict:
                    if v.dtype != model_dict[k].dtype:
                        converted[k] = v.to(model_dict[k].dtype)
                    elif v.shape != model_dict[k].shape:
                        if self.is_main_process:
                            print(f"[Elastic] Skipping key {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
                        skipped += 1
                    else:
                        converted[k] = v
            if self.is_main_process and skipped > 0:
                print(f"[Elastic] Skipped {skipped} keys due to shape mismatch")
            model.load_state_dict(converted)

        if self.optimizer is not None and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint["step"], checkpoint["phase"]

    def _capture_standard_batch(self, batch: Any) -> None:
        """Store masks and explicit future/auxiliary targets."""
        self._standard_targets = {}
        self._standard_target_masks = {}
        self._standard_signal_masks = {}
        self._standard_contract_targets = {}
        self._standard_perturbation = None
        self._standard_baseline_perturbation = None
        if not isinstance(batch, dict):
            return
        for modality in ("eeg", "fmri", "meg"):
            current = batch.get(modality, batch.get(f"{modality}_data"))
            if isinstance(current, torch.Tensor):
                mask = batch.get(f"{modality}_mask")
                if isinstance(mask, torch.Tensor):
                    self._standard_signal_masks[modality] = move_to_device(
                        mask, self.device)
            future = batch.get(f"{modality}_future")
            future_mask = batch.get(f"{modality}_future_mask")
            if not isinstance(future, torch.Tensor):
                future = batch.get(
                    f"{modality}_next", batch.get(f"next_{modality}"))
                future_mask = batch.get(
                    f"{modality}_next_mask",
                    batch.get(f"next_{modality}_mask"))
            if isinstance(future, torch.Tensor):
                self._standard_targets[modality] = move_to_device(
                    future, self.device)
                if isinstance(future_mask, torch.Tensor):
                    self._standard_target_masks[modality] = move_to_device(
                        future_mask, self.device)

        for key in (
                "cross_modal_labels",
                "intervention_target",
                "intervention_baseline",
                "intervention_mask",
                "action_utility_target",
                "replay_target"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                self._standard_contract_targets[key] = move_to_device(
                    value, self.device)
        perturbation = batch.get("perturbation")
        baseline = batch.get(
            "baseline_perturbation", batch.get("intervention_baseline"))
        if isinstance(perturbation, torch.Tensor):
            self._standard_perturbation = move_to_device(
                perturbation, self.device)
        if isinstance(baseline, torch.Tensor):
            self._standard_baseline_perturbation = move_to_device(
                baseline, self.device)
    def _get_batch(self, batch_size: int, step: int = 0, total_steps: int = 1):
        """Get a standard batch and capture its supervision contract."""
        self._capture_standard_batch(None)

        def finish(batch):
            self._capture_standard_batch(batch)
            result = []
            for modality, default in (
                ("eeg", (19, 2560)),
                ("fmri", (400, 100)),
            ):
                value = batch.get(modality, batch.get(f"{modality}_data"))
                if not isinstance(value, torch.Tensor):
                    value = torch.randn(
                        batch_size, *default, device=self.device)
                else:
                    value = move_to_device(value, self.device)
                result.append(value)
            meg = batch.get("meg", batch.get("meg_data"))
            if isinstance(meg, torch.Tensor):
                result.append(move_to_device(meg, self.device))
            if self.require_future_targets:
                required = [
                    m for m in ("eeg", "fmri", "meg")
                    if m in self.species_modalities
                    and isinstance(batch.get(m, batch.get(f"{m}_data")),
                                   torch.Tensor)
                ]
                missing = [m for m in required if m not in self._standard_targets]
                if missing:
                    raise ValueError(
                        "missing future targets for standard modalities: "
                        + ", ".join(missing))
            return tuple(result)

        if self.train_dataloader is not None and hasattr(
                self.train_dataloader, "mixer"):
            batch = self.train_dataloader.mixer.next_batch(step, total_steps)
            if isinstance(batch, dict):
                return finish(batch)

        if self.train_dataloader is not None:
            if self._dataloader_iter is None:
                self._dataloader_iter = iter(self.train_dataloader)
            try:
                batch = next(self._dataloader_iter)
            except StopIteration:
                if self._train_sampler is not None:
                    self._epoch += 1
                    self._train_sampler.set_epoch(self._epoch)
                self._dataloader_iter = iter(self.train_dataloader)
                batch = next(self._dataloader_iter)
            except AttributeError:
                self._dataloader_iter = iter(self.train_dataloader)
                try:
                    batch = next(self._dataloader_iter)
                except StopIteration:
                    batch = {}
            if isinstance(batch, dict):
                return finish(batch)

        if self.require_future_targets:
            raise ValueError(
                "future targets are required but no paired training loader "
                "was supplied")
        return finish({})

    def _next_generic_signals(
        self, batch_size: int, step: int = 0, total_steps: int = 1
    ) -> Dict[str, torch.Tensor]:
        """Read a generic batch while retaining masks and future targets."""
        self._generic_batch_tensors = {}
        self._generic_signal_masks = {}
        if self.train_dataloader is not None:
            if self._dataloader_iter is None:
                self._dataloader_iter = iter(self.train_dataloader)
            try:
                batch = next(self._dataloader_iter)
            except StopIteration:
                if self._train_sampler is not None:
                    self._epoch += 1
                    self._train_sampler.set_epoch(self._epoch)
                self._dataloader_iter = iter(self.train_dataloader)
                batch = next(self._dataloader_iter)
            except AttributeError:
                self._dataloader_iter = iter(self.train_dataloader)
                try:
                    batch = next(self._dataloader_iter)
                except StopIteration:
                    batch = None
            if isinstance(batch, dict):
                batch_values = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_values[key] = move_to_device(value, self.device)
                    elif key in {"dt", "species", "sample_rate_hz", "rate_hz"}:
                        batch_values[key] = value
                tensors = {
                    key: value for key, value in batch_values.items()
                    if isinstance(value, torch.Tensor)
                }
                if tensors:
                    self._generic_batch_tensors = batch_values
                    self._generic_signal_masks = {
                        m: tensors[f"{m}_mask"]
                        for m in self.signal_modalities
                        if f"{m}_mask" in tensors
                    }
                    return {
                        m: tensors[m] for m in self.signal_modalities
                        if m in tensors
                    }
        signals = make_dummy_generic_signals(
            self.signal_modalities, batch_size,
            channels=self.generic_channels, time_len=self.generic_time,
            device=self.device,
        )
        self._generic_batch_tensors = signals
        return signals

    def _sanitize_perturbation(self, perturbation, model):
        """Drop control input when the model has no perturbation channel."""
        if perturbation is None:
            return None
        model_ref = model.module if hasattr(model, "module") else model
        has_channel = getattr(
            getattr(model_ref, "velocity_brain", None), "perturbation_dim", None)
        if has_channel is None:
            if not self._perturbation_warned:
                self._perturbation_warned = True
                print("[Control] perturbation provided but model has no "
                      "perturbation_dim; control input ignored")
            return None
        return perturbation

    def _forward_intervention_baseline(
        self,
        model,
        *,
        signals,
        dummy_eeg,
        dummy_fmri,
        dummy_meg,
        actual_eeg,
        actual_fmri,
        actual_meg,
        signal_masks,
        rollout_steps,
        perturbation,
        cross_modal_labels=None,
    ):
        """Run the matched baseline control for response supervision."""
        if self.generic_model:
            return model.forward_modalities(
                signals,
                masks=signal_masks,
                num_steps=rollout_steps,
                return_sequences=rollout_steps > 1,
                reconstruct=True,
                recon_max_channels=self.recon_max_channels,
                perturbation=perturbation,
                **self._generic_step_kwargs(self._generic_batch_tensors))
        return model(
            dummy_eeg,
            dummy_fmri,
            meg=dummy_meg,
            actual_eeg=actual_eeg,
            actual_fmri=actual_fmri,
            actual_meg=actual_meg,
            masks=signal_masks,
            action=None,
            perturbation=perturbation,
            mode=("forecast" if rollout_steps > 1 else "perception"),
            num_steps=rollout_steps,
            return_sequences=rollout_steps > 1,
            cross_modal_labels=cross_modal_labels,
        )

    @staticmethod
    def _attach_intervention_effect(
        outputs: Dict[str, torch.Tensor],
        baseline_outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> None:
        """Attach treated-minus-baseline latent response with strict shapes."""
        if target.dim() == 3:
            state_key = "z_next_sequence"
        elif target.dim() == 2:
            state_key = "z_next"
        else:
            raise ValueError(
                "intervention_target must have shape (B,D) or (B,K,D)")
        treated = outputs.get(state_key)
        baseline = baseline_outputs.get(state_key)
        if treated is None or baseline is None:
            raise ValueError(
                f"intervention response requires {state_key} from both "
                "treated and baseline rollouts")
        if treated.shape != baseline.shape or treated.shape != target.shape:
            raise ValueError(
                "intervention_target must match treated-minus-baseline "
                f"{state_key} shape")
        outputs["intervention_effect"] = treated - baseline
    def _enable_meg_recon(self) -> None:
        """Enable the recon_meg weight once real MEG batches are present.

        Keeps the default objective unchanged for non-MEG runs while making
        MEG supervision automatic when the data path provides it (phases can
        still override the weight by setting recon_meg explicitly).
        """
        from dataclasses import replace
        weights = self.total_loss.loss_weights
        if weights is None or getattr(weights, "recon_meg", 0.0) <= 0:
            base = LossWeights() if weights is None else weights
            if self.model_use_meg:
                self.total_loss.loss_weights = replace(
                    base, recon_meg=1.0)
                if self.is_main_process:
                    print("[MEG] MEG batches detected; recon_meg enabled (1.0)")
    def _prepare_reconstruction_targets(
        self,
        outputs: Dict[str, torch.Tensor],
        raw_targets: Dict[str, torch.Tensor],
        raw_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Align current or multi-horizon targets to decoder time grids."""
        raw_masks = raw_masks or {}
        targets = {}
        for modality, raw in raw_targets.items():
            prediction = outputs.get(f"{modality}_recon")
            sequence_prediction = outputs.get(
                f"{modality}_recon_sequence")
            if prediction is None:
                continue
            mask = raw_masks.get(modality)
            if raw.dim() == 4:
                if sequence_prediction is not None:
                    if sequence_prediction.dim() != 4:
                        raise ValueError(
                            f"{modality} forecast output must be rank 4")
                    if raw.shape[:3] != sequence_prediction.shape[:3]:
                        raise ValueError(
                            f"{modality} forecast target/output shapes do "
                            f"not match: {tuple(raw.shape)} vs "
                            f"{tuple(sequence_prediction.shape)}")
                    raw = torch.stack([
                        _resize_temporal(
                            raw[:, horizon],
                            sequence_prediction.shape[-1])
                        for horizon in range(raw.shape[1])
                    ], dim=1)
                    if mask is not None:
                        if mask.shape[:3] != sequence_prediction.shape[:3]:
                            raise ValueError(
                                f"{modality} forecast mask shape does not "
                                "match decoder sequence")
                        mask = torch.stack([
                            _resize_temporal(
                                mask[:, horizon],
                                sequence_prediction.shape[-1],
                                is_mask=True)
                            for horizon in range(mask.shape[1])
                        ], dim=1)
                    targets[modality] = raw
                    if mask is not None:
                        targets[f"{modality}_mask"] = mask
                    continue
                raw = raw[:, -1]
                if mask is not None:
                    mask = mask[:, -1]
            if raw.dim() != 3 or prediction.dim() != 3:
                raise ValueError(
                    f"{modality} reconstruction requires rank-3 tensors")
            if raw.shape[:2] != prediction.shape[:2]:
                raise ValueError(
                    f"{modality} target channels do not match decoder: "
                    f"{tuple(raw.shape)} vs {tuple(prediction.shape)}")
            targets[modality] = _resize_temporal(
                raw, prediction.shape[-1])
            if mask is not None:
                if mask.shape[:2] != prediction.shape[:2]:
                    raise ValueError(
                        f"{modality} target mask channels do not match decoder")
                targets[f"{modality}_mask"] = _resize_temporal(
                    mask, prediction.shape[-1], is_mask=True)
        for key in (
                "cross_modal_labels",
                "intervention_target",
                "intervention_baseline",
                "intervention_mask",
                "action_utility_target",
                "replay_target"):
            if key in raw_targets:
                targets[key] = raw_targets[key]
        return targets

    def _standard_raw_targets(
        self, eeg: torch.Tensor, fmri: torch.Tensor,
        meg: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return validated future targets or legacy current observations."""
        fallback = {"eeg": eeg, "fmri": fmri}
        if meg is not None:
            fallback["meg"] = meg
        targets = {}
        for modality, value in fallback.items():
            target = self._standard_targets.get(modality, value)
            if target.dim() == 4:
                expected = max(1, int(getattr(
                    self, "_rollout_steps", self.future_steps)))
                if target.shape[1] != expected:
                    raise ValueError(
                        f"{modality} future target has {target.shape[1]} "
                        f"horizons but rollout_steps={expected}")
            elif getattr(self, "_forecast_enabled", False):
                raise ValueError(
                    f"multi-horizon forecast requires {modality}_future with "
                    f"{getattr(self, '_rollout_steps', 1)} horizons")
            targets[modality] = target
        return targets, dict(self._standard_target_masks)

    @staticmethod
    def _batch_frame_dt(batch: Dict[str, Any]) -> Optional[float]:
        """Median sampling interval (seconds per frame) of a collated batch.

        Metrics that convert frame lags to seconds must use the recording's
        own interval; using a global `latent_dt` silently reports another
        species' (or another stage's) clock.
        """
        values = batch.get("dt") if isinstance(batch, dict) else None
        if isinstance(values, (list, tuple)) and values:
            finite = [float(v) for v in values
                      if v is not None and float(v) > 0]
            if finite:
                return float(sorted(finite)[len(finite) // 2])
        if isinstance(values, torch.Tensor) and values.numel():
            finite = values.detach().flatten().float()
            finite = finite[finite > 0]
            if finite.numel():
                return float(finite.median())
        return None

    def _generic_control_metrics(self, tensors: Optional[Dict[str, Any]] = None
                                 ) -> Dict[str, float]:
        """Control health for the live batch, with a one-time warning.

        A zero-mean alternating drive (salt +/- half-cycles) inside one window
        reduces to ~0 under the default window-mean reduction, so the control
        input silently disappears; say so once instead of training blind.
        """
        tensors = self._generic_batch_tensors if tensors is None else tensors
        metrics: Dict[str, float] = {}
        reduction = getattr(self, "control_reduction", "resample")
        for modality in getattr(self, "control_modalities", ()) or ():
            track = tensors.get(modality)
            if not isinstance(track, torch.Tensor) or track.dim() != 3:
                continue
            stats = control_diagnostics(track, reduction,
                                        getattr(self, "_rollout_steps", 1))
            metrics.update({f"{modality}_{k}": v for k, v in stats.items()})
            if (not getattr(self, "_control_warned", False)
                    and reduction == "resample"
                    and int(getattr(self, "_rollout_steps", 1)) == 1
                    and stats.get("control_active_frac", 0.0) > 0.0
                    and stats.get("control_collapse_ratio", 1.0) < 0.1):
                self._control_warned = True
                print(
                    f"[Control] '{modality}': with rollout_steps=1 the window "
                    f"mean is the only control the model sees "
                    f"(collapse ratio "
                    f"{stats.get('control_collapse_ratio', 0.0):.3f}, "
                    f"active {stats.get('control_active_frac', 0.0):.2f}). "
                    f"For an alternating drive use "
                    f"control_reduction=\"last\" or rollout_steps>1, "
                    f"otherwise the stimulus timing is discarded.")
        return metrics

    def _generic_step_kwargs(self, tensors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Physics/species kwargs for one generic forward from the live batch.

        One latent step advances one *window*, so its physical duration in
        seconds is ``frames_per_window * dt_row``: the sampling interval comes
        from the manifest per sample (worms are imaged at 3.69-5.72 Hz), not
        from a global constant. Passing it makes the MT-KDA/OU time constants
        mean seconds for whatever rate the batch carries, and the per-sample
        species tags keep replay batches from being conditioned as the stage
        species.
        """
        tensors = self._generic_batch_tensors if tensors is None else tensors
        kwargs: Dict[str, Any] = {}
        frame_counts = None
        for modality, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.dim() != 3:
                continue
            mask = tensors.get(f"{modality}_mask")
            if isinstance(mask, torch.Tensor) and mask.shape == value.shape:
                # Collapse channels first: frame_counts must be one value per
                # sample, not one value per channel.
                valid_frames = mask.detach().to(value.device).bool().any(dim=1)
                frame_counts = valid_frames.sum(dim=-1).clamp_min(1)
            else:
                frame_counts = torch.full(
                    (value.shape[0],), float(value.shape[-1]),
                    device=value.device)
            break
        dt_values = tensors.get("dt")
        if frame_counts is not None and dt_values is not None:
            try:
                if isinstance(dt_values, torch.Tensor):
                    per_sample = dt_values.detach().flatten().to(self.device).float()
                elif isinstance(dt_values, (list, tuple)):
                    per_sample = torch.tensor(
                        [float(v) for v in dt_values], device=self.device)
                else:
                    per_sample = torch.tensor(
                        [float(dt_values)], device=self.device)
            except (TypeError, ValueError):
                per_sample = None
            if per_sample is not None and per_sample.numel() == 1:
                per_sample = per_sample.expand(frame_counts.shape[0])
            if (per_sample is not None
                    and per_sample.numel() == frame_counts.numel()
                    and torch.isfinite(per_sample).all()
                    and bool((per_sample > 0).all())):
                kwargs["dt"] = per_sample * frame_counts.to(per_sample.dtype)
                kwargs["frame_dt"] = per_sample
                if int(getattr(self, "_rollout_steps", 1)) > 1:
                    kwargs["step_dt"] = kwargs["dt"]
        species_tags = tensors.get("species")
        if isinstance(species_tags, (list, tuple)) and species_tags:
            kwargs["species_names"] = [str(tag) for tag in species_tags]
        return kwargs

    @staticmethod
    def _validate_runtime_phase(phase_config: Mapping[str, Any]) -> None:
        """Reject legacy dataclass/partial phase objects at the boundary."""
        if not isinstance(phase_config, Mapping):
            raise TypeError(
                "BrainMoETrainer requires a normalized phase mapping; call "
                "TrainingPhase.to_runtime_config() first")
        required = {
            "name", "stage", "task", "total_steps", "learning_rate",
            "loss_weights", "freeze_policy", "rollout", "context",
            "transition_gate",
        }
        missing = sorted(required - set(phase_config))
        if missing:
            raise ValueError(
                "phase runtime mapping is missing required fields: "
                + ", ".join(missing))
        if not isinstance(phase_config["freeze_policy"], Mapping):
            raise TypeError("phase freeze_policy must be a mapping")
        if not isinstance(phase_config["rollout"], Mapping):
            raise TypeError("phase rollout settings must be a mapping")
        if not isinstance(phase_config["context"], Mapping):
            raise TypeError("phase context settings must be a mapping")
        if int(phase_config.get("gradient_accumulation", 1)) < 1:
            raise ValueError("phase gradient_accumulation must be positive")

        if phase_config["task"] not in {
                "model_training", "magi_eeg_pretraining"}:
            raise ValueError(
                f"unsupported phase task: {phase_config['task']!r}")

    def _apply_freeze_policy(self, policy: Mapping[str, Any]) -> None:
        """Apply the phase's explicit freeze policy before optimizer creation."""
        if not isinstance(policy, Mapping):
            raise TypeError("freeze policy must be a mapping")
        model = self.ds_engine if self.ds_engine is not None else self.model
        target = model.module if hasattr(model, "module") else model

        def set_frozen(component, frozen: bool) -> None:
            if component is None:
                return
            setter = getattr(component, "set_freeze", None)
            if callable(setter):
                setter(bool(frozen))
                return
            for parameter in component.parameters():
                parameter.requires_grad = not frozen

        components = {
            "eeg_encoder": getattr(target, "eeg_encoder", None),
            "fmri_encoder": getattr(target, "fmri_encoder", None),
            "decoder": getattr(
                target, "decoder", getattr(target, "decoder_router", None)),
        }
        for name, component in components.items():
            if name in policy:
                set_frozen(component, bool(policy[name]))
        router = getattr(getattr(target, "moe_velocity", None), "router", None)
        if "moe_router" in policy:
            set_frozen(router, bool(policy["moe_router"]))
        self._active_freeze_policy = dict(policy)

    def _apply_timed_freeze_policy(
        self,
        policy: Mapping[str, Any],
        elapsed_steps: int,
        steps_per_epoch: int,
    ) -> bool:
        """Thaw frozen components at their configured phase-local epoch."""
        model = self.ds_engine if self.ds_engine is not None else self.model
        target = model.module if hasattr(model, "module") else model
        components = {
            "eeg_encoder": getattr(target, "eeg_encoder", None),
            "fmri_encoder": getattr(target, "fmri_encoder", None),
        }
        changed = False
        thawed = getattr(self, "_phase_thawed_components", set())
        for name, component in components.items():
            if not policy.get(name, False) or name in thawed or component is None:
                continue
            epochs_key = (
                "eeg_epochs_thawed" if name == "eeg_encoder"
                else "fmri_epochs_thawed")
            thaw_step = max(0, int(policy.get(epochs_key, 1))) * max(
                1, int(steps_per_epoch))
            if elapsed_steps < thaw_step:
                continue
            setter = getattr(component, "set_freeze", None)
            if callable(setter):
                setter(False)
            else:
                for parameter in component.parameters():
                    parameter.requires_grad = True
            thawed.add(name)
            changed = True
        self._phase_thawed_components = thawed
        return changed

    def _validate_context_settings(self, phase_config: Mapping[str, Any]) -> None:
        schedule = phase_config.get("context_expansion_schedule")
        context = phase_config.get("context", {})
        if schedule is None:
            schedule = context.get("expansion_schedule")
        if schedule is None:
            return
        schedule = list(schedule)
        steps = []
        for entry in schedule:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("context expansion entries must be (step, length)")
            step, length = map(int, entry)
            if step < 0 or length < 1:
                raise ValueError("context expansion entries must be non-negative")
            steps.append(step)
        if steps != sorted(steps):
            raise ValueError("context expansion schedule must be ordered")
        model = self.ds_engine if self.ds_engine is not None else self.model
        target = model.module if hasattr(model, "module") else model
        if not callable(getattr(target, "set_context_length", None)):
            raise NotImplementedError(
                "phase requests context expansion, but the model does not "
                "implement set_context_length()")

    def _apply_phase_context(self, phase_config: Mapping[str, Any]) -> None:
        """Apply non-scheduled context declarations instead of ignoring them."""
        requested = phase_config.get("base_context")
        if requested is None:
            sequence_length = phase_config.get("max_seq_len_eeg")
            if sequence_length is not None and int(sequence_length) != 2560:
                requested = int(sequence_length)
        if requested is None:
            return
        model = self.ds_engine if self.ds_engine is not None else self.model
        target = model.module if hasattr(model, "module") else model
        setter = getattr(target, "set_context_length", None)
        if not callable(setter):
            raise NotImplementedError(
                f"phase '{phase_config['name']}' declares context length "
                f"{requested}, but the model does not implement "
                "set_context_length()")
        setter(int(requested))
        self._current_context_length = int(requested)

    def _train_magi_pretraining(
        self,
        phase_config: Mapping[str, Any],
        start_step: int = 0,
    ) -> int:
        """Run Magi self-supervision against the production EEG encoder."""
        if self.magi_pretraining is None:
            raise RuntimeError(
                "Magi pretraining was requested without an initialized Magi "
                "objective; use the v2 EEG backend")
        if self.train_dataloader is None:
            raise ValueError(
                "Magi EEG pretraining requires a real train dataloader")

        self.reset_loss_normalizer()
        self._validate_context_settings(phase_config)
        self._apply_freeze_policy(phase_config["freeze_policy"])
        total_steps = int(phase_config.get("total_steps", 10000))
        warmup_steps = int(phase_config.get("warmup_steps", 500))
        gradient_accumulation = int(
            phase_config.get("gradient_accumulation", 1))
        if gradient_accumulation != 1:
            raise ValueError(
                "Magi pretraining currently requires gradient_accumulation=1")

        use_deepspeed = self.ds_engine is not None
        if not use_deepspeed:
            self.create_optimizer(phase_config)
            self.optimizer.zero_grad(set_to_none=True)
        model = self.ds_engine if use_deepspeed else self.model
        model.train()
        self.magi_pretraining.train()
        iterator = iter(self.train_dataloader)
        step = int(start_step)
        phase_name = str(phase_config.get("name", "Magi EEG Pretraining"))
        weights = dict(phase_config.get("magi_objective_weights") or {})
        masked_weight = float(weights.get("masked", 1.0))
        ntp_weight = float(weights.get("causal_ntp", 1.0))
        contrastive_weight = float(weights.get("contrastive", 0.1))

        while step < total_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_dataloader)
                batch = next(iterator)
            if not isinstance(batch, Mapping):
                raise TypeError("Magi pretraining batches must be mappings")

            eeg = batch.get(
                "eeg",
                batch.get("ecog", batch.get("eeg_data")))
            if not isinstance(eeg, torch.Tensor):
                raise ValueError(
                    "Magi pretraining batches must contain tensor key "
                    "'eeg' or 'ecog'")
            eeg = move_to_device(eeg, self.device)
            channel_types = batch.get(
                "channel_types", batch.get("ecog_channel_types"))
            if isinstance(channel_types, torch.Tensor):
                channel_types = move_to_device(channel_types, self.device)
            channel_names = batch.get(
                "channel_names",
                batch.get("ecog_channel_names", batch.get("eeg_channel_names")))
            view1 = batch.get("eeg_view1", batch.get("ecog_view1"))
            view2 = batch.get("eeg_view2", batch.get("ecog_view2"))
            if isinstance(view1, torch.Tensor):
                view1 = move_to_device(view1, self.device)
            else:
                view1 = eeg
            if isinstance(view2, torch.Tensor):
                view2 = move_to_device(view2, self.device)
            if step < warmup_steps:
                base_lr = float(phase_config.get("learning_rate", 3e-5))
                lr = base_lr * step / max(1, warmup_steps)
            else:
                lr = get_learning_rate_schedule(
                    step,
                    float(phase_config.get("learning_rate", 3e-5)),
                    warmup_steps,
                    total_steps,
                    float(phase_config.get("min_lr", 1e-6)),
                    schedule=phase_config.get("lr_schedule", "cosine"),
                )
            optimizer = self.optimizer
            if optimizer is None:
                optimizer = getattr(model, "optimizer", None)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr

            with autocast_context(self.device, enabled=True):
                # Call through the wrapped model. DDP/DeepSpeed then observes
                # gradients for both the production Magi encoder and the
                # registered pretraining heads.
                result = model(
                    eeg=view1,
                    fmri=None,
                    channel_names=channel_names,
                    channel_types=channel_types,
                    magi_pretraining=True,
                    magi_eeg_view2=view2,
                    magi_objective_weights=weights,
                )
                loss = result["loss"]

            if use_deepspeed:
                model.backward(loss)
                model.step()
                self.magi_pretraining.update_momentum_encoder()
            else:
                old_scale = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(phase_config.get("grad_clip", 1.0)),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.get_scale() >= old_scale:
                    self.magi_pretraining.update_momentum_encoder()

            self._current_step = step
            if self.logger and step % 10 == 0:
                self.logger.log(
                    step,
                    {
                        "total_loss": float(result["loss"].detach()),
                        "masked_loss": float(result["masked_loss"].detach()),
                        "causal_ntp_loss": float(result["ntp_loss"].detach()),
                        "contrastive_loss": float(
                            result["contrastive_loss"].detach()),
                        "lr": lr,
                    },
                    phase_name,
                )
            step += 1

        self.save_checkpoint(step, phase_name)
        return step


    def evaluate_phase_transition_gate(
        self, phase_config: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> bool:
        """Evaluate a normalized phase gate with a deterministic result."""
        self._validate_runtime_phase(phase_config)
        return evaluate_transition_gate(phase_config["transition_gate"], metrics)

    def reset_loss_normalizer(self) -> None:
        """Reset loss-scale calibration at a phase boundary."""
        reset = getattr(self.total_loss, "reset_loss_normalizer", None)
        if callable(reset):
            reset()
        elif hasattr(getattr(self.total_loss, "normalizer", None), "reset"):
            self.total_loss.normalizer.reset()

    def train_phase(self, phase_config: Mapping[str, Any], start_step: int = 0):
        """Train one normalized runtime phase mapping."""
        self._validate_runtime_phase(phase_config)
        if phase_config["task"] == "magi_eeg_pretraining":
            return self._train_magi_pretraining(
                phase_config, start_step=start_step)
        phase_name = phase_config.get("name", "unknown")
        loss_weights = phase_config.get("loss_weights")
        forecast_requested = bool(
            getattr(loss_weights, "forecast", 0.0) > 0)
        self.reset_loss_normalizer()
        self._validate_context_settings(phase_config)
        rollout_steps = max(1, int(
            self._rollout_steps_override
            if getattr(self, "_rollout_steps_override", None)
            else phase_config.get("rollout_steps", 1)))
        self._forecast_enabled = forecast_requested and rollout_steps > 1
        if self._forecast_enabled and not self.require_future_targets:
            raise ValueError(
                f"phase {phase_name!r} enables forecast supervision but "
                "the data contract does not enable explicit future targets")
        self._apply_freeze_policy(phase_config["freeze_policy"])
        self._phase_thawed_components = set()
        total_steps = int(phase_config.get("total_steps", 10000))
        self._apply_phase_context(phase_config)
        warmup_steps = phase_config.get("warmup_steps", 500)
        batch_size = phase_config.get("batch_size", 16)
        if self.train_dataloader is not None and self.train_dataloader.batch_size:
            # The loader (built from the data + species profile) is
            # authoritative: a phase table's batch_size cannot describe a
            # species run whose profile chose another one, and silently
            # training at a different effective batch than the profile says is
            # exactly the failure this assert prevents.
            loader_batch = int(self.train_dataloader.batch_size)
            if int(batch_size) != loader_batch:
                print(f"[BATCH] phase '{phase_name}' declares batch_size="
                      f"{batch_size} but the loader yields {loader_batch}; "
                      f"using the loader (data/species profile is the source "
                      f"of truth)")
            batch_size = loader_batch
        gradient_accumulation = int(phase_config.get("gradient_accumulation", 1))
        if gradient_accumulation < 1:
            raise ValueError("phase gradient_accumulation must be positive")

        if self.ds_config is not None:
            # DeepSpeed fixes its micro batch at engine init and never re-reads
            # it, and the DeepSpeed branch calls model.backward() every step
            # instead of accumulating itself.  A phase whose batch differs from
            # what the engine was told therefore trains at a different
            # effective batch than the config claims, silently.
            configured_micro = int(
                self.ds_config.get("train_micro_batch_size_per_gpu", 0) or 0)
            configured_accum = int(
                self.ds_config.get("gradient_accumulation_steps", 1) or 1)
            if configured_micro and int(batch_size) != configured_micro:
                raise ValueError(
                    f"DeepSpeed was initialised with "
                    f"train_micro_batch_size_per_gpu={configured_micro} but the "
                    f"loader yields {batch_size} samples per step (phase "
                    f"'{phase_name}'). DeepSpeed never re-reads the micro batch, "
                    f"so the run would train at a different effective batch than "
                    f"this configuration states; align the data profile / "
                    f"DeepSpeed config (train.py rewrites the micro batch from "
                    f"the loader, so this means a hand-edited config).")
            if gradient_accumulation != configured_accum:
                raise ValueError(
                    f"DeepSpeed was initialised with gradient_accumulation_steps="
                    f"{configured_accum}, but phase '{phase_name}' requests "
                    f"{gradient_accumulation}. DeepSpeed cannot change "
                    "gradient accumulation after engine initialization; use "
                    "one accumulation value for all selected phases.")
        base_lr = phase_config.get("learning_rate", 3e-5)
        min_lr = phase_config.get("min_lr", 1e-6)

        # Update loss weights for this phase. Generic (non-EEG) runs get
        # their species reconstruction terms auto-enabled here without
        # editing the shared phase tables.
        phase_loss_weights = phase_config.get("loss_weights", None)
        if self.generic_model and self.recon_modalities:
            if phase_loss_weights is None:
                phase_loss_weights = LossWeights()
            phase_loss_weights = augment_phase_loss_weights(
                phase_loss_weights, self.recon_modalities,
                self.recon_loss_types)
        if phase_loss_weights is not None:
            self.total_loss.loss_weights = phase_loss_weights
        elif phase_name != "Magi EEG Encoder Pretraining":
            print(f"[WARNING] Phase '{phase_name}' has no loss_weights; using defaults")

        use_deepspeed = self.ds_engine is not None

        if not use_deepspeed and self.optimizer is None:
            self.create_optimizer(phase_config)
        elif not use_deepspeed and self.optimizer is not None:
            new_opt = phase_config.get("optimizer", "adamw")
            old_opt = getattr(self, "_current_optimizer_type", "adamw")
            if new_opt != old_opt:
                print(f"[OPTIMIZER] Detected optimizer change: {old_opt} -> {new_opt}")
                self.transition_optimizer(
                    new_optimizer_type=new_opt,
                    new_lr=phase_config.get("learning_rate", 5e-4),
                    freeze_warmup_steps=200,
                    gradual_transition_steps=1000,
                )
            else:
                self.create_optimizer(phase_config)
        self._current_optimizer_type = phase_config.get("optimizer", "adamw")

        model = self.ds_engine if use_deepspeed else self.model

        noise_mode = phase_config.get("noise_mode", None)
        if noise_mode is not None:
            tgt = model.module if hasattr(model, "module") else model
            if hasattr(tgt, "noise_mode"):
                tgt.noise_mode = str(noise_mode)
                if self.is_main_process:
                    print(f"[SDE] noise_mode set to {noise_mode}")

        router_tau = phase_config.get("router_tau", None)
        if router_tau is not None:
            tgt = model.module if hasattr(model, "module") else model
            if hasattr(tgt, "moe_velocity") and hasattr(tgt.moe_velocity, "set_router_tau"):
                tgt.moe_velocity.set_router_tau(router_tau)
                if self.is_main_process:
                    print(f"[ROUTER] tau set to {router_tau}")

        imagination_interval = phase_config.get("imagination_interval", 0)
        # Generic species runs exercise forward_modalities, which has no
        # imagination path; do not force modules that would never be used.
        if imagination_interval > 0 and not self.generic_model:
            tgt = model.module if hasattr(model, "module") else model
            if hasattr(tgt, "use_imagination") and not getattr(tgt, "use_imagination", False):
                tgt.use_imagination = True
                if tgt.imagination_sampler is None:
                    from brain_moe_pinn.core.counterfactual_search import (
                        CounterfactualTreeSearch, ImaginationSampler,
                    )
                    tgt.counterfactual_search = CounterfactualTreeSearch(
                        latent_dim=tgt.latent_dim, num_goals=2,
                        num_exploration=1, rollout_steps=3, branch_factor=4,
                    ).to(self.device)
                    tgt.imagination_sampler = ImaginationSampler(
                        latent_dim=tgt.latent_dim,
                    ).to(self.device)
                if self.is_main_process:
                    print(f"[IMAGINATION] Enabled with interval={imagination_interval}")

        model.train()

        if self._train_sampler is not None:
            phase_epoch = start_step // max(1, total_steps) if total_steps > 0 else 0
            self._train_sampler.set_epoch(phase_epoch)
            if self.rank == 0:
                print(f"[Dist] Train sampler epoch={phase_epoch}")

        # Reset SSM router state at the start of each phase
        target = model.module if hasattr(model, "module") else model
        if hasattr(target, "reset_router_state"):
            target.reset_router_state(batch_size=batch_size)

        step = start_step
        accum_loss = 0.0
        micro_step_count = 0
        self._dataloader_iter = None

        if self.train_dataloader is not None:
            self._dataloader_iter = iter(self.train_dataloader)

        # Curriculum: ramp up physics losses over first N steps
        curriculum_steps = phase_config.get("curriculum_steps", warmup_steps)

        # Progressive context expansion scheduling (Stage 2: 4096 → 16384 → 65536)
        ctx_schedule = phase_config.get("context_expansion_schedule", None)
        current_ctx = getattr(self, "_current_context_length", phase_config.get("base_context", 4096))
        if ctx_schedule is not None:
            print(f"[ContextExpansion] Progressive context expansion enabled: {ctx_schedule}")

        while step < total_steps:
            # LR schedule (unified — handles cosine, flat, and transition ramp)
            if hasattr(self, "_transition_end_step") and step < self._transition_end_step:
                t = (step - self._transition_start_step) / max(1, self._transition_end_step - self._transition_start_step)
                lr = self._transition_old_lr + (self._transition_new_lr - self._transition_old_lr) * t
                self._current_lr_transition = lr
            elif step < warmup_steps:
                lr = base_lr * step / max(1, warmup_steps)
            else:
                lr_schedule = phase_config.get("lr_schedule", "cosine")
                lr = get_learning_rate_schedule(step, base_lr, warmup_steps, total_steps, min_lr, schedule=lr_schedule)
            self._current_lr = lr
            curriculum_progress = min(1.0, (step - start_step) / max(1, curriculum_steps))

            steps_per_epoch = (
                len(self.train_dataloader)
                if self.train_dataloader is not None
                else max(1, int(warmup_steps))
            )

            self._apply_pending_thaw(step)
            thaw_changed = self._apply_timed_freeze_policy(
                phase_config["freeze_policy"], step - start_step,
                steps_per_epoch)
            if thaw_changed and not use_deepspeed:
                self.create_optimizer(phase_config)
            self._current_step = step


            if use_deepspeed:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            # Generic species runs feed forward_modalities with raw signals
            # per modality; the EEG/fMRI(+MEG) path is unchanged.
            rollout_steps = max(1, int(
                self._rollout_steps_override
                if getattr(self, "_rollout_steps_override", None)
                else phase_config.get("rollout_steps", 1)))
            self._rollout_steps = rollout_steps
            generic_signals = None
            signals = None
            dummy_meg = None
            actual_eeg = actual_fmri = actual_meg = None
            # model has a MEG branch); phases may still override the weight.
            meg_present = (not self.generic_model and dummy_meg is not None)
            if meg_present and getattr(self, "_meg_recon_enabled", False) is False:
                self._meg_recon_enabled = True
                self._enable_meg_recon()

            # Inject Wiener monitor values from previous step into model
            # for oscillation-tolerant feedback (§11.10.2.1)
            target = model.module if hasattr(model, "module") else model
            q_report = self.q_factor_monitor.get_report()
            ac_report = self.ataxia_catalepsy_monitor.get_report()
            target._wiener_q_factor = q_report.get("q_factor_current", 1.0)
            target._wiener_ataxia = ac_report.get("ataxia_current", 0.0)
            target._wiener_catalepsy = ac_report.get("catalepsy_current", 0.0)
            step_metrics = {}

            step_metrics.update(self._generic_control_metrics())
            if self.generic_model:
                signals, raw_targets, perturbation = partition_generic_batch(
                    self._generic_batch_tensors, self.signal_modalities,
                    self.recon_modalities, self.device,
                    control_modalities=self.control_modalities,
                    control_reduction=self.control_reduction,
                    rollout_steps=rollout_steps,
                    require_future_targets=self.require_future_targets,
                    require_multi_horizon=self._forecast_enabled)
                signal_masks = dict(self._generic_signal_masks)
            else:
                raw_targets, raw_target_masks = self._standard_raw_targets(
                    dummy_eeg, dummy_fmri, dummy_meg)
                actual_eeg = (
                    raw_targets["eeg"]
                    if raw_targets["eeg"].dim() == 3 else None)
                actual_fmri = (
                    raw_targets["fmri"]
                    if raw_targets["fmri"].dim() == 3 else None)
                actual_meg = (
                    raw_targets.get("meg")
                    if (raw_targets.get("meg") is not None
                        and raw_targets["meg"].dim() == 3) else None)
                signal_masks = dict(self._standard_signal_masks)
            if self.generic_model:
                raw_target_masks = {
                    modality: raw_targets[f"{modality}_mask"]
                    for modality in self.recon_modalities
                    if f"{modality}_mask" in raw_targets
                }
                raw_targets = {
                    key: value for key, value in raw_targets.items()
                    if not key.endswith("_mask")
                }
            perturbation = (
                self._sanitize_perturbation(perturbation, model)
                if self.generic_model else
                self._sanitize_perturbation(
                    self._standard_perturbation, model))
            intervention_weight = getattr(
                self.total_loss.loss_weights, "intervention_response", 0.0)
            intervention_enabled = intervention_weight > 0
            baseline_perturbation = (
                self._generic_batch_tensors.get(
                    "intervention_baseline",
                    self._generic_batch_tensors.get(
                        "baseline_perturbation"))
                if self.generic_model
                else self._standard_baseline_perturbation)
            if intervention_enabled:
                target_map = (raw_targets if self.generic_model
                              else self._standard_contract_targets)
                if "intervention_target" not in target_map:
                    raise ValueError(
                        "intervention response loss requires "
                        "intervention_target in each batch")
                if perturbation is None or baseline_perturbation is None:
                    raise ValueError(
                        "intervention response loss requires treated and "
                        "baseline perturbation tensors")
                baseline_perturbation = self._sanitize_perturbation(
                    baseline_perturbation, model)
                if baseline_perturbation is None:
                    raise ValueError(
                        "intervention response loss requires a model "
                        "perturbation channel")
                if baseline_perturbation.shape != perturbation.shape:
                    raise ValueError(
                        "intervention_baseline must match the treated "
                        "perturbation shape")

            imagination_interval = phase_config.get("imagination_interval", 0)
            if imagination_interval > 0:
                tgt_model = model.module if hasattr(model, "module") else model
                is_imag_step = step % imagination_interval == 0 and step > 0
                if hasattr(tgt_model, "_imagination_active"):
                    tgt_model._imagination_active = is_imag_step

            if imagination_interval > 0 and is_imag_step:
                step_metrics["imagination_step"] = 1.0

            if use_deepspeed:
                if self.generic_model:
                    if perturbation is not None:
                        step_metrics["perturbation_norm"] = float(
                            perturbation.abs().mean())
                    outputs = model.forward_modalities(
                        signals, masks=signal_masks,
                        num_steps=rollout_steps,
                        return_sequences=rollout_steps > 1,
                        reconstruct=True,
                        perturbation=perturbation,
                        **self._generic_step_kwargs(
                            self._generic_batch_tensors))
                else:
                    outputs = model(
                        dummy_eeg, dummy_fmri,
                        meg=dummy_meg,
                        actual_eeg=actual_eeg,
                        actual_fmri=actual_fmri,
                        actual_meg=actual_meg,
                        masks=signal_masks,
                        action=None,
                        perturbation=perturbation,
                        mode=("forecast" if rollout_steps > 1
                              else "perception"),
                        num_steps=rollout_steps,
                        return_sequences=rollout_steps > 1,
                        cross_modal_labels=self._standard_contract_targets.get(
                            "cross_modal_labels"),
                    )
                if intervention_enabled:
                    baseline_outputs = self._forward_intervention_baseline(
                        model,
                        signals=(signals if self.generic_model else None),
                        dummy_eeg=dummy_eeg,
                        dummy_fmri=dummy_fmri,
                        dummy_meg=dummy_meg,
                        actual_eeg=actual_eeg,
                        actual_fmri=actual_fmri,
                        actual_meg=actual_meg,
                        signal_masks=signal_masks,
                        rollout_steps=rollout_steps,
                        perturbation=baseline_perturbation,
                        cross_modal_labels=(
                            self._standard_contract_targets.get(
                                "cross_modal_labels")
                            if not self.generic_model else None),
                    )
                    target_effect = (
                        raw_targets["intervention_target"]
                        if self.generic_model
                        else self._standard_contract_targets[
                            "intervention_target"])
                    self._attach_intervention_effect(
                        outputs, baseline_outputs, target_effect)
                targets = self._prepare_reconstruction_targets(
                    outputs, raw_targets, raw_target_masks)
                if not self.generic_model:
                    targets.update(self._standard_contract_targets)
                total_loss, loss_metrics = self.total_loss(outputs, targets)
                model.backward(total_loss)
                model.step()
            else:
                is_accum_boundary = (micro_step_count + 1) % gradient_accumulation == 0
                sync_context = self.model.no_sync() if hasattr(self.model, "no_sync") and not is_accum_boundary else None

                with autocast_context(self.device, enabled=True):
                    if self.generic_model:
                        if perturbation is not None:
                            step_metrics["perturbation_norm"] = float(
                                perturbation.abs().mean())
                        outputs = model.forward_modalities(
                            signals, masks=signal_masks,
                            num_steps=rollout_steps,
                            return_sequences=rollout_steps > 1,
                            reconstruct=True,
                            recon_max_channels=self.recon_max_channels,
                            perturbation=perturbation,
                            **self._generic_step_kwargs(
                                self._generic_batch_tensors))
                    else:
                        outputs = model(
                            dummy_eeg, dummy_fmri,
                            meg=dummy_meg,
                            actual_eeg=actual_eeg,
                            actual_fmri=actual_fmri,
                            actual_meg=actual_meg,
                            masks=signal_masks,
                            action=None,
                            perturbation=perturbation,
                            mode=("forecast" if rollout_steps > 1
                                  else "perception"),
                            num_steps=rollout_steps,
                            return_sequences=rollout_steps > 1,
                            cross_modal_labels=self._standard_contract_targets.get(
                                "cross_modal_labels"),
                        )
                    if intervention_enabled:
                        baseline_outputs = (
                            self._forward_intervention_baseline(
                                model,
                                signals=(signals if self.generic_model
                                         else None),
                                dummy_eeg=dummy_eeg,
                                dummy_fmri=dummy_fmri,
                                dummy_meg=dummy_meg,
                                actual_eeg=actual_eeg,
                                actual_fmri=actual_fmri,
                                actual_meg=actual_meg,
                                signal_masks=signal_masks,
                                rollout_steps=rollout_steps,
                                perturbation=baseline_perturbation,
                                cross_modal_labels=(
                                    self._standard_contract_targets.get(
                                        "cross_modal_labels")
                                    if not self.generic_model else None),
                            ))
                        target_effect = (
                            raw_targets["intervention_target"]
                            if self.generic_model
                            else self._standard_contract_targets[
                                "intervention_target"])
                        self._attach_intervention_effect(
                            outputs, baseline_outputs, target_effect)
                    targets = self._prepare_reconstruction_targets(
                        outputs, raw_targets, raw_target_masks)
                    if not self.generic_model:
                        targets.update(self._standard_contract_targets)
                    total_loss, loss_metrics = self.total_loss(outputs, targets)
                    total_loss = total_loss / gradient_accumulation

                if sync_context is not None:
                    with sync_context:
                        self.scaler.scale(total_loss).backward()
                else:
                    self.scaler.scale(total_loss).backward()
                micro_step_count += 1

                if micro_step_count % gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_clip = getattr(self, "_grad_clip", 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    micro_step_count = 0

            accum_loss += total_loss.item() if hasattr(total_loss, "item") else float(total_loss)


            # L1: KDA state normalization every 64 steps
            if step % 64 == 0:
                tgt = model.module if hasattr(model, "module") else model
                if hasattr(tgt, "kda_state_history") and tgt.kda_state_history is not None:
                    with torch.no_grad():
                        tgt.kda_state_history = self.stability.apply_L1(tgt.kda_state_history)

            # L5: Hebbian spectral normalization every 100 steps
            if step % 100 == 0:
                self._apply_hebbian_spectral_norm(model)

            # QR reorthogonalization for Poisson Router every 1000 steps
            if step % 1000 == 0:
                self._apply_router_orthogonalization(model)

            # Monitor metrics for auto-rollback (run at reduced cadence and
            # only in training mode: monitors hold stateful EMAs, so running
            # them during eval would make evaluation history-dependent)
            monitor_interval = max(1, getattr(self, "monitor_interval", 10))
            run_monitors = (model.training and step % monitor_interval == 0)
            step_metrics["total_loss"] = accum_loss
            step_metrics["lr"] = lr
            step_metrics["curriculum"] = curriculum_progress
            # Extract router entropy if available
            if "moe_routing" in outputs and "router_entropy" in outputs["moe_routing"]:
                step_metrics["router_entropy"] = outputs["moe_routing"]["router_entropy"]

            # StabilityMonitor needs the energy potential for its divergence
            # trigger.  It is free to report: the scalar is produced alongside
            # the energy gradient in the velocity field, so no extra forward.
            if outputs.get("E") is not None:
                step_metrics["E"] = float(outputs["E"])

            # Signed-work ratio, stability trigger 1.  This is a *within
            # trajectory* diagnostic, so it is only emitted when a real
            # multi-step rollout exists; buffering single-step velocities
            # across iterations would mix unrelated batches, which is the same
            # mistake the old latent PSD buffer made.
            delta_seq = outputs.get("delta_z_sequence")
            if (isinstance(delta_seq, torch.Tensor)
                    and delta_seq.dim() == 3 and delta_seq.shape[1] >= 3):
                try:
                    from ..physics.thermodynamics import (
                        compute_trajectory_diagnostics)
                    velocity = delta_seq.transpose(0, 1)          # (T, B, D)
                    path = torch.cumsum(velocity, dim=0) * float(
                        getattr(model, "integration_dt", 1.0) or 1.0)
                    step_metrics["negative_work_ratio"] = float(
                        compute_trajectory_diagnostics(path, velocity)[
                            "negative_work_ratio"])
                except Exception:
                    pass

            # Wiener Monitor #3: Q-factor (latent velocity FFT)
            if run_monitors and "delta_z" in outputs:
                q_result = self.q_factor_monitor.update(outputs["delta_z"])
                step_metrics["q_factor"] = q_result["q_factor"]
                step_metrics["q_f0"] = q_result["q_f0"]
                step_metrics["q_band"] = q_result.get("q_band", "unknown")

            # Wiener Monitor #7: Ataxia/Catalepsy (EEG-signal monitor; runs
            # only when a reconstruction target is available — generic
            # species runs without eeg_recon skip it)
            eeg_recon = outputs.get("eeg_recon")
            z_global = outputs.get("z_global")
            # TotalLoss consumed this exact aligned target mapping.
            actual_signal = targets.get("eeg") if eeg_recon is not None else None
            if run_monitors and eeg_recon is not None and actual_signal is not None:
                ac_result = self.ataxia_catalepsy_monitor.update(
                    predicted_signal=eeg_recon,
                    actual_signal=actual_signal,
                    z=z_global,
                )
                step_metrics["ataxia_score"] = ac_result["ataxia_score"]
                step_metrics["catalepsy_score"] = ac_result["catalepsy_score"]
                step_metrics["ataxia_alert"] = ac_result["ataxia_alert"]
                step_metrics["catalepsy_alert"] = ac_result["catalepsy_alert"]

            # Heuristic energy-change monitor; not work, Jarzynski, or EPR.
            if (run_monitors and self.energy_change_monitor is not None
                    and "z_global" in outputs and "delta_z" in outputs):
                energy_result = self.energy_change_monitor.update(
                    z_current=outputs["z_global"],
                    z_next=outputs["z_global"] + outputs["delta_z"].detach(),
                    energy_fn=getattr(self, "_energy_fn_ref", None),
                )
                step_metrics["energy_change_proxy"] = energy_result[
                    "energy_change_proxy"]
            # Dissipative-structure proxy: this is not stochastic EPR.
            if run_monitors and self.wiener_homeostat is not None and "delta_z" in outputs:
                dissipative_proxy = (outputs["delta_z"] ** 2).sum(dim=-1).mean().item() / max(
                    self.wiener_homeostat.D_eff.item(), 1e-4)
                router_entropy = step_metrics.get("router_entropy", 1.0)
                hm_result = self.wiener_homeostat.update(dissipative_proxy, router_entropy)
                step_metrics["D_eff"] = hm_result["D_eff"]
                step_metrics["homeostat_health"] = hm_result["health"]
                # Apply D_eff only when explicitly enabled.  The vector is
                # an initialized gain, not a guarantee of output matching.
                if getattr(self, "_homeostat_active", False):
                    try:
                        tgt = model.module if hasattr(model, "module") else model
                        if hasattr(tgt, "velocity_brain") and hasattr(tgt.velocity_brain, "ou_noise"):
                            tgt.velocity_brain.ou_noise.set_diffusion(
                                hm_result.get("D_eff_vec", hm_result["D_eff"]))
                    except Exception:
                        pass

            self.stability.monitor.update(step_metrics)

            # L6/L7: Auto-rollback check every 1000 steps
            if step % 1000 == 0 and step > 0:
                triggered, reason = self.stability.monitor.check_triggers(step)
                if triggered:
                    if self.is_main_process:
                        print(f"[STABILITY] Trigger fired at step {step}: {reason}")
                    # The rollback mutates model state (router re-init, Top-1
                    # override), so it runs on EVERY rank.  Guarding it with
                    # is_main_process would let ranks diverge, and the router
                    # re-init draws from the global RNG, so it is seeded from
                    # the step to make every rank produce identical weights.
                    prev_seed = torch.initial_seed()
                    torch.manual_seed(step + 12345)
                    try:
                        level, action = self.rollback.on_trigger(
                            reason, model, self.optimizer, step)
                    finally:
                        torch.manual_seed(prev_seed)
                    self._apply_rollback_action(level, action, model)
                    if self.is_main_process:
                        print(f"[STABILITY] Rollback level {level}: {action}")
                    if level >= 4:
                        if self.is_main_process:
                            print("[STABILITY] Level 4: stopping the phase. No "
                                  "checkpoint is loaded automatically — resume "
                                  "with --resume_from after inspection.")
                        break

            # Wiener monitor sustained alerts (every 1000 steps)
            if step % 1000 == 0 and step > 0:
                ac_report = self.ataxia_catalepsy_monitor.get_report()
                if ac_report.get("ataxia_alert", False) and self.is_main_process:
                    print(f"[WIENER] Ataxia alert at step {step}: score={ac_report['ataxia_current']:.4f} sustained {ac_report['ataxia_sustained']} steps")
                if ac_report.get("catalepsy_alert", False) and self.is_main_process:
                    print(f"[WIENER] Catalepsy alert at step {step}: score={ac_report['catalepsy_current']:.4f} sustained {ac_report['catalepsy_sustained']} steps")

            if step % 5000 == 0 and step > 0:
                self.save_checkpoint(step, phase_name)

            if step % 10 == 0 and self.logger:
                avg_loss = accum_loss / max(step - start_step + 1, 10)
                if self.world_size > 1:
                    loss_tensor = torch.tensor([avg_loss], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss = loss_tensor.item() / self.world_size
                metrics = {
                    "total_loss": avg_loss,
                    "lr": lr,
                    "curriculum": curriculum_progress,
                }
                # Wiener monitor logging (every 100 steps to reduce noise)
                if step % 100 == 0:
                    q_report = self.q_factor_monitor.get_report()
                    ac_report = self.ataxia_catalepsy_monitor.get_report()
                    metrics["q_factor"] = q_report.get("q_factor_current", float("nan"))
                    metrics["ataxia_score"] = ac_report.get("ataxia_current", float("nan"))
                    metrics["catalepsy_score"] = ac_report.get("catalepsy_current", float("nan"))
                    if self.energy_change_monitor is not None:
                        energy_report = self.energy_change_monitor.get_report()
                        metrics["energy_change_proxy"] = energy_report.get(
                            "energy_change_proxy_current", float("nan"))
                    if self.wiener_homeostat is not None:
                        hm_report = self.wiener_homeostat.get_report()
                        metrics["D_eff"] = hm_report.get("D_eff", float("nan"))
                        metrics["homeostat_health"] = hm_report.get("health", float("nan"))
                    if ac_report.get("ataxia_alert", False):
                        metrics["ataxia_alert"] = 1.0
                    if ac_report.get("catalepsy_alert", False):
                        metrics["catalepsy_alert"] = 1.0
                self.logger.log(step, metrics, phase_name)
                accum_loss = 0.0

            # Validation every 5000 steps (or 1000 steps for longer phases)
            val_interval = phase_config.get("val_interval", 5000)
            if step % val_interval == 0 and step > 0 and self.val_dataloader is not None:
                if self.is_main_process:
                    print(f"[Validation] Running validation at step {step}")
                val_metrics = self.validate(model, phase_config, step, total_steps)
                if val_metrics:
                    if self.is_main_process:
                        print(f"[Validation] val_loss={val_metrics.get('val_loss', float('inf')):.4f}")
                    if self.logger:
                        self.logger.log(step, val_metrics, f"{phase_name}_val")

                    # Save best model based on validation loss
                    best_val_loss = getattr(self, "_best_val_loss", float("inf"))
                    if val_metrics.get("val_loss", float("inf")) < best_val_loss:
                        self._best_val_loss = val_metrics["val_loss"]
                        if self.is_main_process:
                            best_path = self.checkpoint_dir / f"best_{phase_name}_step{step}.pt"
                            print(f"[Validation] New best model! Saving to {best_path}")
                            # Reuse save mechanism with explicit filename
                            self.save_checkpoint(step, phase_name, filename=f"best_{phase_name}_step{step}.pt")

                    # Early stopping check
                    early_stop = getattr(self, "early_stopper", None)
                    if early_stop is not None:
                        should_stop = early_stop.step(val_metrics.get("val_loss", float("inf")))
                        if should_stop:
                            if self.is_main_process:
                                print(f"[Validation] Early stopping triggered at step {step}")
                            break

            # Progressive context expansion
            if ctx_schedule is not None:
                for ctx_step, ctx_target in ctx_schedule:
                    if step == ctx_step and current_ctx != ctx_target:
                        if self.is_main_process:
                            print(f"[ContextExpansion] Step {step}: expanding context {current_ctx} → {ctx_target}")
                        current_ctx = ctx_target
                        self._current_context_length = current_ctx
                        # If model supports context expansion, notify it
                        if hasattr(model, "set_context_length"):
                            model.set_context_length(current_ctx)
                        elif hasattr(model, "module") and hasattr(model.module, "set_context_length"):
                            model.module.set_context_length(current_ctx)
                        break

            step += 1

        return step

    def _apply_hebbian_spectral_norm(self, model):
        """L5: Apply power-iteration spectral normalization to Hebbian weights."""
        target = model.module if hasattr(model, "module") else model
        for name, module in target.named_modules():
            if hasattr(module, "W") and hasattr(module, "spectral_bound"):
                with torch.no_grad():
                    module.W.data = module.clip_spectral()

    def _apply_rollback_action(self, level: int, action: Dict, model) -> None:
        """Apply the state mutation implied by a rollback level.

        Level 1 escalates the balance weight and sharpens routing, level 2 is
        performed inside ``AutoRollback`` (router re-init), level 3 forces
        Top-1 routing for a bounded window, level 4 stops the phase.  The
        caller runs this on every rank.
        """
        target = model.module if hasattr(model, "module") else model
        moe = getattr(target, "moe_velocity", None)
        router = getattr(moe, "router", None)

        if level == 1:
            weights = getattr(self.total_loss, "loss_weights", None)
            multiplier = action.get("balance_weight_multiplier")
            if weights is not None and multiplier:
                updated = float(getattr(weights, "moe_load_balance", 0.0)) * float(
                    multiplier)
                weights.moe_load_balance = updated
                if self.is_main_process:
                    print(f"[STABILITY] moe_load_balance -> {updated:.4g}")
            temperature = action.get("temperature")
            if router is not None and temperature is not None:
                setter = getattr(moe, "set_router_tau", None)
                if callable(setter):
                    setter(float(temperature))
                else:
                    router.tau = float(temperature)
                if self.is_main_process:
                    print(f"[STABILITY] router tau -> {temperature}")
        elif level == 3:
            if router is not None and hasattr(router, "apply_top1_override"):
                duration = int(action.get("duration_steps", 1000))
                router.apply_top1_override(duration)
                if self.is_main_process:
                    print(f"[STABILITY] Top-1 routing forced for {duration} steps")

    def _apply_router_orthogonalization(self, model):
        """QR reorthogonalization for Poisson Router antisymmetric matrix."""
        target = model.module if hasattr(model, "module") else model
        for name, module in target.named_modules():
            if hasattr(module, "orthogonalize") and callable(getattr(module, "orthogonalize")):
                with torch.no_grad():
                    module.orthogonalize()

    def validate(
        self,
        model,
        phase_config: Dict,
        step: int,
        total_steps: int,
        dataloader: Optional[Any] = None,
        label: str = "val",
    ) -> Dict[str, float]:
        """Evaluate one deterministic, state-isolated validation split."""
        loader = self.val_dataloader if dataloader is None else dataloader
        if loader is None:
            return {}

        from ..diagnostics.causal_dynamics import forward_reverse_prediction_gap
        from ..diagnostics.free_run_metrics import (
            intrinsic_rollout_stats, run_free_run_suite)

        model_was_training = model.training
        model.eval()
        target_model = model.module if hasattr(model, "module") else model

        def reset_eval_state():
            if hasattr(target_model, "reset_history"):
                target_model.reset_history()
            elif hasattr(target_model, "reset_runtime_state"):
                target_model.reset_runtime_state()

        reset_eval_state()
        total_loss_value = 0.0
        val_metrics = {}
        causal_gaps = []
        one_step_corr = []
        one_step_variance = []
        one_step_autocorr = []
        metric_dt_used = []
        free_step_norm = []
        free_tail_ratio = []
        intervention_enabled = getattr(
            self.total_loss.loss_weights, "intervention_response", 0.0) > 0
        val_steps = 0
        max_val_steps = 100
        rollout_steps = max(1, int(phase_config.get("rollout_steps", 1)))
        phase_weights = phase_config.get("loss_weights")
        if self.generic_model and self.recon_modalities:
            phase_weights = augment_phase_loss_weights(
                phase_weights or LossWeights(), self.recon_modalities,
                getattr(self, "recon_loss_types", {}))
            self.total_loss.loss_weights = phase_weights
        self._forecast_enabled = bool(
            getattr(phase_weights, "forecast", 0.0) > 0
            and rollout_steps > 1)
        try:
            # Physics fields use autograd internally; validation disables
            # parameter updates by omitting backward, not by disabling autograd.
            with torch.enable_grad():
                for i, batch in enumerate(loader):
                    if self.generic_model:
                        batch_tensors = {}
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                batch_tensors[key] = move_to_device(
                                    value, self.device)
                            elif key in {
                                    "dt", "species", "sample_rate_hz",
                                    "rate_hz"}:
                                # Keep non-tensor manifest metadata available
                                # to the same per-recording physics kwargs
                                # used by the training path.
                                batch_tensors[key] = value
                        signals, batch_targets, perturbation = (
                            partition_generic_batch(
                                batch_tensors, self.signal_modalities,
                                self.recon_modalities, self.device,
                                control_modalities=self.control_modalities,
                                control_reduction=self.control_reduction,
                                rollout_steps=rollout_steps,
                                require_future_targets=(
                                    self.require_future_targets),
                                require_multi_horizon=(
                                    self._forecast_enabled)))
                        if not signals:
                            continue
                        signal_masks = {
                            m: batch_tensors[f"{m}_mask"]
                            for m in self.signal_modalities
                            if f"{m}_mask" in batch_tensors
                        }
                        raw_target_masks = {
                            m: batch_targets[f"{m}_mask"]
                            for m in self.recon_modalities
                            if f"{m}_mask" in batch_targets
                        }
                        raw_targets = {
                            key: value for key, value in batch_targets.items()
                            if not key.endswith("_mask")
                        }
                        perturbation = self._sanitize_perturbation(
                            perturbation, model)
                        outputs = model.forward_modalities(
                            signals, masks=signal_masks,
                            num_steps=rollout_steps, return_all=True,
                            return_sequences=rollout_steps > 1,
                            reconstruct=True,
                            recon_max_channels=self.recon_max_channels,
                            **self._generic_step_kwargs(batch_tensors),
                            perturbation=perturbation)
                    else:
                        self._capture_standard_batch(batch)
                        eeg = batch.get("eeg", batch.get("eeg_data"))
                        fmri = batch.get("fmri", batch.get("fmri_data"))
                        if not isinstance(eeg, torch.Tensor) or not isinstance(
                                fmri, torch.Tensor):
                            continue
                        eeg = move_to_device(eeg, self.device)
                        fmri = move_to_device(fmri, self.device)
                        meg = batch.get("meg", batch.get("meg_data"))
                        meg = (move_to_device(meg, self.device)
                               if isinstance(meg, torch.Tensor) else None)
                        raw_targets, raw_target_masks = (
                            self._standard_raw_targets(eeg, fmri, meg))
                        perturbation = self._sanitize_perturbation(
                            self._standard_perturbation, model)
                        if self.require_future_targets:
                            missing = [
                                m for m in ("eeg", "fmri")
                                if m not in self._standard_targets
                            ]
                            if meg is not None and "meg" in self.species_modalities:
                                if "meg" not in self._standard_targets:
                                    missing.append("meg")
                            if missing:
                                raise ValueError(
                                    "missing future targets for validation: "
                                    + ", ".join(missing))
                        outputs = model(
                            eeg, fmri, meg=meg,
                            actual_eeg=(
                                raw_targets["eeg"]
                                if raw_targets["eeg"].dim() == 3 else None),
                            actual_fmri=(
                                raw_targets["fmri"]
                                if raw_targets["fmri"].dim() == 3 else None),
                            actual_meg=(
                                raw_targets.get("meg")
                                if (raw_targets.get("meg") is not None
                                    and raw_targets["meg"].dim() == 3)
                                else None),
                            masks=signal_masks,
                            perturbation=perturbation,
                            mode=("forecast" if rollout_steps > 1
                                  else "perception"),
                            num_steps=rollout_steps,
                            return_all=True,
                            return_sequences=rollout_steps > 1,
                            cross_modal_labels=self._standard_contract_targets.get(
                                "cross_modal_labels"),
                        )

                    if intervention_enabled:
                        target_map = (raw_targets if self.generic_model
                                      else self._standard_contract_targets)
                        target_effect = target_map.get(
                            "intervention_target")
                        baseline_perturbation = (
                            batch_tensors.get(
                                "intervention_baseline",
                                batch_tensors.get("baseline_perturbation"))
                            if self.generic_model
                            else self._standard_baseline_perturbation)
                        if target_effect is None:
                            raise ValueError(
                                "intervention response loss requires "
                                "intervention_target in validation batch")
                        if perturbation is None or baseline_perturbation is None:
                            raise ValueError(
                                "intervention response loss requires treated "
                                "and baseline perturbation tensors")
                        baseline_perturbation = self._sanitize_perturbation(
                            baseline_perturbation, model)
                        if baseline_perturbation is None:
                            raise ValueError(
                                "intervention response loss requires a model "
                                "perturbation channel")
                        if baseline_perturbation.shape != perturbation.shape:
                            raise ValueError(
                                "intervention_baseline must match the "
                                "treated perturbation shape")
                        baseline_outputs = self._forward_intervention_baseline(
                            model,
                            signals=(signals if self.generic_model else None),
                            dummy_eeg=(eeg if not self.generic_model else None),
                            dummy_fmri=(fmri if not self.generic_model else None),
                            dummy_meg=(meg if not self.generic_model else None),
                            actual_eeg=(
                                raw_targets["eeg"]
                                if not self.generic_model
                                and raw_targets["eeg"].dim() == 3 else None),
                            actual_fmri=(
                                raw_targets["fmri"]
                                if not self.generic_model
                                and raw_targets["fmri"].dim() == 3 else None),
                            actual_meg=(
                                raw_targets.get("meg")
                                if not self.generic_model
                                and raw_targets.get("meg") is not None
                                and raw_targets["meg"].dim() == 3 else None),
                            signal_masks=signal_masks,
                            rollout_steps=rollout_steps,
                            perturbation=baseline_perturbation,
                            cross_modal_labels=(
                                self._standard_contract_targets.get(
                                    "cross_modal_labels")
                                if not self.generic_model else None),
                        )
                        self._attach_intervention_effect(
                            outputs, baseline_outputs, target_effect)
                    targets = self._prepare_reconstruction_targets(
                        outputs, raw_targets, raw_target_masks)
                    if not self.generic_model:
                        targets.update(self._standard_contract_targets)
                    batch_loss, loss_metrics = self.total_loss(
                        outputs, targets, update_normalizer=False)
                    total_loss_value += float(batch_loss.item())
                    val_steps += 1

                    if "moe_routing" in outputs:
                        for key, value in outputs["moe_routing"].items():
                            if isinstance(value, torch.Tensor):
                                if not (
                                    value.is_floating_point()
                                    or value.is_complex()
                                ):
                                    continue
                                value = float(value.detach().mean().item())
                            if isinstance(value, (int, float)):
                                val_metrics.setdefault(key, []).append(value)

                    states = outputs.get("states")
                    if isinstance(states, torch.Tensor) and states.dim() == 3:
                        if states.shape[1] >= 3:
                            latent_path = states.transpose(0, 1)
                            def latent_predictor(current):
                                original_shape = current.shape
                                flat = current.reshape(-1, original_shape[-1])
                                if hasattr(target_model, "reset_runtime_state"):
                                    target_model.reset_runtime_state(
                                        batch_size=flat.shape[0])
                                # Probe the full trained latent step (backbone
                                # + MoE + control), not the backbone alone:
                                # the operative question is whether the *trained
                                # field* is time-asymmetric.  Recurrent state is
                                # reset per call so the probe is memoryless and
                                # cannot leak into the main path.
                                step_out = target_model._latent_step(flat)
                                predicted = flat + target_model.integration_dt * (
                                    step_out["delta_z"])
                                return predicted.reshape(original_shape)

                            if hasattr(target_model, "reset_runtime_state"):
                                target_model.reset_runtime_state(
                                    batch_size=states.shape[0])
                            causal = forward_reverse_prediction_gap(
                                latent_path, latent_predictor)
                            causal_gaps.append(
                                float(causal["forward_reverse_gap"].item()))
                            if hasattr(target_model, "reset_runtime_state"):
                                target_model.reset_runtime_state(
                                    batch_size=states.shape[0])

                            # Genuine free run: `states` is the unsupervised
                            # latent rollout, so its autonomy statistics are
                            # what free-run stability actually means.
                            autonomy = intrinsic_rollout_stats(
                                states.detach().cpu().numpy())
                            free_step_norm.append(autonomy["step_norm_mean"])
                            free_tail_ratio.append(autonomy["tail_std_ratio"])

                    for modality in self.recon_modalities:
                        prediction = outputs.get(f"{modality}_recon")
                        real = targets.get(modality)
                        # Trivial baselines on the same targets and mask:
                        # without them a reconstruction correlation is not
                        # interpretable (calcium is a low-passed, aliased
                        # observable, so 'persistence' is a real competitor).
                        observed = signals.get(modality)
                        if (isinstance(observed, torch.Tensor)
                                and isinstance(real, torch.Tensor)
                                and observed.shape == real.shape):
                            tgt_mask = targets.get(f"{modality}_mask")
                            val_metrics.setdefault(
                                f"recon_{modality}_baseline_persistence",
                                []).append(masked_channel_correlation(
                                    observed, real, tgt_mask))
                            val_metrics.setdefault(
                                f"recon_{modality}_baseline_channel_mean",
                                []).append(masked_channel_correlation(
                                    observed.mean(dim=-1, keepdim=True)
                                    .expand_as(real), real, tgt_mask))
                        if (isinstance(prediction, torch.Tensor)
                                and isinstance(real, torch.Tensor)
                                and prediction.shape[-1] >= 3
                                and prediction.shape[1] >= 2):
                            # One-step *prediction* statistics: the decoder
                            # output for the next window scored against that
                            # window.  The latent advanced a single step from a
                            # real observation, so this is not a free run.
                            frame_dt_s = (
                                self._batch_frame_dt(batch)
                                if isinstance(batch, dict) else None) or float(
                                    getattr(target_model, "latent_dt", None)
                                    or 1.0)
                            pair = run_free_run_suite(
                                real[0].detach().transpose(0, 1).cpu().numpy(),
                                prediction[0].detach().transpose(0, 1).cpu().numpy(),
                                max_lag=min(
                                    prediction.shape[-1] // 3, 20),
                                dt=frame_dt_s)
                            metric_dt_used.append(float(frame_dt_s))
                            one_step_corr.append(float(pair["corr_matrix_mse"]))
                            one_step_variance.append(
                                float(pair["variance_ratio"]["mean"]))
                            one_step_autocorr.append(
                                float(pair["autocorr"]["mse"]))
                            break
        finally:
            reset_eval_state()
            if model_was_training:
                model.train()
        if val_steps == 0:
            return {}
        prefix = str(label)
        result = {f"{prefix}_loss": total_loss_value / val_steps}
        for key, values in val_metrics.items():
            if values:
                result[f"{prefix}_{key}"] = sum(values) / len(values)
        if causal_gaps:
            result[f"{prefix}_causal_forward_reverse_gap"] = (
                sum(causal_gaps) / len(causal_gaps))
        if metric_dt_used:
            # Seconds-per-frame actually used by the time-converting metrics,
            # so a reported lag/dwell can always be traced to its clock.
            result[f"{prefix}_metric_dt_s"] = (
                sum(metric_dt_used) / len(metric_dt_used))
        if one_step_corr:
            result[f"{prefix}_one_step_corr_matrix_mse"] = (
                sum(one_step_corr) / len(one_step_corr))
            result[f"{prefix}_one_step_variance_ratio"] = (
                sum(one_step_variance) / len(one_step_variance))
            result[f"{prefix}_one_step_autocorr_mse"] = (
                sum(one_step_autocorr) / len(one_step_autocorr))
        if free_step_norm:
            result[f"{prefix}_free_run_step_norm"] = (
                sum(free_step_norm) / len(free_step_norm))
            result[f"{prefix}_free_run_tail_std_ratio"] = (
                sum(free_tail_ratio) / len(free_tail_ratio))

        if self.logger and prefix == "val" and len(
                self.logger.metrics_history) > 10:
            recent_train = [
                m.get("total_loss", 0)
                for m in self.logger.metrics_history[-10:]
            ]
            train_trend = (recent_train[-1] - recent_train[0]) / max(
                len(recent_train), 1)
            result["overfit_flag"] = float(
                train_trend < -0.01
                and result["val_loss"] > recent_train[-1] * 1.5)
        return result

    def train(
        self,
        phases: List[Mapping[str, Any]],
        resume_from: Optional[str] = None,
    ):
        """Run full training pipeline across all nodes."""
        for phase in phases:
            self._validate_runtime_phase(phase)
        self.global_barrier()

        start_step = 0
        current_phase = 0

        if resume_from:
            if self.is_main_process:
                start_step, loaded_phase = self.load_checkpoint(resume_from)
            else:
                loaded_phase = None
            loaded_phase = self.broadcast_object(loaded_phase if self.is_main_process else None)
            start_step = self.broadcast_object(start_step if self.is_main_process else 0)
            for i, p in enumerate(phases):
                if p["name"] == loaded_phase:
                    current_phase = i
                    break
            if self.is_main_process:
                print(f"Resuming from {loaded_phase} at step {start_step}")

        for i in range(current_phase, len(phases)):
            phase = phases[i]
            if self.is_main_process:
                print(f"\n{'='*60}")
                print(f"Starting {phase['name']}")
                print(f"{'='*60}\n")

            self.global_barrier()
            gate = phase.get("transition_gate")
            if gate is not None:
                status = {"ok": True, "error": None}
                if self.is_main_process:
                    metrics = getattr(self, "_last_phase_metrics", None)
                    if metrics is None:
                        status = {
                            "ok": False,
                            "error": "no completed-phase metrics are available",
                        }
                    else:
                        try:
                            if not self.evaluate_phase_transition_gate(phase, metrics):
                                status = {
                                    "ok": False,
                                    "error": "gate returned false",
                                }
                        except Exception as exc:
                            status = {"ok": False, "error": str(exc)}
                status = self.broadcast_object(status)
                if not status["ok"]:
                    raise RuntimeError(
                        f"transition gate rejected phase '{phase['name']}': "
                        f"{status['error']}")

            # Reset SSM router state at phase boundaries
            # The router accumulates temporal context; old context is stale after a phase change
            target = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(target, "reset_router_state"):
                batch_size = phase.get("batch_size", 16)
                target.reset_router_state(batch_size=batch_size)
                if self.is_main_process:
                    print(f"[Router] SSM state reset for phase {phase['name']}")

            if i == current_phase:
                trained_steps = self.train_phase(phase, start_step)
                start_step = 0
            else:
                trained_steps = self.train_phase(phase)

            if self.logger and self.logger.metrics_history:
                self._last_phase_metrics = dict(self.logger.metrics_history[-1])
            self.global_barrier()

            if self.logger:
                self.logger.save(f"metrics_{phase['name']}.json")

        if self.test_dataloader is not None and phases:
            test_metrics = self.validate(
                self.ds_engine if self.ds_engine is not None else self.model,
                phases[-1],
                start_step,
                phases[-1].get("total_steps", start_step),
                dataloader=self.test_dataloader,
                label="test",
            )
            self.test_metrics = test_metrics
            if self.is_main_process and test_metrics:
                print(f"[Test] test_loss={test_metrics.get('test_loss', float('inf')):.4f}")
        if self.is_main_process:
            print("\nTraining complete!")

        cleanup_distributed()

    def __del__(self):
        """Ensure distributed cleanup on garbage collection."""
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def broadcast_object(self, obj):
        """Broadcast a Python object from rank 0 to all processes."""
        if self.world_size == 1:
            return obj
        if self.is_main_process:
            obj_bytes = json.dumps(obj).encode()
            tensor = torch.frombuffer(bytearray(obj_bytes), dtype=torch.uint8).to(self.device)
            count = torch.tensor([tensor.numel()], dtype=torch.long, device=self.device)
        else:
            count = torch.zeros(1, dtype=torch.long, device=self.device)
        dist.broadcast(count, src=0)
        if not self.is_main_process:
            tensor = torch.zeros(count.item(), dtype=torch.uint8, device=self.device)
        dist.broadcast(tensor, src=0)
        return json.loads(tensor.cpu().tobytes().decode())


def create_deepspeed_config() -> Dict:
    """Create DeepSpeed ZeRO Stage 2 configuration."""
    return {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 16,
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e7,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "contiguous_gradients": True,
            "load_from_fp32_weights": True,
            "round_robin_gradients": True,
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {
            "enabled": False,
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
        "communication_data_type": "fp16",
        "aio": {
            "enabled": False,
        },
        "checkpoint": {
            "save_interval": 5000,
            "tag_checkpoint_major_only": False,
        },
        "sparse_checkpoint": {
            "enabled": False,
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
        "zero_allow_untested_optimizer": True,
    }


if __name__ == "__main__":
    print("BrainMoE Trainer module loaded")
    print("To use DeepSpeed, initialize with create_deepspeed_config()")