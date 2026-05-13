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
from typing import Dict, Optional, List, Tuple, Any
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import deepspeed
    from deepspeed import DeepSpeedConfig
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

try:
    from brain_moe_pinn.physics.thermodynamics import BergsonMonitor
    BERGSON_AVAILABLE = True
except ImportError:
    BERGSON_AVAILABLE = False
    BergsonMonitor = None

try:
    from brain_moe_pinn.core.velocity_brain import WienerHomeostat
    HOMEOSTAT_AVAILABLE = True
except ImportError:
    HOMEOSTAT_AVAILABLE = False
    WienerHomeostat = None

from .stability import StabilityController, AutoRollback, QFactorMonitor, AtaxiaCatalepsyMonitor
from .losses import TotalLoss
from .training_phases import LossWeights


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
    """Initialize distributed training for multi-node cluster.

    Supports:
    - Single node (LOCAL_RANK only)
    - Multi-node (WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT)
    """
    if not dist.is_initialized():
        backend = "nccl"

        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = int(os.environ.get("MASTER_PORT", "29500"))

            init_method = f"tcp://{master_addr}:{master_port}"
            dist.init_process_group(backend=backend, init_method=init_method,
                                    world_size=world_size, rank=rank)
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.init_process_group(backend=backend)
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return int(os.environ.get("LOCAL_RANK", 0)), dist.get_rank(), dist.get_world_size()


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
    ):
        self.model = model
        self.config = config
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.local_rank, self.rank, self.world_size = setup_distributed()
        self.is_main_process = self.rank == 0

        self.model = self.model.cuda()

        self.ds_engine = None
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
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.optimizer = None
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.logger = MetricsLogger(log_dir) if self.is_main_process else None
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.stability = StabilityController(hidden_dim=2048, device=f"cuda:{self.local_rank}")
        self.rollback = AutoRollback(checkpoint_dir=str(self.checkpoint_dir))
        self.rollback_trigger_count = 0

        self.q_factor_monitor = QFactorMonitor(window_size=512)
        self.ataxia_catalepsy_monitor = AtaxiaCatalepsyMonitor()
        if BERGSON_AVAILABLE and BergsonMonitor is not None:
            self.bergson_monitor = BergsonMonitor(beta=1.0, window_size=100)
        else:
            self.bergson_monitor = None

        if HOMEOSTAT_AVAILABLE and WienerHomeostat is not None:
            self.wiener_homeostat = WienerHomeostat(
                D_0=1.0, target_epr=1.0, target_entropy=1.0,
            )
        else:
            self.wiener_homeostat = None

        # TotalLoss with physics constraints (lazily updated per phase)
        self.total_loss = TotalLoss(loss_weights=LossWeights())
        self._loss_weights_cache = None

        # Early stopping for validation
        self.early_stopper = None

    def broadcast_config(self, config: Dict) -> Dict:
        """Broadcast config dict from rank 0 to all nodes."""
        if self.world_size == 1:
            return config
        config_bytes = json.dumps(config).encode()
        if self.is_main_process:
            broadcast_tensor = torch.tensor([len(config_bytes)], dtype=torch.long).cuda()
        else:
            broadcast_tensor = torch.zeros(1, dtype=torch.long).cuda()
        dist.broadcast(broadcast_tensor, src=0)
        config_len = broadcast_tensor.item()
        if self.is_main_process:
            config_tensor = torch.from_buffer(bytearray(config_bytes), dtype=torch.uint8).cuda()
        else:
            config_tensor = torch.zeros(config_len, dtype=torch.uint8).cuda()
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
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
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
        old_lr = self.config.get("learning_rate", 3e-5)
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
        """Save model checkpoint."""
        if not self.is_main_process:
            return

        if filename is None:
            filename = f"checkpoint_{phase}_step{step}.pt"

        path = self.checkpoint_dir / filename

        if self.ds_engine is not None:
            # DeepSpeed save: convert selective FP32 then save via standard API
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
        }

        if self.optimizer is not None:
            state["optimizer_state"] = self.optimizer.state_dict()

        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> Tuple[int, str]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=f"cuda:{self.local_rank}")

        model = self.ds_engine if self.ds_engine is not None else self.model
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            ckpt = checkpoint["model_state"]
            model_dict = model.state_dict()
            converted = {}
            for k, v in ckpt.items():
                if k in model_dict and v.dtype != model_dict[k].dtype:
                    converted[k] = v.to(model_dict[k].dtype)
                else:
                    converted[k] = v
            model.load_state_dict(converted)

        if self.optimizer is not None and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint["step"], checkpoint["phase"]

    def _get_batch(self, batch_size: int, step: int = 0, total_steps: int = 1):
        """Get a training batch from dataloader or generate dummy data."""
        # Support DataMixer (progress-dependent sampling)
        if self.train_dataloader is not None and hasattr(self.train_dataloader, "mixer"):
            batch = self.train_dataloader.mixer.next_batch(step, total_steps)
            eeg = batch.get("eeg", batch.get("eeg_data"))
            fmri = batch.get("fmri", batch.get("fmri_data"))
            if eeg is not None and fmri is not None:
                return eeg.cuda(), fmri.cuda()

        if self.train_dataloader is not None:
            try:
                batch = next(self._dataloader_iter)
            except (StopIteration, AttributeError):
                self._dataloader_iter = iter(self.train_dataloader)
                batch = next(self._dataloader_iter)
            eeg = batch.get("eeg", batch.get("eeg_data"))
            fmri = batch.get("fmri", batch.get("fmri_data"))
            if eeg is None or fmri is None:
                # Fallback to dummy if batch format unexpected
                eeg = torch.randn(batch_size, 19, 2560).cuda()
                fmri = torch.randn(batch_size, 400, 100).cuda()
            else:
                eeg = eeg.cuda()
                fmri = fmri.cuda()
            return eeg, fmri
        else:
            dummy_eeg = torch.randn(batch_size, 19, 2560).cuda()
            dummy_fmri = torch.randn(batch_size, 400, 100).cuda()
            return dummy_eeg, dummy_fmri

    def train_phase(self, phase_config: Dict, start_step: int = 0):
        """Train for one phase."""
        phase_name = phase_config.get("name", "unknown")
        total_steps = phase_config.get("total_steps", 10000)
        warmup_steps = phase_config.get("warmup_steps", 500)
        batch_size = phase_config.get("batch_size", 16)
        gradient_accumulation = phase_config.get("gradient_accumulation", 1)
        base_lr = phase_config.get("learning_rate", 3e-5)
        min_lr = phase_config.get("min_lr", 1e-6)

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
        model.train()

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

            self._apply_pending_thaw(step)
            self._current_step = step

            # Update encoder freeze state
            if hasattr(model, "set_training_step"):
                model.set_training_step(step - start_step, freeze_epochs_steps=warmup_steps)
            elif hasattr(model, "module") and hasattr(model.module, "set_training_step"):
                model.module.set_training_step(step - start_step, freeze_epochs_steps=warmup_steps)

            if use_deepspeed:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            dummy_eeg, dummy_fmri = self._get_batch(batch_size, step, total_steps)

            # Inject Wiener monitor values from previous step into model
            # for oscillation-tolerant feedback (§11.10.2.1)
            target = model.module if hasattr(model, "module") else model
            q_report = self.q_factor_monitor.get_report()
            ac_report = self.ataxia_catalepsy_monitor.get_report()
            target._wiener_q_factor = q_report.get("q_factor_current", float("nan"))
            target._wiener_ataxia = ac_report.get("ataxia_current", float("nan"))
            target._wiener_catalepsy = ac_report.get("catalepsy_current", float("nan"))

            # Pre-compute expected decoder output shape for active inference.
            # EEG encoder uses patch_size=256, stride=128.
            eeg_seq_len = dummy_eeg.shape[-1]
            eeg_patches = max(1, (eeg_seq_len - 256) // 128 + 1) if eeg_seq_len >= 256 else 1
            actual_eeg_slice = dummy_eeg[:, :, :eeg_patches]
            # NOTE: For true online updating, actual_eeg should be the *next*
            # timestep (t+1), not the current input (t). The dataloader must
            # return paired sequences for causal active-inference feedback.

            if use_deepspeed:
                outputs = model(
                    dummy_eeg, dummy_fmri,
                    actual_eeg=actual_eeg_slice,
                    actual_fmri=dummy_fmri,
                    action=torch.randn(batch_size, 2048, device=dummy_eeg.device),
                )
                # Build targets at correct resolution
                eeg_target = actual_eeg_slice
                targets = {"eeg": eeg_target, "fmri": dummy_fmri}
                total_loss, loss_metrics = self.total_loss(outputs, targets)
                model.backward(total_loss)
                model.step()
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(
                        dummy_eeg, dummy_fmri,
                        actual_eeg=actual_eeg_slice,
                        actual_fmri=dummy_fmri,
                        action=torch.randn(batch_size, 2048, device=dummy_eeg.device),
                    )
                    eeg_target = actual_eeg_slice
                    targets = {"eeg": eeg_target, "fmri": dummy_fmri}
                    total_loss, loss_metrics = self.total_loss(outputs, targets)
                    total_loss = total_loss / gradient_accumulation

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
            if step % 64 == 0 and hasattr(model, "kda_state_history") and model.kda_state_history is not None:
                with torch.no_grad():
                    model.kda_state_history = self.stability.apply_L1(model.kda_state_history)

            # L5: Hebbian spectral normalization every 100 steps
            if step % 100 == 0:
                self._apply_hebbian_spectral_norm(model)

            # QR reorthogonalization for Poisson Router every 1000 steps
            if step % 1000 == 0:
                self._apply_router_orthogonalization(model)

            # Monitor metrics for auto-rollback
            step_metrics = {
                "total_loss": accum_loss,
                "lr": lr,
                "curriculum": curriculum_progress,
            }
            # Extract router entropy if available
            if "routing_metrics" in outputs and "router_entropy" in outputs["routing_metrics"]:
                step_metrics["router_entropy"] = outputs["routing_metrics"]["router_entropy"]

            # Wiener Monitor #3: Q-factor (latent velocity FFT)
            if "delta_z" in outputs:
                q_result = self.q_factor_monitor.update(outputs["delta_z"])
                step_metrics["q_factor"] = q_result["q_factor"]
                step_metrics["q_f0"] = q_result["q_f0"]
                step_metrics["q_band"] = q_result.get("q_band", "unknown")

            # Wiener Monitor #7: Ataxia/Catalepsy
            eeg_recon = outputs.get("eeg_recon")
            z_global = outputs.get("z_global")
            # Use the same target slice that was fed to TotalLoss for consistency
            actual_signal = eeg_target if eeg_recon is not None else None
            ac_result = self.ataxia_catalepsy_monitor.update(
                predicted_signal=eeg_recon,
                actual_signal=actual_signal,
                z=z_global,
            )
            step_metrics["ataxia_score"] = ac_result["ataxia_score"]
            step_metrics["catalepsy_score"] = ac_result["catalepsy_score"]
            step_metrics["ataxia_alert"] = ac_result["ataxia_alert"]
            step_metrics["catalepsy_alert"] = ac_result["catalepsy_alert"]

            # Wiener Monitor #1: Bergson irreversibility gap
            if self.bergson_monitor is not None and "z_global" in outputs and "delta_z" in outputs:
                bergson_result = self.bergson_monitor.update(
                    z_current=outputs["z_global"],
                    z_next=outputs["z_global"] + outputs["delta_z"].detach(),
                    energy_fn=getattr(self, "_energy_fn_ref", None),
                )
                step_metrics["i_bergson"] = bergson_result["i_bergson"]
                step_metrics["bergson_mean_work"] = bergson_result["bergson_mean_work"]
                step_metrics["bergson_nan_warning"] = bergson_result["bergson_nan_warning"]
                # Collapse alarm: only fire when both I_Bergson AND router_entropy are low
                router_ent = step_metrics.get("router_entropy", 1.0)
                if self.bergson_monitor.check_collapse(router_ent, threshold=0.5):
                    step_metrics["bergson_collapse_alarm"] = 1.0

            # Wiener Homeostat (#5): adaptive noise regulation
            if self.wiener_homeostat is not None and "delta_z" in outputs:
                # EPR proxy: ||v||^2 / D  (sum over hidden dim, mean over batch)
                epr_proxy = (outputs["delta_z"] ** 2).sum(dim=-1).mean().item() / max(self.wiener_homeostat.D_eff, 1e-4)
                router_entropy = step_metrics.get("router_entropy", 1.0)
                hm_result = self.wiener_homeostat.update(epr_proxy, router_entropy)
                step_metrics["D_eff"] = hm_result["D_eff"]
                step_metrics["homeostat_health"] = hm_result["health"]
                # Apply D_eff to VelocityBrain's OU noise (monitoring-only by default;
                # set self._homeostat_active = True to enable active noise regulation)
                if getattr(self, "_homeostat_active", False):
                    try:
                        tgt = model.module if hasattr(model, "module") else model
                        if hasattr(tgt, "velocity_brain") and hasattr(tgt.velocity_brain, "ou_noise"):
                            tgt.velocity_brain.ou_noise.D = float(hm_result["D_eff"])
                    except Exception:
                        pass

            self.stability.monitor.update(step_metrics)

            # L6/L7: Auto-rollback check every 1000 steps
            if step % 1000 == 0 and step > 0:
                triggered, reason = self.stability.monitor.check_triggers(step)
                if triggered and self.is_main_process:
                    print(f"[STABILITY] Trigger fired at step {step}: {reason}")
                    level, action = self.rollback.on_trigger(
                        reason, model, self.optimizer, step
                    )
                    print(f"[STABILITY] Executing rollback level {level}: {action}")
                    if level >= 4:
                        print("[STABILITY] Level 4 rollback requested — stopping phase for manual intervention")
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
                metrics = {
                    "total_loss": accum_loss / min(step - start_step + 1, 10),
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
                    if self.bergson_monitor is not None:
                        bergson_report = self.bergson_monitor.get_report()
                        metrics["i_bergson"] = bergson_report.get("i_bergson_current", float("nan"))
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

    def _apply_router_orthogonalization(self, model):
        """QR reorthogonalization for Poisson Router antisymmetric matrix."""
        target = model.module if hasattr(model, "module") else model
        for name, module in target.named_modules():
            if hasattr(module, "orthogonalize") and callable(getattr(module, "orthogonalize")):
                with torch.no_grad():
                    module.orthogonalize()

    def validate(self, model, phase_config: Dict, step: int, total_steps: int) -> Dict[str, float]:
        """
        Run one validation pass.

        Args:
            model: Model to evaluate (or ds_engine)
            phase_config: Current phase config
            step: Current training step
            total_steps: Total steps in phase

        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}

        model.eval()
        val_loss = 0.0
        val_metrics = {}
        val_steps = 0
        max_val_steps = 100  # Cap validation steps to avoid long eval time

        use_deepspeed = self.ds_engine is not None

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= max_val_steps:
                    break
                eeg = batch.get("eeg", batch.get("eeg_data"))
                fmri = batch.get("fmri", batch.get("fmri_data"))
                if eeg is None or fmri is None:
                    continue
                eeg = eeg.cuda()
                fmri = fmri.cuda()

                outputs = model(eeg, fmri)
                batch_loss = torch.nn.functional.mse_loss(
                    outputs["eeg_recon"], eeg[:, :, :outputs["eeg_recon"].shape[-1]]
                )
                val_loss += batch_loss.item()
                val_steps += 1

                # Collect routing metrics if available
                if "routing_metrics" in outputs:
                    for k, v in outputs["routing_metrics"].items():
                        if k not in val_metrics:
                            val_metrics[k] = []
                        val_metrics[k].append(v)

        if val_steps == 0:
            model.train()
            return {}

        avg_val_loss = val_loss / val_steps
        result = {"val_loss": avg_val_loss}

        # Average accumulated metrics
        for k, v_list in val_metrics.items():
            if isinstance(v_list[0], (int, float)):
                result[f"val_{k}"] = sum(v_list) / len(v_list)

        model.train()

        # Overfit detection
        if self.logger and len(self.logger.metrics_history) > 10:
            recent_train = [m.get("total_loss", 0) for m in self.logger.metrics_history[-10:]]
            train_trend = (recent_train[-1] - recent_train[0]) / max(len(recent_train), 1)
            # If train loss decreasing but val loss high, flag potential overfit
            if train_trend < -0.01 and avg_val_loss > recent_train[-1] * 1.5:
                result["overfit_flag"] = 1.0
            else:
                result["overfit_flag"] = 0.0

        return result

    def train(
        self,
        phases: List[Dict],
        resume_from: Optional[str] = None,
    ):
        """Run full training pipeline across all nodes."""
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

            # Reset SSM router state at phase boundaries
            # The router accumulates temporal context; old context is stale after a phase change
            target = model.module if hasattr(model, "module") else model
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

            self.global_barrier()

            if self.logger:
                self.logger.save(f"metrics_{phase['name']}.json")

        if self.is_main_process:
            print("\nTraining complete!")

    def broadcast_object(self, obj):
        """Broadcast a Python object from rank 0 to all processes."""
        if self.world_size == 1:
            return obj
        if self.is_main_process:
            obj_bytes = json.dumps(obj).encode()
            tensor = torch.frombuffer(bytearray(obj_bytes), dtype=torch.uint8).cuda()
            count = torch.tensor([tensor.numel()], dtype=torch.long).cuda()
        else:
            count = torch.zeros(1, dtype=torch.long).cuda()
        dist.broadcast(count, src=0)
        if not self.is_main_process:
            tensor = torch.zeros(count.item(), dtype=torch.uint8).cuda()
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