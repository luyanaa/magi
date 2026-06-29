#!/usr/bin/env python3
"""
Brain MoE-PINN Training Entry Point

Supports:
- Single-node and multi-node training
- DeepSpeed ZeRO Stage 2
- Multi-phase training (-1, 0, 1, 2, 3)
- Resume from checkpoint

Usage:
    Single node:
        python train.py --deepspeed_config configs/ds_config_zero2.json

    Multi-node (torchrun):
        torchrun --nnodes=16 --nproc_per_node=4 train.py \
            --deepspeed_config configs/ds_config_zero2_multinode.json

    Multi-node (SLURM):
        sbatch scripts/brain_moe_multinode.sh
"""

import os
import sys
import argparse
import json
import signal
from pathlib import Path

import torch

# Allow importing brain_moe_pinn from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_moe_pinn import BrainMoEPINN, BrainMoEPINNConfig
from brain_moe_pinn.utils.training_loop import BrainMoETrainer
from brain_moe_pinn.utils.training_phases import (
    NEGATIVE_ONE_PHASE,
    STAGE_ONE_PHASE,
    STAGE_TWO_PHASE,
    STAGE_THREE_PHASE,
)

# Slurm preemption signal handler
_trainer_ref = None
_preemption_received = False

def _preemption_handler(signum, frame):
    global _preemption_received
    _preemption_received = True
    print(f"\n[PREEMPTION] Received signal {signum} (SIGTERM={signal.SIGTERM}). Saving checkpoint...")
    if _trainer_ref is not None:
        try:
            step = getattr(_trainer_ref, '_current_step', 0) or 0
            _trainer_ref.save_checkpoint(step=step, phase="preemption")
            print("[PREEMPTION] Checkpoint saved successfully")
        except Exception as e:
            print(f"[PREEMPTION] Failed to save checkpoint: {e}")
    sys.exit(128 + signum)


def _install_preemption_handler(trainer):
    global _trainer_ref
    _trainer_ref = trainer
    for sig in [signal.SIGTERM, getattr(signal, 'SIGINT', 2), getattr(signal, 'SIGUSR1', 10)]:
        try:
            signal.signal(sig, _preemption_handler)
        except (ValueError, AttributeError):
            pass


def parse_phases(phase_str: str):
    """Parse phase string like '-1,1,2,3' into phase configs."""
    phase_map = {
        "-1": NEGATIVE_ONE_PHASE,
        "1": STAGE_ONE_PHASE,
        "2": STAGE_TWO_PHASE,
        "3": STAGE_THREE_PHASE,
    }
    phases = []
    for p in phase_str.split(","):
        p = p.strip()
        if p in phase_map:
            phases.append(phase_map[p])
        else:
            raise ValueError(f"Unknown phase: {p}. Available: {list(phase_map.keys())}")
    return phases


def main():
    parser = argparse.ArgumentParser(description="Brain MoE-PINN Training")
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_config_zero2.json",
                        help="DeepSpeed config JSON file")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (per phase)")
    parser.add_argument("--phase", type=str, default="-1,1,2,3",
                        help="Training phases to run, e.g. '-1,1,2,3'")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Logging directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                        help="Local rank (set by torchrun/deepspeed)")
    parser.add_argument("--eeg_channels", type=int, default=19)
    parser.add_argument("--fmri_regions", type=int, default=400)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--use_kda_decoder", action="store_true", default=True,
                        help="Use KDA-based decoders")
    parser.add_argument("--no_kda_decoder", action="store_false", dest="use_kda_decoder",
                        help="Disable KDA decoders")
    parser.add_argument("--use_active_inference", action="store_true", default=True)
    parser.add_argument("--no_active_inference", action="store_false", dest="use_active_inference")
    parser.add_argument("--use_neurostorm", action="store_true", default=True,
                        help="Use NeuroSTORM fMRI encoder")
    parser.add_argument("--no_neurostorm", action="store_false", dest="use_neurostorm")
    parser.add_argument("--use_mamba2", action="store_true", default=False,
                        help="Use Mamba-2 SSM for context expansion in encoders")
    parser.add_argument("--mamba2_kwargs", type=str, default=None,
                        help="JSON string with Mamba-2 kwargs (e.g. '{\"chunk_size\":256}')")
    parser.add_argument("--tau_delay", type=float, default=0.5)
    parser.add_argument("--use_torch_compile", action="store_true", default=False,
                        help="Enable torch.compile for MoE experts and encoder")
    parser.add_argument("--use_deep_experts", action="store_true", default=False,
                        help="Use DeepExpertNetwork (50-layer) instead of ExpertNetwork (2-layer)")
    parser.add_argument("--shared_depth", type=int, default=50,
                        help="Number of layers in shared deep experts")
    parser.add_argument("--routed_depth", type=int, default=14,
                        help="Number of layers in routed deep experts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eeg_checkpoint", type=str, default=None,
                        help="Path to pretrained Magi EEG encoder checkpoint (.pt)")
    parser.add_argument("--fmri_checkpoint", type=str, default=None,
                        help="Path to pretrained NeuroSTORM/BrainLM fMRI checkpoint (.pt/.safetensors)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # Load DeepSpeed config
    ds_config_path = Path(args.deepspeed_config)
    if not ds_config_path.is_absolute():
        ds_config_path = Path(__file__).parent.parent / ds_config_path

    with open(ds_config_path) as f:
        deepspeed_config = json.load(f)

    # Parse phases
    phases = parse_phases(args.phase)

    # Parse Mamba-2 kwargs
    mamba2_kwargs = None
    if args.mamba2_kwargs:
        try:
            mamba2_kwargs = json.loads(args.mamba2_kwargs)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse --mamba2_kwargs: {args.mamba2_kwargs}")

    # Create model config
    config = BrainMoEPINNConfig(
        eeg_channels=args.eeg_channels,
        fmri_regions=args.fmri_regions,
        latent_dim=args.latent_dim,
        use_neurostorm=args.use_neurostorm,
        use_kda_decoder=args.use_kda_decoder,
        use_active_inference=args.use_active_inference,
        use_mamba2=args.use_mamba2,
        mamba2_kwargs=mamba2_kwargs,
        tau_delay=args.tau_delay,
        use_torch_compile=args.use_torch_compile,
        use_deep_experts=args.use_deep_experts,
        shared_depth=args.shared_depth,
        routed_depth=args.routed_depth,
    )
    model = config.to_model()

    if args.eeg_checkpoint or args.fmri_checkpoint:
        model.load_pretrained(
            eeg_checkpoint=args.eeg_checkpoint,
            fmri_checkpoint=args.fmri_checkpoint,
        )

    # Initialize trainer
    trainer = BrainMoETrainer(
        model=model,
        config=vars(args),
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        deepspeed_config=deepspeed_config,
    )

    # Install preemption handler for SLURM SIGTERM/USR1
    _install_preemption_handler(trainer)

    # Run training
    trainer.train(phases=phases, resume_from=args.resume)


if __name__ == "__main__":
    main()
