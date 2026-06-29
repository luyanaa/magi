#!/usr/bin/env python3
"""
Brain MoE-PINN v2 Training Entry Point with Magi v2 support.

Supports:
- Magi v2 architecture (24L × 1024d)
- ECoG/sEEG multi-modality
- MoE Stage 0 from start (E=4)
- Revised token budgets based on MoE scaling laws
- Cross-modal sync with DANDI datasets
- Context length scheduling

Usage:
    Single node:
        python train_v2.py --deepspeed_config configs/ds_config_zero2.json

    Multi-node (torchrun):
        torchrun --nnodes=16 --nproc_per_node=4 train_v2.py \
            --deepspeed_config configs/ds_config_zero2_multinode.json

    Multi-node (SLURM):
        sbatch scripts/brain_moe_multinode_v2.sh
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path

import torch

# Allow importing brain_moe_pinn from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from brain_moe_pinn.brain_moe_pinn_v2 import BrainMoEPINNV2, BrainMoEPINNConfig
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    warnings.warn("BrainMoEPINNV2 not available, falling back to v1")
    from brain_moe_pinn import BrainMoEPINN

from brain_moe_pinn.utils.training_loop import BrainMoETrainer
from brain_moe_pinn.utils.training_phases import (
    NEGATIVE_ONE_PHASE,
    STAGE_ONE_PHASE,
    STAGE_TWO_PHASE,
    STAGE_THREE_PHASE,
)


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


def create_model(args, phase_config):
    """Create model based on phase configuration."""
    
    # Common kwargs
    common_kwargs = {
        "eeg_channels": args.eeg_channels,
        "fmri_regions": args.fmri_regions,
        "latent_dim": args.latent_dim,
        "use_kda_decoder": args.use_kda_decoder,
        "use_active_inference": args.use_active_inference,
        "use_neurostorm": args.use_neurostorm,
        "use_mamba2": args.use_mamba2,
        "eeg_checkpoint": args.eeg_checkpoint,
        "fmri_checkpoint": args.fmri_checkpoint,
    }
    
    # Mamba-2 kwargs
    if args.use_mamba2 and args.mamba2_kwargs:
        try:
            mamba2_kwargs = json.loads(args.mamba2_kwargs)
            common_kwargs["mamba2_kwargs"] = mamba2_kwargs
        except json.JSONDecodeError:
            warnings.warn(f"Invalid Mamba-2 kwargs JSON: {args.mamba2_kwargs}")
    
    # Phase-specific config
    phase_kwargs = phase_config.get("model_config", {})
    
    # Use v2 if available and requested
    use_v2 = args.use_magi_v2 and V2_AVAILABLE
    
    if use_v2:
        # Create v2 config
        config = BrainMoEPINNConfig(
            eeg_hidden_dim=1024,  # Magi v2
            eeg_num_layers=24,    # Magi v2
            latent_dim=args.latent_dim,
            fmri_regions=args.fmri_regions,
            use_magi_v2=True,
            max_channels=phase_kwargs.get("max_channels", 256),
            ecog_amplitude_scale=phase_kwargs.get("ecog_amplitude_scale", 20.0),
            use_channel_type_embed=True,
            moe_num_experts=phase_kwargs.get("moe_num_experts", 4),
            moe_top_k=phase_kwargs.get("moe_top_k", 2),
            freeze_encoders_epochs=phase_kwargs.get("freeze_encoders_epochs", 1),
            use_mamba2=args.use_mamba2,
            use_kda_decoder=args.use_kda_decoder,
            use_active_inference=args.use_active_inference,
            use_neurostorm=args.use_neurostorm,
            initial_context_length=phase_kwargs.get("initial_context_length", 256),
            max_context_length=phase_kwargs.get("max_context_length", 1024),
        )
        
        # Create v2 model
        model = BrainMoEPINNV2(
            config=config,
            eeg_channels=args.eeg_channels,
            eeg_checkpoint=args.eeg_checkpoint,
            fmri_checkpoint=args.fmri_checkpoint,
        )
        
        print(f"[train_v2] Created BrainMoEPINNV2 with Magi v2 architecture")
        
    else:
        # Fallback to v1
        if not V2_AVAILABLE and args.use_magi_v2:
            warnings.warn("Magi v2 requested but not available. Using v1.")
        
        # Use v1 model
        model = BrainMoEPINN(
            eeg_channels=args.eeg_channels,
            fmri_regions=args.fmri_regions,
            latent_dim=args.latent_dim,
            use_neurostorm=args.use_neurostorm,
            use_kda_decoder=args.use_kda_decoder,
            use_active_inference=args.use_active_inference,
            use_mamba2=args.use_mamba2,
            mamba2_kwargs=common_kwargs.get("mamba2_kwargs"),
            tau_delay=args.tau_delay,
            freeze_encoders_epochs=phase_kwargs.get("freeze_encoders_epochs", 1),
        )
        
        print(f"[train_v2] Created BrainMoEPINN v1 (fallback)")
    
    # Load pretrained checkpoints
    if args.eeg_checkpoint and os.path.exists(args.eeg_checkpoint):
        try:
            if use_v2:
                model.load_pretrained(eeg_checkpoint=args.eeg_checkpoint, strict=False)
            else:
                # v1 loading handled in constructor
                pass
            print(f"[train_v2] Loaded EEG checkpoint: {args.eeg_checkpoint}")
        except Exception as e:
            warnings.warn(f"Failed to load EEG checkpoint: {e}")
    
    if args.fmri_checkpoint and os.path.exists(args.fmri_checkpoint):
        try:
            if use_v2:
                model.load_pretrained(fmri_checkpoint=args.fmri_checkpoint, strict=False)
            else:
                # v1 loading handled in constructor
                pass
            print(f"[train_v2] Loaded fMRI checkpoint: {args.fmri_checkpoint}")
        except Exception as e:
            warnings.warn(f"Failed to load fMRI checkpoint: {e}")
    
    return model, use_v2


def main():
    parser = argparse.ArgumentParser(description="Brain MoE-PINN v2 Training")
    
    # DeepSpeed
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_config_zero2.json",
                        help="DeepSpeed config JSON file")
    
    # Training phases
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (per phase)")
    parser.add_argument("--phase", type=str, default="-1,0,1,2,3",
                        help="Training phases to run, e.g. '-1,0,1,2,3'")
    
    # Checkpoints
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--log_dir", type=str, default="./logs_v2",
                        help="Logging directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_v2",
                        help="Checkpoint directory")
    
    # Distributed
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                        help="Local rank (set by torchrun/deepspeed)")
    
    # Model architecture
    parser.add_argument("--eeg_channels", type=int, default=64,
                        help="Number of EEG channels (default 64 for ECoG)")
    parser.add_argument("--fmri_regions", type=int, default=400)
    parser.add_argument("--latent_dim", type=int, default=1024)
    
    # Magi v2
    parser.add_argument("--use_magi_v2", action="store_true", default=True,
                        help="Use Magi v2 architecture (24L×1024d)")
    parser.add_argument("--no_magi_v2", action="store_false", dest="use_magi_v2",
                        help="Disable Magi v2, use v1 instead")
    
    # Decoders
    parser.add_argument("--use_kda_decoder", action="store_true", default=True,
                        help="Use KDA-based decoders")
    parser.add_argument("--no_kda_decoder", action="store_false", dest="use_kda_decoder",
                        help="Disable KDA decoders")
    
    # Active inference
    parser.add_argument("--use_active_inference", action="store_true", default=True)
    parser.add_argument("--no_active_inference", action="store_false", dest="use_active_inference")
    
    # fMRI encoder
    parser.add_argument("--use_neurostorm", action="store_true", default=True,
                        help="Use NeuroSTORM fMRI encoder")
    parser.add_argument("--no_neurostorm", action="store_false", dest="use_neurostorm")
    
    # Mamba-2
    parser.add_argument("--use_mamba2", action="store_true", default=False,
                        help="Use Mamba-2 SSM for context expansion in encoders")
    parser.add_argument("--mamba2_kwargs", type=str, default=None,
                        help="JSON string with Mamba-2 kwargs (e.g. '{\"chunk_size\":256}')")
    
    # Checkpoints
    parser.add_argument("--eeg_checkpoint", type=str, default=None,
                        help="Path to pretrained Magi EEG encoder checkpoint (.pt)")
    parser.add_argument("--fmri_checkpoint", type=str, default=None,
                        help="Path to pretrained NeuroSTORM/BrainLM fMRI checkpoint")
    
    # ECoG settings
    parser.add_argument("--max_channels", type=int, default=256,
                        help="Maximum number of channels (for ECoG)")
    parser.add_argument("--ecog_amplitude_scale", type=float, default=20.0,
                        help="ECoG amplitude scaling factor (20.0 = 1/20 scaling)")
    
    # MoE settings
    parser.add_argument("--moe_num_experts", type=int, default=4,
                        help="Number of MoE experts (Stage 0: E=4)")
    parser.add_argument("--moe_top_k", type=int, default=2,
                        help="Top-k experts to activate")
    
    # Training
    parser.add_argument("--tau_delay", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_moe_stage0", action="store_true", default=True,
                        help="Use MoE from Stage 0 (instead of dense)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Parse phases - train_v2 uses the same parse function as train.py
    try:
        from train import parse_phases as _parse_phases
        phases = _parse_phases(args.phase)
    except ValueError as e:
        print(f"Error parsing phases: {e}")
        return 1

    print(f"[train_v2] Running phases: {args.phase}")
    print(f"[train_v2] Using Magi v2: {args.use_magi_v2}")
    print(f"[train_v2] MoE Stage 0: {args.use_moe_stage0}")
    print(f"[train_v2] ECoG support: max_channels={args.max_channels}, scale={args.ecog_amplitude_scale}")

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create model
    model, use_v2 = create_model(args, phase_config=phase_configs[0])

    # Create trainer - match BrainMoETrainer.__init__ signature
    ds_cfg = json.load(open(args.deepspeed_config)) if args.deepspeed_config else None
    trainer = BrainMoETrainer(
        model=model,
        config=vars(args),
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        deepspeed_config=ds_cfg,
    )

    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        print(f"[train_v2] Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run each phase
    for phase_idx, phase_config in enumerate(phases):
        phase_name = phase_config.get("name", f"phase_{phase_idx}")
        print(f"\n{'='*60}")
        print(f"Starting phase: {phase_name}")
        print(f"{'='*60}")

        try:
            trainer.train_phase(phase_config)
        except KeyboardInterrupt:
            print(f"[train_v2] Phase {phase_name} interrupted")
            checkpoint_path = trainer.save_checkpoint(step=trainer._current_step if hasattr(trainer, '_current_step') else 0, phase=phase_name)
            print(f"[train_v2] Saved checkpoint: {checkpoint_path}")
            break
        except Exception as e:
            print(f"[train_v2] Error in phase {phase_name}: {e}")
            import traceback
            traceback.print_exc()
            try:
                checkpoint_path = trainer.save_checkpoint(step=0, phase=f"error_{phase_name}")
                print(f"[train_v2] Saved checkpoint after error: {checkpoint_path}")
            except Exception:
                print(f"[train_v2] Could not save checkpoint after error")
            break

    print(f"[train_v2] Training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())