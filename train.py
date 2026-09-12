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
        sbatch brain_moe_multinode.sh
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

from brain_moe_pinn import BrainMoEPINNConfig
from brain_moe_pinn.config import load_experiment_config
from brain_moe_pinn.data import EEGDenoiseNetDataset, PairedBrainDataset
from brain_moe_pinn.data.species_dataset import build_species_dataloaders
from brain_moe_pinn.training.training_loop import BrainMoETrainer
from brain_moe_pinn.training.training_loop import (
    apply_precision,
    rectify_deepspeed_config,
    _PRECISION_DTYPES,
)
from brain_moe_pinn.training.training_phases import (
    STAGE_TWO_PHASE,
    STAGE_THREE_PHASE,
    get_phase_neg_1,
    get_stage_1_p1,
    get_stage_1_p2,
    get_stage_1_p3,
    get_stage_1_p4,
    get_stage_1_p5,
    get_stage_1_p6,
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


def build_data_loaders(profile_path: Path, leave_subject_out=None, *,
                       default_batch_size=None, default_seq_seconds=None,
                       profile_rate_hz=None, rate_hz_source="profile",
                       species=None):
    """Construct train/val loaders from a --data JSON profile.

    ``leave_subject_out`` comes from the species/training profile and binds to
    the split policy: ``True`` forces subject-grouped splits, ``False`` allows
    plain row shuffling (``by="none"``).  An explicit ``split_by`` in the data
    profile still wins.  Previously this flag was declared in every species
    profile and read nowhere, so a configured leakage control silently did
    nothing.

    Profiles:
      {"kind": "paired", "root": "<dir with eeg/, fmri/, meg?>",
       "batch_size": 16, "num_workers": 4}
      {"kind": "species", "root": "<dir with <modality>/ arrays>",
       "modalities": ["calcium", "voltage"], "batch_size": 8,
       "seq_len": 256, "num_workers": 0}
      {"kind": "eegdenoisenet", "root": "<EEGdenoiseNet checkout>",
       "artifact": "EOG", "batch_size": 32, "num_workers": 0}

    Species profiles stay authoritative for quantities they already declare:
    ``batch_size`` and ``sequence_seconds`` are used when the data profile omits
    them, and ``sample_rate_hz`` becomes the validated rate contract unless the
    species profile sets ``sample_rate_hz_source="manifest"`` (per-recording
    rates, e.g. per-worm C. elegans imaging).

    Returns (train_loader, val_loader); paired batches are dicts with
    eeg/fmri/(meg) keys, species batches map modality -> (B, C, T), and
    EEGdenoiseNet batches provide clean/noisy EEG views for Magi pretraining.
    """
    with open(profile_path) as f:
        profile = json.load(f)
    kind = profile.get("kind")
    if kind not in ("paired", "species", "eegdenoisenet"):
        raise ValueError(
            "--data profile needs \"kind\": \"paired\", \"species\", or "
            "\"eegdenoisenet\"")
    root = Path(profile["root"])
    if not root.is_absolute():
        root = profile_path.parent / root
    batch_size = (int(profile["batch_size"])
                  if profile.get("batch_size") is not None else None)
    num_workers = int(profile.get("num_workers", 4))
    if kind == "eegdenoisenet":
        snr_db_range = tuple(
            profile.get("snr_db_range", (-7.0, 2.0)))
        dataset_kwargs = dict(
            root=str(root),
            artifact=profile.get("artifact", "EOG"),
            train_fraction=float(profile.get("train_fraction", 0.8)),
            val_fraction=float(profile.get("val_fraction", 0.1)),
            seed=int(profile.get("seed", 0)),
            snr_db_range=snr_db_range,
        )
        train_set = EEGDenoiseNetDataset(split="train", **dataset_kwargs)
        val_set = EEGDenoiseNetDataset(split="val", **dataset_kwargs)
        common = dict(
            batch_size=int(batch_size if batch_size is not None
                           else (default_batch_size or 32)),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=EEGDenoiseNetDataset.collate_fn,
        )
        return (
            torch.utils.data.DataLoader(
                train_set,
                shuffle=True,
                drop_last=bool(profile.get("drop_last", False)),
                **common,
            ),
            torch.utils.data.DataLoader(
                val_set,
                shuffle=False,
                drop_last=False,
                **common,
            ),
        )

    if kind == "paired":
        paired_next = bool(profile.get(
            "return_next_step_targets",
            profile.get("paired_next_step_targets", False)))
        future_steps = max(1, int(profile.get("future_steps", 1)))
        if future_steps > 1 and not paired_next:
            raise ValueError(
                "future_steps > 1 requires return_next_step_targets=true "
                "or paired_next_step_targets=true")
        alignment_file = profile.get("alignment_file")
        if alignment_file:
            alignment_file = Path(alignment_file)
            alignment_file = (
                alignment_file if alignment_file.is_absolute()
                else profile_path.parent / alignment_file)
        dataset = PairedBrainDataset(
            data_dir=root, alignment_file=alignment_file,
            return_next_step_targets=paired_next,
            future_steps=future_steps)
        if len(dataset) < 2:
            raise ValueError(f"paired dataset at {root} has < 2 aligned samples")

        group_by = profile.get("group_by", "subject")
        seed = int(profile.get("seed", 0))
        groups = {}
        for index, entry in enumerate(dataset.pair_list):
            sample_id = str(entry.get("sample_id", index))
            if group_by in entry:
                group = str(entry[group_by])
            elif group_by == "subject":
                group = sample_id.split("_", 1)[0]
            elif group_by in (None, "none"):
                group = str(index)
            else:
                group = sample_id
            groups.setdefault(group, []).append(index)
        ordered_groups = list(groups)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(ordered_groups), generator=generator).tolist()
        ordered_groups = [ordered_groups[i] for i in order]
        val_frac = float(profile.get("val_frac", 0.1))
        test_frac = float(profile.get("test_frac", 0.0))
        n_val = max(1, int(round(len(ordered_groups) * val_frac)))
        n_test = (max(1, int(round(len(ordered_groups) * test_frac)))
                  if test_frac > 0 and len(ordered_groups) >= 3 else 0)
        while n_val + n_test >= len(ordered_groups):
            if n_test > 0:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        val_groups = ordered_groups[:n_val]
        test_groups = ordered_groups[n_val:n_val + n_test]
        train_groups = ordered_groups[n_val + n_test:]
        index_for = lambda selected: [
            index for group in selected for index in groups[group]]
        train_set = torch.utils.data.Subset(dataset, index_for(train_groups))
        val_set = torch.utils.data.Subset(dataset, index_for(val_groups))
        test_set = torch.utils.data.Subset(dataset, index_for(test_groups))
        common = dict(
            batch_size=int(batch_size if batch_size is not None
                           else (default_batch_size or 16)),
            num_workers=num_workers, pin_memory=True)
        loaders = (
            torch.utils.data.DataLoader(
                train_set, shuffle=True, drop_last=True, **common),
            torch.utils.data.DataLoader(
                val_set, shuffle=False, drop_last=False, **common),
        )
        if len(test_set):
            loaders += (torch.utils.data.DataLoader(
                test_set, shuffle=False, drop_last=False, **common),)
        return loaders
    seq_len = int(profile["seq_len"]) if profile.get("seq_len") else None
    seq_seconds = (float(profile["seq_seconds"])
                   if profile.get("seq_seconds") else None)
    if seq_len is None and seq_seconds is None:
        if default_seq_seconds is None:
            raise ValueError(
                "species data profile sets neither seq_len (frames) nor "
                "seq_seconds, and the species profile declares no "
                "sequence_seconds either")
        seq_seconds = float(default_seq_seconds)
    if batch_size is None:
        batch_size = int(default_batch_size) if default_batch_size else 16
    if rate_hz_source == "manifest":
        expected_rate_hz = None
    else:
        expected_rate_hz = float(
            profile.get("expected_rate_hz") or profile_rate_hz or 0.0) or None

    loader_kwargs = dict(
        modalities=profile.get("modalities", ("calcium", "voltage")),
        batch_size=batch_size,
        seq_len=seq_len,
        seq_seconds=seq_seconds,
        species=species or profile.get("species"),
        expected_rate_hz=expected_rate_hz,
        num_workers=num_workers,
        seed=int(profile.get("seed", 0)),
        val_frac=float(profile.get("val_frac", 0.1)),
        split_by=profile.get(
            "split_by",
            "subject" if leave_subject_out in (None, True) else "none"),
        align_channels=bool(profile.get("align_channels", False)),
        max_union_channels=int(profile.get("max_union_channels", 4096)),
        use_trials=bool(profile.get("use_trials", True)),
        rate_default=(float(profile["rate_default"])
                      if profile.get("rate_default") is not None else None),
        return_next_step_targets=bool(profile.get(
            "return_next_step_targets",
            profile.get("paired_next_step_targets", False))),
        future_steps=max(1, int(profile.get("future_steps", 1))),
        roles=profile.get("roles"),
        normalization=profile.get("normalization"),
    )
    if loader_kwargs["future_steps"] > 1 and not loader_kwargs[
            "return_next_step_targets"]:
        raise ValueError(
            "future_steps > 1 requires return_next_step_targets=true "
            "or paired_next_step_targets=true")
    if profile.get("test_frac"):
        loader_kwargs["test_frac"] = float(profile["test_frac"])
    if profile.get("region_map"):
        region = Path(profile["region_map"])
        loader_kwargs["region_map"] = (
            region if region.is_absolute() else profile_path.parent / region)
    loaders = build_species_dataloaders(root, **loader_kwargs)
    if len(loaders) == 3:
        print(f"[Data] Test loader built ({len(loaders[2].dataset)} samples)")
    return loaders


def parse_phases(phase_str: str):
    """Parse CLI selections into normalized runtime phase mappings.

    ``1`` is intentionally a selector for the concrete Stage 1 P1-P6
    schedule, never a synthetic flat phase with one averaged configuration.
    """
    selectors = {
        "-1": (get_phase_neg_1,),
        "1": (
            get_stage_1_p1, get_stage_1_p2, get_stage_1_p3,
            get_stage_1_p4, get_stage_1_p5, get_stage_1_p6,
        ),
        "2": (lambda: STAGE_TWO_PHASE,),
        "3": (lambda: STAGE_THREE_PHASE,),
    }
    phases = []
    tokens = [token.strip() for token in phase_str.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("phase selection must contain non-empty selectors")
    for token in tokens:
        factories = selectors.get(token)
        if factories is None:
            raise ValueError(
                f"Unknown phase: {token}. Available: {list(selectors)}")
        phases.extend(factory().to_runtime_config() for factory in factories)
    return phases


def main():
    parser = argparse.ArgumentParser(description="Brain MoE-PINN Training")
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_config_zero2.json",
                        help="DeepSpeed config JSON file")
    parser.add_argument("--precision", type=str, default=None,
                        choices=sorted(_PRECISION_DTYPES),
                        help="DeepSpeed autocast precision. Omit to use the "
                             "config's own dtype. 'bf16' avoids fp16's 65504 "
                             "ceiling and is required on TPU/XLA, which has no "
                             "fp16 compute path")
    parser.add_argument("--phase", type=str, default="1,2,3",
                        help="Training phases to run, e.g. '-1,1,2,3'; Phase -1 runs Magi v2 EEG pretraining and requires --eeg_backend v2")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Logging directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                        help="Local rank (set by torchrun/deepspeed)")
    parser.add_argument("--config", type=str, default=None,
                        help="Unified JSON experiment profile")
    parser.add_argument("--data", type=str, default=None,
                        help="JSON dataset profile: {\"kind\": "
                             "\"paired\"|\"species\"|\"eegdenoisenet\", ...}")
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
    parser.add_argument("--species", type=str, default="human")
    parser.add_argument("--eeg_backend", choices=("v1", "v2"), default="v1")
    parser.add_argument("--use_magi_v2", action="store_true", default=False)
    parser.add_argument("--max_channels", type=int, default=256)
    parser.add_argument("--ecog_amplitude_scale", type=float, default=20.0)
    parser.add_argument("--use_meg", action="store_true", default=False)
    parser.add_argument("--noise_mode", choices=("off", "rollout", "train", "always"),
                        default="off",
                        help="SDE noise policy (default off = deterministic steps)")
    parser.add_argument("--use_imagination", action="store_true", default=False)
    parser.add_argument("--use_generic_moe", action="store_true", default=False)
    parser.add_argument("--use_species_conditioning", action="store_true", default=False)
    parser.add_argument("--perturbation_dim", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # Load DeepSpeed config
    ds_config_path = Path(args.deepspeed_config)
    if not ds_config_path.is_absolute():
        ds_config_path = Path(__file__).parent / ds_config_path

    with open(ds_config_path) as f:
        deepspeed_config = json.load(f)
    # The shipped configs' train_batch_size/micro_batch did not match any
    # launch topology; they are rebuilt below from the real loader batch
    # and world size, once both are known.

    # Parse phases
    phases = parse_phases(args.phase)

    # Parse Mamba-2 kwargs
    mamba2_kwargs = None
    if args.mamba2_kwargs:
        try:
            mamba2_kwargs = json.loads(args.mamba2_kwargs)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse --mamba2_kwargs: {args.mamba2_kwargs}")

    # Create the canonical model configuration. A profile is authoritative;
    # legacy flags remain available when no profile is supplied.
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path
        config = BrainMoEPINNConfig.from_file(str(config_path))
    else:
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
            use_meg=args.use_meg,
            use_imagination=args.use_imagination,
            use_generic_moe=args.use_generic_moe,
            perturbation_dim=args.perturbation_dim,
            eeg_backend=args.eeg_backend,
            use_magi_v2=args.use_magi_v2,
            max_channels=args.max_channels,
            ecog_amplitude_scale=args.ecog_amplitude_scale,
            species=args.species,
            use_species_conditioning=args.use_species_conditioning,
            noise_mode=args.noise_mode,
        )
    magi_pretraining_selected = any(
        phase.get("task") == "magi_eeg_pretraining" for phase in phases)
    if magi_pretraining_selected and getattr(config, "eeg_backend", "v1") != "v2":
        raise ValueError(
            "Phase -1 requires the v2 EEG backend; pass --eeg_backend v2 "
            "or --use_magi_v2")
    model = config.to_model()

    if args.eeg_checkpoint or args.fmri_checkpoint:
        model.load_pretrained(
            eeg_checkpoint=args.eeg_checkpoint,
            fmri_checkpoint=args.fmri_checkpoint,
        )

    # Initialize trainer
    trainer_config = vars(args).copy()
    trainer_config["model_config"] = dict(config.__dict__)
    trainer_config["magi_pretraining_enabled"] = magi_pretraining_selected
    if args.config:
        experiment = load_experiment_config(str(config_path))
        trainer_config["experiment_data"] = {
            "modalities": list(experiment.data.modalities),
            "sample_rate_hz": experiment.data.sample_rate_hz,
            "sequence_seconds": experiment.data.sequence_seconds,
            "paired_next_step_targets": (
                experiment.data.paired_next_step_targets),
            "roles": dict(experiment.data.roles),
            "control_specs": dict(experiment.data.control_specs),
        }
    train_loader = None
    val_loader = None
    test_loader = None
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = Path(__file__).parent / data_path
        loaders = build_data_loaders(
            data_path,
            experiment.training.leave_subject_out if args.config else None,
            default_batch_size=(experiment.training.batch_size
                                if args.config else None),
            default_seq_seconds=(experiment.data.sequence_seconds
                                 if args.config else None),
            profile_rate_hz=(experiment.data.sample_rate_hz
                             if args.config else None),
            rate_hz_source=(experiment.data.sample_rate_hz_source
                            if args.config else "profile"),
            species=(experiment.species if args.config else None))
        train_loader, val_loader = loaders[:2]
        if len(loaders) == 3:
            test_loader = loaders[2]
        # Real loaders are used verbatim; DataMixer/auto-splitting applies to
        # the scaffold path only.
        trainer_config["use_mixer"] = False
        with open(data_path) as fh:
            data_profile = json.load(fh)
        if data_profile.get("roles"):
            trainer_config["modality_roles"] = dict(data_profile["roles"])
            trainer_config.setdefault("experiment_data", {})["roles"] = dict(
                data_profile["roles"])
        if data_profile.get("return_next_step_targets") or data_profile.get(
                "paired_next_step_targets"):
            trainer_config["require_next_step_targets"] = True
            trainer_config.setdefault("experiment_data", {})[
                "paired_next_step_targets"] = True
        if data_profile.get("future_steps") is not None:
            future_steps = max(1, int(data_profile["future_steps"]))
            trainer_config["future_steps"] = future_steps
            trainer_config.setdefault("experiment_data", {})[
                "future_steps"] = future_steps
        if data_profile.get("control_modalities"):
            trainer_config["control_modalities"] = tuple(
                data_profile["control_modalities"])
        if data_profile.get("control_reduction"):
            trainer_config["control_reduction"] = str(
                data_profile["control_reduction"])
        if data_profile.get("rollout_steps"):
            trainer_config["rollout_steps"] = int(data_profile["rollout_steps"])
        summary = (f"{len(train_loader.dataset)} train / "
                   f"{len(val_loader.dataset)} val")
        if test_loader is not None:
            summary += f" / {len(test_loader.dataset)} test"
        print(f"[Data] Loaders built from {data_path} ({summary} samples)")
    # DeepSpeed's batch assertion is only satisfiable if the micro batch
    # equals what the loader yields and the accumulation matches the phase.
    deepspeed_config = apply_precision(deepspeed_config, args.precision)
    deepspeed_config = rectify_deepspeed_config(
        deepspeed_config,
        micro_batch=getattr(train_loader, 'batch_size', None),
        gradient_accumulation=phases[0].get('gradient_accumulation', 1)
        if phases else 1,
        world_size=int(os.environ.get('WORLD_SIZE', 1)),
    )

    trainer = BrainMoETrainer(
        model=model,
        config=trainer_config,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        deepspeed_config=deepspeed_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
    )

    # Install preemption handler for SLURM SIGTERM/USR1
    _install_preemption_handler(trainer)

    # Run training
    trainer.train(phases=phases, resume_from=args.resume)


if __name__ == "__main__":
    main()
