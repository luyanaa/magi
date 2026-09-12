#!/usr/bin/env python3
"""Real-data sanity harness: species ladder arrays -> forward+backward run.

Takes converted real recordings (see ``data/ingest_c_elegans.py`` output:
``<root>/calcium/<id>.npy`` etc.) and proves the whole stack on real data:
species config -> canonical model -> forward_modalities(reconstruct=True)
-> species reconstruction criteria -> backward, reported next to trivial
baselines (persistence / channel mean) so a correlation is interpretable.

Usage::

    python tools/real_data_sanity.py \\
        --config configs/species/c_elegans.json \\
        --root /path/to/converted_ladder \\
        [--seq 256 | --seq-seconds 60] [--batch 1] [--steps 1] [--rollout 1]

Batch default is 1: per-recording channel counts differ (146-226 for the
24-worm salt pilot), so a multi-sample batch is only possible with
``align_channels``/``region_map``.  The window defaults to the species
profile's ``sequence_seconds`` (converted per sample with that sample's own
sampling rate), never to a frame count borrowed from another species.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from brain_moe_pinn import BrainMoEPINNConfig
from brain_moe_pinn.config import SPECIES_PROFILES, SUPPORTED_MODALITIES
from brain_moe_pinn.config import ExperimentConfig
from brain_moe_pinn.data.species_dataset import build_species_dataloaders
from brain_moe_pinn.training.losses import TotalLoss
from brain_moe_pinn.training.training_loop import (
    augment_phase_loss_weights,
    masked_channel_correlation,
)
from brain_moe_pinn.training.training_phases import LossWeights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="species JSON profile")
    parser.add_argument("--root", required=True, help="converted ladder root")
    parser.add_argument("--seq", type=int, default=None,
                        help="window length in frames (overrides seconds)")
    parser.add_argument("--seq-seconds", type=float, default=None,
                        help="window length in seconds (default: profile)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--rollout", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    experiment = ExperimentConfig.from_file(args.config)
    cfg = BrainMoEPINNConfig.from_experiment(experiment)
    model = cfg.to_model().train()
    print(f"[sanity] species={cfg.species} latent_dim={cfg.latent_dim} "
          f"nominal_latent_dt={cfg.latent_dt} "
          f"rate_source={experiment.data.sample_rate_hz_source}")

    root = Path(args.root)
    detected = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and d.name in SUPPORTED_MODALITIES + ("stimulus",))
    if not detected:
        raise SystemExit(f"no supported modality dirs under {root}")
    print(f"[sanity] detected modalities: {detected}")

    roles = dict(experiment.data.roles)
    signal_modalities = tuple(
        m for m in detected if roles.get(m, "signal") == "signal")
    control_modalities = tuple(
        m for m in detected if roles.get(m) == "control")
    if not signal_modalities:
        raise SystemExit(f"no signal-role modalities among {detected}")

    seq_kwargs = ({"seq_len": args.seq} if args.seq is not None else {
        "seq_seconds": (args.seq_seconds
                        if args.seq_seconds is not None
                        else experiment.data.sequence_seconds)})
    print(f"[sanity] window: {seq_kwargs}, batch={args.batch}, "
          f"control={control_modalities}")
    loader, _ = build_species_dataloaders(
        root, tuple(detected), batch_size=args.batch, num_workers=0,
        shuffle=False, species=experiment.species,
        return_next_step_targets=True, roles=roles, **seq_kwargs)
    profile = SPECIES_PROFILES.get(cfg.species)
    criteria = dict(profile.recon_loss_types) if profile else {}
    recon_modalities = tuple(m for m in signal_modalities if m != "behavior")
    weights = augment_phase_loss_weights(
        LossWeights(), recon_modalities, criteria)

    for step in range(max(1, args.steps)):
        batch = next(iter(loader))
        signals = {m: batch[m] for m in signal_modalities if m in batch}
        masks = {m: batch[f"{m}_mask"] for m in signal_modalities
                 if f"{m}_mask" in batch}
        targets = {m: batch[f"{m}_next"] for m in recon_modalities
                   if f"{m}_next" in batch}
        for modality in recon_modalities:
            if f"{modality}_next_mask" in batch:
                targets[f"{modality}_mask"] = batch[f"{modality}_next_mask"]
        # One latent step advances one window: dt (seconds) per sample.
        frames = next(iter(signals.values())).shape[-1]
        dt = batch.get("dt")
        dt_kwargs = {}
        if isinstance(dt, (list, tuple)) and dt:
            dt_kwargs["dt"] = torch.tensor(
                [float(v) * frames for v in dt], dtype=torch.float32)
            print(f"[sanity] per-sample dt ({dt_kwargs['dt'].numel()}): "
                  f"{[round(float(v), 1) for v in dt_kwargs['dt']]} s "
                  f"for {frames} frames")

        out = model.forward_modalities(
            signals, masks=masks, num_steps=max(1, args.rollout),
            reconstruct=True, **dt_kwargs)
        total, metrics = TotalLoss(weights)(out, targets)
        total.backward()
        finite = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None)
        recon_metrics = {k: round(v, 6) for k, v in metrics.items()
                         if k.startswith("recon_")}
        baseline_metrics = {}
        for modality in recon_modalities:
            if modality in signals and modality in targets:
                observed = signals[modality]
                target = targets[modality]
                if observed.shape != target.shape:
                    continue
                mask = targets.get(f"{modality}_mask")
                baseline_metrics[f"{modality}/persistence"] = round(
                    masked_channel_correlation(observed, target, mask), 6)
                baseline_metrics[f"{modality}/channel_mean"] = round(
                    masked_channel_correlation(
                        observed.mean(dim=-1, keepdim=True).expand_as(target),
                        target, mask), 6)
                if f"{modality}_recon" in out:
                    baseline_metrics[f"{modality}/model"] = round(
                        masked_channel_correlation(
                            out[f"{modality}_recon"], target, mask), 6)
        print(f"[sanity] step {step}: total={total.item():.4f} "
              f"finite_grads={finite} criteria={criteria}")
        print(f"[sanity] loss recon terms: {recon_metrics}")
        print(f"[sanity] per-channel correlation (model vs trivial): "
              f"{baseline_metrics}")
        if not finite or not torch.isfinite(total):
            raise SystemExit("sanity FAILED: non-finite loss or gradients")

    print("[sanity] OK - real-data forward/backward with species criteria")
    print("[sanity] next: train.py --config %s --data %s "
          "(kind=species, root=%s)" % (
              args.config, "configs/data/c_elegans_salt.json", root))


if __name__ == "__main__":
    main()
