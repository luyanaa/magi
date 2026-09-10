#!/usr/bin/env python3
"""Real-data sanity harness: species ladder arrays -> forward+backward run.

Takes converted real recordings (see ``data/ingest_c_elegans.py`` output:
``<root>/calcium/<id>.npy`` etc.) and proves the whole stack on real data:
species config -> canonical model -> forward_modalities(reconstruct=True)
-> species reconstruction criteria -> backward.

Usage::

    python tools/real_data_sanity.py \\
        --config configs/species/c_elegans.json \\
        --root /path/to/converted_ladder \\
        [--seq 256 --batch 2 --steps 1 --rollout 2]

Cluster training itself runs through ``train.py`` with a ``--data`` profile
built from the same ladder root; this harness is the CPU-gated equivalent.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from brain_moe_pinn import BrainMoEPINNConfig
from brain_moe_pinn.config import SPECIES_PROFILES, SUPPORTED_MODALITIES
from brain_moe_pinn.data.species_dataset import build_species_dataloaders
from brain_moe_pinn.training.losses import TotalLoss
from brain_moe_pinn.training.training_loop import augment_phase_loss_weights
from brain_moe_pinn.training.training_phases import LossWeights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="species JSON profile")
    parser.add_argument("--root", required=True, help="converted ladder root")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--rollout", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = BrainMoEPINNConfig.from_file(args.config)
    model = cfg.to_model().train()
    print(f"[sanity] species={cfg.species} latent_dim={cfg.latent_dim} "
          f"latent_dt={cfg.latent_dt}")

    root = Path(args.root)
    detected = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and d.name in SUPPORTED_MODALITIES)
    if not detected:
        raise SystemExit(f"no supported modality dirs under {root}")
    print(f"[sanity] detected modalities: {detected}")

    loader, _ = build_species_dataloaders(
        root, detected, batch_size=args.batch, seq_len=args.seq,
        num_workers=0, shuffle=False)
    profile = SPECIES_PROFILES.get(cfg.species)
    criteria = dict(profile.recon_loss_types) if profile else {}
    recon_modalities = tuple(m for m in detected if m != "behavior")
    weights = augment_phase_loss_weights(
        LossWeights(), recon_modalities, criteria)

    for step in range(max(1, args.steps)):
        signals = next(iter(loader))
        signals = {k: v for k, v in signals.items()}
        out = model.forward_modalities(
            signals, num_steps=max(1, args.rollout), reconstruct=True)
        targets = {m: signals[m] for m in recon_modalities if m in signals}
        total, metrics = TotalLoss(weights)(out, targets)
        total.backward()
        finite = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None)
        recon_metrics = {k: round(v, 6) for k, v in metrics.items()
                         if k.startswith("recon_")}
        print(f"[sanity] step {step}: total={total.item():.4f} "
              f"finite_grads={finite} recon={recon_metrics} "
              f"criteria={criteria}")
        if not finite or not torch.isfinite(total):
            raise SystemExit("sanity FAILED: non-finite loss or gradients")

    print("[sanity] OK - real-data forward/backward with species criteria")
    print("[sanity] next: train.py --config %s --data <profile> "
          "(kind=species, root=%s)" % (args.config, root))


if __name__ == "__main__":
    main()
