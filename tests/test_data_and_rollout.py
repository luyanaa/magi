"""Regression checks: data loaders (paired+MEG, species dict) and rollout
temporal-T velocity supervision.
"""

import pytest
from pathlib import Path

torch = pytest.importorskip("torch")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.data import PairedBrainDataset
from brain_moe_pinn.data.species_dataset import (
    SpeciesSignalDataset, build_species_dataloaders,
)
from brain_moe_pinn.training.losses import TotalLoss, VelocitySmoothnessLoss
from brain_moe_pinn.training.training_phases import LossWeights


def _write_npy(directory: Path, stem: str, array) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{stem}.npy", array)


def test_species_dataset_windows_and_batches(tmp_path):
    arrays_by_modality = {
        "calcium": [np.random.randn(4, 40), np.random.randn(4, 16),
                    np.random.randn(4, 32), np.random.randn(4, 24)],
        "voltage": [np.random.randn(8, 40), np.random.randn(8, 16),
                    np.random.randn(8, 32), np.random.randn(8, 24)],
    }
    for modality, arrays in arrays_by_modality.items():
        for i, arr in enumerate(arrays):
            _write_npy(tmp_path / modality, f"sample{i}", arr)

    ds = SpeciesSignalDataset(tmp_path, ["calcium", "voltage"], seq_len=32)
    assert len(ds) == 4
    item = ds[1]  # short file is padded
    assert item["calcium"].shape == (4, 32)
    assert item["voltage"].shape == (8, 32)
    assert torch.isfinite(item["calcium"]).all()

    train_loader, val_loader = build_species_dataloaders(
        tmp_path, ["calcium", "voltage"], batch_size=2, seq_len=32)
    batch = next(iter(train_loader))
    assert batch["calcium"].shape == (2, 4, 32)
    assert batch["voltage"].shape == (2, 8, 32)
    assert len(val_loader.dataset) >= 0


def test_paired_dataset_with_optional_meg(tmp_path):
    root = tmp_path / "paired"
    _write_npy(root / "eeg", "sub1", np.random.randn(4, 64))
    _write_npy(root / "fmri", "sub1", np.random.randn(400, 20))
    _write_npy(root / "meg", "sub1", np.random.randn(306, 64))
    _write_npy(root / "eeg", "sub2", np.random.randn(4, 64))
    _write_npy(root / "fmri", "sub2", np.random.randn(400, 20))

    ds = PairedBrainDataset(data_dir=root)
    assert len(ds) == 2
    first = ds[0]
    second = ds[1]
    assert first["eeg"].shape == (4, 64)
    assert first["meg"].shape == (306, 64)
    assert "meg" not in second
    assert first["is_paired"] and second["is_paired"]


def _make_model():
    torch.manual_seed(0)
    return BrainMoEPINN(
        eeg_channels=2,
        fmri_regions=3,
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        use_species_conditioning=True,
        species="c_elegans",
        species_vocab=["c_elegans", "human"],
        perturbation_dim=2,
        moe_num_shared=1,
        moe_num_routed=1,
        moe_top_k=1,
        generic_observation_only=True,
    ).eval()


def test_model_emits_rollout_velocity_sequence():
    model = _make_model()
    signals = {"calcium": torch.randn(2, 12, 64)}
    out = model.forward_modalities(signals, num_steps=3)
    assert "delta_z_sequence" in out
    assert out["delta_z_sequence"].shape == (2, 3, 8)
    # Single-step calls do not emit a (useless) length-1 sequence.
    out1 = model.forward_modalities(signals, num_steps=1)
    assert "delta_z_sequence" not in out1


def test_velocity_smoothness_is_temporal_tv():
    loss = VelocitySmoothnessLoss(alpha=0.5)

    # Constant velocity rollout -> zero temporal TV.
    flat = torch.zeros(2, 4, 8)
    assert loss(flat).item() == 0.0

    # Varying rollout -> mean L1 of successive velocity differences.
    seq = torch.randn(2, 4, 8)
    manual = torch.norm(seq[:, 1:] - seq[:, :-1], p=1, dim=-1).mean()
    assert torch.allclose(loss(seq), 0.5 * manual, atol=1e-6)

    # Single-step velocities have no temporal axis -> zero (legacy batch-TV
    # semantics removed).
    assert loss(torch.randn(2, 8)).item() == 0.0


def _load_train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_entry", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_data_loaders_paired_profile(tmp_path):
    train_mod = _load_train_module()
    root = tmp_path / "paired_root"
    for stem in ("s1", "s2", "s3", "s4"):
        _write_npy(root / "eeg", stem, np.random.randn(4, 64))
        _write_npy(root / "fmri", stem, np.random.randn(400, 20))
        _write_npy(root / "meg", stem, np.random.randn(306, 64))
    profile = tmp_path / "paired.json"
    profile.write_text('{"kind": "paired", "root": "paired_root", '
                       '"batch_size": 2, "num_workers": 0}')
    train_loader, val_loader = train_mod.build_data_loaders(profile)
    batch = next(iter(train_loader))
    assert set(batch) >= {"eeg", "fmri", "meg", "is_paired"}
    assert batch["meg"].shape == (2, 306, 64)
    assert len(val_loader) >= 1


def test_build_data_loaders_species_profile(tmp_path):
    train_mod = _load_train_module()
    root = tmp_path / "species_root"
    for modality in ("calcium", "voltage"):
        for i in range(4):
            _write_npy(root / modality, f"w{i}",
                       np.random.randn(146, 300))
    profile = tmp_path / "species.json"
    profile.write_text('{"kind": "species", "root": "species_root", '
                       '"modalities": ["calcium", "voltage"], '
                       '"batch_size": 2, "seq_len": 256}')
    train_loader, val_loader = train_mod.build_data_loaders(profile)
    batch = next(iter(train_loader))
    assert batch["calcium"].shape == (2, 146, 256)
    assert batch["voltage"].shape == (2, 146, 256)


def test_total_loss_fires_temporal_tv_on_rollouts():
    model = _make_model().train()
    signals = {"calcium": torch.randn(2, 12, 64)}
    out = model.forward_modalities(signals, num_steps=4)
    weights = LossWeights(velocity_smooth=0.5, sigreg=0.0)
    for field in ("recon_eeg", "recon_fmri", "cross_modal", "dissip",
                  "generic_constraint", "moe_load_balance",
                  "grassmannian_reg", "spectrum", "bandpower", "cross"):
        setattr(weights, field, 0.0)
    tl = TotalLoss(weights)
    total, metrics = tl(out, {})
    seq = out["delta_z_sequence"]
    manual = torch.norm(seq[:, 1:] - seq[:, :-1], p=1, dim=-1).mean()
    # Metrics expose the raw component term; phase weights are applied only
    # to the aggregate so a configured weight is not double-counted.
    assert "velocity_smooth" in metrics
    assert torch.allclose(
        torch.tensor(metrics["velocity_smooth"]), manual, atol=1e-6)
    assert torch.isfinite(total)
