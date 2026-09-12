"""P0-P2.1 tests: stimulus control wiring, control gating, noise policy."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.core.velocity_brain import VelocityBrain
from brain_moe_pinn.data.species_dataset import SpeciesSignalDataset, build_species_dataloaders
from brain_moe_pinn.training.training_loop import (
    partition_generic_batch, reduce_control,
)


def _write_npy(directory: Path, stem: str, array) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{stem}.npy", array)


def _model(**kwargs):
    torch.manual_seed(0)
    defaults = dict(
        eeg_channels=2, fmri_regions=3, latent_dim=8, use_neurostorm=False,
        use_kda_decoder=False, use_active_inference=False,
        use_species_conditioning=True, species="c_elegans",
        species_vocab=["c_elegans", "human"], perturbation_dim=2,
        moe_num_shared=1, moe_num_routed=1, moe_top_k=1,
        generic_observation_only=True,
    )
    defaults.update(kwargs)
    return BrainMoEPINN(**defaults)


# ------------------------------------------------------------------ P1 gates
def test_zero_control_is_exact_noop_and_gates_identity():
    torch.manual_seed(0)
    vb = VelocityBrain(hidden_dim=8, num_poisson_layers=2, num_energy_layers=2,
                       poisson_rank=4, perturbation_dim=2,
                       use_lowrank_poisson=True).eval()
    # zero-init guarantees at construction
    last = vb.perturbation_map[-1]
    assert torch.count_nonzero(last.weight) == 0 and torch.count_nonzero(last.bias) == 0
    assert torch.count_nonzero(vb.gate_map[-1].weight) == 0

    z = torch.randn(3, 8)

    def fresh():
        torch.manual_seed(0)
        return VelocityBrain(hidden_dim=8, num_poisson_layers=2,
                             num_energy_layers=2, poisson_rank=4,
                             perturbation_dim=2,
                             use_lowrank_poisson=True).eval()

    # Stateful buffers (MT-KDA) make repeat calls differ; compare fresh,
    # identically-seeded instances instead.
    # At construction the control readout is zero-init (dead start): both
    # zero and nonzero controls are no-ops until the readout learns.
    init_zero = fresh()(z, perturbation=torch.zeros(3, 2),
                        apply_noise=False)
    init_active = fresh()(z, perturbation=torch.ones(3, 2) * 2.0,
                          apply_noise=False)
    assert torch.allclose(init_zero["delta_z"], init_active["delta_z"])
    assert init_zero["control_term"].abs().sum() == 0

    def trained_readout():
        torch.manual_seed(0)
        vb = VelocityBrain(hidden_dim=8, num_poisson_layers=2,
                           num_energy_layers=2, poisson_rank=4,
                           perturbation_dim=2,
                           use_lowrank_poisson=True).eval()
        with torch.no_grad():
            vb.perturbation_map[-1].weight.normal_(0, 0.5)
        return vb

    baseline = trained_readout()(z, apply_noise=False)["delta_z"]
    zero_control = trained_readout()(z, perturbation=torch.zeros(3, 2),
                                     apply_noise=False)["delta_z"]
    active = trained_readout()(z, perturbation=torch.ones(3, 2) * 2.0,
                               apply_noise=False)["delta_z"]
    assert torch.allclose(baseline, zero_control, atol=0)  # exact no-op
    assert not torch.allclose(baseline, active)


def test_control_gating_modulates_mobility_and_arousal():
    torch.manual_seed(0)
    vb = VelocityBrain(hidden_dim=8, num_poisson_layers=2, num_energy_layers=2,
                       poisson_rank=4, perturbation_dim=2,
                       use_lowrank_poisson=True).eval()
    # Learnable gate weights make conditioning influential after training;
    # simulate by writing nonzero gates.
    with torch.no_grad():
        vb.gate_map[-1].weight.normal_(0, 0.5)
    z = torch.randn(3, 8)

    def gated_vb():
        torch.manual_seed(0)
        vb = VelocityBrain(hidden_dim=8, num_poisson_layers=2,
                           num_energy_layers=2, poisson_rank=4,
                           perturbation_dim=2,
                           use_lowrank_poisson=True).eval()
        with torch.no_grad():
            vb.gate_map[-1].weight.normal_(0, 0.5)
        return vb

    base = gated_vb()(z, perturbation=torch.zeros(3, 2),
                      apply_noise=False)["delta_z"]
    gated = gated_vb()(z, perturbation=torch.ones(3, 2),
                       apply_noise=False)["delta_z"]
    assert not torch.allclose(base, gated)


# ----------------------------------------------------------- P2.1 noise policy
def test_noise_policy_modes():
    signals = {"calcium": torch.randn(2, 12, 64)}

    def run(mode, train: bool, seed: int):
        model = _model(noise_mode=mode)  # identically seeded weights
        model.train(train)
        torch.manual_seed(seed)          # control the noise RNG stream
        return model.forward_modalities(signals, num_steps=2)["delta_z"]

    # off: deterministic (eval isolates dropout; noise policy is off)
    assert torch.allclose(run("off", False, 1), run("off", False, 2))

    # always: stochastic regardless of train/eval
    a = run("always", True, 1)
    b = run("always", True, 2)
    assert not torch.allclose(a, b)
    assert torch.isfinite(a).all()

    # train: deterministic in eval, stochastic while training
    assert torch.allclose(run("train", False, 1), run("train", False, 2))
    t1 = run("train", True, 1)
    t2 = run("train", True, 2)
    assert not torch.allclose(t1, t2)

    with pytest.raises(ValueError, match="noise_mode"):
        _model(noise_mode="bogus")


def test_per_step_control_sequence():
    signals = {"calcium": torch.randn(2, 12, 64)}
    zeros = torch.zeros(2, 3, 2)
    ones = torch.ones(2, 3, 2)
    # Zero-init readout => dead start; simulate learning by randomizing it.
    def driven_model():
        model = _model(noise_mode="off")
        with torch.no_grad():
            model.velocity_brain.perturbation_map[-1].weight.normal_(0, 0.5)
        return model

    base = driven_model().forward_modalities(
        signals, num_steps=3, perturbation=zeros)["delta_z"]
    driven = driven_model().forward_modalities(
        signals, num_steps=3, perturbation=ones)["delta_z"]
    assert not torch.allclose(base, driven)


# ------------------------------------------------------- P0 data/control glue
def test_reduce_control_and_partition():
    control = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    mean = reduce_control(control, "mean", rollout_steps=1)
    assert mean.shape == (2, 3)
    assert torch.allclose(mean, control.mean(dim=-1))
    last = reduce_control(control, "last")
    assert torch.allclose(last, control[..., -1])
    seq = reduce_control(control, "resample", rollout_steps=4)
    assert seq.shape == (2, 4, 3)

    pulse = torch.zeros(1, 1, 8)
    pulse[0, 0, 1] = 1.0
    pulse[0, 0, 5] = 2.0
    peak = reduce_control(pulse, "peak", rollout_steps=4)
    assert peak.shape == (1, 4, 1)
    assert float(peak.max()) == 2.0
    assert float(peak[0, 0, 0]) > float(
        reduce_control(pulse, "resample", rollout_steps=1).max())

    batch = {
        "calcium": torch.randn(2, 4, 16),
        "calcium_mask": torch.ones(2, 4, 16, dtype=torch.bool),
        "stimulus": torch.randn(2, 3, 16),
        "subject": ["a", "b"],
    }
    signals, targets, perturbation = partition_generic_batch(
        batch, ("calcium", "stimulus"), ("calcium",), None,
        control_modalities=("stimulus",), control_reduction="mean",
        rollout_steps=1)
    assert set(signals) == {"calcium"}          # control excluded from signals
    assert "calcium_mask" in targets
    assert perturbation.shape == (2, 3)

    event_batch = {
        "calcium": torch.randn(1, 4, 8),
        "opto": torch.cat(
            [torch.ones(1, 3, 4), torch.zeros(1, 3, 4)], dim=-1),
    }
    _, _, event_perturbation = partition_generic_batch(
        event_batch, ("calcium", "opto"), ("calcium",), None,
        control_modalities=("opto",), control_reduction="peak",
        rollout_steps=2)
    assert event_perturbation.shape == (1, 2, 3)
    assert torch.all(event_perturbation[0, 0] == 1)
    assert torch.all(event_perturbation[0, 1] == 0)


def test_roles_declaration_and_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_npy(root / "calcium", "w1", np.random.randn(4, 32))
        _write_npy(root / "stimulus", "w1", np.random.randn(2, 32))
        ds = SpeciesSignalDataset(
            root, ["calcium", "stimulus"], seq_len=16,
            roles={"calcium": "signal", "stimulus": "control"},
            manifest=None)
        assert ds.roles["stimulus"] == "control"
        item = ds[0]
        assert item["calcium"].shape == (4, 16)
        assert item["stimulus"].shape == (2, 16)

        tensor_batch, _ = build_species_dataloaders(
            root, ["calcium", "stimulus"], batch_size=1, seq_len=16,
            split_by="none", val_frac=0.0, shuffle=False,
            roles={"calcium": "signal", "stimulus": "control"})
        batch = next(iter(tensor_batch))
        _, _, perturbation = partition_generic_batch(
            batch, ("calcium", "stimulus"), ("calcium",), None,
            control_modalities=("stimulus",), rollout_steps=1)
        # Default reduction is per-step, so the step axis is always present.
        assert perturbation.shape == (1, 1, 2)

        with pytest.raises(ValueError, match="unknown modality roles"):
            SpeciesSignalDataset(root, ["calcium"], seq_len=16,
                                 roles={"calcium": "nonsense"})
