import pytest

torch = pytest.importorskip("torch")

from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.core.stochastic_transition import LowRankDiffusionHead
from brain_moe_pinn.physics.stochastic_process import sample_low_rank_transition
from brain_moe_pinn.training.losses import EmpiricalCRPSLoss, TotalLoss
from brain_moe_pinn.training.training_loop import BrainMoETrainer
from brain_moe_pinn.training.training_phases import LossWeights


def _small_model(**kwargs):
    torch.manual_seed(0)
    options = dict(
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
    )
    options.update(kwargs)
    return BrainMoEPINN(**options).eval()


def test_low_rank_transition_uses_sqrt_dt_and_per_sample_clock():
    mean = torch.zeros(2, 3)
    factor = torch.zeros(2, 3, 1)
    diagonal_std = torch.ones(2, 3)
    diagonal_noise = torch.ones_like(mean)
    factor_noise = torch.zeros(2, 1)

    unit = sample_low_rank_transition(
        mean, factor, diagonal_std, 1.0,
        factor_noise=factor_noise, diagonal_noise=diagonal_noise)
    four = sample_low_rank_transition(
        mean, factor, diagonal_std, 4.0,
        factor_noise=factor_noise, diagonal_noise=diagonal_noise)
    mixed = sample_low_rank_transition(
        mean, factor, diagonal_std, torch.tensor([1.0, 4.0]),
        factor_noise=factor_noise, diagonal_noise=diagonal_noise)

    assert torch.equal(four, 2.0 * unit)
    assert torch.equal(mixed[0], unit[0])
    assert torch.equal(mixed[1], four[1])


def test_low_rank_diffusion_head_is_control_conditioned_and_differentiable():
    head = LowRankDiffusionHead(
        latent_dim=8, rank=3, control_dim=2, floor=1e-4, scale=1e-2)
    state = torch.randn(4, 8)
    zero_control = torch.zeros(4, 2)

    factor, diagonal_std = head(state, zero_control)
    no_control_factor, no_control_diag = head(state, None)
    assert factor.shape == (4, 8, 3)
    assert diagonal_std.shape == (4, 8)
    assert torch.equal(factor, no_control_factor)
    assert torch.equal(diagonal_std, no_control_diag)
    assert torch.all(diagonal_std > head.floor)

    sample = sample_low_rank_transition(
        state, factor, diagonal_std, 0.5)
    sample.square().mean().backward()
    assert head.factor_basis.grad is not None
    assert torch.isfinite(head.factor_basis.grad).all()
    assert head.rank_scale[-1].weight.grad is not None
    assert torch.isfinite(head.rank_scale[-1].weight.grad).all()
    assert head.diag_std_raw.grad is not None
    assert torch.isfinite(head.diag_std_raw.grad).all()


def test_default_ode_is_seed_independent_and_emits_legacy_shapes():
    model = _small_model()
    signals = {"calcium": torch.randn(2, 12, 64)}

    torch.manual_seed(1)
    first = model.forward_modalities(
        signals, num_steps=3, return_sequences=True, reconstruct=True)
    torch.manual_seed(99)
    second = model.forward_modalities(
        signals, num_steps=3, return_sequences=True, reconstruct=True)

    assert model.transition_mode == "ode"
    assert model.transition_diffusion is None
    assert not any(
        key.startswith("transition_diffusion.")
        for key in model.state_dict())
    assert torch.equal(first["z_next_sequence"], second["z_next_sequence"])
    assert torch.equal(first["delta_z_sequence"], second["delta_z_sequence"])
    assert first["calcium_recon_sequence"].shape == (2, 3, 12, 64)
    assert "transition_factor_sequence" not in first


def test_sde_returns_transition_distribution_and_explicit_samples():
    model = _small_model(transition_mode="sde", diffusion_rank=3)
    signals = {"calcium": torch.randn(2, 12, 64)}

    mean_path = model.forward_modalities(
        signals, num_steps=3, return_sequences=True, reconstruct=True)
    assert torch.equal(mean_path["z_next_sequence"],
                       mean_path["transition_mean_sequence"])
    assert mean_path["transition_factor_sequence"].shape == (2, 3, 8, 3)
    assert mean_path["transition_diag_std_sequence"].shape == (2, 3, 8)
    assert torch.all(mean_path["transition_diag_std_sequence"] > 0)

    torch.manual_seed(4)
    first = model.forward_modalities(
        signals, num_steps=3, return_sequences=True,
        sample_transition=True)
    torch.manual_seed(5)
    second = model.forward_modalities(
        signals, num_steps=3, return_sequences=True,
        sample_transition=True)
    assert not torch.equal(first["z_next_sequence"],
                           second["z_next_sequence"])
    assert torch.isfinite(first["z_next_sequence"]).all()


def test_ode_rejects_sampling_and_sde_rejects_legacy_ou_noise():
    ode = _small_model()
    signals = {"calcium": torch.randn(2, 12, 64)}
    with pytest.raises(ValueError, match="transition_mode='sde'"):
        ode.forward_modalities(signals, sample_transition=True)

    sde = _small_model(transition_mode="sde", noise_mode="always")
    with pytest.raises(ValueError, match="legacy OU noise"):
        sde.forward_modalities(signals, num_steps=2, sample_transition=True)


def test_sample_ensemble_returns_mean_and_crps_ready_paths():
    model = _small_model(transition_mode="sde", diffusion_rank=3)
    signals = {"calcium": torch.randn(2, 12, 64)}
    result = model.forward_modalities(
        signals,
        num_steps=3,
        return_sequences=True,
        reconstruct=True,
        step_dt=torch.tensor([0.5, 1.0]),
        sample_transition=True,
        num_samples=4,
    )

    samples = result["calcium_recon_samples"]
    assert samples.shape == (4, 2, 3, 12, 64)
    assert result["calcium_recon_sequence"].shape == (2, 3, 12, 64)
    assert torch.allclose(
        result["calcium_recon_sequence"], samples.mean(dim=0))
    assert result["z_next_sequence_samples"].shape == (4, 2, 3, 8)
    assert result["z_next_sequence"].shape == (2, 3, 8)
    assert samples.std(dim=0).mean() > 0

    with pytest.raises(ValueError, match="sample_transition=True"):
        model.forward_modalities(
            signals, num_samples=2, return_sequences=True)


def test_empirical_crps_scores_spread_and_masks_invalid_values():
    crps = EmpiricalCRPSLoss()
    target = torch.zeros(1, 1, 1, 1)
    collapsed = torch.tensor([1.0, 1.0]).reshape(2, 1, 1, 1, 1)
    spread = torch.tensor([0.0, 2.0]).reshape(2, 1, 1, 1, 1)

    assert crps(spread, target).item() == pytest.approx(0.5)
    assert crps(spread, target).item() < crps(collapsed, target).item()

    masked_target = torch.tensor([[[[0.0, 100.0]]]])
    masked_samples = torch.tensor([
        [[[[0.0, -1000.0]]]],
        [[[[0.0, 1000.0]]]],
    ])
    mask = torch.tensor([[[[True, False]]]])
    assert crps(masked_samples, masked_target, mask).item() == pytest.approx(0.0)

    trainable_samples = spread.clone().requires_grad_()
    crps(trainable_samples, target).backward()
    assert torch.isfinite(trainable_samples.grad).all()


def test_total_loss_applies_empirical_crps_and_requires_samples():
    loss = TotalLoss(LossWeights(
        recon_extra={"calcium": 0.0},
        forecast_crps=1.0,
    ))
    target = torch.zeros(1, 1, 1, 1)
    samples = torch.tensor([0.0, 2.0]).reshape(2, 1, 1, 1, 1).requires_grad_()
    predictions = {
        "calcium_recon": torch.zeros(1, 1, 1),
        "calcium_recon_samples": samples,
    }
    predictions["behavior_recon"] = torch.zeros(1, 1, 1)
    predictions["behavior_recon_samples"] = torch.zeros(
        2, 1, 1, 1, 1)
    total, metrics = loss(predictions, {"calcium": target})

    assert metrics["forecast_crps_calcium"] == pytest.approx(0.5)
    total.backward()
    assert torch.isfinite(samples.grad).all()

    supervised_loss = TotalLoss(LossWeights(
        recon_extra={"calcium": 1.0},
        forecast_crps=1.0,
    ))
    with pytest.raises(KeyError, match="target 'calcium'"):
        supervised_loss(predictions, {})

    with pytest.raises(ValueError, match="forecast CRPS requires"):
        loss(
            {"calcium_recon": torch.zeros(1, 1, 1)},
            {"calcium": target})


class _CheckpointModel(torch.nn.Module):
    def __init__(self, with_transition: bool):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        if with_transition:
            self.transition_diffusion = torch.nn.Linear(2, 2)


def _checkpoint_trainer(model, optimizer):
    trainer = BrainMoETrainer.__new__(BrainMoETrainer)
    trainer.device = torch.device("cpu")
    trainer.world_size = 1
    trainer.is_main_process = False
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.ds_engine = None
    return trainer


def test_old_checkpoint_loads_with_new_sde_parameters_and_optimizer_slots(
        tmp_path):
    old_model = _CheckpointModel(with_transition=False)
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=1e-2)
    old_model.backbone(torch.ones(1, 2)).square().mean().backward()
    old_optimizer.step()
    path = tmp_path / "old_ode_checkpoint.pt"
    torch.save({
        "model_state": old_model.state_dict(),
        "optimizer_state": old_optimizer.state_dict(),
        "step": 7,
        "phase": "forecast",
    }, path)

    new_model = _CheckpointModel(with_transition=True)
    initial_transition = {
        key: value.detach().clone()
        for key, value in new_model.state_dict().items()
        if key.startswith("transition_diffusion.")
    }
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-2)
    trainer = _checkpoint_trainer(new_model, new_optimizer)

    assert trainer.load_checkpoint(str(path)) == (7, "forecast")
    assert torch.equal(
        new_model.backbone.weight, old_model.backbone.weight)
    assert all(torch.equal(
        new_model.state_dict()[key], value)
        for key, value in initial_transition.items())
    assert len(new_optimizer.param_groups[0]["params"]) == 4
    assert new_model.backbone.weight in new_optimizer.state
    assert new_model.transition_diffusion.weight not in new_optimizer.state



def test_sde_checkpoint_downgrades_and_drops_transition_optimizer_slots(
        tmp_path):
    sde_model = _CheckpointModel(with_transition=True)
    sde_optimizer = torch.optim.AdamW(sde_model.parameters(), lr=1e-2)
    signal = torch.ones(1, 2)
    (sde_model.backbone(signal).square().mean()
     + sde_model.transition_diffusion(signal).square().mean()).backward()
    sde_optimizer.step()
    path = tmp_path / "sde_checkpoint.pt"
    torch.save({
        "model_state": sde_model.state_dict(),
        "optimizer_state": sde_optimizer.state_dict(),
        "step": 9,
        "phase": "forecast",
    }, path)

    ode_model = _CheckpointModel(with_transition=False)
    ode_optimizer = torch.optim.AdamW(ode_model.parameters(), lr=1e-2)
    trainer = _checkpoint_trainer(ode_model, ode_optimizer)

    assert trainer.load_checkpoint(str(path)) == (9, "forecast")
    assert torch.equal(ode_model.backbone.weight, sde_model.backbone.weight)
    assert len(ode_optimizer.param_groups[0]["params"]) == 2
    assert set(ode_optimizer.state) == set(ode_model.parameters())

def test_checkpoint_migration_rejects_unrelated_missing_model_keys(tmp_path):
    model = _CheckpointModel(with_transition=True)
    path = tmp_path / "incomplete_checkpoint.pt"
    state = model.state_dict()
    torch.save({
        "model_state": {"backbone.weight": state["backbone.weight"]},
        "step": 1,
        "phase": "forecast",
    }, path)
    trainer = _checkpoint_trainer(model, optimizer=None)

    with pytest.raises(RuntimeError, match="missing keys"):
        trainer.load_checkpoint(str(path))
