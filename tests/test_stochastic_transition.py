import pytest

torch = pytest.importorskip("torch")

from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.core.stochastic_transition import LowRankDiffusionHead
from brain_moe_pinn.physics.stochastic_process import sample_low_rank_transition
from brain_moe_pinn.training.losses import (
    EmpiricalCRPSLoss,
    ProjectedPathEnergyScore,
    SampledHorizonOverdispersionLoss,
    SampledPathAutocorrelationLoss,
    TotalLoss,
)
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


def test_scale_anchor_is_explicit_and_transition_independent():
    signals = {"calcium": 4.0 * torch.randn(2, 12, 64)}
    reference_std = signals["calcium"].std(dim=-1, unbiased=False)
    for transition_mode in ("ode", "sde"):
        off = _small_model(
            transition_mode=transition_mode, scale_anchor="off")
        context = _small_model(
            transition_mode=transition_mode, scale_anchor="context")
        off_output = off.forward_modalities(
            signals, reconstruct=True)["calcium_recon"]
        context_output = context.forward_modalities(
            signals, reconstruct=True)["calcium_recon"]

        assert off.scale_anchor == "off"
        assert context.scale_anchor == "context"
        assert torch.allclose(
            context_output.std(dim=-1, unbiased=False),
            reference_std,
            atol=1e-5,
        )
        assert not torch.allclose(
            off_output.std(dim=-1, unbiased=False), reference_std)
        if transition_mode == "sde":
            sampled = context.forward_modalities(
                signals,
                num_steps=2,
                return_sequences=True,
                reconstruct=True,
                sample_transition=True,
                num_samples=3,
            )
            sampled_std = sampled["calcium_recon_samples"].std(
                dim=-1, unbiased=False)
            assert sampled_std.shape == (3, 2, 2, 12)
            assert torch.allclose(
                sampled_std,
                reference_std.unsqueeze(0).unsqueeze(2).expand_as(
                    sampled_std),
                atol=1e-5,
            )


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

    with pytest.raises(ValueError, match="SDE transitions require noise_mode"):
        _small_model(transition_mode="sde", noise_mode="always")


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


def test_projected_energy_score_distinguishes_equal_marginals_by_path_order():
    score = ProjectedPathEnergyScore(projection_dim=1, lags=(1,))
    target = torch.ones(1, 1, 1, 2)
    coupled = torch.tensor(
        [[-1.0, -1.0], [1.0, 1.0]]).reshape(2, 1, 1, 1, 2)
    anticoupled = torch.tensor(
        [[-1.0, 1.0], [1.0, -1.0]]).reshape(2, 1, 1, 1, 2)

    assert torch.equal(
        coupled.sort(dim=0).values, anticoupled.sort(dim=0).values)
    assert score(coupled, target) < score(anticoupled, target)


def test_projected_energy_score_ignores_masked_path_values():
    score = ProjectedPathEnergyScore(projection_dim=1, lags=(1, 2))
    target = torch.tensor([0.0, 100.0, 1.0]).reshape(1, 1, 1, 3)
    samples = torch.tensor([
        [0.0, -1000.0, 1.0],
        [0.0, 1000.0, 1.0],
    ]).reshape(2, 1, 1, 1, 3)
    mask = torch.tensor([True, False, True]).reshape(1, 1, 1, 3)
    changed_target = target.clone()
    changed_target[..., 1] = -12345.0
    changed_samples = samples.clone()
    changed_samples[..., 1] = 54321.0

    original = score(samples, target, mask)
    changed = score(changed_samples, changed_target, mask)

    assert torch.allclose(original, changed)


def test_sampled_path_autocorrelation_calibrates_all_configured_lags():
    generator = torch.Generator().manual_seed(21)
    target = torch.randn((1, 1, 1, 64), generator=generator)
    alternating = torch.where(
        torch.arange(64) % 2 == 0, -1.0, 1.0).reshape(1, 1, 1, 64)
    samples = torch.stack((alternating, -alternating), dim=0)
    lag_losses = [
        SampledPathAutocorrelationLoss(lags=(lag,))(samples, target)
        for lag in (1, 2, 4, 8)
    ]

    assert all(loss.item() > 0.5 for loss in lag_losses)
    assert SampledPathAutocorrelationLoss()(samples, target).item() == pytest.approx(
        torch.stack(lag_losses).mean().item())


def test_late_overdispersion_only_penalizes_horizons_five_through_eight():
    scale_target = torch.arange(4, dtype=torch.float32).reshape(
        1, 1, 1, 4).expand(1, 8, 1, 4).clone()
    late_factors = torch.tensor(
        [100.0, 100.0, 100.0, 100.0, 2.0, 2.0, 2.0, 2.0]
    ).reshape(1, 8, 1, 1)
    early_factors = torch.tensor(
        [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    ).reshape(1, 8, 1, 1)
    late_overdispersed = (
        scale_target * late_factors).unsqueeze(0).expand(2, -1, -1, -1, -1)
    early_only_overdispersed = (
        scale_target * early_factors).unsqueeze(0).expand_as(
            late_overdispersed)
    underdispersed = (scale_target * 0.5).unsqueeze(0).expand_as(
        late_overdispersed)
    scale_loss = SampledHorizonOverdispersionLoss(start_horizon=5)

    assert scale_loss(late_overdispersed, scale_target).item() == pytest.approx(1.0)
    assert scale_loss(
        early_only_overdispersed, scale_target).item() == pytest.approx(0.0)
    assert scale_loss(underdispersed, scale_target).item() == pytest.approx(0.0)


def test_total_loss_trains_with_sampled_path_objective_terms():
    target = torch.arange(16, dtype=torch.float32).reshape(
        1, 1, 1, 16).expand(1, 5, 1, 16).clone()
    noise = torch.linspace(-0.2, 0.2, 16).reshape(1, 1, 1, 16)
    samples = torch.stack(
        (target + noise, target - noise), dim=0).requires_grad_()
    loss = TotalLoss(LossWeights(
        forecast=0.0,
        forecast_crps=0.25,
        forecast_path_energy=1.0,
        forecast_path_autocorr=0.05,
        forecast_overdispersion=0.1,
        forecast_corr_diff=0.0,
        forecast_variance=0.0,
        forecast_autocorr=0.0,
        forecast_component=0.0,
        recon_extra={"calcium": 0.0},
    ))
    total, metrics = loss(
        {
            "calcium_recon": samples.mean(dim=0)[:, -1],
            "calcium_recon_sequence": samples.mean(dim=0),
            "calcium_recon_samples": samples,
        },
        {"calcium": target},
    )

    assert all(name in metrics for name in (
        "forecast_crps_calcium",
        "forecast_path_energy_calcium",
        "forecast_path_autocorr_calcium",
        "forecast_overdispersion_calcium",
    ))
    total.backward()
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()


def test_total_loss_applies_empirical_crps_and_requires_samples():
    sampled_weights = dict(
        forecast=0.0,
        forecast_component=0.0,
        forecast_variance=0.0,
        forecast_autocorr=0.0,
        forecast_corr_diff=0.0,
        recon_extra={"calcium": 0.0},
        forecast_crps=1.0,
    )
    loss = TotalLoss(LossWeights(**sampled_weights))
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

    supervised_loss = TotalLoss(LossWeights(**sampled_weights))
    with pytest.raises(
            ValueError, match="sampled modality with a target"):
        supervised_loss(predictions, {})

    with pytest.raises(ValueError, match="forecast CRPS requires"):
        loss(
            {"calcium_recon": torch.zeros(1, 1, 1)},
            {"calcium": target})


def test_total_loss_supports_crps_only_with_forecast_enabled():
    weights = LossWeights(
        forecast=1.0,
        forecast_huber=0.0,
        forecast_corr_diff=0.0,
        forecast_variance=0.0,
        forecast_crps=1.0,
        forecast_autocorr=0.0,
        forecast_component=0.0,
        recon_extra={"calcium": 0.0},
    )
    loss = TotalLoss(weights)
    target = torch.zeros(1, 1, 1, 1)
    samples = torch.tensor([0.0, 2.0]).reshape(2, 1, 1, 1, 1)
    predictions = {
        "calcium_recon": torch.zeros(1, 1, 1),
        "calcium_recon_sequence": torch.zeros(1, 1, 1, 1),
        "calcium_recon_samples": samples,
    }

    total, metrics = loss(predictions, {"calcium": target})

    assert total.item() == pytest.approx(0.5)
    assert metrics["forecast_crps_calcium"] == pytest.approx(0.5)
    assert "forecast_calcium" not in metrics



@pytest.mark.parametrize(
    "legacy_weights",
    [
        {"forecast_component": 1.0},
        {"forecast_variance": 1.0},
        {"forecast_autocorr": 1.0},
        {"forecast_corr_diff": 1.0},
        {"recon_extra": {"calcium": 1.0}},
    ],
)
def test_sampled_forecast_rejects_ensemble_mean_losses(legacy_weights):
    weights = {
        "forecast": 0.1,
        "forecast_crps": 1.0,
        "forecast_corr_diff": 0.0,
        "forecast_variance": 0.0,
        "forecast_autocorr": 0.0,
        "recon_extra": {"calcium": 0.0},
    }
    weights.update(legacy_weights)
    loss = TotalLoss(LossWeights(**weights))
    target = torch.zeros(1, 2, 1, 4)
    samples = torch.stack((target, target + 1.0), dim=0)
    predictions = {
        "calcium_recon": samples.mean(dim=0)[:, -1],
        "calcium_recon_sequence": samples.mean(dim=0),
        "calcium_recon_samples": samples,
    }

    with pytest.raises(ValueError, match="ensemble-mean"):
        loss(predictions, {"calcium": target})


def test_sde_forecast_allows_only_center_and_sampled_crps():
    weights = LossWeights(
        forecast=0.1,
        forecast_crps=1.0,
        forecast_corr_diff=0.0,
        forecast_variance=0.0,
        forecast_autocorr=0.0,
        forecast_component=0.0,
        recon_extra={"calcium": 0.0},
    )
    loss = TotalLoss(weights)
    target = torch.zeros(1, 2, 1, 4)
    samples = torch.stack((target, target + 1.0), dim=0)
    center = samples.mean(dim=0)
    total, metrics = loss(
        {
            "calcium_recon": center[:, -1],
            "calcium_recon_sequence": center,
            "calcium_recon_samples": samples,
        },
        {"calcium": target},
    )

    assert torch.isfinite(total)
    assert "forecast_calcium" in metrics
    assert "forecast_crps_calcium" in metrics


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


def test_optimizer_preserves_parameter_lr_scales_through_schedule_updates():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.velocity_brain = torch.nn.Linear(2, 2)
            self.transition_diffusion = torch.nn.Linear(2, 2)
            self.frozen = torch.nn.Linear(2, 2)

    model = Model()
    trainer = _checkpoint_trainer(model, None)
    trainer.scaler_init_scale = 1.0
    trainer._apply_freeze_policy({
        "trainable_parameter_prefixes": (
            "velocity_brain.", "transition_diffusion."),
        "trainable_parameter_lr_scales": {
            "velocity_brain.": 1.0,
            "transition_diffusion.": 0.1,
        },
    })
    trainer.create_optimizer({
        "learning_rate": 1e-3,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "grad_clip": 1.0,
    })

    def parameter_learning_rates():
        return {
            id(parameter): group["lr"]
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }

    rates = parameter_learning_rates()
    assert all(rates[id(parameter)] == pytest.approx(1e-3)
               for parameter in model.velocity_brain.parameters())
    assert all(rates[id(parameter)] == pytest.approx(1e-4)
               for parameter in model.transition_diffusion.parameters())
    assert all(id(parameter) not in rates
               for parameter in model.frozen.parameters())

    trainer._set_optimizer_learning_rate(trainer.optimizer, 5e-4)
    rates = parameter_learning_rates()
    assert all(rates[id(parameter)] == pytest.approx(5e-4)
               for parameter in model.velocity_brain.parameters())
    assert all(rates[id(parameter)] == pytest.approx(5e-5)
               for parameter in model.transition_diffusion.parameters())



def test_deepspeed_lr_groups_preserve_phase_specific_prefix_scales():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.velocity_brain = torch.nn.Linear(2, 2)
            self.transition_diffusion = torch.nn.Linear(2, 2)
            self.other = torch.nn.Linear(2, 2)

    model = Model()
    phase_scales = {
        "p6": {
            "velocity_brain.": 0.2,
            "transition_diffusion.": 0.02,
        },
        "p7": {
            "velocity_brain.": 0.4,
            "transition_diffusion.": 0.05,
        },
    }
    groups = BrainMoETrainer._deepspeed_parameter_groups(
        model, phase_scales)
    trainer = _checkpoint_trainer(model, torch.optim.AdamW(groups, lr=1e-3))
    trainer._phase_trainable_parameter_lr_scales = phase_scales

    def parameter_learning_rates():
        return {
            id(parameter): group["lr"]
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }

    trainer._apply_deepspeed_phase_lr_multipliers(
        "p6", {"trainable_parameter_lr_scales": phase_scales["p6"]})
    trainer._set_optimizer_learning_rate(trainer.optimizer, 1e-3)
    rates = parameter_learning_rates()
    assert all(rates[id(parameter)] == pytest.approx(2e-4)
               for parameter in model.velocity_brain.parameters())
    assert all(rates[id(parameter)] == pytest.approx(2e-5)
               for parameter in model.transition_diffusion.parameters())
    assert all(rates[id(parameter)] == pytest.approx(1e-3)
               for parameter in model.other.parameters())

    trainer._apply_deepspeed_phase_lr_multipliers(
        "p7", {"trainable_parameter_lr_scales": phase_scales["p7"]})
    trainer._set_optimizer_learning_rate(trainer.optimizer, 5e-4)
    rates = parameter_learning_rates()
    assert all(rates[id(parameter)] == pytest.approx(2e-4)
               for parameter in model.velocity_brain.parameters())
    assert all(rates[id(parameter)] == pytest.approx(2.5e-5)
               for parameter in model.transition_diffusion.parameters())
    assert all(rates[id(parameter)] == pytest.approx(5e-4)
               for parameter in model.other.parameters())


def test_old_checkpoint_loads_with_new_sde_parameters_and_optimizer_slots(
        tmp_path):
    old_model = _CheckpointModel(with_transition=False)
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=1e-2)
    old_model.backbone(torch.ones(1, 2)).square().mean().backward()
    old_optimizer.step()
    legacy_runtime_state = {
        f"velocity_brain.mt_kda.state_{name}": torch.zeros(8, 192)
        for name in ("a", "b", "c")
    }
    path = tmp_path / "old_ode_checkpoint.pt"
    torch.save({
        "model_state": {
            **old_model.state_dict(),
            **legacy_runtime_state,
        },
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


def test_trainable_prefix_freezes_other_parameters_and_modes():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
            self.transition_diffusion = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))

    model = Model()
    trainer = _checkpoint_trainer(model, None)
    trainer._apply_freeze_policy({
        "trainable_parameter_prefixes": ["transition_diffusion."]})
    model.train()
    trainer._apply_frozen_module_eval_modes(model)

    assert all(not parameter.requires_grad
               for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad
               for parameter in model.transition_diffusion.parameters())
    assert not model.backbone.training
    assert not model.backbone[1].training
    assert model.transition_diffusion.training
    assert model.transition_diffusion[1].training


def test_trainable_prefix_rejects_unmatched_names_and_component_policy():
    model = _CheckpointModel(with_transition=True)
    trainer = _checkpoint_trainer(model, None)
    with pytest.raises(ValueError, match="match no model parameters"):
        trainer._apply_freeze_policy(
            {"trainable_parameter_prefixes": ["missing."]})
    with pytest.raises(ValueError, match="cannot be combined"):
        trainer._apply_freeze_policy({
            "trainable_parameter_prefixes": ["transition_diffusion."],
            "velocity_brain": True,
        })


def test_model_only_checkpoint_load_does_not_restore_optimizer(tmp_path):
    old_model = _CheckpointModel(with_transition=False)
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=1e-2)
    old_model.backbone(torch.ones(1, 2)).square().mean().backward()
    old_optimizer.step()
    path = tmp_path / "p6_ode_model.pt"
    torch.save({
        "model_state": old_model.state_dict(),
        "optimizer_state": old_optimizer.state_dict(),
        "step": 3000,
        "phase": "p6",
    }, path)

    new_model = _CheckpointModel(with_transition=True)
    initial_transition = {
        key: value.detach().clone()
        for key, value in new_model.state_dict().items()
        if key.startswith("transition_diffusion.")
    }
    trainer = _checkpoint_trainer(new_model, None)
    missing, unexpected = trainer.load_model_checkpoint(str(path))

    assert torch.equal(new_model.backbone.weight, old_model.backbone.weight)
    assert torch.equal(new_model.backbone.bias, old_model.backbone.bias)
    assert missing == {
        "transition_diffusion.weight", "transition_diffusion.bias"}
    assert unexpected == set()
    assert all(torch.equal(new_model.state_dict()[key], value)
               for key, value in initial_transition.items())
    assert trainer.optimizer is None



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
