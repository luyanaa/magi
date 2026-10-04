"""Focused contracts for the normalized training-phase boundary."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from train import (
    apply_data_profile_loss_overrides,
    apply_trainable_parameter_prefixes,
    apply_data_profile_rollout_contract,
    enforce_sde_forecast_contract,
    parse_phases,
)
from brain_moe_pinn.training.losses import TotalLoss
from brain_moe_pinn.training.training_loop import (
    BrainMoETrainer,
    augment_phase_loss_weights,
)
from brain_moe_pinn.training.training_phases import (
    LossWeights,
    evaluate_transition_gate,
    get_stage_1_p1,
    get_stage_1_p3,
    get_stage_1_p4,
    get_stage_1_p5,
    get_stage_1_p6,
)


def test_cli_phase_selection_is_normalized_and_expands_stage_one():
    phases = parse_phases("-1,1,2,3")

    assert len(phases) == 9
    assert all(isinstance(phase, dict) for phase in phases)
    assert phases[0]["task"] == "magi_eeg_pretraining"
    assert [phase["stage"] for phase in phases[1:7]] == [
        "stage_1_p1", "stage_1_p2", "stage_1_p3",
        "stage_1_p4", "stage_1_p5", "stage_1_p6",
    ]
    assert phases[1]["task"] == "model_training"
    assert "freeze_policy" in phases[1]
    assert "rollout" in phases[1] and "context" in phases[1]


def test_phase_task_identity_and_gate_are_explicit():
    phase = get_stage_1_p1().to_runtime_config()
    assert phase["task"] == "model_training"
    assert evaluate_transition_gate(lambda metrics: metrics["score"] >= 2,
                                    {"score": 2})
    assert not evaluate_transition_gate(lambda metrics: metrics["score"] >= 2,
                                        {"score": 1})


def test_forecast_phase_declares_multi_horizon_contract():
    phase = get_stage_1_p3().to_runtime_config()
    assert phase["rollout"]["steps"] == 2
    assert phase["loss_weights"]["forecast"] == 1.0
    assert phase["loss_weights"]["forecast_horizon_weights"] == (1.0, 0.5)
    assert phase["loss_weights"]["forecast_variance"] == 1.0
    assert phase["loss_weights"]["cross_modal"] == 0.0


def test_data_profile_drives_rollout_and_horizon_weights():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/data/c_elegans_unified.json").open() as handle:
        profile = json.load(handle)

    phases = apply_data_profile_rollout_contract(
        parse_phases("1"), profile)

    assert profile["rollout_steps"] == profile["future_steps"] == 3
    assert profile["rollout_horizon_weights"] == [1.0, 0.5, 0.25]
    assert all(phase["rollout_steps"] == 3 for phase in phases)
    assert all(phase["rollout"]["steps"] == 3 for phase in phases)
    assert all(
        phase["loss_weights"].forecast_horizon_weights
        == (1.0, 0.5, 0.25)
        for phase in phases)


def test_data_profile_can_disable_unstable_autocorrelation():
    phases = apply_data_profile_loss_overrides(
        parse_phases("p6"),
        {"forecast_autocorr": 0.0},
    )

    assert phases[0]["loss_weights"].forecast_autocorr == 0.0


def test_sde_objective_overrides_only_forecast_phases():
    phases = apply_data_profile_loss_overrides(
        parse_phases("1"),
        {
            "forecast_center": 0.1,
            "forecast_corr_diff": 0.0,
            "forecast_variance": 0.0,
            "forecast_autocorr": 0.0,
            "forecast_crps": 1.0,
            "forecast_recon_extra": {"calcium": 0.0},
        },
    )

    warmup, forecast = phases[0]["loss_weights"], phases[2]["loss_weights"]
    assert warmup.forecast == 0.0
    assert warmup.forecast_crps == 0.0
    assert "calcium" not in warmup.recon_extra
    assert forecast.forecast == pytest.approx(0.1)
    assert forecast.forecast_corr_diff == 0.0
    assert forecast.forecast_variance == 0.0
    assert forecast.forecast_autocorr == 0.0
    assert forecast.forecast_crps == pytest.approx(1.0)
    assert forecast.recon_extra["calcium"] == 0.0


def test_crps_only_profile_zeros_other_phase_loss_weights():
    profile = {
        "rollout_steps": 8,
        "future_steps": 8,
        "rollout_horizon_weights": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "forecast_objective": "crps_only",
        "forecast_crps": 1.0,
        "forecast_recon_extra": {"calcium": 0.0},
    }
    phases = apply_data_profile_rollout_contract(parse_phases("p6"), profile)
    phases = apply_data_profile_loss_overrides(phases, profile)
    weights = phases[0]["loss_weights"]

    assert weights.forecast == 1.0
    assert weights.forecast_crps == 1.0
    assert weights.forecast_huber == 0.0
    assert weights.forecast_corr_diff == 0.0
    assert weights.forecast_variance == 0.0
    assert weights.forecast_autocorr == 0.0
    assert weights.forecast_component == 0.0
    assert weights.forecast_component_wasserstein == 0.0
    assert weights.forecast_component_variance == 0.0
    assert weights.recon_eeg == weights.recon_fmri == weights.recon_meg == 0.0
    assert weights.velocity_smooth == weights.dissip == weights.spectrum == 0.0
    assert weights.moe_load_balance == weights.grassmannian_reg == 0.0
    assert weights.bandpower == weights.sigreg == 0.0
    assert weights.sigreg_sketch_dim == 64
    assert weights.forecast_horizon_weights == tuple(
        profile["rollout_horizon_weights"])
    assert weights.recon_extra == {"calcium": 0.0}
    assert enforce_sde_forecast_contract(
        phases,
        SimpleNamespace(
            transition_mode="sde", generic_observation_only=True),
        recon_modalities=("calcium",),
    ) is None



def test_path_calibrated_profile_enables_only_sampled_path_terms():
    profile = {
        "rollout_steps": 8,
        "future_steps": 8,
        "rollout_horizon_weights": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "forecast_objective": "path_calibrated",
        "forecast_crps": 0.25,
        "forecast_path_energy": 1.0,
        "forecast_path_autocorr": 0.05,
        "forecast_overdispersion": 0.1,
        "forecast_recon_extra": {"calcium": 0.0},
    }
    phases = apply_data_profile_rollout_contract(parse_phases("p6"), profile)
    phases = apply_data_profile_loss_overrides(phases, profile)
    weights = phases[0]["loss_weights"]

    assert weights.forecast == 1.0
    assert weights.forecast_crps == pytest.approx(0.25)
    assert weights.forecast_path_energy == pytest.approx(1.0)
    assert weights.forecast_path_autocorr == pytest.approx(0.05)
    assert weights.forecast_overdispersion == pytest.approx(0.1)
    assert weights.forecast_huber == 0.0
    assert weights.forecast_corr_diff == 0.0
    assert weights.forecast_variance == 0.0
    assert weights.forecast_autocorr == 0.0
    assert weights.recon_extra == {"calcium": 0.0}

    with pytest.raises(ValueError, match="forecast_path_autocorr"):
        apply_data_profile_loss_overrides(
            parse_phases("p6"),
            {
                **profile,
                "forecast_path_autocorr": 0.0,
            },
        )

def test_crps_only_profile_requires_positive_crps_and_known_objective():
    with pytest.raises(ValueError, match="requires forecast_crps > 0"):
        apply_data_profile_loss_overrides(
            parse_phases("p6"),
            {"forecast_objective": "crps_only", "forecast_crps": 0.0},
        )
    with pytest.raises(ValueError, match="forecast_objective"):
        apply_data_profile_loss_overrides(
            parse_phases("p6"), {"forecast_objective": "distribution_only"})


def test_trainable_parameter_prefix_replaces_component_freeze_policy():
    phases = apply_trainable_parameter_prefixes(
        parse_phases("p6"), ["transition_diffusion."])

    assert len(phases) == 1
    assert phases[0]["freeze_policy"] == {
        "trainable_parameter_prefixes": ("transition_diffusion.",)
    }
    with pytest.raises(ValueError, match="exactly one"):
        apply_trainable_parameter_prefixes(
            parse_phases("1"), ["transition_diffusion."])



def test_trainable_parameter_prefix_lr_scales_are_recorded_and_validated():
    phases = apply_trainable_parameter_prefixes(
        parse_phases("p6"),
        ["velocity_brain.mobility_op.", "transition_diffusion."],
        [
            "velocity_brain.mobility_op.=0.2",
            "transition_diffusion.=0.02",
        ],
    )

    assert phases[0]["freeze_policy"] == {
        "trainable_parameter_prefixes": (
            "velocity_brain.mobility_op.", "transition_diffusion."),
        "trainable_parameter_lr_scales": {
            "velocity_brain.mobility_op.": 0.2,
            "transition_diffusion.": 0.02,
        },
    }
    with pytest.raises(ValueError, match="not in the trainable"):
        apply_trainable_parameter_prefixes(
            parse_phases("p6"),
            ["transition_diffusion."],
            ["velocity_brain.=0.1"],
        )

def test_sde_forecast_phase_requires_sampled_distribution_loss():
    phases = parse_phases("p6")
    sde = SimpleNamespace(
        transition_mode="sde", generic_observation_only=True)
    ode = SimpleNamespace(transition_mode="ode")

    with pytest.raises(ValueError, match="requires forecast_crps"):
        enforce_sde_forecast_contract(
            phases, sde, recon_modalities=("calcium",))
    assert enforce_sde_forecast_contract(phases, ode) is None

    crps_only = apply_data_profile_loss_overrides(
        phases,
        {
            "forecast_crps": 1.0,
            "forecast_variance": 0.0,
            "forecast_autocorr": 0.0,
            "forecast_recon_extra": {"calcium": 0.0},
        },
    )
    with pytest.raises(ValueError, match="forecast_corr_diff"):
        enforce_sde_forecast_contract(
            crps_only, sde, recon_modalities=("calcium",))

    resolved = apply_data_profile_loss_overrides(
        phases,
        {
            "forecast_center": 0.1,
            "forecast_crps": 1.0,
            "forecast_corr_diff": 0.0,
            "forecast_variance": 0.0,
            "forecast_autocorr": 0.0,
            "forecast_recon_extra": {"calcium": 0.0},
        },
    )
    assert enforce_sde_forecast_contract(
        resolved, sde, recon_modalities=("calcium",)) is None


@pytest.mark.parametrize(
    "term",
    [
        "forecast_component",
        "forecast_variance",
        "forecast_autocorr",
        "forecast_corr_diff",
    ],
)
def test_sde_preflight_rejects_ensemble_mean_forecast_terms(term):
    weights = LossWeights(
        forecast=0.0,
        forecast_crps=1.0,
        forecast_component=0.0,
        forecast_variance=0.0,
        forecast_autocorr=0.0,
        forecast_corr_diff=0.0,
        recon_extra={"calcium": 0.0},
    )
    setattr(weights, term, 1.0)
    phase = {"name": "sde_forecast", "loss_weights": weights}
    sde = SimpleNamespace(
        transition_mode="sde", generic_observation_only=True)

    with pytest.raises(ValueError, match=term):
        enforce_sde_forecast_contract(
            [phase], sde, recon_modalities=("calcium",))


def test_sde_preflight_rejects_reconstruction_and_component_spec_losses():
    sde = SimpleNamespace(
        transition_mode="sde", generic_observation_only=True)
    crps_with_reconstruction = {
        "name": "sde_forecast",
        "loss_weights": LossWeights(
            forecast_crps=1.0,
            forecast_corr_diff=0.0,
            forecast_variance=0.0,
            forecast_autocorr=0.0,
            recon_extra={"calcium": 0.25},
        ),
    }
    with pytest.raises(ValueError, match=r"recon_extra\[calcium\]"):
        enforce_sde_forecast_contract(
            [crps_with_reconstruction],
            sde,
            recon_modalities=("calcium",),
        )

    forecast = {
        "name": "sde_forecast",
        "loss_weights": LossWeights(
            forecast=1.0,
            forecast_crps=1.0,
            forecast_corr_diff=0.0,
            recon_extra={"calcium": 0.0},
        ),
    }
    with pytest.raises(ValueError, match="forecast_component"):
        enforce_sde_forecast_contract(
            [forecast],
            sde,
            component_forecast_spec={"loss_weight": 0.25},
            recon_modalities=("calcium",),
        )

    implicit_reconstruction = {
        "name": "sde_forecast",
        "loss_weights": LossWeights(
            forecast_crps=1.0,
            forecast_component=0.0,
            forecast_variance=0.0,
            forecast_autocorr=0.0,
            forecast_corr_diff=0.0,
        ),
    }
    with pytest.raises(ValueError, match=r"recon_extra\[calcium\]"):
        enforce_sde_forecast_contract(
            [implicit_reconstruction],
            sde,
            recon_modalities=("calcium",),
        )


def test_data_profile_enables_crps_only_for_forecast_phases():
    phases = parse_phases("-1,p6")

    resolved = apply_data_profile_loss_overrides(
        phases, {"forecast_crps": 0.25})

    assert resolved[0]["loss_weights"].forecast_crps == 0.0
    assert resolved[1]["loss_weights"].forecast_crps == 0.25
    assert phases[1]["loss_weights"].forecast_crps == 0.0


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_data_profile_rejects_invalid_crps_override(value):
    with pytest.raises(ValueError, match="forecast_crps"):
        apply_data_profile_loss_overrides(parse_phases("p6"), {
            "forecast_crps": value,
        })


def test_data_profile_crps_requires_a_forecast_phase():
    with pytest.raises(ValueError, match="forecast loss enabled"):
        apply_data_profile_loss_overrides(parse_phases("-1"), {
            "forecast_crps": 0.25,
        })

def test_data_profile_rejects_invalid_autocorrelation_override():
    with pytest.raises(ValueError, match="forecast_autocorr"):
        apply_data_profile_loss_overrides(
            parse_phases("p6"),
            {"forecast_autocorr": -0.1},
        )


def test_data_profile_rejects_inconsistent_rollout_contract():
    with pytest.raises(ValueError, match="exactly one value per"):
        apply_data_profile_rollout_contract(
            parse_phases("1"),
            {
                "future_steps": 3,
                "rollout_steps": 3,
                "rollout_horizon_weights": [1.0, 0.5],
            })


def test_total_loss_syncs_forecast_criteria_to_runtime_horizons():
    phase_weights = LossWeights(
        recon_extra={"calcium": 1.0},
        forecast=1.0,
        forecast_huber=0.25,
        forecast_corr_diff=2.0,
        forecast_horizon_weights=(1.0, 0.5),
        sigreg=0.0,
    )
    total_loss = TotalLoss(LossWeights())
    total_loss.loss_weights = phase_weights
    total_loss.use_loss_normalization = False

    with pytest.raises(ValueError, match="resolve the data-profile"):
        total_loss.configure_forecast(phase_weights, rollout_steps=3)


def test_generic_reconstruction_does_not_duplicate_dedicated_terms():
    weights = LossWeights(recon_eeg=2.0, recon_extra={"calcium": 0.5})
    augmented = augment_phase_loss_weights(
        weights, ("eeg", "fmri", "calcium"), {"eeg": "huber"})

    assert augmented.recon_extra == {"calcium": 0.5}
    assert augmented.recon_loss_types == {}
    total_loss = TotalLoss(
        LossWeights(recon_eeg=1.0, recon_extra={"eeg": 9.0}))
    total_loss.use_loss_normalization = False
    prediction = {"eeg_recon": torch.zeros(1, 1, 4)}
    target = {"eeg": torch.ones(1, 1, 4)}
    value, _ = total_loss(prediction, target)
    assert torch.allclose(value, torch.ones((),), atol=1e-6)


def test_generic_crps_clears_absent_dedicated_reconstruction_terms():
    weights = LossWeights(
        recon_eeg=1.0,
        recon_fmri=1.0,
        recon_meg=1.0,
        recon_extra={"calcium": 0.0},
        forecast=0.1,
        forecast_corr_diff=0.0,
        forecast_variance=0.0,
        forecast_autocorr=0.0,
        forecast_crps=1.0,
        forecast_horizon_weights=(1.0,),
        cross_modal=0.0,
        cross=0.0,
        cross_soft=0.0,
        dissip=0.0,
        spectrum=0.0,
        bandpower=0.0,
        sigreg=0.0,
        moe_load_balance=0.0,
        grassmannian_reg=0.0,
    )
    effective_weights = augment_phase_loss_weights(
        weights, ("calcium",))
    assert effective_weights.recon_eeg == 0.0
    assert effective_weights.recon_fmri == 0.0
    assert effective_weights.recon_meg == 0.0

    total_loss = TotalLoss(effective_weights)
    total_loss.use_loss_normalization = False
    total_loss.configure_forecast(effective_weights, rollout_steps=1)
    target = torch.ones(1, 1, 1, 4)
    predictions = {
        "calcium_recon": torch.zeros(1, 1, 4),
        "calcium_recon_sequence": torch.zeros_like(target),
        "calcium_recon_samples": torch.stack(
            (torch.zeros_like(target), target)),
    }
    value, metrics = total_loss(predictions, {"calcium": target})

    expected = (
        0.1 * metrics["forecast_calcium"]
        + metrics["forecast_crps_calcium"]
    )
    assert torch.isfinite(value)
    assert metrics["forecast_calcium"] > 0
    assert metrics["forecast_crps_calcium"] > 0
    assert value.item() == pytest.approx(expected)


def test_generic_ode_forecast_keeps_reconstruction_without_loss_mix():
    weights = LossWeights(
        recon_eeg=0.0,
        recon_fmri=0.0,
        recon_meg=0.0,
        recon_extra={"calcium": 1.0},
        forecast=0.1,
        forecast_corr_diff=0.0,
        forecast_variance=0.0,
        forecast_autocorr=0.0,
        forecast_crps=0.0,
        forecast_horizon_weights=(1.0,),
        cross_modal=0.0,
        cross=0.0,
        cross_soft=0.0,
        dissip=0.0,
        spectrum=0.0,
        bandpower=0.0,
        sigreg=0.0,
        moe_load_balance=0.0,
        grassmannian_reg=0.0,
    )
    total_loss = TotalLoss(weights)
    total_loss.use_loss_normalization = False
    total_loss.configure_forecast(weights, rollout_steps=1)
    target = torch.ones(1, 1, 1, 4)
    predictions = {
        "calcium_recon": torch.zeros(1, 1, 4),
        "calcium_recon_sequence": torch.zeros_like(target),
    }
    value, metrics = total_loss(predictions, {"calcium": target})

    expected = (
        0.1 * metrics["forecast_calcium"]
        + metrics["recon_calcium"]
    )
    assert metrics["forecast_calcium"] > 0
    assert metrics["recon_calcium"] > 0
    assert value.item() == pytest.approx(expected)


def test_phase_boundary_resets_loss_normalizer_and_gate_rejects_bad_result():
    trainer = object.__new__(BrainMoETrainer)
    trainer.total_loss = TotalLoss(LossWeights())
    trainer.total_loss.normalizer.normalize("probe", torch.tensor(2.0))
    assert trainer.total_loss.normalizer.get_stats()

    trainer.reset_loss_normalizer()
    assert trainer.total_loss.normalizer.get_stats() == {}

    try:
        evaluate_transition_gate(lambda metrics: 1, {})
    except TypeError as exc:
        assert "return bool" in str(exc)
    else:
        raise AssertionError("non-boolean transition gate result was accepted")


def test_policy_objectives_fail_closed_without_observable_targets():
    action_loss = TotalLoss(LossWeights(action=1.0))
    with pytest.raises(ValueError, match="action_utility_target"):
        action_loss({"efe": torch.zeros(2, 1)}, {})

    replay_loss = TotalLoss(LossWeights(replay=1.0))
    with pytest.raises(ValueError, match="replay_target"):
        replay_loss({"z_imagined": torch.zeros(2, 4)}, {})


def test_intervention_response_loss_uses_explicit_effect_target_and_mask():
    total_loss = TotalLoss(LossWeights(intervention_response=1.0))
    predictions = {"intervention_effect": torch.zeros(2, 3)}
    targets = {
        "intervention_target": torch.ones(2, 3),
        "intervention_mask": torch.tensor(
            [[True, True, False], [True, False, False]]),
    }
    value, metrics = total_loss(predictions, targets)
    assert torch.isfinite(value)
    assert metrics["intervention_valid_fraction"] == pytest.approx(0.5)


def test_generic_runs_disable_hub_only_cross_modal_objectives():
    augmented = augment_phase_loss_weights(
        LossWeights(cross_modal=0.3, cross=0.05, cross_soft=0.1),
        ("calcium",))
    assert augmented.cross_modal == 0.0
    assert augmented.cross == 0.0
    assert augmented.cross_soft == 0.0

def test_phase_accumulation_is_consistent_across_stage_one():
    phases = parse_phases("1")
    assert {int(phase["gradient_accumulation"]) for phase in phases} == {1}


def test_phase_six_declares_context_length():
    phase = parse_phases("1")[-1]
    assert phase["max_seq_len_eeg"] == 4096



def test_phase_six_uses_stable_long_rollout_learning_rate():
    phase = get_stage_1_p6().to_runtime_config()
    assert phase["learning_rate"] == pytest.approx(5e-5)
    assert phase["min_lr"] == pytest.approx(5e-5)


def test_trainer_freeze_policy_includes_velocity_brain():
    assert "velocity_brain" in parse_phases("1")[0]["freeze_policy"]
