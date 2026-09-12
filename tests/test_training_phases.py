"""Focused contracts for the normalized training-phase boundary."""

import pytest

torch = pytest.importorskip("torch")

from train import parse_phases
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
    assert phase["loss_weights"]["cross_modal"] == 0.0


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
