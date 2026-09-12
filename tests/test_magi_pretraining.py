"""Observable contracts for Magi v2 self-supervised objectives."""

import pytest


torch = pytest.importorskip("torch")

from brain_moe_pinn.magi.magi_v2 import MagiV2EEGEncoder
from brain_moe_pinn.magi.pretraining import MagiPretrainingObjective
from brain_moe_pinn.training.training_phases import get_phase_neg_1


def make_encoder():
    return MagiV2EEGEncoder(
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        intermediate_dim=32,
        dropout=0.0,
        spatial_heads=2,
        temporal_heads=2,
        window_size=4,
        use_rope=False,
        alternating_pattern=True,
        max_channels=2,
        patch_size_time=4,
        stride_time=2,
        use_biot_embedding=False,
        use_channel_type_embed=False,
    )

def test_masked_representation_does_not_depend_on_masked_raw_support():
    torch.manual_seed(7)
    encoder = make_encoder().eval()
    objective = MagiPretrainingObjective(encoder, mask_ratio=0.2).eval()
    eeg = torch.randn(1, 1, 14)

    torch.manual_seed(11)
    first = objective.masked_prediction(eeg)
    selected = first["sampled_mask"].reshape(1, 1, -1)
    patch_index = int(torch.where(selected[0, 0])[0][0])
    changed = eeg.clone()
    start = patch_index * encoder.stride_time
    changed[:, :, start:start + encoder.patch_size_time] += 100.0

    torch.manual_seed(11)
    second = objective.masked_prediction(changed)
    visible = ~first["mask"]
    torch.testing.assert_close(
        first["hidden"][visible], second["hidden"][visible],
        rtol=0.0, atol=1e-6,
    )


def test_causal_patch_ntp_shifts_targets_and_blocks_future_input():
    torch.manual_seed(13)
    encoder = make_encoder().eval()
    objective = MagiPretrainingObjective(encoder, mask_ratio=0.2).eval()
    eeg = torch.randn(1, 1, 20)
    future_changed = eeg.clone()
    future_changed[:, :, 12:] += 50.0

    first = objective.causal_patch_ntp(eeg)
    second = objective.causal_patch_ntp(future_changed)
    assert first["predictions"].shape == first["targets"].shape
    assert first["predictions"].shape[2] == first["targets"].shape[2]
    torch.testing.assert_close(
        first["hidden"][:, :, :3], second["hidden"][:, :, :3],
        rtol=0.0, atol=1e-6,
    )


def test_contrastive_requires_two_views_and_ema_moves_key_parameters():
    torch.manual_seed(17)
    encoder = make_encoder()
    objective = MagiPretrainingObjective(
        encoder, momentum=0.5, projection_dim=8).eval()
    view1 = torch.randn(2, 1, 20)
    view2 = view1 + 0.05
    result = objective.contrastive(view1, view2)
    assert result["query"].shape == (2, 8)
    assert result["key"].shape == (2, 8)

    online = next(encoder.parameters())
    key = next(objective.momentum_encoder.parameters())
    before = key.detach().clone()
    with torch.no_grad():
        online.add_(2.0)
    objective.update_momentum_encoder()
    assert not torch.equal(before, key.detach())

    with pytest.raises(ValueError, match="distinct"):
        objective.contrastive(view1, view1)


def test_magi_phase_routes_to_executable_task_contract():
    phase = get_phase_neg_1().to_runtime_config()
    assert phase["task"] == "magi_eeg_pretraining"
    assert phase["magi_objective_weights"] == {
        "masked": 1.0,
        "causal_ntp": 1.0,
        "contrastive": 0.1,
    }


def test_objective_forward_combines_all_enabled_terms_and_freezes_teacher():
    torch.manual_seed(23)
    encoder = make_encoder()
    objective = MagiPretrainingObjective(
        encoder, mask_ratio=0.2, projection_dim=8).train()
    view1, view2 = objective.make_two_views(torch.randn(2, 1, 20))

    result = objective(
        view1, eeg_view2=view2, masked_weight=1.0,
        ntp_weight=0.5, contrastive_weight=0.25)

    assert result["loss"].ndim == 0
    assert result["masked_loss"].ndim == 0
    assert result["ntp_loss"].ndim == 0
    assert result["contrastive_loss"].ndim == 0
    assert not objective.momentum_encoder.training
    assert not objective.momentum_proj_head.training
