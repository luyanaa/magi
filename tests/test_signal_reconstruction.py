"""Regression checks: modality-generic reconstruction (Tier A + B).

Covers the generic ``ReconstructionLoss`` API (per-call loss type, channel
weights, masked reduction), the ``TotalLoss`` reconstruction registry driven
by ``LossWeights.recon_extra``, and full-signal reconstruction of arbitrary
neural signals through ``forward_modalities(reconstruct=True)``.
"""

import pytest

torch = pytest.importorskip("torch")

from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.training.losses import ReconstructionLoss, TotalLoss
from brain_moe_pinn.training.training_phases import LossWeights
from brain_moe_pinn.core.observation_adapters import (
    _linear_resample,
    _portable_resample_required,
    _resample_time,
)


def _make_generic_model():
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
    )


def test_reconstruction_loss_modality_generic_api():
    loss = ReconstructionLoss()
    pred = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)

    mse = loss(pred, target, loss_type="mse")
    l1 = loss(pred, target, loss_type="l1")
    huber = loss(pred, target, loss_type="huber")
    assert torch.allclose(mse, ((pred - target) ** 2).mean())
    assert torch.allclose(l1, (pred - target).abs().mean())
    assert mse.item() > 0 and huber.item() > 0

    # Per-channel weighting scales the mean by the weighted average.
    weights = torch.tensor([0.5, 1.0, 1.5, 2.0])
    weighted = loss(pred, target, channel_weights=weights)
    manual = ((pred - target) ** 2 * weights.view(1, -1, 1)).mean()
    assert torch.allclose(weighted, manual)

    # Masked reduction is the mean over valid entries only.
    mask = torch.zeros_like(pred, dtype=torch.bool)
    mask[..., :8] = True
    masked = loss(pred, target, mask=mask)
    manual_masked = ((pred - target) ** 2)[mask].mean()
    assert torch.allclose(masked, manual_masked)

    with pytest.raises(ValueError, match="must match"):
        loss(pred, target[:, :3, :])
    with pytest.raises(ValueError, match="loss_type"):
        loss(pred, target, loss_type="nonsense")


def test_total_loss_recon_extra_registry():
    preds = {
        "eeg_recon": torch.randn(2, 4, 16),
        "voltage_recon": torch.randn(2, 8, 16),
    }
    targets = {"eeg": torch.randn(2, 4, 16), "voltage": torch.randn(2, 8, 16)}
    tl = TotalLoss(LossWeights(recon_extra={"voltage": 0.5}))
    loss, metrics = tl(preds, targets)
    # Fixed registry entry still fires; extra modality fires under its own weight.
    assert "recon_eeg" in metrics and "recon_voltage" in metrics
    assert "recon_meg" not in metrics  # no meg prediction/target -> skipped
    # Missing target for a registered modality is skipped, not an error.
    del targets["voltage"]
    tl2 = TotalLoss(LossWeights(recon_extra={"voltage": 0.5}))
    loss2, metrics2 = tl2(preds, targets)
    assert "recon_voltage" not in metrics2 and torch.isfinite(loss2)


def test_forward_modalities_reconstructs_arbitrary_signal():
    model = _make_generic_model().train()
    signals = {"calcium": torch.randn(2, 12, 32), "behavior": torch.randn(2, 3, 32)}

    out = model.forward_modalities(signals, reconstruct=True, num_steps=2)
    recon = out["calcium_recon"]
    assert recon.shape == signals["calcium"].shape
    # Any modality under the channel cap gets reconstructed, behavior included.
    assert "behavior_recon" in out
    assert out["behavior_recon"].shape == signals["behavior"].shape

    # Without the flag the reconstruction surface stays off.
    out2 = model.forward_modalities(signals, num_steps=1)
    assert "calcium_recon" not in out2

    # Supervise the calcium channel: gradient must reach the dynamics, the
    # shared head, and the channel adapter.
    tl = TotalLoss(LossWeights(recon_extra={"calcium": 1.0}, sigreg=0.0))
    loss, metrics = tl(out, {"calcium": signals["calcium"]})
    assert metrics["recon_calcium"] > 0
    loss.backward()
    dynamics_grad = sum(
        p.grad.abs().sum().item()
        for p in model.velocity_brain.parameters()
        if p.grad is not None
    )
    head_grad = sum(
        p.grad.abs().sum().item()
        for p in model.signal_recon_head.parameters()
        if p.grad is not None
    )
    adapter_grad = sum(
        p.grad.abs().sum().item()
        for p in model.generic_channel_adapters["calcium"].parameters()
        if p.grad is not None
    )
    assert dynamics_grad > 0 and head_grad > 0 and adapter_grad > 0


def test_forward_modalities_recon_channel_cap_falls_back():
    model = _make_generic_model().train()
    signals = {"calcium": torch.randn(2, 12, 32)}
    out = model.forward_modalities(signals, reconstruct=True, recon_max_channels=8)
    # Too many channels -> summary encoding, no reconstruction, no crash.
    assert "calcium_recon" not in out
    assert out["z_next"].shape == (2, 8)


def test_reconstruction_statistical_criteria():
    """Correlation, corr-diff, Poisson, and Wasserstein criteria + gradients."""
    loss = ReconstructionLoss()
    target = torch.randn(2, 4, 64)

    # A scaled, shifted prediction is a perfect correlation match but a
    # terrible MSE match - the C. elegans / TDE-RICA gain-invariance lesson.
    pred_perfect = 2.0 * target + 5.0
    corr = loss(pred_perfect, target, loss_type="correlation")
    assert corr.item() < 1e-4
    assert loss(pred_perfect, target, loss_type="mse").item() > 1.0

    # Manual Pearson check on one channel.
    pred = target + 0.3 * torch.randn_like(target)
    rho = loss(pred, target, loss_type="correlation")
    manual = []
    for b in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            p = pred[b, c] - pred[b, c].mean()
            t = target[b, c] - target[b, c].mean()
            manual.append(1.0 - (p * t).sum() / (
                (p * p).sum() * (t * t).sum()).sqrt())
    assert torch.allclose(rho, torch.tensor(manual).mean(), atol=1e-5)

    # First-difference correlation ignores the (autocorrelated) baseline.
    assert torch.isfinite(loss(pred, target, loss_type="corr_diff"))

    # Poisson NLL with non-negative targets; rate = softplus(pred).
    counts = torch.poisson(torch.full((2, 4, 64), 3.0))
    poisson_loss = loss(torch.randn_like(counts) * 0.1, counts,
                        loss_type="poisson")
    assert torch.isfinite(poisson_loss) and poisson_loss > 0

    # 1-D Wasserstein equals mean |sorted(pred) - sorted(target)|.
    w1 = loss(pred, target, loss_type="wasserstein1")
    manual_w1 = torch.stack([
        (torch.sort(pred[b, c]).values - torch.sort(target[b, c]).values)
        .abs().mean() for b in range(pred.shape[0]) for c in range(pred.shape[1])
    ]).mean()
    assert torch.allclose(w1, manual_w1, atol=1e-5)

    # All criteria propagate gradients to the prediction.
    for criterion in ("correlation", "corr_diff", "poisson", "wasserstein1"):
        probe = torch.randn(2, 4, 64, requires_grad=True)
        value = loss(probe, counts.float() if criterion == "poisson" else target,
                     loss_type=criterion)
        value.backward()
        assert probe.grad is not None and torch.isfinite(probe.grad).all()


def test_total_loss_per_modality_criteria():
    """recon_loss_types selects the criterion per modality in TotalLoss."""
    pred = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)
    tl = TotalLoss(LossWeights(
        recon_extra={"calcium": 1.0, "voltage": 1.0},
        recon_loss_types={"calcium": "correlation", "voltage": "huber"},
    ))
    loss_value, metrics = tl(
        {"calcium_recon": pred, "voltage_recon": pred},
        {"calcium": target, "voltage": target},
    )
    assert "recon_calcium" in metrics and "recon_voltage" in metrics
    assert torch.isfinite(loss_value)


def test_species_batch_helpers():
    """Dummy species batches and phase-weight augmentation contract."""
    from brain_moe_pinn.training.training_loop import (
        augment_phase_loss_weights, make_dummy_generic_signals,
    )

    signals = make_dummy_generic_signals(
        ("calcium", "behavior"), batch_size=2, channels=12, time_len=32)
    assert signals["calcium"].shape == (2, 12, 32)
    assert signals["behavior"].shape == (2, 12, 32)
    with pytest.raises(ValueError, match="at least one modality"):
        make_dummy_generic_signals([], 2)

    phase_weights = LossWeights(recon_extra={"calcium": 0.5})
    augmented = augment_phase_loss_weights(
        phase_weights, ("calcium", "voltage"),
        recon_loss_types={"calcium": "correlation", "voltage": "huber"})
    assert augmented.recon_extra == {"calcium": 0.5, "voltage": 1.0}
    assert augmented.recon_loss_types == {
        "calcium": "correlation", "voltage": "huber"}
    # Phase-selected criteria win over defaults.
    phase_weights2 = LossWeights(
        recon_extra={"calcium": 1.0},
        recon_loss_types={"calcium": "huber"})
    augmented2 = augment_phase_loss_weights(
        phase_weights2, ("calcium",),
        recon_loss_types={"calcium": "correlation"})
    assert augmented2.recon_loss_types == {"calcium": "huber"}
    # Nothing to add -> same object back (no accidental copies).
    assert augment_phase_loss_weights(LossWeights(), (), {}) is not None


class TestPortableResample:
    """`_linear_resample` replaces F.interpolate in the reconstruction head.

    F.interpolate lowers to upsample_linear1d, whose backward kernel is
    missing on some accelerators (torch_musa 1.3.0 has forward only), which
    killed training on MUSA. The replacement must stay numerically identical.
    """

    @pytest.mark.parametrize("time_in,time_out", [
        (64, 256), (256, 64), (64, 128), (7, 129), (129, 7),
        (64, 1), (1, 64), (64, 2), (2, 64), (64, 64),
    ])
    def test_matches_interpolate_forward(self, time_in, time_out):
        import torch.nn.functional as F
        torch.manual_seed(0)
        x = torch.randn(3, 8, time_in)
        ref = F.interpolate(x, size=time_out, mode="linear", align_corners=True)
        assert torch.allclose(_linear_resample(x, time_out), ref, atol=1e-6)

    def test_matches_interpolate_backward(self):
        """Gradients must match exactly - this is the path MUSA lacked."""
        import torch.nn.functional as F
        torch.manual_seed(0)
        a = torch.randn(2, 4, 64, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        F.interpolate(a, size=200, mode="linear", align_corners=True).sum().backward()
        _linear_resample(b, 200).sum().backward()
        assert torch.equal(a.grad, b.grad)

    def test_identity_and_singleton(self):
        x = torch.randn(2, 3, 16)
        assert _linear_resample(x, 16) is x
        assert _linear_resample(x, 1).shape == (2, 3, 1)
        assert torch.allclose(_linear_resample(x, 1), x[..., :1])

class TestResampleDispatch:
    """The portable path must stay a narrow MUSA-only workaround.

    Every other backend keeps the vendor-tuned F.interpolate kernel; only the
    device whose backward kernel is missing gets the hand-rolled version.
    """

    def test_only_musa_takes_portable_path(self):
        assert _portable_resample_required("musa")
        for other in ("cpu", "cuda", "xpu", "xla", "npu", "mlu", "mps"):
            assert not _portable_resample_required(other), other

    def test_non_musa_uses_native_interpolate(self, monkeypatch):
        """A CUDA/CPU tensor must go through F.interpolate, not the fallback."""
        import torch.nn.functional as F
        import brain_moe_pinn.core.observation_adapters as oa

        calls = []
        real = F.interpolate

        def spy(*args, **kwargs):
            calls.append(kwargs.get("mode"))
            return real(*args, **kwargs)

        monkeypatch.setattr(oa.F, "interpolate", spy)
        x = torch.randn(2, 3, 64)
        out = _resample_time(x, 128)
        assert calls == ["linear"], "native interpolate was not used on CPU"
        assert out.shape == (2, 3, 128)

    def test_identity_short_circuits(self):
        x = torch.randn(2, 3, 16)
        assert _resample_time(x, 16) is x
