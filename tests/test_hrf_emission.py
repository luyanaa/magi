"""Internal state -> BOLD: the learnable haemodynamic emission.

The structural reference is Friston, Harrison & Penny (2003), NeuroImage
19:1273-1302 (Balloon-Windkessel forward model, Table 1 priors). Because the
drive here is an internal latent rather than the DCM's neuronal state, the
response is *initialised* from that reference and then learned -- these tests
pin both halves of that contract.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from brain_moe_pinn.core.emission import SensorEmission
from brain_moe_pinn.core.hrf import (
    BalloonParameters, BalloonWindkessel, LearnedHemodynamicResponse,
    bold_k_constants, reference_impulse_response,
)
from brain_moe_pinn.core.sensors import resolve_sensor


def _drive(n: int, channels: int = 1, width: int = 1, amplitude: float = 1.0):
    drive = torch.zeros(1, channels, n)
    drive[:, :, :width] = amplitude
    return drive


def test_bold_k_constants_match_the_paper():
    k1, k2, k3 = bold_k_constants(rho=0.34, v0=0.02)
    assert k1 == pytest.approx(7 * 0.34)      # 2.38
    assert k2 == pytest.approx(2.0)
    assert k3 == pytest.approx(2 * 0.34 - 0.2)  # 0.48


def test_reference_response_has_a_canonical_shape():
    dt = 0.1
    response = reference_impulse_response(duration_s=32.0, dt=dt).numpy()
    peak_s = float(np.argmax(response)) * dt
    assert 1.5 <= peak_s <= 5.0, peak_s                 # ~3 s for an impulse
    assert response[0] < response.max() * 0.5           # rises after onset
    assert np.argmin(response) * dt > peak_s            # undershoot follows
    assert abs(response[-1]) < 0.05 * response.max()    # returns to baseline
    assert np.all(np.isfinite(response))


def test_reference_model_rests_at_zero_bold():
    model = BalloonWindkessel(max_channels=2)
    out = model(_drive(64, channels=2, width=0), dt=0.5)
    assert float(out.abs().max()) < 1e-9


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_learned_emission_starts_as_the_reference_response(dt):
    reference = reference_impulse_response(duration_s=32.0, dt=dt)
    model = LearnedHemodynamicResponse(max_channels=3, init_from=(reference, dt))
    n = int(round(32.0 / dt))
    response = model(_drive(n, channels=1), dt)[0, 0].detach().numpy()
    assert np.corrcoef(response, reference.numpy())[0, 1] > 0.999
    relative = (np.linalg.norm(response - reference.numpy())
                / np.linalg.norm(reference.numpy()))
    assert relative < 0.01, relative


def test_emission_is_causal_and_reflects_the_reference_delay():
    dt = 0.5
    reference = reference_impulse_response(duration_s=32.0, dt=dt)
    model = LearnedHemodynamicResponse(max_channels=2, init_from=(reference, dt))
    summary = model.parameter_summary(1, dt=dt)
    # peak latency is a property of the reference, not of the drive
    assert 1.5 <= summary["peak_delay_s"] <= 5.0
    assert summary["reference_deviation"] < 1e-6   # nothing learned yet
    assert summary["fwhm_s"] > 0.5
    assert 0.0 <= summary["undershoot_ratio"] < 1.0
    # an impulse at the end of the window cannot produce an earlier response
    late = torch.zeros(1, 1, 40)
    late[0, 0, -1] = 1.0
    response = model(late, dt)[0, 0].detach().numpy()
    assert np.abs(response[:-1]).max() < 1e-6


def test_tr_sampling_preserves_the_underlying_response():
    """A longer TR changes only the sampling grid, not the response shape."""
    params = BalloonParameters()
    reference = reference_impulse_response(duration_s=40.0, dt=0.25, params=params)
    fine = LearnedHemodynamicResponse(max_channels=1, init_from=(reference, 0.25))
    fast = fine(_drive(160), 0.25)[0, 0].detach()
    slow = fine(_drive(40), 1.0)[0, 0].detach()
    # both start from the same drive (one impulse), so the TR=1 s response is
    # the TR=0.25 s response decimated by averaging
    decimated = fast.reshape(40, 4).mean(dim=1)
    correlation = float(torch.corrcoef(torch.stack([slow, decimated]))[0, 1])
    assert correlation > 0.95


def _correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-channel correlation loss (what the repo uses for signal recon)."""
    pred = prediction - prediction.mean(dim=-1, keepdim=True)
    tgt = target - target.mean(dim=-1, keepdim=True)
    num = (pred * tgt).sum(dim=-1)
    den = (pred.pow(2).sum(dim=-1) * tgt.pow(2).sum(dim=-1)).clamp_min(1e-12).sqrt()
    return (1.0 - num / den).mean()


def test_corpus_specific_haemodynamics_are_learnable():
    """Training must be able to move the emission off Friston's priors.

    The emission starts as the reference response; fitting it to BOLD generated
    with *different* haemodynamic parameters (a faster/slower corpus) must
    reduce a correlation loss and raise the model-target correlation. This is
    the property that makes the observation channel learned rather than
    imposed -- the drive here is a latent, not the DCM's neuronal state.
    """
    torch.manual_seed(0)
    generator = BalloonWindkessel(
        max_channels=1, params=BalloonParameters(tau=1.6, kappa=0.4))
    drive = torch.zeros(1, 1, 72)
    drive[0, 0, 4:12] = 1.0
    target = generator(drive, 0.5)

    model = LearnedHemodynamicResponse(max_channels=1)
    before = model.parameter_summary(1, dt=0.5)["reference_deviation"]
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(600):
        optimiser.zero_grad()
        loss = _correlation_loss(model(drive, 0.5), target)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))
    after = model.parameter_summary(1, dt=0.5)["reference_deviation"]
    with torch.no_grad():
        prediction = model(drive, 0.5)[0, 0].numpy()
    correlation = float(np.corrcoef(prediction, target[0, 0].numpy())[0, 1])
    assert losses[-1] < losses[0]
    assert correlation > 0.95, correlation
    assert after > before            # the kernel did move away from the prior


def test_frozen_emission_is_deterministic():
    reference = reference_impulse_response(duration_s=16.0, dt=0.5)
    model = LearnedHemodynamicResponse(max_channels=2, init_from=(reference, 0.5),
                                       learnable=False)
    drive = _drive(32, channels=1, width=2)
    first = model(drive, 0.5)
    second = model(drive, 0.5)
    assert torch.allclose(first, second)
    assert not any(p.requires_grad for p in model.parameters())


def test_bold_readout_flows_from_the_profile():
    root = Path(__file__).resolve().parents[1]
    from brain_moe_pinn.config import ExperimentConfig

    for species, tr in (("human", 2.0), ("mouse", 1.0)):
        experiment = ExperimentConfig.from_file(
            root / "configs" / "species" / f"{species}.json")
        spec = experiment.sensor_specs()["fmri"]
        assert spec.readout == "bold"
        assert spec.imaging.name == "bold_epi"
        assert spec.reporter.family == "hemodynamic"
        assert spec.hrf and spec.hrf["tau"] == pytest.approx(0.98)
        emission = SensorEmission(max_channels=4, spec=spec)
        drive = _drive(32, channels=4, width=2)
        out = emission(drive, dt=tr)
        assert out.shape == drive.shape
        summary = emission.parameter_summary(4)
        assert summary["readout"] == "bold"
        assert summary["imaging"] == "bold_epi"
        assert summary["reporter"] == "bold_hemodynamic"
        assert 1.0 <= summary["peak_delay_s"] <= 6.0


def test_bold_sensor_validation_rejects_mismatches():
    with pytest.raises(ValueError, match="haemodynamic"):
        resolve_sensor("fmri", {"imaging": "bold_epi", "reporter": "gcamp6f"})
    with pytest.raises(ValueError, match="needs a haemodynamic"):
        resolve_sensor("fmri", {"imaging": "electrical", "reporter": "voltron"})
    with pytest.raises(ValueError, match="belongs to the fmri modality"):
        resolve_sensor("calcium", {"imaging": "spinning_disk_4d",
                                   "reporter": "bold_hemodynamic",
                                   "readout": "bold"})
    # BOLD is a dynamic readout, so it may carry next-step targets
    spec = resolve_sensor("fmri", {"imaging": "bold_epi",
                                   "reporter": "bold_hemodynamic"})
    assert spec.dynamics_valid is True
