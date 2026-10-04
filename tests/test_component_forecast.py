"""Contracts for frozen TDE-RICA component forecast supervision."""

import os
import sys
from pathlib import Path

import pytest
import numpy as np
torch = pytest.importorskip("torch")

from brain_moe_pinn.training.losses import (
    ComponentForecastLoss,
    FixedComponentProjector,
    TotalLoss,
)
from brain_moe_pinn.training.training_loop import apply_component_forecast_spec
from brain_moe_pinn.training.training_phases import LossWeights

from brain_moe_pinn.data.species_dataset import SpeciesSignalDataset

def _projection(embed_width=2, channels=2):
    # Delay-window flattening is [frame_0/channel_0, frame_0/channel_1,
    # frame_1/channel_0, frame_1/channel_1].
    matrix = torch.zeros(embed_width * channels, 1)
    matrix[0, 0] = 1.0
    matrix[-1, 0] = 1.0
    return matrix


def test_component_profile_overrides_only_forecast_phase_weights():
    spec = {
        "loss_weight": 0.2,
        "wasserstein_weight": 1.0,
        "variance_weight": 0.5,
    }
    disabled = apply_component_forecast_spec(
        LossWeights(forecast=0.0), spec)
    enabled = apply_component_forecast_spec(
        LossWeights(forecast=1.0), spec)

    assert disabled.forecast_component == 0.0
    assert enabled.forecast_component == pytest.approx(0.2)
    assert enabled.forecast_component_variance == pytest.approx(0.5)

def test_declared_channel_order_is_shared_by_each_dataset_split(tmp_path):
    dataset = SpeciesSignalDataset(
        tmp_path,
        modalities=("calcium",),
        seq_len=2,
        rows=[],
        align_channels=True,
        channel_order=("B", "A"),
    )

    assert dataset._global_ids == ["B", "A"]

def test_fixed_projector_matches_tderica_delay_order():
    projector = FixedComponentProjector(_projection(), 2, 2)
    signal = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])

    projected, mask = projector(signal)

    assert projected.shape == (1, 1, 2)
    assert torch.allclose(projected, torch.tensor([[[21.0, 32.0]]]))
    assert mask.shape == projected.shape
    assert bool(mask.all())

def _import_tderica():
    candidates = [
        os.environ.get("TDERICA_PATH"),
        str(Path(__file__).resolve().parents[2] / "TDE-RICA"),
        "/root/TDE-RICA",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            break
    return pytest.importorskip("tderica")


def test_fixed_projector_matches_tderica_project():
    tderica = _import_tderica()
    rng = np.random.default_rng(17)
    embed_width, channels, components, time = 3, 4, 5, 9
    signal = rng.normal(size=(channels, time))
    coeff = rng.normal(size=(components, embed_width, channels))
    embedded = np.stack([
        signal[:, start:start + embed_width].T
        for start in range(time - embed_width + 1)
    ])
    expected = tderica.project(embedded, coeff)
    projection = np.linalg.pinv(coeff.reshape(components, -1).T).T
    projector = FixedComponentProjector(
        torch.from_numpy(projection.astype(np.float32)),
        embed_width,
        channels,
    )

    projected, mask = projector(torch.from_numpy(signal).unsqueeze(0))

    assert bool(mask.all())
    assert np.allclose(
        projected[0].detach().numpy().T,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

def test_fixed_projector_can_score_zero_filled_missing_channels():
    projector = FixedComponentProjector(
        _projection(), 2, 2, missing_channel_as_zero=True)
    signal = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, True], [False, False, False]]])

    projected, component_mask = projector(signal, mask=mask)

    assert bool(component_mask.all())
    assert torch.allclose(projected, torch.tensor([[[1.0, 2.0]]]))

def test_fixed_projector_observed_lstsq_uses_available_channels():
    coefficients = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    projection = torch.linalg.pinv(
        coefficients.reshape(1, -1).transpose(0, 1)).transpose(0, 1)
    projector = FixedComponentProjector(
        projection, 2, 2,
        coefficient_tensor=coefficients,
        missing_channel_mode="observed_lstsq",
    )
    signal = torch.tensor([[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[[True, True, True], [False, False, False]]])

    projected, component_mask = projector(signal, mask=mask)

    assert bool(component_mask.all())
    assert torch.allclose(
        projected,
        torch.tensor([[[1.0, 1.6]]]),
        atol=1e-6,
    )

def test_component_forecast_loss_backpropagates_and_respects_masks():
    criterion = ComponentForecastLoss(
        _projection(), 2, 2, wasserstein_weight=1.0, variance_weight=1.0)
    prediction = torch.randn(1, 2, 2, 3, requires_grad=True)
    target = prediction.detach() + 0.2
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 0] = False

    value, metrics = criterion(prediction, target, mask=mask)

    assert torch.isfinite(value)
    assert value.item() > 0
    assert set(metrics) == {"wasserstein", "variance"}
    value.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_total_loss_wires_component_forecast_term():
    weights = LossWeights(
        recon_extra={"calcium": 1.0},
        forecast=1.0,
        forecast_component=0.5,
        forecast_component_wasserstein=1.0,
        forecast_component_variance=0.0,
        sigreg=0.0,
    )
    total = TotalLoss(weights)
    total.use_loss_normalization = False
    total.configure_component_forecast(_projection(), 2, 2)
    prediction = torch.randn(1, 2, 2, 3, requires_grad=True)
    target = prediction.detach() + 0.1

    value, metrics = total(
        {
            "calcium_recon": prediction[:, -1],
            "calcium_recon_sequence": prediction,
        },
        {"calcium": target},
    )

    assert torch.isfinite(value)
    assert metrics["forecast_component_calcium"] > 0
    assert metrics["forecast_component_calcium_wasserstein"] > 0
    assert metrics["forecast_component_calcium_variance"] == pytest.approx(0.0)
    value.backward()
    assert prediction.grad is not None


def test_component_forecast_rejects_basis_channel_mismatch():
    criterion = ComponentForecastLoss(_projection(), 2, 2)
    prediction = torch.randn(1, 1, 3, 3)
    target = torch.randn_like(prediction)

    with pytest.raises(ValueError, match="fixed basis requires"):
        criterion(prediction, target)
