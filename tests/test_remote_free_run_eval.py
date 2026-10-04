"""Regression coverage for sampled SDE free-run evaluation."""

import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brain_moe_pinn import BrainMoEPINN
from tools.remote_free_run_eval import (
    _fit_tderica_context_projections,
    _forward_free_run_paths,
    _path_ensemble_diagnostics,
    _pooled_tderica_distribution,
    _sampled_horizon_reports,
    _tderica_horizon_reports,
)
from brain_moe_pinn.diagnostics.free_run_metrics import (
    sde_transition_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
TDERICA = ROOT.parent / "TDE-RICA"


def _tiny_sde_model():
    torch.manual_seed(23)
    return BrainMoEPINN(
        eeg_channels=2,
        fmri_regions=3,
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        use_species_conditioning=False,
        perturbation_dim=2,
        moe_num_shared=1,
        moe_num_routed=1,
        moe_top_k=1,
        use_moe=False,
        generic_observation_only=True,
        transition_mode="sde",
        diffusion_rank=2,
        stochastic_samples=4,
        scale_anchor="off",
    ).eval()


def test_evaluator_keeps_mean_path_separate_from_sampled_realizations():
    model = _tiny_sde_model()
    signals = {"calcium": torch.randn(1, 4, 24)}
    outputs = _forward_free_run_paths(
        model,
        signals,
        masks={"calcium": torch.ones_like(signals["calcium"], dtype=torch.bool)},
        rollout_steps=3,
        dt=torch.tensor([6.0]),
        frame_dt=torch.tensor([0.25]),
    )

    mean_path = outputs["transition_mean_path"]
    sampled_paths = outputs["sampled_paths"]
    assert outputs["sample_count"] == 4
    assert torch.equal(
        mean_path["z_next_sequence"],
        mean_path["transition_mean_sequence"],
    )
    assert sampled_paths["z_next_sequence_samples"].shape == (4, 1, 3, 8)
    assert sampled_paths["calcium_recon_samples"].shape == (4, 1, 3, 4, 24)
    assert sampled_paths["z_next_sequence_samples"].std(dim=0).mean() > 0
    assert mean_path is not sampled_paths
    diagnostics = sde_transition_diagnostics(
        sampled_paths["z_global"].detach().cpu().numpy(),
        sampled_paths["z_next_sequence_samples"].detach().cpu().numpy(),
        sampled_paths[
            "transition_mean_sequence_samples"
        ].detach().cpu().numpy(),
        sampled_paths[
            "transition_factor_sequence_samples"
        ].detach().cpu().numpy(),
        sampled_paths[
            "transition_diag_std_sequence_samples"
        ].detach().cpu().numpy(),
        model.transition_diffusion.factor_basis.detach().cpu().numpy(),
    )
    assert diagnostics["sde_noise_increment_rms"] > 0
    assert diagnostics["sde_trace_q"] > 0
    assert diagnostics["sde_diffusion_rms"] == pytest.approx(
        np.sqrt(diagnostics["sde_trace_q"] / 8))



def test_sde_sampled_runtime_state_round_trips_through_checkpoint():
    model = _tiny_sde_model()
    signals = {"calcium": torch.randn(1, 4, 24)}
    outputs = _forward_free_run_paths(
        model,
        signals,
        masks={"calcium": torch.ones_like(
            signals["calcium"], dtype=torch.bool)},
        rollout_steps=3,
        dt=torch.tensor([6.0]),
        frame_dt=torch.tensor([0.25]),
    )
    assert outputs["sample_count"] == 4
    assert model.velocity_brain.mt_kda.state_a.shape == (4, 8)

    buffer = io.BytesIO()
    torch.save({"model_state": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=False)
    restored = _tiny_sde_model()
    restored.load_state_dict(checkpoint["model_state"], strict=False)
    assert torch.equal(
        restored.transition_diffusion.rank_scale[-1].weight,
        model.transition_diffusion.rank_scale[-1].weight,
    )


def test_pooled_tderica_distribution_uses_realizations_not_their_mean():
    if not TDERICA.is_dir():
        pytest.skip("TDE-RICA toolbox is unavailable")
    real = np.zeros((12, 1), dtype=float)
    generated = [
        np.ones((12, 1), dtype=float),
        -np.ones((12, 1), dtype=float),
    ]
    pooled = _pooled_tderica_distribution(
        real, generated, toolbox_path=TDERICA)

    from tderica import kl_divergence_1d, wasserstein_distance

    all_generated = np.concatenate(generated, axis=0)
    mean_path = np.mean(np.stack(generated), axis=0)
    assert pooled["wasserstein_global"] == pytest.approx(
        wasserstein_distance(real, all_generated))
    assert pooled["wasserstein_global"] > wasserstein_distance(real, mean_path)
    assert pooled["kl_divergence"] == pytest.approx(
        np.mean(kl_divergence_1d(real, all_generated)))
    assert pooled["n_realizations"] == 2
    assert pooled["n_sample_features"] == 24


def test_context_tderica_projection_fits_once_for_all_sample_paths():
    if not TDERICA.is_dir():
        pytest.skip("TDE-RICA toolbox is unavailable")
    rng = np.random.default_rng(7)
    time = np.arange(40, dtype=float)
    context = np.stack([
        np.sin(time * 0.17),
        np.cos(time * 0.11),
        np.sin(time * 0.07 + 0.5),
    ], axis=1)
    real = rng.normal(size=(18, 3))
    samples = [rng.normal(size=(18, 3)), rng.normal(size=(18, 3))]

    projected, motifs = _fit_tderica_context_projections(
        context,
        [real, *samples],
        dim_embed=3,
        n_components=2,
        toolbox_path=TDERICA,
    )

    assert len(projected) == 3
    assert motifs.shape == (2, 3, 3)
    assert all(value.shape == (16, 2) for value in projected)
    assert all(np.isfinite(value).all() for value in projected)



def test_sampled_horizons_report_r_sigma_and_pathwise_variance():
    time = np.arange(12, dtype=float)
    real = np.concatenate([
        np.column_stack([
            np.sin(time[:6] * 0.4), np.cos(time[:6] * 0.4)]),
        np.column_stack([
            np.sin(time[6:] * 0.7), np.cos(time[6:] * 0.7)]),
    ])
    samples = np.stack([2.0 * real, 2.0 * real])
    mean_path = 2.0 * real

    reports = _sampled_horizon_reports(real, samples, mean_path, rollouts=2)

    assert [entry["horizon"] for entry in reports] == [1, 2]
    assert [entry["r_sigma"] for entry in reports] == pytest.approx([2.0, 2.0])
    assert all(entry["between_sample_variance"] == pytest.approx(0.0)
               for entry in reports)
    assert all(entry["within_path_temporal_variance"] > 0
               for entry in reports)
    assert all(entry["mean_path_native"] for entry in reports)


def test_path_diagnostics_separate_sample_spread_and_temporal_variance():
    time = np.arange(10, dtype=float)[:, None]
    real = time * np.array([[1.0, 2.0]])
    samples = np.stack([real + 1.0, real - 1.0])

    result = _path_ensemble_diagnostics(real, real, samples)

    assert result["between_sample_variance"] == pytest.approx(1.0)
    assert result["within_path_temporal_variance"] == pytest.approx(
        np.var(real, axis=0).mean())
    assert result["real_temporal_variance"] == pytest.approx(
        np.var(real, axis=0).mean())
    assert result["realization_lag1"]["real"] == pytest.approx(1.0)
    assert result["realization_lag1"]["mean"] == pytest.approx(1.0)
    assert result["realization_lag1"]["per_realization"] == pytest.approx(
        [1.0, 1.0])


def test_horizon_tderica_reports_keep_mean_and_pooled_distributions_separate():
    if not TDERICA.is_dir():
        pytest.skip("TDE-RICA toolbox is unavailable")

    time = np.arange(10, dtype=float)
    real_blocks = []
    mean_blocks = []
    sample_a_blocks = []
    sample_b_blocks = []
    for horizon in range(8):
        scale = float(horizon + 1)
        real = np.column_stack([
            scale * np.sin(time * 0.31),
            scale * np.cos(time * 0.19),
        ])
        mean = 0.6 * real
        offset = np.array([0.3 * scale, -0.2 * scale])
        real_blocks.append(real)
        mean_blocks.append(mean)
        sample_a_blocks.append(mean + offset)
        sample_b_blocks.append(mean - offset)
    real_all = np.concatenate(real_blocks)
    mean_all = np.concatenate(mean_blocks)
    sample_paths = np.stack([
        np.concatenate(sample_a_blocks),
        np.concatenate(sample_b_blocks),
    ])

    reports = _tderica_horizon_reports(
        real_all,
        mean_all,
        sample_paths,
        8,
        projector=lambda signal: signal,
        toolbox_path=TDERICA,
        include_d3=True,
    )
    from tderica import kernel_transition_comparison

    first = reports[0]["tderica"]
    second = reports[1]["tderica"]
    assert first["mean_path_distribution"]["wasserstein_global"] != pytest.approx(
        second["mean_path_distribution"]["wasserstein_global"])
    expected_mean = _pooled_tderica_distribution(
        real_blocks[0], [mean_blocks[0]], toolbox_path=TDERICA)
    expected_pooled = _pooled_tderica_distribution(
        real_blocks[0],
        [sample_a_blocks[0], sample_b_blocks[0]],
        toolbox_path=TDERICA,
    )
    assert first["mean_path_distribution"]["wasserstein_global"] == pytest.approx(
        expected_mean["wasserstein_global"])
    assert first["pooled_sample_distribution"]["wasserstein_global"] == pytest.approx(
        expected_pooled["wasserstein_global"])
    assert first["pooled_sample_distribution"]["n_realizations"] == 2
    assert first["mean_path_kernel_transition"] == pytest.approx(
        kernel_transition_comparison(real_blocks[0], mean_blocks[0]))
    assert len(first["sample_kernel_transition_per_realization"]) == 2


def test_tderica_horizon_reports_skip_windows_too_short_for_embedding(
        monkeypatch):
    monkeypatch.syspath_prepend("unused")
    toolbox = types.ModuleType("tderica")
    toolbox.kernel_transition_comparison = lambda *_: 0.0
    monkeypatch.setitem(sys.modules, "tderica", toolbox)

    real = np.zeros((32, 2), dtype=np.float32)
    sampled = np.stack([real, real])
    projector_calls = []

    def projector(signal):
        projector_calls.append(signal.shape[0])
        raise AssertionError("short horizons must be skipped before projection")

    reports = _tderica_horizon_reports(
        real,
        real,
        sampled,
        8,
        projector=projector,
        toolbox_path=Path("unused"),
        include_d3=False,
        minimum_input_frames=5,
    )

    assert projector_calls == []
    assert len(reports) == 8
    assert all(report["tderica"] is None for report in reports)
    assert all(
        report["skipped_reason"]
        == "requires at least 5 input frames for three projected frames"
        for report in reports
    )
