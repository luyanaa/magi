"""Regression tests for the audit-driven data, dynamics, and objective fixes."""

from pathlib import Path

import numpy as np
import pytest
import torch

from brain_moe_pinn.core.active_inference import (
    ActiveInferenceController, ClosedLoopFeedback, PrecisionGate)
from brain_moe_pinn.core.moe import MoEVelocityField, MoEGenericVelocityField
from brain_moe_pinn.core.observation_adapters import SignalReconstructionHead
from brain_moe_pinn.core.counterfactual_search import (
    CounterfactualTreeSearch, TrajectoryRollout)
from brain_moe_pinn.core.velocity_brain import (
    EnergyEntropyFields, MultiTimeScaleKDA, OUStructuredNoise, VelocityBrain,
    oettinger_mobility_action)
from brain_moe_pinn.diagnostics.free_run_metrics import intrinsic_rollout_stats
from brain_moe_pinn.evaluation.resting_state import compute_avalanche_stats
from brain_moe_pinn.training.losses import (
    DissipationLoss, LatentSparsityLoss, SpectralSlopeLoss, TotalLoss,
    aperiodic_exponent)
from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.data.data_loader import PairedBrainDataset
from brain_moe_pinn.data.species_dataset import SpeciesSignalDataset
from brain_moe_pinn.training.losses import SpectralSlopeLoss, TotalLoss
from brain_moe_pinn.training.training_loop import (
    BrainMoETrainer, partition_generic_batch)
from brain_moe_pinn.training.training_phases import LossWeights

def _write_array(root: Path, modality: str, name: str, value) -> None:
    directory = root / modality
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{name}.npy", np.asarray(value))


def test_paired_dataset_returns_distinct_future_target(tmp_path):
    root = tmp_path / "paired"
    _write_array(root, "eeg", "sample", np.arange(8, dtype=np.float32)[None, :])
    _write_array(root, "fmri", "sample", np.arange(8, dtype=np.float32)[None, :])

    item = PairedBrainDataset(
        root, return_next_step_targets=True)[0]
    assert torch.equal(item["eeg"], torch.tensor([[0., 1., 2., 3.]]))
    assert torch.equal(item["eeg_next"], torch.tensor([[4., 5., 6., 7.]]))
    assert not torch.equal(item["eeg"], item["eeg_next"])


def test_paired_dataset_rejects_ambiguous_odd_length_future(tmp_path):
    root = tmp_path / "paired"
    _write_array(root, "eeg", "sample", np.arange(7, dtype=np.float32)[None, :])
    _write_array(root, "fmri", "sample", np.arange(8, dtype=np.float32)[None, :])
    dataset = PairedBrainDataset(root, return_next_step_targets=True)
    with pytest.raises(ValueError, match="even-length"):
        dataset[0]


def test_species_next_window_uses_physical_time_and_masks_padding(tmp_path):
    # Rates differ, but the second window begins at the same physical time.
    _write_array(tmp_path, "calcium", "s", np.arange(40, dtype=np.float32)[None, :])
    _write_array(tmp_path, "voltage", "s", np.arange(60, dtype=np.float32)[None, :])
    rows = [
        {
            "sample_id": "s",
            "calcium": "calcium/s.npy",
            "voltage": "voltage/s.npy",
            "calcium_rate_hz": 10,
            "voltage_rate_hz": 15,
        }
    ]
    dataset = SpeciesSignalDataset(
        tmp_path,
        ["calcium", "voltage"],
        rows=rows,
        seq_len=None,
        seq_seconds=1.0,
        random_windows=False,
        return_next_step_targets=True,
    )
    item = dataset[0]
    assert item["calcium"].shape[-1] == 10
    assert item["voltage"].shape[-1] == 15
    assert item["calcium_next"][0, 0].item() == 10
    assert item["voltage_next"][0, 0].item() == 15
    assert item["calcium_next_mask"].all()
    assert item["voltage_next_mask"].all()


def test_generic_partition_respects_declared_control_role():
    batch = {
        "calcium": torch.ones(2, 3, 4),
        "calcium_next": torch.full((2, 3, 4), 2.0),
        "stimulus": torch.arange(8, dtype=torch.float32).view(2, 1, 4),
        "stimulus_next": torch.zeros(2, 1, 4),
    }
    signals, targets, perturbation = partition_generic_batch(
        batch,
        signal_modalities=("calcium",),
        recon_modalities=("calcium",),
        device=torch.device("cpu"),
        control_modalities=("stimulus",),
        require_future_targets=True,
    )
    assert set(signals) == {"calcium"}
    assert torch.equal(targets["calcium"], batch["calcium_next"])
    # Default reduction is per-step ("resample"), so the perturbation always
    # carries a step axis: (B, rollout_steps, U).
    assert perturbation.shape == (2, 1, 1)


def test_router_dispatch_is_differentiable():
    torch.manual_seed(2)
    field = MoEVelocityField(
        hidden_dim=4,
        num_shared=1,
        num_routed=2,
        top_k=1,
        dropout=0.0,
    ).train()
    z = torch.randn(3, 4)
    field(z)["velocity"].sum().backward()
    router_grads = [
        parameter.grad.abs().sum()
        for parameter in field.router.parameters()
        if parameter.grad is not None
    ]
    assert router_grads and sum(router_grads).item() > 0


def test_kda_default_isolated_between_batches():
    torch.manual_seed(3)
    kda = MultiTimeScaleKDA(hidden_dim=4)
    first_batch = torch.randn(2, 4)
    unrelated_batch = torch.randn(2, 4)
    first = kda(first_batch)
    kda(unrelated_batch)
    repeated = kda(first_batch)
    assert torch.allclose(first, repeated)


def test_active_feedback_applies_latent_correction():
    torch.manual_seed(4)
    feedback = ClosedLoopFeedback(latent_dim=4, feedback_strength=1.0).eval()
    z = torch.randn(2, 4, requires_grad=True)
    result = feedback(z, torch.ones(2, 4), torch.zeros(2, 4))
    assert result["weighted_error"] is not None
    assert not torch.allclose(result["corrected_z"], z)
    result["corrected_z"].sum().backward()
    assert feedback.feedback_fusion[0].weight.grad is not None


def test_router_state_reset_is_batch_isolated():
    torch.manual_seed(5)
    field = MoEVelocityField(
        hidden_dim=4,
        num_shared=1,
        num_routed=2,
        top_k=1,
        dropout=0.0,
    ).eval()
    first_batch = torch.randn(2, 4)
    unrelated_batch = torch.randn(2, 4)
    field.reset_router_state()
    first = field(first_batch)["velocity"]
    field(unrelated_batch)
    field.reset_router_state()
    repeated = field(first_batch)["velocity"]
    assert torch.allclose(first, repeated)


def test_spectral_slope_uses_signed_pink_noise_target():
    length = 256
    frequencies = torch.arange(1, length // 2, dtype=torch.float32)
    phases = torch.arange(1, length // 2, dtype=torch.float32) * 0.37

    def colored_noise(power):
        spectrum = torch.zeros(length // 2 + 1, dtype=torch.complex64)
        amplitude = frequencies.pow(power / 2)
        spectrum[1:length // 2] = amplitude * torch.exp(1j * phases)
        return torch.fft.irfft(spectrum, n=length)

    pink = colored_noise(-1.0)
    blue = colored_noise(1.0)
    loss = SpectralSlopeLoss(weight=1.0)
    pink_value, pink_metrics = loss(pink.view(1, 1, -1))
    blue_value, blue_metrics = loss(blue.view(1, 1, -1))
    assert pink_metrics["psd_slope"] < -0.5
    assert blue_metrics["psd_slope"] > 0.5
    assert pink_value < blue_value


def test_validation_loss_does_not_update_normalizer():
    weights = LossWeights(
        recon_eeg=1.0,
        recon_fmri=0.0,
        recon_meg=0.0,
        cross_modal=0.0,
        dissip=0.0,
        spectrum=0.0,
        bandpower=0.0,
        sigreg=0.0,
    )
    total_loss = TotalLoss(weights)
    prediction = {"eeg_recon": torch.ones(1, 1, 8)}
    target = {"eeg": torch.zeros(1, 1, 8)}
    total_loss(prediction, target, update_normalizer=True)
    before = total_loss.normalizer.get_stats()
    total_loss({"eeg_recon": torch.full((1, 1, 8), 10.0)}, target,
               update_normalizer=False)
    after = total_loss.normalizer.get_stats()
    assert after == before
 
 
def test_validation_runs_physics_autograd_and_diagnostics():
    torch.manual_seed(6)
    model = BrainMoEPINN(
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
    ).eval()
    trainer = object.__new__(BrainMoETrainer)
    trainer.val_dataloader = [{
        "calcium": torch.randn(1, 4, 16),
        "calcium_future": torch.randn(1, 3, 4, 16),
        "calcium_mask": torch.ones(1, 4, 16, dtype=torch.bool),
        "calcium_future_mask": torch.ones(1, 3, 4, 16, dtype=torch.bool),
    }]
    trainer.generic_model = True
    trainer.device = torch.device("cpu")
    trainer.require_future_targets = True
    trainer.signal_modalities = ("calcium",)
    trainer.recon_modalities = ("calcium",)
    trainer.control_modalities = ()
    trainer.control_reduction = "mean"
    trainer.recon_max_channels = 2048
    trainer.total_loss = TotalLoss(LossWeights(
        recon_eeg=0.0,
        recon_fmri=0.0,
        recon_extra={"calcium": 1.0},
        cross_modal=0.0,
        dissip=0.0,
        spectrum=0.0,
        bandpower=0.0,
        sigreg=0.0,
        generic_constraint=0.0,
        moe_load_balance=0.0,
        grassmannian_reg=0.0,
    ))
    trainer.logger = None
    trainer._rollout_steps_override = 3
    first = trainer.validate(model, {"rollout_steps": 1}, 0, 2)
    second = trainer.validate(model, {"rollout_steps": 1}, 0, 2)
    assert {
        "val_loss",
        "val_causal_forward_reverse_gap",
        "val_one_step_corr_matrix_mse",
        "val_free_run_step_norm",
        "val_free_run_tail_std_ratio",
    } <= set(first)
    assert first["val_loss"] == second["val_loss"]
    assert (
        first["val_causal_forward_reverse_gap"]
        == second["val_causal_forward_reverse_gap"]
    )
 
 
def test_species_builder_preserves_held_out_test_split(tmp_path):
    for index in range(4):
        _write_array(
            tmp_path, "calcium", f"s{index}",
            np.ones((2, 8), dtype=np.float32) * index,
        )
    from brain_moe_pinn.data.species_dataset import build_species_dataloaders

    loaders = build_species_dataloaders(
        tmp_path,
        ["calcium"],
        batch_size=1,
        seq_len=4,
        val_frac=0.25,
        test_frac=0.25,
        manifest=None,
    )
    assert len(loaders) == 3
    assert len(loaders[2].dataset) == 1
 
 
def test_rollout_scales_deterministic_velocity_by_latent_dt():
    class ConstantVelocity(torch.nn.Module):
        integration_dt = 0.25

        def forward(self, z):
            return {"delta_z": torch.ones_like(z)}

    rollout = TrajectoryRollout(latent_dim=2, num_steps=2, dt=1.0)
    rollout.set_velocity_net(ConstantVelocity())
    initial = torch.zeros(1, 1, 2)
    actions = torch.zeros(1, 1, 2)
    trajectories, _ = rollout(initial[:, 0], actions)
    assert torch.allclose(trajectories[0, 0, -1], torch.full((2,), 0.5))
 
 
def test_counterfactual_tree_scores_consistent_prefix_paths():
    class FixedActions(torch.nn.Module):
        def forward_random_exploration(self, z, num_samples):
            values = torch.arange(
                num_samples, device=z.device, dtype=z.dtype).view(
                    1, num_samples, 1)
            return values.expand(z.shape[0], -1, z.shape[-1])

    class FirstStepScore(torch.nn.Module):
        def forward(self, trajectory, goal=None):
            return trajectory[:, 1, 0]

    search = CounterfactualTreeSearch(
        latent_dim=2,
        num_goals=0,
        num_exploration=2,
        rollout_steps=2,
        dt=1.0,
        selection_temperature=0.001,
    )
    search.perturbation = FixedActions()
    search.discriminator = FirstStepScore()
    result = search(torch.zeros(1, 2))
    # The low-energy first branch must remain the prefix at the first
    # trajectory step across all second-level suffixes.
    assert result["selected_trajectory"][0, 1, 0] < 0.1
 
 
def test_generic_forward_emits_all_physics_loss_inputs():
    model = BrainMoEPINN(
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        use_species_conditioning=True,
        species="c_elegans",
        species_vocab=["c_elegans", "human"],
        moe_num_shared=1,
        moe_num_routed=1,
        moe_top_k=1,
        generic_observation_only=True,
    ).train()
    output = model.forward_modalities(
        {"calcium": torch.randn(2, 4, 16)}, num_steps=2)
    assert {
        "grad_E",
        "grad_S",
        "L_z",
        "M_diag",
        "generic_constraint_residual",
        "grassmannian_loss",
    } <= set(output)
 
 
def test_trainer_accepts_non_neural_control_role(tmp_path):
    model = BrainMoEPINN(
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        species="c_elegans",
        species_vocab=["c_elegans"],
        perturbation_dim=1,
        moe_num_shared=1,
        moe_num_routed=1,
        moe_top_k=1,
        generic_observation_only=True,
    )
    trainer = BrainMoETrainer(
        model=model,
        config={
            "model_config": {"species": "c_elegans"},
            "experiment_data": {
                "modalities": ["calcium", "stimulus"],
                "roles": {"calcium": "signal", "stimulus": "control"},
            },
            "control_modalities": ("stimulus",),
            "use_mixer": False,
        },
        log_dir=tmp_path,
        checkpoint_dir=tmp_path,
    )
    assert trainer.signal_modalities == ("calcium",)
    assert trainer.control_modalities == ("stimulus",)

def test_sigreg_is_finite_for_batch_one():
    """Covariance regularization must be a connected zero for one sample."""
    from brain_moe_pinn.training.losses import WeakSIGRegLoss
    embedding = torch.randn(1, 8, requires_grad=True)
    loss = WeakSIGRegLoss()(embedding)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.isfinite(embedding.grad).all()


def test_correlation_loss_ignores_one_degenerate_channel_without_batch_poisoning():
    from brain_moe_pinn.training.losses import ReconstructionLoss
    pred = torch.tensor([[[1., 2., 3., 4.], [0., 0., 0., 0.]],
                         [[1., 2., 3., 4.], [1., 2., 3., 4.]]])
    target = pred.clone()
    mask = torch.ones_like(pred, dtype=torch.bool)
    rho = ReconstructionLoss._pearson_per_channel(pred, target, mask)
    assert rho[0, 0].item() == pytest.approx(1.0)
    assert rho[1, 0].item() == pytest.approx(1.0)
    assert rho[0, 1].item() == pytest.approx(0.0)


def test_loss_normalizer_rejects_nonfinite_terms():
    from brain_moe_pinn.training.losses import LossNormalizer
    normalizer = LossNormalizer()
    with pytest.raises(FloatingPointError, match="non-finite"):
        normalizer.normalize("probe", torch.tensor(float("nan")))


def test_dissipation_coupling_is_finite_at_zero_energy_gradient():
    from brain_moe_pinn.training.losses import DissipationLoss
    value, metrics = DissipationLoss()(torch.ones(2, 4), torch.ones(2, 4),
                                       torch.ones(2, 4), torch.zeros(2, 4))
    assert torch.isfinite(value)
    assert metrics["mobility_energy_coupling"] == 0.0


def test_generic_trainer_fetches_real_loader_batch(tmp_path):
    """A generic trainer step must consume loader signals, not dummy inputs."""
    root = tmp_path / "species"
    _write_array(root, "calcium", "sample", np.arange(32, dtype=np.float32).reshape(2, 16))
    model = BrainMoEPINN(
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        species="c_elegans",
        species_vocab=["c_elegans"],
        moe_num_shared=1,
        moe_num_routed=1,
        moe_top_k=1,
        generic_observation_only=True,
    )
    loader = torch.utils.data.DataLoader(
        SpeciesSignalDataset(root, ["calcium"], seq_len=8,
                             random_windows=False), batch_size=1,
        collate_fn=lambda batch: {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0] if isinstance(batch[0][key], torch.Tensor)
        })
    trainer = BrainMoETrainer(
        model=model,
        config={"model_config": {"species": "c_elegans"},
                "experiment_data": {"modalities": ["calcium"]},
                "use_mixer": False},
        log_dir=tmp_path / "logs", checkpoint_dir=tmp_path / "checkpoints",
        train_dataloader=loader)
    trainer._dataloader_iter = iter(loader)
    signals = trainer._next_generic_signals(1)
    assert set(signals) == {"calcium"}
    expected = torch.tensor(
        [[[0., 1., 2., 3., 4., 5., 6., 7.],
          [16., 17., 18., 19., 20., 21., 22., 23.]]])
    assert torch.equal(signals["calcium"].cpu(), expected)


def test_active_inference_controller_exposes_value_function():
    """The perception path must not depend on an unconstructed submodule."""
    torch.manual_seed(0)
    dim, batch = 12, 2
    controller = ActiveInferenceController(
        latent_dim=dim, use_counterfactual=True, use_feedback=True)
    assert hasattr(controller, "value_function")

    z = torch.randn(batch, dim)
    result = controller.forward_perception(
        z,
        torch.randn(batch, 9),
        torch.randn(batch, 9),
        action=torch.randn(batch, dim),
        kda_state_history=torch.randn(batch, 3, dim),
    )
    assert result["corrected_z"].shape == z.shape
    assert torch.isfinite(result["value"]).all()

    without_action = controller.forward_perception(
        z, torch.randn(batch, 9), torch.randn(batch, 9))
    assert without_action["corrected_z"].shape == z.shape


def test_controller_reuses_supplied_potentials_and_tree_search():
    """EFE and the dynamics must score one landscape, not two."""
    torch.manual_seed(0)
    dim = 12
    potentials = EnergyEntropyFields(hidden_dim=dim)
    search = CounterfactualTreeSearch(
        latent_dim=dim, num_goals=1, num_exploration=1, rollout_steps=2)
    controller = ActiveInferenceController(
        latent_dim=dim,
        use_counterfactual=True,
        use_feedback=True,
        energy_entropy_net=potentials,
        counterfactual_search=search,
    )
    assert controller.value_function.energy_entropy is potentials
    assert controller.cfts is search

    result = controller.forward_imagination(
        torch.randn(2, dim), goal_attractors=torch.randn(2, 1, dim))
    assert torch.isfinite(result["efe"]).all()


def test_precision_gate_accepts_any_action_shape():
    """Precision is driven by the action magnitude, whatever its layout."""
    gate = PrecisionGate(hidden_dim=8)
    delay = torch.full((3,), 0.5)
    for action in (torch.randn(3), torch.randn(3, 1), torch.randn(3, 16)):
        precision, _ = gate(delay, action)
        assert precision.shape == (3,)


def test_feedback_error_history_keeps_time_axis_bounded():
    """Error history is (B, T, d) and bounded, not an ever-widening latent."""
    torch.manual_seed(0)
    dim, batch = 6, 2
    feedback = ClosedLoopFeedback(latent_dim=dim)
    feedback.max_history = 3
    for _ in range(6):
        result = feedback(
            torch.randn(batch, dim),
            torch.randn(batch, 5),
            torch.randn(batch, 5),
            action=torch.randn(batch, dim),
            kda_state_history=torch.randn(batch, 2, dim),
        )
    assert result["error_history"].shape == (batch, 3, dim)


def test_ou_noise_states_scale_as_euler_maruyama():
    """State noise must grow like sqrt(dt), i.e. Var(dt * noise) = 2 D dt."""
    torch.manual_seed(0)
    dt, tau, diffusion = 0.25, 0.4, 0.7
    noise = OUStructuredNoise(dim=4, tau=tau, dt=dt, D=diffusion)

    probe = torch.zeros(4096, 4)
    state = None
    for _ in range(200):  # burn in to the stationary unit-variance state
        _, state = noise(probe, state)
    assert state.var().item() == pytest.approx(1.0, abs=0.05)

    increment = dt * noise(torch.zeros_like(probe), state)[0]
    assert increment.var().item() == pytest.approx(
        2.0 * diffusion * dt, rel=0.15)


def test_velocity_brain_mobility_is_learned_and_degenerate():
    """M(z) is state dependent, nonnegative, and annihilates grad_E."""
    torch.manual_seed(0)
    brain = VelocityBrain(
        hidden_dim=8, num_poisson_layers=1, num_energy_layers=1)
    z = torch.randn(2, 8)
    low = brain(z, salience=torch.full((2, 8), -4.0))
    high = brain(z, salience=torch.full((2, 8), 4.0))

    for output in (low, high):
        assert output["M_diag"].shape == z.shape
        assert (output["M_diag"] >= 0).all()
    assert not torch.allclose(low["M_diag"], high["M_diag"])
    assert low["degeneracy_M_grad_E_norm"].item() < 1e-5


def test_oettinger_projection_diagonal_mobility_annihilates_grad_E():
    """The operator-level projection is what makes M grad_E = 0 exact."""
    torch.manual_seed(0)
    grad_E = torch.randn(3, 10)
    grad_S = torch.randn(3, 10)
    mobility = torch.rand(3, 10) + 0.1

    assert oettinger_mobility_action(mobility, grad_E, grad_E).norm() < 1e-5
    applied = oettinger_mobility_action(mobility, grad_E, grad_S)
    assert (applied * grad_E).sum(dim=-1).abs().max().item() < 1e-5


def test_potential_gradients_are_exact_and_deterministic():
    """grad_E must be the true gradient of E, stable between evaluations."""
    torch.manual_seed(0)
    fields = EnergyEntropyFields(hidden_dim=8)  # train mode on purpose
    state = torch.randn(1, 8)
    grad_E, _ = fields.compute_gradients(state.clone().requires_grad_(True))
    repeated, _ = fields.compute_gradients(state.clone().requires_grad_(True))
    assert torch.allclose(grad_E, repeated)

    scale = abs(fields.energy_scale).item()
    eps = 1e-3
    for index in range(4):
        plus, minus = state.clone(), state.clone()
        plus[0, index] += eps
        minus[0, index] -= eps
        numeric = ((fields(plus)[0][0, 0] - fields(minus)[0][0, 0]) / (2 * eps))
        assert grad_E[0, index].item() * scale == pytest.approx(
            numeric.item(), abs=1e-3)


def test_rollout_control_enters_with_dt_weight():
    """Deterministic actions carry a dt weight, not sqrt(dt)."""
    class ZeroVelocity(torch.nn.Module):
        integration_dt = 0.5

        def forward(self, z):
            return {"delta_z": torch.zeros_like(z)}

    actions = torch.full((1, 1, 2), 3.0)

    # impulse (default): the intervention lands once, at the branch point.
    impulse = TrajectoryRollout(latent_dim=2, num_steps=2, dt=1.0)
    impulse.set_velocity_net(ZeroVelocity())
    traj, _ = impulse(torch.zeros(1, 2), actions)
    assert torch.allclose(traj[0, 0, 1], torch.full((2,), 1.5), atol=1e-5)
    assert torch.allclose(traj[0, 0, -1], torch.full((2,), 1.5), atol=1e-5)

    # sustained: the action is re-applied every step.
    sustained = TrajectoryRollout(
        latent_dim=2, num_steps=2, dt=1.0, action_mode="sustained")
    sustained.set_velocity_net(ZeroVelocity())
    traj2, _ = sustained(torch.zeros(1, 2), actions)
    assert torch.allclose(traj2[0, 0, -1], torch.full((2,), 3.0), atol=1e-5)


def test_efe_charges_control_cost():
    """compute_efe must account for the action it is told about."""
    controller = ActiveInferenceController(
        latent_dim=8, use_counterfactual=False, use_feedback=False)
    z = torch.randn(2, 8)
    goals = torch.randn(2, 1, 8)

    without, _ = controller.value_function.compute_efe(
        z, goal_attractors=goals)
    action = torch.full((2, 4), 2.0)
    with_action, metrics = controller.value_function.compute_efe(
        z, goal_attractors=goals, action=action, action_cost=0.5)

    expected = 0.5 * action.pow(2).sum(dim=-1)
    assert torch.allclose(with_action - without, expected, atol=1e-5)
    assert "control_cost" in metrics


def test_reconstruction_head_generates_from_latent_only():
    """The decoder must not be able to read the current window."""
    torch.manual_seed(0)
    head = SignalReconstructionHead(latent_dim=8, num_time_basis=16)
    first = torch.randn(2, 8)
    second = torch.randn(2, 8)

    out_a = head(first, num_channels=3, target_time_len=32)
    out_b = head(second, num_channels=3, target_time_len=32)
    assert out_a.shape == (2, 3, 32)
    assert not torch.allclose(out_a, out_b)      # latent drives the waveform
    assert not torch.allclose(out_a[0, 0], out_a[0, 1])   # channels differ

    # The old leaky calling convention (channel tokens in) no longer exists.
    with pytest.raises(ValueError, match="latent must have shape"):
        head(torch.randn(2, 3, 4, 8), num_channels=3, target_time_len=32)
    with pytest.raises(ValueError, match="capacity"):
        head(first, num_channels=10_000, target_time_len=32)


def _colored_noise(length, power, seed=0):
    torch.manual_seed(seed)
    freqs = torch.arange(1, length // 2, dtype=torch.float32)
    phases = freqs * 0.37
    spectrum = torch.zeros(length // 2 + 1, dtype=torch.complex64)
    spectrum[1:length // 2] = freqs.pow(power / 2) * torch.exp(1j * phases)
    return torch.fft.irfft(spectrum, n=length)


def test_aperiodic_exponent_recovers_slope_and_survives_peaks():
    """Signed log-log exponent, robust to an oscillatory peak."""
    for power in (-1.0, -2.0, 1.0):
        estimate, valid = aperiodic_exponent(
            _colored_noise(256, power).view(1, 1, -1))
        assert valid
        assert estimate.item() == pytest.approx(power, abs=0.05)

    # A strong alpha peak must not drag the estimated exponent flat.
    clean = _colored_noise(256, -1.0)
    t = torch.arange(256, dtype=torch.float32)
    peaky = clean + 8.0 * torch.sin(2 * torch.pi * 10.0 * t / 256.0)
    robust, _ = aperiodic_exponent(peaky.view(1, 1, -1))
    assert robust.item() == pytest.approx(-1.0, abs=0.15)


def test_spectral_loss_reference_comes_from_target():
    """The objective tracks the data's exponent, not a fixed pink target."""
    recon = _colored_noise(256, -1.0).view(1, 1, -1).repeat(1, 2, 1)
    target = _colored_noise(256, -2.0).view(1, 1, -1).repeat(1, 2, 1)

    loss = SpectralSlopeLoss(weight=1.0)
    fixed_value, _ = loss(recon)
    data_value, metrics = loss(recon, target)

    assert metrics["psd_slope_reference_source"] == "target"
    assert metrics["psd_slope_reference"] == pytest.approx(-2.0, abs=0.1)
    assert metrics["psd_slope"] == pytest.approx(-1.0, abs=0.1)
    # Matching the wrong (pink) reference is penalised; matching the data is not.
    assert fixed_value.item() == pytest.approx(0.0, abs=0.05)
    assert data_value.item() == pytest.approx(1.0, abs=0.1)


def test_avalanche_exponent_is_size_distribution_mle():
    """tau must come from P(size), not from a size-vs-duration regression."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Synthetic critical-like activity: many independent active units.
    steps, units = 6000, 32

    def burst_process(p_burst):
        activity = np.zeros((steps, units), dtype=np.float32)
        for t in range(steps):
            if rng.random() < p_burst:
                n_active = int(rng.integers(1, units))
                idx = rng.choice(units, size=n_active, replace=False)
                activity[t, idx] = rng.uniform(1.0, 3.0, n_active)
        return torch.from_numpy(activity)

    result = compute_avalanche_stats(burst_process(0.02))
    assert "tau" in result and "num_avalanches" in result
    assert result["size_exponent_definition"].startswith("P(size)")
    if not result["insufficient_data"]:
        assert np.isfinite(result["tau"])

    tiny = compute_avalanche_stats(burst_process(0.0001))
    assert tiny["insufficient_data"] is True
    assert np.isnan(tiny["tau"])


def test_intrinsic_rollout_stats_distinguish_regimes():
    """Free-run autonomy must separate contraction, stationarity, expansion."""
    base = torch.randn(2, 12, 6)
    expand = base * torch.linspace(1.0, 3.0, 12).view(1, 12, 1)
    contract = base * torch.linspace(3.0, 1.0, 12).view(1, 12, 1)
    stationary = base.clone()

    assert intrinsic_rollout_stats(expand.numpy())["tail_std_ratio"] > 1.5
    assert intrinsic_rollout_stats(contract.numpy())["tail_std_ratio"] < 0.7
    assert intrinsic_rollout_stats(stationary.numpy())["tail_std_ratio"] == (
        pytest.approx(1.0, abs=0.15))
    with pytest.raises(ValueError, match=r"\(B, K, D\)"):
        intrinsic_rollout_stats(torch.randn(2, 2) .numpy())


def test_structured_moe_perturbs_operators_not_velocity():
    """Experts must perturb L and M, and stay parameter-affordable."""
    torch.manual_seed(0)
    dim, shared, routed = 32, 2, 2
    structured = MoEGenericVelocityField(
        hidden_dim=dim, num_shared=shared, num_routed=routed, top_k=1, rank=4)
    plain = MoEVelocityField(
        hidden_dim=dim, num_shared=shared, num_routed=routed, top_k=1)

    structured_params = sum(p.numel() for p in structured.parameters())
    plain_params = sum(p.numel() for p in plain.parameters())
    assert structured_params / plain_params < 8.0

    z = torch.randn(2, dim, requires_grad=True)
    grad_E = torch.randn(2, dim)
    grad_S = torch.randn(2, dim)
    out = structured(z, grad_E=grad_E, grad_S=grad_S)
    assert out["delta_L"].shape == (2, dim, dim)

    # L is antisymmetric and its entropy-gradient residual is removed.
    assert torch.allclose(out["delta_L"], -out["delta_L"].transpose(-2, -1),
                          atol=1e-4)
    entropy_residual = torch.bmm(
        out["delta_L"], grad_S.unsqueeze(-1)).squeeze(-1)
    assert entropy_residual.norm().item() < 1e-3
    # Mobility is nonnegative by construction.
    assert (out["delta_M_diag"] >= 0).all()


def test_dissipation_loss_does_not_crush_mobility():
    """M must not appear in the objective; degeneracy is enforced by projection."""
    torch.manual_seed(0)
    mobility = torch.rand(2, 8, requires_grad=True) + 0.2
    grad_E = torch.randn(2, 8)
    grad_S = torch.randn(2, 8, requires_grad=True)
    delta_z = torch.randn(2, 8, requires_grad=True).abs()

    loss, metrics = DissipationLoss(weight=1.0)(delta_z, grad_S, mobility, grad_E)
    loss.backward()

    # Penalising the raw diagonal (M * grad_E) would drive M -> 0 everywhere
    # grad_E is nonzero, deleting the dissipative term.  M is now a metric only.
    assert mobility.grad is None
    assert "mobility_energy_coupling" in metrics
    assert "degeneracy_violation" not in metrics


def test_structured_experts_start_generic():
    """The unstructured expert bias is zero at init, so step 0 is GENERIC."""
    torch.manual_seed(0)
    field = MoEGenericVelocityField(
        hidden_dim=16, num_shared=2, num_routed=2, top_k=1, rank=4)
    out = field(torch.randn(2, 16),
                grad_E=torch.randn(2, 16), grad_S=torch.randn(2, 16))
    assert out["velocity_bias"].norm().item() == 0.0
    assert torch.allclose(
        out["delta_L"], -out["delta_L"].transpose(-2, -1), atol=1e-4)


def test_grassmannian_reuses_expert_outputs_consistently():
    """Cached shared-expert outputs must equal a fresh recomputation."""
    torch.manual_seed(0)
    field = MoEVelocityField(
        hidden_dim=16, num_shared=2, num_routed=2, top_k=1, dropout=0.0).train()
    z = torch.randn(2, 16)
    result = field(z)

    outputs = torch.stack([expert(z) for expert in field.all_experts], dim=1)
    normed = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-8)
    gram = normed @ normed.transpose(-2, -1)
    eye = torch.eye(field.num_experts).unsqueeze(0)
    manual = ((1 - eye) * gram).pow(2).sum() / (
        field.num_experts * (field.num_experts - 1))
    assert torch.allclose(result["grassmannian_loss"], manual, atol=1e-6)


def test_sparsity_loss_penalises_concentration():
    """The penalty must rise with peakedness, not reward it."""
    loss = LatentSparsityLoss(weight=1.0)
    peaked = torch.zeros(1, 8)
    peaked[0, 0] = 10.0
    uniform = torch.zeros(1, 8)

    peaked_value, peaked_metrics = loss(peaked)
    uniform_value, uniform_metrics = loss(uniform)
    assert peaked_value.item() > uniform_value.item()
    assert (peaked_metrics["latent_concentration"]
            > uniform_metrics["latent_concentration"])


def test_poisson_operator_is_implicit_and_exact():
    """Action form must match the dense matrix form bit-for-bit."""
    torch.manual_seed(0)
    brain = VelocityBrain(hidden_dim=32, num_poisson_layers=1,
                          num_energy_layers=1)
    z = torch.randn(2, 32)

    brain.materialize_poisson = True
    brain.mt_kda.reset_state(batch_size=2)
    dense = brain(z)["delta_z"]
    brain.materialize_poisson = False
    brain.mt_kda.reset_state(batch_size=2)
    implicit = brain(z)["delta_z"]

    assert torch.allclose(dense, implicit)
    assert brain(z)["L_z"] is None          # no dense matrix is returned

    # The operator's action agrees with explicitly assembling L.
    vec = torch.randn(2, 32)
    op = brain.poisson_op
    assembled = torch.bmm(op.forward(z), vec.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(op.apply_action(z, vec), assembled, atol=1e-5)


def test_poisson_operator_is_deterministic_in_train_mode():
    """L(z) must be a function of z, not a fresh dropout sample."""
    torch.manual_seed(0)
    brain = VelocityBrain(hidden_dim=32, num_poisson_layers=1,
                          num_energy_layers=1).train()
    assert not any(isinstance(m, torch.nn.Dropout)
                   for m in brain.poisson_op.modules())
    z = torch.randn(2, 32)
    brain.mt_kda.reset_state(batch_size=2)
    first = brain(z)["delta_z"]
    brain.mt_kda.reset_state(batch_size=2)
    second = brain(z)["delta_z"]
    assert torch.allclose(first, second)


def test_smith_predictor_gain_mode_is_not_sticky():
    """The Wiener interlock must not change later forwards."""
    torch.manual_seed(0)
    feedback = ClosedLoopFeedback(latent_dim=8)
    z = torch.randn(2, 8)
    history = torch.randn(2, 4, 8)

    def run(**kwargs):
        torch.manual_seed(3)
        return feedback(z, torch.randn(2, 5), torch.randn(2, 5),
                        action=torch.randn(2, 8),
                        kda_state_history=history, **kwargs)

    baseline = run(q_factor=1.0)
    interlocked = run(q_factor=1.0, ataxia_score=0.9)
    repeated = run(q_factor=1.0)

    assert interlocked["metrics"]["wiener_mode"] == "ataxia_interlock"
    assert interlocked["metrics"]["oscillation_tolerant"] is False
    assert torch.allclose(baseline["corrected_z"], repeated["corrected_z"])
    assert feedback.smith_predictor.oscillation_tolerant is True
