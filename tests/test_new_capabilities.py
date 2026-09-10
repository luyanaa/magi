"""Smoke tests for the 2026-09 scalability additions.

Run:  python -m pytest tests/test_new_capabilities.py -x
Modules are loaded by file path so the (heavy) package __init__ chains are
not executed.  Torch-dependent tests skip automatically when torch is
unavailable.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _install_package_shims():
    """Register namespace-only parents so file-loaded modules can use relative
    imports.

    Modules are loaded by file path here to skip the heavy package __init__
    chain, which leaves `from ..runtime.device_utils import x` with no parent
    package to resolve against.  These shims provide the parent namespaces
    without executing the real __init__ files.
    """
    import types
    for dotted, path in (
        ("brain_moe_pinn", ROOT),
        ("brain_moe_pinn.runtime", ROOT / "runtime"),
        ("brain_moe_pinn.core", ROOT / "core"),
    ):
        if dotted in sys.modules:
            continue
        shim = types.ModuleType(dotted)
        shim.__path__ = [str(path)]
        sys.modules[dotted] = shim


def _load(name, rel):
    _install_package_shims()
    # Load under the real dotted name so module-level relative imports resolve.
    dotted = "brain_moe_pinn." + rel[:-3].replace("/", ".")
    spec = importlib.util.spec_from_file_location(dotted, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


subject_adaptation = _load("sa_mod", "core/subject_adaptation.py")
latent_diffusion = _load("ld_mod", "core/latent_diffusion.py")
hierarchical_gating = _load("hg_mod", "core/hierarchical_gating.py")
scaling_probe = _load("sp_mod", "diagnostics/scaling_probe.py")
statistics = _load("statistics_mod", "physics/statistics.py")
velocity_brain = _load("velocity_brain_mod", "core/velocity_brain.py")
stochastic_process = _load("stochastic_process_mod", "physics/stochastic_process.py")
causal_dynamics = _load("causal_dynamics_mod", "diagnostics/causal_dynamics.py")

torch = pytest.importorskip("torch", reason="torch not installed")

LowRankDynamicsAdapter = subject_adaptation.LowRankDynamicsAdapter
CalibrationEncoder = subject_adaptation.CalibrationEncoder
evaluate_subject_adaptation = subject_adaptation.evaluate_subject_adaptation
LatentDiffusion = latent_diffusion.LatentDiffusion
make_windows = latent_diffusion.make_windows
ddpm_posterior_mean = latent_diffusion.ddpm_posterior_mean
GenerickeDegeneracyProjection = velocity_brain.GenerickeDegeneracyProjection
SlowGateTransition = hierarchical_gating.SlowGateTransition
validate_states = hierarchical_gating.validate_states
information_horizon = scaling_probe.information_horizon
spectral_floor = scaling_probe.spectral_floor
variance_maintenance = scaling_probe.variance_maintenance


def _synth_subjects(n=3, D=8, T=400, seed=0):
    """Subjects with shared linear dynamics + subject-specific rotation."""
    rng = np.random.default_rng(seed)
    A = np.eye(D) * 0.98
    subs = []
    for s in range(n):
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        z = np.zeros((T, D))
        noise = 0.05
        for t in range(1, T):
            z[t] = Q @ (A @ (Q.T @ z[t - 1])) + noise * rng.standard_normal(D)
        subs.append((f"sub{s}", z))
    return subs


def test_adapter_noop_at_zero():
    D, r = 8, 4
    ad = LowRankDynamicsAdapter(D, r)
    z = torch.randn(5, D)
    dz = torch.randn(5, D)
    out = ad.adapt(dz, z)
    assert torch.allclose(out, dz, atol=1e-6)


def _synth_lowrank_subjects(D=8, T=300, rho=0.99, w_list=(0.02, 0.03, 0.05),
                            noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    A0 = rho * np.eye(D)
    u = rng.standard_normal(D)
    v = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    subs = []
    for s, w in enumerate(w_list):
        A = A0 + w * np.outer(u, v)
        z = np.zeros((T, D))
        z[0] = rng.standard_normal(D)
        for t in range(1, T):
            z[t] = A @ z[t - 1] + noise * rng.standard_normal(D)
        subs.append((f"sub{s}", z))

    def field(z):
        M = torch.as_tensor(A0 - np.eye(D), dtype=z.dtype)
        return z @ M.t()

    return subs, field, u, v


def test_test_time_adapt_improves():
    # (1) noiseless slow-decay system: exact low-rank route recovers the
    # perturbation almost completely (no noise floor, energy persists)
    subs, field, u, v = _synth_lowrank_subjects()
    ad = LowRankDynamicsAdapter(8, rank=1)
    with torch.no_grad():
        ad.U.copy_(torch.tensor(u)[:, None])
        ad.V.copy_(torch.tensor(v)[:, None])
    res1, _ = evaluate_subject_adaptation(field, subs, adapter=ad,
                                          steps=40, lr=1e-1, calib_len=150)
    assert all(r["adapted_mse"] < 0.2 * r["base_mse"] for r in res1), res1

    # (2) noisy system: a mismatched random rank-8 adapter must degrade
    # gracefully (no strong damage), not necessarily improve
    subs2, field2, _, _ = _synth_lowrank_subjects(noise=0.01, w_list=(0.1, 0.2, 0.3),
                                                  rho=0.8, T=600)
    ad2 = LowRankDynamicsAdapter(8, rank=8)
    res2, imp2 = evaluate_subject_adaptation(field2, subs2, adapter=ad2,
                                             steps=20, lr=1e-1, calib_len=200)
    mean_base2 = float(np.mean([r["base_mse"] for r in res2]))
    assert imp2 > -0.1 * mean_base2, (imp2, mean_base2)


def test_calibration_encoder_shapes():
    enc = CalibrationEncoder(8, rank=4)
    lam = enc(torch.randn(3, 64, 8))
    assert lam.shape == (3, 4)


def test_diffusion_fit_and_sample_shapes():
    D, Lc, Lt = 6, 10, 24
    rng = np.random.default_rng(0)
    z = np.cumsum(rng.standard_normal((300, D)) * 0.05, axis=0)
    win = make_windows(z, Lc, Lt, stride=4)
    diff = LatentDiffusion(dim=D, context=Lc, target=Lt, t_steps=20,
                           hidden=16)
    diff.fit(torch.tensor(win, dtype=torch.float32), epochs=2, batch=16,
             progress=False)
    out = diff.sample(z[:Lc], length=60, n_roll=2, progress=False)
    assert out.shape[1] == D and out.shape[0] >= 60


def test_gate_forward_and_validate():
    gate = SlowGateTransition(dim=8, k_states=3, timescales=(0.5,))
    z = torch.randn(1, 8)
    gate.reset(z[0])
    logits = gate(z, z)
    assert logits.shape == (1, 3)
    states = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0, 0])
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0])
    rep = validate_states(states, labels)
    assert 0.0 <= rep["matched_accuracy"] <= 1.0


def test_scaling_probe_numpy():
    rng = np.random.default_rng(0)
    t = np.arange(2000)
    slow = np.sin(2 * np.pi * 0.01 * t)[:, None]
    white = rng.standard_normal((2000, 1))
    h_slow = information_horizon(slow, fs=1.0)
    h_white = information_horizon(white, fs=1.0)
    assert h_slow > h_white
    assert 0.0 <= spectral_floor(slow) <= 1.0
    vm = variance_maintenance(white, white * 2)
    assert abs(vm["std_ratio"] - 2.0) < 0.2

estimate_local_transition_growth = statistics.estimate_local_transition_growth


def test_transition_growth_uses_dt_aware_step_map():
    x = torch.zeros(2, 4)
    stable = estimate_local_transition_growth(
        lambda z: -0.2 * z, x, dt=1.0, iters=8)
    unstable = estimate_local_transition_growth(
        lambda z: 0.2 * z, x, dt=1.0, iters=8)
    assert abs(stable["transition_growth_mean"] - 0.8) < 1e-5
    assert stable["locally_bounded"]
    assert abs(unstable["transition_growth_mean"] - 1.2) < 1e-5
    assert not unstable["locally_bounded"]


def test_gate_explicit_batch_state_and_stickiness_calibration():
    gate = SlowGateTransition(
        dim=4, k_states=3, timescales=(2.0,), stickiness=5.0, dt=0.1)
    z = torch.randn(4, 4)
    states = torch.tensor([0, 1, 2, 0])
    logits, next_state = gate.forward_with_state(z, states, None)
    assert logits.shape == (4, 3)
    assert next_state.shape == (4, 1, 4)

    gate.stickiness = 0.0
    baseline, _ = gate.forward_with_state(z, states, next_state)
    assert torch.allclose(
        logits - baseline,
        5.0 * torch.nn.functional.one_hot(states, 3).to(logits.dtype),
        atol=1e-5,
    )

    report = gate.calibrate_stickiness(
        z, states, candidates=torch.tensor([0.0, 2.0, 5.0]))
    assert report["calibrated_nll"] <= report["baseline_nll"]


def test_trajectory_diagnostics_do_not_claim_epr():
    thermo = _load("thermo_mod", "physics/thermodynamics.py")
    z = torch.zeros(5, 2)
    v = torch.ones(5, 2)
    report = thermo.compute_trajectory_diagnostics(
        z, v, div_drift_fn=lambda x: torch.arange(x.shape[0], dtype=x.dtype))
    assert "epr_exact_mean" not in report
    assert report["drift_divergence_mean"] == 2.0


def test_energy_change_diagnostic_is_not_jarzynski_estimator():
    thermo = _load("thermo_mod_energy", "physics/thermodynamics.py")
    z = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    report = thermo.compute_energy_change_diagnostic(
        z, lambda x: x.sum(dim=-1))
    assert "jarzynski_left" not in report
    assert report["energy_change_mean"] == 4.0
    monitor = thermo.EnergyChangeMonitor(window_size=2)
    monitor.update(z[:1], z[1:2])
    monitor_report = monitor.update(z[1:2], z[2:3])
    assert "i_bergson" not in monitor_report
    assert "energy_change_proxy" in monitor_report

def test_ddpm_posterior_mean_uses_sqrt_previous_cumulative_alpha():
    x_t = torch.ones(1, 2, 3)
    x0 = torch.zeros_like(x_t)
    got = ddpm_posterior_mean(
        x_t, x0,
        torch.tensor(0.8), torch.tensor(0.6),
        torch.tensor(0.75), torch.tensor(0.2))
    expected = torch.sqrt(torch.tensor(0.8)) * 0.25 / 0.4
    assert torch.allclose(got, torch.full_like(got, expected))


def test_degeneracy_projection_preserves_antisymmetry():
    torch.manual_seed(0)
    raw = torch.randn(2, 5, 5)
    L = raw - raw.transpose(-1, -2)
    grad_s = torch.randn(2, 5)
    projection = GenerickeDegeneracyProjection(5)
    projected = projection.project_L_orthogonal_to_gradS(L, grad_s)
    antisymmetry_error = (projected + projected.transpose(-1, -2)).abs().max()
    null_error = torch.bmm(projected, grad_s.unsqueeze(-1)).abs().max()
    assert antisymmetry_error < 1e-5
    assert null_error < 1e-4


def test_velocity_brain_perturbation_conditioning():
    torch.manual_seed(0)
    model = velocity_brain.VelocityBrain(
        hidden_dim=8,
        num_poisson_layers=1,
        num_energy_layers=1,
        dropout=0.0,
        use_lowrank_poisson=True,
        poisson_rank=4,
        perturbation_dim=2,
    ).eval()
    z = torch.randn(2, 8)
    zero_control = torch.zeros(2, 2)
    active_control = torch.ones(2, 2)
    model.step_counter = 0
    baseline = model(z, perturbation=zero_control, apply_noise=False)
    model.step_counter = 0
    intervened = model(z, perturbation=active_control, apply_noise=False)
    assert intervened["control_term"].shape == z.shape
    assert torch.isfinite(intervened["control_term"]).all()
    assert not torch.allclose(baseline["delta_z"], intervened["delta_z"])

    sde = stochastic_process.ControlledSDE(
        drift_fn=lambda state, control: torch.zeros_like(state),
        diffusion_fn=lambda state, control: torch.ones_like(state) * 0.5,
        contract=stochastic_process.ControlledSDEContract(
            state_dim=2, control_dim=1, dt=0.1),
    )
    state = torch.zeros(2, 2)
    control = torch.ones(2, 1)
    next_state = sde.step(state, control, noise=torch.zeros_like(state))
    assert torch.equal(next_state, state)

    states = torch.zeros(3, 2, 2)
    drifts = torch.zeros(2, 2, 2)
    diffusions = torch.ones(2, 2, 2) * 0.5
    log_likelihood = stochastic_process.path_log_likelihood(
        states, drifts, diffusions, dt=0.1)
    assert log_likelihood.shape == (2,)
    assert torch.isfinite(log_likelihood).all()

    conditional = stochastic_process.estimate_path_entropy_production(
        states, drifts, diffusions, drifts, diffusions, dt=0.1)
    assert conditional["boundary_terms_supplied"] is False
    assert "total_path_entropy_production" not in conditional

    with_boundaries = stochastic_process.estimate_path_entropy_production(
        states, drifts, diffusions, drifts, diffusions, dt=0.1,
        initial_log_prob=torch.zeros(2),
        reverse_initial_log_prob=torch.zeros(2),
    )
    assert "total_path_entropy_production" in with_boundaries


def test_controlled_response_and_temporal_asymmetry():
    initial = torch.zeros(2, 1)
    controls = torch.ones(3, 2, 1)
    treated = causal_dynamics.rollout_controlled(
        lambda state, control: state + control, initial, controls)
    baseline = causal_dynamics.rollout_controlled(
        lambda state, control: state + control, initial, torch.zeros_like(controls))
    response = causal_dynamics.intervention_response_metrics(
        treated, baseline, dt=0.5, intervention_start=1)
    assert response["final_effect"] > 0
    assert response["cumulative_effect"] > 0

    trajectory = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
    asymmetry = causal_dynamics.forward_reverse_prediction_gap(
        trajectory, lambda state: state + 1.0)
    assert asymmetry["forward_mse"] == 0
    assert asymmetry["reverse_mse"] > 0
    assert asymmetry["forward_reverse_gap"] > 0
