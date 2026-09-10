"""Tests for the free-run metric suite (diagnostics/free_run_metrics.py).

The suite is numpy-only offline bookkeeping; these tests are hermetic and
run without torch.  File-path loading follows the repo convention (module
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


frm = _load("frm_mod", "diagnostics/free_run_metrics.py")

rng = np.random.default_rng(0)


def _ar1(n, rho, scale=1.0, n_comp=3):
    x = np.zeros((n, n_comp))
    for c in range(n_comp):
        for t in range(1, n):
            x[t, c] = rho * x[t - 1, c] + scale * rng.standard_normal()
    return x


def test_corr_matrix_mse_identical_zero():
    x = _ar1(500, 0.9)
    assert frm.corr_matrix_mse(x, x) < 1e-10


def test_corr_matrix_mse_white_vs_ar_different():
    x = _ar1(500, 0.9)
    w = rng.standard_normal(x.shape)
    # white noise across columns ~ uncorrelated; AR(0.9) shared slow drift
    # across components only via common noise? build shared-signal case:
    common = _ar1(500, 0.95, n_comp=1)[:, 0]
    x2 = np.stack([common + rng.standard_normal(500),
                   common + rng.standard_normal(500),
                   common + rng.standard_normal(500)], axis=1)
    w2 = rng.standard_normal(x2.shape)
    m_ar = frm.corr_matrix_mse(x2, x2[:, ::-1])  # permuted same cols
    m_wn = frm.corr_matrix_mse(x2, w2)
    assert m_ar < 0.05 and m_wn > 0.1


def test_variance_ratio():
    x = _ar1(500, 0.8, scale=2.0)
    doubled = 2.0 * x
    out = frm.variance_ratio(x, doubled)
    assert out["mean"] == pytest.approx(4.0, rel=0.05)
    assert out["max"] == pytest.approx(4.0, rel=0.05)


def test_variance_ratio_mismatch_raises():
    with pytest.raises(ValueError):
        frm.variance_ratio(np.zeros((100, 2)), np.zeros((100, 3)))


def test_autocorr_white_vs_persistent():
    x = _ar1(800, 0.9)
    w = rng.standard_normal(x.shape)
    # calibrate white to same variance so a variance-only check would pass
    w = w * (x.std(axis=0) / w.std(axis=0))[None, :]
    out = frm.autocorr_profile_mse(x, w, max_lag=100)
    assert out["mse"] > 0.03
    same = frm.autocorr_profile_mse(x, x, max_lag=100)
    assert same["mse"] < 1e-10
    # timescale marker: persistent AR zero-crossing much later than white's
    assert out["lag_of_first_zero_real"] > 5
    assert out["lag_of_first_zero_gen"] <= 2


def test_state_path_stats_and_chatter():
    # path with long dwells: 50 x state0, 50 x state1, 50 x state0
    stable = np.concatenate([np.full(50, 0), np.full(50, 1), np.full(50, 0)])
    st = frm.state_path_stats(stable)
    assert st["n_switches"] == 2
    assert st["mean_dwell"] == pytest.approx(50.0)
    assert st["chattering_fraction"] == 0.0
    # chattering path: alternating every frame
    chat = np.tile([0, 1], 75)
    ct = frm.state_path_stats(chat)
    assert ct["chattering_fraction"] == 1.0
    assert ct["switch_rate"] > 0.9
    assert frm.chattering_index(stable) == 0.0
    assert frm.chattering_index(chat) == 1.0


def test_suite_with_states_and_dt():
    real = _ar1(400, 0.85)
    gen = _ar1(400, 0.85)
    s_real = np.concatenate([np.full(100, i % 2) for i in range(4)])
    s_gen = s_real.copy()
    out = frm.run_free_run_suite(real, gen, s_real, s_gen, dt=0.25)
    assert set(out) >= {"corr_matrix_mse", "variance_ratio", "autocorr",
                        "state_real", "state_gen", "dwell_mean_ratio",
                        "switch_rate_ratio", "chatter_diff",
                        "mean_dwell_s_real", "mean_dwell_s_gen"}
    assert out["corr_matrix_mse"] < 0.05
    assert out["chatter_diff"] == 0.0
    # dt conversion: 100 frames at 4 fps = 25 s
    assert out["mean_dwell_s_real"] == pytest.approx(25.0)


def test_short_input_raises():
    with pytest.raises(ValueError):
        frm.corr_matrix_mse(np.zeros((2, 4)), np.zeros((2, 4)))
