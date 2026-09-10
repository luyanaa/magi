"""Free-run metric suite for generative models of neural dynamics.

Design source: the C. elegans corpus (master thesis Ch. 2-5, rSLDS seminar,
TDE-RICA bake-off).  Its central lesson is that no single metric decides a
generator: correlation-structure fidelity, temporal realism, and state
dynamics are separate axes, and models trade them off (TSMixer wins
correlation MSE 0.031 but loses temporal realism; the GMM family is the
reverse).  Generative-model checks must therefore be reported as a suite,
never as one number.

This module implements the corpus-validated, species-agnostic axes:

  * correlation-structure fidelity  -> corr_matrix_mse()
  * marginal scale fidelity         -> variance_ratio()
  * temporal correlation profile    -> autocorr_profile_mse()
  * discrete-state dynamics         -> state_path_stats() / chattering_index()

Deliberately NOT here: spectral/1-f slope targets (human-EEG-specific per
2026-09 decision; physics.statistics.compute_1f_exponent is the EEG-only
spot), and predictive R2 on time splits (the corpus shows time-split
prediction fails for principled drift reasons - it is not a generator
quality metric).

Offline bookkeeping only (numpy).  All functions accept (T, N) arrays
(time x components/neurons); the axis along the "signal" is always time.
"""

import numpy as np
from typing import Dict, List, Optional, Sequence


def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be (T, N); got shape {a.shape}")
    if a.shape[0] < 3:
        raise ValueError(f"{name} needs >= 3 time steps; got {a.shape[0]}")
    return a


def _corr_matrix(a: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlation with NaN (constant column) -> 0.0."""
    a = a - a.mean(axis=0)
    sd = a.std(axis=0)
    sd[sd < 1e-12] = 1.0
    a = a / sd[None, :]
    c = a.T @ a / a.shape[0]
    if not np.isfinite(c).all():
        c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=-1.0)
    return c


def corr_matrix_mse(real: np.ndarray, generated: np.ndarray) -> float:
    """Mean squared error of the off-diagonal pairwise correlation matrices.

    Correlation structure is the corpus's headline free-run metric (thesis:
    TSMixer 0.031 vs gKDR-GMM 0.041 on this quantity).  The diagonal is
    excluded because it is identically 1 for both and would dilute the
    signal.
    """
    r = _as_2d(real, "real")
    g = _as_2d(generated, "generated")
    cr, cg = _corr_matrix(r), _corr_matrix(g)
    iu = np.triu_indices(cr.shape[0], k=1)
    return float(np.mean((cr[iu] - cg[iu]) ** 2))


def variance_ratio(real: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    """Per-component generated/real variance ratio (1.0 = right scale).

    Corpus anchor: real TDE-RICA occurrence components span 2.7-4.4x in std
    across components, so the mean alone hides scale mismatch; report the
    spread as well.
    """
    r = _as_2d(real, "real")
    g = _as_2d(generated, "generated")
    if r.shape[1] != g.shape[1]:
        raise ValueError(f"component mismatch: {r.shape[1]} vs {g.shape[1]}")
    vr = g.var(axis=0) / np.maximum(r.var(axis=0), 1e-12)
    return {
        "mean": float(np.mean(vr)),
        "median": float(np.median(vr)),
        "min": float(np.min(vr)),
        "max": float(np.max(vr)),
        "n_components": int(r.shape[1]),
        # log-space mean error: 0 = perfect, penalises over/under equally
        "log_rmse": float(np.sqrt(np.mean(np.log(vr) ** 2))),
    }


def autocorr_profile_mse(real: np.ndarray, generated: np.ndarray,
                         max_lag: Optional[int] = None) -> Dict[str, float]:
    """MSE between normalised autocorrelation profiles (per component,
    averaged).

    Catches 'wrong colour but right variance': e.g. white noise and an
    OU process can share variance yet have very different lag structure.
    max_lag defaults to min(T/3, 200).  Returned 'mse' is the mean over
    components and lags of (acf_real - acf_gen)^2.
    """
    r = _as_2d(real, "real")
    g = _as_2d(generated, "generated")
    if max_lag is None:
        max_lag = min(r.shape[0] // 3, 200)
    max_lag = max(1, int(max_lag))

    def _profile(a: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=0)
        sd = a.std(axis=0)
        sd[sd < 1e-12] = 1.0
        a = a / sd[None, :]
        T = a.shape[0]
        out = np.empty((max_lag + 1, a.shape[1]))
        for lag in range(max_lag + 1):
            out[lag] = np.mean(a[: T - lag] * a[lag:], axis=0)
        return out

    pr, pg = _profile(r), _profile(g)
    return {
        "mse": float(np.mean((pr - pg) ** 2)),
        "max_lag": max_lag,
        "lag_of_first_zero_real": float(_first_zero(pr)),
        "lag_of_first_zero_gen": float(_first_zero(pg)),
    }


def _first_zero(profile: np.ndarray) -> float:
    """Return first zero crossing, or the last measured lag if absent."""
    mean_prof = profile.mean(axis=1)
    for lag in range(1, len(mean_prof)):
        if mean_prof[lag] <= 0.0:
            return float(lag)
    return float(len(mean_prof) - 1)


def state_path_stats(path: np.ndarray,
                     min_dwell: int = 2) -> Dict[str, float]:
    """Dwell/switch statistics of a discrete state path (frames).

    Corpus anchor: chattering (rapid unrealistic state switching) was a
    central rSLDS failure mode, cured with stickiness kappa=200; dwell
    statistics are therefore a first-class generator check.
    """
    s = np.asarray(path).ravel().astype(int)
    if s.size == 0:
        return {}
    # run lengths
    changes = np.flatnonzero(s[1:] != s[:-1]) + 1
    bounds = np.concatenate(([0], changes, [s.size]))
    runs = np.diff(bounds)
    dwells_by_state = {k: runs[s[bounds[:-1]] == k].tolist()
                       for k in np.unique(s)}
    mean_dwell = float(np.mean(runs))
    median_dwell = float(np.median(runs))
    n_runs = int(runs.size)
    switch_rate = float((n_runs - 1) / s.size)
    short = runs[runs <= min_dwell]
    return {
        "n_states": int(np.unique(s).size),
        "n_switches": int(n_runs - 1),
        "mean_dwell": mean_dwell,
        "median_dwell": median_dwell,
        "switch_rate": switch_rate,
        "chattering_fraction": float(short.size / n_runs) if n_runs else 0.0,
        "dwells_by_state": dwells_by_state,
    }


def chattering_index(path: np.ndarray, min_dwell: int = 2) -> float:
    """Fraction of state runs shorter than min_dwell (0 = no chatter)."""
    return float(state_path_stats(path, min_dwell=min_dwell)
                 .get("chattering_fraction", 0.0))


def run_free_run_suite(real: np.ndarray, generated: np.ndarray,
                       state_real: Optional[Sequence] = None,
                       state_gen: Optional[Sequence] = None,
                       max_lag: Optional[int] = None,
                       dt: float = 1.0) -> Dict[str, object]:
    """Full three-axis suite for one (real, generated) pair.

    If state paths are given, state dynamics are included.  dt (seconds per
    frame) converts dwell times to seconds so the same numbers are portable
    across species/sampling rates.
    """
    out: Dict[str, object] = {
        "corr_matrix_mse": corr_matrix_mse(real, generated),
        "variance_ratio": variance_ratio(real, generated),
        "autocorr": autocorr_profile_mse(real, generated, max_lag=max_lag),
    }
    if state_real is not None and state_gen is not None:
        sr = state_path_stats(state_real)
        sg = state_path_stats(state_gen)
        out["state_real"] = sr
        out["state_gen"] = sg
        out["dwell_mean_ratio"] = (sg.get("mean_dwell", float("nan"))
                                   / max(sr.get("mean_dwell", 1e-9), 1e-9))
        out["switch_rate_ratio"] = (sg.get("switch_rate", float("nan"))
                                    / max(sr.get("switch_rate", 1e-9), 1e-9))
        out["chatter_diff"] = (sg.get("chattering_fraction", 0.0)
                               - sr.get("chattering_fraction", 0.0))
        for key in ("mean_dwell", "median_dwell"):
            out[f"{key}_s_real"] = sr.get(key, float("nan")) * dt
            out[f"{key}_s_gen"] = sg.get(key, float("nan")) * dt
    return out


def intrinsic_rollout_stats(states: np.ndarray) -> Dict[str, float]:
    """Autonomy statistics for an unforced latent rollout.

    ``states`` is ``(B, K, D)``: a free run of the latent dynamics in which no
    observation is injected after the first step.  This is what "free-run
    stability" means for a dynamical system — how far the state travels per
    step, and whether the trajectory contracts toward a fixed point or expands
    without bound.

    Comparing a one-step decoder output against its target measures *prediction*
    quality, not autonomy, and should never be reported as a free run.

    Returns ``step_norm_mean`` (mean ``||z_{k+1} - z_k||``), ``step_norm_last``
    (the final step, where contraction shows up first) and ``tail_std_ratio``
    (per-component spread over the last third relative to the first third;
    ``1.0`` is stationary, ``< 1`` contracts, ``> 1`` expands).
    """
    s = np.asarray(states, dtype=float)
    if s.ndim != 3 or s.shape[1] < 3:
        raise ValueError("states must be (B, K, D) with K >= 3")

    steps = np.linalg.norm(np.diff(s, axis=1), axis=-1)  # (B, K-1)
    window = max(1, s.shape[1] // 3)
    head = float(s[:, :window, :].std(axis=1).mean())
    tail = float(s[:, -window:, :].std(axis=1).mean())
    return {
        "steps": int(s.shape[1]),
        "step_norm_mean": float(steps.mean()),
        "step_norm_last": float(steps[:, -1].mean()),
        "tail_std_ratio": tail / max(head, 1e-12),
    }
