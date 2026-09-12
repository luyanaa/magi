"""Species-parameter estimators for the C. elegans stage (CBM-informed).

These functions do **not** simulate a conductance-based model. They estimate the
parameters of that model class from observations -- the estimands the audit
identified as identifiable from an optical calcium corpus:

* ``tau`` (per neuron, seconds): AR(1) on the recording's own sampling grid.
  What it estimates is the *observable* time constant, a composition of
  membrane, indicator and smoothing kinetics; it is only attributable to the
  membrane if the indicator kernel is fixed independently (Iwasaki-lab
  indicator kinetics), so the report keeps the composition explicit.
* ``ar_order_gain``: how much an AR(2) term adds over AR(1). Bi-exponential
  indicator kinetics would show up here; a small value means a first-order
  observation model is adequate.
* ``coupling_split``: the fitted one-step coupling matrix split into its
  symmetric (Laplacian-like, gap-junction-compatible) and antisymmetric
  (directed, chemical-synapse-compatible) parts, with the share of weight mass
  in each. The conductance forms ``I_gap = g_gap (V_post - V_pre)`` and
  ``I_syn = g(t)(V_post - V_rev)`` are exactly a symmetric and a directed
  contribution, so this split is the testable signature of the two terms.
* ``zero_lag_network_r2``: how much of a neuron's instantaneous variance is
  linearly predictable from the other neurons. This is the confounding floor
  for any "correlation implies connection" reading (see the caveat quoted from
  the corpus paper), and it bounds what a gap-junction regression can claim.

Everything is computed per recording and returned with its units; nothing here
reads a global time constant, so the estimates cannot silently inherit one
species' clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TauEstimate",
    "estimate_tau",
    "estimate_ar_order_gain",
    "fit_coupling_matrix",
    "coupling_split",
    "zero_lag_network_r2",
    "estimate_ladder_parameters",
]


@dataclass
class TauEstimate:
    """Per-neuron observable time constant (seconds)."""

    names: List[str]
    tau_s: np.ndarray
    ar1: np.ndarray
    dt_s: float
    n_frames: int

    def by_name(self) -> Dict[str, float]:
        return {name: float(tau) for name, tau in zip(self.names, self.tau_s)}

    def summary(self) -> Dict[str, float]:
        finite = self.tau_s[np.isfinite(self.tau_s)]
        if finite.size == 0:
            return {"n": 0.0}
        return {
            "n": float(finite.size),
            "median_tau_s": float(np.median(finite)),
            "iqr_low_s": float(np.percentile(finite, 25)),
            "iqr_high_s": float(np.percentile(finite, 75)),
            "dt_s": float(self.dt_s),
            "n_frames": float(self.n_frames),
        }


def estimate_tau(
    signals: np.ndarray,
    *,
    names: Optional[Sequence[str]] = None,
    dt_s: float,
    valid: Optional[np.ndarray] = None,
    max_tau_s: Optional[float] = None,
) -> TauEstimate:
    """AR(1) time constant per channel on the recording's own grid.

    ``signals`` is ``(C, T)``; ``valid`` is an optional ``(C, T)`` boolean mask
    (padded/invalid frames excluded from the fit). ``dt_s`` is the sampling
    interval in seconds -- required, never defaulted, because an AR(1)
    coefficient is only interpretable in the time unit it was fitted with.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signals must be (C, T)")
    if dt_s is None or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be a positive number of seconds")
    C, T = x.shape
    if names is None:
        names = [str(i) for i in range(C)]
    if len(names) != C:
        raise ValueError("names must have one entry per channel")
    mask = (np.ones_like(x, dtype=bool) if valid is None
            else np.asarray(valid, dtype=bool))
    if mask.shape != x.shape:
        raise ValueError("valid must match signals shape")
    x = np.where(mask, x, np.nan)
    tau = np.full(C, np.nan)
    ar1 = np.full(C, np.nan)
    for c in range(C):
        a, b = x[c, :-1], x[c, 1:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 32:
            continue
        a, b = a[ok], b[ok]
        if a.std() <= 0 or b.std() <= 0:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        ar1[c] = rho
        if 0 < rho < 1:
            tau[c] = -dt_s / np.log(rho)
        elif rho >= 1:
            tau[c] = np.inf
        else:
            tau[c] = np.nan
    if max_tau_s is not None:
        tau = np.where(tau > max_tau_s, np.nan, tau)
    return TauEstimate(names=list(names), tau_s=tau, ar1=ar1,
                       dt_s=float(dt_s), n_frames=int(T))


def estimate_ar_order_gain(
    signals: np.ndarray, *, valid: Optional[np.ndarray] = None
) -> float:
    """Median AR(2)-over-AR(1) residual reduction, per channel.

    Near zero means a first-order observation model suffices; a large value
    would indicate bi-exponential indicator kinetics (tau_on/tau_off) that a
    single time constant cannot capture.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signals must be (C, T)")
    mask = (np.ones_like(x, dtype=bool) if valid is None
            else np.asarray(valid, dtype=bool))
    x = np.where(mask, x, np.nan)
    gains: List[float] = []
    for c in range(x.shape[0]):
        row = x[c]
        y = row[2:]
        a1 = row[1:-1]
        a0 = row[:-2]
        ok = np.isfinite(y) & np.isfinite(a1) & np.isfinite(a0)
        if ok.sum() < 64:
            continue
        y, a1, a0 = y[ok], a1[ok], a0[ok]
        y = y - y.mean()
        X1 = np.stack([a1 - a1.mean()], axis=1)
        X2 = np.stack([a1 - a1.mean(), a0 - a0.mean()], axis=1)
        r1 = _residual_variance(X1, y)
        r2 = _residual_variance(X2, y)
        if r1 > 0:
            gains.append(max(0.0, (r1 - r2) / r1))
    return float(np.median(gains)) if gains else 0.0


def _residual_variance(X: np.ndarray, y: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(resid @ resid / max(1, resid.size))


def fit_coupling_matrix(
    signals: np.ndarray, *, alpha: float = 1.0,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Ridge one-step linear coupling ``x_{t+1} = W x_t`` on z-scored rows.

    Returns ``W`` with shape ``(C, C)``. ``alpha`` is the ridge penalty on
    z-scored data; a positive value keeps a wide, short recording from
    producing a spurious rank-deficient map.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signals must be (C, T)")
    mask = (np.ones_like(x, dtype=bool) if valid is None
            else np.asarray(valid, dtype=bool))
    if mask.shape != x.shape:
        raise ValueError("valid must match signals shape")
    # z-score per channel on valid frames
    count = mask.sum(axis=1, keepdims=True).clip(min=1)
    mean = (np.where(mask, x, 0.0)).sum(axis=1, keepdims=True) / count
    centred = np.where(mask, x - mean, np.nan)
    std = np.sqrt(np.nanmean(centred ** 2, axis=1, keepdims=True))
    std = np.where(std > 0, std, 1.0)
    z = np.nan_to_num((x - mean) / std, nan=0.0)
    pair_ok = mask[:, :-1] & mask[:, 1:]
    src = np.where(pair_ok.T, z[:, :-1].T, 0.0)      # (T-1, C)
    dst = np.where(pair_ok.T, z[:, 1:].T, 0.0)
    gram = src.T @ src + float(alpha) * np.eye(z.shape[0])
    return np.linalg.solve(gram, src.T @ dst).T


def coupling_split(weights: np.ndarray) -> Dict[str, float]:
    """Split a coupling matrix into symmetric and antisymmetric parts.

    ``I_gap = g_gap (V_post - V_pre)`` is symmetric with zero row sums (a graph
    Laplacian) and ``I_syn = g(t)(V_post - V_rev)`` is directed, so the
    symmetric/antisymmetric weight mass is a direct signature of which term
    dominates. The diagonal (self-coupling) is excluded: it is not a synaptic
    term.
    """
    W = np.asarray(weights, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("weights must be a square (C, C) matrix")
    off = ~np.eye(W.shape[0], dtype=bool)
    sym = 0.5 * (W + W.T)
    asym = 0.5 * (W - W.T)
    sym_mass = float(np.abs(sym[off]).sum())
    asym_mass = float(np.abs(asym[off]).sum())
    total = sym_mass + asym_mass
    return {
        "symmetric_share": sym_mass / total if total > 0 else float("nan"),
        "antisymmetric_share": asym_mass / total if total > 0 else float("nan"),
        "symmetric_mass": sym_mass,
        "antisymmetric_mass": asym_mass,
        "row_sum_abs_mean": float(np.abs(W.sum(axis=1)).mean()),
    }


def zero_lag_network_r2(
    signals: np.ndarray, *, alpha: float = 1.0,
    valid: Optional[np.ndarray] = None,
) -> float:
    """Median leave-one-out R^2 of a neuron from the others at the same time.

    This is the confounding floor for connectivity claims from correlations
    (the corpus paper warns that correlated activity need not come from the
    neuron pair itself), and it bounds what a gap-junction regression can
    attribute to an edge.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signals must be (C, T)")
    mask = (np.ones_like(x, dtype=bool) if valid is None
            else np.asarray(valid, dtype=bool))
    z = np.where(mask, x, 0.0)
    z = (z - z.mean(axis=1, keepdims=True)) / (
        z.std(axis=1, keepdims=True) + 1e-12)
    scores: List[float] = []
    C = z.shape[0]
    for c in range(C):
        others = np.delete(z, c, axis=0).T
        target = z[c]
        gram = others.T @ others + float(alpha) * np.eye(C - 1)
        coef = np.linalg.solve(gram, others.T @ target)
        resid = target - others @ coef
        scores.append(1.0 - float(resid @ resid) / max(1e-12, float(target @ target)))
    return float(np.median(scores)) if scores else float("nan")


def estimate_ladder_parameters(
    signals: np.ndarray,
    *,
    dt_s: float,
    names: Optional[Sequence[str]] = None,
    valid: Optional[np.ndarray] = None,
    max_tau_s: Optional[float] = None,
) -> Dict[str, object]:
    """One-call report for a single recording (all units explicit)."""
    tau = estimate_tau(signals, names=names, dt_s=dt_s, valid=valid,
                       max_tau_s=max_tau_s)
    weights = fit_coupling_matrix(signals, valid=valid)
    return {
        "tau": tau.summary(),
        "tau_by_name": tau.by_name(),
        "ar_order_gain": estimate_ar_order_gain(signals, valid=valid),
        "coupling": coupling_split(weights),
        "zero_lag_network_r2": zero_lag_network_r2(signals, valid=valid),
    }
