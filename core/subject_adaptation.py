"""Test-time subject adaptation of the shared latent dynamics.

Design (see design discussion 2026-09):
  - The population model owns a *shared* latent dynamics field f(z).
  - A new subject differs mainly in *which* latent directions are expressed
    and in the transition/gates, not in the encoder stack.
  - Adaptation therefore lives in the DYNAMICS, as a low-rank residual:

        f_subject(z) = f(z) + U @ diag(lambda) @ (V^T z)

    with U, V in R^{D x r} (random, frozen) and lambda in R^r (the only
    trainable/test-time parameters).  At lambda = 0 the subject model is
    exactly the population model (safe default).

  This is the nonlinear generalisation of TDE-RICA's per-animal occurrence
  solve (shared motifs M, per-animal weights W) and is far more parameter
  efficient than weight-space LoRA on every layer.

  Two test-time routes are provided:
    1. amortised:  CalibrationEncoder maps a short calibration window to
                   lambda (one forward pass at test time).
    2. gradient:   test_time_adapt() takes a few gradient steps on the
                   calibration window (CAVIA-lite) - used as refinement or
                   as the primary route when the encoder underfits.

  Falsifiability: use evaluate_subject_adaptation() with leave-subject-out
  splits and compare against (a) no adaptation and (b) per-subject-from-
  scratch models.
"""

import torch
from torch import nn


class LowRankDynamicsAdapter(nn.Module):
    """Low-rank residual on a latent dynamics field: f(z) + U diag(lam) V^T z.

    U, V are random projection pairs (frozen); only `lam` is adapted.
    Zero-initialised so the adapted model starts at the population model.
    """

    def __init__(self, dim: int, rank: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        U = torch.randn(dim, rank, generator=g) / (rank ** 0.5)
        V = torch.randn(dim, rank, generator=g) / (rank ** 0.5)
        self.register_buffer("U", U)
        self.register_buffer("V", V)
        self.lam = nn.Parameter(torch.zeros(rank))

    def reset(self):
        """Return to the population model (no adaptation)."""
        with torch.no_grad():
            self.lam.zero_()

    def adapt(self, dz: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """dz: population-field output (..., D); z: state (..., D)."""
        return dz + self.adapt_direct(z)

    def adapt_direct(self, z: torch.Tensor) -> torch.Tensor:
        """Return the residual U diag(lambda) V^T z (for inspection)."""
        proj = z @ self.V                            # (..., r)
        return (proj * self.lam) @ self.U.t()        # (..., D)


class CalibrationEncoder(nn.Module):
    """Amortised route: short window -> lambda (low-rank coefficients)."""

    def __init__(self, dim: int, rank: int = 16, hidden: int = 128):
        super().__init__()
        self.rank = rank
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, rank),
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: (B, T, D) -> lambda (B, r)."""
        z = window.mean(dim=1)                       # (B, D)
        return self.net(z)


def test_time_adapt(adapter: LowRankDynamicsAdapter,
                    field_fn, window: torch.Tensor,
                    steps: int = 3, lr: float = 1e-2,
                    dt: float = 1.0) -> torch.Tensor:
    """CAVIA-lite: gradient steps on lambda minimising one-step predictive
    error of the adapted field over the calibration window.

    field_fn(z) -> dz of the population model (no adaptation).
    window: (T, D) latent trajectory of the new subject.
    Mutates adapter.lam in place; returns the adapted lambda.
    """
    adapter.reset()
    opt = torch.optim.Adam([adapter.lam], lr=lr)
    z = window
    dz_true = (z[1:] - z[:-1]) / dt
    for _ in range(steps):
        opt.zero_grad()
        with torch.no_grad():
            dz_pop = field_fn(z[:-1])
        dz_hat = adapter.adapt(dz_pop, z[:-1])
        loss = torch.mean((dz_hat - dz_true) ** 2)
        loss.backward()
        opt.step()
    return adapter.lam.detach().clone()


def evaluate_subject_adaptation(field_fn, subjects, adapter=None,
                                steps: int = 3, lr: float = 1e-2,
                                calib_len: int = 64, dt: float = 1.0,
                                calib_encoder: nn.Module = None):
    """Leave-subject-out-style evaluation of dynamics adaptation.

    subjects: list of (name, z) with z: (T, D) latent trajectories.
    For each subject S: fit lambda on S's first `calib_len` frames
    (gradient route, or amortised if calib_encoder is given), then measure
    one-step predictive MSE on the remainder vs (a) no adaptation.
    Returns per-subject dicts and the mean improvement.
    """
    results = []
    for name, z in subjects:
        z = torch.as_tensor(z, dtype=torch.float32)
        calib = z[:calib_len]
        test = z[calib_len:]
        with torch.no_grad():
            dz_test_true = (test[1:] - test[:-1]) / dt
            dz_pop = field_fn(test[:-1])
        base_mse = torch.mean((dz_pop - dz_test_true) ** 2).item()

        if adapter is None:
            adapter = LowRankDynamicsAdapter(z.shape[-1])
        if calib_encoder is not None:
            lam = calib_encoder(calib[None])[0]
            with torch.no_grad():
                adapter.lam.copy_(lam)
        else:
            test_time_adapt(adapter, field_fn, calib, steps=steps, lr=lr, dt=dt)
        with torch.no_grad():
            dz_hat = adapter.adapt(field_fn(test[:-1]), test[:-1])
        adapted_mse = torch.mean((dz_hat - dz_test_true) ** 2).item()
        results.append({"subject": name, "base_mse": float(base_mse),
                        "adapted_mse": float(adapted_mse),
                        "improvement": float(base_mse - adapted_mse)})
        adapter.reset()
    mean_imp = float(sum(r["improvement"] for r in results) / len(results))
    return results, mean_imp
