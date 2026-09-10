"""Hierarchical gating: slow latent variables gating fast-state transitions.

Motivation (2026-09 design pass):
  Macro-states (sleep stages, arousal, task modes) are typically COUPLED -
  e.g. arousal modulates how task states transition.  Forcing one flat set of
  discrete states is therefore questionable as a *mechanism*.  This module
  treats states as *tools*: a two-level structure where slow variables
  (multi-timescale EMAs of the latent, cf. MultiTimeScaleKDA) gate the
  transition logits of faster states.

  Validation-first philosophy: the discrete layer earns its place only if it
  predicts something external (behavioural labels, dwell statistics).  Use
  validate_states() against such labels before trusting the states.
"""

import torch
from torch import nn
from typing import Optional


class SlowGateTransition(nn.Module):
    """Slow-latent-gated transition logits with explicit sequence state.

    ``timescales`` are positive time constants in the same units as ``dt``.
    The EMA decay is ``exp(-dt / tau)``; frame-rate changes therefore do not
    silently change the intended physical time constant.

    The legacy ``forward`` method maintains state only for batch size one.
    Batched or distributed sequence processing should use
    ``forward_with_state`` and carry the returned ``ema_state`` explicitly.
    """

    def __init__(self, dim: int, k_states: int = 6, hidden: int = 64,
                 timescales=(0.1, 0.5, 0.9), stickiness: float = 0.0,
                 dt: float = 1.0):
        super().__init__()
        if dt <= 0:
            raise ValueError("dt must be positive")
        if any(tau <= 0 for tau in timescales):
            raise ValueError("timescales must be positive")
        self.k = k_states
        self.timescales = tuple(float(tau) for tau in timescales)
        self.dt = float(dt)
        self.stickiness = float(stickiness)
        self.net = nn.Sequential(
            nn.Linear(dim * (1 + len(timescales)), hidden), nn.SiLU(),
            nn.Linear(hidden, k_states),
        )
        decay = torch.exp(-torch.as_tensor(self.dt / torch.as_tensor(
            self.timescales, dtype=torch.float32)))
        self.register_buffer("decay", decay)
        # Internal state is intentionally single-sequence only.  Batched
        # callers must pass state explicitly to avoid cross-subject leakage.
        self.register_buffer("ema", torch.zeros(len(timescales), dim))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    def reset(self, z0: torch.Tensor):
        """Reset the internal single-sequence state from one latent vector."""
        if z0.dim() != 1 or z0.shape[0] != self.ema.shape[-1]:
            raise ValueError("reset expects z0 with shape (dim,)")
        with torch.no_grad():
            self.ema.copy_(z0.expand_as(self.ema))
            self.initialized.fill_(True)

    def _update_ema(self, z: torch.Tensor, ema_state: torch.Tensor) -> torch.Tensor:
        decay = self.decay.to(dtype=z.dtype, device=z.device)
        return (
            decay[None, :, None] * ema_state
            + (1.0 - decay[None, :, None]) * z[:, None, :]
        )

    def _state_logits(
        self,
        z: torch.Tensor,
        z_state: Optional[torch.Tensor],
        ema_state: torch.Tensor,
    ) -> torch.Tensor:
        slow = ema_state.reshape(z.shape[0], -1)
        feat = torch.cat([z, slow], dim=-1)
        logits = self.net(feat)
        if self.stickiness != 0.0 and z_state is not None:
            if z_state.dim() == 1:
                if z_state.shape[0] != z.shape[0]:
                    raise ValueError("state indices must have shape (B,)")
                cur = torch.nn.functional.one_hot(z_state.long(), self.k)
                cur = cur.to(dtype=z.dtype, device=z.device)
            elif z_state.dim() == 2 and z_state.shape == (z.shape[0], self.k):
                cur = z_state.to(dtype=z.dtype, device=z.device)
            else:
                raise ValueError(
                    "z_state must be indices (B,) or probabilities (B, K)")
            logits = logits + self.stickiness * cur
        return logits

    def forward_with_state(
        self,
        z: torch.Tensor,
        z_state: Optional[torch.Tensor] = None,
        ema_state: Optional[torch.Tensor] = None,
    ):
        """Return ``(logits, next_ema_state)`` without hidden batch state."""
        if z.dim() != 2:
            raise ValueError("z must have shape (B, D)")
        B = z.shape[0]
        persist_internal = B == 1 and ema_state is None
        if ema_state is None:
            if persist_internal and bool(self.initialized.item()):
                state = self.ema.to(dtype=z.dtype, device=z.device)[None]
            else:
                # Start a new independent sequence at its first observation.
                state = z[:, None, :].clone()
        else:
            if ema_state.shape != (B, len(self.timescales), z.shape[-1]):
                raise ValueError(
                    "ema_state must have shape (B, n_timescales, D)")
            state = ema_state.to(dtype=z.dtype, device=z.device)
        next_state = self._update_ema(z, state)
        logits = self._state_logits(z, z_state, next_state)
        if persist_internal:
            with torch.no_grad():
                self.ema.copy_(next_state[0].to(self.ema))
                self.initialized.fill_(True)
        return logits, next_state

    @torch.no_grad()
    def calibrate_stickiness(
        self,
        z: torch.Tensor,
        states: torch.Tensor,
        candidates=None,
    ) -> dict:
        """Fit one stickiness value on a labelled calibration sequence.

        The fit minimizes next-state cross-entropy over a one-dimensional
        candidate grid while holding the learned transition network fixed.
        This makes stickiness a measurable duration-control parameter rather
        than an uncalibrated copy of an rSLDS hyperparameter.
        """
        if z.dim() != 2 or states.dim() != 1 or z.shape[0] != states.shape[0]:
            raise ValueError("z must be (T, D) and states must be (T,)")
        if z.shape[0] < 2:
            raise ValueError("at least two labelled states are required")
        if states.min() < 0 or states.max() >= self.k:
            raise ValueError("states contain an invalid state index")
        if candidates is None:
            candidates = torch.linspace(0.0, 20.0, 81, device=z.device)
        else:
            candidates = torch.as_tensor(candidates, dtype=z.dtype, device=z.device)
        if candidates.numel() == 0 or (candidates < 0).any():
            raise ValueError("candidates must be a non-empty nonnegative grid")

        was_training = self.training
        self.eval()
        self.stickiness = 0.0
        state = z[:1, None, :].clone()
        baseline = []
        for t in range(z.shape[0] - 1):
            logits, state = self.forward_with_state(z[t:t + 1], ema_state=state)
            baseline.append(logits[0])
        baseline = torch.stack(baseline)
        current = torch.nn.functional.one_hot(states[:-1].long(), self.k).to(z.dtype)
        target = states[1:].long()
        losses = []
        for value in candidates:
            losses.append(torch.nn.functional.cross_entropy(
                baseline + value * current, target))
        best = int(torch.argmin(torch.stack(losses)).item())
        self.stickiness = float(candidates[best].item())
        if was_training:
            self.train()
        with torch.no_grad():
            observed_switch_rate = float((states[1:] != states[:-1]).float().mean())
            target_stay = 1.0 - observed_switch_rate
        return {
            "stickiness": self.stickiness,
            "baseline_nll": float(losses[0].item()),
            "calibrated_nll": float(losses[best].item()),
            "observed_switch_rate": observed_switch_rate,
            "observed_stay_rate": target_stay,
        }

    def forward(self, z: torch.Tensor,
                z_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return next-state logits; explicit state is preferred for batches."""
        logits, _ = self.forward_with_state(z, z_state=z_state)
        return logits


def validate_states(states: torch.Tensor, labels: torch.Tensor) -> dict:
    """External validation of inferred discrete states vs ground-truth labels.

    Returns contingency-based agreement (Hungarian-matched accuracy) and
    per-state dwell statistics.  Offline analysis helper: numpy/scipy used
    here only for the small Hungarian match (single call, not on the
    training path).  States are a *tool*: report these numbers
    before claiming the states are meaningful.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    s = states.detach().cpu().numpy().ravel()
    y = np.asarray(labels).ravel()
    n = min(len(s), len(y))
    s, y = s[:n], y[:n]
    K = max(s.max() + 1, y.max() + 1)
    C = np.zeros((K, K), dtype=int)
    for a, b in zip(s, y):
        C[a, b] += 1
    ri, ci = linear_sum_assignment(-C)
    matched_acc = float(C[ri, ci].sum() / n)

    # dwell statistics (mean run length per state)
    runs = []
    prev, length = s[0], 1
    for v in s[1:]:
        if v == prev:
            length += 1
        else:
            runs.append(length)
            prev, length = v, 1
    runs.append(length)
    return {"matched_accuracy": matched_acc,
            "n_states": int(s.max() + 1),
            "mean_dwell": float(np.mean(runs)),
            "median_dwell": float(np.median(runs)),
            "n_switches": int(len(runs) - 1),
            "chattering_fraction": float(sum(r <= 2 for r in runs) / len(runs))}
