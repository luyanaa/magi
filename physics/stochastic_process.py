"""Explicit controlled-SDE contracts and path-likelihood diagnostics.

The functions in this module are deliberately conditional diagnostics.  They
become thermodynamic claims only when the caller supplies a valid forward and
reverse process, diffusion convention, reversal map, and boundary densities.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class ControlledSDEContract:
    """Contract for an Ito SDE with covariance-rate diffusion.

    The process is interpreted as

        dz = b(z, u, t) dt + sigma(z, u, t) dW,

    where ``diffusion_fn`` returns the covariance rate ``sigma sigma.T`` as
    either a diagonal vector ``(B, D)`` or a full matrix ``(B, D, D)``.
    ``dt`` is a positive numeric step in the caller's declared time unit.
    """

    state_dim: int
    control_dim: Optional[int]
    dt: float
    diffusion_kind: str = "covariance_rate"

    def __post_init__(self):
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.control_dim is not None and self.control_dim <= 0:
            raise ValueError("control_dim must be positive when provided")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.diffusion_kind != "covariance_rate":
            raise ValueError("only covariance_rate diffusion is supported")

    def validate(
        self,
        state: Tensor,
        control: Optional[Tensor],
        drift: Tensor,
        diffusion: Tensor,
    ) -> None:
        if state.dim() != 2 or state.shape[-1] != self.state_dim:
            raise ValueError("state must have shape (B, state_dim)")
        if drift.shape != state.shape:
            raise ValueError("drift must have the same shape as state")
        if self.control_dim is not None and control is None:
            raise ValueError("control is required by this SDE contract")
        if control is not None:
            if control.dim() != 2 or control.shape[0] != state.shape[0]:
                raise ValueError("control must have shape (B, control_dim)")
            if self.control_dim is not None and control.shape[-1] != self.control_dim:
                raise ValueError("control has the wrong control_dim")
        if diffusion.shape not in {
            (state.shape[0], self.state_dim),
            (state.shape[0], self.state_dim, self.state_dim),
        }:
            raise ValueError("diffusion must be (B, D) or (B, D, D)")


class ControlledSDE:
    """Small Euler-Maruyama wrapper around an explicit controlled SDE."""

    def __init__(
        self,
        drift_fn: Callable[[Tensor, Optional[Tensor]], Tensor],
        diffusion_fn: Callable[[Tensor, Optional[Tensor]], Tensor],
        contract: ControlledSDEContract,
    ):
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.contract = contract

    def drift(self, state: Tensor, control: Optional[Tensor] = None) -> Tensor:
        drift = self.drift_fn(state, control)
        diffusion = self.diffusion_fn(state, control)
        self.contract.validate(state, control, drift, diffusion)
        return drift

    def diffusion(self, state: Tensor, control: Optional[Tensor] = None) -> Tensor:
        diffusion = self.diffusion_fn(state, control)
        drift = self.drift_fn(state, control)
        self.contract.validate(state, control, drift, diffusion)
        return diffusion

    def step(
        self,
        state: Tensor,
        control: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Take one Euler-Maruyama step using covariance-rate diffusion."""
        drift = self.drift_fn(state, control)
        diffusion = self.diffusion_fn(state, control)
        self.contract.validate(state, control, drift, diffusion)
        dt = self.contract.dt
        if diffusion.dim() == 2:
            if noise is None:
                noise = torch.randn_like(state)
            if noise.shape != state.shape:
                raise ValueError("diagonal diffusion noise must match state shape")
            stochastic = torch.sqrt(diffusion.clamp_min(0.0) * dt) * noise
        else:
            if noise is None:
                noise = torch.randn_like(state)
            if noise.shape != state.shape:
                raise ValueError("full diffusion noise must match state shape")
            cov = _regularize_covariance(diffusion, dt)
            chol = torch.linalg.cholesky(cov)
            stochastic = torch.bmm(chol, noise.unsqueeze(-1)).squeeze(-1)
        return state + dt * drift + stochastic

    def transition_log_prob(
        self,
        next_state: Tensor,
        state: Tensor,
        control: Optional[Tensor] = None,
    ) -> Tensor:
        """Return conditional Gaussian log probability per batch row."""
        drift = self.drift_fn(state, control)
        diffusion = self.diffusion_fn(state, control)
        self.contract.validate(state, control, drift, diffusion)
        return gaussian_transition_log_prob(
            next_state, state + self.contract.dt * drift, diffusion, self.contract.dt)


def _regularize_covariance(diffusion: Tensor, dt: float, eps: float = 1e-6) -> Tensor:
    cov = 0.5 * (diffusion + diffusion.transpose(-1, -2)) * dt
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    return cov + eps * eye.unsqueeze(0)


def gaussian_transition_log_prob(
    next_state: Tensor,
    mean: Tensor,
    diffusion: Tensor,
    dt: float,
    eps: float = 1e-6,
) -> Tensor:
    """Evaluate a Gaussian transition with covariance ``dt * diffusion``."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if next_state.shape != mean.shape or next_state.dim() != 2:
        raise ValueError("next_state and mean must have shape (B, D)")
    residual = next_state - mean
    if diffusion.shape == next_state.shape:
        variance = diffusion.clamp_min(eps) * dt
        return -0.5 * (
            residual.square() / variance
            + torch.log(2.0 * torch.pi * variance)
        ).sum(dim=-1)
    if diffusion.shape != (next_state.shape[0], next_state.shape[1], next_state.shape[1]):
        raise ValueError("diffusion must be diagonal or full covariance rate")
    covariance = _regularize_covariance(diffusion, dt, eps=eps)
    chol = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
    quadratic = (residual * solved).sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    dimension = next_state.shape[-1]
    return -0.5 * (quadratic + log_det + dimension * torch.log(
        torch.as_tensor(2.0 * torch.pi, dtype=next_state.dtype, device=next_state.device)))


def path_log_likelihood(
    states: Tensor,
    drifts: Tensor,
    diffusions: Tensor,
    dt: float,
) -> Tensor:
    """Return conditional path log likelihood per batch row.

    ``states`` has shape ``(T, B, D)``.  ``drifts`` and ``diffusions`` have
    shape ``(T-1, B, D)`` and ``(T-1, B, D[, D])`` respectively, evaluated
    at each forward transition.  Boundary-state probabilities are excluded.
    """
    if (
        states.dim() != 3
        or drifts.dim() != 3
        or states.shape[0] < 2
        or drifts.shape != (states.shape[0] - 1, states.shape[1], states.shape[2])
    ):
        raise ValueError("states must be (T,B,D) and drifts must be (T-1,B,D)")
    if diffusions.shape[0:2] != (states.shape[0] - 1, states.shape[1]):
        raise ValueError("diffusions must have leading shape (T-1,B)")
    total = torch.zeros(states.shape[1], dtype=states.dtype, device=states.device)
    for t in range(states.shape[0] - 1):
        total = total + gaussian_transition_log_prob(
            states[t + 1],
            states[t] + dt * drifts[t],
            diffusions[t],
            dt,
        )
    return total


def estimate_path_entropy_production(
    states: Tensor,
    forward_drifts: Tensor,
    forward_diffusions: Tensor,
    reverse_drifts: Tensor,
    reverse_diffusions: Tensor,
    dt: float,
    initial_log_prob: Optional[Tensor] = None,
    reverse_initial_log_prob: Optional[Tensor] = None,
    reverse_states: Optional[Tensor] = None,
) -> Dict[str, Tensor | bool]:
    """Compare forward and reverse conditional path likelihoods.

    ``reverse_drifts`` and ``reverse_diffusions`` must be evaluated along the
    supplied ``reverse_states`` path.  If ``reverse_states`` is omitted, the
    time-reversed state sequence is used; callers with odd variables must pass
    a parity-adjusted reversal explicitly.  Without both boundary log
    densities, the result is only a *conditional path log ratio*, not total
    entropy production.
    """
    if reverse_states is None:
        reverse_states = states.flip(0)
    if reverse_states.shape != states.shape:
        raise ValueError("reverse_states must have the same shape as states")
    forward_logp = path_log_likelihood(states, forward_drifts, forward_diffusions, dt)
    reverse_logp = path_log_likelihood(
        reverse_states, reverse_drifts, reverse_diffusions, dt)
    conditional_ratio = forward_logp - reverse_logp
    result: Dict[str, Tensor | bool] = {
        "forward_path_log_prob": forward_logp,
        "reverse_path_log_prob": reverse_logp,
        "conditional_path_log_ratio": conditional_ratio,
        "boundary_terms_supplied": False,
    }
    if initial_log_prob is not None or reverse_initial_log_prob is not None:
        if initial_log_prob is None or reverse_initial_log_prob is None:
            raise ValueError("both boundary log probabilities are required")
        boundary = torch.as_tensor(initial_log_prob) - torch.as_tensor(reverse_initial_log_prob)
        if boundary.shape not in {(), (states.shape[1],)}:
            raise ValueError("boundary log probabilities must be scalar or shape (B,)")
        result["boundary_log_ratio"] = boundary
        result["total_path_entropy_production"] = conditional_ratio + boundary
        result["boundary_terms_supplied"] = True
    return result
