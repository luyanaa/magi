"""Learned low-rank covariance rates for latent stochastic transitions."""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LowRankDiffusionHead(nn.Module):
    """State/control-conditioned low-rank diffusion plus diagonal floor.

    The covariance rate is ``factor @ factor.T + diag(diag_std**2)``.
    ``factor`` has shape ``(B, latent_dim, rank)`` and ``diag_std`` is a
    standard-deviation rate with shape ``(B, latent_dim)``. Both are measured
    per square-root unit of the caller's physical time.

    The learned basis is shared across states; its rank-wise amplitudes depend
    on the latent state and optional control. This keeps parameter and compute
    costs linear in ``latent_dim * rank`` rather than materializing a dense
    covariance matrix.
    """

    def __init__(
        self,
        latent_dim: int,
        rank: int = 16,
        control_dim: Optional[int] = None,
        floor: float = 1e-4,
        scale: float = 1e-2,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if rank <= 0 or rank > latent_dim:
            raise ValueError("rank must be in 1..latent_dim")
        if control_dim is not None and control_dim <= 0:
            raise ValueError("control_dim must be positive when provided")
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("floor must be finite and positive")
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and non-negative")

        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        self.control_dim = control_dim
        self.floor = float(floor)
        self.scale = float(scale)
        input_dim = self.latent_dim + int(control_dim or 0)
        hidden_dim = int(hidden_dim or min(max(self.latent_dim, 64), 512))
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.factor_basis = nn.Parameter(torch.empty(self.latent_dim, self.rank))
        nn.init.orthogonal_(self.factor_basis)
        self.rank_scale = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.rank),
        )
        # Start at equal average diagonal power in the low-rank and diagonal
        # components; the SDE remains opt-in and this scale is intentionally
        # small in physical units.
        initial_rank_scale = math.sqrt(self.latent_dim / self.rank)
        rank_bias = math.log(math.expm1(initial_rank_scale))
        nn.init.zeros_(self.rank_scale[-1].weight)
        nn.init.constant_(self.rank_scale[-1].bias, rank_bias)
        diag_bias = math.log(math.expm1(1.0))
        self.diag_std_raw = nn.Parameter(
            torch.full((self.latent_dim,), diag_bias))

    def forward(
        self,
        state: torch.Tensor,
        control: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() != 2 or state.shape[-1] != self.latent_dim:
            raise ValueError("state must have shape (B, latent_dim)")
        if self.control_dim is None:
            if control is not None:
                raise ValueError("control supplied to an unconditioned diffusion head")
            features = state
        else:
            if control is None:
                control = state.new_zeros((state.shape[0], self.control_dim))
            if control.shape != (state.shape[0], self.control_dim):
                raise ValueError("control must have shape (B, control_dim)")
            features = torch.cat((state, control.to(state)), dim=-1)

        rank_scale = F.softplus(self.rank_scale(features)) * self.scale
        factor = self.factor_basis.to(state).unsqueeze(0) * rank_scale.unsqueeze(1)
        diag_std = (
            self.floor + self.scale * F.softplus(self.diag_std_raw.to(state))
        ).unsqueeze(0).expand(state.shape[0], -1)
        return factor, diag_std
