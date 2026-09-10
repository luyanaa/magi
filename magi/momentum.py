"""Shared exponential-moving-average encoder wrapper."""

from __future__ import annotations

import copy

import torch
from torch import nn


class MomentumEncoder(nn.Module):
    """Run a frozen EMA copy of a trainable encoder."""

    def __init__(self, base_encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1")

        self.base_encoder = base_encoder
        # Keep the v1 state-dict name stable. ``momentum_encoder`` is exposed
        # as a property for callers that use the clearer v2 terminology.
        self.encoder = copy.deepcopy(base_encoder)
        self.momentum = momentum
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @property
    def momentum_encoder(self) -> nn.Module:
        """Return the frozen EMA copy using the v2-compatible name."""
        return self.encoder

    @torch.no_grad()
    def update(self) -> None:
        """Update the EMA copy from the trainable encoder."""
        for query, key in zip(
            self.base_encoder.parameters(), self.encoder.parameters()
        ):
            key.mul_(self.momentum).add_(query, alpha=1.0 - self.momentum)

    def forward(self, *args, **kwargs):
        """Run the frozen EMA copy without constructing a gradient graph."""
        with torch.no_grad():
            return self.encoder(*args, **kwargs)
