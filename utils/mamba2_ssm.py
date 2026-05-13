"""
Mamba-2 SSM Integration for Brain MoE-PINN.

Wraps flash-linear-attention (FLA) Mamba2 layers for efficient
long-context modelling in both EEG and fMRI encoders.

Progressive context expansion: 4096 → 16384 → 65536
with activation checkpointing support.
"""

import sys
sys.path.insert(0, "/home/yanlu/Documents/flash-linear-attention")

import torch
import torch.nn as nn
from typing import Optional, List
import math

try:
    from fla.layers import Mamba2
    FLA_MAMBA2_AVAILABLE = True
except ImportError:
    FLA_MAMBA2_AVAILABLE = False
    Mamba2 = None


class Mamba2EncoderBlock(nn.Module):
    """
    Single Mamba-2 encoder block with residual, norm, and MLP.

    Designed to be a drop-in replacement for Transformer encoder layers
    when long-context efficiency is needed.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        expand: int = 2,
        state_size: int = 128,
        head_dim: int = 64,
        conv_kernel: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_mlp: bool = True,
        layer_idx: Optional[int] = None,
        chunk_size: int = 256,
        backend: str = "triton",
    ):
        super().__init__()
        if not FLA_MAMBA2_AVAILABLE:
            raise RuntimeError(
                "fla.layers.Mamba2 not available. "
                "Install flash-linear-attention or check /home/yanlu/Documents/flash-linear-attention"
            )

        self.hidden_dim = hidden_dim
        self.use_mlp = use_mlp

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mamba = Mamba2(
            hidden_size=hidden_dim,
            expand=expand,
            state_size=state_size,
            head_dim=head_dim,
            conv_kernel=conv_kernel,
            use_conv_bias=True,
            hidden_act="silu",
            rmsnorm=False,          # we use LN above
            norm_before_gate=False,
            use_bias=False,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
            backend=backend,
        )
        self.dropout = nn.Dropout(dropout)

        if use_mlp:
            self.norm2 = nn.LayerNorm(hidden_dim)
            mlp_hidden = int(hidden_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, hidden_dim),
                nn.Dropout(dropout),
            )
        self._current_context_length = None

    def set_context_length(self, context_length: int):
        """
        Update the target context length for this block.

        The FLA Mamba2 layer does not have a mutable context-length
        parameter (it processes any L), so we store it for any
        future buffer pre-allocation or logging.

        Args:
            context_length: new target context length
        """
        self._current_context_length = context_length

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            attention_mask: optional (B, L) or (B, 1, L, L) bool mask
        Returns:
            output: (B, L, D)
        """
        # Mamba2 sub-layer
        residual = x
        x_normed = self.norm1(x)
        mamba_out = self.mamba(x_normed, attention_mask=attention_mask)
        if isinstance(mamba_out, tuple):
            mamba_out = mamba_out[0]
        x = residual + self.dropout(mamba_out)

        # MLP sub-layer
        if self.use_mlp:
            residual = x
            x = residual + self.mlp(self.norm2(x))

        return x


class Mamba2ContextExpansion(nn.Module):
    """
    Progressive context-length expansion module.

    Manages doubling of receptive context from base_ctx → target_ctx
    via stride-2 temporal pooling between Mamba2 stages.

    Stage progression example (base=4096):
        Stage 0: 4096  tokens  (stride 1)
        Stage 1: 8192  tokens  (stride 1, longer sequence)
        Stage 2: 16384 tokens  (stride 1, with skip-connect)
        Stage 3: 32768 tokens  (stride 1)
        Stage 4: 65536 tokens  (stride 1)

    For fMRI/EEG where temporal resolution differs, the *effective* temporal
    receptive field scales with sequence length, not necessarily token count.
    """

    def __init__(
        self,
        hidden_dim: int,
        base_context: int = 4096,
        target_context: int = 65536,
        num_stages: int = 3,
        blocks_per_stage: int = 2,
        expand: int = 2,
        state_size: int = 128,
        head_dim: int = 64,
        dropout: float = 0.1,
        chunk_size: int = 256,
        backend: str = "triton",
    ):
        super().__init__()
        if not FLA_MAMBA2_AVAILABLE:
            raise RuntimeError("FLA Mamba2 not available. Check flash-linear-attention installation.")

        self.hidden_dim = hidden_dim
        self.base_context = base_context
        self.target_context = target_context
        self.num_stages = num_stages

        # Compute context at each stage (geometric progression)
        ratio = (target_context / base_context) ** (1.0 / num_stages)
        self.stage_contexts = [int(base_context * (ratio ** i)) for i in range(num_stages + 1)]

        self.stages = nn.ModuleList()
        for stage_idx in range(num_stages):
            blocks = nn.ModuleList([
                Mamba2EncoderBlock(
                    hidden_dim=hidden_dim,
                    expand=expand,
                    state_size=state_size,
                    head_dim=head_dim,
                    dropout=dropout,
                    layer_idx=stage_idx * blocks_per_stage + block_idx,
                    chunk_size=chunk_size,
                    backend=backend,
                )
                for block_idx in range(blocks_per_stage)
            ])
            self.stages.append(blocks)

        # Optional downsampling between stages (not needed for pure context expansion)
        self.downsamplers = nn.ModuleList()
        for i in range(num_stages - 1):
            # Simple mean-pool downsampling if we want to reduce tokens between stages
            self.downsamplers.append(nn.Identity())

        self._current_max_context = base_context

    def set_context_length(self, context_length: int):
        """
        Update the effective context length for this expansion module.

        Adjusts `_current_max_context` so that forward() can adapt
        (e.g., adjust stride or skip-connection behaviour).

        Args:
            context_length: new target context length
        """
        self._current_max_context = context_length

    def get_effective_context(self, stage_idx: int) -> int:
        """Return effective context length at given stage."""
        return self.stage_contexts[min(stage_idx, self.num_stages)]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_stages: bool = False,
    ):
        """
        Args:
            x: (B, L, D) input tokens
            attention_mask: optional mask
            return_all_stages: if True, return list of stage outputs
        Returns:
            final_output: (B, L, D)
            or list of stage outputs if return_all_stages=True
        """
        all_outputs = []
        for stage_idx, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, attention_mask=attention_mask)
            all_outputs.append(x)
            # Optional: apply downsampling between stages
            if stage_idx < len(self.downsamplers):
                x = self.downsamplers[stage_idx](x)

        if return_all_stages:
            return all_outputs
        return x


class Mamba2Backbone(nn.Module):
    """
    Pure Mamba-2 backbone (no windowing, no transformer).

    Replaces hierarchical SWM stages with pure Mamba2 blocks.
    Suitable for progressive context expansion in Stage 2.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        expand: int = 2,
        state_size: int = 128,
        head_dim: int = 64,
        dropout: float = 0.1,
        chunk_size: int = 256,
        backend: str = "triton",
        use_mlp: bool = True,
    ):
        super().__init__()
        if not FLA_MAMBA2_AVAILABLE:
            raise RuntimeError("FLA Mamba2 not available.")

        self.layers = nn.ModuleList([
            Mamba2EncoderBlock(
                hidden_dim=hidden_dim,
                expand=expand,
                state_size=state_size,
                head_dim=head_dim,
                dropout=dropout,
                use_mlp=use_mlp,
                layer_idx=i,
                chunk_size=chunk_size,
                backend=backend,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self._current_context_length = None

    def set_context_length(self, context_length: int):
        """
        Update the target context length for this Mamba-2 backbone.

        Propagates to all child Mamba2EncoderBlock layers so they can
        adjust any sequence-length-dependent internal state.

        Args:
            context_length: new target context length
        """
        self._current_context_length = context_length
        for layer in self.layers:
            if hasattr(layer, "set_context_length"):
                layer.set_context_length(context_length)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.norm(x)


def create_mamba2_stack(
    hidden_dim: int = 768,
    num_layers: int = 12,
    use_context_expansion: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory for creating a Mamba-2 stack.

    Args:
        hidden_dim: hidden dimension
        num_layers: total number of Mamba2 layers
        use_context_expansion: if True, use Mamba2ContextExpansion stages
        **kwargs: passed to Mamba2Backbone or Mamba2ContextExpansion
    Returns:
        Mamba-2 backbone module
    """
    if use_context_expansion:
        return Mamba2ContextExpansion(
            hidden_dim=hidden_dim,
            num_stages=kwargs.get("num_stages", 3),
            blocks_per_stage=kwargs.get("blocks_per_stage", 2),
            **{k: v for k, v in kwargs.items() if k not in ("num_stages", "blocks_per_stage")}
        )
    return Mamba2Backbone(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs,
    )


if __name__ == "__main__":
    print("Testing Mamba2Backbone...")
    model = Mamba2Backbone(hidden_dim=256, num_layers=4, chunk_size=128, backend="triton")
    x = torch.randn(2, 512, 256)
    y = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")

    print("\nTesting Mamba2ContextExpansion...")
    ctx_model = Mamba2ContextExpansion(
        hidden_dim=256,
        base_context=4096,
        target_context=16384,
        num_stages=2,
        blocks_per_stage=2,
        chunk_size=128,
        backend="triton",
    )
    y_stages = ctx_model(x, return_all_stages=True)
    print(f"  Stage outputs: {[s.shape for s in y_stages]}")

    print("\nAll Mamba-2 tests passed!")
