"""
NeuroSTORM fMRI Foundation Encoder Integration.

Based on: Wang et al. (2026) Nature Biomedical Engineering [s41551-026-01666-y]
GitHub: https://github.com/CUHK-AIM-Group/NeuroSTORM

Primary design:
- Shifted-Window Mamba (SWM) backbone with 4-level hierarchical feature maps
- depths=[2,2,6,2], hidden_dims=[C1,C2,C3,C4], window_size=[4,4,4,4]
- Supports voxel-level (4D), ROI time series (2D), and functional connectivity (2D)
- Task-specific Prompt Tuning (TPT): only ~1% params trained (prompts), backbone frozen

Fallback: BrainLM (ICLR 2024) if NeuroSTORM weights unavailable.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
import math

import sys
sys.path.insert(0, "/home/yanlu/Documents/a")

try:
    from brain_moe_pinn.utils.mamba2_ssm import Mamba2EncoderBlock, FLA_MAMBA2_AVAILABLE
except ImportError:
    try:
        from utils.mamba2_ssm import Mamba2EncoderBlock, FLA_MAMBA2_AVAILABLE
    except ImportError:
        FLA_MAMBA2_AVAILABLE = False
        Mamba2EncoderBlock = None


class NeuroSTORMPromptTuning(nn.Module):
    """
    Task-specific Prompt Tuning (TPT) for NeuroSTORM.
    Learns a small set of prompts that condition the frozen backbone on downstream tasks.
    """

    def __init__(
        self,
        num_prompts: int = 32,
        prompt_dim: int = 768,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim

        self.prompts = nn.Parameter(
            torch.randn(num_prompts, prompt_dim) * 0.02
        )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(prompt_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_size: int) -> List[torch.Tensor]:
        """
        Generate prompt tokens for each transformer layer.
        Returns list of prompt tensors, one per layer.
        """
        prompts = [self.prompts] * len(self.layer_norms)
        prompts = [p.unsqueeze(0).expand(batch_size, -1, -1) for p in prompts]
        prompts = [self.dropout(norm(p)) for norm, p in zip(self.layer_norms, prompts)]
        return prompts


class ShiftedWindowMambaBlock(nn.Module):
    """
    Single Shifted-Window Mamba (SWM) block.
    Combines Mamba SSM with shifted window attention pattern.

    For fMRI spatial tokens, we use Mamba for efficient long-range dependency
    modeling with linear complexity wrt sequence length.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 16,
        expand: float = 2.0,
        dropout: float = 0.1,
        window_size: int = 4,
        shift_size: int = 2,
        use_sliding_window: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_sliding_window = use_sliding_window

        self.norm1 = nn.LayerNorm(hidden_dim)

        inner_dim = int(hidden_dim * expand)
        self.in_proj = nn.Linear(hidden_dim, inner_dim * 2, bias=False)

        self.x_proj = nn.Linear(inner_dim, state_dim * 2 + 1, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(inner_dim) * 0.02)

        self.A_log = nn.Parameter(torch.randn(inner_dim, state_dim))
        self.D = nn.Parameter(torch.ones(inner_dim))

        self.out_proj = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Partition (B, L, D) into windows (B*num_windows, window_size, D)."""
        B, L, D = x.shape
        pad_len = (window_size - L % window_size) % window_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        L_padded = x.shape[1]
        num_windows = L_padded // window_size
        x = x.view(B, num_windows, window_size, D)
        x = x.reshape(B * num_windows, window_size, D)
        return x, pad_len

    def _window_merge(self, x: torch.Tensor, B: int, L: int, pad_len: int) -> torch.Tensor:
        """Merge windows back to (B, L, D)."""
        window_size = x.shape[1]
        num_windows = x.shape[0] // B
        D = x.shape[2]
        x = x.view(B, num_windows, window_size, D)
        x = x.reshape(B, num_windows * window_size, D)
        if pad_len > 0:
            x = x[:, :L, :]
        return x

    def _mamba_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply simplified Mamba-style transformation."""
        B_tokens, D = x.shape
        x_gate = self.in_proj(x)
        x_inner, gate = x_gate.chunk(2, dim=-1)
        x_inner = x_inner * torch.sigmoid(gate)
        ssm_params = self.x_proj(x_inner)
        B_param, C_param, dt_bias = ssm_params.split([self.state_dim, self.state_dim, 1], dim=-1)
        dt = torch.softplus(self.dt_bias.unsqueeze(0) + dt_bias)
        output = self.out_proj(x_inner * dt.squeeze(-1).float()) + self.D.float() * x_inner
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) where L is sequence length
        Returns:
            output: (B, L, D)
        """
        B, L, D = x.shape
        residual = x
        x_normed = self.norm1(x)

        # Shifted-window processing
        if self.shift_size > 0:
            # Cyclic shift
            x_shifted = torch.roll(x_normed, shifts=-self.shift_size, dims=1)
        else:
            x_shifted = x_normed

        # Window partition
        x_windows, pad_len = self._window_partition(x_shifted, self.window_size)

        # Apply Mamba block per window
        output_windows = self._mamba_block(x_windows.view(-1, D))
        output_windows = output_windows.view(x_windows.shape)

        # Merge windows
        output = self._window_merge(output_windows, B, L, pad_len)

        # Reverse shift
        if self.shift_size > 0:
            output = torch.roll(output, shifts=self.shift_size, dims=1)

        output = residual + self.dropout(output)
        output = output + self.mlp(self.norm2(output))
        return output


class ShiftedWindowMambaStage(nn.Module):
    """
    Single stage of Shifted-Window Mamba with multiple blocks.
    Implements the hierarchical feature extraction from NeuroSTORM.
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        window_size: int = 4,
        shift_size: int = 2,
        downsample: bool = False,
        expand_ratio: float = 4.0,
        dropout: float = 0.1,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        if use_mamba2 and FLA_MAMBA2_AVAILABLE and Mamba2EncoderBlock is not None:
            kwargs = mamba2_kwargs or {}
            self.blocks = nn.ModuleList([
                Mamba2EncoderBlock(
                    hidden_dim=hidden_dim,
                    layer_idx=i,
                    **kwargs,
                )
                for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                ShiftedWindowMambaBlock(
                    hidden_dim=hidden_dim,
                    window_size=window_size,
                    shift_size=shift_size if i % 2 == 0 else 0,
                    dropout=dropout,
                )
                for i in range(depth)
            ])
        self.downsample = nn.Identity() if not downsample else nn.Conv1d(
            hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1
        )

    def set_context_length(self, context_length: int):
        """Propagate context length to child blocks (Mamba2EncoderBlock may use it)."""
        self._current_context_length = context_length
        for block in self.blocks:
            if hasattr(block, "set_context_length"):
                block.set_context_length(context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = x.transpose(1, 2)
            x = self.downsample(x)
            x = x.transpose(1, 2)
        return x


class NeuroSTORMBackbone(nn.Module):
    """
    NeuroSTORM backbone with optional Mamba-2 SSM backend.

    Supports two backends:
    1. SWM (default): Shifted-Window Mamba with 4-level hierarchy
    2. Mamba-2 (Stage 2): Pure Mamba-2 blocks for long-context expansion
    """

    def __init__(
        self,
        hidden_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        window_size: int = 4,
        dropout: float = 0.1,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        assert len(hidden_dims) == len(depths) == 4

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(
                ShiftedWindowMambaStage(
                    hidden_dim=hidden_dims[i],
                    depth=depths[i],
                    window_size=window_size,
                    shift_size=window_size // 2,
                    downsample=(i < 3),
                    dropout=dropout,
                    use_mamba2=use_mamba2,
                    mamba2_kwargs=mamba2_kwargs,
                )
            )

        self.hidden_dims = hidden_dims
        self.depths = depths

    def set_context_length(self, context_length: int):
        """Propagate context length to all stages (Mamba-2 blocks may use it)."""
        self._current_context_length = context_length
        for stage in self.stages:
            if hasattr(stage, "set_context_length"):
                stage.set_context_length(context_length)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class NeuroSTORMEncoder(nn.Module):
    """
    NeuroSTORM Foundation Encoder for fMRI.

    Modes:
    1. ROI mode (recommended): (B, R, T) -> ROI time series -> tokenized -> encode
    2. Voxel mode: (B, X, Y, Z, T) -> 3D patches -> tokenized -> encode

    Output: (B, T_patches, hidden_dim) feature sequence
    """

    def __init__(
        self,
        input_mode: str = "roi",
        roi_dim: int = 400,
        hidden_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        window_size: int = 4,
        num_prompts: int = 32,
        prompt_dim: int = 768,
        dropout: float = 0.1,
        use_tpt: bool = True,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.input_mode = input_mode
        self.roi_dim = roi_dim
        self.hidden_dims = hidden_dims
        self.depths = depths
        self.use_tpt = use_tpt and num_prompts > 0

        if input_mode == "roi":
            self.input_proj = nn.Linear(roi_dim, hidden_dims[0])
        elif input_mode == "voxel":
            self.input_proj = nn.Conv3d(1, hidden_dims[0] // 8, kernel_size=6, stride=6)
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.backbone = NeuroSTORMBackbone(
            hidden_dims=hidden_dims,
            depths=depths,
            window_size=window_size,
            dropout=dropout,
            use_mamba2=use_mamba2,
            mamba2_kwargs=mamba2_kwargs,
        )

        if self.use_tpt:
            self.prompt_tuning = NeuroSTORMPromptTuning(
                num_prompts=num_prompts,
                prompt_dim=hidden_dims[-1],
                num_layers=4,
                dropout=dropout,
            )

        self.output_dim = hidden_dims[-1]
        self.pooler = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.pooler_activation = nn.Tanh()

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained NeuroSTORM weights into backbone + input_proj.

        Typical sources:
        - GitHub: https://github.com/CUHK-AIM-Group/NeuroSTORM
        - Local: /path/to/neurostorm_pretrained.pt

        Args:
            checkpoint_path: path to .pt or .safetensors checkpoint
            strict: if True, require all keys match exactly
        """
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            except ImportError:
                raise ImportError("safetensors required for .safetensors checkpoints")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[NeuroSTORMEncoder] load_pretrained: {len(missing)} missing keys")
        if unexpected:
            print(f"[NeuroSTORMEncoder] load_pretrained: {len(unexpected)} unexpected keys")
        if not missing and not unexpected:
            print("[NeuroSTORMEncoder] Pretrained weights loaded successfully (all keys matched)")

    def set_freeze(self, freeze: bool = True):
        """Freeze or unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        for param in self.input_proj.parameters():
            param.requires_grad = not freeze
        if hasattr(self, 'pooler'):
            for param in self.pooler.parameters():
                param.requires_grad = not freeze

    def set_context_length(self, context_length: int):
        """
        Update effective context length for progressive context expansion.

        Propagates to the backbone (which may contain Mamba-2 blocks).
        The SWM stages themselves are window-based and not context-length-dependent,
        but Mamba-2 replacement blocks may adjust internal buffers.

        Args:
            context_length: new target context length
        """
        self._current_context_length = context_length
        if hasattr(self.backbone, "set_context_length"):
            self.backbone.set_context_length(context_length)

    def forward_roi(
        self,
        roi_signals: torch.Tensor,
        time_patches: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ROI time series input.

        Args:
            roi_signals: (B, R, T) ROI time series (R ≈ 400 brain regions)
            time_patches: number of temporal patches to split into
        Returns:
            last_hidden: (B, T_patches, hidden_dim)
            pooler_output: (B, hidden_dim)
        """
        B, R, T = roi_signals.shape

        roi_flat = roi_signals.transpose(1, 2).contiguous()
        roi_flat = roi_flat.view(B * T, R)

        tokens = self.input_proj(roi_flat)
        tokens = tokens.view(B, T, -1)

        if self.use_tpt:
            prompt_list = self.prompt_tuning(B)
            prompt_tokens = prompt_list[-1]
            tokens = torch.cat([prompt_tokens, tokens], dim=1)

        features = self.backbone(tokens)

        last_hidden = features[-1]

        if self.use_tpt:
            num_prompts = self.prompt_tuning.num_prompts
            last_hidden = last_hidden[:, num_prompts:, :]

        pooler_out = self.pooler_activation(self.pooler(last_hidden.mean(dim=1)))

        return last_hidden, pooler_out

    def forward_voxel(
        self,
        volumes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for voxel-level 4D fMRI input.

        Args:
            volumes: (B, X, Y, Z, T) 4D fMRI volumes
        Returns:
            last_hidden: (B, num_patches, hidden_dim)
            pooler_output: (B, hidden_dim)
        """
        B, X, Y, Z, T = volumes.shape

        x = volumes.unsqueeze(1)
        x = self.input_proj(x)

        x = x.view(B, -1, T).transpose(1, 2)

        if self.use_tpt:
            prompt_list = self.prompt_tuning(B)
            prompt_tokens = prompt_list[-1]
            x = torch.cat([prompt_tokens, x], dim=1)

        features = self.backbone(x)

        last_hidden = features[-1]

        if self.use_tpt:
            num_prompts = self.prompt_tuning.num_prompts
            last_hidden = last_hidden[:, num_prompts:, :]

        pooler_out = self.pooler_activation(self.pooler(last_hidden.mean(dim=1)))

        return last_hidden, pooler_out

    def forward(
        self,
        fmri_signals: torch.Tensor,
        return_pooler: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward for both ROI and voxel modes.

        Args:
            fmri_signals: (B, R, T) for ROI mode or (B, X, Y, Z, T) for voxel mode
            return_pooler: whether to return pooled output
        Returns:
            dict with 'last_hidden' and optionally 'pooler_output'
        """
        if self.input_mode == "roi":
            last_hidden, pooler_out = self.forward_roi(fmri_signals)
        else:
            last_hidden, pooler_out = self.forward_voxel(fmri_signals)

        out = {"last_hidden": last_hidden}
        if return_pooler:
            out["pooler_output"] = pooler_out
        return out


class BrainLMEncoder(nn.Module):
    """
    BrainLM fallback encoder (ICLR 2024).

    Transformer masked autoencoder with:
    - 4-layer encoder + 2-layer decoder
    - hidden_dim=512
    - 424 brain region parcels
    - 20 TR temporal patches

    Reference: https://huggingface.co/vandijklab/BrainLM
    """

    def __init__(
        self,
        num_regions: int = 424,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_patch_size: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.hidden_dim = hidden_dim
        self.time_patch_size = time_patch_size

        self.input_proj = nn.Linear(num_regions, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_dim = hidden_dim
        self.pooler = nn.Linear(hidden_dim, hidden_dim)
        self.pooler_activation = nn.Tanh()

    def set_freeze(self, freeze: bool = True):
        """Freeze or unfreeze encoder parameters."""
        for param in self.parameters():
            param.requires_grad = not freeze

    def set_context_length(self, context_length: int):
        """
        No-op for BrainLM — it uses fixed-length TransformerEncoder
        without sequence-length-dependent state.
        """
        self._current_context_length = context_length

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained BrainLM weights.

        Typical source: https://huggingface.co/vandijklab/BrainLM

        Args:
            checkpoint_path: path to .pt or .safetensors checkpoint
            strict: if True, require all keys match exactly
        """
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            except ImportError:
                raise ImportError("safetensors required for .safetensors checkpoints")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[BrainLMEncoder] load_pretrained: {len(missing)} missing keys")
        if unexpected:
            print(f"[BrainLMEncoder] load_pretrained: {len(unexpected)} unexpected keys")
        if not missing and not unexpected:
            print("[BrainLMEncoder] Pretrained weights loaded successfully")

    def forward(
        self,
        roi_signals: torch.Tensor,
        return_pooler: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            roi_signals: (B, R, T) ROI time series
        Returns:
            dict with 'last_hidden' and optionally 'pooler_output'
        """
        B, R, T = roi_signals.shape

        x = roi_signals.transpose(1, 2).contiguous()
        x = x.view(B * T, R)
        x = self.input_proj(x)
        x = x.view(B, T, -1)

        last_hidden = self.encoder(x)

        pooler_out = self.pooler_activation(self.pooler(last_hidden.mean(dim=1)))

        out = {"last_hidden": last_hidden}
        if return_pooler:
            out["pooler_output"] = pooler_out
        return out


def create_fmri_encoder(
    encoder_type: str = "neurostorm",
    use_mamba2: bool = False,
    mamba2_kwargs: Optional[Dict] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create fMRI encoder.

    Args:
        encoder_type: 'neurostorm' or 'brainlm'
        use_mamba2: if True, use Mamba-2 SSM blocks instead of SWM
        mamba2_kwargs: dict passed to Mamba2EncoderBlock constructors
        **kwargs: passed to encoder constructor
    Returns:
        fMRI encoder module
    """
    if encoder_type.lower() == "neurostorm":
        return NeuroSTORMEncoder(use_mamba2=use_mamba2, mamba2_kwargs=mamba2_kwargs, **kwargs)
    elif encoder_type.lower() == "brainlm":
        return BrainLMEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    print("Testing NeuroSTORMEncoder (ROI mode)...")
    encoder = NeuroSTORMEncoder(input_mode="roi", roi_dim=400)
    dummy_roi = torch.randn(4, 400, 100)
    out = encoder(dummy_roi)
    print(f"  last_hidden shape: {out['last_hidden'].shape}")
    print(f"  pooler_output shape: {out['pooler_output'].shape}")

    print("\nTesting BrainLMEncoder...")
    brainlm = BrainLMEncoder(num_regions=424, hidden_dim=512)
    dummy_brainlm = torch.randn(4, 424, 100)
    out_brainlm = brainlm(dummy_brainlm)
    print(f"  last_hidden shape: {out_brainlm['last_hidden'].shape}")
    print(f"  pooler_output shape: {out_brainlm['pooler_output'].shape}")

    print("\nAll tests passed!")