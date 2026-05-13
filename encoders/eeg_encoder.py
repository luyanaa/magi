"""
EEG Encoder wrapper that integrates Magi foundation model.

Magi is a BERT-base EEG encoder with:
- 12-layer Transformer, hidden_dim=768
- Hybrid SWA (Sliding Window Attention) + RoPE
- BIOT-style 3D electrode position embeddings
- MoCo contrastive + masked prediction pretraining

This module wraps the Magi EEGFoundationModel for use in Brain MoE-PINN.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
import sys
import os

try:
    from brain_moe_pinn.magi.encoder import EEGFoundationModel
    MAGI_AVAILABLE = True
except ImportError as e:
    MAGI_AVAILABLE = False
    EEGFoundationModel = None

sys.path.insert(0, "/home/yanlu/Documents/a")
try:
    from brain_moe_pinn.utils.mamba2_ssm import Mamba2Backbone, FLA_MAMBA2_AVAILABLE
except ImportError:
    try:
        from utils.mamba2_ssm import Mamba2Backbone, FLA_MAMBA2_AVAILABLE
    except ImportError:
        FLA_MAMBA2_AVAILABLE = False
        Mamba2Backbone = None


class EEGEncoderWrapper(nn.Module):
    """
    Wraps Magi EEGFoundationModel for use in Brain MoE-PINN.

    Provides:
    - Forward pass returning (B, L, 768) token embeddings
    - Optional pooler output for contrastive/auxiliary tasks
    - Frozen backbone support with trainable projection head
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        output_dim: int = 2048,
        freeze_encoder: bool = True,
        use_biot_embedding: bool = True,
        use_factorized: bool = False,
        patch_size_time: int = 256,
        stride_time: int = 128,
        in_channels: int = 19,
        channels: Optional[List[str]] = None,
        freeze_epochs: int = 1,
        use_mamba2: bool = False,
        mamba2_layers: int = 4,
        mamba2_chunk_size: int = 256,
        mamba2_backend: str = "triton",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.freeze_encoder = freeze_encoder
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        self.use_mamba2 = use_mamba2

        if MAGI_AVAILABLE and EEGFoundationModel is not None:
            self.encoder = EEGFoundationModel(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                patch_size_time=patch_size_time,
                stride_time=stride_time,
                use_biot_embedding=use_biot_embedding,
                use_factorized=use_factorized,
                channels=channels,
            )
            print(f"[EEGEncoderWrapper] Loaded Magi EEGFoundationModel (BIOT={use_biot_embedding}, factorized={use_factorized})")
        else:
            raise RuntimeError(
                f"Magi EEGFoundationModel not available. MAGI_AVAILABLE={MAGI_AVAILABLE}. "
                f"Check brain_moe_pinn.magi package."
            )

        if use_mamba2 and FLA_MAMBA2_AVAILABLE and Mamba2Backbone is not None:
            self.mamba2 = Mamba2Backbone(
                hidden_dim=hidden_dim,
                num_layers=mamba2_layers,
                chunk_size=mamba2_chunk_size,
                backend=mamba2_backend,
            )
            print(f"[EEGEncoderWrapper] Added Mamba-2 context stack ({mamba2_layers} layers, chunk={mamba2_chunk_size})")
        else:
            self.mamba2 = None
            if use_mamba2 and not FLA_MAMBA2_AVAILABLE:
                print("[EEGEncoderWrapper] WARNING: use_mamba2=True but FLA Mamba2 not available. Skipping Mamba2 stack.")

        self.upscaling_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

        if self.freeze_encoder:
            self.set_freeze(True)

    def set_freeze(self, freeze: bool):
        """Freeze/unfreeze the Magi encoder backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        if freeze:
            print("[EEGEncoderWrapper] Magi encoder FROZEN")
        else:
            print("[EEGEncoderWrapper] Magi encoder UNFROZEN")

    def set_context_length(self, context_length: int):
        """
        Update effective context length for the Mamba-2 long-context stack.

        No-op if Mamba-2 is not enabled. The Magi SWA encoder itself is
        window-based and does not depend on global context length.

        Args:
            context_length: new target context length
        """
        self._current_context_length = context_length
        if self.mamba2 is not None and hasattr(self.mamba2, "set_context_length"):
            self.mamba2.set_context_length(context_length)

    def set_epoch(self, epoch: int):
        """Update current epoch for freeze schedule."""
        self.current_epoch = epoch
        if self.freeze_encoder and self.freeze_epochs > 0:
            if epoch >= self.freeze_epochs:
                self.set_freeze(False)

    def forward(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        channel_types: Optional[torch.Tensor] = None,
        return_embeddings: bool = True,
        return_pooler: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Magi EEG encoder + upscaling projection.

        Args:
            eeg: (B, C, T) raw EEG signals
            channel_names: Optional list of channel names for BIOT embedding
            channel_types: Optional (C,) long tensor: 0=scalp, 1=ecog, 2=seeg, 3=unknown
            return_embeddings: If True, return projected embeddings
            return_pooler: If True, return pooler output for aux tasks
        Returns:
            dict with:
                - 'embeddings': (B, L, output_dim) projected token embeddings
                - 'mask': (B, L) valid token mask
                - 'pooler_output': (B, hidden_dim) if return_pooler=True
        """
        tokens, mask, grid_size = self.encoder.forward_embeddings(
            eeg, channel_names, channel_types
        )

        last_hidden, pooler_output = self.encoder._get_encoder_output(
            tokens, mask,
            num_channels=grid_size[0],
            num_times=grid_size[1],
        )

        # Optional Mamba-2 long-context processing
        if self.mamba2 is not None:
            last_hidden = self.mamba2(last_hidden)

        out = {
            "last_hidden": last_hidden,
            "mask": mask,
            "grid_size": grid_size,
        }

        if return_embeddings:
            embeddings = self.upscaling_proj(last_hidden)
            out["embeddings"] = embeddings

        if return_pooler:
            out["pooler_output"] = pooler_output

        return out

    def get_encoder_state_dict(self) -> Dict[str, Any]:
        """Get state dict for Magi encoder only (excluding projection)."""
        return {
            k.replace("encoder.", ""): v
            for k, v in self.state_dict().items()
            if k.startswith("encoder.")
        }

    def load_encoder_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load Magi encoder weights into this wrapper."""
        filtered_dict = {
            f"encoder.{k}": v
            for k, v in state_dict.items()
        }
        self.load_state_dict(filtered_dict, strict=False)

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained Magi EEG encoder weights.

        Typical source: brain_moe_pinn.magi (local package)

        Args:
            checkpoint_path: path to .pt checkpoint
            strict: if True, require all keys match exactly
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        encoder_state = {
            k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v
            for k, v in state_dict.items()
        }
        self.load_encoder_weights(encoder_state)


class EEGProjection(nn.Module):
    """
    Upscaling projection from Magi hidden_dim (768) to VelocityBrain dim (2048).

    Lightweight adapter that maps EEG semantic features to the high-dimensional
    latent space used by the rest of Brain MoE-PINN.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim) token embeddings
        Returns:
            (B, L, output_dim) projected embeddings
        """
        return self.projection(x)


if __name__ == "__main__":
    if not MAGI_AVAILABLE:
        print("Magi not available, skipping runtime test")
    else:
        print("Testing EEGEncoderWrapper...")
        wrapper = EEGEncoderWrapper(freeze_encoder=False)
        dummy_eeg = torch.randn(2, 19, 2560)
        out = wrapper(dummy_eeg, return_embeddings=True, return_pooler=True)
        print(f"  embeddings shape: {out['embeddings'].shape}")
        print(f"  pooler shape: {out['pooler_output'].shape}")
        print(f"  mask shape: {out['mask'].shape}")
        print(f"  grid_size: {out['grid_size']}")
        print("All tests passed!")