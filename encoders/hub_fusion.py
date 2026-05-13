"""
Hub Token Modality Fusion.

Cross-attention based fusion of EEG and fMRI branches via learnable hub tokens.
Based on Brain Harmony (arXiv:2509.24693) hub token design.

Each modality branch has its own hub token that attends to the modality's
token sequence, then hub tokens across modalities interact via cross-attention
to produce a fused global representation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class HubTokenFusion(nn.Module):
    """
    Hub token based modality fusion for dual-branch EEG + fMRI encoding.

    Two learnable hub tokens (one per modality) interact via cross-attention
    to produce a unified latent representation z_global.

    Design:
    - hub_eeg: encodes EEG modality identity + extracts EEG global features
    - hub_fmri: encodes fMRI modality identity + extracts fMRI global features
    - cross_attention: hub tokens attend to each other to align representations
    - output: fused z_global (B, d) for downstream MoE velocity field
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.hub_eeg = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.hub_fmri = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.eeg_to_hub_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fmri_to_hub_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.hub_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.hub_norm_eeg = nn.LayerNorm(hidden_dim)
        self.hub_norm_fmri = nn.LayerNorm(hidden_dim)
        self.hub_norm_cross = nn.LayerNorm(hidden_dim)

        self.gate_eeg = nn.Parameter(torch.ones(hidden_dim) * 0.5) if use_gate else None
        self.gate_fmri = nn.Parameter(torch.ones(hidden_dim) * 0.5) if use_gate else None

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        eeg_tokens: torch.Tensor,
        fmri_tokens: torch.Tensor,
        mask_eeg: Optional[torch.Tensor] = None,
        mask_fmri: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse EEG and fMRI branch outputs via hub tokens.

        Args:
            eeg_tokens: (B, L_eeg, d) EEG branch token sequence
            fmri_tokens: (B, L_fmri, d) fMRI branch token sequence
            mask_eeg: (B, L_eeg) optional padding mask for EEG tokens
            mask_fmri: (B, L_fmri) optional padding mask for fMRI tokens
        Returns:
            z_global: (B, d) fused global latent state
            hub_eeg_out: (B, d) EEG hub token output
            hub_fmri_out: (B, d) fMRI hub token output
        """
        B = eeg_tokens.shape[0]
        device = eeg_tokens.device

        hub_eeg = self.hub_eeg.expand(B, -1, -1)
        hub_fmri = self.hub_fmri.expand(B, -1, -1)

        hub_eeg = self.hub_norm_eeg(
            hub_eeg + self.eeg_to_hub_attn(
                hub_eeg, eeg_tokens, eeg_tokens,
                key_padding_mask=mask_eeg,
            )[0]
        )

        hub_fmri = self.hub_norm_fmri(
            hub_fmri + self.fmri_to_hub_attn(
                hub_fmri, fmri_tokens, fmri_tokens,
                key_padding_mask=mask_fmri,
            )[0]
        )

        hub_cross, _ = self.hub_cross_attn(
            hub_eeg, hub_fmri, hub_fmri,
        )
        hub_eeg = self.hub_norm_cross(hub_eeg + hub_cross)

        hub_cross, _ = self.hub_cross_attn(
            hub_fmri, hub_eeg, hub_eeg,
        )
        hub_fmri = self.hub_norm_cross(hub_fmri + hub_cross)

        hub_eeg_out = hub_eeg.squeeze(1)
        hub_fmri_out = hub_fmri.squeeze(1)

        if self.gate_eeg is not None:
            hub_eeg_out = hub_eeg_out * torch.sigmoid(self.gate_eeg)
            hub_fmri_out = hub_fmri_out * torch.sigmoid(self.gate_fmri)

        z_global = self.fusion_mlp(torch.cat([hub_eeg_out, hub_fmri_out], dim=-1))

        return z_global, hub_eeg_out, hub_fmri_out


class CrossModalAdapter(nn.Module):
    """
    Lightweight cross-modal adapter for aligning EEG and fMRI representations.

    Applied after upscaling projections before hub token fusion.
    Helps bridge the gap between EEG (low spatial resolution, high temporal)
    and fMRI (high spatial resolution, low temporal) characteristics.
    """

    def __init__(
        self,
        dim: int = 2048,
        reduction: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // reduction, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)


if __name__ == "__main__":
    print("Testing HubTokenFusion...")
    fusion = HubTokenFusion(hidden_dim=2048, num_heads=8)
    eeg_tokens = torch.randn(4, 100, 2048)
    fmri_tokens = torch.randn(4, 50, 2048)
    z_global, hub_eeg, hub_fmri = fusion(eeg_tokens, fmri_tokens)
    print(f"  z_global shape: {z_global.shape}")
    print(f"  hub_eeg shape: {hub_eeg.shape}")
    print(f"  hub_fmri shape: {hub_fmri.shape}")
    print("All tests passed!")