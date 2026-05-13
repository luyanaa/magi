"""
Dual-modality Decoder for EEG and fMRI signal reconstruction.

Generates modality-specific outputs from the latent state z_{t+1}.
The decoder is asymmetric from the encoder (not a mirror architecture).
Outputs serve as:
- Efference Copy for closed-loop feedback
- Reconstruction loss targets
- Optional: future trajectory prediction
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class EEGDecoder(nn.Module):
    """
    Lightweight EEG decoder from latent state.

    Takes z_{t+1} and projects back to EEG signal space.
    Uses inverse-style architecture (transposed convolutions or linear layers)
    rather than mirroring the Magi encoder.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        hidden_dim: int = 768,
        output_channels: int = 19,
        patch_size_time: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.patch_size_time = patch_size_time

        layers = []
        prev_dim = latent_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(prev_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = next_dim

        self.decoder = nn.Sequential(*layers)

        self.channel_proj = nn.Linear(hidden_dim, output_channels * patch_size_time)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to EEG signal.

        Args:
            z: (B, latent_dim) latent state at t+1
        Returns:
            (B, C, T) reconstructed EEG signal
        """
        B = z.shape[0]
        h = self.decoder(z)
        out = self.channel_proj(h)
        out = out.view(B, self.output_channels, self.patch_size_time)
        return out


class fMRIDecoder(nn.Module):
    """
    fMRI decoder from latent state.

    Takes z_{t+1} and projects to fMRI signal space (ROI time series or voxel space).
    Simpler architecture than EEG decoder since fMRI temporal resolution is lower.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        hidden_dim: int = 768,
        num_regions: int = 400,
        time_patches: int = 20,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.time_patches = time_patches

        layers = []
        prev_dim = latent_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(prev_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = next_dim

        self.decoder = nn.Sequential(*layers)

        self.region_proj = nn.Linear(hidden_dim, num_regions * time_patches)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to fMRI signal.

        Args:
            z: (B, latent_dim) latent state at t+1
        Returns:
            (B, R, T) reconstructed fMRI ROI time series
        """
        B = z.shape[0]
        h = self.decoder(z)
        out = self.region_proj(h)
        out = out.view(B, self.num_regions, self.time_patches)
        return out


class ModalityDecoderRouter(nn.Module):
    """
    Routes latent state to correct modality decoder based on Hub Token identity.

    Hub tokens carry modality identity information that determines which
    decoder to activate. This allows a single latent state to produce
    modality-specific outputs.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        eeg_hidden: int = 768,
        fmri_hidden: int = 768,
        output_channels: int = 19,
        num_regions: int = 400,
        patch_size_time: int = 256,
        time_patches: int = 20,
    ):
        super().__init__()
        self.eeg_decoder = EEGDecoder(
            latent_dim=latent_dim,
            hidden_dim=eeg_hidden,
            output_channels=output_channels,
            patch_size_time=patch_size_time,
        )
        self.fmri_decoder = fMRIDecoder(
            latent_dim=latent_dim,
            hidden_dim=fmri_hidden,
            num_regions=num_regions,
            time_patches=time_patches,
        )

        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        z: torch.Tensor,
        hub_eeg: torch.Tensor,
        hub_fmri: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate modality-specific decodings from shared latent state.

        Args:
            z: (B, latent_dim) latent state at t+1
            hub_eeg: (B, latent_dim) EEG hub token
            hub_fmri: (B, latent_dim) fMRI hub token
        Returns:
            dict with 'eeg_recon' and 'fmri_recon'
        """
        eeg_out = self.eeg_decoder(z)
        fmri_out = self.fmri_decoder(z)

        gate_weights = self.gate(torch.cat([hub_eeg, hub_fmri], dim=-1))

        weighted_eeg = eeg_out * gate_weights[:, 0:1].unsqueeze(-1)
        weighted_fmri = fmri_out * gate_weights[:, 1:2].unsqueeze(-1)

        return {
            "eeg_recon": eeg_out,
            "fmri_recon": fmri_out,
            "weighted_eeg": weighted_eeg,
            "weighted_fmri": weighted_fmri,
            "gate_weights": gate_weights,
        }


if __name__ == "__main__":
    print("Testing ModalityDecoderRouter...")
    router = ModalityDecoderRouter()
    z = torch.randn(4, 2048)
    hub_eeg = torch.randn(4, 2048)
    hub_fmri = torch.randn(4, 2048)
    out = router(z, hub_eeg, hub_fmri)
    print(f"  eeg_recon shape: {out['eeg_recon'].shape}")
    print(f"  fmri_recon shape: {out['fmri_recon'].shape}")
    print(f"  gate_weights shape: {out['gate_weights'].shape}")
    print("All tests passed!")