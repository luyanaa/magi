"""
Hub Token Modality Fusion with Slow Manifold Bridge.

Cross-attention based fusion of EEG and fMRI branches via learnable hub tokens.
Based on Brain Harmony (arXiv:2509.24693) hub token design.

Also contains:
- SlowManifoldProjector: W_slow (256×1024) extracts <0.25Hz shared subspace
- LatentHRFBridge: Physics-structured learned HRF in latent slow manifold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class HubTokenFusion(nn.Module):
    """
    Hub token based modality fusion for multi-branch encoding.

    N learnable hub tokens (one per modality) interact via cross-attention
    to produce a unified latent representation z_global.

    Design:
    - hub_tokens[i]: encodes modality i identity + extracts global features
    - modal_attns[i]: hub token i attends to modality i token sequence
    - cross_attention: all hub tokens attend to each other pairwise
    - output: fused z_global (B, d) for downstream MoE velocity field
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 2,
        modality_names: Optional[List[str]] = None,
        use_gate: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.modality_names = modality_names or (["eeg", "fmri", "meg"][:num_modalities])

        # One hub token per modality
        self.hub_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            for _ in range(num_modalities)
        ])

        # Per-modality attention: hub -> tokens
        self.modal_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_modalities)
        ])

        # Shared cross-attention for hub-hub interaction
        self.hub_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.hub_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_modalities)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_modalities)
        ])

        self.gates = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim) * 0.5) if use_gate else None
            for _ in range(num_modalities)
        ]) if use_gate else [None] * num_modalities

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        *modal_tokens: torch.Tensor,
        masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse modality branch outputs via hub tokens.

        Args:
            *modal_tokens: variable-length tuple of (B, L_i, d) token sequences
            masks: optional list of (B, L_i) padding masks, one per modality
        Returns:
            z_global: (B, d) fused global latent state
            hub_outputs: dict mapping modality name -> (B, d) hub token output
        """
        assert len(modal_tokens) == self.num_modalities, (
            f"Expected {self.num_modalities} modalities, got {len(modal_tokens)}"
        )
        B = modal_tokens[0].shape[0]
        device = modal_tokens[0].device
        masks = masks or [None] * self.num_modalities

        hubs = []
        for i, tokens in enumerate(modal_tokens):
            hub = self.hub_tokens[i].expand(B, -1, -1)
            hub = self.hub_norms[i](
                hub + self.modal_attns[i](
                    hub, tokens, tokens,
                    key_padding_mask=masks[i],
                )[0]
            )
            hubs.append(hub)

        # Pairwise cross-attention between all hub tokens
        # Note: sequential update order means hub[0] sees hub[2] through hub[1]'s lens.
        # With 2 modalities this is symmetric; with 3+ it introduces order dependence.
        # A full simultaneous update (all-pairs in parallel) would require additional memory.
        for i in range(self.num_modalities):
            for j in range(self.num_modalities):
                if i != j:
                    cross, _ = self.hub_cross_attn(hubs[i], hubs[j], hubs[j])
                    hubs[i] = self.cross_norms[i](hubs[i] + cross)

        hub_outs = [h.squeeze(1) for h in hubs]
        if self.gates[0] is not None:
            hub_outs = [h * torch.sigmoid(g) for h, g in zip(hub_outs, self.gates)]

        z_global = self.fusion_mlp(torch.cat(hub_outs, dim=-1))
        hub_dict = {name: out for name, out in zip(self.modality_names, hub_outs)}
        return z_global, hub_dict


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


class SlowManifoldProjector(nn.Module):
    """
    Extracts the <0.25 Hz slow manifold from the 1024-dim latent space.

    W_slow: 256×1024 learned projection (shared across modalities).
    The slow manifold captures ultra-slow dynamics shared between EEG and fMRI.
    Fast EEG dynamics (alpha/gamma/ERP) remain in the residual space.

    Training: W_slow is jointly trained by L_cross (Stage 2) and L_recon (all stages).
    """
    def __init__(self, latent_dim: int = 1024, slow_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.slow_dim = slow_dim
        # Single shared projection for both modalities
        self.W_slow = nn.Parameter(torch.randn(slow_dim, latent_dim) * 0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., 1024) → z_slow: (..., 256)"""
        return F.linear(z, self.W_slow)

    def inverse_project(self, z_slow: torch.Tensor) -> torch.Tensor:
        """Approximate inverse: z_slow: (..., 256) → z: (..., 1024)"""
        return F.linear(z_slow, self.W_slow.T)


class LatentHRFBridge(nn.Module):
    """
    Learned hemodynamic response function in latent slow manifold.

    Three physics-structured parameters:
    - delay: peak delay (~6s), learned
    - dispersion: response width (~1s), learned
    - undershoot: post-peak undershoot ratio (~0.3), learned

    Optional subject-conditioning via age/sex → HRF parameter offsets.
    Convolution is applied in the 256-dim slow manifold.

    Trainable with 341 synchronized subjects (236M slow-manifold datapoints).
    Physics structure makes this feasible — unstructured requires 18B+ tokens.
    """
    def __init__(
        self,
        slow_dim: int = 256,
        hr_length: int = 32,  # 32 TR samples = 64s window
        use_subject_conditioning: bool = True,
    ):
        super().__init__()
        self.slow_dim = slow_dim
        self.hr_length = hr_length

        # Canonical HRF parameters (learnable)
        self.delay = nn.Parameter(torch.tensor(6.0))       # peak delay (seconds)
        self.dispersion = nn.Parameter(torch.tensor(1.0))   # width
        self.undershoot = nn.Parameter(torch.tensor(0.3))   # undershoot ratio

        # Optional subject conditioning
        self.use_subject_cond = use_subject_conditioning
        if use_subject_conditioning:
            self.subject_adapter = nn.Sequential(
                nn.Linear(3, 64),   # age, sex, brain_region_embed
                nn.GELU(),
                nn.Linear(64, 3),   # Δdelay, Δdispersion, Δundershoot
            )

    def _build_hrf_kernel(self, dt: float = 0.5, subject_features=None):
        """Build canonical HRF kernel with current parameters."""
        # Get base parameters
        delay = self.delay
        dispersion = self.dispersion
        undershoot = self.undershoot

        # Apply subject conditioning
        if self.use_subject_cond and subject_features is not None:
            delta = torch.tanh(self.subject_adapter(subject_features))  # (B, 3)
            delay = delay + delta[:, 0].mean()
            dispersion = dispersion + delta[:, 1].mean()
            undershoot = undershoot + delta[:, 2].mean()

        # Build double-gamma HRF kernel
        t = torch.arange(0, self.hr_length * dt, dt, device=self.delay.device)
        # Ensure t doesn't go negative
        t_safe = torch.clamp(t - delay, min=1e-6)

        peak = (t_safe / dispersion) ** 2 * torch.exp(-t_safe / dispersion)
        undershoot_component = undershoot * torch.exp(-t_safe / (dispersion * 3))
        hrf = peak - undershoot_component

        # Normalize
        hrf = hrf / (hrf.sum() + 1e-8)
        return hrf

    def forward(
        self,
        z_eeg_slow: torch.Tensor,
        z_fmri_slow: Optional[torch.Tensor] = None,
        subject_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply HRF convolution in latent slow manifold.

        Args:
            z_eeg_slow: (B, T_eeg, slow_dim) EEG slow manifold trajectory
            z_fmri_slow: (B, T_fmri, slow_dim) optional, for alignment loss
            subject_features: (B, 3) optional subject features

        Returns:
            z_predicted_fmri: (B, T_out, slow_dim)
            alignment_loss: scalar or None
        """
        B, T, D = z_eeg_slow.shape
        device = z_eeg_slow.device

        # Build HRF kernel
        hrf = self._build_hrf_kernel(dt=2.0, subject_features=subject_features)  # TR=2s
        hrf = hrf.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, K)

        # Downsample EEG slow manifold to fMRI TR rate
        eeg_ds = F.avg_pool1d(
            z_eeg_slow.transpose(1, 2), kernel_size=4, stride=4
        ).transpose(1, 2)  # (B, T_fmri, D)

        # Apply HRF convolution per dimension (depthwise: groups=D)
        # Weight shape for F.conv1d(groups=D): (D, 1, K)
        eeg_ds = eeg_ds.transpose(1, 2)  # (B, D, T)
        hrf_weight = hrf.expand(D, -1, -1)  # (D, 1, K)
        z_pred = F.conv1d(eeg_ds, hrf_weight, padding=hrf.shape[2], groups=D)
        z_pred = z_pred[:, :, :eeg_ds.shape[2]]  # Trim to match
        z_pred = z_pred.transpose(1, 2)  # (B, T, D)

        alignment_loss = None
        if z_fmri_slow is not None:
            T_pred = z_pred.shape[1]
            T_fmri = z_fmri_slow.shape[1]
            min_T = min(T_pred, T_fmri)
            alignment_loss = F.mse_loss(z_pred[:, :min_T, :], z_fmri_slow[:, :min_T, :])

        return z_pred, alignment_loss


if __name__ == "__main__":
    print("Testing HubTokenFusion (2 modalities)...")
    fusion = HubTokenFusion(hidden_dim=1024, num_heads=8, num_modalities=2)
    eeg_tokens = torch.randn(4, 100, 1024)
    fmri_tokens = torch.randn(4, 50, 1024)
    z_global, hubs = fusion(eeg_tokens, fmri_tokens)
    print(f"  z_global shape: {z_global.shape}")
    print(f"  hub_eeg shape: {hubs['eeg'].shape}")
    print(f"  hub_fmri shape: {hubs['fmri'].shape}")

    print("\nTesting HubTokenFusion (3 modalities)...")
    fusion3 = HubTokenFusion(hidden_dim=1024, num_heads=8, num_modalities=3)
    meg_tokens = torch.randn(4, 80, 1024)
    z_global3, hubs3 = fusion3(eeg_tokens, fmri_tokens, meg_tokens)
    print(f"  z_global shape: {z_global3.shape}")
    print(f"  hub_meg shape: {hubs3['meg'].shape}")
    print("All tests passed!")