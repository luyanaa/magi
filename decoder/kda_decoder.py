"""
Kimi Delta Attention Decoder with Flash Linear Attention.

Based on Kimi Delta Attention (KDA) from arXiv:2510.26692
Implemented using the fla (flash-linear-attention) library.

KDA is used in decoders for temporal reconstruction.
The delta rule with per-key-dim gating enables efficient state-tracking
in linear complexity, ideal for autoregressive generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
import math
import warnings

import sys
try:
    import fla
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    KimiDeltaAttention = None
    print(f"[KDA Decoder] Warning: fla not available ({e}). Using PyTorch fallback.")


class KDAEncoderBlock(nn.Module):
    """
    Single KDA encoder block using the fla library.

    Combines KimiDeltaAttention with feed-forward network.
    Uses short convolutions for local context modeling.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        head_dim: int = 128,
        num_heads: Optional[int] = None,
        expand_v: float = 1.0,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        safe_gate: bool = False,
        lower_bound: float = -5.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_heads is None:
            num_heads = hidden_size // head_dim

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        if FLA_AVAILABLE and KimiDeltaAttention is not None:
            self.attention = KimiDeltaAttention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                expand_v=expand_v,
                mode=mode,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                layer_idx=0,
            )
        else:
            self.attention = self._create_fallback_attention(
                hidden_size, head_dim, num_heads, dropout
            )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def _create_fallback_attention(self, hidden_size, head_dim, num_heads, dropout):
        """Fallback attention if fla is not available."""
        return nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: (B, L, D) input sequence
            attention_mask: (B, L) padding mask
            past_key_values: cache for autoregressive decoding
            use_cache: whether to return cache
        Returns:
            output: (B, L, D)
            present: cache state if use_cache=True
        """
        residual = x
        x_normed = self.norm1(x)

        if FLA_AVAILABLE and isinstance(self.attention, KimiDeltaAttention):
            attn_out, _, present = self.attention(
                hidden_states=x_normed.float(),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if attn_out.dtype != residual.dtype:
                attn_out = attn_out.to(residual.dtype)
            x = residual + attn_out
        else:
            attn_out, _ = self.attention(x_normed, x_normed, x_normed, attn_mask=attention_mask)
            x = residual + attn_out
            present = None

        residual = x
        x = residual + self.mlp(self.norm2(x))

        present_dict = {"attn": present} if use_cache else None
        return x, present_dict


class LinearDecoderBlock(nn.Module):
    """
    Simplified decoder block without self-attention.

    Uses cross-attention to fuse latent context with the generated
    sequence, followed by MLP channel mixing. No self-attention
    because all tokens start as identical copies of the latent
    state — self-attention between identical vectors is a no-op.

    Corresponds to the non-KDA layers in the Kimi Linear hybrid
    architecture (75% of layers, interleaved with 25% KDA layers).
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        target_dim: int = 4096,
        head_dim: int = 128,
        num_heads: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_dim = target_dim

        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.output_proj = nn.Linear(hidden_size, target_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: (B, L, D) target sequence
            context: (B, L_ctx, D) latent context for cross-attention
            attention_mask: (B, L) padding mask
            past_key_values: not used (no self-attn state)
            use_cache: not used (no self-attn state)
        Returns:
            output: (B, L, target_dim)
            present: None (no recurrent state)
        """
        if context is not None:
            residual = x
            x_normed = self.norm1(x)
            cross_out, _ = self.cross_attn(x_normed, context, context)
            x = residual + cross_out

        residual = x
        x = residual + self.mlp(self.norm2(x))

        output = self.output_proj(x)
        return output, None


class KDADecoderBlock(nn.Module):
    """
    KDA Decoder block with cross-attention for modality fusion.

    Used for decoding latent state z_{t+1} into EEG/fMRI signals.
    The cross-attention mechanism attends to the latent state
    while KDA handles temporal dynamics.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        target_dim: int = 768,
        head_dim: int = 128,
        num_heads: Optional[int] = None,
        expand_v: float = 1.0,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        safe_gate: bool = False,
        lower_bound: float = -5.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_heads is None:
            num_heads = hidden_size // head_dim

        self.hidden_size = hidden_size
        self.target_dim = target_dim
        self.head_dim = head_dim
        self.num_heads = num_heads

        if FLA_AVAILABLE and KimiDeltaAttention is not None:
            self.self_attn = KimiDeltaAttention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                expand_v=expand_v,
                mode=mode,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                layer_idx=0,
            )
        else:
            self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True)

        # Cross-attention uses standard MultiheadAttention (KimiDeltaAttention is self-attention only)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.output_proj = nn.Linear(hidden_size, target_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: (B, L, D) target sequence (e.g., temporal decoder input)
            context: (B, L_ctx, D) context for cross-attention (e.g., z_next)
            attention_mask: (B, L) padding mask
            past_key_values: cache for autoregressive decoding
            use_cache: whether to return cache
        Returns:
            output: (B, L, target_dim)
            present: cache state if use_cache=True
        """
        residual = x
        x_normed = self.norm1(x)

        if FLA_AVAILABLE and isinstance(self.self_attn, KimiDeltaAttention):
            attn_out, _, present = self.self_attn(
                hidden_states=x_normed.float(),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            # Cast back to original dtype for selective FP32
            if attn_out.dtype != residual.dtype:
                attn_out = attn_out.to(residual.dtype)
            x = residual + attn_out
        else:
            attn_out, _ = self.self_attn(x_normed, x_normed, x_normed, attn_mask=attention_mask)
            x = residual + attn_out
            present = None

        if context is not None:
            residual = x
            x_normed = self.norm2(x)

            cross_out, _ = self.cross_attn(x_normed, context, context)
            x = residual + cross_out

        residual = x
        x = residual + self.mlp(self.norm3(x))

        output = self.output_proj(x)

        return output, present if use_cache else None


class KDATemporalDecoder(nn.Module):
    """
    Temporal decoder with 25% KDA + 75% linear layers (Kimi-Linear style).

    Follows the Kimi Linear paper (arXiv:2510.26692) hybrid architecture:
    - 1 KDA layer (25%): Kimi Delta Attention for expressive temporal modeling
    - 3 linear/MLP layers (75%): cross-attention + MLP, no self-attention

    The KDA layer provides positional awareness (the paper delegates
    positional encoding entirely to KDA, using NoPE for full-attention
    layers). Linear layers handle cross-modal fusion with the latent
    context via standard MultiheadAttention.

    Architecture:
        z -> expand -> [linear x3 | KDA x1] -> temporal projection -> signal
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        output_dim: int = 768,
        num_layers: int = 4,
        kda_ratio: float = 0.25,
        head_dim: int = 128,
        expand_v: float = 1.0,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        safe_gate: bool = False,
        lower_bound: float = -5.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # KDA goes in the last layer (position-aware operator, per Kimi Linear design)
        self.kda_layer_idx = num_layers - 1  # layer 3 of 4

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 2),
        )

        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == self.kda_layer_idx:
                self.decoder_blocks.append(
                    KDADecoderBlock(
                        hidden_size=latent_dim * 2,
                        target_dim=latent_dim * 2,
                        head_dim=head_dim,
                        expand_v=expand_v,
                        mode=mode,
                        use_short_conv=use_short_conv,
                        conv_size=conv_size,
                        safe_gate=safe_gate,
                        lower_bound=lower_bound,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )
            else:
                self.decoder_blocks.append(
                    LinearDecoderBlock(
                        hidden_size=latent_dim * 2,
                        target_dim=latent_dim * 2,
                        head_dim=head_dim,
                        num_heads=(latent_dim * 2) // head_dim,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )

        self.temporal_proj = nn.Linear(latent_dim * 2, output_dim)

    def forward(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        seq_length: int = 100,
        use_cache: bool = False,
        past_key_values: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """
        Decode latent state into temporal sequence.

        Supports two modes:
        - Batch expansion (use_cache=False): expand z to (B, seq_length, D)
          and process all steps in parallel.
        - Autoregressive (use_cache=True): generate token-by-token with
          KDA recurrent state caching. The last KDA layer uses past_key_values
          for step-by-step state management.

        Args:
            z: (B, D) latent state at t+1
            context: (B, L_ctx, D) optional context for cross-attention
            seq_length: length of output sequence
            use_cache: enable autoregressive mode with state caching
            past_key_values: list of cache dicts for each block
        Returns:
            (B, seq_length, output_dim) predicted signal
            present: list of cache dicts if use_cache=True
        """
        B = z.shape[0]
        device = z.device

        x = self.input_proj(z)
        x = x.unsqueeze(1).expand(-1, seq_length, -1)

        presents = [] if use_cache else None
        causal_mask = None
        if use_cache:
            max_mask_len = 16384
            mask_len = min(seq_length, max_mask_len)
            causal_mask = torch.triu(
                torch.ones(mask_len, mask_len, dtype=torch.bool, device=device), diagonal=1
            )
            if seq_length > max_mask_len:
                warnings.warn(
                    f"seq_length={seq_length} exceeds causal mask limit {max_mask_len}. "
                    "Mask truncated; past_key_values beyond this limit use no mask."
                )

        for i, block in enumerate(self.decoder_blocks):
            block_pkv = past_key_values[i] if past_key_values is not None else None

            if isinstance(block, KDADecoderBlock):
                x, present = block(
                    x, context=context, attention_mask=causal_mask,
                    past_key_values=block_pkv, use_cache=use_cache
                )
            else:
                x, present = block(
                    x, context=context, attention_mask=causal_mask,
                    past_key_values=block_pkv, use_cache=use_cache
                )

            if use_cache:
                presents.append(present)

        output = self.temporal_proj(x)
        if use_cache:
            return output, presents
        return output


class EEGKDADecoder(nn.Module):
    """
    EEG-specific KDA decoder.

    Decodes z_{t+1} -> EEG signal (B, C, T).
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 768,
        output_channels: int = 19,
        num_layers: int = 4,
        head_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_channels = output_channels

        self.temporal_decoder = KDATemporalDecoder(
            latent_dim=latent_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.channel_proj = nn.Linear(hidden_dim, output_channels)

    def forward(self, z: torch.Tensor, seq_length: int = 19) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent state
            seq_length: number of temporal patches (= encoder patch count)
        Returns:
            (B, output_channels, seq_length) EEG signal at patch resolution
        """
        temporal_out = self.temporal_decoder(z, seq_length=seq_length)
        # temporal_out: (B, seq_length, hidden_dim)
        out = self.channel_proj(temporal_out)  # (B, seq_length, output_channels)
        out = out.transpose(1, 2)  # (B, output_channels, seq_length)
        return out


class fMRIKDADecoder(nn.Module):
    """
    fMRI-specific KDA decoder.

    Decodes z_{t+1} -> fMRI signal (B, R, T).
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 768,
        num_regions: int = 400,
        num_layers: int = 4,
        head_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_regions = num_regions

        self.temporal_decoder = KDATemporalDecoder(
            latent_dim=latent_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.region_proj = nn.Linear(hidden_dim, num_regions)

    def forward(self, z: torch.Tensor, seq_length: int = 100) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent state
            seq_length: output temporal length (in TRs)
        Returns:
            (B, num_regions, seq_length) fMRI signal
        """
        temporal_out = self.temporal_decoder(z, seq_length=seq_length)
        # temporal_out: (B, seq_length, hidden_dim)
        out = self.region_proj(temporal_out)  # (B, seq_length, num_regions)
        out = out.transpose(1, 2)  # (B, num_regions, seq_length)
        return out


class KIMIKDAMoDeCoderRouter(nn.Module):
    """
    Router for KDA-based modality decoders.

    Uses hub token information to route to correct decoder.
    Combines KDA temporal modeling with routing.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        eeg_hidden: int = 768,
        fmri_hidden: int = 768,
        meg_hidden: int = 768,
        output_channels: int = 19,
        num_regions: int = 400,
        meg_channels: int = 306,
        patch_size: int = 256,
        num_layers: int = 4,
        head_dim: int = 128,
        dropout: float = 0.1,
        use_meg: bool = False,
    ):
        super().__init__()
        self.use_meg = use_meg
        self.num_modalities = 3 if use_meg else 2

        self.eeg_decoder = EEGKDADecoder(
            latent_dim=latent_dim,
            hidden_dim=eeg_hidden,
            output_channels=output_channels,
            num_layers=num_layers,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.fmri_decoder = fMRIKDADecoder(
            latent_dim=latent_dim,
            hidden_dim=fmri_hidden,
            num_regions=num_regions,
            num_layers=num_layers,
            head_dim=head_dim,
            dropout=dropout,
        )

        if use_meg:
            self.meg_decoder = EEGKDADecoder(
                latent_dim=latent_dim,
                hidden_dim=meg_hidden,
                output_channels=meg_channels,
                num_layers=num_layers,
                head_dim=head_dim,
                dropout=dropout,
            )

        gate_in = latent_dim * self.num_modalities
        gate_out = self.num_modalities
        self.gate = nn.Sequential(
            nn.Linear(gate_in, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, gate_out),
            nn.Softmax(dim=-1),
        )

    def set_context_length(self, context_length: int):
        """
        Update the default context length for decoder output sequences.

        When the training loop expands context (4096 → 16384 → 65536),
        the decoders should produce correspondingly longer sequences.

        Args:
            context_length: new target context length
        """
        self._current_context_length = context_length

    def forward(
        self,
        z: torch.Tensor,
        hub_eeg: torch.Tensor,
        hub_fmri: torch.Tensor,
        hub_meg: Optional[torch.Tensor] = None,
        eeg_seq_length: int = 256,
        fmri_seq_length: int = 100,
        meg_seq_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate modality-specific decodings using KDA decoders.

        Args:
            z: (B, latent_dim) latent state at t+1
            hub_eeg: (B, latent_dim) EEG hub token
            hub_fmri: (B, latent_dim) fMRI hub token
            hub_meg: (B, latent_dim) optional MEG hub token
            eeg_seq_length: EEG output sequence length
            fmri_seq_length: fMRI output sequence length
            meg_seq_length: MEG output sequence length
        Returns:
            dict with 'eeg_recon', 'fmri_recon', 'gate_weights', optional 'meg_recon'
        """
        eeg_out = self.eeg_decoder(z, seq_length=eeg_seq_length)
        fmri_out = self.fmri_decoder(z, seq_length=fmri_seq_length)

        if self.use_meg and hub_meg is not None:
            gate_input = torch.cat([hub_eeg, hub_fmri, hub_meg], dim=-1)
        else:
            gate_input = torch.cat([hub_eeg, hub_fmri], dim=-1)
        gate_weights = self.gate(gate_input)

        result = {
            "eeg_recon": eeg_out,
            "fmri_recon": fmri_out,
            "gate_weights": gate_weights,
        }

        if self.use_meg and hub_meg is not None:
            meg_out = self.meg_decoder(z, seq_length=meg_seq_length)
            result["meg_recon"] = meg_out

        return result


if __name__ == "__main__":
    print("Testing KDA-based decoders...")

    if FLA_AVAILABLE:
        print("Using fla library KimiDeltaAttention")
    else:
        print("fla not available, using PyTorch fallback")

    print("\nTesting KDATemporalDecoder...")
    decoder = KDATemporalDecoder(latent_dim= 1024, output_dim=768, num_layers=4)
    z = torch.randn(2, 2048)
    out = decoder(z, seq_length=100)
    print(f"  output shape: {out.shape}")

    print("\nTesting EEGKDADecoder...")
    eeg_dec = EEGKDADecoder(latent_dim= 1024, output_channels=19)
    z = torch.randn(2, 2048)
    eeg_out = eeg_dec(z, seq_length=19)
    print(f"  EEG output shape: {eeg_out.shape}")
    assert eeg_out.shape == (2, 19, 19), f"Expected (2, 19, 19), got {eeg_out.shape}"
    print(f"  EEG output shape: {eeg_out.shape}")

    print("\nTesting fMRIKDADecoder...")
    fmri_dec = fMRIKDADecoder(latent_dim= 1024, num_regions=400)
    fmri_out = fmri_dec(z, seq_length=100)
    print(f"  fMRI output shape: {fmri_out.shape}")

    print("\nTesting KIMIKDAMoDeCoderRouter...")
    router = KIMIKDAMoDeCoderRouter(latent_dim= 1024, output_channels=19, num_regions=400)
    hub_eeg = torch.randn(2, 2048)
    hub_fmri = torch.randn(2, 2048)
    out = router(z, hub_eeg, hub_fmri, eeg_seq_length=19)
    print(f"  EEG recon shape: {out['eeg_recon'].shape}")
    print(f"  fMRI recon shape: {out['fmri_recon'].shape}")

    print("\nAll tests passed!")