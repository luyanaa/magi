"""
Magi v2 Architecture: 24L × 1024d ModernBERT-large scale (~340M params)

Key features:
- 24 layers × 1024 hidden dimension
- RoPE (Rotary Position Embedding) for better sequence modeling
- GeGLU (Gated Linear Unit) activation in FFN
- Alternating attention: global attention every 3 layers, SWA (Sliding Window Attention) otherwise
- Pre-norm architecture (LayerNorm before attention/FFN)
- Tiling initialization from 12L model (initialize 24L by repeating 12L weights)
- Factorized attention for EEG spatial-temporal structure
- Channel type embedding for multi-modality (scalp EEG / ECoG / sEEG)
- BIOT-style arbitrary electrode position embedding
- Support for variable channel counts (cap 256)
- Amplitude normalization for ECoG (~20× scalp EEG)

References:
- ModernBERT: Scaling BERT to Modern Architectures (arXiv:2406.xxxx)
- RoPE: Rotary Position Embedding (Su et al., 2021)
- GeGLU: Gated Linear Units (Shazeer, 2020)
- BIOT: Biosignal Transformer (Chen et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from einops import rearrange, repeat

from .spatio_temporal import (
    STANDARD_10_20_COORDS,
    STANDARD_10_5_COORDS,
    BIOTStyleEmbedding,
    ChannelTypeEmbedding,
)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better sequence modeling."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inv_freq for RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache for forward pass
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Tensor of shape (..., seq_len, dim)
            seq_dim: Dimension containing sequence length
        
        Returns:
            Tensor with RoPE applied
        """
        seq_len = x.shape[seq_dim]
        
        # Ensure cache is large enough
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len * 2
            self._build_cache(self.max_seq_len)
        
        # Get cos/sin for this sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Reshape for rotary application
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotary transformation
        rotated = torch.cat(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
            dim=-1
        )
        
        return rotated.type_as(x)


class GeGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * self.gelu(x2)


class FactorizedAttentionV2(nn.Module):
    """
    Factorized attention with RoPE support and alternating attention patterns.
    
    Supports:
    - Global attention (full attention across sequence)
    - Sliding Window Attention (SWA) with configurable window size
    - Factorized spatial-temporal attention
    - RoPE for better positional encoding
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1,
        spatial_heads: int = 8,
        temporal_heads: int = 8,
        window_size: int = 128,
        use_rope: bool = True,
        attention_type: str = "global",  # "global", "swa", "factorized"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.spatial_heads = spatial_heads
        self.temporal_heads = temporal_heads
        self.window_size = window_size
        self.use_rope = use_rope
        self.attention_type = attention_type
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert spatial_heads + temporal_heads == num_heads, "spatial_heads + temporal_heads must equal num_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # RoPE for temporal dimension
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
        
    def _apply_rope(self, x: torch.Tensor, seq_dim: int = -2, num_heads: int = None) -> torch.Tensor:
        """Apply RoPE to query and key tensors."""
        if not self.use_rope:
            return x
        
        if num_heads is None:
            num_heads = self.num_heads
        x = rearrange(x, '... (h d) -> ... h d', h=num_heads)
        x = self.rope(x, seq_dim=seq_dim)
        x = rearrange(x, '... h d -> ... (h d)')
        return x
    
    def _factorized_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_channels: int,
        num_times: int,
    ) -> torch.Tensor:
        """Factorized attention: spatial attention followed by temporal attention."""
        B, N, D = q.shape
        
        # Reshape to (B, C, T, D)
        q = q.view(B, num_channels, num_times, D)
        k = k.view(B, num_channels, num_times, D)
        v = v.view(B, num_channels, num_times, D)
        
        # Split heads for spatial and temporal attention
        q_spatial = q[:, :, :, :self.spatial_heads * self.head_dim]
        q_temporal = q[:, :, :, self.spatial_heads * self.head_dim:]
        
        k_spatial = k[:, :, :, :self.spatial_heads * self.head_dim]
        k_temporal = k[:, :, :, self.spatial_heads * self.head_dim:]
        
        v_spatial = v[:, :, :, :self.spatial_heads * self.head_dim]
        v_temporal = v[:, :, :, self.spatial_heads * self.head_dim:]
        
        # Spatial attention (across channels at each time point)
        q_spatial = rearrange(q_spatial, 'b c t (h d) -> b t h c d', 
                            h=self.spatial_heads, d=self.head_dim)
        k_spatial = rearrange(k_spatial, 'b c t (h d) -> b t h c d',
                            h=self.spatial_heads, d=self.head_dim)
        v_spatial = rearrange(v_spatial, 'b c t (h d) -> b t h c d',
                            h=self.spatial_heads, d=self.head_dim)
        
        # Apply RoPE to spatial dimension (channel positions)
        q_spatial = self._apply_rope(q_spatial, seq_dim=-2, num_heads=self.spatial_heads)
        k_spatial = self._apply_rope(k_spatial, seq_dim=-2, num_heads=self.spatial_heads)
        
        # Spatial attention scores
        attn_spatial = torch.einsum('bthcd,bthkd->bthck', q_spatial, k_spatial) * self.scale
        attn_spatial = F.softmax(attn_spatial, dim=-1)
        attn_spatial = self.attn_dropout(attn_spatial)
        
        # Spatial attention output
        out_spatial = torch.einsum('bthck,bthkd->bthcd', attn_spatial, v_spatial)
        out_spatial = rearrange(out_spatial, 'b t h c d -> b c t (h d)')
        
        # Temporal attention (across time at each channel)
        q_temporal = rearrange(q_temporal, 'b c t (h d) -> b c h t d',
                             h=self.temporal_heads, d=self.head_dim)
        k_temporal = rearrange(k_temporal, 'b c t (h d) -> b c h t d',
                             h=self.temporal_heads, d=self.head_dim)
        v_temporal = rearrange(v_temporal, 'b c t (h d) -> b c h t d',
                             h=self.temporal_heads, d=self.head_dim)
        
        # Apply RoPE to temporal dimension
        q_temporal = self._apply_rope(q_temporal, seq_dim=-2, num_heads=self.temporal_heads)
        k_temporal = self._apply_rope(k_temporal, seq_dim=-2, num_heads=self.temporal_heads)
        
        # Temporal attention scores
        attn_temporal = torch.einsum('bchtd,bchkd->bchtk', q_temporal, k_temporal) * self.scale
        attn_temporal = F.softmax(attn_temporal, dim=-1)
        attn_temporal = self.attn_dropout(attn_temporal)
        
        # Temporal attention output
        out_temporal = torch.einsum('bchtk,bchkd->bchtd', attn_temporal, v_temporal)
        out_temporal = rearrange(out_temporal, 'b c h t d -> b c t (h d)')
        
        # Concatenate spatial and temporal outputs
        out = torch.cat([out_spatial, out_temporal], dim=-1)
        out = rearrange(out, 'b c t d -> b (c t) d')
        
        return out
    
    def _sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_times: int,
    ) -> torch.Tensor:
        """Sliding Window Attention with configurable window size."""
        B, N, D = q.shape

        num_windows = math.ceil(N / self.window_size)
        pad_len = num_windows * self.window_size - N
        
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        
        q_windows = q.view(B, num_windows, self.window_size, D)
        k_windows = k.view(B, num_windows, self.window_size, D)
        v_windows = v.view(B, num_windows, self.window_size, D)
        
        # Apply attention within each window
        q_windows = rearrange(q_windows, 'b w s (h d) -> b w h s d', h=self.num_heads, d=self.head_dim)
        k_windows = rearrange(k_windows, 'b w s (h d) -> b w h s d', h=self.num_heads, d=self.head_dim)
        v_windows = rearrange(v_windows, 'b w s (h d) -> b w h s d', h=self.num_heads, d=self.head_dim)
        
        # Apply RoPE within windows
        q_windows = self._apply_rope(q_windows, seq_dim=-2, num_heads=self.num_heads)
        k_windows = self._apply_rope(k_windows, seq_dim=-2, num_heads=self.num_heads)
        
        # Window attention
        attn = torch.einsum('bwhsd,bwhkd->bwhsk', q_windows, k_windows) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out_windows = torch.einsum('bwhsk,bwhkd->bwhsd', attn, v_windows)
        out_windows = rearrange(out_windows, 'b w h s d -> b w s (h d)')
        
        # Reshape back
        out = out_windows.view(B, num_windows * self.window_size, D)
        if pad_len > 0:
            out = out[:, :num_times, :]
        
        return out
    
    def _global_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Global full attention across entire sequence."""
        B, N, D = q.shape
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        
        # Apply RoPE
        q = self._apply_rope(q, seq_dim=-2)
        k = self._apply_rope(k, seq_dim=-2)
        
        # Attention
        attn = torch.einsum('bhnd,bhkd->bhnk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.einsum('bhnk,bhkd->bhnd', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        num_channels: Optional[int] = None,
        num_times: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        
        # Project to QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply appropriate attention pattern
        if self.attention_type == "factorized" and num_channels is not None and num_times is not None:
            out = self._factorized_attention(q, k, v, num_channels, num_times)
        elif self.attention_type == "swa" and num_times is not None:
            out = self._sliding_window_attention(q, k, v, num_times)
        else:
            out = self._global_attention(q, k, v)
        
        # Output projection
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        return out


class MagiV2TransformerLayer(nn.Module):
    """Magi v2 transformer layer with GeGLU and alternating attention."""
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        intermediate_dim: int = 4096,  # 4× hidden_dim
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        spatial_heads: int = 8,
        temporal_heads: int = 8,
        window_size: int = 128,
        use_rope: bool = True,
        attention_type: str = "global",  # "global", "swa", "factorized"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Attention
        self.attn = FactorizedAttentionV2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            spatial_heads=spatial_heads,
            temporal_heads=temporal_heads,
            window_size=window_size,
            use_rope=use_rope,
            attention_type=attention_type,
        )
        
        # GeGLU FFN
        self.ffn = nn.Sequential(
            GeGLU(hidden_dim, intermediate_dim),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        num_channels: Optional[int] = None,
        num_times: Optional[int] = None,
    ) -> torch.Tensor:
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, num_channels, num_times)
        x = self.dropout(x)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class MagiV2TransformerEncoder(nn.Module):
    """Magi v2 transformer encoder with 24 layers and alternating attention."""
    
    def __init__(
        self,
        num_layers: int = 24,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        intermediate_dim: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        spatial_heads: int = 8,
        temporal_heads: int = 8,
        window_size: int = 128,
        use_rope: bool = True,
        alternating_pattern: bool = True,  # Global every 3 layers, SWA otherwise
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create alternating attention pattern
        layers = []
        for i in range(num_layers):
            if alternating_pattern:
                # Global attention every 3 layers, SWA otherwise
                if i % 3 == 0:
                    attn_type = "global"
                else:
                    attn_type = "swa"
            else:
                # All factorized attention
                attn_type = "factorized"
            
            layer = MagiV2TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                spatial_heads=spatial_heads,
                temporal_heads=temporal_heads,
                window_size=window_size,
                use_rope=use_rope,
                attention_type=attn_type,
            )
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        num_channels: Optional[int] = None,
        num_times: Optional[int] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, num_channels, num_times)
        return self.norm(x)
    
    def tile_from_12l(self, source_model: nn.Module):
        """
        Tile weights from a 12-layer model to initialize this 24-layer model.
        
        Strategy: Repeat 12-layer weights twice (blocks 0-11 → 0-11, 0-11 → 12-23)
        with small noise added to second copy to break symmetry.
        """
        if source_model.num_layers != 12:
            raise ValueError(f"Source model must have 12 layers, got {source_model.num_layers}")
        
        # Tile layer weights
        for i in range(self.num_layers):
            source_idx = i % 12
            
            # Copy weights from source layer
            source_layer = source_model.layers[source_idx]
            target_layer = self.layers[i]
            
            # Copy attention weights
            target_layer.attn.q_proj.weight.data = source_layer.attn.q_proj.weight.data.clone()
            target_layer.attn.k_proj.weight.data = source_layer.attn.k_proj.weight.data.clone()
            target_layer.attn.v_proj.weight.data = source_layer.attn.v_proj.weight.data.clone()
            target_layer.attn.out_proj.weight.data = source_layer.attn.out_proj.weight.data.clone()
            
            # Copy FFN weights
            target_layer.ffn[0].proj.weight.data = source_layer.ffn[0].weight.data.clone()
            target_layer.ffn[2].weight.data = source_layer.ffn[3].weight.data.clone()
            
            # Copy layer norm weights
            target_layer.norm1.weight.data = source_layer.norm1.weight.data.clone()
            target_layer.norm2.weight.data = source_layer.norm2.weight.data.clone()
            
            # Add small noise to second copy to break symmetry
            if i >= 12:
                noise_scale = 1e-3
                target_layer.attn.q_proj.weight.data += torch.randn_like(target_layer.attn.q_proj.weight.data) * noise_scale
                target_layer.attn.k_proj.weight.data += torch.randn_like(target_layer.attn.k_proj.weight.data) * noise_scale
                target_layer.attn.v_proj.weight.data += torch.randn_like(target_layer.attn.v_proj.weight.data) * noise_scale
                target_layer.attn.out_proj.weight.data += torch.randn_like(target_layer.attn.out_proj.weight.data) * noise_scale
                target_layer.ffn[0].proj.weight.data += torch.randn_like(target_layer.ffn[0].proj.weight.data) * noise_scale
                target_layer.ffn[2].weight.data += torch.randn_like(target_layer.ffn[2].weight.data) * noise_scale
        
        # Copy final norm
        self.norm.weight.data = source_model.norm.weight.data.clone()
        self.norm.bias.data = source_model.norm.bias.data.clone()


class MagiV2EEGEncoder(nn.Module):
    """
    Complete Magi v2 EEG encoder combining:
    - BIOT-style arbitrary electrode embedding
    - Channel type embedding (scalp EEG / ECoG / sEEG)
    - Magi v2 transformer encoder (24L × 1024d)
    - Amplitude normalization for ECoG
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_dim: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        spatial_heads: int = 8,
        temporal_heads: int = 8,
        window_size: int = 128,
        use_rope: bool = True,
        alternating_pattern: bool = True,
        max_channels: int = 256,  # Cap for ECoG/sEEG
        patch_size_time: int = 256,
        use_biot_embedding: bool = True,
        use_channel_type_embed: bool = True,
        ecog_amplitude_scale: float = 20.0,  # ECoG ~20× scalp EEG amplitude
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_channels = max_channels
        self.patch_size_time = patch_size_time
        self.use_biot_embedding = use_biot_embedding
        self.use_channel_type_embed = use_channel_type_embed
        self.ecog_amplitude_scale = ecog_amplitude_scale
        
        # BIOT-style embedding for arbitrary electrodes
        if use_biot_embedding:
            self.biot_embed = BIOTStyleEmbedding(
                embedding_dim=hidden_dim,
                use_3d_coords=True,
            )
        else:
            self.biot_embed = None
        
        # Channel type embedding (0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown)
        if use_channel_type_embed:
            self.channel_type_embed = ChannelTypeEmbedding(
                num_types=4,
                embedding_dim=hidden_dim,
            )
        else:
            self.channel_type_embed = None
        
        # Temporal projection for EEG time series
        self.temporal_proj = nn.Conv1d(
            in_channels=1,  # Per electrode
            out_channels=hidden_dim,
            kernel_size=patch_size_time,
            stride=patch_size_time,
            padding=0,
        )
        
        # Layer norm after projection
        self.proj_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Magi v2 transformer encoder
        self.encoder = MagiV2TransformerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            spatial_heads=spatial_heads,
            temporal_heads=temporal_heads,
            window_size=window_size,
            use_rope=use_rope,
            alternating_pattern=alternating_pattern,
        )
        
        # Pooler
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
    
    def _normalize_amplitude(
        self,
        eeg: torch.Tensor,
        channel_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize amplitude based on channel type.
        ECoG signals are ~20× larger than scalp EEG.
        """
        if channel_types is None:
            return eeg
        
        # Create amplitude scaling factors
        # scalp_EEG: 1.0, ecog_grid: 1/20, seeg_depth: 1/10, unknown: 1.0
        scale_factors = torch.ones_like(channel_types, dtype=eeg.dtype, device=eeg.device)
        
        # ECoG: scale down by ecog_amplitude_scale
        ecog_mask = (channel_types == 1)
        scale_factors[ecog_mask] = 1.0 / self.ecog_amplitude_scale
        
        # sEEG: scale down by half of ECoG
        seeg_mask = (channel_types == 2)
        scale_factors[seeg_mask] = 1.0 / (self.ecog_amplitude_scale / 2)
        
        # Apply scaling
        scale_factors = scale_factors.view(1, -1, 1)  # (1, C, 1)
        eeg = eeg * scale_factors
        
        # Then z-score normalize per channel
        mean = eeg.mean(dim=-1, keepdim=True)
        std = eeg.std(dim=-1, keepdim=True) + 1e-6
        eeg = (eeg - mean) / std
        
        return eeg
    
    def forward(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            eeg: (B, C, T) raw EEG/ECoG/sEEG signals
            channel_names: List of length B, each containing C channel names
            channel_types: (B, C) integer tensor: 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
            attention_mask: (B, C*T) boolean mask (1 for valid, 0 for padding)
            output_hidden_states: whether to return all hidden states
        
        Returns:
            last_hidden_state: (B, C*T/P, D) where P = patch_size_time
            pooler_output: (B, D)
        """
        B, C, T = eeg.shape
        
        # Validate channel count
        if C > self.max_channels:
            raise ValueError(f"Number of channels {C} exceeds max_channels {self.max_channels}")
        
        # Amplitude normalization based on channel type
        if self.use_channel_type_embed and channel_types is not None:
            eeg = self._normalize_amplitude(eeg, channel_types)
        
        # Process each channel separately
        tokens_list = []
        for b in range(B):
            batch_tokens = []
            for c in range(C):
                # Get single channel time series
                channel_signal = eeg[b, c:c+1, :]  # (1, T)
                
                # Temporal projection
                channel_tokens = self.temporal_proj(channel_signal.unsqueeze(1))  # (1, D, T/P)
                channel_tokens = channel_tokens.squeeze(1).transpose(1, 0)  # (T/P, D)
                
                # Add BIOT embedding if available
                if self.biot_embed is not None and channel_names is not None:
                    channel_name = channel_names[b][c] if c < len(channel_names[b]) else "unknown"
                    biot_emb = self.biot_embed.get_embedding(channel_name)
                    if biot_emb is not None:
                        channel_tokens = channel_tokens + biot_emb.unsqueeze(0)
                
                # Add channel type embedding
                if self.channel_type_embed is not None and channel_types is not None:
                    ctype = channel_types[b, c]
                    ctype_emb = self.channel_type_embed(ctype)
                    channel_tokens = channel_tokens + ctype_emb.unsqueeze(0)
                
                batch_tokens.append(channel_tokens)
            
            # Stack tokens for this batch: (C, T/P, D)
            batch_tokens = torch.stack(batch_tokens, dim=0)
            tokens_list.append(batch_tokens)
        
        # Combine batch: reshape to (B, C*T/P, D)
        all_tokens = torch.stack(tokens_list, dim=0)  # (B, C, T/P, D)
        num_tokens_per_channel = all_tokens.shape[2]
        all_tokens = all_tokens.view(B, C * num_tokens_per_channel, self.hidden_dim)
        
        # Apply projection norm
        all_tokens = self.proj_norm(all_tokens)
        
        # Apply encoder
        last_hidden = self.encoder(
            all_tokens,
            num_channels=C,
            num_times=num_tokens_per_channel,
        )
        
        # Pooler
        pooler_output = self.pooler(last_hidden.mean(dim=1))
        
        if output_hidden_states:
            return last_hidden, pooler_output, all_tokens
        else:
            return last_hidden, pooler_output


def create_magi_v2_from_v1(
    v1_model: nn.Module,
    hidden_dim: int = 1024,
    num_layers: int = 24,
    **kwargs,
) -> MagiV2EEGEncoder:
    """
    Create Magi v2 model by tiling weights from v1 model.
    
    Args:
        v1_model: Existing Magi v1 model (12L × 768d)
        hidden_dim: Target hidden dimension (1024)
        num_layers: Target number of layers (24)
        **kwargs: Additional kwargs for MagiV2EEGEncoder
    
    Returns:
        Initialized Magi v2 model
    """
    # Extract v1 config
    v1_hidden_dim = v1_model.hidden_dim
    v1_num_layers = len(v1_model.encoder.layers) if hasattr(v1_model.encoder, 'layers') else 12
    
    # Create v2 model
    v2_model = MagiV2EEGEncoder(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs,
    )
    
    # Handle dimension mismatch: 768 → 1024
    if v1_hidden_dim != hidden_dim:
        print(f"Note: Hidden dimension mismatch: v1={v1_hidden_dim}, v2={hidden_dim}")
        print("Will use random initialization with tiling where possible.")
        # We can still tile layer structure but need projection for weights
        # For now, we'll initialize randomly and tile layer pattern
        pass
    
    # Tile layer weights if dimensions match
    if v1_hidden_dim == hidden_dim and hasattr(v1_model.encoder, 'tile_from_12l'):
        v2_model.encoder.tile_from_12l(v1_model.encoder)
    
    return v2_model