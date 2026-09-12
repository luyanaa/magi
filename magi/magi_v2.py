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
        causal: bool = False,
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
        attn_temporal = torch.einsum(
            'bchtd,bchkd->bchtk', q_temporal, k_temporal) * self.scale
        if causal:
            future = torch.triu(
                torch.ones(
                    num_times, num_times, dtype=torch.bool,
                    device=attn_temporal.device),
                diagonal=1,
            )
            attn_temporal = attn_temporal.masked_fill(
                future.view(1, 1, 1, num_times, num_times),
                torch.finfo(attn_temporal.dtype).min,
            )
        attn_temporal = F.softmax(attn_temporal, dim=-1)
        attn_temporal = self.attn_dropout(attn_temporal)

        # Temporal attention output
        out_temporal = torch.einsum(
            'bchtk,bchkd->bchtd', attn_temporal, v_temporal)
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
        num_channels: Optional[int] = None,
        causal: bool = False,
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
        attn = torch.einsum(
            'bwhsd,bwhkd->bwhsk', q_windows, k_windows) * self.scale
        if causal:
            positions = torch.arange(
                num_windows * self.window_size, device=attn.device
            ).view(num_windows, self.window_size)
            time_index = positions % max(1, int(num_times))
            future = time_index[:, None, :] > time_index[:, :, None]
            invalid_keys = positions[:, None, :] >= N
            causal_mask = future | invalid_keys
            attn = attn.masked_fill(
                causal_mask.view(1, num_windows, 1,
                                 self.window_size, self.window_size),
                torch.finfo(attn.dtype).min,
            )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out_windows = torch.einsum(
            'bwhsk,bwhkd->bwhsd', attn, v_windows)
        out_windows = rearrange(out_windows, 'b w h s d -> b w s (h d)')

        # Reshape back
        out = out_windows.view(B, num_windows * self.window_size, D)
        if pad_len > 0:
            out = out[:, :N, :]

        return out

    def _global_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_channels: Optional[int] = None,
        num_times: Optional[int] = None,
        causal: bool = False,
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
        if causal:
            time_index = torch.arange(N, device=attn.device)
            if num_times is not None:
                time_index = time_index % max(1, int(num_times))
            future = time_index[None, :] > time_index[:, None]
            attn = attn.masked_fill(
                future.view(1, 1, N, N),
                torch.finfo(attn.dtype).min,
            )
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
        causal: bool = False,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # Project to QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply appropriate attention pattern
        if (self.attention_type == "factorized"
                and num_channels is not None and num_times is not None):
            out = self._factorized_attention(
                q, k, v, num_channels, num_times, causal=causal)
        elif self.attention_type == "swa" and num_times is not None:
            out = self._sliding_window_attention(
                q, k, v, num_times, num_channels=num_channels,
                causal=causal)
        else:
            out = self._global_attention(
                q, k, v, num_channels=num_channels, num_times=num_times,
                causal=causal)
        
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
        causal: bool = False,
    ) -> torch.Tensor:
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, num_channels, num_times, causal=causal)
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
        causal: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, num_channels, num_times, causal=causal)
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
        stride_time: Optional[int] = None,
        use_biot_embedding: bool = True,
        use_channel_type_embed: bool = True,
        ecog_amplitude_scale: float = 20.0,  # ECoG ~20× scalp EEG amplitude
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_channels = max_channels
        self.patch_size_time = patch_size_time
        self.stride_time = patch_size_time if stride_time is None else stride_time
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
            stride=self.stride_time,
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
    
    def _channel_amplitude_scale(
        self,
        eeg: torch.Tensor,
        channel_types: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply deterministic modality scaling without data-dependent stats."""
        if channel_types is None:
            return eeg
        scale_factors = torch.ones_like(
            channel_types, dtype=eeg.dtype, device=eeg.device)
        scale_factors[channel_types == 1] = 1.0 / self.ecog_amplitude_scale
        scale_factors[channel_types == 2] = (
            1.0 / (self.ecog_amplitude_scale / 2))
        return eeg * scale_factors.unsqueeze(-1)

    def _normalization_stats(
        self,
        eeg: torch.Tensor,
        channel_types: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample/channel stats for a fixed waveform support."""
        scaled = self._channel_amplitude_scale(eeg, channel_types)
        return (
            scaled.mean(dim=-1, keepdim=True),
            scaled.std(dim=-1, keepdim=True) + 1e-6,
        )

    def _normalize_amplitude(
        self,
        eeg: torch.Tensor,
        channel_types: torch.Tensor,
        stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Normalize amplitude using optional support-independent statistics."""
        if channel_types is None:
            return eeg
        scaled = self._channel_amplitude_scale(eeg, channel_types)
        if stats is None:
            stats = self._normalization_stats(eeg, channel_types)
        mean, std = stats
        return (scaled - mean) / std
    
    def _prepare_inputs(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        normalize_amplitude: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[List[str]]], Optional[torch.Tensor]]:
        """Normalize metadata and, when requested, amplitude."""
        if eeg.dim() != 3:
            raise ValueError("eeg must have shape (batch, channels, time)")
        batch_size, channels, _ = eeg.shape
        if channel_names is not None and channel_names and isinstance(
                channel_names[0], str):
            channel_names = [channel_names for _ in range(batch_size)]
        if channel_types is not None and not isinstance(
                channel_types, torch.Tensor):
            channel_types = torch.as_tensor(
                channel_types, device=eeg.device)
        if channel_types is not None:
            channel_types = channel_types.to(device=eeg.device)
        if channel_types is not None and channel_types.dim() == 1:
            channel_types = channel_types.unsqueeze(0).expand(
                batch_size, -1)
        if channels > self.max_channels:
            raise ValueError(
                f"Number of channels {channels} exceeds max_channels "
                f"{self.max_channels}")
        if (normalize_amplitude and self.use_channel_type_embed
                and channel_types is not None):
            eeg = self._normalize_amplitude(eeg, channel_types)
        return eeg, channel_names, channel_types

    def embed_tokens(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        causal: bool = False,
        normalize_amplitude: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Create per-channel temporal patch tokens without self-attention."""
        if normalize_amplitude is None:
            normalize_amplitude = not causal
        eeg, channel_names, channel_types = self._prepare_inputs(
            eeg, channel_names, channel_types,
            normalize_amplitude=normalize_amplitude)
        batch_size, channels, _ = eeg.shape
        tokens_list = []
        for batch_index in range(batch_size):
            batch_tokens = []
            for channel_index in range(channels):
                channel_signal = eeg[
                    batch_index, channel_index:channel_index + 1, :]
                if causal and self.stride_time != self.patch_size_time:
                    channel_tokens = F.conv1d(
                        channel_signal.unsqueeze(1),
                        self.temporal_proj.weight,
                        self.temporal_proj.bias,
                        stride=self.patch_size_time,
                    )
                else:
                    channel_tokens = self.temporal_proj(
                        channel_signal.unsqueeze(1))
                channel_tokens = channel_tokens.squeeze(0).transpose(0, 1)

                if self.biot_embed is not None and channel_names is not None:
                    channel_name = (
                        channel_names[batch_index][channel_index]
                        if channel_index < len(channel_names[batch_index])
                        else "unknown")
                    biot_emb = self.biot_embed.get_embedding(channel_name)
                    if biot_emb is not None:
                        channel_tokens = channel_tokens + biot_emb.to(
                            channel_tokens).unsqueeze(0)

                if self.channel_type_embed is not None and channel_types is not None:
                    ctype_emb = self.channel_type_embed(
                        channel_types[batch_index, channel_index])
                    channel_tokens = channel_tokens + ctype_emb.to(
                        channel_tokens).unsqueeze(0)
                batch_tokens.append(channel_tokens)
            tokens_list.append(torch.stack(batch_tokens, dim=0))

        all_tokens = torch.stack(tokens_list, dim=0)
        num_times = all_tokens.shape[2]
        all_tokens = all_tokens.reshape(
            batch_size, channels * num_times, self.hidden_dim)
        return self.proj_norm(all_tokens), channels, num_times

    def extract_patches(
        self,
        eeg: torch.Tensor,
        channel_types: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return raw patches as ``(B, C, N, patch_size)``."""
        eeg, _, _ = self._prepare_inputs(
            eeg, None, channel_types,
            normalize_amplitude=normalize)
        return eeg.unfold(
            dimension=-1,
            size=self.patch_size_time,
            step=self.stride_time if step is None else int(step),
        )

    def expand_patch_mask(
        self,
        mask: torch.Tensor,
        num_channels: int,
        num_times: int,
    ) -> torch.Tensor:
        """Mask every token whose convolution window overlaps a masked patch."""
        if mask.shape[1] != num_channels * num_times:
            raise ValueError("patch mask shape does not match token grid")
        starts = torch.arange(
            num_times, device=mask.device, dtype=torch.long
        ) * self.stride_time
        ends = starts + self.patch_size_time
        overlap = (
            starts[:, None] < ends[None, :]
        ) & (
            ends[:, None] > starts[None, :]
        )
        grid = mask.reshape(mask.shape[0], num_channels, num_times)
        expanded = torch.matmul(
            grid.to(dtype=torch.float32),
            overlap.to(dtype=torch.float32).transpose(0, 1),
        ).gt(0)
        return expanded.reshape(mask.shape[0], num_channels * num_times)

    def mask_raw_patches(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        num_channels: int,
        num_times: int,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Replace selected raw patch supports before temporal projection."""
        if mask.shape[1] != num_channels * num_times:
            raise ValueError("patch mask shape does not match token grid")
        masked = eeg.clone()
        grid = mask.reshape(mask.shape[0], num_channels, num_times)
        sample_mask = torch.zeros_like(masked, dtype=torch.bool)
        time_len = eeg.shape[-1]
        for patch_index in range(num_times):
            start = patch_index * self.stride_time
            end = min(time_len, start + self.patch_size_time)
            if start >= end:
                continue
            sample_mask[:, :, start:end] |= grid[:, :, patch_index].unsqueeze(-1)
        return masked.masked_fill(sample_mask, fill_value)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        num_channels: int,
        num_times: int,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer and pool token representations."""
        last_hidden = self.encoder(
            tokens,
            num_channels=num_channels,
            num_times=num_times,
            causal=causal,
        )
        pooler_output = self.pooler(last_hidden.mean(dim=1))
        return last_hidden, pooler_output

    def forward(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode raw EEG, optionally using causal temporal attention."""
        del attention_mask  # Padding masks are not yet part of the Magi contract.
        tokens, channels, num_times = self.embed_tokens(
            eeg, channel_names, channel_types, causal=causal)
        last_hidden, pooler_output = self.encode_tokens(
            tokens, channels, num_times, causal=causal)
        if output_hidden_states:
            return last_hidden, pooler_output, tokens
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