import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors.
    
    RoPE encodes position information by rotating query and key vectors in complex space.
    This allows the model to generalize to longer sequences than seen during training.
    
    Args:
        q: (batch, heads, seq_len, head_dim) query tensor
        k: (batch, heads, seq_len, head_dim) key tensor
        positions: (batch, seq_len) position indices
        base: base for the angular frequency (default: 10000.0)
    Returns:
        q, k: rotated query and key tensors
    """
    B, H, L, d = q.shape
    device = q.device
    
    # Compute frequency bands: (head_dim // 2)
    freqbands = torch.arange(0, d // 2, device=device, dtype=torch.float32)
    freqbands = torch.pow(base, -freqbands * 2 / d)  # (d//2,)
    
    # Compute angles: (L, d//2)
    positions_float = positions.float()  # (B, L) or (1, L)
    if positions.dim() == 2:
        positions_float = positions_float.unsqueeze(1)  # (B, 1, L)
    angles = positions_float.unsqueeze(-1) * freqbands  # (B, L, d//2)
    
    # Compute cos and sin: (B, L, d//2)
    cos = angles.cos()
    sin = angles.sin()
    
    # Split q and k into halves
    q1, q2 = q[..., :d//2], q[..., d//2:]  # each (B, H, L, d//2)
    k1, k2 = k[..., :d//2], k[..., d//2:]
    
    # Rotate using complex multiplication:
    # RoPE(q1, q2) = q1 * cos - q2 * sin, q1 * sin + q2 * cos
    q1_new = q1 * cos - q2 * sin
    q2_new = q1 * sin + q2 * cos
    k1_new = k1 * cos - k2 * sin
    k2_new = k1 * sin + k2 * cos
    
    q_rotated = torch.cat([q1_new, q2_new], dim=-1)
    k_rotated = torch.cat([k1_new, k2_new], dim=-1)
    
    return q_rotated, k_rotated

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    flash_attn_func = None

try:
    from xformers.ops import memory_efficient_attention
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    memory_efficient_attention = None


class MultiHeadAttention(nn.Module):
    """
    Multi‑head self‑attention with optional Flash Attention (via `flash_attn`) and
    xFormers memory‑efficient attention as fallback.
    
    Supports:
    - Rotary Position Embedding (RoPE) for better generalization to variable-length sequences
    - Sliding Window Attention for long-context modeling with linear complexity
    - Hybrid SWA + Global Attention (Mistral/Granite style) with Longformer-style global tokens
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        use_flash: bool = True,
        use_xformers: bool = True,
        bias: bool = True,
        use_rope: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int = 512,
        enable_hybrid_swa: bool = False,  # Mistral-style: SWA in early layers, full attention periodically
        hybrid_full_attn_interval: int = 4,  # Every N layers use full attention (if enable_hybrid_swa)
        use_adaptive_swa_gating: bool = False,
        num_global_heads: int = 0,  # Number of heads that use full attention even when SWA is enabled
        use_rel_pos_bias: bool = False,  # Whether to use learned relative position bias
        max_rel_pos: int = 512,  # Maximum relative position for learned bias
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Validate sliding_window
        if use_sliding_window and sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {sliding_window}")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.enable_hybrid_swa = enable_hybrid_swa
        self.hybrid_full_attn_interval = hybrid_full_attn_interval
        self.use_adaptive_swa_gating = use_adaptive_swa_gating
        self.num_global_heads = num_global_heads
        self.use_rel_pos_bias = use_rel_pos_bias
        self.max_rel_pos = max_rel_pos

        if self.use_adaptive_swa_gating:
            self.swa_gate = nn.Parameter(torch.zeros(1))

        if self.use_rel_pos_bias:
            # T5-style relative position bias
            # We use a simple table-based bias for each head
            self.rel_pos_bias = nn.Embedding(2 * max_rel_pos + 1, num_heads)

        self.use_flash = use_flash and FLASH_AVAILABLE
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        self.use_sdpa = hasattr(F, 'scaled_dot_product_attention')  # PyTorch 2.0+
        
        # Check FlashAttention version for window_size support
        self.flash_supports_window_size = False
        if FLASH_AVAILABLE:
            try:
                import flash_attn
                # FlashAttention 2.0+ supports window_size
                version = getattr(flash_attn, '__version__', '0.0.0')
                major = int(version.split('.')[0]) if version else 0
                self.flash_supports_window_size = major >= 2
            except:
                self.flash_supports_window_size = False

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        layer_idx: Optional[int] = None,
        global_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) or (batch, 1, seq_len, seq_len) boolean mask
                where True positions are allowed to attend, False are masked out.
            key_padding_mask: (batch, seq_len) boolean mask where True positions are padding.
            causal: whether to apply causal masking (for autoregressive generation).
            layer_idx: index of this layer in the encoder (for hybrid SWA decision).
            global_tokens_mask: (batch, seq_len) boolean mask where True = global token
                that attends to/from all positions (Longformer-style).
        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len) or None if using flash/xformers.
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device

        # Project queries, keys, values
        q = self.q_proj(hidden_states)  # (B, L, D)
        k = self.k_proj(hidden_states)  # (B, L, D)
        v = self.v_proj(hidden_states)  # (B, L, D)

        # Reshape for multi‑head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)

        # Apply Rotary Position Embedding (RoPE) if enabled
        if self.use_rope:
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            q, k = apply_rotary_pos_emb(q, k, positions)

        # Determine if this layer uses SWA or full attention (hybrid mode)
        use_swa_this_layer = self.use_sliding_window
        if self.enable_hybrid_swa and layer_idx is not None:
            # SWA for first few layers, then full attention periodically
            # Pattern: layers 0,1,2 use SWA, layer 3 uses full, layers 4,5,6 use SWA, layer 7 uses full, etc.
            # This means full attention at layer_idx % interval == (interval-1)
            use_swa_this_layer = (layer_idx % self.hybrid_full_attn_interval) != (self.hybrid_full_attn_interval - 1)

        # Prepare masks
        if attention_mask is not None and attention_mask.dim() == 2:
            # (B, L) -> (B, 1, 1, L) for broadcast
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if key_padding_mask is not None:
            # (B, L) -> (B, 1, 1, L)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if attention_mask is None:
                attention_mask = key_padding_mask
            else:
                # PyTorch 2.11: bitwise_and not supported for float;
                # cast both to bool before logical AND
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask.to(torch.bool)
                if key_padding_mask.dtype != torch.bool:
                    key_padding_mask = key_padding_mask.to(torch.bool)
                attention_mask = attention_mask & key_padding_mask

        # Convert boolean mask to additive mask (0 for allowed, -inf for masked)
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            additive_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            additive_mask = additive_mask.masked_fill(~attention_mask, float("-inf"))
            attention_mask = additive_mask

        if self.use_adaptive_swa_gating:
            # SWA branch
            swa_out, swa_weights = self._perform_attention(
                q, k, v, L, device, attention_mask, causal, 
                use_swa_this_layer=True, global_tokens_mask=global_tokens_mask
            )
            # Full attention branch
            full_out, full_weights = self._perform_attention(
                q, k, v, L, device, attention_mask, causal, 
                use_swa_this_layer=False, global_tokens_mask=global_tokens_mask
            )
            
            gate = torch.sigmoid(self.swa_gate)
            output = gate * swa_out + (1 - gate) * full_out
            attn_weights = None # Mixed weights are complex to represent
        else:
            output, attn_weights = self._perform_attention(
                q, k, v, L, device, attention_mask, causal, 
                use_swa_this_layer=use_swa_this_layer, global_tokens_mask=global_tokens_mask
            )

        # Merge heads
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)
        return output, attn_weights

    def _perform_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        L: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        use_swa_this_layer: bool = False,
        global_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = q.shape[0]
        # Flash Attention with native windowing (O(N) memory)
        # NOTE: FlashAttention cannot handle per-token variable attention patterns
        # (like Longformer-style global tokens). When global_tokens_mask is present,
        # we must fall back to SDPA which properly handles global attention.
        if self.use_flash and not causal and global_tokens_mask is None:
            # flash_attn expects q, k, v of shape (B, L, H, d) and returns (B, L, H, d)
            q_flash = q.transpose(1, 2).contiguous()  # (B, L, H, d)
            k_flash = k.transpose(1, 2).contiguous()
            v_flash = v.transpose(1, 2).contiguous()
            
            # Convert additive mask to (B, L) if needed
            attn_mask = None
            if attention_mask is not None:
                if attention_mask.size(-1) == L and attention_mask.size(-2) == 1:
                    attn_mask = attention_mask.squeeze(1).squeeze(1)  # (B, L)
                else:
                    attn_mask = attention_mask.squeeze(1)  # (B, L, L)
            
            # Use native FlashAttention window_size for true O(N) memory
            # Only use window_size if FlashAttention version supports it (2.0+)
            window_size = None
            if use_swa_this_layer and self.flash_supports_window_size:
                window_size = (self.sliding_window, self.sliding_window)
                
                output = flash_attn_func(
                    q_flash, k_flash, v_flash,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=causal,
                    attn_mask=attn_mask,
                    window_size=window_size,  # Native SWA support
                )
                output = output.transpose(1, 2).contiguous()  # (B, H, L, d)
                return output, None

            elif use_swa_this_layer and not self.flash_supports_window_size:
                # Fall back to SDPA path if Flash doesn't support window_size
                pass # Continue to xformers/sdpa paths
            else:
                # Full attention with Flash
                output = flash_attn_func(
                    q_flash, k_flash, v_flash,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=causal,
                    attn_mask=attn_mask,
                    window_size=None,
                )
                output = output.transpose(1, 2).contiguous()  # (B, H, L, d)
                return output, None
            
        # xFormers memory‑efficient attention (no native SWA, falls back to dense)
        if self.use_xformers and not use_swa_this_layer:
            # xFormers expects q, k, v of shape (B, H, L, d) and additive mask of shape (B, L, L) or None
            attn_mask = attention_mask.squeeze(1) if attention_mask is not None else None
            output = memory_efficient_attention(
                q, k, v,
                attn_bias=attn_mask,
                p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
            return output, None
            
        # PyTorch SDPA (scaled dot‑product attention) - PyTorch 2.0+
        if self.use_sdpa:
            attn_mask = None
            key_padding_mask_out = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    attn_mask = attention_mask.squeeze(1).squeeze(1)
                elif attention_mask.dim() == 2:
                    key_padding_mask_out = attention_mask
            
            # Apply sliding window mask if enabled (optimized block-diagonal)
            if use_swa_this_layer and not causal:
                attn_mask = self._create_sliding_window_mask(L, device, global_tokens_mask)
            
            # Handle global tokens (Longformer-style): they attend to all positions
            if global_tokens_mask is not None and not use_swa_this_layer:
                # Global tokens mask needs special handling - add to attention mask
                # Global tokens should be allowed to attend to all, others attend to globals
                global_mask = self._create_global_attention_mask(L, B, device, global_tokens_mask)
                if attn_mask is None:
                    attn_mask = global_mask
                else:
                    attn_mask = attn_mask + global_mask
            
            # PyTorch SDPA expects attention mask broadcastable to (B, H, L, L).
            # Ensure mask has head dimension (..., 1, L, L) for proper broadcasting.
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # (B, L) -> (B, 1, 1, L)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)  # (B, L, L) or (1, L, L) -> (B, 1, L, L)

            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask if attn_mask is not None else key_padding_mask_out,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
                scale=self.scale,
            )
            return output, None
            
        # Fallback to standard PyTorch attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias if enabled
        if self.use_rel_pos_bias:
            # Create relative position indices: (L, L)
            range_vec = torch.arange(L, device=device)
            rel_pos = range_vec.unsqueeze(1) - range_vec.unsqueeze(0)  # (L, L)
            # Clip and shift to be in [0, 2*max_rel_pos]
            rel_pos = torch.clamp(rel_pos, -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
            # (L, L, H)
            bias = self.rel_pos_bias(rel_pos)
            # (H, L, L)
            bias = bias.permute(2, 0, 1).unsqueeze(0)
            attn_scores = attn_scores + bias

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        if use_swa_this_layer and not causal:
            swa_mask = self._create_sliding_window_mask(L, device, global_tokens_mask)
            
            # Sparse / Global Head Mixture:
            # For heads < num_global_heads, we use full attention (ignore swa_mask)
            # For heads >= num_global_heads, we use SWA
            if self.num_global_heads > 0 and self.num_global_heads < self.num_heads:
                # attn_scores: (B, H, L, L)
                # swa_mask: (L, L) or (B, L, L)
                if swa_mask.dim() == 2:
                    swa_mask = swa_mask.unsqueeze(0).unsqueeze(0) # (1, 1, L, L)
                else:
                    swa_mask = swa_mask.unsqueeze(1) # (B, 1, L, L)
                
                # Apply SWA only to the subset of heads
                # Heads [num_global_heads:] get the mask
                attn_scores[:, self.num_global_heads:] = attn_scores[:, self.num_global_heads:] + swa_mask
            elif self.num_global_heads >= self.num_heads:
                # All heads are global, ignore SWA
                pass
            else:
                # All heads use SWA (default behavior)
                attn_scores = attn_scores + swa_mask.unsqueeze(0).unsqueeze(0) if swa_mask.dim() == 2 else attn_scores + swa_mask.unsqueeze(1)
            
        if causal:
            causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def _create_sliding_window_mask(
        self,
        L: int,
        device: torch.device,
        global_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create an efficient sliding window attention mask using block-diagonal approach.
        O(L * W) memory instead of O(L^2) where W is the window size.
        
        Args:
            L: sequence length
            device: device to create mask on
            global_tokens_mask: (B, L) boolean mask where True = global token
        Returns:
            additive_mask: (1, L, L) or (B, L, L) tensor with -inf for masked positions
        """
        W = self.sliding_window
        
        # Create band mask using torch.triu_indices for efficiency
        # This creates O(L*W) entries instead of O(L^2)
        row_indices = []
        col_indices = []
        
        for i in range(L):
            start = max(0, i - W)
            end = min(L, i + W + 1)
            row_indices.append(torch.full((end - start,), i, device=device, dtype=torch.long))
            col_indices.append(torch.arange(start, end, device=device, dtype=torch.long))
        
        row_idx = torch.cat(row_indices)
        col_idx = torch.cat(col_indices)
        
        # Create sparse-like mask with batch dimension
        # Start with all-masked, then unmask positions within the sliding window
        mask = torch.full((1, L, L), float("-inf"), device=device, dtype=torch.float)
        mask[0, row_idx, col_idx] = 0.0
        
        # Handle global tokens: global tokens can attend to all positions
        if global_tokens_mask is not None:
            if global_tokens_mask.dim() == 1:
                global_tokens_mask = global_tokens_mask.unsqueeze(0)
            B = global_tokens_mask.size(0)
            
            # Expand mask to batch dimension for per-sample global tokens
            if B > 1:
                mask = mask.expand(B, -1, -1).clone()
            
            # Global tokens attend everywhere (row = global, col = any)
            for b in range(B):
                global_pos = torch.where(global_tokens_mask[b])[0]
                if len(global_pos) > 0:
                    for g in global_pos:
                        mask[b, g, :] = 0.0  # Allow full attention from global token
                        mask[b, :, g] = 0.0  # Allow all to attend to global token
        
        return mask

    def _create_global_attention_mask(
        self,
        L: int,
        B: int,
        device: torch.device,
        global_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a mask for Longformer-style global attention.
        Global tokens attend to all positions, non-global tokens attend only to globals.
        
        Args:
            L: sequence length
            B: batch size
            device: device
            global_tokens_mask: (B, L) boolean where True = global token
        Returns:
            mask: (B, L, L) additive mask
        """
        # Start with all -inf (no attention)
        mask = torch.full((B, L, L), float("-inf"), device=device)
        
        # Global tokens attend to all positions
        global_mask = global_tokens_mask.unsqueeze(-1)  # (B, L, 1)
        mask = mask.masked_fill(global_mask, 0.0)  # Global -> all = allowed
        
        # All tokens attend to global tokens
        mask = mask.masked_fill(global_mask.unsqueeze(1), 0.0)  # All -> global = allowed
        
        return mask


class FeedForwardNetwork(nn.Module):
    """Standard two‑layer feed‑forward network with GELU activation."""

    def __init__(
        self,
        hidden_dim: int = 768,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    A single transformer encoder layer (self‑attention + feed‑forward) with pre‑layer norm.
    Supports hybrid SWA (Sliding Window Attention) and Full Attention layers.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_flash: bool = True,
        use_xformers: bool = True,
        use_sliding_window: bool = False,
        sliding_window: int = 512,
        enable_hybrid_swa: bool = False,
        hybrid_full_attn_interval: int = 4,
        use_adaptive_swa_gating: bool = False,
        num_global_heads: int = 0,
        use_rel_pos_bias: bool = False,
        max_rel_pos: int = 512,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash=use_flash,
            use_xformers=use_xformers,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            enable_hybrid_swa=enable_hybrid_swa,
            hybrid_full_attn_interval=hybrid_full_attn_interval,
            use_adaptive_swa_gating=use_adaptive_swa_gating,
            num_global_heads=num_global_heads,
            use_rel_pos_bias=use_rel_pos_bias,
            max_rel_pos=max_rel_pos,
        )
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        layer_idx: int = 0,
        global_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self‑attention with residual
        residual = hidden_states
        x = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(
            x,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            causal=causal,
            layer_idx=layer_idx,
            global_tokens_mask=global_tokens_mask,
        )
        hidden_states = residual + self.dropout(attn_out)

        # Feed‑forward with residual
        residual = hidden_states
        x = self.norm2(hidden_states)
        ffn_out = self.ffn(x)
        hidden_states = residual + self.dropout(ffn_out)
        return hidden_states


class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderLayer layers with optional hybrid SWA.
    
    In hybrid mode (enable_hybrid_swa=True), the encoder alternates between:
    - SWA layers: Fast local feature extraction with linear memory
    - Full Attention layers: Global context routing every N layers
    
    This is the Mistral/Granite approach for balancing local and global context.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_dim: int = 768,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_flash: bool = True,
        use_xformers: bool = True,
        use_sliding_window: bool = False,
        sliding_window: int = 512,
        enable_hybrid_swa: bool = False,
        hybrid_full_attn_interval: int = 4,
        use_adaptive_swa_gating: bool = False,
        num_global_heads: int = 0,
        use_rel_pos_bias: bool = False,
        max_rel_pos: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                use_flash=use_flash,
                use_xformers=use_xformers,
                use_sliding_window=use_sliding_window,
                sliding_window=sliding_window,
                enable_hybrid_swa=enable_hybrid_swa,
                hybrid_full_attn_interval=hybrid_full_attn_interval,
                use_adaptive_swa_gating=use_adaptive_swa_gating,
                num_global_heads=num_global_heads,
                use_rel_pos_bias=use_rel_pos_bias,
                max_rel_pos=max_rel_pos,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        output_hidden_states: bool = False,
        global_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            global_tokens_mask: (B, L) boolean mask where True = global token
                (e.g., [CLS] token) that attends to all positions globally.
        Returns:
            last_hidden_state: (B, L, D)
            all_hidden_states: tuple of (B, L, D) if output_hidden_states=True
        """
        all_hidden_states = () if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                causal=causal,
                layer_idx=layer_idx,
                global_tokens_mask=global_tokens_mask,
            )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            return hidden_states, all_hidden_states
        return hidden_states


class BERTEncoder(nn.Module):
    """
    Complete BERT‑like encoder with embedding layer and transformer stack.
    Supports hybrid SWA (Sliding Window Attention) via TransformerEncoder.
    """

    def __init__(
        self,
        vocab_size: int = 30522,  # not used for EEG, but kept for compatibility
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_flash: bool = True,
        use_xformers: bool = True,
        use_sliding_window: bool = False,
        sliding_window: int = 512,
        enable_hybrid_swa: bool = False,
        hybrid_full_attn_interval: int = 4,
        use_adaptive_swa_gating: bool = False,
        num_global_heads: int = 0,
        use_rel_pos_bias: bool = False,
        max_rel_pos: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_sliding_window = use_sliding_window
        self.enable_hybrid_swa = enable_hybrid_swa
        
        # Token embedding (not used if we pass inputs_embeds)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        # Position embedding (not used if RoPE is enabled)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_dim)
        self.token_type_embedding = nn.Embedding(2, hidden_dim)  # segment IDs
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            use_flash=use_flash,
            use_xformers=use_xformers,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            enable_hybrid_swa=enable_hybrid_swa,
            hybrid_full_attn_interval=hybrid_full_attn_interval,
            use_adaptive_swa_gating=use_adaptive_swa_gating,
            num_global_heads=num_global_heads,
            use_rel_pos_bias=use_rel_pos_bias,
            max_rel_pos=max_rel_pos,
        )
        # Pooler (like BERT's [CLS] linear+tanh)
        self.pooler = nn.Linear(hidden_dim, hidden_dim)
        self.pooler_activation = nn.Tanh()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        causal: bool = False,
        global_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            inputs_embeds: (B, L, D) optional pre‑computed embeddings (for EEG tokens).
            global_tokens_mask: (B, L) boolean mask where True = global token (e.g., [CLS]).
        Returns:
            last_hidden_state: (B, L, D)
            pooler_output: (B, D) (optional, only if not inputs_embeds)
            all_hidden_states: optional tuple
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            B, L = input_ids.shape
        else:
            B, L, D = inputs_embeds.shape

        # Prepare position IDs
        if position_ids is None:
            position_ids = torch.arange(L, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(B, L)
        # Prepare token type IDs (default to 0)
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, L, dtype=torch.long, device=inputs_embeds.device)

        # Compute embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        token_type_embeds = self.token_type_embedding(token_type_ids)

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Transformer
        # If global_tokens_mask is not provided but we have SWA enabled,
        # default to treating position 0 as a global token ([CLS]-style pooling)
        if global_tokens_mask is None and self.use_sliding_window:
            # Default: treat first token as global for pooling
            global_tokens_mask = torch.zeros(B, L, dtype=torch.bool, device=inputs_embeds.device)
            global_tokens_mask[:, 0] = True  # [CLS] token at position 0 is global
        
        transformer_output = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            key_padding_mask=attention_mask,  # same mask for padding
            causal=causal,
            output_hidden_states=output_hidden_states,
            global_tokens_mask=global_tokens_mask,
        )
        if output_hidden_states:
            last_hidden_state, all_hidden_states = transformer_output
        else:
            last_hidden_state = transformer_output
            all_hidden_states = None

        # Pooler (only if we have [CLS] token at position 0)
        # For EEG we may not have a CLS token; you can adjust accordingly.
        pooler_output = self.pooler_activation(self.pooler(last_hidden_state[:, 0]))

        output = (last_hidden_state, pooler_output)
        if output_hidden_states:
            output = output + (all_hidden_states,)
        if output_attentions:
            # Not supported with flash/xformers; you could disable them to get attentions
            output = output + (None,)  # placeholder
        if len(output) == 1:
            return output[0]
        return output