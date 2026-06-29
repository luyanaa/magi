import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional, Tuple

class EEGPatchEmbedding(nn.Module):
    """
    Convert raw EEG signal (C x T) into a sequence of token embeddings.
    Uses 1D convolutions to extract patches across time and optionally across channels.
    """

    def __init__(
        self,
        in_channels: int = 19,
        hidden_dim: int = 768,
        patch_size: int = 256,  # samples (1 second at 256Hz)
        stride: Optional[int] = None,
        use_channel_mixing: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2  # 50% overlap by default
        self.use_channel_mixing = use_channel_mixing

        # 1D convolution to project patches
        # kernel_size = patch_size, stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=self.stride,
            padding=0,
            bias=False
        )
        # Learnable per‑channel bias
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Optional channel mixer (a small MLP) to mix information across channels before patching
        if use_channel_mixing:
            self.channel_mixer = nn.Sequential(
                nn.Linear(in_channels, in_channels * 2),
                nn.GELU(),
                nn.Linear(in_channels * 2, in_channels),
                nn.Dropout(dropout),
            )
        else:
            self.channel_mixer = None

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # Layer norm (as in ViT)
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.normal_(self.bias, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            tokens: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) boolean mask indicating which tokens are real (True) vs padding (False)
        """
        B, C, T = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Optional channel mixing
        if self.channel_mixer is not None:
            # Mix across channels: (B, C, T) -> (B, T, C) -> mix -> (B, T, C) -> (B, C, T)
            x_mixed = self.channel_mixer(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + x_mixed  # residual

        # Project patches: conv1d expects (B, C, T), outputs (B, hidden_dim, L)
        patches = self.conv(x)  # (B, hidden_dim, L)
        L = patches.size(2)

        # Add bias per channel
        patches = patches + self.bias.view(1, -1, 1)

        # Transpose to sequence: (B, L, hidden_dim)
        tokens = patches.permute(0, 2, 1)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+L, hidden_dim)
        seq_len = tokens.size(1)

        # Add positional embeddings (only up to seq_len)
        tokens = tokens + self.pos_embed[:, :seq_len]

        # Layer norm & dropout
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        # Create mask: all tokens are real (no padding) because convolution length is deterministic
        mask = torch.ones(B, seq_len, dtype=torch.bool, device=x.device)

        # Return tokens, mask, and patch grid dimensions (C_patches=1, T_patches=L)
        return tokens, mask, (1, L)

    def compute_patch_length(self, input_time: int) -> int:
        """
        Compute the number of patches (excluding CLS token) for a given input length.
        """
        # Formula for conv output length: L = floor((T - kernel_size) / stride) + 1
        L = (input_time - self.patch_size) // self.stride + 1
        return max(L, 0)


class EEGMasking(nn.Module):
    """
    Advanced masking strategies for EEG: random, block, and channel-wise masking.
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        mask_token: Optional[torch.Tensor] = None,
        hidden_dim: int = 768,
        masking_type: str = "random",  # "random", "block", "channel", "hybrid"
        block_size: int = 4,  # Number of consecutive patches for block masking
        channel_ratio: float = 0.3,  # Ratio of channels to mask for channel masking
        hybrid_probs: Tuple[float, float, float] = (0.4, 0.3, 0.3),  # Probabilities for (random, block, channel)
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.masking_type = masking_type
        self.block_size = block_size
        self.channel_ratio = channel_ratio
        self.hybrid_probs = hybrid_probs

        if mask_token is None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = mask_token

    def _random_mask(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard random token masking."""
        B, L, D = tokens.shape
        device = tokens.device

        num_tokens = mask.sum(dim=1).float()
        num_mask = (num_tokens * self.mask_ratio).long()

        mask_indices = torch.zeros(B, L, dtype=torch.bool, device=device)
        restore_indices = torch.zeros(B, L, dtype=torch.bool, device=device)

        for i in range(B):
            real_indices = torch.where(mask[i])[0]
            if len(real_indices) == 0:
                continue
            perm = torch.randperm(len(real_indices), device=device)
            selected = real_indices[perm[:num_mask[i]]]
            mask_indices[i, selected] = True
            restore_indices[i, selected] = True

        return mask_indices, restore_indices

    def _block_mask(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask consecutive blocks of patches (temporal segments)."""
        B, L, D = tokens.shape
        device = tokens.device

        num_tokens = mask.sum(dim=1).float()
        num_mask = (num_tokens * self.mask_ratio).long()

        mask_indices = torch.zeros(B, L, dtype=torch.bool, device=device)
        restore_indices = torch.zeros(B, L, dtype=torch.bool, device=device)

        for i in range(B):
            real_indices = torch.where(mask[i])[0]
            if len(real_indices) == 0:
                continue
            
            # Number of blocks to mask
            num_blocks = int(num_mask[i].item()) // self.block_size
            if num_blocks == 0:
                num_blocks = 1
            
            # Get maximum valid start position for blocks
            max_start = len(real_indices) - self.block_size
            if max_start <= 0:
                # Not enough tokens for a full block, use all tokens
                mask_indices[i, real_indices] = True
                restore_indices[i, real_indices] = True
            else:
                # Randomly select num_blocks starting positions
                # Use randperm on possible start positions, not on real_indices
                possible_starts = torch.arange(max_start + 1, device=device)
                selected_indices = torch.randperm(len(possible_starts))[:num_blocks]
                selected_starts = possible_starts[selected_indices]
                
                # Sort starts to enable overlap detection
                sorted_starts, sort_order = torch.sort(selected_starts)
                
                # Greedily select non-overlapping blocks
                # A block starting at position p covers indices [p, p + block_size)
                # So a new block at q only overlaps if q < prev_end
                prev_end = -1
                for start_pos in sorted_starts:
                    if start_pos.item() < prev_end + self.block_size:
                        # This block overlaps with previous, skip it
                        continue
                    # Get the actual token indices for this block
                    block_indices = real_indices[start_pos:start_pos + self.block_size]
                    mask_indices[i, block_indices] = True
                    restore_indices[i, block_indices] = True
                    prev_end = start_pos.item() + self.block_size - 1

        return mask_indices, restore_indices

    def _channel_mask(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        num_channels: int = 19,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask entire channels (all their patches) at random time segments."""
        B, L, D = tokens.shape
        device = tokens.device
        # Estimate patches per channel (assuming L is divisible by num_channels)
        patches_per_channel = L // num_channels

        num_channels_to_mask = max(1, int(num_channels * self.channel_ratio))

        mask_indices = torch.zeros(B, L, dtype=torch.bool, device=device)
        restore_indices = torch.zeros(B, L, dtype=torch.bool, device=device)

        for i in range(B):
            # Select random channels to mask
            channels_to_mask = torch.randperm(num_channels)[:num_channels_to_mask]
            for ch in channels_to_mask:
                start = ch * patches_per_channel
                end = start + patches_per_channel
                mask_indices[i, start:end] = True
                restore_indices[i, start:end] = True

        return mask_indices, restore_indices

    def __call__(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_channels: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask a fraction of tokens, replace with mask_token.
        Args:
            tokens: (B, L, D)
            mask: (B, L) boolean, True for real tokens (optional).
            num_channels: Number of channels (for channel masking)
        Returns:
            masked_tokens: tokens after masking
            mask_indices: (B, L) boolean, True for masked positions
            restore_indices: (B, L) boolean, True for positions to predict
        """
        B, L, D = tokens.shape
        device = tokens.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        if self.masking_type == "random":
            mask_indices, restore_indices = self._random_mask(tokens, mask)
        elif self.masking_type == "block":
            mask_indices, restore_indices = self._block_mask(tokens, mask)
        elif self.masking_type == "channel":
            mask_indices, restore_indices = self._channel_mask(tokens, mask, num_channels)
        elif self.masking_type == "hybrid":
            # Randomly choose masking strategy
            r = torch.rand(1).item()
            if r < self.hybrid_probs[0]:
                mask_indices, restore_indices = self._random_mask(tokens, mask)
            elif r < self.hybrid_probs[0] + self.hybrid_probs[1]:
                mask_indices, restore_indices = self._block_mask(tokens, mask)
            else:
                mask_indices, restore_indices = self._channel_mask(tokens, mask, num_channels)
        else:
            mask_indices, restore_indices = self._random_mask(tokens, mask)

        # Replace masked tokens with mask_token
        masked_tokens = tokens.clone()
        mask_token_expanded = self.mask_token.expand(B, L, D)
        masked_tokens[mask_indices] = mask_token_expanded[mask_indices]

        return masked_tokens, mask_indices, restore_indices


if __name__ == '__main__':
    # Quick test
    embed = EEGPatchEmbedding(in_channels=19, hidden_dim=768, patch_size=256, stride=128)
    x = torch.randn(2, 19, 2560)  # 10 seconds
    tokens, mask = embed(x)
    print(f"Input shape: {x.shape}")
    print(f"Token shape: {tokens.shape}")  # (2, 1 + L, 768)
    print(f"Mask shape: {mask.shape}")
    print(f"Number of patches L = {embed.compute_patch_length(2560)}")

    # Test masking
    masking = EEGMasking(mask_ratio=0.75)
    masked, mask_idx, restore_idx = masking(tokens, mask)
    print(f"Masked shape: {masked.shape}")
    print(f"Masked count: {mask_idx.sum(dim=1)}")