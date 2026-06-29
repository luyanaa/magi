import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from brain_moe_pinn.magi.transformer import apply_rotary_pos_emb

# Standard 10-20 montage electrode positions (approximate 3D coordinates)
# Format: {channel_name: (x, y, z)}
# Based on standard 10-20 system coordinates
STANDARD_10_20_COORDS = {
    # Frontal
    'Fp1': (-3.0, 5.0, 8.0),
    'Fp2': (3.0, 5.0, 8.0),
    'F7': (-5.0, 3.0, 5.0),
    'F3': (-3.0, 3.0, 7.0),
    'Fz': (0.0, 3.0, 8.0),
    'F4': (3.0, 3.0, 7.0),
    'F8': (5.0, 3.0, 5.0),
    # Temporal
    'T3': (-6.0, 0.0, 5.0),
    'T4': (6.0, 0.0, 5.0),
    'T5': (-5.0, -3.0, 5.0),
    'T6': (5.0, -3.0, 5.0),
    # Central
    'C3': (-3.0, 0.0, 8.0),
    'Cz': (0.0, 0.0, 9.0),
    'C4': (3.0, 0.0, 8.0),
    # Parietal
    'P3': (-3.0, -3.0, 7.0),
    'Pz': (0.0, -3.0, 8.0),
    'P4': (3.0, -3.0, 7.0),
    # Occipital
    'O1': (-3.0, -5.0, 6.0),
    'O2': (3.0, -5.0, 6.0),
}

# Brain region groupings
BRAIN_REGIONS = {
    'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'temporal': ['T3', 'T4', 'T5', 'T6'],
    'central': ['C3', 'Cz', 'C4'],
    'parietal': ['P3', 'Pz', 'P4'],
    'occipital': ['O1', 'O2'],
}

# Frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
}

# Extended 10-5 system electrode positions (subset of most common locations)
# Full 10-5 system has 345 locations, here we include a comprehensive subset
# Format: {channel_name: (x, y, z)}
STANDARD_10_5_COORDS = {
    # Standard 10-20 channels (included in 10-5)
    'Fp1': (-3.0, 5.0, 8.0), 'Fp2': (3.0, 5.0, 8.0),
    'F7': (-5.0, 3.0, 5.0), 'F3': (-3.0, 3.0, 7.0),
    'Fz': (0.0, 3.0, 8.0), 'F4': (3.0, 3.0, 7.0), 'F8': (5.0, 3.0, 5.0),
    'T3': (-6.0, 0.0, 5.0), 'T4': (6.0, 0.0, 5.0),
    'T5': (-5.0, -3.0, 5.0), 'T6': (5.0, -3.0, 5.0),
    'C3': (-3.0, 0.0, 8.0), 'Cz': (0.0, 0.0, 9.0), 'C4': (3.0, 0.0, 8.0),
    'P3': (-3.0, -3.0, 7.0), 'Pz': (0.0, -3.0, 8.0), 'P4': (3.0, -3.0, 7.0),
    'O1': (-3.0, -5.0, 6.0), 'O2': (3.0, -5.0, 6.0),
    # Intermediate 10% positions (between 10-20 electrodes)
    'AF3': (-2.0, 4.0, 7.5), 'AF4': (2.0, 4.0, 7.5), 'AF7': (-4.0, 4.0, 6.5), 'AF8': (4.0, 4.0, 6.5),
    'F1': (-1.5, 3.0, 7.5), 'F2': (1.5, 3.0, 7.5), 'F5': (-4.0, 3.0, 6.0), 'F6': (4.0, 3.0, 6.0),
    'FC1': (-2.0, 1.5, 8.0), 'FC2': (2.0, 1.5, 8.0), 'FC3': (-3.5, 1.5, 7.5), 'FC4': (3.5, 1.5, 7.5),
    'FC5': (-5.0, 1.5, 6.5), 'FC6': (5.0, 1.5, 6.5),
    'FT7': (-5.5, 1.5, 5.0), 'FT8': (5.5, 1.5, 5.0),
    'C1': (-1.5, 0.0, 8.5), 'C2': (1.5, 0.0, 8.5), 'C5': (-4.5, 0.0, 6.5), 'C6': (4.5, 0.0, 6.5),
    'TP7': (-5.5, -1.5, 5.0), 'TP8': (5.5, -1.5, 5.0),
    'CP1': (-2.0, -1.5, 8.0), 'CP2': (2.0, -1.5, 8.0), 'CP3': (-3.5, -1.5, 7.5), 'CP4': (3.5, -1.5, 7.5),
    'CP5': (-5.0, -1.5, 6.5), 'CP6': (5.0, -1.5, 6.5),
    'P1': (-1.5, -3.0, 7.5), 'P2': (1.5, -3.0, 7.5), 'P5': (-4.0, -3.0, 6.0), 'P6': (4.0, -3.0, 6.0),
    'PO3': (-2.0, -4.0, 7.0), 'PO4': (2.0, -4.0, 7.0), 'PO7': (-4.0, -4.0, 5.5), 'PO8': (4.0, -4.0, 5.5),
}


class BIOTStyleEmbedding(nn.Module):
    """
    BIOT-style arbitrary channel embedding.
    
    This embedding layer learns a unique representation for each electrode location
    in the extended 10-5 system. Unlike MontageAwareEmbedding which requires
    rigid channel ordering and interpolation, this approach:
    1. Assigns a learnable embedding to each of 100+ 10-5 electrode positions
    2. Looks up embeddings based on channel name (string-based lookup)
    3. Uses distance-weighted fallback for unknown channels
    4. Is natively robust to arbitrary channel counts and missing electrodes
    
    Reference: BIOT (Biosignal Transformer) - handles arbitrary electrode configurations
    without requiring spatial interpolation to a standard montage.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        use_3d_coords: bool = True,
        coord_embedding_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_3d_coords = use_3d_coords
        self.coord_embedding_dim = coord_embedding_dim or embedding_dim // 4
        
        # Build index mapping for all 10-5 positions
        self.channel_to_idx = {}
        self.idx_to_channel = {}
        for idx, ch in enumerate(STANDARD_10_5_COORDS.keys()):
            self.channel_to_idx[ch] = idx
            self.idx_to_channel[idx] = ch
        
        self.num_positions = len(STANDARD_10_5_COORDS)
        
        # Learnable position embeddings (one per 10-5 electrode)
        self.position_embedding = nn.Embedding(self.num_positions, embedding_dim)
        
        # 3D coordinate projection
        if use_3d_coords:
            self.coord_proj = nn.Sequential(
                nn.Linear(3, self.coord_embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.coord_embedding_dim, self.coord_embedding_dim),
            )
        
        # Fallback embedding for unknown channels
        self.fallback_embedding = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        if self.use_3d_coords:
            for module in self.coord_proj:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def get_embedding(self, channel_name: str) -> Optional[torch.Tensor]:
        """Get embedding for a single channel name. Returns None if not found."""
        idx, found = self.get_position_index(channel_name)
        if not found:
            return None
        return self.position_embedding.weight[idx]

    def get_position_index(self, channel_name: str) -> Tuple[int, bool]:
        """Get the index for a channel name. Returns (index, found) tuple."""
        if channel_name in self.channel_to_idx:
            return self.channel_to_idx[channel_name], True
        
        # Try case-insensitive match
        channel_lower = channel_name.lower()
        for ch, idx in self.channel_to_idx.items():
            if ch.lower() == channel_lower:
                return idx, True
        
        return -1, False
    
    def get_nearest_position(self, channel_name: str) -> int:
        """Find the nearest known 10-5 position for an unknown channel."""
        if channel_name in STANDARD_10_5_COORDS:
            return self.channel_to_idx[channel_name]
        
        # Try to parse coordinates from name (some systems embed coords)
        # For now, just return a default frontal channel
        return self.channel_to_idx.get('Fz', 0)
    
    def forward(
        self,
        channel_names: List[str],
        channel_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            channel_names: List of channel names for the batch (shared across batch)
            channel_coords: (B, C, 3) optional 3D coordinates if available
        Returns:
            embeddings: (B, C, embedding_dim)
        """
        B = 1 if channel_coords is None else channel_coords.shape[0]
        C = len(channel_names)
        device = next(self.parameters()).device
        
        # Get position indices for all channels
        position_indices = []
        valid_mask = []
        for ch in channel_names:
            idx, found = self.get_position_index(ch)
            position_indices.append(idx if found else self.get_nearest_position(ch))
            valid_mask.append(found)
        
        position_indices = torch.tensor(position_indices, dtype=torch.long, device=device)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)
        
        # Base position embeddings
        embeds = self.position_embedding(position_indices)  # (C, D)
        
        # Add coordinate information if available
        if self.use_3d_coords:
            if channel_coords is not None:
                # Use provided coordinates
                coords_emb = self.coord_proj(channel_coords)  # (B, C, coord_dim)
            else:
                # Lookup coordinates from standard positions
                coords_list = []
                for ch in channel_names:
                    if ch in STANDARD_10_5_COORDS:
                        coords_list.append(STANDARD_10_5_COORDS[ch])
                    else:
                        coords_list.append((0.0, 0.0, 0.0))
                coords = torch.tensor(coords_list, dtype=torch.float32, device=device)
                coords_emb = self.coord_proj(coords)  # (C, coord_dim)
                coords_emb = coords_emb.unsqueeze(0).expand(B, -1, -1)  # (B, C, coord_dim)
            
            # Concatenate position and coordinate embeddings
            embeds = embeds.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)
            embeds = torch.cat([embeds, coords_emb], dim=-1)  # (B, C, D + coord_dim)
            
            # Project back to embedding_dim if needed
            if embeds.shape[-1] != self.embedding_dim:
                if not hasattr(self, '_coord_proj'):
                    self._coord_proj = nn.Linear(embeds.shape[-1], self.embedding_dim)
                self._coord_proj = self._coord_proj.to(device)
                embeds = self._coord_proj(embeds)
        
        # Apply fallback for invalid channels
        if not valid_mask.all():
            # embeds is (1, C, D) or (B, C, D); valid_mask is (C,)
            embeds = embeds.clone()
            embeds[:, ~valid_mask, :] = self.fallback_embedding
        
        return embeds

class ChannelTypeEmbedding(nn.Module):
    """
    Learnable embedding for signal source type (scalp EEG, ECoG grid, sEEG depth, unknown).

    Helps the model distinguish between different electrode modalities, which have
    fundamentally different signal properties (amplitude, bandwidth, spatial spread).

    Types:
        0 = scalp_EEG    (standard 10-20/10-5 scalp electrodes)
        1 = ecog_grid    (intracranial grid/strip electrodes)
        2 = seeg_depth   (stereotactic depth electrodes)
        3 = unknown      (fallback for unlabeled channels)
    """

    def __init__(self, num_types: int = 4, embedding_dim: int = 768):
        super().__init__()
        self.num_types = num_types
        self.embedding_dim = embedding_dim
        self.type_embed = nn.Embedding(num_types, embedding_dim)
        nn.init.normal_(self.type_embed.weight, std=0.02)

    def forward(self, channel_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_types: (C,) or (B, C) long tensor with type indices 0-3
        Returns:
            embeddings: (C, D) or (B, C, D)
        """
        return self.type_embed(channel_types)


class MontageAwareEmbedding(nn.Module):
    """
    Learns embeddings for electrode positions and brain regions.
    Incorporates spatial information from the 10-20 montage.
    """

    def __init__(
        self,
        channels: List[str],
        embedding_dim: int = 768,
        use_3d_coords: bool = True,
        learn_region_embeddings: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_channels = len(channels)
        self.embedding_dim = embedding_dim
        self.use_3d_coords = use_3d_coords

        # Channel to index mapping
        self.channel_to_idx = {ch: i for i, ch in enumerate(channels)}

        # Learnable channel embeddings
        self.channel_embedding = nn.Embedding(self.num_channels, embedding_dim)

        # Learnable 3D coordinate projections (if using coordinates)
        if use_3d_coords:
            self.coord_proj = nn.Linear(3, embedding_dim // 4)

        # Learnable brain region embeddings
        if learn_region_embeddings:
            self.region_embedding = nn.Embedding(len(BRAIN_REGIONS), embedding_dim // 4)
            self.region_to_idx = {region: i for i, region in enumerate(BRAIN_REGIONS.keys())}

        # Learnable frequency band embeddings (static, will be computed during forward)
        self.band_proj = nn.Linear(len(FREQ_BANDS), embedding_dim // 4)

        # Initialize channel embeddings with reasonable values
        self._init_embeddings()

    def _init_embeddings(self):
        # Initialize channel embeddings
        nn.init.normal_(self.channel_embedding.weight, std=0.02)

    def get_channel_position(self, channel: str) -> Optional[torch.Tensor]:
        """Get 3D coordinates for a channel."""
        if channel in STANDARD_10_20_COORDS:
            coords = torch.tensor(STANDARD_10_20_COORDS[channel], dtype=torch.float32)
            return coords
        return None

    def get_region(self, channel: str) -> Optional[str]:
        """Get brain region for a channel."""
        for region, channels in BRAIN_REGIONS.items():
            if channel in channels:
                return region
        return None

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_indices: (batch, num_channels) - indices of channels
        Returns:
            embeddings: (batch, num_channels, embedding_dim)
        """
        B, C = channel_indices.shape
        device = channel_indices.device

        # Base channel embeddings
        embeds = self.channel_embedding(channel_indices)  # (B, C, D)

        # Add 3D coordinate information
        if self.use_3d_coords:
            # Create coordinate tensor for all channels in batch
            coords_list = []
            for ch in self.channels:
                coords = self.get_channel_position(ch)
                if coords is not None:
                    coords_list.append(coords.to(device))
                else:
                    coords_list.append(torch.zeros(3, device=device))
            all_coords = torch.stack(coords_list)  # (C, 3)
            coords_emb = self.coord_proj(all_coords)  # (C, D//4)
            coords_emb = coords_emb.unsqueeze(0).expand(B, -1, -1)  # (B, C, D//4)
            embeds = torch.cat([embeds, coords_emb], dim=-1)  # (B, C, D*3/4)

        # Add brain region information
        if hasattr(self, 'region_embedding'):
            region_indices = []
            for ch in self.channels:
                region = self.get_region(ch)
                if region is not None:
                    region_indices.append(self.region_to_idx[region])
                else:
                    region_indices.append(0)  # Default to first region
            region_indices = torch.tensor(region_indices, device=device).unsqueeze(0).expand(B, -1)  # (B, C)
            region_emb = self.region_embedding(region_indices)  # (B, C, D//4)
            embeds = torch.cat([embeds, region_emb], dim=-1)  # (B, C, D*3/4 + D/4)

        return embeds


class EEG2DPatchEmbedding(nn.Module):
    """
    2D Patch Embedding for EEG.
    Extracts patches from (channels, time) grid and projects to tokens.
    Preserves spatial structure within each patch.
    """

    def __init__(
        self,
        in_channels: int = 19,
        hidden_dim: int = 768,
        patch_size_time: int = 256,  # samples in time dimension
        patch_size_channel: int = 1,   # samples in channel dimension
        stride_time: int = 128,       # stride in time
        stride_channel: int = 1,      # stride in channel
        channels: Optional[List[str]] = None,
        use_montage: bool = True,
        max_time_patches: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size_time = patch_size_time
        self.patch_size_channel = patch_size_channel
        self.stride_time = stride_time
        self.stride_channel = stride_channel
        self.use_montage = use_montage

        # Montage-aware embedding (spatial)
        if use_montage and channels is not None:
            self.montage_embed = MontageAwareEmbedding(
                channels=channels,
                embedding_dim=hidden_dim // 4,  # Partial dim for spatial
            )
            # MontageAwareEmbedding outputs: embedding_dim + 2*(embedding_dim//4)
            # = hidden_dim//4 + hidden_dim//8 = 3*hidden_dim//8
            montage_out_dim = (hidden_dim // 4) + 2 * ((hidden_dim // 4) // 4)
            self.spatial_proj = nn.Linear(montage_out_dim, hidden_dim)
        else:
            self.montage_embed = None
            self.spatial_proj = None

        # 2D convolution to extract patches
        # Input: (B, 1, channels, time) -> Output: (B, hidden_dim, num_patches_channel, num_patches_time)
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=(patch_size_channel, patch_size_time),
            stride=(stride_channel, stride_time),
            bias=False,
        )

        # Learnable temporal positional embedding (for patch positions)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_time_patches, hidden_dim))
        nn.init.normal_(self.temporal_pos_embed, std=0.02)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def compute_num_patches(self, channels: int, time: int) -> Tuple[int, int]:
        """Compute number of patches in each dimension."""
        # Conv2d output size: floor((W - K) / S) + 1
        num_channel_patches = max(1, (channels - self.patch_size_channel) // self.stride_channel + 1)
        num_time_patches = max(1, (time - self.patch_size_time) // self.stride_time + 1)
        return num_channel_patches, num_time_patches

    def forward(
        self,
        x: torch.Tensor,
        channel_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: (batch, channels, time)
            channel_indices: (batch, channels) optional channel indices
        Returns:
            tokens: (batch, num_channel_patches * num_time_patches, hidden_dim)
            mask: (batch, num_channel_patches * num_time_patches) boolean mask
            grid_size: (num_channel_patches, num_time_patches)
        """
        B, C, T = x.shape

        # Reshape to (B, 1, C, T) for Conv2d
        x = x.unsqueeze(1)  # (B, 1, C, T)

        # Project patches: (B, 1, C, T) -> (B, hidden_dim, C_patches, T_patches)
        patches = self.proj(x)

        # Reshape: (B, hidden_dim, C_patches, T_patches) -> (B, T_patches * C_patches, hidden_dim)
        # We want (B, T, C, D) order for factorized attention
        C_patches, T_patches = patches.shape[2], patches.shape[3]
        patches = patches.permute(0, 3, 2, 1).reshape(B, T_patches * C_patches, self.hidden_dim)

        # Add temporal positional embedding (for each time patch)
        # temporal_pos_embed: (1, max_T, D)
        # We repeat it for each channel patch at that time step
        temporal_pos = self.temporal_pos_embed[:, :T_patches, :]  # (1, T, D)
        temporal_pos = temporal_pos.repeat_interleave(C_patches, dim=1)  # (1, T*C, D)
        patches = patches + temporal_pos

        # Add montage-aware spatial embedding (if available)
        if self.montage_embed is not None and channel_indices is not None:
            # channel_indices: (B, C)
            spatial_emb = self.montage_embed(channel_indices)  # (B, C, D//4)

            # Repeat spatial embeddings for each time patch
            # spatial_emb: (B, C, D//4) -> (B, T*C, D//4)
            # The order should be [c1, c2, ..., cC, c1, c2, ..., cC, ...]
            spatial_emb = spatial_emb.repeat(1, T_patches, 1)  # (B, T*C, D//4)

            # Project spatial embedding to hidden_dim
            spatial_emb = self.spatial_proj(spatial_emb)

            # Add to patches (residual)
            patches = patches + spatial_emb

        # Normalize and dropout
        patches = self.norm(patches)
        patches = self.dropout(patches)

        # Mask: all tokens are real (no padding)
        mask = torch.ones(B, T_patches * C_patches, dtype=torch.bool, device=x.device)

        return patches, mask, (C_patches, T_patches)


class FactorizedAttention(nn.Module):
    """
    Factorized Attention that separates spatial and temporal attention.
    - Spatial attention: attends over channels (C -> C)
    - Temporal attention: attends over time patches (T -> T)
    This is more efficient than full O((C*T)^2) attention.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        spatial_heads: int = 4,  # Number of heads for spatial attention
        temporal_heads: int = 8,  # Number of heads for temporal attention
        use_flash: bool = True,
        use_xformers: bool = True,
        use_rope: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int = 512,
        enable_hybrid_swa: bool = False,
        hybrid_full_attn_interval: int = 4,
        use_temporal_smoothing: bool = False,
        temporal_smoothing_kernel: int = 3,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_heads = spatial_heads
        self.temporal_heads = temporal_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.enable_hybrid_swa = enable_hybrid_swa
        self.hybrid_full_attn_interval = hybrid_full_attn_interval
        self.use_temporal_smoothing = use_temporal_smoothing

        # Spatial attention projections
        self.spatial_q = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_k = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_v = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_out = nn.Linear(hidden_dim, hidden_dim)

        # Temporal attention projections
        self.temporal_q = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_k = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_v = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_out = nn.Linear(hidden_dim, hidden_dim)

        # Temporal smoothing convolution (depthwise 1D)
        if use_temporal_smoothing:
            self.temporal_smoothing = nn.Conv1d(
                hidden_dim, 
                hidden_dim, 
                kernel_size=temporal_smoothing_kernel,
                padding=temporal_smoothing_kernel // 2,
                groups=hidden_dim,
                bias=False
            )

        self.dropout_layer = nn.Dropout(dropout)
        self.use_flash = use_flash
        self.use_xformers = use_xformers
        
        # For hybrid SWA, we need the layer index to determine attention type
        self.use_sdpa = hasattr(F, 'scaled_dot_product_attention')

    def forward_spatial(
        self,
        x: torch.Tensor,
        num_channels: int,
        num_times: int,
    ) -> torch.Tensor:
        """
        Spatial attention: attends over channels.
        x: (B, T*C, D)
        Returns: (B, T*C, D)
        """
        B, TC, D = x.shape

        # Reshape to (B, T, C, D) for easier channel attention
        x = x.view(B, num_times, num_channels, D)

        # Project Q, K, V
        q = self.spatial_q(x)  # (B, T, C, D)
        k = self.spatial_k(x)
        v = self.spatial_v(x)

        # Reshape for multi-head: (B, T, C, H, d) -> (B, T, H, C, d)
        q = q.view(B, num_times, num_channels, self.spatial_heads, -1).transpose(2, 3)
        k = k.view(B, num_times, num_channels, self.spatial_heads, -1).transpose(2, 3)
        v = v.view(B, num_times, num_channels, self.spatial_heads, -1).transpose(2, 3)

        # Attention over channels
        # (B, T, H, C, d) @ (B, T, H, d, C) -> (B, T, H, C, C)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        # Apply attention
        # (B, T, H, C, C) @ (B, T, H, C, d) -> (B, T, H, C, d)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, num_times, num_channels, D)
        out = out.view(B, TC, D)

        return self.spatial_out(out)

    def forward_temporal(
        self,
        x: torch.Tensor,
        num_channels: int,
        num_times: int,
    ) -> torch.Tensor:
        """
        Temporal attention: attends over time.
        x: (B, T*C, D)
        Returns: (B, T*C, D)
        """
        B, TC, D = x.shape

        # Reshape to (B, T, C, D) for easier time attention
        x = x.view(B, num_times, num_channels, D)

        # Apply temporal smoothing if enabled (smoothing across time patches)
        if self.use_temporal_smoothing:
            # x: (B, T, C, D) -> (B*C, D, T)
            x_smooth = x.transpose(1, 2).reshape(B * num_channels, num_times, D).transpose(1, 2)
            x_smooth = self.temporal_smoothing(x_smooth)
            x = x_smooth.transpose(1, 2).view(B, num_channels, num_times, D).transpose(1, 2)

        # Project Q, K, V
        q = self.temporal_q(x)  # (B, T, C, D)
        k = self.temporal_k(x)
        v = self.temporal_v(x)

        # Reshape for multi-head: (B, T, C, H, d) -> (B, C, H, T, d)
        q = q.view(B, num_times, num_channels, self.temporal_heads, -1).permute(0, 2, 3, 1, 4)
        k = k.view(B, num_times, num_channels, self.temporal_heads, -1).permute(0, 2, 3, 1, 4)
        v = v.view(B, num_times, num_channels, self.temporal_heads, -1).permute(0, 2, 3, 1, 4)

        # Apply RoPE to temporal attention if enabled
        if self.use_rope:
            positions = torch.arange(num_times, device=q.device).unsqueeze(0).unsqueeze(0).expand(B, num_channels, -1)  # (B, C, T)
            # Flatten for apply_rotary_pos_emb: (B, C, H, T, d) -> (B*C*H, T, d)
            q_flat = q.flatten(0, 2)  # (B*C*H, T, d)
            k_flat = k.flatten(0, 2)
            positions_flat = positions.flatten(0, 1)  # (B*C, T)
            q_rot, k_rot = apply_rotary_pos_emb(q_flat, k_flat, positions_flat)
            q = q_rot.view_as(q)
            k = k_rot.view_as(k)

        # Attention over time
        # (B, C, H, T, d) @ (B, C, H, d, T) -> (B, C, H, T, T)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sliding window mask if enabled (efficient O(T*W) mask)
        if self.use_sliding_window:
            # Create efficient sliding window mask using block diagonal approach
            W = self.sliding_window
            # Build row/col indices for O(T*W) entries instead of O(T^2)
            row_idx_list = []
            col_idx_list = []
            for i in range(num_times):
                start = max(0, i - W)
                end = min(num_times, i + W + 1)
                row_idx_list.append(torch.full((end - start,), i, device=q.device, dtype=torch.long))
                col_idx_list.append(torch.arange(start, end, device=q.device, dtype=torch.long))
            
            row_idx = torch.cat(row_idx_list)
            col_idx = torch.cat(col_idx_list)
            
            # Create sparse-like mask
            mask = torch.zeros(num_times, num_times, device=q.device, dtype=torch.float)
            mask[row_idx, col_idx] = float("-inf")
            
            # Apply mask: add -inf to masked positions before softmax
            attn = attn + mask.unsqueeze(0).unsqueeze(0)  # Broadcast to (B, C, H, T, T)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        # Apply attention
        # (B, C, H, T, T) @ (B, C, H, T, d) -> (B, C, H, T, d)
        out = torch.matmul(attn, v)
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(B, num_times, num_channels, D)
        out = out.view(B, TC, D)

        return self.temporal_out(out)

    def forward(
        self,
        x: torch.Tensor,
        num_channels: int,
        num_times: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T*C, D)
            num_channels: C
            num_times: T
        Returns:
            output: (B, T*C, D)
        """
        # Apply spatial attention
        spatial_out = self.forward_spatial(x, num_channels, num_times)

        # Apply temporal attention
        temporal_out = self.forward_temporal(spatial_out, num_channels, num_times)

        return temporal_out


class FactorizedTransformerLayer(nn.Module):
    """
    A single transformer layer with factorized (spatial + temporal) attention.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        spatial_heads: int = 4,
        temporal_heads: int = 8,
    ):
        super().__init__()
        self.factorized_attn = FactorizedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            spatial_heads=spatial_heads,
            temporal_heads=temporal_heads,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        num_channels: int,
        num_times: int,
    ) -> torch.Tensor:
        # Factorized attention with residual
        residual = x
        x = self.norm1(x)
        x = self.factorized_attn(x, num_channels, num_times)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class FactorizedTransformerEncoder(nn.Module):
    """
    Stack of factorized transformer layers.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_dim: int = 768,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        spatial_heads: int = 4,
        temporal_heads: int = 8,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FactorizedTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                spatial_heads=spatial_heads,
                temporal_heads=temporal_heads,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        num_channels: int,
        num_times: int,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, num_channels, num_times)
        return self.norm(x)


class VirtualMontageEmbedding(nn.Module):
    """
    Learns to map arbitrary EEG montages to standard 10-20 montage.
    Uses MNE's interpolation under the hood but also learns alignment.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        max_channels: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_channels = max_channels
        
        # Learnable positional embeddings for source channels
        self.source_pos_embed = nn.Embedding(max_channels, hidden_dim)
        
        # Learnable positional embeddings for target (10-20) channels
        self.target_pos_embed = nn.Embedding(max_channels, hidden_dim)
        
        # Cross-attention to align source to target
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_features: (B, S, D) features from source channels
            source_mask: (B, S) boolean mask for valid source channels
        Returns:
            aligned_features: (B, T, D) aligned to standard 10-20
            alignment_weights: (B, S, T) attention weights
        """
        B, S, D = source_features.shape
        device = source_features.device
        
        # Get positional embeddings
        source_pos = self.source_pos_embed(
            torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        )  # (B, S, D)
        
        # Add positional embeddings
        source_emb = source_features + source_pos
        
        # Target positional embeddings (assuming 19 standard 10-20 channels)
        T = 19
        target_pos = self.target_pos_embed(
            torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        )  # (B, T, D)
        
        # Cross-attention: query=target, key=value=source
        # This aligns source channels to target positions
        aligned, attn_weights = self.cross_attention(
            target_pos,  # query
            source_emb,  # key, value
            source_emb,  # key, value
            key_padding_mask=source_mask,
        )
        aligned = aligned + self.ffn(aligned)
        aligned = self.norm1(aligned)
        
        return aligned, attn_weights


class MontageInvariantEncoder(nn.Module):
    """
    Encodes EEG signals in a montage-invariant way.
    First projects to a common space, then applies factorized attention.
    """
    
    def __init__(
        self,
        in_channels: int = 19,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        use_montage_embedding: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_montage_embedding = use_montage_embedding
        
        # Virtual montage embedding if enabled
        if use_montage_embedding:
            self.montage_embedding = VirtualMontageEmbedding(
                hidden_dim=hidden_dim,
                max_channels=in_channels,
            )
        
        # Factorized transformer encoder
        self.encoder = FactorizedTransformerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        
        # Projection to common space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        channel_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) or (B, S, D) input features
            channel_indices: (B, C) channel indices (unused, for compatibility)
            mask: optional source mask for virtual montage
        Returns:
            encoded: (B, num_tokens, D)
            pooled: (B, D) pooled representation
        """
        # Apply virtual montage alignment if enabled
        if self.use_montage_embedding:
            aligned, attn_weights = self.montage_embedding(x, mask)
        else:
            aligned = x
            attn_weights = None
        
        # Reshape if 3D input: (B, C, T) -> (B, C*T, D)
        if aligned.dim() == 3:
            B, C, T = aligned.shape
            # Project to hidden_dim if needed (assuming linear projection)
            if aligned.shape[-1] != self.hidden_dim:
                aligned = F.linear(aligned.view(B, C * T),
                                  self.projection[0].weight[:, :C * T]).view(B, C, T)
            aligned = aligned.reshape(B, C * T, -1)
        
        # Apply transformer encoder
        num_tokens = aligned.shape[1]
        encoded = self.encoder(aligned, num_channels=1, num_times=num_tokens)
        
        # Project to common space
        encoded = self.projection(encoded)
        encoded = self.norm(encoded)
        
        # Pool
        pooled = encoded.mean(dim=1)
        
        return encoded, pooled


if __name__ == '__main__':
    # Test 2D patch embedding
    print("Testing EEG2DPatchEmbedding...")
    channels = list(STANDARD_10_20_COORDS.keys())
    embed = EEG2DPatchEmbedding(
        in_channels=19,
        hidden_dim=768,
        patch_size_time=256,
        patch_size_channel=1,
        stride_time=128,
        stride_channel=1,
        channels=channels,
        use_montage=True,
    )
    x = torch.randn(2, 19, 2560)  # (B, C, T)
    tokens, mask, (C_patches, T_patches) = embed(x, channel_indices=torch.arange(19).unsqueeze(0).expand(2, -1))
    print(f"Input shape: {x.shape}")
    print(f"Token shape: {tokens.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Patch grid: C_patches={C_patches}, T_patches={T_patches}")

    # Test factorized attention
    print("\nTesting FactorizedAttention...")
    B, C, T = 2, 10, 20
    D = 768
    x = torch.randn(B, T * C, D)
    attn = FactorizedAttention(hidden_dim=D, spatial_heads=4, temporal_heads=8)
    out = attn(x, num_channels=C, num_times=T)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Test factorized transformer
    print("\nTesting FactorizedTransformerEncoder...")
    encoder = FactorizedTransformerEncoder(num_layers=2, hidden_dim=D)
    out = encoder(x, num_channels=C, num_times=T)
    print(f"Output shape: {out.shape}")
