"""
EEG Foundation Model v2 with Magi v2 architecture (24L × 1024d).

Key upgrades from v1:
- 24 layers × 1024 hidden dimension (vs 12L × 768d)
- RoPE (Rotary Position Embedding)
- GeGLU activation in FFN
- Alternating attention: global every 3 layers, SWA otherwise
- Pre-norm architecture
- Tiling initialization from 12L model
- Better support for ECoG/sEEG multi-modality
- Amplitude normalization for ECoG (~20× scalp EEG)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from .magi_v2 import (
    MagiV2EEGEncoder,
    create_magi_v2_from_v1,
)
from .momentum import MomentumEncoder
from .patch_embedding import EEGMasking



class EEGFoundationModelV2(nn.Module):
    """
    Complete EEG foundation model v2 with Magi v2 architecture.
    
    Combines:
        - BIOT-style arbitrary electrode embedding
        - Channel type embedding (scalp EEG / ECoG / sEEG)
        - Magi v2 transformer encoder (24L × 1024d)
        - Momentum encoder for contrastive learning
        - Two heads: masked prediction & contrastive projection
        - Adversarial subject classifier (GRL-based)
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
        max_channels: int = 256,
        patch_size_time: int = 256,
        stride_time: Optional[int] = None,
        mask_ratio: float = 0.75,
        momentum: float = 0.999,
        projection_dim: int = 256,
        use_biot_embedding: bool = True,
        use_channel_type_embed: bool = True,
        ecog_amplitude_scale: float = 20.0,
        in_channels: int = 19,
        channels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_channels = max_channels
        self.patch_size_time = patch_size_time
        self.use_biot_embedding = use_biot_embedding
        self.use_channel_type_embed = use_channel_type_embed
        self.ecog_amplitude_scale = ecog_amplitude_scale
        self.in_channels = in_channels
        
        # Store init kwargs for cloning
        self.init_kwargs = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'intermediate_dim': intermediate_dim,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps,
            'spatial_heads': spatial_heads,
            'temporal_heads': temporal_heads,
            'window_size': window_size,
            'use_rope': use_rope,
            'alternating_pattern': alternating_pattern,
            'max_channels': max_channels,
            'patch_size_time': patch_size_time,
            'stride_time': stride_time,
            'use_biot_embedding': use_biot_embedding,
            'use_channel_type_embed': use_channel_type_embed,
            'ecog_amplitude_scale': ecog_amplitude_scale,
            'in_channels': in_channels,
            'channels': channels,
            **kwargs,
        }
        
        # Magi v2 EEG encoder
        self.base_encoder = MagiV2EEGEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            spatial_heads=spatial_heads,
            temporal_heads=temporal_heads,
            window_size=window_size,
            use_rope=use_rope,
            alternating_pattern=alternating_pattern,
            max_channels=max_channels,
            patch_size_time=patch_size_time,
            stride_time=stride_time,
            use_biot_embedding=use_biot_embedding,
            use_channel_type_embed=use_channel_type_embed,
            ecog_amplitude_scale=ecog_amplitude_scale,
        )
        
        # Momentum encoder
        self.momentum_encoder = MomentumEncoder(self.base_encoder, momentum=momentum)
        
        # Masking for masked prediction
        self.masking = EEGMasking(mask_ratio=mask_ratio, hidden_dim=hidden_dim)
        
        # Masked prediction head (reconstruct original time series)
        self.mask_head = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, patch_size_time),  # Predict time series patch
        )
        
        # Contrastive projection head (for InfoNCE)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.momentum_proj_head = copy.deepcopy(self.proj_head)
        for parameter in self.momentum_proj_head.parameters():
            parameter.requires_grad = False
        
        # Adversarial subject classifier (GRL-based regularizer)
        self.subject_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1000),  # Max 1000 subjects
        )
        
        self.projection_dim = projection_dim
        self.mask_ratio = mask_ratio
        self.momentum = momentum
    
    def forward_embeddings(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token embeddings from raw EEG.
        
        Args:
            eeg: (B, C, T) raw EEG/ECoG/sEEG signals
            channel_names: List of length B, each containing C channel names
            channel_types: (B, C) integer tensor: 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
        
        Returns:
            tokens: (B, num_tokens, D) token embeddings
            pooler_output: (B, D) pooled representation
        """
        # Get encoder output
        last_hidden, pooler_output = self.base_encoder(
            eeg=eeg,
            channel_names=channel_names,
            channel_types=channel_types,
        )
        
        return last_hidden, pooler_output
    def forward_masked_prediction(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        return_targets: bool = False,
    ):
        # Metadata normalization is independent of signal values. The
        # amplitude statistics are computed from the masked support below, so
        # changing a masked patch cannot perturb visible tokens.
        prepared_eeg, prepared_names, prepared_types = (
            self.base_encoder._prepare_inputs(
                eeg, channel_names, channel_types,
                normalize_amplitude=False))
        with torch.no_grad():
            shape_tokens, channels, num_times = (
                self.base_encoder.embed_tokens(
                    prepared_eeg, channel_names=prepared_names,
                    channel_types=prepared_types,
                    normalize_amplitude=False))
            _, sampled_mask, _ = self.masking(
                shape_tokens, num_channels=channels)
        mask = self.base_encoder.expand_patch_mask(
            sampled_mask, channels, num_times)
        masked_raw = self.base_encoder.mask_raw_patches(
            prepared_eeg, sampled_mask, channels, num_times)
        if (prepared_types is not None
                and self.base_encoder.use_channel_type_embed):
            stats = self.base_encoder._normalization_stats(
                masked_raw, prepared_types)
            masked_eeg = self.base_encoder._normalize_amplitude(
                masked_raw, prepared_types, stats=stats)
            target_eeg = self.base_encoder._normalize_amplitude(
                prepared_eeg, prepared_types, stats=stats)
        else:
            masked_eeg = masked_raw
            target_eeg = prepared_eeg
        masked_inputs, _, _ = self.base_encoder.embed_tokens(
            masked_eeg, channel_names=prepared_names,
            channel_types=prepared_types, normalize_amplitude=False)
        mask_token = self.masking.mask_token.expand_as(masked_inputs)
        masked_tokens = masked_inputs.clone()
        masked_tokens[mask] = mask_token[mask]
        last_hidden, _ = self.base_encoder.encode_tokens(
            masked_tokens, channels, num_times, causal=False)
        pred_patches = self.mask_head(last_hidden)
        targets = self.base_encoder.extract_patches(
            target_eeg, channel_types=prepared_types,
            normalize=False).reshape(
                eeg.shape[0], channels * num_times, self.patch_size_time)
        if return_targets:
            return masked_tokens, mask, pred_patches, targets
        return masked_tokens, mask, pred_patches
    
    def forward_contrastive(
        self,
        eeg1: torch.Tensor,
        eeg2: Optional[torch.Tensor] = None,
        channel_names1: Optional[List[List[str]]] = None,
        channel_types1: Optional[torch.Tensor] = None,
        channel_names2: Optional[List[List[str]]] = None,
        channel_types2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode distinct query/key views with the EMA key encoder."""
        if eeg2 is None:
            raise ValueError(
                "forward_contrastive requires two augmented EEG views")
        if eeg1.shape != eeg2.shape:
            raise ValueError("contrastive views must have identical shapes")
        if torch.equal(eeg1.detach(), eeg2.detach()):
            raise ValueError("contrastive views must be distinct augmentations")

        _, query_pooler = self.forward_embeddings(
            eeg1, channel_names1, channel_types1)
        query = F.normalize(self.proj_head(query_pooler), dim=-1)
        with torch.no_grad():
            self.momentum_encoder.eval()
            _, key_pooler = self.momentum_encoder(
                eeg=eeg2,
                channel_names=channel_names2 or channel_names1,
                channel_types=(
                    channel_types2
                    if channel_types2 is not None else channel_types1),
            )
            key = F.normalize(self.momentum_proj_head(key_pooler), dim=-1)
        return query, key
    
    def forward_adversarial_subject(
        self,
        pooler_output: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Adversarial subject classification using GRL.
        
        Args:
            pooler_output: (B, D) pooled representation
            alpha: GRL reversal strength
        
        Returns:
            subject_logits: (B, num_subjects) subject classification logits
        """
        from .encoder import grad_reverse
        
        # GRL reverses gradients during backprop
        rev_features = grad_reverse(pooler_output, alpha)
        subject_logits = self.subject_classifier(rev_features)
        
        return subject_logits
    
    @torch.no_grad()
    def update_momentum_encoder(self):
        """Update the EMA encoder and its key projection after an optimizer step."""
        self.momentum_encoder.update()
        for online, key in zip(
                self.proj_head.parameters(),
                self.momentum_proj_head.parameters()):
            key.mul_(self.momentum).add_(
                online.detach(), alpha=1.0 - self.momentum)
    
    @classmethod
    def from_v1_model(
        cls,
        v1_model,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        **kwargs,
    ) -> 'EEGFoundationModelV2':
        """
        Create v2 model by tiling weights from v1 model.
        
        Args:
            v1_model: EEGFoundationModel v1 (12L × 768d)
            hidden_dim: Target hidden dimension
            num_layers: Target number of layers
            **kwargs: Additional kwargs for v2 model
        
        Returns:
            EEGFoundationModelV2 initialized from v1
        """
        # Extract v1 config
        v1_hidden_dim = getattr(v1_model, 'hidden_dim', 768)
        v1_num_layers = getattr(v1_model.base_encoder, 'num_layers', 12)
        
        # Create v2 model
        v2_model = cls(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs,
        )
        
        # Handle dimension mismatch: 768 → 1024
        if v1_hidden_dim != hidden_dim:
            print(f"Note: Hidden dimension mismatch: v1={v1_hidden_dim}, v2={hidden_dim}")
            print("Will use random initialization with tiling where possible.")
        
        # If v1 uses factorized encoder, try to tile weights
        if hasattr(v1_model, 'base_encoder') and hasattr(v1_model.base_encoder, 'encoder'):
            v1_encoder = v1_model.base_encoder.encoder
            
            # Create Magi v2 encoder from v1 encoder
            v2_base_encoder = create_magi_v2_from_v1(
                v1_encoder,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                # Pass relevant config from v1
                num_heads=getattr(v1_encoder, 'num_heads', 12),
                intermediate_dim=getattr(v1_encoder, 'intermediate_dim', 3072),
                dropout=getattr(v1_encoder, 'dropout', 0.1),
                spatial_heads=getattr(v1_encoder, 'spatial_heads', 4),
                temporal_heads=getattr(v1_encoder, 'temporal_heads', 8),
            )
            
            # Replace encoder in v2 model
            v2_model.base_encoder.encoder = v2_base_encoder
        
        return v2_model


def test_magi_v2_architecture():
    """Test function to verify Magi v2 architecture."""
    import torch
    
    # Test 1: Basic encoder
    print("Test 1: Basic MagiV2EEGEncoder")
    encoder = MagiV2EEGEncoder(
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        max_channels=256,
        patch_size_time=256,
    )
    
    # Test input
    B, C, T = 2, 64, 1024  # 64 channels, 1024 time points
    eeg = torch.randn(B, C, T)
    channel_names = [[f'Ch{i}' for i in range(C)] for _ in range(B)]
    channel_types = torch.zeros(B, C, dtype=torch.long)  # All scalp EEG
    
    # Forward pass
    last_hidden, pooler = encoder(eeg, channel_names, channel_types)
    print(f"  Input shape: {eeg.shape}")
    print(f"  Output shape: {last_hidden.shape}")
    print(f"  Pooler shape: {pooler.shape}")
    print(f"  Expected tokens: {C * (T // 256)} = {C * 4}")
    print(f"  Actual tokens: {last_hidden.shape[1]}")
    
    # Test 2: ECoG amplitude normalization
    print("\nTest 2: ECoG amplitude normalization")
    channel_types_ecog = torch.ones(B, C, dtype=torch.long)  # All ECoG
    last_hidden_ecog, pooler_ecog = encoder(eeg, channel_names, channel_types_ecog)
    print(f"  ECoG output shape: {last_hidden_ecog.shape}")
    
    # Test 3: Mixed modality
    print("\nTest 3: Mixed modality (scalp + ECoG + sEEG)")
    mixed_types = torch.randint(0, 4, (B, C), dtype=torch.long)
    last_hidden_mixed, pooler_mixed = encoder(eeg, channel_names, mixed_types)
    print(f"  Mixed output shape: {last_hidden_mixed.shape}")
    
    # Test 4: Foundation model
    print("\nTest 4: EEGFoundationModelV2")
    model = EEGFoundationModelV2(
        hidden_dim=1024,
        num_layers=24,
        max_channels=256,
        patch_size_time=256,
    )
    
    # Test masked prediction
    masked_tokens, mask, pred_patches = model.forward_masked_prediction(
        eeg, channel_names, channel_types
    )
    print(f"  Masked tokens shape: {masked_tokens.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Pred patches shape: {pred_patches.shape}")
    
    # Contrastive views must be distinct augmentations; metadata is keyworded
    # to avoid binding it to the second EEG tensor.
    query, key = model.forward_contrastive(
        eeg, eeg2=eeg + 0.01 * torch.randn_like(eeg),
        channel_names1=channel_names, channel_types1=channel_types,
        channel_names2=channel_names, channel_types2=channel_types)
    print(f"  Query shape: {query.shape}")
    print(f"  Key shape: {key.shape}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_magi_v2_architecture()