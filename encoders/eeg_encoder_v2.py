"""
EEG Encoder wrapper v2 with Magi v2 architecture (24L × 1024d).

Features:
- Magi v2: 24 layers × 1024 hidden dimension
- RoPE (Rotary Position Embedding)
- GeGLU activation in FFN
- Alternating attention: global every 3 layers, SWA otherwise
- BIOT-style arbitrary electrode embeddings
- Channel type embedding (scalp EEG / ECoG / sEEG)
- Amplitude normalization for ECoG (~20× scalp EEG)
- Support for variable channel counts (cap 256)
- Mamba-2 SSM for long-context processing
- ECoG integration via DANDI datasets
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import warnings

from brain_moe_pinn.magi.magi_v2 import MagiV2EEGEncoder
from brain_moe_pinn.sequence.mamba2_ssm import Mamba2Backbone, FLA_MAMBA2_AVAILABLE


class EEGEncoderWrapperV2(nn.Module):
    """
    Wraps Magi v2 EEGFoundationModel for use in Brain MoE-PINN.
    
    Supports:
    - Magi v2 architecture (24L × 1024d)
    - ECoG/sEEG multi-modality via channel type embedding
    - Amplitude normalization for ECoG
    - Variable channel counts (up to 256)
    - Mamba-2 SSM for long-context processing
    - Frozen backbone with trainable projection
    - Context length scheduling
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        output_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        freeze_encoder: bool = True,
        use_biot_embedding: bool = True,
        use_channel_type_embed: bool = True,
        patch_size_time: int = 256,
        stride_time: int = 128,
        max_channels: int = 256,
        ecog_amplitude_scale: float = 20.0,
        freeze_epochs: int = 1,
        use_mamba2: bool = False,
        mamba2_layers: int = 4,
        mamba2_chunk_size: int = 256,
        mamba2_backend: str = "triton",
        use_rope: bool = True,
        alternating_pattern: bool = True,
        window_size: int = 128,
        from_v1_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.freeze_encoder = freeze_encoder
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        self.use_mamba2 = use_mamba2
        self.max_channels = max_channels
        self.ecog_amplitude_scale = ecog_amplitude_scale
        
        # Context length scheduling
        self.current_context_length = patch_size_time
        self.target_context_length = patch_size_time * 4  # Can expand to 4×
        
        # Brain MoE-PINN only needs the Magi representation encoder. The
        # standalone EEGFoundationModelV2 adds pretraining heads that are not
        # used in this production adapter.
        self.encoder = MagiV2EEGEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_size_time=patch_size_time,
            stride_time=stride_time,
            max_channels=max_channels,
            use_biot_embedding=use_biot_embedding,
            use_channel_type_embed=use_channel_type_embed,
            ecog_amplitude_scale=ecog_amplitude_scale,
            use_rope=use_rope,
            alternating_pattern=alternating_pattern,
            window_size=window_size,
        )
        print(f"[EEGEncoderWrapperV2] Loaded MagiV2EEGEncoder "
              f"({num_layers}L×{hidden_dim}d, BIOT={use_biot_embedding}, "
              f"ECoG-scale={ecog_amplitude_scale})")
        
        # Load from v1 checkpoint if specified
        if from_v1_checkpoint:
            self.load_from_v1_checkpoint(from_v1_checkpoint)
        
        # Mamba-2 SSM for long-context processing
        if use_mamba2 and FLA_MAMBA2_AVAILABLE and Mamba2Backbone is not None:
            self.mamba2 = Mamba2Backbone(
                hidden_dim=hidden_dim,
                num_layers=mamba2_layers,
                chunk_size=mamba2_chunk_size,
                backend=mamba2_backend,
            )
            print(f"[EEGEncoderWrapperV2] Added Mamba-2 context stack "
                  f"({mamba2_layers} layers, chunk={mamba2_chunk_size})")
        else:
            self.mamba2 = None
        
        # Projection to output dimension (for Brain MoE-PINN)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
        
        print(f"[EEGEncoderWrapperV2] Initialized: hidden_dim={hidden_dim}, "
              f"output_dim={output_dim}, freeze={freeze_encoder}, mamba2={use_mamba2}")
    

    
    def load_from_v1_checkpoint(self, checkpoint_path: str):
        """Load weights from v1 checkpoint."""
        try:
            if checkpoint_path.endswith('.pt'):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            elif checkpoint_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                checkpoint = load_file(checkpoint_path)
            else:
                raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
            
            # Load encoder weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            encoder_state_dict = {}
            prefixes = ("encoder.base_encoder.", "base_encoder.", "encoder.")
            for key, value in state_dict.items():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        encoder_state_dict[key[len(prefix):]] = value
                        break

            if encoder_state_dict:
                # v1 and v2 differ in dimensions; compatible keys are loaded.
                missing, unexpected = self.encoder.load_state_dict(
                    encoder_state_dict, strict=False
                )
                print(f"[EEGEncoderWrapperV2] Loaded v1 checkpoint: {checkpoint_path}")
                if missing:
                    print(f"  Missing keys: {missing[:5]}...")
                if unexpected:
                    print(f"  Unexpected keys: {unexpected[:5]}...")
            else:
                print(f"[EEGEncoderWrapperV2] No encoder weights found in checkpoint")
        
        except Exception as e:
            warnings.warn(f"Failed to load v1 checkpoint {checkpoint_path}: {e}")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[EEGEncoderWrapperV2] Encoder frozen")
    
    def _thaw_encoder(self):
        """Thaw encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("[EEGEncoderWrapperV2] Encoder thawed")
    
    def set_training_step(self, epoch: int, total_epochs: int):
        """
        Update training state for curriculum learning.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs
        """
        self.current_epoch = epoch
        
        # Thaw encoder after freeze_epochs
        if self.freeze_encoder and epoch >= self.freeze_epochs:
            self._thaw_encoder()
            self.freeze_encoder = False  # Only thaw once
    
    def set_context_length(self, context_length: int):
        """
        Set context length for progressive training.
        
        Args:
            context_length: New context length in samples
        """
        if hasattr(self.encoder, 'set_context_length'):
            self.encoder.set_context_length(context_length)
            self.current_context_length = context_length
            print(f"[EEGEncoderWrapperV2] Context length set to {context_length}")
        
        if self.mamba2 is not None and hasattr(self.mamba2, 'set_context_length'):
            self.mamba2.set_context_length(context_length)
    
    def forward(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        return_pooler: bool = False,
        mamba2_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through EEG encoder.
        
        Args:
            eeg: (B, C, T) EEG/ECoG/sEEG signals
            channel_names: List of length B, each containing C channel names
            channel_types: (B, C) integer tensor: 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
            return_pooler: Whether to return pooler output
            mamba2_only: Skip Magi encoder, only use Mamba-2 (for ablation)
        
        Returns:
            dict with ``last_hidden`` and ``embeddings``. ``pooler_output``
            is included when ``return_pooler`` is true.
        """
        B, C, T = eeg.shape
        
        # Validate channel count
        if C > self.max_channels:
            raise ValueError(f"Number of channels {C} exceeds max_channels {self.max_channels}")
        
        # Magi v2 encoder
        if not mamba2_only:
            last_hidden, pooler_output = self.encoder(
                eeg=eeg,
                channel_names=channel_names,
                channel_types=channel_types,
            )
            # Mamba-2 context processing
            if self.mamba2 is not None:
                # Reshape for Mamba-2: (B, L, D) -> process
                mamba_output = self.mamba2(last_hidden)
                last_hidden = last_hidden + mamba_output  # Residual connection
        else:
            # Mamba-2 only mode (for ablation studies)
            if self.mamba2 is None:
                raise ValueError("mamba2_only=True but Mamba-2 not initialized")
            
            # Direct projection of EEG to token space
            # Simple temporal convolution to match patch size
            if not hasattr(self, '_mamba2_proj'):
                self._mamba2_proj = nn.Conv1d(
                    in_channels=C,
                    out_channels=self.hidden_dim,
                    kernel_size=self.encoder.patch_size_time,
                    stride=self.encoder.stride_time,
                ).to(eeg.device)
            
            # Project to token space: (B, C, T) -> (B, D, T/P) -> (B, T/P, D)
            tokens = self._mamba2_proj(eeg)  # (B, D, T/P)
            tokens = tokens.permute(0, 2, 1)  # (B, T/P, D)
            
            # Add BIOT embeddings if available
            if channel_names is not None and hasattr(self.encoder, 'biot_embed'):
                # Simplified: just use first channel name per position
                biot_embeds = self.encoder.biot_embed(channel_names[0])  # (1, C, D)
                # Average across channels for each token
                biot_embeds = biot_embeds.mean(dim=1, keepdim=True)  # (1, 1, D)
                biot_embeds = biot_embeds.expand(B, tokens.shape[1], -1)  # (B, T/P, D)
                tokens = tokens + biot_embeds
            
            # Mamba-2 processing
            last_hidden = self.mamba2(tokens)
            pooler_output = last_hidden.mean(dim=1)  # Average pooling
        
        # Project to output dimension
        embeddings = self.projection(last_hidden)
        
        out = {
            "last_hidden": last_hidden,
            "embeddings": embeddings,
        }
        if return_pooler:
            out["pooler_output"] = self.projection(
                pooler_output.unsqueeze(1)
            ).squeeze(1)
        return out
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """
        Load pretrained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (.pt or .safetensors)
            strict: Whether to require exact parameter matching
        """
        try:
            if checkpoint_path.endswith('.pt'):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            elif checkpoint_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                checkpoint = load_file(checkpoint_path)
            else:
                raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
            
            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter for encoder weights (ignore projection head)
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    encoder_state_dict[k] = v
                elif not k.startswith('projection.') and not k.startswith('mask_head.') and not k.startswith('proj_head.'):
                    # Try to match other keys
                    encoder_state_dict[k] = v
            
            # Load with compatibility handling
            missing, unexpected = self.load_state_dict(encoder_state_dict, strict=strict)
            
            print(f"[EEGEncoderWrapperV2] Loaded pretrained weights from {checkpoint_path}")
            if missing:
                print(f"  Missing keys: {missing[:5]}... (total: {len(missing)})")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:5]}... (total: {len(unexpected)})")
            
            return missing, unexpected
        
        except Exception as e:
            warnings.warn(f"Failed to load pretrained weights from {checkpoint_path}: {e}")
            raise
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for different components."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count by component
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        mamba2_params = sum(p.numel() for p in self.mamba2.parameters()) if self.mamba2 else 0
        projection_params = sum(p.numel() for p in self.projection.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder_params,
            'mamba2': mamba2_params,
            'projection': projection_params,
        }


def test_eeg_encoder_v2():
    """Test function for EEGEncoderWrapperV2."""
    import torch
    
    print("Testing EEGEncoderWrapperV2...")
    
    # Test 1: Basic initialization
    try:
        encoder = EEGEncoderWrapperV2(
            hidden_dim=1024,
            output_dim= 1024,
            num_layers=24,
            freeze_encoder=False,
            use_biot_embedding=True,
            use_channel_type_embed=True,
            max_channels=256,
            ecog_amplitude_scale=20.0,
            use_mamba2=False,
        )
        
        print(f"✓ Initialized EEGEncoderWrapperV2")
        
        # Test forward pass
        B, C, T = 2, 64, 1024
        eeg = torch.randn(B, C, T)
        channel_names = [[f'Ch{i}' for i in range(C)] for _ in range(B)]
        channel_types = torch.ones(B, C, dtype=torch.long)  # All ECoG
        
        embeddings, pooler = encoder(eeg, channel_names, channel_types, return_pooler=True)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {eeg.shape}")
        print(f"  Output embeddings shape: {embeddings.shape}")
        print(f"  Pooler shape: {pooler.shape if pooler is not None else 'None'}")
        
        # Test parameter counts
        params = encoder.get_num_params()
        print(f"✓ Parameter counts:")
        for k, v in params.items():
            print(f"  {k}: {v:,}")
        
        # Test context length setting
        encoder.set_context_length(512)
        print(f"✓ Context length setting")
        
        # Test training step
        encoder.set_training_step(epoch=0, total_epochs=10)
        print(f"✓ Training step update")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_eeg_encoder_v2()