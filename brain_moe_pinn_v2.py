"""
Brain MoE-PINN v2 with Magi v2 architecture support.

Key upgrades:
- Magi v2 EEG encoder (24L × 1024d)
- ECoG/sEEG multi-modality support via channel type embedding
- Amplitude normalization for ECoG (~20× scalp EEG)
- Variable channel counts (cap 256)
- Revised token budgets based on MoE scaling laws
- Stage 0 MoE from start (E=4) instead of dense init
- Cross-modal sync with DANDI datasets
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import warnings

# Try to import v2 encoders
try:
    from .encoders.eeg_encoder_v2 import EEGEncoderWrapperV2
    EEG_V2_AVAILABLE = True
except ImportError:
    EEG_V2_AVAILABLE = False
    EEGEncoderWrapperV2 = None
    warnings.warn("EEGEncoderWrapperV2 not available, falling back to v1")

# Fallback to v1
try:
    from .encoders.eeg_encoder import EEGEncoderWrapper
    EEG_V1_AVAILABLE = True
except ImportError:
    EEG_V1_AVAILABLE = False
    EEGEncoderWrapper = None

# Other imports
from .encoders.fmri_encoder import NeuroSTORMEncoder, BrainLMEncoder, create_fmri_encoder
from .encoders.hub_fusion import HubTokenFusion, CrossModalAdapter
from .core.velocity_brain import VelocityBrain, MultiTimeScaleKDA
from .core.moe import PoissonRouter, MoEVelocityField, WorkingMemoryRouter, ExpertNetwork
from .core.hebbian_memory import (
    HebbianAssociativeMemory,
    EngramLandscape,
    StructuralPlasticity,
    HippocampalIndex,
)
from .core.counterfactual_search import (
    CounterfactualTreeSearch,
    ImaginationSampler,
    LatentPerturbation,
)
from .core.active_inference import (
    ValueFunction,
    EfferenceCopy,
    SmithPredictor,
    PrecisionGate,
    ClosedLoopFeedback,
    ActiveInferenceController,
)
from .decoder.kda_decoder import (
    KDATemporalDecoder,
    EEGKDADecoder,
    fMRIKDADecoder,
    KIMIKDAMoDeCoderRouter,
)
from .decoder.modality_decoder import EEGDecoder, fMRIDecoder, ModalityDecoderRouter
from .utils.losses import TotalLoss
from .utils.training_phases import LossWeights


class BrainMoEPINNConfig:
    """Configuration for Brain MoE-PINN v2."""
    
    def __init__(
        self,
        # Architecture
        eeg_hidden_dim: int = 1024,  # Magi v2: 1024 vs v1: 768
        eeg_num_layers: int = 24,    # Magi v2: 24 vs v1: 12
        latent_dim: int = 2048,
        fmri_regions: int = 400,
        
        # Modality support
        use_magi_v2: bool = True,
        max_channels: int = 256,      # Cap for ECoG/sEEG
        ecog_amplitude_scale: float = 20.0,
        use_channel_type_embed: bool = True,
        
        # MoE settings
        moe_num_experts: int = 4,     # Stage 0: E=4 (from MoE scaling laws)
        moe_top_k: int = 2,
        
        # Training
        freeze_encoders_epochs: int = 1,
        use_mamba2: bool = False,
        use_kda_decoder: bool = True,
        use_active_inference: bool = True,
        
        # Cross-modal
        use_neurostorm: bool = True,
        use_brainlm: bool = False,
        
        # Context expansion
        initial_context_length: int = 256,
        max_context_length: int = 1024,
        context_expansion_steps: int = 10000,
    ):
        self.eeg_hidden_dim = eeg_hidden_dim
        self.eeg_num_layers = eeg_num_layers
        self.latent_dim = latent_dim
        self.fmri_regions = fmri_regions
        
        self.use_magi_v2 = use_magi_v2
        self.max_channels = max_channels
        self.ecog_amplitude_scale = ecog_amplitude_scale
        self.use_channel_type_embed = use_channel_type_embed
        
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        
        self.freeze_encoders_epochs = freeze_encoders_epochs
        self.use_mamba2 = use_mamba2
        self.use_kda_decoder = use_kda_decoder
        self.use_active_inference = use_active_inference
        
        self.use_neurostorm = use_neurostorm
        self.use_brainlm = use_brainlm
        
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.context_expansion_steps = context_expansion_steps
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration."""
        if self.use_magi_v2 and not EEG_V2_AVAILABLE:
            warnings.warn("Magi v2 requested but not available. Falling back to v1.")
            self.use_magi_v2 = False
        
        if self.use_magi_v2 and self.eeg_hidden_dim != 1024:
            warnings.warn(f"Magi v2 typically uses hidden_dim=1024, got {self.eeg_hidden_dim}")
        
        if self.use_magi_v2 and self.eeg_num_layers != 24:
            warnings.warn(f"Magi v2 typically uses num_layers=24, got {self.eeg_num_layers}")
        
        if self.max_channels > 256:
            warnings.warn(f"max_channels={self.max_channels} > 256 may cause memory issues")
        
        # MoE scaling law constraints
        if self.moe_num_experts < 4:
            warnings.warn(f"MoE scaling laws suggest E≥4 for efficiency, got E={self.moe_num_experts}")
        
        if self.moe_top_k > self.moe_num_experts:
            raise ValueError(f"moe_top_k={self.moe_top_k} > moe_num_experts={self.moe_num_experts}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BrainMoEPINNConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class BrainMoEPINNV2(nn.Module):
    """
    Brain MoE-PINN v2 with Magi v2 architecture support.
    
    Supports:
    - Magi v2 EEG encoder (24L × 1024d)
    - ECoG/sEEG multi-modality
    - Variable channel counts (cap 256)
    - Amplitude normalization for ECoG
    - MoE Stage 0 from start (E=4)
    - Cross-modal sync with DANDI datasets
    - Context length scheduling
    """
    
    def __init__(
        self,
        config: Optional[BrainMoEPINNConfig] = None,
        eeg_channels: int = 19,
        channel_types: Optional[torch.Tensor] = None,
        eeg_checkpoint: Optional[str] = None,
        fmri_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Brain MoE-PINN v2.
        
        Args:
            config: Configuration object
            eeg_channels: Number of EEG channels (default for scalp EEG)
            channel_types: Optional (C,) tensor: 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
            eeg_checkpoint: Path to pretrained EEG encoder checkpoint
            fmri_checkpoint: Path to pretrained fMRI encoder checkpoint
            **kwargs: Additional arguments passed to config
        """
        super().__init__()
        
        # Configuration
        if config is None:
            config = BrainMoEPINNConfig(**kwargs)
        self.config = config
        
        # Channel info
        self.eeg_channels = eeg_channels
        self.channel_types = channel_types
        
        # Store checkpoint paths
        self.eeg_checkpoint = eeg_checkpoint
        self.fmri_checkpoint = fmri_checkpoint
        
        # Initialize components
        self._init_encoders()
        self._init_fusion()
        self._init_core()
        self._init_decoders()
        self._init_auxiliary()
        
        # Context length tracking
        self.current_context_length = config.initial_context_length
        self.context_expansion_counter = 0
        
        print(f"[BrainMoEPINNV2] Initialized with config:")
        for k, v in config.to_dict().items():
            print(f"  {k}: {v}")
    
    def _init_encoders(self):
        """Initialize EEG and fMRI encoders."""
        # EEG Encoder (Magi v2 or v1)
        if self.config.use_magi_v2 and EEG_V2_AVAILABLE:
            self.eeg_encoder = EEGEncoderWrapperV2(
                hidden_dim=self.config.eeg_hidden_dim,
                output_dim=self.config.latent_dim,
                num_layers=self.config.eeg_num_layers,
                freeze_encoder=True,
                use_biot_embedding=True,
                use_channel_type_embed=self.config.use_channel_type_embed,
                max_channels=self.config.max_channels,
                ecog_amplitude_scale=self.config.ecog_amplitude_scale,
                use_mamba2=self.config.use_mamba2,
                from_v1_checkpoint=self.eeg_checkpoint if self.eeg_checkpoint and not self.config.use_magi_v2 else None,
            )
            print(f"[BrainMoEPINNV2] Using Magi v2 EEG encoder (24L×{self.config.eeg_hidden_dim}d)")
        
        elif EEG_V1_AVAILABLE:
            self.eeg_encoder = EEGEncoderWrapper(
                hidden_dim=768,  # v1 default
                output_dim=self.config.latent_dim,
                freeze_encoder=True,
                in_channels=self.eeg_channels,
                use_mamba2=self.config.use_mamba2,
            )
            print(f"[BrainMoEPINNV2] Using Magi v1 EEG encoder (12L×768d)")
        
        else:
            raise RuntimeError("No EEG encoder available")
        
        # fMRI Encoder
        fmri_kwargs = {
            "input_mode": "roi",
            "roi_dim": self.config.fmri_regions,
            "use_mamba2": self.config.use_mamba2,
        }
        
        if self.config.use_neurostorm:
            self.fmri_encoder = create_fmri_encoder(
                encoder_type="neurostorm",
                **fmri_kwargs,
            )
        elif self.config.use_brainlm:
            self.fmri_encoder = create_fmri_encoder(
                encoder_type="brainlm",
                num_regions=self.config.fmri_regions,
            )
        else:
            raise ValueError("Must specify either use_neurostorm=True or use_brainlm=True")
        
        # fMRI projection
        self.fmri_projection = nn.Sequential(
            nn.Linear(self.fmri_encoder.output_dim, self.config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.latent_dim),
        )
        
        # Load pretrained checkpoints
        if self.eeg_checkpoint and self.config.use_magi_v2:
            self.eeg_encoder.load_pretrained(self.eeg_checkpoint, strict=False)
        
        if self.fmri_checkpoint:
            self.fmri_encoder.load_pretrained(self.fmri_checkpoint, strict=False)
    
    def _init_fusion(self):
        """Initialize fusion modules."""
        self.hub_fusion = HubTokenFusion(
            hidden_dim=self.config.latent_dim,
            num_heads=8,
        )
        
        self.cross_modal_adapter = CrossModalAdapter(
            dim=self.config.latent_dim,
            reduction=4,
            dropout=0.1,
        )
    
    def _init_core(self):
        """Initialize core dynamics modules."""
        self.moe_velocity = MoEVelocityField(
            hidden_dim=self.config.latent_dim,
            num_core=self.config.moe_num_experts,
            num_salience=2,
            num_specialized=10,
            top_k=self.config.moe_top_k,
        )
        
        self.velocity_brain = VelocityBrain(
            hidden_dim=self.config.latent_dim,
        )
        
        self.multi_timescale_kda = MultiTimeScaleKDA(
            hidden_dim=self.config.latent_dim,
            alphas=(0.1, 0.5, 0.9),
        )
        
        self.hebbian_memory = HebbianAssociativeMemory(
            hidden_dim=self.config.latent_dim,
            memory_dim=512,
        )
    
    def _init_decoders(self):
        """Initialize decoders."""
        if self.config.use_kda_decoder:
            self.decoder_router = KIMIKDAMoDeCoderRouter(
                latent_dim=self.config.latent_dim,
                output_channels=self.eeg_channels,
                num_regions=self.config.fmri_regions,
                num_layers=4,
            )
        else:
            self.decoder_router = ModalityDecoderRouter(
                latent_dim=self.config.latent_dim,
                output_channels=self.eeg_channels,
                num_regions=self.config.fmri_regions,
            )
    
    def _init_auxiliary(self):
        """Initialize auxiliary modules."""
        if self.config.use_active_inference:
            self.active_inference = ActiveInferenceController(
                latent_dim=self.config.latent_dim,
                tau_delay=0.5,
            )
        
        self.counterfactual_search = CounterfactualTreeSearch(
            latent_dim=self.config.latent_dim,
            rollout_steps=3,
            branch_factor=4,
        )
        
        self.total_loss = TotalLoss(loss_weights=LossWeights())
    
    def set_training_step(self, epoch: int, total_epochs: int):
        """
        Update training state for curriculum learning.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs
        """
        # Update EEG encoder
        if hasattr(self.eeg_encoder, 'set_training_step'):
            self.eeg_encoder.set_training_step(epoch, total_epochs)
        
        # Thaw encoders after freeze period
        if epoch >= self.config.freeze_encoders_epochs:
            if hasattr(self.eeg_encoder, '_thaw_encoder'):
                self.eeg_encoder._thaw_encoder()
            
            # Thaw fMRI encoder projection (encoder may remain frozen)
            for param in self.fmri_projection.parameters():
                param.requires_grad = True
        
        # Context length expansion
        if epoch > 0 and self.context_expansion_counter < self.config.context_expansion_steps:
            self.context_expansion_counter += 1
            
            # Expand context length linearly
            progress = self.context_expansion_counter / self.config.context_expansion_steps
            new_length = int(
                self.config.initial_context_length + 
                progress * (self.config.max_context_length - self.config.initial_context_length)
            )
            
            if new_length > self.current_context_length:
                self.set_context_length(new_length)
    
    def set_context_length(self, context_length: int):
        """
        Set context length for progressive training.
        
        Args:
            context_length: New context length in samples
        """
        # Update EEG encoder
        if hasattr(self.eeg_encoder, 'set_context_length'):
            self.eeg_encoder.set_context_length(context_length)
        
        # Update fMRI encoder
        if hasattr(self.fmri_encoder, 'set_context_length'):
            self.fmri_encoder.set_context_length(context_length)
        
        # Update decoders
        if hasattr(self.decoder_router, 'set_context_length'):
            self.decoder_router.set_context_length(context_length)
        
        self.current_context_length = context_length
        print(f"[BrainMoEPINNV2] Context length updated to {context_length}")

    def reset_history(self):
        """Reset KDA state history and SSM router state."""
        if hasattr(self, "kda_state_history"):
            self.kda_state_history = None
        if hasattr(self, "moe_velocity"):
            self.moe_velocity.reset_router_state()
        if self.config.use_active_inference and hasattr(self, "active_inference"):
            self.active_inference.reset()

    def reset_router_state(self, batch_size: Optional[int] = None):
        """Reset the SSM router state (e.g., at phase boundaries)."""
        if hasattr(self, "moe_velocity"):
            self.moe_velocity.reset_router_state(batch_size)

    def forward(
        self,
        eeg: torch.Tensor,
        fmri: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        mode: str = "perception",
        num_steps: int = 1,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Brain MoE-PINN v2.
        
        Args:
            eeg: (B, C, T) EEG/ECoG/sEEG signals
            fmri: (B, R, T) or (B, H, W, D, T) fMRI data
            channel_names: List of channel names for BIOT embedding
            channel_types: (B, C) integer tensor: 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
            mode: "perception" (sensory-driven) or "imagination" (free evolution)
            num_steps: Number of time steps to evolve
            return_all: Whether to return intermediate states
        
        Returns:
            Dictionary with predictions and intermediate states
        """
        B, C, T = eeg.shape
        device = eeg.device
        
        # Use provided channel_types or default
        if channel_types is None:
            if self.channel_types is not None:
                # Expand to batch size
                if self.channel_types.dim() == 1:
                    channel_types = self.channel_types.unsqueeze(0).expand(B, -1)
                else:
                    channel_types = self.channel_types
            else:
                # Default: all scalp EEG
                channel_types = torch.zeros(B, C, dtype=torch.long, device=device)
        
        # Encode EEG
        eeg_out = self.eeg_encoder(
            eeg=eeg,
            channel_names=channel_names,
            channel_types=channel_types,
            return_pooler=True,
        )
        eeg_tokens = eeg_out["embeddings"]
        eeg_pooler = eeg_out["pooler_output"]
        
        # Encode fMRI
        if fmri.dim() == 3:  # (B, R, T) ROI time series
            fmri_out = self.fmri_encoder(
                fmri,
                return_pooler=True,
            )
            fmri_tokens = fmri_out["last_hidden"]
            fmri_pooler = fmri_out["pooler_output"]
        else:  # (B, H, W, D, T) voxel data
            fmri_tokens, fmri_pooler = self.fmri_encoder.forward_voxel(
                fmri,
                return_pooler=True,
            )
        
        # Project fMRI to latent space
        fmri_tokens = self.fmri_projection(fmri_tokens)
        fmri_pooler = self.fmri_projection(fmri_pooler.unsqueeze(1)).squeeze(1)
        
        # Fusion — use full token sequences (not pooler) for hub cross-attention
        z_global, hub_eeg, hub_fmri = self.hub_fusion(eeg_tokens, fmri_tokens)

        # Cross-modal adaptation
        if hasattr(self, 'cross_modal_adapter'):
            eeg_tokens = self.cross_modal_adapter(eeg_tokens)
            fmri_tokens = self.cross_modal_adapter(fmri_tokens)

        # Store initial state
        z_t = z_global
        states = [z_t] if return_all else []

        # Time evolution
        for t in range(num_steps):
            # MoE velocity field
            moe_out = self.moe_velocity(z_t)
            moe_delta_z = moe_out["velocity"]

            # Apply GENERIC dynamics: VelocityBrain uses MoE velocity as
            # salience signal for the mobility operator, then computes a
            # physics-informed delta_z via Poisson + mobility + arousal.
            vb_out = self.velocity_brain(
                z_t, salience=moe_delta_z, apply_noise=True
            )
            generic_delta_z = vb_out["delta_z"]

            # Multi-timescale KDA decay applied to the velocity
            generic_delta_z = self.multi_timescale_kda(generic_delta_z)

            # Update state
            z_t = z_t + generic_delta_z

            # Hebbian memory update
            if t % 10 == 0:  # Update memory every 10 steps
                self.hebbian_memory.update_kda_state(z_t)

            if return_all:
                states.append(z_t)

        # Decode final state
        if self.config.use_kda_decoder:
            eeg_seq_len = max(1, (eeg.shape[-1] - 256) // 128 + 1) if eeg.shape[-1] >= 256 else 1
            decoder_out = self.decoder_router(
                z_t, hub_eeg, hub_fmri,
                eeg_seq_length=eeg_seq_len,
                fmri_seq_length=fmri.shape[-1],
            )
        else:
            decoder_out = self.decoder_router(z_t, hub_eeg, hub_fmri)
        eeg_recon = decoder_out["eeg_recon"]
        fmri_recon = decoder_out["fmri_recon"]

        pred_error = None
        pred_error = None
        # Active inference (perception mode)
        if mode == "perception" and self.config.use_active_inference:
            # Use actual sensory input for efference copy / reafference
            # NOTE: For true online updating the dataloader should return
            # paired (t, t+1) sequences so actual_eeg is the next timestep.
            ai_out = self.active_inference.forward_perception(
                z_current=z_t,
                predicted_signal=eeg_recon,
                actual_signal=eeg,
                action=None,
                kda_state_history=getattr(self, 'kda_state_history', None),
            )
            if ai_out.get("corrected_z") is not None:
                z_t = ai_out["corrected_z"]
                # Re-decode after correction
                if self.config.use_kda_decoder:
                    decoder_out = self.decoder_router(
                        z_t, hub_eeg, hub_fmri,
                        eeg_seq_length=eeg_seq_len,
                        fmri_seq_length=fmri.shape[-1],
                    )
                else:
                    decoder_out = self.decoder_router(z_t, hub_eeg, hub_fmri)
                eeg_recon = decoder_out["eeg_recon"]
                fmri_recon = decoder_out["fmri_recon"]
            pred_error = ai_out.get("feedback_metrics", {}).get("prediction_error")

        # Counterfactual search (imagination mode)
        elif mode == "imagination" and hasattr(self, 'counterfactual_search'):
            # Generate counterfactual trajectories
            trajectories = self.counterfactual_search.search(
                initial_state=z_t,
                horizon=10,
            )

            # Use best trajectory
            if len(trajectories) > 0:
                z_t = trajectories[0]['final_state']
                if self.config.use_kda_decoder:
                    decoder_out = self.decoder_router(
                        z_t, hub_eeg, hub_fmri,
                        eeg_seq_length=eeg_seq_len,
                        fmri_seq_length=fmri.shape[-1],
                    )
                else:
                    decoder_out = self.decoder_router(z_t, hub_eeg, hub_fmri)
                eeg_recon = decoder_out["eeg_recon"]
                fmri_recon = decoder_out["fmri_recon"]

        # Cross-modal alignment: hub token consistency
        # At initialization these should be near-zero (modalities are independent).
        # During Stage 1 P2+ cross-modal training they should rise.
        # A high value at step 0 means the hub is NOT distinguishing modalities — a bug.
        eeg_norm = hub_eeg / (hub_eeg.norm(dim=-1, keepdim=True) + 1e-8)
        fmri_norm = hub_fmri / (hub_fmri.norm(dim=-1, keepdim=True) + 1e-8)
        cross_modal_corr = (eeg_norm * fmri_norm).sum(dim=-1).mean().item()

        # Prepare output
        output = {
            'z_global': z_global,
            'z_final': z_t,
            'eeg_recon': eeg_recon,
            'fmri_recon': fmri_recon,
            'hub_eeg': hub_eeg,
            'hub_fmri': hub_fmri,
            'cross_modal_correlation': cross_modal_corr,
            'eeg_tokens': eeg_tokens,
            'fmri_tokens': fmri_tokens,
        }

        if return_all:
            output['states'] = torch.stack(states, dim=1)  # (B, num_steps+1, D)

        if mode == "perception" and self.config.use_active_inference and pred_error is not None:
            output['prediction_error'] = pred_error

        return output
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            predictions: Output from forward()
            targets: Dictionary with target data
            epoch: Current epoch for curriculum (unused, kept for API compat)
            **kwargs: Additional arguments for loss computation (unused)

        Returns:
            Dictionary of loss components
        """
        total_loss, metrics = self.total_loss(
            predictions=predictions,
            targets=targets,
        )
        # TotalLoss returns (tensor, metrics_dict); wrap for backward compat
        return {"total_loss": total_loss, **metrics}
    
    def load_pretrained(
        self,
        eeg_checkpoint: Optional[str] = None,
        fmri_checkpoint: Optional[str] = None,
        strict: bool = False,
    ):
        """
        Load pretrained weights for encoders.
        
        Args:
            eeg_checkpoint: Path to EEG encoder checkpoint
            fmri_checkpoint: Path to fMRI encoder checkpoint
            strict: Whether to require exact parameter matching
        """
        if eeg_checkpoint:
            print(f"[BrainMoEPINNV2] Loading EEG encoder from {eeg_checkpoint}")
            self.eeg_encoder.load_pretrained(eeg_checkpoint, strict=strict)
            self.eeg_checkpoint = eeg_checkpoint
        
        if fmri_checkpoint:
            print(f"[BrainMoEPINNV2] Loading fMRI encoder from {fmri_checkpoint}")
            self.fmri_encoder.load_pretrained(fmri_checkpoint, strict=strict)
            self.fmri_checkpoint = fmri_checkpoint
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for different components."""
        components = [
            'eeg_encoder',
            'fmri_encoder',
            'fmri_projection',
            'hub_fusion',
            'moe_velocity',
            'velocity_brain',
            'decoder_router',
            'active_inference' if hasattr(self, 'active_inference') else None,
            'counterfactual_search',
            'hebbian_memory',
        ]
        
        params = {}
        for comp in components:
            if comp is None or not hasattr(self, comp):
                continue
            
            module = getattr(self, comp)
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            params[f'{comp}_total'] = total
            params[f'{comp}_trainable'] = trainable
        
        # Totals
        params['total'] = sum(p.numel() for p in self.parameters())
        params['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return params


def test_brain_moe_pinn_v2():
    """Test function for BrainMoEPINNV2."""
    import torch
    
    print("Testing BrainMoEPINNV2...")
    
    try:
        # Test configuration
        config = BrainMoEPINNConfig(
            use_magi_v2=True,
            eeg_hidden_dim=1024,
            eeg_num_layers=24,
            latent_dim=2048,
            max_channels=256,
            ecog_amplitude_scale=20.0,
            use_channel_type_embed=True,
            moe_num_experts=4,
            moe_top_k=2,
            freeze_encoders_epochs=1,
            use_mamba2=False,
            use_kda_decoder=True,
            use_active_inference=True,
            use_neurostorm=True,
        )
        
        # Initialize model
        model = BrainMoEPINNV2(
            config=config,
            eeg_channels=64,  # Simulating ECoG
        )
        
        print(f"✓ Model initialized")
        
        # Test forward pass
        B, C, T = 2, 64, 1024
        eeg = torch.randn(B, C, T)
        fmri = torch.randn(B, 400, T)  # ROI time series
        
        # Channel types: mix of scalp EEG and ECoG
        channel_types = torch.zeros(B, C, dtype=torch.long)
        channel_types[:, :32] = 0  # First 32 channels: scalp EEG
        channel_types[:, 32:] = 1  # Last 32 channels: ECoG
        
        # Channel names
        channel_names = [[f'Ch{i}' for i in range(C)] for _ in range(B)]
        
        # Perception mode
        output = model(
            eeg=eeg,
            fmri=fmri,
            channel_names=channel_names,
            channel_types=channel_types,
            mode='perception',
            num_steps=5,
            return_all=True,
        )
        
        print(f"✓ Forward pass successful")
        print(f"  Input shapes: EEG {eeg.shape}, fMRI {fmri.shape}")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  z_global shape: {output['z_global'].shape}")
        print(f"  z_final shape: {output['z_final'].shape}")
        print(f"  EEG recon shape: {output['eeg_recon'].shape}")
        print(f"  fMRI recon shape: {output['fmri_recon'].shape}")
        
        if 'states' in output:
            print(f"  States shape: {output['states'].shape}")
        
        # Test loss computation
        targets = {
            'eeg': eeg,
            'fmri': fmri,
        }
        
        losses = model.compute_loss(output, targets, epoch=0)
        print(f"✓ Loss computation successful")
        print(f"  Loss components: {list(losses.keys())}")
        
        # Test parameter counts
        params = model.get_num_params()
        print(f"✓ Parameter counts:")
        print(f"  Total: {params.get('total', 0):,}")
        print(f"  Trainable: {params.get('trainable', 0):,}")
        
        # Test training step
        model.set_training_step(epoch=0, total_epochs=10)
        print(f"✓ Training step update")
        
        # Test context length
        model.set_context_length(512)
        print(f"✓ Context length update")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_brain_moe_pinn_v2()