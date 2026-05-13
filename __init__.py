"""
Brain MoE-PINN: Multi-Attractor Neural Dynamics Model

A 7-10B parameter Mixture-of-Experts Physics-Informed Neural Network
for modeling brain neural dynamics with GENERIC (General Equation for
Non-Equilibrium Reversible-Irreversible Coupling) dynamics constraints.

Architecture:
- Dual-modality encoders: Magi EEG (BERT-base, 768d) + NeuroSTORM fMRI (SWM backbone)
- Hub Token fusion for cross-modal representation
- VelocityBrain GENERIC dynamics core (irreversible Δz prediction)
- MoE expert routing with Poisson Router
- Multi-Time-Scale KDA for memory hierarchies
- KDA-based decoders (Kimi Delta Attention from fla library)
- Counterfactual Tree Search for imagination/action planning
- Active Inference with Value Function, Efference Copy, Smith Predictor, Precision Gate

Reference: Brain MoE-PINN Training Plan (kilo plan: 1778463235296-neon-cabin.md)
"""

from .encoders.eeg_encoder import EEGEncoderWrapper, EEGProjection
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
from .decoder.modality_decoder import EEGDecoder, fMRIDecoder, ModalityDecoderRouter
from .decoder.kda_decoder import (
    KDATemporalDecoder,
    EEGKDADecoder,
    fMRIKDADecoder,
    KIMIKDAMoDeCoderRouter,
)
from .utils.losses import TotalLoss
from .utils.data_loader import (
    EEGDataset,
    fMRIDataset,
    PairedBrainDataset,
    create_brain_dataloaders,
)
import torch
import torch.nn as nn
from typing import Optional, Dict, List

__version__ = "0.3.1"

__all__ = [
    # Encoders
    "EEGEncoderWrapper",
    "EEGProjection",
    "NeuroSTORMEncoder",
    "BrainLMEncoder",
    "create_fmri_encoder",
    # Fusion
    "HubTokenFusion",
    "CrossModalAdapter",
    # Core
    "VelocityBrain",
    "MultiTimeScaleKDA",
    "PoissonRouter",
    "MoEVelocityField",
    "WorkingMemoryRouter",
    "ExpertNetwork",
    # Memory
    "HebbianAssociativeMemory",
    "EngramLandscape",
    "StructuralPlasticity",
    "HippocampalIndex",
    # Counterfactual Search
    "CounterfactualTreeSearch",
    "ImaginationSampler",
    "LatentPerturbation",
    # Active Inference
    "ValueFunction",
    "EfferenceCopy",
    "SmithPredictor",
    "PrecisionGate",
    "ClosedLoopFeedback",
    "ActiveInferenceController",
    # Decoders
    "EEGDecoder",
    "fMRIDecoder",
    "ModalityDecoderRouter",
    "KDATemporalDecoder",
    "EEGKDADecoder",
    "fMRIKDADecoder",
    "KIMIKDAMoDeCoderRouter",
    # Utils
    "TotalLoss",
    "EEGDataset",
    "fMRIDataset",
    "PairedBrainDataset",
    "create_brain_dataloaders",
]


class BrainMoEPINN(nn.Module):
    """
    Complete Brain MoE-PINN model integrating all components.

    Forward pass:
    1. EEG signal -> Magi encoder -> upscaling projection -> eeg_tokens (B, L_eeg, 2048)
    2. fMRI signal -> NeuroSTORM encoder -> projection -> fmri_tokens (B, L_fmri, 2048)
    3. Hub Token fusion -> z_global (B, 2048)
    4. MoEVelocityField -> delta_z (B, 2048)
    5. z_{t+1} = z_t + delta_z (irreversible)
    6. ModalityDecoderRouter -> EEG/fMRI reconstructions
    7. ActiveInferenceController -> Value function, feedback correction

    Modes:
    - 'perception': forced ODE with sensory feedback correction
    - 'imagination': free evolution with counterfactual planning
    """

    def __init__(
        self,
        eeg_channels: int = 19,
        fmri_regions: int = 400,
        latent_dim: int = 2048,
        use_neurostorm: bool = True,
        use_kda_decoder: bool = True,
        use_active_inference: bool = True,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
        tau_delay: float = 0.5,
        freeze_encoders_epochs: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_kda_decoder = use_kda_decoder
        self.use_active_inference = use_active_inference

        eeg_kwargs = {"in_channels": eeg_channels}
        if use_mamba2:
            eeg_kwargs["use_mamba2"] = True
            if mamba2_kwargs:
                eeg_kwargs.update({
                    "mamba2_layers": mamba2_kwargs.get("eeg_layers", 4),
                    "mamba2_chunk_size": mamba2_kwargs.get("chunk_size", 256),
                    "mamba2_backend": mamba2_kwargs.get("backend", "triton"),
                })

        self.eeg_encoder = EEGEncoderWrapper(
            hidden_dim=768,
            output_dim=latent_dim,
            freeze_encoder=True,
            **eeg_kwargs,
        )

        fmri_kwargs = {
            "input_mode": "roi",
            "roi_dim": fmri_regions,
        }
        if use_mamba2:
            fmri_kwargs["use_mamba2"] = True
            if mamba2_kwargs:
                fmri_kwargs["mamba2_kwargs"] = {
                    "expand": mamba2_kwargs.get("fmri_expand", 2),
                    "chunk_size": mamba2_kwargs.get("chunk_size", 256),
                    "backend": mamba2_kwargs.get("backend", "triton"),
                }

        if use_neurostorm:
            self.fmri_encoder = create_fmri_encoder(
                encoder_type="neurostorm",
                **fmri_kwargs,
            )
        else:
            self.fmri_encoder = create_fmri_encoder(
                encoder_type="brainlm",
                num_regions=fmri_regions,
            )

        self.fmri_projection = nn.Sequential(
            nn.Linear(self.fmri_encoder.output_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        self.hub_fusion = HubTokenFusion(
            hidden_dim=latent_dim,
            num_heads=8,
        )

        self.moe_velocity = MoEVelocityField(hidden_dim=latent_dim)

        if use_kda_decoder:
            self.decoder_router = KIMIKDAMoDeCoderRouter(
                latent_dim=latent_dim,
                output_channels=eeg_channels,
                num_regions=fmri_regions,
                num_layers=4,
            )
        else:
            self.decoder_router = ModalityDecoderRouter(
                latent_dim=latent_dim,
            )

        if use_active_inference:
            self.active_inference = ActiveInferenceController(
                latent_dim=latent_dim,
                use_counterfactual=True,
                use_feedback=True,
                tau_delay=tau_delay,
            )

        self.counterfactual_search = CounterfactualTreeSearch(
            latent_dim=latent_dim,
            num_goals=2,
            num_exploration=1,
            rollout_steps=3,
            branch_factor=4,
        )
        self.imagination_sampler = ImaginationSampler(latent_dim=latent_dim)

        self.kda_state_history = None
        self.max_history = 100

    def forward(
        self,
        eeg: torch.Tensor,
        fmri: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        channel_types: Optional[torch.Tensor] = None,
        mode: str = "perception",
        goal_attractors: Optional[torch.Tensor] = None,
        task_cue: Optional[torch.Tensor] = None,
        actual_eeg: Optional[torch.Tensor] = None,
        actual_fmri: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through Brain MoE-PINN.

        Args:
            eeg: (B, C_eeg, T) raw EEG signals (encoder input)
            fmri: (B, R, T_fmri) ROI time series (encoder input)
            channel_names: Optional channel names for BIOT embedding
            mode: 'perception' (forced ODE with feedback) or 'imagination' (free)
            goal_attractors: (B, num_goals, d) goal attractor centers
            task_cue: (B, d) task cue for imagination
            actual_eeg: (B, C_eeg, T) actual EEG at t+1 (for perception feedback)
            actual_fmri: (B, R, T_fmri) actual fMRI at t+1 (for perception feedback)
            action: (B, d) executed action (for Smith predictor)
        Returns:
            dict with all outputs including reconstructions and latent states
        """
        eeg_out = self.eeg_encoder(
            eeg, channel_names, channel_types, return_embeddings=True
        )
        eeg_tokens = eeg_out["embeddings"]

        fmri_out = self.fmri_encoder(fmri)
        fmri_tokens = self.fmri_projection(fmri_out["last_hidden"])

        z_global, hub_eeg, hub_fmri = self.hub_fusion(eeg_tokens, fmri_tokens)

        moe_out = self.moe_velocity(z_global)
        delta_z = moe_out["velocity"]

        z_next = z_global + delta_z

        if self.use_kda_decoder:
            # Decoder seq_length = encoder temporal patches
            # EEG: patch_size=256, stride=128 => num_patches = (T - 256) // 128 + 1
            eeg_patches = (eeg.shape[-1] - 256) // 128 + 1 if eeg.shape[-1] >= 256 else 1
            fmri_seq_len = fmri.shape[-1]
            decoder_out = self.decoder_router(
                z_next,
                hub_eeg,
                hub_fmri,
                eeg_seq_length=max(1, eeg_patches),
                fmri_seq_length=fmri_seq_len,
            )
        else:
            decoder_out = self.decoder_router(z_next, hub_eeg, hub_fmri)

        result = {
            "z_global": z_global,
            "z_next": z_next,
            "delta_z": delta_z,
            "hub_eeg": hub_eeg,
            "hub_fmri": hub_fmri,
            "eeg_recon": decoder_out["eeg_recon"],
            "fmri_recon": decoder_out["fmri_recon"],
            "moe_routing": moe_out["routing_metrics"],
        }

        if self.use_active_inference and mode == "perception":
            self._update_kda_history(z_next)

            if actual_eeg is not None and actual_fmri is not None:
                w_q = getattr(self, "_wiener_q_factor", None)
                w_a = getattr(self, "_wiener_ataxia", None)
                w_c = getattr(self, "_wiener_catalepsy", None)
                ai_result = self.active_inference.forward_perception(
                    z_next,
                    decoder_out["eeg_recon"],
                    actual_eeg,
                    action,
                    self.kda_state_history,
                    q_factor=w_q,
                    ataxia_score=w_a,
                    catalepsy_score=w_c,
                )
                result["corrected_z"] = ai_result.get("corrected_z", z_next)
                result["value"] = ai_result.get("value")
                result["feedback_metrics"] = ai_result.get("metrics", {})

                if ai_result.get("corrected_z") is not None:
                    result["z_next"] = ai_result["corrected_z"]

        if mode == "imagination":
            cfts_result = self.counterfactual_search(
                z_next,
                goal_attractors=goal_attractors,
                task_cue=task_cue,
            )
            result["selected_action"] = cfts_result["selected_action"]
            result["selected_trajectory"] = cfts_result["selected_trajectory"]
            result["imagination_scores"] = cfts_result["scores"]

            imagined = self.imagination_sampler(z_next, num_samples=4)
            result["imagined_states"] = imagined["imagined_states"]

            if self.use_active_inference:
                efe, efe_metrics = self.active_inference.value_function.compute_efe(
                    cfts_result["selected_trajectory"][:, -1],
                    goal_attractors=goal_attractors,
                )
                result["efe"] = efe
                result["efe_metrics"] = efe_metrics

        return result

    def _update_kda_history(self, z: torch.Tensor):
        """Update KDA state history for Smith predictor. Stored in FP32."""
        if self.kda_state_history is None:
            self.kda_state_history = z.detach().float()
        else:
            self.kda_state_history = torch.cat([
                self.kda_state_history, z.detach().float()
            ], dim=1)[:, -self.max_history:]

    def set_training_step(self, step: int, freeze_epochs_steps: int = 1000):
        """
        Update model state based on training step.

        Args:
            step: current training step
            freeze_epochs_steps: number of steps equivalent to 1 epoch of frozen encoders
        """
        if step < freeze_epochs_steps:
            # Freeze encoders
            if hasattr(self.eeg_encoder, 'freeze_encoder'):
                self.eeg_encoder.freeze_encoder = True
            if hasattr(self.fmri_encoder, 'set_freeze'):
                self.fmri_encoder.set_freeze(True)
        else:
            # Thaw encoders with reduced LR
            if hasattr(self.eeg_encoder, 'freeze_encoder'):
                self.eeg_encoder.freeze_encoder = False
            if hasattr(self.fmri_encoder, 'set_freeze'):
                self.fmri_encoder.set_freeze(False)

    def set_context_length(self, context_length: int):
        """
        Update effective context length for progressive context expansion.

        Propagates to encoders (Mamba2 backends) and decoders so they can
        adjust their internal sequence-length-dependent buffers.

        Args:
            context_length: new target context length (e.g. 4096 → 16384 → 65536)
        """
        self._current_context_length = context_length
        if hasattr(self.eeg_encoder, "set_context_length"):
            self.eeg_encoder.set_context_length(context_length)
        if hasattr(self.fmri_encoder, "set_context_length"):
            self.fmri_encoder.set_context_length(context_length)
        if self.use_kda_decoder and hasattr(self.decoder_router, "set_context_length"):
            self.decoder_router.set_context_length(context_length)

    def load_pretrained(
        self,
        eeg_checkpoint: Optional[str] = None,
        fmri_checkpoint: Optional[str] = None,
    ):
        """
        Load pretrained encoder weights into their respective submodules.

        Args:
            eeg_checkpoint: path to Magi pretrained weights (.pt)
            fmri_checkpoint: path to NeuroSTORM or BrainLM pretrained weights (.pt/.safetensors)
        """
        if eeg_checkpoint is not None:
            if hasattr(self.eeg_encoder, "load_pretrained"):
                self.eeg_encoder.load_pretrained(eeg_checkpoint)
            else:
                print("[BrainMoEPINN] EEG encoder has no load_pretrained method")

        if fmri_checkpoint is not None:
            if hasattr(self.fmri_encoder, "load_pretrained"):
                self.fmri_encoder.load_pretrained(fmri_checkpoint)
            else:
                print("[BrainMoEPINN] fMRI encoder has no load_pretrained method")

    def reset_history(self):
        """Reset KDA state history and SSM router state."""
        self.kda_state_history = None
        if hasattr(self, "moe_velocity"):
            self.moe_velocity.reset_router_state()
        if self.use_active_inference:
            self.active_inference.reset()

    def reset_router_state(self, batch_size: Optional[int] = None):
        """Reset the SSM router state (e.g., at phase boundaries)."""
        if hasattr(self, "moe_velocity"):
            self.moe_velocity.reset_router_state(batch_size)


class BrainMoEPINNConfig:
    """Configuration class for BrainMoEPINN."""

    def __init__(
        self,
        eeg_channels: int = 19,
        fmri_regions: int = 400,
        latent_dim: int = 2048,
        use_neurostorm: bool = True,
        use_kda_decoder: bool = True,
        use_active_inference: bool = True,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
        tau_delay: float = 0.5,
    ):
        self.eeg_channels = eeg_channels
        self.fmri_regions = fmri_regions
        self.latent_dim = latent_dim
        self.use_neurostorm = use_neurostorm
        self.use_kda_decoder = use_kda_decoder
        self.use_active_inference = use_active_inference
        self.use_mamba2 = use_mamba2
        self.mamba2_kwargs = mamba2_kwargs or {}
        self.tau_delay = tau_delay

    def to_model(self) -> BrainMoEPINN:
        """Create model from config."""
        return BrainMoEPINN(
            eeg_channels=self.eeg_channels,
            fmri_regions=self.fmri_regions,
            latent_dim=self.latent_dim,
            use_neurostorm=self.use_neurostorm,
            use_kda_decoder=self.use_kda_decoder,
            use_active_inference=self.use_active_inference,
            use_mamba2=self.use_mamba2,
            mamba2_kwargs=self.mamba2_kwargs,
            tau_delay=self.tau_delay,
        )


if __name__ == "__main__":
    print("Testing complete BrainMoEPINN model...")

    model = BrainMoEPINN(
        eeg_channels=19,
        fmri_regions=400,
        use_kda_decoder=True,
        use_active_inference=True,
    )

    B, C_eeg, T_eeg = 2, 19, 2560
    B, R, T_fmri = 2, 400, 100
    eeg_patches = (T_eeg - 256) // 128 + 1  # = 19 (encoder temporal resolution)

    dummy_eeg = torch.randn(B, C_eeg, T_eeg)
    dummy_fmri = torch.randn(B, R, T_fmri)
    dummy_actual_eeg = torch.randn(B, C_eeg, eeg_patches)  # patch resolution
    dummy_actual_fmri = torch.randn(B, R, T_fmri)
    dummy_action = torch.randn(B, 2048)

    print("\n--- Perception Mode ---")
    out = model(dummy_eeg, dummy_fmri, mode="perception")
    print(f"  z_global shape: {out['z_global'].shape}")
    print(f"  delta_z shape: {out['delta_z'].shape}")
    print(f"  eeg_recon shape: {out['eeg_recon'].shape}")
    print(f"  fmri_recon shape: {out['fmri_recon'].shape}")

    print("\n--- Perception Mode with Feedback ---")
    model.reset_history()
    out = model(
        dummy_eeg, dummy_fmri,
        mode="perception",
        actual_eeg=dummy_actual_eeg,
        actual_fmri=dummy_actual_fmri,
        action=dummy_action,
    )
    print(f"  corrected_z shape: {out.get('corrected_z').shape if out.get('corrected_z') is not None else 'None'}")
    print(f"  value: {out.get('value').mean().item() if out.get('value') is not None else 'None':.4f}")

    print("\n--- Imagination Mode ---")
    goals = torch.randn(B, 2, 2048)
    cue = torch.randn(B, 2048)
    out = model(dummy_eeg, dummy_fmri, mode="imagination", goal_attractors=goals, task_cue=cue)
    print(f"  selected_action shape: {out['selected_action'].shape}")
    print(f"  selected_trajectory shape: {out['selected_trajectory'].shape}")
    print(f"  imagined_states shape: {out['imagined_states'].shape}")
    print(f"  efe: {out.get('efe').mean().item() if out.get('efe') is not None else 'None':.4f}")

    print("\nAll tests passed!")