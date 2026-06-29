"""
Brain MoE-PINN: Multi-Attractor Neural Dynamics Model

A ~1.2B parameter Mixture-of-Experts Physics-Informed Neural Network
for modeling brain neural dynamics with GENERIC (General Equation for
Non-Equilibrium Reversible-Irreversible Coupling) dynamics constraints.

Architecture (Revised 2026-05-18):
- Tri-modality encoders: Magi EEG (8L×512d BERT-medium) + NeuroSTORM fMRI (frozen SWM) + MEG (BERT-medium, shared backbone)
- Hub Token fusion for cross-modal representation
- Slow Manifold Projector (256×1024) + Learned Latent HRF Bridge
- VelocityBrain GENERIC dynamics core (irreversible Δz prediction, d=1024)
- MoE: 8 Shared experts (always-on, ~420M baseline dynamics) + 6 Routed experts (Top-3, τ curriculum)
- Routing temperature curriculum (τ: 2.0→0.7) via PoissonSSMRouter
- Multi-Time-Scale KDA for memory hierarchies
- KDA-based decoders (Kimi Delta Attention from fla library)
- Counterfactual Tree Search for imagination/action planning
- Active Inference with Value Function, Efference Copy, Smith Predictor, Precision Gate

Reference: Brain MoE-PINN Training Plan (kilo plan: 1778463235296-neon-cabin.md)
"""

from .encoders.eeg_encoder import EEGEncoderWrapper, EEGProjection
from .encoders.fmri_encoder import NeuroSTORMEncoder, BrainLMEncoder, create_fmri_encoder
from .encoders.meg_encoder import MEGEncoderWrapper, MEGProjection
from .encoders.hub_fusion import HubTokenFusion, CrossModalAdapter
from .core.velocity_brain import VelocityBrain, MultiTimeScaleKDA
from .core.moe import PoissonRouter, MoEVelocityField, MoEGenericVelocityField, WorkingMemoryRouter, ExpertNetwork
from .encoders.hub_fusion import HubTokenFusion, CrossModalAdapter, SlowManifoldProjector, LatentHRFBridge
from .core.hebbian_memory import (
    HebbianAssociativeMemory,
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
from .decoder.modality_decoder import EEGDecoder, fMRIDecoder, MEGDecoder, ModalityDecoderRouter
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
    "MEGEncoderWrapper",
    "MEGProjection",
    # Fusion
    "HubTokenFusion",
    "CrossModalAdapter",
    # Core
    "VelocityBrain",
    "MultiTimeScaleKDA",
    "PoissonRouter",
    "MoEVelocityField",
    "MoEGenericVelocityField",
    "WorkingMemoryRouter",
    "ExpertNetwork",
    # Memory
    "HebbianAssociativeMemory",
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
    """..."""

    def __init__(
        self,
        eeg_channels: int = 19,
        fmri_regions: int = 400,
        meg_channels: int = 306,
        latent_dim: int = 1024,
        use_neurostorm: bool = True,
        use_kda_decoder: bool = True,
        use_active_inference: bool = True,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
        tau_delay: float = 0.5,
        freeze_encoders_epochs: int = 1,
        use_torch_compile: bool = False,
        use_deep_experts: bool = False,
        shared_depth: int = 50,
        routed_depth: int = 14,
        use_meg: bool = False,
        use_imagination: bool = False,
        use_generic_moe: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_kda_decoder = use_kda_decoder
        self.use_active_inference = use_active_inference
        self.use_meg = use_meg
        self.use_imagination = use_imagination
        self.use_generic_moe = use_generic_moe

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
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
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

        # Optional MEG branch (shares BERT-medium backbone with EEG)
        if use_meg:
            self.meg_encoder = MEGEncoderWrapper(
                num_channels=meg_channels,
                hidden_dim=512,
                output_dim=latent_dim,
                num_layers=8,
                num_heads=8,
            )
            self.meg_projection = MEGProjection(
                input_dim=512,
                output_dim=latent_dim,
            )
            num_modalities = 3
        else:
            self.meg_encoder = None
            self.meg_projection = None
            num_modalities = 2

        self.hub_fusion = HubTokenFusion(
            hidden_dim=latent_dim,
            num_heads=8,
            dropout=0.1,
            num_modalities=num_modalities,
        )

        self.slow_projector = SlowManifoldProjector(
            latent_dim=latent_dim,
            slow_dim=256,
        )

        self.latent_hrf = LatentHRFBridge(
            slow_dim=256,
            hr_length=32,
            use_subject_conditioning=True,
        )

        if use_generic_moe:
            self.moe_velocity = MoEGenericVelocityField(
                hidden_dim=latent_dim,
                use_deep_bias=use_deep_experts,
                shared_depth=shared_depth,
                routed_depth=routed_depth,
            )
        else:
            self.moe_velocity = MoEVelocityField(
                hidden_dim=latent_dim,
                use_deep_experts=use_deep_experts,
                shared_depth=shared_depth,
                routed_depth=routed_depth,
            )

        self.velocity_brain = VelocityBrain(
            hidden_dim=latent_dim,
            apply_degeneracy_projection=True,
            use_lowrank_poisson=True,
            poisson_rank=64,
        )

        if use_kda_decoder:
            self.decoder_router = KIMIKDAMoDeCoderRouter(
                latent_dim=latent_dim,
                output_channels=eeg_channels,
                num_regions=fmri_regions,
                meg_channels=meg_channels,
                num_layers=4,
                use_meg=use_meg,
            )
        else:
            self.decoder_router = ModalityDecoderRouter(
                latent_dim=latent_dim,
                use_meg=use_meg,
                meg_channels=meg_channels,
            )

        if use_active_inference:
            self.active_inference = ActiveInferenceController(
                latent_dim=latent_dim,
                use_counterfactual=use_imagination,
                use_feedback=True,
                tau_delay=tau_delay,
            )

        if use_imagination:
            self.counterfactual_search = CounterfactualTreeSearch(
                latent_dim=latent_dim,
                num_goals=2,
                num_exploration=1,
                rollout_steps=3,
                branch_factor=4,
            )
            self.imagination_sampler = ImaginationSampler(latent_dim=latent_dim)
        else:
            self.counterfactual_search = None
            self.imagination_sampler = None

        self.hebbian_memory = HebbianAssociativeMemory(
            hidden_dim=latent_dim,
            memory_dim=512,
        )

        self.kda_state_history = None
        self.max_history = 100
        self._replay_buffer = []

        if use_torch_compile and hasattr(torch, "compile"):
            self.moe_velocity = torch.compile(
                self.moe_velocity,
                mode="reduce-overhead",
                fullgraph=False,
            )
            self.eeg_encoder = torch.compile(
                self.eeg_encoder,
                mode="reduce-overhead",
                fullgraph=False,
            )
            if use_meg and self.meg_encoder is not None:
                self.meg_encoder = torch.compile(
                    self.meg_encoder,
                    mode="reduce-overhead",
                    fullgraph=False,
                )

    def forward(
        self,
        eeg: torch.Tensor,
        fmri: torch.Tensor,
        meg: Optional[torch.Tensor] = None,
        channel_names: Optional[List[str]] = None,
        channel_types: Optional[torch.Tensor] = None,
        mode: str = "perception",
        goal_attractors: Optional[torch.Tensor] = None,
        task_cue: Optional[torch.Tensor] = None,
        actual_eeg: Optional[torch.Tensor] = None,
        actual_fmri: Optional[torch.Tensor] = None,
        actual_meg: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        subject_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through Brain MoE-PINN.

        Args:
            eeg: (B, C_eeg, T) raw EEG signals (encoder input)
            fmri: (B, R, T_fmri) ROI time series (encoder input)
            meg: (B, C_meg, T) optional raw MEG signals
            channel_names: Optional channel names for BIOT embedding
            mode: 'perception' (forced ODE with feedback) or 'imagination' (free)
            goal_attractors: (B, num_goals, d) goal attractor centers
            task_cue: (B, d) task cue for imagination
            actual_eeg: (B, C_eeg, T) actual EEG at t+1 (for perception feedback)
            actual_fmri: (B, R, T_fmri) actual fMRI at t+1 (for perception feedback)
            actual_meg: (B, C_meg, T) actual MEG at t+1
            action: (B, d) executed action (for Smith predictor)
            subject_features: (B, 3) optional age/sex/region for HRF conditioning
        Returns:
            dict with all outputs including reconstructions and latent states
        """
        eeg_out = self.eeg_encoder(
            eeg, channel_names, channel_types, return_embeddings=True
        )
        eeg_tokens = eeg_out["embeddings"]

        fmri_out = self.fmri_encoder(fmri)
        fmri_tokens = self.fmri_projection(fmri_out["last_hidden"])

        # Optional MEG branch (shares BERT-medium backbone with EEG)
        meg_tokens = None
        if self.use_meg and meg is not None and self.meg_encoder is not None:
            meg_out = self.meg_encoder(meg, return_embeddings=True)
            meg_tokens = meg_out["embeddings"]

        # --- Slow Manifold Projector + Latent HRF Bridge ---
        # Operate on full token sequences (not just hub tokens)
        eeg_slow = self.slow_projector(eeg_tokens)   # (B, L_eeg, 256)
        fmri_slow = self.slow_projector(fmri_tokens)  # (B, L_fmri, 256)
        z_pred_fmri_slow, hrf_align_loss = self.latent_hrf(
            eeg_slow, fmri_slow, subject_features=subject_features
        )

        # Hub token fusion
        if self.use_meg and meg_tokens is not None:
            z_global, hubs = self.hub_fusion(eeg_tokens, fmri_tokens, meg_tokens)
            hub_eeg = hubs["eeg"]
            hub_fmri = hubs["fmri"]
            hub_meg = hubs["meg"]
        else:
            z_global, hubs = self.hub_fusion(eeg_tokens, fmri_tokens)
            hub_eeg = hubs["eeg"]
            hub_fmri = hubs["fmri"]
            hub_meg = None

        # VelocityBrain GENERIC dynamics: L(z)∇E + M(z)∇S + v₀e(θ) + √(2D)ξ
        vb_out = self.velocity_brain(z_global, apply_noise=False)
        generic_delta_z = vb_out["delta_z"]

        # Hebbian memory update
        self.hebbian_memory.update_kda_state(z_global)

        if self.use_generic_moe:
            # MoE produces physics-structured perturbations (delta_L, delta_M_diag, delta_M_lowrank, bias)
            moe_out = self.moe_velocity(
                z_global,
                grad_E=vb_out["grad_E"],
                grad_S=vb_out["grad_S"],
                L_base=vb_out["L_z"],
                M_base=vb_out["M_diag"],
            )
            # Expert perturbation to conservative dynamics: delta_L @ grad_E
            expert_poisson = torch.bmm(
                moe_out["delta_L"], vb_out["grad_E"].unsqueeze(-1)
            ).squeeze(-1)
            # Expert perturbation to dissipative dynamics:
            #   (P_E diag(delta_M_diag) P_E) @ grad_S  +  (P_E V V^T P_E) @ grad_S
            # The diagonal action is pre-computed exactly via rank-1 formulation in O(d).
            # The low-rank action uses projected V factors: (P_E V)(P_E V)^T @ grad_S.
            expert_mobility_diag = moe_out["delta_M_diag_action"]
            expert_mobility_lr = torch.bmm(
                moe_out["delta_M_lowrank"], vb_out["grad_S"].unsqueeze(-1)
            ).squeeze(-1)
            expert_mobility = expert_mobility_diag + expert_mobility_lr
            # Combined: base GENERIC + expert perturbations + learned bias
            delta_z = generic_delta_z + expert_poisson + expert_mobility + moe_out["velocity_bias"]
            moe_routing = moe_out["routing_metrics"]
            grassmannian_loss = moe_out.get("grassmannian_loss", torch.tensor(0.0))
        else:
            # Standard MoE: arbitrary velocity addition
            moe_out = self.moe_velocity(z_global)
            delta_z = generic_delta_z + moe_out["velocity"]
            moe_routing = moe_out["routing_metrics"]
            grassmannian_loss = moe_out.get("grassmannian_loss", torch.tensor(0.0))

        # Add small cross-modal HRF residual to z_next for physics grounding
        hrf_residual = self.slow_projector.inverse_project(z_pred_fmri_slow.mean(dim=1))
        z_next = z_global + delta_z + 0.05 * hrf_residual

        # Decoder routing
        if self.use_kda_decoder:
            eeg_patches = (eeg.shape[-1] - 256) // 128 + 1 if eeg.shape[-1] >= 256 else 1
            fmri_seq_len = fmri.shape[-1]
            decoder_kwargs = {
                "eeg_seq_length": max(1, eeg_patches),
                "fmri_seq_length": fmri_seq_len,
            }
            if self.use_meg and hub_meg is not None:
                decoder_kwargs["hub_meg"] = hub_meg
                decoder_kwargs["meg_seq_length"] = meg.shape[-1] if meg is not None else 256
            decoder_out = self.decoder_router(z_next, hub_eeg, hub_fmri, **decoder_kwargs)
        else:
            if self.use_meg and hub_meg is not None:
                decoder_out = self.decoder_router(z_next, hub_eeg=hub_eeg, hub_fmri=hub_fmri, hub_meg=hub_meg)
            else:
                decoder_out = self.decoder_router(z_next, hub_eeg=hub_eeg, hub_fmri=hub_fmri)

        result = {
            "z_global": z_global,
            "z_next": z_next,
            "delta_z": delta_z,
            "hub_eeg": hub_eeg,
            "hub_fmri": hub_fmri,
            "eeg_recon": decoder_out["eeg_recon"],
            "fmri_recon": decoder_out["fmri_recon"],
            "moe_routing": moe_routing,
            "eeg_slow": eeg_slow,
            "fmri_slow": fmri_slow,
            "z_pred_fmri_slow": z_pred_fmri_slow,
            "hrf_align_loss": hrf_align_loss,
            "grad_E": vb_out["grad_E"],
            "grad_S": vb_out["grad_S"],
            "L_z": vb_out["L_z"],
            "M_diag": vb_out["M_diag"],
            "generic_delta_z": generic_delta_z,
            "grassmannian_loss": grassmannian_loss,
            "hebbian_weights": self.hebbian_memory.hebbian_weight.W.data,
        }
        if hub_meg is not None:
            result["hub_meg"] = hub_meg
        if "meg_recon" in decoder_out:
            result["meg_recon"] = decoder_out["meg_recon"]

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

        if mode == "imagination" and self.use_imagination and self.counterfactual_search is not None:
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
            result["z_imagined"] = imagined["mean"][:, 0, :]
            if self.hebbian_memory.kda_state is not None:
                result["z_engram"] = self.hebbian_memory.kda_state

            if self.use_active_inference:
                efe, efe_metrics = self.active_inference.value_function.compute_efe(
                    cfts_result["selected_trajectory"][:, -1],
                    goal_attractors=goal_attractors,
                )
                result["efe"] = efe
                result["efe_metrics"] = efe_metrics


        if mode == "perception" and self.use_imagination and self.counterfactual_search is not None and self.training and getattr(self, "_imagination_active", True):
            cfts_result = self.counterfactual_search(
                z_next,
                goal_attractors=goal_attractors,
                task_cue=task_cue,
            )
            best_action = cfts_result["selected_action"]

            imagined = self.imagination_sampler(z_next, num_samples=4)
            result["z_imagined"] = imagined["mean"][:, 0, :]
            if self.hebbian_memory.kda_state is not None:
                result["z_engram"] = self.hebbian_memory.kda_state

            # EFE: computable whenever we have imagination, even without
            # the full active inference perception loop
            if self.use_active_inference:
                efe, efe_metrics = self.active_inference.value_function.compute_efe(
                    cfts_result["selected_trajectory"][:, -1],
                    goal_attractors=goal_attractors,
                )
            else:
                # Standalone EFE: negative value = -V(z) = E(z) - T*S(z)
                # Approximate from energy/entropy if available, else use
                # trajectory norm as a simple proxy
                traj_endpoint = cfts_result["selected_trajectory"][:, -1, :]
                efe = 0.5 * (traj_endpoint - z_next.detach()).pow(2).mean()
                efe_metrics = {"efe_proxy": efe.item()}
            result["efe"] = efe
            result["efe_metrics"] = efe_metrics

            # EFE-driven velocity correction: the selected action's
            # trajectory endpoint is used to bias the velocity field.
            # This is the "active inference" loop: imagination → EFE
            # → action selection → velocity correction.
            best_traj = cfts_result["selected_trajectory"]
            if best_traj.dim() == 3 and best_traj.shape[1] > 0:
                action_target = best_traj[:, -1, :]
                action_correction = 0.1 * (action_target - z_next.detach())
                result["z_next"] = z_next + action_correction
                result["delta_z"] = delta_z + action_correction

            # Replay: store imagined states for later comparison with real observations
            self._replay_buffer.append(imagined["mean"][:, 0, :].detach())
            if len(self._replay_buffer) > 256:
                self._replay_buffer = self._replay_buffer[-256:]

            # If we have stored replay states, provide a replay pair
            # (imagined past vs actual current) for the replay loss
            if len(self._replay_buffer) >= 2 and self.kda_state_history is not None:
                result["z_engram"] = self._replay_buffer[-2]

        return result

    def _update_kda_history(self, z: torch.Tensor):
        """Update KDA state history for Smith predictor. Shape: (B, T, d)."""
        z_unsqueezed = z.detach().float().unsqueeze(1)
        if self.kda_state_history is None:
            self.kda_state_history = z_unsqueezed
        else:
            self.kda_state_history = torch.cat([
                self.kda_state_history, z_unsqueezed
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
            if self.use_meg and self.meg_encoder is not None and hasattr(self.meg_encoder, 'set_freeze'):
                self.meg_encoder.set_freeze(True)
        else:
            # Thaw encoders with reduced LR
            if hasattr(self.eeg_encoder, 'freeze_encoder'):
                self.eeg_encoder.freeze_encoder = False
            if hasattr(self.fmri_encoder, 'set_freeze'):
                self.fmri_encoder.set_freeze(False)
            if self.use_meg and self.meg_encoder is not None and hasattr(self.meg_encoder, 'set_freeze'):
                self.meg_encoder.set_freeze(False)

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
        if self.use_meg and self.meg_encoder is not None and hasattr(self.meg_encoder, "set_context_length"):
            self.meg_encoder.set_context_length(context_length)
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
        self._replay_buffer = []
        self.kda_state_history = None
        if hasattr(self, "moe_velocity"):
            self.moe_velocity.reset_router_state()
        if self.use_active_inference:
            self.active_inference.reset()

    def reset_router_state(self, batch_size: Optional[int] = None):
        """Reset the SSM router state (e.g., at phase boundaries)."""
        if hasattr(self, "moe_velocity"):
            self.moe_velocity.reset_router_state(batch_size)

    def get_num_params(self, trainable_only: bool = False) -> Dict[str, int]:
        """Return parameter counts by component."""
        counts = {}
        for name, mod in self.named_children():
            n = sum(p.numel() for p in mod.parameters() if (not trainable_only or p.requires_grad))
            counts[name] = n
        total = sum(p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad))
        counts["total"] = total
        return counts


class BrainMoEPINNConfig:
    """Configuration class for BrainMoEPINN."""

    def __init__(
        self,
        eeg_channels: int = 19,
        fmri_regions: int = 400,
        meg_channels: int = 306,
        latent_dim: int = 1024,
        use_neurostorm: bool = True,
        use_kda_decoder: bool = True,
        use_active_inference: bool = True,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
        tau_delay: float = 0.5,
        use_torch_compile: bool = False,
        use_deep_experts: bool = False,
        shared_depth: int = 50,
        routed_depth: int = 14,
        use_meg: bool = False,
    ):
        self.eeg_channels = eeg_channels
        self.fmri_regions = fmri_regions
        self.meg_channels = meg_channels
        self.latent_dim = latent_dim
        self.use_neurostorm = use_neurostorm
        self.use_kda_decoder = use_kda_decoder
        self.use_active_inference = use_active_inference
        self.use_mamba2 = use_mamba2
        self.mamba2_kwargs = mamba2_kwargs or {}
        self.tau_delay = tau_delay
        self.use_torch_compile = use_torch_compile
        self.use_deep_experts = use_deep_experts
        self.shared_depth = shared_depth
        self.routed_depth = routed_depth
        self.use_meg = use_meg

    def to_model(self) -> BrainMoEPINN:
        """Create model from config."""
        return BrainMoEPINN(
            eeg_channels=self.eeg_channels,
            fmri_regions=self.fmri_regions,
            meg_channels=self.meg_channels,
            latent_dim=self.latent_dim,
            use_neurostorm=self.use_neurostorm,
            use_kda_decoder=self.use_kda_decoder,
            use_active_inference=self.use_active_inference,
            use_mamba2=self.use_mamba2,
            mamba2_kwargs=self.mamba2_kwargs,
            tau_delay=self.tau_delay,
            use_torch_compile=self.use_torch_compile,
            use_deep_experts=self.use_deep_experts,
            shared_depth=self.shared_depth,
            routed_depth=self.routed_depth,
            use_meg=self.use_meg,
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