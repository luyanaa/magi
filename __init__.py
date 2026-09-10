"""
Brain MoE-PINN: Multi-Attractor Neural Dynamics Model

A ~1.2B parameter Mixture-of-Experts Physics-Informed Neural Network
for modeling brain neural dynamics with GENERIC (General Equation for
Non-Equilibrium Reversible-Irreversible Coupling) dynamics constraints.

Architecture (Revised 2026-05-18):
- Tri-modality encoders: Magi EEG (8L×512d BERT-medium) + NeuroSTORM fMRI (frozen SWM) + MEG (BERT-medium, shared backbone)
- Hub Token fusion for cross-modal representation
- Slow Manifold Projector (256×1024) + Learned Latent HRF Bridge
- VelocityBrain GENERIC-inspired dynamics core (irreversible Δz prediction, d=1024)
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
from .runtime.device_utils import safe_epsilon
from .core.moe import (
    MoEVelocityField,
    MoEGenericVelocityField,
    PoissonRouter,
    WorkingMemoryRouter,
    ExpertNetwork,
)
from .core.observation_adapters import (
    GenericSignalAdapter,
    ChannelSignalAdapter,
    SignalReconstructionHead,
)
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
from .training.losses import TotalLoss
from .training.training_phases import LossWeights
from .data.data_loader import (
    EEGDataset,
    fMRIDataset,
    PairedBrainDataset,
    create_brain_dataloaders,
)
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Dict, List
from .config import (
    ExperimentConfig,
    SUPPORTED_MODALITIES,
    SUPPORTED_SPECIES,
    load_experiment_config,
)

__version__ = "0.3.1"

__all__ = [
    "BrainMoEPINN",
    "BrainMoEPINNConfig",
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
    "GenericSignalAdapter",
    "ChannelSignalAdapter",
    "SignalReconstructionHead",
    "ExperimentConfig",
    "load_experiment_config",
    "SUPPORTED_SPECIES",
    "SUPPORTED_MODALITIES",
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
        latent_dt: Optional[float] = None,
        noise_mode: str = "off",
        control_gating: bool = True,
        use_neurostorm: bool = True,
        use_kda_decoder: bool = True,
        use_active_inference: bool = True,
        active_inference_step: float = 0.1,
        use_mamba2: bool = False,
        mamba2_kwargs: Optional[Dict] = None,
        tau_delay: float = 0.5,
        use_torch_compile: bool = False,
        freeze_encoders_epochs: int = 1,
        use_deep_experts: bool = False,
        shared_depth: int = 50,
        routed_depth: int = 14,
        use_meg: bool = False,
        use_imagination: bool = False,
        use_generic_moe: bool = False,
        perturbation_dim: Optional[int] = None,
        eeg_backend: str = "v1",
        use_magi_v2: bool = False,
        eeg_hidden_dim: int = 1024,
        eeg_num_layers: int = 24,
        use_channel_type_embed: bool = True,
        max_channels: int = 256,
        ecog_amplitude_scale: float = 20.0,
        moe_num_shared: int = 8,
        moe_num_routed: int = 6,
        moe_top_k: int = 3,
        species: str = "human",
        species_vocab: Optional[List[str]] = None,
        use_species_conditioning: bool = False,
        generic_observation_only: bool = False,
        initial_context_length: int = 256,
        max_context_length: int = 1024,
        context_expansion_steps: int = 10000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dt is not None and latent_dt <= 0:
            raise ValueError("latent_dt must be positive")
        self.latent_dt = latent_dt
        self.integration_dt = (
            float(latent_dt) if latent_dt is not None else 1.0)
        if noise_mode not in ("off", "rollout", "train", "always"):
            raise ValueError("noise_mode must be off|rollout|train|always")
        self.noise_mode = noise_mode
        self.control_gating = bool(control_gating)
        self.use_kda_decoder = use_kda_decoder
        self.use_active_inference = use_active_inference
        self.active_inference_step = float(active_inference_step)
        self.use_meg = use_meg
        self.use_imagination = use_imagination
        self.use_generic_moe = use_generic_moe
        self.perturbation_dim = perturbation_dim
        self.eeg_backend = "v2" if use_magi_v2 else eeg_backend
        self.use_channel_type_embed = use_channel_type_embed
        self.moe_num_shared = moe_num_shared
        self.moe_num_routed = moe_num_routed
        self.moe_top_k = moe_top_k
        self.species = species
        self.species_vocab = tuple(species_vocab or SUPPORTED_SPECIES)
        self.use_species_conditioning = use_species_conditioning
        if self.eeg_backend not in {"v1", "v2"}:
            raise ValueError("eeg_backend must be 'v1' or 'v2'")
        if self.moe_num_shared <= 0 or self.moe_num_routed < 0:
            raise ValueError("MoE expert counts are invalid")
        if self.moe_top_k <= 0:
            raise ValueError("moe_top_k must be positive")
        if self.species not in self.species_vocab:
            raise ValueError("species must be present in species_vocab")
        if initial_context_length <= 0 or max_context_length < initial_context_length:
            raise ValueError("invalid context length bounds")
        if context_expansion_steps <= 0:
            raise ValueError("context_expansion_steps must be positive")
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.context_expansion_steps = context_expansion_steps
        self._current_context_length = initial_context_length
        self.generic_observation_only = generic_observation_only
        if generic_observation_only:
            self.eeg_encoder = None
            self.fmri_encoder = None
            self.fmri_projection = None
            self.meg_encoder = None
            self.meg_projection = None
            num_modalities = 2
        else:
            eeg_kwargs = {"in_channels": eeg_channels}
            if use_mamba2:
                eeg_kwargs["use_mamba2"] = True
                if mamba2_kwargs:
                    eeg_kwargs.update({
                        "mamba2_layers": mamba2_kwargs.get("eeg_layers", 4),
                        "mamba2_chunk_size": mamba2_kwargs.get("chunk_size", 256),
                        "mamba2_backend": mamba2_kwargs.get("backend", "triton"),
                    })

            if self.eeg_backend == "v2":
                from .encoders.eeg_encoder_v2 import EEGEncoderWrapperV2
                self.eeg_encoder = EEGEncoderWrapperV2(
                    hidden_dim=eeg_hidden_dim,
                    output_dim=latent_dim,
                    num_layers=eeg_num_layers,
                    freeze_encoder=True,
                    freeze_epochs=freeze_encoders_epochs,
                    max_channels=max(max_channels, eeg_channels),
                    ecog_amplitude_scale=ecog_amplitude_scale,
                    use_channel_type_embed=use_channel_type_embed,
                    mamba2_layers=eeg_kwargs.get("mamba2_layers", 4),
                    mamba2_chunk_size=eeg_kwargs.get("mamba2_chunk_size", 256),
                    mamba2_backend=eeg_kwargs.get("mamba2_backend", "triton"),
                )
            else:
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
        if self.use_species_conditioning:
            self.species_embedding = nn.Embedding(len(self.species_vocab), latent_dim)
            self.species_film = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, 2 * latent_dim),
            )
        else:
            self.species_embedding = None
            self.species_film = None
        self.generic_signal_adapters = nn.ModuleDict({
            modality: GenericSignalAdapter(latent_dim)
            for modality in SUPPORTED_MODALITIES
        })
        # Channel-preserving encoders + shared reconstruction head power
        # full-signal reconstruction for arbitrary modalities (calcium,
        # voltage, widefield, ...) via forward_modalities(reconstruct=True).
        self.generic_channel_adapters = nn.ModuleDict({
            modality: ChannelSignalAdapter(latent_dim)
            for modality in SUPPORTED_MODALITIES
        })
        self.signal_recon_head = SignalReconstructionHead(latent_dim)
        self.generic_modality_embedding = nn.Embedding(
            len(SUPPORTED_MODALITIES), latent_dim)

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
                num_shared=moe_num_shared,
                num_routed=moe_num_routed,
                top_k=moe_top_k,
                use_deep_bias=use_deep_experts,
                shared_depth=shared_depth,
                routed_depth=routed_depth,
            )
        else:
            self.moe_velocity = MoEVelocityField(
                hidden_dim=latent_dim,
                num_shared=moe_num_shared,
                num_routed=moe_num_routed,
                top_k=moe_top_k,
                use_deep_experts=use_deep_experts,
                shared_depth=shared_depth,
                routed_depth=routed_depth,
            )

        self.velocity_brain = VelocityBrain(
            hidden_dim=latent_dim,
            apply_degeneracy_projection=True,
            use_lowrank_poisson=True,
            poisson_rank=64,
            perturbation_dim=perturbation_dim,
            control_gating=self.control_gating,
            latent_dt=self.latent_dt,
        )

        if generic_observation_only:
            self.decoder_router = None
        elif use_kda_decoder:
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
        if self.counterfactual_search is not None:
            self.counterfactual_search.rollout.set_velocity_net(
                self.velocity_brain)

        if use_active_inference:
            # One tree search shared with the controller, and the controller's
            # value function reuses the velocity field's potentials so EFE and
            # the dynamics score the same landscape.
            self.active_inference = ActiveInferenceController(
                latent_dim=latent_dim,
                use_counterfactual=use_imagination,
                use_feedback=True,
                tau_delay=tau_delay,
                energy_entropy_net=self.velocity_brain.energy_entropy,
                counterfactual_search=self.counterfactual_search,
            )
            if self.active_inference.cfts is not None:
                self.active_inference.cfts.rollout.set_velocity_net(
                    self.velocity_brain)


        self.hebbian_memory = HebbianAssociativeMemory(
            hidden_dim=latent_dim,
            memory_dim=512,
        )

        self.kda_state_history = None
        self.max_history = 100
        self._replay_buffer = []
        self.total_loss = TotalLoss(loss_weights=LossWeights())

        if use_torch_compile and hasattr(torch, "compile"):
            self.moe_velocity = torch.compile(
                self.moe_velocity,
                mode="reduce-overhead",
                fullgraph=False,
            )
            if self.eeg_encoder is not None:
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
    def _resolve_species_ids(
        self,
        batch_size: int,
        device: torch.device,
        species_ids: Optional[torch.Tensor] = None,
        species_names=None,
    ) -> torch.Tensor:
        if species_ids is not None:
            ids = torch.as_tensor(species_ids, device=device, dtype=torch.long)
            if ids.dim() == 0:
                ids = ids.expand(batch_size)
            if ids.shape != (batch_size,):
                raise ValueError("species_ids must have shape (B,)")
            if (ids < 0).any() or (ids >= len(self.species_vocab)).any():
                raise ValueError("species_ids contains an unknown species index")
            return ids

        if species_names is None:
            names = [self.species] * batch_size
        elif isinstance(species_names, str):
            names = [species_names] * batch_size
        else:
            names = list(species_names)
            if len(names) != batch_size:
                raise ValueError("species_names must contain one name per batch row")
        try:
            return torch.tensor(
                [self.species_vocab.index(name) for name in names],
                dtype=torch.long,
                device=device,
            )
        except ValueError as exc:
            raise ValueError(f"unknown species in {names}") from exc

    def _apply_species_conditioning(
        self,
        z: torch.Tensor,
        species_ids: Optional[torch.Tensor],
        species_names=None,
    ) -> torch.Tensor:
        if self.species_embedding is None:
            return z
        ids = self._resolve_species_ids(
            z.shape[0], z.device, species_ids=species_ids, species_names=species_names)
        scale, shift = self.species_film(self.species_embedding(ids)).chunk(2, dim=-1)
        return z * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift

    def _latent_step(
        self,
        z: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
        apply_noise: bool = False,
        step_index: int = 0,
        noise_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if perturbation is not None and perturbation.dim() == 3:
            k = min(step_index, perturbation.shape[1] - 1)
            perturbation = perturbation[:, k]
        vb_out = self.velocity_brain(
            z, perturbation=perturbation, apply_noise=apply_noise,
            noise_state=noise_state)
        generic_delta_z = vb_out["delta_z"]

        if self.use_generic_moe:
            moe_out = self.moe_velocity(
                z,
                grad_E=vb_out["grad_E"],
                grad_S=vb_out["grad_S"],
                L_base=vb_out["L_z"],
                M_base=vb_out["M_diag"],
            )
            expert_poisson = torch.bmm(
                moe_out["delta_L"], vb_out["grad_E"].unsqueeze(-1)
            ).squeeze(-1)
            expert_mobility_diag = moe_out["delta_M_diag_action"]
            expert_mobility_lr = torch.bmm(
                moe_out["delta_M_lowrank"],
                vb_out["grad_S"].unsqueeze(-1),
            ).squeeze(-1)
            delta_z = (
                generic_delta_z
                + expert_poisson
                + expert_mobility_diag
                + expert_mobility_lr
                + moe_out["velocity_bias"])
            constraint_residual = moe_out.get(
                "generic_constraint_residual",
                torch.zeros((), dtype=z.dtype, device=z.device))
        else:
            moe_out = self.moe_velocity(z)
            delta_z = generic_delta_z + moe_out["velocity"]
            constraint_residual = (
                vb_out["L_grad_S"].square().sum(dim=-1).mean()
                + vb_out["M_grad_E"].square().sum(dim=-1).mean())

            # Scope diagnostic.  The cheap MoE adds an unconstrained velocity
            # *after* the degeneracy projections, so with this path the total
            # field is not GENERIC even though the backbone is.  Quantify that
            # instead of leaving it implicit: how much of the applied velocity
            # comes from the unstructured experts, and how strongly that part
            # is aligned with grad_E (i.e. how much it changes the learned
            # energy potential, which is exactly what the structure is
            # supposed to control).
            grad_E = vb_out["grad_E"]
            with torch.no_grad():
                expert_velocity = moe_out["velocity"]
                expert_norm = expert_velocity.norm(dim=-1)
                backbone_norm = generic_delta_z.norm(dim=-1)
                total_norm = (expert_norm + backbone_norm).clamp_min(
                    safe_epsilon(expert_norm, 1e-12))
                moe_out["routing_metrics"]["moe_velocity_share"] = float(
                    (expert_norm / total_norm).mean())
                alignment = (expert_velocity * grad_E).sum(dim=-1).abs() / (
                    expert_norm * grad_E.norm(dim=-1)).clamp_min(
                        safe_epsilon(expert_norm, 1e-12))
                moe_out["routing_metrics"]["moe_energy_alignment"] = float(
                    alignment.mean())

        return {
            "vb_out": vb_out,
            "generic_delta_z": generic_delta_z,
            "delta_z": delta_z,
            "noise_state": vb_out.get("new_noise_state"),
            "moe_routing": moe_out["routing_metrics"],
            "grad_E": vb_out["grad_E"],
            "grad_S": vb_out["grad_S"],
            "E": vb_out.get("E"),
            "S": vb_out.get("S"),
            "L_z": vb_out["L_z"],
            "M_diag": vb_out["M_diag"],
            "generic_constraint_residual": constraint_residual,
            "grassmannian_loss": moe_out.get(
                "grassmannian_loss",
                torch.zeros((), dtype=z.dtype, device=z.device)),
        }


    def forward_modalities(
        self,
        signals: Dict[str, torch.Tensor],
        *,
        perturbation: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_names=None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = 1,
        return_all: bool = False,
        reconstruct: bool = False,
        recon_max_channels: int = 2048,
    ) -> Dict[str, torch.Tensor]:
        """Run the shared latent dynamics on arbitrary neural modalities.

        ``signals`` maps modality names such as ``calcium``, ``voltage``,
        ``widefield``, ``eeg``, or ``fmri`` to tensors shaped ``(B,C,T)``.
        The latent path is modality-agnostic; when ``reconstruct=True`` and a
        modality fits under ``recon_max_channels``, per-channel tokens are
        encoded (``ChannelSignalAdapter``) to build the latent, and the raw
        signal is then *generated* from the evolved latent by
        ``SignalReconstructionHead`` (channel identity + learned temporal
        basis, no current-window tokens), emitting ``{modality}_recon`` so
        ``TotalLoss`` can supervise ANY neural signal through
        ``LossWeights.recon_extra``.
        """
        if not signals:
            raise ValueError("signals must contain at least one modality")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        masks = masks or {}
        batch_size = None
        self.reset_runtime_state()
        pooled = []
        modality_tokens = {}
        recon_channels = {}
        for modality, signal in signals.items():
            if modality not in self.generic_signal_adapters:
                raise ValueError(f"unsupported modality {modality!r}")
            if batch_size is None:
                batch_size = signal.shape[0]
            elif signal.shape[0] != batch_size:
                raise ValueError("all modality tensors must share batch size")
            mask = masks.get(modality)
            input_signal = signal
            if mask is not None:
                if mask.shape != signal.shape:
                    raise ValueError(
                        f"mask for {modality!r} must match signal shape")
                input_signal = signal.masked_fill(
                    ~mask.to(device=signal.device, dtype=torch.bool), 0.0)
            modality_index = SUPPORTED_MODALITIES.index(modality)
            modality_embed = self.generic_modality_embedding.weight[
                modality_index].view(1, 1, -1)
            want_recon = (
                reconstruct
                and input_signal.shape[1] <= min(
                    recon_max_channels, self.signal_recon_head.max_channels))
            if want_recon:
                channel_tokens = self.generic_channel_adapters[
                    modality].forward_channels(input_signal)
                recon_channels[modality] = input_signal.shape[1]
                tokens = channel_tokens.mean(dim=1) + modality_embed
            else:
                tokens = self.generic_signal_adapters[
                    modality](input_signal)
                tokens = tokens + modality_embed
            modality_tokens[modality] = tokens
            pooled.append(tokens.mean(dim=1))

        z_global = torch.stack(pooled, dim=0).mean(dim=0)
        z_global = self._apply_species_conditioning(
            z_global, species_ids=species_ids, species_names=species_names)
        z_t = z_global
        states = [z_t] if return_all else []
        control_terms = []
        delta_z_seq = []
        use_noise = (self.noise_mode == "always"
                     or (self.noise_mode == "train" and self.training)
                     or (self.noise_mode == "rollout" and self.training
                         and num_steps > 1))
        noise_state = None
        for k in range(num_steps):
            step_out = self._latent_step(
                z_t, perturbation=perturbation, apply_noise=use_noise,
                step_index=k, noise_state=noise_state)
            noise_state = step_out.get("noise_state")
            delta_z_seq.append(step_out["delta_z"])
            z_t = z_t + self.integration_dt * step_out["delta_z"]
            control_terms.append(step_out["vb_out"]["control_term"])
            if return_all:
                states.append(z_t)

        result = {
            "z_global": z_global,
            "z_next": z_t,
            "delta_z": step_out["delta_z"],
            "generic_delta_z": step_out["generic_delta_z"],
            "control_term": control_terms[-1],
            "moe_routing": step_out["moe_routing"],
            "modality_tokens": modality_tokens,
            "grad_E": step_out["grad_E"],
            "grad_S": step_out["grad_S"],
            "E": step_out.get("E"),
            "S": step_out.get("S"),
            "L_z": step_out["L_z"],
            "M_diag": step_out["M_diag"],
            "generic_constraint_residual": step_out[
                "generic_constraint_residual"],
            "grassmannian_loss": step_out["grassmannian_loss"],
        }
        if len(delta_z_seq) > 1:
            # Rollout velocity trace for temporal losses (velocity TV, ks).
            result["delta_z_sequence"] = torch.stack(delta_z_seq, dim=1)
        # Generate each reconstruct-capable modality from the evolved latent
        # alone (plus channel identity); the encoder's current-window tokens
        # are deliberately not passed to the decoder.
        for modality, num_channels in recon_channels.items():
            result[f"{modality}_recon"] = self.signal_recon_head(
                z_t,
                num_channels=num_channels,
                target_time_len=signals[modality].shape[-1])
        if return_all:
            result["states"] = torch.stack(states, dim=1)
            result["control_terms"] = torch.stack(control_terms, dim=1)
        return result

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
        perturbation: Optional[torch.Tensor] = None,
        subject_features: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        return_all: bool = False,
        species_ids: Optional[torch.Tensor] = None,
        species_names=None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
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
            action: (B, d) executed action (for Smith predictor)
            perturbation: (B, u) exogenous intervention/control input
            subject_features: (B, 3) optional age/sex/region for HRF conditioning
            num_steps: number of latent rollout steps after encoding
            return_all: return latent states and control terms for every step
            species_ids/species_names: per-row species metadata for shared
                cross-species dynamics; ignored when conditioning is disabled
        Returns:
            dict with all outputs including reconstructions and latent states
        """
        if self.generic_observation_only:
            raise RuntimeError(
                "generic_observation_only models require forward_modalities()")
        masks = masks or {}
        self.reset_runtime_state(batch_size=eeg.shape[0])

        def _masked_input(
            signal: Optional[torch.Tensor], modality: str
        ) -> Optional[torch.Tensor]:
            if signal is None:
                return None
            mask = masks.get(modality)
            if mask is None:
                return signal
            if mask.shape != signal.shape:
                raise ValueError(
                    f"mask for {modality!r} must match signal shape")
            return signal.masked_fill(
                ~mask.to(device=signal.device, dtype=torch.bool), 0.0)

        eeg = _masked_input(eeg, "eeg")
        fmri = _masked_input(fmri, "fmri")
        meg = _masked_input(meg, "meg")
        if isinstance(channel_names, list) and channel_names and isinstance(channel_names[0], list):
            channel_names = channel_names[0]
        eeg_out = self.eeg_encoder(
            eeg,
            channel_names,
            channel_types,
            **({"return_embeddings": True} if self.eeg_backend == "v1" else {"return_pooler": False}),
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

        z_global = self._apply_species_conditioning(
            z_global, species_ids=species_ids, species_names=species_names)
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        # Hebbian memory tracks the encoded state; the shared dynamics are
        # rolled out without re-encoding observations at every step.
        self.hebbian_memory.update_kda_state(z_global)
        hrf_residual = self.slow_projector.inverse_project(z_pred_fmri_slow.mean(dim=1))
        z_t = z_global
        states = [z_t] if return_all else []
        control_terms = []
        delta_z_seq = []
        use_noise = (self.noise_mode == "always"
                     or (self.noise_mode == "train" and self.training)
                     or (self.noise_mode == "rollout" and self.training
                         and num_steps > 1))
        noise_state = None
        for k in range(num_steps):
            step_out = self._latent_step(
                z_t, perturbation=perturbation, apply_noise=use_noise,
                step_index=k, noise_state=noise_state)
            noise_state = step_out.get("noise_state")
            z_t = (
                z_t + self.integration_dt * step_out["delta_z"]
                + 0.05 * hrf_residual)
            control_terms.append(step_out["vb_out"]["control_term"])
            if return_all:
                states.append(z_t)

        z_next = z_t
        vb_out = step_out["vb_out"]
        generic_delta_z = step_out["generic_delta_z"]
        delta_z = step_out["delta_z"]
        moe_routing = step_out["moe_routing"]
        grassmannian_loss = step_out["grassmannian_loss"]

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
            "generic_constraint_residual": step_out[
                "generic_constraint_residual"],
            "grassmannian_loss": grassmannian_loss,
            "control_term": vb_out["control_term"],
            "hebbian_weights": self.hebbian_memory.hebbian_weight.W.data,
        }
        if len(delta_z_seq) > 1:
            # Rollout velocity trace for temporal losses (velocity TV, ks).
            result["delta_z_sequence"] = torch.stack(delta_z_seq, dim=1)
        if return_all:
            result["states"] = torch.stack(states, dim=1)
            result["control_terms"] = torch.stack(control_terms, dim=1)
        if hub_meg is not None:
            result["hub_meg"] = hub_meg
        if "meg_recon" in decoder_out:
            result["meg_recon"] = decoder_out["meg_recon"]

        if self.use_active_inference and mode == "perception":
            self._update_kda_history(z_next)

            predicted_parts = []
            actual_parts = []
            for predicted, actual in (
                (decoder_out.get("eeg_recon"), actual_eeg),
                (decoder_out.get("fmri_recon"), actual_fmri),
                (decoder_out.get("meg_recon"), actual_meg),
            ):
                if predicted is None or actual is None:
                    continue
                if predicted.dim() == 3 and actual.dim() == 3:
                    if predicted.shape[1] != actual.shape[1]:
                        raise ValueError(
                            "feedback prediction and target channel counts "
                            "must match")
                    if predicted.shape[-1] != actual.shape[-1]:
                        actual = F.adaptive_avg_pool1d(
                            actual, predicted.shape[-1])
                predicted_parts.append(predicted.reshape(predicted.shape[0], -1))
                actual_parts.append(actual.reshape(actual.shape[0], -1))

            if predicted_parts:
                predicted_signal = torch.cat(predicted_parts, dim=1)
                actual_signal = torch.cat(actual_parts, dim=1)
                w_q = getattr(self, "_wiener_q_factor", None)
                w_a = getattr(self, "_wiener_ataxia", None)
                w_c = getattr(self, "_wiener_catalepsy", None)
                ai_result = self.active_inference.forward_perception(
                    z_next,
                    predicted_signal,
                    actual_signal,
                    action,
                    self.kda_state_history,
                    q_factor=w_q,
                    ataxia_score=w_a,
                    catalepsy_score=w_c,
                )
                result["corrected_z"] = ai_result.get(
                    "corrected_z", z_next)
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

            # EFE-driven correction.  The latent takes a step *down* the
            # free-energy gradient evaluated at the current state, so the
            # update is a variational step rather than a pull toward a detached
            # rollout endpoint.  The gradient flows through z_next, so the
            # dynamics and potentials are trained by the correction instead of
            # being bypassed by it.
            efe, efe_metrics = self.active_inference.value_function.compute_efe(
                z_next, goal_attractors=goal_attractors)
            result["efe"] = efe
            result["efe_metrics"] = efe_metrics

            if z_next.requires_grad:
                efe_grad = torch.autograd.grad(
                    efe.sum(), z_next, create_graph=True, retain_graph=True)[0]
                correction = -self.active_inference_step * efe_grad
                result["z_next"] = z_next + correction
                result["delta_z"] = delta_z + correction
                result["efe_step_norm"] = correction.norm(dim=-1).mean()

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

    def set_training_step(
        self,
        step: Optional[int] = None,
        freeze_epochs_steps: int = 1000,
        total_epochs: Optional[int] = None,
        *,
        epoch: Optional[int] = None,
    ):
        """
        Update model state based on training step.

        Args:
            step: current training step
            freeze_epochs_steps: number of steps equivalent to 1 epoch of frozen encoders
        """
        if step is None:
            if epoch is None:
                raise ValueError("step or epoch must be provided")
            step = epoch
        if total_epochs is not None and hasattr(self.eeg_encoder, "set_training_step"):
            self.eeg_encoder.set_training_step(
                epoch=step,
                total_epochs=total_epochs,
            )
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

    def reset_runtime_state(
        self, batch_size: Optional[int] = None
    ) -> None:
        """Reset per-sequence router and multiscale dynamical memory."""
        if hasattr(self, "moe_velocity"):
            self.moe_velocity.reset_router_state(batch_size)
        if hasattr(self, "velocity_brain"):
            self.velocity_brain.step_counter = 0
            if hasattr(self.velocity_brain, "mt_kda"):
                self.velocity_brain.mt_kda.reset_state(batch_size)

    def reset_history(self):
        """Reset history, replay, router, and multiscale sequence state."""
        self._replay_buffer = []
        self.kda_state_history = None
        self.reset_runtime_state()
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
    """Flat model-facing view over the unified experiment config."""

    def __init__(
        self,
        eeg_channels: int = 19,
        fmri_regions: int = 400,
        meg_channels: int = 306,
        latent_dim: int = 1024,
        latent_dt: Optional[float] = None,
        noise_mode: str = "off",
        control_gating: bool = True,
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
        use_imagination: bool = False,
        use_generic_moe: bool = False,
        perturbation_dim: Optional[int] = None,
        eeg_backend: str = "v1",
        use_magi_v2: bool = False,
        use_channel_type_embed: bool = True,
        max_channels: int = 256,
        ecog_amplitude_scale: float = 20.0,
        moe_num_shared: int = 8,
        moe_num_routed: int = 6,
        moe_top_k: int = 3,
        species: str = "human",
        species_vocab: Optional[List[str]] = None,
        use_species_conditioning: bool = False,
        generic_observation_only: bool = False,
        initial_context_length: int = 256,
        max_context_length: int = 1024,
        context_expansion_steps: int = 10000,
    ):
        self.eeg_channels = eeg_channels
        self.fmri_regions = fmri_regions
        self.meg_channels = meg_channels
        self.latent_dim = latent_dim
        self.latent_dt = latent_dt
        self.noise_mode = noise_mode
        self.control_gating = bool(control_gating)
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
        self.use_imagination = use_imagination
        self.use_generic_moe = use_generic_moe
        self.perturbation_dim = perturbation_dim
        self.eeg_backend = "v2" if use_magi_v2 else eeg_backend
        self.use_magi_v2 = use_magi_v2
        self.max_channels = max_channels
        self.ecog_amplitude_scale = ecog_amplitude_scale
        self.use_channel_type_embed = use_channel_type_embed
        self.moe_num_shared = moe_num_shared
        self.moe_num_routed = moe_num_routed
        self.moe_top_k = moe_top_k
        self.species = species
        self.species_vocab = tuple(species_vocab or SUPPORTED_SPECIES)
        self.use_species_conditioning = use_species_conditioning
        self.generic_observation_only = generic_observation_only
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.context_expansion_steps = context_expansion_steps


    @classmethod
    def from_experiment(cls, config: ExperimentConfig) -> "BrainMoEPINNConfig":
        features = config.features
        kwargs = dict(
            eeg_channels=config.data.channel_count,
            fmri_regions=config.data.region_count or 400,
            latent_dim=config.latent_dim,
            latent_dt=1.0 / config.data.sample_rate_hz,
            noise_mode=getattr(features, "noise_mode", "off"),
            control_gating=getattr(features, "control_gating", True),
            use_kda_decoder=features.use_kda_decoder,
            use_active_inference=features.use_active_inference,
            use_mamba2=features.use_mamba2,
            use_deep_experts=features.use_deep_experts,
            use_meg=features.use_meg,
            use_imagination=features.use_imagination,
            use_generic_moe=features.use_generic_moe,
            perturbation_dim=features.perturbation_dim,
            eeg_backend=features.eeg_backend,
            use_channel_type_embed=features.use_channel_type_embed,
            moe_num_shared=features.moe_num_shared,
            moe_num_routed=features.moe_num_routed,
            moe_top_k=features.moe_top_k,
            generic_observation_only=features.use_generic_observation_adapter,
            species=config.species,
            species_vocab=list(config.species_vocab),
            use_species_conditioning=features.use_species_conditioning,
            max_channels=config.data.max_channels,
            initial_context_length=config.training.initial_context_length,
            max_context_length=config.training.max_context_length,
            context_expansion_steps=config.training.context_expansion_steps,
        )
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str) -> "BrainMoEPINNConfig":
        return cls.from_experiment(load_experiment_config(path))

    def to_model(self) -> BrainMoEPINN:
        """Create the single canonical BrainMoEPINN implementation."""
        return BrainMoEPINN(
            eeg_channels=self.eeg_channels,
            fmri_regions=self.fmri_regions,
            meg_channels=self.meg_channels,
            latent_dim=self.latent_dim,
            latent_dt=self.latent_dt,
            noise_mode=self.noise_mode,
            control_gating=self.control_gating,
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
            use_imagination=self.use_imagination,
            use_generic_moe=self.use_generic_moe,
            perturbation_dim=self.perturbation_dim,
            eeg_backend=self.eeg_backend,
            use_channel_type_embed=self.use_channel_type_embed,
            max_channels=self.max_channels,
            ecog_amplitude_scale=self.ecog_amplitude_scale,
            moe_num_shared=self.moe_num_shared,
            moe_num_routed=self.moe_num_routed,
            moe_top_k=self.moe_top_k,
            generic_observation_only=self.generic_observation_only,
            initial_context_length=self.initial_context_length,
            max_context_length=self.max_context_length,
            context_expansion_steps=self.context_expansion_steps,
            species=self.species,
            species_vocab=list(self.species_vocab),
            use_species_conditioning=self.use_species_conditioning,
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