"""Unified model, species, modality, and data-ladder configuration.

The configuration is deliberately independent of any one observation model.
Species and modality profiles describe the data contract; the shared latent
model consumes the resulting metadata and can keep its dynamics parameters
shared across ladder stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


SUPPORTED_SPECIES = ("c_elegans", "zebrafish", "mouse", "human")
SUPPORTED_MODALITIES = (
    "calcium",
    "voltage",
    "behavior",
    "widefield",
    "eeg",
    "ecog",
    "meg",
    "fmri",
)


@dataclass(frozen=True)
class SpeciesProfile:
    """Species-level defaults; these are not anatomical equivalence claims."""

    name: str
    modalities: Tuple[str, ...]
    sample_rate_hz: float
    max_channels: int
    region_count: Optional[int]
    channel_count: int
    time_unit: str = "seconds"
    recon_loss_types: Dict[str, str] = field(default_factory=dict)
    """Default ReconstructionLoss criterion per modality.

    Evidence-based defaults (TDE-RICA / C. elegans validation practice):
    MSE is dominated by per-channel gain and the strongly autocorrelated
    slow baseline of calcium fluorescence, so whole-brain calcium is compared
    with correlation-style similarity and per-component marginal/gain
    diagnostics. ``correlation`` is scale/shift invariant per channel;
    ``huber`` tolerates spike outliers in voltage recordings. Any modality
    without an entry keeps the generic MSE default.
    """


SPECIES_PROFILES = {
    "c_elegans": SpeciesProfile(
        name="c_elegans",
        modalities=("calcium", "voltage", "behavior"),
        sample_rate_hz=10.0,
        max_channels=302,
        region_count=302,
        channel_count=302,
        recon_loss_types={"calcium": "correlation", "voltage": "huber"},
    ),
    "zebrafish": SpeciesProfile(
        name="zebrafish",
        modalities=("calcium", "voltage", "behavior"),
        sample_rate_hz=10.0,
        max_channels=10000,
        region_count=None,
        channel_count=1024,
        recon_loss_types={"calcium": "correlation", "voltage": "huber"},
    ),
    "mouse": SpeciesProfile(
        name="mouse",
        modalities=("calcium", "voltage", "widefield", "fmri", "behavior"),
        sample_rate_hz=30.0,
        max_channels=1024,
        region_count=400,
        channel_count=64,
        recon_loss_types={"calcium": "correlation", "voltage": "huber"},
    ),
    "human": SpeciesProfile(
        name="human",
        modalities=("eeg", "ecog", "meg", "fmri", "behavior"),
        sample_rate_hz=256.0,
        max_channels=1024,
        region_count=400,
        channel_count=64,
    ),
}


@dataclass
class FeatureConfig:
    """Optional model features with explicit capability gates."""

    eeg_backend: str = "v1"
    use_kda_decoder: bool = True
    use_active_inference: bool = False
    use_imagination: bool = False
    use_mamba2: bool = False
    use_deep_experts: bool = False
    use_generic_moe: bool = False
    use_species_conditioning: bool = True
    use_generic_observation_adapter: bool = False
    use_stochastic_noise: bool = False
    noise_mode: str = "off"
    """SDE noise policy: off | rollout | train | always (see
    BrainMoEPINN docstring; default preserves deterministic steps)."""
    control_gating: bool = True
    use_meg: bool = False
    use_channel_type_embed: bool = True
    moe_num_shared: int = 8
    moe_num_routed: int = 6
    moe_top_k: int = 3
    perturbation_dim: Optional[int] = None

@dataclass
class DataConfig:
    """Canonical data contract passed to modality adapters."""

    modalities: Tuple[str, ...] = ("eeg", "fmri")
    sample_rate_hz: float = 256.0
    time_unit: str = "seconds"
    sequence_seconds: float = 10.0
    channel_count: int = 19
    max_channels: int = 256
    region_count: Optional[int] = 400
    atlas: Optional[str] = "schaefer_400"
    paired_next_step_targets: bool = False
    roles: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Data-ladder and evaluation controls."""

    ladder_stage: str = "human"
    freeze_shared_dynamics: bool = False
    freeze_observation_adapters: bool = False
    replay_species: Tuple[str, ...] = ()
    leave_subject_out: bool = True
    batch_size: int = 1
    initial_context_length: int = 256
    max_context_length: int = 1024
    context_expansion_steps: int = 10000


@dataclass
class ExperimentConfig:
    """Validated configuration for one species/modality experiment."""

    species: str = "human"
    species_vocab: Tuple[str, ...] = SUPPORTED_SPECIES
    latent_dim: int = 1024
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        self.species_vocab = tuple(self.species_vocab)
        self.data.modalities = tuple(self.data.modalities)
        self.data.roles = dict(self.data.roles)
        self.training.replay_species = tuple(self.training.replay_species)
        self.validate()

    @classmethod
    def from_species(
        cls,
        species: str,
        *,
        modalities: Optional[Iterable[str]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "ExperimentConfig":
        if species not in SPECIES_PROFILES:
            raise ValueError(
                f"unknown species {species!r}; expected one of {SUPPORTED_SPECIES}")
        profile = SPECIES_PROFILES[species]
        values: Dict[str, Any] = {
            "species": species,
            "features": {
                "use_kda_decoder": species == "human",
                "use_generic_observation_adapter": species != "human",
            },
            "data": {
                "modalities": tuple(modalities or profile.modalities),
                "sample_rate_hz": profile.sample_rate_hz,
                "max_channels": profile.max_channels,
                "channel_count": profile.channel_count,
                "region_count": profile.region_count,
                "time_unit": profile.time_unit,
            },
            "training": {"ladder_stage": species},
        }
        if overrides:
            values = _deep_merge(values, dict(overrides))
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        values = dict(values)
        features = FeatureConfig(**dict(values.pop("features", {})))
        data = DataConfig(**dict(values.pop("data", {})))
        training = TrainingConfig(**dict(values.pop("training", {})))
        return cls(features=features, data=data, training=training, **values)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError("configuration files use JSON; YAML loading is not implicit")
        with path.open() as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.species not in SUPPORTED_SPECIES:
            raise ValueError(f"unknown species {self.species!r}")
        if not self.species_vocab or self.species not in self.species_vocab:
            raise ValueError("species must be included in species_vocab")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        unknown_modalities = set(self.data.modalities) - set(SUPPORTED_MODALITIES)
        invalid_roles = set(self.data.roles.values()) - {
            "signal", "control", "aux", "graph"}
        if invalid_roles:
            raise ValueError(
                f"unknown modality roles: {sorted(invalid_roles)}")
        unknown_role_modalities = set(self.data.roles) - set(
            self.data.modalities)
        if unknown_role_modalities:
            raise ValueError(
                f"roles reference modalities not requested: "
                f"{sorted(unknown_role_modalities)}")
        unsupported_signals = {
            modality for modality in unknown_modalities
            if self.data.roles.get(modality, "signal") == "signal"
        }
        if unsupported_signals:
            raise ValueError(
                f"unsupported signal modalities: {sorted(unsupported_signals)}")
        if not self.data.modalities:
            raise ValueError("at least one modality is required")
        if self.data.sample_rate_hz <= 0 or self.data.sequence_seconds <= 0:
            raise ValueError("sample_rate_hz and sequence_seconds must be positive")
        if self.data.max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if self.data.channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if self.data.channel_count > self.data.max_channels:
            raise ValueError("channel_count cannot exceed max_channels")
        if self.data.region_count is not None and self.data.region_count <= 0:
            raise ValueError("region_count must be positive when provided")
        if self.features.noise_mode not in ("off", "rollout", "train", "always"):
            raise ValueError("noise_mode must be off|rollout|train|always")
        if self.features.eeg_backend not in {"v1", "v2"}:
            raise ValueError("eeg_backend must be 'v1' or 'v2'")
        if self.features.eeg_backend == "v2" and not ({"eeg", "ecog"} & set(self.data.modalities)):
            raise ValueError("eeg_backend='v2' requires eeg or ecog data")
        if self.features.use_meg and "meg" not in self.data.modalities:
            raise ValueError("use_meg requires the meg modality")
        if self.features.use_active_inference and not self.data.paired_next_step_targets:
            raise ValueError("use_active_inference requires paired_next_step_targets=true")
        if self.features.moe_num_shared <= 0 or self.features.moe_num_routed < 0:
            raise ValueError("MoE expert counts are invalid")
        if self.features.moe_top_k <= 0:
            raise ValueError("moe_top_k must be positive")
        if self.features.perturbation_dim is not None and self.features.perturbation_dim <= 0:
            raise ValueError("perturbation_dim must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            self.training.initial_context_length <= 0
            or self.training.max_context_length < self.training.initial_context_length
        ):
            raise ValueError("invalid context length bounds")
        if self.training.context_expansion_steps <= 0:
            raise ValueError("context_expansion_steps must be positive")
        if self.training.ladder_stage not in (*SUPPORTED_SPECIES, "pretrain"):
            raise ValueError("ladder_stage must be pretrain or a supported species")
        unknown_replay = set(self.training.replay_species) - set(self.species_vocab)
        if unknown_replay:
            raise ValueError(f"unknown replay species: {sorted(unknown_replay)}")

    def model_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments accepted by the canonical model."""
        return {
            "latent_dim": self.latent_dim,
            "eeg_backend": self.features.eeg_backend,
            "use_kda_decoder": self.features.use_kda_decoder,
            "use_active_inference": self.features.use_active_inference,
            "use_imagination": self.features.use_imagination,
            "use_mamba2": self.features.use_mamba2,
            "use_deep_experts": self.features.use_deep_experts,
            "use_generic_moe": self.features.use_generic_moe,
            "use_species_conditioning": self.features.use_species_conditioning,
            "generic_observation_only": self.features.use_generic_observation_adapter,
            "use_meg": self.features.use_meg,
            "use_channel_type_embed": self.features.use_channel_type_embed,
            "moe_num_shared": self.features.moe_num_shared,
            "moe_num_routed": self.features.moe_num_routed,
            "moe_top_k": self.features.moe_top_k,
            "species": self.species,
            "species_vocab": self.species_vocab,
            "perturbation_dim": self.features.perturbation_dim,
            "initial_context_length": self.training.initial_context_length,
            "max_context_length": self.training.max_context_length,
            "context_expansion_steps": self.training.context_expansion_steps,
        }
def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_file(path)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), dict(value))
        else:
            result[key] = value
    return result
