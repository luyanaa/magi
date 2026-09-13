"""Executable training phases for Brain MoE-PINN.

The schedule is concrete rather than a list of aspirations:

- Phase -1: Magi EEG masked/causal/contrastive pretraining.
- Stage 1: full-model P1-P6 curriculum, including explicit multi-horizon
  forecasts where the loader supplies future targets.
- Stage 2: labelled cross-modal latent/HRF bridging.
- Stage 3: online adaptation; policy and intervention losses stay disabled
  until their executed-action/replay/effect contracts are supplied.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Dict, Mapping, Optional, Tuple
from enum import Enum

class TrainingStage(Enum):
    PHASE_NEG_1 = "phase_-1"
    STAGE_1_P1 = "stage_1_p1"
    STAGE_1_P2 = "stage_1_p2"
    STAGE_1_P3 = "stage_1_p3"
    STAGE_1_P4 = "stage_1_p4"
    STAGE_1_P5 = "stage_1_p5"
    STAGE_1_P6 = "stage_1_p6"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"


class TrainingTask(str, Enum):
    """Executable task selected by a normalized phase."""

    MODEL_TRAINING = "model_training"
    MAGI_EEG_PRETRAINING = "magi_eeg_pretraining"


def evaluate_transition_gate(
    gate: Optional[Callable[[Mapping[str, Any]], bool]],
    metrics: Mapping[str, Any],
) -> bool:
    """Evaluate a phase gate without allowing ambiguous truthiness."""
    if gate is None:
        return True
    if not callable(gate):
        raise TypeError("transition_gate must be callable or None")
    result = gate(metrics)
    if not isinstance(result, bool):
        raise TypeError("transition_gate must return bool")
    return result


@dataclass
class LossWeights:
    """Steering weights per training stage.

    Positive weights are executable optimizer terms; zero-weight terms remain
    available as diagnostics only.  Reconstruction weights cover EEG/fMRI/MEG
    plus arbitrary generic modalities through ``recon_extra``.  The forecast
    weight is separate: it is enabled only when the loader supplies an
    explicit ``(B, K, C, T)`` future target and the phase rolls out K states.
    The composite forecast combines robust signal error with first-difference
    correlation, using optional horizon weights.

    Structural terms are only steering terms when their inputs are observable:
    ``cross_modal`` and ``cross`` need explicit synchronized/async pair labels,
    ``action`` needs an executed-action utility target, ``replay`` needs a
    matched replay target, and ``intervention_response`` needs treated and
    baseline perturbations plus an explicit effect target.  The trainer
    rejects enabled terms whose contracts are absent.
    """
    recon_eeg: float = 1.0
    recon_fmri: float = 1.0
    recon_meg: float = 0.0          # wired; enable once the trainer feeds MEG
                                    # batches (targets currently lack "meg")
    recon_extra: Dict[str, float] = field(default_factory=dict)
                                    # modality -> weight for arbitrary neural
                                    # signals (e.g. {"calcium": 1.0}); fires on
                                    # predictions["<modality>_recon"] vs
                                    # targets["<modality>"] (TotalLoss registry)
    recon_loss_types: Dict[str, str] = field(default_factory=dict)
                                    # modality -> single ReconstructionLoss
                                    # criterion ("mse" default; "correlation" /
                                    # "corr_diff" / "huber" / "poisson" /
                                    # "wasserstein1")
    recon_loss_mix: Dict[str, Dict[str, float]] = field(default_factory=dict)
                                    # modality -> criterion -> non-negative
                                    # mixture weight. The normalized mixture
                                    # can preserve temporal correlation while
                                    # directly constraining marginal scale.
    forecast: float = 0.0              # multi-horizon signal forecast
    forecast_huber: float = 1.0
    forecast_corr_diff: float = 1.0
    forecast_horizon_weights: Optional[Tuple[float, ...]] = None
    velocity_smooth: float = 0.0    # monitor: batch-axis TV; needs time axis
    generic_constraint: float = 0.0  # enforced by projection, not by penalty:
                                     # (P_S L P_S) grad_S == 0 and
                                     # (P_E M P_E) grad_E == 0 hold exactly, so
                                     # the residual is identically zero and this
                                     # weight cannot steer.  Kept for callers
                                     # that disable apply_degeneracy_projection,
                                     # where the residual becomes real.
    moe_load_balance: float = 0.01
    hebbian_reg: float = 0.0        # monitor: Oja norm; gradient-free (.data)
    grassmannian_reg: float = 0.001 # single authority on raw orthogonality
    jacobi_reg: float = 0.0         # diagnostic; 0 default avoids crash path
    nsp: float = 0.0                # monitor: band-power clone of bandpower
    cross_modal: float = 0.0  # requires explicit sync/async pair labels
    dissip: float = 0.1
    spectrum: float = 0.01          # sole 1/f PSD term (recon EEG/MEG)
    tsallis: float = 0.0            # monitor: latent sparsity (sigreg chosen;
                                    # code sign rewards concentration — do not
                                    # enable without a sign fix)
    action: float = 0.0
    replay: float = 0.0
    intervention_response: float = 0.0
    cross: float = 0.05             # Latent HRF alignment (Stage 2, sync data)
    cross_soft: float = 0.0        # soft contrastive alignment; explicit labels
    bandpower: float = 0.01         # per-channel band-power matching
    sigreg: float = 0.02            # Weak-SIGReg covariance regularization
    sigreg_sketch_dim: int = 64     # SIGReg sketch dimension
    def __getitem__(self, key: str) -> Any:
        """Expose phase weights through the runtime mapping boundary."""
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        """Return one weight without coupling callers to dataclass fields."""
        return getattr(self, key, default)


@dataclass
class FreezeConfig:
    """Freeze strategy configuration."""
    eeg_encoder: bool = True
    fmri_encoder: bool = True
    eeg_epochs_thawed: int = 1
    fmri_epochs_thawed: int = 1
    velocity_brain: bool = False
    moe_router: bool = False
    decoder: bool = False


@dataclass
class TrainingPhase:
    """Declarative phase definition with one explicit runtime boundary."""
    name: str
    stage: TrainingStage
    total_steps: int
    learning_rate: float
    min_lr: float = 1e-6
    warmup_steps: int = 500
    batch_size: int = 16
    gradient_accumulation: int = 1
    grad_clip: float = 1.0
    max_seq_len_eeg: int = 2560
    max_seq_len_fmri: int = 100
    loss_weights: LossWeights = field(default_factory=LossWeights)
    freeze_config: FreezeConfig = field(default_factory=FreezeConfig)
    lr_schedule: str = "cosine"
    optimizer: str = "adamw"
    router_tau: Optional[float] = None
    imagination_interval: int = 0
    imagination_num_samples: int = 4
    imagination_rollout_steps: int = 3
    rollout_steps: int = 1
    base_context: Optional[int] = None
    context_expansion_schedule: Optional[List[Tuple[int, int]]] = None
    description: str = ""
    task: TrainingTask = TrainingTask.MODEL_TRAINING
    magi_objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "masked": 1.0,
            "causal_ntp": 1.0,
            "contrastive": 0.1,
        })
    transition_gate: Optional[Callable[[Mapping[str, Any]], bool]] = None

    def to_runtime_config(self) -> Dict[str, Any]:
        """Convert this phase into the plain mapping consumed by the trainer."""
        if not isinstance(self.stage, TrainingStage):
            raise TypeError("TrainingPhase.stage must be a TrainingStage")
        task = self.task.value if isinstance(self.task, TrainingTask) else str(self.task)
        if task not in {TrainingTask.MODEL_TRAINING.value,
                         TrainingTask.MAGI_EEG_PRETRAINING.value}:
            raise ValueError(f"unsupported training task: {task!r}")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        schedule = self.context_expansion_schedule
        if schedule is not None:
            schedule = [tuple(map(int, item)) for item in schedule]
            if any(step < 0 or length < 1 for step, length in schedule):
                raise ValueError("context expansion entries must be non-negative")
            steps = [step for step, _ in schedule]
            if steps != sorted(steps):
                raise ValueError("context expansion schedule must be ordered")
        return {
            "name": self.name,
            "stage": self.stage.value,
            "task": task,
            "total_steps": int(self.total_steps),
            "learning_rate": float(self.learning_rate),
            "min_lr": float(self.min_lr),
            "warmup_steps": int(self.warmup_steps),
            "batch_size": int(self.batch_size),
            "gradient_accumulation": int(self.gradient_accumulation),
            "grad_clip": float(self.grad_clip),
            "max_seq_len_eeg": int(self.max_seq_len_eeg),
            "max_seq_len_fmri": int(self.max_seq_len_fmri),
            "loss_weights": self.loss_weights,
            "freeze_policy": asdict(self.freeze_config),
            "lr_schedule": self.lr_schedule,
            "optimizer": self.optimizer,
            "router_tau": self.router_tau,
            "imagination_interval": int(self.imagination_interval),
            "imagination_num_samples": int(self.imagination_num_samples),
            "imagination_rollout_steps": int(self.imagination_rollout_steps),
            "rollout_steps": int(self.rollout_steps),
            "rollout": {
                "steps": int(self.rollout_steps),
                "imagination_interval": int(self.imagination_interval),
                "imagination_num_samples": int(self.imagination_num_samples),
                "imagination_rollout_steps": int(self.imagination_rollout_steps),
            },
            "base_context": self.base_context,
            "context_expansion_schedule": schedule,
            "context": {
                "base_length": self.base_context,
                "expansion_schedule": schedule,
            },
            "description": self.description,
            "weight_decay": 0.01,
            "noise_mode": None,
            "val_interval": 5000,
            "curriculum_steps": int(self.warmup_steps),
            "transition_gate": self.transition_gate,
            "magi_objective_weights": dict(self.magi_objective_weights),
        }

    to_runtime_mapping = to_runtime_config


def get_phase_neg_1() -> TrainingPhase:
    """Phase -1: Magi EEG encoder standalone pretraining."""
    return TrainingPhase(
        name="Phase -1: Magi EEG Pretraining",
        stage=TrainingStage.PHASE_NEG_1,
        total_steps=10000,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_steps=2000,
        batch_size=32,
        lr_schedule="cosine",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=0.0,
            recon_meg=0.0,
            cross_modal=0.0,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.0,
            hebbian_reg=0.0,
            grassmannian_reg=0.0,
            nsp=0.0,
            dissip=0.0,
            spectrum=0.0,
            tsallis=0.0,
            action=0.0,
            replay=0.0,
            cross=0.0,
            cross_soft=0.0,
            bandpower=0.0,
            sigreg=0.05,
        ),
        freeze_config=FreezeConfig(
            eeg_encoder=False,
            fmri_encoder=True,
            fmri_epochs_thawed=1,
        ),
        task=TrainingTask.MAGI_EEG_PRETRAINING,
        magi_objective_weights={"masked": 1.0, "causal_ntp": 1.0,
                                "contrastive": 0.1},
        description="Masked patch reconstruction + causal patch NTP + EMA two-view contrastive pretraining",
        router_tau=None,
    )


def get_stage_1_p1() -> TrainingPhase:
    """Stage 1 P1: Reconstruction priming and routing differentiation."""
    return TrainingPhase(
        name="Stage 1 P1: Encoder Alignment & MoE Priming",
        stage=TrainingStage.STAGE_1_P1,
        total_steps=30000,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_steps=2000,
        batch_size=16,
        gradient_accumulation=1,
        lr_schedule="cosine",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            cross_modal=0.0,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.001,
            jacobi_reg=0.0,
            nsp=0.0,
            dissip=0.0,
            spectrum=0.0,
            tsallis=0.0,
            action=0.0,
            replay=0.0,
            cross=0.0,
            cross_soft=0.0,
            bandpower=0.0,
            sigreg=0.05,
        ),
        freeze_config=FreezeConfig(
            eeg_encoder=True,
            fmri_encoder=True,
        ),
        description="Merged Stage 0 + P1: MoE from start, frozen encoders 1 epoch, no physics",
        router_tau=2.0,
    )


def get_stage_1_p2() -> TrainingPhase:
    """Stage 1 P2: Add the dissipative structural proxy."""
    return TrainingPhase(
        name="Stage 1 P2: Dissipation Constraint",
        stage=TrainingStage.STAGE_1_P2,
        total_steps=15000,
        learning_rate=1e-4,
        min_lr=1e-4,
        warmup_steps=500,
        batch_size=16,
        lr_schedule="flat",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            cross_modal=0.0,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.001,
            jacobi_reg=0.0,
            nsp=0.0,
            dissip=0.05,
            spectrum=0.01,
            tsallis=0.0,
            action=0.0,
            replay=0.0,
            bandpower=0.01,
            sigreg=0.05,
        ),
        freeze_config=FreezeConfig(
            eeg_encoder=True,
            fmri_encoder=True,
        ),
        description="Add L_dissip + nullspace, encoders remain frozen",
        router_tau=2.0,
    )


def get_stage_1_p3() -> TrainingPhase:
    """Stage 1 P3: Supervise two explicit future windows."""
    return TrainingPhase(
        name="Stage 1 P3: Multi-Horizon Forecasting",
        stage=TrainingStage.STAGE_1_P3,
        total_steps=20000,
        learning_rate=1e-4,
        min_lr=1e-4,
        warmup_steps=500,
        batch_size=16,
        gradient_accumulation=1,
        lr_schedule="flat",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            forecast=1.0,
            forecast_huber=1.0,
            forecast_corr_diff=1.0,
            forecast_horizon_weights=(1.0, 0.5),
            cross_modal=0.0,
            velocity_smooth=0.02,
            generic_constraint=0.0,
            moe_load_balance=0.02,
            hebbian_reg=0.0,
            grassmannian_reg=0.002,
            jacobi_reg=0.0,
            nsp=0.0,
            dissip=0.08,
            spectrum=0.01,
            tsallis=0.0,
            action=0.0,
            replay=0.0,
            cross=0.0,
            cross_soft=0.0,
            bandpower=0.01,
            sigreg=0.05,
        ),
        rollout_steps=2,
        freeze_config=FreezeConfig(eeg_encoder=False, fmri_encoder=False),
        description="Two-horizon forecast with dissipative proxy and thawed encoders",
        router_tau=2.0,
    )


def get_stage_1_p4() -> TrainingPhase:
    """Stage 1 P4: Tighten routing after forecast warm-up."""
    return TrainingPhase(
        name="Stage 1 P4: MoE Expansion",
        stage=TrainingStage.STAGE_1_P4,
        total_steps=50000,
        learning_rate=5e-4,
        min_lr=5e-4,
        warmup_steps=1000,
        batch_size=16,
        lr_schedule="flat",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            forecast=1.0,
            forecast_huber=1.0,
            forecast_corr_diff=1.0,
            forecast_horizon_weights=(1.0, 0.5),
            cross_modal=0.0,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.05,
            hebbian_reg=0.0,
            grassmannian_reg=0.002,
            jacobi_reg=0.0,
            nsp=0.0,
            dissip=0.1,
            spectrum=0.01,
            tsallis=0.0,
            action=0.0,
            replay=0.0,
            bandpower=0.02,
            sigreg=0.02,
        ),
        rollout_steps=2,
        freeze_config=FreezeConfig(eeg_encoder=False, fmri_encoder=False),
        description="Top-2 routing, E=8, LR 5e-4 flat",
        router_tau=0.7,
    )


def get_stage_1_p5() -> TrainingPhase:
    """Stage 1 P5: Physics-inspired constraints and forecast retention."""
    return TrainingPhase(
        name="Stage 1 P5: Physics-Inspired Constraints",
        stage=TrainingStage.STAGE_1_P5,
        total_steps=100000,
        learning_rate=5e-4,
        min_lr=1e-5,
        warmup_steps=2000,
        batch_size=16,
        lr_schedule="cosine",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            forecast=1.0,
            forecast_huber=1.0,
            forecast_corr_diff=1.0,
            forecast_horizon_weights=(1.0, 0.5),
            cross_modal=0.0,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.005,
            jacobi_reg=0.0,
            nsp=0.0,
            dissip=0.1,
            spectrum=0.01,
            tsallis=0.0,
            action=0.0,
            replay=0.0,
            bandpower=0.02,
            sigreg=0.02,
        ),
        rollout_steps=2,
        freeze_config=FreezeConfig(eeg_encoder=False, fmri_encoder=False),
        description="Physics-inspired constraints (degeneracy, dissipation proxy, spectrum); EPR/Jacobi logged as monitors",
    )


def get_stage_1_p6() -> TrainingPhase:
    """Stage 1 P6: Long-context forecast continuation."""
    return TrainingPhase(
        name="Stage 1 P6: Long-Context Expansion",
        stage=TrainingStage.STAGE_1_P6,
        total_steps=20000,
        learning_rate=1e-4,
        min_lr=1e-5,
        warmup_steps=500,
        batch_size=16,
        lr_schedule="cosine",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            forecast=1.0,
            forecast_huber=1.0,
            forecast_corr_diff=1.0,
            forecast_horizon_weights=(1.0, 0.5),
            cross_modal=0.0,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.005,
            jacobi_reg=0.0,
            nsp=0.0,
            dissip=0.1,
            spectrum=0.01,
            tsallis=0.0,
            action=0.0,
            replay=0.0,
            bandpower=0.02,
            sigreg=0.02,
        ),
        rollout_steps=2,
        max_seq_len_eeg=4096,
        freeze_config=FreezeConfig(eeg_encoder=False, fmri_encoder=False),
        description="Sequence length 4096, LR restart to 1e-4",
        router_tau=0.7,
    )


def get_all_phases() -> List[TrainingPhase]:
    """Return ordered list of all training phases."""
    return [
        get_phase_neg_1(),
        get_stage_1_p1(),
        get_stage_1_p2(),
        get_stage_1_p3(),
        get_stage_1_p4(),
        get_stage_1_p5(),
        get_stage_1_p6(),
        STAGE_TWO_PHASE,
        STAGE_THREE_PHASE,
    ]


def get_phase_by_name(name: str) -> Optional[TrainingPhase]:
    """Get phase by name string."""
    phases = get_all_phases()
    for phase in phases:
        if phase.name == name:
            return phase
    return None


STAGE_TRANSITIONS = {
    TrainingStage.PHASE_NEG_1: TrainingStage.STAGE_1_P1,
    TrainingStage.STAGE_1_P1: TrainingStage.STAGE_1_P2,
    TrainingStage.STAGE_1_P2: TrainingStage.STAGE_1_P3,
    TrainingStage.STAGE_1_P3: TrainingStage.STAGE_1_P4,
    TrainingStage.STAGE_1_P4: TrainingStage.STAGE_1_P5,
    TrainingStage.STAGE_1_P5: TrainingStage.STAGE_1_P6,
    TrainingStage.STAGE_1_P6: TrainingStage.STAGE_2,
    TrainingStage.STAGE_2: TrainingStage.STAGE_3,
}


@dataclass
class StageTransition:
    """Defines a transition between stages."""
    from_stage: TrainingStage
    to_stage: TrainingStage
    trigger: str
    condition: Optional[callable] = None


STAGE_TRANSITION_RULES = [
    StageTransition(
        from_stage=TrainingStage.PHASE_NEG_1,
        to_stage=TrainingStage.STAGE_1_P1,
        trigger="magi_pretrain_complete",
    ),
    StageTransition(
        from_stage=TrainingStage.STAGE_1_P1,
        to_stage=TrainingStage.STAGE_1_P2,
        trigger="alignment_stable",
        condition=lambda metrics: metrics.get("recon_loss", float("inf")) < 0.3,
    ),
]


STAGE_TWO_PHASE = TrainingPhase(
    name="Stage 2: Cross-Modal Bridging",
    stage=TrainingStage.STAGE_2,
    total_steps=20000,
    learning_rate=1e-4,
    min_lr=1e-5,
    warmup_steps=2000,
    batch_size=16,
    gradient_accumulation=1,
    grad_clip=0.5,
    lr_schedule="cosine",
    optimizer="adafactor",
    loss_weights=LossWeights(
        recon_eeg=1.0, recon_fmri=1.0, cross_modal=0.3,
        velocity_smooth=0.0, generic_constraint=0.0, moe_load_balance=0.01,
        hebbian_reg=0.0, grassmannian_reg=0.005, jacobi_reg=0.0,
        nsp=0.0, dissip=0.05, spectrum=0.01,
        tsallis=0.0, action=0.0, replay=0.0,
        cross=0.05, cross_soft=0.0, bandpower=0.01, sigreg=0.02,
    ),
    router_tau=0.7,
    context_expansion_schedule=[(0, 4096), (5000, 8192), (15000, 16384)],
)

STAGE_THREE_PHASE = TrainingPhase(
    name="Stage 3: Online Adaptation",
    stage=TrainingStage.STAGE_3,
    total_steps=50000,
    learning_rate=1e-4,
    min_lr=5e-6,
    warmup_steps=2000,
    batch_size=16,
    gradient_accumulation=1,
    grad_clip=0.5,
    lr_schedule="cosine",
    optimizer="adafactor",
    loss_weights=LossWeights(
        recon_eeg=1.0, recon_fmri=1.0, cross_modal=0.2,
        velocity_smooth=0.0, generic_constraint=0.0, moe_load_balance=0.01,
        hebbian_reg=0.0, grassmannian_reg=0.01, jacobi_reg=0.0,
        nsp=0.0, dissip=0.05, spectrum=0.01,
        tsallis=0.0,
        # These policies require executed-action utility and replay targets.
        # They remain opt-in until a loader supplies those contracts.
        action=0.0, replay=0.0, intervention_response=0.0,
        cross=0.03, cross_soft=0.0, bandpower=0.01, sigreg=0.02,
    ),
    router_tau=0.7,
    imagination_interval=100,
    imagination_num_samples=2,
    imagination_rollout_steps=3,
    context_expansion_schedule=[(0, 16384), (20000, 32768), (40000, 65536)],
)


if __name__ == "__main__":
    print("Training phases defined:")
    for phase in get_all_phases():
        print(f"  {phase.name}: {phase.total_steps} steps, lr={phase.learning_rate}")