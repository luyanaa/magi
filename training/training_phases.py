"""
Training Phases and Stage Definitions for Brain MoE-PINN (Revised 2026-05-16).

Defines the multi-stage training schedule:
- Phase -1: Magi EEG encoder standalone pretraining (8L×512d BERT-medium)
- Stage 1: Full model training with Shared+Routed MoE (P1-P6)
  - P1: Alignment + routing differentiation (tau=2.0)
  - P2: Dissipation constraint + L_spectrum activation
  - P3: L_TV weight decay (0.1→0.02), dissipative-proxy regularization
  - P4: Routing tightening (tau→0.7), EMA startup, lr bump
  - P5: Full physics-inspired constraints + trajectory-diagnostic audit
  - P6: Long context (seq=4096)
- Stage 2: Cross-modal latent HRF bridge (L_cross + L_cross_soft)
- Stage 3: Online adaptation with Hebbian memory + active inference

Each stage has specific learning rates, freeze strategies, and loss weights.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
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


@dataclass
class LossWeights:
    """Steering weights per training stage.

    Philosophy (2026-09, Occam pass): weights > 0 steer the optimizer;
    weights == 0 keep the term available as a logged monitor (or dormant).
    No loss is deleted.  Defaults below are the canonical dense-stage
    steering set:
      - reconstruction: recon_eeg, recon_fmri, recon_meg (fires only when
        the model emits ``meg_recon`` AND targets contain ``meg``) plus any
        ``recon_extra`` modality (calcium/voltage/widefield/...) whose
        prediction key ``{modality}_recon`` and target ``{modality}`` exist;
        ``recon_loss_types`` picks the per-modality criterion (default mse;
        correlation/corr_diff/huber/poisson/wasserstein1 available)
      - structure: dissip (grad_S · delta_z proxy), moe_load_balance.
        generic_constraint is listed with the steering set for historical
        reasons but its default weight is 0: the degeneracy residuals are
        identically zero because the projections enforce them, so it can only
        steer when apply_degeneracy_projection is disabled.
      - representation: grassmannian_reg (single authority: MoE emits the
        RAW orthogonality; this weight is applied once), sigreg
      - spectral: spectrum (sole 1/f constraint), bandpower (band matching)
      - cross-modal: cross_modal, cross (HRF bridge, sync data)
    Monitored / dormant at weight 0 (re-enable by setting a phase weight):
    velocity_smooth (batch-axis TV — semantically wrong until rollouts
    supply delta_z_sequence), nsp (duplicates
    bandpower until implemented as latent->stats prediction), tsallis
    (softmax-kurtosis anti-collapse; sigreg is the chosen one), hebbian_reg (Oja keeps
    W^T W ~ I by construction; spectral radius handled by periodic norm),
    jacobi_reg (diagnostic only: raises unless poisson_fn/z supplied),
    cross_soft (hub contrastive; labels are never supplied by the trainer).
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
                                    # modality -> ReconstructionLoss criterion
                                    # ("mse" default; "correlation" /
                                    # "corr_diff" / "huber" / "poisson" /
                                    # "wasserstein1" for non-Gaussian signals)
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
    cross_modal: float = 0.1
    dissip: float = 0.1
    spectrum: float = 0.01          # sole 1/f PSD term (recon EEG/MEG)
    tsallis: float = 0.0            # monitor: latent sparsity (sigreg chosen;
                                    # code sign rewards concentration — do not
                                    # enable without a sign fix)
    action: float = 0.0
    replay: float = 0.0
    cross: float = 0.05             # Latent HRF alignment (Stage 2, sync data)
    cross_soft: float = 0.0         # monitor: labels never supplied
    bandpower: float = 0.01         # per-channel band-power matching
    sigreg: float = 0.02            # Weak-SIGReg covariance regularization
    sigreg_sketch_dim: int = 64     # SIGReg sketch dimension


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
    """Complete configuration for a training phase."""
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
    context_expansion_schedule: Optional[List[Tuple[int, int]]] = None
    description: str = ""


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
            eeg_encoder=False,
            fmri_encoder=True,
            eeg_epochs_thawed=1,
            fmri_epochs_thawed=1,
        ),
        description="Placeholder for standalone Magi pretraining (masked/MoCo/PSD losses live outside TotalLoss); EEG recon + SIGReg warm-up only",
        router_tau=None,
    )


def get_stage_1_p1() -> TrainingPhase:
    """Stage 1 P1 (0-30B): Encoder alignment + routing differentiation + EEG-fMRI alignment."""
    return TrainingPhase(
        name="Stage 1 P1: Encoder Alignment & MoE Priming",
        stage=TrainingStage.STAGE_1_P1,
        total_steps=30000,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_steps=2000,
        batch_size=16,
        gradient_accumulation=3,
        lr_schedule="cosine",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            cross_modal=0.1,
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
    """Stage 1 P2 (30-60B): Add dissipation constraint L_dissip + nullspace regularization."""
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
            cross_modal=0.1,
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
    """Stage 1 P3 (60-90B): Activate the structural dissipative proxy."""
    return TrainingPhase(
        name="Stage 1 P3: Dissipative Proxy",
        stage=TrainingStage.STAGE_1_P3,
        total_steps=20000,
        learning_rate=1e-4,
        min_lr=1e-4,
        warmup_steps=500,
        batch_size=16,
        lr_schedule="flat",
        optimizer="adamw",
        loss_weights=LossWeights(
            recon_eeg=1.0,
            recon_fmri=1.0,
            cross_modal=0.15,
            velocity_smooth=0.0,
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
            bandpower=0.01,
            sigreg=0.05,
        ),
        description="Structural dissipative proxy active; encoders fully thawed",
        router_tau=2.0,
    )


def get_stage_1_p4() -> TrainingPhase:
    """Stage 1 P4 (90-120B): Expand MoE Top-1->Top-2, E=4->8, LR 1e-4->5e-4."""
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
            cross_modal=0.2,
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
        description="Top-2 routing, E=8, LR 5e-4 flat",
        router_tau=0.7,
    )


def get_stage_1_p5() -> TrainingPhase:
    """Stage 1 P5: Physics-inspired constraints, all experts active."""
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
            cross_modal=0.2,
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
        description="Physics-inspired constraints (degeneracy, dissipation proxy, spectrum); EPR/Jacobi logged as monitors",
        router_tau=0.7,
    )


def get_stage_1_p6() -> TrainingPhase:
    """Stage 1 P6 (200-260B): Expand seq to 4096, LR restart to 1e-4."""
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
            cross_modal=0.2,
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

# Flat phase configs for CLI parsing
# NOTE: Phase -1 conceptually pretrains Magi with its own losses (masked
# reconstruction, MoCo, PSD), which are not wired into TotalLoss. Until that
# trainer exists, this dict runs the canonical model as an EEG warm-up.
NEGATIVE_ONE_PHASE = {
    "name": "Phase -1: Magi EEG Pretraining",
    "stage": "phase_neg_1",
    "total_steps": 10000,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 2000,
    "batch_size": 32,
    "gradient_accumulation": 1,
    "grad_clip": 1.0,
    "loss_weights": LossWeights(
        recon_eeg=1.0, recon_fmri=0.0, recon_meg=0.0, cross_modal=0.0,
        generic_constraint=0.0, moe_load_balance=0.0,
        grassmannian_reg=0.0, dissip=0.0, spectrum=0.0,
        cross=0.0, bandpower=0.0, sigreg=0.05,
    ),
}

STAGE_ONE_PHASE = {
    "name": "Stage 1 P1-P6: Pre-Training",
    "stage": "stage_1",
    "total_steps": 235000,
    "learning_rate": 1e-4,
    "min_lr": 1e-6,
    "warmup_steps": 500,
    "batch_size": 16,
    "gradient_accumulation": 1,
    "grad_clip": 1.0,
    "optimizer": "adamw",
    "loss_weights": LossWeights(
        recon_eeg=1.0, recon_fmri=1.0, cross_modal=0.2,
        velocity_smooth=0.0, generic_constraint=0.0, moe_load_balance=0.01,
        hebbian_reg=0.0, grassmannian_reg=0.005, jacobi_reg=0.0,
        nsp=0.0, dissip=0.1, spectrum=0.01,
        tsallis=0.0, action=0.0, replay=0.0,
        bandpower=0.01, sigreg=0.02,
    ),
}

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
        tsallis=0.0, action=0.05, replay=0.02,
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