"""
Training Phases and Stage Definitions for Brain MoE-PINN.

Defines the multi-stage training schedule:
- Phase -1: Magi EEG encoder standalone pretraining
- Stage 0: Dual-encoder alignment with frozen backbones
- Stage 1: Full model training with MoE (P1-P5)
- Stage 2: Cross-modal bridging
- Stage 3: Online adaptation with Hebbian memory

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
    """Loss weight configuration for each stage."""
    recon_eeg: float = 1.0
    recon_fmri: float = 1.0
    velocity_smooth: float = 0.1
    generic_constraint: float = 0.5
    moe_load_balance: float = 0.01
    hebbian_reg: float = 0.0
    grassmannian_reg: float = 0.001
    jacobi_reg: float = 0.001
    nsp: float = 0.1
    cross_modal: float = 0.1
    dissip: float = 0.1
    epr: float = 0.05
    spectrum: float = 0.01
    tsallis: float = 0.05
    ks: float = 0.01
    action: float = 0.1
    replay: float = 0.02
    bandpower: float = 0.0


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
            recon_eeg=0.0,
            recon_fmri=0.0,
            cross_modal=0.0,
            nsp=1.0,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.0,
            hebbian_reg=0.0,
            grassmannian_reg=0.0,
            jacobi_reg=0.0,
            dissip=0.0,
            epr=0.0,
            spectrum=0.0,
            tsallis=0.0,
            ks=0.0,
            action=0.0,
            replay=0.0,
        ),
        freeze_config=FreezeConfig(
            eeg_encoder=True,
            fmri_encoder=True,
            eeg_epochs_thawed=1,
            fmri_epochs_thawed=1,
        ),
        description="MoE E=4 Top-1, encoder alignment + NSP + modal_align, no physics",
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
            nsp=0.05,
            velocity_smooth=0.0,
            generic_constraint=0.0,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.001,
            jacobi_reg=0.0,
            dissip=0.0,
            epr=0.0,
            spectrum=0.0,
            tsallis=0.0,
            ks=0.0,
            action=0.0,
            replay=0.0,
        ),
        freeze_config=FreezeConfig(
            eeg_encoder=True,
            fmri_encoder=True,
        ),
        description="Merged Stage 0 + P1: MoE from start, frozen encoders 1 epoch, no physics",
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
            velocity_smooth=0.05,
            generic_constraint=0.4,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.001,
            jacobi_reg=0.0,
            dissip=0.05,
            epr=0.02,
            spectrum=0.0,
            tsallis=0.02,
            ks=0.0,
            action=0.0,
            replay=0.0,
        ),
        freeze_config=FreezeConfig(
            eeg_encoder=True,
            fmri_encoder=True,
        ),
        description="Add L_dissip + nullspace, encoders partially thawed",
    )


def get_stage_1_p3() -> TrainingPhase:
    """Stage 1 P3 (60-90B): Add L_EPR >= 0 (Second Law hard constraint)."""
    return TrainingPhase(
        name="Stage 1 P3: EPR Constraint",
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
            velocity_smooth=0.05,
            generic_constraint=0.5,
            moe_load_balance=0.02,
            hebbian_reg=0.0,
            grassmannian_reg=0.002,
            jacobi_reg=0.0,
            dissip=0.08,
            epr=0.03,
            spectrum=0.0,
            tsallis=0.03,
            ks=0.0,
            action=0.0,
            replay=0.01,
        ),
        freeze_config=FreezeConfig(),
        description="L_EPR >= 0 constraint, encoders fully thawed",
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
            velocity_smooth=0.05,
            generic_constraint=0.5,
            moe_load_balance=0.05,
            hebbian_reg=0.0,
            grassmannian_reg=0.002,
            jacobi_reg=0.0,
            dissip=0.1,
            epr=0.05,
            spectrum=0.01,
            tsallis=0.05,
            ks=0.01,
            action=0.05,
            replay=0.02,
        ),
        freeze_config=FreezeConfig(),
        description="Top-2 routing, E=8, LR 5e-4 flat",
    )


def get_stage_1_p5() -> TrainingPhase:
    """Stage 1 P5: Full physical constraints, all experts active."""
    return TrainingPhase(
        name="Stage 1 P5: Full Physical Constraints",
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
            velocity_smooth=0.1,
            generic_constraint=1.0,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.005,
            jacobi_reg=0.0,
            dissip=0.1,
            epr=0.05,
            spectrum=0.01,
            tsallis=0.05,
            ks=0.01,
            action=0.1,
            replay=0.02,
        ),
        freeze_config=FreezeConfig(),
        description="Full physical constraints (Jarzynski, Landauer, spectrum, EPR audit)",
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
            velocity_smooth=0.1,
            generic_constraint=1.0,
            moe_load_balance=0.01,
            hebbian_reg=0.0,
            grassmannian_reg=0.005,
            jacobi_reg=0.0,
            dissip=0.1,
            epr=0.05,
            spectrum=0.01,
            tsallis=0.05,
            ks=0.01,
            action=0.1,
            replay=0.02,
        ),
        freeze_config=FreezeConfig(),
        description="Sequence length 4096, LR restart to 1e-4",
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
}

STAGE_ONE_PHASE = {
    "name": "Stage 1 P1-P6: Pre-Training",
    "stage": "stage_1",
    "total_steps": 195000,
    "learning_rate": 1e-4,
    "min_lr": 1e-6,
    "warmup_steps": 500,
    "batch_size": 16,
    "gradient_accumulation": 1,
    "grad_clip": 1.0,
    "optimizer": "adamw",
    "loss_weights": LossWeights(
        recon_eeg=1.0, recon_fmri=1.0, cross_modal=0.2,
        velocity_smooth=0.1, generic_constraint=1.0, moe_load_balance=0.01,
        hebbian_reg=0.0, grassmannian_reg=0.005, jacobi_reg=0.0,
        dissip=0.1, epr=0.05, spectrum=0.01, tsallis=0.05,
        ks=0.01, action=0.1, replay=0.02,
    ),
}

STAGE_TWO_PHASE = {
    "name": "Stage 2: Cross-Modal Bridging",
    "stage": "stage_2",
    "total_steps": 20000,
    "learning_rate": 1e-4,
    "min_lr": 1e-5,
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "grad_clip": 0.5,
    "batch_size": 16,
    "gradient_accumulation": 1,
    "lr_schedule": "cosine",
    "optimizer": "adafactor",
    "loss_weights": LossWeights(
        recon_eeg=1.0, recon_fmri=1.0, cross_modal=0.3,
        velocity_smooth=0.1, generic_constraint=0.5, moe_load_balance=0.01,
        hebbian_reg=0.0, grassmannian_reg=0.005, jacobi_reg=0.0,
        dissip=0.05, epr=0.03, spectrum=0.01, tsallis=0.03,
        ks=0.01, action=0.05, replay=0.01,
    ),
}

STAGE_THREE_PHASE = {
    "name": "Stage 3: Online Adaptation",
    "stage": "stage_3",
    "total_steps": 50000,
    "learning_rate": 1e-4,
    "min_lr": 5e-6,
    "warmup_steps": 2000,
    "weight_decay": 0.05,
    "grad_clip": 0.5,
    "batch_size": 16,
    "gradient_accumulation": 1,
    "lr_schedule": "cosine",
    "optimizer": "adafactor",
    "loss_weights": LossWeights(
        recon_eeg=1.0, recon_fmri=1.0, cross_modal=0.2,
        velocity_smooth=0.05, generic_constraint=0.3, moe_load_balance=0.01,
        hebbian_reg=0.0, grassmannian_reg=0.01, jacobi_reg=0.0,
        dissip=0.05, epr=0.02, spectrum=0.01, tsallis=0.02,
        ks=0.01, action=0.1, replay=0.05,
    ),
}


if __name__ == "__main__":
    print("Training phases defined:")
    for phase in get_all_phases():
        print(f"  {phase.name}: {phase.total_steps} steps, lr={phase.learning_rate}")