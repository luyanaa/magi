# Brain MoE-PINN: Known Remaining Mismatches Log
# Generated: 2026-05-11
# Updated: 2026-05-11 (after alignment fix session)
# This file tracks design-level mismatches between the training plan and
# implementation that are deferred for future analysis / refactoring.

## Table of Remaining Issues

| # | Area | Plan Claim | Implementation Actual | Priority | Status |
|---|------|-----------|----------------------|----------|--------|
| 1 | fMRI Encoder | NeuroSTORM/BrainLM **pretrained weight loading** from GitHub/HuggingFace | `load_pretrained()` methods added to NeuroSTORMEncoder, BrainLMEncoder, EEGEncoderWrapper, and BrainMoEPINN. CLI `--eeg_checkpoint` / `--fmri_checkpoint` in train.py. Requires actual weight files to call | HIGH | **PARTIAL** — infrastructure ready; needs checkpoint download |
| 2 | EEG Decoder | Exact 2-layer spec: `Linear(2048,768) → GELU → Linear(768, C×T_patch)` | Multi-layer MLP with extra LayerNorm, Dropout, intermediate 768→768 | LOW | Design choice; revisit if recon quality poor |
| 3 | Modality Router | Hub Token identity **hard-activates** a single decoder | Soft-blends both decoder outputs via gate weights | LOW | Design choice; soft blend provides gradient to both |
| 4 | Counterfactual Search | Depth **3**, branch factor 4, tree structure (4³=64 leaves) | `rollout_steps=3`, `branch_factor=4`, actual tree expansion in `counterfactual_search.py` | — | **DONE** |
| 5 | Mamba-2 SSM | Stage 2: context expansion 4096→16384→65536 via Mamba-2 | `Mamba2EncoderBlock`, `Mamba2Backbone`, `Mamba2ContextExpansion` in `utils/mamba2_ssm.py` using `fla.layers.Mamba2`. Integrated into EEG encoder (post-Magi stack) and fMRI encoder (optional SWM replacement). CLI `--use_mamba2` in `train.py` | HIGH | **DONE** — FLA library at `/home/yanlu/Documents/flash-linear-attention` |
| 6 | Top-k Router | Core Shared (4) **always active**; Salience (2) gated; Specialized (10) Top-2 | Core Shared always active + Salience gated + Specialized Top-2 in `moe.py` | — | **DONE** |
| 7 | Adafactor Optimizer | AdamW → Adafactor at P4 with lr=5e-4, freeze-warmup transition | `transition_optimizer()` added in `training_loop.py` with 200-step freeze-warmup | — | **DONE** |
| 8 | Data Mixing Schedule | Progressive resting/task_simple/task_complex ratios by training progress | `DataMixer` class added in `data_loader.py` with 5-phase schedule | — | **DONE** |
| 9 | Validation Loop | 5% held-out validation set, evaluate every 5000 steps, overfit detection | `validate()` added to `BrainMoETrainer`, wired into `train_phase()` every `val_interval` steps | — | **DONE** |
| 10 | NSP Loss Band Powers | `compute_band_powers` aggregates per-channel | Refactored to use per-channel temporary tensor (`channel_bands`) then mean; code is correct | — | **DONE** |
| 11 | MNN Moment Embedding | Optional MNN consistency loss in dissip term | Not implemented | LOW | Research-only; requires MNN statistical mechanics |
| 12 | Amortized Personalization | Age/gender/clinical → low-dim conditioning vector in Stage 3 | No conditioning mechanism | LOW | Stage 3 feature; requires clinical metadata pipeline |
| 13 | STRD / TPT | Spatiotemporal Redundancy Dropout, Task-specific Prompt Tuning | `NeuroSTORMPromptTuning` now applied in `forward_roi`/`forward_voxel`: prompts prepended to tokens, stripped from output | — | **DONE** |
| 14 | fMRI FC Input Mode | Functional connectivity (2D) input mode for NeuroSTORM | Only "roi" and "voxel" modes exist | LOW | Alternative input format; ROI mode is primary |
| 15 | Virtual Montage Mapping | Arbitrary montage → standard 10-20 grid via MNE interpolation | Not implemented | LOW | Data preprocessing feature; BIOT fallback handles unknown electrodes |
| 16 | Factorized Attention | Spatial (4 heads) + Temporal (8 heads) decomposition in EEG encoder | Parameter exists but no implementation visible | LOW | Optional mode for computational efficiency |
| 17 | OU Noise Temperature Scan | Temperature T scan for critical phase transition | No scanning infrastructure | LOW | Evaluation feature for §6 critical phase analysis |
| 18 | Loss Normalization EMA | Each loss maintains EMA (β=0.99) of mean/std; normalize to unit variance | `LossNormalizer` in `losses.py` (line 647) wired into `TotalLoss._add_loss()` | — | **DONE** |
| 19 | Checkpoint Phase Detection | `load_checkpoint` detects phase from checkpoint filename | Uses `checkpoint["phase"]` key; robust | — | **DONE** (was false alarm) |
| 20 | DeepSpeed Single-Node Config | Missing `load_from_fp32_weights` and `round_robin_gradients` | Added to `ds_config_zero2.json` + `create_deepspeed_config()` in `training_loop.py` | — | **DONE** |
| 21 | SWM Shifted-Window Mamba | Windowing was non-functional (params ignored) | `_window_partition`/`_window_merge`/`_mamba_block` functional | — | **DONE** |
| 22 | Noise Scaling √(2D) | Missing diffusion coefficient D scaling | `D` parameter + `sqrt(2*D)` in `OUStructuredNoise` | — | **DONE** |
| 23 | Grassmannian on Router | `GrassmannianRegularization` never applied to expert outputs | Computed in `MoEVelocityField.forward()` as `grassmannian_loss` | — | **DONE** |
| 24 | Context Expansion Wiring | `set_context_length()` called by training loop but not implemented on model | Full propagation chain: BrainMoEPINN → EEGEncoderWrapper → Mamba2Backbone → Mamba2EncoderBlock; BrainMoEPINN → NeuroSTORMEncoder → NeuroSTORMBackbone → ShiftedWindowMambaStage; BrainMoEPINN → KIMIKDAMoDeCoderRouter | — | **DONE** |
| 25 | Magi Architecture Scale | 12L × 768d (BERT-base, 110M params) | **Magi v2 implemented**: 24L × 1024d (ModernBERT-large, ~340M params) + RoPE + GeGLU + alternating attention + pre-norm + tiling init | HIGH | **IMPLEMENTED** — `/home/yanlu/Documents/magi/src/model/magi_v2.py` and `encoder_v2.py` |
| 26 | ECoG into Magi | Scalp EEG only (68M tokens) | **ECoG unified dataset**: `utils/ecog_dataset.py` with DANDI/BIDS/EDF support, variable channels (cap 256), MNI coords, amplitude normalization, channel type embedding | HIGH | **IMPLEMENTED** — ready for Phase -1 training |
| 27 | ChannelTypeEmbedding | No modality type discrimination | `ChannelTypeEmbedding` class added to Magi `spatio_temporal.py`; wired into `EEGFoundationModel`; `EEGEncoderWrapper` and `BrainMoEPINN` accept `channel_types` arg | MEDIUM | **DONE** |
| 28 | ECoG Preprocessing | No ECoG/iEEG data loading | `utils/ecog_preprocessing.py` — MNE-based pipeline: resample, bandpass, notch, re-reference, z-score, bad channel detection, MNI coord extraction | HIGH | **DONE** |
| 29 | ModalityAlignmentLoss | No cross-modality alignment loss | `ModalityAlignmentLoss` in `utils/losses.py`: cosine alignment for simultaneous scalp+iEEG | MEDIUM | **DONE** |

| 30 | MoE Chinchilla Ratio | Dense Chinchilla 20× used for token budget | MoE scaling laws (3 papers) show D/N_act ≈ 12× for E=8; see DATA_AUDIT.md §8 | HIGH | **RESOLVED** — budget revised to MoE-aware 67B |
| 31 | Stage 0 Dense Init | Stage 0 trains dense model (E=1) first, then converts to MoE | Dense Stage 0 needs 140B tokens (infeasible); MoE scaling laws say start with E=4 MoE directly (needs 16B tokens) | HIGH | **PLAN UPDATE** — skip dense init, start E=4 |
| 32 | Expert Count Optimum | E=8 chosen by design | Paper 3 (Zhao et al., 2025) shows G_opt≈7, S_opt≈0.31 — our 3-tier design (G≈6-8, S≈33%) aligns | — | **VALIDATED** by MoE scaling laws |
| 33 | Sparsity Ratio N_a/N | 10-15% by design | Paper 3: practical optimal 5-9%; theoretical optimal 20-43% for N<10B | LOW | Our 12-21% is within theoretical range; could go sparser |
| 34 | EEG Encoder Wrapper v2 | Magi v1 wrapper only | `encoders/eeg_encoder_v2.py`: EEGEncoderWrapperV2 with Magi v2, channel type embedding, amplitude normalization, variable channel count | HIGH | **IMPLEMENTED** |
| 35 | Brain MoE-PINN v2 | v1 model with 12L×768d EEG | `brain_moe_pinn_v2.py`: BrainMoEPINNV2 with Magi v2 support, ECoG multi-modality, MoE Stage 0, context expansion | HIGH | **IMPLEMENTED** |
| 36 | Training Script v2 | train.py for v1 only | `scripts/train_v2.py`: supports Magi v2, ECoG datasets, MoE Stage 0, revised token budgets | HIGH | **IMPLEMENTED** |

## Action Plan by Priority

### HIGH (blocks pretraining quality)
1. **Pretrained weight loading** (Item 1): `load_pretrained()` infrastructure is in place. Requires actual NeuroSTORM/BrainLM/Magi checkpoint downloads. CLI: `--eeg_checkpoint` / `--fmri_checkpoint`.
2. **Stage 0 MoE from start** (Item 31): MoE scaling laws show dense init needs 140B tokens (infeasible). **Implemented**: train_v2.py uses MoE Stage 0 (E=4) directly, requiring 16B tokens.
3. **Magi v2 integration**: Items 34-36 implemented. Need to test integration and update DeepSpeed configs for 340M Magi + 7B MoE memory budget.

### MEDIUM (improves training stability / curriculum)
_No remaining open MEDIUM items. All have been addressed._

### LOW (nice-to-have / research features)
3. **EEG decoder exact 2-layer** (Item 2): Only if reconstruction quality is poor.
4. **Hard modality activation** (Item 3): Only if soft blend causes gradient conflict.
5. ~~**STRD/TPT** (Item 13):~~ **DONE** — TPT now wired into forward_roi/forward_voxel.
6. **MNN consistency loss** (Item 11): If adding pulse-network physics constraints.
7. **Amortized personalization** (Item 12): Stage 3 only, requires clinical data.
8. **Temperature scan** (Item 17): Evaluation script for critical dynamics.
9. **Virtual montage mapping** (Item 15): Data preprocessing pipeline.
10. **Factorized attention** (Item 16): Memory-constrained scenarios only.
11. **fMRI FC mode** (Item 14): Alternative to ROI mode.

## Completed in This Session (2026-05-11)

- **Items 4, 6, 7, 8, 9, 10, 18, 20**: All previously listed MEDIUM/LOW items are now fully implemented.
- **`DataMixer`** (`utils/data_loader.py`): Progress-dependent task-type sampling with 5-phase schedule.
- **`validate()`** (`utils/training_loop.py`): Periodic evaluation loop with val_loss tracking, best-model checkpoint saving, overfit flag detection, EarlyStopping integration.
- **Adafactor transition** (`utils/training_loop.py`): `transition_optimizer()` with freeze-warmup (freeze existing params, thaw adapters immediately, full thaw after N steps).
- **`_get_batch()`**: Updated to support `DataMixer` via `train_dataloader.mixer` attribute.
- **`train_phase()`**: Wired validation, early stopping, freeze-thaw, and step tracking.
- **DeepSpeed configs**: Single-node `ds_config_zero2.json` now includes `load_from_fp32_weights`, `round_robin_gradients`, activation checkpointing, checkpoint/save settings, matching multinode config.
- **`create_deepspeed_config()`**: In `training_loop.py`, updated with all new fields for runtime consistency.
- **NSPLoss**: `compute_band_powers` refactored to use per-channel temporary accumulator (`channel_bands`), eliminating dead `powers` variable and clarifying the averaging logic.

## Completed in This Session (2026-05-12) — MoE Scaling Law Audit

- **Item 30 (MoE Chinchilla Ratio)**: Resolved. Dense Chinchilla 20× ratio does not apply to MoE. Three papers analyzed:
  1. Krajewski et al. (2024, arXiv:2402.07871): Fine-grained MoE scaling laws, β_MoE=0.147 > β_dense=0.127 (MoE benefits more from data)
  2. Ludziejewski et al. (2025, arXiv:2502.05172): Joint MoE scaling, E=8 → D/N_act≈12× (not 20×), MoE is memory-efficient
  3. Zhao et al. (2025, arXiv:2509.23678): Comprehensive 5-factor MoE scaling, G_opt≈7, S_opt≈0.31, N_a/N_opt=5-9% (practical)
- **Item 32 (Expert count validation)**: Our 3-tier hierarchy (G≈6-8, S≈33%) validated as aligned with MoE scaling law optima.
- **DATA_AUDIT.md §8**: New section with full MoE-specific analysis, revised token budget 67B (MoE-justified).
- **Token budget revised**: Stage 0 should start with E=4 MoE (16B tokens) instead of dense (140B infeasible).
- **Stage 2 cross-modal gap**: Still critical — 18B tokens needed vs ~3M available.

## Earlier Completions (2026-05-12)

- **Item 24 (set_context_length wiring)**: Full propagation chain from `BrainMoEPINN` → encoders → Mamba2Backbone → Mamba2EncoderBlock; also to `NeuroSTORMBackbone` → `ShiftedWindowMambaStage`; and `KIMIKDAMoDeCoderRouter`.
- **Item 13 (TPT in forward pass)**: `NeuroSTORMPromptTuning` now applied in `forward_roi()` and `forward_voxel()`: prompt tokens prepended before backbone, stripped after.
- **Item 1 partial (load_pretrained infrastructure)**: `load_pretrained()` added to `NeuroSTORMEncoder`, `BrainLMEncoder`, `EEGEncoderWrapper`, and `BrainMoEPINN`. CLI `--eeg_checkpoint` and `--fmri_checkpoint` added to `train.py`. Supports `.pt` and `.safetensors` formats.

## Notes

- All **runtime CRITICAL bugs** have been fixed (shape mismatches, undefined variables, wrong paths).
- All **physics losses** have been implemented and wired into `TotalLoss`.
- **7-layer stability defense** and **auto-rollback** are implemented and integrated into training loop.
- **Smoke tests** and **evaluation suite** are in place.
- **`set_context_length()`** propagation chain is fully wired from training loop through all submodules.
- **TPT (Task-specific Prompt Tuning)** is now active in NeuroSTORM forward passes.
- **`load_pretrained()`** infrastructure is ready for all encoders; only the actual checkpoint downloads remain.
- The remaining open items are: Item 1 (needs checkpoint files), Item 31 (needs plan update + code change), Items 2,3,11,12,14-17,33 (LOW design/research choices).
- Design-choice items (2, 3, 11, 12, 14-17) are documented but deferred until results indicate they are needed.
