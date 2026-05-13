# Brain MoE-PINN: Project Status
# Generated: 2026-05-12

## Completed (2026-05-11 — 2026-05-12)

### Architecture & Code (brain_moe_pinn/)
- [x] `set_context_length()` propagation chain across 5 files (BrainMoEPINN → encoders → Mamba2 → decoder)
- [x] `load_pretrained()` infrastructure for all encoders + CLI `--eeg_checkpoint` / `--fmri_checkpoint`
- [x] TPT wired into NeuroSTORM `forward_roi()` / `forward_voxel()`
- [x] Counterfactual search: depth=3, branch_factor=4 (actual tree expansion)
- [x] All 32 Python files pass `py_compile`
- [x] All 23 KNOWN_MISMATCHES items resolved or documented

### ECoG Integration (P0 critical)
- [x] `ChannelTypeEmbedding` (4 types: scalp/ecog/seeg/unknown) in Magi `spatio_temporal.py`
- [x] `EEGFoundationModel` accepts `channel_types` in `forward_embeddings()`
- [x] `EEGEncoderWrapper` + `BrainMoEPINN` pass `channel_types` through
- [x] `ecog_preprocessing.py` — MNE-based pipeline: resample → filter → notch → re-reference → z-score → bad channel detection → MNI coord extraction

### Documentation
- [x] `DATA_AUDIT.md` — Chinchilla ratio analysis with ~62B revised budget
- [x] `DATASETS_BY_MODALITY.md` — 3 catalogs: fMRI (~83K subj), ECoG (~735 subj), low-res EEG (~6K subj)
- [x] `ECOG_INTEGRATION_PLAN.md` — Full ECoG → Magi integration plan with ModernBERT adaptations

### Magi Integration (external repo)
- [x] `ChannelTypeEmbedding` added to `spatio_temporal.py`
- [x] `EEGFoundationModel` accepts `channel_types` in `forward_embeddings()`
- [x] `EEGEncoderWrapper` passes `channel_types` through
- [x] `BrainMoEPINN` accepts `channel_types` in `forward()`

## Planned (ECOG_INTEGRATION_PLAN.md)

### P0: Critical for ECoG integration
- [ ] **ECoGDataset** — `MNE`-based dataset class for NWB/EDF/BIDS ECoG data
- [x] **ECoG preprocessing pipeline** — `utils/ecog_preprocessing.py`: bandpass, z-score, MNI coords, bad channel detection
- [x] **L_modality_align** — `utils/losses.py`: cosine alignment loss for simultaneous scalp+iEEG
- [ ] **Electrode registry** — extend 10-5 coordinates with ECoG/sEEG MNI positions
- [x] **Amplitude normalization** — included in `ecog_preprocessing.py` (z-score per channel)

### P1: Magi architecture upgrade
- [ ] **Scale to 24L × 1024d** — ModernBERT-large architecture (340M params)
- [ ] **RoPE** — replace learned position embeddings with rotary embeddings
- [ ] **GeGLU FFN** — replace GELU MLP with gated variant
- [ ] **Alternating attention** — global every 3 layers, SWA otherwise
- [ ] **Pre-norm** — move LayerNorm before attention/FFN
- [ ] **Tiling init** — initialize 24L from existing 12L Magi checkpoint
- [ ] **3-phase training** — warmup_stable → long_context → annealing

### P2: Data acquisition
- [ ] Download DANDI:000055 (AJILE12, 845 GB) — multi-day iEEG
- [ ] Download DANDI:000623 (sync iEEG+fMRI, 27.7 GB) — cross-modal bridge
- [ ] Download DANDI:000574 (scalp+iEEG, 107 GB) — source localization ground truth
- [ ] Download DANDI:000465/000554 (PtNRGrids, 129 GB) — ultra-high-density ECoG
- [ ] Download HBCD fMRI+EEG (7K subjects) — infant multimodal

### P3: Brain MoE-PINN updates
- [ ] Update `EEGProjection` (768→2048) → (1024→2048)
- [ ] Update DeepSpeed configs for 340M Magi + 7B MoE
- [ ] Memory audit: V100 16GB with 24L×1024d encoder
- [ ] Phase -1 training script with mixed-modality data mixing

## Chinchilla Status (with ECoG + 24L×1024d)

| Component | Params | Chinchilla (20×) | Available | Ratio |
|-----------|--------|-----------------|-----------|-------|
| Magi Phase -1 | 340M | 6.8B | 32B (aug) | **4.7×** ✅ |
| fMRI adapter | 5M | 100M | 39M | 0.4× ⚠️ |
| Brain MoE-PINN | 7B | 140B | 62B (rev. budget) | 0.44× ⚠️ |

Magi Phase -1 is healthy with ECoG. fMRI and main model still need augmentation + multi-epoch.

## Next Immediate Action

Implement **ECoGDataset** and **ECoG preprocessing pipeline** (P0 items).
These are additive (no breaking changes) and unlock the 2.6B ECoG tokens.
