# Magi v2 Implementation Summary

## What Has Been Implemented

### 1. Magi v2 Architecture (`/home/yanlu/Documents/magi/src/model/magi_v2.py`)
- **24 layers × 1024 hidden dimension** (ModernBERT-large scale, ~340M params)
- **RoPE (Rotary Position Embedding)** for better sequence modeling
- **GeGLU (Gated Linear Unit)** activation in FFN
- **Alternating attention**: global attention every 3 layers, SWA (Sliding Window Attention) otherwise
- **Pre-norm architecture** (LayerNorm before attention/FFN)
- **Tiling initialization** from 12L model (initialize 24L by repeating 12L weights)
- **Factorized attention** for EEG spatial-temporal structure
- **Channel type embedding** for multi-modality (scalp EEG / ECoG / sEEG)
- **BIOT-style arbitrary electrode position embedding**
- **Support for variable channel counts** (cap 256)
- **Amplitude normalization for ECoG** (~20× scalp EEG)

### 2. EEGFoundationModelV2 (`/home/yanlu/Documents/magi/src/model/encoder_v2.py`)
- Complete EEG foundation model v2 with Magi v2 architecture
- Momentum encoder for contrastive learning
- Masked prediction + contrastive projection heads
- Adversarial subject classifier (GRL-based)
- Support for ECoG/sEEG amplitude normalization

### 3. ECoG Dataset (`/home/yanlu/Documents/a/brain_moe_pinn/utils/ecog_dataset.py`)
- **DANDI NWB file support** (AJILE12, iEEG+fMRI sync, etc.)
- **EDF/BIDS format support** for ECoG/sEEG data
- **Variable channel counts** (1-256) with capping
- **MNI coordinate extraction** for BIOT embedding
- **Channel type embedding** (ECoG grid vs sEEG depth)
- **Amplitude normalization** (ECoG ~20× scalp EEG)
- **Paired ECoG-fMRI dataset** for cross-modal sync

### 4. EEGEncoderWrapperV2 (`/home/yanlu/Documents/a/brain_moe_pinn/encoders/eeg_encoder_v2.py`)
- Wraps Magi v2 EEGFoundationModelV2 for Brain MoE-PINN
- **Channel type embedding** support
- **Amplitude normalization** for ECoG
- **Variable channel count** handling (cap 256)
- **Mamba-2 SSM integration** for long-context processing
- **Context length scheduling** support
- **Pretrained weight loading** from v1 checkpoints

### 5. BrainMoEPINNV2 (`/home/yanlu/Documents/a/brain_moe_pinn/brain_moe_pinn_v2.py`)
- Complete Brain MoE-PINN v2 with Magi v2 support
- **MoE Stage 0 from start** (E=4, not dense init)
- **ECoG/sEEG multi-modality** support
- **Revised token budgets** based on MoE scaling laws
- **Cross-modal sync** with DANDI datasets
- **Context length scheduling**
- **Configuration object** for easy parameter management

### 6. Training Script v2 (`/home/yanlu/Documents/a/brain_moe_pinn/scripts/train_v2.py`)
- Supports **Magi v2 architecture**
- **MoE Stage 0 from start** (skip dense phase)
- **ECoG dataset integration**
- **Revised phase configurations** based on MoE scaling laws
- **Backward compatible** with v1

### 7. Updated Documentation
- **KNOWN_MISMATCHES.md** updated with v2 implementation status
- **DATA_AUDIT.md** includes MoE scaling law analysis (§8)
- **ECOG_INTEGRATION_PLAN.md** with full architecture and training strategy

## Key Architectural Decisions

### 1. Magi v2: 24L × 1024d (vs v1: 12L × 768d)
- **Rationale**: ECoG signals are denser than scalp EEG but less spatial than fMRI
- **Chinchilla scaling**: 340M params needs 6.8B tokens; have 2.67B unique × 12× aug = 32B effective → 4.7× overtraining (healthy)
- **Memory impact**: 340M vs 110M → 3× larger, but still manageable with 64×V100

### 2. MoE Stage 0 from Start (E=4)
- **Problem**: Dense Stage 0 would need 140B tokens (infeasible)
- **Solution**: Start with MoE (E=4) from Stage 0, needs only 16B tokens
- **MoE scaling law**: D/N_act ≈ 8-12× for E=4-8 (not dense 20×)
- **Paper support**: Krajewski et al. (2024), Ludziejewski et al. (2025), Zhao et al. (2025)

### 3. ECoG Amplitude Normalization (×20 scaling)
- **Fact**: ECoG signals are ~20× larger than scalp EEG
- **Implementation**: Per-channel z-score after amplitude scaling
- **Channel types**: scalp_EEG (1.0), ecog_grid (1/20), seeg_depth (1/10), unknown (1.0)

### 4. Variable Channel Counts (cap 256)
- **ECoG reality**: Grids have 64-256 channels, sEEG has 8-16 depth electrodes
- **BIOT embedding**: Handles arbitrary electrode positions via MNI coordinates
- **Memory optimization**: Cap at 256 channels for practical memory limits

## Remaining Tasks

### HIGH Priority
1. **Download pretrained weights**
   - NeuroSTORM/BrainLM fMRI encoders
   - Magi v1/v2 EEG encoder checkpoints
   - Test `load_pretrained()` infrastructure

2. **Download DANDI datasets**
   - DANDI:000055 (AJILE12, 845 GB)
   - DANDI:000623 (sync iEEG+fMRI)
   - DANDI:000574 (scalp+iEEG)
   - DANDI:000465/000554 (PtNRGrids)

3. **Update DeepSpeed configs**
   - Memory budget for 340M Magi + 7B MoE
   - ZeRO-2 configuration for mixed precision
   - Multi-node scaling for 64×V100

4. **Set up Python environment**
   - Install torch + dependencies
   - Test runtime smoke tests
   - Verify FLA Mamba-2 CUDA kernels

### MEDIUM Priority
5. **Test integration**
   - Run `test_v2_integration.py`
   - Verify forward/backward passes
   - Check memory usage

6. **Update training phases**
   - Stage 0: MoE from start (E=4)
   - Phase -1: ECoG integrated pretraining
   - Token budgets revised per MoE scaling laws

7. **Hardware topology confirmation**
   - AOM-SXMV vs IBM AC922
   - TP/EP allocation on 4-GPU NVLink nodes

### LOW Priority
8. **Performance optimization**
   - Kernel fusion for MoE routing
   - Gradient checkpointing for 24L Magi
   - Mixed precision optimization

## Token Budget Revisions (MoE-aware)

### Original (Dense Chinchilla 20×)
- Stage 0: 140B tokens (infeasible)
- Total: 372B tokens

### Revised (MoE-aware, D/N_act ≈ 12× for E=8)
- **Stage 0**: 16B tokens (MoE E=4, N_act=2B, D/N_act=8×)
- **Stage 1**: 18B tokens
- **Stage 2**: 20B tokens  
- **Stage 3**: 13B tokens
- **Total**: ~67B tokens (5.5× less than dense budget)

### Justification
- **Krajewski et al., 2024**: α=0.115, β=0.147, γ=0.58; MoE β>dense β → MoE benefits MORE from data
- **Ludziejewski et al., 2025**: D/N_act varies with E — E=4→~8-12×, E=8→~11-17×
- **Zhao et al., 2025**: G_opt≈6.78, S_opt≈0.31; our 3-tier hierarchy (G≈8, S≈0.33) aligns

## Cross-Modal Data Pipeline

### ECoG Datasets (735 subjects, ~8.5K hours)
1. **AJILE12** (DANDI:000055): 12 subjects, multi-day continuous, 845 GB
2. **iEEG+fMRI sync** (DANDI:000623): Unique for cross-modal bridge
3. **Scalp+iEEG** (DANDI:000574): Source localization ground truth
4. **PtNRGrids** (DANDI:000465/000554): Clinical epilepsy data

### fMRI Datasets (83K subjects, ~22K hours)
1. **UK Biobank**: 40K subjects, resting-state + task fMRI
2. **HCP**: 1.2K subjects, high-quality multimodal
3. **ABCD**: 11.8K subjects, developmental
4. **HBCD**: 7K+ subjects with fMRI+EEG (valuable for pretraining, ⚠️ NOT simultaneous — separate protocols, different subjects)

### Low-res EEG (6.2K subjects, ~59K hours)
1. **TUH EEG**: 45M tokens (75% of high-res EEG)
2. **LEMON**: 227 subjects, resting-state + structural
3. **MPI-Leipzig**: 227 subjects, multimodal

## Next Immediate Actions

1. **Run integration tests**: `python test_v2_integration.py`
2. **Download priority datasets**: Start with DANDI:000623 (sync iEEG+fMRI)
3. **Set up torch environment**: Install dependencies for runtime testing
4. **Update KNOWN_MISMATCHES.md**: Mark v2 items as IMPLEMENTED
5. **Create launch scripts**: For 64×V100 cluster with v2 configuration

## Success Metrics

1. **Model compiles** without syntax errors ✓
2. **Forward passes** execute without runtime errors
3. **Memory usage** fits within 16GB V100 constraints
4. **Training converges** on small validation set
5. **ECoG normalization** works correctly (20× scaling)
6. **MoE routing** activates correct experts
7. **Cross-modal alignment** improves with training

## Files Created/Updated

### New Files
1. `/home/yanlu/Documents/magi/src/model/magi_v2.py` - Magi v2 architecture
2. `/home/yanlu/Documents/magi/src/model/encoder_v2.py` - EEGFoundationModelV2
3. `/home/yanlu/Documents/a/brain_moe_pinn/encoders/eeg_encoder_v2.py` - EEGEncoderWrapperV2
4. `/home/yanlu/Documents/a/brain_moe_pinn/brain_moe_pinn_v2.py` - BrainMoEPINNV2
5. `/home/yanlu/Documents/a/brain_moe_pinn/scripts/train_v2.py` - Training script v2
6. `/home/yanlu/Documents/a/brain_moe_pinn/utils/ecog_dataset.py` - ECoG dataset
7. `/home/yanlu/Documents/a/brain_moe_pinn/test_v2_integration.py` - Integration tests

### Updated Files
1. `/home/yanlu/Documents/a/brain_moe_pinn/KNOWN_MISMATCHES.md` - Status tracking
2. `/home/yanlu/Documents/a/brain_moe_pinn/DATA_AUDIT.md` - MoE scaling laws
3. `/home/yanlu/Documents/magi/src/model/spatio_temporal.py` - ChannelTypeEmbedding
4. `/home/yanlu/Documents/magi/src/model/encoder.py` - EEGFoundationModel updates

## Conclusion

The Magi v2 architecture has been fully implemented with support for ECoG/sEEG multi-modality, MoE-aware token budgeting, and cross-modal synchronization. The implementation follows ModernBERT-large scale (24L×1024d) with RoPE, GeGLU, and alternating attention patterns. 

The key innovation is starting with MoE from Stage 0 (E=4) instead of dense initialization, reducing token requirements from 140B to 16B while maintaining model capacity. ECoG integration provides 2.6B additional tokens for pretraining, resolving the Chinchilla deficit.

Next steps focus on downloading pretrained weights and datasets, updating DeepSpeed configurations, and running integration tests on the target hardware (64×V100 SXM2).