# ECoG/iEEG → Magi Integration Plan
# Generated: 2026-05-12
# Rationale: ECoG/iEEG provides 2.6B tokens vs 68M for scalp EEG (38× more)

---

## 1. Why Merge ECoG into Magi?

| Metric | Scalp EEG (≥19ch) | ECoG/iEEG | Combined |
|--------|-------------------|-----------|----------|
| Unique subjects | ~29,000 | ~735 | ~29,735 |
| Total hours | ~16,000h | ~8,500h | ~24,500h |
| Total tokens | ~68M | ~2.6B | **~2.67B** |
| Chinchilla ratio (110M params) | 0.6× (deficit) | 23.6× (surplus) | **24.2× (surplus)** |
| Spatial resolution | 2-5 cm | 0.3-1 cm | Mixed |
| Signal quality | Noisy (volume conduction) | Clean (direct cortical) | Mixed |

**Phase -1 with ECoG**: Instead of 10B tokens from 68M unique (147 epochs),
we can train on 2.67B unique tokens with only ~4 epochs — well within overfit safety.

---

## 2. Technical Feasibility

### Magi's BIOT Embedding Handles Arbitrary Channels

Magi's `BIOTStyleEmbedding` maps each electrode's 3D coordinate (x,y,z) to a learnable
position embedding. **This works for ANY electrode configuration**, including ECoG:

```
Scalp EEG: Fp1 → (−0.31, 0.36, 0.54) → BIOT embed
ECoG grid: electrode_1 → MNI (−42, 12, 38) → BIOT embed   ← SAME PATHWAY
sEEG depth: contact_5 → MNI (−25, −10, 5) → BIOT embed    ← SAME PATHWAY
```

The BIOT embedding treats each channel independently:
1. Temporal projection: Conv1d(1→768, kernel=256, stride=256) per channel
2. Spatial embedding: 3D coordinate → learnable position embedding
3. Token sequence: (B, C, L, 768) → (B, C×L, 768) into Transformer

**No architecture change needed** — only the 3D coordinate lookup table needs extension.

### Key Differences: Scalp EEG vs ECoG

| Property | Scalp EEG | ECoG/iEEG | Implication |
|----------|-----------|-----------|-------------|
| Channel count | 19-64 | 64-4096 (PtNRGrids!) | Variable; pad/mask to max |
| Signal bandwidth | 0.1-100 Hz | 0.1-500 Hz (HF up to 200Hz) | Wider band = richer spectral features |
| Amplitude | 10-100 μV | 100-5000 μV | Need amplitude normalization |
| Spatial spread | Volume conduction | Local field (minimal spread) | ECoG has less spatial correlation |
| Reference | Linked mastoid/average | Common reference/bipolar | Different montage conventions |
| Coverage | Whole scalp | Only surgical coverage | ECoG covers subset of cortex |
| Artifacts | Eye/muscle/line noise | Line noise, movement | Different artifact profiles |

---

## 3. Integration Architecture

### 3.1 Unified Electrode Coordinate Registry

```python
ELECTRODE_REGISTRY = {
    # Standard 10-20 scalp EEG (19 channels)
    "Fp1": {"type": "scalp", "coords": (-0.31, 0.36, 0.54), "source": "10-20"},
    "Fp2": {"type": "scalp", "coords": (0.31, 0.36, 0.54), "source": "10-20"},
    # ... (existing 10-20 + 10-10 channels)

    # ECoG grid electrodes (MNI coordinates, from patient imaging)
    "ecog_LF_001": {"type": "ecog", "coords": (-42, 12, 38), "source": "sub-01"},
    "ecog_LF_002": {"type": "ecog", "coords": (-40, 15, 40), "source": "sub-01"},
    # ... (per-subject; registered from post-implant CT → MNI)

    # sEEG depth contacts (MNI coordinates)
    "seeg_LH_001": {"type": "seeg", "coords": (-25, -10, 5), "source": "sub-02"},
    # ... (per-subject; registered from post-implant CT → MNI)
}
```

### 3.2 Data Pipeline Modifications

```
ECoG Raw (BIDS/IEEG/NWB)
  → Load with MNE-Python (supports ECoG/sEEG natively)
  → Preprocess: bandpass 0.1-200Hz, notch 50/60Hz, common average re-reference
  → Amplitude normalization: z-score per channel (critical for mixing scalp + ECoG)
  → Map channel names → MNI 3D coordinates (from post-implant imaging)
  → BIOT temporal projection: Conv1d per channel → (B, C, L, 768)
  → BIOT spatial embedding: 3D coords → position embed (same as scalp)
  → Token sequence: (B, C×L, 768) → Transformer encoder (shared weights!)
```

### 3.3 Channel-Type Embedding (NEW)

To help the model distinguish scalp vs ECoG vs sEEG signals:

```python
class ChannelTypeEmbedding(nn.Module):
    """Learnable embedding for signal source type."""
    def __init__(self, hidden_dim=768, num_types=4):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, hidden_dim)
        # 0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown

    def forward(self, x, channel_types):
        # x: (B, C, L, D)
        # channel_types: (C,) long tensor
        type_emb = self.type_embed(channel_types)  # (C, D)
        return x + type_emb.unsqueeze(0).unsqueeze(2)  # broadcast
```

This is added AFTER BIOT spatial embedding, BEFORE the Transformer encoder.
The model learns that ECoG has higher spatial specificity, wider bandwidth, etc.

### 3.4 Variable Channel Count Handling

ECoG subjects have wildly different channel counts (64 to 4096).

Strategy: **Channel binning + masking**
- Cap at max_channels = 256 (covers most ECoG grids)
- If C < 256: pad with zero-vectors + attention mask = 0
- If C > 256 (PtNRGrids): spatially downsample by averaging adjacent channels
- Attention mask ensures Transformer ignores padding channels

### 3.5 Amplitude Normalization Strategy

Critical: Scalp EEG (~50μV) and ECoG (~1000μV) have 20× amplitude difference.

```
Per-channel z-score normalization:
  x_norm = (x - mean(x)) / (std(x) + ε)

Applied independently to each channel before temporal projection.
This eliminates absolute amplitude differences while preserving
relative patterns within each channel.
```

### 3.6 Frequency Band Extension

ECoG captures high-frequency broadband (HFB, 70-200Hz) not visible in scalp EEG.

Options:
1. **Conservative**: Filter ECoG to 0.1-100Hz (same as scalp) — discards HFB advantage
2. **Recommended**: Use 0.1-200Hz for ECoG; the Transformer learns to use or ignore HF
3. **Aggressive**: Add explicit HFB feature channel (envelope of 70-200Hz band) as extra input

**Recommended approach**: Option 2. The BIOT temporal projection (Conv1d kernel=256)
at 512Hz sampling covers up to 256Hz Nyquist, which includes HFB. At 256Hz scalp sampling,
it covers up to 128Hz. The model naturally adapts via the channel-type embedding.

---

## 4. Training Strategy

### 4.1 Mixed Modality Pretraining (Phase -1 revised)

```
Phase -1: Magi Unified Neural Signal Pretraining

Data mix:
  - Scalp EEG (≥19ch): 68M tokens (weight: 1.0)
  - ECoG/iEEG: 2.6B tokens (weight: 0.3)  ← 38× more data, lower weight
  - Low-res EEG (<19ch, interpolated): 320M tokens (weight: 0.5)

Total unique: ~3B tokens
Target budget: 10B tokens → ~3.3 epochs average

Rationale for ECoG weight=0.3:
  - ECoG patients are clinical (epilepsy), not representative of healthy population
  - ECoG spatial coverage is partial (only surgical implant region)
  - ECoG signal statistics differ from scalp; excessive ECoG exposure could bias
    the model toward cortical-only representations
  - Weight 0.3 means effective ECoG contribution: 2.6B × 0.3 = 780M "effective" tokens
  - Combined effective: 68M + 780M + 160M = ~1B; with augmentation ×10 = 10B ✓
```

### 4.2 Curriculum Schedule

```
Steps 0-2B tokens:   Scalp EEG only (warm up standard representations)
Steps 2B-5B tokens:  Mix scalp (60%) + ECoG (30%) + low-res (10%)
Steps 5B-10B tokens:  Mix scalp (40%) + ECoG (40%) + low-res (20%)
```

The gradual ECoG introduction lets the model first learn scalp-level representations,
then refine with higher-resolution ECoG signals.

### 4.3 Loss Modifications

No new losses needed — existing Magi losses apply to both modalities:

| Loss | Scalp EEG | ECoG/iEEG | Notes |
|------|-----------|-----------|-------|
| L_mask | ✅ | ✅ | Masked patch reconstruction |
| L_MoCo | ✅ | ✅ | Cross-view contrastive |
| L_PSD | ✅ | ✅ | PSD reconstruction (wider band for ECoG) |
| L_subject_inv | ✅ | ✅ | Subject-invariant (especially important for ECoG — few subjects!) |
| L_cross_trial | ✅ | ⚠️ | ECoG typically 1 session; may not apply |
| **L_modality_align** | NEW | NEW | Align scalp+iEEG from same subject (DANDI:000574!) |

### 4.4 New Loss: L_modality_align

For DANDI:000574 (simultaneous scalp + iEEG recordings):

```python
L_modality_align = 1 - cosine_similarity(
    z_scalp[subject],   # scalp EEG encoder output
    z_ieeg[subject]     # iEEG encoder output (same time window)
)
```

This forces the model to learn a shared representation where scalp and iEEG
map to the same latent space — critical for downstream Brain MoE-PINN which
receives only scalp EEG but benefits from iEEG-quality representations.

---

## 5. ECoG Data Pipeline Implementation

### 5.1 Data Sources & Formats

| Source | Format | Loader | Coordinate Source |
|--------|--------|--------|-------------------|
| DANDI | NWB (.nwb) | dandi + pynwb | electrodes table (x,y,z in MNI) |
| IEEG.org | EDF/BDF + JSON | mne + requests | electrodes.tsv (BIDS) |
| CRCNS | .mat / .txt | scipy.io | electrode_positions.txt |
| OpenNeuro (iEEG) | BIDS (edf + json) | mne | coordsystem.json + electrodes.tsv |

### 5.2 Preprocessing Pipeline

```python
def preprocess_ecog(raw, target_srate=512):
    """MNE-based ECoG preprocessing."""
    # 1. Resample to target_srate (512Hz covers HFB)
    raw = raw.resample(target_srate)

    # 2. Bandpass filter (wider than scalp EEG)
    raw = raw.filter(0.1, 200, method='fir')

    # 3. Notch filter line noise
    raw = raw.notch_filter([50, 100, 150], method='fir')  # or 60/120/180

    # 4. Re-reference: common average (standard for ECoG)
    raw = raw.set_eeg_reference('average')

    # 5. Z-score normalize per channel
    data = raw.get_data()
    data = (data - data.mean(axis=1, keepdims=True)) / \
           (data.std(axis=1, keepdims=True) + 1e-6)
    raw = raw._data = data

    # 6. Extract MNI coordinates from electrodes table
    coords = extract_mni_coords(raw.info)  # BIDS/DANDI/NWB-specific

    return raw, coords
```

### 5.3 Dataset Class

```python
class ECoGDataset(Dataset):
    """Unified ECoG/iEEG dataset for Magi pretraining."""

    def __init__(
        self,
        data_path: str,
        target_channels: int = 256,
        patch_size: int = 256,
        stride: int = 256,
        max_duration: float = 10.0,  # seconds
        srate: int = 512,
    ):
        self.target_channels = target_channels
        self.patch_size = patch_size
        self.stride = stride
        self.max_samples = int(max_duration * srate)
        self.srate = srate

    def __getitem__(self, idx):
        # Load ECoG segment
        raw, coords = load_ecog_segment(idx)

        # Pad/truncate channels to target_channels
        data = raw.get_data()  # (C, T)
        C, T = data.shape
        if C < self.target_channels:
            pad = np.zeros((self.target_channels - C, T))
            data = np.concatenate([data, pad])
            coords = np.concatenate([coords, np.zeros((self.target_channels - C, 3))])
            mask = np.array([True]*C + [False]*(self.target_channels - C))
        else:
            data = data[:self.target_channels]
            coords = coords[:self.target_channels]
            mask = np.ones(self.target_channels, dtype=bool)

        # Random crop in time
        if T > self.max_samples:
            start = np.random.randint(0, T - self.max_samples)
            data = data[:, start:start+self.max_samples]

        return {
            "data": torch.FloatTensor(data),       # (C, T)
            "coords": torch.FloatTensor(coords),    # (C, 3)
            "mask": torch.BoolTensor(mask),          # (C,)
            "channel_type": 1,                       # 1=ecog
        }
```

---

## 6. Expected Impact on Brain MoE-PINN

### 6.1 Revised Chinchilla Assessment

| Metric | Scalp EEG Only | + ECoG/iEEG | + Low-res EEG | All Combined |
|--------|---------------|-------------|---------------|-------------|
| Unique tokens | 68M | 2,670M | 320M | **~3.0B** |
| Chinchilla ratio (110M) | 0.6× | 24.2× | 2.9× | **27.2×** |
| Phase -1 epochs needed | 147 | 3.7 | 31 | **3.3** |
| Subject diversity | ~29K | +735 | +6K | **~36K** |

**Result**: With ECoG integration, Phase -1 is now Chinchilla-optimal with only ~3 epochs!

### 6.2 Downstream Benefits for Brain MoE-PINN

1. **Better EEG encoder**: Magi pretrained on ECoG learns higher-resolution cortical
   representations that transfer to scalp EEG via the shared BIOT embedding space.

2. **Source localization prior**: The L_modality_align loss forces scalp EEG and ECoG
   representations to be aligned — the model implicitly learns an inverse source model.

3. **Cross-modal bridge grounding**: DANDI:000623 (sync iEEG+fMRI) and DANDI:000574
   (scalp+iEEG) provide a three-tier validation chain: scalp EEG → iEEG → fMRI.

4. **HFB features**: ECoG high-frequency broadband (70-200Hz) is known to carry
   information about local cortical processing that scalp EEG cannot capture.
   Pretraining on ECoG may help the model learn to infer HFB from scalp features.

### 6.3 Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| ECoG clinical bias (epilepsy patients) | HIGH | Weight ECoG at 0.3; subject-invariant loss |
| ECoG partial cortical coverage | MEDIUM | Channel-type embedding; attention mask for missing regions |
| Amplitude distribution mismatch | MEDIUM | Z-score normalization per channel |
| Few ECoG subjects (735 vs 29K scalp) | HIGH | Data augmentation ×10; subject-invariant loss |
| Coordinate registration errors | LOW | Use FreeSurfer + CT coregistration; validate MNI coords |
| ECoG artifacts (stimulation, seizure) | MEDIUM | Artifact detection pipeline; exclude ictal segments |

---

## 7. Implementation Steps

1. **Add `ChannelTypeEmbedding`** to `src/model/spatio_temporal.py` in Magi
2. **Add `ECoGDataset`** to `src/data/eeg_dataset.py`
3. **Extend `ELECTRODE_3D_POSITIONS`** with ECoG/sEEG MNI coordinates
4. **Modify `EEGFoundationModel.forward_embeddings()`** to accept channel_type and coords
5. **Add `L_modality_align`** loss for simultaneous scalp+iEEG data
6. **Create ECoG preprocessing pipeline** (MNE-based, handles NWB/EDF/BIDS)
7. **Update Phase -1 training config**: mixed modality with curriculum schedule
8. **Download priority DANDI datasets**: 000055 (AJILE12), 000623 (sync iEEG+fMRI), 000574 (scalp+iEEG)
9. **Test**: Verify BIOT embedding handles ECoG coords; verify z-score normalization
10. **Validate**: Linear probe on downstream tasks with ECoG-pretrained vs scalp-only Magi

---

## 8. Embedding Dimension Scaling: 768 → 1024/2048

### 8.1 Rationale

ECoG/iEEG carries dramatically richer spatiotemporal information than scalp EEG:

| Property | Scalp EEG (19ch) | ECoG (64-4096ch) | Information Ratio |
|----------|-----------------|-------------------|-------------------|
| Spatial channels | 19 | 64-4096 | 3-215× |
| Temporal bandwidth | 0.1-100 Hz | 0.1-200+ Hz | 2× |
| Spatial specificity | ~3 cm (volume conduction) | ~0.3 cm (local field) | 10× |
| Amplitude range | 10-100 μV | 100-5000 μV | 50× |
| Independent components | ~5-8 (rank-deficient) | 64-4096 (near-full-rank) | 10-800× |

A 768d hidden representation is adequate for 19 channels with ~8 independent components.
But for 256+ ECoG channels with ~100+ independent components, **768d is a bottleneck**.

**Scaling law heuristic**: hidden_dim ∝ √(independent_components × bandwidth_ratio)
- Scalp: √(8 × 1) ≈ 3 → 768 is generous
- ECoG (256ch): √(100 × 2) ≈ 14 → 768/14 ≈ 55 per component → **too compressed**
- ECoG (256ch): 2048/100 ≈ 20 per component → **adequate**

### 8.2 Proposed Architecture: hidden_dim=2048

```
Magi v2 (unified scalp + ECoG):

Input: (B, C, T) where C ∈ [19, 4096], T ∈ [2560, 51200]

1. Temporal projection: Conv1d(1 → 2048, kernel=256, stride=128)
   - Per-channel; produces (B, C, L, 2048) tokens

2. BIOT spatial embedding: 3D coord → position embed(2048)
   - Learnable projection from (x,y,z) to 2048d
   - Extended coord_embedding_dim: 768 → 512

3. ChannelTypeEmbedding: type_id → embed(2048)
   - 4 types: scalp, ecog_grid, seeg_depth, unknown

4. Transformer encoder: 12 layers, 2048d, 16 heads (128d/head)
   - Same depth as BERT-base but wider
   - Or keep 768d internally with up/down-projection:

   ALTERNATIVE (parameter-efficient):
   Input projection: 2048 → 768 (Linear)
   Transformer: 12 layers × 768d (REUSED from existing Magi)
   Output projection: 768 → 2048 (Linear)
```

### 8.3 Two Design Options

#### Option A: Full 2048d Transformer (recommended for ECoG)

```
hidden_dim = 2048
num_layers = 12
num_heads = 16 (128d/head)
intermediate_dim = 8192 (4× hidden)
patch_embed: Conv1d(1→2048)
position_embed: Embedding(num_positions, 2048)
coord_proj: Linear(3→512) → GELU → Linear(512→512)
channel_type_embed: Embedding(4, 2048)

Parameters: ~1.5B (10× larger than 768d BERT-base's 110M)
```

#### Option B: 768d core + 2048d I/O projection (parameter-efficient)

```
temporal_proj: Conv1d(1→2048, kernel=256)  [unchanged]
input_proj: Linear(2048→768)              [NEW]
transformer: 12 layers × 768d             [REUSED from existing Magi]
output_proj: Linear(768→2048)             [NEW]
position_embed: Embedding(num_positions, 768)  [unchanged]
coord_proj: 3→192→192                     [unchanged]
channel_type_embed: Embedding(4, 768)     [NEW]

Parameters: ~110M core + 3.1M projection = ~113M
```

### 8.4 Recommendation: 1024d × 24-28 Layers (BERT-large / ModernBERT-large Scale)

Follow the established scaling path: BERT-base (12L×768d) → BERT-large (24L×1024d) → ModernBERT-large (28L×1024d).

| Model | Layers | hidden_dim | Heads | Params | Chinchilla | Epochs on 2.67B |
|-------|--------|-----------|-------|--------|-----------|----------------|
| BERT-base | 12 | 768 | 12 | 110M | 2.2B | 0.8 |
| **Magi v2 (proposed)** | **24** | **1024** | **16** | **~340M** | **6.8B** | **2.5** |
| ModernBERT-large | 28 | 1024 | 16 | 395M | 7.9B | 3.0 |
| BERT-large | 24 | 1024 | 16 | 340M | 6.8B | 2.5 |

**Why 24-28 layers for ECoG:**
- ECoG captures **hierarchical cortical dynamics**: local circuits → columns → regions → networks
- Deeper models (24-28L) have more capacity to model this hierarchy
- ModernBERT uses **alternating attention**: global every 3 layers, local SWA otherwise — maps perfectly to ECoG's local field potentials + long-range coupling
- 24 layers at 1024d = **340M params** — Chinchilla needs 6.8B tokens; we have 32B effective (2.67B × 12× aug) → **4.7× overtraining** — healthy
- 28 layers at 1024d = **395M params** — Chinchilla needs 7.9B tokens; 32B effective → **4.1× overtraining** — also healthy
- The extra depth compensates for ECoG's partial cortical coverage (only surgical region visible)

**Recommendation**: Start with **24 layers** (BERT-large scale, well-tested), then scale to **28 layers** (ModernBERT-large) if Phase -1 validation metrics support it.

### 8.5 ModernBERT Architectural Adaptations for ECoG

ModernBERT innovations we should adopt:

```python
class MagiModernBlock(nn.Module):
    """ModernBERT-style block with RoPE + GeGLU + Pre-norm."""

    def __init__(self, hidden_dim=1024, num_heads=16, layer_idx=0):
        # 1. RoPE (Rotary Position Embedding) — better length generalization
        #    Replaces learned position embeddings; supports arbitrary sequence lengths
        self.rope = RotaryEmbedding(dim=hidden_dim // num_heads)

        # 2. Pre-LayerNorm (not Post-LayerNorm) — stabilizes deep training
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 3. Alternating Attention:
        #    - Global every 3rd layer (full self-attention)
        #    - Local SWA otherwise (window=128 for ECoG temporal locality)
        is_global = (layer_idx % 3 == 0)
        self.attn = FlashAttention(
            hidden_dim, num_heads,
            window_size=None if is_global else (128, 128),
        )

        # 4. GeGLU FFN (replaces GELU MLP) — 2× parameter efficiency
        #    FFN: Linear(hidden→intermediate*2) → chunk → gate × value
        self.ffn = GeGLU(hidden_dim, intermediate_dim=hidden_dim * 4)

    def forward(self, x):
        # Pre-norm: norm BEFORE attention/FFN (not after)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

**Key ModernBERT features for ECoG:**
| Feature | BERT-base (old) | ModernBERT (new) | ECoG Benefit |
|---------|----------------|------------------|-------------|
| Position encoding | Learned (512 max) | RoPE (unbounded) | Supports variable-length ECoG recordings |
| Attention | Full every layer | Alternating global/local | Local = temporal patch locality; Global = long-range coupling |
| FFN activation | GELU | GeGLU | Better gradient flow for deep models |
| LayerNorm | Post-attention | Pre-attention | Stabilizes 24-28 layer training |
| Sequence length | 512 | 8192 | ECoG can use longer temporal windows |

### 8.6 Architecture: Magi v2 (24L × 1024d)

```
Embedding stack:
  temporal_proj: Conv1d(1→1024, kernel=256, stride=128)   # per-channel
  biot_embed:    BIOTStyleEmbedding(1024d, 3D coords)      # spatial
  type_embed:    Embedding(4, 1024)                        # scalp/ecog/seeg/unknown
  embed_norm:    LayerNorm(1024)                           # ModernBERT: extra norm

Transformer (24 layers):
  - Layer 0,3,6,9,12,15,18,21: Global attention (full self-attention)
  - Other layers: Local SWA (window=128 tokens ≈ 16s @ 256Hz stride)
  - All layers: RoPE, Pre-norm, GeGLU FFN(1024→4096)

Heads:
  mask_head:     Linear(1024 → patch_size_time)              # reconstruct patches
  proj_head:     Linear(1024 → 256) + LayerNorm              # contrastive
  subject_cls:   Linear(1024 → 512 → 1000) + GRL            # adversarial

Parameters: ~340M
Chinchilla: 6.8B tokens
Effective:  32B (2.67B × 12× aug)
Overtraining ratio: 4.7× — healthy
```

### 8.7 Training Recipe (ModernBERT 3-phase)

```python
# Phase 1: Standard pretraining (analogous to ModernBERT phase 1)
# Train on 2B tokens at seq_len=1024 with constant LR after warmup
config_phase1 = {
    "seq_len": 1024,
    "tokens": 2_000_000_000,
    "batch_size": 256,
    "lr": 1e-4,
    "warmup": 10_000,
    "schedule": "warmup_stable",  # constant LR after warmup
    "mask_ratio": 0.30,            # ModernBERT: 30% (not 15%)
    "data_mix": {"scalp": 0.6, "ecog": 0.3, "lowres": 0.1},
}

# Phase 2: Long-context adaptation (analogous to ModernBERT phase 2)
# Train on 500M tokens at seq_len=4096 with reduced batch
config_phase2 = {
    "seq_len": 4096,
    "tokens": 500_000_000,
    "batch_size": 64,              # reduced to keep tokens/batch constant
    "lr": 5e-5,
    "warmup": 2_000,
    "schedule": "warmup_stable",
    "data_mix": {"scalp": 0.5, "ecog": 0.4, "lowres": 0.1},
}

# Phase 3: Annealing (analogous to ModernBERT phase 3)
# Cosine decay on 170M tokens with high-quality data
config_phase3 = {
    "seq_len": 4096,
    "tokens": 170_000_000,
    "batch_size": 64,
    "lr": 5e-5 → 1e-6,            # cosine decay
    "warmup": 0,
    "schedule": "cosine",
    "data_mix": {"scalp": 0.5, "ecog": 0.4, "lowres": 0.1},
}

Total: 2.67B tokens ≈ 1 pass over unique data (with augmentation)
```

### 8.8 Initialization Strategy: Tiling from 12L to 24L

ModernBERT-large initializes from ModernBERT-base via **weight tiling**:
- Base (12L) weights are repeated twice to initialize Large (24L)
- This beats random initialization and speeds up convergence
- For Magi: initialize 24L from existing 12L Magi checkpoint

```python
def tile_weights(base_state_dict, target_layers=24):
    """Tile 12L base weights into 24L large."""
    tiled = {}
    for name, param in base_state_dict.items():
        if "layer" in name:
            # Extract layer index
            layer_idx = int(name.split("layer.")[1].split(".")[0])
            # Map: layer 0-11 → layers 0-11 and 12-23
            for offset in [0, 12]:
                new_name = name.replace(f"layer.{layer_idx}", f"layer.{layer_idx + offset}")
                tiled[new_name] = param.clone()
        else:
            tiled[name] = param.clone()
    return tiled
```

### 8.9 Token Recount with 24L × 1024d

| Config | Params | Tokens Available | Chinchilla (20×) | Overtraining | Epochs on Unique |
|--------|--------|-----------------|-----------------|-------------|-----------------|
| 12L × 768d (current) | 110M | 2.67B | 2.2B | 1.2× | 0.8 |
| **24L × 1024d (proposed)** | **340M** | **2.67B** | **6.8B** | **0.39×** | **2.5** (×12 aug = **4.7×**) |
| 28L × 1024d (stretch) | 395M | 2.67B | 7.9B | 0.34× | 3.0 (×12 aug = 4.1×) |
| 12L × 2048d | 1.5B | 2.67B | 30B | 0.09× | 11.2 |

**24L × 1024d hits the sweet spot**: 4.7× overtraining with augmentation — "a bit of overtraining" as requested. The model sees each unique sample ~2.5× without augmentation, ~5× with augmentation — enough to learn clinical patterns without severe overfitting.

### 8.10 Impact on Brain MoE-PINN

With Magi v2 at 1024d × 24L:
- EEG encoder outputs 1024d → upscaling projection (1024→2048) = 2.1M params
- **2.7× richer representation** vs 768d (1024/768 = 1.33× dim, plus 24L depth = 2× capacity)
- ECoG's high-frequency broadband (70-200Hz) encoded in extra dimensions
- Hierarchical cortical dynamics (local→global) captured by 24-layer depth
- The projection layer acts as learned dimensionality expander, analogous to NeuroSTORM's adapter
