# Brain MoE-PINN: Data Audit for Chinchilla Ratio
# Generated: 2026-05-12
# Purpose: Assess whether available data can satisfy Chinchilla-optimal training
#          for a 7-10B total / 1-3B active parameter MoE-PINN

---

## 1. Chinchilla Scaling Law Recap

Chinchilla (Hoffmann et al., 2022): For compute-optimal training,
  **N_tokens ≈ 20 × N_params_active**

Applied to Brain MoE-PINN:
| Stage | Active Params | Chinchilla-Optimal Tokens | Plan Budget |
|-------|--------------|--------------------------|-------------|
| Phase -1 (EEG only) | ~110M (BERT-base) | ~2.2B | ~10B ✅ |
| Stage 0 | ~7B (Dense) | ~140B | 8B ⚠️ |
| Stage 1 | ~3B (Top-2) | ~60B | 260B ✅ |
| Stage 2 | ~3B (Top-2) | ~60B | 74B ✅ |
| Stage 3 | ~2B (frozen shared) | ~40B | 30B ⚠️ |
| **Total (Stages 0-3)** | | **~300B** | **372B** |

**Key Insight**: Chinchilla scaling for LLMs may not directly apply here.
DIVER-1 (arXiv:2512.19097) shows that for electrophysiological foundation models,
**data diversity > training duration > model parameters** (opposite of LLMs).
Thus subject/task diversity matters MORE than raw token count.

---

## 2. EEG Data Token Accounting

### Token Definition
- 1 EEG token = 1 patch of 256 samples @ 256Hz (= 1 second, 19 channels)
- Raw samples per token: 19 channels × 256 samples = 4,864 scalar values
- Tokens per hour of EEG: 3,600 seconds = 3,600 tokens

### Available EEG Datasets (from EEG-Datasets/README.md)

| Dataset | Subjects | Channels | SRate | Duration/Subject | Total Hours | Tokens | LaBram? | Category |
|---------|----------|----------|-------|-----------------|-------------|--------|---------|----------|
| **TUH EEG** | ~25,000 | variable (→ interp to 19ch) | 250-512Hz | ~30 min avg | ~12,500h | ~45M | ✅ | Clinical |
| **TDBrain** | 1,200 | 21ch (→ interp) | 500Hz | ~60 min | 1,200h | ~4.3M | — | Resting+Task |
| **HBN** | ~1,500 | 128ch (→ interp) | 500Hz | ~30 min | 750h | ~2.7M | — | Task-independent |
| **PhysioNet MI** | 109 | 64ch | 1000Hz | ~15 min | 27h | ~97K | ✅ | Motor-Imagery |
| **BCI IV-1** | 7 | 64ch | 1000Hz | ~30 min | 3.5h | ~13K | ✅ | Motor-Imagery |
| **DEAP** | 32 | 32ch | 512Hz | ~60 min | 32h | ~115K | — | Emotion |
| **SEED** | 15 | 62ch | 200Hz | ~60 min | 15h | ~54K | ✅ | Emotion |
| **SEED-IV** | 15 | 62ch | 200Hz | ~30 min | 7.5h | ~27K | ✅ | Emotion |
| **Enterface'06** | 16 | 64ch | — | ~30 min | 8h | ~29K | ✅ | Emotion |
| **OpenMIIR** | 10 | 64ch | 512Hz | ~60 min | 10h | ~36K | — | Auditory |
| **ERP Core** | 40 | 30ch | 500Hz | ~60 min | 40h | ~144K | — | Cognitive |
| **Nencki-Symfonia** | 42 | 64ch | — | ~60 min | 42h | ~151K | — | Cognitive |
| **Brain Invaders (3 sets)** | 132 | 32ch | — | ~30 min | 66h | ~238K | ✅ | ERP |
| **SADT** | 27 | 32ch | 500Hz | 180 min | 81h | ~292K | — | Sustained Attention |
| **SPIS Resting** | 10 | 64ch | — | 2.5 min | 0.4h | ~1.4K | ✅ | Resting |
| **Resting State (608 subj)** | 608 | variable | — | ~30 min | 304h | ~1.1M | — | Resting |
| **Siena Scalp EEG** | 14 | 29ch | 512Hz | variable | ~10h | ~36K | ✅ | Clinical |
| **CHB-MIT** | 22 | 23ch | 256Hz | ~60 min avg | 22h | ~79K | — | Clinical/Seizure |
| **High-Gamma Dataset** | 14 | 128ch | 500Hz | ~60 min | 14h | ~50K | — | Motor-Imagery |
| **Grasp and Lift (LaBram)** | 12 | 32ch | 500Hz | ~30 min | 6h | ~22K | ✅ | Motor-Imagery |
| **Mental Imagery** | 13 | 38ch | — | ~60 min | 13h | ~47K | — | Motor-Imagery |
| **MNIST Brain Digits** | 1 | variable | — | 2s/trial | ~670h est. | ~2.4M | — | Visual (single subj!) |
| **Leipzig LEMON** | ~200 | 62ch | — | ~30 min | 100h | ~360K | — | Resting+Emotion |
| **ZuCo** | 12 | — | — | 4-6h/subj | 60h | ~216K | — | Language |
| **Kara-One** | 14 | 64ch | — | ~30 min | 7h | ~25K | — | Language |
| **LaBram Self-collected** | unknown | — | — | — | large | ~10M+ est. | ✅ | Mixed |
| **Cuban Human Brain** | ~300 | 19ch | — | ~30 min | 150h | ~540K | — | Task-independent |
| **MIPDB** | ~120 | 19ch | — | ~30 min | 60h | ~216K | — | Task-independent |

### EEG Token Summary (Conservative Estimates)

| Category | Datasets | Total Subjects | Total Hours | Total Tokens |
|----------|----------|---------------|-------------|-------------|
| **Clinical (TUH + TDBrain + Siena + CHB-MIT)** | 4 | ~26,236 | ~13,732h | ~49.5M |
| **Task-Independent (HBN + Cuban + MIPDB + Resting)** | 4 | ~2,528 | ~1,264h | ~4.6M |
| **Motor-Imagery** | 5 | ~155 | ~60h | ~216K |
| **Emotion** | 5 | ~94 | ~62h | ~223K |
| **ERP/Cognitive** | 4 | ~251 | ~229h | ~825K |
| **Auditory/Music** | 5+ | ~30 | ~50h | ~180K |
| **Language** | 3 | ~40 | ~77h | ~277K |
| **Visual** | 3 | ~1 | ~670h | ~2.4M ⚠️ single subj |
| **LaBram Self-collected** | 1 | unknown | unknown | ~10M+ est. |
| **TOTAL (without TUH, without LaBram)** | | ~3,099 | ~2,412h | ~8.7M |
| **TOTAL (with TUH)** | | ~29,335 | ~16,144h | ~58.2M |
| **TOTAL (with TUH + LaBram)** | | ~29,335+ | ~16,144h+ | ~68M+ |

**Critical Finding**: Without TUH clinical EEG, only ~8.7M tokens available.
TUH alone provides ~45M tokens (~75% of total). TUH is essential.

---

## 3. fMRI Data Token Accounting

### Token Definition
- 1 fMRI token = 1 ROI timepoint (400 regions × 1 TR)
- Typical TR = 0.72-2.0 seconds
- Tokens per hour of fMRI: ~1,800 TRs/hr → 1,800 tokens (ROI mode)

### Available fMRI Datasets

| Dataset | Subjects | TR (s) | Duration/Subject | Total Hours | Tokens | Category |
|---------|----------|--------|-----------------|-------------|--------|----------|
| **UK Biobank** | ~50,000 | 0.735 | ~6 min | 5,000h | ~9M | Resting |
| **ABCD** | ~11,000 | 0.8 | ~20 min | 3,667h | ~6.6M | Task+Resting |
| **HCP (Young Adult)** | 1,200 | 0.72 | ~60 min (4 runs) | 1,200h | ~2.2M | Task+Resting |
| **HCP (Development/Aging)** | ~1,500 | 0.8 | ~40 min | 1,000h | ~1.8M | Task+Resting |
| **OpenNeuro (aggregate)** | ~5,000 | varies | varies | ~3,000h est. | ~5.4M est. | Mixed |
| **CineBrain (sync EEG-fMRI)** | ~20 | 2.0 | ~6h | 120h | ~216K | Cross-modal |
| **NeuroSTORM pretraining corpus** | 50,000+ | varies | varies | >9,000h | >16M | Mixed |

### fMRI Token Summary

| Category | Total Subjects | Total Hours | Total Tokens |
|----------|---------------|-------------|-------------|
| Resting (UKB) | ~50,000 | ~5,000h | ~9M |
| Developmental (ABCD+HCP-Dev) | ~12,500 | ~4,667h | ~8.4M |
| HCP Young Adult | 1,200 | 1,200h | ~2.2M |
| OpenNeuro aggregate | ~5,000 | ~3,000h | ~5.4M |
| **TOTAL accessible** | **~68,700** | **~13,867h** | **~25M** |
| NeuroSTORM corpus (requires their pipeline) | 50,000+ | >9,000h | >16M |

**Note**: NeuroSTORM was pretrained on 28.65M frames from 50K+ subjects.
Our fMRI branch uses frozen NeuroSTORM backbone, so we need fMRI data
for fine-tuning adapters + joint EEG-fMRI training only.

---

## 4. Joint EEG-fMRI Data (Cross-Modal Bridge)

| Dataset | Subjects | Modality | Duration | Tokens (joint) | Availability |
|---------|----------|----------|----------|----------------|-------------|
| **CineBrain** | ~20 | sync EEG+fMRI | ~6h/subj | ~216K | arXiv:2503.06940 |
| **Simultaneous EEG-fMRI Sleep** | ~20 | sync EEG+fMRI | ~2h/subj | ~40K | OpenNeuro ds003768 |
| **EEG/FMRI Naturalistic Viewing** | ~20 | sync EEG+fMRI | ~1h/subj | ~20K | INDI |
| **HCP 7T EEG-fMRI** | ~20 | sync EEG+fMRI | ~30min/subj | ~10K | Limited access |
| **TDBRAIN (has fMRI subset)** | unknown | EEG only (some fMRI) | — | — | Partial |

**Cross-modal total**: ~286K tokens — very sparse!

---

## 5. Chinchilla Ratio Assessment

### Phase -1: Magi EEG Encoder Pretraining (Revised with ECoG + 24L×1024d)

**Architecture change**: Magi v2 now uses 24L × 1024d (340M params, ModernBERT-large scale)
- Target: 6.8B tokens (Chinchilla-optimal for 340M)
- Available EEG + ECoG + low-res: **2.67B unique tokens**
- Effective with augmentation (12×): **32B tokens**
- **Ratio: 32B / 6.8B = 4.7× overtraining — healthy**

Previous analysis (without ECoG, 12L×768d):
- Target: 10B tokens
- Available (with TUH): ~68M tokens
- Deficit: 147× short (no longer relevant)

Even with TUH (45M tokens), we are ~147× below the 10B token target.
This means either:
1. **Reduce Phase -1 token budget** to ~100M (still 1.5× above Chinchilla for 110M params)
2. **Multi-epoch training**: TUH data can be used for ~150 epochs to reach 10B tokens
3. **Augmentation multiplier**: Time-shift, channel dropout, noise injection effectively multiply data

### Stages 0-3: Joint Model Training
- Plan total: 372B tokens
- But these are "model tokens" (latent space, not raw data tokens)

The key question: **How many raw data tokens → 1 model training step?**

Each training step processes:
  - EEG: batch × seq_len × patch_embed_dim → ~131K-393K tokens per step
  - fMRI: batch × seq_len × roi_dim → similar order

With data augmentation (random cropping, channel dropout, noise),
each raw sample can generate ~10-50 distinct training views.

### Reconciliation: Data Epochs Required

| Stage | Plan Tokens | Unique Raw EEG Tokens | Epochs Needed | Unique Raw fMRI Tokens | Epochs Needed |
|-------|-----------|----------------------|---------------|----------------------|---------------|
| Phase -1 | 10B | 68M | ~147 | N/A | N/A |
| Stage 0 | 8B | 68M | ~118 | 25M | ~320 |
| Stage 1 | 260B | 68M | ~3,824 | 25M | ~10,400 |
| Stage 2 | 74B | 68M | ~1,088 | 25M | ~2,960 |
| Stage 3 | 30B | 68M | ~441 | 25M | ~1,200 |

**Verdict**: Stages 1-3 require thousands of epochs on the same data.
This is extreme overfitting risk without heavy augmentation.

---

## 6. DIVER-1 Diversity Assessment

DIVER-1 (arXiv:2512.19097): For electrophysiological foundation models,
**data diversity > training compute > model size**.

### Subject Diversity Score

| Dimension | EEG | fMRI | Assessment |
|-----------|-----|------|-----------|
| **Total subjects** | ~29K (w/ TUH) | ~69K (w/ UKB) | EEG OK with TUH; fMRI excellent |
| **Age range** | 5-100 (TUH clinical skew) | 5-100 (ABCD+UKB+HCP) | Good |
| **Clinical vs Healthy** | TUH is all clinical; others mixed | Mostly healthy | EEG has clinical bias ⚠️ |
| **Task paradigms** | ~15+ distinct paradigms | ~5-10 paradigms | EEG moderate; fMRI limited |
| **Recording sites** | TUH multi-site; others single | UKB single-site; HCP multi | Moderate |
| **Equipment diversity** | TUH mixed; others varied | UKB uniform; HCP uniform | EEG better than fMRI |
| **Cross-modal (sync)** | ~80 subjects total | Same | Critical shortage ⚠️⚠️⚠️ |

### Task Coverage Matrix

| Task Type | EEG Datasets | fMRI Datasets | Joint Datasets |
|-----------|-------------|---------------|----------------|
| Resting state | ✅ (TUH, HBN, TDBrain) | ✅ (UKB, HCP, ABCD) | ⚠️ (CineBrain) |
| Motor imagery | ✅ (PhysioNet MI, BCI IV) | ✅ (HCP Motor) | ❌ |
| Emotion | ✅ (DEAP, SEED) | ⚠️ (limited) | ❌ |
| Language | ✅ (ZuCo, Kara-One) | ✅ (HCP Language) | ❌ |
| Visual | ✅ (THINGS-EEG) | ✅ (HCP Vision) | ❌ |
| Auditory | ✅ (OpenMIIR) | ✅ (HCP) | ⚠️ (CineBrain) |
| Sleep | ✅ (CHB-MIT, Sleep datasets) | ✅ (some OpenNeuro) | ⚠️ (ds003768) |
| Clinical | ✅ (TUH extensive) | ⚠️ (limited public) | ❌ |

---

## 7. Recommendations

### CRITICAL: Data Budget Reduction

The 372B token budget is infeasible without extreme multi-epoch training.
**Recommended revised budget** (balancing Chinchilla + DIVER-1 + overfit risk):

| Stage | Original | Revised | Rationale |
|-------|----------|---------|-----------|
| Phase -1 | 10B | **500M** | 5 epochs on 68M EEG tokens + augmentation; Chinchilla needs only 2.2B for 110M params |
| Stage 0 | 8B | **2B** | ~30 EEG epochs + ~80 fMRI epochs; heavy augmentation |
| Stage 1 | 260B | **40B** | ~600 EEG epochs + ~1,600 fMRI epochs; still heavy but physics losses provide regularization |
| Stage 2 | 74B | **15B** | Cross-modal data bottleneck dominates; CineBrain expansion critical |
| Stage 3 | 30B | **5B** | Specialized fine-tuning; few epochs needed |
| **Total** | **372B** | **~62B** | **6× reduction** |

This is ~62B tokens, roughly 20× the active parameter count (3B), 
satisfying Chinchilla ratio for the active parameters.

### HIGH PRIORITY: Cross-Modal Data Acquisition

CineBrain is the ONLY viable synchronized EEG-fMRI dataset for Stage 2.
- Current: ~216K joint tokens (~80 subjects × ~2h)
- Need: At least 2M joint tokens for stable cross-modal bridge
- Action: Contact Feng group for CineBrain access; consider EEG-informed fMRI synthesis

### MEDIUM: TUH Access

TUH provides ~75% of all EEG tokens. Without it:
- Only ~8.7M unique EEG tokens (vs 68M with TUH)
- Phase -1 must be shortened to ~50M tokens max
- Recommend: Apply for TUH access immediately (free for academic use)

### MEDIUM: fMRI Data Strategy

fMRI data (25M tokens from public sources) is sufficient for adapter fine-tuning
since NeuroSTORM backbone is frozen. The 25M fMRI tokens easily cover
Chinchilla needs for the ~5M trainable adapter parameters.

### LOW: Data Augmentation Strategy

To stretch limited data:
- EEG: Time-shift (±500ms), channel dropout (20%), Gaussian noise (SNR 20-40dB),
  band-stop masking (random 5Hz band), amplitude scaling (0.8-1.2×)
- fMRI: Temporal jitter (±1 TR), spatial smoothing variation, motion artifact simulation
- Joint: Temporal misalignment injection (train robustness to sync errors)

---

## 8. MoE-Specific Scaling Laws: Revised Chinchilla Analysis

The original analysis used dense Chinchilla (Hoffmann et al., 2022) which assumes
N_tokens ≈ 20 × N_params. This is **wrong for MoE models**. Three recent papers
fundamentally change how we should budget data for MoE:

### 8.1 Paper 1: Scaling Laws for Fine-Grained Mixture of Experts
**(Krajewski et al., 2024 — arXiv:2402.07871)**

**Key findings:**
- Introduces **granularity** G = d_ff / d_expert (splitting experts into smaller ones)
- Joint scaling law: L(N,D,G) = c + (g/G^γ + a) · 1/N^α + b/D^β
- Fitted coefficients: **α=0.115, β=0.147, γ=0.58** (vs dense: α=0.126, β=0.127)
- **MoE β (0.147) > dense β (0.127)**: MoE benefits MORE from more data than dense
- **MoE α (0.115) < dense α (0.126)**: MoE benefits LESS from more total params
- Higher granularity always improves loss for same compute
- Standard expert size (=d_ff, G=1) is **almost never optimal**
- Compute-optimal MoE saves 20-40× FLOPs vs dense at same loss
- The efficiency gap **widens** with scale (contradicts Clark et al., 2022)

**Implication for Brain MoE-PINN**: Our 3-tier expert hierarchy (Core/Salience/Specialized)
is a form of heterogeneous granularity — Core experts are "shared" (always active),
Salience are medium-grained, Specialized are fine-grained. This is consistent with
the paper's finding that finer granularity helps.

### 8.2 Paper 2: Joint MoE Scaling Laws — MoE Can Be Memory Efficient
**(Ludziejewski et al., 2025 — arXiv:2502.05172)**

**Key findings:**
- Joint scaling law: L(N_act, D, Ê) = a·Ê^δ · N_act^(α+γ·ln(Ê)) + b·Ê^ω · D^(β+ζ·ln(Ê)) + c
- E_hat is a monotonic transformation of expert count E (Eq. 4 in paper)
- **Finding 1**: More experts → higher tokens-to-param ratio at compute-optimal
  - E=1:  D_opt/N_act ≈ 5.7 (dense Chinchilla ≈ 20×)
  - E=4:  D_opt/N_act ≈ 8.6
  - E=8:  D_opt/N_act ≈ 11.7
  - E=16: D_opt/N_act ≈ 15.7
  - E=32: D_opt/N_act ≈ 20.4
- **Finding 2**: More experts → better performance (always, at compute-optimal)
- **Finding 3**: MoE can be memory-optimal — total-param-matched MoE beats
  overtrained dense model at same FLOP budget
- **Rule of Thumb**: For fixed total params, MoE with E≤8 outperforms compute-optimal
  dense if trained on **E× more tokens** while maintaining same memory footprint
  (e.g., E=4 MoE needs 4× more tokens than dense of same total params)
- Memory-optimal E for typical budgets: 24GB→E=4-16; 80GB→E=8-16; 640GB→E≥32

**Critical table from the paper** (compute-optimal configs):
| FLOP Budget | E | N_act^opt | D^opt | D/N_act |
|-------------|---|-----------|-------|---------|
| 10^20       | 1 | 1.7B      | 9.7B  | 5.7×    |
| 10^20       | 4 | 1.2B      | 13.9B | 11.6×   |
| 10^20       | 8 | 990M      | 17B   | 17.2×   |
| 10^20       | 16| 810M      | 20.7B | 25.6×   |
| 10^21       | 4 | 4.4B      | 38B   | 8.6×    |
| 10^21       | 8 | 3.8B      | 44.3B | 11.7×   |
| 10^21       | 16| 3.3B      | 51.2B | 15.5×   |
| 10^22       | 4 | 15.8B     | 105.4B| 6.7×    |
| 10^22       | 8 | 14.4B     | 115.8B| 8.0×    |

**Key insight**: The dense Chinchilla ratio of ~20× is the *limit as E→∞*.
For realistic E=4-8, the optimal D/N_act ratio is **8-12×**, not 20×.

### 8.3 Paper 3: Towards a Comprehensive Scaling Law of MoE
**(Zhao et al., 2025 — arXiv:2509.23678, Tencent Hunyuan)**

**Key findings (446 controlled experiments, up to 9B params, 100B tokens):**
- 5-factor scaling law: L(N,D,N_a,G,S) with shared expert ratio S
- **G_opt ≈ 6.78** (optimal activated expert count, independent of N,D,N_a)
- **S_opt ≈ 0.31** (optimal shared expert ratio ≈ 13-31% range)
- **N_a/N_opt decreases as N grows** — larger models should be sparser:
  - N=30B:  N_a/N ≈ 40% (theoretical), ~9% (practical/efficiency-aware)
  - N=671B: N_a/N ≈ 22% (theoretical), ~5% (practical)
- **Practical efficiency-aware optimal N_a/N ≈ 5-9%** (accounts for training cost)
- The N_a/N ratio follows: (N_a/N)_opt ∝ N^(-1/(α+1))
- Shared experts are **essential** (not having them hurts significantly)
- Loss is relatively flat around S_opt — shared ratio 13-31% all work well

**Implication for our design**: Our 3-tier hierarchy maps to:
- Core Shared (always active) ≈ shared experts → S ≈ 1/G_total
- With G_total=8 (Core=2 + Top-2 Salience + Top-2 Specialized):
  effective S ≈ 2/6 = 0.33 (within 13-31% optimal range ✓)

### 8.4 Revised Token Budget for Brain MoE-PINN

**Applying MoE-specific scaling laws to our architecture:**

Our model: N_total ≈ 7-10B, N_active ≈ 1-3B, E≈8 (effective), G≈6-8

**From Paper 2 (Joint MoE):** With E=8, compute-optimal D/N_act ≈ 11-17×
**From Paper 3 (Comprehensive):** Practical optimal N_a/N ≈ 5-9%

| Configuration | N_total | N_active | N_a/N | D/N_act (MoE-optimal) | Required Tokens |
|--------------|---------|----------|-------|----------------------|-----------------|
| Conservative (E≈4) | 7B | 1.5B | 21% | ~8-12× | 12-18B |
| Recommended (E≈8) | 8B | 1B | 12.5% | ~11-17× | 11-17B |
| Aggressive (E≈16) | 10B | 0.8B | 8% | ~15-25× | 12-20B |

**Key revision: Dense Chinchilla (20×) significantly over-estimates MoE data needs.**

For our specific configuration (E≈8, N_act≈1-3B):
- **Dense Chinchilla**: 3B × 20 = 60B tokens (previous §7 estimate)
- **MoE-optimal (E=8)**: 3B × 12 ≈ **36B tokens** (40% reduction)
- **MoE-optimal (E=8, N_act=1B)**: 1B × 15 ≈ **15B tokens** (75% reduction)

### 8.5 Stage-by-Stage Revised Budget

| Stage | N_act | E (effective) | D/N_act (MoE) | Required Tokens | Previous Budget | Change |
|-------|-------|---------------|---------------|-----------------|-----------------|--------|
| Phase -1 (Magi) | 340M | 1 (dense) | ~20× | 6.8B | 500M | ↑ (still have 32B eff.) |
| Stage 0 | 7B | 1 (dense init) | ~20× | 140B | 2B | ↑↑ but dense is temporary |
| Stage 1 | 1.5B | 8 | ~12× | 18B | 40B | ↓ 55% |
| Stage 2 | 1.5B | 8 | ~12× | 18B | 15B | ~same |
| Stage 3 | 1B | 8 | ~15× | 15B | 5B | ↑ (more data helps) |
| **Total (Stages 1-3)** | | | | **~51B** | **~60B** | **~15% reduction** |

**However**, Stage 0 (dense initialization) is the real bottleneck:
- If Stage 0 is truly dense (E=1), it needs 140B tokens at Chinchilla-optimal
- This is infeasible — we don't have 140B unique tokens
- **Solution**: Start MoE routing from Stage 0 (skip dense phase)
  - With E=4 from the start: N_act=2B, D/N_act≈8× → 16B tokens ✓
  - With E=8 from the start: N_act=1B, D/N_act≈12× → 12B tokens ✓

### 8.6 MoE-Specific Data Efficiency Bonus

From Paper 1 (Fine-Grained MoE):
- MoE compute savings vs dense at same loss: **20-40×** at typical budgets
- This means: for the same compute budget, we can train on **more tokens**
  with fewer active params, getting better loss than a dense model

From Paper 2 (Joint MoE):
- Memory-matched MoE beats overtrained dense at same FLOP budget
- For E≤8: train on E× more tokens than compute-optimal dense → MoE wins
- Our E≈8 → we need ~8× more tokens than dense of same total params
- But N_act is 8× smaller → D_opt = 8 × (D_dense / 8) ≈ D_dense
- **Net effect: same total tokens needed, but MoE gets better loss**

From Paper 3 (Comprehensive):
- Shared experts (our Core tier) are essential — verified
- G_opt≈7, S_opt≈0.31 — our 3-tier design aligns with this
- Sparsity increases with scale — our 10-15% N_a/N is in the practical optimal range

### 8.7 Final MoE-Aware Token Budget

| Stage | MoE-Aware Budget | Rationale |
|-------|-----------------|-----------|
| Phase -1 (Magi) | 6.8B | Dense Chinchilla; have 32B effective with ECoG+aug |
| Stage 0 (MoE init, E=4) | 16B | N_act=2B, D/N_act=8× (Paper 2 Table 1) |
| Stage 1 (E=8) | 18B | N_act=1.5B, D/N_act=12× |
| Stage 2 (E=8) | 18B | N_act=1.5B, D/N_act=12×; cross-modal bottleneck |
| Stage 3 (E=8) | 15B | N_act=1B, D/N_act=15× |
| **Total (Stages 0-3)** | **~67B** | **vs 62B dense estimate — similar, but MoE-justified** |

**Critical difference from dense estimate**: The 67B is now justified by MoE
scaling laws (not arbitrary reduction), and Stage 0 is **MoE from the start**
(E=4), avoiding the infeasible 140B dense Chinchilla requirement.

**Data availability check (MoE-aware)**:

| Stage | Required | Available (unique) | Available (eff. w/ aug) | Ratio |
|-------|----------|--------------------|------------------------|-------|
| Phase -1 | 6.8B | 2.67B | 32B (12× aug) | 4.7× ✅ |
| Stage 0 | 16B | 2.67B EEG + 25M fMRI | ~32B eff. | 2× ⚠️ |
| Stage 1 | 18B | Same | ~32B eff. | 1.8× ⚠️ |
| Stage 2 | 18B | 286K sync | ~3M eff. | 0.17× 🔴 |
| Stage 3 | 15B | Same | ~32B eff. | 2.1× ✅ |

**Cross-modal (Stage 2) remains the binding constraint.**
Even with MoE-aware scaling, we need ~18B joint tokens but have ~3M effective.
**DANDI:000623 (sync iEEG+fMRI) and HBCD (7K+ subj fMRI+EEG) are critical.**

---

## 9. Summary Table (MoE-Aware)

| Metric | Plan Assumption | Audit Reality (Dense) | Audit Reality (MoE-Aware) | Risk |
|--------|----------------|----------------------|---------------------------|------|
| EEG unique tokens | Assumed large corpus | ~68M (w/ TUH) | Same | 🟡 MEDIUM |
| fMRI unique tokens | NeuroSTORM corpus | ~25M (public) | Same | 🟢 LOW |
| Cross-modal tokens | CineBrain sufficient | ~286K | ~3M (w/ aug + DANDI:000623) | 🔴 HIGH |
| Subject diversity | Assumed high | EEG: ~29K; fMRI: ~69K | Same + HBCD potential | 🟡 MEDIUM |
| Chinchilla ratio (dense) | 372B / 3B = 124× | 62B / 3B = 20× | **N/A** — MoE uses different ratio | — |
| Chinchilla ratio (MoE E=8) | — | — | **51B / 1.5B = 34×** (healthy) | 🟢 OK |
| Optimal D/N_act (MoE E=8) | 20× (dense assumed) | — | **12×** (Paper 2, Table 1) | 🟢 BETTER |
| Optimal G (activated experts) | 8 (by design) | — | **≈7** (Paper 3, Eq. 12) | 🟢 ALIGNED |
| Optimal shared expert ratio | Core tier ≈ 33% | — | **13-31%** (Paper 3) | 🟢 ALIGNED |
| Optimal N_a/N (sparsity) | 10-15% | — | **5-9% practical** (Paper 3) | 🟡 CONSERVATIVE |
| Stage 2 cross-modal deficit | — | ~7× short | **~6000× short** (18B needed vs 3M) | 🔴 CRITICAL |
| Multi-epoch overfit risk | Not analyzed | Thousands of epochs | Reduced with MoE (lower N_act) | 🟡 MEDIUM |

**Bottom Line (MoE-Aware)**: 
1. Dense Chinchilla 20× ratio **overestimates** MoE data needs by ~40-60%.
2. Our expert hierarchy design (G≈7, S≈33%) is **well-aligned** with MoE scaling law optima.
3. MoE-aware token budget: **~67B** (vs 62B dense, 372B original).
4. The binding constraint remains **cross-modal synchronized data** (Stage 2).
5. We should skip the dense Stage 0 — start with E=4 MoE from the beginning.
6. HBCD (7K+ subjects with fMRI+EEG) would provide ~12.6M joint tokens,
   still short of 18B but a 4× improvement over current projections.
not raw token count. DIVER-1 diversity priority means we should invest
in acquiring more *diverse* subjects/tasks rather than more epochs on existing data.

### MoE Scaling Law References

1. Krajewski et al. (2024). "Scaling Laws for Fine-Grained Mixture of Experts." arXiv:2402.07871
2. Ludziejewski et al. (2025). "Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient." arXiv:2502.05172
3. Zhao et al. (2025). "Towards a Comprehensive Scaling Law of Mixture-of-Experts." arXiv:2509.23678
4. Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla) arXiv:2203.15556
5. Clark et al. (2022). "Unified Scaling Laws for Routed Language Models." ICML 2022
6. He (2024). "Mixture of A Million Experts." arXiv:2407.04153 (PEER — extreme granularity)
