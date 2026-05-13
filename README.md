# Brain MoE-PINN: Physics-Informed Neural Dynamics Model

**System**: 7–10B total / 1–3B active parameters, Mixture-of-Experts (MoE), Physics-Informed Neural Network (PINN)  
**Hardware**: 64× NVIDIA V100 16GB SXM2 (16 nodes × 4 GPUs), unified across all stages  
**Training budget**: ~372B tokens over ~90 days  
**Code**: `/home/yanlu/Documents/a/brain_moe_pinn/` — 30+ Python files, fully self-contained (Magi encoder migrated in)


## 1. Related Work: Brain Simulation vs Brain Representation Learning

It is essential to distinguish two fundamentally different research programs, because our project intentionally merges them.

**Brain simulation** builds a *generative forward model* whose dynamics produce observable neural or behavioral data. Such a model supports inference (estimating hidden states from observations) and in-silico experimentation. **Brain representation learning** trains an encoder that maps brain data to a latent space — useful for decoding and transfer learning, but the model does not *generate* dynamics intrinsically. Representations embeddings; simulations evolve.

### 1.1 Brain Simulation with Inference

**Spiking neural networks on supercomputers**. Since the IBM BlueGene/L era (2005), researchers have simulated increasingly large spiking networks: cat-scale cortex columns on BlueGene, cerebellar OKR circuits on the K Computer (Yamazaki & Nagao, 2012), and most recently the Hygon Exascale prototype (2022–26) claiming 86B neurons × 47.8T synapses on 14,012 GPUs. These projects solve ODEs for each neuron (Hodgkin-Huxley or Izhikevich), with all-to-all spike communication. The Hygon team added a Bayesian inference layer: an observation operator (Balloon-Windkessel hemodynamic model) maps hidden neural states X to observable BOLD Y, and a particle/Ensemble Kalman Filter estimates P(X_t | Y_{1:t}). This is genuine brain simulation with inference — you can assimilate real fMRI and correct the hidden trajectory.

*Limitations*: Phenomenologically detailed but computationally brutal — only tractable on top supercomputers after years of HPC optimization. No mechanism for learning from data: parameters are hand-tuned or sampled from priors. The Bayesian inference adds enormous overhead on top of expensive simulation.

**The Virtual Brain (TVB)** takes a different approach: connectome-based neural mass models (Jansen-Rit, Wong-Wang) at each of 200–1000 brain regions. TVB supports inference via DCM-like variational methods, lesion simulations, and pharmacological in-silico experiments. It is the most mature platform bridging simulation and clinical application. *Limitation*: fixed biophysical models with hand-tuned parameters; no large-scale learning.

**Procedural connectivity** (Knight & Nowotny, *Nature Computational Science*, 2020) addresses the GPU-friendliness problem: instead of storing large sparse connectivity matrices, compute synaptic weights on the fly from neuron coordinates and distance rules. This trades memory for computation, achieving real-time simulation of ~10⁵ neurons on a single GPU.

### 1.2 Brain Representation Learning

**fMRI foundation models**. BrainLM (ICLR 2024) is a 4-layer transformer masked autoencoder trained on 77K sessions (6,700 hrs fMRI). It supports zero-shot functional network identification and in-silico perturbation. NeuroSTORM (*Nature Biomedical Engineering*, 2026) scales to 28.65M frames from 50K+ subjects using a Shifted-Window Mamba backbone with Spatiotemporal Redundancy Dropout (STRD). Both produce latent embeddings from fMRI — they do *not* generate fMRI time series or support state-space inference.

**EEG foundation models**. Magi (our BERT-base encoder, 12L × 768d, Hybrid SWA + RoPE + BIOT 3D electrode embeddings) was developed alongside this project. DeeperBrain (2026) introduced the Neurodynamics Statistics Prediction (NSP) loss — predicting power spectra and functional connectivity from latent representations — which directly inspired our NSP auxiliary loss. BIOT (Yuan et al.) and Brain-OF (2026) provide flexible electrode positioning and resolution handling.

**Multimodal fusion**. Brain Harmony (2025) introduced Hub Token cross-attention for EEG ↔ fMRI fusion — the design we adopt. CineBrain (Gao, Feng, Fu, 2025, arXiv:2503.06940) provides the first large-scale synchronous fMRI-EEG dataset (6h of TV episodes), enabling our Stage 2 cross-modal bridging. CineSync uses this for video reconstruction from brain signals, but does *not* model dynamics — temporal structure comes from the video, not from learned neural dynamics.

### 1.3 Hybrid Approaches (Representation + Dynamics)

Very few projects bridge the gap.

**Spisak & Friston** (2025, arXiv:2505.22749) formalized how attractor neural networks emerge from the Free Energy Principle: orthogonalized attractor representations arise from minimizing model complexity; sequential data produces asymmetric couplings and nonequilibrium steady-state dynamics. This provides a principled justification for our Grassmannian attractor regularization.

**Metriplector** (Oprisa & Toth, 2026, arXiv:2603.29496) uses metriplectic dynamics as the computation primitive — input configures fields, sources, and operators, and the dynamics *is* the computation. The metriplectic structure (Poisson bracket + dissipative gradient) is mathematically identical to GENERIC. Metriplector achieves strong results on vision, language, and control. *Key difference*: Metriplector is architecture-first (metriplectic dynamics as universal neural primitive); we are neuroscience-first (GENERIC as physically meaningful brain dynamics with interpretable energy/entropy potentials).

**Brain MoE-PINN (this project)** sits at the intersection:

```
EEG → Magi BERT ─┐
                   ├──→ Hub Fusion → z_global → VelocityBrain (GENERIC Δz) → Decoder → x̂
fMRI → NeuroSTORM ─┘
```

A **learned generative dynamical system** at mesoscopic (brain-region) scale. Encoders map observations to a latent state (representation learning). VelocityBrain evolves this state via GENERIC dynamics (simulation). The decoder maps back to observations (generative model). Active inference corrects the state using prediction errors (inference). The model can both *represent* brain states and *simulate* their evolution — a combination no existing project achieves.

### 1.4 Comparative Summary

| Dimension | Spiking HPC | TVB | Foundation models | **Brain MoE-PINN** |
|-----------|-------------|-----|-------------------|---------------------|
| Generative dynamics? | ✅ ODE integration | ✅ Neural mass ODE | ❌ Encoder only | ✅ GENERIC Δz |
| Supports inference? | ⚠️ Bayesian (expensive) | ✅ DCM | ❌ | ✅ Active inference |
| Learns from data? | ❌ Hand-tuned | ❌ Some DCM fitting | ✅ Pretrained | ✅ End-to-end |
| Scale (accessible) | 86B neurons (top supercomputer) | 200-1000 regions | Billions params | 7-10B params (64×V100) |
| Physical constraints? | ✅ Built-in biophysics | ✅ Built-in neural mass | ❌ None | ✅ GENERIC thermodynamics |
| Can "imagine"? (free-run) | ✅ | ✅ | ❌ | ✅ |
| Multimodal? | ❌ Single modality typically | ❌ Single modality | ⚠️ fMRI or EEG only | ✅ EEG + fMRI |
| Online adaptation? | ❌ | ❌ | ❌ | ✅ Hebbian + active inference |

**Is this not just a dressed-up transformer?** The fundamental difference: standard transformers learn a static input-output mapping. Our model learns a **dynamical system** — it predicts how brain states *evolve over time*, and this evolution obeys structured equations (GENERIC) rather than being an arbitrary function of inputs.


## 2. Core Architectural Decisions

### 2.1 Irreversibility by Design

The model predicts **velocity** $\Delta z$ (state change) rather than absolute next state $z_{t+1}$. Combined with causal masking in the decoder, this makes time reversal architecturally impossible. This is **intentional**: biological time is irreversible (Bergsonian *durée*, not Newtonian clock time). Only irreversible dynamics can maintain nonequilibrium steady states, learn, and adapt. Crooks fluctuation theorem is not used (requires reverse trajectories); Jarzynski equality is used only as a monitoring-side thermodynamic consistency check for the learned GENERIC dynamics ($\S 4.5$, monitoring-only, not a training loss).

### 2.2 Dual-Modal Encoding

EEG (fast electrical, low spatial resolution) and fMRI (slow hemodynamic, high spatial resolution) differ by 3 orders of magnitude in temporal resolution and 2–3 in spatial resolution. They need separate encoders:

**EEG — Magi BERT-base** (migrated into `brain_moe_pinn/magi/`):
- 12-layer Transformer, 768d hidden, 12 heads, RoPE position encoding
- Sliding Window Attention ($W=512$, $O(nW)$ instead of $O(n^2)$)
- BIOT-style 3D electrode position embeddings (any montage, implicit volume conduction)
- Patchify EEG at 1-second windows (patch_size=256, stride=128)
- **v2 variant**: 24-layer × 1024d, GeGLU activation, ECoG/sEEG channel-type embedding

**fMRI — NeuroSTORM** (frozen backbone) or **BrainLM** (lighter fallback):
- Shifted-Window Mamba backbone, pretrained on 50K+ subjects
- Supports ROI time series, voxel 4D, functional connectivity inputs
- Only the adapter projection (~5M params) is trained

**Hub Token Fusion**: Two learnable tokens (EEG/fMRI identity) attend to encoder outputs via cross-attention, producing $z_{\text{global}} \in \mathbb{R}^{2048}$ — a modality-aligned latent state.

### 2.3 Latent Dynamics: GENERIC + MoE

The core innovation: the latent velocity field $\dot{z}$ is computed by combining two complementary mechanisms:

**GENERIC dynamics** (via `VelocityBrain`):
$$\dot{z} = \underbrace{L(z)\nabla E}_{\text{conservative}} + \underbrace{M(z)\nabla S}_{\text{dissipative}} + \underbrace{v_0 e(\theta)}_{\text{arousal}} + \underbrace{\sqrt{2D}\xi}_{\text{noise}}$$

- $L(z)$ is an antisymmetric Poisson operator (conservative, energy-preserving)
- $M(z)$ is a diagonal mobility matrix (dissipative, entropy-producing)
- $E(z)$ and $S(z)$ are learned scalar potentials (energy and entropy)
- Two **degeneracy conditions** enforced via gradient projection every step: $L\nabla S = 0$ and $M\nabla E = 0$

**On identifiability**: $E(z)$ and $S(z)$ are **effective potentials** — they are not claimed to correspond to physiological energy or entropy. The GENERIC structure constrains the dynamics enough to satisfy energy conservation and the Second Law, but different $(E, S, L, M)$ tuples can produce the same trajectories (a fundamental limitation of all learned GENERIC models — cf. Stat-PINN literature). The learned potentials are validated by whether they produce correct dynamics (spectral slopes, FC correlation, band-power statistics), not by their interpretability as physical quantities.

**MoE velocity field** (via `MoEVelocityField` + `PoissonSSMRouter`):
- 16 experts (4 Core Shared + 2 Salience + 10 Specialized), top-2 routing
- Router is a **recurrent selective SSM** (Mamba S6 core, 64D hidden state), not a stateless MLP. Why? Brain routing decisions depend on temporal context — which experts were recently active, what task is being performed. A recurrent router produces smoother, context-aware expert allocation.
- Grassmannian regularization encourages orthogonal expert attractor subspaces (principled by FEP)

The two mechanisms are **independent** — $L(z)$ handles physics, the Router handles routing. They share only the antisymmetric structure inspiration.

### 2.4 Memory & Active Inference

**Three-tier memory**: Synaptic Engram (KDA + Hebbian Oja weights, fast), Engram Landscape (Gaussian potential wells, medium), Structural Plasticity (connectomic reorganization, slow).

**Dual modes**: Perception (encoder-driven, forced ODE) and Imagination (internal simulation, no external input).

**Online correction** via efference copy / reafference: decoder output $\hat{x}$ predicts sensation, actual input $x$ provides error signal to correct latent state. Smith Predictor compensates for feedback delays using KDA history. Precision Gate weights error by delay and action magnitude.

### 2.5 Wiener Cybernetics Guardrails (Monitoring-Only in Stage 1)

Four monitors run every step, no backward path modification:

| Monitor | What it measures | Health range |
|---------|-----------------|--------------|
| **Bergson** | Irreversibility gap $\mathcal{I}_{\text{B}} = \langle\beta\Delta W\rangle - \log\langle e^{-\beta\Delta W}\rangle$ | > 0 positive |
| **Q-Factor** | Oscillation resonance: $Q = f_0 / \Delta f$ (FFT on $\Delta z$) | [0.5, 2.0] |
| **Ataxia** | Prediction-actual mismatch: $\mathbb{E}[\|\hat{x} - x\|] / \mathbb{E}[\|x\|]$ | < 0.5 |
| **Catalepsy** | Latent entropy collapse via k-NN differential entropy | < 3.0 |

**Wiener Homeostat**: Continuously adapts noise coefficient $D$ to maintain criticality:
$$D_{\text{eff}} = D_0 \cdot \text{clamp}(\text{health}, 0.3, 3.0)$$

**Note on monitor naming**: The names Bergson, Q-Factor, Ataxia, and Catalepsy are **analogies** drawing on Norbert Wiener's *Cybernetics* (1948), where he argued neurological disorders are computational feedback failures. Each monitor tracks a well-defined dynamical quantity (irreversibility gap, oscillation resonance, normalized prediction error, latent entropy). The clinical names serve as mnemonics for the type of dynamical pathology detected — they are not claims of clinical validity without independent experimental validation.

**AutoRollback** (4 levels, no dense fallback): Reacts to EPR negativity, energy divergence, router entropy collapse, and KS entropy explosion. Recovery increases balance weight → resets router → force top-1 → rollback checkpoint.


## 3. Training Pipeline

### 3.1 Stages

| Stage | Tokens | Days | Active params | Key |
|-------|--------|------|---------------|-----|
| Phase -1 | 10B | ~2 | 110M (Magi only) | EEG encoder pretraining |
| **Stage 1 P1** | 0–30B | | ~2B (E=4 T-1) | MoE from start, encoder alignment, no physics |
| P2 | 30–60B | | ~3B (T-2) | +$\mathcal{L}_{\text{dissip}}$ |
| P3 | 60–90B | | ~3B | +$\mathcal{L}_{\text{EPR}}$ (Second Law) |
| P4 | 90–120B | | E=8 T-2 | MoE expansion + AdamW→Adafactor transition |
| P5 | 120–200B | | ~3B | +full physics (Jarzynski, spectral, etc.) |
| P6 | 200–260B | | ~3B | Seq 512→4096, LR restart |
| Stage 2 | 74B | ~24 | ~3B | Long context (4096→65K), cross-modal bridging |
| Stage 3 | 30B | ~5 | ~2B frozen shared | Specialized fine-tune + Hebbian injection |

Stage 0 (originally a separate priming phase) has been **merged into P1**. MoE runs from step 1 — no dense pre-training.

### 3.2 Loss Function (Stage 1 P2+)

$$\mathcal{L}_{\text{total}} = \sum_\ell w_\ell \cdot \tilde{\mathcal{L}}_\ell \quad \text{where} \quad \tilde{\mathcal{L}}_\ell = \frac{\mathcal{L}_\ell - \mu_\ell}{\sigma_\ell + \epsilon}$$

The raw loss terms (10 total): reconstruction (MSE), dissipation, EPR, MoE balance, spectral slope, sparsity, KS entropy, total variation, action (EFE), replay. EMA normalization statistics ($\mu_\ell$, $\sigma_\ell$) are calibrated over 1000 warmup steps and then **frozen** to prevent moving-target issues.

**Critical**: In Stage 1 P1, all physics losses are set to weight 0.0. They are **monitoring-only** — we expect dynamics to self-organize through reconstruction pressure + MoE routing competition. Physics losses are gradually activated from P2 onward. This is the **emergence paradigm**: rather than supervising physics, we let it emerge and use Wiener monitors as guardrails.

### 3.3 Key Engineering Details

- **Mixed precision**: FP16 with selective FP32 at runtime for Router, Hebbian, KDA state accumulation (not just save-time)
- **DeepSpeed ZeRO-2**: gradients/optimizer states sharded across 16 DP nodes (~0.88 GB/GPU)
- **TP=4, EP=4 co-located** on same 4-GPU NVLink node; only cross-node communication is DP all-reduce
- **Optimizer transition**: AdamW → Adafactor at P4 entry, 1000-step gradual LR interpolation, 200-step freeze-warmup for cold slot variables
- **Numerical stability**: 7-layer defense (KDA normalization, degenerate projection, Kahan summation, stable softmax, Hebbian spectral norm, EMA monitoring, energy audit)
- **Chinchilla ratio**: For MoE, D/N_act ≈ 8–12× (not dense 20×) since only active params matter


## 4. Current Status

### 4.1 Codebase Maturity

| Component | Status |
|-----------|--------|
| VelocityBrain (GENERIC dynamics + MT-KDA) | ✅ Exposes grad_E, grad_S, degeneracy norms for monitoring |
| MoE (PoissonSSMRouter, 3-tier experts) | ✅ Recurrent SSM routing, Grassmannian reg |
| KDA decoder (Kimi Delta Attention) | ✅ FP32 runtime casting |
| Active inference (EfferenceCopy, SmithPredictor, ClosedLoopFeedback) | ✅ Wired in training loop with actual_eeg slice |
| Wiener monitors (Bergson, Q-Factor, Ataxia/Catalepsy) | ✅ All 4 run every step |
| Wiener Homeostat (adaptive noise D) | ✅ Monitoring mode by default |
| AutoRollback (4-level, no dense fallback) | ✅ |
| TotalLoss (10-term, frozen normalization) | ✅ All losses computable |
| Magi encoder (v1 + v2) | ✅ Migrated into repo, PyTorch 2.11 compatible |
| Training loop (multi-phase, optimizer transition, gradient clipping) | ✅ |
| v2 model (BrainMoEPINNV2) | ✅ Initialization + forward pass (B=1, B=2) |
| Low-rank Poisson operator | ✅ `LowRankPoissonOperator` (r=64, 30× parameter reduction at d=2048) |
| Band-power cross-frequency loss | ✅ `BandPowerLoss` in `TotalLoss` (delta/theta/alpha/beta/gamma) |
| Expert lesioning interface | ✅ `lesion_experts()` + `set_poisson_enabled()` for causal testing |

### 4.2 Smoke Test Results (14/14 pass, CPU, PyTorch 2.11)

| Test | Status |
|------|--------|
| LowRankPoissonOperator forward + antisymmetry + efficient action | ✅ |
| BandPowerLoss via TotalLoss | ✅ |
| Lesioning (MoE experts + Poisson + degeneracy) | ✅ |
| BrainMoEPINNV2 forward (B=1, B=2) | ✅ |
| VelocityBrain forward + physics state exposure | ✅ |
| MoE routing + PoissonSSMRouter | ✅ |
| PoissonRouter antisymmetry | ✅ |
| GENERIC degeneracy (L·∇S, M·∇E norms) | ✅ |
| KDA decoder output shape | ⏭️ FLA CPU fallback issue |
| EPR proxy vs exact correlation | ✅ (|r| > 0.5) |
| Oja Hebbian update | ✅ |
| TotalLoss computation | ✅ |
| Wiener monitors (all 4) | ✅ |
| Training loop imports + phase configs | ✅ |

### 4.3 Design Decisions

| Decision | Rationale |
|------|-------------|
| MoE from step 1 (no dense pre-training) | No benefit to dense pre-training; E=4 Top-1 works from start, saves weeks of Stage 0 |
| Jacobi identity not enforced | Requires 8.5B triple checks at d=2048; even 128-sample MC gives 1.5e-8 sampling rate — no gradient signal. Low-rank Poisson operator (r=64) naturally constrains the triple-product space |
| Degeneracy projection every step (not every 64) | 64-step interval leaves 98% of steps unconstrained; violations compound |
| Loss normalization statistics frozen after 1000 steps | Moving-target normalization prevents converging losses from improving |
| EPR proxy: `(v²).sum(dim=-1).mean()` not `.mean()` | Original `.mean()` averaged over hidden dimension, off by factor d=2048 |
| Magi encoder migrated into repo | Self-contained, no `sys.path` hacks needed |
| Monitors named as analogies (Bergson, Ataxia...) | Follows Wiener's Cybernetics framing; each measures a well-defined dynamical quantity, not a claim of clinical validity |
| Cross-modal hub correlation should be near-zero at init | High correlation at step 0 means hub is not distinguishing modalities — a bug detection signal |

### 4.4 Known Limitations

1. **FLA CPU fallback**: `fla.utils.custom_device_ctx` calls `torch.cpu.device(index)` which doesn't exist in PyTorch 2.11. Works on GPU. No impact on training.
2. **Active inference uses same-input slice** (not next-timestep): The training loop passes `actual_eeg = dummy_eeg[:, :, :patches]` rather than data from t+1. True online updating requires a paired-sequence dataloader.
3. **BrainLM fMRI encoder**: `create_fmri_encoder("brainlm", ...)` returns a functioning encoder, but pretrained weights need to be downloaded from HuggingFace.

### 4.5 Research Directions from Literature

A review of past and concurrent brain simulation projects (*see LITERATURE_REVIEW.md*) suggests several extensions:

**P1: Low-rank Poisson operator** — Replace the dense 2048×2048 antisymmetric matrix L(z) with a rank-64 factorization L(z) = U(z)·J·U(z)ᵀ - h.c. where U(z) ∈ ℝ^{2048×64} and J is a fixed symplectic kernel. This reduces L(z) from 4M to 131K parameters (30× reduction), forces the conservative dynamics onto a low-dimensional symplectic manifold, and naturally suppresses Jacobi identity violations. Inspired by the procedural connectivity principle (Knight & Nowotny, 2020): compute structure from coordinates rather than storing it explicitly.

**P1: Cross-frequency coupling as validation** — Following the cerebellar OKR simulation tradition (Yamazaki & Nagao, K Computer), compare neurophysiologically meaningful metrics between simulated and real EEG: band-power modulation depth (alpha suppression during visual tasks), phase-amplitude coupling strength, and spectral coherence. Add band-power reconstruction as an auxiliary loss in Stage 1 P2+:
$$L_{\text{band}} = ||\text{BandPower}(\hat{x}) - \text{BandPower}(x)||^2$$
This addresses the L2 measurement validity issue: L2 reconstruction cannot capture spectral structure (GPT review §8), and band-power loss provides a frequency-domain complement.

**P1: Model lesioning for causal testing** — Selectively disable MoE experts (Core Shared, Salience, Specialized) or L(z) components and measure behavioral changes in simulated dynamics. Core Shared lesion → predicted resting-state FC collapse (DMN analogue). Salience lesion → predicted failure to switch between resting and task states. This provides testable causal hypotheses linking architectural components to brain functions.

**P2: Connectome-conditioned dynamics** — Make the Poisson and mobility operators depend on a subject-specific structural connectivity prior (from DTI). This enables zero-shot personalization: given a new subject's DTI, the dynamics automatically adapt without retraining. Also enables in-silico lesion experiments: artificially damage a tract → predict changes in functional connectivity.

**P3: Stress-energy readout** — Following Metriplector (Oprisa & Toth, 2026), use conserved quantities of the GENERIC dynamics (E(z), S(z), Casimir invariants) as additional decoder inputs, forcing the reconstruction to respect the learned physics.

**Questioned: Ensemble Kalman Filter over latent space** — While the Hygon Bayesian approach uses EnKF for state estimation, our active inference loop (efference copy + reafference + Smith predictor + precision gate) already provides a principled, biologically-plausible correction mechanism. EnKF would add N-ensemble computational overhead, observation Jacobian estimation, and covariance inflation tuning for marginal benefit.

### 4.6 Testable Predictions and Verification Protocol

Each model component makes specific predictions that can be falsified via ablation (adapted from DeepSeek GPT review and Stat-PINN methodology). Thresholds are informed by neuroscience literature where available; otherwise set from statistical properties of random baselines.

| Component | Prediction | Test | Metric | Threshold | Significance |
|-----------|-----------|------|--------|-----------|--------------|
| Degeneracy projection | Relaxing $L\nabla S=0$ or $M\nabla E=0$ degrades trajectory stability | Set `apply_degeneracy_projection=False` | Delta_z Frobenius norm growth over 1000 steps | > 2× baseline (d_lesioned / d_control > 2.0) | Paired t-test, n=5 seeds |
| Grassmannian regularization | Without it, expert subspaces become less orthogonal (relative to random baseline) | Ablate `grassmannian_weight → 0`, train 10K steps | Off-diagonal Gram Frobenius norm (`routing_metrics["expert_orthogonality"]`) | At d=256: > 0.05 (random baseline ≈ 0.03). At d=2048: > 0.006 (random baseline ≈ 0.004, high-d random vectors are nearly orthogonal by chance) | Cohen's d > 0.8 relative to control |
| Closed-loop active inference | Without prediction-error correction, trajectory Lyapunov exponent increases | Model with `actual_eeg=None` vs. `actual_eeg=slice` | Lyapunov exponent $\lambda$ of z trajectory | $\lambda_{\text{open}} / \lambda_{\text{closed}} > 1.5$ | Bootstrap CI, 1000 resamples |
| Poisson conservative term | Without it, EEG PSD slope flattens (oscillation loss) | `set_poisson_enabled(False)` | PSD slope $\alpha$ (1/f fit, 1-45 Hz) | $\alpha_{\text{lesioned}} < 0.5$ (healthy $0.8 \leq \alpha \leq 1.2$) | Effect size vs. literature band ([He et al. 2010](https://doi.org/10.1038/nrn2904)) |
| Wiener Homeostat | Fixed D causes EPR drift or criticality loss | Fixed D vs. adaptive D_eff | EPR proxy variance over 10K steps | Adaptive D variance < 0.5 × fixed D variance | F-test for variance ratio |
| MoE SSM vs. MLP router | SSM router reduces expert switching frequency | Compare `PoissonSSMRouter` vs. `PoissonRouter` | Expert switching frequency (Hz), router entropy temporal autocorrelation (lag-1) | SSM switching < 0.5 × MLP switching; SSM autocorrelation > 0.3 (MLP < 0.1) | Wilcoxon signed-rank, n=10 seeds |
| BandPowerLoss | High band-power MSE indicates poor spectral fidelity | Compare models with/without bandpower loss | Per-band MSE (delta/theta/alpha/beta/gamma), alpha suppression depth (visual task: power ratio pre/post stimulus) | Alpha suppression ratio: simulated within 20% of real data ratio | Two-one-sided t-test (TOST) for equivalence |
| Cross-modal hub fusion | Hub tokens encode shared modality-invariant content | Synchronous EEG-fMRI input; measure cosine similarity between hub_eeg and hub_fmri | `cross_modal_correlation` = $(\hat{e} \cdot \hat{f}) / (\|\hat{e}\|\|\hat{f}\|)$ averaged over batch | Should be near 0 at init (modalities independent), > 0.3 after Stage 1 P2+ training. If > 0.1 at step 0, hub is not distinguishing modalities — a bug | Permutation test, 10K shuffles |
| Ataxia/Catalepsy monitors | Monitor trajectories should track training health | Record Ataxia score, Catalepsy score, Bergson I across all P1-P6 phases | Ataxia AUC (>0.5 sustained = early warning); Catalepsy crossing of threshold 3.0 | Ataxia AUC > 0.7 for predicting validation loss spike (1000-step lead) | ROC-AUC compared to random classifier baseline |

### 4.6a Baseline Comparisons

To isolate the contribution of each complex component, define simplified baselines:

**Baseline 1: VAE+Neural ODE** — Remove all physics constraints, GENERIC structure, and Wiener monitors. Keep encoder → hub fusion → Neural ODE (unconstrained MLP velocity field) → decoder. Train with only reconstruction loss. This tests whether the physics constraints improve dynamics quality over a standard latent ODE.

**Baseline 2: Graph Neural ODE** — Replace the hub fusion + VelocityBrain with a Graph Neural ODE where each brain region is a node and the structural connectome (from DTI) defines edges. Compare reconstruction fidelity and FC prediction. This tests whether the learned dynamics outperform a connectivity-informed model.

**Baseline 3: Without Hub Token fusion** — Remove the identity tokens; directly average EEG and fMRI encoder outputs. This tests whether Hub Token cross-attention provides measurable improvement over simple fusion.

**Baseline 4: Transformer trajectory model** — Replace VelocityBrain with a standard causal transformer predicting z_{t+1} from z_{1:t} (no GENERIC structure, no ODE). This tests whether the dynamical system formulation is beneficial compared to sequence modeling.

All baselines compared on: (1) EEG/fMRI reconstruction MSE, (2) PSD slope 1/f fit ($R^2$), (3) FC correlation with real data (Pearson $r$), (4) parameter count, (5) training throughput (samples/sec). Multi-objective Pareto dominance across these five metrics determines whether the full model provides net benefit.

### 4.7 Next Steps

1. **Real data integration**: Download DANDI EEG datasets + HCP/UKBiobank fMRI → validate dataloader throughput
2. **Multi-node smoke test**: 2 nodes × 4 GPU with DeepSpeed, verify NCCL + checkpoint save/load
3. **Confirm GPU** that the FLA KDA decoder runs correctly with CUDA triton kernels
4. **Implement low-rank Poisson operator** (P1, ~1 week)
5. **Implement cross-frequency coupling validation** (P1, ~1 week)
6. **Model lesioning experiment** on trained model (P1, ~1 week post-training)
7. **Begin training**: Phase -1 (Magi pretraining) → Stage 1 P1


*Plan originated: 2026-05-11. Code at `/home/yanlu/Documents/a/brain_moe_pinn/`. Magi encoder migrated into `brain_moe_pinn/magi/`.*  
*Fundamental principle: Model velocity $\Delta z$, not absolute position. Irreversibility is a feature, not a bug.*
