# Brain MoE-PINN: A Learned Generative Dynamical System for Multimodal Brain Data

**A technical introduction for computational neuroscience and bioinformatics researchers.**

Brain MoE-PINN is a ~770-million-parameter model (default configuration; up to ~1.2B+ with deep experts enabled) that learns to *generate* brain dynamics from multimodal neuroimaging data — EEG, fMRI, and optionally MEG — while obeying physical constraints from nonequilibrium thermodynamics. Unlike a conventional encoder that maps brain data to a static embedding, this model learns a **velocity field** over a latent state space: given the current brain state, it predicts how the state will *evolve*. This lets it both *represent* brain activity (like a foundation model) and *simulate* its temporal evolution (like a biophysical model). The architecture sits at the intersection of representation learning and dynamical systems theory, targeting a regime — mesoscopic whole-brain dynamics with learned physics — that neither tradition currently addresses.

---

## 1. The Problem: Representation vs. Simulation

Two research traditions dominate computational brain modeling, and they rarely speak to each other.

**Brain simulation** solves differential equations for each neuron or neural population. Projects like The Virtual Brain (TVB) use connectome-based neural mass models at 200–1000 brain regions, supporting in-silico lesion experiments and inference via Dynamical Causal Modeling. At the extreme, spiking network simulations on supercomputers (Hygon Exascale, K Computer cerebellar models) integrate Hodgkin-Huxley equations for billions of neurons. These models are *generative* — you can run them forward without external input — but their parameters are hand-tuned or sampled from priors. They do not learn from data.

**Brain representation learning** trains encoders that map brain recordings to a latent space. BrainLM (2024) applies masked autoencoding to 6,700 hours of fMRI; NeuroSTORM (2026) scales this to 50,000+ subjects using a Shifted-Window Mamba backbone. For EEG, Magi provides a BERT-style encoder with 3D electrode embeddings. These models support decoding and transfer learning, but they do not *generate dynamics* — they produce a static embedding for each time window, with no notion of temporal evolution.

**Brain MoE-PINN bridges this gap.** It combines an encoder stack (representation learning) with a physics-structured velocity field (simulation), enabling both encoding *and* generation in a single model. The velocity field obeys the GENERIC (General Equation for Non-Equilibrium Reversible-Irreversible Coupling) formalism from nonequilibrium thermodynamics — a structured decomposition into conservative (energy-preserving) and dissipative (entropy-producing) components. This is not an arbitrary neural network: the dynamics are architecturally constrained to satisfy energy conservation and the Second Law of Thermodynamics.

---

## 2. Architecture Overview

The model processes EEG and fMRI simultaneously, with optional MEG — EEG (19–128 channels at 256 Hz), fMRI (400 brain regions at ~0.5 Hz), and MEG (306 channels at 256 Hz, `use_meg=True`) — through a pipeline with five stages:

```
EEG → Magi encoder ──────────┐
fMRI → NeuroSTORM encoder ───┼──→ Hub Token Fusion → z_global
MEG → BERT-medium encoder ───┘ (optional)                        │
                                                                    ▼
                                               Velocity Field (GENERIC + MoE)
                                                                    │
                                                                    ▼
                                               Decoder → EEĜ, fMRÎ, MEĜ (if enabled)
```

### 2.1 Encoders

**EEG — Magi (8 layers, 512 dimensions).** A BERT-medium transformer with Sliding Window Attention, RoPE position encoding, GeGLU activations, and BIOT-style 3D electrode embeddings that encode the physical electrode positions. Optionally pretrained during Phase -1 on ~800M EEG+MEG tokens before integration. The encoder is frozen during early training and gradually thawed.

**fMRI — NeuroSTORM (frozen backbone).** A Shifted-Window Mamba backbone pretrained on 28.65 million frames from 50,000+ subjects. We freeze the backbone and train only a small adapter projection (~5M parameters). Task-specific Prompt Tuning (TPT) prepends learnable prompt vectors to the token sequence, enabling task-conditioned processing without modifying the backbone weights.

**MEG — Shared BERT-medium backbone (optional).** No pretrained MEG foundation model exists, so we share the EEG encoder's architecture and initialize from EEG weights via `load_from_eeg()`. This is justified by Maxwell's equations: EEG and MEG both arise from the same post-synaptic currents, differing only in measurement physics (volume conduction vs. magnetic induction). Sharing the backbone enforces this physical correspondence. MEG is disabled by default (`use_meg=False`); when enabled, the hub fusion expands from two to three tokens and the decoder gains a MEG reconstruction head.

### 2.2 Hub Token Fusion

Each encoder produces a sequence of token embeddings. Two learnable *hub tokens* — one per active modality (EEG and fMRI by default) — attend to these sequences via cross-attention, producing modality-aligned representations. A final fusion step produces a single global latent state $z_{\text{global}} \in \mathbb{R}^{1024}$ that integrates information from all available modalities. When MEG is enabled, a third hub token is added. The hub tokens are also preserved for downstream decoding.

### 2.3 The Velocity Field: GENERIC + Mixture of Experts

This is the core of the model. Given a latent state $z$, we predict its velocity $\dot{z}$ — how fast and in which direction the state changes:

$$\dot{z} = \underbrace{L(z)\nabla E}_{\text{conservative}} + \underbrace{M(z)\nabla S}_{\text{dissipative}} + \underbrace{\text{MoE}(z)}_{\text{learned experts}}$$

**The GENERIC contribution** comes from three scalar fields defined over the latent space:
- $E(z)$: an **energy** potential. The conservative term $L(z)\nabla E$ rotates the state without changing its energy, powered by an antisymmetric Poisson operator $L(z)$.
- $S(z)$: an **entropy** potential. The dissipative term $M(z)\nabla S$ drives the state toward higher entropy, powered by a positive-semidefinite mobility matrix $M(z)$.
- The decomposition is constrained by two **degeneracy conditions**: $L(z)\nabla S = 0$ (the Poisson bracket is orthogonal to entropy) and $M(z)\nabla E = 0$ (mobility is orthogonal to energy). These are enforced via gradient projection at every step, keeping the GENERIC structure valid throughout training.

The energy and entropy potentials are *learned* scalar fields — they are not claimed to correspond to physiological energy or entropy. They are effective potentials, validated by whether the resulting dynamics match observed brain activity (spectral slopes, functional connectivity, band-power statistics), not by their interpretability as physical quantities. This is a standard approach in physics-informed neural networks: the structure constrains the dynamics, and the learned fields capture whatever representations produce correct behavior.

**The MoE contribution** adds learned complexity. Eight *shared experts* (50-layer residual MLPs, 52.5M parameters each) are always active, providing a dense backbone. Six *routed experts* (14-layer residual MLPs) are selectively activated via a Top-3 gating mechanism, giving the model the capacity to learn specialized dynamical modes — perhaps one expert for resting-state dynamics, another for visual task processing, etc.

The router is a **recurrent selective state-space model** (a Mamba S6 core with 64-dimensional hidden state), not a stateless MLP. This is important: expert routing at time $t$ depends on the entire history of latent states $z_{<t}$, not just the current state. A stateful router produces smoother, more contextually appropriate expert allocation. The router's state is a registered PyTorch buffer, surviving device transfers and checkpoint save/load. A temperature parameter $\tau$ controls routing softness, annealed from 2.0 (near-uniform, encouraging exploration) to 0.7 (sharp, encouraging specialization) over the course of training.

The Poisson operator $L(z)$ uses a **low-rank factorization**: $L(z) = U(z) \cdot J \cdot U(z)^\top$ where $U(z) \in \mathbb{R}^{1024 \times 64}$ and $J$ is a fixed antisymmetric symplectic matrix. This reduces $L(z)$ from 4M parameters (dense) to 131K (rank-64), a 30× reduction, and forces the conservative dynamics onto a low-dimensional symplectic submanifold. An important consequence: the rank constraint reduces the Jacobi identity violation space from $\sim d^3 \approx 10^9$ directions (for a dense antisymmetric matrix) to $\sim 3dr \approx 2 \times 10^5$ directions — a 5000× reduction. Each $\partial L/\partial z_m$ is rank at most $2r$ because it decomposes as $(\partial U/\partial z_m) J U^\top + U J (\partial U/\partial z_m)^\top$, keeping the entire $L(z)$ trajectory on a rank-$r$ manifold. The $L(z)$ MLP is unconstrained, but $L(z)$ itself always lands on this manifold.

### 2.4 Multi-Time-Scale Dynamics

The velocity field operates at multiple timescales through three mechanisms:

**Multi-Time-Scale KDA (MT-KDA)** — not to be confused with the decoder's Kimi Delta Attention — maintains three parallel exponential moving averages of the velocity with decay rates $\alpha = 0.1, 0.5, 0.9$, corresponding to fast (synaptic, ~10ms), medium (working memory, ~100ms), and slow (contextual, ~1s) dynamics. A learned gating network combines these, and the running states are registered as buffers (not optimizer parameters) to avoid polluting the optimizer with mutable state.

**Slow Manifold Projector** extracts sub-0.25 Hz ultra-slow dynamics via a learned 256×1024 projection. This is where cross-modal alignment between EEG and fMRI lives — the fast electrical signals and slow hemodynamic signals meet on this manifold through a physics-structured Latent HRF Bridge.

**OU-Structured Noise** injects colored (Ornstein-Uhlenbeck) noise with a learnable diffusion coefficient $D$, regulated by a Wiener Homeostat that continuously adapts the noise level to maintain criticality — balancing exploration (noise) against exploitation (signal).

### 2.5 Decoders

The decoder uses **KDA (Kimi Delta Attention)**, a linear-complexity attention mechanism from Kimi (arXiv:2510.26692). Unlike standard softmax attention, KDA uses a delta rule — the attention output is computed as an incremental update to a recurrent hidden state, with per-key-dimension gating for selective state-tracking. This makes it ideal for autoregressive generation: each new token updates the state in $O(d)$ rather than $O(n^2)$. In our architecture, KDA layers are interleaved with standard linear layers in a 1:3 ratio (25% KDA, 75% linear), following the Kimi Linear hybrid design. On CPU, the FLA library's `causal_conv1d` component requires CUDA, so a fallback decoder (`ModalityDecoderRouter`) is available for CPU-only development.

The decoder gate concatenates hub tokens from all active modalities. For a model with EEG + fMRI + MEG enabled, this produces a 3072-dimensional gate input (3 × 1024), enabling cross-modal information to influence reconstruction of each modality.

EEG and fMRI are always reconstructed; MEG is reconstructed only when enabled. The MEG reconstruction loss closes a gradient gap that would otherwise leave the MEG decoder supervised only through the cross-modal hub fusion path.

### 2.6 Active Inference and Imagination

The model operates in two modes:

**Perception mode** processes real sensor data through the encoders. After predicting $z_{t+1}$ via the velocity field, an active inference loop compares the decoder's reconstruction $\hat{x}_{t+1}$ against the actual observation $x_{t+1}$ and corrects the latent state using the prediction error. A Smith Predictor compensates for feedback delays using a history of KDA states (stored in FP32 for numerical stability), and a Precision Gate weights corrections by delay and action magnitude.

**Imagination mode** runs the model forward without external input — the velocity field drives the dynamics autonomously. A Counterfactual Tree Search (depth 3, branch factor 4) explores possible futures, and Expected Free Energy (EFE) scores each trajectory. The best trajectory's endpoint biases the velocity field, creating a closed action-perception loop: imagination directly improves perception.

During training, imagination runs on designated steps (controlled by an `imagination_interval` curriculum) to manage computational cost. Imagination components are gated behind a `use_imagination` flag — when disabled, they consume no memory.

### 2.7 Memory Systems

**Hebbian Associative Memory** stores latent states via Oja's rule — a biologically plausible Hebbian update that maintains normalized weights ($W^\top W \approx I$). The engram trace provides a long-term reference for replay loss, comparing imagined states against stored memories. Hebbian weights are spectrally normalized every 100 steps to prevent runaway growth.

A **replay buffer** accumulates imagined states (capacity 256) and provides temporal pairs for replay loss computation. The buffer is cleared alongside other state when `reset_history()` is called at phase boundaries.

---

## 3. The Loss Function: Physics Through Supervision

The total loss combines 22 terms, each with a stage-dependent weight:

$$\mathcal{L}_{\text{total}} = \sum_{\ell=1}^{22} w_\ell \cdot \tilde{\mathcal{L}}_\ell$$

Where $\tilde{\mathcal{L}}_\ell$ is an EMA-normalized version of each loss (mean/std calibrated over 1,000 warmup steps and frozen thereafter). This prevents one loss from dominating due to scale differences.

**Reconstruction losses** (EEG, fMRI, MEG): Standard MSE between decoder output and input signal. These are the primary training signal — all other losses are auxiliary.

**GENERIC structure losses:**
- *Generic constraint*: Penalizes violations of the degeneracy conditions $L\nabla S \neq 0$ and $M\nabla E \neq 0$. These are the architectural guarantees that the dynamics remain valid GENERIC.
- *Jacobi regularization*: Penalizes violations of the Jacobi identity for $L(z)$, ensuring it defines a valid Poisson bracket. **Currently dormant** — all training phases set `jacobi_reg=0.0` because verifying the Jacobi identity at $d=1024$ requires checking ~8.5 billion triple products, making the sampling-based regularizer statistically ineffective. The low-rank factorization of $L(z)$ (rank-64) suppresses violations structurally without an explicit loss.
- *Grassmannian regularization*: Encourages expert attractor subspaces to be orthogonal, preventing expert collapse. Grounded in the Free Energy Principle (Spisak & Friston, 2025), which predicts attractor orthogonalization emerges naturally from minimizing model complexity.

**Thermodynamic losses:**
- *Dissipation*: Measures entropy production rate $\dot{S} = \nabla S \cdot M \nabla S$. Must be non-negative (Second Law).
- *Entropy Production Rate (EPR)*: Monitors the magnitude of dissipative dynamics. Used as an early warning signal — negative EPR triggers AutoRollback.

**Spectral losses:**
- *Band-power loss*: Compares power in five frequency bands (delta/theta/alpha/beta/gamma, 0.5–50 Hz) between reconstructed and real EEG. This enforces frequency-domain fidelity beyond pixel-level MSE. Includes a 1/f slope constraint: the reconstructed EEG's power spectral density should follow $P(f) \propto 1/f$, a hallmark of healthy brain dynamics (He et al., *Nature Reviews Neuroscience*, 2010).
- *Spectral slope loss*: Enforces the same 1/f constraint on reconstructed EEG and MEG via direct Fourier-domain slope fitting, complementing the band-power approach. Operates on the reconstructed signal (not latent states) because the 1/f property is an observable signature, not a dynamical invariant. Requires at least 16 temporal samples for meaningful slope estimation; shorter sequences return zero loss. fMRI is excluded from this loss because the BOLD signal operates at 0.01–0.1 Hz — a fundamentally different timescale where temporal 1/f constraints are not meaningful at our sampling rate.

**Why spectral constraints operate on reconstructed signals, not latent states.** We initially tracked a rolling buffer of latent states (`z_sequence`) and computed the spectral slope over this trajectory. This was incorrect for two reasons: (1) the buffer mixed states from 128 different batches/subjects, and computing the PSD of a mixed sequence is mathematically meaningless; (2) the 1/f property is an observable signature of neural signals, not a guaranteed property of latent dynamical trajectories. The corrected approach computes spectral constraints on the decoder output — the actual signal a neuroscientist would measure — which is where neurophysiological constraints properly belong.

**Auxiliary losses:**
- *NSP (Neurodynamics Statistics Prediction)*: Predicts band powers and functional connectivity from latent representations, inspired by the DeeperBrain model.
- *Cross-modal alignment*: Cosine similarity and soft contrastive losses between active hub tokens (EEG/fMRI by default; MEG included when enabled). Synchronous data (from paired EEG-fMRI recordings like CineBrain) receives stronger weight; asynchronous data uses soft contrastive comparison.
- *Velocity smoothness*: Penalizes sudden jumps in the velocity field (total variation in latent space), encouraging smooth trajectories.
- *MoE load balancing*: Encourages uniform utilization across routed experts, preventing a single expert from dominating.
- *Hebbian regularization*: Maintains stable Hebbian weight norms via spectral radius constraints.
- *Latent sparsity*: Prevents the latent representation from collapsing to a single dimension (Tsallis-type penalty).
- *Velocity difference regularizer*: Penalizes large velocity differences between consecutive time steps by computing the log-norm of velocity deltas along a trajectory. This encourages smooth temporal dynamics. The name was changed from "KS entropy" to avoid false claims about Lyapunov exponent estimation. Currently preserved in the loss function but dormant — it requires proper temporal sequences (not single-step training) to fire.
- *Action loss*: Minimizes Expected Free Energy for active inference; active in Stage 3 when imagination is enabled.
- *Replay loss*: Compares imagined states against engram traces for memory consolidation; active in Stage 3.

---

## 4. Training Strategy

### 4.1 Phased Curriculum

Training proceeds through a single continuous run with nine sub-phases, progressively introducing complexity. No checkpoint save/restore gaps — the schedule is continuous.

| Phase | Tokens | Key Changes |
|-------|--------|-------------|
| **P0 (warmup)** | 0–0.8B | Encoder warmup. Magi EEG+MEG encoder trained from scratch or loaded from Phase -1. Backbone frozen after warmup. |
| **P1** | 0.8–30B | Alignment + routing differentiation (τ=2.0). NSP + modal_align + meg_align active. Physics losses are **monitoring-only** (weight=0). |
| **P2** | 30–60B | Dissipation (`L_dissip`) + spectral losses activated. NSP decays. Physical constraints begin guiding dynamics. |
| **P3** | 60–90B | TV weight decay 0.1→0.02. EPR enforcement. Velocity difference guard (0.005) active. |
| **P4** | 90–120B | Router temperature tightens (2.0→0.7). EMA startup. Expert specialization begins. |
| **P5** | 120–200B | Full physics constraint suite. Jarzynski + Landauer monitoring. EPR audited (not as loss). |
| **P6** | 200–260B | Long context (seq=4096). Async cross-modal contrastive (`L_cross_soft`) + slow manifold projection. |
| **P7 (Stage 2)** | 260–334B | Seq→16384. Mamba-2 SSM. Full latent HRF bridge (`L_cross`). Adafactor switch. Grassmannian + Waddington. |
| **P8 (Stage 3)** | 334–364B | Shared experts frozen. Hebbian Oja + online assimilation. Imagination 50%. Seq returns to 4096. |

> **Phase -1** (Magi encoder standalone pretraining, ~800M tokens) is optional. If skipped, P0 trains the encoder from scratch.

**Total**: ~364B tokens, ~74 days on 64× V100 (16 nodes × 4 GPUs) at ~14.5% effective MFU.

### 4.2 Why MoE From the Start (No Dense Pretraining)

Standard practice in MoE models (e.g., Mixtral) pretrains a dense model and then converts to MoE. MoE scaling laws (Krajewski et al., 2024; Ludziejewski et al., 2025; Zhao et al., 2025) show this is unnecessary for our scale: starting with E=4 MoE from step 1 requires 16B tokens, while dense initialization would need 140B tokens — infeasible on our hardware. We start with 8 shared + 6 routed experts directly, with a temperature curriculum that encourages broad exploration early and sharp specialization later.

### 4.3 Hardware and Distributed Training

- **64× NVIDIA V100 16GB SXM2** (16 nodes × 4 GPUs)
- **DeepSpeed ZeRO-2**: Gradients and optimizer states sharded across data-parallel nodes (~0.88 GB/GPU)
- **Mixed precision**: FP16 compute with selective FP32 casting for numerically sensitive operations (router, Hebbian, KDA state accumulation). Non-CUDA backends (Intel XPU, Huawei Ascend NPU, Moore Threads MUSA, Cambricon MLU) default to FP16 in autocast (Ascend NPU is not bf16-friendly).
- **Device-agnostic design**: All device references are centralized in `utils/device_utils.py` with auto-detection priority CUDA→XPU→NPU→MUSA→MLU→CPU. No hardcoded `torch.cuda` or `.cuda()` calls outside this module.

---

## 5. Stability and Monitoring

Training a ~770M-parameter model (default) with 22 loss terms and physical constraints requires layered defenses:

**Seven-layer stability defense:**
1. KDA state L1 normalization every 64 steps
2. Degeneracy projection ($L\nabla S = 0$, $M\nabla E = 0$) every step
3. Kahan summation for floating-point accumulation
4. Stable softmax (subtract max before exp)
5. Hebbian spectral normalization every 100 steps
6. EMA monitoring of key metrics (energy, EPR, router entropy)
7. Energy audit: track total energy $E(z)$ to detect divergence

**AutoRollback** reacts to four failure modes without a dense fallback: EPR negativity triggers balance weight increase; energy divergence triggers router reset; router entropy collapse forces Top-1 routing; velocity divergence triggers checkpoint rollback.

**Wiener monitors** run every step and report read-only metrics (no backward path modification):
- Bergson irreversibility gap: measures departure from equilibrium
- Q-Factor: oscillation resonance in the velocity field
- Ataxia: normalized prediction-actual mismatch
- Catalepsy: latent entropy collapse detection

### 5.1 Key Fixes for Training Stability

During development, several stability-critical issues were identified and resolved:

**State management.** Two components stored mutable running state as plain Python attributes or `nn.Parameter` tensors — patterns that fail under `.to(device)`, `state_dict()`, or distributed training. The **PoissonSSMRouter's recurrent SSM state** (which accumulates temporal context for expert routing) and the **MultiTimeScaleKDA's three timescale states** (which maintain fast/medium/slow velocity averages) were both converted to registered PyTorch buffers with in-place `.data` updates. This ensures they survive device transfers, appear in checkpoints, and are properly synchronized across distributed workers.

**Imagination timing.** The joint perception-imagination forward pass (which closes the action-perception loop by running CFTS and EFE computation inside perception mode) was accidentally duplicated — the code block appeared twice, causing CFTS to run twice per step. The duplicate was removed, leaving a single coherent path. Additionally, the `_imagination_active` flag that controls whether imagination runs on a given step was being set *after* the forward call, creating a one-step lag and incorrectly activating imagination on step 0. It now executes before the forward pass, ensuring the flag reflects the current step.

**Dimension hygiene.** The `action` tensor passed to the active inference loop was hardcoded at 2048 dimensions while the latent space is 1024-dimensional. This was fixed to match. In the Poisson router, the standard Mamba S6 discretization $B\Delta = \Delta \cdot B_{\text{proj}}(z)$ was incorrectly written as $B_{\text{proj}}(z) \cdot \Delta.\text{unsqueeze}(-1)$, causing a broadcast failure when batch and state dimensions collided.

---

## 6. Validation and Testable Predictions

The model makes specific, falsifiable predictions that can be tested through ablation experiments:

| Component | Prediction | Test |
|-----------|-----------|------|
| Degeneracy projection | Disabling $L\nabla S = 0$ or $M\nabla E = 0$ causes trajectory instability | Delta-z Frobenius norm growth > 2× |
| Grassmannian regularization | Without it, expert subspaces collapse (become non-orthogonal) | Off-diagonal Gram norm increases |
| Poisson conservative term | Without it, EEG PSD flattens (loses oscillations) | 1/f slope drops below 0.5 (healthy: 0.8–1.2) |
| SSM router vs. MLP router | Recurrent routing reduces expert switching frequency | Switching frequency < 0.5× MLP baseline |
| Band-power + spectral slope loss | Models without these fail to reproduce correct spectral profiles | Per-band MSE; alpha suppression ratio |

**Baseline comparisons** against four simplified architectures (VAE+Neural ODE, Graph Neural ODE, No-hub-token fusion, Transformer trajectory model) on five metrics (reconstruction MSE, PSD slope fit, FC correlation, parameter count, throughput) determine whether each complex component provides net benefit.

---

## 7. Design Principles

Several design choices distinguish this project from conventional deep learning architectures:

**Irreversibility by design.** The model predicts velocity $\Delta z$ (state change), not absolute next state $z_{t+1}$. Combined with causal masking in the decoder, time reversal is architecturally impossible. This is intentional: biological time is irreversible (a Bergsonian *durée*, not a Newtonian clock), and only irreversible dynamics can maintain nonequilibrium steady states.

**Physics constraints as architecture, not loss.** The GENERIC structure is enforced via the architecture itself — the decomposition into $L(z)\nabla E + M(z)\nabla S$ is baked into the forward pass, not imposed as a soft penalty. Loss terms reinforce the constraints but the structure is present regardless of loss weight. This follows the metriplectic dynamics paradigm (Oprisa & Toth, 2026): the dynamics *is* the computation.

**Emergence paradigm.** In Stage 1 P1, all physics losses have zero weight. The model is expected to self-organize through reconstruction pressure and MoE routing competition — complex dynamics should emerge from simple objectives. Physics losses are gradually activated from P2 onward, acting as guardrails rather than hand-holding.

**State is explicit, not hidden.** All temporal state in the model — the router's SSM state, the KDA timescale states, the Hebbian weight matrix, the replay buffer — is registered as named PyTorch buffers. Nothing is stored as a plain Python attribute that would be lost on `.to(device)` or checkpoint save. This is a deliberate engineering choice, not just a fix: in a model where temporal continuity is essential, losing state is losing the computation.

---

## 8. Current Status and Limitations

**What works (verified by smoke test on CPU):**
- Full forward and backward pass with ~770M parameters (default config) or 147M (reduced-dimension test configuration)
- All 20 loss terms compute correctly, including the spectral slope loss on reconstructed EEG/MEG
- Gradient flow verified across all components (encoder stack, velocity field, MoE, decoder)
- Router state persists correctly across forward calls and resets
- Multi-Time-Scale KDA states update correctly as buffers
- Spectral slope guard correctly returns zero loss for sequences with fewer than 16 temporal samples

**Known limitations:**
- The KDA decoder uses `causal_conv1d` from the Flash Linear Attention (FLA) library, which requires CUDA. A fallback decoder is available for CPU development.
- The training loop currently passes `actual_eeg` as a slice of the *input* EEG, not data from time $t+1$. True online active inference requires a paired-sequence dataloader.
- Pretrained NeuroSTORM and BrainLM weights need to be downloaded (loading infrastructure is in place).
- The VelocityDifferenceRegularizer is preserved in the loss function but will not fire until sequence-level training (multi-timestep batches) is implemented.
- Spectral slope loss on eeg_recon partially overlaps with BandPowerLoss's built-in 1/f constraint — weight tuning is needed to avoid double-counting.

**Hardware requirements:**
- 64× V100 16GB SXM2 (16 nodes × 4 GPUs) for full-scale training
- DeepSpeed ZeRO-2/3 with torchrun elastic training
- ~74 days estimated training time at full scale

---

## 9. Further Reading

The research directions in `BRAINSTORM.md` explore extensions including:
- Ensemble Kalman Filtering over latent space for principled uncertainty quantification
- Connectome-conditioned dynamics for subject-specific personalization from DTI
- Model lesioning for causal testing of expert function (e.g., does dropping Core Shared experts collapse resting-state FC?)
- Cross-frequency coupling metrics as neurophysiological validation beyond MSE

**KDA (Kimi Delta Attention)**: arXiv:2510.26692 — the linear-complexity attention mechanism used in our decoder. See `decoder/kda_decoder.py` for the full implementation.

**XMMM Consciousness Algorithm** (Van Schalkwyk, Xzistor LAB, 2026): A control-theoretic architecture for emotion, cognition, and adaptive behavior that aligns closely with the Brain MoE-PINN design — homeostatic drives map to GENERIC dissipative terms, and the Epistemic Isolation Principle has a natural interpretation in GENERIC degeneracy. See §11.11 of the training plan for a detailed mapping.

The literature review in `LITERATURE_REVIEW.md` provides context on the brain simulation and representation learning projects that informed this design.

---

*Code: `/home/yanlu/Documents/a/brain_moe_pinn/` — 65+ Python files, self-contained.*  
*Fundamental principle: Model velocity $\Delta z$, not absolute position. Irreversibility is a feature, not a bug.*
