"""
MoE Expert Routing with Poisson Router.

The Poisson Router uses an antisymmetric matrix parameterization:
    W = W_sym + (A - A^T)

This combines:
- W_sym: symmetric part for baseline expert affinity scores
- (A - A^T): antisymmetric part derived from FEP/Spisak & Friston 2025
  (Grassmannian geometry of attractor orthogonalization)

Revised architecture (2026-05-16):
- 8 Shared experts (1.0x width, always active, uniform weights = dense backbone)
- 6 Routed experts (0.7x width, Top-3 routing with temperature curriculum tau: 2.0→0.7)
- No salience gating (simplified: shared always-on + routed conditional)
- Routing temperature annealing over training phases replaces Top-K changes

Every 1000 steps: QR reorthogonalization to preserve antisymmetry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class PoissonRouter(nn.Module):
    """
    Poisson Router for MoE expert selection. (Legacy stateless router — kept for ablation.)

    Uses explicit antisymmetric parameterization W = A - A^T
    combined with symmetric baseline scores.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_experts: int = 14,
        num_selected: int = 3,
        bias_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_selected = num_selected

        self.token_proj = nn.Linear(hidden_dim, bias_dim, bias=False)
        self.expert_proj = nn.Linear(bias_dim, num_experts, bias=False)

        self.A_antisym = nn.Parameter(torch.randn(bias_dim, bias_dim).float() * 0.02)
        self.sym_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("steps_since_orthog", torch.tensor(0))
        self.orthog_interval = 1000

    def forward(
        self,
        z: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expert routing scores and selection.

        Args:
            z: (B, d) latent state
            return_scores: If True, return raw routing scores
        Returns:
            gate_weights: (B, num_selected) normalized weights for selected experts
            selected_indices: (B, num_selected) indices of selected experts
            scores: (B, num_experts) raw scores (if return_scores=True)
        """
        B = z.shape[0]

        token_bias = self.token_proj(z)
        token_bias = F.gelu(token_bias)
        token_bias = self.dropout(token_bias)

        scores = self.expert_proj(token_bias)

        A_antisym = self.A_antisym - self.A_antisym.T
        proj_A = torch.matmul(token_bias, A_antisym.T)
        antisym_contrib = self.expert_proj(proj_A)

        combined_scores = self.sym_scale * scores + (1 - self.sym_scale) * antisym_contrib

        topk_result = torch.topk(combined_scores, k=self.num_selected, dim=-1)
        selected_indices = topk_result.indices
        gate_weights = F.softmax(topk_result.values, dim=-1)

        if return_scores:
            return gate_weights, selected_indices, combined_scores
        return gate_weights, selected_indices

    def orthogonalize(self):
        """
        QR reorthogonalization of antisymmetric matrix A.

        Preserves antisymmetry: A_new = Q @ R @ Q^T where Q is orthogonal.
        Then enforces A_new = (A_new - A_new^T) / 2.
        """
        A = self.A_antisym.data
        A_copy = A.clone()

        try:
            Q, R = torch.linalg.qr(A_copy)
            A_new = Q @ R @ Q.T
            A_new = (A_new - A_new.T) / 2
            self.A_antisym.data = A_new
        except Exception:
            pass

    def try_orthogonalize(self, step: int):
        """Call orthogonalize if orthog_interval steps have passed."""
        if step - self.steps_since_orthog >= self.orthog_interval:
            self.orthogonalize()
            self.steps_since_orthog = step


class PoissonSSMRouter(nn.Module):
    """
    Poisson-structured Selective SSM Router (Mamba S6 style).

    Replaces the stateless MLP router with a recurrent state-space model
    that maintains temporal context across steps. The routing decision at
    time t depends on the entire history of latent states z_{<t}, not just
    the current z_t.

    Architecture:
        h_t = A_bar(z_t) ⊙ h_{t-1} + B_bar(z_t) ⊙ x(z_t)   (selective SSM)
        logits_t = C(z_t) · h_t   +   W(z_t)   +   Poisson(z_t)
                  ↑ SSM readout    ↑ direct    ↑ antisymmetric

    where the Poisson term uses A_skew - A_skew^T for GENERIC-inspired
    routing geometry (Grassmannian attractor orthogonalization).

    Design choices:
    - State size s=64: enough for routing context, negligible overhead
    - Temperature curriculum: tau parameter controls routing softness
      (tau=2.0 early = soft/near-uniform; tau=0.7 late = sparse differentiation)

    Usage:
        router = PoissonSSMRouter(hidden_dim=1024, num_experts=6, num_selected=3, tau=2.0)
        # Later in training:
        router.tau = 0.7  # tighten routing
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_experts: int = 6,
        num_selected: int = 3,
        state_dim: int = 64,
        bias_dim: int = 64,
        dropout: float = 0.0,
        tau: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.state_dim = state_dim
        self.bias_dim = bias_dim
        self.tau = tau

        # --- SSM parameters ---
        self.A_log = nn.Parameter(
            torch.log(torch.linspace(0.5, 0.99, state_dim)).unsqueeze(0).float()
        )

        self.x_proj = nn.Linear(hidden_dim, state_dim, bias=False)
        self.dt_proj = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )
        self.B_proj = nn.Linear(hidden_dim, state_dim, bias=False)
        self.C_proj = nn.Linear(hidden_dim, num_experts * state_dim, bias=False)

        self.D_param = nn.Parameter(torch.ones(state_dim).float())

        self.ssm_norm = nn.LayerNorm(state_dim)
        self.ssm_out_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_experts),
        )

        # --- Direct (non-recurrent) path ---
        self.skip_proj = nn.Sequential(
            nn.Linear(hidden_dim, bias_dim),
            nn.GELU(),
            nn.Linear(bias_dim, num_experts),
        )

        # --- Poisson antisymmetric path ---
        self.z_bias_proj = nn.Linear(hidden_dim, bias_dim, bias=False)
        self.A_antisym = nn.Parameter(torch.randn(bias_dim, bias_dim).float() * 0.02)
        self.poisson_head = nn.Linear(bias_dim, num_experts, bias=False)
        self.sym_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("steps_since_orthog", torch.tensor(0))
        self.orthog_interval = 1000

        self.register_buffer("_state", torch.zeros(1, self.state_dim))

    def reset_state(self, batch_size: Optional[int] = None):
        """
        Reset the recurrent SSM state.

        Called at phase boundaries or when starting a new sequence.

        Args:
            batch_size: if provided, reinitialize state to zeros (B, state_dim)
                        if None, keep current state size (for epoch resets
                        where batch dimension is unchanged).
        """
        if batch_size is not None:
            device = self._state.device
            self._state.data = torch.zeros(batch_size, self.state_dim, device=device)

    def forward(
        self,
        z: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expert routing scores via selective SSM + Poisson structure.

        Args:
            z: (B, d) latent state at current step
            return_scores: If True, return raw routing scores
        Returns:
            gate_weights: (B, num_selected) normalized weights
            selected_indices: (B, num_selected) expert indices
            scores: (B, num_experts) raw scores (if return_scores=True)
        """
        B, d = z.shape
        device = z.device

        # Ensure state buffer is on correct device and batch size
        if self._state.device != device:
            self._state.data = self._state.data.to(device)
        if self._state.shape[0] != B:
            self._state.data = torch.zeros(B, self.state_dim, device=device)

        # SSM input projection
        x_in = self.x_proj(z)                           # (B, state_dim)

        # Discretization: dt = softplus(z) controls integration step size
        dt = self.dt_proj(z)                            # (B, state_dim)
        A = -torch.exp(self.A_log.float())              # (1, state_dim)
        A_bar = torch.exp(A * dt)                       # (B, state_dim)
        B_bar = dt * self.B_proj(z)       # (B, state_dim)

        # SSM state update (in-place via .data to preserve buffer registration)
        self._state.data = A_bar * self._state + B_bar * x_in

        # SSM readout
        c_out = self.C_proj(z)                          # (B, num_experts * state_dim)
        c_out = c_out.view(B, self.num_experts, self.state_dim)
        h = self.ssm_norm(self._state)                  # (B, state_dim)
        ssm_logits = (c_out * h.unsqueeze(1)).sum(dim=-1)  # (B, num_experts)

        # Skip connection (non-recurrent direct path)
        skip_logits = self.skip_proj(z)                 # (B, num_experts)

        # Poisson antisymmetric path (GENERIC-inspired routing geometry)
        z_bias = self.z_bias_proj(z)                    # (B, bias_dim)
        A_skew = self.A_antisym - self.A_antisym.T      # (bias_dim, bias_dim)
        poisson_logits = self.poisson_head(z_bias @ A_skew)  # (B, num_experts)

        # Combine all paths
        logits = ssm_logits + skip_logits + self.sym_scale * poisson_logits
        logits = self.dropout(logits)

        # Temperature-scaled routing
        scores = logits / self.tau

        # Top-k selection
        _, selected_indices = torch.topk(scores, self.num_selected, dim=-1)
        gate_scores = torch.gather(scores, -1, selected_indices)
        gate_weights = torch.softmax(gate_scores, dim=-1)

        if return_scores:
            return gate_weights, selected_indices, scores
        return gate_weights, selected_indices

    def get_state_norm(self) -> float:
        """Return the norm of the current SSM state for monitoring."""
        return torch.norm(self._state).item()

    def reset_orthog_counter(self):
        """Reset the orthogonalization step counter."""
        self.steps_since_orthog.zero_()


class ExpertNetwork(nn.Module):
    """
    Single expert network in the MoE.

    Each expert is a 2-layer FFN with LayerNorm and residual connection.
    Width varies by expert type:
    - Shared: 1.0x hidden_dim (always active, dense backbone)
    - Routed: 0.7x hidden_dim (task modulation, Top-3 routed)

    For deep experts, use DeepExpertNetwork instead.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 1024,
        width_multiplier: float = 1.0,
        dropout: float = 0.1,
        use_snorm: bool = True,
    ):
        super().__init__()
        self.width_multiplier = width_multiplier
        self.hidden_dim = int(hidden_dim * width_multiplier)

        self.norm = nn.LayerNorm(input_dim)

        self.w1 = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, input_dim, bias=False)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if use_snorm:
            self.spectral_norm = True
        else:
            self.spectral_norm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        h = self.act(self.w1(x_norm))
        h = self.dropout(h)
        out = self.w2(h)
        return out


class DeepExpertNetwork(nn.Module):
    """
    Deep expert network with pre-LayerNorm, residual connections, and GELU.

    Architecture:
        x -> LN -> Linear(d, d×W) -> GELU -> Dropout
        -> [LN -> Linear(d×W, d×W) -> GELU -> Dropout] × (N-2)
        -> LN -> Linear(d×W, d) -> residual

    Residual connections are added every ``residual_interval`` hidden blocks.
    A 50-layer stack at d=1024, W=1.0 yields ~52 M parameters per expert,
    providing the nonlinear depth needed for complex attractor dynamics.

    Key parameters:
        num_layers: 50-60 for DMN Shared experts, ~30 for Salience,
                    ~14 for Task Specialized
        width_multiplier: 1.0 (Shared) / 0.7 (Routed)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        width_multiplier: float = 1.0,
        num_layers: int = 50,
        dropout: float = 0.1,
        residual_interval: int = 2,
        use_snorm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.width_multiplier = width_multiplier
        hidden_dim = int(input_dim * width_multiplier)
        self.num_layers = num_layers
        self.residual_interval = residual_interval

        # Input projection: input_dim -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Hidden residual blocks: hidden_dim -> hidden_dim
        self.blocks = nn.ModuleList()
        for i in range(max(0, num_layers - 2)):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ))

        # Output projection: hidden_dim -> input_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )

        self.use_snorm = use_snorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_proj(x)

        for i, block in enumerate(self.blocks):
            h_in = h
            h = block(h)
            if (i + 1) % self.residual_interval == 0:
                h = h + h_in

        out = self.output_proj(h)
        return out + residual


class MoEVelocityField(nn.Module):
    """
    Mixture-of-Experts velocity field module (Revised 2026-05-16).

    Architecture:
        8 Shared experts (always active, 1.0x width, uniform weights = dense backbone)
        + 6 Routed experts (Top-3, 0.7x width, temperature curriculum tau: 2.0→0.7)

    Velocity field:
        dz/dt = sum_i shared_i(z) * (1/N_shared)  +  sum_j m_j(z) * routed_j(z)
                └── baseline dynamics ──┘           └── task modulation ──┘
                (always active, 4.0B backbone)       (0.5-0.75B Top-3 active)
                                                     where m_j = Top-3 softmax(router(z)/tau)
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_shared: int = 8,
        num_routed: int = 6,
        width_shared: float = 1.0,
        width_routed: float = 0.7,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        top_k: int = 3,
        grassmannian_weight: float = 0.001,
        router_tau: float = 2.0,
        use_deep_experts: bool = False,
        shared_depth: int = 50,
        routed_depth: int = 14,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_shared + num_routed
        self.router_tau = router_tau
        self.grassmannian_weight = grassmannian_weight
        self.expert_types = ['shared'] * num_shared + ['routed'] * num_routed

        expert_ctor = DeepExpertNetwork if use_deep_experts else ExpertNetwork

        if use_deep_experts:
            self.shared_experts = nn.ModuleList([
                expert_ctor(hidden_dim, width_multiplier=width_shared, num_layers=shared_depth, dropout=dropout)
                for _ in range(num_shared)
            ])
            self.routed_experts = nn.ModuleList([
                expert_ctor(hidden_dim, width_multiplier=width_routed, num_layers=routed_depth, dropout=dropout)
                for _ in range(num_routed)
            ])
        else:
            self.shared_experts = nn.ModuleList([
                expert_ctor(hidden_dim, hidden_dim, width_multiplier=width_shared, dropout=dropout)
                for _ in range(num_shared)
            ])
            self.routed_experts = nn.ModuleList([
                expert_ctor(hidden_dim, hidden_dim, width_multiplier=width_routed, dropout=dropout)
                for _ in range(num_routed)
            ])

        self.all_experts = nn.ModuleList(list(self.shared_experts) + list(self.routed_experts))

        self.router = PoissonSSMRouter(
            hidden_dim=hidden_dim,
            num_experts=num_routed,
            num_selected=top_k,
            tau=router_tau,
        )

        self.gate = nn.Parameter(torch.ones(1))

    def set_router_tau(self, tau: float):
        """Set routing temperature for curriculum learning (sync to router)."""
        self.router_tau = tau
        self.router.tau = tau

    def forward(
        self,
        z: torch.Tensor,
        force_shared_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: compute MoE velocity field. Wraps set_routing_temp()."""
        return self.set_routing_temp(z, force_shared_only=force_shared_only)

    def set_routing_temp(
        self,
        z: torch.Tensor,
        force_shared_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MoE velocity field with Shared+Routed architecture.

        Architecture:
        - Shared experts (num_shared): always active, uniform weight 1/num_shared
        - Routed experts (num_routed): Top-K routing with temperature tau

        dz/dt = mean(shared_i(z)) + sum_j m_j(z) * routed_j(z)
                where m_j = Top-K softmax(router(z) / tau)

        Args:
            z: (B, d) latent state
            force_shared_only: If True, only activate shared experts (dense fallback)
        Returns:
            dict with 'velocity' and routing diagnostics
        """
        B, d = z.shape
        device = z.device
        max_capacity = int(B * self.capacity_factor)
        expert_outputs = [torch.zeros_like(z) for _ in self.all_experts]
        expert_counts = [0] * self.num_experts

        # --- Shared experts: always active, uniform weights ---
        for i in range(self.num_shared):
            expert_outputs[i] = self.all_experts[i](z) / self.num_shared
            expert_counts[i] += B

        if not force_shared_only:
            # --- Routed experts: Top-K with temperature ---
            gate_weights, selected_indices = self.router(z)  # Relative indices [0, num_routed)

            # Map selected indices to absolute expert positions
            selected_indices = selected_indices + self.num_shared
            selected_indices = torch.clamp(selected_indices, min=self.num_shared, max=self.num_experts - 1)

            for b in range(B):
                for k in range(self.top_k):
                    exp_idx = selected_indices[b, k].item()
                    weight = gate_weights[b, k].item()

                    if expert_counts[exp_idx] < max_capacity:
                        expert_outputs[exp_idx][b] += weight * self.all_experts[exp_idx](z[b:b+1]).squeeze(0)
                        expert_counts[exp_idx] += 1

        velocity = torch.stack(expert_outputs).sum(dim=0)

        # Apply lesion mask if set
        lesion_mask = getattr(self, "_lesion_mask", None)
        if lesion_mask is not None and lesion_mask.any():
            lesioned_outputs = [
                torch.zeros_like(z) if lesion_mask[i].item() else expert_outputs[i]
                for i in range(self.num_experts)
            ]
            velocity = torch.stack(lesioned_outputs).sum(dim=0)

        # Compute all expert outputs for Grassmannian regularization (training only)
        grassmannian_loss = torch.tensor(0.0, device=z.device)
        if self.training and self.grassmannian_weight > 0 and B > 0:
            all_expert_outputs = []
            for expert in self.all_experts:
                all_expert_outputs.append(expert(z))
            expert_out_stack = torch.stack(all_expert_outputs, dim=1)
            outputs_norm = expert_out_stack / (expert_out_stack.norm(dim=-1, keepdim=True) + 1e-8)
            gram = torch.matmul(outputs_norm, outputs_norm.transpose(-2, -1))
            mask = torch.eye(self.num_experts, device=gram.device).unsqueeze(0)
            gram_off_diag = (1 - mask) * gram
            orthogonality = (gram_off_diag ** 2).sum() / (self.num_experts * (self.num_experts - 1))
            grassmannian_loss = self.grassmannian_weight * orthogonality

        total_active = sum(1 for c in expert_counts if c > 0)
        routing_metrics = {
            "expert_utilization": [c / max(max_capacity, 1) for c in expert_counts],
            "max_load": max(expert_counts) / (max(max_capacity, 1) / self.num_experts) if max_capacity > 0 else 0,
            "selected_experts": selected_indices,
            "gate_weights": gate_weights,
            "router_entropy": -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean(),
            "num_active_experts": total_active,
            "expert_orthogonality": orthogonality,
        }

        return {
            "velocity": velocity,
            "routing_metrics": routing_metrics,
            "grassmannian_loss": grassmannian_loss,
        }

    def lesion_experts(self, types_to_lesion: List[str]):
        """Lesion: zero out outputs of specified expert types.

        Args:
            types_to_lesion: list of expert types to disable.
                Valid: 'shared', 'routed'.

        This creates a per-expert mask that is applied during forward().
        Use empty list to clear lesions.

        Testable predictions:
        - shared lesioned: resting-state FC should collapse (dense backbone degraded)
        - routed lesioned: task modulation absent, shared-only baseline
        """
        self._lesion_mask = torch.zeros(self.num_experts, dtype=torch.bool)
        for i, t in enumerate(self.expert_types):
            if t in types_to_lesion:
                self._lesion_mask[i] = True

    def reset_router_state(self, batch_size: Optional[int] = None):
        """
        Reset the SSM router state at phase/epoch boundaries.

        The SSM router accumulates temporal context across training steps.
        When the training phase changes (new data distribution, new loss
        schedule, optimizer reset), the accumulated state is no longer
        relevant and should be cleared.

        Args:
            batch_size: batch size for reinitializing state to zeros
        """
        if hasattr(self.router, "reset_state"):
            self.router.reset_state(batch_size)


class ExpertGenericPerturbation(nn.Module):
    """
    Physics-structured expert perturbation for GENERIC dynamics.

    Each expert produces three additive contributions:
        - delta_L: low-rank antisymmetric perturbation to Poisson operator
        - delta_M: diagonal positive perturbation to mobility operator
        - velocity_bias: learned velocity residual (analogous to current expert output)

    Architecture:
        delta_L_i(z) = U_i(z) @ J @ U_i(z)^T   (rank-r antisymmetric)
        delta_M_i(z) = diag( sigmoid( gate_i(z) ) )   (positive diagonal)
        bias_i(z)    = MLP_i(z)   (standard residual velocity)

    The combined dynamics are:
        L_eff = L_base + sum_i w_i * delta_L_i
        M_eff = M_base + sum_i w_i * delta_M_i
        delta_z = L_eff @ grad_E + M_eff * grad_S + sum_i w_i * bias_i + arousal + noise

    This preserves GENERIC structure because:
        1. Each delta_L_i is antisymmetric by construction (UJU^T with J skew-symmetric)
        2. Each delta_M_i is positive-semidefinite by construction (diag(sigmoid))
        3. The weighted sum preserves both properties
        4. Degeneracy projection can be applied to the combined L_eff, M_eff

    Reference: §MoE-GENERIC discussion in training plan.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        rank: int = 16,
        num_bias_layers: int = 2,
        dropout: float = 0.1,
        use_deep_bias: bool = False,
        bias_depth: int = 14,
        width_multiplier: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # ---- delta_L: U_i(z) @ J @ U_i(z)^T ----
        self.U_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * rank),
        )

        half = rank // 2
        J = torch.zeros(rank, rank)
        J[:half, half:] = torch.eye(half)
        J[half:, :half] = -torch.eye(half)
        self.register_buffer("J", J)
        # Explicit antisymmetry guard: J^T = -J
        assert torch.allclose(J.T, -J, atol=1e-6), "J must be antisymmetric (J^T = -J)"

        # ---- delta_M: diag(softplus(g_diag)) + V @ V^T ----
        # PSD by construction, unbounded positive via softplus, with low-rank coupling.
        # Diagonal-only (sigmoid) is too restrictive: bounded and no cross-coordinate coupling.
        self.M_diag_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Low-rank coupling: V(z) in R^{d x r_mob} for cross-coordinate dissipation
        self.M_coupling_rank = max(4, rank // 4)
        self.M_coupling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * self.M_coupling_rank),
        )

        # ---- velocity_bias: standard MLP ----
        if use_deep_bias:
            self.bias_net = DeepExpertNetwork(
                input_dim=hidden_dim,
                width_multiplier=width_multiplier,
                num_layers=bias_depth,
                dropout=dropout,
            )
        else:
            bias_dim = int(hidden_dim * width_multiplier)
            self.bias_net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, bias_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bias_dim, hidden_dim, bias=False),
            )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, d) latent state
        Returns:
            delta_L: (B, d, d) antisymmetric perturbation
            delta_M: (B, d) diagonal positive perturbation
            bias:    (B, d) velocity bias
        """
        B, d = z.shape

        # delta_L via low-rank factorization: U @ J @ U^T, antisymmetric by construction
        U = self.U_proj(z).view(B, d, self.rank)  # (B, d, r)
        J_batch = self.J.unsqueeze(0).expand(B, -1, -1)  # (B, r, r)
        delta_L = torch.bmm(torch.bmm(U, J_batch), U.transpose(-2, -1))  # (B, d, d)

        # delta_M: diag(softplus(g_diag)) + V @ V^T, PSD by construction
        g_diag = self.M_diag_gate(z)  # (B, d)
        M_diag = F.softplus(g_diag)    # (B, d), unbounded positive

        V = self.M_coupling(z).view(B, d, self.M_coupling_rank)  # (B, d, r_mob)

        # Return V factor (not VV^T) so caller can apply Öttinger projection:
        #   P_E (V V^T) P_E = (P_E V) (P_E V)^T
        # This preserves low-rank structure while guaranteeing exact degeneracy.
        delta_M_diag = M_diag  # (B, d)

        # velocity bias
        bias = self.bias_net(z)  # (B, d)

        return delta_L, delta_M_diag, V, bias


class MoEGenericVelocityField(nn.Module):
    """
    GENERIC-preserving Mixture-of-Experts velocity field.

    Each expert produces physics-structured perturbations (delta_L, delta_M, bias)
    rather than arbitrary velocity vectors. The combined perturbations are added
    to a base GENERIC backbone (from VelocityBrain) to produce expert-modulated
    but physically consistent dynamics.

    Architecture:
        - 8 Shared experts (always active): small baseline perturbations
        - 6 Routed experts (Top-3): task-specific modulation
        - Router: PoissonSSMRouter (same as MoEVelocityField)

    Forward signature differs from MoEVelocityField:
        Inputs:  z, grad_E, grad_S, L_base, M_base
        Outputs: delta_L_eff, delta_M_eff, velocity_bias, routing_metrics

    The caller (BrainMoEPINN) assembles the full velocity:
        L_eff = L_base + delta_L_eff
        M_eff = M_base + delta_M_eff
        delta_z = L_eff @ grad_E + M_eff * grad_S + velocity_bias + arousal + noise
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_shared: int = 8,
        num_routed: int = 6,
        rank: int = 16,
        width_shared: float = 1.0,
        width_routed: float = 0.7,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        top_k: int = 3,
        grassmannian_weight: float = 0.001,
        router_tau: float = 2.0,
        use_deep_bias: bool = False,
        shared_depth: int = 50,
        routed_depth: int = 14,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_shared + num_routed
        self.router_tau = router_tau
        self.grassmannian_weight = grassmannian_weight
        self.expert_types = ["shared"] * num_shared + ["routed"] * num_routed

        # All experts produce (delta_L, delta_M, bias)
        self.shared_experts = nn.ModuleList([
            ExpertGenericPerturbation(
                hidden_dim=hidden_dim,
                rank=rank,
                dropout=dropout,
                use_deep_bias=use_deep_bias,
                bias_depth=shared_depth,
                width_multiplier=width_shared,
            )
            for _ in range(num_shared)
        ])
        self.routed_experts = nn.ModuleList([
            ExpertGenericPerturbation(
                hidden_dim=hidden_dim,
                rank=rank,
                dropout=dropout,
                use_deep_bias=use_deep_bias,
                bias_depth=routed_depth,
                width_multiplier=width_routed,
            )
            for _ in range(num_routed)
        ])

        self.all_experts = nn.ModuleList(list(self.shared_experts) + list(self.routed_experts))

        self.router = PoissonSSMRouter(
            hidden_dim=hidden_dim,
            num_experts=num_routed,
            num_selected=top_k,
            tau=router_tau,
        )

    def set_router_tau(self, tau: float):
        """Set routing temperature for curriculum learning."""
        self.router_tau = tau
        self.router.tau = tau

    def forward(
        self,
        z: torch.Tensor,
        grad_E: Optional[torch.Tensor] = None,
        grad_S: Optional[torch.Tensor] = None,
        L_base: Optional[torch.Tensor] = None,
        M_base: Optional[torch.Tensor] = None,
        force_shared_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute expert-conditioned GENERIC perturbations.

        Args:
            z: (B, d) latent state
            grad_E: (B, d) energy gradient (optional, for diagnostics)
            grad_S: (B, d) entropy gradient (optional, for diagnostics)
            L_base: (B, d, d) base Poisson operator (optional, for combined diagnostics)
            M_base: (B, d) base mobility (optional, for combined diagnostics)
            force_shared_only: If True, only shared experts activate
        Returns:
            dict with:
                - delta_L: (B, d, d) combined antisymmetric perturbation
                - delta_M: (B, d) combined positive diagonal perturbation
                - velocity_bias: (B, d) combined velocity bias
                - routing_metrics: dict
                - grassmannian_loss: scalar
        """
        B, d = z.shape
        device = z.device
        max_capacity = int(B * self.capacity_factor)

        # --- Öttinger projector helper ---
        def project_V_orthogonal(V: torch.Tensor, gE: torch.Tensor) -> torch.Tensor:
            """Apply P_E = I - (gE ⊗ gE) / ||gE||^2 to each column of V.
            P_E (V V^T) P_E = (P_E V) (P_E V)^T preserves low-rank + PSD."""
            if gE is None:
                return V
            # V: (B, d, r), gE: (B, d)
            norm_sq = (gE ** 2).sum(dim=-1, keepdim=True) + 1e-8  # (B, 1)
            coeffs = torch.bmm(V.transpose(-2, -1), gE.unsqueeze(-1)).squeeze(-1) / norm_sq  # (B, r)
            # P_E V = V - gE[:, None] * coeffs[:, None, :]
            return V - gE.unsqueeze(-1) * coeffs.unsqueeze(1)

        def apply_diag_projected_to_gradS(m: torch.Tensor, gE: torch.Tensor, gS: torch.Tensor) -> torch.Tensor:
            """
            Exact action of P_E @ diag(m) @ P_E on grad_S, computed in O(d).

            For D = diag(m), P_E = I - (e e^T)/(e^T e):
                P_E D P_E = D - [(m⊙e)e^T + e(m⊙e)^T]/(e^T e)
                            + [(∑ m_i e_i^2) e e^T] / (e^T e)^2
            This is D plus two rank-1 corrections — never dense.
            """
            if gE is None or gS is None:
                return m * gS
            e = gE
            v = gS
            e_dot_e = (e ** 2).sum(dim=-1, keepdim=True) + 1e-8      # (B, 1)
            e_dot_v = (e * v).sum(dim=-1, keepdim=True)              # (B, 1)
            m_dot_e = m * e                                          # (B, d)
            m_dot_e_dot_v = (m_dot_e * v).sum(dim=-1, keepdim=True)  # (B, 1)
            sum_m_e2 = (m * e ** 2).sum(dim=-1, keepdim=True)        # (B, 1)

            # (P_E D P_E) v = m⊙v - (m⊙e)(e·v)/(e·e) - e((m⊙e)·v)/(e·e)
            #                 + e(e·v)(∑m_i e_i^2)/(e·e)^2
            result = m * v
            result -= m_dot_e * e_dot_v / e_dot_e
            result -= e * m_dot_e_dot_v / e_dot_e
            result += e * e_dot_v * sum_m_e2 / (e_dot_e ** 2)
            return result

        # Accumulators for combined perturbations
        delta_L_sum = torch.zeros(B, d, d, device=device, dtype=z.dtype)
        delta_M_diag_sum = torch.zeros(B, d, device=device, dtype=z.dtype)
        V_list = []  # list of (B, d, r_mob) tensors with weights, for exact Öttinger
        bias_sum = torch.zeros(B, d, device=device, dtype=z.dtype)
        expert_counts = [0] * self.num_experts

        # --- Shared experts: always active, uniform weights ---
        for i in range(self.num_shared):
            dL, dM_diag, V, bias = self.all_experts[i](z)
            delta_L_sum += dL / self.num_shared
            delta_M_diag_sum += dM_diag / self.num_shared
            V_list.append((1.0 / self.num_shared, V))
            bias_sum += bias / self.num_shared
            expert_counts[i] += B

        # --- Routed experts: Top-K with temperature ---
        selected_indices = torch.zeros(B, self.top_k, dtype=torch.long, device=device)
        gate_weights = torch.zeros(B, self.top_k, device=device)

        if not force_shared_only:
            gate_weights, selected_indices = self.router(z)  # relative indices [0, num_routed)
            selected_indices = selected_indices + self.num_shared
            selected_indices = torch.clamp(selected_indices, min=self.num_shared, max=self.num_experts - 1)

            for b in range(B):
                for k in range(self.top_k):
                    exp_idx = selected_indices[b, k].item()
                    weight = gate_weights[b, k].item()

                    if expert_counts[exp_idx] < max_capacity:
                        dL, dM_diag, V, bias = self.all_experts[exp_idx](z[b : b + 1])
                        delta_L_sum[b] += weight * dL.squeeze(0)
                        delta_M_diag_sum[b] += weight * dM_diag.squeeze(0)
                        # V from single-sample forward; grad_E must match batch dim
                        V_b = V.squeeze(0).unsqueeze(0)  # (1, d, r)
                        gE_b = grad_E[b : b + 1] if grad_E is not None else None
                        V_list.append((weight, V_b, gE_b))
                        bias_sum[b] += weight * bias.squeeze(0)
                        expert_counts[exp_idx] += 1

        # Apply lesion mask if set
        lesion_mask = getattr(self, "_lesion_mask", None)
        if lesion_mask is not None and lesion_mask.any():
            delta_L_sum.zero_()
            delta_M_diag_sum.zero_()
            V_list.clear()
            bias_sum.zero_()
            for i in range(self.num_experts):
                if not lesion_mask[i].item():
                    if i < self.num_shared:
                        w = 1.0 / self.num_shared
                        dL, dM_diag, V, bias = self.all_experts[i](z)
                        delta_L_sum += w * dL
                        delta_M_diag_sum += w * dM_diag
                        V_list.append((w, V, grad_E))
                        bias_sum += w * bias
                    else:
                        pass

        # --- Öttinger projection on mobility ---
        # Low-rank: P_E (Σ w_i V_i V_i^T) P_E = Σ w_i (P_E V_i)(P_E V_i)^T
        delta_M_lowrank_sum = torch.zeros(B, d, d, device=device, dtype=z.dtype)
        for item in V_list:
            if len(item) == 3:
                w, V, gE_slice = item
                V_proj = project_V_orthogonal(V, gE_slice)
            else:
                w, V = item
                V_proj = project_V_orthogonal(V, grad_E)
            delta_M_lowrank_sum += w * torch.bmm(V_proj, V_proj.transpose(-2, -1))

        # Diagonal: exact rank-1 formulation, action computed on-the-fly in __init__.py
        # We keep delta_M_diag_sum as the raw diagonal weights; the projected action
        # is (P_E diag(M_diag) P_E) @ grad_S, computed directly without materialization.

        # Grassmannian regularization on expert velocity biases
        grassmannian_loss = torch.tensor(0.0, device=device)
        orthogonality = torch.tensor(0.0, device=device)
        if self.training and self.grassmannian_weight > 0 and B > 0:
            all_biases = []
            for expert in self.all_experts:
                _, _, _, bias = expert(z)
                all_biases.append(bias)
            bias_stack = torch.stack(all_biases, dim=1)  # (B, num_experts, d)
            bias_norm = bias_stack / (bias_stack.norm(dim=-1, keepdim=True) + 1e-8)
            gram = torch.bmm(bias_norm, bias_norm.transpose(-2, -1))
            mask = torch.eye(self.num_experts, device=gram.device).unsqueeze(0)
            gram_off_diag = (1 - mask) * gram
            orthogonality = (gram_off_diag ** 2).sum() / (self.num_experts * (self.num_experts - 1))
            grassmannian_loss = self.grassmannian_weight * orthogonality

        total_active = sum(1 for c in expert_counts if c > 0)
        expert_ortho = orthogonality if self.training else torch.tensor(0.0, device=device)

        routing_metrics = {
            "expert_utilization": [c / max(max_capacity, 1) for c in expert_counts],
            "max_load": max(expert_counts) / (max(max_capacity, 1) / self.num_experts) if max_capacity > 0 else 0,
            "selected_experts": selected_indices,
            "gate_weights": gate_weights,
            "router_entropy": -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean(),
            "num_active_experts": total_active,
            "expert_orthogonality": expert_ortho,
        }

        # Pre-compute projected diagonal action on grad_S (exact, O(d))
        delta_M_diag_action = torch.zeros(B, d, device=device, dtype=z.dtype)
        if grad_S is not None:
            delta_M_diag_action = apply_diag_projected_to_gradS(
                delta_M_diag_sum, grad_E, grad_S
            )

        # Optional: diagnostics on combined degeneracy
        if grad_E is not None and grad_S is not None and L_base is not None and M_base is not None:
            L_eff = L_base + delta_L_sum
            L_grad_S = torch.bmm(L_eff, grad_S.unsqueeze(-1)).squeeze(-1)
            # Full M @ grad_E = (P_E diag(M_diag) P_E) @ grad_E + M_lowrank @ grad_E
            # For degeneracy check: M_lowrank was already projected, so M_lowrank @ grad_E = 0
            M_grad_E_diag = apply_diag_projected_to_gradS(delta_M_diag_sum, grad_E, grad_E)
            routing_metrics["combined_L_grad_S_norm"] = torch.norm(L_grad_S, dim=-1).mean().item()
            routing_metrics["combined_M_grad_E_norm"] = torch.norm(M_grad_E_diag, dim=-1).mean().item()

        return {
            "delta_L": delta_L_sum,
            "delta_M_diag": delta_M_diag_sum,
            "delta_M_diag_action": delta_M_diag_action,
            "delta_M_lowrank": delta_M_lowrank_sum,
            "velocity_bias": bias_sum,
            "routing_metrics": routing_metrics,
            "grassmannian_loss": grassmannian_loss,
        }

    def lesion_experts(self, types_to_lesion: List[str]):
        """Lesion: zero out outputs of specified expert types."""
        self._lesion_mask = torch.zeros(self.num_experts, dtype=torch.bool)
        for i, t in enumerate(self.expert_types):
            if t in types_to_lesion:
                self._lesion_mask[i] = True

    def reset_router_state(self, batch_size: Optional[int] = None):
        """Reset the SSM router state at phase/epoch boundaries."""
        if hasattr(self.router, "reset_state"):
            self.router.reset_state(batch_size)


class WorkingMemoryRouter(nn.Module):
    """
    Dual-router system for working memory.

    - Fast Router: driven by immediate perceptual input
    - Slow Router: driven by task-relevant state (alpha=0.95 smoothing)
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_experts: int = 16,
        top_k: int = 2,
    ):
        super().__init__()
        self.fast_router = PoissonSSMRouter(hidden_dim, num_experts, top_k)
        self.slow_router = PoissonSSMRouter(hidden_dim, num_experts, top_k)

        self.slow_state = None
        self.alpha = 0.95

    def forward(
        self,
        z: torch.Tensor,
        task_cue: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fast and slow routing decisions.

        Args:
            z: (B, d) current latent state
            task_cue: (B, d) optional task cue for slow router
        Returns:
            fast_gate, fast_indices, slow_gate, slow_indices
        """
        fast_gate, fast_indices = self.fast_router(z)

        if task_cue is not None:
            slow_input = task_cue
        else:
            slow_input = z

        if self.slow_state is None:
            self.slow_state = slow_input.detach()

        self.slow_state = self.alpha * self.slow_state + (1 - self.alpha) * slow_input
        slow_gate, slow_indices = self.slow_router(self.slow_state.detach())

        return fast_gate, fast_indices, slow_gate, slow_indices


if __name__ == "__main__":
    print("Testing MoE components...")

    moe = MoEVelocityField(hidden_dim=2048)
    z = torch.randn(4, 2048)

    print("\n1. MoEVelocityField with SSM router...")
    out = moe(z)
    print(f"  velocity shape: {out['velocity'].shape}")
    print(f"  router_entropy: {out['routing_metrics']['router_entropy']:.4f}")
    print(f"  active experts: {out['routing_metrics']['num_active_experts']}")

    print("\n2. SSM state continuity (multi-step)...")
    moe.reset_router_state(batch_size=4)
    for step in range(5):
        z_step = torch.randn(4, 2048)
        out_step = moe(z_step)
        print(f"  step {step}: entropy={out_step['routing_metrics']['router_entropy']:.4f}")

    print("\n3. PoissonSSMRouter (standalone)...")
    ssm_router = PoissonSSMRouter(hidden_dim=2048, num_experts=16)
    ssm_router.reset_state(4)
    for step in range(4):
        z_step = torch.randn(4, 2048)
        gw, si = ssm_router(z_step)
        print(f"  step {step}: gate_weights={gw[0].tolist()}, state_norm={ssm_router.get_state_norm():.4f}")

    print("\n4. WorkingMemoryRouter with SSM...")
    wm_router = WorkingMemoryRouter(hidden_dim=2048)
    f_gate, f_idx, s_gate, s_idx = wm_router(z)
    print(f"  fast_gate shape: {f_gate.shape}")

    print("\n5. Router state reset...")
    moe.reset_router_state(batch_size=8)
    out_reset = moe(torch.randn(8, 2048))
    print(f"  after reset with batch 8: velocity shape={out_reset['velocity'].shape}")

    print("\nAll SSM router tests passed!")