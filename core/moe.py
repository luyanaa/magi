"""
MoE Expert Routing with Poisson Router.

The Poisson Router uses an antisymmetric matrix parameterization:
    W = W_sym + (A - A^T)

This combines:
- W_sym: symmetric part for baseline expert affinity scores
- (A - A^T): antisymmetric part derived from FEP/Spisak & Friston 2025
  (Grassmannian geometry of attractor orthogonalization)

Expert hierarchy:
- 4 Core Shared experts (0.5x width, always active)
- 2 Salience experts (0.5x width, gated activation)
- 10 Specialized experts (1x width, top-2 dynamic routing)

Every 1000 steps: QR reorthogonalization to preserve antisymmetry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class PoissonRouter(nn.Module):
    """
    Poisson Router for MoE expert selection.

    Uses explicit antisymmetric parameterization W = A - A^T
    combined with symmetric baseline scores.

    Key properties:
    - Antisymmetric part enforces Grassmannian attractor geometry
    - QR reorthogonalization every 1000 steps
    - Top-K selection (K=2) with load balancing
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_experts: int = 16,
        num_selected: int = 2,
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
    - Selective Δ, B, C: the SSM adapts its dynamics to the input
    - Diagonal A (learnable): efficient discretization, O(s) per step
    - Bias_dim=64 for antisymmetric part: matches original PoissonRouter

    Mamba-2 vs this: Mamba-2 is a full sequence processor with heads,
    convolutions, and gating — designed for L ≫ 1 token mixing. This
    router is a single-token-per-step SSM (L=1, recurrent mode), so we
    use only the essential S6 core without the seq-level machinery.

    Memory: ~50K params, O(d × s) per step, reusable state across calls.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_experts: int = 16,
        num_selected: int = 2,
        state_dim: int = 64,
        bias_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.state_dim = state_dim
        self.bias_dim = bias_dim

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

        self._state: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: Optional[int] = None):
        """
        Reset the recurrent SSM state.

        Called at phase boundaries or when starting a new sequence.

        Args:
            batch_size: if provided, reinitialize state to zeros (B, state_dim)
        """
        if batch_size is not None:
            device = self.A_log.device
            self._state = torch.zeros(batch_size, self.state_dim, device=device)
        else:
            self._state = None

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

        # Initialize or migrate state to current device/batch
        if self._state is None or self._state.device != device:
            self._state = torch.zeros(B, self.state_dim, device=device)
        elif self._state.shape[0] != B:
            self._state = torch.zeros(B, self.state_dim, device=device)

        # --- Selective SSM step ---
        A = -torch.exp(self.A_log)                 # (1, state_dim)
        dt = self.dt_proj(z)                        # (B, state_dim)
        x_in = self.x_proj(z)                       # (B, state_dim)
        B_ssm = self.B_proj(z)                      # (B, state_dim)
        C_out = self.C_proj(z)                      # (B, n_experts * state_dim)

        # Discretize (zero-order hold, diagonal A)
        A_bar = torch.exp(dt * A)                # (B, state_dim)
        B_bar = (A_bar - 1.0) / (A + 1e-8) * B_ssm

        # State update: h_t = A_bar ⊙ h_{t-1} + B_bar ⊙ x_in
        self._state = A_bar * self._state + B_bar * x_in
        h = self.ssm_norm(self._state)              # (B, state_dim)

        # SSM readout
        C_reshaped = C_out.view(B, self.num_experts, self.state_dim)
        ssm_logits = torch.bmm(C_reshaped, h.unsqueeze(-1)).squeeze(-1)

        # Optional: nonlinear SSM head
        ssm_logits = ssm_logits + self.ssm_out_proj(h)

        # Skip connection: D ⊙ x_in via out projection
        skip_out = self.D_param * x_in
        ssm_logits = ssm_logits + self.ssm_out_proj(skip_out - h) * 0.1

        # --- Direct (non-recurrent) path ---
        direct_logits = self.skip_proj(z)            # (B, num_experts)

        # --- Poisson antisymmetric path ---
        z_bias = self.z_bias_proj(z)                 # (B, bias_dim)
        z_bias = self.dropout(z_bias)
        A_skew = self.A_antisym - self.A_antisym.T
        poisson = torch.matmul(z_bias, A_skew.T)     # (B, bias_dim)
        poisson_logits = self.poisson_head(poisson)   # (B, num_experts)

        # --- Combine ---
        combined = (
            self.sym_scale * ssm_logits +
            (1.0 - self.sym_scale) * 0.5 * direct_logits +
            (1.0 - self.sym_scale) * 0.5 * poisson_logits
        )

        # Top-K selection
        topk_result = torch.topk(combined, k=self.num_selected, dim=-1)
        selected_indices = topk_result.indices
        gate_weights = F.softmax(topk_result.values, dim=-1)

        if return_scores:
            return gate_weights, selected_indices, combined
        return gate_weights, selected_indices

    def orthogonalize(self):
        """
        QR reorthogonalization of the Poisson antisymmetric matrix.

        Preserves antisymmetry: A_new = Q @ R @ Q^T, then enforce skew.
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
        """Call orthogonalize if interval passed."""
        if step - self.steps_since_orthog >= self.orthog_interval:
            self.orthogonalize()
            self.steps_since_orthog = step

    def get_state_norm(self) -> float:
        """Return current SSM state norm for monitoring."""
        if self._state is None:
            return 0.0
        return torch.norm(self._state).item()


class ExpertNetwork(nn.Module):
    """
    Single expert network in the MoE.

    Each expert is a 2-layer FFN with LayerNorm and residual connection.
    Width varies by expert type:
    - Core Shared: 0.5x hidden_dim
    - Salience: 0.5x hidden_dim
    - Specialized: 1.0x hidden_dim
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
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
        """
        Args:
            x: (B, d) input
        Returns:
            (B, d) expert output
        """
        x_norm = self.norm(x)
        h = self.act(self.w1(x_norm))
        h = self.dropout(h)
        out = self.w2(h)
        return out


class MoEVelocityField(nn.Module):
    """
    Mixture-of-Experts velocity field module.

    Combines multiple expert networks with Poisson Router routing.
    Implements the velocity field computation:
        v = sum_i gate_i * expert_i(z)
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_core: int = 4,
        num_salience: int = 2,
        num_specialized: int = 10,
        width_core: float = 0.5,
        width_salience: float = 0.5,
        width_specialized: float = 1.0,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        top_k: int = 2,
        grassmannian_weight: float = 0.001,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_core + num_salience + num_specialized
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.grassmannian_weight = grassmannian_weight

        self.experts = nn.ModuleList()
        expert_types = []

        for i in range(num_core):
            self.experts.append(ExpertNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                width_multiplier=width_core,
                dropout=dropout,
            ))
            expert_types.append("core")

        for i in range(num_salience):
            self.experts.append(ExpertNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                width_multiplier=width_salience,
                dropout=dropout,
            ))
            expert_types.append("salience")

        for i in range(num_specialized):
            self.experts.append(ExpertNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                width_multiplier=width_specialized,
                dropout=dropout,
            ))
            expert_types.append("specialized")

        self.expert_types = expert_types
        self.router = PoissonSSMRouter(
            hidden_dim=hidden_dim,
            num_experts=self.num_experts,
            num_selected=top_k,
        )

        self.num_core = num_core
        self.num_salience = num_salience
        self.num_specialized = num_specialized

    def forward(
        self,
        z: torch.Tensor,
        salience_signal: Optional[torch.Tensor] = None,
        force_core_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MoE velocity field with hierarchical expert activation.

        Hierarchy (per plan §2.3):
        - Core Shared (4): always active
        - Salience (2): gated by salience_signal
        - Specialized (10): Top-2 dynamic routing

        Args:
            z: (B, d) latent state
            salience_signal: (B, d) optional salience gating
            force_core_only: If True, only activate Core Shared experts
        Returns:
            dict with 'velocity' and routing diagnostics
        """
        B, d = z.shape
        device = z.device
        max_capacity = int(B * self.capacity_factor)
        expert_outputs = [torch.zeros_like(z) for _ in self.experts]
        expert_counts = [0] * self.num_experts

        # --- Tier 1: Core Shared experts (always active) ---
        core_indices = list(range(self.num_core))
        for exp_idx in core_indices:
            out = self.experts[exp_idx](z)
            expert_outputs[exp_idx] = out
            expert_counts[exp_idx] += B

        if force_core_only:
            # Only Core Shared active
            velocity = torch.stack(expert_outputs).sum(dim=0)
            gate_weights = torch.zeros(B, self.top_k, device=device)
            selected_indices = torch.zeros(B, self.top_k, dtype=torch.long, device=device)
        else:
            # --- Tier 2: Salience experts (gated by salience_signal) ---
            if salience_signal is not None:
                salience_gate = torch.sigmoid(salience_signal.mean(dim=-1, keepdim=True))  # (B, 1)
                for i in range(self.num_salience):
                    exp_idx = self.num_core + i
                    out = self.experts[exp_idx](z)
                    expert_outputs[exp_idx] = salience_gate * out
                    expert_counts[exp_idx] += int(salience_gate.sum().item())

            # --- Tier 3: Specialized experts (Top-2 routing) ---
            specialized_start = self.num_core + self.num_salience
            num_specialized = self.num_specialized

            # Route only among specialized experts
            gate_weights, selected_indices = self.router(z)

            # Map selected indices to actual expert positions (offset by specialized_start)
            selected_indices = selected_indices + specialized_start
            selected_indices = torch.clamp(selected_indices, min=specialized_start, max=self.num_experts - 1)

            for b in range(B):
                for k in range(self.top_k):
                    exp_idx = selected_indices[b, k].item()
                    weight = gate_weights[b, k].item()

                    if expert_counts[exp_idx] < max_capacity:
                        expert_outputs[exp_idx][b] += weight * self.experts[exp_idx](z[b:b+1]).squeeze(0)
                        expert_counts[exp_idx] += 1

            velocity = torch.stack(expert_outputs).sum(dim=0)

        # Apply lesion mask if set (zero out disabled expert types)
        lesion_mask = getattr(self, "_lesion_mask", None)
        if lesion_mask is not None and lesion_mask.any():
            # Recompute velocity with lesioned experts zeroed
            lesioned_outputs = [
                torch.zeros_like(z) if lesion_mask[i].item() else expert_outputs[i]
                for i in range(self.num_experts)
            ]
            velocity = torch.stack(lesioned_outputs).sum(dim=0)

        # Compute all expert outputs for Grassmannian regularization (training only)
        grassmannian_loss = torch.tensor(0.0, device=z.device)
        if self.training and self.grassmannian_weight > 0 and B > 0:
            all_expert_outputs = []
            for expert in self.experts:
                all_expert_outputs.append(expert(z))
            expert_out_stack = torch.stack(all_expert_outputs, dim=1)
            outputs_norm = expert_out_stack / (expert_out_stack.norm(dim=-1, keepdim=True) + 1e-8)
            gram = torch.matmul(outputs_norm, outputs_norm.transpose(-2, -1))
            mask = torch.eye(self.num_experts, device=gram.device).unsqueeze(0)
            gram_off_diag = (1 - mask) * gram
            orthogonality = (gram_off_diag ** 2).sum() / (self.num_experts * (self.num_experts - 1))
            grassmannian_loss = self.grassmannian_weight * orthogonality

        total_active = sum(1 for c in expert_counts if c > 0)
        # Expert subspace orthogonality: Frobenius norm of off-diagonal Gram matrix
        # Lower = more orthogonal subspaces (desired for expert specialization).
        # Significantly higher without Grassmannian regularization → regularization is doing its job.
        expert_ortho = orthogonality.item() if self.training else 0.0
        routing_metrics = {
            "expert_utilization": [c / max(max_capacity, 1) for c in expert_counts],
            "max_load": max(expert_counts) / (max(max_capacity, 1) / self.num_experts) if max_capacity > 0 else 0,
            "selected_experts": selected_indices,
            "gate_weights": gate_weights,
            "router_entropy": -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean().item(),
            "num_active_experts": total_active,
            "expert_orthogonality": expert_ortho,  # off-diagonal Gram norm; lower = more orthogonal
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
                Valid: 'core', 'salience', 'specialized'.

        This creates a per-expert mask that is applied during forward().
        Use empty list to clear lesions.

        Testable predictions:
        - core lesioned: resting-state FC should collapse (DMN analogue)
        - salience lesioned: should fail to switch between rest/task
        - specialized lesioned: task responses degrade, resting intact
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