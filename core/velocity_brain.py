"""
VelocityBrain: Latent Dynamics Core.

Implements the GENERIC (General Equation for Non-Equilibrium Reversible-Irreversible Coupling)
dynamics for the brain MoE-PINN model.

The state space is (z, v) where v = dz/dt = delta_z / delta_t.
The model predicts velocity field delta_z rather than absolute position z_{t+1},
making the architecture irreversibile by design (no inverse mapping possible).

GENERIC Equation:
    dz/dt = L(z) * grad_E + M(z) * grad_S + v_0 * e(theta) + sqrt(2D) * xi

Where:
- L(z): antisymmetric Poisson operator (conservative dynamics)
- M(z): diagonal mobility tensor (dissipative dynamics)
- E(z): energy potential
- S(z): entropy potential
- v_0 * e(theta): dopamine-like arousal drive
- xi: OU structured noise (tau=20ms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np


class OUStructuredNoise(nn.Module):
    """
    Ornstein-Uhlenbeck structured noise for neuro-modulator-like fluctuations.
    tau = 20ms corresponds to typical neural oscillation timescales.
    """

    def __init__(self, dim: int = 2048, tau: float = 0.02, dt: float = 0.001, D: float = 1.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.dt = dt
        self.D = D
        self.noise_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, noise_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply OU noise to input tensor.

        Args:
            x: (B, d) input tensor
            noise_state: Optional (B, d) previous noise state for continuity
        Returns:
            x_noisy: (B, d) with OU noise applied
            noise_state: (B, d) new noise state for next step
        """
        if noise_state is None:
            noise_state = torch.zeros_like(x)

        dW = torch.randn_like(x) * math.sqrt(self.dt)
        noise_state = noise_state + (self.dt / self.tau) * (-noise_state) + dW

        noise = torch.tanh(self.noise_proj(noise_state))
        # Scale by sqrt(2D) per GENERIC equation: sqrt(2D) * xi
        noise = math.sqrt(2.0 * self.D) * noise
        return x + noise, noise_state


class GenericPoissonOperator(nn.Module):
    """
    Antisymmetric Poisson operator L(z) for GENERIC dynamics.

    L(z) = A_L(z) - A_L(z)^T where A_L(z) is a learnable matrix function of z.
    Must satisfy Jacobi identity for valid Poisson bracket (enforced via regularization).

    Args:
        hidden_dim: dimension d of latent space
        num_layers: number of MLP layers to compute A_L(z)
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute L(z) for given input z.

        Args:
            z: (B, d) current state
        Returns:
            L_z: (B, d, d) antisymmetric matrices
        """
        B, d = z.shape
        A_z = self.mlp(z)  # (B, d)
        # Broadcast state-dependent vector across matrix rows + add base matrix
        A_z = A_z.unsqueeze(1) + self.A.unsqueeze(0)  # (B, 1, d) + (1, d, d) = (B, d, d)

        L_z = A_z - A_z.transpose(-2, -1)
        return L_z

    def compute_poisson_action(self, z: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Compute L(z) @ grad (matrix-vector product).

        Args:
            z: (B, d) current state
            grad: (B, d) gradient (e.g., energy or entropy gradient)
        Returns:
            (B, d) Poisson bracket action
        """
        L_z = self.forward(z)
        return torch.bmm(L_z, grad.unsqueeze(-1)).squeeze(-1)


class LowRankPoissonOperator(nn.Module):
    """
    Low-rank Poisson operator with symplectic inductive bias.

    Factorizes L(z) as U(z) @ J @ U(z)^T, where:
    - U(z) in R^{d x r} is computed by a small MLP from state z
    - J in R^{r x r} is a fixed antisymmetric symplectic matrix
    - r << d (default 64)

    This reduces parameters from O(d^2) to O(d*r), forces the
    conservative dynamics onto a low-dimensional symplectic
    submanifold, and naturally suppresses Jacobi identity violations
    (the rank-r structure limits the triple-product space).

    Reference: Knight & Nowotny (2020), procedural connectivity;
              literature review P1 low-rank Poisson (§4.5).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        rank: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, hidden_dim * rank))
        self.mlp = nn.Sequential(*layers)

        # Fixed antisymmetric symplectic-like matrix J in R^{r x r}
        # Initialize as block-off-diagonal: J = [[0, I], [-I, 0]]
        half = rank // 2
        J = torch.zeros(rank, rank)
        for i in range(half):
            J[i, half + i] = 1.0
            J[half + i, i] = -1.0
        self.register_buffer("J", J)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute L(z) = U @ J @ U^T for each batch element.

        Args:
            z: (B, d) current state
        Returns:
            L_z: (B, d, d) antisymmetric low-rank matrices
        """
        B, d = z.shape
        U = self.mlp(z).view(B, d, self.rank)  # (B, d, r)
        # U @ J @ U^T is automatically antisymmetric when J is antisymmetric
        L_z = torch.bmm(torch.bmm(U, self.J.unsqueeze(0).expand(B, -1, -1)),
                        U.transpose(-2, -1))  # (B, d, d)
        return L_z

    def compute_poisson_action(self, z: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Compute L(z) @ grad efficiently without materializing L(z).
        Uses: L(z) @ g = U @ J @ (U^T @ g)

        Args:
            z: (B, d) current state
            grad: (B, d) gradient vector
        Returns:
            (B, d) Poisson bracket action
        """
        B, d = z.shape
        U = self.mlp(z).view(B, d, self.rank)  # (B, d, r)
        Utg = torch.bmm(U.transpose(-2, -1), grad.unsqueeze(-1))  # (B, r, 1)
        J_Utg = torch.bmm(self.J.unsqueeze(0).expand(B, -1, -1), Utg)  # (B, r, 1)
        result = torch.bmm(U, J_Utg).squeeze(-1)  # (B, d)
        return result


class GenericMobilityOperator(nn.Module):
    """
    Diagonal mobility operator M(z) for GENERIC dynamics.

    M(z) = diag(g_sal) where g_sal is the salience expert output.
    This represents the rate at which the system dissipates energy into entropy.

    Args:
        hidden_dim: dimension d of latent space
    """

    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.salience_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, salience: torch.Tensor) -> torch.Tensor:
        """
        Compute M(z) @ grad_S.

        Args:
            z: (B, d) current state (unused for diagonal)
            salience: (B, d) salience activation
        Returns:
            (B, d) mobility-scaled entropy gradient
        """
        g_sal = self.salience_gate(salience)
        return g_sal


class EnergyEntropyFields(nn.Module):
    """
    Energy E(z) and entropy S(z) potential fields.

    Both are modeled as neural networks that map latent state z to scalar potentials.
    The gradients of these fields drive the GENERIC dynamics via L(z) and M(z).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        energy_layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            energy_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if i < num_layers - 1 else nn.Identity(),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
        self.energy_net = nn.Sequential(*energy_layers)
        self.energy_head = nn.Linear(hidden_dim, 1)

        entropy_layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            entropy_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if i < num_layers - 1 else nn.Identity(),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
        self.entropy_net = nn.Sequential(*entropy_layers)
        self.entropy_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute E(z) and S(z) potentials.

        Args:
            z: (B, d) latent state
        Returns:
            E: (B, 1) energy potential
            S: (B, 1) entropy potential
        """
        E = self.energy_head(self.energy_net(z))
        S = self.entropy_head(self.entropy_net(z))
        return E, S

    def compute_gradients(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of E and S with respect to z.

        Args:
            z: (B, d) latent state
        Returns:
            grad_E: (B, d) energy gradient
            grad_S: (B, d) entropy gradient
        """
        z.requires_grad_(True)
        E, S = self.forward(z)
        grad_E = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
        grad_S = torch.autograd.grad(S.sum(), z, create_graph=True)[0]
        return grad_E, grad_S


class GenerickeDegeneracyProjection(nn.Module):
    """
    Enforces GENERIC degeneracy conditions periodically.

    Condition 1: L(z) * grad_S(z) = 0 (Poisson operator orthogonal to entropy gradient)
    Condition 2: M(z) * grad_E(z) = 0 (Mobility operator orthogonal to energy gradient)

    These are enforced via gradient projection to maintain valid GENERIC structure.
    """

    def __init__(self, hidden_dim: int = 2048, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def project_L_orthogonal_to_gradS(self, L: torch.Tensor, grad_S: torch.Tensor) -> torch.Tensor:
        """
        Project L to make it orthogonal to grad_S:
        L <- L - (L * grad_S * grad_S^T) / (grad_S^T * grad_S)
        """
        grad_S_norm_sq = torch.sum(grad_S ** 2, dim=-1, keepdim=True) + self.eps
        proj = torch.bmm(L, grad_S.unsqueeze(-1)) * grad_S.unsqueeze(-2)
        proj = proj / grad_S_norm_sq.unsqueeze(-1)
        return L - proj

    def project_M_orthogonal_to_gradE(self, M_diag: torch.Tensor, grad_E: torch.Tensor) -> torch.Tensor:
        """
        Project diagonal M to make it orthogonal to grad_E:
        M_diag <- M_diag - (M_diag * grad_E^2) / (grad_E^2 + eps)
        """
        grad_E_sq = grad_E ** 2 + self.eps
        M_proj = M_diag * grad_E_sq
        M_proj = M_proj / (grad_E_sq + self.eps)
        return M_diag - M_proj

    def forward(
        self,
        L: torch.Tensor,
        M_diag: torch.Tensor,
        grad_E: torch.Tensor,
        grad_S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply both degeneracy projections.

        Returns:
            L_proj: Projected antisymmetric matrix
            M_diag_proj: Projected diagonal mobility
        """
        L_proj = self.project_L_orthogonal_to_gradS(L, grad_S)
        M_diag_proj = self.project_M_orthogonal_to_gradE(M_diag, grad_E)
        return L_proj, M_diag_proj


class VelocityBrain(nn.Module):
    """
    Core latent dynamics module implementing GENERIC equations.

    Predicts velocity field delta_z (irreversible by design) from latent state z
    and optional salience/arousal inputs.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_poisson_layers: int = 3,
        num_energy_layers: int = 3,
        dropout: float = 0.1,
        ou_tau: float = 0.02,
        noise_dim: int = 2048,
        apply_degeneracy_projection: bool = True,
        degeneracy_check_interval: int = 1,
        use_lowrank_poisson: bool = False,
        poisson_rank: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.apply_degeneracy_projection = apply_degeneracy_projection
        self.degeneracy_check_interval = degeneracy_check_interval
        self.step_counter = 0
        self._poisson_enabled = True
        self._degeneracy_enabled = True

        if use_lowrank_poisson:
            self.poisson_op = LowRankPoissonOperator(
                hidden_dim, rank=poisson_rank, num_layers=min(num_poisson_layers, 2), dropout=dropout,
            )
        else:
            self.poisson_op = GenericPoissonOperator(hidden_dim, num_poisson_layers, dropout)
        self.mobility_op = GenericMobilityOperator(hidden_dim)
        self.energy_entropy = EnergyEntropyFields(hidden_dim, num_energy_layers, dropout)
        self.mt_kda = MultiTimeScaleKDA(hidden_dim)

        self.ou_noise = OUStructuredNoise(hidden_dim, ou_tau)

        self.arousal_vector = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.arousal_scale = nn.Parameter(torch.tensor(0.1))

        if apply_degeneracy_projection:
            self.degeneracy_proj = GenerickeDegeneracyProjection(hidden_dim)

    def forward(
        self,
        z: torch.Tensor,
        salience: Optional[torch.Tensor] = None,
        noise_state: Optional[torch.Tensor] = None,
        apply_noise: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute velocity field delta_z from current state z.

        Args:
            z: (B, d) current latent state
            salience: (B, d) optional salience activation for mobility
            noise_state: (B, d) OU noise state for continuity
            apply_noise: whether to apply OU noise
        Returns:
            dict with:
                - delta_z: (B, d) predicted velocity
                - new_noise_state: (B, d) updated noise state
                - metrics: dict of diagnostics
        """
        grad_E, grad_S = self.energy_entropy.compute_gradients(z)

        L_z = self.poisson_op(z)
        if getattr(self, "_poisson_enabled", True):
            poisson_term = torch.bmm(L_z, grad_E.unsqueeze(-1)).squeeze(-1)
        else:
            poisson_term = torch.zeros_like(grad_E)

        if salience is not None:
            M_diag = self.mobility_op(z, salience)
            mobility_term = M_diag * grad_S
        else:
            M_diag = torch.ones_like(grad_S) * 0.1
            mobility_term = M_diag * grad_S

        arousal_term = self.arousal_scale * self.arousal_vector

        if (self.apply_degeneracy_projection
                and self.step_counter % self.degeneracy_check_interval == 0
                and getattr(self, "_degeneracy_enabled", True)):
            L_z, M_diag = self.degeneracy_proj(L_z, M_diag, grad_E, grad_S)

        delta_z = poisson_term + mobility_term + arousal_term

        # Apply multi-time-scale KDA decay
        delta_z = self.mt_kda(delta_z)

        new_noise_state = None
        if apply_noise:
            delta_z, new_noise_state = self.ou_noise(delta_z, noise_state)

        # Post-projection degeneracy violations for monitoring
        # L_z is (B, d, d) from poisson_op; grad_S is (B, d)
        L_grad_S = torch.bmm(L_z, grad_S.unsqueeze(-1)).squeeze(-1)
        L_grad_S_norm = torch.norm(L_grad_S, dim=-1).mean()
        M_grad_E = M_diag * grad_E
        M_grad_E_norm = torch.norm(M_grad_E, dim=-1).mean()

        metrics = {
            "||L_z||_F": torch.norm(L_z, p="fro") / self.hidden_dim,
            "||grad_E||": torch.norm(grad_E, dim=-1).mean(),
            "||grad_S||": torch.norm(grad_S, dim=-1).mean(),
            "||arousal||": torch.norm(self.arousal_vector, dim=-1).mean(),
            "degeneracy_L_grad_S": L_grad_S_norm.item(),
            "degeneracy_M_grad_E": M_grad_E_norm.item(),
        }

        self.step_counter += 1

        return {
            "delta_z": delta_z,
            "new_noise_state": new_noise_state,
            "grad_E": grad_E,
            "grad_S": grad_S,
            "degeneracy_L_grad_S_norm": L_grad_S_norm,
            "degeneracy_M_grad_E_norm": M_grad_E_norm,
            "metrics": metrics,
        }

    def set_poisson_enabled(self, enabled: bool = True):
        """Lesion: enable/disable the Poisson (conservative) dynamics.

        When disabled, the conservative term L(z)∇E is zeroed,
        leaving only dissipative + arousal dynamics. This tests
        the causal contribution of conservative oscillations to
        the model's behavior.
        """
        self._poisson_enabled = enabled

    def set_degeneracy_enabled(self, enabled: bool = True):
        """Enable/disable degeneracy projection.

        When disabled, the constraints L∇S = 0 and M∇E = 0 are
        NOT enforced numerically. Use this to test whether explicit
        projection is necessary, or whether the optimizer satisfies
        the constraints implicitly through the loss landscape.
        """
        self._degeneracy_enabled = enabled


class MultiTimeScaleKDA(nn.Module):
    """
    Multi-Time-Scale Kernel Dynamical Analysis (MT-KDA).

    Three parallel KDA states with different decay rates:
    - alpha=0.1: synaptic post-synaptic potentials (fast)
    - alpha=0.5: working memory (medium)
    - alpha=0.9: long-term context bias (slow)

    These interact via learned dynamic mixing weights.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        alphas: Tuple[float, float, float] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.alphas = alphas
        self.hidden_dim = hidden_dim

        self.state_a = nn.Parameter(torch.zeros(1, hidden_dim))
        self.state_b = nn.Parameter(torch.zeros(1, hidden_dim))
        self.state_c = nn.Parameter(torch.zeros(1, hidden_dim))

        self.mix_weights = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, delta_z: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-time-scale decay to velocity input.

        Args:
            delta_z: (B, d) velocity input
        Returns:
            (B, d) decay-weighted combined output
        """
        B = delta_z.shape[0]
        device = delta_z.device

        self.state_a.data = self.state_a.data.to(device)
        self.state_b.data = self.state_b.data.to(device)
        self.state_c.data = self.state_c.data.to(device)

        state_a = self.state_a.expand(B, -1)
        state_b = self.state_b.expand(B, -1)
        state_c = self.state_c.expand(B, -1)

        state_a = self.alphas[0] * state_a + (1 - self.alphas[0]) * delta_z
        state_b = self.alphas[1] * state_b + (1 - self.alphas[1]) * delta_z
        state_c = self.alphas[2] * state_c + (1 - self.alphas[2]) * delta_z

        combined = torch.cat([state_a, state_b, state_c], dim=-1)
        weights = self.mix_weights(combined)  # (B, 3)

        w_a = weights[:, 0:1]
        w_b = weights[:, 1:2]
        w_c = weights[:, 2:3]

        output = w_a * state_a + w_b * state_b + w_c * state_c

        self.state_a.data = state_a.detach()
        self.state_b.data = state_b.detach()
        self.state_c.data = state_c.detach()

        return output


class WienerHomeostat(nn.Module):
    """
    Wiener Homeostat: adaptive noise regulation maintaining criticality.

    Wiener Cybernetics §11.10.4 / Ch.4, Ch.10: Noise is not an obstacle but an
    essential feature of communication. A noiseless system cannot learn (no
    exploration) nor adapt (no variation). But excessive noise destroys
    homeostasis. The optimal noise level keeps the system at the critical edge.

    This module continuously adapts the noise coefficient D based on:
      - EPR (entropy production rate): proxy for thermodynamic drive
      - Router entropy: proxy for expert diversity / information processing

    The health index combines both:
      health = 0.5 * (epr / target_epr) + 0.5 * (router_entropy / target_entropy)
      D_eff = D_0 * clamp(health, clamp_min, clamp_max)

    Interpretation:
      health ≈ 1.0 : critical (optimal exploration-exploitation balance)
      health < 0.5  : over-damped, too ordered — increase noise
      health > 1.5  : under-damped, too chaotic — decrease noise

    Difference from AutoRollback: the homeostat is continuous, preventive
    regulation (every step). AutoRollback is discrete, reactive (triggered).
    Both are complementary: homeostat maintains daily criticality, AutoRollback
    handles catastrophic collapse.

    P2 priority (Stage 1+). Monitoring-only: no backward path changes.
    """

    def __init__(
        self,
        D_0: float = 1.0,
        target_epr: float = 1.0,
        target_entropy: float = 1.0,
        clamp_min: float = 0.3,
        clamp_max: float = 3.0,
        ema_decay: float = 0.99,
        clip_min: float = 1e-4,
    ):
        super().__init__()
        self.D_0 = D_0
        self.target_epr = target_epr
        self.target_entropy = target_entropy
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.ema_decay = ema_decay
        self.clip_min = clip_min

        self.D_eff = D_0
        self.ema_epr = target_epr
        self.ema_entropy = target_entropy
        self.health_ema = 1.0
        self.step_counter = 0

        self.D_history: List[float] = []
        self.health_history: List[float] = []

    def update(
        self,
        current_epr: float,
        router_entropy: float,
    ) -> Dict[str, float]:
        """
        Compute adapted noise D_eff for current step.

        Args:
            current_epr: entropy production rate (scalar per step)
            router_entropy: router entropy in bits (scalar per step)

        Returns:
            dict with D_eff, health, epr_ratio, entropy_ratio
        """
        epr = max(current_epr, self.clip_min)
        entropy = max(router_entropy, self.clip_min)

        self.ema_epr = self.ema_decay * self.ema_epr + (1.0 - self.ema_decay) * epr
        self.ema_entropy = self.ema_decay * self.ema_entropy + (1.0 - self.ema_decay) * entropy

        target_epr = max(self.target_epr, self.clip_min)
        target_entropy = max(self.target_entropy, self.clip_min)

        epr_ratio = self.ema_epr / target_epr
        entropy_ratio = self.ema_entropy / target_entropy

        health = 0.5 * epr_ratio + 0.5 * entropy_ratio

        self.health_ema = self.ema_decay * self.health_ema + (1.0 - self.ema_decay) * health

        D_eff = self.D_0 * max(self.clamp_min, min(self.clamp_max, self.health_ema))

        self.D_eff = D_eff
        self.step_counter += 1

        self.D_history.append(D_eff)
        self.health_history.append(self.health_ema)
        if len(self.D_history) > 10000:
            self.D_history = self.D_history[-5000:]
            self.health_history = self.health_history[-5000:]

        return {
            "D_eff": D_eff,
            "health": self.health_ema,
            "epr_ratio": epr_ratio,
            "entropy_ratio": entropy_ratio,
            "ema_epr": self.ema_epr,
            "ema_entropy": self.ema_entropy,
        }

    def get_D(self) -> float:
        """Return current effective noise coefficient."""
        return self.D_eff

    def get_report(self) -> Dict:
        return {
            "D_eff": self.D_eff,
            "D_0": self.D_0,
            "health": self.health_ema,
            "ema_epr": self.ema_epr,
            "ema_entropy": self.ema_entropy,
            "target_epr": self.target_epr,
            "target_entropy": self.target_entropy,
            "steps": self.step_counter,
            "D_mean": float(np.mean(self.D_history)) if self.D_history else self.D_0,
            "health_mean": float(np.mean(self.health_history)) if self.health_history else 1.0,
        }


if __name__ == "__main__":
    print("Testing VelocityBrain...")
    vb = VelocityBrain(hidden_dim=2048)
    z = torch.randn(4, 2048)
    salience = torch.randn(4, 2048) * 0.1
    result = vb(z, salience, apply_noise=False)
    print(f"  delta_z shape: {result['delta_z'].shape}")
    print(f"  ||delta_z||: {torch.norm(result['delta_z'], dim=-1).mean().item():.4f}")
    print("  metrics:", {k: f"{v.mean().item():.4f}" for k, v in result['metrics'].items()})

    print("\nTesting MultiTimeScaleKDA...")
    kda = MultiTimeScaleKDA(hidden_dim=2048)
    out = kda(result['delta_z'])
    print(f"  KDA output shape: {out.shape}")
    print("All tests passed!")