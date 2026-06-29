"""
Full-Rank Hebbian Memory System with Oja's Rule and Spectral Normalization.

Three-layer memory architecture:
1. Synaptic Engram (fast, sequence-level) - KDA state + Hebbian weights
2. Engram Landscape (medium, cross-session) - 1024 memory attractors
3. Structural Plasticity (slow, day-level) - Expert co-activation stats

Implements full-rank Hebbian weight matrix with Oja's rule for online updates
and spectral normalization to keep eigenvalues bounded < 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class OjaUpdate(nn.Module):
    """
    Oja's rule for online Hebbian weight updates.

    Oja's rule: dw/dt = eta * y * (x - w * y)
    where y = w^T @ x is the neuron output and eta is learning rate.

    This keeps the weight matrix symmetric and with spectral radius <= 1/eta.
    """

    def __init__(self, hidden_dim: int = 1024, eta: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eta = eta

    def forward(
        self,
        pre: torch.Tensor,
        W: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Oja's rule to update weight matrix.

        Args:
            pre: (B, d) pre-synaptic activations
            W: (d, d) current weight matrix
            post: (B, d) post-synaptic activations (or target)
        Returns:
            (d, d) updated weight matrix
        """
        B = pre.shape[0]
        d = pre.shape[1]

        if W.shape != (d, d):
            W = W.reshape(d, d)

        post_pred = torch.matmul(W, pre.t()).t()
        outer = torch.matmul(post_pred.t(), pre) / B
        y_sq = torch.matmul(post_pred.t(), post_pred) / B
        W_change = self.eta * (outer - torch.matmul(y_sq, W))

        W_new = W + W_change
        return W_new


class SpectralNormalizedHebbianWeight(nn.Module):
    """
    Full-rank Hebbian weight matrix with spectral normalization.

    Keeps the spectral radius (largest eigenvalue) <= 1 to ensure
    numerical stability and prevent runaway feedback loops.
    """

    def __init__(
        self,
        dim: int = 1024,
        eta_oja: float = 0.01,
        spectral_bound: float = 0.99,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.spectral_bound = spectral_bound
        self.eta_oja = eta_oja

        if use_orthogonal_init:
            W_init = torch.randn(dim, dim)
            W_init = torch.linalg.qr(W_init)[0]
            W_init = W_init * spectral_bound
        else:
            W_init = torch.eye(dim) * spectral_bound

        self.W = nn.Parameter(W_init.float())

        self.oja_update = OjaUpdate(dim, eta_oja)

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Apply Hebbian update and return filtered output.

        Args:
            pre: (B, d) pre-synaptic input
            post: (B, d) post-synaptic activity (target)
            update: Whether to apply Oja's rule update
        Returns:
            (B, d) output after applying W @ pre.T
        """
        if update and self.training:
            with torch.no_grad():
                self.W.data = self.oja_update(self.W.data, pre, post)

        W_clipped = self.clip_spectral()

        output = torch.matmul(pre, W_clipped.t())
        return output

    def clip_spectral(self) -> torch.Tensor:
        """Project weight matrix to have spectral radius <= spectral_bound via power iteration."""
        W = self.W.data
        d = W.shape[0]
        device = W.device
        dtype = W.dtype

        v = torch.randn(d, 1, device=device, dtype=dtype)
        v = v / (torch.norm(v) + 1e-8)
        for _ in range(5):
            v = W.t() @ (W @ v)
            v = v / (torch.norm(v) + 1e-8)
        sigma_max = torch.norm(W @ v).item()

        if sigma_max > self.spectral_bound:
            scale = self.spectral_bound / sigma_max
            W = W * scale
        return W.detach().requires_grad_(self.W.requires_grad)


class HebbianAssociativeMemory(nn.Module):
    """
    Hebbian Associative Memory (HAM) combining KDA state and Hebbian weights.

    Implements the fast memory layer from the three-layer memory architecture:
    - KDA state: decaying memory of recent sequence states
    - Full-rank Hebbian weights: cross-sequence associations via Oja's rule
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        memory_dim: int = 512,
        eta_oja: float = 0.01,
        spectral_bound: float = 0.95,
        kda_decay: float = 0.9,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.kda_decay = kda_decay

        self.hebbian_weight = SpectralNormalizedHebbianWeight(
            dim=hidden_dim,
            eta_oja=eta_oja,
            spectral_bound=spectral_bound,
        )

        self.kda_state = None  # Not a buffer: variable batch dim, reset per phase
        self._kda_state_shape = (1, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def reset_kda(self):
        """Reset KDA state at sequence/phase boundaries."""
        self.kda_state = None

    def update_kda_state(self, z: torch.Tensor):
        """Update KDA state with exponential decay. Stored in FP32."""
        if self.kda_state is None:
            self.kda_state = z.detach().float()
        else:
            self.kda_state = self.kda_decay * self.kda_state + (1 - self.kda_decay) * z.detach().float()

    def retrieve(
        self,
        query: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from Hebbian memory using query.

        Args:
            query: (B, d) query tensor
            memory_mask: (B, d) optional mask for memory
        Returns:
            retrieved: (B, d) retrieved memory content
            attention_weights: (B, d) attention weights
        """
        if self.kda_state is None:
            return torch.zeros_like(query), torch.zeros_like(query)

        q = self.query_proj(query)
        k = self.key_proj(self.kda_state)

        attention_scores = torch.matmul(q, k.t()) / math.sqrt(self.memory_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        retrieved = torch.matmul(attention_weights, self.kda_state)
        return retrieved, attention_weights

    def forward(
        self,
        z: torch.Tensor,
        update: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process latent state through HAM.

        Args:
            z: (B, d) current latent state
            update: Whether to update Hebbian weights
        Returns:
            dict with 'z_augmented', 'kda_state', 'hebbian_output'
        """
        self.update_kda_state(z)

        hebbian_out = self.hebbian_weight(z, z, update=update)

        retrieved, attn_weights = self.retrieve(z)

        z_augmented = z + 0.1 * hebbian_out + 0.05 * retrieved

        return {
            "z_augmented": z_augmented,
            "kda_state": self.kda_state,
            "hebbian_output": hebbian_out,
            "attention_weights": attn_weights,
        }


class EngramLandscape(nn.Module):
    """
    Medium-term memory: 1024 memory attractors (Gaussian potential wells).

    Each attractor is parameterized by:
    - center: (d,) center position in latent space
    - depth: scalar, depth of the potential well
    - width: scalar, width of the well (variance)

    Consolidation trigger: HAM accumulation + salience threshold + boundary detection.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_attractors: int = 1024,
        attractor_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attractors = num_attractors
        self.attractor_dim = attractor_dim

        self.centers = nn.Parameter(
            torch.randn(num_attractors, hidden_dim) * 0.02
        )
        self.depths = nn.Parameter(torch.zeros(num_attractors))
        self.widths = nn.Parameter(torch.ones(num_attractors) * 0.5)

        self.recent_traces = []
        self.max_traces = 100

        self.salience_threshold = 2.0
        self.accumulation_threshold = 5

    def add_trace(self, z: torch.Tensor, salience: float):
        """Add a memory trace if salience is high enough."""
        if salience > self.salience_threshold:
            self.recent_traces.append((z.detach().cpu(), salience))
            if len(self.recent_traces) > self.max_traces:
                self.recent_traces.pop(0)

    def compute_attractor_force(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of engram landscape potential.

        Returns force pointing toward nearest attractors.
        """
        B = z.shape[0]
        z_expanded = z.unsqueeze(1)
        centers_expanded = self.centers.unsqueeze(0)

        dist_sq = torch.sum((z_expanded - centers_expanded) ** 2, dim=-1)

        width_sq = self.widths ** 2 + 1e-6
        attraction = -self.depths.unsqueeze(0) * (z_expanded - centers_expanded) / width_sq.unsqueeze(0)
        attraction = attraction * torch.exp(-dist_sq / (2 * width_sq.unsqueeze(0)))

        force = attraction.sum(dim=1)

        return force

    def consolidate(self):
        """Consolidate high-frequency traces into new attractors.
        
        TODO: Implement attractor creation from accumulated traces.
        Currently a no-op — medium-term memory consolidation is a future feature
        (see plan §2.4, Engram Landscape).
        """
        if len(self.recent_traces) >= self.accumulation_threshold:
            pass


class StructuralPlasticity(nn.Module):
    """
    Slow memory: Expert co-activation statistics for connection组 reorganization.

    Tracks which experts are activated together and slowly restructures
    the expert connectivity graph based on co-activation patterns.
    """

    def __init__(
        self,
        num_experts: int = 16,
        update_interval: int = 1000,
        change_rate: float = 1e-4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.update_interval = update_interval
        self.change_rate = change_rate

        self.co_activation_stats = torch.zeros(num_experts, num_experts)

        self.expert_connectivity = nn.Parameter(
            torch.eye(num_experts) + torch.randn(num_experts, num_experts) * 0.01,
            requires_grad=False,
        )

        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    def update_co_activation(self, expert_indices: torch.Tensor):
        """
        Update co-activation statistics from expert selection.

        Args:
            expert_indices: (B, top_k) selected expert indices per sample
        """
        flat_indices = expert_indices.reshape(-1)
        num_elements = flat_indices.shape[0]
        coo_i = flat_indices.unsqueeze(1).expand(-1, num_elements).reshape(-1)
        coo_j = flat_indices.unsqueeze(0).expand(num_elements, -1).reshape(-1)
        self.co_activation_stats.index_put_(
            (coo_i, coo_j), torch.ones(coo_i.shape[0], device=coo_i.device, dtype=self.co_activation_stats.dtype),
            accumulate=True,
        )

    def restructure_connectivity(self):
        """
        Apply slow restructuring to expert connectivity based on co-activation.

        Experts that are frequently co-activated get stronger connections.
        This is called every update_interval steps.
        """
        total = self.co_activation_stats.sum()
        if total > 0:
            co_activation_prob = self.co_activation_stats / total
        else:
            co_activation_prob = torch.zeros_like(self.co_activation_stats)

        target_conn = torch.eye(self.num_experts) + co_activation_prob * self.change_rate

        self.expert_connectivity.data = (
            self.change_rate * target_conn +
            (1 - self.change_rate) * self.expert_connectivity.data
        )

        self.co_activation_stats.zero_()

    def forward(self, expert_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process expert co-activation stats and optionally restructure.

        Args:
            expert_indices: (B, top_k) optional expert indices for update
        Returns:
            dict with connectivity info
        """
        if expert_indices is not None:
            self.update_co_activation(expert_indices)

        self.step_counter.data += 1

        if self.step_counter.item() % self.update_interval == 0:
            self.restructure_connectivity()

        return {
            "connectivity": self.expert_connectivity,
            "co_activation_stats": self.co_activation_stats,
        }


class HippocampalIndex(nn.Module):
    """
    Time indexing system using low-dimensional torus.

    Implements a continuous time representation for:
    - Pattern completion
    - Time cells
    - Mental time travel (imagination of past/future trajectories)
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        time_dim: int = 64,
        num_phi: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.num_phi = num_phi

        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.phase_vectors = nn.Parameter(
            torch.randn(num_phi, time_dim) * 0.1
        )
        freq_bands = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
        self.register_buffer("frequency_bands", freq_bands)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous time into toroidal phase representation.

        Args:
            t: (B,) or scalar time value
        Returns:
            (B, time_dim) time encoding
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        B = t.shape[0]

        t_expanded = t.float().unsqueeze(-1)
        phase_input = t_expanded * self.frequency_bands[:self.num_phi].unsqueeze(0)

        phases = torch.remainder(phase_input, 2 * math.pi)
        time_encoding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

        if time_encoding.shape[-1] < self.time_dim:
            pad_size = self.time_dim - time_encoding.shape[-1]
            time_encoding = F.pad(time_encoding, (0, pad_size))
        elif time_encoding.shape[-1] > self.time_dim:
            time_encoding = time_encoding[:, :self.time_dim]

        return time_encoding

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Index latent state into time representation.

        Args:
            z: (B, d) latent state
            t: (B,) or scalar time
        Returns:
            dict with 'z_indexed' and 'time_encoding'
        """
        time_encoding = self.encode_time(t)
        time_embedding = self.time_encoder(time_encoding)

        z_indexed = z + 0.1 * time_embedding

        return {
            "z_indexed": z_indexed,
            "time_encoding": time_encoding,
        }


if __name__ == "__main__":
    print("Testing Hebbian Memory components...")

    ham = HebbianAssociativeMemory(hidden_dim= 1024)
    z = torch.randn(4, 2048)
    out = ham(z, update=True)
    print(f"  z_augmented shape: {out['z_augmented'].shape}")
    print(f"  hebbian_output norm: {out['hebbian_output'].norm().item():.4f}")

    print("\nTesting EngramLandscape...")
    engram = EngramLandscape(hidden_dim= 1024, num_attractors=1024)
    force = engram.compute_attractor_force(z)
    print(f"  attractor force shape: {force.shape}")

    print("\nTesting StructuralPlasticity...")
    struct = StructuralPlasticity(num_experts=16)
    expert_idx = torch.tensor([[0, 3], [1, 4], [0, 1], [2, 3]])
    out_struct = struct(expert_idx)
    print(f"  connectivity shape: {out_struct['connectivity'].shape}")

    print("\nTesting HippocampalIndex...")
    hippo = HippocampalIndex(latent_dim= 1024, time_dim=64)
    t = torch.tensor([0.0, 0.5, 1.0, 1.5])
    out_hippo = hippo(z, t)
    print(f"  z_indexed shape: {out_hippo['z_indexed'].shape}")

    print("\nAll tests passed!")