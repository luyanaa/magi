"""
VelocityBrain: structured latent dynamics core.

The module uses a GENERIC-inspired decomposition as an inductive bias:
antisymmetric conservative candidates, nonnegative diagonal mobility, and
pointwise degeneracy projections.  These algebraic properties do not by
themselves establish a valid Poisson bracket, stochastic entropy production,
or physiological energy/entropy interpretation.

The state space is (z, v) where v = dz/dt = delta_z / delta_t.
The model predicts a velocity field delta_z rather than absolute position
z_{t+1}, making the architecture irreversible by design (no inverse mapping
is implied).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..runtime.device_utils import safe_epsilon
from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np


def project_along(
    direction: torch.Tensor,
    vector: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply ``P_e v`` with ``P_e = I - e eᵀ / |e|²`` in ``O(d)``.

    Used to apply the GENERIC degeneracy projectors without ever materialising
    ``P`` (or the operator it acts on).  ``project_along(e, e) == 0`` by
    construction, which is what makes the degeneracy conditions hold exactly.
    """
    if direction.shape != vector.shape:
        raise ValueError("direction and vector must share shape")
    norm_sq = (direction ** 2).sum(dim=-1, keepdim=True) + eps
    coeff = (direction * vector).sum(dim=-1, keepdim=True) / norm_sq
    return vector - coeff * direction


def oettinger_mobility_action(
    M_diag: torch.Tensor,
    grad_E: torch.Tensor,
    grad_S: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply the Öttinger-projected diagonal mobility to an entropy gradient.

    Returns ``(P_E diag(M) P_E) @ grad_S`` with ``P_E = I - e eᵀ / |e|²`` and
    ``e = grad_E``, evaluated in ``O(d)`` without materializing ``M`` as a
    dense matrix.

    Because ``P_E @ grad_E = 0``, the effective mobility satisfies the GENERIC
    degeneracy condition ``M_eff @ grad_E = 0`` *exactly* for any nonnegative
    diagonal ``M``.  That is why the projection is applied to the operator
    rather than to its action: projecting the action
    (``M grad_S - ((grad_E · M grad_S) / |grad_E|²) grad_E``) only makes the
    force orthogonal to ``grad_E`` and leaves ``M @ grad_E`` untouched, so a
    penalty on ``M @ grad_E`` would drive the mobility itself to zero.
    """
    if M_diag.shape != grad_S.shape:
        raise ValueError("M_diag and grad_S must share shape")
    e = grad_E
    v = grad_S
    e_dot_e = (e ** 2).sum(dim=-1, keepdim=True) + eps
    e_dot_v = (e * v).sum(dim=-1, keepdim=True)
    m_dot_e = M_diag * e
    m_dot_e_dot_v = (m_dot_e * v).sum(dim=-1, keepdim=True)
    sum_m_e2 = (M_diag * e ** 2).sum(dim=-1, keepdim=True)

    result = M_diag * v
    result = result - m_dot_e * e_dot_v / e_dot_e
    result = result - e * m_dot_e_dot_v / e_dot_e
    result = result + e * e_dot_v * sum_m_e2 / (e_dot_e ** 2)
    return result


class OUStructuredNoise(nn.Module):
    """
    Ornstein-Uhlenbeck structured noise for neuro-modulator-like fluctuations.
    tau = 20ms corresponds to typical neural oscillation timescales.
    """

    def __init__(self, dim: int = 1024, tau: float = 0.02, dt: float = 0.001, D: float = 1.0):
        super().__init__()
        if tau <= 0 or dt <= 0:
            raise ValueError("tau and dt must be positive")
        if D < 0:
            raise ValueError("diffusion D must be nonnegative")
        self.dim = dim
        self.tau = tau
        self.dt = dt
        self.register_buffer("D", torch.tensor(D, dtype=torch.float32))
        self.noise_proj = nn.Linear(dim, dim)

    def set_diffusion(self, value):
        """Set a nonnegative scalar or per-output-dimension diffusion gain."""
        if isinstance(value, torch.Tensor):
            v = value.detach().to(self.D.dtype).reshape(-1)
            if v.numel() not in (1, self.dim):
                raise ValueError(f"D must be scalar or ({self.dim},); got {v.numel()}")
            if (v < 0).any():
                raise ValueError("diffusion D must be nonnegative")
            v = v.to(self.D.device)
        else:
            if float(value) < 0:
                raise ValueError("diffusion D must be nonnegative")
            v = torch.tensor(float(value), dtype=self.D.dtype, device=self.D.device)
        self.D = v

    @property
    def is_per_dim(self) -> bool:
        return self.D.numel() > 1

    def forward(self, x: torch.Tensor, noise_state: Optional[torch.Tensor] = None,
                dt=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply OU-structured noise to a velocity input.

        ``x`` is a velocity increment, so the returned tensor carries the
        noise term in *velocity* units.  The caller integrates
        ``z <- z + dt * (delta_z_noisy)``; scaling the velocity noise by
        ``sqrt(2 D / dt)`` therefore gives the state an Ornstein-Uhlenbeck
        colored-noise increment with the correct ``sqrt(2 D dt)`` Euler-
        Maruyama magnitude.  Returning ``sqrt(2 D) * state`` (a velocity-sized
        term) would instead make state noise linear in ``dt``.

        The state update uses the exact discrete OU transition for step
        ``dt``:

            a = exp(-dt / tau)
            state <- a * state + sqrt(1 - a^2) * N(0, I)

        so the state has unit stationary variance and ``D`` alone sets the
        diffusion scale.  The exact form stays valid when ``dt`` approaches or
        exceeds ``tau``, where an Euler step would be unstable.

        Args:
            x: (B, d) velocity input
            noise_state: Optional (B, d) previous unit-variance OU state
            dt: Optional step duration in seconds (scalar or per-sample);
                defaults to the constructor value. Pass the *physical*
                duration of the latent step, so ``tau``/``D`` keep their
                seconds/units meaning regardless of the sampling rate.
        Returns:
            x_noisy: (B, d) with OU noise added in velocity units
            noise_state: (B, d) new OU state for the next step
        """
        if noise_state is None:
            noise_state = torch.zeros_like(x)

        step = (torch.as_tensor(self.dt, dtype=x.dtype, device=x.device)
                if dt is None else as_step_dt(dt, x.shape[0], x.device, x.dtype))
        decay = torch.exp(-step / self.tau)
        innovation = torch.sqrt((1.0 - decay * decay).clamp_min(0.0))
        noise_state = decay * noise_state + innovation * torch.randn_like(x)

        noise = torch.sqrt(2.0 * self.D / step) * noise_state
        return x + noise, noise_state


def as_step_dt(dt, batch_size: int, device, dtype) -> torch.Tensor:
    """Normalize a latent step duration to a broadcastable tensor.

    ``dt`` may be a python number, a 0-dim tensor (one duration for the whole
    batch) or a ``(B,)``/``(B, 1)`` tensor (one duration per sample, as
    required when a batch mixes recordings with different sampling rates).
    Returned shape is ``()`` or ``(B, 1)`` so it broadcasts against ``(B, d)``
    latent tensors.
    """
    if isinstance(dt, torch.Tensor):
        value = dt.detach().to(device=device, dtype=dtype)
    else:
        value = torch.as_tensor(float(dt), device=device, dtype=dtype)
    if value.dim() == 0:
        if float(value) <= 0:
            raise ValueError("dt must be positive")
        return value
    if value.numel() != int(batch_size):
        raise ValueError(
            f"dt must be a scalar or carry one value per sample "
            f"({batch_size}); got {value.numel()}")
    if bool((value <= 0).any()):
        raise ValueError("dt must be positive")
    return value.reshape(-1, 1)


class GenericPoissonOperator(nn.Module):
    """Antisymmetric candidate bivector for GENERIC-inspired dynamics.

    ``L = A - A.T`` guarantees antisymmetry only.  A valid Poisson bracket
    additionally requires the differential Jacobi identity, which this
    parameterization does not enforce.

    The candidate is deterministic: dropout inside the operator map would make
    ``L(z)`` a random function of the state, so two evaluations at the same
    ``z`` would disagree and the antisymmetry/degeneracy properties would hold
    only for the sampled draw.  Regularization belongs in the loss, not in the
    definition of the operator.

    Note this parameterization carries a dense ``(d, d)`` matrix (1M entries at
    d=1024) and its state dependence is a broadcast rank-1 row shift;
    :class:`LowRankPoissonOperator` is the default and is O(d * rank).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 3,
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
            ])
        self.mlp = nn.Sequential(*layers)

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute antisymmetric candidate matrices for each batch row."""
        B, d = z.shape
        A_z = self.mlp(z)
        A_z = A_z.unsqueeze(1) + self.A.unsqueeze(0)
        return A_z - A_z.transpose(-2, -1)

    def compute_poisson_action(self, z: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Apply the candidate bivector to a gradient."""
        L_z = self.forward(z)
        return torch.bmm(L_z, grad.unsqueeze(-1)).squeeze(-1)

    def apply_action(self, z: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """NOTE: deliberately not named ``apply`` -- that would shadow
        ``nn.Module.apply``, which DeepSpeed (and any ``model.apply(fn)``
        traversal) relies on to walk the module tree."""
        """Action ``L(z) @ vector``. Assembles the dense operator."""
        return self.compute_poisson_action(z, vector)
class LowRankPoissonOperator(nn.Module):
    """Low-rank antisymmetric candidate with a symplectic-like factor.

    ``U @ J @ U.T`` guarantees antisymmetry and limits parameter count.  A
    state-dependent ``U`` does not guarantee the Jacobi identity; treat this
    as a geometric inductive bias, not a valid Poisson structure by theorem.

    Deterministic by construction (no dropout in the factor map): ``L(z)`` must
    be a function of ``z`` for antisymmetry and the degeneracy conditions to
    describe the operator rather than one sample of it.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        rank: int = 64,
        num_layers: int = 2,
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
            ])
        layers.append(nn.Linear(hidden_dim, hidden_dim * rank))
        self.mlp = nn.Sequential(*layers)

        half = rank // 2
        J = torch.zeros(rank, rank)
        for i in range(half):
            J[i, half + i] = 1.0
            J[half + i, i] = -1.0
        self.register_buffer("J", J)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute low-rank antisymmetric candidate matrices."""
        B, d = z.shape
        U = self.mlp(z).view(B, d, self.rank)
        return torch.bmm(
            torch.bmm(U, self.J.unsqueeze(0).expand(B, -1, -1)),
            U.transpose(-2, -1),
        )

    def compute_poisson_action(self, z: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank candidate without materializing its action."""
        B, d = z.shape
        U = self.mlp(z).view(B, d, self.rank)
        Utg = torch.bmm(U.transpose(-2, -1), grad.unsqueeze(-1))
        # J is a structural constant; follow the activation dtype so the bmm
        # does not mix a float32 buffer with a half-precision factor.
        J = self.J.to(dtype=Utg.dtype)
        J_Utg = torch.bmm(J.unsqueeze(0).expand(B, -1, -1), Utg)
        return torch.bmm(U, J_Utg).squeeze(-1)

    def apply_action(self, z: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Action ``L(z) @ vector`` in O(d * rank); never assembles ``L``."""
        return self.compute_poisson_action(z, vector)
class GenericMobilityOperator(nn.Module):
    """Diagonal nonnegative mobility candidate.

    The sigmoid output supplies a nonnegative dissipative direction.  It is
    not, by itself, a calibrated physical mobility tensor or proof of entropy
    production.
    """

    def __init__(self, hidden_dim: int = 1024, base_value: float = 0.1):
        super().__init__()
        if base_value <= 0:
            raise ValueError("base_value must be positive")
        self.hidden_dim = hidden_dim
        self.base_value = float(base_value)
        self.salience_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, z: torch.Tensor, salience: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the nonnegative diagonal mobility values ``M(z)``.

        A freshly initialized ``Linear`` maps an arbitrary latent to
        approximately zero, so ``sigmoid`` starts near 0.5 and the
        ``2 * base_value`` gain reproduces the historical constant mobility
        (``base_value``) at initialization.  The gate then learns a
        state-dependent mobility without a discontinuity at step 0.
        """
        source = z if salience is None else salience
        return (2.0 * self.base_value) * self.salience_gate(source)
class GenerickeDegeneracyProjection(nn.Module):
    """Pointwise residual projection for GENERIC-inspired constraints.

    The projection can reduce ``L @ grad_S`` and ``M @ grad_E`` at evaluated
    points.  It does not establish global GENERIC validity or the Jacobi
    identity.
    """

    def __init__(self, hidden_dim: int = 1024, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def project_L_orthogonal_to_gradS(self, L: torch.Tensor, grad_S: torch.Tensor) -> torch.Tensor:
        """Project ``L`` while preserving antisymmetry and ``L @ grad_S=0``."""
        d = L.shape[-1]
        norm_sq = torch.sum(grad_S ** 2, dim=-1, keepdim=True) + self.eps
        outer = grad_S.unsqueeze(-1) * grad_S.unsqueeze(-2)
        identity = torch.eye(d, dtype=L.dtype, device=L.device).expand_as(L)
        projector = identity - outer / norm_sq.unsqueeze(-1)
        # P is symmetric, so P L P remains antisymmetric whenever L is.
        return torch.bmm(torch.bmm(projector, L), projector)

    def project_M_action_onto_gradS(
        self,
        M_diag: torch.Tensor,
        grad_S: torch.Tensor,
        grad_E: torch.Tensor,
    ) -> torch.Tensor:
        """Return the applied dissipative force ``(P_E diag(M) P_E) grad_S``.

        The projection acts on the mobility *operator*, so the resulting
        effective mobility annihilates ``grad_E`` and the GENERIC degeneracy
        condition ``M @ grad_E = 0`` holds by construction.  See
        :func:`oettinger_mobility_action` for why the operator-level form is
        required instead of projecting the action.
        """
        return oettinger_mobility_action(M_diag, grad_E, grad_S, eps=self.eps)

    def forward(
        self,
        L: torch.Tensor,
        M_diag: torch.Tensor,
        grad_E: torch.Tensor,
        grad_S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pointwise projected candidate and mobility action."""
        L_proj = self.project_L_orthogonal_to_gradS(L, grad_S)
        Mv_proj = self.project_M_action_onto_gradS(M_diag, grad_S, grad_E)
        return L_proj, Mv_proj


class EnergyEntropyFields(nn.Module):
    """
    Energy E(z) and entropy S(z) potential fields.

    Both are modeled as neural networks that map latent state z to scalar potentials.
    When ``scale_gradients=True``, grad_E and grad_S stay true gradients of the
    scalar fields and are multiplied by learned positive scale factors.  A
    positive scalar times a gradient is still the gradient of a scaled
    potential, so ``L @ grad_S`` and ``M @ grad_E`` keep their GENERIC meaning.
    Earlier revisions unit-normalized the gradients before scaling, which
    replaced the gradient field with an arbitrary direction field and voided
    every degeneracy projection built on it.
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        normalize_gradients: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.normalize_gradients = normalize_gradients

        if normalize_gradients:
            self.energy_scale = nn.Parameter(torch.tensor(1.0))
            self.entropy_scale = nn.Parameter(torch.tensor(1.0))

        # The potential networks are deliberately deterministic: dropout would
        # make E(z) and S(z) random functions of the state, so grad_E would
        # change between two evaluations at the same z and the velocity field
        # would stop being an autonomous vector field.  Regularization for
        # these fields belongs in the loss, not in the potential definition.
        energy_layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            energy_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if i < num_layers - 1 else nn.Identity(),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
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

        ``grad_E`` and ``grad_S`` are always ``autograd`` derivatives of the
        scalar fields.  The optional learned scales multiply those true
        gradients (equivalently, they rescale the potentials), so
        ``L @ grad_S = 0`` and ``M @ grad_E = 0`` remain meaningful
        degeneracy conditions.

        The differentiation variable is detached from the caller's graph: the
        fields are evaluated at a fixed state, so second-order terms flow to
        the potential parameters but not back through the encoder history that
        produced ``z``.  The caller's tensor is never mutated in place.

        Autograd is enabled locally, so evaluating the physics under an outer
        ``torch.no_grad()`` (as the evaluation helpers do) still produces the
        gradients the velocity field needs instead of raising.
        """
        grad_E, grad_S, _, _ = self.compute_gradients_and_values(z)
        return grad_E, grad_S

    def compute_gradients_and_values(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """As :meth:`compute_gradients`, also returning the scalar potentials.

        ``E`` and ``S`` are needed by the stability monitor (energy-divergence
        rollback trigger) and are free to return here: they are already
        evaluated on the way to the gradients, so no second forward is needed.
        """
        with torch.enable_grad():
            grad_source = (
                z if z.requires_grad else z.detach().requires_grad_(True))
            E, S = self.forward(grad_source)
            grad_E = torch.autograd.grad(
                E.sum(), grad_source, create_graph=True)[0]
            grad_S = torch.autograd.grad(
                S.sum(), grad_source, create_graph=True)[0]

            if self.normalize_gradients:
                grad_E = grad_E * torch.abs(self.energy_scale)
                grad_S = grad_S * torch.abs(self.entropy_scale)

        return grad_E, grad_S, E.detach(), S.detach()



class VelocityBrain(nn.Module):
    """Core latent dynamics module with GENERIC-inspired structure.

    It predicts a velocity field from the latent state.  The algebraic
    candidates and pointwise projections are diagnostics/inductive biases,
    not a theorem that the learned process is physically GENERIC.
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_poisson_layers: int = 3,
        num_energy_layers: int = 3,
        dropout: float = 0.1,
        ou_tau: float = 0.02,
        noise_dim: int = 1024,
        apply_degeneracy_projection: bool = True,
        degeneracy_check_interval: int = 1,
        use_lowrank_poisson: bool = True,
        poisson_rank: int = 64,
        materialize_poisson: bool = False,
        perturbation_dim: Optional[int] = None,
        control_gating: bool = True,
        latent_dt: Optional[float] = None,
    ):
        super().__init__()
        if latent_dt is not None and latent_dt <= 0:
            raise ValueError("latent_dt must be positive")
        self.hidden_dim = hidden_dim
        self.integration_dt = float(latent_dt) if latent_dt is not None else 1.0
        self.apply_degeneracy_projection = apply_degeneracy_projection
        self.degeneracy_check_interval = degeneracy_check_interval
        # The Poisson operator is applied implicitly (action on a vector) by
        # default.  Materialising L is O(d^2) memory and turns the degeneracy
        # projection into two d x d matmuls; keep it available only for
        # diagnostics that genuinely need the matrix (e.g. the Jacobi check).
        self.materialize_poisson = bool(materialize_poisson)
        self.step_counter = 0
        self._poisson_enabled = True
        self._degeneracy_enabled = True

        if use_lowrank_poisson:
            self.poisson_op = LowRankPoissonOperator(
                hidden_dim, rank=poisson_rank, num_layers=min(num_poisson_layers, 2),
            )
        else:
            self.poisson_op = GenericPoissonOperator(hidden_dim, num_poisson_layers)
        kda_taus = None
        if latent_dt is not None:
            kda_taus = tuple(
                -1.0 / math.log(alpha)
                for alpha in (0.1, 0.5, 0.9))
        self.mt_kda = MultiTimeScaleKDA(
            hidden_dim, time_constants=kda_taus, stateful=True)
        self.energy_entropy = EnergyEntropyFields(hidden_dim, num_energy_layers)
        self.mobility_op = GenericMobilityOperator(hidden_dim)

        # The OU integrator shares the latent clock by construction, so the
        # noise time constant means the same thing with or without an
        # explicit latent_dt.
        self.ou_noise = OUStructuredNoise(
            hidden_dim, ou_tau, dt=self.integration_dt)

        self.arousal_vector = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.arousal_scale = nn.Parameter(torch.tensor(0.1))

        if apply_degeneracy_projection:
            self.degeneracy_proj = GenerickeDegeneracyProjection(hidden_dim)

        if perturbation_dim is not None:
            if perturbation_dim <= 0:
                raise ValueError("perturbation_dim must be positive")
            self.perturbation_dim = int(perturbation_dim)
            # Bias-free input layer + zero-init readout: u = 0 is an
            # exact no-op for any later parameter state.
            self.perturbation_map = nn.Sequential(
                nn.Linear(self.perturbation_dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Zero-init the control readout: zero control must be a no-op
            # and conditioning starts as an identity modification.
            last = self.perturbation_map[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            # Input-conditioned gates (P1): multiply mobility/arousal by
            # 1 + tanh(g(u)); zero-init g keeps the unforced model intact.
            self.control_gating = bool(control_gating)
            if self.control_gating:
                self.gate_map = nn.Sequential(
                    nn.Linear(self.perturbation_dim, 32, bias=False),
                    nn.SiLU(),
                    nn.Linear(32, 2),
                )
                gate_last = self.gate_map[-1]
                nn.init.zeros_(gate_last.weight)
                nn.init.zeros_(gate_last.bias)
            else:
                self.gate_map = None
        else:
            self.perturbation_dim = None
            self.perturbation_map = None
            self.control_gating = False
            self.gate_map = None

    def forward(
        self,
        z: torch.Tensor,
        salience: Optional[torch.Tensor] = None,
        noise_state: Optional[torch.Tensor] = None,
        perturbation: Optional[torch.Tensor] = None,
        apply_noise: bool = False,
        dt=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute velocity field delta_z from current state z.

        Args:
            z: (B, d) current latent state
            salience: (B, d) optional salience activation for mobility
            perturbation: (B, u) optional exogenous intervention/control input
            apply_noise: whether to apply OU noise
            dt: Optional physical duration of this latent step (seconds;
                scalar or one value per sample). Defaults to the module's
                ``integration_dt``. The MT-KDA time constants and the OU
                transition are evaluated with this duration, so their
                seconds semantics hold for any sampling rate.
            dict with:
                - delta_z: (B, d) predicted velocity
                - new_noise_state: (B, d) updated noise state
                - metrics: dict of diagnostics
        """
        grad_E, grad_S, E_value, S_value = (
            self.energy_entropy.compute_gradients_and_values(z))

        poisson_enabled = getattr(self, "_poisson_enabled", True)
        project = (
            self.apply_degeneracy_projection
            and self.step_counter % self.degeneracy_check_interval == 0
            and getattr(self, "_degeneracy_enabled", True))
        eps = getattr(getattr(self, "degeneracy_proj", None), "eps", 1e-6)

        L_z = None
        if self.materialize_poisson:
            L_z = self.poisson_op(z)

        # Conservative term.  The degeneracy projector commutes with the
        # action -- (P_S L P_S) grad_E == P_S (L (P_S grad_E)) -- so neither L
        # nor P_S is ever assembled.  The dense alternative builds L (B, d, d)
        # and computes P L P, i.e. two d x d matmuls (2 d^3 flops per sample),
        # to produce this same vector.
        if poisson_enabled:
            if project:
                poisson_term = project_along(
                    grad_S,
                    self.poisson_op.apply_action(
                        z, project_along(grad_S, grad_E, eps)),
                    eps)
            else:
                poisson_term = self.poisson_op.apply_action(z, grad_E)
        else:
            poisson_term = torch.zeros_like(grad_E)

        if self.mobility_op is not None:
            M_diag = self.mobility_op(z, salience)
        else:
            M_diag = torch.ones_like(grad_S) * 0.1

        if project:
            # Oettinger projection on the mobility *operator*: the effective
            # mobility annihilates grad_E exactly, so M grad_E = 0 holds by
            # construction rather than by penalty.
            mobility_term = oettinger_mobility_action(
                M_diag, grad_E, grad_S, eps)
        else:
            mobility_term = M_diag * grad_S

        control_term = torch.zeros_like(z)
        if perturbation is not None:
            if self.perturbation_map is None:
                raise ValueError(
                    "perturbation_dim must be configured before passing perturbation")
            if perturbation.shape != (z.shape[0], self.perturbation_dim):
                raise ValueError(
                    "perturbation must have shape (B, perturbation_dim)")
            control_term = self.perturbation_map(
                perturbation.to(dtype=z.dtype, device=z.device))

        arousal_term = self.arousal_scale * self.arousal_vector

        if perturbation is not None and self.gate_map is not None:
            gates = 1.0 + torch.tanh(
                self.gate_map(perturbation.to(dtype=z.dtype, device=z.device)))
            mobility_term = mobility_term * gates[:, 0:1]
            arousal_term = arousal_term * gates[:, 1:2]

        delta_z = poisson_term + mobility_term + arousal_term + control_term

        # Apply multi-time-scale KDA decay
        delta_z = self.mt_kda(delta_z, dt=self.integration_dt if dt is None else dt)

        new_noise_state = None
        if apply_noise:
            delta_z, new_noise_state = self.ou_noise(
                delta_z, noise_state, dt=dt)

        # Degeneracy residuals.  With the projectors active both conditions
        # hold by construction (P_S grad_S == 0 and P_E grad_E == 0), so the
        # residuals are reported as zero; the mobility one is still evaluated
        # through the helper as a cheap check that the projector is correct.
        if poisson_enabled and not project:
            L_grad_S = self.poisson_op.apply_action(z, grad_S)
        else:
            L_grad_S = torch.zeros_like(grad_S)
        L_grad_S_norm = torch.norm(L_grad_S, dim=-1).mean()
        if project:
            M_grad_E = oettinger_mobility_action(M_diag, grad_E, grad_E, eps)
        else:
            M_grad_E = M_diag * grad_E
        M_grad_E_norm = torch.norm(M_grad_E, dim=-1).mean()

        if L_z is not None and project and poisson_enabled:
            L_z = self.degeneracy_proj.project_L_orthogonal_to_gradS(
                L_z, grad_S)

        metrics = {
            "||L_z||_F": (torch.norm(L_z, p="fro") / self.hidden_dim
                          if L_z is not None else torch.zeros(())),
            "||grad_E||": torch.norm(grad_E, dim=-1).mean(),
            "||grad_S||": torch.norm(grad_S, dim=-1).mean(),
            "||arousal||": torch.norm(
                self.arousal_vector, dim=-1).mean(),
            "control_norm": torch.norm(
                control_term, dim=-1).mean(),
            "degeneracy_L_grad_S": L_grad_S_norm,
        }

        self.step_counter += 1
        return {
            "delta_z": delta_z,
            "new_noise_state": new_noise_state,
            "grad_E": grad_E,
            "grad_S": grad_S,
            "E": E_value.mean().detach(),
            "S": S_value.mean().detach(),
            "L_z": L_z,
            "M_diag": M_diag,
            "L_grad_S": L_grad_S,
            "M_grad_E": M_grad_E,
            "degeneracy_L_grad_S_norm": L_grad_S_norm,
            "degeneracy_M_grad_E_norm": M_grad_E_norm,
            "control_term": control_term,
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

    Three parallel exponential moving averages of the incoming velocity, mixed
    by a learned gate.  With ``time_constants`` set, the decay per call is
    ``exp(-dt / tau)`` with ``dt`` and ``tau`` in the same physical unit, so
    the three branches are literal fast / medium / slow time constants of the
    latent process.

    Without ``time_constants`` the historical per-step ``alphas`` are used
    directly (decay ``alpha`` per call, no time unit implied).

    These are latent-state filters: they are *not* synaptic, working-memory,
    or hemodynamic time constants, and they do not model the biophysical
    processes those names refer to.  ``decay`` values derived from the default
    alphas are ``{0.43, 1.44, 9.49}`` in units of ``1 / (-ln alpha)`` steps,
    which at 256 Hz latents is seconds — three orders of magnitude slower than
    a synaptic membrane time constant.

    The default is stateless across calls so unrelated batches cannot leak
    into one another. ``stateful=True`` retains state within an explicit
    sequence; callers must reset it at sequence boundaries.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        alphas: Tuple[float, float, float] = (0.1, 0.5, 0.9),
        time_constants: Optional[Tuple[float, float, float]] = None,
        stateful: bool = False,
    ):
        super().__init__()
        if len(alphas) != 3 or any(
            not 0 <= alpha < 1 for alpha in alphas):
            raise ValueError("alphas must contain three values in [0, 1)")
        if time_constants is not None and (
            len(time_constants) != 3 or any(
                tau <= 0 for tau in time_constants)):
            raise ValueError("time_constants must contain three positives")
        self.alphas = tuple(float(alpha) for alpha in alphas)
        self.time_constants = time_constants
        self.stateful = bool(stateful)
        self.hidden_dim = hidden_dim

        self.register_buffer("state_a", torch.zeros(1, hidden_dim))
        self.register_buffer("state_b", torch.zeros(1, hidden_dim))
        self.register_buffer("state_c", torch.zeros(1, hidden_dim))

        self.mix_weights = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1),
        )

    def reset_state(self, batch_size: Optional[int] = None) -> None:
        """Clear all KDA memory, optionally preparing a batch-sized state."""
        size = 1 if batch_size is None else int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        for name in ("state_a", "state_b", "state_c"):
            state = getattr(self, name)
            state.data = torch.zeros(
                size, self.hidden_dim, device=state.device, dtype=state.dtype)

    def _state_for_batch(
        self, state: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if state.shape[0] == batch_size:
            return state
        # A batch-size mismatch marks a new batch, not a request to broadcast
        # state from an unrelated sequence.
        return torch.zeros(
            batch_size, self.hidden_dim,
            device=state.device, dtype=state.dtype)

    def forward(
        self, delta_z: torch.Tensor, dt: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply multi-time-scale decay to velocity input.

        When ``time_constants`` are configured, ``dt`` is in the same
        physical time unit as those constants.  Without them, the historical
        per-step ``alphas`` remain the default API behavior.
        """
        batch_size = delta_z.shape[0]
        state_a = self._state_for_batch(self.state_a, batch_size)
        state_b = self._state_for_batch(self.state_b, batch_size)
        state_c = self._state_for_batch(self.state_c, batch_size)

        if self.time_constants is not None and dt is not None:
            step = as_step_dt(dt, batch_size, delta_z.device, delta_z.dtype)
            taus = torch.as_tensor(
                self.time_constants, device=delta_z.device,
                dtype=delta_z.dtype).reshape(-1, 1, 1)
            decays = torch.exp(-step.reshape(1, *step.shape) / taus)
        else:
            decays = torch.as_tensor(
                self.alphas, device=delta_z.device,
                dtype=delta_z.dtype).reshape(-1, 1, 1)

        state_a = decays[0] * state_a + (1.0 - decays[0]) * delta_z
        state_b = decays[1] * state_b + (1.0 - decays[1]) * delta_z
        state_c = decays[2] * state_c + (1.0 - decays[2]) * delta_z

        combined = torch.cat([state_a, state_b, state_c], dim=-1)
        weights = self.mix_weights(combined)
        output = (
            weights[:, 0:1] * state_a
            + weights[:, 1:2] * state_b
            + weights[:, 2:3] * state_c)

        if self.stateful:
            self.state_a.data = state_a.detach()
            self.state_b.data = state_b.detach()
            self.state_c.data = state_c.detach()
        return output


class WienerHomeostat(nn.Module):
    """Adaptive stochastic-noise regulation using explicit proxies.

    The controller combines a dissipative-structure proxy and router entropy
    to keep the noise gain near user-supplied calibration targets.  These
    targets are control setpoints, not universal criticality observables, and
    the resulting ``D_eff`` is not a thermodynamic entropy-production
    estimate.

    Per-component gains are an initialization heuristic based on observed
    variance ratios.  Because the OU state passes through a learned,
    nonlinear projection, generated-output statistics must be recalibrated
    on held-out rollouts before claiming marginal preservation.

    Monitoring is read-only unless the trainer explicitly enables active
    noise regulation.
    """

    def __init__(
        self,
        D_0: float = 1.0,
        target_dissipative_proxy: float = 1.0,
        target_entropy: float = 1.0,
        clamp_min: float = 0.3,
        clamp_max: float = 3.0,
        ema_decay: float = 0.99,
        clip_min: float = 1e-4,
        dim: Optional[int] = None,
    ):
        super().__init__()
        self.D_0 = D_0
        self.target_dissipative_proxy = target_dissipative_proxy
        self.target_entropy = target_entropy
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.ema_decay = ema_decay
        self.clip_min = clip_min
        self.dim = dim
        # history (logging-only) cadence: append every N updates to avoid a
        # CPU-GPU sync on every training step
        self.history_interval = 10

        self.register_buffer("D_eff", torch.tensor(D_0, dtype=torch.float32))
        self.register_buffer(
            "ema_dissipative_proxy",
            torch.tensor(target_dissipative_proxy, dtype=torch.float32))
        self.register_buffer("ema_entropy", torch.tensor(target_entropy, dtype=torch.float32))
        self.register_buffer("health_ema", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        # per-component relative gain (dim,), all ones until calibrated.
        # Scalar D_eff * dim_gain reproduces the observed per-component
        # variance spread (C. elegans occurrence components span 2.7-4.4x);
        # a scalar alone cannot match such marginals (2026-09 bake-off).
        if dim is not None and dim > 0:
            self.register_buffer("dim_gain", torch.ones(dim, dtype=torch.float32))
        else:
            self.register_buffer("dim_gain", torch.tensor([]))

        self.D_history: List[float] = []
        self.health_history: List[float] = []

    def calibrate_dim_gain(self, windows: torch.Tensor) -> Dict[str, float]:
        """Initialize relative gains from observed component variances.

        This is only a starting-point calibration.  The learned OU
        projection and nonlinear saturation mean that it does not guarantee
        generated-output variance matching; validate the resulting gains on
        held-out rollouts.
        """
        if windows.shape[-1] != self.dim_gain.numel():
            raise ValueError(
                f"window dim {windows.shape[-1]} != homeostat dim {self.dim_gain.numel()}")
        var = windows.detach().float().var(dim=0)
        mean_var = var.mean().clamp_min(safe_epsilon(var, 1e-12))
        gain = torch.sqrt(
            var.clamp_min(safe_epsilon(var, 1e-12)) / mean_var).clamp(0.1, 10.0)
        self.dim_gain.copy_(gain)
        return {
            "gain_min": float(gain.min()),
            "gain_max": float(gain.max()),
            "gain_mean": float(gain.mean()),
        }

    def update(
        self,
        current_dissipative_proxy: float,
        router_entropy: float,
    ) -> Dict[str, Any]:
        """Update the noise controller from explicit control proxies.

        ``current_dissipative_proxy`` is the model's algebraic
        ``grad_S · M · grad_S`` proxy or another documented scalar.  It is not
        a stochastic entropy-production estimate.
        """
        proxy = torch.as_tensor(
            max(current_dissipative_proxy, self.clip_min),
            dtype=self.ema_dissipative_proxy.dtype,
            device=self.ema_dissipative_proxy.device,
        )
        entropy = torch.as_tensor(
            max(router_entropy, self.clip_min),
            dtype=self.ema_dissipative_proxy.dtype,
            device=self.ema_dissipative_proxy.device,
        )

        self.ema_dissipative_proxy.mul_(self.ema_decay).add_(
            (1.0 - self.ema_decay) * proxy)
        self.ema_entropy.mul_(self.ema_decay).add_(
            (1.0 - self.ema_decay) * entropy)

        target_proxy = max(self.target_dissipative_proxy, self.clip_min)
        target_entropy = max(self.target_entropy, self.clip_min)
        proxy_ratio = self.ema_dissipative_proxy / target_proxy
        entropy_ratio = self.ema_entropy / target_entropy
        health = 0.5 * proxy_ratio + 0.5 * entropy_ratio

        self.health_ema.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * health)
        D_eff = self.D_0 * torch.clamp(
            self.health_ema, self.clamp_min, self.clamp_max)
        self.D_eff.copy_(D_eff)
        self.step_counter.add_(1)

        if int(self.step_counter.item()) % self.history_interval == 0:
            self.D_history.append(float(D_eff.item()))
            self.health_history.append(float(self.health_ema.item()))
            if len(self.D_history) > 10000:
                self.D_history = self.D_history[-5000:]
                self.health_history = self.health_history[-5000:]

        out: Dict[str, Any] = {
            "D_eff": float(D_eff.item()),
            "health": float(self.health_ema.item()),
            "dissipative_proxy_ratio": float(proxy_ratio.item()),
            "entropy_ratio": float(entropy_ratio.item()),
            "ema_dissipative_proxy": float(self.ema_dissipative_proxy.item()),
            "ema_entropy": float(self.ema_entropy.item()),
        }
        if self.dim_gain.numel() > 0:
            out["D_eff_vec"] = D_eff * self.dim_gain
        return out

    def get_D(self) -> float:
        return self.D_eff.item()

    def get_report(self) -> Dict:
        return {
            "D_eff": self.D_eff.item(),
            "D_0": self.D_0,
            "health": self.health_ema.item(),
            "ema_dissipative_proxy": self.ema_dissipative_proxy.item(),
            "ema_entropy": self.ema_entropy.item(),
            "target_dissipative_proxy": self.target_dissipative_proxy,
            "target_entropy": self.target_entropy,
            "steps": self.step_counter.item(),
            "D_mean": float(np.mean(self.D_history)) if self.D_history else self.D_0,
            "health_mean": float(np.mean(self.health_history)) if self.health_history else 1.0,
        }


if __name__ == "__main__":
    print("Testing VelocityBrain...")
    vb = VelocityBrain(hidden_dim= 1024)
    z = torch.randn(4, 2048)
    salience = torch.randn(4, 2048) * 0.1
    result = vb(z, salience, apply_noise=False)
    print(f"  delta_z shape: {result['delta_z'].shape}")
    print(f"  ||delta_z||: {torch.norm(result['delta_z'], dim=-1).mean().item():.4f}")
    print("  metrics:", {k: f"{v.mean().item():.4f}" for k, v in result['metrics'].items()})

    print("\nTesting MultiTimeScaleKDA...")
    kda = MultiTimeScaleKDA(hidden_dim= 1024)
    out = kda(result['delta_z'])
    print(f"  KDA output shape: {out.shape}")
    print("All tests passed!")