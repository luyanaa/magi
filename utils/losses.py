"""
Loss Functions for Brain MoE-PINN Training.

Implements all loss terms:
- Reconstruction losses (EEG, fMRI)
- Velocity smoothness constraint
- GENERIC physical constraints (degeneracy, Jacobi)
- MoE load balancing
- Hebbian regularization
- Grassmannian/orthogonal attractor regularization
- NSP (Neural Signal Processing) auxiliary losses
- Cross-modal alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ReconstructionLoss(nn.Module):
    """EEG and fMRI reconstruction losses."""

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward_eeg(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, T) predicted EEG
            target: (B, C, T) target EEG
        """
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=1.0)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward_fmri(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, R, T_patches) predicted fMRI
            target: (B, R, T_patches) target fMRI
        """
        return self.forward_eeg(pred, target)


class VelocitySmoothnessLoss(nn.Module):
    """
    Penalizes sudden jumps in velocity field.

    Encourages smooth trajectories in latent space, consistent with
    continuous neural dynamics.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, delta_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_z: (B, d) velocity vectors
        """
        if delta_z.shape[0] < 2:
            return torch.tensor(0.0, device=delta_z.device)

        diff = delta_z[1:] - delta_z[:-1]
        smoothness = torch.norm(diff, p=1, dim=-1).mean()
        return self.alpha * smoothness


class GenerickeConstraintLoss(nn.Module):
    """
    Enforces GENERIC degeneracy conditions.

    Condition 1: L(z) @ grad_S(z) ≈ 0 (Poisson bracket orthogonal to entropy)
    Condition 2: M(z) @ grad_E(z) ≈ 0 (Mobility orthogonal to energy)

    Also computes the Jacobi identity violation for regularization.
    """

    def __init__(self, weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(
        self,
        L_z: torch.Tensor,
        grad_S: torch.Tensor,
        M_diag: torch.Tensor,
        grad_E: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            L_z: (d, d) antisymmetric Poisson matrix
            grad_S: (B, d) entropy gradient
            M_diag: (B, d) diagonal mobility
            grad_E: (B, d) energy gradient
        Returns:
            total_loss, metrics dict
        """
        B = grad_S.shape[0]
        device = grad_S.device

        L_grad_S = torch.bmm(L_z.unsqueeze(0).expand(B, -1, -1), grad_S.unsqueeze(-1)).squeeze(-1)
        condition1_violation = torch.norm(L_grad_S, dim=-1).mean()

        M_grad_E = M_diag * grad_E
        condition2_violation = torch.norm(M_grad_E, dim=-1).mean()

        metrics = {
            "L_grad_S_norm": condition1_violation.item(),
            "M_grad_E_norm": condition2_violation.item(),
        }

        total_loss = self.weight * (condition1_violation + condition2_violation)
        return total_loss, metrics


class MoELoadBalancingLoss(nn.Module):
    """
    Encourages balanced expert utilization.

    Based on Switch Transformer auxiliary load balancing loss.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        gate_weights: torch.Tensor,
        selected_indices: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            gate_weights: (B, top_k) normalized gate weights
            selected_indices: (B, top_k) selected expert indices
            num_experts: total number of experts
        Returns:
            loss, metrics
        """
        B, top_k = gate_weights.shape

        expert_counts = torch.zeros(num_experts, device=gate_weights.device)
        for idx in selected_indices.view(-1):
            expert_counts[idx] += 1

        expert_fraction = expert_counts / (B * top_k)

        load_imbalance = torch.var(expert_fraction)

        router_probs = gate_weights.mean(dim=0)
        router_entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum()

        metrics = {
            "expert_util_variance": load_imbalance.item(),
            "router_entropy": router_entropy.item(),
        }

        return self.alpha * load_imbalance, metrics


class GrassmannianRegularization(nn.Module):
    """
    Orthogonal attractor regularization from FEP (Spisak & Friston 2025).

    Minimizes pairwise inner products between normalized expert output directions,
    encouraging orthogonal attractor representations on Grassmannian manifold.
    """

    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight

    def forward(self, expert_outputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            expert_outputs: (B, num_experts, d) output from each expert
        Returns:
            loss, metrics
        """
        B, num_experts, d = expert_outputs.shape

        outputs_norm = expert_outputs / (expert_outputs.norm(dim=-1, keepdim=True) + 1e-8)

        gram = torch.matmul(outputs_norm, outputs_norm.transpose(-2, -1))

        mask = torch.eye(num_experts, device=gram.device).unsqueeze(0)
        gram_off_diag = (1 - mask) * gram

        orthogonality = (gram_off_diag ** 2).sum() / (num_experts * (num_experts - 1))

        metrics = {
            "grassmannian_violation": orthogonality.item(),
        }

        return self.weight * orthogonality, metrics


class JacobiRegularization(nn.Module):
    """
    Jacobi identity regularization for valid Poisson bracket.

    For L(z) to define a valid Poisson bracket, it must satisfy:
    {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0

    This is approximated via Monte Carlo sampling of random index triplets.
    """

    def __init__(self, weight: float = 0.001, num_samples: int = 128):
        super().__init__()
        self.weight = weight
        self.num_samples = num_samples

    def _compute_jacobi_violation(
        self,
        L: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobi identity violation for sampled triplets.

        Args:
            L: (d, d) antisymmetric matrix
            indices: (num_samples, 3) random index triplets
        """
        d = L.shape[0]
        violation = torch.zeros(self.num_samples, device=L.device)

        for s in range(self.num_samples):
            i, j, k = indices[s]

            L_ij = L[i, j]
            L_jk = L[j, k]
            L_ki = L[k, i]

            L_ik = L[i, k]
            L_kj = L[k, j]
            L_ji = L[j, i]

            L_ijk = L_ij * L_jk * L_ki
            L_kij = L_kj * L_ij * L_ik
            L_jki = L_jk * L_ki * L_ji

            jacobi = L_ijk + L_kij + L_jki
            violation[s] = jacobi ** 2

        return violation.mean()

    def forward(self, L_z: torch.Tensor, d: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            L_z: (d, d) Poisson matrix
            d: dimension
        Returns:
            loss, metrics
        """
        indices = torch.randint(0, d, (self.num_samples, 3), device=L_z.device)

        violation = self._compute_jacobi_violation(L_z, indices)

        metrics = {
            "jacobi_violation": violation.item(),
        }

        return self.weight * violation, metrics


class HebbianRegularization(nn.Module):
    """
    Regularization for Hebbian memory system.

    Encourages stable fixed points and prevents runaway weight growth.
    """

    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight

    def forward(self, hebbian_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            hebbian_weights: (d, d) Hebbian weight matrix
        Returns:
            loss, metrics
        """
        d = hebbian_weights.shape[0]

        weight_norm = torch.norm(hebbian_weights, p="fro")
        spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(hebbian_weights.float())))

        stability_loss = F.relu(spectral_radius - 0.95)

        metrics = {
            "hebbian_weight_norm": weight_norm.item(),
            "hebbian_spectral_radius": spectral_radius.item(),
        }

        return self.weight * stability_loss, metrics


class NSPLoss(nn.Module):
    """
    Neural Signal Processing auxiliary losses from DeeperBrain.

    Includes:
    - Spectral power matching (delta/theta/alpha/beta/gamma)
    - Functional connectivity matching
    - Cross-frequency coupling
    - Complexity metrics (entropy, LZ complexity)
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def compute_band_powers(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Compute relative band powers using Welch's method approximation.

        Returns tensor of shape (B, 5) for delta/theta/alpha/beta/gamma.
        """
        B, C, T = eeg.shape
        device = eeg.device

        band_powers = torch.zeros(B, 5, device=device)

        for b in range(B):
            eeg_b = eeg[b]
            # Temporary accumulator: (channels, bands)
            channel_bands = torch.zeros(C, 5, device=device)
            for ch in range(C):
                x = eeg_b[ch]
                fft_vals = torch.fft.rfft(x)
                psd = torch.abs(fft_vals) ** 2

                freqs = torch.fft.rfftfreq(T, d=1.0/256)

                delta_mask = (freqs >= 1) & (freqs < 4)
                theta_mask = (freqs >= 4) & (freqs < 8)
                alpha_mask = (freqs >= 8) & (freqs < 13)
                beta_mask = (freqs >= 13) & (freqs < 30)
                gamma_mask = (freqs >= 30) & (freqs < 100)

                total_power = psd.sum() + 1e-8

                channel_bands[ch, 0] = psd[delta_mask].sum() / total_power
                channel_bands[ch, 1] = psd[theta_mask].sum() / total_power
                channel_bands[ch, 2] = psd[alpha_mask].sum() / total_power
                channel_bands[ch, 3] = psd[beta_mask].sum() / total_power
                channel_bands[ch, 4] = psd[gamma_mask].sum() / total_power

            band_powers[b] = channel_bands.mean(dim=0)

        return band_powers

    def forward(
        self,
        pred_eeg: torch.Tensor,
        target_eeg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_eeg: (B, C, T) predicted EEG
            target_eeg: (B, C, T) target EEG
        Returns:
            loss, metrics
        """
        B = pred_eeg.shape[0]
        device = pred_eeg.device

        pred_bands = self.compute_band_powers(pred_eeg)
        target_bands = self.compute_band_powers(target_eeg)

        band_loss = F.mse_loss(pred_bands, target_bands)

        metrics = {
            "band_power_loss": band_loss.item(),
        }

        return self.weight * band_loss, metrics


class CrossModalAlignmentLoss(nn.Module):
    """
    Aligns EEG and fMRI representations via hub tokens.

    Encourages:
    - EEG and fMRI hub tokens to be similar when data is aligned
    - Orthogonal when data is from different subjects/sessions
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        hub_eeg: torch.Tensor,
        hub_fmri: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            hub_eeg: (B, d) EEG hub token
            hub_fmri: (B, d) fMRI hub token
            labels: (B,) optional labels (1=aligned, 0=unaligned)
        Returns:
            loss, metrics
        """
        hub_eeg_norm = hub_eeg / (hub_eeg.norm(dim=-1, keepdim=True) + 1e-8)
        hub_fmri_norm = hub_fmri / (hub_fmri.norm(dim=-1, keepdim=True) + 1e-8)

        similarity = (hub_eeg_norm * hub_fmri_norm).sum(dim=-1)

        if labels is not None:
            aligned_mask = labels == 1
            loss = F.mse_loss(similarity[aligned_mask], torch.ones_like(similarity[aligned_mask]))
            if (~aligned_mask).sum() > 0:
                loss += 0.1 * F.mse_loss(similarity[~aligned_mask], torch.zeros_like(similarity[~aligned_mask]))
        else:
            loss = -similarity.mean()

        metrics = {
            "cross_modal_similarity": similarity.mean().item(),
        }

        return self.weight * loss, metrics


class DissipationLoss(nn.Module):
    """
    Dissipation constraint: non-negative entropy production + GENERIC degeneracy.

    L_dissip = max(0, -dS/dt) + ||M(z) @ grad_E||^2
    where dS/dt ≈ grad_S · delta_z (entropy change rate).
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        delta_z: torch.Tensor,
        grad_S: torch.Tensor,
        M_diag: torch.Tensor,
        grad_E: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dS_dt = (grad_S * delta_z).sum(dim=-1)
        entropy_production = F.relu(-dS_dt).mean()

        M_grad_E = M_diag * grad_E
        degeneracy_violation = (M_grad_E ** 2).sum(dim=-1).mean()

        loss = entropy_production + degeneracy_violation
        metrics = {
            "entropy_production": entropy_production.item(),
            "degeneracy_violation": degeneracy_violation.item(),
        }
        return self.weight * loss, metrics


class EPRLoss(nn.Module):
    """
    Entropy Production Rate (EPR) constraint.

    sigma = ||v||^2 / D, ensure >= 0 (Second Law of Thermodynamics).
    L_EPR = max(0, -sigma) acts as a hard constraint.
    """

    def __init__(self, weight: float = 0.05, D: float = 1.0):
        super().__init__()
        self.weight = weight
        self.D = D

    def forward(self, delta_z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        sigma = (delta_z ** 2).sum(dim=-1) / (self.D + 1e-8)
        epr_negative = F.relu(-sigma).mean()

        metrics = {
            "epr_mean": sigma.mean().item(),
            "epr_negative_ratio": (sigma < 0).float().mean().item(),
        }
        return self.weight * epr_negative, metrics


class SpectralSlopeLoss(nn.Module):
    """
    PSD spectral slope constraint: target 1/f noise (slope ≈ 1.0).

    Only computed when a sequence of z states is available.
    L_spectrum = ||slope(z-PSD) - 1.0||^2
    """

    def __init__(self, weight: float = 0.01, target_slope: float = 1.0):
        super().__init__()
        self.weight = weight
        self.target_slope = target_slope

    def forward(self, z_sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_sequence: (B, T, d) sequence of latent states
        """
        B, T, d = z_sequence.shape
        slopes = []
        for b in range(B):
            for i in range(min(d, 16)):  # sample subset of dimensions for efficiency
                x = z_sequence[b, :, i]
                fft_vals = torch.fft.rfft(x)
                psd = torch.abs(fft_vals) ** 2 + 1e-8
                freqs = torch.fft.rfftfreq(T, d=1.0)
                # Avoid DC component
                valid = freqs > 0
                if valid.sum() < 2:
                    continue
                log_psd = torch.log(psd[valid])
                log_freq = torch.log(freqs[valid])
                # Simple linear regression for slope
                slope = ((log_freq - log_freq.mean()) * (log_psd - log_psd.mean())).sum() / \
                        ((log_freq - log_freq.mean()) ** 2).sum()
                slopes.append(slope)

        if len(slopes) == 0:
            return torch.tensor(0.0, device=z_sequence.device), {"psd_slope": 0.0}

        avg_slope = torch.stack(slopes).mean()
        loss = (avg_slope - self.target_slope) ** 2

        metrics = {
            "psd_slope": avg_slope.item(),
        }
        return self.weight * loss, metrics


class LatentSparsityLoss(nn.Module):
    """
    Latent sparsity regularization via Tsallis-type penalty (q=2).

    Penalizes highly concentrated latent representations by computing:
      H_q = -1/(q-1) * (1 - sum_i p_i^q)
    where p_i = softmax(z_i) over latent dimensions.

    NOTE: This is NOT Tsallis entropy in the physical sense — softmax(z)
    does not define a probability distribution in information-theoretic space.
    The formula is a sparsity/kurtosis penalty that penalizes "peaked" latent
    vectors. Higher values = more uniform (less sparse) latent representation.
    Lower values = more concentrated/peaked (potentially collapsing) representation.

    Renamed from TsallisEntropyLoss to avoid false physics claims.
    """

    def __init__(self, weight: float = 0.05, q: float = 2.0):
        super().__init__()
        self.weight = weight
        self.q = q

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        p = F.softmax(z, dim=-1)
        tsallis = -(1.0 - (p ** self.q).sum(dim=-1)) / (self.q - 1.0)
        loss = -tsallis.mean()

        metrics = {
            "latent_sparsity": tsallis.mean().item(),
        }
        return self.weight * loss, metrics


class KSEntropyLoss(nn.Module):
    """
    KS entropy approximation via local Lyapunov exponent estimate.

    L_KS = -1/dt * <log ||J_v · delta_z||>
    where J_v is approximated by finite differences.
    Computed only when delta_z sequence is available.
    """

    def __init__(self, weight: float = 0.01, dt: float = 1.0):
        super().__init__()
        self.weight = weight
        self.dt = dt

    def forward(self, delta_z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            delta_z: (B, T, d) velocity sequence
        """
        if delta_z.dim() < 3 or delta_z.shape[1] < 2:
            return torch.tensor(0.0, device=delta_z.device), {"ks_entropy": 0.0}

        B, T, d = delta_z.shape
        # Finite-difference Jacobian approximation: J_v[i,j] ≈ d(v_i)/d(z_j)
        # Using velocity differences as proxy: ||delta_z[t+1] - delta_z[t]||
        dv = delta_z[:, 1:] - delta_z[:, :-1]
        norm_dv = torch.norm(dv, p=2, dim=-1) + 1e-8
        ks = -torch.log(norm_dv).mean() / self.dt

        metrics = {
            "ks_entropy": ks.item(),
        }
        return self.weight * F.relu(-ks), metrics  # penalize negative KS (explosive)


class ActionLoss(nn.Module):
    """
    Active inference action loss: minimize Expected Free Energy (EFE).

    L_action = E_{q(a)}[EFE(a)]
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, efe: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = efe.mean()
        metrics = {
            "efe_mean": loss.item(),
        }
        return self.weight * loss, metrics


class ReplayLoss(nn.Module):
    """
    Memory replay loss: imagined trajectories should match engram traces.

    L_replay = ||z_imagined - z_engram||^2
    """

    def __init__(self, weight: float = 0.02):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        z_imagined: torch.Tensor,
        z_engram: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = F.mse_loss(z_imagined, z_engram)
        metrics = {
            "replay_mse": loss.item(),
        }
        return self.weight * loss, metrics


class ModalityAlignmentLoss(nn.Module):
    """
    Cross-modality alignment loss: forces scalp EEG and iEEG representations
    of the same subject/time-window to be close in latent space.

    Used for DANDI:000574 (simultaneous scalp + iEEG recordings).
    L_align = 1 - cosine_similarity(z_scalp, z_ieeg)

    Reference: ECOG_INTEGRATION_PLAN.md §4.4
    """

    def __init__(self, weight: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(
        self,
        z_scalp: torch.Tensor,
        z_ieeg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_scalp: (B, D) scalp EEG encoder output (pooled)
            z_ieeg: (B, D) iEEG encoder output (pooled)
        Returns:
            loss, metrics
        """
        # Cosine similarity
        z_scalp_norm = F.normalize(z_scalp, dim=-1)
        z_ieeg_norm = F.normalize(z_ieeg, dim=-1)
        sim = (z_scalp_norm * z_ieeg_norm).sum(dim=-1)  # (B,)

        # Loss: push similarity toward 1.0
        loss = (1.0 - sim).mean()

        metrics = {
            "modality_align_loss": loss.item(),
            "modality_sim_mean": sim.mean().item(),
            "modality_sim_min": sim.min().item(),
            "modality_sim_max": sim.max().item(),
        }
        return self.weight * loss, metrics


class LossNormalizer:
    """
    Per-loss EMA normalization with calibration freeze.

    During calibration (first `warmup_steps` steps), EMA statistics are
    accumulated. After calibration, statistics are FROZEN to prevent the
    normalization from becoming a moving target (a converging loss would
    have shrinking variance, which artificially amplifies small residuals).

    Args:
        beta: EMA decay rate
        eps: numerical stability
        warmup_steps: number of steps to calibrate before freezing
    """

    def __init__(self, beta: float = 0.99, eps: float = 1e-8, warmup_steps: int = 1000):
        self.beta = beta
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.ema_mean: Dict[str, float] = {}
        self.ema_var: Dict[str, float] = {}
        self.step_count: Dict[str, int] = {}
        self._frozen: Dict[str, bool] = {}
        self._frozen_mean: Dict[str, float] = {}
        self._frozen_std: Dict[str, float] = {}

    def normalize(self, loss_name: str, loss_value: torch.Tensor) -> torch.Tensor:
        """Normalize loss to unit variance using EMA statistics (frozen after warmup)."""
        val = loss_value.detach().item()

        if loss_name not in self.ema_mean:
            self.ema_mean[loss_name] = val
            self.ema_var[loss_name] = 0.0
            self.step_count[loss_name] = 1
            self._frozen[loss_name] = False
            return loss_value

        n = self.step_count[loss_name]

        if self._frozen.get(loss_name, False):
            return (loss_value - self._frozen_mean[loss_name]) / (self._frozen_std[loss_name] + self.eps)

        mean = self.ema_mean[loss_name]
        var = self.ema_var[loss_name]

        self.ema_mean[loss_name] = self.beta * mean + (1 - self.beta) * val
        self.ema_var[loss_name] = self.beta * var + (1 - self.beta) * (val - mean) ** 2
        self.step_count[loss_name] = n + 1

        if n + 1 >= self.warmup_steps:
            self._frozen[loss_name] = True
            mean_corr = self.ema_mean[loss_name] / (1 - self.beta ** (n + 1))
            var_corr = self.ema_var[loss_name] / (1 - self.beta ** (n + 1))
            self._frozen_mean[loss_name] = mean_corr
            self._frozen_std[loss_name] = math.sqrt(var_corr + self.eps)

        mean_corrected = self.ema_mean[loss_name] / (1 - self.beta ** (n + 1))
        var_corrected = self.ema_var[loss_name] / (1 - self.beta ** (n + 1))
        std_corrected = math.sqrt(var_corrected + self.eps)

        return (loss_value - mean_corrected) / std_corrected

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return current EMA statistics."""
        return {
            name: {"mean": self.ema_mean[name], "std": math.sqrt(self.ema_var[name] + self.eps)}
            for name in self.ema_mean
        }


class BandPowerLoss(nn.Module):
    """
    Cross-frequency coupling loss via band-power reconstruction.

    Compares band-limited power between predicted and target EEG signals
    across delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz),
    beta (13-30 Hz), gamma (30-50 Hz) bands using STFT.

    This provides a neurophysiologically meaningful validation metric
    beyond pixel-level MSE: the model must reproduce the correct
    spectral power distribution, not just the raw waveform shape.

    Reference: cerebellar OKR validation tradition (Yamazaki & Nagao);
              INTRODUCTION §4.5 P1.
    """

    FREQ_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    def __init__(self, weight: float = 0.0, sample_rate: float = 256.0, n_fft: int = 128):
        super().__init__()
        self.weight = weight
        self.sample_rate = sample_rate
        self.n_fft = n_fft

    def band_power(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute band power for each frequency band.

        Args:
            x: (B, C, T) EEG signal
        Returns:
            dict mapping band name -> (B, C) power values
        """
        B, C, T = x.shape
        if T < self.n_fft:
            return {name: torch.zeros(B, C, device=x.device) for name in self.FREQ_BANDS}

        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(
            x.view(B * C, T),
            n_fft=self.n_fft,
            hop_length=self.n_fft // 4,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2).mean(dim=-1)
        power = power.view(B, C, -1)

        freqs = torch.linspace(0, self.sample_rate / 2, power.shape[-1], device=x.device)
        result = {}
        for name, (lo, hi) in self.FREQ_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                result[name] = power[:, :, mask].sum(dim=-1)
            else:
                result[name] = torch.zeros(B, C, device=x.device)
        return result

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute band-power MSE loss.

        Args:
            pred: (B, C, T) predicted EEG
            target: (B, C, T) target EEG
        Returns:
            total_loss, metrics dict
        """
        pred_bp = self.band_power(pred)
        target_bp = self.band_power(target)

        total = torch.tensor(0.0, device=pred.device)
        metrics = {}
        for name in self.FREQ_BANDS:
            loss = F.mse_loss(pred_bp[name], target_bp[name])
            total = total + loss
            metrics[f"band_{name}"] = loss.item()
        metrics["bandpower_loss"] = total.item()
        return total, metrics


class TotalLoss(nn.Module):
    """
    Combined loss function for Brain MoE-PINN.

    Aggregates all individual loss terms with stage-specific weights.
    """

    def __init__(
        self,
        loss_weights: "LossWeights",
    ):
        super().__init__()
        self.loss_weights = loss_weights

        self.recon_loss = ReconstructionLoss()
        self.velocity_smooth = VelocitySmoothnessLoss(alpha=loss_weights.velocity_smooth)
        self.generic_constraint = GenerickeConstraintLoss(weight=loss_weights.generic_constraint)
        self.moe_balance = MoELoadBalancingLoss(alpha=loss_weights.moe_load_balance)
        self.grassmannian = GrassmannianRegularization(weight=loss_weights.grassmannian_reg)
        self.nsp = NSPLoss(weight=loss_weights.nsp)
        self.cross_modal = CrossModalAlignmentLoss(weight=loss_weights.cross_modal)
        self.dissip = DissipationLoss(weight=getattr(loss_weights, "dissip", 0.1))
        self.epr = EPRLoss(weight=getattr(loss_weights, "epr", 0.05))
        self.spectrum = SpectralSlopeLoss(weight=getattr(loss_weights, "spectrum", 0.01))
        self.sparsity = LatentSparsityLoss(weight=getattr(loss_weights, "tsallis", 0.05))
        self.ks = KSEntropyLoss(weight=getattr(loss_weights, "ks", 0.01))
        self.action = ActionLoss(weight=getattr(loss_weights, "action", 0.1))
        self.replay = ReplayLoss(weight=getattr(loss_weights, "replay", 0.02))
        self.bandpower = BandPowerLoss(weight=getattr(loss_weights, "bandpower", 0.0))

        self.normalizer = LossNormalizer(beta=0.99)
        self.use_loss_normalization = getattr(loss_weights, "use_loss_normalization", True)

    def _add_loss(
        self,
        total: torch.Tensor,
        loss_value: torch.Tensor,
        weight: float,
        name: str,
        metrics: Dict[str, float],
    ) -> torch.Tensor:
        """Add a loss term with optional EMA normalization."""
        if self.use_loss_normalization and weight > 0:
            loss_norm = self.normalizer.normalize(name, loss_value)
        else:
            loss_norm = loss_value
        return total + weight * loss_norm

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss from predictions and targets.

        Args:
            predictions: dict of model outputs
            targets: dict of ground truth
        Returns:
            total_loss, metrics dict
        """
        total_loss = 0.0
        metrics = {}

        if self.loss_weights.recon_eeg > 0 and "eeg_recon" in predictions:
            recon_eeg = self.recon_loss.forward_eeg(predictions["eeg_recon"], targets["eeg"])
            total_loss = self._add_loss(total_loss, recon_eeg, self.loss_weights.recon_eeg, "recon_eeg", metrics)
            metrics["recon_eeg"] = recon_eeg.item()

        if self.loss_weights.recon_fmri > 0 and "fmri_recon" in predictions:
            recon_fmri = self.recon_loss.forward_fmri(predictions["fmri_recon"], targets["fmri"])
            total_loss = self._add_loss(total_loss, recon_fmri, self.loss_weights.recon_fmri, "recon_fmri", metrics)
            metrics["recon_fmri"] = recon_fmri.item()

        if self.loss_weights.velocity_smooth > 0 and "delta_z" in predictions:
            vel_smooth = self.velocity_smooth(predictions["delta_z"])
            total_loss = self._add_loss(total_loss, vel_smooth, self.loss_weights.velocity_smooth, "velocity_smooth", metrics)
            metrics["velocity_smooth"] = vel_smooth.item()

        if self.loss_weights.nsp > 0 and "eeg_recon" in predictions:
            nsp_loss, nsp_metrics = self.nsp(predictions["eeg_recon"], targets["eeg"])
            total_loss = self._add_loss(total_loss, nsp_loss, self.loss_weights.nsp, "nsp", metrics)
            metrics.update({f"nsp_{k}": v for k, v in nsp_metrics.items()})

        if self.loss_weights.cross_modal > 0 and "hub_eeg" in predictions and "hub_fmri" in predictions:
            cm_loss, cm_metrics = self.cross_modal(
                predictions["hub_eeg"],
                predictions["hub_fmri"],
                targets.get("cross_modal_labels"),
            )
            total_loss = self._add_loss(total_loss, cm_loss, self.loss_weights.cross_modal, "cross_modal", metrics)
            metrics.update({f"cross_modal_{k}": v for k, v in cm_metrics.items()})

        if getattr(self.loss_weights, "dissip", 0) > 0 and "delta_z" in predictions and "grad_S" in predictions:
            dissip_loss, dissip_metrics = self.dissip(
                predictions["delta_z"],
                predictions["grad_S"],
                predictions.get("M_diag", torch.ones_like(predictions["delta_z"]) * 0.1),
                predictions.get("grad_E", torch.zeros_like(predictions["delta_z"])),
            )
            total_loss = self._add_loss(total_loss, dissip_loss, self.loss_weights.dissip, "dissip", metrics)
            metrics.update({f"dissip_{k}": v for k, v in dissip_metrics.items()})

        if getattr(self.loss_weights, "epr", 0) > 0 and "delta_z" in predictions:
            epr_loss, epr_metrics = self.epr(predictions["delta_z"])
            total_loss = self._add_loss(total_loss, epr_loss, self.loss_weights.epr, "epr", metrics)
            metrics.update({f"epr_{k}": v for k, v in epr_metrics.items()})

        if getattr(self.loss_weights, "spectrum", 0) > 0 and "z_sequence" in predictions:
            spec_loss, spec_metrics = self.spectrum(predictions["z_sequence"])
            total_loss = self._add_loss(total_loss, spec_loss, self.loss_weights.spectrum, "spectrum", metrics)
            metrics.update({f"spectrum_{k}": v for k, v in spec_metrics.items()})

        if getattr(self.loss_weights, "tsallis", 0) > 0 and "z_global" in predictions:
            sparsity_loss, sparsity_metrics = self.sparsity(predictions["z_global"])
            total_loss = self._add_loss(total_loss, sparsity_loss, self.loss_weights.tsallis, "latent_sparsity", metrics)
            metrics.update({f"sparsity_{k}": v for k, v in sparsity_metrics.items()})

        if getattr(self.loss_weights, "moe_load_balance", 0) > 0 and "routing_metrics" in predictions:
            rm = predictions["routing_metrics"]
            if "gate_weights" in rm and "selected_experts" in rm:
                num_experts = getattr(self.loss_weights, "num_experts", 16)
                balance_loss, balance_metrics = self.moe_balance(
                    rm["gate_weights"], rm["selected_experts"], num_experts
                )
                total_loss = self._add_loss(total_loss, balance_loss, self.loss_weights.moe_load_balance, "moe_balance", metrics)
                metrics.update({f"balance_{k}": v for k, v in balance_metrics.items()})

        if getattr(self.loss_weights, "ks", 0) > 0 and "delta_z_sequence" in predictions:
            ks_loss, ks_metrics = self.ks(predictions["delta_z_sequence"])
            total_loss = self._add_loss(total_loss, ks_loss, self.loss_weights.ks, "ks", metrics)
            metrics.update({f"ks_{k}": v for k, v in ks_metrics.items()})

        if getattr(self.loss_weights, "action", 0) > 0 and "efe" in predictions:
            action_loss, action_metrics = self.action(predictions["efe"])
            total_loss = self._add_loss(total_loss, action_loss, self.loss_weights.action, "action", metrics)
            metrics.update({f"action_{k}": v for k, v in action_metrics.items()})

        if getattr(self.loss_weights, "replay", 0) > 0 and "z_imagined" in predictions and "z_engram" in predictions:
            replay_loss, replay_metrics = self.replay(predictions["z_imagined"], predictions["z_engram"])
            total_loss = self._add_loss(total_loss, replay_loss, self.loss_weights.replay, "replay", metrics)
            metrics.update({f"replay_{k}": v for k, v in replay_metrics.items()})

        if getattr(self.loss_weights, "bandpower", 0) > 0 and "eeg_recon" in predictions and "eeg" in targets:
            bp_loss, bp_metrics = self.bandpower(predictions["eeg_recon"], targets["eeg"])
            total_loss = self._add_loss(total_loss, bp_loss, self.loss_weights.bandpower, "bandpower", metrics)
            metrics.update({f"bandpower_{k}": v for k, v in bp_metrics.items()})

        if self.use_loss_normalization:
            metrics["loss_norm_stats"] = self.normalizer.get_stats()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


if __name__ == "__main__":
    print("Testing loss functions...")

    recon = ReconstructionLoss()
    x = torch.randn(4, 19, 256)
    y = torch.randn(4, 19, 256)
    loss = recon.forward_eeg(x, y)
    print(f"  Reconstruction loss: {loss.item():.4f}")

    vel_smooth = VelocitySmoothnessLoss()
    delta_z = torch.randn(4, 2048)
    loss = vel_smooth(delta_z)
    print(f"  Velocity smoothness: {loss.item():.4f}")

    moe_balance = MoELoadBalancingLoss()
    gate_weights = torch.softmax(torch.randn(4, 2), dim=-1)
    selected_indices = torch.tensor([[0, 3], [1, 4], [2, 5], [3, 6]])
    loss, m = moe_balance(gate_weights, selected_indices, num_experts=16)
    print(f"  MoE balance loss: {loss.item():.4f}, variance: {m['expert_util_variance']:.4f}")

    print("\nAll tests passed!")