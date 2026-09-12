"""
Loss Functions for Brain MoE-PINN Training.

Implements all loss terms:
- Reconstruction losses (EEG, fMRI)
- Velocity smoothness constraint
- GENERIC-inspired residuals (degeneracy, optional Jacobi diagnostic)
- MoE load balancing
- Hebbian regularization
- Grassmannian/orthogonal attractor regularization
- NSP (Neural Signal Processing) auxiliary losses
- Cross-modal alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..runtime.device_utils import safe_epsilon
from typing import Dict, Tuple, Optional, Union
import math


class ReconstructionLoss(nn.Module):
    """Signal reconstruction loss for any same-shaped prediction/target pair.

    Modality-generic: EEG, fMRI, MEG, calcium, voltage, widefield, and any
    other neural signal differ only in tensor shape and noise model. The
    former ``forward_eeg``/``forward_fmri``/``forward_meg`` aliases were
    identical code; call ``forward(pred, target)`` instead and select a
    per-modality error model via ``loss_type``/``channel_weights``/``mask``.

    Criteria (all backprop-safe; choice motivated by TDE-RICA / C. elegans
    validation practice, which compares signals with correlation-style,
    distributional, and increment-based measures rather than raw MSE):

    - ``mse`` / ``l1`` / ``huber``: elementwise errors (homoscedastic
      Gaussian, Laplace, and robust Huber noise models).
    - ``poisson``: Poisson NLL with rate = softplus(pred) for count-like
      signals (spike counts); target must be non-negative.
    - ``correlation``: per-channel Pearson correlation (1 - rho, mean over
      channels). Scale/shift invariant per channel - TDE-RICA compares
      real-vs-sim trajectories with cosine/correlation similarity, and the
      C. elegans lesson flags per-component gain mismatch as a key failure
      mode of plain MSE.
    - ``corr_diff``: correlation on first temporal differences; targets the
      *dynamics* (increments) instead of the strongly autocorrelated
      baseline, penalizing trivial lag-copy solutions.
    - ``wasserstein1``: per-channel 1-D Wasserstein distance between the
      sorted prediction and target marginals (exact, differentiable via
      sort); matches the distribution axis of the multi-axis validation
      suite.
    """

    LOSS_TYPES = (
        "mse", "l1", "huber", "poisson", "correlation", "corr_diff",
        "wasserstein1",
    )
    ELEMENTWISE_TYPES = ("mse", "l1", "huber", "poisson")

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        if loss_type not in self.LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {self.LOSS_TYPES}")
        self.loss_type = loss_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Optional[str] = None,
        channel_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruction criterion between same-shaped ``(..., C, T)`` tensors.

        Args:
            pred, target: any matching-shaped signal tensors
            loss_type: overrides the constructor default per call
            channel_weights: optional ``(C,)`` per-channel weights; for
                statistical criteria the loss is a weighted mean over
                channels instead of the unweighted mean
            mask: optional boolean ``(B, C, T)`` validity mask; elementwise
                criteria average over valid entries, statistical criteria
                compute over valid time points per channel
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shapes must match; got {tuple(pred.shape)} "
                f"vs {tuple(target.shape)}")
        loss_type = loss_type or self.loss_type
        if loss_type not in self.LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {self.LOSS_TYPES}")

        if loss_type in self.ELEMENTWISE_TYPES:
            return self._elementwise_loss(
                pred, target, loss_type, channel_weights, mask)
        if loss_type in ("correlation", "corr_diff"):
            return self._correlation_loss(
                pred, target, on_difference=(loss_type == "corr_diff"),
                channel_weights=channel_weights, mask=mask)
        return self._wasserstein_loss(pred, target, channel_weights, mask)

    def _elementwise_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str,
        channel_weights: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if loss_type == "mse":
            elementwise = (pred - target) ** 2
        elif loss_type == "l1":
            elementwise = (pred - target).abs()
        elif loss_type == "huber":
            elementwise = F.huber_loss(pred, target, delta=1.0, reduction="none")
        else:  # poisson NLL, rate = softplus(pred)
            rate = F.softplus(pred)
            elementwise = rate - target.clamp_min(0.0) * torch.log(
                rate + safe_epsilon(rate))

        if channel_weights is not None:
            elementwise = elementwise * self._channel_weights_shape(
                elementwise, channel_weights)

        if mask is not None:
            if mask.shape != pred.shape:
                raise ValueError("mask must match pred shape")
            if mask.dtype != torch.bool:
                mask = mask.bool()
            return elementwise[mask].mean()
        return elementwise.mean()

    @staticmethod
    def _channel_weights_shape(
        tensor: torch.Tensor, channel_weights: torch.Tensor
    ) -> torch.Tensor:
        return channel_weights.reshape(
            (1,) * (tensor.dim() - 2) + (-1, 1))

    @staticmethod
    def _pearson_per_channel(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-(B, C) Pearson correlation of pred/target along the time axis.

        Returns ``rho`` with shape ``(B, C)`` in [-1, 1] (0 for degenerate
        channels with zero variance on either side).
        """
        def center(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            mean = x.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True) / (
                valid.sum(dim=-1, keepdim=True).clamp_min(1.0))
            return x - mean

        B, C, T = pred.shape
        if mask is None:
            valid = torch.ones_like(pred, dtype=torch.bool)
        else:
            valid = mask.bool()
            valid = valid.reshape(B, C, T)
        pred = pred.reshape(B, C, T)
        target = target.reshape(B, C, T)
        if T < 2 or (mask is not None and valid.sum(dim=-1).min() < 2):
            return torch.zeros(B, C, device=pred.device, dtype=pred.dtype)

        pc = center(pred, valid)
        tc = center(target, valid)
        num = (pc * tc * valid).sum(dim=-1)
        den = ((pc * pc * valid).sum(dim=-1)
               * (tc * tc * valid).sum(dim=-1)).clamp_min(1e-6).sqrt()
        rho = num / den
        # Guard against division artifacts when a channel is constant.
        valid_var = valid.sum(dim=-1) >= 2
        return torch.where(valid_var, rho, torch.zeros_like(rho))

    def _correlation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        on_difference: bool,
        channel_weights: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if on_difference:
            if pred.shape[-1] < 2:
                return pred.new_zeros(())
            diff_mask = None
            if mask is not None:
                diff_mask = mask[..., 1:] & mask[..., :-1]
            rho = self._pearson_per_channel(
                pred[..., 1:] - pred[..., :-1],
                target[..., 1:] - target[..., :-1],
                diff_mask,
            )
        else:
            rho = self._pearson_per_channel(pred, target, mask)
        per_channel = 1.0 - rho  # (B, C)
        if channel_weights is not None:
            if channel_weights.shape[0] != per_channel.shape[-1]:
                raise ValueError("channel_weights length must match channel dim")
            per_channel = per_channel * channel_weights.to(per_channel.device)
        return per_channel.mean()

    def _wasserstein_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_weights: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Exact 1-D Wasserstein distance per channel on the time axis.

        W1(pred_row, target_row) = mean(|sorted(pred) - sorted(target)|);
        differentiable through the sort. With a mask, only valid time points
        are sorted per channel.
        """
        B, C, T = pred.shape
        if T < 1:
            raise ValueError("cannot compute Wasserstein distance over empty axis")
        if mask is not None and mask.shape != pred.shape:
            raise ValueError("mask must match pred shape")

        distances = []
        for b in range(B):
            for c in range(C):
                if mask is not None:
                    valid = mask[b, c].bool()
                    if valid.sum() == 0:
                        distances.append(pred.new_zeros(()))
                        continue
                    p = pred[b, c][valid]
                    t = target[b, c][valid]
                else:
                    p = pred[b, c]
                    t = target[b, c]
                sp, _ = torch.sort(p)
                st, _ = torch.sort(t)
                distances.append((sp - st).abs().mean())
        stacked = torch.stack(distances).reshape(B, C)
        if channel_weights is not None:
            if channel_weights.shape[0] != C:
                raise ValueError("channel_weights length must match channel dim")
            stacked = stacked * channel_weights.to(stacked.device)
        return stacked.mean()


class CompositeForecastLoss(nn.Module):
    """Horizon-weighted robust signal and increment forecast loss.

    Inputs are ``(B, K, C, T)`` tensors. Each horizon combines Huber error on
    the signal with correlation error on first differences, so a forecast
    cannot win only by copying a smooth baseline or by matching amplitude.
    """

    def __init__(
        self,
        huber_weight: float = 1.0,
        corr_diff_weight: float = 1.0,
        horizon_weights: Optional[Union[torch.Tensor, list]] = None,
    ):
        super().__init__()
        if huber_weight < 0 or corr_diff_weight < 0:
            raise ValueError("forecast loss weights must be non-negative")
        if huber_weight == 0 and corr_diff_weight == 0:
            raise ValueError("at least one forecast criterion is required")
        self.huber_weight = float(huber_weight)
        self.corr_diff_weight = float(corr_diff_weight)
        self.horizon_weights = (
            None if horizon_weights is None else torch.as_tensor(
                horizon_weights, dtype=torch.float32))
        if self.horizon_weights is not None:
            if self.horizon_weights.dim() != 1:
                raise ValueError("horizon_weights must be one-dimensional")
            if bool((self.horizon_weights < 0).any()):
                raise ValueError("horizon_weights must be non-negative")
            if float(self.horizon_weights.sum()) <= 0:
                raise ValueError("horizon_weights must have positive mass")
        self.huber = ReconstructionLoss("huber")
        self.corr_diff = ReconstructionLoss("corr_diff")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.dim() != 4 or target.dim() != 4:
            raise ValueError(
                "forecast prediction and target must have shape (B,K,C,T)")
        if prediction.shape != target.shape:
            raise ValueError(
                "forecast prediction and target shapes must match")
        if mask is not None and mask.shape != target.shape:
            raise ValueError("forecast mask must match prediction shape")
        horizons = prediction.shape[1]
        if self.horizon_weights is None:
            weights = prediction.new_ones(horizons)
        else:
            if self.horizon_weights.numel() != horizons:
                raise ValueError(
                    f"horizon_weights has {self.horizon_weights.numel()} values "
                    f"for {horizons} horizons")
            weights = self.horizon_weights.to(prediction)
        weights = weights / weights.sum().clamp_min(1e-12)
        total = prediction.sum() * 0.0
        for horizon in range(horizons):
            horizon_mask = None if mask is None else mask[:, horizon]
            robust = self.huber(
                prediction[:, horizon], target[:, horizon],
                mask=horizon_mask)
            increment = self.corr_diff(
                prediction[:, horizon], target[:, horizon],
                mask=horizon_mask)
            total = total + weights[horizon] * (
                self.huber_weight * robust
                + self.corr_diff_weight * increment)
        return total


class VelocitySmoothnessLoss(nn.Module):
    """Temporal total-variation of the latent velocity along a rollout.

    Penalizes sudden velocity jumps across TIME: given a rollout of latent
    velocities ``delta_z_sequence`` shaped ``(B, T, d)`` with ``T >= 2``, the
    loss is the mean L1 norm of the per-step velocity differences.

    The pre-2026-09 implementation took the difference over the BATCH axis of
    a single-step ``(B, d)`` tensor, which shrank inter-sample velocity
    diversity instead of enforcing temporal smoothness (and could not bound
    velocity magnitude). Single-step inputs have no temporal axis and now
    return zero; enable the term only for rollout training
    (``rollout_steps > 1``), which also feeds the sibling
    ``VelocityDifferenceRegularizer``.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, delta_z_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_z_sequence: (B, d) single-step velocities (returns 0 —
                no temporal axis) or (B, T, d) rollout velocities with T >= 2
        """
        if delta_z_sequence.dim() < 3 or delta_z_sequence.shape[1] < 2:
            return torch.tensor(0.0, device=delta_z_sequence.device)

        diff = delta_z_sequence[:, 1:] - delta_z_sequence[:, :-1]
        smoothness = torch.norm(diff, p=1, dim=-1).mean()
        return self.alpha * smoothness


class GenerickeConstraintLoss(nn.Module):
    """Penalize pointwise residuals for GENERIC-inspired degeneracy.

    The two residuals are evaluated at supplied samples:
    ``L(z) @ grad_S(z) ≈ 0`` and ``M(z) @ grad_E(z) ≈ 0``.  This does not
    enforce a globally valid bracket and does not compute a Jacobi residual.
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
            L_z: (d, d) or (B, d, d) candidate antisymmetric matrix
            grad_S: (B, d) entropy-potential gradient
            M_diag: (B, d) diagonal mobility candidate
            grad_E: (B, d) energy-potential gradient
        Returns:
            total_loss, metrics dict
        """
        B = grad_S.shape[0]
        device = grad_S.device

        if L_z.dim() == 2:
            L_grad_S = torch.bmm(L_z.unsqueeze(0).expand(B, -1, -1), grad_S.unsqueeze(-1)).squeeze(-1)
        else:
            L_grad_S = torch.bmm(L_z, grad_S.unsqueeze(-1)).squeeze(-1)
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
    Encourage balanced expert utilization with differentiable importance.

    The selected-expert load is discrete by design, but the auxiliary
    objective must still provide gradients through the router probabilities.
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
            gate_weights: (B, top_k) normalized weights for selected experts
            selected_indices: (B, top_k) selected absolute expert indices
            num_experts: total number of experts
        """
        B, top_k = gate_weights.shape
        if B == 0 or top_k == 0:
            zero = gate_weights.sum() * 0.0
            return zero, {"expert_util_variance": 0.0,
                          "router_entropy": 0.0}

        indices = selected_indices.clamp(0, num_experts - 1)
        one_hot = F.one_hot(indices, num_classes=num_experts).to(
            dtype=gate_weights.dtype)
        expert_load = one_hot.mean(dim=(0, 1))

        flat_indices = indices.reshape(-1)
        flat_weights = gate_weights.reshape(-1)
        importance = torch.zeros(
            num_experts, device=gate_weights.device,
            dtype=gate_weights.dtype).index_add(
                0, flat_indices, flat_weights) / B
        balance = num_experts * (importance * expert_load).sum()
        router_entropy = -(
            importance * torch.log(importance + safe_epsilon(importance))).sum()

        metrics = {
            "expert_util_variance": torch.var(expert_load).item(),
            "router_entropy": router_entropy.item(),
            "router_importance": importance.detach().cpu().tolist(),
        }
        return self.alpha * balance, metrics


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

        outputs_norm = expert_outputs / (
            expert_outputs.norm(dim=-1, keepdim=True) + safe_epsilon(expert_outputs))

        gram = torch.matmul(outputs_norm, outputs_norm.transpose(-2, -1))

        mask = torch.eye(num_experts, device=gram.device).unsqueeze(0)
        gram_off_diag = (1 - mask) * gram

        orthogonality = (gram_off_diag ** 2).sum() / (num_experts * (num_experts - 1))

        metrics = {
            "grassmannian_violation": orthogonality.item(),
        }

        return self.weight * orthogonality, metrics


class JacobiRegularization(nn.Module):
    """Optional sampled Jacobi residual for a differentiable Poisson map.

    Antisymmetry of ``L(z)`` is not the Jacobi identity.  A genuine sampled
    residual requires both state points and derivatives of a callable
    ``poisson_fn(z)``.  The high-dimensional training path does not provide
    those inputs, so enabling this term without them raises an explicit error
    instead of reporting a false Jacobi score.
    """

    def __init__(self, weight: float = 0.001, num_samples: int = 128,
                 max_dimension: int = 128):
        super().__init__()
        self.weight = weight
        self.num_samples = num_samples
        self.max_dimension = max_dimension

    def _compute_jacobi_violation(self, poisson_fn, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[0] != 1:
            raise ValueError("Jacobi diagnostic expects one state with shape (1, d)")
        d = z.shape[-1]
        if d > self.max_dimension:
            raise ValueError(
                f"sampled Jacobi check is limited to d <= {self.max_dimension}; got {d}")
        z = z.detach().requires_grad_(True)
        L = poisson_fn(z)[0]
        if L.shape != (d, d):
            raise ValueError("poisson_fn must return shape (1, d, d)")
        dL = torch.autograd.functional.jacobian(
            lambda q: poisson_fn(q[None])[0], z[0], create_graph=True)
        indices = torch.randint(0, d, (self.num_samples, 3), device=z.device)
        residuals = []
        for i, j, k in indices.tolist():
            value = torch.zeros((), dtype=L.dtype, device=L.device)
            for m in range(d):
                value = value + (
                    L[i, m] * dL[j, k, m]
                    + L[j, m] * dL[k, i, m]
                    + L[k, m] * dL[i, j, m]
                )
            residuals.append(value.square())
        return torch.stack(residuals).mean()

    def forward(
        self,
        L_z: torch.Tensor,
        d: int,
        poisson_fn=None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if poisson_fn is None or z is None:
            raise ValueError(
                "JacobiRegularization requires poisson_fn and z; sampled "
                "matrices alone cannot identify the Jacobi identity")
        violation = self._compute_jacobi_violation(poisson_fn, z)
        metrics = {"jacobi_violation": violation.item()}
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

    def __init__(self, weight: float = 0.1, sample_rate: float = 256.0):
        super().__init__()
        self.weight = weight
        self.sample_rate = sample_rate

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
            channel_bands = torch.zeros(C, 5, device=device)
            for ch in range(C):
                x = eeg_b[ch]
                fft_vals = torch.fft.rfft(x)
                psd = torch.abs(fft_vals) ** 2

                freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate)

                delta_mask = (freqs >= 1) & (freqs < 4)
                theta_mask = (freqs >= 4) & (freqs < 8)
                alpha_mask = (freqs >= 8) & (freqs < 13)
                beta_mask = (freqs >= 13) & (freqs < 30)
                gamma_mask = (freqs >= 30) & (freqs < 100)

                total_power = psd.sum() + safe_epsilon(psd)

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
    """Align EEG and fMRI representations using explicit pair labels.

    ``labels`` has one binary value per batch row: 1 means the two modalities
    are synchronized/aligned and 0 means they are intentionally unmatched.
    There is no unlabeled fallback because aligning every row would turn
    subject/session mismatches into false positives.
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
        """Compute labeled same-row alignment loss."""
        if labels is None:
            raise ValueError(
                "cross-modal alignment requires cross_modal_labels")
        if hub_eeg.shape != hub_fmri.shape or hub_eeg.dim() != 2:
            raise ValueError(
                "cross-modal hubs must have matching shape (B, latent_dim)")
        labels = labels.to(device=hub_eeg.device).flatten()
        if labels.numel() != hub_eeg.shape[0]:
            raise ValueError(
                "cross_modal_labels must have one value per batch row")
        if (not torch.isfinite(labels).all()
                or not bool(((labels == 0) | (labels == 1)).all())):
            raise ValueError("cross_modal_labels must contain only 0 or 1")

        similarity = F.cosine_similarity(hub_eeg, hub_fmri, dim=-1)
        aligned = labels == 1
        terms = []
        if bool(aligned.any()):
            terms.append(F.mse_loss(
                similarity[aligned], torch.ones_like(similarity[aligned])))
        if bool((~aligned).any()):
            terms.append(F.mse_loss(
                similarity[~aligned], torch.zeros_like(similarity[~aligned])))
        loss = torch.stack(terms).mean()
        metrics = {
            "cross_modal_similarity": similarity.mean().item(),
            "cross_modal_aligned_fraction": aligned.float().mean().item(),
        }
        return self.weight * loss, metrics


class DissipationLoss(nn.Module):
    """Penalize learned entropy-potential decrease along the model update.

    ``grad_S · delta_z`` is the directional change of the learned scalar
    field along the total model update.  It is an entropy-potential change
    proxy, not stochastic entropy production: ``delta_z`` also contains
    conservative, learned-bias, and possibly noisy terms.

    The GENERIC mobility condition ``M grad_E = 0`` is *not* penalized here.
    It is enforced architecturally: the applied dissipative force is
    ``(P_E diag(M) P_E) grad_S`` with ``P_E = I - grad_E grad_E^T / |grad_E|^2``,
    so the effective operator annihilates ``grad_E`` exactly.  Penalizing the
    raw diagonal ``M * grad_E`` instead would force ``M -> 0`` wherever
    ``grad_E != 0`` — it would delete the dissipative term it is meant to
    shape.  The raw coupling is still reported, scale-free, as a diagnostic.
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
        entropy_change_violation = F.relu(-dS_dt).mean()

        # Scale-free coupling between the raw diagonal mobility and grad_E:
        # 1.0 means M acts entirely along grad_E, 0.0 means M is orthogonal to
        # it.  Reported only; see the class docstring for why it is not a loss.
        coupling = (M_diag * grad_E).norm(dim=-1) / (
            M_diag.norm(dim=-1) * grad_E.norm(dim=-1)).clamp_min(1e-12)

        metrics = {
            "entropy_change_violation": entropy_change_violation.item(),
            "mobility_energy_coupling": coupling.mean().item(),
        }
        return self.weight * entropy_change_violation, metrics


class DissipativeFormMonitor:
    """Dissipative-form diagnostic. Metrics only — there is no loss here.

    Reports ``sigma = grad_S . M . grad_S / D``: the quadratic form of the
    mobility along the entropy gradient, i.e. the magnitude of the dissipative
    drive.  It is worth watching precisely because it collapses to zero when
    the mobility dies, which is the failure mode the degeneracy projection is
    built to avoid.

    It cannot be a steering term.  ``sigma`` is non-negative for any
    positive-semidefinite ``M``, so ``relu(-sigma)`` is identically zero: the
    previous `EPRLoss` carried a weight and a gradient path that could never
    fire.  It is also not stochastic entropy production; that would need the
    Ito/divergence term ``div(M grad_S)``, a Jacobian trace not worth its cost
    here.
    """

    def __init__(self, D: float = 1.0):
        if D <= 0:
            raise ValueError("D must be positive")
        self.D = float(D)

    def __call__(
        self, grad_S: torch.Tensor, M_diag: torch.Tensor
    ) -> Dict[str, float]:
        sigma = (grad_S * M_diag * grad_S).sum(dim=-1) / self.D
        return {
            "dissipative_form_mean": sigma.mean().item(),
            "mobility_mean": M_diag.mean().item(),
            "mobility_min": M_diag.min().item(),
        }


def aperiodic_exponent(
    signal: torch.Tensor,
    sample_rate: float = 256.0,
    f_min: float = 1.0,
    f_max: float = 45.0,
    max_channels: int = 8,
    peak_rejection: float = 2.0,
    num_iterations: int = 3,
) -> Tuple[torch.Tensor, bool]:
    """Estimate the aperiodic (1/f) exponent of ``(B, C, T)`` signals.

    Returns ``(exponent, valid)``.  ``exponent`` is the mean slope of
    ``log(power)`` against ``log(frequency)`` over ``[f_min, f_max]``;
    ``valid`` is False when the window leaves too few usable bins.

    Oscillatory peaks rise *above* the aperiodic background, and a plain
    least-squares line fit is strongly biased by them — the specparam
    comparison in Donoghue et al. (2020) reports Cohen's d ≈ 2.3 between naive
    linear fitting and peak-aware model fitting.  The fit here is therefore an
    iteratively reweighted regression that drops positive residuals beyond
    ``peak_rejection`` robust standard deviations before the final pass, so an
    alpha or beta peak cannot drag the estimated exponent flat.

    The aperiodic exponent is a state/age/excitation-inhibition marker that
    varies across subjects, regions, and tasks, and its estimated value depends
    on the frequency range used, so callers must treat the range as part of the
    estimate rather than as a fixed physical constant.
    """
    if signal.dim() != 3:
        raise ValueError("signal must have shape (B, C, T)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 < f_min < f_max:
        raise ValueError("f_min and f_max must satisfy 0 < f_min < f_max")

    time_len = signal.shape[-1]
    channels = min(signal.shape[1], int(max_channels))
    if channels < 1:
        return signal.new_zeros(()), False

    # Compute the spectrum in float32 on the signal's own device.  Casting is
    # required for two independent reasons:
    #   * torch.fft.rfftfreq returns a CPU tensor, so comparing it against a
    #     CUDA/XPU/NPU `power` raises a device mismatch;
    #   * cuFFT only supports half precision for power-of-two lengths, so a
    #     half-precision rfft at the production EEG length (2560) raises
    #     "cuFFT only supports dimensions whose sizes are powers of two".
    # .float() is differentiable, so gradients still reach a half-precision
    # reconstruction.
    window = signal[:, :channels, :].float()
    spectrum = torch.fft.rfft(window, dim=-1)
    power = spectrum.abs().square()
    freqs = torch.fft.rfftfreq(
        time_len, d=1.0 / sample_rate, device=window.device)
    band = (freqs >= f_min) & (freqs <= f_max)
    if int(band.sum()) < 4:
        return signal.new_zeros(()), False

    log_f = torch.log(freqs[band])
    log_p = torch.log(power[..., band] + 1e-10)
    flat = log_p.reshape(-1, log_p.shape[-1])
    slope = _robust_loglog_slope(
        log_f, flat, clip_sigma=peak_rejection,
        iterations=num_iterations).mean()
    return slope, True


def _robust_loglog_slope(
    log_f: torch.Tensor,
    log_p: torch.Tensor,
    clip_sigma: float = 2.0,
    iterations: int = 3,
) -> torch.Tensor:
    """Iteratively reweighted log-log regression, rejecting peak bins."""
    weight = torch.ones_like(log_p)
    slope = log_p.new_zeros(log_p.shape[0])
    for _ in range(max(1, int(iterations))):
        count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        x_mean = (log_f * weight).sum(dim=-1, keepdim=True) / count
        y_mean = (log_p * weight).sum(dim=-1, keepdim=True) / count
        x_dev = (log_f - x_mean) * weight
        denom = (x_dev * (log_f - x_mean)).sum(dim=-1, keepdim=True)
        denom = torch.where(
            denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
        slope = ((x_dev * (log_p - y_mean)).sum(dim=-1, keepdim=True) / denom)
        intercept = y_mean - slope * x_mean
        residual = log_p - (intercept + slope * log_f)
        limit = clip_sigma * residual.std(dim=-1, keepdim=True).clamp_min(1e-6)
        weight = (residual <= limit).to(log_p.dtype)
        # Never let rejection starve the second pass.
        starved = weight.sum(dim=-1, keepdim=True) < 3
        weight = torch.where(starved, torch.ones_like(weight), weight)
    return slope.squeeze(-1)


class SpectralSlopeLoss(nn.Module):
    """Aperiodic (1/f) exponent constraint on reconstructed EEG/MEG.

    Two modes:

    * **Data-driven** — ``forward(recon, target)`` estimates the target
      signal's exponent and penalises the reconstruction for deviating from
      that value.  This is the meaningful objective: the aperiodic exponent
      varies with subject, region, and cognitive state, so there is no single
      "healthy" exponent to steer toward, and a fixed target would fight any
      dataset whose spectra differ from the convention.
    * **Fixed** — ``forward(recon)`` falls back to ``target_slope`` for
      standalone/ablation use.  ``-1.0`` (pink noise) is the conventional
      reference point, not a physiological constant.

    Both modes use the peak-robust estimator in :func:`aperiodic_exponent`;
    the peak bias of a naive full-band line fit is what makes a fixed slope
    target misleading in the first place.  The fitting range and the reference
    exponent are reported so estimates stay comparable across runs.
    """

    def __init__(self, weight: float = 0.01, target_slope: float = -1.0,
                 sample_rate: float = 256.0, min_timesteps: int = 16,
                 f_min: float = 1.0, f_max: float = 45.0,
                 peak_rejection: float = 2.0, num_iterations: int = 3,
                 max_channels: int = 8):
        super().__init__()
        self.weight = weight
        self.target_slope = target_slope
        self.sample_rate = sample_rate
        self.min_timesteps = min_timesteps
        self.f_min = f_min
        self.f_max = f_max
        self.peak_rejection = peak_rejection
        self.num_iterations = num_iterations
        self.max_channels = max_channels

    def _exponent(self, signal: torch.Tensor):
        return aperiodic_exponent(
            signal,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            max_channels=self.max_channels,
            peak_rejection=self.peak_rejection,
            num_iterations=self.num_iterations,
        )

    def forward(
        self,
        recon: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            recon: (B, C, T) reconstructed EEG or MEG signal
            target: optional (B, C, T) reference signal; its exponent defines
                the objective instead of ``target_slope``
        Returns:
            loss, metrics dict
        """
        zero = recon.new_zeros(())
        if (recon.dim() != 3
                or recon.shape[-1] < self.min_timesteps
                or recon.shape[1] < 1):
            return zero, {"psd_slope": 0.0}

        slope, valid = self._exponent(recon)
        if not valid:
            return zero, {"psd_slope": 0.0}

        reference: Union[float, torch.Tensor] = self.target_slope
        reference_source = "fixed"
        if (target is not None and target.dim() == 3
                and target.shape == recon.shape
                and target.shape[-1] >= self.min_timesteps):
            target_slope, target_valid = self._exponent(target)
            if target_valid:
                # The reference is a property of the data, not a parameter.
                reference = target_slope.detach()
                reference_source = "target"

        loss = (slope - reference) ** 2
        metrics = {
            "psd_slope": float(slope.detach()),
            "psd_slope_reference": float(reference.detach())
            if torch.is_tensor(reference) else float(reference),
            "psd_slope_reference_source": reference_source,
            "psd_fit_range_hz": f"{self.f_min}-{self.f_max}",
        }
        return self.weight * loss, metrics


class LatentSparsityLoss(nn.Module):
    """
    Softmax-concentration penalty (Tsallis-type, q=2).

    Penalizes highly concentrated latent representations by computing:
      H_q = -1/(q-1) * (1 - sum_i p_i^q)
    where p_i = softmax(z_i) over latent dimensions.

    NOTE: This is NOT Tsallis entropy in the physical sense — softmax(z)
    does not define a probability distribution in information-theoretic space.
    The formula is a concentration/kurtosis penalty on "peaked" latent vectors.
    ``H_q`` increases with uniformity, so minimising it pushes the softmax
    toward uniform and *penalises* collapse.  The logged
    ``latent_concentration`` is that same quantity: higher = more uniform.

    Renamed from TsallisEntropyLoss to avoid false physics claims.
    """

    def __init__(self, weight: float = 0.05, q: float = 2.0):
        super().__init__()
        self.weight = weight
        self.q = q

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        p = F.softmax(z, dim=-1)
        tsallis = -(1.0 - (p ** self.q).sum(dim=-1)) / (self.q - 1.0)
        # Minimise the penalty, i.e. push the softmax toward uniform.  The
        # previous revision returned ``-tsallis.mean()``, which *maximised*
        # concentration and therefore rewarded the collapse this term exists
        # to prevent (and logged a metric that read the opposite way).
        loss = tsallis.mean()

        metrics = {
            "latent_concentration": tsallis.mean().item(),
        }
        return self.weight * loss, metrics


class ActionLoss(nn.Module):
    """Supervise expected free energy against executed-action utility."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        efe: torch.Tensor,
        utility_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if efe.shape != utility_target.shape:
            raise ValueError(
                "action_utility_target must match the EFE prediction shape")
        loss = F.smooth_l1_loss(efe, utility_target)
        metrics = {
            "efe_mean": efe.mean().item(),
            "action_utility_huber": loss.item(),
        }
        return self.weight * loss, metrics


class ReplayLoss(nn.Module):
    """Supervise imagined latent states against an explicit replay target."""

    def __init__(self, weight: float = 0.02):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        z_imagined: torch.Tensor,
        replay_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if z_imagined.shape != replay_target.shape:
            raise ValueError(
                "replay_target must match the imagined latent shape")
        loss = F.mse_loss(z_imagined, replay_target)
        metrics = {
            "replay_mse": loss.item(),
        }
        return self.weight * loss, metrics


class InterventionResponseLoss(nn.Module):
    """Fit a predicted treated-minus-baseline response to observed effects.

    The target is an explicit response/effect vector, not a proxy generated
    from the model's own rollout.  A mask may be supplied per element or per
    trailing latent vector.
    """

    def __init__(self, weight: float = 1.0, beta: float = 1.0):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.weight = weight
        self.beta = float(beta)

    def forward(
        self,
        predicted_effect: torch.Tensor,
        target_effect: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if predicted_effect.shape != target_effect.shape:
            raise ValueError(
                "intervention_target must match the predicted effect shape")
        if predicted_effect.dim() not in (2, 3):
            raise ValueError(
                "intervention effects must have shape (B,D) or (B,K,D)")
        error = F.smooth_l1_loss(
            predicted_effect, target_effect, beta=self.beta, reduction="none")
        if mask is not None:
            if mask.shape == predicted_effect.shape[:-1]:
                mask = mask.unsqueeze(-1)
            if mask.shape != predicted_effect.shape:
                raise ValueError(
                    "intervention_mask must match effect shape or omit latent dim")
            mask = mask.to(device=error.device, dtype=error.dtype)
            denominator = mask.sum().clamp_min(1.0)
            loss = (error * mask).sum() / denominator
            valid_fraction = (mask > 0).float().mean()
        else:
            loss = error.mean()
            valid_fraction = error.new_tensor(1.0)
        metrics = {
            "response_huber": loss.item(),
            "valid_fraction": valid_fraction.item(),
        }

        return self.weight * loss, metrics

class CrossSoftContrastiveLoss(nn.Module):
    """Contrastive alignment with explicit synchronized/async labels.

    ``labels`` has one binary value per row. Synchronized rows use their
    same-row cross-modal pair as the positive; async rows are required to
    have low diagonal similarity. An unlabeled identity fallback would
    silently train false positives and is therefore rejected.
    """

    def __init__(self, weight: float = 0.02, temperature: float = 0.07):
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(
        self,
        hub_eeg: torch.Tensor,
        hub_fmri: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if labels is None:
            raise ValueError(
                "soft cross-modal contrastive loss requires "
                "cross_modal_labels")
        if hub_eeg.shape != hub_fmri.shape or hub_eeg.dim() != 2:
            raise ValueError(
                "cross-modal hubs must have matching shape (B, latent_dim)")
        labels = labels.to(device=hub_eeg.device).flatten()
        if labels.numel() != hub_eeg.shape[0]:
            raise ValueError(
                "cross_modal_labels must have one value per batch row")
        if (not torch.isfinite(labels).all()
                or not bool(((labels == 0) | (labels == 1)).all())):
            raise ValueError("cross_modal_labels must contain only 0 or 1")

        z_eeg = F.normalize(hub_eeg, dim=-1)
        z_fmri = F.normalize(hub_fmri, dim=-1)
        logits = torch.matmul(z_eeg, z_fmri.t()) / self.temperature
        synchronized = labels == 1
        terms = []
        if bool(synchronized.any()):
            log_prob = logits - torch.logsumexp(
                logits, dim=-1, keepdim=True)
            terms.append(-log_prob.diagonal()[synchronized].mean())
        if bool((~synchronized).any()):
            diagonal = logits.diagonal()[~synchronized] * self.temperature
            terms.append(F.mse_loss(
                diagonal, torch.zeros_like(diagonal)))
        loss = torch.stack(terms).mean()
        metrics = {
            "cross_soft_loss": loss.item(),
            "cross_soft_sim_mean": (
                (logits * self.temperature).mean().item()),
            "cross_soft_aligned_fraction": synchronized.float().mean().item(),
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
    Per-loss scale equalization with calibration freeze.

    Each steering term is divided by an EMA of its own absolute value, so every
    term contributes at a comparable magnitude regardless of its natural units.
    The scale is frozen after ``warmup_steps`` so the weighting stops moving.

    Why magnitude rather than mean/std: normalizing by the run-to-run standard
    deviation makes the divisor vanish for any term that is converging or
    nearly constant, and the normalized value then explodes.  The previous
    revision tracked the variance of the *innovation* (``(v - prev_mean)^2``)
    rather than the spread of the term, so a steady loss was divided by the
    ``eps`` floor: a constant input of 1.0 was normalized to -492512 on the
    second step, and of two terms at the same scale the one with *less* jitter
    was amplified 40x more.  Freezing at ``warmup_steps`` did not help because
    the explosion starts at step 2, exactly when training is most fragile.
    Dividing by the magnitude is stable by construction -- a constant value
    normalizes to 1.0 -- which is what "prevent one loss from dominating due to
    scale differences" requires.  It rescales magnitude while preserving every
    term's gradient direction.

    Args:
        beta: EMA decay rate
        eps: numerical stability
        warmup_steps: steps of scale estimation before freezing
        min_scale: floor for the divisor (a dead term stays dead)
        max_scale: ceiling for the divisor
    """

    def __init__(
        self, beta: float = 0.99, eps: float = 1e-8, warmup_steps: int = 1000,
        min_scale: float = 1e-6, max_scale: float = 1e6,
    ):
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if min_scale <= 0 or max_scale < min_scale:
            raise ValueError("require 0 < min_scale <= max_scale")
        self.beta = beta
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ema_abs: Dict[str, float] = {}
        self.step_count: Dict[str, int] = {}
        self._frozen: Dict[str, bool] = {}
        self._frozen_scale: Dict[str, float] = {}

    def normalize(
        self, loss_name: str, loss_value: torch.Tensor, update: bool = True
    ) -> torch.Tensor:
        """Normalize by the EMA magnitude, optionally without mutating it."""
        val = loss_value.detach().item()

        if loss_name not in self.ema_abs:
            if not update:
                return loss_value
            self.ema_abs[loss_name] = abs(val)
            self.step_count[loss_name] = 1
            self._frozen[loss_name] = False
            return loss_value

        if self._frozen.get(loss_name, False):
            scale = self._frozen_scale[loss_name]
        else:
            if update:
                self.ema_abs[loss_name] = (
                    self.beta * self.ema_abs[loss_name]
                    + (1.0 - self.beta) * abs(val))
                self.step_count[loss_name] += 1
            scale = self.ema_abs[loss_name]
            if update and self.step_count[loss_name] >= self.warmup_steps:
                self._frozen[loss_name] = True
                self._frozen_scale[loss_name] = scale

        scale = min(max(scale, self.min_scale), self.max_scale)
        return loss_value / (scale + self.eps)

    def reset(self) -> None:
        """Discard calibration state at a phase boundary."""
        self.ema_abs.clear()
        self.step_count.clear()
        self._frozen.clear()
        self._frozen_scale.clear()

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return current scale statistics."""
        return {
            name: {
                "scale": self._frozen_scale.get(name, self.ema_abs[name]),
                "ema_abs": self.ema_abs[name],
                "steps": float(self.step_count[name]),
                "frozen": float(self._frozen.get(name, False)),
            }
            for name in self.ema_abs
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

    def __init__(self, weight: float = 0.0, sample_rate: float = 256.0,
                 n_fft: int = 128, include_one_over_f: bool = False):
        super().__init__()
        self.weight = weight
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        # SpectralSlopeLoss is the single 1/f authority in TotalLoss; this
        # flag exists for standalone use/ablation only.
        self.include_one_over_f = include_one_over_f

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
        Compute per-channel band-power MSE between prediction and target.

        The 1/f slope is deliberately NOT included by default: SpectralSlopeLoss
        is the single 1/f authority (on reconstructed EEG/MEG) inside TotalLoss,
        and enforcing it twice with different estimators double-counts the
        same spectral constraint. Pass ``include_one_over_f=True`` only for
        standalone ablation runs.

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

        if self.include_one_over_f:
            # 1/f slope constraint on predicted EEG (ablation/standalone only)
            slope_loss = self._one_over_f_loss(pred)
            total = total + slope_loss
            metrics["one_over_f_loss"] = slope_loss.item()

        metrics["bandpower_loss"] = total.item()
        return total, metrics

    def _one_over_f_loss(
        self, x: torch.Tensor, target_slope: float = -1.0
    ) -> torch.Tensor:
        """
        Penalize deviation from a 1/f power law in the EEG PSD.

        A 1/f power spectrum has ``log(PSD)`` slope -1 against
        ``log(f)``.  The signed slope is retained so blue-noise spectra are
        not mistaken for pink noise.
        """
        B, C, T = x.shape
        if T < self.n_fft:
            return torch.tensor(0.0, device=x.device)

        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(
            x.reshape(B * C, T),
            n_fft=self.n_fft,
            hop_length=self.n_fft // 4,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2).mean(dim=-1)
        freqs = torch.linspace(
            1, self.sample_rate / 2, power.shape[-1], device=x.device)

        log_f = torch.log(freqs)
        log_p = torch.log(power + 1e-10)
        log_f_centered = log_f - log_f.mean()
        log_p_centered = log_p - log_p.mean()
        slope = (log_f_centered * log_p_centered).sum() / (
            log_f_centered.pow(2).sum() + 1e-10)

        return F.mse_loss(
            slope, torch.tensor(target_slope, device=x.device))


class WeakSIGRegLoss(nn.Module):
    """
    Weak-SIGReg: Covariance regularization via random sketching.

    Constrains encoder embeddings toward an isotropic Gaussian by forcing
    the sketched covariance toward identity. Prevents representation collapse
    without a momentum teacher network.

    Reference: Akbar (2026), "Weak-SIGReg: Covariance Regularization for
    Stable Deep Learning", ICLR 2026 GRaM Workshop.
    """

    def __init__(self, sketch_dim: int = 64, alpha: float = 0.1):
        super().__init__()
        self.sketch_dim = sketch_dim
        # Retained for API compatibility; TotalLoss owns the phase weight and
        # applies it once, so forward() returns the RAW residual.
        self.alpha = alpha

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C) encoder output embeddings (pooled or CLS token)
        Returns:
            loss: scalar SIGReg loss
        """
        if z.dim() == 3:
            # If z is (B, L, C), pool to (B, C) via mean
            z = z.mean(dim=1)

        N, C = z.shape
        K = min(self.sketch_dim, C)

        if C > K:
            # Deterministic cosine sketch. A random projection here would make
            # validation metrics depend on the global RNG state.
            row = torch.arange(
                1, K + 1, device=z.device, dtype=z.dtype).unsqueeze(1)
            col = torch.arange(
                C, device=z.device, dtype=z.dtype).unsqueeze(0) + 0.5
            S = torch.cos(torch.pi * row * col / C) * (2.0 / C) ** 0.5
            z = z @ S.T
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (N - 1 + 1e-6)  # (K, K)
        target = torch.eye(K, device=z.device, dtype=z.dtype)

        return torch.norm(cov - target, p='fro')


class TotalLoss(nn.Module):
    """
    Combined loss function for Brain MoE-PINN.

    Aggregates all individual loss terms with stage-specific weights.
    """

    def __init__(
        self,
        loss_weights: "LossWeights",
        monitor_terms: bool = True,
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.monitor_terms = bool(monitor_terms)
        self.recon_loss = ReconstructionLoss()
        # Component modules return raw terms; phase weights are applied once
        # by ``_add_loss`` below.
        self.velocity_smooth = VelocitySmoothnessLoss(alpha=1.0)
        self.generic_constraint = GenerickeConstraintLoss(weight=1.0)
        self.moe_balance = MoELoadBalancingLoss(alpha=1.0)
        self.grassmannian = GrassmannianRegularization(weight=1.0)
        self.jacobi = JacobiRegularization(weight=1.0)
        self.hebbian = HebbianRegularization(weight=1.0)
        self.nsp = NSPLoss(weight=1.0)
        self.cross_modal = CrossModalAlignmentLoss(weight=1.0)
        self.cross_soft = CrossSoftContrastiveLoss(
            weight=1.0, temperature=0.07)
        self.cross_hrf = CrossModalAlignmentLoss(weight=1.0)
        self.dissip = DissipationLoss(weight=1.0)
        self.dissipative_form = DissipativeFormMonitor()
        self.spectrum = SpectralSlopeLoss(weight=1.0)
        self.sparsity = LatentSparsityLoss(weight=1.0)
        self.action = ActionLoss(weight=1.0)
        self.replay = ReplayLoss(weight=1.0)
        self.intervention_response = InterventionResponseLoss(weight=1.0)
        self.bandpower = BandPowerLoss(weight=1.0)
        self.sigreg = WeakSIGRegLoss(
            sketch_dim=getattr(loss_weights, "sigreg_sketch_dim", 64),
            alpha=1.0,
        )

        self.normalizer = LossNormalizer(beta=0.99)
        self.use_loss_normalization = getattr(
            loss_weights, "use_loss_normalization", True)
        self.forecast = CompositeForecastLoss(
            huber_weight=getattr(loss_weights, "forecast_huber", 1.0),
            corr_diff_weight=getattr(
                loss_weights, "forecast_corr_diff", 1.0),
            horizon_weights=getattr(
                loss_weights, "forecast_horizon_weights", None),
        )

    def reset_loss_normalizer(self) -> None:
        """Reset per-term scale calibration when a new phase begins."""
        self.normalizer.reset()

    def _add_loss(
        self,
        total: torch.Tensor,
        loss_value: torch.Tensor,
        weight: float,
        name: str,
        metrics: Dict[str, float],
        update_normalizer: Optional[bool] = None,
    ) -> torch.Tensor:
        """Add one raw loss term with exactly one phase weight."""
        if update_normalizer is None:
            update_normalizer = getattr(
                self, "_update_normalizer", True)
        if self.use_loss_normalization and weight > 0:
            loss_norm = self.normalizer.normalize(
                name, loss_value, update=update_normalizer)
        else:
            loss_norm = loss_value
        return total + weight * loss_norm

    def _iter_recon_weights(self):
        """Yield ``(modality, weight)`` for every configured reconstruction term.

        Fixed fields (``recon_eeg``/``recon_fmri``/``recon_meg``) come first,
        followed by ``LossWeights.recon_extra`` entries so arbitrary neural
        signals (calcium, voltage, widefield, ...) need config only — the
        convention is ``{modality}_recon`` in predictions and ``{modality}``
        in targets.
        """
        for modality in ("eeg", "fmri", "meg"):
            yield modality, getattr(
                self.loss_weights, f"recon_{modality}", 0.0)
        extras = getattr(self.loss_weights, "recon_extra", None) or {}
        dedicated = {"eeg", "fmri", "meg"}
        for modality, weight in extras.items():
            if modality not in dedicated:
                yield modality, float(weight)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        update_normalizer: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss from predictions and targets.

        ``update_normalizer=False`` is used for validation/test so evaluation
        cannot change training-time loss scales.
        """
        self._update_normalizer = bool(update_normalizer)

        def first_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                for nested in value.values():
                    found = first_tensor(nested)
                    if found is not None:
                        return found
            return None

        reference = first_tensor(predictions)
        if reference is None:
            reference = first_tensor(targets)
        total_loss = (
            torch.zeros((), requires_grad=True)
            if reference is None else reference.sum() * 0.0
        )
        metrics = {}

        for modality, weight in self._iter_recon_weights():
            if weight <= 0:
                continue
            pred_key = f"{modality}_recon"
            if pred_key not in predictions or modality not in targets:
                continue
            target_value = targets[modality]
            sequence_prediction = predictions.get(
                f"{modality}_recon_sequence")
            if (target_value.dim() == 4
                    and sequence_prediction is not None):
                forecast_weight = float(
                    getattr(self.loss_weights, "forecast", 0.0))
                if forecast_weight > 0:
                    forecast_value = self.forecast(
                        sequence_prediction, target_value,
                        mask=targets.get(f"{modality}_mask"))
                    total_loss = self._add_loss(
                        total_loss, forecast_value, forecast_weight,
                        f"forecast_{modality}", metrics)
                    metrics[f"forecast_{modality}"] = forecast_value.item()
                    continue
            if target_value.dim() == 4:
                target_value = target_value[:, -1]
            mask = targets.get(f"{modality}_mask")
            if mask is not None and mask.dim() == 4:
                mask = mask[:, -1]
            criteria = (getattr(self.loss_weights, "recon_loss_types", None)
                        or {}).get(modality)
            recon_value = self.recon_loss(
                predictions[pred_key], target_value,
                loss_type=criteria, mask=mask)
            total_loss = self._add_loss(
                total_loss, recon_value, weight, f"recon_{modality}", metrics)
            metrics[f"recon_{modality}"] = recon_value.item()

        if self.loss_weights.velocity_smooth > 0:
            # Temporal TV needs the rollout sequence; single-step "delta_z"
            # has no time axis and contributes zero (legacy batch-axis TV
            # semantics were removed 2026-09).
            vel_sequence = predictions.get(
                "delta_z_sequence", predictions.get("delta_z"))
            if vel_sequence is not None:
                vel_smooth = self.velocity_smooth(vel_sequence)
                total_loss = self._add_loss(
                    total_loss, vel_smooth,
                    self.loss_weights.velocity_smooth, "velocity_smooth", metrics)
                metrics["velocity_smooth"] = vel_smooth.item()

        if (
            self.loss_weights.nsp > 0
            and "eeg_recon" in predictions
            and "eeg" in targets
        ):
            nsp_target = targets["eeg"]
            if nsp_target.dim() == 4:
                nsp_target = nsp_target[:, -1]
            nsp_loss, nsp_metrics = self.nsp(
                predictions["eeg_recon"], nsp_target)
            total_loss = self._add_loss(
                total_loss, nsp_loss, self.loss_weights.nsp, "nsp", metrics)
            metrics.update({f"nsp_{k}": v for k, v in nsp_metrics.items()})

        if (self.loss_weights.cross_modal > 0
                and "hub_eeg" in predictions and "hub_fmri" in predictions):
            cm_loss, cm_metrics = self.cross_modal(
                predictions["hub_eeg"],
                predictions["hub_fmri"],
                targets.get("cross_modal_labels"),
            )
            total_loss = self._add_loss(
                total_loss, cm_loss, self.loss_weights.cross_modal,
                "cross_modal", metrics)
            metrics.update({
                f"cross_modal_{k}": v for k, v in cm_metrics.items()})
        if getattr(self.loss_weights, "dissip", 0) > 0 and "delta_z" in predictions and "grad_S" in predictions:
            dissip_loss, dissip_metrics = self.dissip(
                predictions["delta_z"],
                predictions["grad_S"],
                predictions.get("M_diag", torch.ones_like(predictions["delta_z"]) * 0.1),
                predictions.get("grad_E", torch.zeros_like(predictions["delta_z"])),
            )
            total_loss = self._add_loss(total_loss, dissip_loss, self.loss_weights.dissip, "dissip", metrics)
            metrics.update({f"dissip_{k}": v for k, v in dissip_metrics.items()})

        # NOTE: there is deliberately no dissipative-production steering term.
        # sigma = grad_S . M . grad_S / D is non-negative for PSD M, so the
        # penalty could never fire.  It is reported by the monitor instead.

        if getattr(self.loss_weights, "spectrum", 0) > 0:
            if "eeg_recon" in predictions:
                spectrum_target = targets.get("eeg")
                if (isinstance(spectrum_target, torch.Tensor)
                        and spectrum_target.dim() == 4):
                    spectrum_target = spectrum_target[:, -1]
                if spectrum_target is not None:
                    spec_eeg_loss, spec_eeg_metrics = self.spectrum(
                        predictions["eeg_recon"], spectrum_target)
                    total_loss = self._add_loss(
                        total_loss, spec_eeg_loss, self.loss_weights.spectrum,
                        "spectrum_eeg", metrics)
                    metrics.update({
                        f"spectrum_eeg_{k}": v
                        for k, v in spec_eeg_metrics.items()})
            if "meg_recon" in predictions:
                spectrum_target = targets.get("meg")
                if (isinstance(spectrum_target, torch.Tensor)
                        and spectrum_target.dim() == 4):
                    spectrum_target = spectrum_target[:, -1]
                if spectrum_target is not None:
                    spec_meg_loss, spec_meg_metrics = self.spectrum(
                        predictions["meg_recon"], spectrum_target)
                    total_loss = self._add_loss(
                        total_loss, spec_meg_loss, self.loss_weights.spectrum,
                        "spectrum_meg", metrics)
                    metrics.update({
                        f"spectrum_meg_{k}": v
                        for k, v in spec_meg_metrics.items()})
        if getattr(self.loss_weights, "moe_load_balance", 0) > 0 and "moe_routing" in predictions:
            rm = predictions["moe_routing"]
            if "gate_weights" in rm and "selected_experts" in rm:
                num_experts = rm.get(
                    "num_experts",
                    getattr(self.loss_weights, "num_experts", 16),
                )
                balance_loss, balance_metrics = self.moe_balance(
                    rm["gate_weights"], rm["selected_experts"], num_experts)
                total_loss = self._add_loss(
                    total_loss, balance_loss,
                    self.loss_weights.moe_load_balance,
                    "moe_balance", metrics)
                metrics.update({
                    f"balance_{k}": v
                    for k, v in balance_metrics.items()
                })

        action_weight = getattr(self.loss_weights, "action", 0)
        if action_weight > 0:
            if "efe" not in predictions:
                raise ValueError(
                    "action loss requires an executed-action EFE prediction")
            if "action_utility_target" not in targets:
                raise ValueError(
                    "action loss requires action_utility_target")
            action_loss, action_metrics = self.action(
                predictions["efe"], targets["action_utility_target"])
            total_loss = self._add_loss(
                total_loss, action_loss, action_weight, "action", metrics)
            metrics.update({
                f"action_{k}": v for k, v in action_metrics.items()})

        replay_weight = getattr(self.loss_weights, "replay", 0)
        if replay_weight > 0:
            if "z_imagined" not in predictions:
                raise ValueError(
                    "replay loss requires an imagined latent prediction")
            if "replay_target" not in targets:
                raise ValueError("replay loss requires replay_target")
            replay_loss, replay_metrics = self.replay(
                predictions["z_imagined"], targets["replay_target"])
            total_loss = self._add_loss(
                total_loss, replay_loss, replay_weight, "replay", metrics)
            metrics.update({
                f"replay_{k}": v for k, v in replay_metrics.items()})

        intervention_weight = getattr(
            self.loss_weights, "intervention_response", 0)
        if intervention_weight > 0:
            if "intervention_effect" not in predictions:
                raise ValueError(
                    "intervention response loss requires "
                    "intervention_effect prediction")
            if "intervention_target" not in targets:
                raise ValueError(
                    "intervention response loss requires intervention_target")
            response_loss, response_metrics = self.intervention_response(
                predictions["intervention_effect"],
                targets["intervention_target"],
                targets.get("intervention_mask"))
            total_loss = self._add_loss(
                total_loss, response_loss, intervention_weight,
                "intervention_response", metrics)
            metrics.update({
                f"intervention_{k}": v
                for k, v in response_metrics.items()})

        if (getattr(self.loss_weights, "bandpower", 0) > 0
                and "eeg_recon" in predictions and "eeg" in targets):
            bandpower_target = targets["eeg"]
            if bandpower_target.dim() == 4:
                bandpower_target = bandpower_target[:, -1]
            bp_loss, bp_metrics = self.bandpower(
                predictions["eeg_recon"], bandpower_target)

            total_loss = self._add_loss(
                total_loss, bp_loss, self.loss_weights.bandpower,
                "bandpower", metrics)
            metrics.update({
                f"bandpower_{k}": v for k, v in bp_metrics.items()})
        if self.loss_weights.generic_constraint > 0:
            if "generic_constraint_residual" in predictions:
                gen_loss = predictions["generic_constraint_residual"]
                gen_metrics = {
                    "constraint_residual": gen_loss.item()}
            elif "L_z" in predictions and "grad_S" in predictions:
                gen_loss, gen_metrics = self.generic_constraint(
                    predictions["L_z"],
                    predictions["grad_S"],
                    predictions.get(
                        "M_diag",
                        torch.ones_like(predictions["grad_S"]) * 0.1),
                    predictions.get(
                        "grad_E",
                        torch.zeros_like(predictions["grad_S"])),
                )
            else:
                gen_loss = None
                gen_metrics = {}
            if gen_loss is not None:
                total_loss = self._add_loss(
                    total_loss, gen_loss,
                    self.loss_weights.generic_constraint,
                    "generic_constraint", metrics,
                    update_normalizer=update_normalizer)
                metrics.update({
                    f"generic_{k}": v for k, v in gen_metrics.items()})

        if self.loss_weights.hebbian_reg > 0 and "hebbian_weights" in predictions:
            heb_loss, heb_metrics = self.hebbian(predictions["hebbian_weights"])
            total_loss = self._add_loss(total_loss, heb_loss, self.loss_weights.hebbian_reg, "hebbian_reg", metrics)
            metrics.update({f"hebbian_{k}": v for k, v in heb_metrics.items()})

        if self.loss_weights.grassmannian_reg > 0 and "grassmannian_loss" in predictions:
            grass_loss = predictions["grassmannian_loss"]
            total_loss = self._add_loss(total_loss, grass_loss, self.loss_weights.grassmannian_reg, "grassmannian_reg", metrics)
            metrics["grassmannian_loss"] = grass_loss.item()

        if self.loss_weights.jacobi_reg > 0 and "L_z" in predictions:
            poisson_fn = predictions.get("poisson_fn")
            jacobi_z = predictions.get("jacobi_z")
            jac_loss, jac_metrics = self.jacobi(
                predictions["L_z"],
                predictions["L_z"].shape[-1],
                poisson_fn=poisson_fn,
                z=jacobi_z,
            )
            total_loss = self._add_loss(
                total_loss, jac_loss, self.loss_weights.jacobi_reg,
                "jacobi_reg", metrics)
            metrics.update({f"jacobi_{k}": v for k, v in jac_metrics.items()})

        if getattr(self.loss_weights, "cross", 0) > 0 and "hrf_align_loss" in predictions:
            if "cross_modal_labels" not in targets:
                raise ValueError(
                    "cross-HRF loss requires cross_modal_labels")
            cross_loss = predictions["hrf_align_loss"]
            cross_w = getattr(self.loss_weights, "cross", 0)
            total_loss = self._add_loss(total_loss, cross_loss, cross_w, "cross_hrf", metrics)
            metrics["cross_hrf_loss"] = cross_loss.item()

        if getattr(self.loss_weights, "cross_soft", 0) > 0 and "hub_eeg" in predictions and "hub_fmri" in predictions:
            cs_loss, cs_metrics = self.cross_soft(
                predictions["hub_eeg"],
                predictions["hub_fmri"],
                targets.get("cross_modal_labels"),
            )
            total_loss = self._add_loss(total_loss, cs_loss, getattr(self.loss_weights, "cross_soft", 0), "cross_soft", metrics)
            metrics.update({f"cross_soft_{k}": v for k, v in cs_metrics.items()})

        if getattr(self.loss_weights, "sigreg", 0) > 0 and "z_global" in predictions:
            sigreg_loss = self.sigreg(predictions["z_global"])
            total_loss = self._add_loss(total_loss, sigreg_loss, self.loss_weights.sigreg, "sigreg", metrics)
            metrics["sigreg"] = sigreg_loss.item()

        if self.monitor_terms:
            self._update_monitors(predictions, metrics)

        if self.use_loss_normalization:
            metrics["loss_norm_stats"] = self.normalizer.get_stats()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    def _update_monitors(
        self,
        predictions: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
    ) -> None:
        """Log weight-0 terms as monitors without contributing gradients.

        Only cheap algebraic monitors run every step (EPR proxy, latent
        sparsity, Hebbian weight norm). Spectral/NSP monitors stay gated
        behind their phase weight because FFT/STFT cost is not worth paying
        at zero steering weight; enabling the weight restores their metrics.
        Monitor failures never raise into training.
        """
        try:
            if "grad_S" in predictions and "M_diag" in predictions:
                with torch.no_grad():
                    metrics.update({
                        f"mon_{key}": value
                        for key, value in self.dissipative_form(
                            predictions["grad_S"],
                            predictions["M_diag"],
                        ).items()
                    })
            if self.loss_weights.tsallis == 0 and "z_global" in predictions:
                with torch.no_grad():
                    _, mon = self.sparsity(predictions["z_global"])
                    metrics["mon_latent_concentration"] = mon[
                        "latent_concentration"]
            if (self.loss_weights.hebbian_reg == 0
                    and "hebbian_weights" in predictions):
                with torch.no_grad():
                    metrics["mon_hebbian_weight_norm"] = torch.norm(
                        predictions["hebbian_weights"]
                    ).item()
        except Exception as exc:  # pragma: no cover - monitors never crash runs
            metrics["mon_error"] = str(exc)[:80]


if __name__ == "__main__":
    print("Testing loss functions...")

    recon = ReconstructionLoss()
    x = torch.randn(4, 19, 256)
    y = torch.randn(4, 19, 256)
    loss = recon(x, y)
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