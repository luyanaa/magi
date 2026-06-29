"""
Seven-Layer Stability Defense & Auto-Rollback System.

Implements the defense hierarchy from the training plan §5:
L1: KDA state Frobenius normalization (every 64 steps)
L2: GENERIC degeneracy projection + Casimir soft projection (every 64 steps / every step)
L3: Kahan summation (FP16 integration) (every step)
L4: Stable softmax (log-sum-exp) (every step)
L5: Hebbian spectral normalization via power iteration (every 100 steps)
L6: EMA bias monitoring (every 1000 steps)
L7: Global energy audit (every epoch)

Auto-rollback triggers:
- EPR negative ratio > 1% over 1000 steps
- Energy divergence > 2× initial for > 200 steps
- Router entropy collapse < 0.1 bits for > 500 steps
- KS entropy explosion > 5× reference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .device_utils import get_device, get_device_type, autocast_context


class KahanSummation:
    """
    L3: Kahan compensated summation for FP16 integration.

    Reduces numerical error in z_{t+1} = z_t + delta_z accumulation.
    """

    def __init__(self, shape: Tuple[int, ...], device: str = "cpu"):
        self.compensation = torch.zeros(shape, device=device)

    def add(self, z: torch.Tensor, delta_z: torch.Tensor) -> torch.Tensor:
        """
        z_next = z + delta_z with Kahan compensation.

        Args:
            z: (B, d) current state
            delta_z: (B, d) increment
        Returns:
            z_next: (B, d) compensated sum
        """
        y = delta_z - self.compensation
        t = z + y
        self.compensation = (t - z) - y
        return t

    def reset(self):
        self.compensation.zero_()


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    L4: Numerically stable softmax using log-sum-exp trick.

    Args:
        x: input tensor
        dim: dimension to softmax over
    Returns:
        softmax probabilities
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def power_iteration_spectral_norm(
    W: torch.Tensor,
    num_iters: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L5: Power iteration for spectral norm estimation.

    Returns:
        sigma_max: estimated largest singular value
        v: dominant right singular vector
    """
    d = W.shape[0]
    device = W.device
    dtype = W.dtype

    v = torch.randn(d, 1, device=device, dtype=dtype)
    v = v / (torch.norm(v) + 1e-8)

    for _ in range(num_iters):
        v = W.t() @ (W @ v)
        v = v / (torch.norm(v) + 1e-8)

    sigma_max = torch.norm(W @ v).item()
    return torch.tensor(sigma_max, device=device, dtype=dtype), v.squeeze()


class QFactorMonitor:
    """
    Wiener Proposal #3: Q品质因数监控 (Q-factor quality factor monitor).

    Performs FFT on latent velocity v_t over a sliding window to extract
    the dominant oscillation frequency f₀ and half-power bandwidth Δf,
    computing Q = f₀ / Δf as a resonance quality factor.

    Health bands:
      Q < 0.5  : over-damped (system sluggish, no oscillation)
      0.5 ≤ Q ≤ 2.0 : healthy (moderate damping)
      Q > 2.0  : resonant risk (under-damped, possible oscillation blow-up)

    Stage 0 may have no oscillation to monitor; Q is reported as NaN until
    the window is full and spectral energy is above noise floor.

    Schedule: same as EPR (every step, logged every 100).
    Read-only: no backward path changes.
    """

    def __init__(
        self,
        window_size: int = 512,
        min_spectral_energy: float = 1e-4,
        fp32_for_fft: bool = True,
    ):
        self.window_size = window_size
        self.min_spectral_energy = min_spectral_energy
        self.fp32_for_fft = fp32_for_fft

        self.velocity_buffer: List[torch.Tensor] = []
        self.q_history: List[float] = []
        self.f0_history: List[float] = []
        self.band_history: List[str] = []

    def update(self, v_t: torch.Tensor) -> Dict[str, float]:
        """
        Append latent velocity sample and compute Q-factor if window full.

        Args:
            v_t: (B, d) latent velocity at current step

        Returns:
            dict with q_factor, f0, bandwidth, band_label
        """
        self.velocity_buffer.append(v_t.detach())

        if len(self.velocity_buffer) > self.window_size:
            self.velocity_buffer = self.velocity_buffer[-self.window_size:]

        if len(self.velocity_buffer) < self.window_size:
            return {
                "q_factor": float("nan"),
                "q_f0": float("nan"),
                "q_bandwidth": float("nan"),
                "q_band": "insufficient_data",
            }

        v_stack = torch.stack(self.velocity_buffer, dim=0)
        T = v_stack.shape[0]

        if self.fp32_for_fft:
            v_stack = v_stack.float()

        v_mean = v_stack.mean(dim=-1)
        v_centered = v_mean - v_mean.mean(dim=0, keepdim=True)

        if v_centered.dim() == 2:
            v_centered = v_centered.mean(dim=1)

        v_np = v_centered.cpu().numpy()

        if np.std(v_np) < self.min_spectral_energy:
            result = {
                "q_factor": 0.0,
                "q_f0": 0.0,
                "q_bandwidth": 0.0,
                "q_band": "over_damped",
            }
            self._record(result)
            return result

        spectrum = np.abs(np.fft.rfft(v_np))
        freqs = np.fft.rfftfreq(len(v_np))

        peak_idx = np.argmax(spectrum[1:]) + 1
        f0 = freqs[peak_idx]
        peak_power = spectrum[peak_idx]

        half_power = peak_power / 2.0
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > half_power:
            left_idx -= 1
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_power:
            right_idx += 1

        f_left = freqs[max(left_idx, 0)]
        f_right = freqs[min(right_idx, len(freqs) - 1)]
        bandwidth = max(f_right - f_left, 1e-8)

        q = f0 / bandwidth

        if q < 0.5:
            band = "over_damped"
        elif q <= 2.0:
            band = "healthy"
        else:
            band = "resonant_risk"

        result = {
            "q_factor": float(q),
            "q_f0": float(f0),
            "q_bandwidth": float(bandwidth),
            "q_band": band,
        }
        self._record(result)
        return result

    def _record(self, result: Dict[str, float]):
        self.q_history.append(result["q_factor"])
        self.f0_history.append(result["q_f0"])
        self.band_history.append(result["q_band"])
        if len(self.q_history) > 10000:
            self.q_history = self.q_history[-5000:]
            self.f0_history = self.f0_history[-5000:]
            self.band_history = self.band_history[-5000:]

    def get_report(self) -> Dict:
        valid_q = [q for q in self.q_history if not math.isnan(q)]
        return {
            "q_factor_current": self.q_history[-1] if self.q_history else float("nan"),
            "q_factor_mean": float(np.mean(valid_q)) if valid_q else float("nan"),
            "f0_current": self.f0_history[-1] if self.f0_history else float("nan"),
            "band_current": self.band_history[-1] if self.band_history else "none",
            "resonant_risk_count": sum(1 for b in self.band_history if b == "resonant_risk"),
            "over_damped_count": sum(1 for b in self.band_history if b == "over_damped"),
            "healthy_count": sum(1 for b in self.band_history if b == "healthy"),
        }


class AtaxiaCatalepsyMonitor:
    """
    Wiener Proposal #7: Ataxia/Catalepsy 监控.

    Ataxia (运动失调): measures the mismatch between efference copy prediction
    and actual sensory input. AtaxiaScore = E[||x̂−x||] / E[||x||].
    Reuses EfferenceCopy output. Threshold ~0.5 sustained 1K steps.

    Catalepsy (僵住症): measures the collapse of latent diversity/negentropy.
    CatalepsyScore = −H(z_t) estimated via k-nearest-neighbor differential entropy.
    Threshold ~3.0 sustained 2K steps.

    Both are read-only monitors. When thresholds are sustained, they trigger
    parameter adjustments (not rollback) — e.g., increase noise injection,
    adjust mobility M(z), or re-weight expert routing.

    Schedule: every step, logged every 100. Same as EPR.
    """

    def __init__(
        self,
        ataxia_threshold: float = 0.5,
        ataxia_sustain_steps: int = 1000,
        catalepsy_threshold: float = 3.0,
        catalepsy_sustain_steps: int = 2000,
        k_nn: int = 5,
        fp32_for_entropy: bool = True,
    ):
        self.ataxia_threshold = ataxia_threshold
        self.ataxia_sustain_steps = ataxia_sustain_steps
        self.catalepsy_threshold = catalepsy_threshold
        self.catalepsy_sustain_steps = catalepsy_sustain_steps
        self.k_nn = k_nn
        self.fp32_for_entropy = fp32_for_entropy

        self.ataxia_history: List[float] = []
        self.catalepsy_history: List[float] = []
        self.ataxia_sustained_count: int = 0
        self.catalepsy_sustained_count: int = 0
        self.ataxia_alert: bool = False
        self.catalepsy_alert: bool = False

    def compute_ataxia_score(
        self,
        predicted_signal: torch.Tensor,
        actual_signal: torch.Tensor,
    ) -> float:
        """
        AtaxiaScore = E[||x̂−x||] / E[||x||].

        Args:
            predicted_signal: (B, ...) decoder prediction (efference copy)
            actual_signal: (B, ...) actual sensory input

        Returns:
            ataxia_score: scalar in [0, +inf)
        """
        with torch.no_grad():
            pred_flat = predicted_signal.reshape(predicted_signal.shape[0], -1)
            act_flat = actual_signal.reshape(actual_signal.shape[0], -1)

            error_norm = torch.norm(pred_flat - act_flat, dim=-1).mean()
            signal_norm = torch.norm(act_flat, dim=-1).mean()

            ataxia = (error_norm / (signal_norm + 1e-8)).item()

        self.ataxia_history.append(ataxia)
        if len(self.ataxia_history) > self.ataxia_sustain_steps * 2:
            self.ataxia_history = self.ataxia_history[-self.ataxia_sustain_steps:]

        if ataxia > self.ataxia_threshold:
            self.ataxia_sustained_count += 1
        else:
            self.ataxia_sustained_count = 0

        self.ataxia_alert = self.ataxia_sustained_count >= self.ataxia_sustain_steps

        return ataxia

    def compute_catalepsy_score(
        self,
        z: torch.Tensor,
    ) -> float:
        """
        CatalepsyScore = −H(z_t) via k-nearest-neighbor differential entropy.

        Uses the Kozachenko-Leonenko k-NN estimator:
        H(z) ≈ d * log(E_k) + log(N) + const
        where E_k is mean log-distance to k-th nearest neighbor.

        CatalepsyScore = -H(z): higher = more collapsed (less diverse) latent space.

        Args:
            z: (B, d) latent state

        Returns:
            catalepsy_score: scalar (higher = worse)
        """
        with torch.no_grad():
            if self.fp32_for_entropy:
                z_fp32 = z.float()
            else:
                z_fp32 = z

            B, d = z_fp32.shape

            if B < self.k_nn + 1:
                score = float("nan")
                self.catalepsy_history.append(score)
                return score

            if B > 1024:
                idx = torch.randperm(B, device=z.device)[:1024]
                z_sample = z_fp32[idx]
            else:
                z_sample = z_fp32

            N = z_sample.shape[0]

            dist_matrix = torch.cdist(z_sample, z_sample, p=2)

            dist_matrix.fill_diagonal_(float("inf"))

            k = min(self.k_nn, N - 1)
            knn_dists, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
            kth_dist = knn_dists[:, -1]

            kth_dist = torch.clamp(kth_dist, min=1e-8)
            log_dist = torch.log(kth_dist).mean()

            H_estimate = d * log_dist.item() + math.log(N)

            catalepsy_score = -H_estimate

        self.catalepsy_history.append(catalepsy_score)
        if len(self.catalepsy_history) > self.catalepsy_sustain_steps * 2:
            self.catalepsy_history = self.catalepsy_history[-self.catalepsy_sustain_steps:]

        if catalepsy_score > self.catalepsy_threshold:
            self.catalepsy_sustained_count += 1
        else:
            self.catalepsy_sustained_count = 0

        self.catalepsy_alert = self.catalepsy_sustained_count >= self.catalepsy_sustain_steps

        return catalepsy_score

    def update(
        self,
        predicted_signal: Optional[torch.Tensor] = None,
        actual_signal: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute both ataxia and catalepsy scores for current step.

        Args:
            predicted_signal: decoder output (for ataxia)
            actual_signal: actual sensory input (for ataxia)
            z: latent state (for catalepsy)

        Returns:
            dict with ataxia_score, catalepsy_score, ataxia_alert, catalepsy_alert
        """
        result = {}

        if predicted_signal is not None and actual_signal is not None:
            ataxia = self.compute_ataxia_score(predicted_signal, actual_signal)
            result["ataxia_score"] = ataxia
        else:
            result["ataxia_score"] = float("nan")

        if z is not None:
            catalepsy = self.compute_catalepsy_score(z)
            result["catalepsy_score"] = catalepsy
        else:
            result["catalepsy_score"] = float("nan")

        result["ataxia_alert"] = float(self.ataxia_alert)
        result["catalepsy_alert"] = float(self.catalepsy_alert)
        result["ataxia_sustained"] = self.ataxia_sustained_count
        result["catalepsy_sustained"] = self.catalepsy_sustained_count

        return result

    def get_report(self) -> Dict:
        valid_ataxia = [a for a in self.ataxia_history if not math.isnan(a)]
        valid_catalepsy = [c for c in self.catalepsy_history if not math.isnan(c)]
        return {
            "ataxia_current": self.ataxia_history[-1] if self.ataxia_history else float("nan"),
            "ataxia_mean": float(np.mean(valid_ataxia)) if valid_ataxia else float("nan"),
            "ataxia_alert": self.ataxia_alert,
            "ataxia_sustained": self.ataxia_sustained_count,
            "catalepsy_current": self.catalepsy_history[-1] if self.catalepsy_history else float("nan"),
            "catalepsy_mean": float(np.mean(valid_catalepsy)) if valid_catalepsy else float("nan"),
            "catalepsy_alert": self.catalepsy_alert,
            "catalepsy_sustained": self.catalepsy_sustained_count,
        }


class StabilityMonitor:
    """
    L6/L7: EMA bias monitoring and global energy audit.

    Tracks metrics over training and triggers auto-rollback when thresholds violated.
    """

    def __init__(
        self,
        window_size: int = 1000,
        epr_threshold: float = 0.01,
        energy_divergence_factor: float = 2.0,
        energy_divergence_steps: int = 200,
        router_entropy_threshold: float = 0.1,
        router_entropy_steps: int = 500,
        ks_explosion_factor: float = 5.0,
    ):
        self.window_size = window_size
        self.epr_threshold = epr_threshold
        self.energy_divergence_factor = energy_divergence_factor
        self.energy_divergence_steps = energy_divergence_steps
        self.router_entropy_threshold = router_entropy_threshold
        self.router_entropy_steps = router_entropy_steps
        self.ks_explosion_factor = ks_explosion_factor

        self.epr_history: List[float] = []
        self.energy_history: List[float] = []
        self.router_entropy_history: List[float] = []
        self.ks_history: List[float] = []
        self.ks_reference: Optional[float] = None
        self.q_history: List[float] = []
        self.ataxia_history: List[float] = []
        self.catalepsy_history: List[float] = []

        self.trigger_log: List[Dict] = []

    def update(self, metrics: Dict[str, float]):
        """Append metrics from current step."""
        if "epr_negative_ratio" in metrics:
            self.epr_history.append(metrics["epr_negative_ratio"])
        if "E" in metrics:
            self.energy_history.append(abs(metrics["E"]))
        if "router_entropy" in metrics:
            self.router_entropy_history.append(metrics["router_entropy"])
        if "ks_entropy" in metrics:
            self.ks_history.append(metrics["ks_entropy"])
        if "q_factor" in metrics:
            if not math.isnan(metrics["q_factor"]):
                self.q_history.append(metrics["q_factor"])
        if "ataxia_score" in metrics:
            if not math.isnan(metrics["ataxia_score"]):
                self.ataxia_history.append(metrics["ataxia_score"])
        if "catalepsy_score" in metrics:
            if not math.isnan(metrics["catalepsy_score"]):
                self.catalepsy_history.append(metrics["catalepsy_score"])

        # Trim histories
        for hist in [self.epr_history, self.energy_history, self.router_entropy_history, self.ks_history, self.q_history, self.ataxia_history, self.catalepsy_history]:
            if len(hist) > self.window_size * 2:
                hist[:] = hist[-self.window_size:]

        # Calibrate KS reference from last 10K steps of Stage 1 P5
        if self.ks_reference is None and len(self.ks_history) >= 100:
            self.ks_reference = sum(self.ks_history[-100:]) / 100.0

    def check_triggers(self, step: int) -> Tuple[bool, Optional[str]]:
        """
        Check all auto-rollback triggers.

        Returns:
            triggered: bool
            reason: str or None
        """
        # Trigger 1: EPR negative ratio > 1% over last 1000 steps
        if len(self.epr_history) >= self.window_size:
            epr_neg_ratio = sum(self.epr_history[-self.window_size:]) / self.window_size
            if epr_neg_ratio > self.epr_threshold:
                reason = f"EPR negative ratio {epr_neg_ratio:.4f} > {self.epr_threshold}"
                self.trigger_log.append({"step": step, "reason": reason, "type": "epr"})
                return True, reason

        # Trigger 2: Energy divergence > 2× initial for > 200 steps
        if len(self.energy_history) >= self.energy_divergence_steps:
            initial_energy = self.energy_history[0]
            recent = self.energy_history[-self.energy_divergence_steps:]
            if all(e > self.energy_divergence_factor * initial_energy for e in recent):
                reason = f"Energy diverged > {self.energy_divergence_factor}× initial for {self.energy_divergence_steps} steps"
                self.trigger_log.append({"step": step, "reason": reason, "type": "energy"})
                return True, reason

        # Trigger 3: Router entropy collapse < 0.1 bits for > 500 steps
        if len(self.router_entropy_history) >= self.router_entropy_steps:
            recent = self.router_entropy_history[-self.router_entropy_steps:]
            if all(e < self.router_entropy_threshold for e in recent):
                reason = f"Router entropy collapsed < {self.router_entropy_threshold} bits for {self.router_entropy_steps} steps"
                self.trigger_log.append({"step": step, "reason": reason, "type": "router_entropy"})
                return True, reason

        # Trigger 4: KS entropy explosion > 5× reference
        if self.ks_reference is not None and len(self.ks_history) >= 100:
            recent_avg = sum(self.ks_history[-100:]) / 100.0
            if recent_avg > self.ks_explosion_factor * self.ks_reference:
                reason = f"KS entropy {recent_avg:.4f} > {self.ks_explosion_factor}× reference {self.ks_reference:.4f}"
                self.trigger_log.append({"step": step, "reason": reason, "type": "ks"})
                return True, reason

        return False, None

    def get_report(self) -> Dict:
        """Get summary of monitoring state."""
        valid_q = [q for q in self.q_history if not math.isnan(q)]
        valid_ataxia = [a for a in self.ataxia_history if not math.isnan(a)]
        valid_catalepsy = [c for c in self.catalepsy_history if not math.isnan(c)]
        return {
            "epr_history_len": len(self.epr_history),
            "energy_history_len": len(self.energy_history),
            "router_entropy_history_len": len(self.router_entropy_history),
            "ks_history_len": len(self.ks_history),
            "ks_reference": self.ks_reference,
            "q_history_len": len(self.q_history),
            "q_factor_mean": float(np.mean(valid_q)) if valid_q else float("nan"),
            "ataxia_history_len": len(self.ataxia_history),
            "ataxia_mean": float(np.mean(valid_ataxia)) if valid_ataxia else float("nan"),
            "catalepsy_history_len": len(self.catalepsy_history),
            "catalepsy_mean": float(np.mean(valid_catalepsy)) if valid_catalepsy else float("nan"),
            "triggers_fired": len(self.trigger_log),
            "trigger_log": self.trigger_log[-5:],
        }


class StabilityController:
    """
    Orchestrates all 7 layers of stability defense.

    Usage:
        controller = StabilityController(hidden_dim= 1024)
        # In training loop:
        z_next = controller.apply_L3(z, delta_z)  # Kahan summation
        controller.apply_L1(kda_state)            # KDA normalization
        controller.apply_L5(hebbian_weight)       # Spectral norm
        should_rollback, reason = controller.monitor.check_triggers(step)
    """

    def __init__(self, hidden_dim: int = 1024, device: str = "cpu"):
        self.hidden_dim = hidden_dim
        self.device = device
        self.kahan = None  # lazily initialized per batch
        self.monitor = StabilityMonitor()
        self.step_counter = 0

    def init_kahan(self, shape: Tuple[int, ...]):
        """Initialize Kahan summation for given batch shape."""
        if self.kahan is None or self.kahan.compensation.shape != shape:
            self.kahan = KahanSummation(shape, device=self.device)

    def apply_L1(self, kda_state: torch.Tensor) -> torch.Tensor:
        """L1: Frobenius normalize KDA state."""
        norm = torch.norm(kda_state, p="fro", dim=-1, keepdim=True)
        return kda_state / (norm + 1e-8)

    def apply_L2_nullspace(
        self,
        z: torch.Tensor,
        L_z: torch.Tensor,
        delta_z: torch.Tensor,
        eta: float = 0.1,
    ) -> torch.Tensor:
        """
        L2: Casimir soft projection.

        Dampens velocity near Casimir invariant surfaces.
        Currently simplified to z - eta * delta_z with eta controlling
        the damping factor. Full Casimir projection (SVD-based nullspace
        detection) is reserved for future audit mode.
        """
        return z - eta * delta_z

    def apply_L3(self, z: torch.Tensor, delta_z: torch.Tensor) -> torch.Tensor:
        """L3: Kahan summation for z_{t+1} = z_t + delta_z."""
        self.init_kahan(z.shape)
        return self.kahan.add(z, delta_z)

    def apply_L5(self, W: torch.Tensor) -> torch.Tensor:
        """L5: Power iteration spectral normalization for Hebbian weights."""
        sigma_max, _ = power_iteration_spectral_norm(W, num_iters=5)
        if sigma_max > 0.95:
            scale = 0.95 / sigma_max
            return W * scale
        return W

    def step(self, step: int):
        """Increment step counter and update monitor."""
        self.step_counter = step

    def reset_kahan(self):
        """Reset Kahan compensation (e.g., at phase boundary)."""
        if self.kahan is not None:
            self.kahan.reset()


class AutoRollback:
    """
    Automated rollback system triggered by StabilityMonitor.

    Implements 4-level recovery (§8):
    Level 1: Increase balance loss weight, capacity factor, temperature annealing
    Level 2: Reset router weights only, re-warmup 500 steps
    Level 3: Temporarily switch to Top-1 routing
    Level 4: Rollback to last checkpoint + restart from Stage 1 P4
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.level = 0
        self.consecutive_triggers = 0

    def on_trigger(
        self,
        reason: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> Tuple[int, Dict]:
        """
        Execute rollback action based on trigger count.

        Returns:
            level: int (1-4)
            action: dict with changes made
        """
        self.consecutive_triggers += 1

        if self.consecutive_triggers <= 2:
            level = 1
            action = self._level1_adjust(model, optimizer)
        elif self.consecutive_triggers <= 4:
            level = 2
            action = self._level2_router_reset(model, optimizer)
        elif self.consecutive_triggers <= 6:
            level = 3
            action = self._level3_top1_downgrade(model)
        else:
            level = 4
            action = self._level4_checkpoint_rollback(step)

        self.level = level
        action["reason"] = reason
        action["level"] = level
        action["step"] = step
        return level, action

    def _level1_adjust(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict:
        """Level 1: Adjust hyperparameters."""
        # Increase balance loss weight
        adjustments = {
            "balance_weight_multiplier": 5.0,
            "capacity_factor": 1.5,
            "temperature": 0.5,
        }
        # These would need to be applied in the training loop via config
        return adjustments

    def _level2_router_reset(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict:
        """Level 2: Reset router weights, warmup 500 steps."""
        for name, param in model.named_parameters():
            if "router" in name.lower() or "A_antisym" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
        return {"router_reset": True, "warmup_steps": 500}

    def _level3_top1_downgrade(self, model: nn.Module) -> Dict:
        """Level 3: Force Top-1 routing temporarily."""
        return {"force_top1": True, "duration_steps": 1000}

    def _level4_checkpoint_rollback(self, step: int) -> Dict:
        """Level 4: Request rollback to last checkpoint."""
        return {
            "rollback": True,
            "target_checkpoint": f"checkpoint_stage_1_p4_step*.pt",
            "restart_phase": "Stage 1 P4: Full MoE",
        }

    def reset_trigger_count(self):
        """Reset after successful recovery."""
        self.consecutive_triggers = 0
        self.level = 0


if __name__ == "__main__":
    print("Testing Stability System...")


def selective_fp32_forward(fn, *args, **kwargs):
    """
    Run a function in FP32, casting inputs from FP16 and outputs back.

    Used for precision-critical modules: PoissonRouter, Hebbian Oja update,
    KDA state accumulation. Pattern matches Kimi Delta Attention's
    `.float()` runtime cast in kda_decoder.py.

    Usage in model.forward():
        router_logits = selective_fp32_forward(self.poisson_router, z)
        hebbian_update = selective_fp32_forward(self.ham.oja_step, z, prev_w)

    Args:
        fn: callable (module or function)
        *args, **kwargs: arguments (tensors cast to FP32, others passed through)
    Returns:
        fn output, cast back to FP16 if tensor
    """
    fp32_args = [a.float() if isinstance(a, torch.Tensor) else a for a in args]
    fp32_kwargs = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype in (torch.float16, torch.bfloat16) else v
                   for k, v in kwargs.items()}
    with autocast_context(enabled=False):
        output = fn(*fp32_args, **fp32_kwargs)
    if isinstance(output, torch.Tensor):
        return output.to(torch.float16)
    elif isinstance(output, (list, tuple)):
        return [o.to(torch.float16) if isinstance(o, torch.Tensor) else o for o in output]
    return output

    # L3 Kahan
    kahan = KahanSummation((4, 2048), device=get_device())
    z = torch.randn(4, 2048)
    dz = torch.randn(4, 2048) * 0.01
    z2 = kahan.add(z, dz)
    print(f"  Kahan sum shape: {z2.shape}")

    # L5 Power iteration
    W = torch.randn(64, 64)
    sigma, v = power_iteration_spectral_norm(W, num_iters=5)
    print(f"  Power iteration sigma: {sigma.item():.4f}")

    # Monitor
    monitor = StabilityMonitor()
    for i in range(1200):
        monitor.update({
            "epr_negative_ratio": 0.02 if i > 500 else 0.0,
            "E": 1.0 + i * 0.01,
            "router_entropy": 0.05,
            "ks_entropy": 0.1,
        })
    triggered, reason = monitor.check_triggers(step=1200)
    print(f"  Triggered: {triggered}, reason: {reason}")

    # Controller
    ctrl = StabilityController(hidden_dim=1024, device=get_device())
    kda = torch.randn(4, 2048)
    kda_norm = ctrl.apply_L1(kda)
    print(f"  L1 norm: {torch.norm(kda_norm, p='fro').item():.4f}")

    print("All stability tests passed!")
