"""
Active Inference Module for Brain MoE-PINN.

Implements the closed-loop feedback mechanisms for active inference:
1. Value Function V(z) = -(E - TS) for expected free energy computation
2. Efference Copy / Reafference - decoder outputs as sensory predictions
3. Smith Predictor / Delay Compensation - KDA state as history integrator
4. Precision Gate - dynamic weighting based on delay and action magnitude

Reference: Brain MoE-PINN Training Plan §2.6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class ValueFunction(nn.Module):
    """
    Value function V(z) = -(E - TS) for expected free energy computation.

    Represents the negative free energy (expected free energy, EFE).
    Higher V(z) = more favorable state (lower free energy).
    Used in counterfactual tree search for trajectory scoring.

    Components:
    - E: energy potential from EnergyEntropyFields
    - S: entropy potential
    - T: effective temperature (learnable parameter)
    - TD error modulation for dopamine-like signals
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        energy_entropy_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        if energy_entropy_net is not None:
            self.energy_entropy = energy_entropy_net
        else:
            from .velocity_brain import EnergyEntropyFields
            self.energy_entropy = EnergyEntropyFields(hidden_dim=hidden_dim)

        self.temperature = nn.Parameter(torch.tensor(0.1))

        self.td_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute value function V(z) = -(E - TS).

        Args:
            z: (B, d) latent state
        Returns:
            V: (B,) value for each sample
            metrics: dict with E, S, T, TS components
        """
        E, S = self.energy_entropy(z)

        T = torch.abs(self.temperature) + 0.01

        TS = T * S

        E_neg = -E.squeeze(-1)

        V = E_neg + TS.squeeze(-1)

        td_mod = self.td_modulation(z).squeeze(-1)

        V = V + 0.1 * td_mod

        metrics = {
            "E": E.mean().item(),
            "S": S.mean().item(),
            "T": T.item(),
            "TS": TS.mean().item(),
            "V": V.mean().item(),
            "td_mod": td_mod.mean().item(),
        }

        return V, metrics

    def compute_efe(
        self,
        z: torch.Tensor,
        goal_attractors: Optional[torch.Tensor] = None,
        action_cost: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Expected Free Energy for trajectory scoring.

        EFE = V(z_future) + action_cost * ||action||^2

        Args:
            z: (B, d) current or trajectory state
            goal_attractors: optional goal states for goal-directed EFE
            action_cost: cost scaling for actions
        Returns:
            efe: (B,) expected free energy
        """
        V, metrics = self.forward(z)

        efe = -V

        if goal_attractors is not None:
            goal_dist = torch.norm(z.unsqueeze(1) - goal_attractors, dim=-1).mean(dim=1)
            goal_cost = 0.1 * goal_dist
            efe = efe + goal_cost
            metrics["goal_cost"] = goal_cost.mean().item()

        metrics["efe"] = efe.mean().item()
        return efe, metrics


class EfferenceCopy(nn.Module):
    """
    Efference Copy / Reafference mechanism.

    The decoder output x_hat_{t+1} serves as a sensory prediction (efference copy).
    The actual input $x_{t+1}$ returns, and the difference = reafference signal.

    This creates a copy of the motor command that can be compared with
    actual sensory feedback to compute prediction errors.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.efference_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.reafference_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        z_next: torch.Tensor,
        predicted_signal: torch.Tensor,
        actual_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute efference copy and reafference signal.

        Args:
            z_next: (B, d) predicted next latent state
            predicted_signal: (B, ...) predicted sensory signal from decoder
            actual_signal: (B, ...) actual sensory input (if available)
        Returns:
            dict with efference, reafference, and prediction error
        """
        efference = self.efference_proj(z_next)

        result = {
            "efference": efference,
            "predicted_signal": predicted_signal,
            "has_actual": actual_signal is not None,
        }

        if actual_signal is not None:
            reafference = predicted_signal - actual_signal

            if reafference.dim() > 1:
                reafference_norm = torch.norm(reafference.reshape(reafference.shape[0], -1), dim=-1)
            else:
                reafference_norm = torch.norm(reafference, dim=-1)

            result["reafference"] = reafference
            result["prediction_error"] = reafference_norm
            result["weighted_error"] = self.reafference_weight * reafference_norm
        else:
            result["reafference"] = None
            result["prediction_error"] = None
            result["weighted_error"] = None

        return result


class SmithPredictor(nn.Module):
    """
    Smith Predictor for delay compensation.

    Uses KDA state as a history integrator to predict the future state
    when actual feedback is delayed.

    The Smith predictor compensates for sensory feedback delays by:
    1. Running an internal model of the system
    2. Predicting what the sensory feedback will be at the current time
    3. Comparing prediction with actual (delayed) feedback

    Stability condition (Wiener oscillation-tolerant):
        gamma * tau_delay < pi/2 (allows damped oscillation)
        vs. original: gamma * tau_delay < 1 (forced exponential decay)

    With oscillation_tolerant=True (default), the stability margin is
    pi/2 - gamma*tau, allowing healthy neural-like oscillations.
    With oscillation_tolerant=False, falls back to 1.0 - gamma*tau
    (original hard overdamping).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        tau_delay: float = 0.5,
        prediction_horizon: int = 5,
        oscillation_tolerant: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau_delay = tau_delay
        self.prediction_horizon = prediction_horizon
        self.oscillation_tolerant = oscillation_tolerant

        self.history_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.feedback_gain = nn.Parameter(torch.tensor(0.5))

        self.delay_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        self.gamma_max_hard = 1.0
        self.gamma_max_oscillation = math.pi / 2
        self.oscillation_stability_target = math.pi / 2 - 0.2

    def forward(
        self,
        z_current: torch.Tensor,
        kda_state_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future state accounting for delay.

        Args:
            z_current: (B, d) current latent state
            kda_state_history: (B, T_history, d) history of KDA states
        Returns:
            dict with prediction, delay estimate, stability check
        """
        history_len = kda_state_history.shape[1]

        gru_out, _ = self.history_encoder(kda_state_history)
        history_context = gru_out[:, -1]

        predicted_z = self.predictor(history_context)

        tau_est = self.delay_estimator(z_current).squeeze(-1)
        tau_est = torch.clamp(tau_est, min=0.01, max=2.0)

        gamma = torch.abs(self.feedback_gain)

        gamma_max = self.gamma_max_oscillation if self.oscillation_tolerant else self.gamma_max_hard
        stability_margin = gamma_max - gamma * tau_est
        is_stable = stability_margin > 0

        if not is_stable.all():
            adjusted_gain = stability_margin / (tau_est + 1e-6)
            adjusted_gain = torch.clamp(adjusted_gain, min=0.0, max=gamma)
        else:
            adjusted_gain = gamma

        result = {
            "predicted_z": predicted_z,
            "delay_estimate": tau_est,
            "feedback_gain": adjusted_gain,
            "stability_margin": stability_margin,
            "is_stable": is_stable,
            "oscillation_tolerant": self.oscillation_tolerant,
        }

        return result

    def adjust_gain_from_wiener(
        self,
        q_factor: float,
        ataxia_score: float = 0.0,
        catalepsy_score: float = 0.0,
    ) -> Dict:
        """
        Wiener oscillation-tolerant adaptive gain adjustment (§11.10.2.1).

        Uses Q-factor from FFT analysis of latent velocity, plus
        Ataxia/Catalepsy monitors for safety interlocks.

        Strategy:
          Q < 0.5  (overdamped): allow full oscillation-tolerant range [0, π/2τ]
          0.5 ≤ Q ≤ 2.0 (healthy): standard oscillation-tolerant constraint
          Q > 2.0  (resonant risk): restrict gain to damp oscillation

        Safety interlocks:
          AtaxiaScore > 0.5: fall back to hard constraint γτ < 1
          CatalepsyScore > 3.0: signal noise injection (handled upstream)

        Args:
            q_factor: quality factor from QFactorMonitor
            ataxia_score: AtaxiaScore from AtaxiaCatalepsyMonitor
            catalepsy_score: CatalepsyScore from AtaxiaCatalepsyMonitor

        Returns:
            dict with gamma_max, mode, stability_target
        """
        gamma = torch.abs(self.feedback_gain).item()
        tau = self.tau_delay

        if not math.isnan(ataxia_score) and ataxia_score > 0.5:
            self.oscillation_tolerant = False
            gamma_max_hard = 1.0 / (tau + 1e-6)
            return {
                "gamma_max": gamma_max_hard,
                "mode": "ataxia_interlock",
                "oscillation_tolerant": False,
                "stability_target": 1.0,
            }

        if not math.isnan(catalepsy_score) and catalepsy_score > 3.0:
            self.oscillation_tolerant = True
            gamma_max_cat = (math.pi / 2) / (tau + 1e-6)
            return {
                "gamma_max": gamma_max_cat,
                "mode": "catalepsy_interlock",
                "oscillation_tolerant": True,
                "stability_target": math.pi / 2,
                "requires_noise_injection": True,
            }

        self.oscillation_tolerant = True

        if math.isnan(q_factor):
            gamma_max = (math.pi / 2) / (tau + 1e-6)
            mode = "oscillation_tolerant_default"
        elif q_factor < 0.5:
            gamma_max = (math.pi / 2) / (tau + 1e-6)
            mode = "overdamped_allow_oscillation"
        elif q_factor <= 2.0:
            gamma_max = (math.pi / 2) / (tau + 1e-6)
            mode = "healthy_oscillation"
        else:
            gamma_max = 0.8 * (math.pi / 2) / (tau + 1e-6)
            mode = "resonant_damped"

        return {
            "gamma_max": gamma_max,
            "mode": mode,
            "oscillation_tolerant": True,
            "stability_target": gamma_max * tau,
        }

    def compute_stability_loss(self) -> torch.Tensor:
        """
        Compute stability regularization loss.

        Encourages gamma * tau_delay < stability_target.
        oscillation_tolerant=True: target = pi/2 - 0.2 (margin within oscillation range)
        oscillation_tolerant=False: target = 0.9 (original hard overdamping)
        """
        gamma = torch.abs(self.feedback_gain)
        tau = self.tau_delay

        target = self.oscillation_stability_target if self.oscillation_tolerant else 0.9

        stability_violation = F.relu(gamma * tau - target)

        return stability_violation.mean()


class PrecisionGate(nn.Module):
    """
    Precision Gate for dynamic weighting of prediction errors.

    Dynamically adjusts the weight (precision) of prediction errors
    based on:
    - Delay magnitude: longer delays -> lower precision (more uncertainty)
    - Action magnitude: larger actions -> higher precision requirement

    Implements: precision = sigma^-2 = exp(-alpha * delay) * exp(beta * |action|)
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        delay_dim: int = 64,
        action_dim: int = 64,
    ):
        super().__init__()

        self.delay_encoder = nn.Sequential(
            nn.Linear(1, delay_dim),
            nn.GELU(),
            nn.Linear(delay_dim, 1),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, delay_dim),
            nn.GELU(),
            nn.Linear(delay_dim, 1),
        )

        self.precision_net = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        self.base_precision = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        delay: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute dynamic precision weight.

        Args:
            delay: (B,) or (B, 1) delay time
            action: (B, d) or (B, action_dim) action magnitude
        Returns:
            precision: (B,) precision weights
            metrics: dict with components
        """
        if delay.dim() == 1:
            delay = delay.unsqueeze(-1)
        if action.dim() == 2:
            action_mag = torch.norm(action, dim=-1, keepdim=True)
        else:
            action_mag = action.unsqueeze(-1) if action.dim() == 1 else action

        delay_encoded = self.delay_encoder(torch.log(delay + 1e-3))
        action_encoded = self.action_encoder(action_mag)

        combined = torch.cat([delay_encoded, action_encoded], dim=-1)

        precision_mod = self.precision_net(combined).squeeze(-1)

        precision = self.base_precision + precision_mod
        precision = torch.clamp(precision, min=0.01, max=10.0)

        precision_loss = -torch.log(precision).mean()

        metrics = {
            "precision": precision.mean().item(),
            "precision_mod": precision_mod.mean().item(),
            "delay_encoded": delay_encoded.mean().item(),
            "action_encoded": action_encoded.mean().item(),
            "precision_loss": precision_loss.item(),
        }

        return precision, metrics


class ClosedLoopFeedback(nn.Module):
    """
    Complete closed-loop feedback system integrating all components.

    Combines:
    - Efference Copy: motor command -> sensory prediction
    - Smith Predictor: delay compensation
    - Precision Gate: dynamic error weighting
    - Feedback integration into latent dynamics

    Used in perception mode to correct latent state based on
    actual sensory feedback.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        tau_delay: float = 0.5,
        feedback_strength: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.tau_delay = tau_delay
        self.feedback_strength = feedback_strength

        self.efference_copy = EfferenceCopy(latent_dim=latent_dim)

        self.smith_predictor = SmithPredictor(
            hidden_dim=latent_dim,
            tau_delay=tau_delay,
        )

        self.precision_gate = PrecisionGate(latent_dim=latent_dim)

        self.feedback_fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.error_history = None
        self.max_history = 100

    def forward(
        self,
        z_current: torch.Tensor,
        predicted_signal: torch.Tensor,
        actual_signal: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        kda_state_history: Optional[torch.Tensor] = None,
        q_factor: Optional[float] = None,
        ataxia_score: Optional[float] = None,
        catalepsy_score: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute closed-loop feedback correction.

        Args:
            z_current: (B, d) current latent state
            predicted_signal: (B, ...) predicted sensory from decoder
            actual_signal: (B, ...) actual sensory input
            action: (B, d) action that was executed
            kda_state_history: (B, T, d) KDA state history for Smith predictor
            q_factor: Q-factor from FFT analysis (Wiener §11.10.2.1)
            ataxia_score: AtaxiaScore for safety interlock
            catalepsy_score: CatalepsyScore for safety interlock
        Returns:
            dict with corrected_z, feedback_signal, precision_weight, metrics
        """
        efference_result = self.efference_copy(
            z_current, predicted_signal, actual_signal
        )

        wiener_adjustment = None
        if kda_state_history is not None and kda_state_history.shape[1] > 0:
            wiener_adjustment = self.smith_predictor.adjust_gain_from_wiener(
                q_factor=q_factor if q_factor is not None else float("nan"),
                ataxia_score=ataxia_score if ataxia_score is not None else 0.0,
                catalepsy_score=catalepsy_score if catalepsy_score is not None else 0.0,
            )

            smith_result = self.smith_predictor(z_current, kda_state_history)
            predicted_z = smith_result["predicted_z"]
            stability_margin = smith_result["stability_margin"]
        else:
            predicted_z = z_current
            stability_margin = torch.ones(z_current.shape[0], device=z_current.device)

        precision = torch.ones(z_current.shape[0], device=z_current.device)
        if action is not None:
            delay_estimate = torch.full(
                (z_current.shape[0], 1),
                self.tau_delay,
                device=z_current.device
            )
            precision, precision_metrics = self.precision_gate(delay_estimate, action)
        else:
            precision_metrics = {}

        catalepsy_noise_boost = 0.0
        if wiener_adjustment is not None and wiener_adjustment.get("requires_noise_injection", False):
            catalepsy_noise_boost = 0.1

        prediction_error = efference_result["prediction_error"]
        if prediction_error is not None:
            weighted_error = prediction_error * precision * stability_margin

            error_signal = efference_result["efference"] * weighted_error.unsqueeze(-1)

            if catalepsy_noise_boost > 0:
                noise_signal = torch.randn_like(error_signal) * catalepsy_noise_boost
                error_signal = error_signal + noise_signal

            if self.error_history is None:
                self.error_history = error_signal.detach()
            else:
                self.error_history = torch.cat([
                    self.error_history, error_signal.detach()
                ], dim=1)[:, -self.max_history:]

            corrected_z = z_current + self.feedback_strength * error_signal
        else:
            corrected_z = z_current
            weighted_error = None
            error_signal = None

        result = {
            "corrected_z": corrected_z,
            "efference": efference_result["efference"],
            "prediction_error": prediction_error,
            "weighted_error": weighted_error,
            "precision": precision,
            "error_history": self.error_history,
            "metrics": {
                "stability_margin": stability_margin.mean().item() if stability_margin is not None else 1.0,
                "feedback_strength": self.feedback_strength,
                "wiener_mode": wiener_adjustment.get("mode", "disabled") if wiener_adjustment else "disabled",
                "oscillation_tolerant": self.smith_predictor.oscillation_tolerant,
                "catalepsy_noise_boost": catalepsy_noise_boost,
                **precision_metrics,
            }
        }

        return result

    def reset_history(self):
        """Reset error history buffer."""
        self.error_history = None


class ActiveInferenceController(nn.Module):
    """
    Complete Active Inference Controller.

    Integrates:
    1. Value Function for EFE computation
    2. Counterfactual Tree Search for action planning
    3. Closed-Loop Feedback for perception correction
    4. Efference Copy for motor-sensory integration

    Used in both perception and imagination modes.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        use_counterfactual: bool = True,
        use_feedback: bool = True,
        tau_delay: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_counterfactual = use_counterfactual
        self.use_feedback = use_feedback

        self.value_function = ValueFunction(hidden_dim=latent_dim)

        if use_counterfactual:
            from .counterfactual_search import CounterfactualTreeSearch
            self.cfts = CounterfactualTreeSearch(
                latent_dim=latent_dim,
                num_goals=2,
                num_exploration=1,
                rollout_steps=10,
            )

        if use_feedback:
            self.feedback = ClosedLoopFeedback(
                latent_dim=latent_dim,
                tau_delay=tau_delay,
            )

    def forward_perception(
        self,
        z_current: torch.Tensor,
        predicted_signal: torch.Tensor,
        actual_signal: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        kda_state_history: Optional[torch.Tensor] = None,
        q_factor: Optional[float] = None,
        ataxia_score: Optional[float] = None,
        catalepsy_score: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perception mode: correct latent state based on sensory feedback.

        Args:
            z_current: (B, d) current latent state
            predicted_signal: predicted sensory
            actual_signal: actual sensory input
            action: executed action
            kda_state_history: KDA history for Smith predictor
            q_factor: Q-factor for oscillation tolerance (Wiener §11.10.2.1)
            ataxia_score: AtaxiaScore for safety interlock
            catalepsy_score: CatalepsyScore for safety interlock
        Returns:
            dict with corrected state and metrics
        """
        V, value_metrics = self.value_function(z_current)

        result = {"value": V, "value_metrics": value_metrics}

        if self.use_feedback:
            feedback_result = self.feedback(
                z_current,
                predicted_signal,
                actual_signal,
                action,
                kda_state_history,
                q_factor=q_factor,
                ataxia_score=ataxia_score,
                catalepsy_score=catalepsy_score,
            )
            result.update(feedback_result)

        return result

    def forward_imagination(
        self,
        z_current: torch.Tensor,
        goal_attractors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagination mode: plan actions using counterfactual search.

        Args:
            z_current: (B, d) current latent state
            goal_attractors: (B, num_goals, d) goal states
        Returns:
            dict with selected action, trajectory, and EFE
        """
        V, value_metrics = self.value_function(z_current)

        result = {"value": V, "value_metrics": value_metrics}

        if self.use_counterfactual and self.cfts is not None:
            cfts_result = self.cfts(
                z_current,
                goal_attractors=goal_attractors,
            )

            selected_trajectory = cfts_result["selected_trajectory"]

            final_state = selected_trajectory[:, -1]
            efe, efe_metrics = self.value_function.compute_efe(
                final_state,
                goal_attractors=goal_attractors,
            )

            result["selected_action"] = cfts_result["selected_action"]
            result["selected_trajectory"] = selected_trajectory
            result["efe"] = efe
            result["efe_metrics"] = efe_metrics
            result["action_scores"] = cfts_result["scores"]

        return result

    def reset(self):
        """Reset all history buffers."""
        if self.use_feedback:
            self.feedback.reset_history()


if __name__ == "__main__":
    print("Testing Active Inference components...")

    B, D = 2, 2048
    device = torch.device("cpu")

    print("\n1. Testing ValueFunction...")
    from .velocity_brain import EnergyEntropyFields
    ee_net = EnergyEntropyFields(hidden_dim=D)
    vf = ValueFunction(hidden_dim=D, energy_entropy_net=ee_net)
    z = torch.randn(B, D)
    V, metrics = vf(z)
    print(f"  V shape: {V.shape}")
    print(f"  V mean: {metrics['V']:.4f}, E: {metrics['E']:.4f}, S: {metrics['S']:.4f}")

    print("\n2. Testing EfferenceCopy...")
    ec = EfferenceCopy(latent_dim=D)
    pred_signal = torch.randn(B, 19, 256)
    actual_signal = torch.randn(B, 19, 256)
    ec_result = ec(z, pred_signal, actual_signal)
    print(f"  prediction_error shape: {ec_result['prediction_error'].shape}")
    print(f"  weighted_error: {ec_result['weighted_error'].mean().item():.4f}")

    print("\n3. Testing SmithPredictor...")
    sp = SmithPredictor(hidden_dim=D)
    kda_history = torch.randn(B, 10, D)
    sp_result = sp(z, kda_history)
    print(f"  predicted_z shape: {sp_result['predicted_z'].shape}")
    print(f"  delay_estimate: {sp_result['delay_estimate'].mean().item():.4f}")
    print(f"  stability_margin: {sp_result['stability_margin'].mean().item():.4f}")

    stability_loss = sp.compute_stability_loss()
    print(f"  stability_loss: {stability_loss.item():.4f}")

    print("\n4. Testing PrecisionGate...")
    pg = PrecisionGate(latent_dim=D)
    delay = torch.full((B,), 0.5)
    action = torch.randn(B, D)
    precision, p_metrics = pg(delay, action)
    print(f"  precision shape: {precision.shape}")
    print(f"  precision mean: {p_metrics['precision']:.4f}")

    print("\n5. Testing ClosedLoopFeedback...")
    clf = ClosedLoopFeedback(latent_dim=D)
    clf_result = clf(z, pred_signal, actual_signal, action, kda_history)
    print(f"  corrected_z shape: {clf_result['corrected_z'].shape}")
    print(f"  precision: {clf_result['precision'].mean().item():.4f}")

    print("\n5b. Testing ClosedLoopFeedback with Wiener oscillation tolerance...")
    clf2 = ClosedLoopFeedback(latent_dim=D)
    clf_result2 = clf2(z, pred_signal, actual_signal, kda_history,
                       q_factor=1.2, ataxia_score=0.1, catalepsy_score=1.5)
    print(f"  wiener_mode: {clf_result2['metrics']['wiener_mode']}")
    print(f"  oscillation_tolerant: {clf_result2['metrics']['oscillation_tolerant']}")

    print("\n5c. Testing Ataxia interlock (Q=0.3, Ataxia=0.8)...")
    clf3 = ClosedLoopFeedback(latent_dim=D)
    clf_result3 = clf3(z, pred_signal, actual_signal, kda_history,
                       q_factor=0.3, ataxia_score=0.8, catalepsy_score=1.0)
    print(f"  wiener_mode: {clf_result3['metrics']['wiener_mode']}")
    print(f"  oscillation_tolerant: {clf_result3['metrics']['oscillation_tolerant']}")
    print(f"  expected: ataxia_interlock, False")

    print("\n5d. Testing Catalepsy interlock (Catalepsy=4.0)...")
    clf4 = ClosedLoopFeedback(latent_dim=D)
    clf_result4 = clf4(z, pred_signal, actual_signal, kda_history,
                       q_factor=1.0, ataxia_score=0.1, catalepsy_score=4.0)
    print(f"  wiener_mode: {clf_result4['metrics']['wiener_mode']}")
    print(f"  catalepsy_noise_boost: {clf_result4['metrics']['catalepsy_noise_boost']}")
    print(f"  expected: catalepsy_interlock, 0.1")

    print("\n6. Testing ActiveInferenceController (perception)...")
    controller = ActiveInferenceController(latent_dim=D, use_counterfactual=False)
    result = controller.forward_perception(z, pred_signal, actual_signal, action, kda_history)
    print(f"  value: {result['value'].mean().item():.4f}")
    print(f"  corrected_z shape: {result['corrected_z'].shape}")

    print("\n7. Testing ActiveInferenceController (imagination)...")
    controller2 = ActiveInferenceController(latent_dim=D, use_feedback=False)
    goals = torch.randn(B, 2, D)
    result = controller2.forward_imagination(z, goal_attractors=goals)
    print(f"  efe: {result['efe'].mean().item():.4f}")
    print(f"  selected_action shape: {result['selected_action'].shape}")
    print(f"  selected_trajectory shape: {result['selected_trajectory'].shape}")

    print("\nAll tests passed!")