"""
Counterfactual Tree Search for Brain MoE-PINN.

Implements imagination-based trajectory prediction from latent state z_t.
Used in the World Model and Active Inference framework (§2.5).

Key idea: From current state z_t, explore multiple action/goal hypotheticals
by rolling out future trajectories. Actions are defined as latent perturbations:
- From memory engram: goal attractor-based (z_goal - z_t)
- Random exploration: Casimir invariant direction perturbation
- Task cue: external task prompt encoding

Each trajectory is scored by a discriminator/reward model and the
highest-scoring trajectory is selected for action.

Reference: Brain MoE-PINN Training Plan §2.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..runtime.device_utils import device_aware_gru
from typing import Tuple, Optional, List, Dict
import math


class LatentPerturbation(nn.Module):
    """
    Defines perturbations (actions) in latent space.

    Actions are latent perturbations that represent different
    "mental simulations" or counterfactual paths.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        perturbation_dim: int = 64,
        num_actions: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.perturbation_dim = perturbation_dim
        self.num_actions = num_actions

        self.action_proj = nn.Sequential(
            nn.Linear(perturbation_dim, latent_dim),
            nn.Tanh(),
        )

        self.casimir_directions = nn.Parameter(
            torch.randn(num_actions, perturbation_dim) * 0.02
        )

    def forward_from_engram(
        self,
        z_current: torch.Tensor,
        goal_attractors: torch.Tensor,
        gamma: float = 0.1,
    ) -> torch.Tensor:
        """
        Generate action from goal attractor in memory engram.

        Args:
            z_current: (B, d) current latent state
            goal_attractors: (B, num_goals, d) goal attractor centers
            gamma: scaling factor
        Returns:
            (B, num_goals, d) action perturbations
        """
        goal_diff = goal_attractors - z_current.unsqueeze(1)
        actions = gamma * goal_diff
        return actions

    def forward_random_exploration(
        self,
        z_current: torch.Tensor,
        num_samples: int = 4,
    ) -> torch.Tensor:
        """
        Generate exploratory action candidates.

        The candidates are learned latent directions plus per-sample noise,
        projected by ``action_proj``.  They are not Casimir invariants: a
        Casimir invariant of a Poisson structure lies in the kernel of the
        structure matrix, and nothing here constrains
        ``L(z) @ casimir_directions`` to vanish.  Treat them as learned
        exploration directions.

        Args:
            z_current: (B, d) current latent state
            num_samples: number of random actions to generate
        Returns:
            (B, num_samples, d) exploratory action perturbations
        """
        B = z_current.shape[0]

        casimir = self.casimir_directions[:num_samples]
        casimir = casimir.unsqueeze(0).expand(B, -1, -1)

        random_coef = torch.randn(B, num_samples, self.perturbation_dim, device=z_current.device) * 0.1

        perturbations = casimir + random_coef
        perturbations = self.action_proj(perturbations)

        return perturbations

    def forward_from_cue(
        self,
        z_current: torch.Tensor,
        task_cue: torch.Tensor,
        beta: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate action from task cue encoding.

        Args:
            z_current: (B, d) current latent state
            task_cue: (B, d) task prompt/cue encoding
            beta: scaling factor
        Returns:
            (B, d) action perturbation
        """
        action = beta * (task_cue - z_current)
        return action


class TrajectoryRollout(nn.Module):
    """
    Roll out candidate futures from a latent state under the model dynamics.

    The injected ``velocity_net`` is the generic velocity backbone
    (``VelocityBrain``).  MoE expert modulation, stimulus control, species
    conditioning, and the multi-time-scale filter are **not** part of the
    rollout: the search therefore explores the base GENERIC-inspired field at
    the state it is given.  Keep that in mind when interpreting the selected
    trajectory as "what the model would do" — it is the backbone's
    continuation, not a full forward pass.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        num_steps: int = 10,
        dt: float = 0.001,
        action_mode: str = "impulse",
    ):
        super().__init__()
        if action_mode not in ("impulse", "sustained"):
            raise ValueError("action_mode must be 'impulse' or 'sustained'")
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.dt = dt
        self.action_mode = action_mode

        self.velocity_net = None

    def set_velocity_net(self, velocity_net: nn.Module):
        """Set the velocity/dynamics network for rollouts."""
        self.velocity_net = velocity_net
        integration_dt = getattr(velocity_net, "integration_dt", None)
        if integration_dt is not None:
            self.dt = float(integration_dt)

    def _velocity(self, z: torch.Tensor) -> torch.Tensor:
        if self.velocity_net is None:
            return torch.zeros_like(z)
        output = self.velocity_net(z)
        if isinstance(output, dict):
            velocity = output.get("velocity", output.get("delta_z"))
        else:
            velocity = output
        if velocity is None:
            raise ValueError(
                "velocity_net must return a tensor or a velocity/delta_z key")
        return velocity

    def forward(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out trajectories for each action.

        Returns:
            trajectories: (B, num_actions, num_steps+1, d)
            rewards: action-norm scores retained for compatibility
        """
        B, num_actions, d = actions.shape
        device = z_init.device

        trajectories = torch.zeros(
            B, num_actions, self.num_steps + 1, d, device=device,
            dtype=z_init.dtype)
        trajectories[:, :, 0] = z_init.unsqueeze(1).expand(
            -1, num_actions, -1)
        z_current = z_init.unsqueeze(1).expand(
            -1, num_actions, -1).clone()

        for step in range(self.num_steps):
            z_flat = z_current.reshape(B * num_actions, d)
            delta_z = self._velocity(z_flat).view(B, num_actions, d)
            # Counterfactual semantics: the intervention is applied once, at
            # the branch point, and the dynamics then evolve on their own.
            # "sustained" keeps the action on every step instead.
            if step > 0 and self.action_mode == "impulse":
                step_action = torch.zeros_like(actions)
            else:
                step_action = actions
            # Actions are deterministic control inputs, so they enter the
            # Euler step with a dt weight like the drift.  A sqrt(dt) weight
            # belongs to stochastic increments.
            z_current = z_current + self.dt * (delta_z + step_action)
            trajectories[:, :, step + 1] = z_current

        return trajectories, actions.norm(dim=-1)


class TrajectoryDiscriminator(nn.Module):
    """
    Scores trajectory rollouts based on downstream reward/objectives.

    Simple energy-based scoring:
    - Lower energy = better trajectory
    - Energy computed from distance to goal, novelty, task fit
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.energy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        trajectory: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score trajectories (lower energy = better).

        Args:
            trajectory: (B, num_steps, d) trajectory states
            goal: (B, d) optional goal state
        Returns:
            (B,) energy scores per trajectory
        """
        final_state = trajectory[:, -1]

        energy = self.energy_net(final_state).squeeze(-1)

        if goal is not None:
            goal_distance = torch.norm(final_state - goal, dim=-1)
            energy = energy + 0.1 * goal_distance

        return energy


class CounterfactualTreeSearch(nn.Module):
    """
    Counterfactual Tree Search for action selection.

    Candidate actions are expanded for ``rollout_steps`` levels.  Each
    branch is advanced by the configured latent dynamics and scored on its
    exact path by the trajectory discriminator.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        num_goals: int = 2,
        num_exploration: int = 1,
        rollout_steps: int = 3,
        branch_factor: int = 4,
        dt: float = 0.001,
        selection_temperature: float = 0.1,
    ):
        super().__init__()
        if selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        self.latent_dim = latent_dim
        self.num_goals = num_goals
        self.num_exploration = num_exploration
        self.num_actions = num_goals + num_exploration + 1
        self.branch_factor = branch_factor
        self.selection_temperature = selection_temperature

        self.perturbation = LatentPerturbation(
            latent_dim=latent_dim,
            perturbation_dim=64,
            num_actions=self.num_actions,
        )

        self.rollout = TrajectoryRollout(
            latent_dim=latent_dim,
            num_steps=rollout_steps,
            dt=dt,
        )

        self.discriminator = TrajectoryDiscriminator(latent_dim=latent_dim)

    def _generate_actions(
        self,
        z_current: torch.Tensor,
        goal_attractors: Optional[torch.Tensor] = None,
        task_cue: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Generate action candidates for a given state."""
        B = z_current.shape[0]
        device = z_current.device
        actions_list = []
        action_types = []

        if goal_attractors is not None:
            goal_actions = self.perturbation.forward_from_engram(z_current, goal_attractors)
            actions_list.append(goal_actions)
            action_types.extend(["goal"] * goal_actions.shape[1])

        exploration_actions = self.perturbation.forward_random_exploration(
            z_current, num_samples=self.num_exploration
        )
        actions_list.append(exploration_actions)
        action_types.extend(["exploration"] * self.num_exploration)

        if task_cue is not None:
            cue_action = self.perturbation.forward_from_cue(z_current, task_cue)
            cue_action = cue_action.unsqueeze(1)
            actions_list.append(cue_action)
            action_types.extend(["task_cue"])

        all_actions = torch.cat(actions_list, dim=1)

        if all_actions.shape[1] == 0:
            all_actions = torch.zeros(B, 1, self.latent_dim, device=device)

        return all_actions, action_types

    def _advance(
        self, parents: torch.Tensor, actions: torch.Tensor, level: int = 0
    ) -> torch.Tensor:
        """Advance every branch with the configured dynamics plus its action.

        Under the default ``impulse`` mode the action is applied only at the
        branch point (``level == 0``); deeper levels evolve freely.
        """
        B, num_parents, d = parents.shape
        num_actions = actions.shape[2]
        parent_expanded = parents.unsqueeze(2).expand(
            -1, -1, num_actions, -1)
        z_flat = parent_expanded.reshape(-1, d)
        action_flat = actions.reshape(-1, d)
        delta_z = self.rollout._velocity(z_flat).reshape(
            B, num_parents, num_actions, d)
        if level > 0 and self.rollout.action_mode == "impulse":
            action_flat = torch.zeros_like(action_flat)
        return (
            parent_expanded + self.rollout.dt * (
                delta_z + action_flat.reshape(
                    B, num_parents, num_actions, d)))

    def forward(
        self,
        z_current: torch.Tensor,
        goal_attractors: Optional[torch.Tensor] = None,
        task_cue: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a differentiable three-level counterfactual tree search.

        The discriminator scores exact dynamics paths.  A soft minimum over
        those scores selects the returned path for training, while
        ``best_indices`` remains a hard reporting/inspection value.
        """
        B, d = z_current.shape
        if self.rollout.num_steps < 1:
            raise ValueError("rollout_steps must be positive")
        device = z_current.device
        depth = self.rollout.num_steps

        states = [z_current.unsqueeze(1)]
        actions_by_level = []
        action_types = []
        action_counts = []

        for level in range(depth):
            parent_states = states[-1]
            num_parents = parent_states.shape[1]
            parent_flat = parent_states.reshape(B * num_parents, d)
            if goal_attractors is not None:
                goal_flat = goal_attractors.unsqueeze(1).expand(
                    -1, num_parents, -1, -1).reshape(
                        B * num_parents, *goal_attractors.shape[1:])
            else:
                goal_flat = None
            if task_cue is not None:
                cue_flat = task_cue.unsqueeze(1).expand(
                    -1, num_parents, -1).reshape(B * num_parents, d)
            else:
                cue_flat = None

            actions_flat, types = self._generate_actions(
                parent_flat, goal_flat, cue_flat)
            num_actions = actions_flat.shape[1]
            actions = actions_flat.reshape(
                B, num_parents, num_actions, d)
            next_states = self._advance(parent_states, actions, level=level)
            actions_by_level.append(actions)
            action_counts.append(num_actions)
            if level == 0:
                action_types = types
            states.append(next_states.reshape(B, -1, d))

        num_leaves = states[-1].shape[1]
        leaf_ids = torch.arange(num_leaves, device=device)
        path_states = []
        for level, level_states in enumerate(states):
            # State at level l is indexed by the prefix of actions through l.
            # Each prefix is repeated for all suffix combinations.
            suffix = math.prod(action_counts[level + 1:])
            level_ids = (leaf_ids // suffix) % level_states.shape[1]
            path_states.append(level_states[:, level_ids])
        candidate_paths = torch.stack(path_states, dim=2)
        candidate_paths = candidate_paths.reshape(
            B, num_leaves, depth + 1, d)

        traj_for_disc = candidate_paths.reshape(
            B * num_leaves, depth + 1, d)
        scores_flat = self.discriminator(traj_for_disc).reshape(
            B, num_leaves)
        best_leaf_indices = scores_flat.argmin(dim=1)

        soft_weights = torch.softmax(
            -scores_flat / self.selection_temperature, dim=1)
        selected_trajectory = torch.einsum(
            "bn,bntd->btd", soft_weights, candidate_paths)

        first_suffix = math.prod(action_counts[1:]) if depth > 1 else 1
        first_ids = leaf_ids // first_suffix
        first_actions = actions_by_level[0][:, 0, first_ids]
        selected_action = torch.einsum(
            "bn,bnd->bd", soft_weights, first_actions)

        final_shape = [B, *action_counts, d]
        all_trajectories = states[-1].reshape(final_shape)
        return {
            "selected_action": selected_action,
            "selected_trajectory": selected_trajectory,
            "all_trajectories": all_trajectories,
            "scores": scores_flat,
            "best_indices": best_leaf_indices,
            "action_types": action_types,
        }


class ImaginationSampler(nn.Module):
    """
    Sampling from imagined future trajectories.

    Used for mental time travel - imagining past/future scenarios
    based on current state and memory engram seeds.

    Reference: Hippocampal Index + Engram Landscape integration
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 512,
        num_timesteps: int = 20,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        # device_aware_gru(): see runtime.device_utils -- torch_xla rebinds nn.GRU
        # globally and its scan variant rejects CPU tensors.
        self.temporal_encoder = device_aware_gru()(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(
        self,
        seed_state: torch.Tensor,
        num_samples: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate imagined future trajectories.

        Args:
            seed_state: (B, d) seed state from memory engram
            num_samples: number of trajectories to generate
        Returns:
            dict with imagined trajectories and prior parameters
        """
        B = seed_state.shape[0]
        device = seed_state.device

        seed_expanded = seed_state.unsqueeze(1).expand(-1, num_samples, -1)
        seed_flat = seed_expanded.reshape(B * num_samples, 1, self.latent_dim)

        gru_out, _ = self.temporal_encoder(seed_flat)
        gru_out = gru_out.reshape(B, num_samples, -1)

        prior_params = self.prior_net(gru_out)
        mean = prior_params[:, :, :self.latent_dim]
        log_std = prior_params[:, :, self.latent_dim:]

        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(mean)
        imagined_states = mean + std * eps

        return {
            "imagined_states": imagined_states,
            "mean": mean,
            "std": std,
            "num_samples": num_samples,
        }


if __name__ == "__main__":
    print("Testing Counterfactual Tree Search components...")

    B, D = 2, 2048
    z = torch.randn(B, D)

    print("\n1. Testing LatentPerturbation...")
    perturb = LatentPerturbation(latent_dim=D, num_actions=4)
    goals = torch.randn(B, 2, D)
    goal_actions = perturb.forward_from_engram(z, goals)
    print(f"  Goal actions shape: {goal_actions.shape}")

    rand_actions = perturb.forward_random_exploration(z, num_samples=2)
    print(f"  Random actions shape: {rand_actions.shape}")

    cue = torch.randn(B, D)
    cue_action = perturb.forward_from_cue(z, cue)
    print(f"  Cue action shape: {cue_action.shape}")

    print("\n2. Testing TrajectoryRollout...")
    rollout = TrajectoryRollout(latent_dim=D, num_steps=5)
    actions = torch.randn(B, 4, D)
    trajectories, rewards = rollout(z, actions)
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Rewards shape: {rewards.shape}")

    print("\n3. Testing TrajectoryDiscriminator...")
    disc = TrajectoryDiscriminator(latent_dim=D)
    energies = disc(trajectories, goal=goals[:, 0])
    print(f"  Energies shape: {energies.shape}")

    print("\n4. Testing CounterfactualTreeSearch...")
    cfts = CounterfactualTreeSearch(latent_dim=D, num_goals=2, num_exploration=1, rollout_steps=5)
    result = cfts(z, goal_attractors=goals, task_cue=cue)
    print(f"  Selected action shape: {result['selected_action'].shape}")
    print(f"  Selected trajectory shape: {result['selected_trajectory'].shape}")
    print(f"  Best indices: {result['best_indices']}")

    print("\n5. Testing ImaginationSampler...")
    sampler = ImaginationSampler(latent_dim=D)
    imagined = sampler(z, num_samples=4)
    print(f"  Imagined states shape: {imagined['imagined_states'].shape}")
    print(f"  Mean shape: {imagined['mean'].shape}")

    print("\nAll tests passed!")