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
        latent_dim: int = 2048,
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
        Generate random exploration actions along Casimir invariant directions.

        Casimir invariants are quantities that commute with all generators
        of the Poisson algebra, meaning perturbations along these directions
        don't affect the energy/conservation properties.

        Args:
            z_current: (B, d) current latent state
            num_samples: number of random actions to generate
        Returns:
            (B, num_samples, d) random action perturbations
        """
        B = z_current.shape[0]

        casimir = self.casimir_directions[:num_samples]
        casimir = casimir.unsqueeze(0).expand(B, -1, -1)

        random_coef = torch.randn(B, num_samples, self.perturbation_dim, device=z_current.device) * 0.1

        perturbations = random_coef @ casimir.transpose(-2, -1)
        perturbations = perturbations.squeeze(-1)
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
    Roll out future trajectories from current state using the dynamics model.

    Given current state z_t and action perturbations, simulates future
    trajectories using the VelocityBrain/MoE dynamics model.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        num_steps: int = 10,
        dt: float = 0.001,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.dt = dt

        self.velocity_net = None

    def set_velocity_net(self, velocity_net: nn.Module):
        """Set the velocity/dynamics network for rollouts."""
        self.velocity_net = velocity_net

    def forward(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out trajectories for each action.

        Args:
            z_init: (B, d) initial latent state
            actions: (B, num_actions, d) action perturbations per sample
        Returns:
            trajectories: (B, num_actions, num_steps+1, d) trajectory states
            rewards: (B, num_actions) reward scores for each trajectory
        """
        B, num_actions, d = actions.shape
        device = z_init.device

        trajectories = torch.zeros(B, num_actions, self.num_steps + 1, d, device=device)
        trajectories[:, :, 0] = z_init.unsqueeze(1).expand(-1, num_actions, -1)

        z_current = z_init.unsqueeze(1).expand(-1, num_actions, -1).clone()

        for step in range(self.num_steps):
            z_flat = z_current.reshape(B * num_actions, d)

            if self.velocity_net is not None:
                with torch.set_grad_enabled(self.training):
                    delta_z = self.velocity_net(z_flat)["velocity"]
            else:
                delta_z = torch.randn_like(z_flat) * 0.01

            delta_z = delta_z.view(B, num_actions, d)

            z_current = z_current + self.dt * delta_z + actions * math.sqrt(self.dt)

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
        latent_dim: int = 2048,
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

    From current state z_t, generates multiple counterfactual trajectories
    by applying different latent perturbations (actions), evaluates them
    via discriminator, and selects the best action.

    Branching factor: 4 (per plan §2.5)
    - 2 goal attractor-based actions
    - 1 random exploration action
    - 1 task cue action
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        num_goals: int = 2,
        num_exploration: int = 1,
        rollout_steps: int = 3,
        branch_factor: int = 4,
        dt: float = 0.001,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_goals = num_goals
        self.num_exploration = num_exploration
        self.num_actions = num_goals + num_exploration + 1
        self.branch_factor = branch_factor

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

    def forward(
        self,
        z_current: torch.Tensor,
        goal_attractors: Optional[torch.Tensor] = None,
        task_cue: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform counterfactual tree search with actual branching.

        Branch factor: 4 (per plan §2.5)
        Depth: rollout_steps (default 3)
        Total leaf trajectories: 4^3 = 64

        Args:
            z_current: (B, d) current latent state
            goal_attractors: (B, num_goals, d) goal attractor centers from memory
            task_cue: (B, d) optional task cue
        Returns:
            dict with:
                - selected_action: (B, d) chosen action perturbation
                - selected_trajectory: (B, num_steps+1, d) rollout with best action
                - all_trajectories: (B, num_actions, num_steps+1, d) all rollouts
                - scores: (B, num_actions) scores for all actions
                - best_indices: (B,) index of best action per sample
        """
        B = z_current.shape[0]
        device = z_current.device
        num_steps = self.rollout.num_steps

        # Build tree: root -> actions -> actions -> actions
        # Use batching to process all branches in parallel
        z_root = z_current  # (B, d)

        # Level 0: generate actions from root
        actions_0, types_0 = self._generate_actions(z_root, goal_attractors, task_cue)
        num_a0 = actions_0.shape[1]  # branch_factor

        # Rollout step 0 with all level-0 actions
        z_expanded_0 = z_root.unsqueeze(1).expand(-1, num_a0, -1)  # (B, num_a0, d)
        z_next_0 = z_expanded_0 + actions_0 * math.sqrt(self.rollout.dt)  # (B, num_a0, d)

        # Level 1: generate actions from each level-0 state
        z_flat_1 = z_next_0.reshape(B * num_a0, self.latent_dim)  # (B*num_a0, d)

        if goal_attractors is not None:
            goal_attractors_expanded = goal_attractors.unsqueeze(1).expand(-1, num_a0, -1, -1)
            goal_attractors_flat = goal_attractors_expanded.reshape(B * num_a0, *goal_attractors.shape[1:])
        else:
            goal_attractors_flat = None

        if task_cue is not None:
            task_cue_expanded = task_cue.unsqueeze(1).expand(-1, num_a0, -1)
            task_cue_flat = task_cue_expanded.reshape(B * num_a0, self.latent_dim)
        else:
            task_cue_flat = None

        actions_1, types_1 = self._generate_actions(z_flat_1, goal_attractors_flat, task_cue_flat)
        num_a1 = actions_1.shape[1]  # branch_factor

        z_expanded_1 = z_next_0.unsqueeze(2).expand(-1, -1, num_a1, -1)  # (B, num_a0, num_a1, d)
        z_next_1 = z_expanded_1 + actions_1.reshape(B, num_a0, num_a1, self.latent_dim) * math.sqrt(self.rollout.dt)

        # Level 2: generate actions from each level-1 state
        z_flat_2 = z_next_1.reshape(B * num_a0 * num_a1, self.latent_dim)

        if goal_attractors_flat is not None:
            goal_attractors_flat2 = goal_attractors_flat.unsqueeze(1).expand(-1, num_a1, -1, -1)
            goal_attractors_flat2 = goal_attractors_flat2.reshape(B * num_a0 * num_a1, *goal_attractors.shape[1:])
        else:
            goal_attractors_flat2 = None

        if task_cue_flat is not None:
            task_cue_flat2 = task_cue_flat.unsqueeze(1).expand(-1, num_a1, -1)
            task_cue_flat2 = task_cue_flat2.reshape(B * num_a0 * num_a1, self.latent_dim)
        else:
            task_cue_flat2 = None

        actions_2, types_2 = self._generate_actions(z_flat_2, goal_attractors_flat2, task_cue_flat2)
        num_a2 = actions_2.shape[1]  # branch_factor

        # Final states after 3 levels of branching
        z_expanded_2 = z_next_1.unsqueeze(3).expand(-1, -1, -1, num_a2, -1)  # (B, num_a0, num_a1, num_a2, d)
        z_leaf = z_expanded_2 + actions_2.reshape(B, num_a0, num_a1, num_a2, self.latent_dim) * math.sqrt(self.rollout.dt)

        # Score all leaf states
        z_leaf_flat = z_leaf.reshape(B, num_a0 * num_a1 * num_a2, self.latent_dim)

        # Create pseudo-trajectories for discriminator
        traj_for_disc = z_leaf_flat.unsqueeze(2)  # (B, num_leaves, 1, d)
        scores_flat = self.discriminator(traj_for_disc)  # (B, num_leaves)

        best_leaf_indices = scores_flat.argmin(dim=1)  # (B,)

        # Decode best leaf index to action path
        batch_indices = torch.arange(B, device=device)
        best_leaf = z_leaf_flat[batch_indices, best_leaf_indices]  # (B, d)

        # Build selected trajectory: root -> best path
        selected_trajectory = torch.zeros(B, num_steps + 1, self.latent_dim, device=device)
        selected_trajectory[:, 0] = z_root
        # For tree search, we use the leaf as the final state and interpolate
        for t in range(1, num_steps + 1):
            alpha = t / num_steps
            selected_trajectory[:, t] = (1 - alpha) * z_root + alpha * best_leaf

        # Return first action along best path as selected_action
        idx0 = best_leaf_indices // (num_a1 * num_a2)
        selected_action = actions_0[batch_indices, idx0]

        return {
            "selected_action": selected_action,
            "selected_trajectory": selected_trajectory,
            "all_trajectories": z_leaf,  # (B, num_a0, num_a1, num_a2, d)
            "scores": scores_flat,
            "best_indices": best_leaf_indices,
            "action_types": types_0,
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
        latent_dim: int = 2048,
        hidden_dim: int = 512,
        num_timesteps: int = 20,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        self.temporal_encoder = nn.GRU(
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

        imagined_states = imagined_states.transpose(1, 2)

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