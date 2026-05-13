"""
Task switching and basin transfer evaluation.

Measures how well the model transitions between cognitive states
when cued with different task attractors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


def evaluate_basin_transfer(
    model: nn.Module,
    initial_z: torch.Tensor,
    goal_attractors: torch.Tensor,
    num_steps: int = 500,
    transfer_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate basin transfer from initial state to goal attractor.

    Args:
        model: BrainMoEPINN
        initial_z: (1, d) initial latent state
        goal_attractors: (1, num_goals, d) goal states
        num_steps: max steps to evaluate
        transfer_threshold: cosine similarity threshold for transfer success
    Returns:
        dict with transfer_success, latency, energy_profile
    """
    model.eval()
    device = initial_z.device
    latent_dim = initial_z.shape[-1]

    z = initial_z.clone()
    energies = []
    similarities = []

    with torch.no_grad():
        for step in range(num_steps):
            dummy_eeg = torch.zeros(1, 19, 2560, device=device)
            dummy_fmri = torch.zeros(1, 400, 100, device=device)
            cue = goal_attractors[:, 0] if goal_attractors.shape[1] > 0 else None

            out = model(
                dummy_eeg, dummy_fmri,
                mode="imagination",
                goal_attractors=goal_attractors,
                task_cue=cue,
            )
            z = out["z_next"]

            # Energy from ValueFunction if available
            if hasattr(model, "active_inference") and model.active_inference is not None:
                V, v_metrics = model.active_inference.value_function(z)
                energies.append(v_metrics.get("E", 0.0))

            # Similarity to closest goal
            if goal_attractors is not None:
                goal = goal_attractors[0]
                sims = torch.cosine_similarity(z, goal, dim=-1)
                similarities.append(sims.max().item())

    # Detect transfer
    transfer_step = None
    for i, sim in enumerate(similarities):
        if sim > transfer_threshold:
            transfer_step = i
            break

    energy_rise = 0.0
    if len(energies) > 10:
        pre_peak = max(energies[:len(energies) // 2]) if max(energies[:len(energies) // 2]) > 0 else 1e-8
        post_valley = min(energies[len(energies) // 2:])
        energy_rise = (pre_peak - post_valley) / pre_peak

    return {
        "transfer_success": transfer_step is not None,
        "transfer_latency": transfer_step if transfer_step is not None else num_steps,
        "energy_rise": energy_rise,
        "max_similarity": max(similarities) if similarities else 0.0,
    }


def evaluate_expert_reorganization(
    model: nn.Module,
    num_tasks: int = 3,
    steps_per_task: int = 100,
) -> Dict[str, float]:
    """
    Evaluate whether expert activation patterns reorganize across tasks.

    Returns:
        dict with reorganization_score, task_expert_overlap
    """
    model.eval()
    device = next(model.parameters()).device
    latent_dim = model.latent_dim

    task_experts = []

    for task_id in range(num_tasks):
        # Use different random goal attractors per task
        goals = torch.randn(1, 2, latent_dim, device=device)
        expert_counts = torch.zeros(16, device=device)

        with torch.no_grad():
            for _ in range(steps_per_task):
                dummy_eeg = torch.zeros(1, 19, 2560, device=device)
                dummy_fmri = torch.zeros(1, 400, 100, device=device)
                out = model(dummy_eeg, dummy_fmri, mode="imagination", goal_attractors=goals)

                # Extract routing metrics
                if "moe_routing" in out and "selected_experts" in out["moe_routing"]:
                    selected = out["moe_routing"]["selected_experts"]
                    for idx in selected.view(-1):
                        expert_counts[idx.item()] += 1

        task_experts.append(expert_counts / (expert_counts.sum() + 1e-8))

    # Compute pairwise overlap (Jensen-Shannon divergence)
    js_scores = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            p = task_experts[i]
            q = task_experts[j]
            m = 0.5 * (p + q)
            kl_p = (p * torch.log((p + 1e-10) / (m + 1e-10))).sum()
            kl_q = (q * torch.log((q + 1e-10) / (m + 1e-10))).sum()
            js = 0.5 * (kl_p + kl_q)
            js_scores.append(js.item())

    avg_js = sum(js_scores) / len(js_scores) if js_scores else 0.0

    return {
        "reorganization_score": avg_js,
        "num_tasks": num_tasks,
    }
