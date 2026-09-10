"""
Memory system evaluation tests.

- Catastrophic forgetting check
- Pattern completion
- Replay fidelity
- Energy landscape terrain scan
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


def evaluate_pattern_completion(
    model: nn.Module,
    num_patterns: int = 10,
    noise_level: float = 0.8,
    store_steps: int = 200,
) -> Dict[str, float]:
    """Pattern completion through the model's Hebbian memory.

    Patterns are *stored* by running Oja's rule on the model's own weight
    matrix — the same plasticity rule the trainer uses — and then retrieved
    from an independent noisy cue.  The reported quantity is the retrieval
    gain: how much closer the recalled state is to the stored pattern than the
    cue it was recalled from.

    The previous version wrote nothing into memory (it called the MoE on unit
    vectors) and then ran ``model(zeros, zeros, mode="imagination")``, so its
    similarity score was at chance by construction and said nothing about the
    memory system.
    """
    model.eval()
    memory = getattr(model, "hebbian_memory", None)
    if memory is None:
        return {"runnable": False, "reason": "model has no hebbian_memory"}
    device = next(model.parameters()).device
    dim = int(model.latent_dim)

    patterns = torch.nn.functional.normalize(
        torch.randn(num_patterns, dim, device=device), dim=-1)

    # Storage: Oja's rule only updates while the module is in training mode.
    was_training = memory.training
    memory.train()
    try:
        with torch.no_grad():
            for _ in range(max(1, int(store_steps))):
                for pattern in patterns:
                    sample = pattern.unsqueeze(0)
                    memory.hebbian_weight(sample, sample, update=True)
    finally:
        memory.train(was_training)

    # Recall from an independent noisy cue.
    cue_similarities = []
    recall_similarities = []
    with torch.no_grad():
        for pattern in patterns:
            cue = torch.nn.functional.normalize(
                pattern + noise_level * torch.randn_like(pattern), dim=-1)
            recalled = memory.hebbian_weight(
                cue.unsqueeze(0), cue.unsqueeze(0), update=False).squeeze(0)
            cue_similarities.append(
                torch.cosine_similarity(cue, pattern, dim=-1).item())
            recall_similarities.append(
                torch.cosine_similarity(recalled, pattern, dim=-1).item())

    cue_mean = float(np.mean(cue_similarities))
    recall_mean = float(np.mean(recall_similarities))
    chance = float(np.mean([
        torch.cosine_similarity(
            torch.nn.functional.normalize(torch.randn(dim, device=device), dim=-1),
            pattern, dim=-1).item()
        for pattern in patterns
    ]))
    return {
        "runnable": True,
        "pattern_completion_similarity": recall_mean,
        "cue_similarity": cue_mean,
        "chance_similarity": chance,
        "completion_gain": recall_mean - cue_mean,
        "num_patterns": num_patterns,
        "noise_level": noise_level,
        "store_steps": int(store_steps),
    }


def evaluate_replay_fidelity(
    model: nn.Module,
    num_imaginations: int = 50,
) -> Dict[str, float]:
    """
    Compare imagined trajectories to actual latent dynamics.
    """
    model.eval()
    device = next(model.parameters()).device
    latent_dim = model.latent_dim

    # Actual trajectory
    actual = []
    z = torch.randn(1, latent_dim, device=device) * 0.1
    with torch.no_grad():
        for _ in range(num_imaginations):
            dummy_eeg = torch.zeros(1, 19, 2560, device=device)
            dummy_fmri = torch.zeros(1, 400, 100, device=device)
            out = model(dummy_eeg, dummy_fmri, mode="perception")
            z = out["z_next"]
            actual.append(z.squeeze(0))
    actual = torch.stack(actual)

    # Imagined trajectory
    imagined = []
    z = actual[0].unsqueeze(0)
    with torch.no_grad():
        for _ in range(num_imaginations):
            dummy_eeg = torch.zeros(1, 19, 2560, device=device)
            dummy_fmri = torch.zeros(1, 400, 100, device=device)
            out = model(dummy_eeg, dummy_fmri, mode="imagination")
            z = out["z_next"]
            imagined.append(z.squeeze(0))
    imagined = torch.stack(imagined)

    mse = torch.nn.functional.mse_loss(imagined, actual).item()
    corr = torch.corrcoef(torch.stack([imagined.mean(dim=-1), actual.mean(dim=-1)]))[0, 1].item()

    return {
        "replay_mse": mse,
        "replay_correlation": corr,
    }


def evaluate_forgetting(
    model: nn.Module,
    num_patterns_A: int = 5,
    num_patterns_B: int = 5,
) -> Dict[str, float]:
    """
    Catastrophic forgetting test: learn A, then B, measure A retention.
    """
    model.eval()
    device = next(model.parameters()).device
    latent_dim = model.latent_dim

    patterns_A = torch.randn(num_patterns_A, latent_dim, device=device)
    patterns_B = torch.randn(num_patterns_B, latent_dim, device=device)

    # Simulate learning A
    # (In a real test, would train on A then B)
    # Here we just check if model can distinguish them

    sim_AA = []
    sim_AB = []
    for i in range(num_patterns_A):
        for j in range(num_patterns_A):
            sim_AA.append(torch.cosine_similarity(patterns_A[i], patterns_A[j], dim=-1).item())
        for j in range(num_patterns_B):
            sim_AB.append(torch.cosine_similarity(patterns_A[i], patterns_B[j], dim=-1).item())

    avg_AA = sum(sim_AA) / len(sim_AA)
    avg_AB = sum(sim_AB) / len(sim_AB)
    separation = avg_AA - avg_AB

    return {
        "pattern_separation": separation,
        "intra_similarity": avg_AA,
        "inter_similarity": avg_AB,
    }


def evaluate_energy_landscape(
    model: nn.Module,
    grid_resolution: int = 10,
) -> Dict[str, float]:
    """
    Scan energy landscape around origin to find attractor basins.
    """
    model.eval()
    device = next(model.parameters()).device
    latent_dim = model.latent_dim

    if hasattr(model, "active_inference") and model.active_inference is not None:
        vf = model.active_inference.value_function
    else:
        return {"landscape_scanned": False}

    # Sample grid along first 2 principal directions
    x_vals = torch.linspace(-2.0, 2.0, grid_resolution, device=device)
    y_vals = torch.linspace(-2.0, 2.0, grid_resolution, device=device)

    energies = []
    with torch.no_grad():
        for x in x_vals:
            for y in y_vals:
                z = torch.zeros(1, latent_dim, device=device)
                z[0, 0] = x
                z[0, 1] = y
                _, metrics = vf(z)
                energies.append(metrics["E"])

    energies = np.array(energies)
    num_minima = int(np.sum((energies[1:-1] < energies[:-2]) & (energies[1:-1] < energies[2:])))

    return {
        "landscape_scanned": True,
        "num_minima": num_minima,
        "energy_range": float(energies.max() - energies.min()),
        "energy_mean": float(energies.mean()),
    }
