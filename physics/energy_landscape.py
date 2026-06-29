"""
Energy landscape analysis for Brain MoE-PINN.

Waddington epigenetic landscape + Casimir invariants + attractor basins.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def compute_waddington_landscape(
    energy_fn,
    grid_range: Tuple[float, float] = (-3.0, 3.0),
    resolution: int = 50,
    latent_dim: int = 1024,
    device: str = "cuda",
) -> Dict:
    """
    Compute 2D Waddington landscape slice along first 2 principal axes.

    Args:
        energy_fn: callable(z) -> E
        grid_range: (min, max) for each axis
        resolution: grid points per axis
    Returns:
        dict with grid, energies, gradients
    """
    x = torch.linspace(grid_range[0], grid_range[1], resolution, device=device)
    y = torch.linspace(grid_range[0], grid_range[1], resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    energies = torch.zeros(resolution, resolution, device=device)
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                z = torch.zeros(1, latent_dim, device=device)
                z[0, 0] = X[i, j]
                z[0, 1] = Y[i, j]
                E = energy_fn(z)
                energies[i, j] = E.item() if hasattr(E, "item") else E

    # Find basins (local minima)
    padded = torch.nn.functional.pad(energies.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    is_min = (
        (energies < padded[0, 0, :-2, 1:-1]) &
        (energies < padded[0, 0, 2:, 1:-1]) &
        (energies < padded[0, 0, 1:-1, :-2]) &
        (energies < padded[0, 0, 1:-1, 2:])
    )
    num_basins = is_min.sum().item()

    # Compute barrier heights between basins
    barrier_heights = []
    minima_coords = torch.argwhere(is_min)
    for k in range(minima_coords.shape[0]):
        for l in range(k + 1, minima_coords.shape[0]):
            i1, j1 = minima_coords[k]
            i2, j2 = minima_coords[l]
            # Simple path: straight line
            num_steps = max(abs(i2 - i1), abs(j2 - j1))
            if num_steps > 0:
                path_i = torch.linspace(i1, i2, num_steps, device=device).long()
                path_j = torch.linspace(j1, j2, num_steps, device=device).long()
                path_energies = energies[path_i, path_j]
                barrier = path_energies.max().item() - max(energies[i1, j1].item(), energies[i2, j2].item())
                barrier_heights.append(barrier)

    return {
        "grid_x": X.cpu().numpy(),
        "grid_y": Y.cpu().numpy(),
        "energies": energies.cpu().numpy(),
        "num_basins": num_basins,
        "barrier_heights": barrier_heights,
        "mean_barrier": float(np.mean(barrier_heights)) if barrier_heights else 0.0,
    }


def compute_casimir_invariants(L_z: torch.Tensor, num_samples: int = 128) -> Dict[str, float]:
    """
    Approximate Casimir invariants of Poisson operator L(z).

    Casimir functions C(z) satisfy L(z) @ grad_C = 0.
    We approximate by finding near-nullspace directions of L(z).
    """
    d = L_z.shape[0]
    # Use SVD to find nullspace directions
    try:
        U, S, Vh = torch.linalg.svd(L_z)
        threshold = S.mean() * 0.05
        null_dim = (S < threshold).sum().item()

        # Compute how well these directions satisfy L @ v ≈ 0
        null_vectors = Vh[S < threshold] if null_dim > 0 else Vh[-1:]
        violations = []
        for v in null_vectors:
            Lv = L_z @ v
            violations.append(torch.norm(Lv).item())

        return {
            "nullspace_dim": null_dim,
            "casimir_violation": float(np.mean(violations)) if violations else 0.0,
        }
    except Exception:
        return {"nullspace_dim": 0, "casimir_violation": 0.0}
