"""
GPU-accelerated spatial operations for fMRI: smoothing, interpolation, transforms.

Uses torch.nn.functional primitives for 3D spatial operations.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ...runtime.device_utils import get_device


def gaussian_kernel_3d(
    sigma: float,
    kernel_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 3D Gaussian kernel for spatial smoothing.

    Args:
        sigma: Standard deviation in voxels
        kernel_size: Size of kernel (default: ceil(6*sigma) | 1)
        device: torch device
        dtype: torch dtype

    Returns:
        kernel: (kD, kH, kW) Gaussian kernel
    """
    device = get_device(device)
    if kernel_size is None:
        kernel_size = int(math.ceil(6 * sigma)) | 1

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    grid_d, grid_h, grid_w = torch.meshgrid(coords, coords, coords, indexing="ij")

    kernel = torch.exp(-(grid_d ** 2 + grid_h ** 2 + grid_w ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def smooth_3d(
    volume: torch.Tensor,
    fwhm: float,
    voxel_size: float = 1.0,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    """Apply Gaussian smoothing to 3D volume on GPU.

    Args:
        volume: (D, H, W) or (B, D, H, W) or (B, C, D, H, W)
        fwhm: Full width at half maximum in mm
        voxel_size: Voxel size in mm
        kernel_size: override kernel size

    Returns:
        Smoothed volume with same shape
    """
    sigma_voxels = (fwhm / voxel_size) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    kernel = gaussian_kernel_3d(sigma_voxels, kernel_size, device=volume.device, dtype=volume.dtype)

    orig_dim = volume.dim()
    if orig_dim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    elif orig_dim == 4:
        volume = volume.unsqueeze(1)  # (B, 1, D, H, W)

    B, C, D, H, W = volume.shape
    kD, kH, kW = kernel.shape
    pad = (kW // 2, kW // 2, kH // 2, kH // 2, kD // 2, kD // 2)
    padded = F.pad(volume, pad, mode="reflect")

    kernel_5d = kernel.view(1, 1, kD, kH, kW).expand(C, -1, -1, -1, -1)

    result = F.conv3d(
        padded.reshape(B * C, 1, D + kD - 1, H + kH - 1, W + kW - 1),
        kernel_5d[:1].expand(B * C, -1, -1, -1, -1),
        groups=1,
    )
    result = result.reshape(B, C, D, H, W)

    if orig_dim == 3:
        result = result.squeeze(0).squeeze(0)
    elif orig_dim == 4:
        result = result.squeeze(1)

    return result


def smooth_4d_fmri(
    bold: torch.Tensor,
    fwhm: float,
    voxel_size: float = 1.0,
) -> torch.Tensor:
    """Apply Gaussian smoothing to 4D fMRI BOLD data.

    Args:
        bold: (X, Y, Z, T) or (B, X, Y, Z, T)
        fwhm: smoothing FWHM in mm
        voxel_size: voxel size in mm

    Returns:
        Smoothed BOLD data
    """
    if bold.dim() == 4:
        X, Y, Z, T = bold.shape
        bold_5d = bold.permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, X, Y, Z)
        squeeze = True
    else:
        B, X, Y, Z, T = bold.shape
        bold_5d = bold.permute(0, 4, 1, 2, 3)  # (B, T, X, Y, Z)
        squeeze = False

    sigma_voxels = (fwhm / voxel_size) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    kernel = gaussian_kernel_3d(sigma_voxels, device=bold.device, dtype=bold.dtype)
    kD, kH, kW = kernel.shape
    pad = (kW // 2, kW // 2, kH // 2, kH // 2, kD // 2, kD // 2)

    B5, C5, D5, H5, W5 = bold_5d.shape
    padded = F.pad(bold_5d, pad, mode="reflect")
    kernel_5d = kernel.view(1, 1, kD, kH, kW)

    result = torch.empty_like(bold_5d)
    for b in range(B5):
        for c in range(C5):
            vol = padded[b, c].unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            out = F.conv3d(vol, kernel_5d)
            result[b, c] = out.squeeze(0).squeeze(0)

    if squeeze:
        result = result.squeeze(0).permute(1, 2, 3, 0)  # (X, Y, Z, T)
    else:
        result = result.permute(0, 2, 3, 4, 1)  # (B, X, Y, Z, T)

    return result


def apply_affine_transform(
    volume: torch.Tensor,
    affine_matrix: torch.Tensor,
    target_shape: Optional[Tuple[int, int, int]] = None,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Apply affine transformation to 3D volume using grid_sample.

    Args:
        volume: (D, H, W) or (1, 1, D, H, W) volume
        affine_matrix: (4, 4) or (3, 4) affine transformation matrix
        target_shape: (D', H', W') output shape (default: same as input)
        mode: interpolation mode

    Returns:
        Transformed volume
    """
    if volume.dim() == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    D, H, W = volume.shape[-3:]
    if target_shape is None:
        target_shape = (D, H, W)

    D2, H2, W2 = target_shape

    grid_d = torch.linspace(-1, 1, D2, device=volume.device, dtype=volume.dtype)
    grid_h = torch.linspace(-1, 1, H2, device=volume.device, dtype=volume.dtype)
    grid_w = torch.linspace(-1, 1, W2, device=volume.device, dtype=volume.dtype)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, grid_d, indexing="ij"), dim=-1)  # (W2, H2, D2, 3)
    grid = grid.reshape(1, -1, 3)

    ones = torch.ones(1, grid.shape[1], 1, device=volume.device, dtype=volume.dtype)
    grid_hom = torch.cat([grid, ones], dim=-1)  # (1, N, 4)

    if affine_matrix.shape[0] == 3:
        full_affine = torch.eye(4, device=volume.device, dtype=volume.dtype)
        full_affine[:3] = affine_matrix
        affine_matrix = full_affine

    inv_affine = torch.linalg.inv(affine_matrix)
    source_coords = (grid_hom @ inv_affine.T)[:, :, :3]  # (1, N, 3)

    source_coords = source_coords.reshape(1, W2, H2, D2, 3)
    source_coords = source_coords.permute(0, 3, 2, 1, 4)  # (1, D2, H2, W2, 3) for grid_sample

    transformed = F.grid_sample(
        volume,
        source_coords,
        mode=mode,
        align_corners=True,
        padding_mode="zeros",
    )

    if squeeze:
        transformed = transformed.squeeze(0).squeeze(0)
    return transformed


def slice_timing_correction(
    bold: torch.Tensor,
    slice_times: torch.Tensor,
    tr: float,
    ref_slice: float = 0.0,
    mode: str = "linear",
) -> torch.Tensor:
    """Correct slice timing differences in 4D fMRI data.

    Args:
        bold: (X, Y, Z, T) BOLD data
        slice_times: (Z,) acquisition time for each slice (seconds)
        tr: Repetition time (seconds)
        ref_slice: reference slice time (default: 0.0 = first slice)
        mode: interpolation mode ("linear" or "nearest")

    Returns:
        Slice-timing corrected BOLD data
    """
    X, Y, Z, T = bold.shape
    corrected = torch.empty_like(bold)

    for z in range(Z):
        time_shift = ref_slice - slice_times[z]
        if abs(float(time_shift)) < 1e-6:
            corrected[:, :, z, :] = bold[:, :, z, :]
            continue

        shift_frac = float(time_shift) / tr
        n_shift = int(round(shift_frac))
        frac = shift_frac - n_shift

        slice_data = bold[:, :, z, :]  # (X, Y, T)

        if n_shift != 0:
            slice_data = torch.roll(slice_data, shifts=n_shift, dims=-1)
            if n_shift > 0:
                slice_data[:, :, :n_shift] = 0
            else:
                slice_data[:, :, n_shift:] = 0

        if abs(frac) > 1e-6 and mode == "linear":
            rolled = torch.roll(slice_data, shifts=1 if frac > 0 else -1, dims=-1)
            alpha = abs(frac)
            slice_data = (1 - alpha) * slice_data + alpha * rolled

        corrected[:, :, z, :] = slice_data

    return corrected
