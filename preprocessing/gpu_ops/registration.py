"""
GPU-accelerated brain registration using VoxelMorph (optional).

VoxelMorph provides learned deformable registration that is
500-3600× faster than ANTs SyN. Pre-trained SynthMorph models
are available for brain MRI registration.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from ...utils.device_utils import get_device
from typing import Optional, Tuple, Dict
import warnings


VOXELMORPH_AVAILABLE = False
try:
    import voxelmorph as vxm
    VOXELMORPH_AVAILABLE = True
except ImportError:
    pass


class GPURegistration:
    """GPU-accelerated brain registration wrapper.

    Supports:
    1. VoxelMorph (learned, GPU, ~2-10s per subject) — optional
    2. Affine estimation via torch optimization
    3. Fallback to pre-computed fMRIPrep transforms

    Usage:
        reg = GPURegistration(template_path="MNI152.nii.gz")
        warped, flow = reg.register(moving_volume)
    """

    def __init__(
        self,
        template_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_voxelmorph: bool = True,
    ):
        self.device = get_device(device) if device is not None else get_device()
        self.use_voxelmorph = use_voxelmorph and VOXELMORPH_AVAILABLE
        self.template = None
        self.model = None

        if template_path is not None:
            self._load_template(template_path)

        if model_path is not None and self.use_voxelmorph:
            self._load_model(model_path)

    def _load_template(self, path: str):
        """Load MNI template from NIfTI."""
        try:
            import nibabel as nib
            img = nib.load(path)
            data = torch.from_numpy(img.get_fdata()).float().to(self.device)
            self.template = data
        except ImportError:
            warnings.warn("nibabel not available. Cannot load template from NIfTI.")

    def _load_model(self, path: str):
        """Load pre-trained VoxelMorph model."""
        if not VOXELMORPH_AVAILABLE:
            warnings.warn("VoxelMorph not installed. Registration will use affine estimation.")
            return
        try:
            self.model = vxm.networks.VxmDense.load(path, device=self.device)
            self.model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load VoxelMorph model: {e}")
            self.model = None

    def register(
        self,
        moving: torch.Tensor,
        moving_affine: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Register moving volume to template space.

        Args:
            moving: (D, H, W) or (1, D, H, W) brain volume on GPU
            moving_affine: optional (4, 4) affine matrix

        Returns:
            warped: (D, H, W) registered volume
            flow: (D, H, W, 3) deformation field (None if affine-only)
        """
        if moving.dim() == 3:
            moving = moving.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            squeeze = True
        elif moving.dim() == 4:
            moving = moving.unsqueeze(0)  # (1, 1, D, H, W)
            squeeze = False
        else:
            squeeze = False

        if self.use_voxelmorph and self.model is not None and self.template is not None:
            warped, flow = self._register_voxelmorph(moving)
        else:
            warped, flow = self._register_affine(moving, moving_affine)

        if squeeze:
            warped = warped.squeeze(0).squeeze(0)
            if flow is not None:
                flow = flow.squeeze(0)

        return warped, flow

    def _register_voxelmorph(
        self,
        moving: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """VoxelMorph deformable registration."""
        template = self.template.unsqueeze(0).unsqueeze(0).to(self.device)
        if template.shape[-3:] != moving.shape[-3:]:
            template = F.interpolate(
                template,
                size=moving.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        with torch.no_grad():
            warped, flow = self.model(moving, template, registration=True)

        return warped, flow

    def _register_affine(
        self,
        moving: torch.Tensor,
        initial_affine: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple affine registration via gradient descent on GPU.

        Estimates a 12-parameter affine transform (3×4 matrix)
        that maximizes normalized cross-correlation between moving and template.
        """
        if self.template is None:
            warnings.warn("No template loaded. Returning original volume.")
            return moving.squeeze(0).squeeze(0), None

        template = self.template.unsqueeze(0).unsqueeze(0).to(self.device)
        if template.shape != moving.shape:
            template = F.interpolate(
                template,
                size=moving.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        if initial_affine is not None:
            affine = initial_affine.clone().to(self.device).requires_grad_(True)
        else:
            affine = torch.eye(3, 4, device=self.device, dtype=moving.dtype, requires_grad=True)

        optimizer = torch.optim.LBFGS([affine], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            warped = self._apply_affine(moving, affine, template.shape[-3:])
            loss = -self._ncc(warped, template)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            warped = self._apply_affine(moving, affine, template.shape[-3:])

        return warped, None

    def _apply_affine(
        self,
        volume: torch.Tensor,
        affine: torch.Tensor,
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Apply affine transform to volume using grid_sample."""
        from ..gpu_ops.spatial import apply_affine_transform
        return apply_affine_transform(volume.squeeze(0), affine, target_shape).unsqueeze(0)

    @staticmethod
    def _ncc(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Normalized cross-correlation loss."""
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        y_pred_centered = y_pred_flat - y_pred_flat.mean()
        y_true_centered = y_true_flat - y_true_flat.mean()
        ncc = (y_pred_centered * y_true_centered).sum() / (
            y_pred_centered.norm() * y_true_centered.norm() + 1e-12
        )
        return ncc
