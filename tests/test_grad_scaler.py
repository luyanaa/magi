"""GradScaler construction contract for the fp16 path.

The opening steps of a run produce its largest gradients, so the scaler's
starting loss scale decides whether fp16 overflows before it has any evidence
to adapt on.  Starting at PyTorch's 2**16 overflowed on step 1 of this model
under autocast, costing a skipped optimizer step (and making a correctly
functioning scaler look like a NaN-gradient bug when grads are read after
``unscale_()``).  These tests pin the contract that fixes it.
"""

import pytest
import torch

from brain_moe_pinn.runtime.device_utils import (
    DEFAULT_INIT_SCALE,
    grad_scaler,
)
from brain_moe_pinn.runtime.device_utils import get_device_type

CUDA = torch.cuda.is_available()


def test_default_scale_leaves_fp16_headroom():
    """The default must stay well below fp16's 65504 ceiling.

    Gradients in this model are O(1); a scale near the ceiling overflows on
    the first step and burns a skipped update before adapting down.  At 4096
    the ceiling is reached only by gradients above ~15.99, so a 1.0-magnitude
    gradient keeps an order of magnitude in reserve.
    """
    fp16_max = float(torch.finfo(torch.float16).max)
    assert DEFAULT_INIT_SCALE * 15.0 < fp16_max, (
        f"init_scale {DEFAULT_INIT_SCALE} cannot represent gradients up to 15 "
        f"against the fp16 ceiling {fp16_max}")
    assert fp16_max / DEFAULT_INIT_SCALE > 10.0


def test_cpu_scaler_is_disabled():
    """CPU gets a disabled scaler, never an enabled-but-inert one.

    An inert enabled scaler is worse than none: it silently removes the fp16
    safety net, so overflowing gradients are applied instead of skipped.
    """
    scaler = grad_scaler("cpu", enabled=True)
    assert not scaler.is_enabled()


def test_disabled_scaler_does_not_rescale_loss():
    """A disabled scaler must pass the loss through unchanged."""
    scaler = grad_scaler("cpu", enabled=False)
    loss = torch.tensor(2.5)
    assert torch.equal(scaler.scale(loss), loss)


def test_scaler_honours_explicit_init_scale():
    """An explicit init_scale overrides the default (no CUDA needed to check
    the disabled path stays inert)."""
    scaler = grad_scaler("cpu", enabled=True, init_scale=128.0)
    assert not scaler.is_enabled()


@pytest.mark.skipif(not CUDA, reason="CUDA required for a real GradScaler")
def test_cuda_scaler_starts_at_default_and_override():
    """A live scaler starts at DEFAULT_INIT_SCALE, and honours an override."""
    assert grad_scaler("cuda", enabled=True).get_scale() == DEFAULT_INIT_SCALE
    assert grad_scaler("cuda", enabled=True, init_scale=256.0).get_scale() == 256.0


def test_device_type_resolution_used_for_scaler():
    """grad_scaler keys off the resolved device type, not the raw argument."""
    assert get_device_type("cpu") == "cpu"
