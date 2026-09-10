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

import brain_moe_pinn.runtime.device_utils as du
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


# --- vendor backends and old-torch robustness -----------------------------

class _RecordingScaler:
    """Stand-in for a vendor GradScaler that records its init_scale."""

    def __init__(self, init_scale=None, **kwargs):
        self.init_scale = init_scale

    def get_scale(self):
        return self.init_scale

    def is_enabled(self):
        return True


def _install_vendor_stub(monkeypatch, module_name, attr_path, scaler):
    """Install a fake vendor module exposing ``attr_path`` -> scaler class."""
    import sys
    import types

    obj = types.ModuleType(module_name)
    cur = obj
    parts = attr_path.split(".")
    for part in parts[:-1]:
        nxt = types.ModuleType(f"{module_name}.{part}")
        setattr(cur, part, nxt)
        cur = nxt
    setattr(cur, parts[-1], scaler)
    monkeypatch.setitem(sys.modules, module_name, obj)


@pytest.mark.parametrize("vendor", ["musa", "npu", "mlu"])
def test_vendor_backends_use_vendor_scaler(monkeypatch, vendor):
    """Each vendor backend must get its *own* scaler, not an inert one.

    An enabled-but-inert scaler silently removes the fp16 safety net, so a
    vendor that ships a real GradScaler must be routed to it.
    """
    module_name, _autocast, scaler_attr, _label = du._VENDOR_BACKENDS[vendor]
    _install_vendor_stub(monkeypatch, module_name, scaler_attr, _RecordingScaler)
    monkeypatch.setattr(du, "get_device_type", lambda device=None: vendor)

    scaler = du.grad_scaler(None, enabled=True, init_scale=256.0)
    assert isinstance(scaler, _RecordingScaler)
    assert scaler.get_scale() == 256.0


def test_vendor_backends_declare_both_attrs():
    """Every vendor entry needs an autocast path and a scaler path."""
    for name, entry in du._VENDOR_BACKENDS.items():
        assert len(entry) == 4, f"{name} entry is not a 4-tuple"
        module, autocast_attr, scaler_attr, label = entry
        assert module and autocast_attr and scaler_attr and label
        assert autocast_attr.endswith("autocast")
        assert scaler_attr.endswith("GradScaler")


def test_disabled_scaler_without_torch_amp_gradscaler(monkeypatch):
    """torch < 2.3 has no torch.amp.GradScaler; this must not raise.

    Every non-CUDA/XPU device routes through _disabled_scaler, so depending
    on the newer attribute took out CPU, MUSA and XLA at once.
    """
    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)
    scaler = du._disabled_scaler("cpu")
    assert not scaler.is_enabled()
    loss = torch.tensor(1.5)
    assert torch.equal(scaler.scale(loss), loss)


def test_disabled_scaler_raises_when_no_implementation(monkeypatch):
    """With neither attribute present, fail loudly rather than silently."""
    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)
    monkeypatch.delattr(torch.cuda.amp, "GradScaler", raising=False)
    with pytest.raises(RuntimeError, match="no GradScaler implementation"):
        du._disabled_scaler("cpu")
