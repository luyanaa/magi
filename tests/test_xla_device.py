"""XLA/TPU device handling.

Two failure modes are guarded here, both of which are silent rather than
loud: a TPU host reporting no accelerator (so training runs on CPU at ~1/1000
speed with no error), and an fp16 autocast context built on a backend that has
no fp16 compute path.
"""

import sys
import types

import pytest
import torch

import brain_moe_pinn.runtime.device_utils as du


@pytest.fixture
def fake_tpu(monkeypatch):
    """Install a stub torch_xla and an XLA env hint, then clear caches.

    The detector is lru_cached and reads os.environ, so both must be reset
    around each test.
    """
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    xla_model = types.ModuleType("torch_xla.core.xla_model")

    class _Dev:
        type = "xla"

    xla_model.xla_device = lambda: _Dev()
    torch_xla = types.ModuleType("torch_xla")
    core = types.ModuleType("torch_xla.core")
    torch_xla.core = core
    core.xla_model = xla_model
    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model)
    du._detect_xla.cache_clear()
    du._detect_accelerator.cache_clear()
    yield
    du._detect_xla.cache_clear()
    du._detect_accelerator.cache_clear()


def test_xla_detected_when_hinted(fake_tpu):
    """A TPU hint plus torch_xla must resolve to the xla device.

    Without this, _detect_accelerator() returns None on a TPU host and
    get_device() silently returns cpu.
    """
    assert du._detect_xla() == "xla"
    assert du._detect_accelerator() == "xla"
    assert du.get_device_type() == "xla"
    assert du.get_device().type == "xla"


def test_xla_not_probed_without_hint(monkeypatch, fake_tpu):
    """No hint -> XLA is never selected, even with torch_xla importable.

    Importing torch_xla initialises the PJRT runtime, so ordinary hosts must
    not be probed.
    """
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    for name in du._XLA_ENV_HINTS:
        monkeypatch.delenv(name, raising=False)
    du._detect_xla.cache_clear()
    du._detect_accelerator.cache_clear()
    assert du._detect_xla() is None
    assert du._detect_accelerator() != "xla"


def test_missing_torch_xla_degrades_to_non_xla(monkeypatch):
    """A hint without torch_xla must fall back, not raise."""
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.setitem(sys.modules, "torch_xla", None)
    du._detect_xla.cache_clear()
    du._detect_accelerator.cache_clear()
    try:
        assert du._detect_xla() is None
        assert du._detect_accelerator() != "xla"
    finally:
        du._detect_xla.cache_clear()
        du._detect_accelerator.cache_clear()


def test_xla_autocast_defaults_to_bfloat16():
    """A default XLA autocast request must be bf16, not the fp16 CUDA default."""
    assert du._xla_autocast_dtype(None) is torch.bfloat16


def test_xla_promotes_fp16_to_bf16_with_warning():
    """An explicit fp16 request is promoted, and says so.

    Silently honouring fp16 on TPU would give numerics the caller did not ask
    for; silently substituting without a word would be the same defect.
    """
    with pytest.warns(UserWarning, match="no fp16 compute path"):
        resolved = du._xla_autocast_dtype(torch.float16)
    assert resolved is torch.bfloat16


def test_xla_preserves_explicit_fp32():
    """fp32 is a valid XLA autocast dtype and must not be rewritten."""
    assert du._xla_autocast_dtype(torch.float32) is torch.float32


def test_xla_autocast_context_is_constructible():
    """autocast_context('xla') yields a usable context with a bf16 dtype."""
    ctx = du.autocast_context("xla", enabled=True)
    assert ctx is not None
    with ctx:
        pass


def test_xla_scaler_is_disabled():
    """XLA gets a disabled scaler: bf16 has fp32's range and needs no scaling."""
    assert not du.grad_scaler("xla", enabled=True).is_enabled()


class TestDeviceAwareGRU:
    """torch_xla rebinds nn.GRU globally; calls must still route correctly.

    torch_xla/_patched_functions.py does
    `nn.GRU = _pathch_module(nn.GRU, ScanGRU)`, and that scan variant requires
    XLA tensors - so CPU/CUDA nn.GRU raises as soon as torch_xla is importable.
    """

    def test_untouched_without_torch_xla(self, monkeypatch):
        """No shim -> plain nn.GRU, so CUDA/CPU code paths are unchanged."""
        real = torch.nn.modules.rnn.GRU
        monkeypatch.setattr(torch.nn, "GRU", real)
        assert du.device_aware_gru() is real

    def test_non_xla_input_routes_to_native(self, monkeypatch):
        real = torch.nn.modules.rnn.GRU
        seen = {}

        class ScanShim(real):
            _orig = real

            def forward(self, input, hx=None):
                seen["shim"] = True
                return super().forward(input, hx)

        monkeypatch.setattr(torch.nn, "GRU", ScanShim)
        gru = du.device_aware_gru()(4, 3, batch_first=True)
        out, _ = gru(torch.randn(2, 5, 4))          # CPU tensor
        assert tuple(out.shape) == (2, 5, 3)
        assert "shim" not in seen, "XLA scan path was taken for CPU input"

    def test_xla_input_keeps_scan_path(self, monkeypatch):
        """XLA input must still reach torch_xla's implementation.

        A tensor's `is_xla` attribute is not writable, so routing is exercised
        with a sentinel whose `is_xla` is True; the shim stands in for the scan
        implementation and simply records that it was reached.
        """
        import types

        real = torch.nn.modules.rnn.GRU
        seen = {}

        class ScanShim(real):
            _orig = real

            def forward(self, input, hx=None):
                seen["shim"] = True
                return "scan-result", None

        monkeypatch.setattr(torch.nn, "GRU", ScanShim)
        gru = du.device_aware_gru()(4, 3, batch_first=True)
        gru(types.SimpleNamespace(is_xla=True))
        assert seen.get("shim") is True, "scan implementation was bypassed"

    def test_parameter_names_unchanged(self, monkeypatch):
        """Subclassing must not alter state_dict keys (checkpoint compat)."""
        real = torch.nn.modules.rnn.GRU

        class ScanShim(real):
            _orig = real

        monkeypatch.setattr(torch.nn, "GRU", ScanShim)
        names = set(du.device_aware_gru()(4, 3, batch_first=True).state_dict())
        assert names == set(real(4, 3, batch_first=True).state_dict())
