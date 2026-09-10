"""
Device-agnostic utilities for Brain MoE-PINN.

Supports: CUDA, XPU (Intel), MUSA (Moore Threads), NPU (Huawei Ascend),
MLU (Cambricon), CPU — and any future PyTorch accelerator backend.

Design principles:
  1. Never hardcode "cpu" or "cuda" outside this module.
  2. All device moves go through move_to_device().
  3. AMP uses the unified torch.amp API (PyTorch >= 2.0).
  4. Backend detection is lazy and cached.
"""

import contextlib
import os
import warnings

import torch
from typing import Optional, Union
import functools

_BACKEND_PRIORITY = [
    ("cuda", "cuda"),
    ("xpu", "xpu"),
    ("npu", "npu"),
    ("musa", "musa"),
    ("mlu", "mlu"),
]


# XLA/TPU is probed only when the runtime advertises it: importing torch_xla
# initialises the PJRT runtime as a side effect, so probing unconditionally
# would perturb ordinary CPU/GPU hosts.
_XLA_ENV_HINTS = (
    "PJRT_DEVICE", "TPU_NAME", "COLAB_TPU_ADDR", "XRT_TPU_CONFIG",
    "TPU_WORKER_ID", "TPU_PROCESS_ADDRESSES",
)


def _xla_expected() -> bool:
    """Whether the environment advertises a TPU/XLA runtime."""
    return any(os.environ.get(name) for name in _XLA_ENV_HINTS)


@functools.lru_cache(maxsize=1)
def _detect_xla() -> Optional[str]:
    """Return the XLA device type when a TPU runtime is actually present.

    Returns None (rather than raising) when the hint is set but torch_xla is
    missing, so a misconfigured host degrades to the normal accelerators
    instead of failing at import.
    """
    if not _xla_expected():
        return None
    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        return None
    try:
        return str(xm.xla_device().type)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _detect_accelerator() -> Optional[str]:
    # An explicit XLA hint wins: PJRT_DEVICE is the user asking for torch_xla,
    # which manages the underlying accelerator itself.  Without this branch a
    # TPU host reports no accelerator at all and training silently runs on CPU.
    xla = _detect_xla()
    if xla is not None:
        return xla
    for attr, name in _BACKEND_PRIORITY:
        mod = getattr(torch, attr, None)
        if mod is not None and hasattr(mod, "is_available") and mod.is_available():
            return name
    return None


def get_device(device: Optional[Union[str, torch.device]] = None, index: Optional[int] = None) -> torch.device:
    """
    Resolve a torch.device from user input or auto-detect.

    Args:
        device: Explicit device string/device, or None to auto-detect.
        index: Optional device index (e.g. GPU rank). Ignored if device
               already contains an index.
    Returns:
        torch.device
    """
    if device is not None:
        d = torch.device(device)
        if index is not None and d.index is None:
            d = torch.device(d.type, index)
        return d

    acc = _detect_accelerator()
    if acc is not None:
        if index is not None:
            return torch.device(acc, index)
        return torch.device(acc)
    return torch.device("cpu")


def get_device_type(device: Optional[Union[str, torch.device]] = None) -> str:
    """Return the device type string (e.g. 'cuda', 'xpu', 'cpu')."""
    return get_device(device).type


def move_to_device(obj, device: Optional[Union[str, torch.device]] = None):
    """
    Move a tensor, module, or dict/list/tuple thereof to device.

    Replaces .cuda(), .to("cuda"), .to(device="cpu") etc.
    If device is None, auto-detects the accelerator.
    """
    d = get_device(device)
    if isinstance(obj, torch.Tensor):
        return obj.to(d)
    if isinstance(obj, torch.nn.Module):
        return obj.to(d)
    if isinstance(obj, dict):
        return {k: move_to_device(v, d) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, d) for v in obj]
        return type(obj)(moved)
    return obj


def safe_epsilon(reference, default: float = 1e-8) -> float:
    """Return an additive epsilon that is representable in the input's dtype.

    Guards written as ``x + 1e-8`` are silently broken in float16: the smallest
    positive half value is ~5.96e-08, so 1e-8 (and 1e-10, 1e-12) flush to
    exactly zero.  ``log(w + 1e-8)`` then evaluates ``log(0) -> -inf`` and
    ``0 * -inf -> NaN``, and ``x / (norm + 1e-8)`` divides by zero.  Both show
    up as NaN gradients that originate in whichever reduction underflowed.
    float32/float64 keep the default; bfloat16 shares float32's exponent range
    so it also keeps it.
    """
    if torch.is_tensor(reference) and reference.dtype == torch.float16:
        return max(float(default), 1e-6)
    return float(default)


# Starting loss scale for fp16 GradScaler.  PyTorch defaults to 2**16, which
# overflows fp16's 65504 ceiling on the opening steps of this model (largest
# gradients of the run) and costs a skipped optimizer step while the scaler
# halves down.  2**12 leaves ~16x headroom over the observed O(1) gradients
# while keeping even 1e-11 gradients representable; the scaler doubles back up
# on its own once a clean window is seen, so dynamic range is unaffected.
DEFAULT_INIT_SCALE = 4096.0

_VENDOR_AMP = {
    # device type -> (module, attribute path to the vendor autocast factory)
    "npu": ("torch_npu", "npu.amp.autocast", "Ascend NPU"),
    "musa": ("torch_musa", "core.amp.autocast", "Moore Threads MUSA"),
    "mlu": ("torch_mlu", "core.amp.autocast", "Cambricon MLU"),
}


def _require_vendor(module_name: str, backend_label: str):
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - depends on host hardware
        raise RuntimeError(
            f"device type requires the {backend_label} extension "
            f"'{module_name}', which is not installed; install it or select a "
            f"different device") from exc


def _resolve_attr(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


@functools.lru_cache(maxsize=16)
def _torch_amp_accepts(device_type: str) -> bool:
    """Whether stock torch.amp.autocast accepts this device type.

    The context object is *constructed* but never entered, so this probes the
    constructor's device validation without changing autocast state.  The probe
    must not pass ``enabled=False``: that short-circuits the validation and
    accepts any string, which is how an earlier revision silently routed
    ``'npu'`` into ``torch.amp.autocast`` and raised the raw
    "Expected one of cpu, cuda, ipu, xpu, ..." error.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.amp.autocast(device_type=device_type, dtype=torch.float16)
            return True
        except Exception:
            return False


def _xla_autocast_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Resolve the autocast dtype for XLA, which has no fp16 compute path.

    TPUs execute bfloat16 natively; fp16 is not a supported XLA compute
    format, so an fp16 autocast context on this device is either rejected by
    the backend or silently ignored.  bfloat16 also carries fp32's exponent
    range, so it needs no loss scaling -- which is why ``grad_scaler`` stays
    *disabled* on XLA rather than being switched to fp16 scaling.

    An explicit fp16 request is promoted rather than accepted, with a warning
    naming the substitution: silently honouring it would hand back a context
    whose numerics differ from what the caller asked for.
    """
    if dtype is None:
        return torch.bfloat16
    if dtype == torch.float16:
        warnings.warn(
            "XLA/TPU has no fp16 compute path; using bfloat16 instead. "
            "bfloat16 shares fp32's exponent range, so no loss scaling is "
            "required and grad_scaler() remains disabled on this device.",
            stacklevel=3,
        )
        return torch.bfloat16
    return dtype


def autocast_context(device: Optional[Union[str, torch.device]] = None, enabled: bool = True, dtype: Optional[torch.dtype] = None):
    """
    Return an autocast context manager for the resolved device type.

    Stock ``torch.amp.autocast`` only accepts device types PyTorch itself
    implements (cpu/cuda/xpu/hpu/mps/...).  Vendor backends register their own:
    Ascend NPU needs ``torch_npu.npu.amp.autocast`` and Moore Threads needs
    ``torch_musa``.  Passing ``'npu'`` to ``torch.amp`` raises
    "Expected one of cpu, cuda, ipu, xpu, ...", so those are dispatched to the
    vendor factory with an actionable error when the package is missing.

    XLA is accepted by ``torch.amp`` but must not be given an fp16 dtype: TPU
    has no fp16 compute path.  See :func:`_xla_autocast_dtype`.

    Args:
        device: target device or None to auto-detect
        enabled: False yields a context that is valid on every backend
        dtype: autocast dtype; None keeps each backend's default
    """
    dt = get_device_type(device)

    if not enabled:
        if _torch_amp_accepts(dt):
            return torch.amp.autocast(device_type=dt, enabled=False)
        return contextlib.nullcontext()

    if dt == "xla":
        return torch.amp.autocast(
            device_type=dt, dtype=_xla_autocast_dtype(dtype), enabled=True)

    if _torch_amp_accepts(dt):
        return torch.amp.autocast(device_type=dt, dtype=dtype, enabled=True)

    vendor = _VENDOR_AMP.get(dt)
    if vendor is not None:
        module_name, attr_path, label = vendor
        factory = _resolve_attr(_require_vendor(module_name, label), attr_path)
        return factory(dtype=dtype) if dtype is not None else factory()

    raise RuntimeError(
        f"no autocast implementation available for device type {dt!r}")


def grad_scaler(device: Optional[Union[str, torch.device]] = None,
                enabled: bool = True, init_scale: Optional[float] = None):
    """
    Return a GradScaler for the resolved device, or a disabled one.

    Loss scaling is implemented for CUDA in stock PyTorch (and for XPU in
    recent releases); Ascend NPU ships its own.  Everywhere else the scaler is
    returned *disabled*: an enabled-but-inert scaler is worse than none,
    because it silently removes the fp16 safety net (overflowing gradients are
    never detected, so bad steps are applied instead of skipped).

    XLA/TPU lands in that last group, and *should* stay there: autocast on
    that device is bfloat16 (see :func:`_xla_autocast_dtype`), whose fp32
    exponent range means gradients do not underflow and loss scaling has
    nothing to protect.  Enabling a scaler there would be the inert case.

    ``init_scale`` defaults to :data:`DEFAULT_INIT_SCALE` rather than PyTorch's
    2**16.  The opening steps of a run are the largest gradients it will ever
    produce, so starting at 2**16 routinely overflows fp16's 65504 ceiling on
    the first or second step: the scaler then has to detect inf, skip the
    update and halve the scale.  That is the mechanism working correctly, not
    a fault -- but it costs real skipped optimizer steps at exactly the point
    where the model is most sensitive to them, and it makes a correctly
    functioning scaler look like a NaN-gradient bug to anyone reading grads
    after ``unscale_()``.  Starting lower costs nothing: the scaler doubles
    the scale back up on its own once it sees a clean window, so the dynamic
    range is unchanged and only the wasted warmup is avoided.

    The previous revision accepted ``device_type=``, which no released PyTorch
    accepts, so it always fell through to the positional form and enabled the
    scaler unconditionally -- including on CPU/MPS where nothing scales.
    """
    dt = get_device_type(device)

    if not enabled:
        return _disabled_scaler(dt)

    scale = DEFAULT_INIT_SCALE if init_scale is None else float(init_scale)

    if dt == "npu":
        # Ascend ships its own scaler; stock torch would be inert here.
        scaler_cls = _resolve_attr(
            _require_vendor("torch_npu", "Ascend NPU"), "npu.amp.GradScaler")
        return scaler_cls(init_scale=scale)

    if dt in ("cuda", "xpu") and _device_available(dt):
        return torch.amp.GradScaler(device=dt, enabled=True, init_scale=scale)

    return _disabled_scaler(dt)


def _device_available(device_type: str) -> bool:
    module = getattr(torch, device_type, None)
    return bool(module is not None and getattr(module, "is_available", lambda: False)())


def _disabled_scaler(device_type: str):
    try:
        return torch.amp.GradScaler(device=device_type, enabled=False)
    except Exception:
        return torch.amp.GradScaler(enabled=False)


def set_device(local_rank: int):
    """
    Set the current device for the given local rank.

    Replaces torch.cuda.set_device(local_rank).
    Works with CUDA, XPU, NPU, MUSA, MLU.
    """
    dt = get_device_type()
    mod = getattr(torch, dt, None)
    if mod is not None and hasattr(mod, "set_device"):
        mod.set_device(local_rank)


def current_device_name() -> str:
    """Human-readable name of the current accelerator."""
    return _detect_accelerator() or "cpu"
