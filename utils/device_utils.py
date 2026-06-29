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


@functools.lru_cache(maxsize=1)
def _detect_accelerator() -> Optional[str]:
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


def autocast_context(device: Optional[Union[str, torch.device]] = None, enabled: bool = True, dtype: Optional[torch.dtype] = None):
    """
    Return a torch.autocast context manager for the detected device type.

    Replaces torch.cuda.amp.autocast and torch.cpu.amp.autocast.
    """
    dt = get_device_type(device)
    if dtype is None:
        dtype = torch.float16 if dt != "cpu" else torch.bfloat16
    return torch.amp.autocast(device_type=dt, dtype=dtype, enabled=enabled)


def grad_scaler(device: Optional[Union[str, torch.device]] = None, enabled: bool = True):
    """
    Return a GradScaler appropriate for the device.

    Replaces torch.cuda.amp.GradScaler.
    Uses torch.amp.GradScaler (unified API, PyTorch >= 2.0).
    """
    dt = get_device_type(device)
    return torch.amp.GradScaler(device_type=dt, enabled=enabled)


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
