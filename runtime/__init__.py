"""Device and accelerator runtime utilities."""

from .device_utils import (
    get_device, get_device_type, move_to_device, autocast_context,
    grad_scaler, set_device, current_device_name,
)

__all__ = [
    "get_device", "get_device_type", "move_to_device", "autocast_context",
    "grad_scaler", "set_device", "current_device_name",
]
