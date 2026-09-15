"""Bounded observation models for C. elegans calcium recordings.

The historical Onuma/Iwasaki reconstruction is a forward chain::

    membrane voltage -> intracellular Ca2+ -> indicator fluorescence.

The inverse chain is only identifiable after the fluorescence operating point,
indicator parameters, and voltage-to-calcium mapping have been supplied.  This
module keeps that distinction explicit: invalid observations become a mask, not
an arbitrary clamp, and current is reported as a capacitive *current density*
unless a membrane area and ionic-current model are supplied.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import math

import torch


VoltageCalciumModel = Literal["linear", "gaussian", "bell_shape"]
VoltageBranch = Literal["ascending", "descending"]


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    """Preserve floating precision while making integer inputs differentiable."""
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.get_default_dtype())
    return tensor


def _finite_parameters(values: Tuple[float, ...]) -> bool:
    return all(math.isfinite(float(value)) for value in values)



DEFAULT_C_M_UF_PER_CM2 = 38.5
"""Membrane capacitance density reported in the local Onuma reference.

This is a reference default, not a total-cell capacitance.  Use an explicit
value and cell area when reporting total current.
"""


@dataclass(frozen=True)
class FluorescenceCalibration:
    """Parameters needed to invert an indicator observation.

    ``f0`` and ``fmax`` must be in the same raw fluorescence units as the
    supplied observation.  A z-scored YC2.60 trace has no such operating point
    and must not be passed here without an external calibration.
    """

    f0: float
    fmax: float
    hill_kd_nM: float
    hill_h: float
    voltage_model: VoltageCalciumModel = "linear"
    c0_nM_per_mV: float = 1.0
    c1_nM: float = 0.0
    voltage_mu_mV: float = 0.0
    voltage_sigma_mV: float = 1.0
    voltage_shift_mV: float = 0.0
    voltage_eta_per_mV: float = 1.0
    capacitance_uF_per_cm2: float = DEFAULT_C_M_UF_PER_CM2

    def validate(self) -> None:
        values = {
            "f0": self.f0,
            "fmax": self.fmax,
            "hill_kd_nM": self.hill_kd_nM,
            "hill_h": self.hill_h,
            "c0_nM_per_mV": self.c0_nM_per_mV,
            "c1_nM": self.c1_nM,
            "voltage_mu_mV": self.voltage_mu_mV,
            "voltage_sigma_mV": self.voltage_sigma_mV,
            "voltage_shift_mV": self.voltage_shift_mV,
            "voltage_eta_per_mV": self.voltage_eta_per_mV,
            "capacitance_uF_per_cm2": self.capacitance_uF_per_cm2,
        }
        if not _finite_parameters(tuple(values.values())):
            raise ValueError("fluorescence calibration parameters must be finite")
        if self.fmax <= self.f0:
            raise ValueError("fmax must be greater than f0")
        if self.hill_kd_nM <= 0 or self.hill_h <= 0:
            raise ValueError("Hill Kd and coefficient must be positive")
        if self.voltage_model not in ("linear", "gaussian", "bell_shape"):
            raise ValueError(f"unknown voltage model {self.voltage_model!r}")
        if self.voltage_model == "gaussian" and self.voltage_sigma_mV <= 0:
            raise ValueError("gaussian voltage_sigma_mV must be positive")
        if self.voltage_model == "bell_shape" and self.voltage_eta_per_mV == 0:
            raise ValueError("bell_shape voltage_eta_per_mV must be non-zero")
        if self.capacitance_uF_per_cm2 <= 0:
            raise ValueError("capacitance density must be positive")


def hill_fluorescence(
    calcium_nM: torch.Tensor,
    *,
    f0: float,
    fmax: float,
    kd_nM: float,
    hill_h: float,
) -> torch.Tensor:
    """Map non-negative Ca2+ concentration to fluorescence units."""
    _validate_hill(f0, fmax, kd_nM, hill_h)
    calcium = _as_float_tensor(calcium_nM)
    activated = calcium.clamp_min(0.0)
    powered = activated.pow(float(hill_h))
    fraction = powered / (powered + float(kd_nM) ** float(hill_h))
    return float(f0) + (float(fmax) - float(f0)) * fraction


def invert_hill_fluorescence(
    fluorescence: torch.Tensor,
    *,
    f0: float,
    fmax: float,
    kd_nM: float,
    hill_h: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert a Hill readout without hiding out-of-range observations.

    Returns ``(calcium_nM, valid)``.  Values outside ``[f0, fmax]`` are
    returned as NaN and marked invalid; they are not silently clipped into a
    plausible concentration.
    """
    _validate_hill(f0, fmax, kd_nM, hill_h)
    values = _as_float_tensor(fluorescence)
    fraction = (values - float(f0)) / (float(fmax) - float(f0))
    valid = torch.isfinite(values) & (fraction >= 0.0) & (fraction <= 1.0)
    safe_fraction = fraction.clamp(min=0.0, max=1.0)
    eps = torch.finfo(values.dtype).eps if values.is_floating_point() else 1e-7
    safe_fraction = safe_fraction.clamp(min=float(eps), max=1.0 - float(eps))
    calcium = float(kd_nM) * (
        safe_fraction / (1.0 - safe_fraction)
    ).pow(1.0 / float(hill_h))
    calcium = torch.where(valid, calcium, torch.full_like(calcium, torch.nan))
    return calcium, valid


def calcium_from_voltage(
    voltage_mV: torch.Tensor,
    *,
    model: VoltageCalciumModel = "linear",
    c0_nM_per_mV: float,
    c1_nM: float,
    mu_mV: float = 0.0,
    sigma_mV: float = 1.0,
    vshift_mV: float = 0.0,
    eta_per_mV: float = 1.0,
) -> torch.Tensor:
    """Apply the voltage-to-Ca mapping used by the historical reconstruction."""
    voltage = _as_float_tensor(voltage_mV)
    if model == "linear":
        calcium = float(c0_nM_per_mV) * voltage + float(c1_nM)
    elif model == "gaussian":
        if sigma_mV <= 0:
            raise ValueError("gaussian sigma must be positive")
        calcium = float(c0_nM_per_mV) * torch.exp(
            -((voltage - float(mu_mV)) ** 2) / (2.0 * float(sigma_mV) ** 2)
        ) + float(c1_nM)
    elif model == "bell_shape":
        if eta_per_mV == 0:
            raise ValueError("bell-shape eta must be non-zero")
        calcium = float(c0_nM_per_mV) * torch.sigmoid(
            float(eta_per_mV) * (voltage - float(vshift_mV))
        ) + float(c1_nM)
    else:
        raise ValueError(f"unknown voltage model {model!r}")
    return calcium.clamp_min(0.0)


def voltage_from_calcium(
    calcium_nM: torch.Tensor,
    *,
    model: VoltageCalciumModel = "linear",
    c0_nM_per_mV: float,
    c1_nM: float,
    mu_mV: float = 0.0,
    sigma_mV: float = 1.0,
    vshift_mV: float = 0.0,
    eta_per_mV: float = 1.0,
    branch: Optional[VoltageBranch] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert a voltage-to-Ca mapping and return its identifiability mask.

    Gaussian mappings are two-to-one.  They require an explicit ``branch``;
    omitting it raises instead of inventing a voltage sign.
    """
    calcium = _as_float_tensor(calcium_nM)
    if model == "linear":
        if c0_nM_per_mV == 0:
            raise ValueError("linear c0_nM_per_mV must be non-zero")
        voltage = (calcium - float(c1_nM)) / float(c0_nM_per_mV)
        valid = torch.isfinite(calcium) & (calcium >= 0.0)
        return torch.where(valid, voltage, torch.full_like(voltage, torch.nan)), valid
    if model == "gaussian":
        if branch not in ("ascending", "descending"):
            raise ValueError(
                "gaussian voltage inversion is non-unique; choose ascending "
                "or descending branch")
        if c0_nM_per_mV <= 0 or sigma_mV <= 0:
            raise ValueError("gaussian c0 and sigma must be positive")
        normalized = (calcium - float(c1_nM)) / float(c0_nM_per_mV)
        valid = torch.isfinite(calcium) & (normalized > 0.0) & (normalized <= 1.0)
        safe = normalized.clamp(min=torch.finfo(calcium.dtype).eps, max=1.0)
        distance = float(sigma_mV) * torch.sqrt(-2.0 * torch.log(safe))
        sign = -1.0 if branch == "ascending" else 1.0
        voltage = float(mu_mV) + sign * distance
        return torch.where(valid, voltage, torch.full_like(voltage, torch.nan)), valid
    if model == "bell_shape":
        if c0_nM_per_mV == 0 or eta_per_mV == 0:
            raise ValueError("bell-shape c0 and eta must be non-zero")
        normalized = (calcium - float(c1_nM)) / float(c0_nM_per_mV)
        valid = torch.isfinite(calcium) & (normalized > 0.0) & (normalized < 1.0)
        eps = torch.finfo(calcium.dtype).eps
        safe = normalized.clamp(min=float(eps), max=1.0 - float(eps))
        voltage = float(vshift_mV) + torch.log(safe / (1.0 - safe)) / float(eta_per_mV)
        return torch.where(valid, voltage, torch.full_like(voltage, torch.nan)), valid
    raise ValueError(f"unknown voltage model {model!r}")


def capacitive_current_density(
    voltage_mV: torch.Tensor,
    *,
    dt_s: float,
    capacitance_uF_per_cm2: float = DEFAULT_C_M_UF_PER_CM2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``C_m dV/dt`` in A/cm² and a derivative-validity mask.

    The first frame has no backward difference and is marked invalid.  This is
    capacitive current density only; ionic current requires an explicit
    conductance/I-V model and cannot be inferred from fluorescence alone.
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if capacitance_uF_per_cm2 <= 0:
        raise ValueError("capacitance density must be positive")
    voltage = _as_float_tensor(voltage_mV)
    if voltage.shape[-1] < 2:
        raise ValueError("voltage trace needs at least two frames")
    current = torch.full_like(voltage, torch.nan)
    dv_dt_v_per_s = (voltage[..., 1:] - voltage[..., :-1]) * 1e-3 / float(dt_s)
    current[..., 1:] = dv_dt_v_per_s * float(capacitance_uF_per_cm2) * 1e-6
    valid = torch.isfinite(voltage)
    derivative_valid = torch.zeros_like(valid, dtype=torch.bool)
    derivative_valid[..., 1:] = valid[..., 1:] & valid[..., :-1]
    return current, derivative_valid


def reconstruct_voltage_current(
    fluorescence: torch.Tensor,
    calibration: FluorescenceCalibration,
    *,
    dt_s: float,
    gaussian_branch: Optional[VoltageBranch] = None,
) -> Dict[str, torch.Tensor]:
    """Invert calibrated fluorescence and derive capacitive current density."""
    calibration.validate()
    calcium, fluorescence_valid = invert_hill_fluorescence(
        fluorescence,
        f0=calibration.f0,
        fmax=calibration.fmax,
        kd_nM=calibration.hill_kd_nM,
        hill_h=calibration.hill_h,
    )
    voltage, voltage_valid = voltage_from_calcium(
        calcium,
        model=calibration.voltage_model,
        c0_nM_per_mV=calibration.c0_nM_per_mV,
        c1_nM=calibration.c1_nM,
        mu_mV=calibration.voltage_mu_mV,
        sigma_mV=calibration.voltage_sigma_mV,
        vshift_mV=calibration.voltage_shift_mV,
        eta_per_mV=calibration.voltage_eta_per_mV,
        branch=gaussian_branch,
    )
    current, current_valid = capacitive_current_density(
        voltage,
        dt_s=dt_s,
        capacitance_uF_per_cm2=calibration.capacitance_uF_per_cm2,
    )
    valid = fluorescence_valid & voltage_valid
    current_valid = current_valid & valid
    return {
        "calcium_nM": calcium,
        "voltage_mV": voltage,
        "capacitive_current_density_A_per_cm2": current,
        "fluorescence_valid": fluorescence_valid,
        "voltage_valid": voltage_valid,
        "current_valid": current_valid,
    }


def _validate_hill(f0: float, fmax: float, kd_nM: float, hill_h: float) -> None:
    if not _finite_parameters((f0, fmax, kd_nM, hill_h)):
        raise ValueError("Hill parameters must be finite")
    if fmax <= f0:
        raise ValueError("fmax must be greater than f0")
    if kd_nM <= 0 or hill_h <= 0:
        raise ValueError("Hill Kd and coefficient must be positive")


__all__ = [
    "DEFAULT_C_M_UF_PER_CM2",
    "FluorescenceCalibration",
    "hill_fluorescence",
    "invert_hill_fluorescence",
    "calcium_from_voltage",
    "voltage_from_calcium",
    "capacitive_current_density",
    "reconstruct_voltage_current",
]
