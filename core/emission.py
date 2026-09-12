"""Observation-channel model for calcium imaging (opt-in).

The salt/homogenized *C. elegans* corpora do not contain "calcium": they
contain a per-neuron z-scored YFP/CFP ratio of a FRET indicator (YC2.60),
sampled at a few volumes per second, median-filtered, detrended and
standardised. Two properties of that channel are structural, not noise:

1. **Low-pass + aliasing.** The indicator's own kinetics and the release's
   smoothing make the observable band-limited well below Nyquist (measured on
   the pilot: 95%-power bandwidth ~1.0 Hz, spectral centroid ~0.17 Hz at
   ~4 Hz sampling), so the observable is a *filtered* version of the state;
   its frame-to-frame autocorrelation is ~0.79 while the 60 s window
   correlation is ~0.
2. **Saturating non-linear converter.** Fluorescence follows a Hill function
   of [Ca2+] (``F = F0 + (Fmax - F0) * [Ca]^h / ([Ca]^h + Kd^h)``), per
   individual because expression level varies -- the Iwasaki-lab/CREST
   treatment of this chain. A per-neuron z-score removes the operating point,
   so the observable is a monotone *re-parameterisation* of the underlying
   state, and a linear decoder in observed units can only approximate it.

``CalciumEmission`` makes both explicit and *estimable*: it maps the decoder's
pre-readout signal to the observed units through a learnable per-channel
first-order low-pass (time constant in **seconds**, applied with the batch's
``dt``) and a learnable per-channel Hill saturation. The fitted time constants
are the quantity of interest -- they can be compared against the independent
AR(1) estimates on the recording (see ``diagnostics/species_parameters.py``)
-- and nothing here touches the shared latent dynamics: it is the observation
channel only.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .hrf import LearnedHemodynamicResponse, BalloonParameters


class SensorEmission(nn.Module):
    """Decoder signal -> observed readout for one modality, per its sensor spec.

    The low-pass time constant is initialised from the reporter's documented
    kinetics when the registry has them (GCaMP8f 67 ms, Voltron 0.8 ms, ...)
    and otherwise from ``init_tau_s``, which is then a fit target rather than a
    literature value. Voltage readouts (GEVI) are treated as linear in the
    state and skip the Hill term; saturating calcium readouts keep it.

    Args:
        max_channels: channel capacity (per-channel parameters)
        spec: resolved :class:`core.sensors.SensorSpec` (imaging + reporter)
        init_tau_s: fallback low-pass time constant in seconds
        init_hill_h / init_hill_kd: initial Hill coefficient / half-saturation
        dt_default_s: frame interval used when a batch supplies no ``dt``
    """

    def __init__(
        self,
        max_channels: int = 2048,
        spec: Optional["object"] = None,
        init_tau_s: float = 1.5,
        init_hill_h: float = 2.0,
        init_hill_kd: float = 1.0,
        dt_default_s: float = 1.0,
        hrf: Optional[Mapping[str, float]] = None,
    ):
        super().__init__()
        if max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if dt_default_s <= 0:
            raise ValueError("dt_default_s must be positive")
        if init_hill_h <= 0 or init_hill_kd <= 0:
            raise ValueError("Hill parameters must be positive")
        self.max_channels = int(max_channels)
        self.dt_default_s = float(dt_default_s)
        self.spec = spec
        self.readout = getattr(spec, "readout", "dff")
        self.apply_hill = self.readout not in ("voltage", "absolute", "bold")
        self.apply_lowpass = self.readout != "bold"
        # BOLD: the haemodynamic response *is* the low-pass, so the first-order
        # indicator filter is replaced by the (learned, HRF-initialised) drive
        # -> BOLD emission below.
        self.hrf = None
        if self.readout == "bold":
            params = BalloonParameters(**{
                k: float(v) for k, v in (hrf or {}).items()
                if k in ("kappa", "gamma", "tau", "alpha", "rho", "v0")})
            self.hrf = LearnedHemodynamicResponse(
                max_channels=max_channels,
                init_from=(self._reference_hrf(params), 0.5))
        if getattr(spec, "dynamics_valid", True) is False:
            raise ValueError(
                "a fixed-tissue/static reporter has no window dynamics to "
                "filter; resolve the profile with a dynamic sensor instead")
        spec_tau = getattr(spec, "emission_tau_s", None)
        if spec_tau:
            init_tau_s = float(spec_tau)
        if init_tau_s <= 0:
            raise ValueError("init_tau_s must be positive")
        # softplus^{-1} of the initial values so the constrained parameters
        # start exactly at the requested values.
        self.log_tau = nn.Parameter(
            torch.full((max_channels,), _inv_softplus(init_tau_s)))
        self.log_hill_h = nn.Parameter(
            torch.full((max_channels,), _inv_softplus(init_hill_h)))
        self.log_hill_kd = nn.Parameter(
            torch.full((max_channels,), _inv_softplus(init_hill_kd)))
        self.log_gain = nn.Parameter(torch.zeros(max_channels))
        self.baseline = nn.Parameter(torch.zeros(max_channels))

    # ---------------------------------------------------------------- views
    def tau_s(self, num_channels: int) -> torch.Tensor:
        return F.softplus(self.log_tau[:num_channels]) + 1e-6

    def hill_parameters(self, num_channels: int) -> Dict[str, torch.Tensor]:
        return {
            "h": F.softplus(self.log_hill_h[:num_channels]) + 1e-6,
            "kd": F.softplus(self.log_hill_kd[:num_channels]) + 1e-6,
        }

    def parameter_summary(self, num_channels: int) -> Dict[str, float]:
        """Detached scalars for logging/reporting (seconds, dimensionless)."""
        tau = self.tau_s(num_channels).detach()
        out = {
            "tau_s_median": float(tau.median()),
            "tau_s_min": float(tau.min()),
            "tau_s_max": float(tau.max()),
            "readout": self.readout,
        }
        spec = getattr(self, "spec", None)
        if spec is not None:
            out["imaging"] = getattr(getattr(spec, "imaging", None), "name", "")
            out["reporter"] = getattr(getattr(spec, "reporter", None), "name", "")
            out["calibration_required"] = bool(
                getattr(spec, "calibration_required", False))
        if self.apply_hill:
            hill = {k: v.detach() for k, v in
                    self.hill_parameters(num_channels).items()}
            out["hill_h_median"] = float(hill["h"].median())
            out["hill_kd_median"] = float(hill["kd"].median())
        if self.hrf is not None:
            out.update(self.hrf.parameter_summary(num_channels, dt=0.5))
        return out

    @staticmethod
    def _reference_hrf(params: BalloonParameters):
        from .hrf import reference_impulse_response
        return reference_impulse_response(duration_s=32.0, dt=0.5, params=params)

    # -------------------------------------------------------------- forward
    def forward(
        self,
        signal: torch.Tensor,
        dt=None,
        channel_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map ``(B, C, T)`` decoder output to observed readout units.

        ``dt`` is the sampling interval of ``signal`` in seconds (scalar or
        per-sample); the low-pass uses it, so the fitted ``tau`` keeps the
        meaning "seconds" for any corpus rate. ``channel_index`` selects the
        per-channel parameters when a batch holds a subset of channels.
        """
        if signal.dim() != 3:
            raise ValueError("signal must have shape (B, channels, time)")
        C = signal.shape[1]
        if C > self.max_channels:
            raise ValueError(
                f"channels {C} exceed emission capacity {self.max_channels}")
        tau = F.softplus(self.log_tau[:C]) + 1e-6
        gain = F.softplus(self.log_gain[:C]) + 1e-6
        bias = self.baseline[:C]
        kd = F.softplus(self.log_hill_kd[:C]) + 1e-6
        h = F.softplus(self.log_hill_h[:C]) + 1e-6

        scaled = signal * gain.view(1, -1, 1) + bias.view(1, -1, 1)
        if self.hrf is not None:
            # latent -> drive was learned above; drive -> BOLD is the
            # structured, HRF-initialised emission (dt is the TR here).
            return self.hrf(scaled, dt if dt is not None else self.dt_default_s)
        filtered = _first_order_lowpass(scaled, tau, dt, self.dt_default_s)
        if not self.apply_hill:
            # Voltage readouts are treated as linear in the (low-passed) state;
            # there is no saturation curve to invert.
            return filtered
        activated = filtered.clamp_min(0.0)
        kd_b = kd.view(1, -1, 1)
        h_b = h.view(1, -1, 1)
        powered = activated.pow(h_b)
        return powered / (powered + kd_b.pow(h_b) + 1e-12)


def _inv_softplus(value: float) -> float:
    import math
    return math.log(math.expm1(max(value, 1e-6)))


def _first_order_lowpass(
    x: torch.Tensor, tau: torch.Tensor, dt, dt_default: float
) -> torch.Tensor:
    """Exact first-order (exponential) filter along time, per channel.

    ``y_t = a * y_{t-1} + (1 - a) * x_t`` with ``a = exp(-dt / tau)``; the
    exact form stays valid when ``dt`` approaches or exceeds ``tau``.
    """
    if dt is None:
        step = torch.as_tensor(dt_default, dtype=x.dtype, device=x.device)
    elif isinstance(dt, torch.Tensor):
        step = dt.detach().to(device=x.device, dtype=x.dtype)
    else:
        step = torch.as_tensor(float(dt), device=x.device, dtype=x.dtype)
    if step.dim() == 0:
        decay = torch.exp(-step / tau)
    else:
        decay = torch.exp(-step.reshape(-1, 1) / tau.reshape(1, -1))
    out = torch.empty_like(x)
    state = torch.zeros_like(x[..., 0])
    for t in range(x.shape[-1]):
        if decay.dim() == 1:
            state = decay * state + (1.0 - decay) * x[..., t]
        else:
            state = (decay * state
                     + (1.0 - decay) * x[..., t].view(x.shape[0], -1))
        out[..., t] = state
    return out
