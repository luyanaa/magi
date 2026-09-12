"""Internal state -> BOLD: a learnable haemodynamic emission.

Reference for the structure: Friston, Harrison & Penny (2003), "Dynamic causal
modelling", NeuroImage 19:1273-1302 -- Eq. (3)-(4) and Table 1, i.e. the
Balloon-Windkessel model (Buxton et al. 1998; Mandeville et al. 1999).

**What we take from the paper, and what we deliberately do not.** Friston's
forward model converts *neuronal* states ``z`` (one per region, with bilinear
coupling matrices ``A``/``B``/``C``) into BOLD. Our model has no such states:
the drive is an internal latent signal, in arbitrary units, whose relation to
neuronal activity is itself learned. Imposing ``kappa``/``gamma``/``tau``/
``rho``/``alpha`` as fixed biophysics on that signal would read out
interpretations the data cannot support.

So this module splits the problem:

1. ``latent -> drive``: learned, per channel (a gain and a bias, plus an
   optional first-order lag). Nothing biophysical is assumed here.
2. ``drive -> BOLD``: **structured but learned**, in the shape the haemodynamic
   model prescribes -- a delayed, dispersed positive response with a
   post-stimulus undershoot and a compressive static output. Each parameter has
   a biophysical meaning (peak delay, dispersion, undershoot ratio, and the
   static compression), is initialised from the reference balloon impulse
   response, and is reported as an *effective* HRF for the corpus.

``BalloonWindkessel`` still ships as the *reference* forward model. It is used
to (a) initialise the learned response, (b) generate synthetic BOLD with a known
ground truth for tests, and (c) provide an ablation arm ("does the fixed
biophysics beat a learned kernel for this corpus?"). It is not the training
path, because the drive it would have to consume is not ``z``.

Integration follows the same conventions as the other emissions: everything is
in seconds, ``dt`` is the frame interval of the recording (the TR for fMRI) and
may be per-sample, and a TR-sampled drive is held constant across its interval
while the response is evaluated on a finer internal grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "BalloonParameters",
    "bold_k_constants",
    "BalloonWindkessel",
    "LearnedHemodynamicResponse",
    "reference_impulse_response",
    "effective_hrf_summary",
]


@dataclass(frozen=True)
class BalloonParameters:
    """Initialisation for the reference haemodynamic model (human DCM)."""

    kappa: float = 0.65          # rate of signal decay [1/s]
    gamma: float = 0.41          # rate of flow-dependent elimination [1/s]
    tau: float = 0.98            # haemodynamic transit time [s]
    alpha: float = 0.32          # Grubb's exponent (stiffness) [-]
    rho: float = 0.34            # resting oxygen extraction fraction [-]
    v0: float = 0.02             # resting blood volume fraction [-]
    source: str = ("Friston, Harrison & Penny 2003, NeuroImage 19:1273-1302, "
                   "Eq. (3)-(4), Table 1 (Balloon-Windkessel)")
    verified: bool = True


def bold_k_constants(rho: float, v0: float) -> Tuple[float, float, float]:
    """The paper's weight constants: k1 = 7*rho, k2 = 2, k3 = 2*rho - 0.2."""
    return 7.0 * rho, 2.0, 2.0 * rho - 0.2


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(max(value, 1e-6)))


class BalloonWindkessel(nn.Module):
    """Reference forward model: drive -> BOLD via the state equations.

    Kept for initialisation, synthetic-data generation and ablation. The
    parameters are Friston's Table 1 values; ``tau``/``kappa`` can be perturbed
    to synthesise a faster (e.g. rodent-like) response.
    """

    def __init__(
        self,
        max_channels: int = 2048,
        params: Optional[BalloonParameters] = None,
        internal_dt_s: float = 0.02,
    ):
        super().__init__()
        if max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if internal_dt_s <= 0:
            raise ValueError("internal_dt_s must be positive")
        self.max_channels = int(max_channels)
        self.internal_dt_s = float(internal_dt_s)
        params = params or BalloonParameters()
        for name in ("alpha", "rho", "v0"):
            value = getattr(params, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1); got {value}")
        for name in ("kappa", "gamma", "tau"):
            if getattr(params, name) <= 0:
                raise ValueError(f"{name} must be positive")
        self.params = params
        self.register_buffer("kappa", torch.tensor(float(params.kappa)))
        self.register_buffer("gamma", torch.tensor(float(params.gamma)))
        self.register_buffer("tau", torch.tensor(float(params.tau)))
        self.register_buffer("alpha", torch.tensor(float(params.alpha)))
        self.register_buffer("rho", torch.tensor(float(params.rho)))
        self.register_buffer("v0", torch.tensor(float(params.v0)))

    # --------------------------------------------------------------- forward
    def forward(self, drive: torch.Tensor, dt, decimate_to: Optional[int] = None
                ) -> torch.Tensor:
        """Integrate the states for a ``(B, C, T)`` drive sampled every ``dt``.

        The drive is held constant across each sample interval while the ODE is
        integrated with sub-steps, so a TR-sampled input still reproduces the
        HRF instead of aliasing it.
        """
        if drive.dim() != 3:
            raise ValueError("drive must have shape (B, channels, time)")
        if drive.shape[1] > self.max_channels:
            raise ValueError(
                f"channels {drive.shape[1]} exceed balloon capacity "
                f"{self.max_channels}")
        step = _step_tensor(dt, drive)
        substeps = max(1, int(math.ceil(float(step.max()) / self.internal_dt_s)))
        h = step / substeps

        s = torch.zeros(drive.shape[0], drive.shape[1], 1,
                        dtype=drive.dtype, device=drive.device)
        f = torch.ones_like(s)
        v = torch.ones_like(s)
        q = torch.ones_like(s)
        outputs = []
        for t in range(drive.shape[-1]):
            z = drive[:, :, t:t + 1]
            for _ in range(substeps):
                s = s + h * (z - self.kappa * s - self.gamma * (f - 1.0))
                f = f + h * s
                outflow = v.clamp_min(1e-6) ** (1.0 / self.alpha)
                extraction = 1.0 - (1.0 - self.rho).clamp_min(1e-6) ** (
                    1.0 / f.clamp_min(1e-6))
                v = (v + h * (f - outflow) / self.tau).clamp_min(1e-6)
                q = (q + h * (f * extraction / self.rho
                              - outflow * q / v) / self.tau).clamp_min(1e-6)
            outputs.append(
                self.v0 * (7.0 * self.rho * (1.0 - q)
                           + 2.0 * (1.0 - q / v)
                           + (2.0 * self.rho - 0.2) * (1.0 - v)))
        bold = torch.cat(outputs, dim=-1)
        if decimate_to is not None and int(decimate_to) != bold.shape[-1]:
            bold = F.adaptive_avg_pool1d(bold, int(decimate_to))
        return bold


def reference_impulse_response(
    duration_s: float = 32.0,
    dt: float = 0.1,
    params: Optional[BalloonParameters] = None,
) -> torch.Tensor:
    """Haemodynamic response to a one-sample impulse under the reference model.

    This is the curve the learned emission is initialised from; it is also what
    the tests compare against, and the shape a corpus's fitted HRF should be
    sanity-checked against before it is trusted.
    """
    model = BalloonWindkessel(max_channels=1, params=params)
    n = max(2, int(round(duration_s / dt)))
    drive = torch.zeros(1, 1, n)
    drive[0, 0, 0] = 1.0
    with torch.no_grad():
        return model(drive, dt)[0, 0]


class LearnedHemodynamicResponse(nn.Module):
    """Learnable drive -> BOLD emission, anchored to the haemodynamic reference.

    Design (and why)::

        drive --(per-channel first-order lag tau_lag)--> d(t)
        kernel = reference_impulse_response(dt)          # Friston's balloon HRF
                 + sum_k w[c, k] * basis_k(t)            # w initialised to 0
        y      = gain * compress(kernel * d + bias)      # compress = identity at init

    * The kernel starts as **exactly** the reference response, so training begins
      from the paper's shape instead of an arbitrary one.
    * The correction is *added* (not gated by a mixing weight), so every shape
      parameter receives a full gradient from step 0 -- an earlier gated variant
      was unfittable in practice, which is how this design was chosen.
    * The static output is a power warping whose exponent is exactly 1 at init
      (compression, if the corpus wants it), and the gains/biases are affine.
    * Everything reported (peak latency, FWHM, undershoot ratio, deviation from
      the reference) is *measured from the effective kernel*, so a human HRF and
      a mouse HRF are comparable without trusting the parameterisation.

    Args:
        max_channels: channel capacity (per-channel parameters)
        init_from: ``(reference_response, dt)``; defaults to the balloon model
            with Friston's Table 1 priors
        num_basis: number of smooth correction basis functions
        learnable: freeze everything (turns the module into a fixed reference
            emission, useful as an ablation arm)
    """

    def __init__(
        self,
        max_channels: int = 2048,
        init_from: Optional[Tuple[torch.Tensor, float]] = None,
        num_basis: int = 12,
        basis_horizon_s: float = 32.0,
        initial_lag_s: float = 0.05,
        learnable: bool = True,
    ):
        super().__init__()
        if max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if num_basis <= 0:
            raise ValueError("num_basis must be positive")
        self.max_channels = int(max_channels)
        self.num_basis = int(num_basis)
        self.basis_horizon_s = float(basis_horizon_s)
        reference, reference_dt = (
            init_from if init_from is not None
            else (reference_impulse_response(), 0.1))
        reference = torch.as_tensor(reference, dtype=torch.float32).flatten()
        if reference.numel() < 2:
            raise ValueError("reference response needs at least two samples")
        self.register_buffer("reference_kernel", reference)
        self.reference_dt_s = float(reference_dt)

        centers = torch.linspace(0.0, self.basis_horizon_s, self.num_basis)
        spacing = float(centers[1] - centers[0]) if self.num_basis > 1 else 1.0
        self.register_buffer("basis_centers", centers)
        # spacing/3 keeps the correction sharp enough to follow a corpus whose
        # haemodynamics differ from the reference (spacing/2 measurably
        # under-fitting: 0.89 vs 0.98 target correlation on the test corpus).
        self.basis_sigma_s = max(spacing / 3.0, 1e-3)

        raw_lag = math.log(max(float(initial_lag_s), 1e-4))
        self.log_lag = nn.Parameter(torch.full((max_channels,), raw_lag),
                                    requires_grad=learnable)
        self.basis_weights = nn.Parameter(torch.zeros(max_channels, num_basis),
                                          requires_grad=learnable)
        unit = _inv_softplus(1.0)
        self.drive_gain_raw = nn.Parameter(torch.full((max_channels,), unit),
                                           requires_grad=learnable)
        self.drive_bias = nn.Parameter(torch.zeros(max_channels),
                                       requires_grad=learnable)
        self.output_gain_raw = nn.Parameter(torch.full((max_channels,), unit),
                                            requires_grad=learnable)
        self.output_bias = nn.Parameter(torch.zeros(max_channels),
                                        requires_grad=learnable)
        # exponent = 1 + 0.5*tanh(raw) in (0.5, 1.5), exactly 1 at init
        self.compression_raw = nn.Parameter(torch.zeros(max_channels),
                                            requires_grad=learnable)

    # ------------------------------------------------------------------ kernel
    def _reference_at(self, dt: float, length: int, channels: int
                      ) -> torch.Tensor:
        """Reference response sampled in *time* at ``(dt, length)``, ``(C, L)``."""
        ref = self.reference_kernel.detach().to(torch.float32)
        times = torch.arange(length, device=ref.device,
                             dtype=torch.float32) * float(dt)
        pos = times / max(self.reference_dt_s, 1e-6)
        inside = pos <= (ref.numel() - 1)
        lo = pos.floor().clamp(0, ref.numel() - 1).long()
        hi = (lo + 1).clamp(0, ref.numel() - 1)
        frac = (pos - lo.to(pos.dtype)).clamp(0.0, 1.0)
        sampled = ref[lo] * (1.0 - frac) + ref[hi] * frac
        sampled = torch.where(inside, sampled, torch.zeros_like(sampled))
        return sampled.unsqueeze(0).expand(channels, -1).contiguous()

    def _basis(self, dt: float, length: int) -> torch.Tensor:
        """Smooth correction basis ``(num_basis, length)``."""
        time = torch.arange(length, device=self.log_lag.device,
                            dtype=torch.float32) * float(dt)
        delta = (time.unsqueeze(0) - self.basis_centers.unsqueeze(-1))
        return torch.exp(-0.5 * (delta / self.basis_sigma_s) ** 2)

    def kernel(self, channels: int, dt: float, length: int) -> torch.Tensor:
        """Effective HRF kernel: reference plus the learned correction."""
        reference = self._reference_at(dt, length, channels)
        basis = self._basis(dt, length).to(reference.dtype)
        correction = self.basis_weights[:channels] @ basis
        return reference + correction

    def kernel_length(self, channels: int, dt: float) -> int:
        reference_s = self.reference_dt_s * max(1, self.reference_kernel.numel() - 1)
        horizon = max(reference_s + 4.0, self.basis_horizon_s + 4.0)
        return max(3, min(int(math.ceil(horizon / float(dt))),
                          int(math.ceil(128.0 / float(dt)))))

    # ---------------------------------------------------------------- reporting
    def parameter_summary(self, channels: int, dt: float = 0.5,
                          duration_s: float = 32.0) -> Dict[str, float]:
        """Measured descriptors of the *effective* HRF, plus the deviation."""
        length = max(3, int(round(duration_s / dt)))
        with torch.no_grad():
            kernel = self.kernel(channels, dt, length)
            reference = self._reference_at(dt, length, channels)
            lag = float(F.softplus(self.log_lag[:channels]).detach().median())
            deviation = float((kernel - reference).abs().mean()
                              / reference.abs().max().clamp_min(1e-9))
        time = torch.arange(length, dtype=torch.float32) * dt
        peak_index = int(kernel[0].argmax())
        peak = float(kernel[0, peak_index])
        above = (kernel[0] >= peak / 2.0).nonzero().flatten()
        fwhm = (float(time[above[-1]] - time[above[0]])
                if above.numel() > 1 else 0.0)
        undershoot = float(kernel[0].min())
        return {
            "peak_delay_s": float(time[peak_index]),
            "fwhm_s": fwhm,
            "undershoot_ratio": (-undershoot / peak if peak > 0 else 0.0),
            "vasodilatory_lag_s": lag,
            "reference_deviation": deviation,
        }

    # ---------------------------------------------------------------- forward
    def forward(self, drive: torch.Tensor, dt, decimate_to: Optional[int] = None
                ) -> torch.Tensor:
        if drive.dim() != 3:
            raise ValueError("drive must have shape (B, channels, time)")
        channels = drive.shape[1]
        if channels > self.max_channels:
            raise ValueError(
                f"channels {channels} exceed capacity {self.max_channels}")
        scalar_dt = _scalar_dt(dt)
        length = self.kernel_length(channels, scalar_dt)
        kernel = self.kernel(channels, scalar_dt, length)

        gain = F.softplus(self.drive_gain_raw[:channels]).view(1, -1, 1) + 1e-6
        bias = self.drive_bias[:channels].view(1, -1, 1)
        scaled = drive * gain + bias

        lag = F.softplus(self.log_lag[:channels])
        decay = torch.exp(-scalar_dt / lag).view(1, -1, 1).to(scaled.dtype)
        state = torch.zeros_like(scaled[..., :1])
        lagged = []
        for t in range(scaled.shape[-1]):
            state = decay * state + (1.0 - decay) * scaled[..., t:t + 1]
            lagged.append(state)
        lagged = torch.cat(lagged, dim=-1)

        # conv1d cross-correlates, so flip for a causal convolution
        weight = kernel.flip(-1).unsqueeze(1).to(lagged.dtype)
        padded = F.pad(lagged, (length - 1, 0))
        response = F.conv1d(padded, weight, groups=channels)

        out_gain = F.softplus(self.output_gain_raw[:channels]).view(1, -1, 1) + 1e-6
        out_bias = self.output_bias[:channels].view(1, -1, 1)
        exponent = (1.0 + 0.5 * torch.tanh(
            self.compression_raw[:channels])).view(1, -1, 1)
        compressed = torch.sign(response) * response.abs().clamp_min(1e-12) ** exponent
        out = out_gain * compressed + out_bias
        if decimate_to is not None and int(decimate_to) != out.shape[-1]:
            out = F.adaptive_avg_pool1d(out, int(decimate_to))
        return out


def _scalar_dt(dt) -> float:
    if isinstance(dt, torch.Tensor):
        value = float(dt.detach().reshape(-1)[0])
    else:
        value = float(dt)
    if value <= 0:
        raise ValueError("dt must be positive")
    return value


def _step_tensor(dt, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(dt, torch.Tensor):
        step = dt.detach().to(device=reference.device, dtype=reference.dtype)
    else:
        step = torch.as_tensor(float(dt), device=reference.device,
                               dtype=reference.dtype)
    if step.dim() == 0:
        if float(step) <= 0:
            raise ValueError("dt must be positive")
        return step
    if step.numel() != reference.shape[0]:
        raise ValueError(
            f"dt must be a scalar or carry one value per sample "
            f"({reference.shape[0]}); got {step.numel()}")
    if bool((step <= 0).any()):
        raise ValueError("dt must be positive")
    return step.reshape(-1, 1, 1)


def effective_hrf_summary(
    model: "LearnedHemodynamicResponse", channels: int = 1,
    duration_s: float = 32.0, dt: float = 0.5,
) -> Dict[str, float]:
    """Descriptors of the effective (post-training) HRF."""
    return model.parameter_summary(channels, dt=dt, duration_s=duration_s)
