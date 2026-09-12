"""Observation adapters and signal-reconstruction companions.

The cross-species latent-dynamics path deliberately keeps shared dynamics
independent of channel count and sampling grid. ``ChannelSignalAdapter`` +
``SignalReconstructionHead`` extend that contract to full-signal
reconstruction of ANY neural signal (calcium, voltage, widefield, ...).
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple


class GenericSignalAdapter(nn.Module):
    """Encode variable-channel temporal signals without modality-specific shape assumptions.

    The adapter summarizes channel-wise mean, standard deviation, and RMS before
    a compact temporal convolution. Spatial/anatomical metadata can be added by
    a modality-specific adapter later; this baseline deliberately keeps the
    shared dynamics independent of channel count and sampling grid.
    """

    def __init__(self, latent_dim: int, kernel_size: int = 8, stride: int = 8):
        super().__init__()
        if latent_dim <= 0 or kernel_size <= 0 or stride <= 0:
            raise ValueError("latent_dim, kernel_size, and stride must be positive")
        self.kernel_size = kernel_size
        self.stride = stride
        self.temporal = nn.Sequential(
            nn.Conv1d(3, latent_dim, kernel_size=kernel_size, stride=stride),
            nn.GELU(),
            nn.GroupNorm(1, latent_dim),
        )

    def forward(self, signal: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if signal.dim() != 3:
            raise ValueError("signal must have shape (B, channels, time)")
        if signal.shape[1] < 1 or signal.shape[2] < self.kernel_size:
            raise ValueError(
                f"signal requires at least one channel and {self.kernel_size} samples")
        # Accumulate the statistics in float32 for stability, then return to
        # the input dtype before the convolution.  Forcing float32 output here
        # breaks any mode where the module weights are half: DeepSpeed fp16
        # casts parameters once at engine init, so a float32 activation
        # reaching a half-precision conv raises "Input type (float) and bias
        # type (c10::Half) should be the same".
        compute = signal.float() if signal.dtype != torch.float32 else signal
        mean, std, rms = _masked_channel_statistics(compute, mask)
        summary = torch.cat((mean, std, rms), dim=1).to(signal.dtype)
        return self.temporal(summary).transpose(1, 2)


def _masked_channel_statistics(
    compute: torch.Tensor, mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel mean/std/rms over *valid* frames only.

    Padded or invalid frames are zero-filled upstream, so an unmasked mean
    would dilute every summary statistic by the invalid fraction (a batch of
    union-aligned worm samples is ~30-60% valid). ``mask`` is a ``(B, C, T)``
    boolean validity tensor; ``None`` keeps the unmasked behaviour.
    """
    if mask is None:
        mean = compute.mean(dim=1, keepdim=True)
        std = compute.std(dim=1, keepdim=True, unbiased=False)
        rms = compute.square().mean(dim=1, keepdim=True).sqrt()
        return mean, std, rms
    if mask.shape != compute.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match signal shape "
            f"{tuple(compute.shape)}")
    valid = mask.to(device=compute.device, dtype=compute.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (compute * valid).sum(dim=1, keepdim=True) / count
    centred = (compute - mean) * valid
    var = centred.square().sum(dim=1, keepdim=True) / count
    std = var.clamp_min(0.0).sqrt()
    rms = ((compute.square() * valid).sum(dim=1, keepdim=True) / count).sqrt()
    return mean, std, rms


class ChannelSignalAdapter(nn.Module):
    """Per-channel temporal encoder that keeps the channel axis.

    Each channel is projected by a SHARED temporal convolution (channel-
    count independent weights), producing channel tokens ``(B, C, T', d)``.
    Pooling over channels reproduces the token layout of
    ``GenericSignalAdapter``, so the two adapters are drop-in compatible for
    the latent-dynamics path; the un-pooled tokens additionally enable
    full-signal reconstruction via ``SignalReconstructionHead``.
    """

    def __init__(self, latent_dim: int, kernel_size: int = 8, stride: int = 8):
        super().__init__()
        if latent_dim <= 0 or kernel_size <= 0 or stride <= 0:
            raise ValueError("latent_dim, kernel_size, and stride must be positive")
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.temporal = nn.Sequential(
            nn.Conv1d(1, latent_dim, kernel_size=kernel_size, stride=stride),
            nn.GELU(),
            nn.GroupNorm(1, latent_dim),
        )

    def _num_frames(self, time_len: int) -> int:
        return (time_len - self.kernel_size) // self.stride + 1

    def forward_channels(self, signal: torch.Tensor) -> torch.Tensor:
        """Tokenize every channel: (B, C, T) -> (B, C, T', d)."""
        if signal.dim() != 3:
            raise ValueError("signal must have shape (B, channels, time)")
        if signal.shape[1] < 1 or signal.shape[2] < self.kernel_size:
            raise ValueError(
                f"signal requires at least one channel and {self.kernel_size} samples")
        B, C, T = signal.shape
        # See GenericSignalAdapter.forward: statistics in fp32, but the conv
        # input must match the weight dtype under DeepSpeed fp16.
        compute = signal.float() if signal.dtype != torch.float32 else signal
        flat = compute.reshape(B * C, 1, T).to(signal.dtype)
        tokens = self.temporal(flat)  # (B*C, d, T')
        T_prime = tokens.shape[-1]
        return tokens.reshape(B, C, T_prime, self.latent_dim)

    def forward(self, signal: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Summary tokens (B, T', d), pooled over channels — same layout as
        ``GenericSignalAdapter`` for the latent path.

        With a validity mask the pool is weighted by each channel's valid
        fraction, so union-aligned padding channels (all-zero) do not dilute
        the token that drives the shared dynamics.
        """
        channel_tokens = self.forward_channels(signal)  # (B, C, T', d)
        if mask is None:
            return channel_tokens.mean(dim=1)
        if mask.dim() != 3 or mask.shape[:2] != signal.shape[:2]:
            raise ValueError(
                "mask must be (B, C, T) matching the signal's channel axis")
        weight = mask.to(dtype=channel_tokens.dtype).mean(dim=-1, keepdim=True)
        denominator = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (channel_tokens * weight.unsqueeze(-1)).sum(dim=1) / denominator


def _linear_resample(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linear resample along the last axis (``align_corners=True``).

    Drop-in equivalent to ``F.interpolate(x, size=target_len, mode="linear",
    align_corners=True)`` -- verified to 2.4e-07 forward and *bit-identical*
    backward -- expressed with indexing and multiplication.

    This exists only as a fallback for accelerators whose ``upsample_linear1d``
    has no backward kernel.  torch_musa 1.3.0 is the known case: forward is
    implemented, backward is not, so training dies with "Could not run
    'aten::upsample_linear1d_backward.grad_input' with arguments from the
    'musa' backend".  Everywhere else the vendor-tuned ``F.interpolate``
    kernel is preferred, which is what :func:`_resample_time` selects.
    """
    time_in = x.shape[-1]
    if target_len == time_in:
        return x
    if target_len == 1:
        return x[..., :1]
    src = torch.arange(target_len, device=x.device, dtype=x.dtype) * (
        (time_in - 1) / (target_len - 1))
    lower = src.floor().long().clamp(0, time_in - 1)
    upper = (lower + 1).clamp(0, time_in - 1)
    frac = (src - lower.to(src.dtype)).to(x.dtype)
    return (x[..., lower] * (1.0 - frac).view(1, 1, -1)
            + x[..., upper] * frac.view(1, 1, -1))


def _portable_resample_required(device_type: str) -> bool:
    """Whether this device needs the hand-rolled resample.

    Only MUSA is known to ship ``upsample_linear1d`` without its backward
    kernel.  Keeping the workaround narrow means every other backend keeps
    the tuned native kernel.
    """
    return device_type == "musa"


def _resample_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample the time axis to ``target_len`` on the backend's best path."""
    if x.shape[-1] == target_len:
        return x
    if _portable_resample_required(x.device.type):
        return _linear_resample(x, target_len)
    return F.interpolate(x, size=target_len, mode="linear", align_corners=True)


class SignalReconstructionHead(nn.Module):
    """Generative decoder: evolved latent (+ channel identity) -> signal.

    The head deliberately does **not** receive the encoder's channel tokens.
    Readout over ``channel_tokens + z_t`` passes the current window through a
    linear map into the prediction of the next window, which admits a lag-copy
    solution: the model scores well by emitting a gain-matched copy of its own
    input while the latent dynamics contribute nothing.  With the tokens
    removed the waveform must be synthesised from the latent, so reconstruction
    gradients carry real information about the dynamics.

    Layout: channel-count independent, channel-preserving, and **mode
    factored**::

        signal[b, c, t] = sum_k gain[b, k] * channel_embedding[c, k] * time_basis[t, k]
                          + channel_bias[c] + time_bias[t] + output_bias

    ``gain`` is produced by a FiLM head from the evolved latent, so the
    waveform is synthesised from the latent alone, while the ``(channel, mode)``
    embedding gives every channel its own temporal shape ("channel identity
    x latent coefficients x shared temporal modes").  This is what makes
    per-channel reconstruction supervision meaningful: an additive-only
    channel term would force every channel onto one shared waveform up to a
    constant, which no per-channel correlation objective can resolve.

    Frames are interpolated to the raw sampling grid.  Cost is
    O(B * K * (C + T')) for the factorised evaluation, with K modes
    (``num_modes``, default ``num_time_basis``), and no (B, C, T', d) tensor is
    materialised.
    """

    def __init__(
        self,
        latent_dim: int,
        max_channels: int = 2048,
        num_time_basis: int = 64,
        hidden_dim: int = 128,
        num_modes: Optional[int] = None,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if num_time_basis <= 0:
            raise ValueError("num_time_basis must be positive")
        if num_modes is not None and num_modes <= 0:
            raise ValueError("num_modes must be positive")
        self.latent_dim = latent_dim
        self.max_channels = max_channels
        self.num_time_basis = num_time_basis
        self.num_modes = int(num_modes or num_time_basis)
        # Mode-factorised init: with K modes, var(channel * basis * gain) is
        # K * s_E^2 * s_B^2 * var(gain); K^-1/4 on each factor keeps the
        # initial waveform O(1) for a unit-variance latent (a vanishing start
        # would leave the head scaling up for most of training).
        mode_scale = float(self.num_modes) ** -0.25
        self.channel_embedding = nn.Parameter(
            torch.randn(max_channels, self.num_modes) * mode_scale)
        self.temporal_basis = nn.Parameter(
            torch.randn(num_time_basis, self.num_modes) * mode_scale)
        self.modulation = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_modes),
        )
        self.channel_bias = nn.Parameter(torch.zeros(max_channels))
        self.time_bias = nn.Parameter(torch.zeros(num_time_basis))
        self.output_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        latent: torch.Tensor,
        num_channels: int,
        target_time_len: int,
    ) -> torch.Tensor:
        """Decode a (B, C, target_time_len) signal from the evolved latent.

        Args:
            latent: (B, d) evolved latent state driving the generation
            num_channels: C, channel count of the signal to generate
            target_time_len: raw sample count T to interpolate back to
        """
        if latent.dim() != 2:
            raise ValueError("latent must have shape (B, d)")
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent dim {latent.shape[-1]} does not match head "
                f"latent_dim {self.latent_dim}")
        if not 0 < num_channels <= self.max_channels:
            raise ValueError(
                f"num_channels {num_channels} outside head capacity "
                f"1..{self.max_channels}")
        if target_time_len < 1:
            raise ValueError("target_time_len must be positive")

        gain = self.modulation(latent)                          # (B, K)
        channels = self.channel_embedding[:num_channels]         # (C, K)
        basis = self.temporal_basis                              # (T', K)
        frames = torch.einsum("bk,ck,tk->bct", gain, channels, basis)
        frames = frames + self.channel_bias[:num_channels].view(1, -1, 1)
        frames = frames + self.time_bias.view(1, 1, -1) + self.output_bias
        if target_time_len == self.num_time_basis:
            return frames
        return _resample_time(frames, target_time_len)
