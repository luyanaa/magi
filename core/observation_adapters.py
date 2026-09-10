"""Observation adapters and signal-reconstruction companions.

The cross-species latent-dynamics path deliberately keeps shared dynamics
independent of channel count and sampling grid. ``ChannelSignalAdapter`` +
``SignalReconstructionHead`` extend that contract to full-signal
reconstruction of ANY neural signal (calcium, voltage, widefield, ...).
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


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

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
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
        mean = compute.mean(dim=1, keepdim=True)
        std = compute.std(dim=1, keepdim=True, unbiased=False)
        rms = compute.square().mean(dim=1, keepdim=True).sqrt()
        summary = torch.cat((mean, std, rms), dim=1).to(signal.dtype)
        return self.temporal(summary).transpose(1, 2)


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

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Summary tokens (B, T', d), pooled over channels — same layout as
        ``GenericSignalAdapter`` for the latent path."""
        channel_tokens = self.forward_channels(signal)  # (B, C, T', d)
        return channel_tokens.mean(dim=1)


class SignalReconstructionHead(nn.Module):
    """Generative decoder: evolved latent (+ channel identity) -> signal.

    The head deliberately does **not** receive the encoder's channel tokens.
    Readout over ``channel_tokens + z_t`` passes the current window through a
    linear map into the prediction of the next window, which admits a lag-copy
    solution: the model scores well by emitting a gain-matched copy of its own
    input while the latent dynamics contribute nothing.  With the tokens
    removed the waveform must be synthesised from the latent, so reconstruction
    gradients carry real information about the dynamics.

    Layout: channel-count independent, channel-preserving.  A learned temporal
    basis supplies the time axis, a per-channel embedding supplies identity,
    and the latent supplies the coefficients (through a FiLM modulation whose
    gain acts as the per-sample coefficient vector), so the generated waveform
    shape varies per sample and channel.  Frames are interpolated to the raw
    sampling grid.

    The readout is evaluated in factorised form: because every term is linear in
    the ``latent_dim`` feature axis, the (B, C, T', d) activation tensor is
    never materialised and the cost stays O((C + T') * d).
    """

    def __init__(
        self,
        latent_dim: int,
        max_channels: int = 2048,
        num_time_basis: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if num_time_basis <= 0:
            raise ValueError("num_time_basis must be positive")
        self.latent_dim = latent_dim
        self.max_channels = max_channels
        self.num_time_basis = num_time_basis
        self.channel_embedding = nn.Parameter(
            torch.randn(max_channels, latent_dim) * 0.02)
        self.temporal_basis = nn.Parameter(
            torch.randn(num_time_basis, latent_dim) * 0.02)
        self.modulation = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.readout = nn.Linear(latent_dim, 1)

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

        gain, shift = self.modulation(latent).chunk(2, dim=-1)   # (B, d)
        weight = self.readout.weight.reshape(-1)                 # (d,)

        time_coef = weight.unsqueeze(0) * self.temporal_basis     # (T', d)
        chan_coef = (weight.unsqueeze(0)
                     * self.channel_embedding[:num_channels])     # (C, d)

        time_mod = time_coef @ gain.transpose(0, 1)               # (T', B)
        chan_mod = chan_coef @ gain.transpose(0, 1)               # (C, B)

        frames = (
            time_coef.sum(dim=-1).reshape(1, 1, -1)               # (1, 1, T')
            + chan_coef.sum(dim=-1).reshape(1, -1, 1)             # (1, C, 1)
            + time_mod.transpose(0, 1).unsqueeze(1)               # (B, 1, T')
            + chan_mod.transpose(0, 1).unsqueeze(-1)              # (B, C, 1)
            + (shift @ weight + self.readout.bias.reshape(())).reshape(-1, 1, 1)
        )
        if target_time_len == self.num_time_basis:
            return frames
        return F.interpolate(
            frames, size=target_time_len, mode="linear", align_corners=True)
