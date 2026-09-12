"""Executable Magi v2 self-supervised pretraining objectives.

The production EEG wrapper owns the online :class:`MagiV2EEGEncoder`.  This
module keeps only objective heads and an EMA copy of that encoder, so masked
and causal objectives update the same encoder used by BrainMoEPINN.
"""

from __future__ import annotations

import copy
import weakref
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embedding import EEGMasking
from .magi_v2 import MagiV2EEGEncoder


class MagiPretrainingObjective(nn.Module):
    """Masked patch, causal patch-NTP, and two-view EMA objectives."""

    def __init__(
        self,
        base_encoder: MagiV2EEGEncoder,
        mask_ratio: float = 0.75,
        momentum: float = 0.999,
        projection_dim: int = 256,
        temperature: float = 0.2,
    ):
        super().__init__()
        if not isinstance(base_encoder, MagiV2EEGEncoder):
            raise TypeError(
                "MagiPretrainingObjective requires a MagiV2EEGEncoder")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self._base_encoder_ref = weakref.ref(base_encoder)
        self.hidden_dim = base_encoder.hidden_dim
        self.patch_size_time = base_encoder.patch_size_time
        self.mask_ratio = float(mask_ratio)
        self.momentum = float(momentum)
        self.temperature = float(temperature)

        self.momentum_encoder = copy.deepcopy(base_encoder)
        for parameter in self.momentum_encoder.parameters():
            parameter.requires_grad = False
        self.momentum_encoder.eval()

        self.masking = EEGMasking(
            mask_ratio=mask_ratio,
            hidden_dim=self.hidden_dim,
        )
        self.mask_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.patch_size_time),
        )
        self.ntp_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.patch_size_time),
        )
        self.proj_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, projection_dim),
        )
        self.momentum_proj_head = copy.deepcopy(self.proj_head)
        for parameter in self.momentum_proj_head.parameters():
            parameter.requires_grad = False
        self.projection_dim = int(projection_dim)
    def train(self, mode: bool = True):
        """Train online heads while keeping the EMA key path deterministic."""
        super().train(mode)
        self.momentum_encoder.eval()
        self.momentum_proj_head.eval()
        return self


    @property
    def base_encoder(self) -> MagiV2EEGEncoder:
        """Return the production encoder without registering it twice."""
        encoder = self._base_encoder_ref()
        if encoder is None:
            raise RuntimeError("the production Magi encoder was released")
        return encoder

    @staticmethod
    def make_two_views(
        eeg: torch.Tensor,
        noise_scale: float = 0.01,
        shift: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create two non-identical, shape-preserving EEG augmentations."""
        if eeg.dim() != 3:
            raise ValueError("eeg must have shape (batch, channels, time)")
        scale = eeg.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
        view1 = eeg
        view2 = eeg + torch.randn_like(eeg) * scale * float(noise_scale)
        if eeg.shape[-1] > 1 and shift:
            view2 = torch.roll(view2, shifts=int(shift), dims=-1)
        return view1, view2

    def masked_prediction(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mask raw patch supports before the transformer and predict them."""
        base = self.base_encoder
        # Compute normalization from the masked waveform support. This keeps
        # all visible tokens invariant to the values hidden by the mask.
        prepared_eeg, prepared_names, prepared_types = (
            base._prepare_inputs(
                eeg, channel_names, channel_types,
                normalize_amplitude=False))
        with torch.no_grad():
            shape_tokens, channels, num_times = base.embed_tokens(
                prepared_eeg, channel_names=prepared_names,
                channel_types=prepared_types, normalize_amplitude=False)
            _, sampled_mask, _ = self.masking(
                shape_tokens, num_channels=channels)
        mask = base.expand_patch_mask(
            sampled_mask, channels, num_times)
        masked_raw = base.mask_raw_patches(
            prepared_eeg, sampled_mask, channels, num_times)
        if (prepared_types is not None
                and base.use_channel_type_embed):
            stats = base._normalization_stats(masked_raw, prepared_types)
            masked_eeg = base._normalize_amplitude(
                masked_raw, prepared_types, stats=stats)
            target_eeg = base._normalize_amplitude(
                prepared_eeg, prepared_types, stats=stats)
        else:
            masked_eeg = masked_raw
            target_eeg = prepared_eeg
        masked_inputs, _, _ = base.embed_tokens(
            masked_eeg, channel_names=prepared_names,
            channel_types=prepared_types, normalize_amplitude=False)
        masked_tokens = masked_inputs.clone()
        mask_token = self.masking.mask_token.expand_as(masked_tokens)
        masked_tokens[mask] = mask_token[mask]
        hidden, _ = base.encode_tokens(
            masked_tokens, channels, num_times, causal=False)
        targets = base.extract_patches(
            target_eeg, channel_types=prepared_types,
            normalize=False).reshape(
                eeg.shape[0], channels * num_times, self.patch_size_time)
        predictions = self.mask_head(hidden)
        return {
            "predictions": predictions,
            "targets": targets,
            "mask": mask,
            "sampled_mask": sampled_mask,
            "masked_tokens": masked_tokens,
            "hidden": hidden,
        }

    def causal_patch_ntp(
        self,
        eeg: torch.Tensor,
        channel_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict patch ``t+1`` from token ``t`` with causal attention."""
        base = self.base_encoder
        tokens, channels, num_times = base.embed_tokens(
            eeg, channel_types=channel_types, causal=True)
        hidden, _ = base.encode_tokens(
            tokens, channels, num_times, causal=True)
        hidden = hidden.reshape(eeg.shape[0], channels, num_times, -1)
        if num_times < 2:
            raise ValueError(
                "causal patch NTP needs at least two non-overlapping patches")
        predictions = self.ntp_head(hidden[:, :, :-1])
        targets = base.extract_patches(
            eeg, channel_types=channel_types,
            step=base.patch_size_time, normalize=False)
        return {
            "predictions": predictions,
            "targets": targets[:, :, 1:],
            "hidden": hidden,
        }

    def contrastive(
        self,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor,
        channel_names1: Optional[List[List[str]]] = None,
        channel_types1: Optional[torch.Tensor] = None,
        channel_names2: Optional[List[List[str]]] = None,
        channel_types2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode distinct query/key views with an EMA key encoder."""
        if eeg1.shape != eeg2.shape:
            raise ValueError("contrastive views must have identical shapes")
        if torch.equal(eeg1.detach(), eeg2.detach()):
            raise ValueError("contrastive views must be distinct augmentations")
        base = self.base_encoder
        _, query_pooler = base(
            eeg1, channel_names=channel_names1,
            channel_types=channel_types1)
        query = F.normalize(self.proj_head(query_pooler), dim=-1)
        with torch.no_grad():
            _, key_pooler = self.momentum_encoder(
                eeg=eeg2,
                channel_names=channel_names2,
                channel_types=channel_types2)
            key = F.normalize(self.momentum_proj_head(key_pooler), dim=-1)
        logits = query @ key.detach().transpose(0, 1)
        logits = logits / self.temperature
        if logits.shape[0] > 1:
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            loss = 1.0 - (query * key.detach()).sum(dim=-1).mean()
        return {"query": query, "key": key, "loss": loss}

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        """EMA-update encoder and key projector after an optimizer step."""
        base = self.base_encoder
        for online, momentum in zip(
                base.parameters(), self.momentum_encoder.parameters()):
            momentum.mul_(self.momentum).add_(
                online.detach(), alpha=1.0 - self.momentum)
        for online, momentum in zip(
                self.proj_head.parameters(), self.momentum_proj_head.parameters()):
            momentum.mul_(self.momentum).add_(
                online.detach(), alpha=1.0 - self.momentum)

    def forward(
        self,
        eeg: torch.Tensor,
        eeg_view2: Optional[torch.Tensor] = None,
        channel_names: Optional[List[List[str]]] = None,
        channel_types: Optional[torch.Tensor] = None,
        channel_names2: Optional[List[List[str]]] = None,
        channel_types2: Optional[torch.Tensor] = None,
        masked_weight: float = 1.0,
        ntp_weight: float = 1.0,
        contrastive_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        masked = self.masked_prediction(
            eeg, channel_names=channel_names, channel_types=channel_types)
        mask = masked["mask"]
        if bool(mask.any()):
            masked_loss = F.smooth_l1_loss(
                masked["predictions"][mask], masked["targets"][mask])
        else:
            masked_loss = masked["predictions"].sum() * 0.0

        ntp = self.causal_patch_ntp(eeg, channel_types=channel_types)
        ntp_loss = F.smooth_l1_loss(ntp["predictions"], ntp["targets"])

        contrastive = None
        contrastive_loss = masked_loss.new_zeros(())
        if eeg_view2 is not None and contrastive_weight:
            contrastive = self.contrastive(
                eeg, eeg_view2,
                channel_names1=channel_names,
                channel_types1=channel_types,
                channel_names2=(
                    channel_names2
                    if channel_names2 is not None else channel_names),
                channel_types2=(
                    channel_types2
                    if channel_types2 is not None else channel_types),
            )
            contrastive_loss = contrastive["loss"]
        total = (
            float(masked_weight) * masked_loss
            + float(ntp_weight) * ntp_loss
            + float(contrastive_weight) * contrastive_loss
        )
        return {
            "loss": total,
            "masked_loss": masked_loss,
            "ntp_loss": ntp_loss,
            "contrastive_loss": contrastive_loss,
            "mask": mask,
            "contrastive": contrastive,
        }


__all__ = ["MagiPretrainingObjective"]
