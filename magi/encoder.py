import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Optional, Tuple, Dict, Any, List
import copy

from brain_moe_pinn.magi.patch_embedding import EEGMasking
from brain_moe_pinn.magi.transformer import BERTEncoder
from brain_moe_pinn.magi.spatio_temporal import (
    EEG2DPatchEmbedding,
    FactorizedTransformerEncoder,
    STANDARD_10_20_COORDS,
    BIOTStyleEmbedding,
    ChannelTypeEmbedding,
)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class FactorizedEEGEncoder(nn.Module):
    """
    Factorized EEG encoder using spatial + temporal attention.
    More efficient for EEG data with structured spatial dimensions.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        spatial_heads: int = 4,
        temporal_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transformer = FactorizedTransformerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            spatial_heads=spatial_heads,
            temporal_heads=temporal_heads,
        )
        # Pooler (average pooling over spatial-temporal tokens)
        self.pooler = nn.Linear(hidden_dim, hidden_dim)
        self.pooler_activation = nn.Tanh()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        num_channels: int,
        num_times: int,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs_embeds: (B, C*T, D) token embeddings
            num_channels: C
            num_times: T
            attention_mask: (B, C*T) boolean mask (unused for now)
        Returns:
            last_hidden_state: (B, C*T, D)
            pooler_output: (B, D)
        """
        last_hidden = self.transformer(inputs_embeds, num_channels, num_times)
        # Pool over all tokens (average)
        pooler_output = self.pooler_activation(self.pooler(last_hidden.mean(dim=1)))
        return last_hidden, pooler_output


class BERTBaseEncoder(nn.Module):
    """
    Custom BERT‑base encoder with Flash Attention / xFormers support.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_flash: bool = True,
        use_xformers: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bert = BERTEncoder(
            vocab_size=30522,  # not used, but required by BERTEncoder
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            use_flash=use_flash,
            use_xformers=use_xformers,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            inputs_embeds: (B, L, D) token embeddings (from patch embedding)
            attention_mask: (B, L) boolean mask (True for real tokens)
        Returns:
            last_hidden_state: (B, L, D)
            pooler_output: (B, D) (CLS token after a linear+tanh)
            (optional) hidden_states, attentions
        """
        # Convert boolean mask to float mask (1 for real, 0 for padding)
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # outputs: (last_hidden_state, pooler_output, ...)
        if isinstance(outputs, tuple):
            return outputs
        else:
            # If outputs is a single tensor, wrap it
            return (outputs,)


class MomentumEncoder(nn.Module):
    """
    Momentum encoder with exponential moving average (EMA) updates.
    Architecturally identical to the base encoder, but its weights are updated via
    m * momentum_encoder + (1 - m) * base_encoder.
    """

    def __init__(self, base_encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        # Create a deep copy of the base encoder's architecture
        self.encoder = copy.deepcopy(base_encoder)
        # Make sure momentum encoder parameters are not updated by gradient descent
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.momentum = momentum
        self.base_encoder = base_encoder

    @torch.no_grad()
    def update(self):
        """
        Update momentum encoder weights via EMA.
        Should be called after each training step.
        """
        for param_q, param_k in zip(
            self.base_encoder.parameters(), self.encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the momentum encoder (no gradients).
        """
        with torch.no_grad():
            return self.encoder(*args, **kwargs)


class EEGFoundationModel(nn.Module):
    """
    Complete EEG foundation model combining:
        - 2D Patch embedding with montage-aware spatial embeddings
        - Factorized Transformer encoder (spatial + temporal attention)
        - Momentum encoder (for contrastive learning)
        - Two heads: masked prediction & contrastive projection
    """

    def __init__(
        self,
        in_channels: int = 19,
        hidden_dim: int = 768,
        patch_size_time: int = 256,
        patch_size_channel: int = 1,
        stride_time: int = 128,
        mask_ratio: float = 0.75,
        momentum: float = 0.999,
        projection_dim: int = 256,
        use_factorized: bool = True,
        channels: Optional[List[str]] = None,
        use_biot_embedding: bool = False,
        use_momentum_encoder: bool = True,
        **bert_kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_factorized = use_factorized
        self.patch_size_time = patch_size_time
        self.patch_size_channel = patch_size_channel
        self.use_biot_embedding = use_biot_embedding
        self.use_momentum_encoder = use_momentum_encoder

        # BIOT-style arbitrary channel embedding (option)
        # When enabled, uses learnable position embeddings for 100+ 10-5 locations
        # instead of requiring rigid channel ordering to standard montage
        if use_biot_embedding:
            self.biot_embed = BIOTStyleEmbedding(
                embedding_dim=hidden_dim,
                use_3d_coords=True,
            )
            # When using BIOT, we don't use montage-aware patching
            # Instead, we process each channel separately and combine
            self.patch_embed = None
            
            # Temporal projection: project time-series to hidden_dim
            # Kernel size = patch_size to extract temporal patches
            self.biot_temporal_proj = nn.Conv1d(
                in_channels=1,  # Single channel at a time (per electrode)
                out_channels=hidden_dim,
                kernel_size=patch_size_time,
                stride=patch_size_time,
            )
        else:
            self.biot_embed = None
            self.biot_temporal_proj = None
            # 2D Patch embedding (with montage awareness)
            if channels is None:
                channels = list(STANDARD_10_20_COORDS.keys())
            self.patch_embed = EEG2DPatchEmbedding(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                patch_size_time=patch_size_time,
                patch_size_channel=patch_size_channel,
                stride_time=stride_time,
                channels=channels,
                use_montage=True,
            )

        # Encoder: either factorized or standard BERT
        if use_factorized:
            self.base_encoder = FactorizedEEGEncoder(hidden_dim=hidden_dim, **bert_kwargs)
        else:
            self.base_encoder = BERTBaseEncoder(hidden_dim=hidden_dim, **bert_kwargs)

        # Channel type embedding for multi-modality (scalp EEG / ECoG / sEEG)
        self.channel_type_embed = ChannelTypeEmbedding(
            num_types=4, embedding_dim=hidden_dim
        )

        if use_momentum_encoder:
            self.momentum_encoder = MomentumEncoder(self.base_encoder, momentum=momentum)
        else:
            self.momentum_encoder = None
        self.masking = EEGMasking(mask_ratio=mask_ratio, hidden_dim=hidden_dim)

        # Masked prediction head (reconstruct original patches)
        # For 2D patching, each token predicts a patch of size (patch_size_channel, patch_size_time)
        self.mask_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, patch_size_time * patch_size_channel),
        )

        # Contrastive projection head (for InfoNCE)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Adversarial subject classifier (GRL-based regularizer)
        # Encourages the representation to be subject-invariant
        self.subject_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1000),  # Assuming max 1000 subjects
        )

        self.projection_dim = projection_dim
        self.in_channels = in_channels

    def forward_adversarial_subject(self, pooler_output: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Adversarial subject classification using GRL.
        """
        # GRL reverses gradients during backprop
        rev_features = grad_reverse(pooler_output, alpha)
        subject_logits = self.subject_classifier(rev_features)
        return subject_logits

    def _get_encoder_output(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        num_channels: int = 1,
        num_times: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to get encoder output regardless of encoder type."""
        if num_times is None:
            num_times = tokens.shape[1]
        if self.use_factorized:
            return self.base_encoder(
                inputs_embeds=tokens,
                num_channels=num_channels,
                num_times=num_times,
                attention_mask=mask,
            )
        else:
            return self.base_encoder(inputs_embeds=tokens, attention_mask=mask)

    def forward_embeddings(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        channel_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Get token embeddings, mask, and patch grid size from raw EEG.

        Args:
            eeg: (B, C, T) raw EEG signals
            channel_names: Optional list of channel names for BIOT-style embedding
            channel_types: Optional (C,) or (B, C) long tensor with modality type:
                0=scalp_EEG, 1=ecog_grid, 2=seeg_depth, 3=unknown
        Returns:
            tokens: (B, num_tokens, D) token embeddings
            mask: (B, num_tokens) valid token mask
            grid_size: (C_patches, T_patches) spatial-temporal patch grid
        """
        if self.use_biot_embedding and self.biot_embed is not None:
            # BIOT-style: Each electrode gets its own positional embedding
            # based on its 10-5 location. Handles arbitrary channel configs.
            # Input: (B, C, T) raw EEG
            # Output: (B, C * T_patches, D) where each (channel, time_patch) is a token
            B, C, T = eeg.shape
            device = eeg.device

            # Get channel names if not provided
            if channel_names is None:
                channel_names = [f'Ch{i}' for i in range(C)]

            # BIOT position embeddings for each channel: (1, C, D)
            biot_embeds = self.biot_embed(channel_names)  # (1, C, D)

            # Project each channel's time series: (B, C, T) -> per-channel patches
            # eeg[:, ch, :] -> (B, 1, T) -> conv -> (B, D, T_patches)
            tokens_list = []
            for ch in range(C):
                eeg_ch = eeg[:, ch:ch+1, :]  # (B, 1, T)
                eeg_proj = self.biot_temporal_proj(eeg_ch)  # (B, D, T_patches)
                eeg_proj = eeg_proj.permute(0, 2, 1)  # (B, T_patches, D)
                # Add channel-specific BIOT embedding
                eeg_proj = eeg_proj + biot_embeds[:, ch:ch+1, :]  # (B, T_patches, D)
                tokens_list.append(eeg_proj)

            # Concatenate all channels: (B, C * T_patches, D)
            tokens = torch.cat(tokens_list, dim=1)  # (B, C*T_patches, D)

            # Add channel type embedding (scalp vs ECoG vs sEEG)
            if channel_types is not None:
                if channel_types.dim() == 1:
                    # (C,) -> expand for all temporal patches
                    type_emb = self.channel_type_embed(channel_types)  # (C, D)
                    type_emb = type_emb.unsqueeze(1)  # (C, 1, D)
                    type_emb = type_emb.expand(-1, tokens_list[0].shape[1], -1)  # (C, T_patches, D)
                    type_emb = type_emb.reshape(1, C * tokens_list[0].shape[1], tokens.shape[-1])  # (1, C*T_patches, D)
                else:
                    # (B, C) -> per-sample type
                    type_emb = self.channel_type_embed(channel_types)  # (B, C, D)
                    type_emb = type_emb.unsqueeze(2).expand(-1, -1, tokens_list[0].shape[1], -1)  # (B, C, T_patches, D)
                    type_emb = type_emb.reshape(B, C * tokens_list[0].shape[1], tokens.shape[-1])  # (B, C*T_patches, D)
                tokens = tokens + type_emb.to(device)

            # Number of temporal patches per channel
            T_patches = tokens_list[0].shape[1]

            mask = torch.ones(B, C * T_patches, dtype=torch.bool, device=device)
            return tokens, mask, (C, T_patches)
        else:
            tokens, mask, (C_patches, T_patches) = self.patch_embed(eeg)
            return tokens, mask, (C_patches, T_patches)

    def forward_masked(
        self,
        eeg: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        return_masked_tokens: bool = False,
        return_pooler: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for masked pretraining.
        
        Args:
            eeg: (B, C, T) raw EEG signals
            channel_names: Optional channel names for BIOT-style embedding
            return_masked_tokens: Whether to return masked tokens
            return_pooler: Whether to return pooler output (e.g. for adversarial loss)
        """
        tokens, mask, (C_patches, T_patches) = self.forward_embeddings(eeg, channel_names)
        masked_tokens, mask_indices, restore_indices = self.masking(tokens, mask)

        # Encode masked tokens
        last_hidden, pooler_output = self._get_encoder_output(
            masked_tokens, mask,
            num_channels=C_patches,
            num_times=T_patches,
        )

        # Predict patches for masked positions
        hidden_masked = last_hidden[restore_indices]  # (num_masked, D)
        pred_patches = self.mask_head(hidden_masked)  # (num_masked, patch_size_time * patch_size_channel)
        pred_patches = pred_patches.view(-1, self.patch_size_channel, self.patch_size_time)

        out = {
            'restored_patches': pred_patches,
            'mask_indices': mask_indices,
            'restore_indices': restore_indices,
        }
        if return_masked_tokens:
            out['masked_tokens'] = masked_tokens
        if return_pooler:
            out['pooler_output'] = pooler_output
        return out

    def forward_contrastive(
        self,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        Args:
            eeg1, eeg2: two augmented views (B, C, T)
        Returns:
            q: projection of view1 from base encoder (B, projection_dim)
            k: projection of view2 from momentum encoder (B, projection_dim)
        """
        # Base encoder on view1
        tokens1, mask1, (C1, T1) = self.patch_embed(eeg1)
        last_hidden1, pooler1 = self._get_encoder_output(
            tokens1, mask1,
            num_channels=C1,
            num_times=T1,
        )
        q = self.proj_head(pooler1)  # (B, projection_dim)

        # Momentum encoder on view2 (or base encoder with no_grad if disabled)
        tokens2, mask2, (C2, T2) = self.patch_embed(eeg2)
        encoder2 = self.momentum_encoder if self.momentum_encoder is not None else self.base_encoder
        with torch.no_grad() if self.momentum_encoder is None else torch.enable_grad():
            if self.use_factorized:
                last_hidden2, pooler2 = encoder2(
                    inputs_embeds=tokens2,
                    num_channels=C2,
                    num_times=T2,
                    attention_mask=mask2,
                )
            else:
                last_hidden2, pooler2 = encoder2(inputs_embeds=tokens2, attention_mask=mask2)
        k = self.proj_head(pooler2)  # (B, projection_dim)

        return q, k

    def update_momentum_encoder(self):
        """Update momentum encoder weights via EMA."""
        if self.momentum_encoder is not None:
            self.momentum_encoder.update()


if __name__ == '__main__':
    # Test standard BERT encoder
    print("Testing EEGFoundationModel with standard BERT encoder...")
    model = EEGFoundationModel(use_factorized=False)
    dummy = torch.randn(4, 19, 2560)
    print("Testing masked forward...")
    masked_out = model.forward_masked(dummy)
    print(f"Restored patches shape: {masked_out['restored_patches'].shape}")

    dummy1 = torch.randn(4, 19, 2560)
    dummy2 = torch.randn(4, 19, 2560)
    print("\nTesting contrastive forward...")
    q, k = model.forward_contrastive(dummy1, dummy2)
    print(f"q shape: {q.shape}, k shape: {k.shape}")

    # Test factorized encoder
    print("\n" + "="*50)
    print("Testing EEGFoundationModel with factorized encoder...")
    model_factorized = EEGFoundationModel(use_factorized=True)
    masked_out2 = model_factorized.forward_masked(dummy)
    print(f"Restored patches shape: {masked_out2['restored_patches'].shape}")

    q2, k2 = model_factorized.forward_contrastive(dummy1, dummy2)
    print(f"q shape: {q2.shape}, k shape: {k2.shape}")

    print("\nAll tests passed!")
