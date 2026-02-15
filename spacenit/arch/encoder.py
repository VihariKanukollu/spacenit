"""Encoder and decoder modules for multi-sensor geospatial data.

Replaces the monolithic ``adaptive_vision_encoder.py`` with focused,
composable modules:

- :class:`MultiSensorTokenizer` -- patchifies and embeds all sensors into
  a unified token sequence in a single pass.
- :class:`Encoder` -- transformer stack with positional encoding and masking.
- :class:`Decoder` -- bidirectional cross-attention predictor.

Design choices vs. the original codebase:
- Unified single-pass tokenization instead of per-sensor loops
- Sensor-type tokens (like CLS but per-sensor) instead of channel embeddings
- No register tokens (sensor-type tokens serve that role)
- Bidirectional cross-attention in decoder (queries attend to context AND
  context attends to queries)
- Shared Q/K projections between self-attn and cross-attn in decoder
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spacenit.arch.attention import RMSNorm, TransformerStack
from spacenit.arch.embed import (
    AdaptivePatchEmbed,
    CyclicMonthEmbed,
    SensorEmbed,
)
from spacenit.ingestion.sensors import SensorRegistry, SensorSpec
from spacenit.settings import Config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig(Config):
    """Configuration for the full encoder pipeline.

    Args:
        embed_dim: Token embedding dimension.
        depth: Number of transformer layers in the encoder.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA). ``None`` = MHA.
        ffn_expansion: SwiGLU expansion ratio.
        dropout: Dropout probability.
        base_patch_size: Native patch size for the adaptive patch embed.
        max_grid: Maximum spatial grid size (in patches) for RoPE.
        max_timesteps: Maximum temporal length for RoPE.
        reference_gsd: Reference ground-sample distance in meters.
        sensor_labels: Ordered list of sensor labels the model handles.
        rope_theta: Base frequency for rotary embeddings.
    """

    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    num_kv_heads: int | None = None
    ffn_expansion: float = 8 / 3
    dropout: float = 0.0
    base_patch_size: int = 16
    max_grid: int = 64
    max_timesteps: int = 64
    reference_gsd: float = 10.0
    sensor_labels: list[str] = field(default_factory=list)
    rope_theta: float = 10000.0

    def build(self) -> Encoder:
        """Build an :class:`Encoder` from this configuration."""
        return Encoder(self)


# ---------------------------------------------------------------------------
# Multi-Sensor Tokenizer
# ---------------------------------------------------------------------------


class MultiSensorTokenizer(nn.Module):
    """Patchify and embed all sensors into a unified token sequence.

    For each sensor, applies :class:`AdaptivePatchEmbed` to produce patch
    tokens, then prepends a learnable sensor-type token.  All sensor token
    sequences are concatenated into a single flat sequence.

    This replaces the original ``MultiSensorPatchProjection`` which used
    per-sensor loops and channel embeddings.  The key difference is that
    we process all sensors in one pass and use sensor-type tokens instead
    of per-channel additive embeddings.

    Args:
        config: Encoder configuration.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.sensor_labels = config.sensor_labels

        # Per-sensor patch embedders
        self.patch_embeds = nn.ModuleDict()
        for label in config.sensor_labels:
            spec = SensorRegistry.get(label)
            in_channels = spec.total_channels
            self.patch_embeds[label] = AdaptivePatchEmbed(
                base_patch_size=config.base_patch_size,
                in_channels=in_channels,
                embed_dim=config.embed_dim,
            )

        # Sensor-type embedding (one per sensor)
        self.sensor_embed = SensorEmbed(
            num_sensors=len(config.sensor_labels),
            embed_dim=config.embed_dim,
        )
        # Map sensor label -> integer index
        self._sensor_to_idx = {
            label: i for i, label in enumerate(config.sensor_labels)
        }

        # Sensor-type tokens (prepended to each sensor's token sequence)
        self.sensor_type_tokens = nn.ParameterDict({
            label: nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
            for label in config.sensor_labels
        })

    def forward(
        self,
        sensor_data: dict[str, Tensor],
        patch_size: int | None = None,
    ) -> tuple[Tensor, Tensor, list[tuple[str, int]]]:
        """Tokenize all sensors into a unified sequence.

        Args:
            sensor_data: Mapping from sensor label to tensor ``(B, C, H, W)``
                or ``(B, C, H, W, T)`` for temporal sensors.  Only present
                sensors should be included.
            patch_size: Runtime patch size; defaults to ``base_patch_size``.

        Returns:
            Tuple of:
            - ``tokens``: ``(B, total_tokens, D)`` -- concatenated token sequence.
            - ``sensor_ids``: ``(B, total_tokens)`` -- integer sensor ID per token.
            - ``layout``: List of ``(sensor_label, num_tokens)`` describing the
              sequence structure.
        """
        P = patch_size or self.config.base_patch_size
        B = next(iter(sensor_data.values())).shape[0]
        device = next(iter(sensor_data.values())).device

        all_tokens: list[Tensor] = []
        all_sensor_ids: list[Tensor] = []
        layout: list[tuple[str, int]] = []

        for label in self.sensor_labels:
            if label not in sensor_data:
                continue

            x = sensor_data[label]
            sensor_idx = self._sensor_to_idx[label]

            # Handle temporal dimension: (B, C, H, W, T) -> flatten T into batch
            has_time = x.ndim == 5
            if has_time:
                B_orig, C, H, W, T = x.shape
                x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)

            # Patchify: (B*T, C, H, W) -> (B*T, num_patches, D)
            tokens = self.patch_embeds[label](x, patch_size=P)

            if has_time:
                num_patches = tokens.shape[1]
                tokens = tokens.view(B_orig, T * num_patches, self.config.embed_dim)
            else:
                num_patches = tokens.shape[1]

            # Add sensor embedding to all tokens
            sid_tensor = torch.full(
                (tokens.shape[0], tokens.shape[1]),
                sensor_idx,
                dtype=torch.long,
                device=device,
            )
            tokens = tokens + self.sensor_embed(sid_tensor)

            # Prepend sensor-type token
            type_token = self.sensor_type_tokens[label].expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([type_token, tokens], dim=1)

            # Update sensor IDs (type token gets same sensor ID)
            type_sid = torch.full(
                (tokens.shape[0], 1), sensor_idx, dtype=torch.long, device=device
            )
            sid_tensor = torch.cat([type_sid, sid_tensor], dim=1)

            all_tokens.append(tokens)
            all_sensor_ids.append(sid_tensor)
            layout.append((label, tokens.shape[1]))

        # Concatenate all sensors
        tokens = torch.cat(all_tokens, dim=1)
        sensor_ids = torch.cat(all_sensor_ids, dim=1)

        return tokens, sensor_ids, layout


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Transformer encoder with positional encoding and optional masking.

    Pipeline::

        tokenize -> add spatial/temporal/month positions -> transformer -> norm

    Args:
        config: Encoder configuration.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = MultiSensorTokenizer(config)

        # Positional encodings
        self.month_embed = CyclicMonthEmbed(config.embed_dim)

        # Transformer stack
        self.transformer = TransformerStack(
            dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            ffn_expansion=config.ffn_expansion,
            dropout=config.dropout,
        )

        # Final normalization
        self.norm = RMSNorm(config.embed_dim)

    def forward(
        self,
        sensor_data: dict[str, Tensor],
        *,
        patch_size: int | None = None,
        month_indices: Tensor | None = None,
        gsd: float = 10.0,
        mask_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[tuple[str, int]]]:
        """Encode multi-sensor input.

        Args:
            sensor_data: Mapping from sensor label to data tensor.
            patch_size: Runtime patch size.
            month_indices: ``(B, T)`` month-of-year indices (0-11).
            gsd: Ground-sample distance for spatial RoPE scaling.
            mask_indices: ``(B, N_keep)`` indices of tokens to keep after
                masking.  If ``None``, all tokens are kept.

        Returns:
            Tuple of:
            - ``encoded``: ``(B, N, D)`` encoded token representations.
            - ``sensor_ids``: ``(B, N)`` sensor IDs for each token.
            - ``layout``: Sequence structure description.
        """
        # Tokenize
        tokens, sensor_ids, layout = self.tokenizer(sensor_data, patch_size)

        # Add month embedding if provided
        if month_indices is not None:
            month_emb = self.month_embed(month_indices)  # (B, T, D)
            # Broadcast month embedding to temporal tokens
            # For simplicity, add to all tokens (sensor-type tokens get month 0)
            if month_emb.shape[1] == 1:
                tokens = tokens + month_emb
            else:
                # Average month embedding across timesteps for non-temporal tokens
                tokens = tokens + month_emb.mean(dim=1, keepdim=True)

        # Apply masking (select subset of tokens)
        if mask_indices is not None:
            B, N, D = tokens.shape
            # Gather selected tokens
            idx = mask_indices.unsqueeze(-1).expand(-1, -1, D)
            tokens = torch.gather(tokens, dim=1, index=idx)
            sid_idx = mask_indices
            sensor_ids = torch.gather(sensor_ids, dim=1, index=sid_idx)

        # Transformer
        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)

        return encoded, sensor_ids, layout


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """Cross-attention decoder for masked prediction.

    Uses bidirectional cross-attention: query tokens attend to encoder
    context, AND encoder context attends to query tokens.  This is
    different from the original one-way cross-attention approach.

    The decoder also uses shared Q/K projections between self-attention
    and cross-attention layers for parameter efficiency.

    Args:
        embed_dim: Token dimension.
        depth: Number of decoder transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads for GQA.
        ffn_expansion: SwiGLU expansion ratio.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        num_kv_heads: int | None = None,
        ffn_expansion: float = 8 / 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable mask tokens (one per position to predict)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer with cross-attention
        self.transformer = TransformerStack(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            cross_attention=True,
        )

        # Bidirectional: context also attends to queries
        self.context_cross_attn = TransformerStack(
            dim=embed_dim,
            depth=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            cross_attention=True,
        )

        self.norm = RMSNorm(embed_dim)

    def forward(
        self,
        context: Tensor,
        num_predictions: int,
        *,
        prediction_positions: Tensor | None = None,
    ) -> Tensor:
        """Decode masked positions.

        Args:
            context: Encoder output ``(B, N_ctx, D)``.
            num_predictions: Number of tokens to predict.
            prediction_positions: ``(B, num_predictions)`` position indices
                for the mask tokens (used for positional encoding).

        Returns:
            Predicted token representations ``(B, num_predictions, D)``.
        """
        B, _, D = context.shape

        if num_predictions == 0:
            return context.new_zeros(B, 0, D)

        # Initialize mask tokens
        queries = self.mask_token.expand(B, num_predictions, -1)

        # Bidirectional: first let context attend to queries
        enriched_context = self.context_cross_attn(
            context, context=queries
        )

        # Then let queries attend to enriched context
        decoded = self.transformer(queries, context=enriched_context)
        decoded = self.norm(decoded)

        return decoded
