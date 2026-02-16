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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spacenit.arch.attention import TransformerStack
from spacenit.arch.band_tokenization import TokenizationConfig
from spacenit.arch.embed import (
    AdaptivePatchEmbed,
    CyclicMonthEmbed,
    SensorEmbed,
    SpatialRoPE,
    TemporalRoPE,
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
        tokenization_config: Optional :class:`TokenizationConfig` for
            per-spectral-group tokenization.  When ``None``, each sensor's
            channels are tokenized as a single group (legacy behaviour).
    """

    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    num_kv_heads: int | None = None
    ffn_expansion: float = 8 / 3
    dropout: float = 0.0
    drop_path: float = 0.0
    base_patch_size: int = 16
    max_grid: int = 64
    max_timesteps: int = 64
    reference_gsd: float = 10.0
    sensor_labels: list[str] = field(default_factory=list)
    rope_theta: float = 10000.0
    tokenization_config: TokenizationConfig | None = None

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

    When a :class:`TokenizationConfig` is provided, each sensor's spectral
    channels are split into groups (e.g., Sentinel-2's 10 m / 20 m / 60 m
    bands) and each group gets its own :class:`AdaptivePatchEmbed`.  This
    preserves multi-resolution structure instead of forcing all bands
    through a single patch embed.

    Args:
        config: Encoder configuration.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.sensor_labels = config.sensor_labels
        tok_cfg = config.tokenization_config

        # Per-sensor (or per-group) patch embedders.
        # _group_info[label] = list of (channel_indices, embed_key) tuples.
        self.patch_embeds = nn.ModuleDict()
        self._group_info: dict[str, list[tuple[list[int], str]]] = {}

        for label in config.sensor_labels:
            spec = SensorRegistry.get(label)

            if tok_cfg is not None:
                # Multi-group tokenization
                group_indices = tok_cfg.group_indices_for(label)
            else:
                # Legacy: single group with all channels
                group_indices = [list(range(spec.total_channels))]

            group_entries: list[tuple[list[int], str]] = []
            for g_idx, ch_indices in enumerate(group_indices):
                key = f"{label}__g{g_idx}" if len(group_indices) > 1 else label
                self.patch_embeds[key] = AdaptivePatchEmbed(
                    base_patch_size=config.base_patch_size,
                    in_channels=len(ch_indices),
                    embed_dim=config.embed_dim,
                )
                group_entries.append((ch_indices, key))

            self._group_info[label] = group_entries

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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[tuple[str, int]]]:
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
            - ``spatial_ids``: ``(B, total_tokens, 2)`` -- (row, col) patch grid
              position per token.  Sensor-type tokens get ``(-1, -1)``.
            - ``temporal_ids``: ``(B, total_tokens)`` -- timestep index per token.
              Non-temporal tokens and sensor-type tokens get ``-1``.
            - ``layout``: List of ``(sensor_label, num_tokens)`` describing the
              sequence structure.
        """
        P = patch_size or self.config.base_patch_size
        B = next(iter(sensor_data.values())).shape[0]
        device = next(iter(sensor_data.values())).device

        all_tokens: list[Tensor] = []
        all_sensor_ids: list[Tensor] = []
        all_spatial_ids: list[Tensor] = []
        all_temporal_ids: list[Tensor] = []
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
            else:
                _, C, H, W = x.shape
                T = 1

            Hp = H // P
            Wp = W // P

            # Tokenize each spectral group separately, then concatenate.
            group_tokens_list: list[Tensor] = []
            for ch_indices, embed_key in self._group_info[label]:
                # Select channels for this group: (B*T, len(ch_indices), H, W)
                x_group = x[:, ch_indices]
                # Patchify: (B*T, num_patches, D)
                g_tokens = self.patch_embeds[embed_key](x_group, patch_size=P)
                group_tokens_list.append(g_tokens)

            # Concatenate groups: (B*T, num_groups * Hp * Wp, D)
            tokens = torch.cat(group_tokens_list, dim=1)
            num_patches_per_group = Hp * Wp
            num_groups = len(self._group_info[label])
            total_patches = num_patches_per_group * num_groups

            if has_time:
                tokens = tokens.view(B, T * total_patches, self.config.embed_dim)
            # else: already (B, total_patches, D)

            # Build spatial position ids: (row, col) for each patch.
            # Each group produces Hp*Wp patches at the same spatial positions.
            rows = torch.arange(Hp, device=device)
            cols = torch.arange(Wp, device=device)
            grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
            # (Hp*Wp, 2) -- flattened patch grid (same for every group)
            single_group_spatial = torch.stack(
                [grid_r.flatten(), grid_c.flatten()], dim=-1
            )
            # Repeat for each group: (num_groups * Hp*Wp, 2)
            patch_spatial = single_group_spatial.repeat(num_groups, 1)

            if has_time:
                # Repeat spatial ids for each timestep: (T * total_patches, 2)
                patch_spatial = patch_spatial.unsqueeze(0).expand(T, -1, -1).reshape(-1, 2)
                # Temporal ids: each patch in timestep t gets id t
                patch_temporal = (
                    torch.arange(T, device=device)
                    .unsqueeze(1)
                    .expand(-1, total_patches)
                    .reshape(-1)
                )
            else:
                # Non-temporal: all get temporal id -1
                patch_temporal = torch.full(
                    (total_patches,), -1, dtype=torch.long, device=device
                )

            # Expand to batch: (B, N_sensor_tokens, 2) and (B, N_sensor_tokens)
            N_sensor = patch_spatial.shape[0]
            spatial_ids = patch_spatial.unsqueeze(0).expand(B, -1, -1)
            temporal_ids = patch_temporal.unsqueeze(0).expand(B, -1)

            # Add sensor embedding to all tokens
            sid_tensor = torch.full(
                (B, N_sensor),
                sensor_idx,
                dtype=torch.long,
                device=device,
            )
            tokens = tokens + self.sensor_embed(sid_tensor)

            # Prepend sensor-type token
            type_token = self.sensor_type_tokens[label].expand(B, -1, -1)
            tokens = torch.cat([type_token, tokens], dim=1)

            # Sensor-type token gets sensor id, spatial (-1,-1), temporal -1
            type_sid = torch.full((B, 1), sensor_idx, dtype=torch.long, device=device)
            sid_tensor = torch.cat([type_sid, sid_tensor], dim=1)

            type_spatial = torch.full((B, 1, 2), -1, dtype=torch.long, device=device)
            spatial_ids = torch.cat([type_spatial, spatial_ids], dim=1)

            type_temporal = torch.full((B, 1), -1, dtype=torch.long, device=device)
            temporal_ids = torch.cat([type_temporal, temporal_ids], dim=1)

            all_tokens.append(tokens)
            all_sensor_ids.append(sid_tensor)
            all_spatial_ids.append(spatial_ids)
            all_temporal_ids.append(temporal_ids)
            layout.append((label, tokens.shape[1]))

        # Concatenate all sensors
        tokens = torch.cat(all_tokens, dim=1)
        sensor_ids = torch.cat(all_sensor_ids, dim=1)
        spatial_ids = torch.cat(all_spatial_ids, dim=1)
        temporal_ids = torch.cat(all_temporal_ids, dim=1)

        return tokens, sensor_ids, spatial_ids, temporal_ids, layout


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Transformer encoder with positional encoding and optional masking.

    Pipeline::

        tokenize -> add month embedding per-timestep
                  -> build spatial & temporal RoPE frequencies
                  -> transformer(tokens, rope_freqs)
                  -> norm

    Positional information:
    - **Spatial**: 2D rotary embeddings (row, col) scaled by GSD, applied
      inside the attention layers via ``rope_freqs``.
    - **Temporal**: 1D rotary embeddings for the timestep dimension, added
      to the spatial RoPE frequencies.
    - **Month**: Cyclic sin/cos month embedding added *per-timestep* to
      each token based on the token's temporal index.

    Args:
        config: Encoder configuration.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        # Exposed for downstream evaluation: if present, eval uses the model's
        # patch size instead of task defaults (avoids DTensor/FSDP resizing ops).
        self.patch_size = int(config.base_patch_size)

        self.tokenizer = MultiSensorTokenizer(config)

        # Positional encodings
        self.month_embed = CyclicMonthEmbed(config.embed_dim)

        head_dim = config.embed_dim // config.num_heads
        self.spatial_rope = SpatialRoPE(
            dim=head_dim,
            max_grid=config.max_grid,
            theta=config.rope_theta,
            reference_gsd=config.reference_gsd,
        )
        self.temporal_rope = TemporalRoPE(
            dim=head_dim,
            max_timesteps=config.max_timesteps,
            theta=config.rope_theta,
        )

        # Transformer stack
        self.transformer = TransformerStack(
            dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            ffn_expansion=config.ffn_expansion,
            dropout=config.dropout,
            drop_path=config.drop_path,
        )

        # Final normalization (OLMo-Earth uses LayerNorm)
        self.norm = nn.LayerNorm(config.embed_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rope_freqs(
        self,
        spatial_ids: Tensor,
        temporal_ids: Tensor,
        gsd: float,
    ) -> Tensor:
        """Build per-token RoPE frequencies from spatial/temporal metadata.

        For each token we look up its (row, col) spatial RoPE and its
        timestep temporal RoPE, then combine them.  Sensor-type tokens
        (spatial_ids == -1) get zero-phase rotations (identity).

        Args:
            spatial_ids: ``(B, N, 2)`` -- (row, col) per token.
            temporal_ids: ``(B, N)`` -- timestep index per token.
            gsd: Ground-sample distance for spatial scaling.

        Returns:
            ``(B, N, head_dim // 2)`` complex rotation factors.
        """
        B, N, _ = spatial_ids.shape
        device = spatial_ids.device
        head_dim = self.config.embed_dim // self.config.num_heads

        # Compute spatial RoPE table for the grid range present
        max_row = int(spatial_ids[..., 0].max().clamp(min=0).item()) + 1
        max_col = int(spatial_ids[..., 1].max().clamp(min=0).item()) + 1
        # spatial_table: (max_row * max_col, head_dim // 2) complex
        spatial_table = self.spatial_rope(max_row, max_col, gsd)

        # Compute temporal RoPE table
        max_t = int(temporal_ids.max().clamp(min=0).item()) + 1
        # temporal_table: (max_t, head_dim // 2) complex
        temporal_table = self.temporal_rope(max_t)

        # Map each token to its spatial RoPE entry
        rows = spatial_ids[..., 0].clamp(min=0)  # (B, N)
        cols = spatial_ids[..., 1].clamp(min=0)  # (B, N)
        flat_spatial_idx = rows * max_col + cols  # (B, N)
        # (B, N, head_dim//2) -- index into the spatial table
        spatial_freqs = spatial_table[flat_spatial_idx.long()]

        # Map each token to its temporal RoPE entry
        t_idx = temporal_ids.clamp(min=0)  # (B, N)
        temporal_freqs = temporal_table[t_idx.long()]  # (B, N, head_dim//2)

        # Combine: multiply complex rotations (adds angles)
        combined = spatial_freqs * temporal_freqs

        # Zero out (set to identity = 1+0j) for sensor-type tokens
        is_type_token = (spatial_ids[..., 0] == -1)  # (B, N)
        identity = torch.ones(1, dtype=combined.dtype, device=device)
        combined = torch.where(
            is_type_token.unsqueeze(-1).expand_as(combined),
            identity,
            combined,
        )

        return combined

    def _apply_month_embedding(
        self,
        tokens: Tensor,
        temporal_ids: Tensor,
        month_indices: Tensor,
    ) -> Tensor:
        """Add per-timestep month embedding to each token.

        Each token gets the month embedding corresponding to its timestep.
        Sensor-type tokens (temporal_id == -1) and non-temporal tokens
        get the mean month embedding as a fallback.

        Args:
            tokens: ``(B, N, D)`` token embeddings.
            temporal_ids: ``(B, N)`` timestep index per token (-1 for
                sensor-type / non-temporal tokens).
            month_indices: ``(B, T)`` month-of-year indices (0-11).

        Returns:
            ``(B, N, D)`` tokens with month embeddings added.
        """
        B, T = month_indices.shape
        # (B, T, D)
        month_emb = self.month_embed(month_indices)

        if T == 1:
            # Single timestep: broadcast to all tokens
            return tokens + month_emb

        # For tokens with a valid temporal id, use the corresponding month.
        # For sensor-type tokens (temporal_id == -1), use the mean.
        mean_month = month_emb.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Clamp temporal ids to valid range for gathering
        t_idx = temporal_ids.clamp(min=0, max=T - 1)  # (B, N)
        # (B, N, D)
        per_token_month = torch.gather(
            month_emb, dim=1,
            index=t_idx.unsqueeze(-1).expand(-1, -1, month_emb.shape[-1]),
        )

        # Mask: use mean for tokens without a valid timestep
        has_timestep = (temporal_ids >= 0).unsqueeze(-1)  # (B, N, 1)
        per_token_month = torch.where(has_timestep, per_token_month, mean_month)

        return tokens + per_token_month

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sensor_data: dict[str, Tensor],
        *,
        patch_size: int | None = None,
        month_indices: Tensor | None = None,
        gsd: float = 10.0,
        mask_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[tuple[str, int]]]:
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
            - ``spatial_ids``: ``(B, N, 2)`` (row, col) per token
              (``-1`` for sensor-type tokens).
            - ``temporal_ids``: ``(B, N)`` timestep index per token
              (``-1`` for sensor-type / non-temporal tokens).
            - ``layout``: Sequence structure description.
        """
        # Tokenize -- now also returns spatial and temporal position ids
        tokens, sensor_ids, spatial_ids, temporal_ids, layout = self.tokenizer(
            sensor_data, patch_size
        )

        # Add per-timestep month embedding
        if month_indices is not None:
            tokens = self._apply_month_embedding(tokens, temporal_ids, month_indices)

        # Build RoPE frequencies from spatial/temporal metadata.
        # IMPORTANT: incorporate patch_size so spatial positions are in
        # consistent physical units across variable patch sizes (matches
        # OLMo-Earth's resolution-scaled sincos encodings).
        P = patch_size or self.config.base_patch_size
        rope_freqs = self._build_rope_freqs(spatial_ids, temporal_ids, gsd * float(P))

        # Apply masking (select subset of tokens)
        if mask_indices is not None:
            B, N, D = tokens.shape
            # Gather selected tokens
            idx = mask_indices.unsqueeze(-1).expand(-1, -1, D)
            tokens = torch.gather(tokens, dim=1, index=idx)
            sid_idx = mask_indices
            sensor_ids = torch.gather(sensor_ids, dim=1, index=sid_idx)
            temporal_ids = torch.gather(temporal_ids, dim=1, index=sid_idx)
            # Subset spatial_ids: (B, N, 2) -> gather on dim=1
            sp_idx = mask_indices.unsqueeze(-1).expand(-1, -1, 2)
            spatial_ids = torch.gather(spatial_ids, dim=1, index=sp_idx)
            # Also subset the RoPE frequencies (complex tensors need
            # real-view gather since torch.gather doesn't support complex)
            rope_real = torch.view_as_real(rope_freqs)  # (B, N, F, 2)
            rope_idx = mask_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, rope_real.shape[2], 2
            )
            rope_real = torch.gather(rope_real, dim=1, index=rope_idx)
            rope_freqs = torch.view_as_complex(rope_real)

        # Transformer with RoPE
        encoded = self.transformer(tokens, rope_freqs=rope_freqs)
        encoded = self.norm(encoded)

        return encoded, sensor_ids, spatial_ids, temporal_ids, layout


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """Cross-attention decoder for latent masked-image modelling.

    Follows the OLMo-Earth decoder pattern:

    1. Receive the **full** token sequence (visible + masked positions).
    2. Apply ``input_norm`` then ``encoder_to_decoder_embed`` (learned
       linear projection, even when encoder and decoder dims match).
    3. Replace tokens at *masked* positions with a learnable
       ``mask_token``.
    4. **Re-apply positional encodings** (month + sensor embeddings) to
       ALL tokens, including mask tokens.  This gives mask tokens
       explicit positional identity -- matching OLMo-Earth's
       ``composite_encodings`` call inside the decoder.
    5. Sort tokens so that decode-targets come first and context
       (visible) tokens come last.
    6. Run cross-attention blocks: decode-target tokens attend to
       context tokens.
    7. Recombine tokens into the original ordering and return.

    The caller is responsible for extracting the positions it cares
    about (e.g. ``PREDICTED`` positions for the loss).

    Args:
        embed_dim: Encoder token dimension (input to the decoder).
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
        num_sensors: int = 32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Encoder-to-decoder projection (learned transform, even same-dim).
        self.input_norm = nn.LayerNorm(embed_dim)
        self.encoder_to_decoder_embed = nn.Linear(embed_dim, embed_dim)

        # Learnable mask token (replaces masked positions).
        # Shape (1, 1, D) for checkpoint compatibility with existing saves.
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional re-encoding modules (matching OLMo-Earth's
        # ``composite_encodings`` in the decoder).  These give mask
        # tokens explicit spatial/temporal/sensor identity.
        self.month_embed = CyclicMonthEmbed(embed_dim)
        self.sensor_embed = SensorEmbed(
            num_sensors=num_sensors, embed_dim=embed_dim,
        )

        # Additive sincos spatial + temporal embeddings for the decoder.
        # OLMo-Earth splits the embedding dim into 4 equal parts and
        # fills each with a different sincos encoding.  We follow the
        # same approach: the decoder gets its own additive positional
        # encodings (separate from the encoder's RoPE).
        #
        # We precompute the sincos inverse-frequency vectors and build
        # the actual embeddings on-the-fly from spatial_ids / temporal_ids
        # in _apply_positional_reencoding().
        self._sincos_dim = embed_dim  # full dim, split 50/50 spatial/temporal
        self._spatial_dim = embed_dim // 2  # half for spatial (row+col)
        self._temporal_dim = embed_dim - self._spatial_dim  # other half

        # Cross-attention transformer: decode-targets attend to context.
        self.transformer = TransformerStack(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            cross_attention=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _sincos_1d(positions: Tensor, dim: int) -> Tensor:
        """1D sincos positional encoding (matching OLMo-Earth).

        Args:
            positions: ``(L,)`` or ``(B, L)`` integer positions.
            dim: Encoding dimension (must be even).

        Returns:
            ``(*positions.shape, dim)`` sincos encoding.
        """
        # Mirror olmoearth_pretrain.nn.encodings.get_1d_sincos_pos_encoding.
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        omega = torch.arange(dim // 2, device=positions.device).float() / dim / 2.0
        omega = 1.0 / (10000.0**omega)  # (dim/2,)

        pos = positions.reshape(-1).float()  # (L,)
        out = torch.einsum("l,d->ld", pos, omega)  # (L, dim/2)
        enc = torch.cat([out.sin(), out.cos()], dim=1)  # (L, dim)
        return enc.view(*positions.shape, dim)

    def _apply_positional_reencoding(
        self,
        x: Tensor,
        sensor_ids: Tensor,
        spatial_ids: Tensor | None,
        temporal_ids: Tensor | None,
        month_indices: Tensor | None,
        patch_size: int | None,
        gsd: float,
    ) -> Tensor:
        """Re-apply additive positional encodings to all tokens.

        This is the SpaceNit analogue of OLMo-Earth calling
        ``composite_encodings`` inside the decoder after mask token
        replacement.  It ensures that mask tokens (which start as
        position-blind learned vectors) receive explicit positional
        identity before cross-attention.

        OLMo-Earth's ``composite_encodings`` adds four components
        (channel, time-position, month, spatial) each occupying 25%
        of the embedding dimension.  We add:

        - **Sensor embedding** (learnable, full dim) -- analogous to
          OLMo's channel embeddings.
        - **Month embedding** (cyclic sin/cos, full dim) -- analogous
          to OLMo's month encoding.
        - **Spatial sincos** (frozen, ``_spatial_dim``) -- 2D sincos
          from (row, col), matching OLMo's spatial encoding.
        - **Temporal sincos** (frozen, ``_temporal_dim``) -- 1D sincos
          from timestep index, matching OLMo's time-position encoding.

        Args:
            x: ``(B, N, D)`` decoder-space tokens.
            sensor_ids: ``(B, N)`` sensor ID per token.
            spatial_ids: ``(B, N, 2)`` (row, col) per token (``-1``
                for sensor-type tokens).  ``None`` to skip spatial.
            temporal_ids: ``(B, N)`` timestep index per token (``-1``
                for sensor-type tokens).  ``None`` to skip temporal.
            month_indices: ``(B, T)`` month-of-year indices.  ``None``
                to skip month.

        Returns:
            ``(B, N, D)`` tokens with positional encodings re-added.
        """
        D = x.shape[-1]

        # --- Sensor embedding (always available) ---
        x = x + self.sensor_embed(sensor_ids)

        # --- Month embedding (per-timestep, matching encoder logic) ---
        if month_indices is not None and temporal_ids is not None:
            B, T = month_indices.shape
            month_emb = self.month_embed(month_indices)  # (B, T, D)

            if T == 1:
                x = x + month_emb
            else:
                mean_month = month_emb.mean(dim=1, keepdim=True)
                t_idx = temporal_ids.clamp(min=0, max=T - 1)
                per_token_month = torch.gather(
                    month_emb,
                    dim=1,
                    index=t_idx.unsqueeze(-1).expand(-1, -1, month_emb.shape[-1]),
                )
                has_timestep = (temporal_ids >= 0).unsqueeze(-1)
                per_token_month = torch.where(has_timestep, per_token_month, mean_month)
                x = x + per_token_month

        # --- Spatial sincos (2D: row + col) ---
        # Matching OLMo-Earth's get_2d_sincos_pos_encoding_with_resolution.
        # Each row and col position gets a 1D sincos encoding of half the
        # spatial dim, then they are concatenated.
        if spatial_ids is not None:
            rows = spatial_ids[..., 0].clamp(min=0)  # (B, N)
            cols = spatial_ids[..., 1].clamp(min=0)  # (B, N)
            half_sp = self._spatial_dim // 2

            # Resolution scaling (OLMo-Earth): scale spatial coords by
            # gsd_ratio = (input_res * patch_size / BASE_GSD). Here, gsd
            # is meters/pixel and BASE_GSD == 10.
            P = patch_size or 1
            gsd_ratio = (float(gsd) * float(P)) / 10.0
            rows_f = rows.float() * gsd_ratio
            cols_f = cols.float() * gsd_ratio

            row_enc = self._sincos_1d(rows_f, half_sp)  # (B, N, half_sp)
            col_enc = self._sincos_1d(cols_f, half_sp)  # (B, N, half_sp)
            spatial_enc = torch.cat([row_enc, col_enc], dim=-1)  # (B, N, _spatial_dim)

            # Zero out for sensor-type tokens (spatial_ids == -1)
            is_type = (spatial_ids[..., 0] == -1).unsqueeze(-1)  # (B, N, 1)
            spatial_enc = spatial_enc.where(~is_type, spatial_enc.new_zeros(1))

            # Add to the first _spatial_dim dimensions of x
            x[..., : self._spatial_dim] = x[..., : self._spatial_dim] + spatial_enc.to(x.dtype)

        # --- Temporal sincos (1D: timestep index) ---
        if temporal_ids is not None:
            t_pos = temporal_ids.clamp(min=0)  # (B, N)
            temporal_enc = self._sincos_1d(t_pos, self._temporal_dim)  # (B, N, _temporal_dim)

            # Zero out for non-temporal tokens (temporal_ids == -1)
            is_non_temporal = (temporal_ids == -1).unsqueeze(-1)
            temporal_enc = temporal_enc.where(~is_non_temporal, temporal_enc.new_zeros(1))

            # Add to the last _temporal_dim dimensions of x
            x[..., D - self._temporal_dim :] = (
                x[..., D - self._temporal_dim :] + temporal_enc.to(x.dtype)
            )

        return x

    def forward(
        self,
        all_tokens: Tensor,
        visibility_mask: Tensor,
        sensor_ids: Tensor | None = None,
        spatial_ids: Tensor | None = None,
        temporal_ids: Tensor | None = None,
        month_indices: Tensor | None = None,
        patch_size: int | None = None,
        gsd: float = 10.0,
    ) -> Tensor:
        """Decode the full token sequence.

        Args:
            all_tokens: ``(B, N, D_enc)`` -- full encoder output (visible
                positions have real representations; masked positions can
                be anything -- they will be replaced).
            visibility_mask: ``(B, N)`` integer mask with
                :class:`TokenVisibility` values.  Tokens with value
                ``VISIBLE_ENCODER`` (0) are context; tokens with value
                ``PREDICTED`` (2) are decode targets that get replaced by
                ``mask_token``.  All other values (``TARGET_ONLY``,
                ``ABSENT``) are treated as neither context nor target and
                are dropped from the attention.
            sensor_ids: ``(B, N)`` sensor ID per token (from the
                encoder's tokenizer).  Used for positional re-encoding.
            spatial_ids: ``(B, N, 2)`` (row, col) per token.  Used
                for sincos spatial re-encoding.
            temporal_ids: ``(B, N)`` timestep index per token.  Used
                for sincos temporal re-encoding.
            month_indices: ``(B, T)`` month-of-year indices (0-11).
                Used for positional re-encoding.

        Returns:
            ``(B, N, D)`` -- decoded token sequence in the **original**
            ordering.  The caller should gather the positions it needs
            (typically ``PREDICTED``).
        """
        VIS = 0   # TokenVisibility.VISIBLE_ENCODER
        PRED = 2  # TokenVisibility.PREDICTED

        B, N, D_enc = all_tokens.shape

        # --- 1. Project encoder representations into decoder space ---
        x = self.encoder_to_decoder_embed(self.input_norm(all_tokens))
        D = x.shape[-1]

        # --- 2. Replace masked (PREDICTED) positions with mask_token ---
        is_pred = (visibility_mask == PRED)  # (B, N)
        mask_expand = is_pred.unsqueeze(-1).expand_as(x)  # (B, N, D)
        x = torch.where(mask_expand, self.mask_token.expand_as(x), x)

        # --- 3. Re-apply positional encodings to ALL tokens ---
        # This is critical: mask tokens are position-blind learned vectors.
        # OLMo-Earth re-applies composite_encodings after mask replacement
        # so that mask tokens receive spatial/temporal/sensor identity.
        if sensor_ids is not None:
            x = self._apply_positional_reencoding(
                x,
                sensor_ids,
                spatial_ids,
                temporal_ids,
                month_indices,
                patch_size=patch_size,
                gsd=gsd,
            )

        # --- 4. Sort: PREDICTED first, then VISIBLE, then inactive ---
        is_active = (visibility_mask == VIS) | (visibility_mask == PRED)  # (B, N)

        # Build a sort key: PREDICTED=2 > VISIBLE=0, inactive gets -1
        sort_key = torch.where(is_active, visibility_mask, torch.full_like(visibility_mask, -1))
        sorted_key, sort_indices = torch.sort(sort_key.int(), dim=1, descending=True, stable=True)

        # Gather tokens in sorted order
        idx_expand = sort_indices.unsqueeze(-1).expand(-1, -1, D)
        x_sorted = torch.gather(x, dim=1, index=idx_expand)

        # Count decode-targets and context per sample
        max_pred = int((sorted_key == PRED).sum(dim=1).max().item())
        max_vis = int((sorted_key == VIS).sum(dim=1).max().item())

        if max_pred == 0:
            # Nothing to decode -- return projected tokens as-is.
            return self.norm(x)

        # Split: sorted order is [PRED, VIS, inactive].
        tokens_to_decode = x_sorted[:, :max_pred]                          # (B, max_pred, D)
        context_tokens = (
            x_sorted[:, max_pred : max_pred + max_vis]
            if max_vis > 0
            else x_sorted[:, :0]
        )  # (B, max_vis, D)

        # Cross-attention mask: each decode token can attend to valid
        # context tokens.  Shape ``(B, 1, max_pred, max_vis)`` for
        # broadcasting across attention heads.  ``True`` = attend.
        cross_attn_mask: Tensor | None = None
        if max_vis > 0:
            context_mask = (sorted_key[:, max_pred : max_pred + max_vis] == VIS)  # (B, max_vis)
            # (B, 1, 1, max_vis) -> broadcasts to (B, num_heads, max_pred, max_vis)
            cross_attn_mask = context_mask[:, None, None, :].expand(-1, -1, max_pred, -1)

        # --- 5. Cross-attention: decode-targets attend to context ---
        tokens_to_decode = self.transformer(
            tokens_to_decode,
            context=context_tokens,
            cross_attn_mask=cross_attn_mask,
        )

        # --- 6. Recombine into original ordering ---
        out_sorted = x_sorted.clone()
        out_sorted[:, :max_pred] = tokens_to_decode

        # Unsort: scatter back to original positions
        _, unsort_indices = sort_indices.sort(dim=1)
        unsort_expand = unsort_indices.unsqueeze(-1).expand(-1, -1, D)
        out = torch.gather(out_sorted, dim=1, index=unsort_expand)

        out = self.norm(out)
        return out
