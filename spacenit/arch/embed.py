"""Patch embedding and positional encoding modules.

Combines adaptive patch embedding with rotary and cyclic positional
encodings into a single module.

Design choices vs. the original codebase:
- ``nn.Unfold`` + ``nn.Linear`` instead of ``nn.Conv2d`` for patch embedding
- ``F.interpolate`` with area mode (not bicubic) for resolution adaptation
- 2D rotary embeddings scaled by GSD for spatial positions
- 1D rotary embeddings for temporal positions
- Cyclic month encoding via a precomputed sin/cos lookup table
- Simple ``nn.Embedding`` for sensor-type embeddings
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spacenit.arch.attention import _build_rope_freqs


# ---------------------------------------------------------------------------
# Adaptive Patch Embedding (FlexiViT-style)
# ---------------------------------------------------------------------------


class AdaptivePatchEmbed(nn.Module):
    """Patchify and linearly project image patches to token embeddings.

    Uses ``nn.Unfold`` to extract patches and ``nn.Linear`` to project them,
    which is mathematically equivalent to ``nn.Conv2d`` but uses a different
    code path.  Supports runtime patch-size adaptation by resizing the
    projection weights via area-mode interpolation (FlexiViT idea).

    Args:
        base_patch_size: The native patch size the weights are trained for.
        in_channels: Number of input channels.
        embed_dim: Output embedding dimension.
        bias: Whether the linear projection includes bias.
    """

    def __init__(
        self,
        base_patch_size: int,
        in_channels: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.base_patch_size = base_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        flat_dim = in_channels * base_patch_size * base_patch_size
        self.proj = nn.Linear(flat_dim, embed_dim, bias=bias)

    def _get_weight_for_patch_size(self, patch_size: int) -> Tensor:
        """Adapt the projection weight matrix to a different patch size.

        Reshapes the weight to ``(embed_dim, C, base_P, base_P)`` and
        resizes it to ``(embed_dim, C, P, P)`` using area interpolation,
        then flattens back.
        """
        if patch_size == self.base_patch_size:
            return self.proj.weight

        P0 = self.base_patch_size
        W = self.proj.weight.view(self.embed_dim, self.in_channels, P0, P0)
        W = F.interpolate(W.float(), size=(patch_size, patch_size), mode="area")
        return W.reshape(self.embed_dim, -1).to(self.proj.weight.dtype)

    def forward(self, x: Tensor, patch_size: int | None = None) -> Tensor:
        """Patchify and embed.

        Args:
            x: Input tensor ``(B, C, H, W)``.
            patch_size: Runtime patch size; defaults to ``base_patch_size``.

        Returns:
            Token embeddings ``(B, num_patches, embed_dim)`` where
            ``num_patches = (H // P) * (W // P)``.
        """
        P = patch_size or self.base_patch_size
        B, C, H, W = x.shape
        assert H % P == 0 and W % P == 0, (
            f"Spatial dims ({H}, {W}) must be divisible by patch_size {P}"
        )

        # Unfold into patches: (B, C*P*P, num_patches)
        patches = F.unfold(x, kernel_size=P, stride=P)
        # -> (B, num_patches, C*P*P)
        patches = patches.transpose(1, 2)

        # Project with (possibly resized) weight
        weight = self._get_weight_for_patch_size(P)
        bias = self.proj.bias
        tokens = F.linear(patches, weight, bias)

        return tokens


# ---------------------------------------------------------------------------
# Spatial Rotary Positional Embedding (2D, GSD-aware)
# ---------------------------------------------------------------------------


class SpatialRoPE(nn.Module):
    """2D rotary positional embeddings scaled by ground-sample distance.

    Encodes (row, col) grid positions using rotary embeddings.  The
    frequencies are scaled by the sensor's GSD so that sensors with
    different resolutions share a common spatial reference frame.

    Args:
        dim: Embedding dimension (must be divisible by 4 -- half for row,
            half for col, each needing pairs).
        max_grid: Maximum grid size in either dimension.
        theta: Base frequency for RoPE.
        reference_gsd: The reference GSD in meters.  Actual GSD values
            are divided by this to produce a scale factor.
    """

    def __init__(
        self,
        dim: int,
        max_grid: int = 256,
        theta: float = 10000.0,
        reference_gsd: float = 10.0,
    ) -> None:
        super().__init__()
        assert dim % 4 == 0, f"dim ({dim}) must be divisible by 4 for 2D RoPE"
        self.dim = dim
        self.half_dim = dim // 2
        self.theta = theta
        self.reference_gsd = reference_gsd

        # Precompute base frequencies for one axis (dim//4 pairs)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute position indices
        positions = torch.arange(max_grid).float()
        self.register_buffer("positions", positions, persistent=False)

    def forward(
        self,
        num_rows: int,
        num_cols: int,
        gsd: float | Tensor = 10.0,
    ) -> Tensor:
        """Compute 2D rotary frequencies.

        Args:
            num_rows: Number of patch rows.
            num_cols: Number of patch columns.
            gsd: Ground-sample distance in meters (scalar or per-batch tensor).

        Returns:
            Complex tensor ``(num_rows * num_cols, dim // 2)`` of rotation
            factors suitable for :func:`apply_rotary_emb`.
        """
        scale = gsd / self.reference_gsd
        if isinstance(scale, Tensor):
            scale = scale.float().mean()  # collapse to scalar for grid

        row_pos = self.positions[:num_rows] * scale
        col_pos = self.positions[:num_cols] * scale

        # Compute angles for each axis
        row_angles = torch.outer(row_pos, self.inv_freq)  # (R, dim//4)
        col_angles = torch.outer(col_pos, self.inv_freq)  # (C, dim//4)

        # Build 2D grid: (R, C, dim//4) for each axis
        row_grid = row_angles.unsqueeze(1).expand(-1, num_cols, -1)
        col_grid = col_angles.unsqueeze(0).expand(num_rows, -1, -1)

        # Concatenate row and col angles -> (R*C, dim//2)
        angles = torch.cat([
            row_grid.reshape(-1, row_grid.shape[-1]),
            col_grid.reshape(-1, col_grid.shape[-1]),
        ], dim=-1)

        return torch.polar(torch.ones_like(angles), angles)


# ---------------------------------------------------------------------------
# Temporal Rotary Positional Embedding (1D)
# ---------------------------------------------------------------------------


class TemporalRoPE(nn.Module):
    """1D rotary positional embeddings for the temporal dimension.

    Enables length generalization -- the model can handle longer temporal
    sequences at inference than it saw during training.

    Args:
        dim: Embedding dimension (must be even).
        max_timesteps: Maximum number of temporal positions.
        theta: Base frequency for RoPE.
    """

    def __init__(
        self,
        dim: int,
        max_timesteps: int = 128,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f"dim ({dim}) must be even for 1D RoPE"
        self.dim = dim

        freqs = _build_rope_freqs(dim, max_timesteps, theta)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, num_timesteps: int) -> Tensor:
        """Return complex rotation factors ``(T, dim//2)``."""
        return self.freqs[:num_timesteps]


# ---------------------------------------------------------------------------
# Cyclic Month Embedding
# ---------------------------------------------------------------------------


class CyclicMonthEmbed(nn.Module):
    """Encode month-of-year on a circle using sin/cos.

    Months are mapped to angles on a unit circle so that December and
    January are close together.  The encoding is a precomputed lookup
    table projected to the desired dimension.

    Args:
        embed_dim: Output embedding dimension.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # 12 months, encoded as (sin, cos) pairs
        months = torch.arange(12).float()
        angles = 2.0 * math.pi * months / 12.0
        raw = torch.stack([angles.sin(), angles.cos()], dim=-1)  # (12, 2)

        self.register_buffer("raw_embeddings", raw, persistent=False)
        self.proj = nn.Linear(2, embed_dim, bias=False)

    def forward(self, month_indices: Tensor) -> Tensor:
        """Look up and project month embeddings.

        Args:
            month_indices: Integer tensor of month indices (0-11), any shape.

        Returns:
            Embeddings of shape ``(*month_indices.shape, embed_dim)``.
        """
        raw = self.raw_embeddings[month_indices.long()]  # (..., 2)
        # Under FSDP mixed precision the projection weights may be bfloat16.
        # Ensure dtype matches to avoid matmul dtype mismatches.
        raw = raw.to(dtype=self.proj.weight.dtype)
        return self.proj(raw)


# ---------------------------------------------------------------------------
# Sensor Embedding
# ---------------------------------------------------------------------------


class SensorEmbed(nn.Module):
    """Learnable per-sensor-group embedding.

    Each sensor (or sensor group) gets a unique embedding that is added to
    all tokens originating from that sensor.  Implemented as a simple
    ``nn.Embedding`` indexed by sensor ID, rather than the original
    ``nn.ParameterDict`` approach.

    Args:
        num_sensors: Number of distinct sensor types.
        embed_dim: Embedding dimension.
    """

    def __init__(self, num_sensors: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_sensors, embed_dim)

    def forward(self, sensor_ids: Tensor) -> Tensor:
        """Look up sensor embeddings.

        Args:
            sensor_ids: Integer tensor of sensor IDs, any shape.

        Returns:
            Embeddings of shape ``(*sensor_ids.shape, embed_dim)``.
        """
        return self.embedding(sensor_ids)
