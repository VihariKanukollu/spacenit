"""Output heads for different pretraining objectives.

Provides:

- :class:`PixelHead` -- reconstructs pixels from tokens using
  ``nn.Fold`` + ``nn.Linear`` (replaces transposed convolution).
- :class:`ProjectionHead` -- MLP projection for contrastive learning
  with L2 normalization.
- :class:`PoolingHead` -- single-query cross-attention pooling
  (replaces multi-query gated averaging).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spacenit.arch.attention import GroupedQueryAttention, RMSNorm


# ---------------------------------------------------------------------------
# Pixel Reconstruction Head
# ---------------------------------------------------------------------------


class PixelHead(nn.Module):
    """Reconstruct pixels from token embeddings.

    Projects each token to a flat patch of pixels, then folds the patches
    back into a spatial grid using ``nn.Fold``.

    This replaces the original ``PixelReconstructor`` which used transposed
    convolutions.

    Args:
        embed_dim: Input token dimension.
        patch_size: Spatial patch size.
        out_channels: Number of output channels (spectral bands).
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        flat_dim = out_channels * patch_size * patch_size
        self.proj = nn.Linear(embed_dim, flat_dim)

    def forward(
        self, tokens: Tensor, height: int, width: int
    ) -> Tensor:
        """Reconstruct a spatial image from tokens.

        Args:
            tokens: ``(B, N, D)`` where ``N = (H // P) * (W // P)``.
            height: Output image height in pixels.
            width: Output image width in pixels.

        Returns:
            Reconstructed image ``(B, C, H, W)``.
        """
        B, N, _ = tokens.shape
        P = self.patch_size

        # Project to flat patches: (B, N, C*P*P)
        patches = self.proj(tokens)

        # Transpose for fold: (B, C*P*P, N)
        patches = patches.transpose(1, 2)

        # Fold patches back into spatial grid
        output = F.fold(
            patches,
            output_size=(height, width),
            kernel_size=P,
            stride=P,
        )

        return output


# ---------------------------------------------------------------------------
# Projection Head (Contrastive)
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Simple two-layer MLP with GELU activation and L2 normalization on
    the output.  Replaces the original ``ProjectionAndPooling`` which
    combined projection with complex pooling logic.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output (projection) dimension.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalize.

        Args:
            x: Input tensor ``(..., in_dim)``.

        Returns:
            L2-normalized projections ``(..., out_dim)``.
        """
        projected = self.net(x)
        return F.normalize(projected, dim=-1)


# ---------------------------------------------------------------------------
# Attention Pooling Head
# ---------------------------------------------------------------------------


class PoolingHead(nn.Module):
    """Single-query cross-attention pooling.

    Uses a single learnable query token that attends to the full token
    sequence via cross-attention, producing a single pooled representation.

    This is simpler than the original ``AttnPool`` which used multi-query
    gated averaging.

    Args:
        embed_dim: Token dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads for GQA.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cross_attn = GroupedQueryAttention(
            dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        self.norm = RMSNorm(embed_dim)

    def forward(self, tokens: Tensor, *, attn_mask: Tensor | None = None) -> Tensor:
        """Pool a token sequence into a single vector.

        Args:
            tokens: ``(B, N, D)`` token sequence.
            attn_mask: Optional boolean mask of shape ``(B, 1, N)`` or
                ``(B, N_query, N_ctx)`` where ``True`` means the query may
                attend to that context token. When ``None``, attends to all
                tokens.

        Returns:
            Pooled representation ``(B, D)``.
        """
        B = tokens.shape[0]
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        pooled = self.cross_attn(query, context=tokens, attn_mask=attn_mask)  # (B, 1, D)
        pooled = self.norm(pooled)
        return pooled.squeeze(1)  # (B, D)
