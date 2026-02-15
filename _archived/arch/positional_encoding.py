"""Sinusoidal positional encodings for spatial and temporal dimensions.

Provides factory functions that produce fixed (non-learnable) positional
embeddings based on sinusoidal frequency decomposition.  The 2-D variant
optionally scales positions by a per-pixel ground-sample distance so that
the encoding is resolution-aware.
"""

import numpy as np
import torch


def sinusoidal_1d(pos: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Compute a 1-D sinusoidal positional encoding.

    Each scalar position is mapped to a vector of interleaved sine and
    cosine components at geometrically increasing frequencies, following
    the scheme introduced in *Attention Is All You Need*.

    Args:
        pos: Position indices of shape ``(L,)`` (or any shape that can be
            flattened to 1-D).  Values may be non-integer.
        encoding_dim: Target embedding dimensionality.  Must be even.

    Returns:
        Encoding matrix of shape ``(L, encoding_dim)`` on the same device
        as *pos*.

    Raises:
        AssertionError: If *encoding_dim* is odd.
    """
    assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    omega = torch.arange(encoding_dim // 2, device=pos.device) / encoding_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (L,)
    out = torch.einsum("l,d->ld", pos, omega)  # (L, D/2)
    encoding_sin = torch.sin(out)  # (L, D/2)
    encoding_cos = torch.cos(out)  # (L, D/2)

    encoding = torch.cat([encoding_sin, encoding_cos], dim=1)  # (L, D)
    return encoding


def sinusoidal_2d(grid: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """Compute a 2-D sinusoidal positional encoding from a coordinate grid.

    The embedding dimension is split evenly between the two spatial axes.
    Each axis is encoded independently with :func:`sinusoidal_1d` and the
    results are concatenated.

    Args:
        grid: Coordinate grid of shape ``(2, H, W)`` where ``grid[0]`` holds
            row positions and ``grid[1]`` holds column positions.
        encoding_dim: Target embedding dimensionality.  Must be even.

    Returns:
        Encoding matrix of shape ``(H*W, encoding_dim)``.

    Raises:
        AssertionError: If *encoding_dim* is odd.
    """
    assert encoding_dim % 2 == 0

    encoding_dim_1d = encoding_dim // 2
    emb_h = sinusoidal_1d(grid[0], encoding_dim_1d)  # (H*W, D/2)
    emb_w = sinusoidal_1d(grid[1], encoding_dim_1d)  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def sinusoidal_2d_with_gsd(
    side_length: int,
    pixel_pitch: torch.Tensor,
    encoding_dim: int,
    device: torch.device,
    prepend_global_token: bool = False,
) -> torch.Tensor:
    """Produce resolution-aware 2-D sinusoidal positional encodings.

    Constructs a regular spatial grid of size ``side_length x side_length``,
    scales every coordinate by the supplied ground-sample distance(s), and
    encodes the result with :func:`sinusoidal_2d`.  When multiple GSD
    values are provided the output contains one encoding per value, stacked
    along a leading dimension.

    Args:
        side_length: Number of grid cells along each spatial axis.
        pixel_pitch: 1-D tensor of ground-sample distances (metres per
            pixel).  Each value produces an independent encoding.
        encoding_dim: Target embedding dimensionality.
        device: Torch device on which to allocate the grid.
        prepend_global_token: If ``True``, a zero-vector is prepended to
            the spatial sequence (position 0) to serve as a global /
            class-level token embedding.

    Returns:
        Positional encoding of shape ``(N, S, encoding_dim)`` where
        ``N = len(pixel_pitch)`` and ``S = side_length**2`` (or
        ``S = side_length**2 + 1`` when *prepend_global_token* is set).
    """
    grid_h = torch.arange(side_length, device=device)
    grid_w = torch.arange(side_length, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # (H, W) each
    grid = torch.stack(grid, dim=0)  # (2, H, W)

    # Scale coordinates by each ground-sample distance
    grid = torch.einsum("chw,n->cnhw", grid, pixel_pitch)  # (2, N, H, W)
    _, n, h, w = grid.shape
    pos_embed = sinusoidal_2d(grid, encoding_dim)  # (N*H*W, D)
    pos_embed = pos_embed.reshape(n, h * w, encoding_dim)

    if prepend_global_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, encoding_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def cyclic_month_table(encoding_dim: int) -> torch.Tensor:
    """Build a sinusoidal encoding table for the twelve calendar months.

    Months are placed at equal angular intervals around a full circle so
    that January and December are close in embedding space.  The table is
    deterministic and independent of any learned parameters.

    Args:
        encoding_dim: Dimensionality of each month embedding.  Must be even.

    Returns:
        Encoding table of shape ``(12, encoding_dim)`` where row *i*
        corresponds to month *i* (0 = January, 11 = December).

    Raises:
        AssertionError: If *encoding_dim* is odd.
    """
    assert encoding_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    dim_per_table = encoding_dim // 2
    sin_table = torch.sin(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    return month_table  # (12, D)
