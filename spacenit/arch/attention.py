"""Transformer building blocks with modern design choices.

Implements grouped-query attention (GQA), RMSNorm, SwiGLU feed-forward,
and a post-norm transformer layer.  Uses PyTorch-native
``scaled_dot_product_attention`` for automatic flash/memory-efficient
dispatch.

Design choices vs. the original codebase:
- Post-norm (residual then norm) instead of pre-norm
- RMSNorm instead of LayerNorm
- SwiGLU instead of GELU MLP
- Grouped-query attention instead of standard multi-head attention
- No stochastic depth or LayerScale
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2019).

    Faster than LayerNorm because it skips the mean-centering step.
    Used in LLaMA, Gemma, and other modern architectures.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embedding helpers
# ---------------------------------------------------------------------------


def _build_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> Tensor:
    """Precompute complex-valued rotation frequencies for RoPE.

    Returns a ``(max_len, dim // 2)`` tensor of complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embeddings to the last dimension of *x*.

    Supports both position-table and per-token frequency tensors:

    - **Position table** (legacy): ``freqs`` is ``(T, D//2)`` and is
      broadcast over batch and head dimensions.
    - **Per-token** (new): ``freqs`` is ``(B, T, D//2)`` and is broadcast
      over the head dimension only.

    Args:
        x: ``(B, H, T, D)`` real-valued tensor (queries or keys after
            head reshaping) where ``D`` is even.
        freqs: Complex rotation factors -- either ``(T, D//2)`` or
            ``(B, T, D//2)``.

    Returns:
        Tensor of same shape as *x* with rotary embeddings applied.
    """
    # Reshape x into pairs: (B, H, T, D//2, 2) -> view as complex
    orig_shape = x.shape
    x_pairs = x.float().reshape(*orig_shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_pairs)
    # x_complex: (B, H, T, D//2)

    # Broadcast freqs to match x_complex
    if freqs.ndim == 2:
        # (T, D//2) -> (1, 1, T, D//2)
        freqs = freqs.unsqueeze(0).unsqueeze(0)
    elif freqs.ndim == 3:
        # (B, T, D//2) -> (B, 1, T, D//2)
        freqs = freqs.unsqueeze(1)

    freqs = freqs.to(x_complex.device)

    rotated = x_complex * freqs
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with grouped queries (Ainslie et al., 2023).

    When ``num_kv_heads < num_heads``, key/value heads are shared across
    groups of query heads, reducing KV-cache size and compute.  Setting
    ``num_kv_heads == num_heads`` recovers standard multi-head attention;
    ``num_kv_heads == 1`` gives multi-query attention.

    Args:
        dim: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (must divide ``num_heads``).
        dropout: Attention dropout probability.
        bias: Whether linear projections include bias.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.head_dim = dim // num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        *,
        context: Tensor | None = None,
        rope_freqs: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute (optionally cross-) attention.

        Args:
            x: Query input ``(B, N, D)``.
            context: Key/value source ``(B, M, D)``; defaults to *x* for self-attention.
            rope_freqs: Rotary frequencies ``(T, head_dim//2)`` applied to Q and K.
            attn_mask: Boolean mask ``(B, N, M)`` or ``(N, M)``; ``True`` = attend.

        Returns:
            Output tensor ``(B, N, D)``.
        """
        B, N, _ = x.shape
        kv_src = context if context is not None else x
        M = kv_src.shape[1]

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_src).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_src).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to Q and K
        if rope_freqs is not None:
            # rope_freqs may be (T, D//2) or (B, T, D//2).
            # Slice the sequence dimension appropriately.
            if rope_freqs.ndim == 2:
                q = apply_rotary_emb(q, rope_freqs[:N])
                k = apply_rotary_emb(k, rope_freqs[:M])
            else:
                # (B, T, D//2) -- per-token frequencies
                q = apply_rotary_emb(q, rope_freqs[:, :N])
                k = apply_rotary_emb(k, rope_freqs[:, :M])

        # Expand KV heads for grouped-query attention
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, M, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, M, self.head_dim)

        # Use PyTorch native SDPA (auto-dispatches to flash/efficient kernels)
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_p
        )

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020; Dauphin et al., 2017).

    Replaces the standard ``Linear -> GELU -> Linear`` MLP with a gated
    variant: ``Linear_gate * SiLU(Linear_up)`` followed by ``Linear_down``.
    Uses 2/3 of the hidden dimension per gate to keep parameter count
    comparable to a standard MLP with the same expansion ratio.

    Args:
        dim: Input/output dimension.
        expansion: Hidden dimension multiplier (applied then rounded to
            nearest multiple of 8 for hardware efficiency).
        dropout: Dropout probability after the gated activation.
        bias: Whether linear layers include bias.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 8 / 3,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        # Round to nearest multiple of 8 for tensor-core alignment
        hidden = ((hidden + 7) // 8) * 8

        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Post-Norm Transformer Layer
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer layer with post-norm residual connections.

    Architecture::

        x = norm(x + self_attn(x))
        x = norm(x + ffn(x))

    Optionally includes a cross-attention sub-layer between self-attention
    and the feed-forward network.

    Args:
        dim: Model dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        ffn_expansion: Feed-forward expansion ratio.
        dropout: Dropout probability for attention and FFN.
        bias: Whether to use bias in linear layers.
        cross_attention: Whether to include a cross-attention sub-layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        ffn_expansion: float = 8 / 3,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            dim, num_heads, num_kv_heads, dropout=dropout, bias=bias
        )
        self.self_attn_norm = RMSNorm(dim)

        self.has_cross_attn = cross_attention
        if cross_attention:
            self.cross_attn = GroupedQueryAttention(
                dim, num_heads, num_kv_heads, dropout=dropout, bias=bias
            )
            self.cross_attn_norm = RMSNorm(dim)

        self.ffn = SwiGLUFFN(dim, expansion=ffn_expansion, dropout=dropout, bias=bias)
        self.ffn_norm = RMSNorm(dim)

    def forward(
        self,
        x: Tensor,
        *,
        context: Tensor | None = None,
        rope_freqs: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tokens ``(B, N, D)``.
            context: Cross-attention context ``(B, M, D)``; ignored if the
                block was created without ``cross_attention=True``.
            rope_freqs: Rotary frequencies for positional encoding.
            self_attn_mask: Mask for self-attention.
            cross_attn_mask: Mask for cross-attention.

        Returns:
            Output tokens ``(B, N, D)``.
        """
        # Self-attention + post-norm
        x = self.self_attn_norm(
            x + self.self_attn(x, rope_freqs=rope_freqs, attn_mask=self_attn_mask)
        )

        # Cross-attention + post-norm (optional)
        if self.has_cross_attn and context is not None:
            x = self.cross_attn_norm(
                x + self.cross_attn(x, context=context, attn_mask=cross_attn_mask)
            )

        # Feed-forward + post-norm
        x = self.ffn_norm(x + self.ffn(x))

        return x


# ---------------------------------------------------------------------------
# Transformer Stack
# ---------------------------------------------------------------------------


class TransformerStack(nn.Module):
    """A stack of ``TransformerBlock`` layers.

    Args:
        dim: Model dimension.
        depth: Number of transformer layers.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        ffn_expansion: Feed-forward expansion ratio.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        cross_attention: Whether layers include cross-attention.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        ffn_expansion: float = 8 / 3,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                bias=bias,
                cross_attention=cross_attention,
            )
            for _ in range(depth)
        ])

    def forward(
        self,
        x: Tensor,
        *,
        context: Tensor | None = None,
        rope_freqs: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                rope_freqs=rope_freqs,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
        return x
