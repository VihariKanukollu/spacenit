"""Self-attention, feed-forward, and transformer-layer building blocks.

Provides the core components for constructing vision transformers:
multi-head self/cross attention, a two-layer feed-forward network,
learnable per-channel gain, stochastic depth, and a full transformer
layer that composes them together.
"""

from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.fsdp import fully_shard
from torch.jit import Final

try:
    import flash_attn
except ImportError:
    flash_attn = None

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Flash-attention dispatch
# ---------------------------------------------------------------------------


@torch._dynamo.disable()
def invoke_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cumulative_lengths: torch.Tensor | None = None,
    cumulative_lengths_q: torch.Tensor | None = None,
    cumulative_lengths_k: torch.Tensor | None = None,
    longest_sequence: int | None = None,
    longest_sequence_q: int | None = None,
    longest_sequence_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Route a query/key/value triple through the appropriate flash-attn kernel.

    When cumulative sequence-length tensors are provided the variable-length
    kernel (``flash_attn_varlen_func``) is used; otherwise the fixed-length
    kernel (``flash_attn_func``) is selected.

    Args:
        query: Query tensor – ``(B, N, H, D)`` for fixed-length or
            ``(total_tokens, H, D)`` for variable-length.
        key: Key tensor with the same layout as *query*.
        value: Value tensor with the same layout as *query*.
        cumulative_lengths: Shared cumulative lengths for both query and key
            sequences.  Overridden by the per-stream variants when given.
        cumulative_lengths_q: Cumulative lengths specific to the query stream.
        cumulative_lengths_k: Cumulative lengths specific to the key stream.
        longest_sequence: Shared maximum sequence length.  Overridden by the
            per-stream variants when given.
        longest_sequence_q: Maximum query sequence length.
        longest_sequence_k: Maximum key sequence length.
        dropout_p: Dropout probability applied inside the attention kernel.
        softmax_scale: Explicit scaling factor; ``None`` uses ``1/sqrt(D)``.
        causal: Whether to apply a causal mask.

    Returns:
        Attention output with the same spatial layout as the inputs.

    Raises:
        RuntimeError: If the ``flash-attn`` package is not installed.
    """
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")

    if cumulative_lengths is not None:
        if cumulative_lengths_q is None:
            cumulative_lengths_q = cumulative_lengths
        if cumulative_lengths_k is None:
            cumulative_lengths_k = cumulative_lengths
    if longest_sequence is not None:
        if longest_sequence_q is None:
            longest_sequence_q = longest_sequence
        if longest_sequence_k is None:
            longest_sequence_k = longest_sequence

    varlen = all(
        x is not None
        for x in (
            cumulative_lengths_q,
            cumulative_lengths_k,
            longest_sequence_q,
            longest_sequence_k,
        )
    )

    if varlen:
        assert query.ndim == 3, "query must be pre-packed for variable-length attention"
        logger.debug("using variable-length flash attention")

        return flash_attn.flash_attn_varlen_func(
            query,
            key,
            value,
            cumulative_lengths_q,
            cumulative_lengths_k,
            longest_sequence_q,
            longest_sequence_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    else:
        return flash_attn.flash_attn_func(
            query,
            key,
            value,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head self- or cross-attention with optional flash-attention support.

    Projects the input into separate query, key, and value representations,
    optionally normalises the query and key, computes (scaled) dot-product
    attention, and projects the result back to the model dimension.

    Args:
        dim: Model / input dimension.
        num_heads: Number of parallel attention heads.
        qkv_bias: If ``True``, the Q/K/V linear projections include a bias term.
        qk_norm: If ``True``, layer-normalisation is applied to Q and K after
            projection.
        attn_drop: Dropout probability inside the attention computation.
        proj_drop: Dropout probability on the final output projection.
        norm_layer: Normalisation constructor used when *qk_norm* is enabled.
        cross_attn: If ``True`` the module expects a separate context tensor
            for keys and values.
        use_flash_attn: If ``True`` the flash-attention kernel is used instead
            of the PyTorch SDPA path.
    """

    has_sdpa: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.scale = self.dim_per_head ** -0.5

        self.cross_attn = cross_attn
        self.use_flash_attn = use_flash_attn
        self.has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.query_norm = norm_layer(self.dim_per_head) if qk_norm else nn.Identity()
        self.key_norm = norm_layer(self.dim_per_head) if qk_norm else nn.Identity()
        self.attention_dropout = nn.Dropout(attn_drop)
        self.output_proj = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(proj_drop)

    def scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n: int,
        cumulative_lengths: torch.Tensor | None = None,
        cumulative_lengths_q: torch.Tensor | None = None,
        cumulative_lengths_k: torch.Tensor | None = None,
        longest_sequence: int | None = None,
        longest_sequence_q: int | None = None,
        longest_sequence_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention via the best available backend.

        Tries, in order: flash-attention kernel, PyTorch SDPA, and a manual
        fallback for older PyTorch builds.

        Args:
            q: Query tensor – ``(B, H, N, D)`` (standard) or
                ``(total_tokens, H, D)`` (flash varlen).
            k: Key tensor with matching layout.
            v: Value tensor with matching layout.
            n: Number of tokens per sample (used to expand the attention mask).
            cumulative_lengths: Shared cumulative lengths for variable-length
                flash attention.
            cumulative_lengths_q: Per-query cumulative lengths.
            cumulative_lengths_k: Per-key cumulative lengths.
            longest_sequence: Shared max sequence length.
            longest_sequence_q: Max query sequence length.
            longest_sequence_k: Max key sequence length.
            attn_mask: Optional boolean mask broadcastable over the attention
                logits.  ``True`` means *attend*.

        Returns:
            Attention output with shape ``(B, H, N, D)``.
        """
        if self.use_flash_attn:
            x = invoke_flash_attention(
                q,
                k,
                v,
                cumulative_lengths=cumulative_lengths,
                cumulative_lengths_q=cumulative_lengths_q,
                cumulative_lengths_k=cumulative_lengths_k,
                longest_sequence=longest_sequence,
                longest_sequence_q=longest_sequence_q,
                longest_sequence_k=longest_sequence_k,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            # flash-attn returns (B, Nq, H, D); transpose to (B, H, Nq, D)
            x = x.transpose(1, 2)
        elif self.has_sdpa:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, n, 1))
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout.p,
            )
        else:
            # Manual fallback for PyTorch versions without SDPA
            if attn_mask is not None:
                raise NotImplementedError(
                    "Attention masks require PyTorch >= 2.0 (scaled_dot_product_attention)"
                )
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attention_dropout(attn)
            x = attn @ v

        return x

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        cumulative_lengths: torch.Tensor | None = None,
        cumulative_lengths_q: torch.Tensor | None = None,
        cumulative_lengths_k: torch.Tensor | None = None,
        longest_sequence: int | None = None,
        longest_sequence_q: int | None = None,
        longest_sequence_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full multi-head attention computation.

        Args:
            x: Primary input – ``(B, N, C)`` or ``(B*N, C)`` when packed.
            y: Optional context tensor for cross-attention – ``(B, M, C)``.
            cumulative_lengths: Cumulative sequence lengths for variable-length
                flash attention (self-attention).
            cumulative_lengths_q: Cumulative lengths for the query stream
                (cross-attention).
            cumulative_lengths_k: Cumulative lengths for the key stream
                (cross-attention).
            longest_sequence: Maximum sequence length (self-attention).
            longest_sequence_q: Maximum query length (cross-attention).
            longest_sequence_k: Maximum key length (cross-attention).
            attn_mask: Boolean attention mask.

        Returns:
            Output tensor with the same shape as *x*.
        """
        original_shape = x.shape

        q = self.query_proj(x)

        if y is None:
            assert not self.cross_attn
            k = self.key_proj(x)
            v = self.value_proj(x)
        else:
            assert self.cross_attn
            k = self.key_proj(y)
            v = self.value_proj(y)

        if not self.use_flash_attn:
            q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
            k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        else:
            q = rearrange(q, "bn (h d) -> bn h d", h=self.num_heads)
            k = rearrange(k, "bn (h d) -> bn h d", h=self.num_heads)
            v = rearrange(v, "bn (h d) -> bn h d", h=self.num_heads)

        q, k = self.query_norm(q), self.key_norm(k)

        x = self.scaled_dot_product(
            q,
            k,
            v,
            n=original_shape[-2],
            cumulative_lengths=cumulative_lengths,
            cumulative_lengths_q=cumulative_lengths_q,
            cumulative_lengths_k=cumulative_lengths_k,
            longest_sequence=longest_sequence,
            longest_sequence_q=longest_sequence_q,
            longest_sequence_k=longest_sequence_k,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(original_shape)
        x = self.output_proj(x)
        x = self.output_dropout(x)
        return x


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network.

    Expands the input to a wider hidden dimension, applies an activation,
    then projects back.  Dropout is applied after each linear layer.

    Args:
        in_features: Dimensionality of the input.
        hidden_features: Width of the hidden layer.  Defaults to
            *in_features* when ``None``.
        out_features: Dimensionality of the output.  Defaults to
            *in_features* when ``None``.
        act_layer: Activation constructor.
        bias: Whether the linear layers include bias terms.
        drop: Dropout probability applied after each linear layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.hidden_dropout = nn.Dropout(drop)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)
        self.output_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation.

        Args:
            x: Input tensor of arbitrary batch shape with last dim equal to
                *in_features*.

        Returns:
            Tensor with the same batch shape and last dim *out_features*.
        """
        x = self.up_proj(x)
        x = self.act(x)
        x = self.hidden_dropout(x)
        x = self.down_proj(x)
        x = self.output_dropout(x)
        return x


# ---------------------------------------------------------------------------
# Learnable per-channel gain (LayerScale)
# ---------------------------------------------------------------------------


class LearnableGain(nn.Module):
    """Element-wise learnable scaling applied to residual branches.

    Initialised to a small constant so that early in training the residual
    contribution is damped, stabilising optimisation in deep networks.

    Args:
        dim: Number of channels.
        init_values: Initial value for every element of the gain vector.
        inplace: If ``True`` the multiplication is performed in-place.
    """

    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.scale_param = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale *x* element-wise by the learned gain vector.

        Args:
            x: Input tensor whose last dimension equals *dim*.

        Returns:
            Scaled tensor with the same shape.
        """
        return x.mul_(self.scale_param) if self.inplace else x * self.scale_param


# ---------------------------------------------------------------------------
# Stochastic depth (DropPath)
# ---------------------------------------------------------------------------


class StochasticDepth(nn.Module):
    """Per-sample stochastic depth (drop-path) regularisation.

    During training, entire residual branches are randomly zeroed out with
    probability *drop_rate*.  At inference time the input is returned
    unchanged.

    Args:
        drop_rate: Probability that a given sample's residual path is
            dropped during training.

    Reference:
        Huang et al., "Deep Networks with Stochastic Depth", 2016.
        https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop the residual path for a subset of batch samples.

        Args:
            x: Input tensor of shape ``(B, ...)``.

        Returns:
            Tensor of the same shape, with some samples zeroed and the
            remainder rescaled to preserve the expected value.
        """
        if self.drop_rate is None or self.drop_rate == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarise
        return x.div(keep_prob) * random_tensor


# ---------------------------------------------------------------------------
# Full transformer layer
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """Single transformer layer: norm -> attention -> norm -> feed-forward.

    Combines :class:`MultiHeadAttention`, :class:`FeedForward`,
    :class:`LearnableGain`, and :class:`StochasticDepth` into a pre-norm
    residual block.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion factor for the feed-forward hidden dimension.
        qkv_bias: Bias in Q/K/V projections.
        qk_norm: Normalise Q and K after projection.
        drop: Dropout rate for projections and feed-forward layers.
        attn_drop: Dropout rate inside the attention computation.
        drop_path: Stochastic-depth drop rate for the residual branches.
        init_values: Initial gain for :class:`LearnableGain`.  ``None``
            disables learnable scaling.
        act_layer: Activation constructor for the feed-forward block.
        norm_layer: Normalisation constructor.
        cross_attn: Enable cross-attention (expects a context tensor).
        use_flash_attn: Use the flash-attention kernel.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        self.pre_attn_norm = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            cross_attn=cross_attn,
            use_flash_attn=use_flash_attn,
        )
        self.attn_scale = (
            LearnableGain(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.pre_ffn_norm = norm_layer(dim)
        self.mlp = FeedForward(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ffn_scale = (
            LearnableGain(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        cumulative_lengths: torch.Tensor | None = None,
        cumulative_lengths_q: torch.Tensor | None = None,
        cumulative_lengths_k: torch.Tensor | None = None,
        longest_sequence: int | None = None,
        longest_sequence_q: int | None = None,
        longest_sequence_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute one transformer layer.

        Args:
            x: Input tensor – ``(B, N, C)`` or packed ``(B*N, C)``.
            y: Optional context for cross-attention – ``(B, M, C)``.
            cumulative_lengths: Cumulative sequence offsets for variable-length
                flash attention.
            cumulative_lengths_q: Query-specific cumulative offsets.
            cumulative_lengths_k: Key-specific cumulative offsets.
            longest_sequence: Maximum sequence length (self-attention).
            longest_sequence_q: Maximum query length (cross-attention).
            longest_sequence_k: Maximum key length (cross-attention).
            attn_mask: Boolean attention mask.

        Returns:
            Output tensor with the same shape as *x*.
        """
        x = x + self.drop_path(
            self.attn_scale(
                self.attn(
                    x=self.pre_attn_norm(x),
                    y=y,
                    cumulative_lengths=cumulative_lengths,
                    cumulative_lengths_q=cumulative_lengths_q,
                    cumulative_lengths_k=cumulative_lengths_k,
                    longest_sequence=longest_sequence,
                    longest_sequence_q=longest_sequence_q,
                    longest_sequence_k=longest_sequence_k,
                    attn_mask=attn_mask,
                )
            )
        )

        x = x + self.drop_path(self.ffn_scale(self.mlp(self.pre_ffn_norm(x))))
        return x

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Wrap this layer with Fully Sharded Data Parallelism."""
        fully_shard(self, **fsdp_kwargs)

    def enable_compile(self) -> None:
        """Compile this layer with ``torch.compile`` for maximum throughput."""
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)
