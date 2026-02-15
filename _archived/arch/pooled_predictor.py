"""Encoder–predictor pairs with attention-based pooling across token dimensions.

Tokens can be pooled at different stages of the model (across sensors,
time-steps, spatial patches, or combinations thereof) before the decoder
reconstructs the original token grid from the pooled representation.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from spacenit.ingestion.sensors import (
    REFERENCE_GROUND_RESOLUTION,
    SensorRegistry,
    SensorSpec,
    specs_from_labels,
)
from spacenit.structures import MaskedGeoSample, TokenVisibility
from spacenit.arch.self_attention import FeedForward
from spacenit.arch.adaptive_vision_encoder import (
    AdaptiveVisionBase,
    CompositePositionalEncodings,
    EmbeddingsAndMasks,
    LatentPredictorBase,
    MultiSensorPatchProjection,
    ProjectionAndPooling,
    VisionEncoder,
    VisionEncoderConfig,
    LatentPredictorBase,
    LatentPredictorConfig,
    collect_sensor_outputs,
    sensors_to_process,
)
from spacenit.arch.band_tokenization import TokenizationConfig
from spacenit.arch.helpers import cumulative_seq_offsets
from spacenit.settings import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PoolingDimension – which axes to collapse via attention pooling
# ---------------------------------------------------------------------------


class PoolingDimension(StrEnum):
    """Dimensions to pool over."""

    SENSOR = "sensor"  # 1
    TEMPORAL = "temporal"  # 2
    SPATIAL = "spatial"
    SENSOR_TEMPORAL = "sensor_temporal"  # 3
    ALL = "all"  # 4


# ---------------------------------------------------------------------------
# AttnPool – multi-query attention pooling with gated averaging
# ---------------------------------------------------------------------------


class AttnPool(nn.Module):
    """Multi-query attention pooling with gated averaging.

    Args:
        in_dim: Token dimension (must be divisible by 64; head_dim=64).
        attn_dim: Internal attention dimension (defaults to *in_dim*).
        hidden_dim: MLP hidden/out dimension (defaults to *in_dim* unless
            *mlp_ratio* is provided).
        mlp_ratio: If set, ``hidden_dim := int(in_dim * mlp_ratio)``.
        num_queries: Number of learned queries per pooling group.
        num_heads: Number of attention heads.
        gate_temperature: Temperature for softmax gating (>0).
        use_mlp: Whether to apply the MLP head after pooling.
    """

    def __init__(
        self,
        in_dim: int,
        attn_dim: int | None = None,
        hidden_dim: int | None = None,
        mlp_ratio: float | None = None,
        num_queries: int = 1,
        num_heads: int | None = None,
        gate_temperature: float = 1.0,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()
        assert in_dim % 64 == 0, "in_dim must be divisible by 64"
        self.attn_dim = attn_dim or in_dim
        self.num_heads: int = num_heads or self.attn_dim // 64
        self.num_queries: int = num_queries
        self.gate_temperature: float = gate_temperature
        self.use_mlp = use_mlp

        self.query_tokens: nn.Parameter = nn.Parameter(
            torch.empty(num_queries, self.attn_dim)
        )

        # Shared KV projection
        self.kv: nn.Linear = nn.Linear(in_dim, self.attn_dim * 2)

        # Output MLP (+ optional expansion via mlp_ratio)
        if mlp_ratio is not None:
            hidden_dim = int(in_dim * mlp_ratio)
        hidden_dim = hidden_dim or in_dim
        self.out_layer: FeedForward = FeedForward(self.attn_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(self.attn_dim)
        self.out_proj = (
            nn.Linear(self.attn_dim, in_dim)
            if in_dim != self.attn_dim
            else nn.Identity()
        )

        # Gating over k query outputs (maps D -> 1 per query)
        self.gate: nn.Linear | None = (
            nn.Linear(in_dim, 1, bias=False) if num_queries > 1 else None
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights."""
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)  # start near uniform mix

    def forward(
        self, feat_tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply attention pooling to the tokens."""
        Bc, N, _ = feat_tokens.shape
        H = self.num_heads
        D = self.attn_dim
        Dh = D // H

        # queries: [B*, k, D] -> [B*, H, k, Dh]
        q = (
            self.query_tokens[None, :, :]
            .expand(Bc, -1, -1)
            .reshape(Bc, self.num_queries, H, Dh)
        )
        q = rearrange(q, "b k h d -> b h k d")

        # K/V: [B*, N, D] -> [2, B*, H, N, Dh]
        feat_tokens = feat_tokens.to(self.kv.weight.dtype)
        kv = self.kv(feat_tokens).reshape(Bc, N, 2, H, Dh)
        kv = rearrange(kv, "b n two h d -> two b h n d")
        k, v = torch.unbind(kv, dim=0)  # [B*, H, N, Dh] each

        # mask -> [B*, H, k, N] (broadcastable)
        attn_mask = None
        if mask is not None:
            m = mask[:, None, None, :]  # [B*,1,1,N]
            attn_mask = m.expand(Bc, H, self.num_queries, N)

        # H100 chunking on batch axis
        max_size = 63488
        x_chunks = []
        for i in range(0, Bc, max_size):
            q_chunk = q[i : i + max_size, ...]
            k_chunk = k[i : i + max_size, ...]
            v_chunk = v[i : i + max_size, ...]
            m_chunk = (
                attn_mask[i : i + max_size, ...] if attn_mask is not None else None
            )
            # SDPA expects [B,H,Q,D] x [B,H,K,D] -> [B,H,Q,D]
            x_chunk = F.scaled_dot_product_attention(
                q_chunk, k_chunk, v_chunk, attn_mask=m_chunk
            )
            x_chunks.append(x_chunk)

        # [B*, H, k, Dh] -> [B*, k, D]
        x = torch.cat(x_chunks, dim=0)
        o = rearrange(x, "b h k d -> b k (h d)")

        # Gated average across k, or pass-through if k=1
        if self.num_queries > 1 and self.gate is not None:
            o_for_gate = F.layer_norm(o, (D,))  # normalize only for gating
            logits = self.gate(o_for_gate).squeeze(-1)  # [B*, k]
            w = torch.softmax(logits, dim=1)
            z = (w.unsqueeze(-1) * o).sum(dim=1)  # mix the *unnormalized* values
        else:
            z = o.squeeze(1)

        # MLP + LN head
        if self.use_mlp:
            z = self.out_norm(self.out_layer(z))
        return self.out_proj(z)


# ---------------------------------------------------------------------------
# PooledSensorEncoder – encoder that pools tokens before later layers
# ---------------------------------------------------------------------------


class PooledSensorEncoder(VisionEncoder):
    """Encoder that pools tokens across configurable dimensions.

    Extends :class:`VisionEncoder` by inserting an attention-pooling step
    between two groups of transformer layers.  The first
    ``num_pre_sensor_pooling_layers`` layers operate on the full (unpooled)
    token set; the remaining layers attend over the pooled representation.
    """

    def __init__(
        self,
        dims_to_pool: str,
        pooling_attn_dim: int | None = None,
        attn_pool_mlp_ratio: float | None = None,
        num_queries: int = 1,
        use_mlp: bool = False,
        num_pre_sensor_pooling_layers: int = 0,
        num_attn_pool_heads: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the PooledSensorEncoder."""
        super().__init__(*args, **kwargs)
        self.attn_pool = AttnPool(
            in_dim=self.embed_dim,
            attn_dim=pooling_attn_dim,
            mlp_ratio=attn_pool_mlp_ratio,
            num_queries=num_queries,
            use_mlp=use_mlp,
            num_heads=num_attn_pool_heads,
        )
        self.num_pre_sensor_pooling_layers = num_pre_sensor_pooling_layers

        self.dims_to_pool = dims_to_pool
        if self.use_flash_attn:
            raise NotImplementedError("Flash attn not implemented for PooledSensorEncoder")

    # -----------------------------------------------------------------
    # Reduction / expansion helpers
    # -----------------------------------------------------------------

    def _get_reduce_and_expand_args(
        self, shape: tuple[int, ...]
    ) -> tuple[str, str, str, str, str, str, dict[str, int], dict[str, int]]:
        """Get the reduction and expansion arguments for the dimensions to pool."""
        B, H, W, T, M, D = shape
        if self.dims_to_pool == PoolingDimension.SENSOR:
            reduction_args = "(b h w t) m d"
            reduction_mask_args = "(b h w t) m"
            pre_expand_args = "(b h w t) d"
            expand_args = "b h w t d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W, "t": T}
            expand_kwargs = {"b": B, "h": H, "w": W, "t": T, "d": D}
        elif self.dims_to_pool == PoolingDimension.TEMPORAL:
            reduction_args = "(b h w m) t d"
            reduction_mask_args = "(b h w m) t"
            pre_expand_args = "(b h w m) d"
            expand_args = "b h w m d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W, "m": M}
            expand_kwargs = {"b": B, "h": H, "w": W, "m": M, "d": D}
        elif self.dims_to_pool == PoolingDimension.SPATIAL:
            reduction_args = "(b t m) (h w) d"
            reduction_mask_args = "(b t m) (h w)"
            pre_expand_args = "(b t m) d"
            expand_args = "b t m d"
            expand_mask_kwargs = {"b": B, "t": T, "m": M}
            expand_kwargs = {"b": B, "t": T, "m": M, "d": D}
        elif self.dims_to_pool == PoolingDimension.SENSOR_TEMPORAL:
            reduction_args = "(b h w ) (t m) d"
            reduction_mask_args = "(b h w ) (t m)"
            pre_expand_args = "(b h w) d"
            expand_args = "b h w d"
            expand_mask_kwargs = {"b": B, "h": H, "w": W}
            expand_kwargs = {"b": B, "h": H, "w": W, "d": D}
        elif self.dims_to_pool == PoolingDimension.ALL:
            reduction_args = "b (h w t m)  d"
            reduction_mask_args = "b (h w t m)"
            pre_expand_args = "(b n) d"
            expand_args = "b n d"
            expand_mask_kwargs = {"b": B, "n": 1}
            expand_kwargs = {"b": B, "n": 1, "d": D}
        else:
            raise ValueError(f"Invalid dimensions to pool: {self.dims_to_pool}")
        pre_expand_mask_args = pre_expand_args.replace(" d", "")
        expand_mask_args = expand_args.replace(" d", "")
        return (
            reduction_args,
            reduction_mask_args,
            pre_expand_args,
            pre_expand_mask_args,
            expand_args,
            expand_mask_args,
            expand_mask_kwargs,
            expand_kwargs,
        )

    def apply_attn_pooling(
        self, spatial_tokens: torch.Tensor, spatial_masks: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Attentive pool the tokens across the dimensions specified in self.dims_to_pool."""
        (
            reduction_args,
            reduction_mask_args,
            pre_expand_args,
            pre_expand_mask_args,
            expand_args,
            expand_mask_args,
            expand_mask_kwargs,
            expand_kwargs,
        ) = self._get_reduce_and_expand_args(spatial_tokens.shape)
        # Collapse the chosen dimensions
        spatial_tokens = rearrange(spatial_tokens, f"b h w t m d -> {reduction_args}")
        spatial_masks = rearrange(spatial_masks, f"b h w t m -> {reduction_mask_args}")

        pooled_attn_mask = spatial_masks == TokenVisibility.VISIBLE_ENCODER.value
        pooled_tokens = self.attn_pool(spatial_tokens, pooled_attn_mask)
        logger.info(f"shape of pooled tokens: {pooled_tokens.shape}")
        pooled_tokens = rearrange(
            pooled_tokens, f"{pre_expand_args} -> {expand_args}", **expand_kwargs
        )
        # If any token in the reduction dimension is visible, mark the pooled token as visible
        online_encoder_only_mask = (
            spatial_masks == TokenVisibility.VISIBLE_ENCODER.value
        ).any(dim=-1)
        pooled_attn_mask = torch.where(
            online_encoder_only_mask,
            TokenVisibility.VISIBLE_ENCODER.value,
            TokenVisibility.ABSENT.value,
        )

        pooled_attn_mask = rearrange(
            pooled_attn_mask,
            f"{pre_expand_mask_args} -> {expand_mask_args}",
            **expand_mask_kwargs,
        )
        pooled_tokens_and_masks = {
            "sensor_pooled_tokens": pooled_tokens,
            "sensor_pooled_masks": pooled_attn_mask,
        }
        return pooled_tokens_and_masks

    def collapse_and_combine_hwtc_pooled_tokens(
        self, x: dict[str, Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse and combine the pooled tokens and masks."""
        pooled_tokens = x["sensor_pooled_tokens"]
        pooled_masks = x["sensor_pooled_masks"]
        pooled_tokens = rearrange(pooled_tokens, "b ... d -> b (...) d")
        pooled_masks = rearrange(pooled_masks, "b ...  -> b (...) ")
        return pooled_tokens, pooled_masks

    def reshape_pooled_tokens(
        self, pooled_tokens: torch.Tensor, pooled_dims: tuple[int, ...]
    ) -> torch.Tensor:
        """Reshape the pooled tokens to the dimensions specified in pooled_dims."""
        b = pooled_tokens.shape[0]
        middle_dims = pooled_dims[1:-1]
        d = pooled_tokens.shape[-1]
        tokens_reshaped = pooled_tokens.view(b, *middle_dims, d)
        return tokens_reshaped

    def stack_spatial_sensors_and_masks(
        self,
        tokens_dict: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Stack the spatial sensor tokens together along the sensor dimension."""
        available_sensors = collect_sensor_outputs(tokens_dict)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        mask_list = []
        data_list = []
        for sensor in sensors_list:
            # Only include spatial + multitemporal sensors
            sensor_spec = SensorRegistry.get(sensor)
            if sensor_spec.varies_in_space and sensor_spec.has_temporal_axis:
                logger.info(f"sensor: {sensor} is spatial and multitemporal")
                masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
                logger.info(
                    f"sensor: {sensor}, masked_sensor_name: {masked_sensor_name}"
                )
                data = tokens_dict[sensor]
                mask = tokens_dict[masked_sensor_name]
                data_list.append(data)
                mask_list.append(mask)
        # Stack in the sensor dimension
        return torch.cat(data_list, dim=4), torch.cat(mask_list, dim=4)

    @staticmethod
    def remove_masked_tokens(
        x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks.

        On Input:
        0 means this token should be removed
        1 means this token should be kept

        Args:
            x: Tokens to remove masked tokens from
            mask: Mask to remove masked tokens from

        Returns:
            tokens: [B, T, D]
            indices: [B, T]
            updated_mask: [B, T]
            seqlens: [B]
            longest_sequence: [1]
            where T is the max number of unmasked tokens for an instance
        """
        logger.info(f"remove masked tokens shape of x: {x.shape}")
        logger.info(f"remove masked tokens shape of mask: {mask.shape}")
        sorted_mask, indices = torch.sort(mask, dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))

        # Set masked values to 0
        x = x * sorted_mask.unsqueeze(-1)

        # Cut off to the length of the longest sequence
        seq_lengths = sorted_mask.sum(-1)
        longest_sequence = seq_lengths.max()
        x = x[:, :longest_sequence]
        updated_mask = sorted_mask[:, :longest_sequence]

        return x, indices, updated_mask, seq_lengths, longest_sequence

    def apply_unpooled_attn(
        self,
        tokens_and_masks_dict: dict[str, Tensor],
        sensors_to_dims_dict: dict,
        exit_ids_seq: Tensor | None = None,
        exited_tokens: Tensor | None = None,
        fast_pass: bool = False,
    ) -> tuple[dict[str, Tensor], Tensor | None]:
        """Apply attention to the unpooled tokens (pre-pooling layers)."""
        tokens, mask = self.collapse_and_combine_hwtc(tokens_and_masks_dict)

        bool_mask = mask == TokenVisibility.VISIBLE_ENCODER.value

        tokens, indices, new_mask, seq_lengths, longest_sequence = (
            self.remove_masked_tokens(tokens, bool_mask)
        )
        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )
        cu_seqlens = cumulative_seq_offsets(seq_lengths)
        # Pack x tokens
        if self.use_flash_attn:
            og_shape = tokens.shape
            tokens = self.pack_tokens(tokens, new_mask)

        attn_mask = self._maybe_get_attn_mask(new_mask, fast_pass)

        # Add register tokens before attention layers
        register_tokens = None
        if self.has_register_tokens:
            tokens, attn_mask = self.add_register_tokens_and_masks(tokens, attn_mask)

        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            if i_blk == self.num_pre_sensor_pooling_layers:
                break
            logger.debug(f"i_blk pre-sensor pooling: {i_blk}")
            if (exit_ids_seq is not None) and (i_blk > 0):
                assert exited_tokens is not None
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )
            tokens = blk(
                x=tokens,
                cumulative_lengths=cu_seqlens,
                longest_sequence=longest_sequence,
                attn_mask=attn_mask,
            )

        # Remove register tokens after attention layers
        if self.has_register_tokens:
            tokens, register_tokens = self.pop_register_tokens(tokens)

        if self.use_flash_attn:
            tokens = self.unpack_tokens(tokens, new_mask, og_shape)

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),
                input=tokens,
                other=exited_tokens,
            )
        tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
        tokens_per_sensor_dict = self.split_and_expand_per_sensor(
            tokens, sensors_to_dims_dict
        )
        return tokens_per_sensor_dict, register_tokens

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        fast_pass: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, pre_pooled_sensor_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        # Exited tokens are just the linear projection
        exited_tokens, _ = self.collapse_and_combine_hwtc(x)

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            tile_patch_size,
            input_res,
        )
        tokens_dict.update(original_masks_dict)

        # Apply pre-pooling attention
        tokens_dict, register_tokens = self.apply_unpooled_attn(
            tokens_dict,
            pre_pooled_sensor_to_dims_dict,
            exit_ids_seq,
            exited_tokens,
            fast_pass,
        )
        # Update the tokens_dict with the original masks
        tokens_dict.update(original_masks_dict)
        logger.info(f"tokens_dict keys: {tokens_dict.keys()}")

        # Stack spatial sensors and apply attention pooling
        spatial_tokens, spatial_masks = self.stack_spatial_sensors_and_masks(
            tokens_dict
        )

        tokens_dict = self.apply_attn_pooling(spatial_tokens, spatial_masks)
        pooled_dims = tokens_dict["sensor_pooled_tokens"].shape
        original_pooled_masks = tokens_dict["sensor_pooled_masks"]
        tokens, mask = self.collapse_and_combine_hwtc_pooled_tokens(tokens_dict)
        bool_mask = mask == TokenVisibility.VISIBLE_ENCODER.value

        tokens, indices, new_mask, seq_lengths, longest_sequence = (
            self.remove_masked_tokens(tokens, bool_mask)
        )
        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )
        cu_seqlens = cumulative_seq_offsets(seq_lengths)

        attn_mask = self._maybe_get_attn_mask(new_mask, fast_pass)

        # Add register tokens before post-sensor pooling attention layers
        if self.has_register_tokens and register_tokens is not None:
            tokens, attn_mask = self.add_register_tokens_and_masks(
                tokens, attn_mask, register_tokens
            )

        # Apply attn with varying encoder depths (post-pooling)
        for i_blk, blk in enumerate(self.blocks):
            if i_blk < self.num_pre_sensor_pooling_layers:
                continue
            logger.debug(f"i_blk post-sensor pooling: {i_blk}")
            if (exit_ids_seq is not None) and (i_blk > 0):
                assert exited_tokens is not None
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )
            tokens = blk(
                x=tokens,
                cumulative_lengths=cu_seqlens,
                longest_sequence=longest_sequence,
                attn_mask=attn_mask,
            )

        token_norm_stats = None
        if self.has_register_tokens and register_tokens is not None:
            tokens, register_tokens = self.pop_register_tokens(tokens)
            token_norm_stats = self.get_token_norm_stats(tokens, register_tokens)

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),
                input=tokens,
                other=exited_tokens,
            )
        # Apply norm before adding removed tokens
        tokens = self.norm(tokens)
        tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
        out_dict = {}
        out_dict["sensor_pooled_tokens"] = self.reshape_pooled_tokens(
            tokens, pooled_dims
        )
        out_dict["sensor_pooled_masks"] = original_pooled_masks
        return out_dict, token_norm_stats

    def forward(
        self,
        x: MaskedGeoSample,
        tile_patch_size: int,
        input_res: int = REFERENCE_GROUND_RESOLUTION,
        token_exit_cfg: dict | None = None,
        fast_pass: bool = False,
    ) -> dict[str, Any]:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            tile_patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            token_exit_cfg: Configuration for token exit
            fast_pass: Whether to always pass None as the mask to the
                transformer, enabling torch-based flash attention

        Returns:
            Dictionary containing the encoded representations and their masks
        """
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, tile_patch_size)
        tokenized_output = EmbeddingsAndMasks(**patchified_tokens_and_masks)
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            pooled_tokens_and_masks, token_norm_stats = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                tile_patch_size=tile_patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                fast_pass=fast_pass,
            )
        else:
            pooled_tokens_and_masks = {}
            token_norm_stats = None

        output_dict: dict[str, Any] = {
            "embeddings_and_masks": tokenized_output,
        }
        if pooled_tokens_and_masks:
            output_dict["project_aggregated"] = self.project_and_aggregate(
                pooled_tokens_and_masks["sensor_pooled_tokens"]
            )
            output_dict["pooled_tokens_and_masks"] = pooled_tokens_and_masks
        else:
            output_dict["project_aggregated"] = self.project_and_aggregate(
                tokenized_output
            )
        if token_norm_stats is not None:
            output_dict["token_norm_stats"] = token_norm_stats

        return output_dict


# ---------------------------------------------------------------------------
# PooledSensorEncoderConfig
# ---------------------------------------------------------------------------


@dataclass
class PooledSensorEncoderConfig(VisionEncoderConfig):
    """Configuration for the PooledSensorEncoder."""

    dims_to_pool: PoolingDimension = PoolingDimension.SENSOR
    num_queries: int = 1
    attn_pool_mlp_ratio: float | None = None
    num_pre_sensor_pooling_layers: int = 0
    use_mlp: bool = False
    num_attn_pool_heads: int | None = None
    pooling_attn_dim: int | None = None

    def build(self) -> "PooledSensorEncoder":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_sensor_labels is replaced by supported_sensors
        kwargs.pop("supported_sensor_labels")
        kwargs["supported_sensors"] = self.supported_sensors
        logger.info(f"PooledSensorEncoder kwargs: {kwargs}")
        return PooledSensorEncoder(**kwargs)


# ---------------------------------------------------------------------------
# PooledSensorPredictor
# ---------------------------------------------------------------------------


class PooledSensorPredictor(LatentPredictorBase):
    """Predictor that decodes from pooled sensor tokens via cross-attention."""

    def __init__(
        self,
        include_encoder_encodings: bool = True,
        dims_to_pool: str = "sensor",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the PooledSensorPredictor."""
        super().__init__(*args, **kwargs)
        self.include_encoder_encodings = include_encoder_encodings
        self.dims_to_pool = dims_to_pool
        if self.use_flash_attn:
            raise NotImplementedError("Flash attn not implemented for PooledSensorPredictor")

    def _which_encodings_to_use(self) -> dict[str, bool]:
        """Determine which positional encodings to apply to pooled tokens."""
        if self.dims_to_pool == PoolingDimension.SENSOR:
            return {"use_sensor_encodings": False, "use_temporal_encodings": True}
        elif self.dims_to_pool == PoolingDimension.TEMPORAL:
            return {"use_sensor_encodings": True, "use_temporal_encodings": False}
        elif self.dims_to_pool == PoolingDimension.SENSOR_TEMPORAL:
            return {"use_sensor_encodings": False, "use_temporal_encodings": False}
        else:
            raise NotImplementedError(
                f"Dims to pool {self.dims_to_pool} not implemented"
            )

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int,
        pooled_tokens_and_masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        logger.warning("Calling apply_attn for PooledSensorPredictor")
        tokens_only_dict, original_masks_dict, sensors_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, tile_patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)

        pooled_tokens = pooled_tokens_and_masks["sensor_pooled_tokens"]
        if self.include_encoder_encodings:
            encoding_kwargs = self._which_encodings_to_use()
            logger.info(f"encoding_kwargs: {encoding_kwargs}")
            pooled_tokens = self.composite_encodings._apply_encodings_per_sensor(
                SensorRegistry.get("sentinel2_l2a").label,
                pooled_tokens,
                timestamps,
                tile_patch_size,
                input_res,
                **encoding_kwargs,
            )
        pooled_tokens = rearrange(pooled_tokens, "b ... d -> b (...) d")
        pooled_attn_mask = rearrange(
            pooled_tokens_and_masks["sensor_pooled_masks"], "b ... -> b (...)"
        )

        (
            _,
            pooled_tokens,
            _,
            pooled_attn_mask,
            _,
            _,
            _,
            _,
            _,
        ) = self.split_x_y(pooled_tokens, pooled_attn_mask)

        all_tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)
        # X contains the tokens to decode, Y contains the tokens to attend to for context
        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            longest_decoded_sequence,
            longest_unmasked_sequence,
        ) = self.split_x_y(all_tokens, mask)
        # Pack x tokens
        if self.use_flash_attn:
            raise NotImplementedError("Flash attn not implemented")

        for blk in self.blocks:
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=pooled_tokens,
                attn_mask=(
                    pooled_attn_mask.bool() if not self.use_flash_attn else None
                ),
            )

        x = self.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        tokens_per_sensor_dict = self.split_and_expand_per_sensor(
            x, sensors_to_dims_dict
        )
        tokens_per_sensor_dict.update(original_masks_dict)
        return tokens_per_sensor_dict

    def forward(
        self,
        x: EmbeddingsAndMasks,
        pooled_tokens_and_masks: dict[str, Tensor],
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int = REFERENCE_GROUND_RESOLUTION,
    ) -> EmbeddingsAndMasks:
        """Generate predictions from encoded token representations.

        Args:
            x: EmbeddingsAndMasks containing the masks to use to make decodings.
                These tokens are discarded – only the masks are used.
            pooled_tokens_and_masks: Dictionary containing the pooled tokens
                and their masks.
            timestamps: Timestamps of the tokens.
            tile_patch_size: Patch size of the tokens.
            input_res: Input resolution of the tokens.

        Returns:
            EmbeddingsAndMasks containing the predicted tokens and their masks.
        """
        # Apply Input Norms and encoder to decoder embeds to each sensor
        available_sensors = x.sensors
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )

        # Apply input norm and projection on pooled tokens
        pooled_tokens = pooled_tokens_and_masks["sensor_pooled_tokens"]
        pooled_tokens = self.input_norm(pooled_tokens)
        pooled_tokens = self.encoder_to_decoder_embed(pooled_tokens)
        pooled_tokens_and_masks["sensor_pooled_tokens"] = pooled_tokens

        # Prepare the Learnable Masked Outputs on the original Unpooled Tokens
        decoder_embedded_dict = x.as_dict(return_none=False)
        tokens_only_dict = self.add_masks(decoder_embedded_dict)
        decoder_embedded_dict.update(tokens_only_dict)
        tokens_and_masks = self.apply_attn(
            decoder_embedded_dict,
            timestamps,
            tile_patch_size,
            input_res,
            pooled_tokens_and_masks=pooled_tokens_and_masks,
        )

        # Project and Normalize Output Tokens
        output_dict = {}
        available_sensors = collect_sensor_outputs(tokens_and_masks)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
            sensor_mask = tokens_and_masks[masked_sensor_name]
            per_sensor_output_tokens = []
            sensor_data = tokens_and_masks[sensor]

            num_band_sets = self.tokenization_config.group_count_for(sensor)
            for idx in range(num_band_sets):
                per_channel_sensor_data = sensor_data[..., idx, :]
                output_data = self.to_output_embed(self.norm(per_channel_sensor_data))
                per_sensor_output_tokens.append(output_data)
            output_dict[sensor] = torch.stack(per_sensor_output_tokens, dim=-2)
            output_dict[masked_sensor_name] = sensor_mask
        return EmbeddingsAndMasks(**output_dict)


# ---------------------------------------------------------------------------
# PooledSensorPredictorConfig
# ---------------------------------------------------------------------------


@dataclass
class PooledSensorPredictorConfig(LatentPredictorConfig):
    """Configuration for the PooledSensorPredictor."""

    include_encoder_encodings: bool = True
    dims_to_pool: PoolingDimension = PoolingDimension.SENSOR

    def build(self) -> "PooledSensorPredictor":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_sensor_labels is replaced by supported_sensors
        kwargs.pop("supported_sensor_labels")
        kwargs["supported_sensors"] = self.supported_sensors
        logger.info(f"PooledSensorPredictor kwargs: {kwargs}")
        return PooledSensorPredictor(**kwargs)
