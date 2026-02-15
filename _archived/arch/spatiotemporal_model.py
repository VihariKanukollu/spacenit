"""Spatiotemporal encoder–predictor that applies spatial and temporal attention separately.

The :class:`SpatioTemporalEncoder` alternates between spatial-only and
temporal-only attention layers (or uses windowed / full attention when
configured), producing per-token representations that are aware of both
spatial context and temporal dynamics.  :class:`SpatioTemporalPredictor`
decodes masked tokens from these representations via cross-attention.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from spacenit.settings import Config
from spacenit.ingestion.sensors import (
    REFERENCE_GROUND_RESOLUTION,
    SensorRegistry,
    SensorSpec,
    specs_from_labels,
)
from spacenit.structures import MaskedGeoSample, TokenVisibility
from spacenit.arch.self_attention import TransformerLayer
from spacenit.arch.adaptive_vision_encoder import (
    CompositePositionalEncodings,
    EmbeddingsAndMasks,
    MultiSensorPatchProjection,
    ProjectionAndPooling,
    collect_sensor_outputs,
    sensors_to_process,
)
from spacenit.arch.band_tokenization import TokenizationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TemporalSpatialMode – how attention is applied at each layer
# ---------------------------------------------------------------------------


class TemporalSpatialMode(Enum):
    """Mode to perform attention."""

    FULL = 0
    SPATIAL = 1
    TEMPORAL = 2
    WINDOWED = 3


# ---------------------------------------------------------------------------
# SpatioTemporalBase – shared base class
# ---------------------------------------------------------------------------


class SpatioTemporalBase(nn.Module):
    """Base class for spatiotemporal encoder/predictor models."""

    cross_attn: bool = False

    def __init__(
        self,
        embed_dim: int,
        max_sequence_length: int,
        head_count: int,
        ffn_expansion: float,
        depth: int,
        stochastic_depth_rate: float,
        supported_sensors: list[SensorSpec],
        windowed_attention_size: int | None = None,
        learned_channel_embed: bool = True,
        random_channel_embed: bool = False,
        last_layer_cross_attn: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ) -> None:
        """Initialize the SpatioTemporalBase class."""
        super().__init__()

        self.embed_dim = embed_dim
        self.supported_sensors = supported_sensors
        self.supported_sensor_labels = [x.label for x in supported_sensors]
        logger.info(f"sensors being used by model: {self.supported_sensor_labels}")

        self.max_sequence_length = max_sequence_length
        self.windowed_attention_size = windowed_attention_size
        self.learned_channel_embed = learned_channel_embed
        self.random_channel_embed = random_channel_embed
        self._base_tokenization_config = tokenization_config or TokenizationConfig()

        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim,
                    head_count,
                    ffn_expansion,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    cross_attn=self.cross_attn
                    or (last_layer_cross_attn and idx == depth - 1),
                    drop_path=stochastic_depth_rate,
                )
                for idx in range(depth)
            ]
        )

        self.composite_encodings = CompositePositionalEncodings(
            embed_dim,
            self.supported_sensors,
            max_sequence_length,
            learned_channel_embed,
            random_channel_embed,
            tokenization_config=self._base_tokenization_config,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def grab_sensor_specific_dims(sensor_data: Tensor) -> tuple[int, ...]:
        """Grab the sensor specific dimensions from the sensor data.

        Assumes [B, ..., C, D]

        Every sensor will have a batch dimension, a channel dimension and embedding dimension.

        Args:
            sensor_data: Sensor data

        Returns:
            Sensor specific dimensions
        """
        return sensor_data.shape[1:-2] if sensor_data.ndim > 3 else ()

    # -----------------------------------------------------------------
    # Collapse-and-combine dispatch
    # -----------------------------------------------------------------

    def collapse_and_combine(
        self, x: dict[str, Tensor], mode: TemporalSpatialMode, block_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks into two tensors.

        Args:
            x: the dictionary with tokens and masks.
            mode: what kind of attention to apply.
            block_idx: the block index, used by some attention modes.
        """
        if mode == TemporalSpatialMode.FULL:
            return self.collapse_and_combine_full(x)
        elif mode == TemporalSpatialMode.SPATIAL:
            return self.collapse_and_combine_spatial(x)
        elif mode == TemporalSpatialMode.TEMPORAL:
            return self.collapse_and_combine_temporal(x)
        elif mode == TemporalSpatialMode.WINDOWED:
            return self.collapse_and_combine_windowed(x, block_idx)
        # Should not be possible.
        assert False

    def collapse_and_combine_full(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks into two tensors.

        This is for attention across all tokens in each example.
        """
        tokens, masks = [], []
        available_sensors = collect_sensor_outputs(x)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
            x_sensor = x[sensor]
            x_sensor_mask = x[masked_sensor_name]
            tokens.append(rearrange(x_sensor, "b ... d -> b (...) d"))
            masks.append(rearrange(x_sensor_mask, "b ... -> b (...)"))
        tokens_tensor = torch.cat(tokens, dim=1)
        masks_tensor = torch.cat(masks, dim=1)

        return tokens_tensor, masks_tensor

    def collapse_and_combine_temporal(
        self, x: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks for temporal-only attention.

        Combines the batch/height/width dimensions so that attention is applied
        temporally but not spatially.
        """
        available_sensors = collect_sensor_outputs(x)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )

        # Determine spatial dimensions from spatially-varying sensors.
        h: int | None = None
        w: int | None = None
        for sensor in sensors_list:
            x_sensor = x[sensor]
            if len(x_sensor.shape) != 6:
                continue
            cur_h = x_sensor.shape[1]
            cur_w = x_sensor.shape[2]
            if h is None:
                h = cur_h
                w = cur_w
            elif h != cur_h or w != cur_w:
                raise ValueError(
                    "expected all sensors to have the same spatial dimensions"
                )

        if h is None or w is None:
            raise ValueError("expected at least one spatial sensor")

        tokens, masks = [], []
        for sensor in sensors_list:
            masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
            x_sensor = x[sensor]
            x_sensor_mask = x[masked_sensor_name]

            if len(x_sensor.shape) == 3:
                # Static in space/time – duplicate across spatial positions.
                flattened_tokens = repeat(
                    x_sensor, "b b_s d -> (b h w) b_s d", h=h, w=w
                )
                flattened_masks = repeat(
                    x_sensor_mask, "b b_s -> (b h w) b_s", h=h, w=w
                )

            elif len(x_sensor.shape) == 6:
                flattened_tokens = rearrange(
                    x_sensor, "b h w ... d -> (b h w) (...) d"
                )
                flattened_masks = rearrange(
                    x_sensor_mask, "b h w ... -> (b h w) (...)"
                )

            else:
                raise NotImplementedError(
                    f"not implemented for {len(x_sensor.shape)} dimensions"
                )

            num_tokens = flattened_tokens.shape[1]
            logger.debug(f"Sensor {sensor} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along temporal (token) dimension.
        tokens_tensor = torch.cat(tokens, dim=1)
        masks_tensor = torch.cat(masks, dim=1)
        return tokens_tensor, masks_tensor

    def collapse_and_combine_spatial(
        self, x: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks for spatial-only attention.

        Combines the batch/time dimensions so that attention is applied spatially
        but not temporally.
        """
        available_sensors = collect_sensor_outputs(x)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )

        # Determine spatial dimensions.
        h: int | None = None
        w: int | None = None
        for sensor in sensors_list:
            x_sensor = x[sensor]
            if len(x_sensor.shape) != 6:
                continue
            cur_h = x_sensor.shape[1]
            cur_w = x_sensor.shape[2]
            if h is None:
                h = cur_h
                w = cur_w
            elif h != cur_h or w != cur_w:
                raise ValueError(
                    "expected all sensors to have the same spatial dimensions"
                )

        if h is None or w is None:
            raise ValueError("expected at least one spatial sensor")

        tokens, masks = [], []
        for sensor in sensors_list:
            masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
            x_sensor = x[sensor]
            x_sensor_mask = x[masked_sensor_name]

            if len(x_sensor.shape) == 3:
                # Static in space/time – pad to h*w.
                b_s = x_sensor.shape[1]
                amount_to_pad = h * w - b_s
                flattened_tokens = torch.nn.functional.pad(
                    x_sensor, (0, 0, 0, amount_to_pad)
                )
                flattened_masks = torch.nn.functional.pad(
                    x_sensor_mask, (0, amount_to_pad), value=TokenVisibility.ABSENT.value
                )

            elif len(x_sensor.shape) == 6:
                flattened_tokens = rearrange(
                    x_sensor, "b h w ... d -> (b ...) (h w) d"
                )
                flattened_masks = rearrange(
                    x_sensor_mask, "b h w ... -> (b ...) (h w)"
                )

            else:
                raise NotImplementedError(
                    f"not implemented for {len(x_sensor.shape)} dimensions"
                )

            num_tokens = flattened_tokens.shape[0]
            logger.debug(f"Sensor {sensor} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along temporal (batch) dimension.
        tokens_tensor = torch.cat(tokens, dim=0)
        masks_tensor = torch.cat(masks, dim=0)
        return tokens_tensor, masks_tensor

    def collapse_and_combine_windowed(
        self, x: dict[str, Tensor], block_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks for windowed attention.

        Applies attention that is full along the temporal dimension but windowed
        along the spatial dimension. Even blocks are shifted by half the window
        size.

        Args:
            x: the tokens and masks dictionary.
            block_idx: the index of this block in the transformer.

        Returns:
            the (tokens, masks) tuple.
        """
        size = self.windowed_attention_size
        assert size is not None
        if block_idx % 2 == 0:
            offset_padding = size // 2
        else:
            offset_padding = 0

        available_sensors = collect_sensor_outputs(x)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )

        tokens, masks = [], []
        for sensor in sensors_list:
            masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
            x_sensor = x[sensor]
            x_sensor_mask = x[masked_sensor_name]

            if len(x_sensor.shape) == 6:
                # Collapse the temporal and band set dimensions.
                cur_tokens = rearrange(x_sensor, "b h w ... d -> b (...) d h w")
                cur_masks = rearrange(x_sensor_mask, "b h w ... -> b (...) h w")
                # Add the offset padding.
                cur_tokens = torch.nn.functional.pad(
                    cur_tokens, (offset_padding, 0, offset_padding, 0)
                )
                cur_masks = torch.nn.functional.pad(
                    cur_masks,
                    (offset_padding, 0, offset_padding, 0),
                    value=TokenVisibility.ABSENT.value,
                )
                # Pad to multiple of window size.
                w_padding = (-cur_tokens.shape[-1]) % size
                h_padding = (-cur_tokens.shape[-2]) % size
                cur_tokens = torch.nn.functional.pad(
                    cur_tokens, (0, w_padding, 0, h_padding)
                )
                cur_masks = torch.nn.functional.pad(
                    cur_masks,
                    (0, w_padding, 0, h_padding),
                    value=TokenVisibility.ABSENT.value,
                )
                # Split into windows.
                flattened_tokens = rearrange(
                    cur_tokens,
                    "b tbs d (hn hs) (wn ws) -> (b hn wn) (tbs hs ws) d",
                    hs=size,
                    ws=size,
                )
                flattened_masks = rearrange(
                    cur_masks,
                    "b tbs (hn hs) (wn ws) -> (b hn wn) (tbs hs ws)",
                    hs=size,
                    ws=size,
                )

            else:
                raise NotImplementedError(
                    f"not implemented for {len(x_sensor.shape)} dimensions"
                )

            num_tokens = flattened_tokens.shape[0]
            logger.debug(f"Sensor {sensor} has {num_tokens} tokens")
            tokens.append(flattened_tokens)
            masks.append(flattened_masks)

        # Concatenate along the token dimension.
        tokens_tensor = torch.cat(tokens, dim=1)
        masks_tensor = torch.cat(masks, dim=1)
        logger.info(
            f"collapse_and_combine_windowed: end up with {tokens_tensor.shape[0]} "
            f"batches of {tokens_tensor.shape[1]} tokens"
        )
        return tokens_tensor, masks_tensor

    @staticmethod
    def _construct_einops_pattern(
        spatial_dims: tuple[int, ...],
    ) -> tuple[str, dict[str, int]]:
        """Build an einops rearrange pattern for the given spatial dimensions."""
        dim_dict = {f"dim{i}": size for i, size in enumerate(spatial_dims)}
        pattern_input = (
            "d -> " + " ".join(f"(dim{i})" for i in range(len(spatial_dims))) + " d"
        )
        return pattern_input, dim_dict

    def split_tokens_masks_and_dims(
        self, x: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, tuple]]:
        """Split the tokens, masks, and dimensions out into separate dicts."""
        tokens_only_dict = {}
        original_masks_dict = {}
        sensors_to_dims_dict = {}
        available_sensors = collect_sensor_outputs(x)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            x_sensor = x[sensor]
            tokens_only_dict[sensor] = x_sensor
            sensors_to_dims_dict[sensor] = x_sensor.shape
            masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
            original_masks_dict[masked_sensor_name] = x[masked_sensor_name]
        return tokens_only_dict, original_masks_dict, sensors_to_dims_dict

    # -----------------------------------------------------------------
    # Split-and-expand dispatch
    # -----------------------------------------------------------------

    def split_and_expand_per_sensor(
        self,
        x: dict[str, Tensor],
        sensors_to_dims_dict: dict[str, tuple],
        mode: TemporalSpatialMode,
        block_idx: int,
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per sensor."""
        if mode == TemporalSpatialMode.FULL:
            return self.split_and_expand_per_sensor_full(x, sensors_to_dims_dict)
        elif mode == TemporalSpatialMode.SPATIAL:
            return self.split_and_expand_per_sensor_spatial(
                x, sensors_to_dims_dict
            )
        elif mode == TemporalSpatialMode.TEMPORAL:
            return self.split_and_expand_per_sensor_temporal(
                x, sensors_to_dims_dict
            )
        elif mode == TemporalSpatialMode.WINDOWED:
            assert self.windowed_attention_size is not None
            return self.split_and_expand_per_sensor_windowed(
                x, sensors_to_dims_dict, self.windowed_attention_size, block_idx
            )
        # Should not be possible.
        assert False

    @staticmethod
    def split_and_expand_per_sensor_full(
        x: Tensor, sensors_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per sensor (full attention).

        Args:
            x: Tokens to split and expand (b n d)
            sensors_to_dims_dict: Dictionary mapping sensors to their dimensions
        Returns:
            tokens_only_dict: mapping sensors to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for sensor, dims in sensors_to_dims_dict.items():
            middle_dims = dims[1:-1]
            num_tokens_for_sensor = math.prod(middle_dims)

            sensor_tokens = x[
                :, tokens_reshaped : tokens_reshaped + num_tokens_for_sensor
            ]

            x_sensor = sensor_tokens.view(x.shape[0], *middle_dims, x.shape[-1])

            tokens_reshaped += num_tokens_for_sensor
            tokens_only_dict[sensor] = x_sensor

        return tokens_only_dict

    @staticmethod
    def split_and_expand_per_sensor_temporal(
        x: Tensor, sensors_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per sensor (temporal attention).

        Args:
            x: Tokens to split and expand (b*h*w t*b_s d)
            sensors_to_dims_dict: Dictionary mapping sensors to their dimensions
        Returns:
            tokens_only_dict: mapping sensors to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for sensor, dims in sensors_to_dims_dict.items():
            if len(dims) == 3:
                batch, b_s, _ = dims
                num_tokens_for_sensor = b_s
                sensor_tokens = x[
                    :, tokens_reshaped : tokens_reshaped + num_tokens_for_sensor, :
                ]
                # Pool the tokens across space back down into a single one.
                sensor_tokens = rearrange(
                    sensor_tokens, "(b hw) b_s d -> b hw b_s d", b=batch
                )
                x_sensor = torch.mean(sensor_tokens, dim=1)

            elif len(dims) == 6:
                batch, h, w, t, b_s, _ = dims

                num_tokens_for_sensor = t * b_s
                sensor_tokens = x[
                    :, tokens_reshaped : tokens_reshaped + num_tokens_for_sensor, :
                ]

                x_sensor = rearrange(
                    sensor_tokens,
                    "(b h w) (t b_s) d -> b h w t b_s d",
                    b=batch,
                    h=h,
                    w=w,
                    t=t,
                    b_s=b_s,
                )

            else:
                raise NotImplementedError(f"not implemented for {len(dims)} dimensions")

            tokens_reshaped += num_tokens_for_sensor
            tokens_only_dict[sensor] = x_sensor

        assert tokens_reshaped == x.shape[1]

        return tokens_only_dict

    @staticmethod
    def split_and_expand_per_sensor_spatial(
        x: Tensor, sensors_to_dims_dict: dict[str, tuple]
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per sensor (spatial attention).

        Args:
            x: Tokens to split and expand (b*t*b_s h*w d)
            sensors_to_dims_dict: Dictionary mapping sensors to their dimensions
        Returns:
            tokens_only_dict: mapping sensors to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for sensor, dims in sensors_to_dims_dict.items():
            if len(dims) == 3:
                batch, b_s, _ = dims
                num_tokens_for_sensor = batch
                sensor_tokens = x[
                    tokens_reshaped : tokens_reshaped + num_tokens_for_sensor, :, :
                ]
                x_sensor = sensor_tokens[:, 0:b_s, :]

            elif len(dims) == 6:
                batch, h, w, t, b_s, _ = dims
                num_tokens_for_sensor = batch * t * b_s
                sensor_tokens = x[
                    tokens_reshaped : tokens_reshaped + num_tokens_for_sensor, :, :
                ]

                x_sensor = rearrange(
                    sensor_tokens,
                    "(b t b_s) (h w) d -> b h w t b_s d",
                    b=batch,
                    h=h,
                    w=w,
                    t=t,
                    b_s=b_s,
                )

            tokens_reshaped += num_tokens_for_sensor
            tokens_only_dict[sensor] = x_sensor

        assert tokens_reshaped == x.shape[0]

        return tokens_only_dict

    @staticmethod
    def split_and_expand_per_sensor_windowed(
        x: Tensor,
        sensors_to_dims_dict: dict[str, tuple],
        window_size: int,
        block_idx: int,
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per sensor (windowed attention).

        Args:
            x: Tokens to split and expand (b*hn*wn t*bs*hs*ws d)
            sensors_to_dims_dict: Dictionary mapping sensors to their dimensions
            window_size: the window size to use.
            block_idx: the block index. Even blocks are shifted.

        Returns:
            tokens_only_dict: mapping sensors to their tokens
        """
        if block_idx % 2 == 0:
            offset_padding = window_size // 2
        else:
            offset_padding = 0
        tokens_only_dict = {}
        tokens_reshaped = 0
        for sensor, dims in sensors_to_dims_dict.items():
            if len(dims) != 6:
                raise NotImplementedError(f"not implemented for {len(dims)} dimensions")

            batch, h, w, t, b_s, _ = dims
            hn = (h + offset_padding + window_size - 1) // window_size
            wn = (w + offset_padding + window_size - 1) // window_size
            num_tokens_for_sensor = t * b_s * window_size * window_size
            sensor_tokens = x[
                :, tokens_reshaped : tokens_reshaped + num_tokens_for_sensor, :
            ]
            # Rearrange to padded form.
            sensor_tokens = rearrange(
                sensor_tokens,
                "(b hn wn) (t bs hs ws) d -> b (hn hs) (wn ws) t bs d",
                b=batch,
                hn=hn,
                wn=wn,
                hs=window_size,
                ws=window_size,
                t=t,
                bs=b_s,
            )
            # Remove beginning padding.
            sensor_tokens = sensor_tokens[:, offset_padding:, offset_padding:]
            # Remove end padding.
            x_sensor = sensor_tokens[:, 0:h, 0:w]

            tokens_reshaped += num_tokens_for_sensor
            tokens_only_dict[sensor] = x_sensor

        assert tokens_reshaped == x.shape[1]
        return tokens_only_dict

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        for block in self.blocks:
            block.enable_fsdp(**fsdp_kwargs)

    def enable_compile(self) -> None:
        """Apply torch.compile to the model."""
        for block in self.blocks:
            block.enable_compile()


# ---------------------------------------------------------------------------
# SpatioTemporalEncoder
# ---------------------------------------------------------------------------


class SpatioTemporalEncoder(SpatioTemporalBase):
    """Encoder module that processes masked input samples into token representations.

    Alternates between spatial and temporal attention (or uses windowed / full
    attention) to capture both spatial context and temporal dynamics.
    """

    cross_attn: bool = False

    def __init__(
        self,
        embed_dim: int,
        tile_patch_size: int,
        min_patch_size: int,
        head_count: int,
        ffn_expansion: float,
        depth: int,
        stochastic_depth_rate: float,
        supported_sensors: list[SensorSpec],
        max_sequence_length: int,
        windowed_attention_size: int | None = None,
        learned_channel_embed: bool = True,
        random_channel_embed: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        fuse_layers: int | None = None,
        layer_attention_modes: list[TemporalSpatialMode] | None = None,
        fuse_using_cross_attn: bool = True,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the encoder.

        Args:
            embed_dim: Size of token embeddings
            tile_patch_size: Maximum patch size for patchification
            min_patch_size: Minimum patch size for patchification
            head_count: Number of attention heads
            ffn_expansion: Ratio for MLP hidden dimension
            depth: Number of transformer layers
            stochastic_depth_rate: Stochastic depth drop rate
            supported_sensors: list documenting sensors used in a given model instantiation
            max_sequence_length: Maximum sequence length
            windowed_attention_size: Window size for windowed attention instead of
                spatial/temporal attention.
            learned_channel_embed: Whether to use learnable channel embeddings
            random_channel_embed: Initialize channel embeddings randomly (zeros if False)
            num_projection_layers: Number of projection layers
            aggregate_then_project: Whether to aggregate then project
            fuse_layers: Do spatial attention for the first portion of the model, then
                do full attention for this many layers, and then on the last layer do
                cross attention to compute a fused representation for each spatial patch.
            layer_attention_modes: Directly specify the attention mode to use at each layer.
            fuse_using_cross_attn: Fuse using cross attention. If disabled, we perform
                self-attention and then pick one unmasked token at each spatial patch to
                copy to all the other tokens at that patch.
            tokenization_config: Optional config for custom band groupings
        """
        self.tokenization_config = tokenization_config or TokenizationConfig()
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            ffn_expansion=ffn_expansion,
            head_count=head_count,
            max_sequence_length=max_sequence_length,
            learned_channel_embed=learned_channel_embed,
            stochastic_depth_rate=stochastic_depth_rate,
            supported_sensors=supported_sensors,
            windowed_attention_size=windowed_attention_size,
            random_channel_embed=random_channel_embed,
            last_layer_cross_attn=fuse_layers is not None and fuse_using_cross_attn,
            tokenization_config=self.tokenization_config,
        )
        self.min_patch_size = min_patch_size
        self.tile_patch_size = tile_patch_size
        self.embed_dim = embed_dim
        self.fuse_layers = fuse_layers
        self.layer_attention_modes = layer_attention_modes
        self.fuse_using_cross_attn = fuse_using_cross_attn
        self.patch_embeddings = MultiSensorPatchProjection(
            self.supported_sensor_labels,
            self.tile_patch_size,
            self.embed_dim,
            tokenization_config=self.tokenization_config,
        )
        self.project_and_aggregate = ProjectionAndPooling(
            embed_dim=self.embed_dim,
            num_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        if self.fuse_layers is not None:
            self.fusing_token = nn.Parameter(torch.zeros(embed_dim))

        self.apply(self._init_weights)

    def create_token_exit_ids(
        self, x: dict[str, Tensor], token_exit_cfg: dict[str, int]
    ) -> dict[str, Tensor]:
        """Create the token exit ids for # of layers of attention for each band group.

        Assumes sensor channel groups are in the second to last dimension of the tokens.
        """
        exit_ids_per_sensor_dict = {}
        available_sensors = collect_sensor_outputs(x)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            num_exit_layers = token_exit_cfg[sensor]
            exit_seq_sensor = torch.full_like(x[sensor], fill_value=num_exit_layers)
            exit_ids_per_sensor_dict[sensor] = exit_seq_sensor
        return exit_ids_per_sensor_dict

    @staticmethod
    def remove_masked_tokens(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
            where T is the max number of unmasked tokens for an instance
        """
        sorted_mask, indices = torch.sort(mask, dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))

        # Set masked values to 0
        x = x * sorted_mask.unsqueeze(-1)

        # Cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = sorted_mask[:, :max_length]

        return x, indices, updated_mask

    @staticmethod
    def add_removed_tokens(
        x: Tensor, indices: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Add removed tokens to the tokens and masks.

        Args:
            x: Tokens to add removed tokens to
            indices: Original indices of the masked tokens
            mask: Mask to add removed tokens to

        Returns:
            tokens: Tokens with removed tokens added
            mask: Mask with removed tokens added
        """
        assert x.shape[1] > 0, (
            "x must have at least one token we should not mask all tokens"
        )
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1]
        )
        full_mask = torch.cat(
            (
                mask,
                torch.zeros(
                    (x.shape[0], indices.shape[1] - x.shape[1]),
                    device=x.device,
                    dtype=mask.dtype,
                ),
            ),
            dim=-1,
        )
        out = masked_tokens.clone()
        out[full_mask] = x[mask]
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        return out, full_mask

    def create_exit_seqs(
        self,
        tokens_only_dict: dict[str, Tensor],
        mask_only_dict: dict[str, Tensor],
        token_exit_cfg: dict[str, int] | None,
    ) -> tuple[Tensor | None]:
        """Create the exit sequences and tokens."""
        assert all(not key.endswith("_mask") for key in tokens_only_dict), (
            "tokens_only_dict should not contain mask keys"
        )
        if token_exit_cfg:
            exit_ids_per_sensor = self.create_token_exit_ids(
                tokens_only_dict, token_exit_cfg
            )
            exit_ids_per_sensor.update(mask_only_dict)
            exit_ids_seq, _ = self.collapse_and_combine_full(exit_ids_per_sensor)
        else:
            exit_ids_seq = None
        return exit_ids_seq

    def copy_first_unmasked_token(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """For each batch, find the first unmasked token and copy it to all unmasked positions.

        Args:
            tokens: Tensor of shape [B, T, D] with token embeddings.
            mask: Tensor of shape [B, T] with 1 for unmasked and 0 for masked tokens.

        Returns:
            Updated tokens of shape [B, T, D].
        """
        B, T, D = tokens.shape

        # Get indices of the first unmasked token for each batch
        first_unmasked_idx = (mask == 1).float().cumsum(dim=1)
        first_unmasked_idx[first_unmasked_idx != 1] = 0
        first_unmasked_idx[first_unmasked_idx == 1] = 1
        idx = first_unmasked_idx.argmax(dim=1)  # shape: [B]

        # Gather the first unmasked tokens
        idx_expanded = idx.view(B, 1, 1).expand(-1, 1, D)  # shape: [B, 1, D]
        first_tokens = torch.gather(tokens, dim=1, index=idx_expanded).squeeze(
            1
        )  # shape: [B, D]

        # Expand to [B, T, D] and mask
        output = tokens.clone()
        output[mask == 1] = first_tokens.unsqueeze(1).expand(-1, T, -1)[mask == 1]

        return output

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
    ) -> dict[str, Tensor]:
        """Apply the attention to the tokens and masks."""
        tokens_only_dict, original_masks_dict, sensors_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        exited_tokens = None

        if exit_ids_seq is not None:
            exited_tokens, mask = self.collapse_and_combine_full(x)
            bool_mask = mask == TokenVisibility.VISIBLE_ENCODER.value
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, bool_mask)
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, bool_mask)

        x = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            tile_patch_size,
            input_res,
        )
        x.update(original_masks_dict)

        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            if (exit_ids_seq is not None) and (i_blk > 0):
                tokens, _ = self.collapse_and_combine_full(x)
                assert exited_tokens is not None
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )

            # Determine attention mode for this layer
            do_token_fusing = False
            if self.layer_attention_modes:
                attention_mode = self.layer_attention_modes[i_blk]
                # With fusing, the last layer must be temporal attention.
                if self.fuse_layers is not None and i_blk == len(self.blocks) - 1:
                    if attention_mode != TemporalSpatialMode.TEMPORAL:
                        raise ValueError(
                            f"with fusing enabled, the last layer must be temporal "
                            f"attention but got {attention_mode}"
                        )
                    do_token_fusing = True
            elif self.windowed_attention_size is not None:
                attention_mode = TemporalSpatialMode.WINDOWED
            elif self.fuse_layers is not None:
                if i_blk < len(self.blocks) - self.fuse_layers - 1:
                    attention_mode = TemporalSpatialMode.SPATIAL
                elif i_blk < len(self.blocks) - 1:
                    attention_mode = TemporalSpatialMode.FULL
                else:
                    attention_mode = TemporalSpatialMode.TEMPORAL
                    do_token_fusing = True
            elif i_blk % 2 == 0:
                attention_mode = TemporalSpatialMode.TEMPORAL
            else:
                attention_mode = TemporalSpatialMode.SPATIAL

            logger.debug(f"Layer {i_blk} applying attention mode {attention_mode}")
            x, mask = self.collapse_and_combine(x, attention_mode, i_blk)
            bool_mask = mask == TokenVisibility.VISIBLE_ENCODER.value
            tokens, indices, new_mask = self.remove_masked_tokens(x, bool_mask)

            if do_token_fusing and self.fuse_using_cross_attn:
                # Last layer with fusing: cross attention for per-spatial-patch tokens.
                logger.debug(f"Layer {i_blk} fusing tokens using cross attention")
                attention_batch_size = tokens.shape[0]
                attention_seq_len = tokens.shape[1]
                fuse_x = (
                    self.fusing_token.unsqueeze(0)
                    .unsqueeze(1)
                    .repeat(attention_batch_size, 1, 1)
                )
                tokens = blk(x=fuse_x, y=tokens, attn_mask=new_mask)
                tokens = tokens.expand(-1, attention_seq_len, -1)
            else:
                tokens = blk(x=tokens, y=None, attn_mask=new_mask)

            # Apply normalization on last block.
            if i_blk == len(self.blocks) - 1:
                tokens = self.norm(tokens)

            if do_token_fusing and not self.fuse_using_cross_attn:
                logger.debug(
                    f"Layer {i_blk} fusing tokens by replicating the first unmasked token"
                )
                tokens = self.copy_first_unmasked_token(tokens, new_mask)

            tokens, _ = self.add_removed_tokens(tokens, indices, new_mask)
            x = self.split_and_expand_per_sensor(
                tokens, sensors_to_dims_dict, attention_mode, i_blk
            )
            x.update(original_masks_dict)

        if exit_ids_seq is not None:
            tokens, _ = self.collapse_and_combine_full(x)
            assert exited_tokens is not None
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),
                input=tokens,
                other=exited_tokens,
            )
            x = self.split_and_expand_per_sensor_full(tokens, sensors_to_dims_dict)
            x.update(original_masks_dict)

        return x

    def forward(
        self,
        x: MaskedGeoSample,
        tile_patch_size: int,
        input_res: int = REFERENCE_GROUND_RESOLUTION,
        token_exit_cfg: dict | None = None,
    ) -> tuple[EmbeddingsAndMasks, Tensor]:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            tile_patch_size: Size of patches to divide the input into
            input_res: Resolution of the input data
            token_exit_cfg: Configuration for token exit

        Returns:
            EmbeddingsAndMasks containing the encoded representations and their masks
        """
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, tile_patch_size)
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            patchified_tokens_and_masks = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                tile_patch_size=tile_patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
            )
        output = EmbeddingsAndMasks(**patchified_tokens_and_masks)
        return output, self.project_and_aggregate(output)

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().enable_fsdp(**fsdp_kwargs)
        fully_shard(self.patch_embeddings, **fsdp_kwargs)
        register_fsdp_forward_method(self.patch_embeddings, "forward")
        fully_shard(self, **fsdp_kwargs)


# ---------------------------------------------------------------------------
# SpatioTemporalPredictor
# ---------------------------------------------------------------------------


class SpatioTemporalPredictor(SpatioTemporalBase):
    """Predictor module that generates predictions from encoded tokens.

    Uses cross-attention to decode masked tokens from the encoder's
    spatiotemporal representations.
    """

    cross_attn = True

    def __init__(
        self,
        supported_sensors: list[SensorSpec],
        encoder_embed_dim: int = 128,
        decoder_embed_dim: int = 128,
        depth: int = 2,
        ffn_expansion: float = 2.0,
        head_count: int = 8,
        max_sequence_length: int = 24,
        stochastic_depth_rate: float = 0.0,
        learned_channel_embed: bool = True,
        random_channel_embed: bool = False,
        output_embedding_size: int | None = None,
        windowed_attention_size: int | None = None,
        layer_attention_modes: list[TemporalSpatialMode] | None = None,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the predictor.

        Args:
            supported_sensors: sensors this model instantiation supports
            encoder_embed_dim: Size of encoder embeddings
            decoder_embed_dim: Size of decoder embeddings
            depth: Number of transformer layers
            ffn_expansion: Ratio for MLP hidden dimension
            head_count: Number of attention heads
            max_sequence_length: Maximum sequence length
            stochastic_depth_rate: Stochastic depth drop rate
            learned_channel_embed: Whether to use learnable channel embeddings
            random_channel_embed: Whether to randomly initialize channel embeddings
            output_embedding_size: Size of output embeddings
            windowed_attention_size: Size for windowed attention. If set, we do
                windowed attention instead of spatial/temporal attention.
            layer_attention_modes: Directly specify the attention mode to use at
                each layer.
            tokenization_config: Optional config for custom band groupings
        """
        self.tokenization_config = tokenization_config or TokenizationConfig()
        super().__init__(
            embed_dim=decoder_embed_dim,
            depth=depth,
            ffn_expansion=ffn_expansion,
            head_count=head_count,
            max_sequence_length=max_sequence_length,
            stochastic_depth_rate=stochastic_depth_rate,
            learned_channel_embed=learned_channel_embed,
            random_channel_embed=random_channel_embed,
            supported_sensors=supported_sensors,
            windowed_attention_size=windowed_attention_size,
            tokenization_config=self.tokenization_config,
        )
        self.learned_channel_embed = learned_channel_embed
        self.random_channel_embed = random_channel_embed
        self.encoder_embed_dim = encoder_embed_dim
        self.layer_attention_modes = layer_attention_modes
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True
        )
        if output_embedding_size is None:
            output_embedding_size = encoder_embed_dim
        self.output_embedding_size = output_embedding_size
        self.to_output_embed = nn.Linear(
            decoder_embed_dim, output_embedding_size, bias=True
        )
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        self.input_norm = nn.LayerNorm(encoder_embed_dim)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.apply(self._init_weights)

    def add_masks(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Replace tokens that should be decoded (TokenVisibility.PREDICTED) with the learnable mask token.

        Uses einops for dimension-agnostic broadcasting. The final dimension of
        each token tensor is assumed to match ``self.mask_token``'s size.
        """
        output_dict = {}
        available_sensors = collect_sensor_outputs(x)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            x_sensor = x[sensor]
            mask_name = MaskedGeoSample.mask_field_for(sensor)
            mask_sensor = x[mask_name]
            # A boolean mask: True where tokens must be replaced by the mask token
            kept_mask = mask_sensor == TokenVisibility.PREDICTED.value

            # Build the einops pattern and dimension dict
            spatial_dims = x_sensor.shape[:-1]
            pattern_input, dim_dict = self._construct_einops_pattern(spatial_dims)

            mask_token_broadcasted = repeat(self.mask_token, pattern_input, **dim_dict)

            x_sensor = torch.where(
                kept_mask.unsqueeze(-1).bool(), mask_token_broadcasted, x_sensor
            )

            output_dict[sensor] = x_sensor

        return output_dict

    @staticmethod
    def split_x_y(
        tokens: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Splits tokens into groups based on mask values.

        This function:
        1. Sorts tokens according to the mask and gathers them in order.
        2. Chooses tokens to be decoded (x) based on the mask value PREDICTED.
        3. Chooses tokens to be used as context (y) based on the mask value VISIBLE_ENCODER.
        4. Identifies missing tokens (z) based on the mask value ABSENT.
        5. Returns boolean masks for x, y, and z along with indices to revert
           to the original ordering.

        Args:
            tokens: Tokens to split of shape [B, T, D].
            mask: Mask of shape [B, T].

        Returns:
            tokens_to_decode: Tokens to be decoded of shape [B, X_len, D].
            unmasked_tokens: Tokens to be used as context of shape [B, Y_len, D].
            tokens_to_decode_mask: Binary mask for x tokens of shape [B, X_len].
            unmasked_tokens_mask: Binary mask for y tokens of shape [B, Y_len].
            indices: Indices for restoring the original token ordering of shape [B, T].
        """
        org_mask_dtype = mask.dtype
        missing_mask = mask == TokenVisibility.ABSENT.value
        mask[missing_mask] = TokenVisibility.TARGET_ONLY.value

        # Sort tokens by mask value (descending order)
        sorted_mask, indices = torch.sort(
            mask.int(), dim=1, descending=True, stable=True
        )
        tokens = tokens.gather(1, indices[:, :, None].expand_as(tokens))

        # Create binary masks for Encoder and Decoder
        binarized_decoder_mask = sorted_mask == TokenVisibility.PREDICTED.value
        binarized_online_encoder_mask = sorted_mask == TokenVisibility.VISIBLE_ENCODER.value

        max_length_of_unmasked_tokens = binarized_online_encoder_mask.sum(dim=-1).max()
        max_length_of_decoded_tokens = binarized_decoder_mask.sum(dim=-1).max()

        tokens_to_decode = tokens[:, :max_length_of_decoded_tokens]
        tokens_to_decode_mask = binarized_decoder_mask[
            :, :max_length_of_decoded_tokens
        ].to(org_mask_dtype)

        unmasked_tokens = tokens[:, -max_length_of_unmasked_tokens:]
        unmasked_tokens_mask = binarized_online_encoder_mask[
            :, -max_length_of_unmasked_tokens:
        ].to(org_mask_dtype)

        return (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
        )

    @staticmethod
    def combine_x_y(
        tokens_to_decode: Tensor,
        unmasked_tokens: Tensor,
        tokens_to_decode_mask: Tensor,
        unmasked_tokens_mask: Tensor,
        indices: Tensor,
    ) -> Tensor:
        """Reintegrate the separated token sequences into their original order.

        Args:
            tokens_to_decode: Key/value tokens of shape [B, X_len, D].
            unmasked_tokens: Query tokens of shape [B, Y_len, D].
            tokens_to_decode_mask: Binary mask for tokens to decode of shape [B, X_len].
            unmasked_tokens_mask: Binary mask for unmasked tokens of shape [B, Y_len].
            indices: Indices for restoring the original token ordering of shape [B, T].

        Returns:
            A merged tokens tensor of shape [B, T, D] with all tokens in their
            original positions.
        """
        B, T = indices.shape[0], indices.shape[1]
        D = tokens_to_decode.shape[-1]
        tokens = torch.zeros(
            (B, T, D), dtype=tokens_to_decode.dtype, device=tokens_to_decode.device
        )
        tokens[:, -unmasked_tokens.shape[1] :] = (
            unmasked_tokens * unmasked_tokens_mask.unsqueeze(-1)
        )
        tokens[:, : tokens_to_decode.shape[1]] += (
            tokens_to_decode * tokens_to_decode_mask.unsqueeze(-1)
        )
        tokens = tokens.scatter(1, indices[:, :, None].expand_as(tokens), tokens)
        return tokens

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        tokens_only_dict, original_masks_dict, sensors_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, tile_patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)
        x = tokens_dict

        for i_blk, blk in enumerate(self.blocks):
            # Determine attention mode
            if self.layer_attention_modes is not None:
                attention_mode = self.layer_attention_modes[i_blk]
            elif self.windowed_attention_size is not None:
                attention_mode = TemporalSpatialMode.WINDOWED
            elif i_blk % 2 == 0:
                attention_mode = TemporalSpatialMode.TEMPORAL
            else:
                attention_mode = TemporalSpatialMode.SPATIAL

            x, mask = self.collapse_and_combine(x, attention_mode, i_blk)
            x, y, x_mask, y_mask, indices = self.split_x_y(x, mask)

            x = blk(x=x, y=y, attn_mask=y_mask.bool())

            x = self.combine_x_y(
                tokens_to_decode=x,
                unmasked_tokens=y,
                tokens_to_decode_mask=x_mask,
                unmasked_tokens_mask=y_mask,
                indices=indices,
            )
            x = self.split_and_expand_per_sensor(
                x, sensors_to_dims_dict, attention_mode, i_blk
            )
            x.update(original_masks_dict)

        return x

    def is_any_data_to_be_decoded(self, sensor_mask: Tensor) -> bool:
        """Check if any data is to be decoded for a given sensor."""
        return (TokenVisibility.PREDICTED.value == sensor_mask).any()

    def forward(
        self,
        x: EmbeddingsAndMasks,
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int = REFERENCE_GROUND_RESOLUTION,
    ) -> EmbeddingsAndMasks:
        """Generate predictions from encoded token representations.

        Args:
            x: EmbeddingsAndMasks containing the encoded tokens to make predictions from
            timestamps: Timestamps of the tokens
            tile_patch_size: Patch size of the tokens
            input_res: Input resolution of the tokens

        Returns:
            EmbeddingsAndMasks containing the predicted tokens and their masks
        """
        decoder_embedded_dict = x._asdict()
        # Apply Input Norms and encoder to decoder embeds to each sensor
        available_sensors = x.sensors
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            x_sensor = getattr(x, sensor)
            x_sensor = self.input_norm(x_sensor)
            x_sensor = self.encoder_to_decoder_embed(x_sensor)
            masked_sensor_name = x.get_masked_sensor_name(sensor)
            decoder_embedded_dict[sensor] = x_sensor
            decoder_embedded_dict[masked_sensor_name] = getattr(
                x, masked_sensor_name
            )

        tokens_only_dict = self.add_masks(decoder_embedded_dict)
        decoder_embedded_dict.update(tokens_only_dict)
        tokens_and_masks = self.apply_attn(
            decoder_embedded_dict, timestamps, tile_patch_size, input_res
        )
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

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().enable_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)


# ---------------------------------------------------------------------------
# SpatioTemporalEncoderConfig
# ---------------------------------------------------------------------------


@dataclass
class SpatioTemporalEncoderConfig(Config):
    """Configuration for the SpatioTemporalEncoder."""

    supported_sensor_labels: list[str]
    embed_dim: int = 16
    # This is the base patch size for the patch embedder
    tile_patch_size: int = 8
    min_patch_size: int = 1
    head_count: int = 2
    ffn_expansion: float = 1.0
    depth: int = 2
    stochastic_depth_rate: float = 0.1
    max_sequence_length: int = 12
    windowed_attention_size: int | None = None
    fuse_layers: int | None = None
    learned_channel_embed: bool = True
    random_channel_embed: bool = False
    layer_attention_modes: list[str] | None = None
    fuse_using_cross_attn: bool = True
    tokenization_config: TokenizationConfig | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_sensors) == 0:
            raise ValueError("At least one sensor must be added!")
        else:
            for sensor in self.supported_sensors:
                if sensor not in SensorRegistry.all_specs():
                    raise ValueError(f"Sensor {sensor} is not supported")
        if self.tokenization_config is not None:
            self.tokenization_config.check_consistency()

        if self.layer_attention_modes is not None:
            if len(self.layer_attention_modes) != self.depth:
                raise ValueError(
                    f"got {len(self.layer_attention_modes)} layer attention modes "
                    f"but depth is {self.depth}"
                )
            for mode in self.layer_attention_modes:
                if mode not in TemporalSpatialMode.__members__:
                    raise ValueError(f"Invalid attention mode {mode}")

    @property
    def supported_sensors(self) -> list[SensorSpec]:
        """Get the supported sensors."""
        return specs_from_labels(self.supported_sensor_labels)

    def build(self) -> "SpatioTemporalEncoder":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_sensor_labels is replaced by supported_sensors
        kwargs.pop("supported_sensor_labels")
        kwargs["supported_sensors"] = self.supported_sensors
        kwargs["layer_attention_modes"] = (
            [TemporalSpatialMode[mode] for mode in self.layer_attention_modes]
            if self.layer_attention_modes
            else None
        )
        logger.info(f"SpatioTemporalEncoder kwargs: {kwargs}")
        return SpatioTemporalEncoder(**kwargs)


# ---------------------------------------------------------------------------
# SpatioTemporalPredictorConfig
# ---------------------------------------------------------------------------


@dataclass
class SpatioTemporalPredictorConfig(Config):
    """Configuration for the SpatioTemporalPredictor."""

    supported_sensor_labels: list[str]
    encoder_embed_dim: int = 16
    decoder_embed_dim: int = 16
    depth: int = 2
    ffn_expansion: float = 1.0
    head_count: int = 2
    max_sequence_length: int = 12
    stochastic_depth_rate: float = 0.0
    learned_channel_embed: bool = True
    random_channel_embed: bool = False
    output_embedding_size: int | None = None
    windowed_attention_size: int | None = None
    layer_attention_modes: list[str] | None = None
    tokenization_config: TokenizationConfig | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_sensors) == 0:
            raise ValueError("At least one sensor must be added!")
        else:
            for sensor in self.supported_sensors:
                if sensor not in SensorRegistry.all_specs():
                    raise ValueError(f"Sensor {sensor} is not supported")
        if self.tokenization_config is not None:
            self.tokenization_config.check_consistency()

        if self.layer_attention_modes is not None:
            if len(self.layer_attention_modes) != self.depth:
                raise ValueError(
                    f"got {len(self.layer_attention_modes)} layer attention modes "
                    f"but depth is {self.depth}"
                )
            for mode in self.layer_attention_modes:
                if mode not in TemporalSpatialMode.__members__:
                    raise ValueError(f"Invalid attention mode {mode}")

    @property
    def supported_sensors(self) -> list[SensorSpec]:
        """Get the supported sensors."""
        return specs_from_labels(self.supported_sensor_labels)

    def build(self) -> "SpatioTemporalPredictor":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_sensor_labels is replaced by supported_sensors
        kwargs.pop("supported_sensor_labels")
        kwargs["supported_sensors"] = self.supported_sensors
        kwargs["layer_attention_modes"] = (
            [TemporalSpatialMode[mode] for mode in self.layer_attention_modes]
            if self.layer_attention_modes
            else None
        )
        logger.info(f"SpatioTemporalPredictor kwargs: {kwargs}")
        return SpatioTemporalPredictor(**kwargs)
