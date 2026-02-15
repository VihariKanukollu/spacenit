"""Adaptive vision encoder for multi-sensor geospatial foundation models.

Implements the full encoder–predictor–reconstructor stack used by Spacenit.
The :class:`VisionEncoder` patchifies heterogeneous sensor inputs, applies
composite positional encodings, and runs a variable-depth transformer.
:class:`LatentPredictor` generates latent predictions from encoded tokens via
cross-attention, and :class:`PixelReconstructor` maps latent representations
back to pixel space.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard

from spacenit.settings import Config
from spacenit.ingestion.sensors import (
    REFERENCE_GROUND_RESOLUTION,
    SensorRegistry,
    SensorSpec,
    specs_from_labels,
)
from spacenit.structures import MaskedGeoSample, TokenVisibility
from spacenit.arch.self_attention import TransformerLayer
from spacenit.arch.positional_encoding import (
    sinusoidal_1d,
    sinusoidal_2d_with_gsd,
    cyclic_month_table,
)
from spacenit.arch.adaptive_patch_embed import (
    AdaptivePatchEmbedding,
    AdaptivePatchReconstruction,
)
from spacenit.arch.band_tokenization import TokenizationConfig
from spacenit.arch.helpers import cumulative_seq_offsets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def sensors_to_process(
    available_sensors: list[str], supported_sensor_labels: list[str]
) -> list[str]:
    """Return the intersection of available and supported sensor labels."""
    sensors_set = set(supported_sensor_labels).intersection(set(available_sensors))
    return list(sensors_set)


def collect_sensor_outputs(
    per_sensor_input_tokens: dict[str, Tensor],
) -> list[str]:
    """Return sensor labels from a dictionary of per-sensor input tokens."""
    return [
        key for key in per_sensor_input_tokens.keys() if not key.endswith("_mask")
    ]


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


class PoolingType(StrEnum):
    """Strategy for pooling the tokens."""

    MAX = "max"
    MEAN = "mean"


# ---------------------------------------------------------------------------
# EmbeddingsAndMasks – the primary token container
# ---------------------------------------------------------------------------


class EmbeddingsAndMasks:
    """Container for per-sensor token embeddings and their visibility masks.

    Backed by two internal dicts (embeddings and masks) keyed by sensor label.
    Embedding tensors have shape ``(B, P_H, P_W, T, BandSets, D)`` and mask
    tensors have shape ``(B, P_H, P_W, T, BandSets)`` with values from
    :class:`TokenVisibility`.

    Supports keyword construction and attribute access for backward compat::

        em = EmbeddingsAndMasks(sentinel2_l2a=emb, sentinel2_l2a_mask=mask)
        em.sentinel2_l2a  # attribute access
    """

    __slots__ = ("_embeddings", "_masks")

    def __init__(self, **kwargs: Tensor | None) -> None:
        object.__setattr__(self, "_embeddings", {})
        object.__setattr__(self, "_masks", {})
        for key, val in kwargs.items():
            if val is None:
                continue
            if key.endswith("_mask"):
                self._masks[key] = val
            else:
                self._embeddings[key] = val

    # -- access ---------------------------------------------------------------

    def __getattr__(self, key: str) -> Tensor | None:
        if key.startswith("_"):
            raise AttributeError(key)
        if key.endswith("_mask"):
            return self._masks.get(key)
        return self._embeddings.get(key)

    def __getitem__(self, key: str) -> Tensor | None:
        if key.endswith("_mask"):
            return self._masks.get(key)
        return self._embeddings.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        elif key.endswith("_mask"):
            if value is None:
                self._masks.pop(key, None)
            else:
                self._masks[key] = value
        else:
            if value is None:
                self._embeddings.pop(key, None)
            else:
                self._embeddings[key] = value

    # -- NamedTuple compat shims ----------------------------------------------

    @property
    def _fields(self) -> tuple[str, ...]:
        return tuple(list(self._embeddings.keys()) + list(self._masks.keys()))

    def _asdict(self) -> dict[str, Tensor | None]:
        return {**self._embeddings, **self._masks}

    def _replace(self, **kwargs: Any) -> EmbeddingsAndMasks:
        merged = {**self._embeddings, **self._masks}
        for k, v in kwargs.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        return EmbeddingsAndMasks(**merged)

    # -- core API -------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        for val in self._embeddings.values():
            if val is not None:
                return val.device
        raise ValueError("No data to get device from")

    @classmethod
    def get_masked_sensor_name(cls, sensor: str) -> str:
        return f"{sensor}_mask"

    def as_dict(self, return_none: bool = True) -> dict[str, Any]:
        if not return_none:
            return {**self._embeddings, **self._masks}
        # Include None for missing keys -- not commonly needed
        result: dict[str, Any] = {}
        for k, v in self._embeddings.items():
            result[k] = v
        for k, v in self._masks.items():
            result[k] = v
        return result

    @property
    def sensors(self) -> list[str]:
        """Return sensor labels that have embeddings (non-None, non-mask)."""
        return list(self._embeddings.keys())

    def get_shape_dict(self) -> dict[str, tuple]:
        result = {}
        for k, v in self._embeddings.items():
            result[k] = v.shape
        for k, v in self._masks.items():
            result[k] = v.shape
        return result

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        return rearrange(x, "b ... d -> b (...) d")

    def flatten_embeddings_and_masks(
        self, return_lists: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Return the flattened embeddings and masks.

        Embeddings will have shape [B, T, D] and masks will have shape [B, T].
        """
        flattened_x, flattened_masks = [], []
        for attr_name in self.sensors:
            mask_attr_name = self.get_masked_sensor_name(attr_name)
            attr = self._embeddings.get(attr_name)
            masked_attr = self._masks.get(mask_attr_name)
            if attr is not None:
                if masked_attr is None:
                    raise ValueError(
                        f"Can't have present {attr_name} but None {mask_attr_name}"
                    )
                masked_attr = masked_attr.unsqueeze(dim=-1)
                flattened_x.append(self._flatten(attr))
                flattened_masks.append(self._flatten(masked_attr))

        if return_lists:
            flattened_masks = [mask[:, :, 0] for mask in flattened_masks]
            return flattened_x, flattened_masks

        x = torch.cat(flattened_x, dim=1)
        masks = torch.cat(flattened_masks, dim=1)[:, :, 0]
        return x, masks

    def pool_spatially_and_concat_sensors(self) -> Tensor:
        """Pool sensors across time for spatial features and concatenate."""
        from spacenit.ingestion.sensors import SensorRegistry

        spatial_stacked_features = []
        for attr_name in self.sensors:
            if SensorRegistry.get(attr_name).varies_in_space:
                mask_attr_name = self.get_masked_sensor_name(attr_name)
                masked_attr = self._masks.get(mask_attr_name)
                if masked_attr is None:
                    continue
                if (masked_attr == TokenVisibility.VISIBLE_ENCODER.value).all():
                    attr = self._embeddings[attr_name]
                    pooled_attr = torch.mean(attr, dim=(-3))
                    spatial_stacked_features.append(pooled_attr)
        if len(spatial_stacked_features) == 0:
            raise ValueError("Missing unmasked spatial sensors for spatial pooling.")
        spatial_stacked_features = torch.cat(spatial_stacked_features, dim=-2)
        return spatial_stacked_features

    def pool_spatially(self, pooling_type: PoolingType) -> Tensor:
        """Pool sensors across time to get spatial features."""
        from spacenit.ingestion.sensors import SensorRegistry

        spatial_average = []
        for attr_name in self.sensors:
            if SensorRegistry.get(attr_name).varies_in_space:
                mask_attr_name = self.get_masked_sensor_name(attr_name)
                masked_attr = self._masks.get(mask_attr_name)
                if masked_attr is None:
                    continue
                if (masked_attr == TokenVisibility.VISIBLE_ENCODER.value).all():
                    attr = self._embeddings[attr_name]
                    if pooling_type == PoolingType.MEAN:
                        spatial_average.append(torch.mean(attr, dim=(-2, -3)))
                    else:
                        spatial_average.append(
                            torch.max(torch.max(attr, dim=-2).values, dim=-2).values
                        )
        if len(spatial_average) == 0:
            raise ValueError("Missing unmasked spatial sensors for spatial pooling.")
        spatial_average_t = torch.stack(spatial_average, dim=-1)
        if pooling_type == PoolingType.MEAN:
            return spatial_average_t.mean(dim=-1)
        else:
            return spatial_average_t.max(dim=-1).values

    def pool_instance_wise(self, pooling_type: PoolingType) -> Tensor:
        """Pool all the tokens in the instance."""
        x, mask = self.flatten_embeddings_and_masks()
        mask = (mask == TokenVisibility.VISIBLE_ENCODER.value).long()
        x_for_pooling = x * mask.unsqueeze(-1)
        if pooling_type == PoolingType.MAX:
            x_for_pooling = x_for_pooling.masked_fill(
                ~mask.bool().unsqueeze(-1), -float("inf")
            )
            return x_for_pooling.max(dim=1).values
        elif pooling_type == PoolingType.MEAN:
            num_encoded_tokens = torch.sum(mask, -1, keepdim=True)
            logger.debug(f"num_encoded_tokens: {num_encoded_tokens}")
            if (num_encoded_tokens == 0).any():
                raise ValueError(
                    f"num_encoded_tokens is 0 for some samples {num_encoded_tokens}"
                )
            return x_for_pooling.sum(dim=1) / num_encoded_tokens
        else:
            raise ValueError(f"Invalid pooling type: {pooling_type}")

    def pool_unmasked_tokens(
        self,
        pooling_type: PoolingType = PoolingType.MAX,
        spatial_pooling: bool = False,
        concat_features: bool = False,
    ) -> Tensor:
        """Pool the unmasked tokens."""
        if concat_features and spatial_pooling:
            return self.pool_spatially_and_concat_sensors()
        if concat_features:
            raise ValueError("concat_features is not supported for non-spatial pooling")
        if not spatial_pooling:
            return self.pool_instance_wise(pooling_type)
        else:
            return self.pool_spatially(pooling_type)


# ---------------------------------------------------------------------------
# ProjectionAndPooling
# ---------------------------------------------------------------------------


class ProjectionAndPooling(nn.Module):
    """Module that applies a linear projection to embeddings and masks."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        aggregate_then_project: bool = True,
    ):
        """Initialize the linear module.

        embed_dim: The embedding dimension of the input EmbeddingsAndMasks
        num_layers: The number of layers to use in the projection. If >1, then
            a ReLU activation will be applied between layers
        aggregate_then_project: If True, then we will average the tokens before applying
            the projection. If False, we will apply the projection first.
        """
        super().__init__()
        projections = [nn.Linear(embed_dim, embed_dim)]
        for _ in range(1, num_layers):
            projections.append(nn.ReLU())
            projections.append(nn.Linear(embed_dim, embed_dim))
        self.projection = nn.Sequential(*projections)
        self.aggregate_then_project = aggregate_then_project

    def apply_aggregate_then_project(
        self, x: EmbeddingsAndMasks | torch.Tensor
    ) -> torch.Tensor:
        """Apply the aggregate operation to the input."""
        if isinstance(x, EmbeddingsAndMasks):
            pooled_for_contrastive = x.pool_unmasked_tokens(
                PoolingType.MEAN, spatial_pooling=False
            )
        elif isinstance(x, torch.Tensor):
            pooled_for_contrastive = reduce(x, "b ... d -> b  d", "mean")
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        return self.projection(pooled_for_contrastive)

    def apply_project_then_aggregate(
        self, x: EmbeddingsAndMasks | torch.Tensor
    ) -> torch.Tensor:
        """Apply the project operation to the input then aggregate."""
        if isinstance(x, EmbeddingsAndMasks):
            decoder_embedded_dict = x._asdict()
            for sensor in x.sensors:
                x_sensor = getattr(x, sensor)
                x_sensor = self.projection(x_sensor)
                masked_sensor_name = x.get_masked_sensor_name(sensor)
                decoder_embedded_dict[sensor] = x_sensor
                decoder_embedded_dict[masked_sensor_name] = getattr(
                    x, masked_sensor_name
                )
            x_projected = EmbeddingsAndMasks(**decoder_embedded_dict)
            projected_pooled = x_projected.pool_unmasked_tokens(
                PoolingType.MEAN, spatial_pooling=False
            )
        elif isinstance(x, torch.Tensor):
            x_projected = self.projection(x)
            projected_pooled = reduce(x_projected, "b ... d -> b  d", "mean")
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        return projected_pooled

    def forward(self, x: EmbeddingsAndMasks | torch.Tensor) -> torch.Tensor:
        """Apply a (non)linear projection to an input EmbeddingsAndMasks.

        This can be applied either before or after pooling the tokens.
        """
        return (
            self.apply_aggregate_then_project(x)
            if self.aggregate_then_project
            else self.apply_project_then_aggregate(x)
        )


# ---------------------------------------------------------------------------
# MultiSensorPatchProjection
# ---------------------------------------------------------------------------


class MultiSensorPatchProjection(nn.Module):
    """Module that patchifies and encodes the input data for multiple sensors."""

    def __init__(
        self,
        supported_sensor_labels: list[str],
        tile_patch_size: int,
        embed_dim: int,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the patch projections.

        Args:
            supported_sensor_labels: Which sensors this model instantiation supports
            tile_patch_size: Maximum size of patches
            embed_dim: Size of embeddings
            tokenization_config: Optional config for custom band groupings
        """
        super().__init__()
        self.tile_patch_size = tile_patch_size
        self.embed_dim = embed_dim
        self.supported_sensor_labels = supported_sensor_labels
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.per_sensor_embeddings = nn.ModuleDict({})

        for sensor in self.supported_sensor_labels:
            self.per_sensor_embeddings[sensor] = (
                self._get_patch_embedding_module_for_sensor(sensor)
            )

        # For every patch embedding module we want to create a unique buffer
        # for selecting the correct band indices from the data tensor
        for sensor in self.supported_sensor_labels:
            for idx, bandset_indices in enumerate(
                self.tokenization_config.group_indices_for(sensor)
            ):
                buffer_name = self._get_buffer_name(sensor, idx)
                bandset_indices_tensor = torch.tensor(bandset_indices, dtype=torch.long)
                self.register_buffer(
                    buffer_name, bandset_indices_tensor, persistent=False
                )

    @staticmethod
    def _get_buffer_name(sensor: str, idx: int) -> str:
        """Get the buffer name."""
        return f"{sensor}__{idx}_buffer"

    @staticmethod
    def _get_embedding_module_name(sensor: str, idx: int) -> str:
        """Get the embedding module name.

        Module Dicts require string keys
        """
        return f"{sensor}__{idx}"

    def _get_patch_embedding_module_for_sensor(self, sensor: str) -> nn.Module:
        """Get the patch embedding module for a sensor."""
        sensor_spec = SensorRegistry.get(sensor)
        # Get bandset indices from tokenization config (may be overridden)
        bandset_indices = self.tokenization_config.group_indices_for(sensor)

        # Based on the sensor label we choose the way to embed the data
        if not sensor_spec.varies_in_space:
            # static in space
            return nn.ModuleDict(
                {
                    self._get_embedding_module_name(sensor, idx): nn.Linear(
                        len(channel_set_idxs), self.embed_dim
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )
        else:
            return nn.ModuleDict(
                {
                    self._get_embedding_module_name(sensor, idx): AdaptivePatchEmbedding(
                        input_channels=len(channel_set_idxs),
                        embed_dim=self.embed_dim,
                        base_patch_size=self.tile_patch_size,
                        sensor_spec=sensor_spec,
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )

    def apply_embedding_to_sensor(
        self,
        sensor: str,
        input_data: MaskedGeoSample,
        tile_patch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Apply embedding to a sensor."""
        logger.debug(f"applying embedding to sensor:{sensor}")
        masked_sensor_name = input_data.mask_field_for(sensor)
        sensor_mask = getattr(input_data, masked_sensor_name)
        sensor_data = getattr(input_data, sensor)

        sensor_spec = SensorRegistry.get(sensor)
        num_band_sets = self.tokenization_config.group_count_for(sensor)

        sensor_tokens, sensor_masks = [], []
        for idx in range(num_band_sets):
            sensor_specific_kwargs = {}
            if not sensor_spec.varies_in_space:
                # static in time
                token_mask = sensor_mask[..., idx]
            else:
                token_mask = sensor_mask[
                    :,
                    0 :: tile_patch_size * sensor_spec.tile_size_multiplier,
                    0 :: tile_patch_size * sensor_spec.tile_size_multiplier,
                    ...,
                    idx,
                ]
                sensor_specific_kwargs = {"patch_size": tile_patch_size}

            buffer_name = self._get_buffer_name(sensor, idx)
            patchified_data = torch.index_select(
                sensor_data, -1, getattr(self, buffer_name)
            )
            embedding_module = self.per_sensor_embeddings[sensor][
                self._get_embedding_module_name(sensor, idx)
            ]
            patchified_data = embedding_module(
                patchified_data, **sensor_specific_kwargs
            )

            sensor_tokens.append(patchified_data)
            sensor_masks.append(token_mask)
        return torch.stack(sensor_tokens, dim=-2), torch.stack(sensor_masks, dim=-1)

    @staticmethod
    def is_any_data_seen_by_encoder(sensor_mask: Tensor) -> bool:
        """Check if any data is seen by the encoder."""
        return (TokenVisibility.VISIBLE_ENCODER.value == sensor_mask).any()

    def enable_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)

    def forward(
        self,
        input_data: MaskedGeoSample,
        tile_patch_size: int,
    ) -> dict[str, Tensor]:
        """Return flexibly patchified embeddings for each sensor of the input data.

        Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), b_s, D] output.

        We assume that the spatial masks are consistent for the given patch size,
        so that if tile_patch_size == 2 then one possible mask would be
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
        for the H, W dimensions
        """
        output_dict = {}
        sensors_list = sensors_to_process(
            input_data.present_keys, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            sensor_tokens, sensor_masks = self.apply_embedding_to_sensor(
                sensor, input_data, tile_patch_size
            )
            output_dict[sensor] = sensor_tokens
            sensor_mask_name = input_data.mask_field_for(sensor)
            output_dict[sensor_mask_name] = sensor_masks
        return output_dict


# ---------------------------------------------------------------------------
# PixelReconstructor
# ---------------------------------------------------------------------------


class PixelReconstructor(nn.Module):
    """Module that reconstructs pixel-level data from token embeddings."""

    def __init__(
        self,
        decoder: nn.Module,
        supported_sensors: list[SensorSpec],
        tile_patch_size: int,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the pixel reconstructor.

        Args:
            decoder: Predictor nn module to use before reconstruction on input
            supported_sensors: Which sensors this model instantiation supports
            tile_patch_size: Maximum size of patches
            tokenization_config: Optional config for custom band groupings
        """
        super().__init__()
        self.tile_patch_size = tile_patch_size
        self.embed_dim = decoder.output_embedding_size
        self.supported_sensors = supported_sensors
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.decoder = decoder
        self.per_sensor_reconstructions = nn.ModuleDict({})
        for sensor in self.supported_sensors:
            self.per_sensor_reconstructions[sensor.label] = (
                self._get_patch_reconstruction_module_for_sensor(sensor)
            )

    def enable_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.decoder.enable_compile()

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        self.decoder.enable_fsdp(**fsdp_kwargs)

    @staticmethod
    def _get_reconstruction_module_name(sensor: str, idx: int) -> str:
        """Get the reconstruction module name.

        Module Dicts require string keys
        """
        return f"{sensor}__{idx}"

    def _get_patch_reconstruction_module_for_sensor(
        self, sensor: SensorSpec
    ) -> nn.Module:
        """Get the patch reconstruction module for a sensor."""
        # Get bandset indices from tokenization config (may be overridden)
        bandset_indices = self.tokenization_config.group_indices_for(sensor.label)

        # Based on the sensor we choose the way to reconstruct the data
        if sensor.compute_coverage_pitch() == 0:
            # static in space
            return nn.ModuleDict(
                {
                    self._get_reconstruction_module_name(sensor.label, idx): nn.Linear(
                        self.embed_dim, len(channel_set_idxs)
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )
        else:
            return nn.ModuleDict(
                {
                    self._get_reconstruction_module_name(
                        sensor.label, idx
                    ): AdaptivePatchReconstruction(
                        output_channels=len(channel_set_idxs),
                        embed_dim=self.embed_dim,
                        largest_patch_size=self.tile_patch_size,
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )

    def apply_reconstruction_to_sensor(
        self, sensor: str, input_data: EmbeddingsAndMasks, tile_patch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Apply reconstruction to a sensor."""
        masked_sensor_name = input_data.get_masked_sensor_name(sensor)
        sensor_mask = getattr(input_data, masked_sensor_name)
        sensor_data = getattr(input_data, sensor)

        sensor_spec = SensorRegistry.get(sensor)
        bandset_indices = self.tokenization_config.group_indices_for(sensor)

        # x: Input tensor with shape [b, h, w, (t), b_s, d]
        sensor_tokens, sensor_masks = [], []
        for idx, channel_set_indices in enumerate(bandset_indices):
            data = sensor_data[..., idx, :]
            masks = sensor_mask[..., idx]
            r_model = self.per_sensor_reconstructions[sensor][
                self._get_reconstruction_module_name(sensor, idx)
            ]
            if sensor_spec.compute_coverage_pitch() == 0:
                data = r_model(data)
            else:
                data = r_model(data, patch_size=tile_patch_size)
            sensor_tokens.append(data)
            masks = repeat(
                masks,
                "b h w ... -> b (h p_h) (w p_w) ...",
                p_h=tile_patch_size,
                p_w=tile_patch_size,
            )
            sensor_masks.append(masks)
        sensor_mask = repeat(
            sensor_mask,
            "b h w ... -> b (h p_h) (w p_w) ...",
            p_h=tile_patch_size,
            p_w=tile_patch_size,
        )
        return torch.cat(sensor_tokens, dim=-1), sensor_mask

    def forward(
        self,
        x: EmbeddingsAndMasks,
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int = REFERENCE_GROUND_RESOLUTION,
    ) -> EmbeddingsAndMasks:
        """Return flexibly patchified reconstruction for each sensor of the input data.

        Given a [B, H, W, (T), b_s, D] inputs, returns a [B, H, W, (T), C] output.
        """
        input_data = self.decoder(x, timestamps, tile_patch_size, input_res)
        output_dict = {}
        sensors_list = sensors_to_process(
            input_data.sensors, [m.label for m in self.supported_sensors]
        )
        for sensor in sensors_list:
            sensor_tokens, sensor_masks = self.apply_reconstruction_to_sensor(
                sensor, input_data, tile_patch_size
            )
            output_dict[sensor] = sensor_tokens
            sensor_mask_name = input_data.get_masked_sensor_name(sensor)
            output_dict[sensor_mask_name] = sensor_masks
        return EmbeddingsAndMasks(**output_dict)


# ---------------------------------------------------------------------------
# PixelReconstructorConfig
# ---------------------------------------------------------------------------


@dataclass
class PixelReconstructorConfig(Config):
    """Configuration for the PixelReconstructor."""

    decoder_config: "Config"
    supported_sensor_labels: list[str]
    tile_patch_size: int = 8
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

    @property
    def supported_sensors(self) -> list[SensorSpec]:
        """Get the supported sensors."""
        return specs_from_labels(self.supported_sensor_labels)

    def build(self) -> "PixelReconstructor":
        """Build the pixel reconstructor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_sensor_labels")
        kwargs["supported_sensors"] = self.supported_sensors
        kwargs.pop("decoder_config")
        kwargs["decoder"] = self.decoder_config.build()
        logger.info(f"LatentPredictor kwargs: {kwargs}")
        return PixelReconstructor(**kwargs)


# ---------------------------------------------------------------------------
# CompositePositionalEncodings
# ---------------------------------------------------------------------------


class CompositePositionalEncodings(nn.Module):
    """Composite encodings for adaptive vision encoder models."""

    def __init__(
        self,
        embed_dim: int,
        supported_sensors: list[SensorSpec],
        max_sequence_length: int,
        learned_channel_embed: bool = True,
        random_channel_embed: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the composite encodings.

        Args:
            embed_dim: Size of token embeddings
            supported_sensors: Which sensors this model instantiation supports
            max_sequence_length: Maximum sequence length
            learned_channel_embed: Whether to use learnable channel embeddings
            random_channel_embed: Initialize channel embeddings randomly (zeros if False)
            tokenization_config: Optional config for custom band groupings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.supported_sensors = supported_sensors
        self.supported_sensor_labels = [
            sensor.label for sensor in supported_sensors
        ]
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.max_sequence_length = (
            max_sequence_length  # This max sequence length is a time dim thing
        )

        # we have 4 embeddings types (pos_in_time, pos_in_space, month, channel) so each get
        # 0.25 of the dimension
        self.embedding_dim_per_embedding_type = int(embed_dim * 0.25)
        # Position encodings for time dimension initialized to 1D sinusoidal encodings
        self.pos_embed = nn.Parameter(
            sinusoidal_1d(
                torch.arange(max_sequence_length),
                self.embedding_dim_per_embedding_type,
            ),
            requires_grad=False,
        )
        # Month encodings
        month_tab = cyclic_month_table(self.embedding_dim_per_embedding_type)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if not learned_channel_embed and not random_channel_embed:
            self.per_sensor_channel_embeddings = nn.ParameterDict()
            for sensor in self.supported_sensors:
                num_bandsets = self.tokenization_config.group_count_for(sensor.label)
                shape = (num_bandsets, self.embedding_dim_per_embedding_type)
                channel_embeddings = nn.Parameter(
                    torch.zeros(shape), requires_grad=False
                )
                self.per_sensor_channel_embeddings[sensor.label] = channel_embeddings
        else:
            # Channel embeddings
            if learned_channel_embed:
                args = {"requires_grad": True}
            else:
                args = {"requires_grad": False}

            self.per_sensor_channel_embeddings = nn.ParameterDict()
            for sensor in self.supported_sensors:
                num_bandsets = self.tokenization_config.group_count_for(sensor.label)
                shape = (num_bandsets, self.embedding_dim_per_embedding_type)
                if random_channel_embed:
                    channel_embeddings = nn.Parameter(torch.rand(shape), **args)
                else:
                    channel_embeddings = nn.Parameter(torch.zeros(shape), **args)
                self.per_sensor_channel_embeddings[sensor.label] = channel_embeddings

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0).to(torch.float32)

    @staticmethod
    def calculate_gsd_ratio(input_res: float, tile_patch_size: int) -> float:
        """Calculate the Ground Sample Distance ratio."""
        return input_res * tile_patch_size / REFERENCE_GROUND_RESOLUTION

    def _apply_encodings_per_sensor(
        self,
        sensor_label: str,
        sensor_tokens: Tensor,
        timestamps: Tensor | None = None,
        tile_patch_size: int | None = None,
        input_res: int | None = None,
        use_sensor_encodings: bool = True,
        use_temporal_encodings: bool = True,
    ) -> Tensor:
        """Apply the encodings to the patchified data based on sensor type.

        Args:
            sensor_label: Label of the sensor being processed
            sensor_tokens: Token embeddings for the sensor
            timestamps: Optional timestamps for temporal encodings
            tile_patch_size: Optional patch size for spatial encodings
            input_res: Optional input resolution for spatial encodings
            use_sensor_encodings: Whether to use sensor encodings
            use_temporal_encodings: Whether to use temporal encodings

        Returns:
            Tensor with encodings applied based on sensor type
        """
        logger.debug(
            f"use_sensor_encodings: {use_sensor_encodings}, use_temporal_encodings: {use_temporal_encodings}"
        )

        sensor_spec = SensorRegistry.get(sensor_label)
        logger.debug(f"Applying encodings to sensor {sensor_spec}")
        if not use_sensor_encodings and use_temporal_encodings:
            b, h, w, t, _ = sensor_tokens.shape
            ein_string, ein_dict = (
                "b h w t d",
                {"b": b, "h": h, "w": w, "t": t},
            )
        elif not use_temporal_encodings and not use_sensor_encodings:
            b, h, w, _ = sensor_tokens.shape
            ein_string, ein_dict = (
                "b h w d",
                {"b": b, "h": h, "w": w},
            )
        elif not use_temporal_encodings and use_sensor_encodings:
            raise NotImplementedError("Not implemented")
        else:
            if sensor_tokens.ndim == 3:
                # sensor_tokens = [B, Band_Sets, D]; static in space, static in time
                b, b_s, _ = sensor_tokens.shape
                ein_string, ein_dict = "b b_s d", {"b": b, "b_s": b_s}
            elif sensor_tokens.ndim == 4:
                b, t, b_s, _ = sensor_tokens.shape
                ein_string, ein_dict = "b t b_s d", {"b": b, "t": t, "b_s": b_s}
            elif sensor_tokens.ndim == 5:
                b, h, w, b_s, _ = sensor_tokens.shape
                ein_string, ein_dict = (
                    "b h w b_s d",
                    {"b": b, "h": h, "w": w, "b_s": b_s},
                )
            elif sensor_tokens.ndim == 6:
                b, h, w, t, b_s, _ = sensor_tokens.shape
                ein_string, ein_dict = (
                    "b h w t b_s d",
                    {"b": b, "h": h, "w": w, "t": t, "b_s": b_s},
                )
            else:
                raise ValueError(f"Unsupported tokens shape: {sensor_tokens.shape}")

        device = sensor_tokens.device
        sensor_embed = torch.zeros(sensor_tokens.shape, device=device)
        n = self.embedding_dim_per_embedding_type
        actual_bandsets = sensor_tokens.shape[-2]

        # Channel embeddings
        if use_sensor_encodings:
            channel_embed = self.per_sensor_channel_embeddings[sensor_spec.label]
            if channel_embed.shape[0] != actual_bandsets:
                raise ValueError(
                    f"Channel embeddings for {sensor_spec.label} expect "
                    f"{channel_embed.shape[0]} bandsets but tokens have "
                    f"{actual_bandsets}. Ensure tokenization_config is "
                    "consistently passed to the encoder/decoder and masking strategy."
                )
            channel_embed = repeat(
                channel_embed, f"b_s d -> {ein_string}", **ein_dict
            ).to(device)
            sensor_embed[..., :n] += channel_embed

        if sensor_spec.has_temporal_axis and use_temporal_encodings:
            # Time position encodings
            time_embed = repeat(self.pos_embed[:t], f"t d -> {ein_string}", **ein_dict)
            sensor_embed[..., n : n * 2] += time_embed.to(device)

            # Month encodings
            assert timestamps is not None
            months = timestamps[:, :, 1]
            month_embed = self.month_embed(months)
            month_embed = repeat(month_embed, f"b t d -> {ein_string}", **ein_dict)
            sensor_embed[..., n * 2 : n * 3] += month_embed.to(device)
        if sensor_spec.varies_in_space:
            # Spatial encodings
            assert input_res is not None
            assert tile_patch_size is not None
            gsd_ratio = self.calculate_gsd_ratio(input_res, tile_patch_size)
            spatial_embed = sinusoidal_2d_with_gsd(
                side_length=h,
                pixel_pitch=torch.ones(b, device=device) * gsd_ratio,
                encoding_dim=self.embedding_dim_per_embedding_type,
            )
            spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
            spatial_embed = repeat(
                spatial_embed, f"b h w d -> {ein_string}", **ein_dict
            )
            sensor_embed[..., n * 3 : n * 4] += spatial_embed
        return sensor_tokens + sensor_embed

    def forward(
        self,
        per_sensor_input_tokens: dict[str, Tensor],
        timestamps: Tensor,
        tile_patch_size: int,
        input_res: int = REFERENCE_GROUND_RESOLUTION,
    ) -> dict[str, Tensor]:
        """Apply the encodings to the patchified data.

        Args:
            per_sensor_input_tokens: Tokens only for each sensor
            timestamps: Timestamps of the data
            tile_patch_size: Size of patches
            input_res: Resolution of the input data

        Returns:
            Tokens only for each sensor
        """
        output_dict = {}
        available_sensors = collect_sensor_outputs(per_sensor_input_tokens)
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor_label in sensors_list:
            output_dict[sensor_label] = self._apply_encodings_per_sensor(
                sensor_label,
                per_sensor_input_tokens[sensor_label],
                timestamps=timestamps,
                tile_patch_size=tile_patch_size,
                input_res=input_res,
            )
        return output_dict


# ---------------------------------------------------------------------------
# AdaptiveVisionBase – shared base for encoder and predictor
# ---------------------------------------------------------------------------


class AdaptiveVisionBase(nn.Module):
    """Base class for adaptive vision encoder models."""

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
        learned_channel_embed: bool = True,
        random_channel_embed: bool = False,
        use_flash_attn: bool = False,
        qk_norm: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ) -> None:
        """Initialize the AdaptiveVisionBase class."""
        super().__init__()

        self.embed_dim = embed_dim
        self.supported_sensors = supported_sensors
        self.supported_sensor_labels = [x.label for x in supported_sensors]
        logger.info(f"sensors being used by model: {self.supported_sensor_labels}")

        self.max_sequence_length = max_sequence_length
        self._base_tokenization_config = tokenization_config or TokenizationConfig()

        self.use_flash_attn = use_flash_attn
        self.learned_channel_embed = learned_channel_embed
        self.random_channel_embed = random_channel_embed
        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim,
                    head_count,
                    ffn_expansion,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=nn.LayerNorm,
                    cross_attn=self.cross_attn,
                    drop_path=stochastic_depth_rate,
                    use_flash_attn=self.use_flash_attn,
                )
                for _ in range(depth)
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
            # we use xavier_uniform following official JAX ViT:
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

    def collapse_and_combine_hwtc(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Collapse the tokens and masks, respectively, into two tensors."""
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
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)

        return tokens, masks

    @staticmethod
    def _construct_einops_pattern(
        spatial_dims: tuple[int, ...],
    ) -> tuple[str, dict[str, int]]:
        """Given a tuple of spatial dimensions (e.g. [B, H, W, T, ...]).

        build (1) an einops rearrange pattern of the form:
            "d -> (dim0) (dim1) (dim2)... d"
        and (2) a dictionary mapping dim0..dimN to the actual sizes.

        This allows reshaping a single-dimensional tensor [D] into
        [B, H, W, T, ..., D] using einops.
        """
        dim_dict = {f"dim{i}": size for i, size in enumerate(spatial_dims)}
        # e.g., "d -> (dim0) (dim1) (dim2) (dim3) d"
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

    @staticmethod
    def split_and_expand_per_sensor(
        x: Tensor, sensors_to_dims_dict: dict
    ) -> dict[str, Tensor]:
        """Split and expand the tokens per sensor.

        Args:
            x: Tokens to split and expand (b n d)
            sensors_to_dims_dict: Dictionary mapping sensors to their dimensions
        Returns:
            tokens_only_dict: mapping sensors to their tokens
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for sensor, dims in sensors_to_dims_dict.items():
            # Skip batch (first) and embedding (last) dimensions
            middle_dims = dims[1:-1]
            num_tokens_for_sensor = math.prod(middle_dims)

            # Extract tokens for this sensor (b n d)
            sensor_tokens = x[
                :, tokens_reshaped : tokens_reshaped + num_tokens_for_sensor
            ]

            # Reshape to original dimensions (e.g., for 4D spatial dims: b d1 d2 d3 d4 e)
            x_sensor = sensor_tokens.view(x.shape[0], *middle_dims, x.shape[-1])

            tokens_reshaped += num_tokens_for_sensor
            tokens_only_dict[sensor] = x_sensor

        return tokens_only_dict

    @staticmethod
    def pack_tokens(tokens: Tensor, mask: Tensor) -> Tensor:
        """Pack the Batch and sequence length dimensions of tokens and mask into a single tensor.

        Args:
            tokens: Tokens to pack
            mask: Mask to pack

        Returns:
            Packed tokens enabling varlen flash attention
        """
        tokens_packed = torch.flatten(tokens, end_dim=1)
        mask = torch.flatten(mask)
        tokens = tokens_packed[mask]
        return tokens

    @staticmethod
    def unpack_tokens(tokens: Tensor, mask: Tensor, og_shape: tuple) -> Tensor:
        """Unpack the Batch and sequence length dimensions of tokens and mask into a single tensor.

        Args:
            tokens: Tokens to unpack
            mask: Mask to unpack
            og_shape: Original shape of the tokens
        """
        tokens_new = tokens.new_zeros(og_shape[0] * og_shape[1], og_shape[2])
        mask = torch.flatten(mask)
        tokens_new[mask] = tokens
        tokens = tokens_new.reshape(og_shape[0], og_shape[1], -1)
        return tokens

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        for block in self.blocks:
            block.enable_fsdp(**fsdp_kwargs)

    def enable_compile(self) -> None:
        """Apply torch.compile to the model."""
        for block in self.blocks:
            block.enable_compile()


# ---------------------------------------------------------------------------
# VisionEncoder
# ---------------------------------------------------------------------------


class VisionEncoder(AdaptiveVisionBase):
    """Encoder module that processes masked input samples into token representations."""

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
        num_register_tokens: int = 0,
        learned_channel_embed: bool = True,
        random_channel_embed: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        use_flash_attn: bool = False,
        frozen_patch_embeddings: bool = False,
        qk_norm: bool = False,
        log_token_norm_stats: bool = False,
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
            num_register_tokens: Number of register tokens to use
            learned_channel_embed: Whether to use learnable channel embeddings
            random_channel_embed: Initialize channel embeddings randomly (zeros if False)
            num_projection_layers: The number of layers to use in the projection. If >1, then
                a ReLU activation will be applied between layers
            aggregate_then_project: If True, then we will average the tokens before applying
                the projection. If False, we will apply the projection first.
            use_flash_attn: Whether to use flash attention
            frozen_patch_embeddings: If True, we freeze the embedding layer, as recommended in
                https://arxiv.org/pdf/2104.02057, Section 4.2
            qk_norm: Whether to apply normalization to Q and K in attention
            log_token_norm_stats: Whether to log the token norm stats
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
            use_flash_attn=use_flash_attn,
            random_channel_embed=random_channel_embed,
            qk_norm=qk_norm,
            tokenization_config=self.tokenization_config,
        )
        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0
        self.log_token_norm_stats = log_token_norm_stats
        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(
                torch.zeros(num_register_tokens, embed_dim)
            )
        self.min_patch_size = min_patch_size
        self.tile_patch_size = tile_patch_size
        self.embed_dim = embed_dim
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
        self.apply(self._init_weights)

        if frozen_patch_embeddings:
            for p in self.patch_embeddings.parameters():
                p.requires_grad = False
        if self.has_register_tokens:
            self._init_register_tokens()

    def _init_register_tokens(self) -> None:
        """Initialize the register tokens."""
        nn.init.xavier_uniform_(self.register_tokens)

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
    def remove_masked_tokens(
        x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks.

        Implementation from https://stackoverflow.com/a/68621610/2332296

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
        sorted_mask, indices = torch.sort(mask, dim=1, descending=True, stable=True)
        # Now all the places where we want to keep the token are at the front of the tensor
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # Now all tokens that should be kept are first in the tensor

        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        seq_lengths = sorted_mask.sum(-1)
        longest_sequence = seq_lengths.max()
        x = x[:, :longest_sequence]
        # New mask chopped to the longest sequence
        updated_mask = sorted_mask[:, :longest_sequence]

        return x, indices, updated_mask, seq_lengths, longest_sequence

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
        # can't set value on leaf variable
        out = masked_tokens.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[full_mask] = x[mask]
        # then move them to their original positions
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        # Values that were masked out are not returned but the values that are still there are returned to the original positions
        return out, full_mask

    def create_exit_seqs(
        self,
        tokens_only_dict: dict[str, Tensor],
        mask_only_dict: dict[str, Tensor],
        token_exit_cfg: dict[str, int] | None,
    ) -> tuple[Tensor | None]:
        """Create the exit sequences and tokens."""
        # Check that tokens_only_dict doesn't contain any mask keys
        assert all(not key.endswith("_mask") for key in tokens_only_dict), (
            "tokens_only_dict should not contain mask keys"
        )
        if token_exit_cfg:
            exit_ids_per_sensor = self.create_token_exit_ids(
                tokens_only_dict, token_exit_cfg
            )
            exit_ids_per_sensor.update(mask_only_dict)
            # Exit ids seqs tells us which layer to exit each token
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(exit_ids_per_sensor)
        else:
            exit_ids_seq = None
        return exit_ids_seq

    def _maybe_get_attn_mask(
        self,
        new_mask: Tensor,
        fast_pass: bool,
    ) -> Tensor | None:
        """Get the attention mask or None if we should pass None to the transformer."""
        if fast_pass or not self.training:
            return None
        else:
            return new_mask

    def add_register_tokens_and_masks(
        self,
        tokens: Tensor,
        attn_mask: Tensor | None,
        processed_register_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Concatenate register tokens to the tokens."""
        batch_size = tokens.shape[0]
        # Expand register tokens to match batch size: [num_register_tokens, embed_dim] -> [batch_size, num_register_tokens, embed_dim]
        if processed_register_tokens is None:
            reg_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            reg_tokens = processed_register_tokens
        # Concatenate register tokens at the beginning: [batch_size, seq_len, embed_dim] -> [batch_size, num_register_tokens + seq_len, embed_dim]
        tokens = torch.cat([reg_tokens, tokens], dim=1)
        if attn_mask is not None:
            # Create mask for register tokens (all True - they should participate in attention)
            reg_mask = torch.ones(
                batch_size,
                self.num_register_tokens,
                dtype=attn_mask.dtype,
                device=attn_mask.device,
            )
            attn_mask = torch.cat([reg_mask, attn_mask], dim=1)
        else:
            reg_mask = None
        return tokens, attn_mask

    def pop_register_tokens(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Pop the register tokens from the tokens."""
        register_tokens = tokens[:, : self.num_register_tokens, :]
        tokens = tokens[:, self.num_register_tokens :, :]
        return tokens, register_tokens

    def get_token_norm_stats(
        self, tokens: Tensor, register_tokens: Tensor
    ) -> dict[str, float]:
        """Get the token norm stats."""
        # Compute norms for register tokens: [batch_size, num_register_tokens]
        register_tokens_norms = torch.norm(register_tokens, dim=2)
        reg_norms_flat = register_tokens_norms.flatten()
        reg_stats = {
            "register_mean": reg_norms_flat.mean().item(),
            "register_min": reg_norms_flat.min().item(),
            "register_max": reg_norms_flat.max().item(),
        }

        # Compute norms for non-register tokens: [batch_size, seq_len]
        nonreg_tokens_norms = torch.norm(tokens, dim=2)
        nonreg_norms_flat = nonreg_tokens_norms.flatten()
        percentiles = [25.0, 75.0, 90.0, 95.0, 99.0]
        nonreg_percentiles = torch.quantile(
            nonreg_norms_flat.float(),
            torch.tensor(
                [p / 100.0 for p in percentiles], device=nonreg_norms_flat.device
            ),
        ).tolist()
        nonreg_stats = {
            "nonregister_mean": nonreg_norms_flat.mean().item(),
            "nonregister_min": nonreg_norms_flat.min().item(),
            "nonregister_max": nonreg_norms_flat.max().item(),
            "nonregister_std": nonreg_norms_flat.std().item(),
            "nonregister_25th": nonreg_percentiles[0],
            "nonregister_75th": nonreg_percentiles[1],
            "nonregister_90th": nonreg_percentiles[2],
            "nonregister_95th": nonreg_percentiles[3],
            "nonregister_99th": nonreg_percentiles[4],
        }

        token_norm_stats = {**reg_stats, **nonreg_stats}
        return token_norm_stats

    def _maybe_remove_masked_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        fast_pass: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks."""
        if fast_pass and not self.use_flash_attn:
            # This is the inference fast pass
            indices = None
            new_mask = None
            seq_lengths = None
            longest_sequence = None
            bool_mask = None
        else:
            bool_mask = mask == TokenVisibility.VISIBLE_ENCODER.value
            tokens, indices, new_mask, seq_lengths, longest_sequence = (
                self.remove_masked_tokens(tokens, bool_mask)
            )
        return tokens, indices, new_mask, seq_lengths, longest_sequence, bool_mask

    def _maybe_add_removed_tokens(
        self,
        tokens: Tensor,
        indices: Tensor,
        mask: Tensor,
        fast_pass: bool,
    ) -> Tensor:
        """Add removed tokens to the tokens and masks."""
        if not fast_pass:
            tokens, _ = self.add_removed_tokens(tokens, indices, mask)
        return tokens

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
        tokens_only_dict, original_masks_dict, sensors_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        # already a no-op but we could remove entirely
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        # exited tokens are just the linear projection
        exited_tokens, _ = self.collapse_and_combine_hwtc(x)

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            tile_patch_size,
            input_res,
        )
        tokens_dict.update(original_masks_dict)
        tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)

        tokens, indices, new_mask, seq_lengths, longest_sequence, bool_mask = (
            self._maybe_remove_masked_tokens(tokens, mask, fast_pass)
        )

        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )

        # Pack x tokens
        if self.use_flash_attn:
            cumulative_lengths = cumulative_seq_offsets(seq_lengths)
            og_shape = tokens.shape
            tokens = self.pack_tokens(tokens, new_mask)
        else:
            cumulative_lengths = None

        attn_mask = self._maybe_get_attn_mask(
            new_mask,
            fast_pass=fast_pass,
        )

        if self.has_register_tokens:
            tokens, attn_mask = self.add_register_tokens_and_masks(tokens, attn_mask)

        # Apply attn with varying encoder depths
        for i_blk, blk in enumerate(self.blocks):
            # Skip the zeroth block because we want to use the exited tokens that don't have encodings as this allows trivial solution of predicting the shared encodings
            if (exit_ids_seq is not None) and (i_blk > 0):
                # this should only ever be called by the target encoder,
                # in a torch.no_grad context
                assert exited_tokens is not None
                # If a token should exit, then we update the exit token with the current token at the same position
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,
                    other=exited_tokens,
                )
            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            # WARNING: THIS MAY CHANGE DEPENDING ON THE ATTENTION IMPLEMENTATION

            tokens = blk(
                x=tokens,
                cumulative_lengths=cumulative_lengths,
                longest_sequence=longest_sequence,
                # we will have to specify k and q lens for cross attention
                attn_mask=attn_mask,
            )

        if self.has_register_tokens:
            tokens, register_tokens = self.pop_register_tokens(tokens)
            token_norm_stats = (
                self.get_token_norm_stats(tokens, register_tokens)
                if self.log_token_norm_stats
                else None
            )
        else:
            token_norm_stats = None

        if self.use_flash_attn:
            tokens = self.unpack_tokens(tokens, new_mask, og_shape)

        if exit_ids_seq is not None:
            # this should only ever be called by the target encoder,
            # in a torch.no_grad context
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=tokens,
                other=exited_tokens,
            )
        # we apply the norm before we add the removed tokens,
        # so that the norm is only computed against "real" tokens
        tokens = self.norm(tokens)
        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        tokens = self._maybe_add_removed_tokens(tokens, indices, new_mask, fast_pass)

        tokens_per_sensor_dict = self.split_and_expand_per_sensor(
            tokens, sensors_to_dims_dict
        )
        # merge original masks and the processed tokens
        tokens_per_sensor_dict.update(original_masks_dict)
        return tokens_per_sensor_dict, token_norm_stats

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
            fast_pass: Whether to always pass None as the mask to the transformer, this enables torch based flash attention, and skips mask construction and sorting

        Returns:
            EmbeddingsAndMasks containing the encoded representations and their masks
        """
        if fast_pass and token_exit_cfg is not None:
            raise ValueError("token_exit_cfg cannot be set when fast_pass is True")

        patchified_tokens_and_masks = self.patch_embeddings.forward(x, tile_patch_size)
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            patchified_tokens_and_masks, token_norm_stats = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                tile_patch_size=tile_patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                fast_pass=fast_pass,
            )
        else:
            token_norm_stats = {}
        output = EmbeddingsAndMasks(**patchified_tokens_and_masks)
        output_dict: dict[str, Any] = {
            "embeddings_and_masks": output,
        }
        if token_norm_stats:
            output_dict["token_norm_stats"] = token_norm_stats

        if not fast_pass:
            output_dict["project_aggregated"] = self.project_and_aggregate(output)
        return output_dict

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().enable_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)

    def enable_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Compiling blocks")
        for block in self.blocks:
            block.enable_compile()


# ---------------------------------------------------------------------------
# LatentPredictorBase
# ---------------------------------------------------------------------------


class LatentPredictorBase(AdaptiveVisionBase):
    """Predictor module that generates predictions from encoded tokens."""

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
        use_flash_attn: bool = False,
        qk_norm: bool = False,
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
            use_flash_attn: Whether to use flash attention
            qk_norm: Whether to apply normalization to Q and K in attention
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
            use_flash_attn=use_flash_attn,
            qk_norm=qk_norm,
            tokenization_config=self.tokenization_config,
        )
        self.learned_channel_embed = learned_channel_embed
        self.random_channel_embed = random_channel_embed
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True
        )
        if output_embedding_size is None:
            output_embedding_size = encoder_embed_dim
        self.output_embedding_size = output_embedding_size
        self.to_output_embed = nn.Linear(
            decoder_embed_dim, output_embedding_size, bias=True
        )
        # THIS is the learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        self.input_norm = nn.LayerNorm(encoder_embed_dim)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.apply(self._init_weights)

    def add_masks(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Replace tokens that should be decoded (TokenVisibility.PREDICTED) with the learnable mask token.

        in a dimension-agnostic way using einops. We assume the final dimension of each token tensor
        is the embedding dimension matching self.mask_token's size.
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
            spatial_dims = x_sensor.shape[
                :-1
            ]  # all dimensions except the last (embedding)
            pattern_input, dim_dict = self._construct_einops_pattern(spatial_dims)

            mask_token_broadcasted = repeat(self.mask_token, pattern_input, **dim_dict)

            # Where kept_mask is True, use the broadcasted mask token
            x_sensor = torch.where(
                kept_mask.unsqueeze(-1).bool(), mask_token_broadcasted, x_sensor
            )

            output_dict[sensor] = x_sensor

        return output_dict

    @staticmethod
    def split_x_y(tokens: Tensor, mask: Tensor) -> tuple[Tensor, ...]:
        """Splits tokens into three groups based on mask values.

        This function:
        1. Sorts tokens according to the mask and gathers them in order.
        2. Chooses tokens to be decoded (x) based on the mask value PREDICTED.
        3. Chooses tokens to be used as context (y) based on the mask value VISIBLE_ENCODER.
        4. Identifies missing tokens (z) based on the mask value ABSENT.
        5. Returns boolean masks for x, y, and z along with indices to revert to the original ordering.

        Args:
            tokens: Tokens to split of shape [B, T, D].
            mask: Mask of shape [B, T].

        Returns:
            tokens_to_decode: Tokens to be decoded of shape [B, X_len, D].
            unmasked_tokens: Tokens to be used as context of shape [B, Y_len, D].
            tokens_to_decode_mask: Binary mask for x tokens of shape [B, X_len].
            unmasked_tokens_mask: Binary mask for y tokens of shape [B, Y_len].
            indices: Indices for restoring the original token ordering of shape [B, T].
            seqlens_tokens_to_decode: Sequence lengths of tokens to decode of shape [B].
            seqlens_unmasked_tokens: Sequence lengths of unmasked tokens of shape [B].
            longest_decoded_sequence: Maximum length of decoded tokens of shape [1].
            longest_unmasked_sequence: Maximum length of unmasked tokens of shape [1].
        """
        # Set Missing Masks to Target Encoder ONLY so that we can have all unused tokens in the middle
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

        seqlens_unmasked_tokens = binarized_online_encoder_mask.sum(dim=-1)
        longest_unmasked_sequence = seqlens_unmasked_tokens.max()
        seqlens_tokens_to_decode = binarized_decoder_mask.sum(dim=-1)
        longest_decoded_sequence = seqlens_tokens_to_decode.max()

        # the y mask is going to be used to determine which of the y values take. True values
        # take part in the attention (we don't take the inverse here, unlike in the decoder)
        tokens_to_decode = tokens[:, :longest_decoded_sequence]
        tokens_to_decode_mask = binarized_decoder_mask[
            :, :longest_decoded_sequence
        ].to(org_mask_dtype)

        unmasked_tokens = tokens[:, -longest_unmasked_sequence:]
        # the x_mask is just going to be used in the reconstruction, to know which
        # x tokens to add back into the token list.
        unmasked_tokens_mask = binarized_online_encoder_mask[
            :, -longest_unmasked_sequence:
        ].to(org_mask_dtype)

        return (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            longest_decoded_sequence,
            longest_unmasked_sequence,
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

        The token masks zero out positions which are not used/needed,
        and the final scatter step re-applies the original ordering tracked in 'indices'.

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
        # Get dimensions
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

    def is_any_data_to_be_decoded(self, sensor_mask: Tensor) -> bool:
        """Check if any data is to be decoded for a given sensor."""
        return (TokenVisibility.PREDICTED.value == sensor_mask).any()

    def enable_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().enable_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)


# ---------------------------------------------------------------------------
# LatentPredictor
# ---------------------------------------------------------------------------


class LatentPredictor(LatentPredictorBase):
    """Predictor module that generates predictions from encoded tokens."""

    cross_attn = True

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
            og_shape_tokens_to_decode = tokens_to_decode.shape
            tokens_to_decode = self.pack_tokens(
                tokens_to_decode, tokens_to_decode_mask.bool()
            )
            og_shape_unmasked_tokens = unmasked_tokens.shape
            unmasked_tokens = self.pack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool()
            )
            cumulative_lengths_tokens_to_decode = cumulative_seq_offsets(
                seqlens_tokens_to_decode
            )
            cumulative_lengths_unmasked_tokens = cumulative_seq_offsets(
                seqlens_unmasked_tokens
            )
        else:
            cumulative_lengths_tokens_to_decode = None
            cumulative_lengths_unmasked_tokens = None

        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=unmasked_tokens,
                attn_mask=(
                    unmasked_tokens_mask.bool() if not self.use_flash_attn else None
                ),
                cumulative_lengths_q=cumulative_lengths_tokens_to_decode,
                cumulative_lengths_k=cumulative_lengths_unmasked_tokens,
                longest_sequence_q=longest_decoded_sequence,
                longest_sequence_k=longest_unmasked_sequence,
            )

        if self.use_flash_attn:
            tokens_to_decode = self.unpack_tokens(
                tokens_to_decode,
                tokens_to_decode_mask.bool(),
                og_shape_tokens_to_decode,
            )
            unmasked_tokens = self.unpack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool(), og_shape_unmasked_tokens
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
        decoder_embedded_dict = x.as_dict(return_none=False)
        # Apply Input Norms and encoder to decoder embeds to each sensor
        available_sensors = x.sensors
        sensors_list = sensors_to_process(
            available_sensors, self.supported_sensor_labels
        )
        for sensor in sensors_list:
            x_sensor = getattr(x, sensor)
            # Although, we do not account for missing tokens both proj and normalize are on token dimension so there is no mixing with real tokens
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
            # patchify masked data
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
# VisionEncoderConfig
# ---------------------------------------------------------------------------


@dataclass
class VisionEncoderConfig(Config):
    """Configuration for the VisionEncoder."""

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
    num_register_tokens: int = 0
    learned_channel_embed: bool = True
    random_channel_embed: bool = False
    num_projection_layers: int = 1
    aggregate_then_project: bool = True
    use_flash_attn: bool = False
    frozen_patch_embeddings: bool = False
    qk_norm: bool = False
    log_token_norm_stats: bool = False
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

    @property
    def supported_sensors(self) -> list[SensorSpec]:
        """Get the supported sensors."""
        return specs_from_labels(self.supported_sensor_labels)

    def build(self) -> "VisionEncoder":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_sensor_labels is replaced by supported_sensors
        kwargs.pop("supported_sensor_labels")
        kwargs["supported_sensors"] = self.supported_sensors
        logger.info(f"VisionEncoder kwargs: {kwargs}")
        return VisionEncoder(**kwargs)


# ---------------------------------------------------------------------------
# LatentPredictorConfig
# ---------------------------------------------------------------------------


@dataclass
class LatentPredictorConfig(Config):
    """Configuration for the LatentPredictor."""

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
    use_flash_attn: bool = False
    qk_norm: bool = False
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

    @property
    def supported_sensors(self) -> list[SensorSpec]:
        """Get the supported sensors."""
        return specs_from_labels(self.supported_sensor_labels)

    def build(self) -> "LatentPredictorBase":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_sensor_labels is replaced by supported_sensors
        kwargs.pop("supported_sensor_labels")
        kwargs["supported_sensors"] = self.supported_sensors
        logger.info(f"LatentPredictor kwargs: {kwargs}")
        return LatentPredictor(**kwargs)
