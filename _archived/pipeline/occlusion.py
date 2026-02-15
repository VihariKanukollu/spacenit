"""Occlusion (masking) policies for SpaceNit training."""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from class_registry import ClassRegistry
from einops import rearrange, repeat

from spacenit.aliases import NdTensor
from spacenit.arch.band_tokenization import TokenizationConfig
from spacenit.ingestion.sensors import ABSENT_INDICATOR, SensorRegistry, SensorSpec
from spacenit.settings import Config
from spacenit.structures import GeoSample, MaskedGeoSample, TokenVisibility

logger = logging.getLogger(__name__)

# all bandset indices should be tuples of (sensor, bandset_idx) so we can create
# a power set of these combinations from it
ALL_BANDSET_IDXS: list[tuple[str, int]] = []
for _sensor in SensorRegistry.all_specs():
    for _bandset_idx in range(_sensor.group_count):
        ALL_BANDSET_IDXS.append((_sensor.label, _bandset_idx))


class OcclusionPolicy:
    """Abstract base class for occlusion (masking) policies.

    Be sure to implement apply_mask in subclasses.
    """

    tokenization_config: TokenizationConfig | None = None

    @property
    def name(self) -> str:
        """Return the name of the occlusion policy."""
        return self.__class__.__name__.replace("OcclusionPolicy", "").lower()

    @property
    def encode_ratio(self) -> float:
        """Return the encode ratio."""
        if not hasattr(self, "_encode_ratio"):
            raise AttributeError("Encode ratio not set")
        return self._encode_ratio

    @property
    def decode_ratio(self) -> float:
        """Return the decode ratio."""
        if not hasattr(self, "_decode_ratio"):
            raise AttributeError("Decode ratio not set")
        return self._decode_ratio

    def _get_num_bandsets(self, sensor_name: str) -> int:
        """Get the number of bandsets for a sensor, using tokenization config if available."""
        if self.tokenization_config is not None:
            return self.tokenization_config.group_count_for(sensor_name)
        return SensorRegistry.get(sensor_name).group_count

    def _get_bandset_indices(self, sensor_name: str) -> list[list[int]]:
        """Get the bandset indices for a sensor, using tokenization config if available."""
        if self.tokenization_config is not None:
            return self.tokenization_config.group_indices_for(sensor_name)
        return SensorRegistry.get(sensor_name).group_indices()

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply occlusion to the input data.

        Args:
            batch: Input data of type GeoSample
            patch_size: Optional patch size for spatial occlusion policies
            **kwargs: Additional arguments for occlusion
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_missing_mask(
        self, instance: torch.Tensor, sensor: SensorSpec, mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the missing mask for the input data."""
        missing_mask = mask.new_zeros(mask.shape, dtype=torch.bool)
        bandset_indices = self._get_bandset_indices(sensor.label)
        for i, band_set_indices in enumerate(bandset_indices):
            instance_band_set = instance[..., band_set_indices]
            missing_mask_band_set = instance_band_set == ABSENT_INDICATOR
            missing_mask_band_set_any = missing_mask_band_set.any(dim=-1)
            # If any band in the band set is missing, set the whole band set to missing
            missing_mask[..., i] = missing_mask_band_set_any
        return missing_mask

    def fill_mask_with_missing_values(
        self, instance: torch.Tensor, mask: torch.Tensor, sensor: SensorSpec
    ) -> torch.Tensor:
        """Apply a missing mask to the input data."""
        missing_mask = self.get_missing_mask(instance, sensor, mask)
        # If we are changing the mask, we need to clone it as it may be a view of a mask used by different sensors
        if missing_mask.any():
            output_mask = mask.clone()
            output_mask[missing_mask] = TokenVisibility.ABSENT.value
        else:
            output_mask = mask
        return output_mask

    def _create_random_mask(
        self,
        sensor: SensorSpec,
        shape: torch.Size,
        patch_size_at_16: int,
        device: torch.device | None = None,
        encode_ratio: float | None = None,
        decode_ratio: float | None = None,
    ) -> NdTensor:
        mask_shape = list(shape)
        mask_shape[-1] = self._get_num_bandsets(sensor.label)
        if sensor.varies_in_space:
            patch_size = patch_size_at_16 * sensor.tile_size_multiplier
            mask_shape[1] //= patch_size
            mask_shape[2] //= patch_size

        if sensor.varies_in_space or sensor.has_temporal_axis:
            b = shape[0]
            num_tokens = math.prod(mask_shape[1:])
        else:
            num_tokens = math.prod(mask_shape[:-1])

        if encode_ratio is None:
            encode_ratio = self.encode_ratio
        if decode_ratio is None:
            decode_ratio = self.decode_ratio

        encode_tokens = int(num_tokens * encode_ratio)
        decode_tokens = int(num_tokens * decode_ratio)
        target_tokens = int(num_tokens - (encode_tokens + decode_tokens))
        flat_mask_tokens = torch.cat(
            [
                torch.full(
                    (encode_tokens,), TokenVisibility.VISIBLE_ENCODER.value, device=device
                ),
                torch.full((decode_tokens,), TokenVisibility.PREDICTED.value, device=device),
                torch.full(
                    (target_tokens,), TokenVisibility.TARGET_ONLY.value, device=device
                ),
            ]
        )

        if sensor.varies_in_space or sensor.has_temporal_axis:
            masks = [
                flat_mask_tokens[torch.randperm(num_tokens, device=device)]
                for i in range(b)
            ]
            flat_mask_tokens = torch.stack(masks)
        else:
            flat_mask_tokens = flat_mask_tokens[
                torch.randperm(num_tokens, device=device)
            ]

        mask = flat_mask_tokens.view(*mask_shape)
        if sensor.varies_in_space:
            mask = repeat(
                mask, "b h w ... -> b (h hp) (w wp) ...", hp=patch_size, wp=patch_size
            )

        return mask

    def _random_fill_unmasked(
        self,
        mask: torch.Tensor,
        sensor: SensorSpec,
        patch_size_at_16: int,
        encode_ratio: float | None = None,
        decode_ratio: float | None = None,
    ) -> NdTensor:
        """This function assumes B=1."""
        assert mask.shape[0] == 1, (
            f"_random_fill_unmasked does not support B != 1, got input shape {mask.shape}"
        )
        device = mask.device
        if sensor.varies_in_space:
            patch_size = patch_size_at_16 * sensor.tile_size_multiplier
            # the first two dimensions are spatial; lets turn them
            # from h, w to p_h, p_w
            mask = mask[:, 0::patch_size, 0::patch_size]

        original_shape = mask.shape
        # this only works because we assume B = 1
        flat_mask = mask.flatten()  # N tokens
        not_missing_tokens = flat_mask != TokenVisibility.ABSENT.value
        num_not_missing_tokens = sum(not_missing_tokens)

        if encode_ratio is None:
            encode_ratio = self.encode_ratio
        if decode_ratio is None:
            decode_ratio = self.decode_ratio

        if num_not_missing_tokens == 1:
            encode_tokens = 1
            decode_tokens = 0
        else:
            encode_tokens = int(num_not_missing_tokens * encode_ratio)
            decode_tokens = int(num_not_missing_tokens * decode_ratio)

        target_tokens = int(num_not_missing_tokens - (encode_tokens + decode_tokens))
        flat_mask_tokens = torch.cat(
            [
                torch.full(
                    (encode_tokens,), TokenVisibility.VISIBLE_ENCODER.value, device=device
                ),
                torch.full((decode_tokens,), TokenVisibility.PREDICTED.value, device=device),
                torch.full(
                    (target_tokens,), TokenVisibility.TARGET_ONLY.value, device=device
                ),
            ]
        )

        flat_mask_tokens = flat_mask_tokens[
            torch.randperm(num_not_missing_tokens, device=device)
        ]
        flat_mask[not_missing_tokens] = flat_mask_tokens
        mask = flat_mask.view(*original_shape)
        if sensor.varies_in_space:
            mask = repeat(
                mask, "b h w ... -> b (h hp) (w wp) ...", hp=patch_size, wp=patch_size
            )

        return mask


OCCLUSION_POLICY_REGISTRY = ClassRegistry[OcclusionPolicy]()


@OCCLUSION_POLICY_REGISTRY.register("time")
class TemporalOcclusionPolicy(OcclusionPolicy):
    """Time-structured random occlusion of the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def _create_temporal_mask(
        self,
        shape: torch.Size,
        timesteps_with_at_least_one_sensor: torch.Tensor,
        device: torch.device | None = None,
    ) -> NdTensor:
        b = shape[0]
        t = shape[-2]
        # timesteps with at least one sensor are the only ones we can put as either encoder and decoder
        present_t = timesteps_with_at_least_one_sensor.shape[0]  # across all samples
        assert present_t >= 3
        logger.info(f"Present timesteps: {present_t}")
        encode_times = max(int(self.encode_ratio * present_t), 1)
        decode_times = max(int(self.decode_ratio * present_t), 1)
        target_times = present_t - encode_times - decode_times
        logger.info(
            f"Encode times: {encode_times}, Decode times: {decode_times}, Target times: {target_times}"
        )
        # Create mask values only for the encodable timesteps
        encodable_mask_values = torch.cat(
            [
                torch.full(
                    (encode_times,), TokenVisibility.VISIBLE_ENCODER.value, device=device
                ),
                torch.full((decode_times,), TokenVisibility.PREDICTED.value, device=device),
                torch.full(
                    (target_times,), TokenVisibility.TARGET_ONLY.value, device=device
                ),
            ]
        )

        # Create masks for each sample in the batch
        masks = [
            torch.full(
                (t,), TokenVisibility.TARGET_ONLY.value, device=device
            ).index_put_(
                (timesteps_with_at_least_one_sensor,),
                encodable_mask_values[torch.randperm(present_t, device=device)],
            )
            for _ in range(b)
        ]

        mask = torch.stack(masks)
        return mask

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply temporal occlusion to the input data.

        Occlusion happens temporally, with whole time steps having the same mask.
        Non-temporal data is randomly masked.

        Args:
            batch: Input data of type GeoSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments

        Returns:
            MaskedGeoSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for temporal occlusion")
        output_dict: dict[str, NdTensor | None] = {}
        temporal_mask = None
        timesteps_with_at_least_one_sensor = (
            batch.occupied_timesteps
        )
        num_valid_timesteps = timesteps_with_at_least_one_sensor.shape[0]
        for sensor_name in batch.present_keys:
            instance = getattr(batch, sensor_name)
            if instance is None:
                # set instance and mask to None
                output_dict[sensor_name] = None
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = None
            else:
                if sensor_name == "timestamps":
                    output_dict[sensor_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None

                sensor = SensorRegistry.get(sensor_name)
                shape = instance.shape
                if not sensor.has_temporal_axis or num_valid_timesteps < 3:
                    mask = self._create_random_mask(sensor, shape, patch_size, device)
                else:
                    if temporal_mask is None:
                        logger.info(
                            f"Creating temporal mask for sensor {sensor.label}"
                        )
                        temporal_mask = self._create_temporal_mask(
                            shape, timesteps_with_at_least_one_sensor, device
                        )
                    b_s = self._get_num_bandsets(sensor.label)
                    b, h, w = list(shape[:-2]) + [1] * (3 - len(shape[:-2]))
                    # Repeat shares a view of the temporal masks so if we don't clone future changes may propagate across sensors
                    mask = repeat(
                        temporal_mask, "b t -> b h w t b_s", h=h, w=w, b_s=b_s
                    )
                    mask = mask.view(*shape[:-1], b_s).clone()
                # After setting up encoder and decoder masks, fill in missing values

                mask = self.fill_mask_with_missing_values(instance, mask, sensor)
                output_dict[sensor_name] = instance
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = mask
        return MaskedGeoSample(**output_dict)


@OCCLUSION_POLICY_REGISTRY.register("space")
class SpatialOcclusionPolicy(OcclusionPolicy):
    """Spatially structured random occlusion of the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def _create_patch_spatial_mask(
        self,
        sensor: SensorSpec,
        shape: torch.Size,
        patch_size_at_16: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Create a h_p x w_p spatial mask.

        Here, h_p and w_p are the number of patches along height and width dimension
        respectively.

        The mask computed here is sensor-agnostic, but we still expect a specific
        sensor to be passed since it will be used to compute h_p/w_p. The mask will
        then need to be resized using _resize_spatial_mask_for_sensor to the
        sensor's patch size.

        Args:
            sensor: the sensor we are using to compute h_p/w_p.
            shape: the shape of the image for that sensor.
            patch_size_at_16: the patch size measured in 10 m/pixel pixels.
            device: the device to use.
        """
        if not sensor.varies_in_space:
            raise ValueError("Non-spatial sensor {sensor}")

        b, h, w = shape[:3]

        patch_size = patch_size_at_16 * sensor.tile_size_multiplier
        assert (h % patch_size == 0) and (w % patch_size == 0)
        h_p = h // patch_size
        w_p = w // patch_size

        patches = h_p * w_p
        encode_patches = int(self.encode_ratio * patches)
        decode_patches = int(self.decode_ratio * patches)
        target_patches = patches - encode_patches - decode_patches

        flat_mask = torch.cat(
            [
                torch.full(
                    (encode_patches,), TokenVisibility.VISIBLE_ENCODER.value, device=device
                ),
                torch.full((decode_patches,), TokenVisibility.PREDICTED.value, device=device),
                torch.full(
                    (target_patches,),
                    TokenVisibility.TARGET_ONLY.value,
                    device=device,
                ),
            ]
        )

        masks = [flat_mask[torch.randperm(patches, device=device)] for i in range(b)]
        random_batch_mask = torch.stack(masks)
        return rearrange(random_batch_mask, "b (h w) -> b h w", h=h_p, w=w_p)

    def _resize_spatial_mask_for_sensor(
        self,
        patch_mask: torch.Tensor,
        sensor: SensorSpec,
        patch_size_at_16: int,
    ) -> NdTensor:
        """Resize the mask computed by _create_patch_spatial_mask for the given sensor.

        Args:
            patch_mask: the mask computed by _create_patch_spatial_mask.
            sensor: the sensor to compute the mask for.
            patch_size_at_16: the patch size measured in 10 m/pixel pixels.
        """
        if not sensor.varies_in_space:
            raise ValueError("Non-spatial sensor {sensor}")

        patch_size = patch_size_at_16 * sensor.tile_size_multiplier
        mask = repeat(
            patch_mask, "b h w -> b (h hps) (w wps)", hps=patch_size, wps=patch_size
        )
        return mask

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply spatial occlusion to the input data.

        Occlusion happens in patchified form, with whole patches having the same mask.
        Non-spatial data is randomly masked.

        Args:
            batch: Input data of type GeoSample
            patch_size: patch size applied to sample, at a coverage_scale == 16
            **kwargs: Additional arguments

        Returns:
            MaskedGeoSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for spatial occlusion")
        output_dict: dict[str, NdTensor | None] = {}
        patch_spatial_mask = None
        # Same spatial mask for all sensors
        for sensor_name in batch.present_keys:
            instance = getattr(batch, sensor_name)
            if instance is None:
                # set instance and mask to None
                output_dict[sensor_name] = None
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = None
                continue

            if sensor_name == "timestamps":
                output_dict[sensor_name] = instance
                continue

            if isinstance(instance, torch.Tensor):
                device: torch.device | None = instance.device
            else:
                device = None

            sensor = SensorRegistry.get(sensor_name)
            shape = instance.shape
            if not sensor.varies_in_space:
                logger.warning(
                    f"Sensor {sensor.label} is not spatial, random occlusion will be applied"
                )
                mask = self._create_random_mask(sensor, shape, patch_size, device)
            else:
                if patch_spatial_mask is None:
                    logger.info(f"Creating spatial mask for sensor {sensor.label}")
                    patch_spatial_mask = self._create_patch_spatial_mask(
                        sensor, shape, patch_size, device
                    )
                resized_spatial_mask = self._resize_spatial_mask_for_sensor(
                    patch_spatial_mask, sensor, patch_size
                )

                if resized_spatial_mask.shape[0:3] != shape[0:3]:
                    raise ValueError(
                        f"Mismatched shapes for {sensor.label}: "
                        f"computed mask {resized_spatial_mask.shape} but image shape is {shape}"
                    )

                if len(shape) == 5:
                    t = shape[-2]
                else:
                    t = 1
                b_s = self._get_num_bandsets(sensor.label)
                # Mask is a view of the spatial mask, so changes to mask will change spatial_mask
                mask = repeat(resized_spatial_mask, "... -> ... t b_s", t=t, b_s=b_s)
                mask = mask.view(*shape[:-1], b_s).clone()
            mask = self.fill_mask_with_missing_values(instance, mask, sensor)

            # Keep data as is
            output_dict[sensor_name] = instance
            output_dict[
                MaskedGeoSample.mask_field_for(sensor_name)
            ] = mask
        return MaskedGeoSample(**output_dict)


@OCCLUSION_POLICY_REGISTRY.register("space_time")
class SpatioTemporalOcclusionPolicy(OcclusionPolicy):
    """Randomly select spatial or temporal occlusion and apply it to the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

        self.space_strategy = SpatialOcclusionPolicy(encode_ratio, decode_ratio)
        self.time_strategy = TemporalOcclusionPolicy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply spatial or temporal occlusion to the input data."""
        has_enough_timesteps = batch.active_temporal_length >= 3

        if not has_enough_timesteps:
            logger.debug(f"Valid time: {batch.active_temporal_length}, Time: {batch.temporal_length}")
        if (np.random.random() < 0.5) or (not has_enough_timesteps):
            logger.info("Applying spatial occlusion")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying temporal occlusion")
            return self.time_strategy.apply_mask(batch, patch_size, **kwargs)


@OCCLUSION_POLICY_REGISTRY.register("random_space")
class RandomSpatialOcclusionPolicy(OcclusionPolicy):
    """Randomly select spatial or random occlusion."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

        self.random_strategy = RandomOcclusionPolicy(encode_ratio, decode_ratio)
        self.space_strategy = SpatialOcclusionPolicy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply spatial or random occlusion to the input data."""
        if np.random.random() < 0.5:
            logger.info("Applying spatial occlusion")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying random occlusion")
            return self.random_strategy.apply_mask(batch, patch_size, **kwargs)


class CrossSensorOcclusionPolicy(OcclusionPolicy):
    """Abstract class for occlusion policies that select separate sets of bandsets to encode and decode on top of another policy."""

    def __init__(
        self,
        strategy: OcclusionPolicy,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int | None = None,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_sensors: list[str] | None = None,
    ) -> None:
        """Initialize the occlusion policy.

        Args:
            strategy: The base occlusion policy to apply before cross-sensor occlusion.
            encode_ratio: Ratio of tokens to encode (default: 0.5). Used by the base strategy.
            decode_ratio: Ratio of tokens to decode (default: 0.5). Used by the base strategy.
            allow_encoding_decoding_same_bandset: If True, allows the same bandset to be both
                encoded and decoded. If False (default), encoded and decoded bandsets are disjoint.
            min_encoded_bandsets: Minimum number of bandsets to encode per sample.
            max_encoded_bandsets: Maximum number of bandsets to encode per sample.
            min_decoded_bandsets: Minimum number of bandsets to decode per sample.
            max_decoded_bandsets: Maximum number of bandsets to decode per sample.
            only_decode_sensors: List of sensor names that should only be used for decoding,
                never for encoding. Empty list by default (all sensors can be encoded).
        """
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.strategy = strategy
        self.allow_encoding_decoding_same_bandset = allow_encoding_decoding_same_bandset
        if min_encoded_bandsets is None:
            assert max_encoded_bandsets is None, (
                "max_encoded_bandsets must be set if min_encoded_bandsets is set"
            )
        else:
            assert min_encoded_bandsets > 1, (
                "min_encoded_bandsets must be greater than 1 so that we don't only "
                "encode a sensor that is randomly masked on batch dimension ie latlon"
            )
        self.min_encoded_bandsets = min_encoded_bandsets
        self.max_encoded_bandsets = max_encoded_bandsets
        self.min_decoded_bandsets = min_decoded_bandsets
        self.max_decoded_bandsets = max_decoded_bandsets
        self.only_decode_sensors = only_decode_sensors or []

    def get_sample_present_sensor_bandsets(
        self, batch: MaskedGeoSample
    ) -> list[list[tuple[str, int]]]:
        """Get the sensors that are present for each sample."""
        masked_sample_dict = batch.to_dict(return_none=False)
        batch_size = batch.timestamps.shape[0]
        present_sensor_bandsets: list[list[tuple[str, int]]] = [
            [] for _ in range(batch_size)
        ]
        for sensor in batch.present_keys:
            if sensor == "timestamps":
                continue
            sensor_mask_name = MaskedGeoSample.mask_field_for(sensor)
            sensor_mask = masked_sample_dict[sensor_mask_name]
            missing_values_mask = sensor_mask == TokenVisibility.ABSENT.value
            # Find the samples where the sensor is completely missing
            is_sensor_completely_missing_for_samples = torch.all(
                missing_values_mask.view(batch_size, -1), dim=1
            )
            is_sensor_present_for_samples = (
                ~is_sensor_completely_missing_for_samples
            )
            num_bandsets = sensor_mask.shape[-1]

            present_sample_indices = torch.where(is_sensor_present_for_samples)[0]
            for sample_idx in present_sample_indices:
                sample_idx = sample_idx.item()
                for bandset_idx in range(num_bandsets):
                    # check if that sensor bandset has any encoded tokens
                    is_any_tokens_encoded_for_sample = (
                        torch.sum(
                            sensor_mask[sample_idx, ..., bandset_idx]
                            == TokenVisibility.VISIBLE_ENCODER.value
                        )
                        > 0
                    )
                    # only say something is present if it has any encoded tokens
                    if (
                        not is_any_tokens_encoded_for_sample
                        and sensor not in self.only_decode_sensors
                    ):
                        continue
                    present_sensor_bandsets[sample_idx].append(
                        (sensor, bandset_idx)
                    )
        return present_sensor_bandsets

    def select_encoded_decoded_bandsets(
        self, present_sensor_bandsets: list[list[tuple[str, int]]]
    ) -> list[tuple[set[tuple[str, int]], set[tuple[str, int]]]]:
        """Select the encoded and decoded bandsets for each sample."""
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ] = []
        for sample_idx in range(len(present_sensor_bandsets)):
            present_sensor_bandsets_for_sample = present_sensor_bandsets[
                sample_idx
            ]
            # If there is only one sensor, we only encode not decode
            if len(present_sensor_bandsets_for_sample) == 1:
                encoded_bandset_idxs = set(present_sensor_bandsets_for_sample)
                decoded_bandset_idxs = set()
            # If there are two sensors, we encode one and decode the other
            elif len(present_sensor_bandsets_for_sample) == 2:
                encoded_bandset_idxs = set([present_sensor_bandsets_for_sample[0]])
                decoded_bandset_idxs = set([present_sensor_bandsets_for_sample[1]])
            # If there are more than two sensors, we randomly select some to encode and the rest to decode
            else:
                # Select Indices to Encode
                num_present_sensors = len(present_sensor_bandsets_for_sample)
                encodable_sensor_bandsets = [
                    sensor_bandset
                    for sensor_bandset in present_sensor_bandsets_for_sample
                    if sensor_bandset[0] not in self.only_decode_sensors
                ]
                num_encodable_sensor_bandsets = len(encodable_sensor_bandsets)
                upper_limit = num_encodable_sensor_bandsets
                if not self.allow_encoding_decoding_same_bandset:
                    upper_limit -= 1
                if self.max_encoded_bandsets is None:
                    max_encoded_bandsets = upper_limit
                else:
                    max_encoded_bandsets = min(self.max_encoded_bandsets, upper_limit)

                if self.min_encoded_bandsets is None:
                    min_encoded_bandsets = num_encodable_sensor_bandsets
                else:
                    min_encoded_bandsets = min(
                        self.min_encoded_bandsets, num_encodable_sensor_bandsets
                    )
                # Ensure min is less than max
                min_encoded_bandsets = min(min_encoded_bandsets, max_encoded_bandsets)
                num_bandsets_to_encode = np.random.randint(
                    min_encoded_bandsets, max_encoded_bandsets + 1
                )
                encoded_idxs = np.random.choice(
                    len(encodable_sensor_bandsets),
                    size=num_bandsets_to_encode,
                    replace=False,
                )
                encoded_bandset_idxs = set(
                    [encodable_sensor_bandsets[i] for i in encoded_idxs]
                )
                # Select Indices to Decode
                min_decoded_bandsets = min(
                    self.min_decoded_bandsets or 1, num_present_sensors
                )
                max_decoded_bandsets = min(
                    self.max_decoded_bandsets or num_present_sensors,
                    num_present_sensors,
                )
                if self.allow_encoding_decoding_same_bandset:
                    num_decoded_bandsets = np.random.randint(
                        min_decoded_bandsets, max_decoded_bandsets + 1
                    )
                    decoded_idxs = np.random.choice(
                        len(present_sensor_bandsets_for_sample),
                        size=num_decoded_bandsets,
                        replace=False,
                    )
                    decoded_bandset_idxs = set(
                        [
                            present_sensor_bandsets_for_sample[i]
                            for i in decoded_idxs
                        ]
                    )
                else:
                    available_decoded_bandset_idxs = list(
                        set(present_sensor_bandsets_for_sample)
                        - encoded_bandset_idxs
                    )
                    num_decoded_bandsets = len(available_decoded_bandset_idxs)
                    min_decoded_bandsets = min(
                        min_decoded_bandsets, num_decoded_bandsets
                    )
                    max_decoded_bandsets = min(
                        max_decoded_bandsets, num_decoded_bandsets
                    )
                    decoded_idxs = np.random.choice(
                        len(available_decoded_bandset_idxs),
                        size=num_decoded_bandsets,
                        replace=False,
                    )
                    decoded_bandset_idxs = set(
                        [available_decoded_bandset_idxs[i] for i in decoded_idxs]
                    )
            encoded_decoded_bandsets.append(
                (encoded_bandset_idxs, decoded_bandset_idxs)
            )
        return encoded_decoded_bandsets

    def override_strategy_mask(self, sensor_spec: SensorSpec) -> bool:
        """Override the mask for a sensor depending on the strategy being cross-sensor masked."""
        return False

    def apply_bandset_mask_rules(
        self,
        masked_batch: MaskedGeoSample,
        encoded_decoded_bandsets: list[
            tuple[set[tuple[str, int]], set[tuple[str, int]]]
        ],
        present_sensor_bandsets: list[list[tuple[str, int]]],
        patch_size: int,
    ) -> MaskedGeoSample:
        """Compute masks for each band set based on the encode and decode selections.

        Args:
            masked_batch: The masked batch to apply the mask to.
            encoded_decoded_bandsets: The encoded and decoded bandsets for each sample.
            present_sensor_bandsets: The present sensors and bandsets for each sample.
            patch_size: The patch size being applied

        Returns:
            The masked batch with the masks applied.
        """
        masked_batch_dict = masked_batch.to_dict(return_none=False)
        num_encoded: None | torch.Tensor = None
        num_decoded: None | torch.Tensor = None
        for sensor in masked_batch.present_keys:
            if sensor == "timestamps":
                continue
            masked_sensor_name = MaskedGeoSample.mask_field_for(sensor)
            sensor_spec = SensorRegistry.get(sensor)
            sensor_mask = masked_batch_dict[masked_sensor_name]
            out_sensor_mask = sensor_mask.clone()
            num_bandsets = sensor_mask.shape[-1]

            for sample_idx in range(masked_batch.timestamps.shape[0]):
                encoded_bandset_idxs, decoded_bandset_idxs = encoded_decoded_bandsets[
                    sample_idx
                ]
                available_sensors = [
                    sensor_bandset[0]
                    for sensor_bandset in present_sensor_bandsets[sample_idx]
                ]
                if sensor not in available_sensors:
                    logger.debug(
                        f"Sensor {sensor} not present for sample {sample_idx}"
                    )
                    continue

                for bandset_idx in range(num_bandsets):
                    is_encoded = (sensor, bandset_idx) in encoded_bandset_idxs
                    is_decoded = (sensor, bandset_idx) in decoded_bandset_idxs

                    if self.override_strategy_mask(sensor_spec):
                        if is_encoded:
                            forced_mask_value = TokenVisibility.VISIBLE_ENCODER.value
                        elif is_decoded:
                            forced_mask_value = TokenVisibility.PREDICTED.value
                        else:
                            continue
                        logger.debug(
                            f"Setting {sensor} bandset {bandset_idx} to {forced_mask_value}"
                        )
                        not_missing_mask = (
                            sensor_mask[sample_idx, ..., bandset_idx]
                            != TokenVisibility.ABSENT.value
                        )
                        out_sensor_mask[sample_idx, ..., bandset_idx] = torch.where(
                            not_missing_mask,
                            forced_mask_value,
                            sensor_mask[sample_idx, ..., bandset_idx],
                        )
                        continue

                    if not is_encoded:
                        # Suppress all encoded values for a not encoded bandset
                        online_encoder_mask = (
                            sensor_mask[sample_idx, ..., bandset_idx]
                            == TokenVisibility.VISIBLE_ENCODER.value
                        )

                        out_sensor_mask[sample_idx, ..., bandset_idx] = torch.where(
                            online_encoder_mask.clone(),
                            TokenVisibility.TARGET_ONLY.value,
                            sensor_mask[sample_idx, ..., bandset_idx],
                        )
                        continue

                    if not is_decoded:
                        predicted_mask = (
                            sensor_mask[sample_idx, ..., bandset_idx]
                            == TokenVisibility.PREDICTED.value
                        )

                        out_sensor_mask[sample_idx, ..., bandset_idx] = torch.where(
                            predicted_mask,
                            TokenVisibility.TARGET_ONLY.value,
                            sensor_mask[sample_idx, ..., bandset_idx],
                        )
            # check we have more than 0 encoded and decoded tokens.
            flat_mask = torch.flatten(out_sensor_mask, start_dim=1)
            encoded_for_sensor = (flat_mask == TokenVisibility.VISIBLE_ENCODER.value).sum(
                dim=-1
            )
            decoded_for_sensor = (flat_mask == TokenVisibility.PREDICTED.value).sum(dim=-1)
            if num_encoded is None:
                num_encoded = encoded_for_sensor
            else:
                num_encoded += encoded_for_sensor
            if num_decoded is None:
                num_decoded = decoded_for_sensor
            else:
                num_decoded += decoded_for_sensor
            masked_batch_dict[masked_sensor_name] = out_sensor_mask

        no_encoded_indices = torch.argwhere(num_encoded == 0)
        no_decoded_indices = torch.argwhere(num_decoded == 0)
        for i in no_encoded_indices:
            for key, val in masked_batch_dict.items():
                if key.endswith("_mask"):
                    sensor_mask = val[i]
                    sensor_name = MaskedGeoSample.data_field_for(key)
                    if sensor_name in self.only_decode_sensors:
                        continue
                    sensor_spec = SensorRegistry.get(sensor_name)
                    masked_batch_dict[key][i] = self._random_fill_unmasked(
                        sensor_mask, sensor_spec, patch_size
                    )
        for i in no_decoded_indices:
            for key, val in masked_batch_dict.items():
                if key.endswith("_mask"):
                    sensor_mask = val[i]
                    sensor_name = MaskedGeoSample.data_field_for(key)
                    if sensor_name in self.only_decode_sensors:
                        continue
                    sensor_spec = SensorRegistry.get(sensor_name)
                    masked_batch_dict[key][i] = self._random_fill_unmasked(
                        sensor_mask, sensor_spec, patch_size
                    )
        masked_batch = MaskedGeoSample(**masked_batch_dict)

        return masked_batch

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply cross-sensor occlusion to the input data."""
        if patch_size is None:
            raise ValueError("patch_size must be provided for cross-sensor occlusion")

        masked_sample = self.strategy.apply_mask(batch, patch_size, **kwargs)
        present_sensor_bandsets = self.get_sample_present_sensor_bandsets(
            masked_sample
        )
        encoded_decoded_bandsets = self.select_encoded_decoded_bandsets(
            present_sensor_bandsets
        )
        masked_sample = self.apply_bandset_mask_rules(
            masked_sample,
            encoded_decoded_bandsets,
            present_sensor_bandsets,
            patch_size,
        )

        return masked_sample


@OCCLUSION_POLICY_REGISTRY.register("cross_sensor_space")
class CrossSensorSpatialOcclusionPolicy(CrossSensorOcclusionPolicy):
    """Randomly select a sensor and apply spatial occlusion to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_sensors: list[str] | None = None,
    ) -> None:
        """Initialize the occlusion policy."""
        space_strategy = SpatialOcclusionPolicy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_sensors=only_decode_sensors,
        )

    def override_strategy_mask(self, sensor_spec: SensorSpec) -> bool:
        """Override the random mask for the given sensor by the encoding and decoding bandsets."""
        return not sensor_spec.varies_in_space


@OCCLUSION_POLICY_REGISTRY.register("cross_sensor_time")
class CrossSensorTemporalOcclusionPolicy(CrossSensorOcclusionPolicy):
    """Randomly select a sensor and apply temporal occlusion to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_sensors: list[str] | None = None,
    ) -> None:
        """Initialize the occlusion policy."""
        space_strategy = SpatialOcclusionPolicy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=space_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_sensors=only_decode_sensors,
        )

    def override_strategy_mask(self, sensor_spec: SensorSpec) -> bool:
        """Override the random mask for the given sensor by the encoding and decoding bandsets."""
        return not sensor_spec.varies_in_space


@OCCLUSION_POLICY_REGISTRY.register("cross_sensor_space_time")
class CrossSensorSpatioTemporalOcclusionPolicy(OcclusionPolicy):
    """Randomly apply spatial or temporal cross-sensor occlusion."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_sensors: list[str] | None = None,
    ) -> None:
        """Initialize the occlusion policy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.time_strategy = CrossSensorTemporalOcclusionPolicy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_sensors=only_decode_sensors,
        )
        self.space_strategy = CrossSensorSpatialOcclusionPolicy(
            encode_ratio,
            decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_sensors=only_decode_sensors,
        )

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply spatial or temporal cross-sensor occlusion to the input data."""
        has_enough_timesteps = batch.active_temporal_length >= 3
        if (np.random.random() < 0.5) or (not has_enough_timesteps):
            logger.info("Applying spatial occlusion")
            return self.space_strategy.apply_mask(batch, patch_size, **kwargs)
        else:
            logger.info("Applying temporal occlusion")
            return self.time_strategy.apply_mask(batch, patch_size, **kwargs)


@OCCLUSION_POLICY_REGISTRY.register("random")
class RandomOcclusionPolicy(OcclusionPolicy):
    """Randomly occludes the input data."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply random occlusion to the input data.

        Args:
            batch: Input data of type GeoSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments

        Returns:
            MaskedGeoSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for random occlusion")
        output_dict: dict[str, NdTensor | None] = {}
        for sensor_name in batch.present_keys:
            instance = getattr(batch, sensor_name)
            if instance is None:
                output_dict[sensor_name] = None
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = None
            else:
                if sensor_name == "timestamps":
                    output_dict[sensor_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None
                sensor = SensorRegistry.get(sensor_name)
                mask = self._create_random_mask(
                    sensor, instance.shape, patch_size, device
                )
                mask = self.fill_mask_with_missing_values(instance, mask, sensor)
                output_dict[sensor_name] = instance
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = mask
        return MaskedGeoSample(**output_dict)


@OCCLUSION_POLICY_REGISTRY.register("cross_sensor_random")
class CrossSensorRandomOcclusionPolicy(CrossSensorOcclusionPolicy):
    """Randomly select a sensor and apply random occlusion to it."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        allow_encoding_decoding_same_bandset: bool = False,
        min_encoded_bandsets: int = 2,
        max_encoded_bandsets: int | None = None,
        min_decoded_bandsets: int | None = None,
        max_decoded_bandsets: int | None = None,
        only_decode_sensors: list[str] | None = None,
    ) -> None:
        """Initialize the occlusion policy."""
        random_strategy = RandomOcclusionPolicy(encode_ratio, decode_ratio)
        super().__init__(
            strategy=random_strategy,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            allow_encoding_decoding_same_bandset=allow_encoding_decoding_same_bandset,
            min_encoded_bandsets=min_encoded_bandsets,
            max_encoded_bandsets=max_encoded_bandsets,
            min_decoded_bandsets=min_decoded_bandsets,
            max_decoded_bandsets=max_decoded_bandsets,
            only_decode_sensors=only_decode_sensors,
        )


@OCCLUSION_POLICY_REGISTRY.register("random_increasing")
class RandomIncreasingOcclusionPolicy(RandomOcclusionPolicy):
    """Gradually increase the occluded tokens (reduce encode ratio)."""

    def __init__(
        self,
        initial_encode_ratio: float = 0.5,
        final_encode_ratio: float = 0.1,
        initial_decode_ratio: float = 0.5,
        final_decode_ratio: float = 0.9,
        steps: int = 1000,
    ) -> None:
        """Initialize the occlusion policy."""
        super().__init__(initial_encode_ratio, initial_decode_ratio)
        self.initial_encode_ratio = initial_encode_ratio
        self.final_encode_ratio = final_encode_ratio
        self.initial_decode_ratio = initial_decode_ratio
        self.final_decode_ratio = final_decode_ratio
        self.steps = steps
        self.elapsed = 0

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply occlusion while changing the encode and decode ratio over time."""
        self.elapsed += 1
        if self.elapsed >= self.steps:
            self._encode_ratio = self.final_encode_ratio
            self._decode_ratio = self.final_decode_ratio
        else:
            factor = self.elapsed / self.steps
            self._encode_ratio = (
                self.initial_encode_ratio
                + (self.final_encode_ratio - self.initial_encode_ratio) * factor
            )
            self._decode_ratio = (
                self.initial_decode_ratio
                + (self.final_decode_ratio - self.initial_decode_ratio) * factor
            )
        return super().apply_mask(batch, patch_size, **kwargs)


@OCCLUSION_POLICY_REGISTRY.register("random_range")
class RandomRangeOcclusionPolicy(OcclusionPolicy):
    """Randomly occludes the input data with a range of encode/decode ratios."""

    def __init__(
        self,
        min_encode_ratio: float = 0.1,
        max_encode_ratio: float = 0.5,
        min_decode_ratio: float | None = None,
        max_decode_ratio: float | None = None,
    ) -> None:
        """Initialize the occlusion policy.

        Args:
            min_encode_ratio: lower bound of range to sample encode ratio.
            max_encode_ratio: upper bound of range to sample encode ratio.
            min_decode_ratio: lower bound of range to sample decode ratio.
            max_decode_ratio: upper bound of range to sample decode ratio.
        """
        self.min_encode_ratio = min_encode_ratio
        self.max_encode_ratio = max_encode_ratio
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio
        self._encode_ratio = (min_encode_ratio + max_encode_ratio) / 2

        if min_decode_ratio is not None and max_decode_ratio is not None:
            self._decode_ratio = (min_decode_ratio + max_decode_ratio) / 2
        elif min_decode_ratio is not None or max_decode_ratio is not None:
            raise ValueError(
                "min_decode_ratio and max_decode_ratio must be both None or both not None"
            )
        else:
            self._decode_ratio = 1 - self._encode_ratio

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply random occlusion with sampled ratios to the input data.

        Args:
            batch: Input data of type GeoSample
            patch_size: patch size applied to sample
            **kwargs: Additional arguments

        Returns:
            MaskedGeoSample containing the masked data and mask
        """
        if patch_size is None:
            raise ValueError("patch_size must be provided for random occlusion")
        output_dict: dict[str, NdTensor | None] = {}
        for sensor_name in batch.present_keys:
            instance = getattr(batch, sensor_name)
            if instance is None:
                output_dict[sensor_name] = None
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = None
            else:
                if sensor_name == "timestamps":
                    output_dict[sensor_name] = instance
                    continue

                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None

                sensor = SensorRegistry.get(sensor_name)

                if sensor.varies_in_space or sensor.has_temporal_axis:
                    batch_size = instance.shape[0]
                    example_encode_ratios = np.random.uniform(
                        self.min_encode_ratio, self.max_encode_ratio, (batch_size,)
                    )
                    if self.min_decode_ratio is not None:
                        example_decode_ratios = np.random.uniform(
                            self.min_decode_ratio, self.max_decode_ratio, (batch_size,)
                        )
                    else:
                        example_decode_ratios = 1 - example_encode_ratios

                    example_masks = []
                    for batch_idx in range(batch_size):
                        example_masks.append(
                            self._create_random_mask(
                                sensor,
                                instance[batch_idx : batch_idx + 1].shape,
                                patch_size,
                                device,
                                encode_ratio=example_encode_ratios[batch_idx],
                                decode_ratio=example_decode_ratios[batch_idx],
                            )
                        )
                    mask = torch.cat(example_masks, dim=0)

                else:
                    mask = self._create_random_mask(
                        sensor, instance.shape, patch_size, device
                    )

                mask = self.fill_mask_with_missing_values(instance, mask, sensor)
                output_dict[sensor_name] = instance
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = mask
        return MaskedGeoSample(**output_dict)


@OCCLUSION_POLICY_REGISTRY.register("selectable_sensor")
class SelectableSensorOcclusionPolicy(OcclusionPolicy):
    """Like sensor occlusion but we mask some for decoding and others fully.

    Plus we also apply random occlusion for the remaining sensors.
    """

    def __init__(
        self,
        decodable_sensors: list[str],
        fully_mask_sensors: list[str],
        max_to_mask: int,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        self.decodable_sensors = decodable_sensors
        self.fully_mask_sensors = fully_mask_sensors
        self.max_to_mask = max_to_mask
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.random_strategy = RandomOcclusionPolicy(encode_ratio, decode_ratio)

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply random occlusion, plus mask certain additional sensors."""
        masked_sample = self.random_strategy.apply_mask(batch, patch_size, **kwargs)

        all_sensors = self.decodable_sensors + self.fully_mask_sensors
        sensor_indices = np.arange(len(all_sensors))
        np.random.shuffle(sensor_indices)
        num_to_mask = np.random.randint(self.max_to_mask + 1)
        cur_mask_sensors = [
            all_sensors[idx] for idx in sensor_indices[0:num_to_mask]
        ]

        logger.debug("Decided to mask sensors: %s", cur_mask_sensors)
        for sensor in cur_mask_sensors:
            if sensor in self.decodable_sensors:
                value = TokenVisibility.PREDICTED.value
            else:
                value = TokenVisibility.ABSENT.value
            logger.debug("Filling sensor %s mask with %s", sensor, value)
            getattr(
                masked_sample, MaskedGeoSample.mask_field_for(sensor)
            )[:] = value

        return masked_sample


@OCCLUSION_POLICY_REGISTRY.register("selectable_random_range_sensor")
class SelectableRandomRangeSensorOcclusionPolicy(OcclusionPolicy):
    """Like sensor occlusion but we mask some for decoding and others fully.

    Plus we also apply random range occlusion for the remaining sensors.
    """

    def __init__(
        self,
        decodable_sensors: list[str],
        fully_mask_sensors: list[str],
        max_to_mask: int,
        min_encode_ratio: float = 0.1,
        max_encode_ratio: float = 0.5,
        min_decode_ratio: float | None = None,
        max_decode_ratio: float | None = None,
    ) -> None:
        """Initialize the occlusion policy."""
        self.decodable_sensors = decodable_sensors
        self.fully_mask_sensors = fully_mask_sensors
        self.max_to_mask = max_to_mask
        self.random_strategy = RandomRangeOcclusionPolicy(
            min_encode_ratio, max_encode_ratio, min_decode_ratio, max_decode_ratio
        )
        self._encode_ratio = self.random_strategy._encode_ratio
        self._decode_ratio = self.random_strategy._decode_ratio

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply random range occlusion, plus mask certain additional sensors."""
        masked_sample = self.random_strategy.apply_mask(batch, patch_size, **kwargs)

        all_sensors = self.decodable_sensors + self.fully_mask_sensors
        batch_size = getattr(batch, all_sensors[0]).shape[0]

        for batch_idx in range(batch_size):
            sensor_indices = np.arange(len(all_sensors))
            np.random.shuffle(sensor_indices)
            num_to_mask = np.random.randint(self.max_to_mask + 1)
            cur_mask_sensors = [
                all_sensors[idx] for idx in sensor_indices[0:num_to_mask]
            ]

            for sensor in cur_mask_sensors:
                if sensor in self.decodable_sensors:
                    value = TokenVisibility.PREDICTED.value
                else:
                    value = TokenVisibility.ABSENT.value
                getattr(
                    masked_sample,
                    MaskedGeoSample.mask_field_for(sensor),
                )[batch_idx] = value

        return masked_sample


class FixedSensorOcclusionPolicy(OcclusionPolicy):
    """Abstract class for occlusion policies that always mask certain sensors on top of another policy."""

    def __init__(
        self,
        strategy: OcclusionPolicy,
        decoded_sensors: list[str],
        randomize_missing_sensors: list[str] | None = None,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.strategy = strategy
        self.decoded_sensors = decoded_sensors
        self.randomize_missing_sensors = randomize_missing_sensors or []

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply occlusion to the input data."""
        masked_sample = self.strategy.apply_mask(batch, patch_size, **kwargs)

        for sensor in self.decoded_sensors:
            mask = getattr(
                masked_sample, MaskedGeoSample.mask_field_for(sensor)
            )
            if mask is None:
                continue
            instance = getattr(masked_sample, sensor)
            mask[:] = TokenVisibility.PREDICTED.value
            mask[:] = self.fill_mask_with_missing_values(
                instance, mask, SensorRegistry.get(sensor)
            )

        if len(self.randomize_missing_sensors) > 0:
            batch_size = getattr(batch, self.randomize_missing_sensors[0]).shape[0]
            for batch_idx in range(batch_size):
                cur_available_sensors = []
                for sensor in self.randomize_missing_sensors:
                    mask = getattr(
                        masked_sample,
                        MaskedGeoSample.mask_field_for(sensor),
                    )
                    is_available = torch.all(mask != TokenVisibility.ABSENT.value)
                    if is_available:
                        cur_available_sensors.append(sensor)

                if len(cur_available_sensors) <= 1:
                    continue

                sensor_indices = np.arange(len(cur_available_sensors))
                np.random.shuffle(sensor_indices)
                num_to_mask = np.random.randint(len(cur_available_sensors))
                cur_mask_sensors = [
                    cur_available_sensors[idx]
                    for idx in sensor_indices[0:num_to_mask]
                ]

                for sensor in cur_mask_sensors:
                    getattr(
                        masked_sample,
                        MaskedGeoSample.mask_field_for(sensor),
                    )[batch_idx] = TokenVisibility.ABSENT.value

        return masked_sample


@OCCLUSION_POLICY_REGISTRY.register("random_fixed_sensor")
class RandomFixedSensorOcclusionPolicy(FixedSensorOcclusionPolicy):
    """Fixed sensor occlusion + random occlusion."""

    def __init__(
        self,
        decoded_sensors: list[str],
        randomize_missing_sensors: list[str] | None = None,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
    ) -> None:
        """Initialize the occlusion policy."""
        super().__init__(
            strategy=RandomOcclusionPolicy(encode_ratio, decode_ratio),
            decoded_sensors=decoded_sensors,
            randomize_missing_sensors=randomize_missing_sensors,
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
        )


@OCCLUSION_POLICY_REGISTRY.register("random_with_decode")
class RandomWithDecodeOcclusionPolicy(OcclusionPolicy):
    """Random occlusion that separates band sets into encode-only and decode-only roles."""

    def __init__(
        self,
        encode_ratio: float = 0.5,
        decode_ratio: float = 0.5,
        only_decode_sensors: list[str] | None = None,
    ):
        """Random occlusion except for decode sensors, which only get decoded."""
        self._encode_ratio = encode_ratio
        self._decode_ratio = decode_ratio
        self.only_decode_sensors = only_decode_sensors or []

    def apply_mask(
        self, batch: GeoSample, patch_size: int | None = None, **kwargs: Any
    ) -> MaskedGeoSample:
        """Apply occlusion to the input data."""
        if patch_size is None:
            raise ValueError("patch_size must be provided for random occlusion")
        output_dict: dict[str, NdTensor | None] = {"timestamps": batch.timestamps}
        none_sensors: list[str] = []
        for sensor_name in batch.present_keys:
            instance = getattr(batch, sensor_name)
            if instance is None:
                none_sensors.append(sensor_name)
                output_dict[sensor_name] = None
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = None
            elif sensor_name == "timestamps":
                continue
            else:
                if isinstance(instance, torch.Tensor):
                    device: torch.device | None = instance.device
                else:
                    device = None
                sensor = SensorRegistry.get(sensor_name)

                mask_shape = instance.shape[:-1] + (
                    self._get_num_bandsets(sensor_name),
                )
                mask = torch.full(
                    mask_shape, fill_value=TokenVisibility.PREDICTED.value, device=device
                )
                mask = self.fill_mask_with_missing_values(instance, mask, sensor)
                output_dict[sensor_name] = instance
                output_dict[
                    MaskedGeoSample.mask_field_for(sensor_name)
                ] = mask

        # now for the trickier encode-decode sensors
        encode_decode_sensors = [
            m
            for m in batch.present_keys
            if m not in self.only_decode_sensors + ["timestamps"] + none_sensors
        ]
        for i in range(batch.batch_size):
            encode_decode_bandsets: list[tuple[str, int]] = []

            for sensor_name in encode_decode_sensors:
                not_missing = (
                    output_dict[
                        MaskedGeoSample.mask_field_for(sensor_name)
                    ][i]  # type: ignore
                    != TokenVisibility.ABSENT.value
                )
                for bandset_idx in range(not_missing.shape[-1]):
                    if not_missing[..., bandset_idx].sum() >= 1:
                        encode_decode_bandsets.append((sensor_name, bandset_idx))

            if len(encode_decode_bandsets) == 1:
                sensor_name, bandset_idx = encode_decode_bandsets[0]
                masked_sensor_name = MaskedGeoSample.mask_field_for(
                    sensor_name
                )  # type: ignore
                output_dict[masked_sensor_name][
                    i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                ] = self._random_fill_unmasked(
                    output_dict[masked_sensor_name][
                        i : i + 1, ..., bandset_idx : bandset_idx + 1
                    ],  # type: ignore
                    SensorRegistry.get(sensor_name),
                    patch_size,
                    self.encode_ratio,
                    self.decode_ratio,
                )
            else:
                np.random.shuffle(encode_decode_bandsets)
                num_encode = math.ceil(len(encode_decode_bandsets) * self.encode_ratio)
                encode_bandsets = encode_decode_bandsets[:num_encode]
                decode_bandsets = encode_decode_bandsets[num_encode:]

                for sensor_name, bandset_idx in encode_bandsets:
                    masked_sensor_name = (
                        MaskedGeoSample.mask_field_for(sensor_name)
                    )
                    output_dict[masked_sensor_name][
                        i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                    ] = self._random_fill_unmasked(
                        output_dict[masked_sensor_name][
                            i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                        ],
                        SensorRegistry.get(sensor_name),
                        patch_size,
                        self.encode_ratio,
                        0,
                    )
                for sensor_name, bandset_idx in decode_bandsets:
                    masked_sensor_name = (
                        MaskedGeoSample.mask_field_for(sensor_name)
                    )
                    output_dict[masked_sensor_name][
                        i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                    ] = self._random_fill_unmasked(
                        output_dict[masked_sensor_name][
                            i : i + 1, ..., bandset_idx : bandset_idx + 1  # type: ignore
                        ],
                        SensorRegistry.get(sensor_name),
                        patch_size,
                        0,
                        self.decode_ratio,
                    )

        return MaskedGeoSample(**output_dict)


def propagate_tokenization_config(
    occlusion_policy: OcclusionPolicy,
    tokenization_config: "TokenizationConfig",
) -> None:
    """Attach the tokenization config to an occlusion policy (recursively).

    Some occlusion policies wrap other policies (e.g., FixedSensorOcclusionPolicy).
    We need the tokenization config on every policy instance so that mask shapes
    match the model's band-grouping configuration.

    Args:
        occlusion_policy: The occlusion policy to configure.
        tokenization_config: The tokenization config to propagate.
    """
    visited: set[int] = set()

    def _set_config(policy: OcclusionPolicy) -> None:
        policy_id = id(policy)
        if policy_id in visited:
            return
        visited.add(policy_id)

        policy.tokenization_config = tokenization_config

        for child in vars(policy).values():
            if isinstance(child, OcclusionPolicy):
                _set_config(child)

    _set_config(occlusion_policy)


@dataclass
class OcclusionConfig(Config):
    """Configuration for occlusion policies.

    Args:
        strategy_config: Occlusion policy to use in the format of
        {
            "type": "random", # registry key
            # rest of init kwargs
        }
        tokenization_config: Optional tokenization config for custom band groupings.
            If provided, propagated to the occlusion policy so mask shapes match
            the model's band-grouping configuration.
    """

    strategy_config: dict[str, Any]
    tokenization_config: "TokenizationConfig | None" = None

    def build(self) -> OcclusionPolicy:
        """Build an OcclusionPolicy from the config."""
        # Copy strategy_config since we pop from it
        config = dict(self.strategy_config)
        occlusion_key = config.pop("type")
        policy = OCCLUSION_POLICY_REGISTRY.get_class(occlusion_key)(**config)

        # Propagate tokenization config if provided
        if self.tokenization_config is not None:
            propagate_tokenization_config(policy, self.tokenization_config)

        return policy
