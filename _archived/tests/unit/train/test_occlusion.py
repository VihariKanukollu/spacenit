"""Test occlusion (masking) policies."""

import logging
import random

import torch

from spacenit.ingestion.sensors import (
    ABSENT_INDICATOR,
    SensorRegistry,
    SENTINEL2_L2A,
    SENTINEL1,
    WORLDCOVER,
    LATLON,
)
from spacenit.structures import GeoSample, MaskedGeoSample, TokenVisibility
from spacenit.pipeline.occlusion import (
    CrossSensorRandomOcclusionPolicy,
    CrossSensorSpatialOcclusionPolicy,
    RandomOcclusionPolicy,
    RandomRangeOcclusionPolicy,
    RandomWithDecodeOcclusionPolicy,
    SpatialOcclusionPolicy,
    TemporalOcclusionPolicy,
)

logger = logging.getLogger(__name__)


def test_clear_masks() -> None:
    """Test clear_masks functionality."""
    b, t, h, w = 1, 2, 4, 4

    mask = torch.ones(b, t, h, w, 3) * TokenVisibility.PREDICTED.value
    # the first timestep is missing
    mask[:, 0] = TokenVisibility.ABSENT.value
    s = MaskedGeoSample(
        sentinel2_l2a=torch.ones(b, t, h, w, 12),
        sentinel2_l2a_mask=mask,
        timestamps=torch.ones(b, t, 3),
    )
    s = s.clear_masks()
    assert (s.sentinel2_l2a_mask[:, 0] == TokenVisibility.ABSENT.value).all()  # type: ignore
    assert (s.sentinel2_l2a_mask[:, 1:] == TokenVisibility.VISIBLE_ENCODER.value).all()  # type: ignore


def test_random_occlusion_and_clear_masks() -> None:
    """Test random occlusion ratios."""
    b, h, w, t = 4, 16, 16, 8

    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_total_channels = SENTINEL2_L2A.total_channels
    worldcover_total_channels = WORLDCOVER.total_channels
    latlon_total_channels = LATLON.total_channels
    batch = GeoSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_total_channels)),
        latlon=torch.ones((b, latlon_total_channels)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, 1, worldcover_total_channels)),
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = RandomOcclusionPolicy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
        patch_size=patch_size,
    )
    # Check that all values in the first patch are the same (consistent masking)
    assert masked_sample.sentinel2_l2a_mask is not None
    first_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 0, 0]
    first_value: int = first_patch[0, 0]
    assert (first_patch == first_value).all()
    second_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 1, 0]
    second_value: int = second_patch[0, 0]
    assert (second_patch == second_value).all()
    worldcover_patch: torch.Tensor = masked_sample.worldcover_mask[0, :4, :4, 0]  # type: ignore
    worldcover_value: int = worldcover_patch[0, 0]
    assert (worldcover_patch == worldcover_value).all()
    # check that each sensor has the right masking ratio
    for field_name in masked_sample._fields:
        if field_name.endswith("mask"):
            data_field_name = MaskedGeoSample.data_field_for(field_name)
            sensor = SensorRegistry.get(data_field_name)
            mask = getattr(masked_sample, field_name)
            data = getattr(masked_sample, data_field_name)
            logger.info(f"Mask name: {field_name}")
            if mask is None:
                continue
            total_elements = mask.numel()
            num_encoder = len(mask[mask == TokenVisibility.VISIBLE_ENCODER.value])
            num_predicted = len(mask[mask == TokenVisibility.PREDICTED.value])
            assert (num_encoder / total_elements) == encode_ratio, (
                f"{field_name} has incorrect encode mask ratio"
            )
            assert (num_predicted / total_elements) == decode_ratio, (
                f"{field_name} has incorrect decode mask ratio"
            )
            assert mask.shape[:-1] == data.shape[:-1], (
                f"{field_name} has incorrect shape"
            )
            assert mask.shape[-1] == sensor.group_count, (
                f"{field_name} has incorrect group count"
            )

    unmasked_sample = masked_sample.clear_masks()
    for field_name in unmasked_sample._fields:
        if field_name.endswith("mask"):
            mask = getattr(unmasked_sample, field_name)
            if mask is not None:
                assert (mask == 0).all()


def test_space_structure_occlusion_and_clear_masks() -> None:
    """Test space structure occlusion ratios."""
    b, h, w, t = 100, 16, 16, 8

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_total_channels = SENTINEL2_L2A.total_channels
    latlon_total_channels = LATLON.total_channels
    worldcover_total_channels = WORLDCOVER.total_channels
    batch = GeoSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_total_channels)),
        latlon=torch.ones((b, latlon_total_channels)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_total_channels)),
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = SpatialOcclusionPolicy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
        patch_size=4,
    )
    # check that each sensor has the right masking ratio
    for field_name in masked_sample._fields:
        if field_name.endswith("mask"):
            data_field_name = MaskedGeoSample.data_field_for(field_name)
            sensor = SensorRegistry.get(data_field_name)
            mask = getattr(masked_sample, field_name)
            data = getattr(masked_sample, data_field_name)
            logger.info(f"Mask name: {field_name}")
            if mask is None:
                continue
            total_elements = mask.numel()
            num_encoder = len(mask[mask == TokenVisibility.VISIBLE_ENCODER.value])
            num_predicted = len(mask[mask == TokenVisibility.PREDICTED.value])
            assert (num_encoder / total_elements) == encode_ratio, (
                f"{field_name} has incorrect encode mask ratio"
            )
            assert (num_predicted / total_elements) == decode_ratio, (
                f"{field_name} has incorrect decode mask ratio"
            )
            assert mask.shape[:-1] == data.shape[:-1], (
                f"{field_name} has incorrect shape"
            )
            assert mask.shape[-1] == sensor.group_count, (
                f"{field_name} has incorrect group count"
            )

    unmasked_sample = masked_sample.clear_masks()
    for field_name in unmasked_sample._fields:
        if field_name.endswith("mask"):
            mask = getattr(unmasked_sample, field_name)
            if mask is not None:
                assert (mask == 0).all()


def test_time_structure_occlusion_and_clear_masks() -> None:
    """Test time structure occlusion ratios."""
    b, h, w, t = 100, 16, 16, 8

    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_total_channels = SENTINEL2_L2A.total_channels
    latlon_total_channels = LATLON.total_channels
    worldcover_total_channels = WORLDCOVER.total_channels
    batch = GeoSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_total_channels)),
        latlon=torch.ones((b, latlon_total_channels)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_total_channels)),
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = TemporalOcclusionPolicy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
        patch_size=patch_size,
    )
    # check that each sensor has the right masking ratio
    for field_name in masked_sample._fields:
        if field_name.endswith("mask"):
            data_field_name = MaskedGeoSample.data_field_for(field_name)
            sensor = SensorRegistry.get(data_field_name)
            mask = getattr(masked_sample, field_name)
            data = getattr(masked_sample, data_field_name)
            logger.info(f"Mask name: {field_name}")
            if mask is None:
                continue
            total_elements = mask.numel()
            num_encoder = len(mask[mask == TokenVisibility.VISIBLE_ENCODER.value])
            num_predicted = len(mask[mask == TokenVisibility.PREDICTED.value])
            assert (num_encoder / total_elements) == encode_ratio, (
                f"{field_name} has incorrect encode mask ratio"
            )
            assert (num_predicted / total_elements) == decode_ratio, (
                f"{field_name} has incorrect decode mask ratio"
            )
            assert mask.shape[:-1] == data.shape[:-1], (
                f"{field_name} has incorrect shape"
            )
            assert mask.shape[-1] == sensor.group_count, (
                f"{field_name} has incorrect group count"
            )

    unmasked_sample = masked_sample.clear_masks()
    for field_name in unmasked_sample._fields:
        if field_name.endswith("mask"):
            mask = getattr(unmasked_sample, field_name)
            if mask is not None:
                assert (mask == 0).all()


def test_create_random_mask_with_absent_indicator() -> None:
    """Test that ABSENT_INDICATOR in GeoSample is respected during occlusion."""
    b, h, w, t = 5, 8, 8, 4

    # Create a sample with sentinel1 data where some samples are missing
    sentinel1 = torch.ones((b, h, w, t, 2))  # 2 channels for simplicity

    # Create an absent mask for sentinel1 where half the batch is missing
    sentinel1[b // 2 :] = ABSENT_INDICATOR

    # Create the GeoSample
    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)

    batch = GeoSample(
        sentinel2_l2a=torch.ones((b, h, w, t, 12)),  # 12 channels for sentinel2
        sentinel1=sentinel1,
        timestamps=timestamps,
    )

    # Apply random occlusion
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = RandomOcclusionPolicy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    ).apply_mask(batch, patch_size=1)

    # Check the sentinel1 mask
    sentinel1_mask = masked_sample.sentinel1_mask
    assert sentinel1_mask is not None

    # For non-missing samples, check the ratios
    non_missing_indices = torch.where(sentinel1 != ABSENT_INDICATOR)[0]
    for idx in non_missing_indices:
        mask_slice = sentinel1_mask[idx]
        total_elements = mask_slice.numel()

        num_encoder = (mask_slice == TokenVisibility.VISIBLE_ENCODER.value).sum().item()
        num_predicted = (mask_slice == TokenVisibility.PREDICTED.value).sum().item()
        num_target = (mask_slice == TokenVisibility.TARGET_ONLY.value).sum().item()

        # Check with tolerance for rounding
        assert abs(num_encoder / total_elements - encode_ratio) < 0.05, (
            "Encoder ratio incorrect for non-missing samples"
        )
        assert abs(num_predicted / total_elements - decode_ratio) < 0.05, (
            "Predicted ratio incorrect for non-missing samples"
        )
        assert (
            abs(num_target / total_elements - (1 - encode_ratio - decode_ratio)) < 0.05
        ), "Target ratio incorrect for non-missing samples"

    # Check that missing samples have the absent value
    missing_indices = torch.where(sentinel1 == ABSENT_INDICATOR)[0]
    for idx in missing_indices:
        mask_slice = sentinel1_mask[idx]
        # All values for missing samples should be set to TokenVisibility.ABSENT.value
        assert (mask_slice == TokenVisibility.ABSENT.value).all(), (
            f"Missing sample {idx} should have all mask values set to ABSENT"
        )


def test_create_spatial_mask_with_patch_size() -> None:
    """Test the _create_patch_spatial_mask function with different patch sizes."""
    b = 4
    h, w = 16, 16
    shape = (b, h, w)
    patch_size = 4

    encode_ratio, decode_ratio = 0.25, 0.5
    policy = SpatialOcclusionPolicy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    )

    # Call the _create_patch_spatial_mask function directly
    patch_mask = policy._create_patch_spatial_mask(
        sensor=SENTINEL2_L2A, shape=shape, patch_size_at_16=patch_size
    )
    mask = policy._resize_spatial_mask_for_sensor(
        patch_mask,
        sensor=SENTINEL2_L2A,
        patch_size_at_16=patch_size,
    )

    # Check that the mask has the right shape
    assert mask.shape == shape, "Mask shape should match the input shape"

    # Check that patches have consistent values
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            for b_idx in range(b):
                patch = mask[b_idx, i : i + patch_size, j : j + patch_size]
                # All values within a patch should be the same
                assert (patch == patch[0, 0]).all(), (
                    f"Patch at ({b_idx},{i},{j}) has inconsistent values"
                )

    # Check the ratios across all values
    total_elements = mask.numel()
    num_encoder = len(mask[mask == TokenVisibility.VISIBLE_ENCODER.value])
    num_predicted = len(mask[mask == TokenVisibility.PREDICTED.value])
    num_target = len(mask[mask == TokenVisibility.TARGET_ONLY.value])

    assert num_encoder / total_elements == encode_ratio, "Incorrect encode mask ratio"
    assert num_predicted / total_elements == decode_ratio, "Incorrect decode mask ratio"
    assert num_target / total_elements == 1 - encode_ratio - decode_ratio, (
        "Incorrect target mask ratio"
    )


def test_create_temporal_mask() -> None:
    """Test the _create_temporal_mask function."""
    b = 10
    t = 8
    shape = (b, t)

    encode_ratio, decode_ratio = 0.25, 0.5
    policy = TemporalOcclusionPolicy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    )

    # Call the _create_temporal_mask function directly
    timesteps_with_at_least_one_sensor = torch.tensor(list(range(t)))
    mask = policy._create_temporal_mask(
        shape=shape,
        timesteps_with_at_least_one_sensor=timesteps_with_at_least_one_sensor,
    )

    # Check the masking ratios for non-missing timesteps
    total_non_missing = mask.numel()
    num_encoder = len(mask[mask == TokenVisibility.VISIBLE_ENCODER.value])
    num_predicted = len(mask[mask == TokenVisibility.PREDICTED.value])
    num_target = len(mask[mask == TokenVisibility.TARGET_ONLY.value])

    # Check that the ratios are close to expected for non-missing values
    # Note: With small values of t, the ratios might not be exactly as expected
    assert abs(num_encoder / total_non_missing - encode_ratio) < 0.2, (
        "Encode mask ratio too far from expected"
    )
    assert abs(num_predicted / total_non_missing - decode_ratio) < 0.2, (
        "Decode mask ratio too far from expected"
    )
    assert (
        abs(num_target / total_non_missing - (1 - encode_ratio - decode_ratio)) < 0.2
    ), "Target mask ratio too far from expected"


def test_random_range_occlusion() -> None:
    """Test random range occlusion."""
    b, h, w, t = 100, 16, 16, 8
    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_total_channels = SENTINEL2_L2A.total_channels
    worldcover_total_channels = WORLDCOVER.total_channels
    latlon_total_channels = LATLON.total_channels
    batch = GeoSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_total_channels)),
        latlon=torch.ones((b, latlon_total_channels)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, 1, worldcover_total_channels)),
    )
    min_encode_ratio = 0.4
    max_encode_ratio = 0.9
    masked_sample = RandomRangeOcclusionPolicy(
        min_encode_ratio=min_encode_ratio,
        max_encode_ratio=max_encode_ratio,
    ).apply_mask(
        batch,
        patch_size=patch_size,
    )
    # Check that all values in the first patch are the same (consistent masking)
    assert masked_sample.sentinel2_l2a_mask is not None
    first_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 0, 0]
    first_value: int = first_patch[0, 0]
    assert (first_patch == first_value).all()
    second_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 1, 0]
    second_value: int = second_patch[0, 0]
    assert (second_patch == second_value).all()
    worldcover_patch: torch.Tensor = masked_sample.worldcover_mask[0, :4, :4, 0]  # type: ignore
    worldcover_value: int = worldcover_patch[0, 0]
    assert (worldcover_patch == worldcover_value).all()
    # Check that the distribution of masking ratios is roughly correct.
    encode_ratios = []
    decode_ratios = []
    for example_idx in range(b):
        mask = masked_sample.sentinel2_l2a_mask[example_idx]
        total_elements = mask.numel()
        num_encoder = len(mask[mask == TokenVisibility.VISIBLE_ENCODER.value])
        num_predicted = len(mask[mask == TokenVisibility.PREDICTED.value])
        encode_ratios.append(num_encoder / total_elements)
        decode_ratios.append(num_predicted / total_elements)
    eps = 0.02
    assert min_encode_ratio - eps <= min(encode_ratios) < min_encode_ratio + 0.1
    assert max_encode_ratio + eps >= max(encode_ratios) > max_encode_ratio - 0.1
    min_decode_ratio = 1 - max_encode_ratio
    max_decode_ratio = 1 - min_encode_ratio
    assert min_decode_ratio - eps <= min(decode_ratios) < min_decode_ratio + 0.1
    assert max_decode_ratio + eps >= max(decode_ratios) > max_decode_ratio - 0.1
