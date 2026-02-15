"""Conftest for the tests."""

import random
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch

from spacenit.ingestion.sensors import (
    ABSENT_INDICATOR,
    SensorRegistry,
    SensorSpec,
    SENTINEL2_L2A,
    LATLON,
)
from spacenit.structures import GeoSample, TokenVisibility

# Avoid triton imports from olmo-core during tests
sys.modules["triton"] = types.SimpleNamespace(
    runtime=types.SimpleNamespace(autotuner=object(), driver=object())  # type: ignore
)


@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)


@pytest.fixture
def supported_sensors() -> list[SensorSpec]:
    """Create a list of supported sensors for testing."""
    return [SENTINEL2_L2A, LATLON]


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory: Any) -> Any:
    """Session-scoped temporary directory."""
    return tmp_path_factory.mktemp("session")


@pytest.fixture
def masked_sample_dict(
    sensor_group_count_and_total_channels: dict[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Get a masked sample dictionary."""
    sentinel2_l2a_total_channels = sensor_group_count_and_total_channels[
        "sentinel2_l2a"
    ][1]
    latlon_total_channels = sensor_group_count_and_total_channels["latlon"][1]
    B, H, W, T, C = (
        2,
        4,
        4,
        2,
        sentinel2_l2a_total_channels,
    )
    # Create dummy sentinel2_l2a data: shape (B, H, W, T, C)
    sentinel2_l2a = torch.randn(B, H, W, T, C, requires_grad=True)
    # Here we assume 0 (VISIBLE_ENCODER) means the token is visible.
    sentinel2_l2a_mask = torch.full(
        (B, H, W, T, C),
        fill_value=TokenVisibility.VISIBLE_ENCODER.value,
        dtype=torch.long,
    )
    # Dummy latitude-longitude data.
    latlon = torch.randn(B, latlon_total_channels, requires_grad=True)
    latlon_mask = torch.full(
        (B, latlon_total_channels),
        fill_value=TokenVisibility.PREDICTED.value,
        dtype=torch.float32,
    )
    worldcover = torch.randn(B, H, W, 1, 1, requires_grad=True)
    worldcover_mask = torch.full(
        (B, H, W, 1, 1),
        fill_value=TokenVisibility.PREDICTED.value,
        dtype=torch.float32,
    )
    # Generate valid timestamps:
    # - days: range 1..31,
    # - months: range 1..13,
    # - years: e.g. 2018-2019.
    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

    masked_sample_dict = {
        "sentinel2_l2a": sentinel2_l2a,
        "sentinel2_l2a_mask": sentinel2_l2a_mask,
        "latlon": latlon,
        "latlon_mask": latlon_mask,
        "worldcover": worldcover,
        "worldcover_mask": worldcover_mask,
        "timestamps": timestamps,
    }
    return masked_sample_dict


@pytest.fixture
def samples_with_missing_sensors() -> list[tuple[int, GeoSample]]:
    """Samples with missing sensors."""
    s2_H, s2_W, s2_T, s2_C = 8, 8, 12, 12
    s1_H, s1_W, s1_T, s1_C = 8, 8, 12, 2
    wc_H, wc_W, wc_T, wc_C = 8, 8, 1, 1
    na_H, na_W, na_T, na_C = 128, 128, 1, 4

    example_s2_data = np.random.randn(s2_H, s2_W, s2_T, s2_C).astype(np.float32)
    example_s1_data = np.random.randn(s1_H, s1_W, s1_T, s1_C).astype(np.float32)
    example_wc_data = np.random.randn(wc_H, wc_W, wc_T, wc_C).astype(np.float32)
    example_na_data = np.random.randn(na_H, na_W, na_T, na_C).astype(np.float32)
    example_latlon_data = np.random.randn(2).astype(np.float32)

    missing_s1_data = np.full((s1_H, s1_W, s1_T, s1_C), ABSENT_INDICATOR).astype(
        np.float32
    )
    missing_wc_data = np.full((wc_H, wc_W, wc_T, wc_C), ABSENT_INDICATOR).astype(
        np.float32
    )

    timestamps = np.array(
        [
            [15, 7, 2023],
            [15, 8, 2023],
            [15, 9, 2023],
            [15, 10, 2023],
            [15, 11, 2023],
            [15, 11, 2023],
            [15, 1, 2024],
            [15, 2, 2024],
            [15, 3, 2024],
            [15, 4, 2024],
            [15, 5, 2024],
            [15, 6, 2024],
        ],
        dtype=np.int32,
    )

    sample1 = GeoSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        naip_10=example_na_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample2 = GeoSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=missing_s1_data,
        worldcover=example_wc_data,
        naip_10=example_na_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample_3 = GeoSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=missing_wc_data,
        naip_10=example_na_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    batch = [(1, sample1), (1, sample2), (1, sample_3)]
    return batch


@pytest.fixture
def samples_without_missing_sensors(
    set_random_seeds: None,
) -> list[tuple[int, GeoSample]]:
    """Samples without missing sensors."""
    s2_H, s2_W, s2_T, s2_C = 8, 8, 12, 12
    s1_H, s1_W, s1_T, s1_C = 8, 8, 12, 2
    wc_H, wc_W, wc_T, wc_C = 8, 8, 1, 1
    na_H, na_W, na_T, na_C = 128, 128, 1, 4
    example_s2_data = np.random.randn(s2_H, s2_W, s2_T, s2_C).astype(np.float32)
    example_s1_data = np.random.randn(s1_H, s1_W, s1_T, s1_C).astype(np.float32)
    example_wc_data = np.random.randn(wc_H, wc_W, wc_T, wc_C).astype(np.float32)
    example_latlon_data = np.random.randn(2).astype(np.float32)
    example_na_data = np.random.randn(na_H, na_W, na_T, na_C).astype(np.float32)
    timestamps = np.array(
        [
            [15, 7, 2023],
            [15, 8, 2023],
            [15, 9, 2023],
            [15, 10, 2023],
            [15, 11, 2023],
            [15, 11, 2023],
            [15, 1, 2024],
            [15, 2, 2024],
            [15, 3, 2024],
            [15, 4, 2024],
            [15, 5, 2024],
            [15, 6, 2024],
        ],
        dtype=np.int32,
    )

    sample1 = GeoSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        naip_10=example_na_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample2 = GeoSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        naip_10=example_na_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample_3 = GeoSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        naip_10=example_na_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    batch = [(1, sample1), (1, sample2), (1, sample_3)]
    return batch


@pytest.fixture
def sensor_group_count_and_total_channels(
    supported_sensors: list[SensorSpec],
) -> dict[str, tuple[int, int]]:
    """Get the number of spectral groups and total channels for each sensor.

    Returns:
        Dictionary mapping sensor label to tuple of (group_count, total_channels)
    """
    return {
        sensor.label: (
            sensor.group_count,
            sensor.total_channels,
        )
        for sensor in supported_sensors
    }
