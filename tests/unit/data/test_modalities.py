"""Test sensor modality definitions."""

from spacenit.ingestion.sensors import SENTINEL2, LANDSAT, SensorRegistry


def test_sensor_spec_channel_order() -> None:
    """Test that the channel order is correct.

    This should be the order the data is stacked in.
    """
    expected_channel_order_sentinel2 = [
        "B02",
        "B03",
        "B04",
        "B08",
        "B05",
        "B06",
        "B07",
        "B8A",
        "B11",
        "B12",
        "B01",
        "B09",
        "B10",
    ]

    expected_channel_order_landsat = [
        "B8",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B9",
        "B10",
        "B11",
    ]
    assert SENTINEL2.all_channel_names == expected_channel_order_sentinel2
    assert LANDSAT.all_channel_names == expected_channel_order_landsat


def test_sensor_spec_total_channels() -> None:
    """Test that the number of channels is correct."""
    assert SENTINEL2.total_channels == 13
    assert LANDSAT.total_channels == 11


def test_group_indices() -> None:
    """Test that the group indices are correct."""
    assert SENTINEL2.group_indices() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7, 8, 9],
        [10, 11, 12],
    ]
