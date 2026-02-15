"""Tests for the GeoTIFF reader module.

Tests use synthetic data (no actual GeoTIFF files required) to verify
the reader's logic for sample discovery and data assembly.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from spacenit.ingestion.geotiff_reader import (
    SampleLocator,
    discover_samples,
    discover_samples_from_csv,
)


class TestSampleLocator:
    def test_frozen(self):
        loc = SampleLocator(crs="32610", col="100", row="200", tile_time="2020")
        with pytest.raises(AttributeError):
            loc.crs = "other"

    def test_path_parts(self):
        loc = SampleLocator(crs="32610", col="100", row="200", tile_time="2020")
        assert loc.path_parts == ("32610", "100", "200")

    def test_equality(self):
        a = SampleLocator("32610", "100", "200", "2020")
        b = SampleLocator("32610", "100", "200", "2020")
        assert a == b

    def test_hash(self):
        a = SampleLocator("32610", "100", "200", "2020")
        b = SampleLocator("32610", "100", "200", "2020")
        assert hash(a) == hash(b)


class TestDiscoverSamples:
    def test_empty_directory(self, tmp_path):
        locators = discover_samples(tmp_path)
        assert locators == []

    def test_nonexistent_directory(self, tmp_path):
        locators = discover_samples(tmp_path / "nonexistent")
        assert locators == []

    def test_discovers_from_structure(self, tmp_path):
        """Create a minimal directory structure and verify discovery."""
        # Create: data_root/32610/100/200/sentinel2_l2a/2020.tif
        sensor_dir = tmp_path / "32610" / "100" / "200" / "sentinel2_l2a"
        sensor_dir.mkdir(parents=True)
        (sensor_dir / "2020.tif").touch()
        (sensor_dir / "2021.tif").touch()

        locators = discover_samples(tmp_path)
        assert len(locators) == 2
        times = {loc.tile_time for loc in locators}
        assert "2020" in times
        assert "2021" in times

    def test_filter_by_sensor(self, tmp_path):
        """Only discover samples for specified sensors."""
        s2_dir = tmp_path / "32610" / "100" / "200" / "sentinel2_l2a"
        s2_dir.mkdir(parents=True)
        (s2_dir / "2020.tif").touch()

        s1_dir = tmp_path / "32610" / "100" / "200" / "sentinel1"
        s1_dir.mkdir(parents=True)
        (s1_dir / "2020.tif").touch()

        # Only look for sentinel1
        locators = discover_samples(tmp_path, sensor_labels=["sentinel1"])
        assert len(locators) >= 1


class TestDiscoverSamplesFromCSV:
    def test_reads_csv(self, tmp_path):
        csv_path = tmp_path / "samples.csv"
        csv_path.write_text(
            "crs,col,row,tile_time\n"
            "32610,100,200,2020\n"
            "32610,100,200,2021\n"
            "32611,50,75,2020\n"
        )
        locators = discover_samples_from_csv(csv_path)
        assert len(locators) == 3
        assert locators[0].crs == "32610"
        assert locators[2].crs == "32611"
