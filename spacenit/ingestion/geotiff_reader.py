"""GeoTIFF reader for the ``allenai/olmoearth_pretrain_dataset`` layout.

Reads multi-sensor, multi-resolution GeoTIFF files directly from the
HuggingFace dataset directory structure.  Replaces the H5-based
``tile_loader.py`` / ``tile_dataset.py`` pipeline.

The dataset layout is::

    data_root/
      <crs>/
        <col>/
          <row>/
            <sensor_label>/
              <tile_time>.tif   (or multiple files for multi-resolution sensors)

Each sensor may have multiple GeoTIFF files per sample when it contains
spectral groups at different spatial resolutions (e.g. Sentinel-2 has
10m, 20m, and 60m groups stored as separate files).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from spacenit.ingestion.sensors import (
    ABSENT_INDICATOR,
    SensorRegistry,
    SensorSpec,
)
from spacenit.structures import GeoSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample locator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleLocator:
    """Identifies a single sample in the dataset.

    Attributes:
        crs: Coordinate reference system identifier (directory name).
        col: Column index (directory name).
        row: Row index (directory name).
        tile_time: Temporal identifier (e.g. timestamp or index).
    """

    crs: str
    col: str
    row: str
    tile_time: str

    @property
    def path_parts(self) -> tuple[str, str, str]:
        """Return ``(crs, col, row)`` for constructing file paths."""
        return (self.crs, self.col, self.row)


# ---------------------------------------------------------------------------
# Single-sensor reader
# ---------------------------------------------------------------------------


def read_sensor_geotiff(
    locator: SampleLocator,
    sensor_label: str,
    data_root: Path | str,
) -> np.ndarray | None:
    """Read a single sensor's GeoTIFF(s) for one sample.

    Handles multi-resolution sensors by reading each resolution file
    separately and returning them stacked along the channel axis.

    Args:
        locator: Sample identifier.
        sensor_label: Sensor label (e.g. ``"sentinel2_l2a"``).
        data_root: Root directory of the dataset.

    Returns:
        Array of shape ``(H, W, C)`` or ``(H, W, T, C)`` for temporal
        sensors, or ``None`` if the file does not exist.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF reading. "
            "Install it with: pip install rasterio"
        )

    data_root = Path(data_root)
    spec = SensorRegistry.get(sensor_label)
    crs, col, row = locator.path_parts

    sensor_dir = data_root / crs / col / row / sensor_label

    if not sensor_dir.exists():
        logger.debug(f"Sensor directory not found: {sensor_dir}")
        return None

    # Find GeoTIFF files for this sample
    tif_pattern = f"{locator.tile_time}*.tif"
    tif_files = sorted(sensor_dir.glob(tif_pattern))

    if not tif_files:
        # Try without tile_time prefix (some sensors have single files)
        tif_files = sorted(sensor_dir.glob("*.tif"))

    if not tif_files:
        logger.debug(f"No GeoTIFF files found in {sensor_dir}")
        return None

    arrays = []
    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                # Read all bands: (C, H, W)
                data = src.read()
                # Transpose to (H, W, C)
                data = np.transpose(data, (1, 2, 0))
                arrays.append(data)
        except Exception as e:
            logger.warning(f"Failed to read {tif_path}: {e}")
            return None

    if len(arrays) == 1:
        return arrays[0]

    # Multi-resolution: resize smaller arrays to match the largest
    max_h = max(a.shape[0] for a in arrays)
    max_w = max(a.shape[1] for a in arrays)

    resized = []
    for arr in arrays:
        if arr.shape[0] != max_h or arr.shape[1] != max_w:
            # Simple nearest-neighbor resize for multi-resolution alignment
            from scipy.ndimage import zoom

            scale_h = max_h / arr.shape[0]
            scale_w = max_w / arr.shape[1]
            arr = zoom(arr, (scale_h, scale_w, 1), order=0)
        resized.append(arr)

    # Stack along channel axis
    return np.concatenate(resized, axis=-1)


# ---------------------------------------------------------------------------
# Full sample reader
# ---------------------------------------------------------------------------


def read_sample(
    locator: SampleLocator,
    sensor_labels: list[str],
    data_root: Path | str,
    max_timesteps: int | None = None,
) -> GeoSample:
    """Read all sensors for one sample.

    Args:
        locator: Sample identifier.
        sensor_labels: List of sensor labels to read.
        data_root: Root directory of the dataset.
        max_timesteps: Maximum number of temporal steps to read.
            If ``None``, reads all available timesteps.

    Returns:
        A :class:`GeoSample` with data for each present sensor.
    """
    fields: dict[str, np.ndarray] = {}

    for label in sensor_labels:
        data = read_sensor_geotiff(locator, label, data_root)
        if data is None:
            continue

        spec = SensorRegistry.get(label)

        # Truncate temporal dimension if needed
        if spec.has_temporal_axis and max_timesteps is not None:
            if data.ndim == 4:  # (H, W, T, C)
                data = data[:, :, :max_timesteps, :]
            elif data.ndim == 3:  # (T, C) for time-only sensors
                data = data[:max_timesteps, :]

        fields[label] = data

    return GeoSample(**{k: v for k, v in fields.items()})


# ---------------------------------------------------------------------------
# Metadata / sample discovery
# ---------------------------------------------------------------------------


def discover_samples(
    data_root: Path | str,
    sensor_labels: list[str] | None = None,
) -> list[SampleLocator]:
    """Discover all available samples by scanning the directory structure.

    Walks the ``data_root/<crs>/<col>/<row>/`` hierarchy and returns a
    :class:`SampleLocator` for each unique (crs, col, row, tile_time)
    combination.

    Args:
        data_root: Root directory of the dataset.
        sensor_labels: If provided, only consider directories that contain
            at least one of these sensors.

    Returns:
        List of :class:`SampleLocator` instances.
    """
    data_root = Path(data_root)
    locators: list[SampleLocator] = []
    seen: set[tuple[str, str, str, str]] = set()

    if not data_root.exists():
        logger.warning(f"Data root does not exist: {data_root}")
        return locators

    for crs_dir in sorted(data_root.iterdir()):
        if not crs_dir.is_dir():
            continue
        for col_dir in sorted(crs_dir.iterdir()):
            if not col_dir.is_dir():
                continue
            for row_dir in sorted(col_dir.iterdir()):
                if not row_dir.is_dir():
                    continue

                # Find sensor directories
                sensor_dirs = [
                    d for d in row_dir.iterdir()
                    if d.is_dir() and (
                        sensor_labels is None
                        or d.name in sensor_labels
                    )
                ]

                if not sensor_dirs:
                    continue

                # Discover tile times from GeoTIFF filenames
                tile_times: set[str] = set()
                for sd in sensor_dirs:
                    for tif in sd.glob("*.tif"):
                        # Extract tile_time from filename
                        # Convention: <tile_time>.tif or <tile_time>_<resolution>.tif
                        stem = tif.stem
                        # Take everything before the first underscore as tile_time
                        # (or the whole stem if no underscore)
                        parts = stem.split("_")
                        tile_time = parts[0]
                        tile_times.add(tile_time)

                for tt in sorted(tile_times):
                    key = (crs_dir.name, col_dir.name, row_dir.name, tt)
                    if key not in seen:
                        seen.add(key)
                        locators.append(SampleLocator(
                            crs=crs_dir.name,
                            col=col_dir.name,
                            row=row_dir.name,
                            tile_time=tt,
                        ))

    logger.info(f"Discovered {len(locators)} samples in {data_root}")
    return locators


def discover_samples_from_csv(
    csv_path: Path | str,
) -> list[SampleLocator]:
    """Discover samples from a CSV metadata file.

    The CSV should have columns: ``crs``, ``col``, ``row``, ``tile_time``.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of :class:`SampleLocator` instances.
    """
    import csv

    csv_path = Path(csv_path)
    locators: list[SampleLocator] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            locators.append(SampleLocator(
                crs=row_dict["crs"],
                col=row_dict["col"],
                row=row_dict["row"],
                tile_time=row_dict["tile_time"],
            ))

    logger.info(f"Loaded {len(locators)} samples from {csv_path}")
    return locators
