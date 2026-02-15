"""Construct training samples from parsed Spacenit tile manifests.

Given the per-sensor tile dictionaries produced by
:func:`~spacenit.conversion.csv_parser.parse_tile_dataset`, this module
assembles :class:`TileSample` objects that each represent one training
example.  It also provides the raster-loading logic that reads the
appropriate crop from a GeoTIFF.
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import rasterio.windows
from pyproj import Transformer

from spacenit.ingestion.sensors import (
    COORDINATE_SYSTEM,
    FINEST_PIXEL_PITCH,
    TILE_EDGE_PIXELS,
    SensorRegistry,
    SensorSpec,
    TemporalCadence,
)

from .csv_parser import GridCell, SensorTile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TileSample – one training example
# ---------------------------------------------------------------------------


@dataclass
class TileSample:
    """Specification of a training example.

    The example corresponds to one :class:`GridCell` that appears in the
    dataset.  It includes all of the information needed to load sensor data
    at this cell, along with crops from coarser-grained tiles that contain
    this cell.

    Attributes:
        grid_cell: The grid position of this sample.
        cadence: Whether this sample covers an annual or biweekly period.
            Should never be :attr:`TemporalCadence.SNAPSHOT` since a training
            sample is always tied to a specific time range.
        sensors: Mapping from sensor to the tile that provides data for this
            sample.
    """

    grid_cell: GridCell
    cadence: TemporalCadence
    sensors: dict[SensorSpec, SensorTile]

    def compute_latlon(self) -> np.ndarray:
        """Compute the latitude / longitude of this sample's centre.

        Returns:
            A 1-D array ``[lat, lon]``.
        """
        grid_pitch = self.grid_cell.scale_factor * FINEST_PIXEL_PITCH
        x = (self.grid_cell.col + 0.5) * grid_pitch * TILE_EDGE_PIXELS
        y = (self.grid_cell.row + 0.5) * -grid_pitch * TILE_EDGE_PIXELS
        transformer = Transformer.from_crs(
            self.grid_cell.crs, COORDINATE_SYSTEM, always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def extract_timestamps(self) -> dict[SensorSpec, np.ndarray]:
        """Extract per-sensor timestamp arrays.

        Returns:
            A mapping from sensor to an ``(N, 3)`` array of
            ``[day, month, year]`` rows.
        """
        timestamps_dict: dict[SensorSpec, np.ndarray] = {}
        for sensor, tile in self.sensors.items():
            if sensor.has_temporal_axis:
                timestamps = [img.start_time for img in tile.images]
                dt = pd.to_datetime(timestamps)
                timestamps_dict[sensor] = np.array(
                    [dt.day, dt.month - 1, dt.year]
                ).T
        return timestamps_dict


# ---------------------------------------------------------------------------
# Tile -> sample assembly
# ---------------------------------------------------------------------------


def assemble_samples_from_tiles(
    tile_index: dict[SensorSpec, dict[TemporalCadence, list[SensorTile]]],
    supported_sensors: list[SensorSpec] | None = None,
) -> list[TileSample]:
    """Compute training samples from parsed per-sensor tile dictionaries.

    Args:
        tile_index: the parsed dataset from
            :func:`~spacenit.conversion.csv_parser.parse_tile_dataset`.
        supported_sensors: restrict samples to these sensors.  Defaults to
            every sensor registered in :class:`Sensor`.

    Returns:
        a list of :class:`TileSample` objects.
    """
    if supported_sensors is None:
        supported_sensors = SensorRegistry.all_specs()

    # Flatten (sensor, grid_cell, cadence) -> tile for fast lookup.
    flat_index: dict[tuple[SensorSpec, GridCell, TemporalCadence], SensorTile] = {}
    for sensor, cadence_tiles in tile_index.items():
        for cadence, tiles in cadence_tiles.items():
            for tile in tiles:
                flat_index[(sensor, tile.grid_cell, cadence)] = tile

    # Enumerate unique (grid_cell, cadence) pairs.
    unique_cells: set[tuple[GridCell, TemporalCadence]] = set()
    for sensor, grid_cell, cadence in flat_index:
        if cadence == TemporalCadence.SNAPSHOT:
            if grid_cell.scale_factor > 1:
                logger.debug(
                    f"ignoring static tile scale_factor={grid_cell.scale_factor} "
                    f"because it is coarser than the finest pitch for sensor {sensor.label}"
                )
                continue
            else:
                unique_cells.add((grid_cell, TemporalCadence.BIWEEKLY))
                unique_cells.add((grid_cell, TemporalCadence.ANNUAL))
        else:
            unique_cells.add((grid_cell, cadence))

    # Build a TileSample for each unique cell.
    samples: list[TileSample] = []
    for grid_cell, cadence in unique_cells:
        sample = TileSample(
            grid_cell=grid_cell,
            cadence=cadence,
            sensors={},
        )

        for sensor in tile_index:
            if sensor not in supported_sensors:
                logger.warning(
                    f"ignoring sensor {sensor.label} not in supported_sensors"
                )
                continue

            # Only use sensors at an equal or coarser resolution.
            if sensor.coverage_scale < sample.grid_cell.scale_factor:
                logger.debug(
                    f"ignoring sensor {sensor.label} with coverage_scale "
                    f"{sensor.coverage_scale} because it is finer than "
                    f"the sample grid cell scale_factor {sample.grid_cell.scale_factor}"
                )
                continue

            downscale = sensor.coverage_scale // sample.grid_cell.scale_factor

            # Determine the lookup cadence.
            if sensor.has_temporal_axis:
                lookup_cadence = sample.cadence
            else:
                lookup_cadence = TemporalCadence.SNAPSHOT

            # Downscale the grid cell for the lookup.
            sensor_cell = GridCell(
                crs=grid_cell.crs,
                scale_factor=sensor.coverage_scale,
                col=grid_cell.col // downscale,
                row=grid_cell.row // downscale,
            )

            key = (sensor, sensor_cell, lookup_cadence)
            if key not in flat_index:
                logger.debug(
                    f"ignoring sensor {sensor.label} because no tile found for key={key}"
                )
                continue

            sample.sensors[sensor] = flat_index[key]

        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------


def load_raster_for_sample(
    tile: SensorTile, sample: TileSample
) -> npt.NDArray:
    """Load the raster crop corresponding to a sample from a sensor tile.

    If the tile and sample share the same resolution the entire raster is
    loaded.  Otherwise, only the sub-region aligned with the sample is read
    via a windowed read.

    Args:
        tile: the sensor tile to read from.
        sample: the :class:`TileSample` that determines the crop region.

    Returns:
        the image as a NumPy array in ``TCHW`` layout (time on the first
        dimension).  Non-spatial sensors return a 2-D ``(T, C)`` array.
    """
    factor = tile.grid_cell.scale_factor // sample.grid_cell.scale_factor

    group_images = []
    for group, fname in tile.spectral_files.items():
        logger.debug(f"spectral_group={group}, fname={fname}")
        with fname.open("rb") as f:
            with rasterio.open(f) as raster:
                if raster.width != raster.height:
                    raise ValueError(
                        f"expected tile to be square but width={raster.width} "
                        f"!= height={raster.height}"
                    )

                # Non-spatial sensors: read the entire tile and flatten.
                if not tile.sensor.varies_in_space:
                    logger.debug(
                        f"reading entire tile {fname} for sensor {tile.sensor.label}"
                    )
                    image: npt.NDArray = raster.read()
                    image = image.reshape(-1, len(group.channel_names))
                    group_images.append(image)
                    continue

                # Spatial sensors: compute the sub-tile window.
                subtile_size = raster.width // factor
                col_offset = subtile_size * (sample.grid_cell.col % factor)
                row_offset = subtile_size * (sample.grid_cell.row % factor)

                window = rasterio.windows.Window(
                    col_off=col_offset,
                    row_off=row_offset,
                    width=subtile_size,
                    height=subtile_size,
                )
                logger.debug(f"reading window={window} from {fname}")
                image = raster.read(window=window)
                logger.debug(f"image.shape={image.shape}")

                # Resample to the expected tile edge length.
                desired_edge = int(
                    tile.sensor.expected_tile_edge() // factor
                )
                if desired_edge < subtile_size:
                    ds = subtile_size // desired_edge
                    image = image[:, ::ds, ::ds]
                elif desired_edge > subtile_size:
                    logger.debug(
                        f"desired_edge={desired_edge}, subtile_size={subtile_size}"
                    )
                    us = desired_edge // subtile_size
                    image = image.repeat(repeats=us, axis=1).repeat(
                        repeats=us, axis=2
                    )

                shape = (
                    -1,
                    len(group.channel_names),
                    desired_edge,
                    desired_edge,
                )
                image = image.reshape(shape)
                logger.debug(f"shape after scaling image.shape={image.shape}")
                group_images.append(image)

    return np.concatenate(group_images, axis=1)
