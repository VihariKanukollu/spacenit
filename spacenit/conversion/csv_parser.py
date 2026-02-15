"""Parse the Spacenit tile dataset from per-sensor manifest CSVs.

Each sensor's tiles are described by a CSV that records grid coordinates,
timestamps, and image indices.  This module reads those CSVs and produces
structured :class:`SensorTile` objects that downstream code can consume.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime

from upath import UPath

from spacenit.ingestion.sensors import (
    FINEST_PIXEL_PITCH,
    SensorRegistry,
    SensorSpec,
    SpectralGroup,
    TemporalCadence,
)

from .helpers import TileWindowMetadata, sensor_tile_filepath

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorImage:
    """A single observation within a sensor tile's time series.

    Attributes:
        start_time: Beginning of the observation window.
        end_time: End of the observation window.
    """

    start_time: datetime
    end_time: datetime

    def __eq__(self, other: object) -> bool:
        """Check equality based on observation time bounds."""
        if not isinstance(other, SensorImage):
            return False
        return self.start_time == other.start_time and self.end_time == other.end_time


@dataclass(frozen=True)
class GridCell:
    """The position of a tile along a grid of a certain resolution.

    Attributes:
        crs: The coordinate reference system, e.g. ``"EPSG:32610"``.
        scale_factor: The coverage scale factor relative to
            :data:`FINEST_PIXEL_PITCH`.
        col: Column index along the grid.
        row: Row index along the grid.
    """

    crs: str
    scale_factor: int
    col: int
    row: int


@dataclass
class SensorTile:
    """All information about one tile pertaining to a sensor.

    Attributes:
        grid_cell: The grid position of this tile.
        images: Ordered list of observations in the tile's time series.
        centre_time: The reference time that defines time ranges for this tile.
        spectral_files: Mapping from spectral group to the file containing it.
        sensor: The sensor specification this tile belongs to.
    """

    grid_cell: GridCell
    images: list[SensorImage]
    centre_time: datetime
    spectral_files: dict[SpectralGroup, UPath]
    sensor: SensorSpec

    def flat_channel_names(self) -> list[str]:
        """Return all channel names as a flat list.

        The order matches the band-concatenation order used when loading
        spectral groups into a single tensor.
        """
        channels: list[str] = []
        for group in self.spectral_files:
            channels.extend(group.channel_names)
        return channels


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_sensor_csv(
    root: UPath,
    sensor: SensorSpec,
    cadence: TemporalCadence,
    csv_path: UPath,
) -> list[SensorTile]:
    """Parse a manifest CSV for one sensor and temporal cadence.

    Args:
        root: the Spacenit dataset root path.
        sensor: the sensor specification to parse.
        cadence: the temporal cadence of this CSV.
        csv_path: path to the CSV file.

    Returns:
        list of :class:`SensorTile` objects extracted from the CSV.
    """
    sensor_tiles: dict[GridCell, SensorTile] = {}

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            grid_cell = GridCell(
                crs=csv_row["crs"],
                scale_factor=sensor.coverage_scale,
                col=int(csv_row["col"]),
                row=int(csv_row["row"]),
            )
            image = SensorImage(
                start_time=datetime.fromisoformat(csv_row["start_time"]),
                end_time=datetime.fromisoformat(csv_row["end_time"]),
            )
            image_idx = int(csv_row["image_idx"])

            if grid_cell not in sensor_tiles:
                sensor_tiles[grid_cell] = SensorTile(
                    grid_cell=grid_cell,
                    images=[],
                    centre_time=datetime.fromisoformat(csv_row["tile_time"]),
                    spectral_files={},
                    sensor=sensor,
                )

            # Image indices must appear in order in the CSV.
            if image_idx != len(sensor_tiles[grid_cell].images):
                # Rare edge case: duplicate timestamp entries in the original
                # rslearn dataset.  Skip the duplicate rather than raising.
                continue
            sensor_tiles[grid_cell].images.append(image)

    # Resolve file paths for each spectral group.
    for tile in sensor_tiles.values():
        gc = tile.grid_cell
        window = TileWindowMetadata(
            crs=gc.crs,
            resolution=FINEST_PIXEL_PITCH * gc.scale_factor,
            col=gc.col,
            row=gc.row,
            time=tile.centre_time,
        )
        for group in sensor.spectral_groups:
            fname = sensor_tile_filepath(
                root,
                sensor,
                cadence,
                window,
                group.compute_pixel_pitch(),
                "tif",
            )
            tile.spectral_files[group] = fname

    return list(sensor_tiles.values())


def parse_tile_dataset(
    root: UPath,
    supported_sensors: list[SensorSpec] | None = None,
) -> dict[SensorSpec, dict[TemporalCadence, list[SensorTile]]]:
    """Parse all per-sensor manifest CSVs in a Spacenit tile dataset.

    Args:
        root: the dataset root directory containing CSVs and tile
            sub-directories.
        supported_sensors: restrict parsing to these sensors.  Defaults to
            every sensor registered in :class:`Sensor`.

    Returns:
        a mapping from sensor -> temporal cadence -> list of tiles.
    """
    if supported_sensors is None:
        supported_sensors = SensorRegistry.all_specs()

    tiles: dict[SensorSpec, dict[TemporalCadence, list[SensorTile]]] = {}

    for sensor in SensorRegistry.all_specs():
        if sensor.skip_csv_parsing:
            continue
        if sensor not in supported_sensors:
            logger.warning(
                f"ignoring sensor {sensor.label} not in supported_sensors"
            )
            continue

        if sensor.has_temporal_axis:
            cadences = [TemporalCadence.ANNUAL]
        else:
            cadences = [TemporalCadence.SNAPSHOT]

        tiles[sensor] = {}
        for cadence in cadences:
            coverage_pitch = sensor.compute_coverage_pitch()
            csv_fname = (
                root / f"{coverage_pitch}_{sensor.label}{cadence.file_suffix()}.csv"
            )
            logger.debug(f"Parsing {sensor.label} {cadence} {csv_fname}")
            tiles[sensor][cadence] = parse_sensor_csv(
                root,
                sensor,
                cadence,
                csv_fname,
            )

    return tiles
