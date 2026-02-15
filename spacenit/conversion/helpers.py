"""Utilities and base classes for the Spacenit tile-based dataset structure.

Provides path-resolution helpers for sensor data stored in the standard
Spacenit tiling layout.  Every sensor's raster files are organised by
coverage pitch, sensor label, temporal cadence, and grid coordinates.
"""

from datetime import datetime

from upath import UPath

from spacenit.ingestion.sensors import (
    FINEST_PIXEL_PITCH,
    SensorSpec,
    TemporalCadence,
)


class TileWindowMetadata:
    """Metadata describing a single tile window in the Spacenit grid.

    The window name encodes the CRS, column, row, resolution, and timestamp.
    These can also be derived from the rslearn window metadata.
    """

    def __init__(
        self,
        crs: str,
        resolution: float,
        col: int,
        row: int,
        time: datetime,
    ):
        """Create a new TileWindowMetadata.

        Args:
            crs: the UTM CRS that the tile is in.
            resolution: the pixel pitch of the grid this window sits on.
            col: the column of the tile in the grid.
            row: the row of the tile in the grid.
            time: the centre time used at this tile.
        """
        self.crs = crs
        self.resolution = resolution
        self.col = col
        self.row = row
        self.time = time

    def encode_window_name(self) -> str:
        """Encode the metadata back to a window name string."""
        return f"{self.crs}_{self.resolution}_{self.col}_{self.row}"

    def derive_scale_factor(self) -> int:
        """Derive the coverage scale factor from the stored resolution.

        See :pymod:`spacenit.ingestion.sensors` for scale-factor semantics.
        """
        return round(self.resolution / FINEST_PIXEL_PITCH)


def sensor_tile_directory(
    root: UPath, sensor: SensorSpec, cadence: TemporalCadence
) -> UPath:
    """Return the directory where raster files for a sensor are stored.

    Args:
        root: the Spacenit dataset root.
        sensor: the sensor specification.
        cadence: the temporal cadence, which determines the directory suffix.

    Returns:
        directory within *root* that holds this sensor's tiles.
    """
    suffix = cadence.file_suffix()
    dir_name = f"{sensor.compute_coverage_pitch()}_{sensor.label}{suffix}"
    return root / dir_name


def enumerate_tile_ids(
    root: UPath, sensor: SensorSpec, cadence: TemporalCadence
) -> list[str]:
    """List the tile IDs available for a given sensor and cadence.

    This is determined by listing the contents of the sensor directory.
    Manifest CSVs are not used.

    Args:
        root: the Spacenit dataset root.
        sensor: the sensor to check.
        cadence: the temporal cadence to check.

    Returns:
        a list of tile ID strings.
    """
    tile_dir = sensor_tile_directory(root, sensor, cadence)
    if not tile_dir.exists():
        return []

    tile_ids: list[str] = []
    for fname in tile_dir.iterdir():
        tile_ids.append(fname.name.split(".")[0])
    return tile_ids


def sensor_tile_filepath(
    root: UPath,
    sensor: SensorSpec,
    cadence: TemporalCadence,
    window: TileWindowMetadata,
    resolution: float,
    ext: str,
) -> UPath:
    """Return the file path for a specific tile of a sensor.

    Args:
        root: the Spacenit dataset root.
        sensor: the sensor specification.
        cadence: the temporal cadence of this data.
        window: metadata extracted from the tile window name.
        resolution: the pixel pitch of this band.  This should be a power of 2
            multiplied by the window resolution.
        ext: the filename extension, e.g. ``"tif"`` or ``"geojson"``.

    Returns:
        the full path to the raster file.
    """
    tile_dir = sensor_tile_directory(root, sensor, cadence)
    crs = window.crs
    col = window.col
    row = window.row
    fname = f"{crs}_{col}_{row}_{resolution}.{ext}"
    return tile_dir / fname
