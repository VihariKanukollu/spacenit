"""Dataset conversion helpers (CSV / H5 / tile sampling).

Public API
----------
.. autosummary::

    helpers.TileWindowMetadata
    helpers.sensor_tile_directory
    helpers.enumerate_tile_ids
    helpers.sensor_tile_filepath

    csv_parser.SensorImage
    csv_parser.GridCell
    csv_parser.SensorTile
    csv_parser.parse_sensor_csv
    csv_parser.parse_tile_dataset

    tile_sampler.TileSample
    tile_sampler.assemble_samples_from_tiles
    tile_sampler.load_raster_for_sample

    h5_converter.GeoTileH5WriterConfig
    h5_converter.GeoTileH5Writer
"""

from spacenit.conversion.helpers import (
    TileWindowMetadata,
    enumerate_tile_ids,
    sensor_tile_directory,
    sensor_tile_filepath,
)
from spacenit.conversion.csv_parser import (
    GridCell,
    SensorImage,
    SensorTile,
    parse_sensor_csv,
    parse_tile_dataset,
)
from spacenit.conversion.tile_sampler import (
    TileSample,
    assemble_samples_from_tiles,
    load_raster_for_sample,
)
from spacenit.conversion.h5_converter import (
    GeoTileH5Writer,
    GeoTileH5WriterConfig,
)

__all__ = [
    # helpers
    "TileWindowMetadata",
    "sensor_tile_directory",
    "enumerate_tile_ids",
    "sensor_tile_filepath",
    # csv_parser
    "SensorImage",
    "GridCell",
    "SensorTile",
    "parse_sensor_csv",
    "parse_tile_dataset",
    # tile_sampler
    "TileSample",
    "assemble_samples_from_tiles",
    "load_raster_for_sample",
    # h5_converter
    "GeoTileH5WriterConfig",
    "GeoTileH5Writer",
]
