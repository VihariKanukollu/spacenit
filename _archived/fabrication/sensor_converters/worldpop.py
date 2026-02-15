"""Post-process ingested WorldPop population data into the SpaceNit dataset."""

from upath import UPath


def convert_worldpop(window_path: UPath, spacenit_path: UPath) -> None:
    """Add WorldPop population density data for this window to the SpaceNit dataset.

    Reads the WorldPop raster from an rslearn window, clips negative values
    (NODATA = -99999) to zero, skips fully-zero tiles, and writes the result
    as a GeoTIFF with per-example metadata CSV. Uses a fixed time range of
    2020-01-01 to 2021-01-01.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("WorldPop converter not yet implemented for SpaceNit")
