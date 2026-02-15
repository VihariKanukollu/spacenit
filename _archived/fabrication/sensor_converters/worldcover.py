"""Post-process ingested WorldCover land-cover data into the SpaceNit dataset."""

from upath import UPath


def convert_worldcover(window_path: UPath, spacenit_path: UPath) -> None:
    """Add ESA WorldCover land-cover data for this window to the SpaceNit dataset.

    Reads the WorldCover classification raster from an rslearn window and
    writes it as a GeoTIFF with per-example metadata CSV. Uses a fixed time
    range of 2021-01-01 to 2022-01-01.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("WorldCover converter not yet implemented for SpaceNit")
