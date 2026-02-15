"""Post-process ingested CDL crop type data into the SpaceNit dataset."""

from upath import UPath


def convert_cdl(window_path: UPath, spacenit_path: UPath) -> None:
    """Add CDL crop type data for this window to the SpaceNit dataset.

    Reads CDL (Cropland Data Layer) raster data from an rslearn window,
    validates that no background/nodata pixels are present, and writes the
    result as a GeoTIFF into the SpaceNit dataset along with per-example
    metadata CSV.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("CDL converter not yet implemented for SpaceNit")
