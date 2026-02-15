"""Post-process ingested SRTM elevation data into the SpaceNit dataset."""

from upath import UPath


def convert_srtm(window_path: UPath, spacenit_path: UPath) -> None:
    """Add SRTM elevation data for this window to the SpaceNit dataset.

    Reads the SRTM digital elevation model raster from an rslearn window and
    writes it as a GeoTIFF with per-example metadata CSV. Uses a fixed time
    range of 2000-01-01 to 2001-01-01 for the SRTM mission period.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("SRTM converter not yet implemented for SpaceNit")
