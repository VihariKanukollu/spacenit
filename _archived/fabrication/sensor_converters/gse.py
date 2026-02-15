"""Post-process ingested Google Satellite Embedding data into the SpaceNit dataset."""

from upath import UPath


def convert_gse(window_path: UPath, spacenit_path: UPath) -> None:
    """Add Google Satellite Embedding data for this window to the SpaceNit dataset.

    Reads GSE mosaic raster from an rslearn window, computes the time range
    from the earliest item timestamp plus one year, and writes the GeoTIFF
    with per-example metadata CSV.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("GSE converter not yet implemented for SpaceNit")
