"""Post-process ingested NAIP data into the SpaceNit dataset."""

from upath import UPath


def convert_naip(window_path: UPath, spacenit_path: UPath) -> None:
    """Add NAIP aerial imagery data for this window to the SpaceNit dataset.

    Reads the NAIP mosaic from an rslearn window at 0.625 m/pixel resolution,
    determines the time range from the item metadata, and writes the GeoTIFF
    with per-example metadata CSV.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("NAIP converter not yet implemented for SpaceNit")
