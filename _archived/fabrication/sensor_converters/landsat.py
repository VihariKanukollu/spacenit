"""Post-process ingested Landsat data into the SpaceNit dataset."""

from upath import UPath


def convert_landsat(window_path: UPath, spacenit_path: UPath) -> None:
    """Add Landsat data for this window to the SpaceNit dataset.

    Delegates to the multitemporal raster helpers to produce both frequent
    (two-week) and monthly (one-year) composites from Landsat imagery layers
    in the rslearn window.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("Landsat converter not yet implemented for SpaceNit")
