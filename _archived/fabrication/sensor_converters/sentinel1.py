"""Post-process ingested Sentinel-1 SAR data into the SpaceNit dataset."""

from upath import UPath


def convert_sentinel1(window_path: UPath, spacenit_path: UPath) -> None:
    """Add Sentinel-1 SAR data for this window to the SpaceNit dataset.

    Delegates to the multitemporal raster helpers to produce both frequent
    (two-week) and monthly (one-year) composites from Sentinel-1 imagery
    layers in the rslearn window. Errors during conversion are caught and
    logged rather than propagated.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("Sentinel-1 converter not yet implemented for SpaceNit")
