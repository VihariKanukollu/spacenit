"""Post-process ingested Sentinel-2 data into the SpaceNit dataset."""

from upath import UPath


def convert_sentinel2(window_path: UPath, spacenit_path: UPath) -> None:
    """Add Sentinel-2 multispectral data for this window to the SpaceNit dataset.

    Delegates to the multitemporal raster helpers to produce both frequent
    (two-week) and monthly (one-year) composites from Sentinel-2 L1C imagery
    layers in the rslearn window.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("Sentinel-2 converter not yet implemented for SpaceNit")
