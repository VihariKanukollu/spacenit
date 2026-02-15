"""Post-process ingested ERA5 data into the SpaceNit dataset."""

from upath import UPath


def convert_era5(window_path: UPath, spacenit_path: UPath) -> None:
    """Add ERA5 climate reanalysis data for this window to the SpaceNit dataset.

    Reads monthly ERA5 images from an rslearn window, stacks them into a
    one-year composite and extracts the two-week slice matching the window's
    time range. Both are written as GeoTIFFs with accompanying metadata CSVs.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("ERA5 converter not yet implemented for SpaceNit")
