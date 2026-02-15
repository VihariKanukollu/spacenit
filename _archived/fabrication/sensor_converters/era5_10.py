"""Post-process ingested ERA5_10 data into the SpaceNit dataset."""

from upath import UPath


def convert_era5_10(window_path: UPath, spacenit_path: UPath) -> None:
    """Add ERA5 data at 10 m tiling for this window to the SpaceNit dataset.

    Similar to the ERA5 converter but operates on the 10 m/pixel tiling grid,
    adjusting projection and bounds to match the band set resolution. Produces
    both one-year and two-week GeoTIFFs with metadata.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("ERA5_10 converter not yet implemented for SpaceNit")
