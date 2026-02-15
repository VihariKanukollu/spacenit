"""Post-process ingested NAIP data at 10 m tiling into the SpaceNit dataset."""

from upath import UPath


def convert_naip_10(window_path: UPath, spacenit_path: UPath) -> None:
    """Add NAIP data at 10 m/pixel tiling for this window to the SpaceNit dataset.

    Similar to the standard NAIP converter but operates on the 10 m/pixel tiling
    grid. The actual image data is still at 0.625 m/pixel resolution (4096x4096
    pixels), but the tile grid alignment uses 10 m cells. Adjusts projection
    and bounds accordingly.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("NAIP_10 converter not yet implemented for SpaceNit")
