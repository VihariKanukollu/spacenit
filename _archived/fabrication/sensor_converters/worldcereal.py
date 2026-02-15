"""Post-process ingested WorldCereal data into the SpaceNit dataset."""

from upath import UPath


def convert_worldcereal(window_path: UPath, spacenit_path: UPath) -> None:
    """Add WorldCereal crop type data for this window to the SpaceNit dataset.

    Reads per-band WorldCereal rasters from an rslearn window, concatenates
    them (filling missing bands with zeros), clamps values above 100 to 0
    (handling nodata/not-cropland codes), and writes the result as a GeoTIFF
    with per-example metadata CSV.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError("WorldCereal converter not yet implemented for SpaceNit")
