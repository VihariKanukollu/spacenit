"""Helper functions to convert multitemporal rasters into the SpaceNit dataset.

Provides shared logic for sensors that produce both frequent (two-week) and
monthly (one-year) composites, such as Landsat, Sentinel-1, and Sentinel-2.
"""

from upath import UPath

from spacenit.ingestion.modalities import SensorSpec


def get_adjusted_projection_and_bounds(
    modality: SensorSpec,
    band_set: object,
    projection: object,
    window_bounds: tuple[int, int, int, int],
) -> tuple[object, tuple[int, int, int, int]]:
    """Compute projection and bounds adjusted for a band set's resolution.

    Some bands may be stored at lower resolutions than the window bounds. Given
    the window projection and bounds, computes the coarser projection
    corresponding to the band set, as well as the appropriate bounds in pixel
    coordinates under that projection.

    Args:
        modality: the SensorSpec specifying a grid resolution.
        band_set: the BandSet specifying an image resolution.
        projection: the projection of the window.
        window_bounds: the bounds of the window.

    Returns:
        Tuple of (adjusted_projection, adjusted_bounds).
    """
    raise NotImplementedError(
        "Multitemporal raster helpers not yet implemented for SpaceNit"
    )


def convert_freq(
    window_path: UPath,
    spacenit_path: UPath,
    layer_name: str,
    modality: SensorSpec,
    missing_okay: bool = False,
    unprepared_okay: bool = False,
) -> None:
    """Add frequent (two-week) data from this window to the SpaceNit dataset.

    Reads individual images and their timestamps from the rslearn window,
    stacks them, and writes the result as a GeoTIFF with metadata CSV.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
        layer_name: the name of the layer containing frequent data.
        modality: the modality.
        missing_okay: whether missing images are acceptable.
        unprepared_okay: whether unprepared windows should be silently skipped.
    """
    raise NotImplementedError(
        "Multitemporal raster helpers not yet implemented for SpaceNit"
    )


def convert_monthly(
    window_path: UPath,
    spacenit_path: UPath,
    layer_prefix: str,
    modality: SensorSpec,
) -> None:
    """Add monthly (one-year) data from this window to the SpaceNit dataset.

    Reads monthly mosaic layers (suffixed _mo01 through _mo12), stacks them,
    and writes the result as a GeoTIFF with metadata CSV.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
        layer_prefix: the prefix for the monthly layer names.
        modality: the modality.
    """
    raise NotImplementedError(
        "Multitemporal raster helpers not yet implemented for SpaceNit"
    )
