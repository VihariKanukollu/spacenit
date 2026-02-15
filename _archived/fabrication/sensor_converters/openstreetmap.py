"""Post-process ingested OpenStreetMap vector data into the SpaceNit dataset.

OpenStreetMap is vector data, so we want to keep the precision of the data as high as
possible, but the data size (i.e. bytes) is also small enough that we can store it
under the 10 m/pixel tiles without needing too much storage space.

We use the 10 m/pixel grid, but store it with 16x zoomed in coordinates (meaning
the coordinates actually match those of the 0.625 m/pixel tiles). This way we can use
the data for training even at coarser resolution.
"""

from upath import UPath


def convert_openstreetmap(window_path: UPath, spacenit_path: UPath) -> None:
    """Add OpenStreetMap vector data for this window to the SpaceNit dataset.

    Loads GeoJSON features from the rslearn window, concatenates features
    across all item groups, and writes the result as a GeoJSON file with
    per-example metadata CSV.

    Args:
        window_path: the rslearn window directory to read data from.
        spacenit_path: SpaceNit dataset path to write to.
    """
    raise NotImplementedError(
        "OpenStreetMap converter not yet implemented for SpaceNit"
    )
