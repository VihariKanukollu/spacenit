"""Create rasterized OpenStreetMap data from vector OpenStreetMap in the SpaceNit dataset.

Rasterizes GeoJSON features into a multi-channel binary mask (one channel per
OSM category) at 4x the tile resolution (1024x1024 output from 256x256 tiles).
"""

from upath import UPath

# OSM feature categories that are rasterized into separate channels.
CATEGORIES = [
    "aerialway_pylon",
    "aerodrome",
    "airstrip",
    "amenity_fuel",
    "building",
    "chimney",
    "communications_tower",
    "crane",
    "flagpole",
    "fountain",
    "generator_wind",
    "helipad",
    "highway",
    "leisure",
    "lighthouse",
    "obelisk",
    "observatory",
    "parking",
    "petroleum_well",
    "power_plant",
    "power_substation",
    "power_tower",
    "river",
    "runway",
    "satellite_dish",
    "silo",
    "storage_tank",
    "taxiway",
    "water_tower",
    "works",
]


def rasterize_openstreetmap(spacenit_path: UPath, in_fname: UPath) -> None:
    """Rasterize OpenStreetMap GeoJSON data into a multi-channel raster.

    Parses the input GeoJSON, draws polygons / line strings / points onto a
    (len(CATEGORIES), 1024, 1024) uint8 array, and writes the result as a
    GeoTIFF in the SpaceNit dataset.

    Args:
        spacenit_path: path to SpaceNit dataset where OpenStreetMap vector
            data has been written.
        in_fname: the input filename containing the GeoJSON data.
    """
    raise NotImplementedError(
        "OpenStreetMap rasterizer not yet implemented for SpaceNit"
    )
