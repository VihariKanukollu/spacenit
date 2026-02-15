"""Backward-compatibility shim -- all definitions live in ``sensors.py`` now.

Existing code that does ``from spacenit.ingestion.modalities import ...`` will
continue to work.  New code should import from ``spacenit.ingestion.sensors``.
"""

from spacenit.ingestion.sensors import (  # noqa: F401 -- re-exports
    ABSENT_INDICATOR,
    ANNUAL_STEP_COUNT,
    COORDINATE_SYSTEM,
    FINEST_PIXEL_PITCH,
    GEOGRAPHIC_COORDS,
    MAX_TEMPORAL_DEPTH,
    REFERENCE_GROUND_RESOLUTION,
    SAR_NODATA_SENTINEL,
    TEMPORAL_FIELDS,
    TILE_EDGE_PIXELS,
    TemporalCadence,
    SpectralGroup,
    SensorSpec,
    SensorRegistry,
    Sensor,
    compute_pixel_pitch,
    specs_from_labels,
    # Built-in sensor constants
    SENTINEL2_L2A,
    SENTINEL1,
    SENTINEL2,
    LANDSAT,
    NAIP,
    NAIP_10,
    WORLDCOVER,
    WORLDCEREAL,
    SRTM,
    OPENSTREETMAP,
    OPENSTREETMAP_RASTER,
    ERA5,
    ERA5_10,
    LATLON,
    GSE,
    CDL,
    WORLDPOP,
    WRI_CANOPY_HEIGHT_MAP,
)
