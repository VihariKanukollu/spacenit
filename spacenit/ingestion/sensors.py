"""Satellite sensor definitions and registry for SpaceNit.

Provides :class:`SensorSpec` (immutable description of a satellite sensor or
derived product), :class:`SensorRegistry` (the single source of truth for
which sensors the system knows about), and all built-in sensor constants.

Usage::

    from spacenit.ingestion.sensors import SensorRegistry, SENTINEL2_L2A

    spec = SensorRegistry.get("sentinel2_l2a")
    assert spec is SENTINEL2_L2A
    print(spec.total_channels)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

# ---------------------------------------------------------------------------
# Global pixel / grid constants
# ---------------------------------------------------------------------------

# Finest native pixel pitch (metres per pixel) that the tiling grid is built
# around.  Every other resolution is expressed as an integer multiple of this.
FINEST_PIXEL_PITCH: float = 0.625

# Number of pixels along one edge of the standard square tile.
TILE_EDGE_PIXELS: int = 256

# Coordinate reference system used for all geospatial data.
COORDINATE_SYSTEM: str = "EPSG:4326"

# Sentinel value written into raster cells that carry no valid observation.
ABSENT_INDICATOR: int = -99999

# Upper bound on the number of time-steps kept per sample.
MAX_TEMPORAL_DEPTH: int = 12

# Reference ground-sample distance (metres) used as the baseline for
# computing scale factors.
REFERENCE_GROUND_RESOLUTION: int = 10

# Nodata marker specific to Synthetic Aperture Radar (SAR) imagery.
SAR_NODATA_SENTINEL: int = -32768

# How many discrete time-steps make up one annual cycle.
ANNUAL_STEP_COUNT: int = 12

# Geographic coordinate field names.
GEOGRAPHIC_COORDS: list[str] = ["lat", "lon"]

# Temporal metadata field names.
TEMPORAL_FIELDS: list[str] = ["day", "month", "year"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_pixel_pitch(scale_factor: int) -> float | int:
    """Derive the effective pixel pitch from a scale factor.

    When the result is a whole number it is returned as ``int`` so that
    file-naming conventions that embed the resolution as an integer work
    without modification.
    """
    pitch = FINEST_PIXEL_PITCH * scale_factor
    if float(int(pitch)) == pitch:
        return int(pitch)
    return pitch


# ---------------------------------------------------------------------------
# SpectralGroup -- a set of channels stored at the same resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralGroup:
    """A collection of spectral channels sharing a common spatial resolution.

    Most sensors expose a single group, but multi-resolution instruments
    (e.g. Sentinel-2) are represented by several groups.

    Attributes:
        channel_names: Ordered list of channel / band identifiers.
        scale_factor: Multiplier applied to ``FINEST_PIXEL_PITCH`` to obtain
            the native pixel pitch for these channels.  A value of ``0``
            indicates non-spatial data.
    """

    channel_names: list[str]
    scale_factor: int

    def __hash__(self) -> int:
        return hash((tuple(self.channel_names), self.scale_factor))

    def compute_pixel_pitch(self) -> float:
        return compute_pixel_pitch(self.scale_factor)

    def expected_tile_edge(self, coverage_scale: int) -> int:
        """Compute the expected tile edge length for these channels."""
        return TILE_EDGE_PIXELS // (self.scale_factor // coverage_scale)


# ---------------------------------------------------------------------------
# TemporalCadence
# ---------------------------------------------------------------------------


class TemporalCadence(str, Enum):
    """Describes the temporal sampling pattern of a data source."""

    SNAPSHOT = "static"
    ANNUAL = "year"
    BIWEEKLY = "two_week"

    def file_suffix(self) -> str:
        if self == TemporalCadence.SNAPSHOT:
            return ""
        if self == TemporalCadence.ANNUAL:
            return "_monthly"
        if self == TemporalCadence.BIWEEKLY:
            return "_freq"
        raise ValueError("Unrecognised temporal cadence")


# ---------------------------------------------------------------------------
# SensorSpec -- full specification for one sensor / data product
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorSpec:
    """Complete specification of a single satellite sensor or derived product.

    Attributes:
        label: Short unique identifier (e.g. ``"sentinel2_l2a"``).
        coverage_scale: Factor expressing how much more ground area a tile
            covers compared with a ``TILE_EDGE_PIXELS`` tile at the finest
            pixel pitch.
        spectral_groups: Ordered list of :class:`SpectralGroup` instances.
        has_temporal_axis: ``True`` when the data delivers a time series.
        skip_csv_parsing: ``True`` when the sensor is not loaded from CSV.
        tile_size_multiplier: Scaling factor for stored tile edge length.
            Negative values indicate a *divisor*.
    """

    label: str
    coverage_scale: int
    spectral_groups: list[SpectralGroup]
    has_temporal_axis: bool
    skip_csv_parsing: bool
    tile_size_multiplier: int = 1

    def __hash__(self) -> int:
        return hash(self.label)

    def compute_coverage_pitch(self) -> float:
        return compute_pixel_pitch(self.coverage_scale)

    def group_indices(self) -> list[list[int]]:
        """Map each spectral group to a list of flat channel indices."""
        indices: list[list[int]] = []
        offset = 0
        for group in self.spectral_groups:
            n = len(group.channel_names)
            indices.append(list(range(offset, offset + n)))
            offset += n
        return indices

    @property
    def all_channel_names(self) -> list[str]:
        """Flat, ordered list of every channel name across all groups."""
        return sum((list(g.channel_names) for g in self.spectral_groups), [])

    @property
    def group_count(self) -> int:
        """Number of spectral groups in this sensor."""
        return len(self.spectral_groups)

    @property
    def total_channels(self) -> int:
        """Total number of channels across all spectral groups."""
        return sum(len(g.channel_names) for g in self.spectral_groups)

    def expected_tile_edge(self) -> int:
        """Compute the stored tile edge length in pixels."""
        if self.tile_size_multiplier < 0:
            return TILE_EDGE_PIXELS // abs(self.tile_size_multiplier)
        return TILE_EDGE_PIXELS * self.tile_size_multiplier

    @property
    def varies_in_space(self) -> bool:
        return self.compute_coverage_pitch() > 0 and self.expected_tile_edge() > 1

    @property
    def varies_in_space_and_time(self) -> bool:
        return self.varies_in_space and self.has_temporal_axis

    @property
    def varies_in_space_only(self) -> bool:
        return self.varies_in_space and not self.has_temporal_axis

    @property
    def varies_in_time_only(self) -> bool:
        return not self.varies_in_space and self.has_temporal_axis

    @property
    def is_constant(self) -> bool:
        return not self.varies_in_space and not self.has_temporal_axis


# ---------------------------------------------------------------------------
# SensorRegistry -- the single source of truth
# ---------------------------------------------------------------------------


class SensorRegistry:
    """Central registry of all known :class:`SensorSpec` instances.

    Sensors are registered at module-load time via :meth:`register`.  The
    registry is the **single source of truth** for which sensors exist --
    data containers (:class:`~spacenit.structures.GeoSample`, etc.) validate
    keys against it.

    Usage::

        spec = SensorRegistry.get("sentinel2_l2a")
        for s in SensorRegistry:
            print(s.label)
    """

    _specs: OrderedDict[str, SensorSpec] = OrderedDict()

    @classmethod
    def register(cls, spec: SensorSpec) -> SensorSpec:
        """Add a sensor to the registry and return it (for assignment)."""
        if spec.label in cls._specs:
            raise ValueError(f"Sensor '{spec.label}' is already registered")
        cls._specs[spec.label] = spec
        return spec

    @classmethod
    def get(cls, label: str) -> SensorSpec:
        """Look up a :class:`SensorSpec` by its label.

        Raises:
            KeyError: If *label* is not registered.
        """
        try:
            return cls._specs[label]
        except KeyError:
            raise KeyError(
                f"Unknown sensor '{label}'. "
                f"Registered: {list(cls._specs.keys())}"
            ) from None

    @classmethod
    def all_specs(cls) -> list[SensorSpec]:
        """Return every registered :class:`SensorSpec` in registration order."""
        return list(cls._specs.values())

    @classmethod
    def all_labels(cls) -> list[str]:
        """Return every registered sensor label in registration order."""
        return list(cls._specs.keys())

    @classmethod
    def contains(cls, label: str) -> bool:
        """Return ``True`` if *label* is a registered sensor."""
        return label in cls._specs

    @classmethod
    def __iter__(cls) -> Iterator[SensorSpec]:
        return iter(cls._specs.values())

    @classmethod
    def __len__(cls) -> int:
        return len(cls._specs)


# ---------------------------------------------------------------------------
# Built-in sensor definitions (auto-registered)
# ---------------------------------------------------------------------------

SENTINEL2_L2A = SensorRegistry.register(
    SensorSpec(
        label="sentinel2_l2a",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup(["B02", "B03", "B04", "B08"], 16),
            SpectralGroup(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            SpectralGroup(["B01", "B09"], 64),
        ],
        has_temporal_axis=True,
        skip_csv_parsing=False,
    )
)

SENTINEL1 = SensorRegistry.register(
    SensorSpec(
        label="sentinel1",
        coverage_scale=16,
        spectral_groups=[SpectralGroup(["vv", "vh"], 16)],
        has_temporal_axis=True,
        skip_csv_parsing=False,
    )
)

SENTINEL2 = SensorRegistry.register(
    SensorSpec(
        label="sentinel2",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup(["B02", "B03", "B04", "B08"], 16),
            SpectralGroup(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            SpectralGroup(["B01", "B09", "B10"], 64),
        ],
        has_temporal_axis=True,
        skip_csv_parsing=False,
    )
)

LANDSAT = SensorRegistry.register(
    SensorSpec(
        label="landsat",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup(["B8"], 16),
            SpectralGroup(
                ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 32
            ),
        ],
        has_temporal_axis=True,
        skip_csv_parsing=False,
    )
)

NAIP = SensorRegistry.register(
    SensorSpec(
        label="naip",
        coverage_scale=1,
        spectral_groups=[SpectralGroup(["R", "G", "B", "IR"], 1)],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

NAIP_10 = SensorRegistry.register(
    SensorSpec(
        label="naip_10",
        coverage_scale=16,
        spectral_groups=[SpectralGroup(["R", "G", "B", "IR"], 1)],
        has_temporal_axis=False,
        skip_csv_parsing=False,
        tile_size_multiplier=4,
    )
)

WORLDCOVER = SensorRegistry.register(
    SensorSpec(
        label="worldcover",
        coverage_scale=16,
        spectral_groups=[SpectralGroup(["B1"], 16)],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

WORLDCEREAL = SensorRegistry.register(
    SensorSpec(
        label="worldcereal",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup(
                [
                    "tc-annual-temporarycrops-classification",
                    "tc-maize-main-irrigation-classification",
                    "tc-maize-main-maize-classification",
                    "tc-maize-second-irrigation-classification",
                    "tc-maize-second-maize-classification",
                    "tc-springcereals-springcereals-classification",
                    "tc-wintercereals-irrigation-classification",
                    "tc-wintercereals-wintercereals-classification",
                ],
                16,
            )
        ],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

SRTM = SensorRegistry.register(
    SensorSpec(
        label="srtm",
        coverage_scale=16,
        spectral_groups=[SpectralGroup(["srtm"], 16)],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

OPENSTREETMAP = SensorRegistry.register(
    SensorSpec(
        label="openstreetmap",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup(
                [
                    "aerialway_pylon", "aerodrome", "airstrip", "amenity_fuel",
                    "building", "chimney", "communications_tower", "crane",
                    "flagpole", "fountain", "generator_wind", "helipad",
                    "highway", "leisure", "lighthouse", "obelisk",
                    "observatory", "parking", "petroleum_well", "power_plant",
                    "power_substation", "power_tower", "river", "runway",
                    "satellite_dish", "silo", "storage_tank", "taxiway",
                    "water_tower", "works",
                ],
                1,
            )
        ],
        has_temporal_axis=False,
        skip_csv_parsing=True,
    )
)

OPENSTREETMAP_RASTER = SensorRegistry.register(
    SensorSpec(
        label="openstreetmap_raster",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup(
                [
                    "aerialway_pylon", "aerodrome", "airstrip", "amenity_fuel",
                    "building", "chimney", "communications_tower", "crane",
                    "flagpole", "fountain", "generator_wind", "helipad",
                    "highway", "leisure", "lighthouse", "obelisk",
                    "observatory", "parking", "petroleum_well", "power_plant",
                    "power_substation", "power_tower", "river", "runway",
                    "satellite_dish", "silo", "storage_tank", "taxiway",
                    "water_tower", "works",
                ],
                4,
            )
        ],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

ERA5 = SensorRegistry.register(
    SensorSpec(
        label="era5",
        coverage_scale=256,
        spectral_groups=[
            SpectralGroup(
                [
                    "2m-temperature", "2m-dewpoint-temperature",
                    "surface-pressure", "10m-u-component-of-wind",
                    "10m-v-component-of-wind", "total-precipitation",
                ],
                256,
            ),
        ],
        has_temporal_axis=True,
        skip_csv_parsing=True,
    )
)

ERA5_10 = SensorRegistry.register(
    SensorSpec(
        label="era5_10",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup(
                [
                    "2m-temperature", "2m-dewpoint-temperature",
                    "surface-pressure", "10m-u-component-of-wind",
                    "10m-v-component-of-wind", "total-precipitation",
                ],
                4096,
            ),
        ],
        has_temporal_axis=True,
        skip_csv_parsing=False,
        tile_size_multiplier=-256,
    )
)

LATLON = SensorRegistry.register(
    SensorSpec(
        label="latlon",
        coverage_scale=0,
        spectral_groups=[SpectralGroup(["lat", "lon"], 0)],
        has_temporal_axis=False,
        skip_csv_parsing=True,
    )
)

GSE = SensorRegistry.register(
    SensorSpec(
        label="gse",
        coverage_scale=16,
        spectral_groups=[
            SpectralGroup([f"A{idx:02d}" for idx in range(64)], 16),
        ],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

CDL = SensorRegistry.register(
    SensorSpec(
        label="cdl",
        coverage_scale=16,
        spectral_groups=[SpectralGroup(["cdl"], 16)],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

WORLDPOP = SensorRegistry.register(
    SensorSpec(
        label="worldpop",
        coverage_scale=16,
        spectral_groups=[SpectralGroup(["B1"], 16)],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)

WRI_CANOPY_HEIGHT_MAP = SensorRegistry.register(
    SensorSpec(
        label="wri_canopy_height_map",
        coverage_scale=16,
        spectral_groups=[SpectralGroup(["B1"], 16)],
        has_temporal_axis=False,
        skip_csv_parsing=False,
    )
)
