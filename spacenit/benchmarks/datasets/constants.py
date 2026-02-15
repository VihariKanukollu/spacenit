"""Useful constants for benchmarks."""

from spacenit.data.constants import Sensor

BENCH_S2_BAND_NAMES = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "10 - SWIR - Cirrus",
    "11 - SWIR",
    "12 - SWIR",
]

BENCH_S2_L2A_BAND_NAMES = [b for b in BENCH_S2_BAND_NAMES if b != "10 - SWIR - Cirrus"]

BENCH_S1_BAND_NAMES = [
    "vv",
    "vh",
]

BENCH_L8_BAND_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
]

BENCH_SRTM_BAND_NAMES = ["srtm"]


# Get the corresponding index from either Sentinel2 L1C or L2A band names
def _bench_s2_band_index_from_spacenit_name(
    spacenit_name: str, band_names: list[str]
) -> int:
    for idx, band_name in enumerate(band_names):
        if spacenit_name.endswith(band_name.split(" ")[0][-2:]):
            return idx
    raise ValueError(f"Unmatched band name {spacenit_name}")


def _bench_s1_band_index_from_spacenit_name(spacenit_name: str) -> int:
    for idx, band_name in enumerate(BENCH_S1_BAND_NAMES):
        if spacenit_name == band_name:
            return idx
    raise ValueError(f"Unmatched band name {spacenit_name}")


def _bench_l8_band_index_from_spacenit_name(spacenit_name: str) -> int:
    for idx, band_name in enumerate(BENCH_L8_BAND_NAMES):
        if spacenit_name == band_name:
            return idx
    raise ValueError(f"Unmatched band name {spacenit_name}")


BENCH_TO_SPACENIT_S2_BANDS = [
    _bench_s2_band_index_from_spacenit_name(b, BENCH_S2_BAND_NAMES)
    for b in Sensor.SENTINEL2_L2A.all_channel_names
]

BENCH_TO_SPACENIT_S2_L2A_BANDS = [
    _bench_s2_band_index_from_spacenit_name(b, BENCH_S2_L2A_BAND_NAMES)
    for b in Sensor.SENTINEL2_L2A.all_channel_names
]

BENCH_TO_SPACENIT_S1_BANDS = [
    _bench_s1_band_index_from_spacenit_name(b) for b in Sensor.SENTINEL1.all_channel_names
]

BENCH_TO_SPACENIT_L8_BANDS = [
    _bench_l8_band_index_from_spacenit_name(b) for b in Sensor.LANDSAT.all_channel_names
]

# one to one mapping
BENCH_TO_SPACENIT_SRTM_BANDS = [0]
