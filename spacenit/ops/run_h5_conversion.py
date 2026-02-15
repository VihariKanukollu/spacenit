r"""Run the conversion of a SpaceNit dataset to h5py files.

This script is used to convert a dataset to h5py files.

Usage:
    python -m spacenit.ops.run_h5_conversion \
        --tile-path=TILE_PATH \
        --supported-modality-names="\[sentinel2_l2a,sentinel1,worldcover\]" \
        --compression=zstd \
        --compression_opts=3 \
        --tile_size=128
"""

import logging
import sys
from collections.abc import Callable
from typing import Any

from olmo_core.utils import prepare_cli_environment

from spacenit.ingestion.sensors import (
    SENTINEL2_L2A,
    SENTINEL1,
    LANDSAT,
    WORLDCOVER,
    OPENSTREETMAP_RASTER,
    WORLDCEREAL,
    SRTM,
    ERA5_10,
    NAIP_10,
)
from spacenit.conversion.h5_converter import ConvertToH5pyConfig

logger = logging.getLogger(__name__)


def build_default_config() -> ConvertToH5pyConfig:
    """Build the default configuration for H5 conversion."""
    return ConvertToH5pyConfig(
        tile_path="",
        supported_modality_names=[
            SENTINEL2_L2A.label,
            SENTINEL1.label,
            LANDSAT.label,
            WORLDCOVER.label,
            OPENSTREETMAP_RASTER.label,
            WORLDCEREAL.label,
            SRTM.label,
            ERA5_10.label,
            NAIP_10.label,
        ],
        multiprocessed_h5_creation=True,
    )


def main(config_builder: Callable = build_default_config, *args: Any) -> None:
    """Parse arguments, build config, and run the H5 conversion."""
    prepare_cli_environment()

    script, *overrides = sys.argv

    default_config = config_builder()
    config = default_config.merge(overrides)
    logger.info(f"Configuration overrides: {overrides}")
    logger.info(f"Configuration loaded: {config}")

    converter = config.build()
    logger.info("Starting H5 conversion process...")
    converter.run()
    logger.info("H5 conversion process finished.")


if __name__ == "__main__":
    main()
