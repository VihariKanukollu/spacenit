"""Model adapters for benchmark evaluation."""

from enum import StrEnum
from typing import Any

from spacenit.benchmarks.adapters.anysat import AnySat, AnySatConfig
from spacenit.benchmarks.adapters.clay import Clay, ClayConfig
from spacenit.benchmarks.adapters.croma import Croma, CromaConfig
from spacenit.benchmarks.adapters.dinov3 import DINOv3, DINOv3Config
from spacenit.benchmarks.adapters.galileo import GalileoAdapter, GalileoConfig
from spacenit.benchmarks.adapters.panopticon import Panopticon, PanopticonConfig
from spacenit.benchmarks.adapters.presto import PrestoAdapter, PrestoConfig
from spacenit.benchmarks.adapters.prithviv2 import PrithviV2, PrithviV2Config
from spacenit.benchmarks.adapters.satlas import Satlas, SatlasConfig
from spacenit.benchmarks.adapters.tessera import Tessera, TesseraConfig


class BaselineModelName(StrEnum):
    """Enum for baseline model names."""

    DINO_V3 = "dino_v3"
    PANOPTICON = "panopticon"
    GALILEO = "galileo"
    SATLAS = "satlas"
    CROMA = "croma"
    PRESTO = "presto"
    ANYSAT = "anysat"
    TESSERA = "tessera"
    PRITHVI_V2 = "prithvi_v2"
    CLAY = "clay"


__all__ = [
    "Panopticon",
    "PanopticonConfig",
    "GalileoAdapter",
    "GalileoConfig",
    "DINOv3",
    "DINOv3Config",
    "Satlas",
    "SatlasConfig",
    "Croma",
    "CromaConfig",
    "Clay",
    "ClayConfig",
    "PrestoAdapter",
    "PrestoConfig",
    "AnySat",
    "AnySatConfig",
    "Tessera",
    "TesseraConfig",
    "PrithviV2",
    "PrithviV2Config",
    "BaselineModelName",
]
