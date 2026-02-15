"""Shared types and enums used across the architecture modules."""

from enum import StrEnum


class PoolingType(StrEnum):
    """Strategy for pooling token sequences into fixed-size representations."""

    MAX = "max"
    MEAN = "mean"
