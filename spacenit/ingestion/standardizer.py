"""Standardizer for the SpaceNit ingestion pipeline."""

import json
import logging
from enum import Enum
from importlib.resources import files

import numpy as np

from spacenit.ingestion.sensors import SensorSpec

logger = logging.getLogger(__name__)


def load_predefined_config() -> dict[str, dict[str, dict[str, float]]]:
    """Load the predefined config.

    The standardization config maps from sensor label -> band name to a
    dictionary with ``min`` and ``max`` keys.
    """
    with (
        files("spacenit.ingestion.norm_configs") / "predefined.json"
    ).open() as f:
        return json.load(f)


def load_computed_config() -> dict[str, dict]:
    """Load the computed config.

    The standardization config maps from sensor label -> band name to a
    dictionary with ``mean`` and ``std`` keys.
    """
    with (files("spacenit.ingestion.norm_configs") / "computed.json").open() as f:
        return json.load(f)


class Strategy(Enum):
    """The strategy to use for standardization."""

    # Whether to use predefined or computed values for standardization.
    PREDEFINED = "predefined"
    COMPUTED = "computed"


class Standardizer:
    """Standardize raster data to a common numeric range."""

    def __init__(
        self,
        strategy: Strategy,
        std_multiplier: float | None = 2,
    ) -> None:
        """Initialize the standardizer.

        Args:
            strategy: The strategy to use for standardization (predefined or
                computed).
            std_multiplier: Optional, only for strategy COMPUTED.
                The multiplier for the standard deviation when using computed
                values.

        Returns:
            None
        """
        self.strategy = strategy
        self.std_multiplier = std_multiplier
        self.norm_config = self._load_config()

    def _load_config(self) -> dict:
        """Load the appropriate config based on the strategy."""
        if self.strategy == Strategy.PREDEFINED:
            return load_predefined_config()
        elif self.strategy == Strategy.COMPUTED:
            return load_computed_config()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def _standardize_predefined(
        self, sensor: SensorSpec, data: np.ndarray
    ) -> np.ndarray:
        """Standardize the data using predefined values."""
        # When using predefined values, we have the min and max values for each band.
        band_names = sensor.all_channel_names
        sensor_norm_values = self.norm_config[sensor.label]
        min_vals = []
        max_vals = []
        for band in band_names:
            if band not in sensor_norm_values:
                raise ValueError(f"Band {band} not found in config")
            min_val = sensor_norm_values[band]["min"]
            max_val = sensor_norm_values[band]["max"]
            min_vals.append(min_val)
            max_vals.append(max_val)
        # The last dimension of data is always the number of bands (channels).
        return (data - np.array(min_vals)) / (np.array(max_vals) - np.array(min_vals))

    def _standardize_computed(
        self, sensor: SensorSpec, data: np.ndarray
    ) -> np.ndarray:
        """Standardize the data using computed values."""
        # When using computed values, we compute the mean and std of each band
        # in advance, then convert to min and max values that cover ~90% of the
        # data.
        band_names = sensor.all_channel_names
        sensor_norm_values = self.norm_config[sensor.label]
        mean_vals = []
        std_vals = []
        for band in band_names:
            if band not in sensor_norm_values:
                raise ValueError(f"Band {band} not found in config")
            mean_val = sensor_norm_values[band]["mean"]
            std_val = sensor_norm_values[band]["std"]
            mean_vals.append(mean_val)
            std_vals.append(std_val)
        min_vals = np.array(mean_vals) - self.std_multiplier * np.array(std_vals)
        max_vals = np.array(mean_vals) + self.std_multiplier * np.array(std_vals)
        return (data - min_vals) / (max_vals - min_vals)  # type: ignore

    def standardize(self, sensor: SensorSpec, data: np.ndarray) -> np.ndarray:
        """Standardize the data.

        Args:
            sensor: The sensor specification whose data to standardize.
            data: The data to standardize.

        Returns:
            The standardized data.
        """
        if self.strategy == Strategy.PREDEFINED:
            return self._standardize_predefined(sensor, data)
        elif self.strategy == Strategy.COMPUTED:
            return self._standardize_computed(sensor, data)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def normalize(self, sensor: SensorSpec, data: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for :meth:`standardize`.

        Several benchmark dataset adapters historically used ``normalize()``.
        """
        return self.standardize(sensor, data)
