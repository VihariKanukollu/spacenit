"""Spectral-band tokenisation strategies for multi-sensor architectures.

Controls how a sensor's spectral channels are partitioned into token
groups.  Each group becomes a separate patch token at every spatial
location, allowing the model to handle heterogeneous channel layouts
across sensors.

Example::

    from spacenit.arch.band_tokenization import TokenizationConfig, SensorTokenLayout
    from spacenit.ingestion.sensors import SensorRegistry

    # Per-channel tokenisation for Sentinel-2
    s2_channels = SensorRegistry.get("sentinel2").all_channel_names
    config = TokenizationConfig(
        custom_layouts={
            "sentinel2": SensorTokenLayout(
                channel_groups=[[ch] for ch in s2_channels]
            )
        }
    )

    # Default grouping for other sensors
    n_groups = config.group_count_for("sentinel1")
"""

from dataclasses import dataclass, field

from spacenit.ingestion.sensors import SensorRegistry, SensorSpec


# ---------------------------------------------------------------------------
# Per-sensor layout
# ---------------------------------------------------------------------------


@dataclass
class SensorTokenLayout:
    """Describes how a single sensor's channels are grouped into tokens.

    Each element of *channel_groups* is a list of channel names that will
    be concatenated and treated as one token.

    Attributes:
        channel_groups: Ordered list of channel-name lists.  Every name
            must appear in the parent :class:`SensorSpec`.
    """

    channel_groups: list[list[str]]

    def resolve_indices(self, base_sensor: SensorSpec) -> list[list[int]]:
        """Convert channel names to flat integer indices.

        The indices reference the position of each channel in the sensor's
        canonical :pyattr:`SensorSpec.all_channel_names` ordering.

        Args:
            base_sensor: The :class:`SensorSpec` that defines the canonical
                channel order.

        Returns:
            A list of index lists, one per channel group.

        Raises:
            ValueError: If a channel name is not present in *base_sensor*.
        """
        name_to_idx = {name: i for i, name in enumerate(base_sensor.all_channel_names)}
        result: list[list[int]] = []
        for group in self.channel_groups:
            group_indices: list[int] = []
            for channel in group:
                if channel not in name_to_idx:
                    raise ValueError(
                        f"Channel '{channel}' not found in sensor "
                        f"'{base_sensor.label}'. "
                        f"Valid channels: {list(base_sensor.all_channel_names)}"
                    )
                group_indices.append(name_to_idx[channel])
            result.append(group_indices)
        return result

    def channels_per_group(self) -> list[int]:
        """Return the number of channels in each group."""
        return [len(group) for group in self.channel_groups]

    @property
    def group_count(self) -> int:
        """Total number of token groups defined by this layout."""
        return len(self.channel_groups)

    def check_consistency(self, base_sensor: SensorSpec) -> None:
        """Verify that every channel name exists in the sensor specification.

        Args:
            base_sensor: The :class:`SensorSpec` to validate against.

        Raises:
            ValueError: If any channel name is missing from the sensor's
                channel list.
        """
        valid_channels = set(base_sensor.all_channel_names)
        for group in self.channel_groups:
            for channel in group:
                if channel not in valid_channels:
                    raise ValueError(
                        f"Channel '{channel}' not found in sensor "
                        f"'{base_sensor.label}'. "
                        f"Valid channels: {valid_channels}"
                    )


# ---------------------------------------------------------------------------
# Global tokenisation configuration
# ---------------------------------------------------------------------------


@dataclass
class TokenizationConfig:
    """Top-level configuration governing channel-to-token mapping.

    Sensors without an explicit entry in *custom_layouts* fall back to
    the default grouping defined by their :class:`SensorSpec` spectral
    groups.

    Attributes:
        custom_layouts: Mapping from sensor label to a custom
            :class:`SensorTokenLayout`.
    """

    custom_layouts: dict[str, SensorTokenLayout] = field(default_factory=dict)
    _index_cache: dict[str, list[list[int]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def group_indices_for(self, sensor_label: str) -> list[list[int]]:
        """Return per-group channel indices for the given sensor.

        Results are cached after the first call for a given label.

        Args:
            sensor_label: Label of the sensor (e.g. ``"sentinel2"``).

        Returns:
            List of integer-index lists, one per token group.

        Raises:
            ValueError: If *sensor_label* does not match any known sensor,
                or if a custom layout references invalid channel names.
        """
        if sensor_label in self._index_cache:
            return self._index_cache[sensor_label]

        try:
            base_spec = SensorRegistry.get(sensor_label)
        except (AttributeError, AssertionError, KeyError) as e:
            raise ValueError(f"Unknown sensor: {sensor_label}") from e

        if sensor_label in self.custom_layouts:
            result = self.custom_layouts[sensor_label].resolve_indices(base_spec)
        else:
            result = base_spec.group_indices()

        self._index_cache[sensor_label] = result
        return result

    def group_count_for(self, sensor_label: str) -> int:
        """Return the number of token groups for the given sensor.

        Args:
            sensor_label: Label of the sensor.

        Returns:
            Number of token groups.

        Raises:
            ValueError: If the sensor label is unknown.
        """
        if sensor_label in self.custom_layouts:
            return self.custom_layouts[sensor_label].group_count
        try:
            return SensorRegistry.get(sensor_label).group_count
        except (AttributeError, AssertionError, KeyError) as e:
            raise ValueError(f"Unknown sensor: {sensor_label}") from e

    def channels_per_group_for(self, sensor_label: str) -> list[int]:
        """Return the channel count in each token group for a sensor.

        Args:
            sensor_label: Label of the sensor.

        Returns:
            List of integers, one per group.

        Raises:
            ValueError: If the sensor label is unknown.
        """
        if sensor_label in self.custom_layouts:
            return self.custom_layouts[sensor_label].channels_per_group()
        try:
            base_spec = SensorRegistry.get(sensor_label)
        except (AttributeError, AssertionError, KeyError) as e:
            raise ValueError(f"Unknown sensor: {sensor_label}") from e
        return [len(g.channel_names) for g in base_spec.spectral_groups]

    def check_consistency(self) -> None:
        """Validate every custom layout against its sensor specification.

        Raises:
            ValueError: If any sensor label is unrecognised or any channel
                name within a layout is invalid.
        """
        for sensor_label, layout in self.custom_layouts.items():
            try:
                base_spec = SensorRegistry.get(sensor_label)
            except (AttributeError, AssertionError, KeyError):
                raise ValueError(
                    f"Unknown sensor label in custom_layouts: '{sensor_label}'. "
                    f"Valid labels: {SensorRegistry.all_labels()}"
                )
            layout.check_consistency(base_spec)
