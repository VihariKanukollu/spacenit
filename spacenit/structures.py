"""Core data structures for geospatial sample representation.

Provides :class:`GeoSample` (dict-backed container for multi-sensor
observations), :class:`MaskedGeoSample` (adds per-sensor visibility masks),
and :class:`TokenVisibility` (mask states for masked modelling).

Both container classes validate keys against :class:`SensorRegistry` and
support attribute access (``sample.sentinel2_l2a``), dict-style access
(``sample["sentinel2_l2a"]``), and keyword construction::

    sample = GeoSample(sentinel2_l2a=tensor, timestamps=ts)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from math import floor
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from spacenit.aliases import NdTensor
from spacenit.ingestion.sensors import (
    ABSENT_INDICATOR,
    TEMPORAL_FIELDS,
    SensorRegistry,
    SensorSpec,
)

if TYPE_CHECKING:
    from spacenit.arch.band_tokenization import TokenizationConfig

logger = logging.getLogger(__name__)

# Reserved (non-sensor) field names that every GeoSample carries.
_META_FIELDS = ("timestamps", "latlon")


def _resolve_sensor(label: str) -> SensorSpec:
    """Look up a sensor spec, raising a clear error on failure."""
    return SensorRegistry.get(label)


# ---------------------------------------------------------------------------
# GeoSample
# ---------------------------------------------------------------------------


class GeoSample:
    """Container for a single geospatial observation or a batch of observations.

    Sensor data is stored in an internal dict keyed by sensor label
    (e.g. ``"sentinel2_l2a"``).  Metadata fields ``timestamps`` and ``latlon``
    are stored alongside sensor data but are *not* treated as sensors.

    Supports attribute access, dict-style ``[]`` access, and keyword
    construction for backward compatibility::

        sample = GeoSample(sentinel2_l2a=t, timestamps=ts)
        sample.sentinel2_l2a   # attribute access
        sample["sentinel2_l2a"]  # dict access
    """

    __slots__ = ("_data",)

    def __init__(self, **kwargs: NdTensor | None) -> None:
        object.__setattr__(self, "_data", {})
        for key, val in kwargs.items():
            if val is not None:
                self._data[key] = val

    # -- access ---------------------------------------------------------------

    def __getattr__(self, key: str) -> NdTensor | None:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._data.get(key)
        except AttributeError:
            raise AttributeError(key) from None

    def __getitem__(self, key: str) -> NdTensor | None:
        return self._data.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            if value is None:
                self._data.pop(key, None)
            else:
                self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{k}={_shape_str(v)}" for k, v in self._data.items()
        )
        return f"GeoSample({fields})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeoSample):
            return NotImplemented
        return self._data.keys() == other._data.keys() and all(
            _tensor_eq(self._data[k], other._data[k]) for k in self._data
        )

    # -- NamedTuple compatibility shims ---------------------------------------

    @property
    def _fields(self) -> tuple[str, ...]:
        """Return field names (for code that used NamedTuple._fields)."""
        return tuple(self._data.keys())

    def _replace(self, **kwargs: Any) -> GeoSample:
        """Return a copy with specified fields replaced."""
        merged = {**self._data, **{k: v for k, v in kwargs.items() if v is not None}}
        # Remove keys set to None
        for k, v in kwargs.items():
            if v is None:
                merged.pop(k, None)
        return GeoSample(**merged)

    # -- dict interop ---------------------------------------------------------

    def to_dict(self, ignore_nones: bool = True) -> dict[str, NdTensor | None]:
        """Serialise to a plain dictionary.

        Args:
            ignore_nones: When ``True`` (default), only populated fields are
                included.  When ``False``, every registered sensor label plus
                meta fields are included with ``None`` for missing ones.
        """
        if ignore_nones:
            return dict(self._data)
        result: dict[str, NdTensor | None] = {}
        for key in list(_META_FIELDS) + SensorRegistry.all_labels():
            result[key] = self._data.get(key)
        return result

    def as_dict(self, return_none: bool = True) -> dict[str, NdTensor | None]:
        """Alias for :meth:`to_dict` with inverted flag semantics.

        Args:
            return_none: When ``True``, include ``None`` entries for missing
                fields.  When ``False``, only populated fields are returned.
        """
        return self.to_dict(ignore_nones=not return_none)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeoSample:
        """Construct a :class:`GeoSample` from a plain dictionary."""
        return cls(**{k: v for k, v in d.items() if v is not None})

    # -- sensor queries -------------------------------------------------------

    @property
    def present_keys(self) -> list[str]:
        """List field names that are populated (non-``None``).

        Includes ``timestamps`` and ``latlon`` when present.
        """
        return list(self._data.keys())

    @property
    def present_sensors(self) -> list[str]:
        """Sensor labels that carry data (excludes meta fields)."""
        return [k for k in self._data if k not in _META_FIELDS]

    # -- timestamps / latlon shortcuts ----------------------------------------

    @property
    def timestamps(self) -> NdTensor | None:
        return self._data.get("timestamps")

    @property
    def latlon(self) -> NdTensor | None:
        return self._data.get("latlon")

    # -- geometry helpers -----------------------------------------------------

    @staticmethod
    def channel_count(attribute: str) -> int:
        """Return the number of spectral channels for a given field."""
        if attribute == "timestamps":
            return len(TEMPORAL_FIELDS)
        return _resolve_sensor(attribute).total_channels

    def dimensions_of(self, attribute: str, mask: bool = False) -> Sequence[int]:
        """Return the shape of a given field."""
        if attribute == "timestamps":
            if mask:
                raise ValueError("Timestamps do not support masking")
            ts = self.timestamps
            if ts is None:
                raise ValueError("Timestamps are not available in this sample")
            return ts.shape
        return self.expected_dimensions(attribute, mask)

    @property
    def num_samples(self) -> int:
        """Return the batch size."""
        for v in self._data.values():
            if v is not None:
                return v.shape[0] if len(v.shape) > 1 else 1
        return 1

    @property
    def spatial_height(self) -> int:
        """Return the height of the spatial grid at the base resolution."""
        for key in self.present_sensors:
            spec = _resolve_sensor(key)
            if not spec.varies_in_space:
                continue
            x = self._data[key]
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[1] // spec.tile_size_multiplier
                elif len(x.shape) == 4:
                    return x.shape[0] // spec.tile_size_multiplier
                else:
                    raise ValueError(f"Unexpected shape {x.shape} for {key}")
        raise ValueError("No spatial sensor present to determine height")

    @property
    def spatial_width(self) -> int:
        """Return the width of the spatial grid at the base resolution."""
        for key in self.present_sensors:
            spec = _resolve_sensor(key)
            if not spec.varies_in_space:
                continue
            x = self._data[key]
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[2] // spec.tile_size_multiplier
                elif len(x.shape) == 4:
                    return x.shape[1] // spec.tile_size_multiplier
                else:
                    raise ValueError(f"Unexpected shape {x.shape} for {key}")
        raise ValueError("No spatial sensor present to determine width")

    @property
    def temporal_length(self) -> int:
        """Return the total number of time steps recorded."""
        ts = self.timestamps
        if ts is None:
            raise ValueError("Timestamps are not available in this sample")
        return ts.shape[-2]

    @property
    def active_temporal_length(self) -> int:
        """Return the count of time steps with at least one sensor reading."""
        return self.occupied_timesteps.shape[0]

    @property
    def occupied_timesteps(self) -> torch.Tensor:
        """Identify time indices where at least one sensor has valid data."""
        per_sensor_masks: list[torch.Tensor] = []
        for key in self.present_sensors:
            spec = _resolve_sensor(key)
            if spec.has_temporal_axis:
                data = self._data[key]
                if isinstance(data, np.ndarray):
                    raise ValueError(
                        "occupied_timesteps does not yet support numpy arrays"
                    )
                present_mask = (data != ABSENT_INDICATOR).all(dim=(0, 1, 2, 4))
                per_sensor_masks.append(present_mask)
        any_sensor_present = torch.stack(per_sensor_masks, dim=1).any(dim=1)
        return torch.where(any_sensor_present)[0]

    # -- shape inference ------------------------------------------------------

    @staticmethod
    def infer_shape(
        attribute: str,
        height: int | None,
        width: int | None,
        time: int,
        mask: bool = False,
    ) -> tuple[int, ...]:
        """Compute the expected tensor shape for a sensor field."""
        spec = _resolve_sensor(attribute)
        n_bands = spec.group_count if mask else spec.total_channels

        if spec.varies_in_space_and_time:
            assert height is not None and width is not None
            return (
                height * spec.tile_size_multiplier,
                width * spec.tile_size_multiplier,
                time,
                n_bands,
            )
        elif spec.varies_in_space_only:
            assert height is not None and width is not None
            return (
                height * spec.tile_size_multiplier,
                width * spec.tile_size_multiplier,
                1,
                n_bands,
            )
        elif spec.varies_in_time_only:
            return (time, n_bands)
        else:
            return (n_bands,)

    def expected_dimensions(
        self, attribute: str, mask: bool = False
    ) -> tuple[int, ...]:
        """Derive the expected shape of *attribute* from this sample's geometry."""
        return GeoSample.infer_shape(
            attribute,
            self.spatial_height,
            self.spatial_width,
            self.temporal_length,
            mask,
        )

    # -- device / distribution ------------------------------------------------

    def transfer_to(
        self, device: torch.device, non_blocking: bool = True
    ) -> GeoSample:
        """Move every tensor to the specified device."""
        return GeoSample(
            **{
                k: v.to(device, non_blocking=non_blocking)
                for k, v in self._data.items()
                if v is not None
            }
        )

    def shard_across(self, device_mesh: DeviceMesh) -> GeoSample:
        """Distribute tensors across a device mesh."""
        return GeoSample(
            **{k: distribute_tensor(v, device_mesh) for k, v in self._data.items()}
        )

    # -- cropping -------------------------------------------------------------

    def _max_steps_within_budget(
        self,
        h_w_p: int,
        max_tokens_per_instance: int,
        tokenization_config: TokenizationConfig | None = None,
    ) -> int:
        """Determine the largest temporal window that fits the token budget."""
        fixed_tokens = 0
        per_step_tokens = 0
        for attr_name in self.present_sensors:
            spec = _resolve_sensor(attr_name)
            n_band_sets = (
                tokenization_config.group_count_for(attr_name)
                if tokenization_config is not None
                else spec.group_count
            )
            if spec.varies_in_space_and_time:
                per_step_tokens += (h_w_p ** 2) * n_band_sets
            elif spec.varies_in_space_only:
                fixed_tokens += (h_w_p ** 2) * n_band_sets
            elif spec.varies_in_time_only:
                per_step_tokens += n_band_sets
            elif spec.is_constant:
                fixed_tokens += n_band_sets
        if per_step_tokens == 0:
            return 1
        budget_remaining = max_tokens_per_instance - fixed_tokens
        max_t = budget_remaining / per_step_tokens
        if max_t < 1:
            raise ValueError(
                f"Patch size too small for this sample and budget, h_w_p: {h_w_p}, "
                f"max_tokens: {max_tokens_per_instance}"
            )
        return min(floor(max_t), self.temporal_length)

    @staticmethod
    def _feasible_start_indices(
        absent_step_flags: dict[str, Any], max_t: int, current_length: int
    ) -> list[int]:
        """Return sorted list of valid starting time indices for cropping."""
        if current_length > max_t:
            if not absent_step_flags:
                feasible = list(range(current_length - max_t + 1))
            else:
                candidates: set[int] = set()
                for sensor_name in absent_step_flags:
                    valid_steps = np.flatnonzero(absent_step_flags[sensor_name])
                    valid_steps = valid_steps[valid_steps + max_t <= current_length]
                    candidates.update(valid_steps)
                feasible = list(candidates)
        else:
            feasible = [0]
        if len(feasible) == 0:
            logger.warning(
                f"No feasible start indices for {absent_step_flags} with "
                f"max_t {max_t} and current_length {current_length}"
            )
            raise ValueError(
                f"No feasible start indices for {absent_step_flags} with "
                f"max_t {max_t} and current_length {current_length}"
            )
        return sorted(feasible)

    def crop_rectangular(
        self,
        patch_size: int,
        max_tokens_per_instance: int | None,
        sampled_hw_p: int,
        current_length: int,
        missing_timesteps_masks: dict[str, Any] | None = None,
        tokenization_config: TokenizationConfig | None = None,
    ) -> GeoSample:
        """Crop with a contiguous rectangular spatial window."""
        if missing_timesteps_masks is None:
            missing_timesteps_masks = {}
        if max_tokens_per_instance is None:
            return self
        max_t = self._max_steps_within_budget(
            sampled_hw_p, max_tokens_per_instance, tokenization_config
        )
        valid_starts = self._feasible_start_indices(
            missing_timesteps_masks, max_t, current_length
        )
        start_t = np.random.choice(valid_starts)
        new_fields: dict[str, NdTensor] = {}

        sampled_hw = sampled_hw_p * patch_size
        start_h = np.random.choice(self.spatial_height - sampled_hw + 1)
        start_w = np.random.choice(self.spatial_width - sampled_hw + 1)

        for attr_name, tensor in self._data.items():
            if attr_name == "timestamps":
                new_fields[attr_name] = tensor[start_t : start_t + max_t]
                continue
            if attr_name in _META_FIELDS:
                new_fields[attr_name] = tensor
                continue
            spec = _resolve_sensor(attr_name)
            if spec.varies_in_space_and_time:
                new_fields[attr_name] = tensor[
                    start_h * spec.tile_size_multiplier : (start_h + sampled_hw)
                    * spec.tile_size_multiplier,
                    start_w * spec.tile_size_multiplier : (start_w + sampled_hw)
                    * spec.tile_size_multiplier,
                    start_t : start_t + max_t,
                ]
            elif spec.varies_in_space_only:
                new_fields[attr_name] = tensor[
                    start_h * spec.tile_size_multiplier : (start_h + sampled_hw)
                    * spec.tile_size_multiplier,
                    start_w * spec.tile_size_multiplier : (start_w + sampled_hw)
                    * spec.tile_size_multiplier,
                ]
            elif spec.varies_in_time_only:
                new_fields[attr_name] = tensor[start_t : start_t + max_t]
            elif spec.is_constant:
                new_fields[attr_name] = tensor

        return GeoSample(**new_fields)

    def crop_patchwise(
        self,
        patch_size: int,
        max_tokens_per_instance: int | None,
        sampled_hw_p: int,
        current_length: int,
        missing_timesteps_masks: dict[str, Any] | None = None,
        tokenization_config: TokenizationConfig | None = None,
    ) -> GeoSample:
        """Crop by randomly selecting non-contiguous patches."""
        if missing_timesteps_masks is None:
            missing_timesteps_masks = {}
        if max_tokens_per_instance is None:
            return self
        max_t = self._max_steps_within_budget(
            sampled_hw_p, max_tokens_per_instance, tokenization_config
        )
        valid_starts = self._feasible_start_indices(
            missing_timesteps_masks, max_t, current_length
        )
        start_t = np.random.choice(valid_starts)
        new_fields: dict[str, NdTensor] = {}

        height_p = self.spatial_height // patch_size
        width_p = self.spatial_width // patch_size
        h_p_indices = np.random.choice(height_p, size=sampled_hw_p, replace=False)
        w_p_indices = np.random.choice(width_p, size=sampled_hw_p, replace=False)
        h_indices = [
            i
            for hp in h_p_indices
            for i in range(hp * patch_size, (hp + 1) * patch_size)
        ]
        w_indices = [
            i
            for wp in w_p_indices
            for i in range(wp * patch_size, (wp + 1) * patch_size)
        ]
        hh, ww = np.meshgrid(h_indices, w_indices, indexing="ij")

        for attr_name, tensor in self._data.items():
            if attr_name == "timestamps":
                new_fields[attr_name] = tensor[start_t : start_t + max_t]
                continue
            if attr_name in _META_FIELDS:
                new_fields[attr_name] = tensor
                continue
            spec = _resolve_sensor(attr_name)
            if spec.varies_in_space_and_time:
                new_fields[attr_name] = tensor[
                    hh * spec.tile_size_multiplier,
                    ww * spec.tile_size_multiplier,
                    start_t : start_t + max_t,
                ]
            elif spec.varies_in_space_only:
                new_fields[attr_name] = tensor[
                    hh * spec.tile_size_multiplier,
                    ww * spec.tile_size_multiplier,
                ]
            elif spec.varies_in_time_only:
                new_fields[attr_name] = tensor[start_t : start_t + max_t]
            elif spec.is_constant:
                new_fields[attr_name] = tensor

        return GeoSample(**new_fields)

    # -- arithmetic -----------------------------------------------------------

    def multiply(self, s: float) -> GeoSample:
        """Scale every tensor by a scalar factor."""
        return GeoSample(**{k: v * s for k, v in self._data.items()})

    def combine(self, other: GeoSample, timestamps_to_keep: NdTensor) -> GeoSample:
        """Element-wise addition of two samples."""
        if not isinstance(other, GeoSample):
            raise ValueError("combine only supports GeoSample operands")
        summed: dict[str, NdTensor] = {}
        for key, val in self._data.items():
            other_val = other[key]
            if other_val is None:
                raise ValueError(
                    f"combine requires both samples to share the same fields; "
                    f"the other sample is missing '{key}'"
                )
            summed[key] = val + other_val
        summed["timestamps"] = timestamps_to_keep
        return GeoSample(**summed)

    def shift_batch(self) -> GeoSample:
        """Cyclically shift the batch dimension by one position."""
        shifted: dict[str, NdTensor] = {}
        for key, v in self._data.items():
            if isinstance(v, np.ndarray):
                shifted[key] = np.concatenate((v[1:], v[:1]), axis=0)
            elif isinstance(v, torch.Tensor):
                shifted[key] = torch.cat((v[1:], v[:1]), dim=0)
        return GeoSample(**shifted)


# ---------------------------------------------------------------------------
# TokenVisibility
# ---------------------------------------------------------------------------


class TokenVisibility(Enum):
    """Visibility states a token can assume during masked modelling.

    * ``VISIBLE_ENCODER`` -- token is fed to the online encoder.
    * ``TARGET_ONLY`` -- token is visible only to the target (momentum) encoder.
    * ``PREDICTED`` -- token must be reconstructed by the decoder.
    * ``ABSENT`` -- the underlying data is missing entirely.
    """

    VISIBLE_ENCODER = 0
    TARGET_ONLY = 1
    PREDICTED = 2
    ABSENT = 3


# ---------------------------------------------------------------------------
# MaskedGeoSample
# ---------------------------------------------------------------------------


class MaskedGeoSample:
    """A geospatial sample augmented with per-sensor visibility masks.

    Data tensors and mask tensors are stored in separate dicts.  Mask keys
    use the pattern ``<sensor_label>_mask``.  ``timestamps`` is always
    present as a data field (no mask).

    Supports the same construction patterns as the old NamedTuple::

        MaskedGeoSample(timestamps=ts, sentinel2_l2a=data, sentinel2_l2a_mask=mask)
    """

    __slots__ = ("_data", "_masks")

    def __init__(self, **kwargs: NdTensor | None) -> None:
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_masks", {})
        for key, val in kwargs.items():
            if val is None:
                continue
            if key.endswith("_mask"):
                self._masks[key] = val
            else:
                self._data[key] = val

    # -- access ---------------------------------------------------------------

    def __getattr__(self, key: str) -> NdTensor | None:
        if key.startswith("_"):
            raise AttributeError(key)
        if key.endswith("_mask"):
            return self._masks.get(key)
        return self._data.get(key)

    def __getitem__(self, key: str) -> NdTensor | None:
        if key.endswith("_mask"):
            return self._masks.get(key)
        return self._data.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        elif key.endswith("_mask"):
            if value is None:
                self._masks.pop(key, None)
            else:
                self._masks[key] = value
        else:
            if value is None:
                self._data.pop(key, None)
            else:
                self._data[key] = value

    def __repr__(self) -> str:
        data_keys = list(self._data.keys())
        mask_keys = list(self._masks.keys())
        return f"MaskedGeoSample(data={data_keys}, masks={mask_keys})"

    # -- NamedTuple compatibility shims ---------------------------------------

    @property
    def _fields(self) -> tuple[str, ...]:
        """All field names (data + mask), for backward compat."""
        return tuple(list(self._data.keys()) + list(self._masks.keys()))

    def _replace(self, **kwargs: Any) -> MaskedGeoSample:
        """Return a copy with specified fields replaced."""
        all_fields = {**self._data, **self._masks}
        for k, v in kwargs.items():
            if v is None:
                all_fields.pop(k, None)
            else:
                all_fields[k] = v
        return MaskedGeoSample(**all_fields)

    # -- dict interop ---------------------------------------------------------

    def to_dict(self, return_none: bool = True) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        if not return_none:
            return {**self._data, **self._masks}
        # Include None for missing sensors/masks
        result: dict[str, Any] = {}
        result["timestamps"] = self._data.get("timestamps")
        for label in SensorRegistry.all_labels():
            result[label] = self._data.get(label)
            mask_key = f"{label}_mask"
            result[mask_key] = self._masks.get(mask_key)
        # Also include latlon if present
        if "latlon" in self._data:
            result["latlon"] = self._data["latlon"]
            result["latlon_mask"] = self._masks.get("latlon_mask")
        return result

    def as_dict(self, return_none: bool = True) -> dict[str, Any]:
        """Alias for :meth:`to_dict` with the same semantics.

        Args:
            return_none: When ``True``, include ``None`` entries for missing
                fields.  When ``False``, only populated fields are returned.
        """
        return self.to_dict(return_none=return_none)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MaskedGeoSample:
        """Construct a :class:`MaskedGeoSample` from a plain dictionary."""
        return cls(**{k: v for k, v in d.items() if v is not None})

    @classmethod
    def from_geosample(cls, sample: GeoSample) -> MaskedGeoSample:
        """Wrap a :class:`GeoSample` in a :class:`MaskedGeoSample`.

        For each sensor present in *sample*, a mask is created with all
        tokens set to :attr:`TokenVisibility.VISIBLE_ENCODER`.  Absent
        sensors are not included.
        """
        kwargs: dict[str, Any] = {}
        for key in sample._fields:
            val = sample[key]
            if val is None:
                continue
            kwargs[key] = val
            # Add a mask for non-meta fields
            if key not in _META_FIELDS:
                if isinstance(val, torch.Tensor):
                    kwargs[f"{key}_mask"] = torch.full_like(
                        val[..., 0:1].squeeze(-1) if val.ndim > 1 else val,
                        TokenVisibility.VISIBLE_ENCODER.value,
                        dtype=torch.long,
                    )
                else:
                    import numpy as _np

                    kwargs[f"{key}_mask"] = _np.full_like(
                        val[..., 0:1].squeeze(-1) if val.ndim > 1 else val,
                        TokenVisibility.VISIBLE_ENCODER.value,
                        dtype=_np.int64,
                    )
        return cls(**kwargs)

    # Alias for backward compatibility with benchmark code that uses the old name.
    from_spacenitsample = from_geosample

    # -- sensor queries -------------------------------------------------------

    @property
    def present_keys(self) -> list[str]:
        """Sensor labels that carry data (excludes meta fields and masks)."""
        return [
            k for k in self._data if k != "timestamps" and k not in _META_FIELDS
        ]

    @property
    def timestamps(self) -> NdTensor:
        return self._data["timestamps"]

    @property
    def latlon(self) -> NdTensor | None:
        return self._data.get("latlon")

    @property
    def num_samples(self) -> int:
        return self.timestamps.shape[0]

    # -- mask helpers ---------------------------------------------------------

    @staticmethod
    def mask_field_for(sensor: str) -> str:
        return f"{sensor}_mask"

    @staticmethod
    def data_field_for(mask_field_name: str) -> str:
        return mask_field_name.replace("_mask", "")

    def clear_masks(self) -> MaskedGeoSample:
        """Reset all masks: ABSENT stays ABSENT, everything else -> VISIBLE_ENCODER."""
        new_masks: dict[str, Any] = {}
        for mask_key, val in self._masks.items():
            new_masks[mask_key] = val * (val == TokenVisibility.ABSENT.value)
        all_fields = {**self._data, **new_masks}
        return MaskedGeoSample(**all_fields)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_geo_sample(cls, sample: GeoSample) -> MaskedGeoSample:
        """Construct from an unmasked GeoSample.

        Every present sensor receives a mask filled with VISIBLE_ENCODER.
        """
        fields: dict[str, Any] = {}
        for key, tensor in sample._data.items():
            if key == "timestamps":
                fields[key] = tensor
            elif key in _META_FIELDS:
                fields[key] = tensor
                fields[cls.mask_field_for(key)] = torch.ones(
                    sample.dimensions_of(key, mask=False)
                ) * TokenVisibility.VISIBLE_ENCODER.value
            else:
                fields[key] = tensor
                fields[cls.mask_field_for(key)] = (
                    torch.ones(sample.dimensions_of(key, mask=False))
                    * TokenVisibility.VISIBLE_ENCODER.value
                )
        return cls(**fields)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> MaskedGeoSample:
        """Instantiate from a dictionary."""
        return cls(**{k: v for k, v in mapping.items() if v is not None})

    # -- device ---------------------------------------------------------------

    def transfer_to(
        self, device: torch.device, non_blocking: bool = True
    ) -> MaskedGeoSample:
        """Move every tensor to the specified device."""
        all_fields: dict[str, Any] = {}
        for k, v in self._data.items():
            all_fields[k] = v.to(device, non_blocking=non_blocking)
        for k, v in self._masks.items():
            all_fields[k] = v.to(device, non_blocking=non_blocking)
        return MaskedGeoSample(**all_fields)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_str(v: Any) -> str:
    """Compact shape string for repr."""
    if hasattr(v, "shape"):
        return f"shape={tuple(v.shape)}"
    return repr(v)


def _tensor_eq(a: Any, b: Any) -> bool:
    """Element-wise equality check that works for tensors and arrays."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b
