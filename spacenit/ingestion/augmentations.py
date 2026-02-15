"""Data augmentation transforms for geospatial samples.

Rewritten from scratch with different augmentation strategies and a
simple dict-based dispatch instead of ``class_registry``.

Transforms:
- :class:`RandomDihedralTransform` -- the 8 symmetries of a square,
  implemented via ``torch.rot90`` and ``torch.flip`` directly.
- :class:`CutMix` -- replaces the original Mixup with CutMix, a
  different augmentation strategy that cuts and pastes rectangular
  regions between samples.
- :class:`SensorDropout` -- randomly drops entire sensors during
  training (new, not in original).
- :class:`NoTransform` -- identity transform.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from torch import Tensor

from spacenit.ingestion.sensors import SensorRegistry
from spacenit.structures import GeoSample


# ---------------------------------------------------------------------------
# Transform protocol
# ---------------------------------------------------------------------------


class Transform(Protocol):
    """Protocol for sample transforms."""

    def __call__(self, sample: GeoSample) -> GeoSample:
        ...


# ---------------------------------------------------------------------------
# No-op transform
# ---------------------------------------------------------------------------


class NoTransform:
    """Identity transform -- returns the sample unchanged."""

    def __call__(self, sample: GeoSample) -> GeoSample:
        return sample


# ---------------------------------------------------------------------------
# Random Dihedral Transform (8 symmetries of a square)
# ---------------------------------------------------------------------------


class RandomDihedralTransform:
    """Apply a random element of the dihedral group D4 to spatial data.

    The 8 symmetries are: identity, 3 rotations (90/180/270), horizontal
    flip, vertical flip, and the two diagonal reflections (hflip+rot90,
    vflip+rot90).

    Implementation uses ``torch.rot90`` and ``torch.flip`` directly on
    the spatial dimensions, which is different from the original approach
    that used ``torchvision.transforms.v2.functional``.
    """

    def __call__(self, sample: GeoSample) -> GeoSample:
        # Pick a random symmetry index 0-7
        sym = random.randint(0, 7)
        return self._apply_symmetry(sample, sym)

    @staticmethod
    def _apply_symmetry(sample: GeoSample, sym: int) -> GeoSample:
        """Apply a specific symmetry to all spatial tensors in the sample."""
        new_fields: dict[str, Any] = {}

        for key in sample.present_keys:
            data = sample[key]
            if data is None:
                continue

            if key in ("timestamps", "latlon"):
                new_fields[key] = data
                continue

            try:
                spec = SensorRegistry.get(key)
            except (ValueError, KeyError):
                new_fields[key] = data
                continue

            if spec.varies_in_space_and_time or spec.varies_in_space_only:
                data = _apply_dihedral_to_spatial(data, sym)

            new_fields[key] = data

        return GeoSample(**new_fields)


def _apply_dihedral_to_spatial(x: Tensor, sym: int) -> Tensor:
    """Apply dihedral symmetry to a tensor with spatial dimensions.

    Expects the first two (non-batch) dimensions to be spatial (H, W).
    Works with shapes like:
    - ``(H, W, T, C)`` -- unbatched spatiotemporal
    - ``(B, H, W, T, C)`` -- batched spatiotemporal
    - ``(H, W, C)`` -- unbatched spatial-only
    - ``(B, H, W, C)`` -- batched spatial-only

    The spatial dimensions are always the first two after any batch dim.
    """
    # Determine spatial dims based on tensor rank
    # For our data layout, spatial dims are always 0,1 (unbatched) or 1,2 (batched)
    has_batch = x.ndim >= 4 and x.shape[0] < x.shape[1]  # heuristic

    if has_batch:
        h_dim, w_dim = 1, 2
    else:
        h_dim, w_dim = 0, 1

    if sym == 0:
        return x  # identity
    elif sym == 1:
        return torch.rot90(x, k=1, dims=[h_dim, w_dim])
    elif sym == 2:
        return torch.rot90(x, k=2, dims=[h_dim, w_dim])
    elif sym == 3:
        return torch.rot90(x, k=3, dims=[h_dim, w_dim])
    elif sym == 4:
        return torch.flip(x, dims=[w_dim])  # horizontal flip
    elif sym == 5:
        return torch.flip(x, dims=[h_dim])  # vertical flip
    elif sym == 6:
        return torch.flip(torch.rot90(x, k=1, dims=[h_dim, w_dim]), dims=[w_dim])
    elif sym == 7:
        return torch.flip(torch.rot90(x, k=1, dims=[h_dim, w_dim]), dims=[h_dim])
    else:
        return x


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------


class CutMix:
    """CutMix augmentation for geospatial samples.

    Replaces the original Mixup with CutMix (Yun et al., 2019), which
    cuts a rectangular region from one sample and pastes it onto another.
    This is a fundamentally different augmentation strategy.

    The cut region size is sampled from a Beta distribution, and the
    region is placed at a random position.

    Args:
        alpha: Parameter for the Beta distribution controlling the
            cut size.  Larger values produce more uniform size distribution.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(self, sample: GeoSample) -> GeoSample:
        """Apply CutMix to a batched sample.

        Mixes each sample with a cyclically shifted version of the batch.
        """
        other = sample.shift_batch()

        # Sample lambda from Beta distribution
        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())

        # Determine cut region
        try:
            H = sample.spatial_height
            W = sample.spatial_width
        except ValueError:
            # No spatial data -- return original
            return sample

        # Cut region dimensions
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Random position
        cy = random.randint(0, H)
        cx = random.randint(0, W)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        new_fields: dict[str, Any] = {}

        for key in sample.present_keys:
            data = sample[key]
            other_data = other[key]
            if data is None:
                continue

            if key in ("timestamps", "latlon"):
                new_fields[key] = data
                continue

            try:
                spec = SensorRegistry.get(key)
            except (ValueError, KeyError):
                new_fields[key] = data
                continue

            if spec.varies_in_space_and_time or spec.varies_in_space_only:
                # Apply CutMix to spatial region
                mixed = data.clone()
                s = spec.tile_size_multiplier
                sy1, sy2 = y1 * s, y2 * s
                sx1, sx2 = x1 * s, x2 * s

                if other_data is not None:
                    if mixed.ndim >= 4:
                        # Batched: (B, H, W, ...) or spatial dims at 0,1
                        mixed[:, sy1:sy2, sx1:sx2] = other_data[:, sy1:sy2, sx1:sx2]
                    else:
                        mixed[sy1:sy2, sx1:sx2] = other_data[sy1:sy2, sx1:sx2]

                new_fields[key] = mixed
            else:
                new_fields[key] = data

        return GeoSample(**new_fields)


# ---------------------------------------------------------------------------
# Sensor Dropout
# ---------------------------------------------------------------------------


class SensorDropout:
    """Randomly drop entire sensors during training.

    This is a new augmentation not present in the original codebase.
    It encourages the model to be robust to missing sensors and prevents
    over-reliance on any single sensor.

    Args:
        drop_prob: Probability of dropping each sensor.
        min_sensors: Minimum number of sensors to keep (never drop all).
    """

    def __init__(self, drop_prob: float = 0.2, min_sensors: int = 1) -> None:
        self.drop_prob = drop_prob
        self.min_sensors = min_sensors

    def __call__(self, sample: GeoSample) -> GeoSample:
        sensors = sample.present_sensors
        if len(sensors) <= self.min_sensors:
            return sample

        # Decide which sensors to keep
        keep = []
        drop = []
        for s in sensors:
            if random.random() < self.drop_prob:
                drop.append(s)
            else:
                keep.append(s)

        # Ensure minimum sensors are kept
        while len(keep) < self.min_sensors and drop:
            restored = drop.pop(random.randint(0, len(drop) - 1))
            keep.append(restored)

        # Build new sample without dropped sensors
        new_fields: dict[str, Any] = {}
        for key in sample.present_keys:
            if key in ("timestamps", "latlon"):
                new_fields[key] = sample[key]
            elif key in keep:
                new_fields[key] = sample[key]
            # Dropped sensors are simply omitted

        return GeoSample(**new_fields)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class Compose:
    """Apply a sequence of transforms."""

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, sample: GeoSample) -> GeoSample:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---------------------------------------------------------------------------
# Registry-based construction (simple dict dispatch)
# ---------------------------------------------------------------------------

_TRANSFORM_REGISTRY: dict[str, type] = {
    "no_transform": NoTransform,
    "dihedral": RandomDihedralTransform,
    "cutmix": CutMix,
    "sensor_dropout": SensorDropout,
}


def build_transform(config: dict[str, Any]) -> Transform:
    """Build a transform from a configuration dictionary.

    Args:
        config: Dictionary with ``"type"`` key and optional kwargs.
            For composed transforms, use ``"type": "compose"`` with a
            ``"transforms"`` list.

    Returns:
        A callable transform.

    Example::

        build_transform({"type": "dihedral"})
        build_transform({
            "type": "compose",
            "transforms": [
                {"type": "dihedral"},
                {"type": "sensor_dropout", "drop_prob": 0.3},
            ]
        })
    """
    transform_type = config.get("type", "no_transform")

    if transform_type == "compose":
        sub_configs = config.get("transforms", [])
        transforms = [build_transform(c) for c in sub_configs]
        return Compose(transforms)

    cls = _TRANSFORM_REGISTRY.get(transform_type)
    if cls is None:
        raise ValueError(
            f"Unknown transform type: {transform_type!r}. "
            f"Available: {list(_TRANSFORM_REGISTRY.keys())}"
        )

    # Pass remaining config as kwargs (excluding "type")
    kwargs = {k: v for k, v in config.items() if k != "type"}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Backward-compatible TransformConfig
# ---------------------------------------------------------------------------


@dataclass
class TransformConfig:
    """Configuration for transforms (backward-compatible interface).

    Args:
        transform_type: Name of the transform.
        transform_kwargs: Additional keyword arguments for the transform.
    """

    transform_type: str = "no_transform"
    transform_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Transform:
        """Build the transform."""
        config = {"type": self.transform_type, **self.transform_kwargs}
        return build_transform(config)
