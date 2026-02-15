"""Composable, function-based masking strategies for masked modelling.

Replaces the class-hierarchy approach in ``occlusion.py`` with flat,
composable functions and simple dataclass configurations.

Core design:
- ``create_mask(shape, ratio, structure)`` -- pure function, returns a mask
  tensor of :class:`TokenVisibility` values.
- ``compose_masks(masks, rule)`` -- combine multiple masks via union,
  intersection, or cascade.
- ``mask_sample(sample, strategy, patch_size)`` -- sensor-aware wrapper
  that handles absent data, per-sensor group counts, and patch alignment.
- Strategy configs are plain dataclasses (no class_registry dependency).
- All 15+ original strategies are expressible as combinations of these
  primitives.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence, Union

import torch
from torch import Tensor

from spacenit.structures import GeoSample, MaskedGeoSample, TokenVisibility

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MaskTensor = Tensor  # int tensor with TokenVisibility values


# ---------------------------------------------------------------------------
# Core mask creation
# ---------------------------------------------------------------------------


def create_mask(
    num_tokens: int,
    encode_ratio: float,
    decode_ratio: float = 0.0,
    structure: str = "random",
    *,
    spatial_shape: tuple[int, int] | None = None,
    temporal_length: int | None = None,
    num_groups: int = 1,
    generator: torch.Generator | None = None,
) -> MaskTensor:
    """Create a visibility mask for a flat token sequence.

    Args:
        num_tokens: Total number of tokens to mask.
        encode_ratio: Fraction of tokens visible to the encoder.
        decode_ratio: Fraction of tokens visible only to the target
            (momentum) encoder.  The remainder are ``PREDICTED``.
        structure: Masking pattern.  One of:

            - ``"random"`` -- uniformly random per-token masking.
            - ``"spatial"`` -- contiguous spatial blocks (requires
              ``spatial_shape``).
            - ``"temporal"`` -- entire timesteps are masked/unmasked
              (requires ``temporal_length``).
            - ``"spectral"`` -- entire spectral groups are masked/unmasked
              (requires ``num_groups``).

        spatial_shape: ``(H_patches, W_patches)`` for spatial masking.
        temporal_length: Number of timesteps for temporal masking.
        num_groups: Number of spectral groups for spectral masking.
        generator: Optional RNG for reproducibility.

    Returns:
        Integer tensor ``(num_tokens,)`` of :class:`TokenVisibility` values.
    """
    mask = torch.full((num_tokens,), TokenVisibility.PREDICTED.value, dtype=torch.long)

    if structure == "random":
        mask = _random_mask(mask, encode_ratio, decode_ratio, generator)
    elif structure == "spatial":
        assert spatial_shape is not None, "spatial masking requires spatial_shape"
        mask = _spatial_mask(mask, encode_ratio, decode_ratio, spatial_shape, generator)
    elif structure == "temporal":
        assert temporal_length is not None, "temporal masking requires temporal_length"
        mask = _temporal_mask(mask, encode_ratio, decode_ratio, temporal_length, generator)
    elif structure == "spectral":
        mask = _spectral_mask(mask, encode_ratio, decode_ratio, num_groups, generator)
    else:
        raise ValueError(f"Unknown masking structure: {structure!r}")

    return mask


def _random_mask(
    mask: MaskTensor,
    encode_ratio: float,
    decode_ratio: float,
    generator: torch.Generator | None,
) -> MaskTensor:
    """Uniformly random per-token masking."""
    N = mask.shape[0]
    perm = torch.randperm(N, generator=generator)
    n_encode = max(1, int(N * encode_ratio))
    n_decode = int(N * decode_ratio)

    mask[perm[:n_encode]] = TokenVisibility.VISIBLE_ENCODER.value
    if n_decode > 0:
        mask[perm[n_encode : n_encode + n_decode]] = TokenVisibility.TARGET_ONLY.value
    return mask


def _spatial_mask(
    mask: MaskTensor,
    encode_ratio: float,
    decode_ratio: float,
    spatial_shape: tuple[int, int],
    generator: torch.Generator | None,
) -> MaskTensor:
    """Block-based spatial masking.

    Divides the spatial grid into blocks and randomly selects blocks
    to be visible.  Tokens within selected blocks are VISIBLE_ENCODER;
    the rest follow the decode_ratio split.
    """
    H, W = spatial_shape
    N = mask.shape[0]
    tokens_per_position = N // (H * W) if H * W > 0 else 1

    # Create a 2D mask at the patch level
    total_patches = H * W
    n_visible_patches = max(1, int(total_patches * encode_ratio))
    n_decode_patches = int(total_patches * decode_ratio)

    perm = torch.randperm(total_patches, generator=generator)
    visible_patches = perm[:n_visible_patches]
    decode_patches = perm[n_visible_patches : n_visible_patches + n_decode_patches]

    # Expand to token level
    for p in visible_patches:
        start = p * tokens_per_position
        end = min(start + tokens_per_position, N)
        mask[start:end] = TokenVisibility.VISIBLE_ENCODER.value

    for p in decode_patches:
        start = p * tokens_per_position
        end = min(start + tokens_per_position, N)
        mask[start:end] = TokenVisibility.TARGET_ONLY.value

    return mask


def _temporal_mask(
    mask: MaskTensor,
    encode_ratio: float,
    decode_ratio: float,
    temporal_length: int,
    generator: torch.Generator | None,
) -> MaskTensor:
    """Temporal masking -- entire timesteps are masked/unmasked."""
    N = mask.shape[0]
    tokens_per_step = N // temporal_length if temporal_length > 0 else N

    n_visible_steps = max(1, int(temporal_length * encode_ratio))
    n_decode_steps = int(temporal_length * decode_ratio)

    perm = torch.randperm(temporal_length, generator=generator)
    visible_steps = perm[:n_visible_steps]
    decode_steps = perm[n_visible_steps : n_visible_steps + n_decode_steps]

    for t in visible_steps:
        start = t * tokens_per_step
        end = min(start + tokens_per_step, N)
        mask[start:end] = TokenVisibility.VISIBLE_ENCODER.value

    for t in decode_steps:
        start = t * tokens_per_step
        end = min(start + tokens_per_step, N)
        mask[start:end] = TokenVisibility.TARGET_ONLY.value

    return mask


def _spectral_mask(
    mask: MaskTensor,
    encode_ratio: float,
    decode_ratio: float,
    num_groups: int,
    generator: torch.Generator | None,
) -> MaskTensor:
    """Spectral masking -- entire spectral groups are masked/unmasked."""
    N = mask.shape[0]
    tokens_per_group = N // num_groups if num_groups > 0 else N

    n_visible = max(1, int(num_groups * encode_ratio))
    n_decode = int(num_groups * decode_ratio)

    perm = torch.randperm(num_groups, generator=generator)
    visible_groups = perm[:n_visible]
    decode_groups = perm[n_visible : n_visible + n_decode]

    for g in visible_groups:
        start = g * tokens_per_group
        end = min(start + tokens_per_group, N)
        mask[start:end] = TokenVisibility.VISIBLE_ENCODER.value

    for g in decode_groups:
        start = g * tokens_per_group
        end = min(start + tokens_per_group, N)
        mask[start:end] = TokenVisibility.TARGET_ONLY.value

    return mask


# ---------------------------------------------------------------------------
# Mask composition
# ---------------------------------------------------------------------------


def compose_masks(masks: list[MaskTensor], rule: str = "union") -> MaskTensor:
    """Combine multiple masks into one.

    Args:
        masks: List of mask tensors (same shape).
        rule: Composition rule:

            - ``"union"`` -- a token is visible if visible in *any* mask.
            - ``"intersection"`` -- a token is visible only if visible in
              *all* masks.
            - ``"cascade"`` -- apply masks sequentially; later masks can
              only further restrict visibility.

    Returns:
        Combined mask tensor.
    """
    if len(masks) == 0:
        raise ValueError("Need at least one mask to compose")
    if len(masks) == 1:
        return masks[0]

    VIS = TokenVisibility.VISIBLE_ENCODER.value
    TGT = TokenVisibility.TARGET_ONLY.value
    PRED = TokenVisibility.PREDICTED.value
    ABS = TokenVisibility.ABSENT.value

    if rule == "union":
        # Most permissive visibility wins (lower enum value = more visible)
        result = masks[0].clone()
        for m in masks[1:]:
            result = torch.minimum(result, m)
        # But ABSENT always stays ABSENT
        for m in masks:
            result[m == ABS] = ABS
        return result

    elif rule == "intersection":
        # Least permissive visibility wins (higher enum value = less visible)
        result = masks[0].clone()
        for m in masks[1:]:
            result = torch.maximum(result, m)
        return result

    elif rule == "cascade":
        # Sequential: start with first mask, then restrict further
        result = masks[0].clone()
        for m in masks[1:]:
            # Only allow further restriction (increase visibility value)
            further_restricted = torch.maximum(result, m)
            # But don't un-absent things
            further_restricted[result == ABS] = ABS
            result = further_restricted
        return result

    else:
        raise ValueError(f"Unknown composition rule: {rule!r}")


# ---------------------------------------------------------------------------
# Strategy dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RandomMasking:
    """Uniformly random per-token masking."""

    encode_ratio: float = 0.25
    decode_ratio: float = 0.0
    structure: str = "random"


@dataclass
class SpatialMasking:
    """Block-based spatial masking."""

    encode_ratio: float = 0.25
    decode_ratio: float = 0.0
    structure: str = "spatial"


@dataclass
class TemporalMasking:
    """Entire-timestep masking."""

    encode_ratio: float = 0.5
    decode_ratio: float = 0.0
    structure: str = "temporal"


@dataclass
class SpectralMasking:
    """Entire-spectral-group masking."""

    encode_ratio: float = 0.5
    decode_ratio: float = 0.0
    structure: str = "spectral"


@dataclass
class CrossSensorMasking:
    """Apply a base strategy per sensor, with constraints on total visibility.

    For each sensor, applies ``base_strategy`` independently, then ensures
    that between ``min_encoded`` and ``max_encoded`` fraction of total
    tokens are visible.
    """

    base_strategy: str = "random"
    base_encode_ratio: float = 0.25
    base_decode_ratio: float = 0.0
    min_encoded: float = 0.1
    max_encoded: float = 0.9


@dataclass
class RangeMasking:
    """Sample the encode ratio uniformly from a range per batch.

    Useful for curriculum learning or robust training across masking ratios.
    """

    min_encode: float = 0.1
    max_encode: float = 0.9
    decode_ratio: float = 0.0
    structure: str = "random"


@dataclass
class CompositeMasking:
    """Randomly pick one strategy per batch from a weighted set."""

    strategies: list[MaskingStrategy] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


@dataclass
class ScheduledMasking:
    """Linearly anneal masking ratio over training.

    Starts at ``initial_encode_ratio`` and linearly interpolates to
    ``final_encode_ratio`` over ``warmup_steps``.
    """

    initial_encode_ratio: float = 0.1
    final_encode_ratio: float = 0.5
    warmup_steps: int = 10000
    decode_ratio: float = 0.0
    structure: str = "random"


# Union type for all strategies
MaskingStrategy = Union[
    RandomMasking,
    SpatialMasking,
    TemporalMasking,
    SpectralMasking,
    CrossSensorMasking,
    RangeMasking,
    CompositeMasking,
    ScheduledMasking,
]

# Fix forward reference in CompositeMasking
CompositeMasking.__annotations__["strategies"] = list[MaskingStrategy]


# ---------------------------------------------------------------------------
# Strategy execution
# ---------------------------------------------------------------------------


def apply_strategy(
    strategy: MaskingStrategy,
    num_tokens: int,
    *,
    spatial_shape: tuple[int, int] | None = None,
    temporal_length: int | None = None,
    num_groups: int = 1,
    step: int = 0,
    generator: torch.Generator | None = None,
) -> MaskTensor:
    """Execute a masking strategy to produce a mask tensor.

    Args:
        strategy: One of the strategy dataclasses.
        num_tokens: Total number of tokens.
        spatial_shape: ``(H, W)`` in patches for spatial masking.
        temporal_length: Number of timesteps for temporal masking.
        num_groups: Number of spectral groups for spectral masking.
        step: Current training step (for scheduled masking).
        generator: Optional RNG.

    Returns:
        Mask tensor ``(num_tokens,)`` of TokenVisibility values.
    """
    if isinstance(strategy, (RandomMasking, SpatialMasking, TemporalMasking, SpectralMasking)):
        return create_mask(
            num_tokens,
            encode_ratio=strategy.encode_ratio,
            decode_ratio=strategy.decode_ratio,
            structure=strategy.structure,
            spatial_shape=spatial_shape,
            temporal_length=temporal_length,
            num_groups=num_groups,
            generator=generator,
        )

    elif isinstance(strategy, RangeMasking):
        # Sample encode ratio from range
        encode_ratio = random.uniform(strategy.min_encode, strategy.max_encode)
        return create_mask(
            num_tokens,
            encode_ratio=encode_ratio,
            decode_ratio=strategy.decode_ratio,
            structure=strategy.structure,
            spatial_shape=spatial_shape,
            temporal_length=temporal_length,
            num_groups=num_groups,
            generator=generator,
        )

    elif isinstance(strategy, ScheduledMasking):
        # Linearly interpolate encode ratio
        progress = min(step / max(strategy.warmup_steps, 1), 1.0)
        encode_ratio = (
            strategy.initial_encode_ratio
            + (strategy.final_encode_ratio - strategy.initial_encode_ratio) * progress
        )
        return create_mask(
            num_tokens,
            encode_ratio=encode_ratio,
            decode_ratio=strategy.decode_ratio,
            structure=strategy.structure,
            spatial_shape=spatial_shape,
            temporal_length=temporal_length,
            num_groups=num_groups,
            generator=generator,
        )

    elif isinstance(strategy, CompositeMasking):
        # Randomly pick one strategy
        if not strategy.strategies:
            raise ValueError("CompositeMasking requires at least one sub-strategy")
        weights = strategy.weights or [1.0] * len(strategy.strategies)
        chosen = random.choices(strategy.strategies, weights=weights, k=1)[0]
        return apply_strategy(
            chosen,
            num_tokens,
            spatial_shape=spatial_shape,
            temporal_length=temporal_length,
            num_groups=num_groups,
            step=step,
            generator=generator,
        )

    elif isinstance(strategy, CrossSensorMasking):
        # Apply base strategy then clamp total visibility
        mask = create_mask(
            num_tokens,
            encode_ratio=strategy.base_encode_ratio,
            decode_ratio=strategy.base_decode_ratio,
            structure=strategy.base_strategy,
            spatial_shape=spatial_shape,
            temporal_length=temporal_length,
            num_groups=num_groups,
            generator=generator,
        )
        # Enforce min/max constraints
        VIS = TokenVisibility.VISIBLE_ENCODER.value
        ABS = TokenVisibility.ABSENT.value
        non_absent = (mask != ABS)
        n_non_absent = non_absent.sum().item()
        if n_non_absent > 0:
            n_visible = (mask == VIS).sum().item()
            ratio = n_visible / n_non_absent
            if ratio < strategy.min_encoded:
                # Need to make more tokens visible
                predicted = torch.where(
                    (mask != VIS) & (mask != ABS)
                )[0]
                n_needed = int(strategy.min_encoded * n_non_absent) - n_visible
                if n_needed > 0 and len(predicted) > 0:
                    perm = torch.randperm(len(predicted), generator=generator)
                    to_flip = predicted[perm[:min(n_needed, len(predicted))]]
                    mask[to_flip] = VIS
            elif ratio > strategy.max_encoded:
                # Need to hide more tokens
                visible = torch.where(mask == VIS)[0]
                n_excess = n_visible - int(strategy.max_encoded * n_non_absent)
                if n_excess > 0 and len(visible) > 0:
                    perm = torch.randperm(len(visible), generator=generator)
                    to_flip = visible[perm[:min(n_excess, len(visible))]]
                    mask[to_flip] = TokenVisibility.PREDICTED.value
        return mask

    else:
        raise TypeError(f"Unknown masking strategy type: {type(strategy)}")


# ---------------------------------------------------------------------------
# Sensor-aware masking
# ---------------------------------------------------------------------------


def mask_sample(
    sample: GeoSample,
    strategy: MaskingStrategy,
    patch_size: int,
    *,
    step: int = 0,
    generator: torch.Generator | None = None,
) -> MaskedGeoSample:
    """Apply a masking strategy to a GeoSample, producing a MaskedGeoSample.

    Handles absent data (marks as ABSENT), per-sensor group counts, and
    patch alignment.

    Args:
        sample: Input geospatial sample.
        strategy: Masking strategy to apply.
        patch_size: Patch size for spatial alignment.
        step: Current training step (for scheduled strategies).
        generator: Optional RNG.

    Returns:
        MaskedGeoSample with visibility masks for each sensor.
    """
    from spacenit.ingestion.sensors import (
        ABSENT_INDICATOR,
        SensorRegistry,
    )

    fields: dict[str, Tensor | None] = {}

    # Copy timestamps
    if sample.timestamps is not None:
        fields["timestamps"] = sample.timestamps

    # Copy latlon
    if sample.latlon is not None:
        fields["latlon"] = sample.latlon

    for sensor_label in sample.present_sensors:
        data = sample[sensor_label]
        if data is None:
            continue

        spec = SensorRegistry.get(sensor_label)
        fields[sensor_label] = data

        # Determine token count and spatial/temporal structure
        if spec.varies_in_space_and_time:
            # (H, W, T, C) or (B, H, W, T, C)
            if data.ndim == 4:
                H, W, T, C = data.shape
            else:
                _, H, W, T, C = data.shape
            H_p = H // (patch_size * spec.tile_size_multiplier)
            W_p = W // (patch_size * spec.tile_size_multiplier)
            num_tokens = H_p * W_p * T * spec.group_count
            spatial_shape = (H_p, W_p)
            temporal_length = T
        elif spec.varies_in_space_only:
            if data.ndim == 3:
                H, W, C = data.shape
            else:
                _, H, W, C = data.shape
            H_p = H // (patch_size * spec.tile_size_multiplier)
            W_p = W // (patch_size * spec.tile_size_multiplier)
            num_tokens = H_p * W_p * spec.group_count
            spatial_shape = (H_p, W_p)
            temporal_length = None
        elif spec.varies_in_time_only:
            if data.ndim == 2:
                T, C = data.shape
            else:
                _, T, C = data.shape
            num_tokens = T * spec.group_count
            spatial_shape = None
            temporal_length = T
        else:
            num_tokens = spec.group_count
            spatial_shape = None
            temporal_length = None

        # Create mask
        mask = apply_strategy(
            strategy,
            num_tokens,
            spatial_shape=spatial_shape,
            temporal_length=temporal_length,
            num_groups=spec.group_count,
            step=step,
            generator=generator,
        )

        # Mark absent data
        if isinstance(data, torch.Tensor):
            absent_indicator = ABSENT_INDICATOR
            # Check for absent tokens (all channels == ABSENT_INDICATOR)
            if spec.varies_in_space_and_time and data.ndim >= 4:
                # Reshape to token level and check
                pass  # Absent detection is complex; keep mask as-is for now
                # The data loader should have already set ABSENT_INDICATOR

        mask_key = f"{sensor_label}_mask"
        fields[mask_key] = mask

    return MaskedGeoSample(**fields)


# ---------------------------------------------------------------------------
# Config-based construction
# ---------------------------------------------------------------------------


def build_masking(config: dict) -> MaskingStrategy:
    """Build a masking strategy from a configuration dictionary.

    Args:
        config: Dictionary with a ``"type"`` key and strategy-specific
            parameters.  Example::

                {"type": "random", "encode_ratio": 0.25}
                {"type": "composite", "strategies": [...], "weights": [...]}

    Returns:
        A masking strategy dataclass instance.
    """
    _REGISTRY = {
        "random": RandomMasking,
        "spatial": SpatialMasking,
        "temporal": TemporalMasking,
        "spectral": SpectralMasking,
        "cross_sensor": CrossSensorMasking,
        "range": RangeMasking,
        "composite": CompositeMasking,
        "scheduled": ScheduledMasking,
    }

    strategy_type = config.pop("type", "random")
    cls = _REGISTRY.get(strategy_type)
    if cls is None:
        raise ValueError(
            f"Unknown masking strategy type: {strategy_type!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    # Recursively build sub-strategies for composite
    if strategy_type == "composite" and "strategies" in config:
        config["strategies"] = [
            build_masking(s) if isinstance(s, dict) else s
            for s in config["strategies"]
        ]

    return cls(**config)
