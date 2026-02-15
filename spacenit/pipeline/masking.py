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
from typing import Any, Protocol, Sequence, Union

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

    Sensors listed in ``decode_only_sensors`` are never visible to the
    encoder -- all their tokens are prediction targets.
    """

    base_strategy: str = "random"
    base_encode_ratio: float = 0.25
    base_decode_ratio: float = 0.0
    min_encoded: float = 0.1
    max_encoded: float = 0.9
    decode_only_sensors: list[str] = field(default_factory=list)


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


def _mark_absent_tokens(
    mask: Tensor,
    data: Tensor,
    spec: Any,
    patch_size: int,
    absent_indicator: int,
    absent_value: int,
) -> None:
    """In-place: set mask entries to *absent_value* where data is missing.

    A token is considered absent when **all** of its underlying data values
    equal ``absent_indicator``.  The check is performed at the token level
    (after spatial patchification and temporal slicing).

    Supports spatial (``H, W``), spatio-temporal (``H, W, T``), temporal
    (``T``), and scalar sensor layouts.  Batch dimensions are stripped
    before comparison (the mask is 1-D over tokens).
    """
    # Work on the last-few dims regardless of batch prefix.
    d = data.detach()

    # Squeeze leading batch dims to get a canonical shape.
    while d.ndim > 4 and d.shape[0] == 1:
        d = d.squeeze(0)

    total_channels = getattr(spec, "total_channels", None)

    if spec.varies_in_space_and_time:
        # Canonical: (C, H, W, T)
        if d.ndim == 5:
            if total_channels is not None and d.shape[1] == total_channels:
                d = d[0]  # drop batch -> (C, H, W, T)
            else:
                d = d[0].permute(3, 0, 1, 2)  # (H,W,T,C) -> (C,H,W,T)
        elif d.ndim == 4:
            if total_channels is None or d.shape[0] != total_channels:
                d = d.permute(3, 0, 1, 2)  # (H,W,T,C) -> (C,H,W,T)
        else:
            return  # can't determine layout

        C, H, W, T = d.shape
        P = patch_size * spec.tile_size_multiplier
        Hp, Wp = H // P, W // P
        # Reshape to (C, Hp, P, Wp, P, T) -> check per-patch-per-timestep
        if H % P != 0 or W % P != 0:
            return
        d = d.reshape(C, Hp, P, Wp, P, T)
        # All channels absent in a patch-timestep? -> (Hp, Wp, T) bool
        absent_map = (d == absent_indicator).all(dim=(0, 2, 4))
        # Repeat for each spectral group -> (num_groups * Hp * Wp * T,)
        absent_flat = absent_map.reshape(-1).repeat(spec.group_count)
        if absent_flat.shape[0] == mask.shape[0]:
            mask[absent_flat] = absent_value

    elif spec.varies_in_space_only:
        if d.ndim == 5:
            d = d[0]  # drop batch
        if d.ndim == 4:
            if total_channels is not None and d.shape[1] == total_channels:
                d = d[0]  # drop batch -> (C, H, W)
            elif total_channels is not None and d.shape[0] == total_channels:
                pass  # already (C, H, W)
            else:
                d = d.permute(2, 0, 1) if d.ndim == 3 else d[0].permute(2, 0, 1)
        elif d.ndim == 3:
            if total_channels is None or d.shape[0] != total_channels:
                d = d.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:
            return

        C, H, W = d.shape
        P = patch_size * spec.tile_size_multiplier
        Hp, Wp = H // P, W // P
        if H % P != 0 or W % P != 0:
            return
        d = d.reshape(C, Hp, P, Wp, P)
        absent_map = (d == absent_indicator).all(dim=(0, 2, 4))  # (Hp, Wp)
        absent_flat = absent_map.reshape(-1).repeat(spec.group_count)
        if absent_flat.shape[0] == mask.shape[0]:
            mask[absent_flat] = absent_value

    elif spec.varies_in_time_only:
        if d.ndim >= 3:
            d = d.reshape(-1, d.shape[-1]) if d.ndim > 2 else d
        if d.ndim == 2:
            # (C, T) or (T, C)
            if total_channels is not None and d.shape[0] == total_channels:
                absent_per_t = (d == absent_indicator).all(dim=0)  # (T,)
            else:
                absent_per_t = (d == absent_indicator).all(dim=1)  # (T,)
            absent_flat = absent_per_t.repeat(spec.group_count)
            if absent_flat.shape[0] == mask.shape[0]:
                mask[absent_flat] = absent_value

    # Scalar sensors: if all values are absent, mark the single token
    elif spec.group_count == 1 and mask.shape[0] == 1:
        if (d == absent_indicator).all():
            mask[0] = absent_value


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
        total_channels = getattr(spec, "total_channels", None)

        if spec.varies_in_space_and_time:
            # Accept either channel-last (H,W,T,C)/(B,H,W,T,C) or channel-first
            # (C,H,W,T)/(B,C,H,W,T) layouts (the ingestion collator may permute).
            if data.ndim == 5:
                if total_channels is not None and data.shape[1] == total_channels:
                    _, C, H, W, T = data.shape
                else:
                    _, H, W, T, C = data.shape
            elif data.ndim == 4:
                if total_channels is not None and data.shape[0] == total_channels:
                    C, H, W, T = data.shape
                else:
                    H, W, T, C = data.shape
            else:
                raise ValueError(
                    f"Unexpected ndim={data.ndim} for spacetime sensor {sensor_label}"
                )
            H_p = H // (patch_size * spec.tile_size_multiplier)
            W_p = W // (patch_size * spec.tile_size_multiplier)
            num_tokens = H_p * W_p * T * spec.group_count
            spatial_shape = (H_p, W_p)
            temporal_length = T
        elif spec.varies_in_space_only:
            # Accept (H,W,C)/(B,H,W,C) or channel-first (C,H,W)/(B,C,H,W).
            # Some spatial-only sensors may be stored with a singleton time dim:
            # (H,W,1,C)/(B,H,W,1,C) or (C,H,W,1)/(B,C,H,W,1).
            if data.ndim == 5:
                if total_channels is not None and data.shape[1] == total_channels:
                    _, C, H, W, T = data.shape
                else:
                    _, H, W, T, C = data.shape
                # Treat singleton-T as spatial-only.
                if T != 1:
                    raise ValueError(
                        f"Expected singleton time dim for spatial-only sensor {sensor_label}, got T={T}"
                    )
            elif data.ndim == 4:
                if total_channels is not None and data.shape[1] == total_channels:
                    _, C, H, W = data.shape
                else:
                    _, H, W, C = data.shape
            elif data.ndim == 3:
                if total_channels is not None and data.shape[0] == total_channels:
                    C, H, W = data.shape
                else:
                    H, W, C = data.shape
            else:
                raise ValueError(
                    f"Unexpected ndim={data.ndim} for spatial sensor {sensor_label}"
                )
            H_p = H // (patch_size * spec.tile_size_multiplier)
            W_p = W // (patch_size * spec.tile_size_multiplier)
            num_tokens = H_p * W_p * spec.group_count
            spatial_shape = (H_p, W_p)
            temporal_length = None
        elif spec.varies_in_time_only:
            # Accept (T,C)/(B,T,C) or channel-first (C,T)/(B,C,T).
            if data.ndim == 3:
                if total_channels is not None and data.shape[1] == total_channels:
                    _, C, T = data.shape
                else:
                    _, T, C = data.shape
            elif data.ndim == 2:
                if total_channels is not None and data.shape[0] == total_channels:
                    C, T = data.shape
                else:
                    T, C = data.shape
            else:
                raise ValueError(
                    f"Unexpected ndim={data.ndim} for time-only sensor {sensor_label}"
                )
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

        # Mark absent data: any token whose underlying data is entirely
        # ABSENT_INDICATOR should be marked ABSENT regardless of the
        # masking strategy's decision.
        if isinstance(data, torch.Tensor):
            ABS = TokenVisibility.ABSENT.value
            _mark_absent_tokens(
                mask, data, spec, patch_size, ABSENT_INDICATOR, ABS,
            )

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
        "cross_sensor_random": CrossSensorMasking,
        "range": RangeMasking,
        "composite": CompositeMasking,
        "scheduled": ScheduledMasking,
    }

    strategy_type = config.get("type", "random")
    cls = _REGISTRY.get(strategy_type)
    if cls is None:
        raise ValueError(
            f"Unknown masking strategy type: {strategy_type!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    # Build kwargs without the "type" key (don't mutate the caller's dict)
    kwargs = {k: v for k, v in config.items() if k != "type"}

    # Recursively build sub-strategies for composite
    if strategy_type == "composite" and "strategies" in kwargs:
        kwargs["strategies"] = [
            build_masking(s) if isinstance(s, dict) else s
            for s in kwargs["strategies"]
        ]

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Ingestion adapter (DataLoader expects apply_mask())
# ---------------------------------------------------------------------------


class MaskingPolicy(Protocol):
    """Dataloader-facing masking interface."""

    def apply_mask(self, sample: GeoSample, patch_size: int) -> MaskedGeoSample: ...


class _MaskingPolicyFromStrategy:
    def __init__(self, strategy: MaskingStrategy) -> None:
        self._strategy = strategy
        self._step = 0

    def apply_mask(self, sample: GeoSample, patch_size: int) -> MaskedGeoSample:
        masked = mask_sample(
            sample,
            self._strategy,
            patch_size=patch_size,
            step=self._step,
            generator=None,
        )
        self._step += 1
        return masked


def build_masking_policy(config: dict[str, Any]) -> MaskingPolicy:
    """Build a dataloader masking policy from a config dict."""
    strategy = build_masking(dict(config))
    return _MaskingPolicyFromStrategy(strategy)
