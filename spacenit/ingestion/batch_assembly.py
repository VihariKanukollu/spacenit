"""Collate / batch-assembly functions for the SpaceNit ingestion pipeline."""

from __future__ import annotations

import torch

from spacenit.ingestion.augmentations import Transform
from spacenit.ingestion.sensors import SensorRegistry
from spacenit.structures import (
    GeoSample,
    MaskedGeoSample,
)

# MaskingStrategy is expected to live in the pipeline package.  We import it
# with a TYPE_CHECKING guard so the module can still be loaded without the
# full training stack.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class _MaskingStrategyProtocol:
    """Structural type for masking strategies (duck-typed)."""

    def apply_mask(
        self, sample: GeoSample, patch_size: int
    ) -> MaskedGeoSample: ...


# We accept any object that satisfies the protocol above.
MaskingStrategyLike = _MaskingStrategyProtocol


def collate_geo_tiles(
    batch: list[tuple[int, GeoSample]],
) -> tuple[int, GeoSample]:
    """Collate function that automatically handles any modalities present in the samples."""

    # Stack tensors while handling None values
    def stack_or_none(attr: str) -> torch.Tensor | None:
        """Stack the tensors while handling None values."""
        # For partially missing samples we use ABSENT_INDICATOR so we only
        # check the first sample.
        if getattr(batch[0][1], attr) is None:
            return None
        stacked_tensor = torch.stack(
            [torch.from_numpy(getattr(sample, attr)) for _, sample in batch],
            dim=0,
        )
        # Reorder into model-friendly layouts.
        #
        # Ingestion H5 convention is typically (H, W, T, C) for spacetime sensors
        # and (H, W, C) for spatial-only sensors. The model expects channel-first:
        # - temporal: (B, C, H, W, T)
        # - spatial:  (B, C, H, W)
        if attr in ("timestamps", "latlon"):
            return stacked_tensor
        spec = SensorRegistry.get(attr)
        if spec.varies_in_space_and_time and stacked_tensor.ndim == 5:
            # (B,H,W,T,C) -> (B,C,H,W,T)
            return stacked_tensor.permute(0, 4, 1, 2, 3).contiguous()
        if spec.varies_in_space_only:
            # Some spatial-only sensors are stored with a singleton temporal dim:
            # (H,W,1,C) -> batched (B,H,W,1,C). Squeeze it.
            if stacked_tensor.ndim == 5 and stacked_tensor.shape[3] == 1:
                stacked_tensor = stacked_tensor.squeeze(3)
            if stacked_tensor.ndim == 4:
                # (B,H,W,C) -> (B,C,H,W)
                return stacked_tensor.permute(0, 3, 1, 2).contiguous()
        return stacked_tensor

    patch_size, batch_zero = batch[0]
    sample_fields = batch_zero.present_keys

    # Create a dictionary of stacked tensors for each field
    collated_dict = {field: stack_or_none(field) for field in sample_fields}
    return patch_size, GeoSample(**collated_dict)


def collate_single_masked_batched(
    batch: list[tuple[int, GeoSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategyLike,
) -> tuple[int, MaskedGeoSample]:
    """Collate function that applies transform and masking to the full batch.

    This function first collates raw GeoSamples into a batched tensor, then
    applies transform and masking to the entire batch at once, enabling
    vectorized operations.

    Args:
        batch: List of (patch_size, GeoSample) tuples.
        transform: Optional transform to apply to the batch.
        masking_strategy: Masking strategy to apply to the batch.

    Returns:
        A tuple of (patch_size, MaskedGeoSample).
    """
    # First, collate raw samples into a batched GeoSample
    patch_size, stacked_sample = collate_geo_tiles(batch)

    # Apply transform to the batch (if configured)
    if transform is not None:
        stacked_sample = transform(stacked_sample)

    # Apply masking to the batch
    masked_sample = masking_strategy.apply_mask(stacked_sample, patch_size)

    return patch_size, masked_sample


def collate_double_masked_batched(
    batch: list[tuple[int, GeoSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategyLike,
    masking_strategy_b: MaskingStrategyLike | None,
) -> tuple[int, MaskedGeoSample, MaskedGeoSample]:
    """Collate function that applies transform and two masking strategies to the full batch.

    This function first collates raw GeoSamples into a batched tensor, then
    applies transform and two independent masking strategies to the entire
    batch at once, enabling vectorized operations.

    Args:
        batch: List of (patch_size, GeoSample) tuples.
        transform: Optional transform to apply to the batch.
        masking_strategy: First masking strategy to apply.
        masking_strategy_b: Second masking strategy to apply.  If None, uses
            *masking_strategy*.

    Returns:
        A tuple of (patch_size, MaskedGeoSample_a, MaskedGeoSample_b).
    """
    # First, collate raw samples into a batched GeoSample
    patch_size, stacked_sample = collate_geo_tiles(batch)

    # Apply transform independently for each view (if configured). For random
    # transforms (e.g., dihedral), this yields two different augmentations of
    # the same underlying batch, matching common SSL protocols.
    if transform is not None:
        sample_a = transform(stacked_sample)
        sample_b = transform(stacked_sample)
    else:
        sample_a = stacked_sample
        sample_b = stacked_sample

    # Apply masking strategies to each view.
    masked_sample_a = masking_strategy.apply_mask(sample_a, patch_size)
    strategy_b = masking_strategy_b if masking_strategy_b is not None else masking_strategy
    masked_sample_b = strategy_b.apply_mask(sample_b, patch_size)

    return patch_size, masked_sample_a, masked_sample_b
