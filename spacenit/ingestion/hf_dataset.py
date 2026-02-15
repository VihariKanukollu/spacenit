"""HuggingFace dataset wrapper for geospatial data.

Wraps the ``allenai/olmoearth_pretrain_dataset`` (or any compatible
GeoTIFF-based dataset) as a PyTorch ``Dataset`` with distributed-aware
data loading.

Replaces the H5-based ``tile_dataset.py`` / ``merged_dataset.py`` pipeline.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from spacenit.ingestion.geotiff_reader import (
    SampleLocator,
    discover_samples,
    discover_samples_from_csv,
    read_sample,
)
from spacenit.ingestion.sensors import SensorRegistry, SensorSpec
from spacenit.ingestion.standardizer import Standardizer, Strategy
from spacenit.structures import GeoSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class HFGeoDatasetConfig:
    """Configuration for :class:`HFGeoDataset`.

    Args:
        data_root: Root directory containing the GeoTIFF data.
        sensor_labels: List of sensor labels to load.
        max_timesteps: Maximum temporal depth per sample.
        csv_path: Optional CSV file listing samples (instead of scanning).
        normalization_strategy: Normalization approach (``"computed"`` or
            ``"predefined"``).
        std_multiplier: Standard deviation multiplier for z-score normalization.
    """

    data_root: str = ""
    sensor_labels: list[str] = field(default_factory=list)
    max_timesteps: int = 12
    csv_path: str | None = None
    normalization_strategy: str = "computed"
    std_multiplier: float = 2.0


class HFGeoDataset(Dataset):
    """PyTorch Dataset wrapping HuggingFace geospatial data.

    Reads GeoTIFF files via :mod:`geotiff_reader`, applies normalization,
    and optionally applies transforms.

    Args:
        config: Dataset configuration.
        transform: Optional callable transform applied to each sample.
    """

    def __init__(
        self,
        config: HFGeoDatasetConfig,
        transform: Callable[[GeoSample], GeoSample] | None = None,
    ) -> None:
        self.config = config
        self.transform = transform

        # Discover samples
        if config.csv_path:
            self.locators = discover_samples_from_csv(config.csv_path)
        else:
            self.locators = discover_samples(
                config.data_root, config.sensor_labels or None
            )

        if not self.locators:
            logger.warning(
                f"No samples found in {config.data_root}. "
                "Check the data_root path and sensor_labels."
            )

        # Initialize standardizer
        strategy = Strategy(config.normalization_strategy)
        self.standardizer = Standardizer(
            strategy=strategy,
            std_multiplier=config.std_multiplier,
        )

        logger.info(
            f"HFGeoDataset initialized with {len(self.locators)} samples, "
            f"sensors={config.sensor_labels}"
        )

    def __len__(self) -> int:
        return len(self.locators)

    def __getitem__(self, idx: int) -> GeoSample:
        locator = self.locators[idx]

        # Read raw data
        sample = read_sample(
            locator,
            self.config.sensor_labels,
            self.config.data_root,
            max_timesteps=self.config.max_timesteps,
        )

        # Normalize each sensor
        normalized_fields: dict[str, Any] = {}
        for key in sample.present_keys:
            data = sample[key]
            if data is None:
                continue

            if key in ("timestamps", "latlon"):
                normalized_fields[key] = torch.as_tensor(data, dtype=torch.float32)
                continue

            try:
                spec = SensorRegistry.get(key)
                data_np = data if isinstance(data, np.ndarray) else data.numpy()
                data_np = self.standardizer.standardize(spec, data_np)
                normalized_fields[key] = torch.as_tensor(
                    data_np, dtype=torch.float32
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping normalization for {key}: {e}")
                normalized_fields[key] = torch.as_tensor(data, dtype=torch.float32)

        sample = GeoSample(**normalized_fields)

        # Apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def geo_collate_fn(
    samples: list[GeoSample],
) -> GeoSample:
    """Collate a list of GeoSamples into a batched GeoSample.

    Stacks tensors along a new batch dimension (dim=0).  Sensors that
    are missing in some samples are filled with the ``ABSENT_INDICATOR``.

    Args:
        samples: List of individual GeoSamples.

    Returns:
        Batched GeoSample.
    """
    from spacenit.ingestion.sensors import ABSENT_INDICATOR

    if not samples:
        return GeoSample()

    # Collect all keys across samples
    all_keys: set[str] = set()
    for s in samples:
        all_keys.update(s.present_keys)

    batched: dict[str, Tensor] = {}
    for key in sorted(all_keys):
        tensors = []
        for s in samples:
            val = s[key]
            if val is not None:
                if not isinstance(val, Tensor):
                    val = torch.as_tensor(val, dtype=torch.float32)
                tensors.append(val)
            else:
                # Need a placeholder -- use the shape from another sample
                ref_shape = None
                for other in samples:
                    other_val = other[key]
                    if other_val is not None:
                        ref_shape = other_val.shape
                        break
                if ref_shape is not None:
                    placeholder = torch.full(ref_shape, ABSENT_INDICATOR, dtype=torch.float32)
                    tensors.append(placeholder)

        if tensors:
            try:
                batched[key] = torch.stack(tensors, dim=0)
            except RuntimeError:
                # Shape mismatch -- skip this key
                logger.warning(
                    f"Could not stack tensors for key '{key}' due to shape mismatch"
                )

    return GeoSample(**batched)


# ---------------------------------------------------------------------------
# DataLoader wrapper
# ---------------------------------------------------------------------------


@dataclass
class HFGeoDataLoaderConfig:
    """Configuration for :class:`HFGeoDataLoader`.

    Args:
        batch_size: Samples per batch.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle samples.
        max_tokens_per_instance: Token budget per sample (for cropping).
        patch_size_range: ``(min, max)`` range for random patch size sampling.
    """

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    max_tokens_per_instance: int | None = None
    patch_size_range: tuple[int, int] = (8, 32)


class HFGeoDataLoader:
    """Distributed-aware DataLoader wrapper for geospatial data.

    Handles patch size sampling, token budget enforcement, and proper
    collation of multi-sensor samples.

    Args:
        dataset: The underlying dataset.
        config: DataLoader configuration.
        rank: Distributed rank (for DistributedSampler).
        world_size: Distributed world size.
    """

    def __init__(
        self,
        dataset: HFGeoDataset,
        config: HFGeoDataLoaderConfig,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.config = config

        # Set up sampler
        if world_size > 1:
            sampler: Sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=config.shuffle,
                drop_last=config.drop_last,
            )
        else:
            sampler = None

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=config.shuffle if sampler is None else False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            collate_fn=geo_collate_fn,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)

    def sample_patch_size(self) -> int:
        """Sample a random patch size from the configured range."""
        lo, hi = self.config.patch_size_range
        # Sample a power of 2 within range
        valid_sizes = [s for s in [8, 16, 32, 64] if lo <= s <= hi]
        if not valid_sizes:
            valid_sizes = [lo]
        return random.choice(valid_sizes)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the distributed sampler."""
        if hasattr(self.dataloader, "sampler") and isinstance(
            self.dataloader.sampler, DistributedSampler
        ):
            self.dataloader.sampler.set_epoch(epoch)
