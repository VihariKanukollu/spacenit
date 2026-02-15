"""Benchmark adapter contract to be able to run benchmarks on a model."""

from logging import getLogger
from typing import Any

import torch
from einops import rearrange, reduce
from torch import nn

from spacenit.benchmarks.datasets.registry import TaskType
from spacenit.benchmarks.adapters import (
    AnySat,
    Clay,
    Croma,
    DINOv3,
    GalileoAdapter,
    Panopticon,
    PrestoAdapter,
    PrithviV2,
    Satlas,
    Tessera,
)
from spacenit.nn.flexi_vit import (
    FlexiVitBase,
    PoolingType,
    TokensAndMasks,
)
from spacenit.nn.pooled_modality_predictor import EncodeEarlyAttnPool
from spacenit.nn.st_model import STBase
from spacenit.train.masking import MaskedSpaceNitSample

logger = getLogger(__name__)


class BenchmarkAdapter:
    """Base class for benchmark adapters.

    This is the common interface to run our benchmarks on any model.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: TaskType,
        patch_size: int,
        pooling_type: PoolingType,
        concat_features: bool = False,
        use_pooled_tokens: bool = False,
    ):
        """Initialize the benchmark adapter.

        Args:
            model: The model to evaluate.
            task_type: The type of task to evaluate.
            patch_size: The patch size to use for the model.
            pooling_type: The pooling type to use for the model.
            concat_features: Whether to concatenate features across modalities.
            use_pooled_tokens: Whether to use pooled tokens.
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.patch_size = patch_size
        self.pooling_type = pooling_type
        self.concat_features = concat_features
        self.spatial_pool = task_type == TaskType.SEGMENTATION
        self.use_pooled_tokens = use_pooled_tokens
        if self.use_pooled_tokens:
            assert isinstance(self.model, EncodeEarlyAttnPool), (
                "Pooled tokens are only supported for EncodeEarlyAttnPool"
            )

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        dev = getattr(self.model, "device", None)

        if isinstance(dev, torch.device):
            return dev

        if isinstance(dev, str):
            return torch.device(dev)

        # For FSDP wrapped models, fall back to device of model parameters
        return next(self.model.parameters()).device

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model if the attribute is not found on the adapter."""
        return getattr(self.model, name)

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        raise NotImplementedError("Subclasses must implement this method")


class SpaceNitBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for SpaceNit models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        if not self.use_pooled_tokens:
            batch_features: TokensAndMasks = self.model(
                masked_spacenit_sample, patch_size=self.patch_size, fast_pass=True
            )["tokens_and_masks"]  # (bsz, dim)
            # Concat features across modalities in space averaged across time
            batch_features = batch_features.pool_unmasked_tokens(
                self.pooling_type,
                spatial_pooling=self.spatial_pool,
                concat_features=self.concat_features,
            )
        else:
            pooled_tokens_dict = self.model(
                masked_spacenit_sample, patch_size=self.patch_size, fast_pass=True
            )["pooled_tokens_and_masks"]
            pooled_tokens = pooled_tokens_dict["modality_pooled_tokens"]
            # spatial pool is true means we want to keep the spatial dimensions
            # so here we just need to pool across time
            logger.info(f"pooled tokens shape in benchmark adapter: {pooled_tokens.shape}")

            if self.spatial_pool:
                # B H W T C
                if pooled_tokens.shape[1] == 1 and pooled_tokens.ndim == 3:
                    # unsqueeze to get a W H C T
                    pooled_tokens = pooled_tokens.unsqueeze(1)
                pooled_tokens = reduce(
                    pooled_tokens, "b h w ... d -> b h w d", self.pooling_type
                )
            else:
                # Take the mean of all dims except the first and last
                pooled_tokens = reduce(
                    pooled_tokens, "b ... d -> b d", self.pooling_type
                )
            batch_features = pooled_tokens
        return batch_features, labels


class PanopticonBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Panopticon models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        if self.spatial_pool:
            batch_features = self.model.forward_features(
                masked_spacenit_sample, pooling=self.pooling_type
            )
        else:
            batch_features = self.model(
                masked_spacenit_sample, pooling=self.pooling_type
            )
        return batch_features, labels


class GalileoBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Galileo models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return features, labels


class AnySatBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for AnySat model."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        if is_train and (self.task_type == TaskType.SEGMENTATION):
            # Special case for AnySat: subsample training pixels for segmentation
            # to keep memory requirements reasonable.
            subsample_by = 1 / 16
            features = rearrange(features, "b h w d -> b (h w) d")
            labels = rearrange(labels, "b h w -> b (h w)")

            assert features.shape[1] == labels.shape[1]
            num_tokens = features.shape[1]
            num_tokens_to_keep = int(num_tokens * subsample_by)
            sampled_indices = torch.randperm(num_tokens)[:num_tokens_to_keep]
            features = features[:, sampled_indices]
            labels = labels[:, sampled_indices]

            new_hw = int(num_tokens_to_keep**0.5)
            features = rearrange(
                features, "b (h w) d -> b h w d", h=new_hw, w=new_hw
            )
            labels = rearrange(labels, "b (h w) -> b h w", h=new_hw, w=new_hw)
        return features, labels


class PrithviV2BenchmarkAdapter(BenchmarkAdapter):
    """Adapter for PrithviV2 model."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return features, labels


class ClayBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Clay models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        batch_features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class CromaBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Croma models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        batch_features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class PrestoBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Presto model."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        batch_features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class DINOv3BenchmarkAdapter(BenchmarkAdapter):
    """Adapter for DINOv3 models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        if self.spatial_pool:
            batch_features = self.model.forward_features(
                masked_spacenit_sample,
                pooling=self.pooling_type,
            )
        else:
            batch_features = self.model(
                masked_spacenit_sample,
                pooling=self.pooling_type,
            )
        return batch_features, labels


class SatlasBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Satlas models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        batch_features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class TesseraBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Tessera models."""

    def __call__(
        self,
        masked_spacenit_sample: MaskedSpaceNitSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        batch_features = self.model(
            masked_spacenit_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


def get_benchmark_adapter(model: nn.Module, **kwargs: Any) -> BenchmarkAdapter:
    """Factory function to get the appropriate benchmark adapter for a given model.

    Args:
        model: The model to evaluate.
        **kwargs: Additional keyword arguments.

    Returns:
        The appropriate benchmark adapter for the given model.
    """
    if isinstance(model, FlexiVitBase) or isinstance(model, STBase):
        logger.info("Using SpaceNitBenchmarkAdapter")
        return SpaceNitBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, Panopticon):
        logger.info("Using PanopticonBenchmarkAdapter")
        return PanopticonBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, DINOv3):
        logger.info("Using DINOv3BenchmarkAdapter")
        return DINOv3BenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, Croma):
        logger.info("Using CromaBenchmarkAdapter")
        return CromaBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, Clay):
        logger.info("Using ClayBenchmarkAdapter")
        return ClayBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, GalileoAdapter):
        logger.info("Using GalileoBenchmarkAdapter")
        return GalileoBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, PrestoAdapter):
        logger.info("Using PrestoBenchmarkAdapter")
        return PrestoBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, AnySat):
        logger.info("Using AnySatBenchmarkAdapter")
        return AnySatBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, Satlas):
        logger.info("Using SatlasBenchmarkAdapter")
        return SatlasBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, Tessera):
        logger.info("Using TesseraBenchmarkAdapter")
        return TesseraBenchmarkAdapter(model=model, **kwargs)
    elif isinstance(model, PrithviV2):
        logger.info("Using PrithviV2BenchmarkAdapter")
        return PrithviV2BenchmarkAdapter(model=model, **kwargs)
    else:
        raise NotImplementedError(f"No BenchmarkAdapter for model type {type(model)}")
