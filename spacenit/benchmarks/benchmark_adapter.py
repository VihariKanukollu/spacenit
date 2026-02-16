"""Benchmark adapter contract to be able to run benchmarks on a model."""

from logging import getLogger
from typing import Any

import torch
from einops import rearrange, reduce
from torch import nn

from spacenit.arch.common import PoolingType
from spacenit.arch.encoder import Encoder
from spacenit.arch.models import LatentPredictor, SpatioTemporalEncoder
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
from spacenit.benchmarks.datasets.registry import TaskType
from spacenit.structures import MaskedGeoSample

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
        pooling_type: str,
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
        """Delegate attribute access to the underlying model."""
        return getattr(self.model, name)

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the feature specified by initialization."""
        raise NotImplementedError("Subclasses must implement this method")


def _extract_sensor_data(sample: MaskedGeoSample) -> dict[str, torch.Tensor]:
    """Convert a MaskedGeoSample into the sensor_data dict the encoder expects."""
    return {k: sample[k] for k in sample.present_keys if sample[k] is not None}


def _pool_tokens(
    tokens: torch.Tensor,
    pooling_type: str,
    spatial_pool: bool = False,
) -> torch.Tensor:
    """Pool token embeddings.

    For classification (spatial_pool=False): returns (B, D).
    For segmentation (spatial_pool=True): returns (B, N, D) to preserve
    spatial structure.
    """
    if spatial_pool:
        return tokens

    if pooling_type == PoolingType.MEAN:
        return tokens.mean(dim=1)
    elif pooling_type == PoolingType.MAX:
        return tokens.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


def _pool_patch_tokens_only(
    tokens: torch.Tensor,
    spatial_ids: torch.Tensor | None,
    pooling_type: str,
    *,
    spatial_pool: bool = False,
) -> torch.Tensor:
    """Pool tokens while excluding per-sensor type tokens.

    SpaceNit's tokenizer prepends a sensor-type token per sensor with
    ``spatial_ids == (-1, -1)``. OLMo-Earth does not have these tokens, so
    for downstream evaluation we pool over patch tokens only.
    """
    if spatial_pool or spatial_ids is None:
        return _pool_tokens(tokens, pooling_type, spatial_pool=spatial_pool)

    is_patch = spatial_ids[..., 0] != -1  # (B, N)
    if pooling_type == PoolingType.MEAN:
        denom = is_patch.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        return (tokens * is_patch.unsqueeze(-1)).sum(dim=1) / denom
    elif pooling_type == PoolingType.MAX:
        masked = tokens.masked_fill(~is_patch.unsqueeze(-1), float("-inf"))
        return masked.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


class SpaceNitBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for SpaceNit models (Encoder, LatentPredictor, etc.)."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model, then return pooled features.

        Default behaviour (matching OLMo-Earth): run the **encoder** on the
        input and mean-pool the raw token embeddings.  This works whether
        ``self.model`` is a bare :class:`Encoder` or a composite model that
        has an ``.encoder`` attribute.

        When the caller explicitly passes a full
        :class:`~spacenit.arch.models.LatentPredictor` (e.g. via
        ``SPACENIT_EVAL_REPRESENTATION=proj``), we use the contrastive
        projection features instead.  Projection-head features are optimised
        for the contrastive loss and often transfer worse to downstream
        KNN / linear-probe tasks.
        """
        sensor_data = _extract_sensor_data(sample)

        # ------------------------------------------------------------
        # LatentPredictor passed explicitly: projection-head features
        # (only reached when SPACENIT_EVAL_REPRESENTATION=proj)
        # ------------------------------------------------------------
        if isinstance(self.model, LatentPredictor) and not self.spatial_pool:
            month_indices = None
            try:
                ts = getattr(sample, "timestamps", None)
                if isinstance(ts, torch.Tensor) and ts.ndim >= 3 and ts.shape[-1] >= 2:
                    month_indices = ts[..., 1].to(dtype=torch.long)
                    month_indices = month_indices.clamp(min=0, max=11)
            except Exception:
                month_indices = None

            outputs = self.model(
                sensor_data,
                patch_size=self.patch_size,
                month_indices=month_indices,
                contrastive_only=True,
            )
            features = outputs.get("target_proj", outputs["online_proj"])
            return features, labels

        # ------------------------------------------------------------
        # Default: raw encoder tokens, mean-pooled (matches OLMo-Earth)
        # ------------------------------------------------------------
        encoder = self.model
        if hasattr(self.model, "encoder"):
            encoder = self.model.encoder

        encoded, _sensor_ids, spatial_ids, _temporal_ids, _layout = encoder(
            sensor_data, patch_size=self.patch_size
        )

        batch_features = _pool_patch_tokens_only(
            encoded,
            spatial_ids,
            self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class PanopticonBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Panopticon models."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spatial_pool:
            batch_features = self.model.forward_features(
                sample, pooling=self.pooling_type
            )
        else:
            batch_features = self.model(
                sample, pooling=self.pooling_type
            )
        return batch_features, labels


class GalileoBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Galileo models."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(
            sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return features, labels


class AnySatBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for AnySat model."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(
            sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        if is_train and (self.task_type == TaskType.SEGMENTATION):
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
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(
            sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return features, labels


class ClayBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Clay models."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_features = self.model(
            sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class CromaBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Croma models."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_features = self.model(
            sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class PrestoBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Presto model."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_features = self.model(
            sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class DINOv3BenchmarkAdapter(BenchmarkAdapter):
    """Adapter for DINOv3 models."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spatial_pool:
            batch_features = self.model.forward_features(
                sample,
                pooling=self.pooling_type,
            )
        else:
            batch_features = self.model(
                sample,
                pooling=self.pooling_type,
            )
        return batch_features, labels


class SatlasBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Satlas models."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_features = self.model(
            sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_features, labels


class TesseraBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Tessera models."""

    def __call__(
        self,
        sample: MaskedGeoSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_features = self.model(
            sample,
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
    # SpaceNit's own models
    if isinstance(model, (Encoder, LatentPredictor, SpatioTemporalEncoder)):
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
