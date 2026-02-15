"""Latent prediction training runner.

Rewritten to use the new model (:class:`LatentPredictor`), loss
(:class:`LatentPredictionLoss`, :class:`ContrastiveLoss`), and masking
(:mod:`masking`) APIs.  EMA updates are delegated to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_rank, get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from spacenit.arch.models import LatentPredictor, LatentPredictorConfig
from spacenit.ingestion.augmentations import TransformConfig
from spacenit.pipeline.losses import (
    CompositeLoss,
    ContrastiveLoss,
    LatentPredictionLoss,
    UniformityLoss,
)
from spacenit.pipeline.masking import (
    MaskingStrategy,
    RandomMasking,
    apply_strategy,
    build_masking,
)
from spacenit.pipeline.runners.base_runner import (
    SpaceNitTrainRunner,
    SpaceNitTrainRunnerConfig,
)
from spacenit.structures import MaskedGeoSample, TokenVisibility

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LatentPredictionRunnerConfig(SpaceNitTrainRunnerConfig):
    """Configuration for :class:`LatentPredictionRunner`.

    Args:
        masking_config: Masking strategy configuration dict.
        loss_weights: Weights for each loss component.
        smooth_l1_beta: Beta for smooth L1 loss.
        contrastive_temperature: Initial temperature for contrastive loss.
        uniformity_weight: Weight for uniformity regularizer (0 to disable).
        max_grad_norm: Gradient clipping threshold.
    """

    masking_config: dict[str, Any] = field(
        default_factory=lambda: {"type": "random", "encode_ratio": 0.25}
    )
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"latent": 1.0, "contrastive": 0.1}
    )
    smooth_l1_beta: float = 1.0
    contrastive_temperature: float = 0.07
    uniformity_weight: float = 0.0
    max_grad_norm: float = 1.0

    def build(
        self,
        model: LatentPredictor,
        device: torch.device | None = None,
    ) -> LatentPredictionRunner:
        kwargs = self.prepare_kwargs()
        return LatentPredictionRunner(model=model, device=device, **kwargs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LatentPredictionRunner(SpaceNitTrainRunner):
    """Training runner for latent prediction (JEPA-style).

    Uses the new :class:`LatentPredictor` model which handles EMA
    internally, the new composable masking strategies, and the new
    loss functions.
    """

    def __init__(
        self,
        model: LatentPredictor,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        masking_config: dict[str, Any] | None = None,
        loss_weights: dict[str, float] | None = None,
        smooth_l1_beta: float = 1.0,
        contrastive_temperature: float = 0.07,
        uniformity_weight: float = 0.0,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        find_unused_parameters: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            optim_config=optim_config,
            transform_config=transform_config,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            dp_config=dp_config,
            compile_loss=compile_loss,
            autocast_precision=autocast_precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
            find_unused_parameters=find_unused_parameters,
        )

        # Build masking strategy
        masking_config = masking_config or {"type": "random", "encode_ratio": 0.25}
        self.masking_strategy = build_masking(dict(masking_config))

        # Build loss functions
        loss_weights = loss_weights or {"latent": 1.0, "contrastive": 0.1}
        losses: dict[str, tuple[torch.nn.Module, float]] = {}

        losses["latent"] = (
            LatentPredictionLoss(beta=smooth_l1_beta),
            loss_weights.get("latent", 1.0),
        )

        if loss_weights.get("contrastive", 0.0) > 0:
            losses["contrastive"] = (
                ContrastiveLoss(initial_temperature=contrastive_temperature),
                loss_weights["contrastive"],
            )

        if uniformity_weight > 0:
            losses["uniformity"] = (
                UniformityLoss(),
                uniformity_weight,
            )

        self.loss_fn = CompositeLoss(losses)

    def train_batch(
        self,
        batch: tuple[int, MaskedGeoSample],
        dry_run: bool = False,
    ) -> None:
        """Train on a single batch.

        Args:
            batch: Tuple of ``(patch_size, masked_sample)`` from the
                data loader.
            dry_run: If ``True``, skip metric recording.
        """
        self.model.train()
        total_loss = torch.zeros([], device=self.device)
        patch_size = batch[0]
        batch_data = batch[1]

        # Split into microbatches
        microbatches = _partition_masked_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(microbatches)

        for mb_idx, mb in enumerate(microbatches):
            with self._train_microbatch_context(mb_idx, num_microbatches):
                mb = mb.transfer_to(self.device)

                # Extract sensor data and masks
                sensor_data = {}

                for key in mb.present_keys:
                    data = mb[key]
                    if data is not None:
                        sensor_data[key] = data

                # Forward pass through model
                with self._model_forward_context():
                    outputs = self.model(
                        sensor_data,
                        patch_size=patch_size,
                    )

                # Compute losses
                loss_inputs = {
                    "latent_predictions": outputs["predictions"],
                    "latent_targets": outputs["targets"],
                }

                if "online_proj" in outputs:
                    loss_inputs["contrastive_anchors"] = outputs["online_proj"]
                    loss_inputs["contrastive_positives"] = outputs["target_proj"]

                if "online_proj" in outputs:
                    loss_inputs["uniformity_embeddings"] = outputs["online_proj"]

                loss, individual_losses = self.loss_fn(**loss_inputs)
                loss = loss / num_microbatches

                # Check for NaN/Inf
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN/Inf in loss at microbatch {mb_idx}, "
                        f"rank {get_local_rank()}"
                    )
                    continue

                loss.backward()
                total_loss += get_local_tensor(loss.detach())

        # EMA update (delegated to model)
        if not dry_run and hasattr(self.model, "step_ema"):
            self.model.step_ema()

        # Log metrics
        if not dry_run:
            self.trainer.record_metric(
                "train/total_loss", total_loss, ReduceType.mean
            )


def _partition_masked_batch(
    batch: MaskedGeoSample, microbatch_size: int
) -> list[MaskedGeoSample]:
    """Split a batch into microbatches."""
    B = batch.num_samples
    if B <= microbatch_size:
        return [batch]

    microbatches = []
    for start in range(0, B, microbatch_size):
        end = min(start + microbatch_size, B)
        fields: dict[str, Any] = {}
        for key in batch._fields:
            val = batch[key]
            if val is not None and hasattr(val, "__getitem__"):
                fields[key] = val[start:end]
        microbatches.append(MaskedGeoSample(**fields))

    return microbatches
