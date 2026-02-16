"""Dual-branch training runner.

Rewritten to use the new :class:`DualBranch` model which combines
latent prediction with pixel reconstruction.
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

from spacenit.arch.models import DualBranch
from spacenit.ingestion.augmentations import TransformConfig
from spacenit.pipeline.helpers import partition_masked_batch
from spacenit.pipeline.losses import (
    CompositeLoss,
    ContrastiveLoss,
    LatentPredictionLoss,
    ReconstructionLoss,
    UniformityLoss,
)
from spacenit.pipeline.masking import build_masking
from spacenit.pipeline.runners.base_runner import (
    SpaceNitTrainRunner,
    SpaceNitTrainRunnerConfig,
)
from spacenit.structures import MaskedGeoSample

logger = getLogger(__name__)


@dataclass
class DualBranchRunnerConfig(SpaceNitTrainRunnerConfig):
    """Configuration for :class:`DualBranchRunner`."""

    masking_config: dict[str, Any] = field(
        default_factory=lambda: {"type": "random", "encode_ratio": 0.25}
    )
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "latent": 1.0,
            "contrastive": 0.1,
            "reconstruction": 0.5,
        }
    )
    smooth_l1_beta: float = 1.0
    contrastive_temperature: float = 0.07
    reconstruction_loss_type: str = "mse"
    max_grad_norm: float = 1.0

    def build(
        self,
        model: DualBranch,
        device: torch.device | None = None,
    ) -> DualBranchRunner:
        kwargs = self.prepare_kwargs()
        return DualBranchRunner(model=model, device=device, **kwargs)


class DualBranchRunner(SpaceNitTrainRunner):
    """Training runner for dual-branch (latent + reconstruction) training."""

    def __init__(
        self,
        model: DualBranch,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        masking_config: dict[str, Any] | None = None,
        loss_weights: dict[str, float] | None = None,
        smooth_l1_beta: float = 1.0,
        contrastive_temperature: float = 0.07,
        reconstruction_loss_type: str = "mse",
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

        masking_config = masking_config or {"type": "random", "encode_ratio": 0.25}
        self.masking_strategy = build_masking(dict(masking_config))

        loss_weights = loss_weights or {
            "latent": 1.0,
            "contrastive": 0.1,
            "reconstruction": 0.5,
        }

        losses: dict[str, tuple[torch.nn.Module, float]] = {
            "latent": (
                LatentPredictionLoss(beta=smooth_l1_beta),
                loss_weights.get("latent", 1.0),
            ),
            "reconstruction": (
                ReconstructionLoss(loss_type=reconstruction_loss_type),
                loss_weights.get("reconstruction", 0.5),
            ),
        }

        if loss_weights.get("contrastive", 0.0) > 0:
            losses["contrastive"] = (
                ContrastiveLoss(initial_temperature=contrastive_temperature),
                loss_weights["contrastive"],
            )

        self.loss_fn = CompositeLoss(losses)

    def train_batch(
        self,
        batch: tuple[int, MaskedGeoSample],
        dry_run: bool = False,
    ) -> None:
        self.model.train()

        # EMA update BEFORE forward pass (matching OLMo-Earth).
        if not dry_run and hasattr(self.model, "step_ema"):
            self.model.step_ema()

        total_loss = torch.zeros([], device=self.device)
        patch_size, batch_data = batch

        microbatches = partition_masked_batch(batch_data, self.rank_microbatch_size)
        num_mb = len(microbatches)

        for mb_idx, mb in enumerate(microbatches):
            with self._train_microbatch_context(mb_idx, num_mb):
                mb = mb.transfer_to(self.device)

                sensor_data = {
                    k: mb[k] for k in mb.present_keys if mb[k] is not None
                }

                with self._model_forward_context():
                    outputs = self.model(sensor_data, patch_size=patch_size)

                loss_inputs = {
                    "latent_predictions": outputs["latent_predictions"],
                    "latent_targets": outputs["targets"],
                    "reconstruction_predictions": outputs["reconstructed"],
                    "reconstruction_targets": next(iter(sensor_data.values())),
                }

                if "online_proj" in outputs:
                    loss_inputs["contrastive_anchors"] = outputs["online_proj"]
                    loss_inputs["contrastive_positives"] = outputs["target_proj"]

                loss, individual = self.loss_fn(**loss_inputs)
                loss = loss / num_mb

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"NaN/Inf at microbatch {mb_idx}")
                    continue

                loss.backward()
                total_loss += get_local_tensor(loss.detach())

        if not dry_run:
            self.trainer.record_metric(
                "train/dual_branch_loss", total_loss, ReduceType.mean
            )
