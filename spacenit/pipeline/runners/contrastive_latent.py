"""Contrastive latent training runner.

Rewritten to use the new :class:`LatentPredictor` model with
:class:`ContrastiveLoss` as the primary objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_rank, get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from spacenit.arch.models import LatentPredictor
from spacenit.ingestion.augmentations import TransformConfig
from spacenit.pipeline.helpers import partition_masked_batch
from spacenit.pipeline.losses import ContrastiveLoss, UniformityLoss
from spacenit.pipeline.masking import build_masking
from spacenit.pipeline.runners.base_runner import (
    SpaceNitTrainRunner,
    SpaceNitTrainRunnerConfig,
)
from spacenit.structures import MaskedGeoSample

logger = getLogger(__name__)


@dataclass
class ContrastiveLatentRunnerConfig(SpaceNitTrainRunnerConfig):
    """Configuration for :class:`ContrastiveLatentRunner`."""

    masking_config: dict[str, Any] = field(
        default_factory=lambda: {"type": "random", "encode_ratio": 0.25}
    )
    contrastive_temperature: float = 0.07
    uniformity_weight: float = 0.01
    max_grad_norm: float = 1.0

    def build(
        self,
        model: LatentPredictor,
        device: torch.device | None = None,
    ) -> ContrastiveLatentRunner:
        kwargs = self.prepare_kwargs()
        return ContrastiveLatentRunner(model=model, device=device, **kwargs)


class ContrastiveLatentRunner(SpaceNitTrainRunner):
    """Training runner for contrastive learning in latent space."""

    def __init__(
        self,
        model: LatentPredictor,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        masking_config: dict[str, Any] | None = None,
        contrastive_temperature: float = 0.07,
        uniformity_weight: float = 0.01,
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
        # This runner currently uses encoder-pooled representations for InfoNCE
        # and does not backprop through the decoder. Freeze decoder params so
        # they are excluded from the optimizer (and don't break checkpoint reload
        # due to missing optimizer state entries).
        for p in model.decoder.parameters():
            p.requires_grad = False

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
        self.contrastive_loss = ContrastiveLoss(
            initial_temperature=contrastive_temperature
        )
        self.uniformity_loss = UniformityLoss() if uniformity_weight > 0 else None
        self.uniformity_weight = uniformity_weight

    def train_batch(
        self,
        batch: tuple[int, MaskedGeoSample],
        dry_run: bool = False,
    ) -> None:
        self.model.train()
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

                # Provide month indices so month embeddings get gradients (and
                # optimizer state is present for checkpoint reload).
                month_indices = None
                try:
                    ts = mb.timestamps
                    if isinstance(ts, torch.Tensor) and ts.ndim >= 3 and ts.shape[-1] >= 2:
                        month_indices = ts[..., 1].to(dtype=torch.long)  # (B, T)
                        month_indices = month_indices.clamp(min=0, max=11)
                except Exception:
                    month_indices = None

                with self._model_forward_context():
                    # Use model.forward() with contrastive_only=True so that
                    # any future changes to the encoder/target-encoder API are
                    # handled in one place.
                    outputs = self.model(
                        sensor_data,
                        patch_size=patch_size,
                        month_indices=month_indices,
                        contrastive_only=True,
                    )

                online_z = F.normalize(outputs["online_proj"], dim=-1)
                with torch.no_grad():
                    target_z = F.normalize(outputs["target_proj"], dim=-1)

                loss = self.contrastive_loss(online_z, target_z)

                # Uniformity regularizer
                if self.uniformity_loss is not None:
                    uni_loss = self.uniformity_loss(online_z)
                    loss = loss + self.uniformity_weight * uni_loss

                loss = loss / num_mb

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"NaN/Inf at microbatch {mb_idx}")
                    continue

                loss.backward()
                total_loss += get_local_tensor(loss.detach())

        # EMA update
        if not dry_run and hasattr(self.model, "step_ema"):
            self.model.step_ema()

        if not dry_run:
            self.trainer.record_metric(
                "train/contrastive_loss", total_loss, ReduceType.mean
            )
