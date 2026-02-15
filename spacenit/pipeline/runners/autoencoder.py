"""Auto-encoder (MAE-style) training runner.

Rewritten to use the new :class:`AutoEncoder` model and
:class:`ReconstructionLoss`.
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

from spacenit.arch.models import AutoEncoder
from spacenit.ingestion.augmentations import TransformConfig
from spacenit.pipeline.losses import ReconstructionLoss
from spacenit.pipeline.masking import build_masking
from spacenit.pipeline.runners.base_runner import (
    SpaceNitTrainRunner,
    SpaceNitTrainRunnerConfig,
)
from spacenit.structures import MaskedGeoSample

logger = getLogger(__name__)


@dataclass
class AutoEncoderRunnerConfig(SpaceNitTrainRunnerConfig):
    """Configuration for :class:`AutoEncoderRunner`."""

    masking_config: dict[str, Any] = field(
        default_factory=lambda: {"type": "random", "encode_ratio": 0.25}
    )
    loss_type: str = "mse"
    normalize_target: bool = True
    max_grad_norm: float = 1.0

    def build(
        self,
        model: AutoEncoder,
        device: torch.device | None = None,
    ) -> AutoEncoderRunner:
        kwargs = self.prepare_kwargs()
        return AutoEncoderRunner(model=model, device=device, **kwargs)


class AutoEncoderRunner(SpaceNitTrainRunner):
    """Training runner for masked auto-encoding (MAE-style)."""

    def __init__(
        self,
        model: AutoEncoder,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        masking_config: dict[str, Any] | None = None,
        loss_type: str = "mse",
        normalize_target: bool = True,
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
        self.reconstruction_loss = ReconstructionLoss(
            loss_type=loss_type, normalize_target=normalize_target
        )

    def train_batch(
        self,
        batch: tuple[int, MaskedGeoSample],
        dry_run: bool = False,
    ) -> None:
        self.model.train()
        total_loss = torch.zeros([], device=self.device)
        patch_size, batch_data = batch

        microbatches = _partition(batch_data, self.rank_microbatch_size)
        num_mb = len(microbatches)

        for mb_idx, mb in enumerate(microbatches):
            with self._train_microbatch_context(mb_idx, num_mb):
                mb = mb.transfer_to(self.device)

                sensor_data = {
                    k: mb[k] for k in mb.present_keys if mb[k] is not None
                }

                with self._model_forward_context():
                    outputs = self.model(sensor_data, patch_size=patch_size)

                loss = self.reconstruction_loss(
                    outputs["reconstructed"],
                    # Target is the original sensor data (first available)
                    next(iter(sensor_data.values())),
                )
                loss = loss / num_mb

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"NaN/Inf at microbatch {mb_idx}")
                    continue

                loss.backward()
                total_loss += get_local_tensor(loss.detach())

        if not dry_run:
            self.trainer.record_metric(
                "train/reconstruction_loss", total_loss, ReduceType.mean
            )


def _partition(batch: MaskedGeoSample, size: int) -> list[MaskedGeoSample]:
    B = batch.num_samples
    if B <= size:
        return [batch]
    parts = []
    for s in range(0, B, size):
        e = min(s + size, B)
        fields = {}
        for key in batch._fields:
            val = batch[key]
            if val is not None and hasattr(val, "__getitem__"):
                fields[key] = val[s:e]
        parts.append(MaskedGeoSample(**fields))
    return parts
