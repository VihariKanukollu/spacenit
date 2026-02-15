"""Base training runner for the SpaceNit pipeline.

Integrates with olmo-core's ``TrainModule`` for distributed training
(FSDP/DDP).  Provides the common training loop infrastructure that
concrete runners extend.

Uses the model/loss/masking APIs with olmo-core integration for
distributed training.  Key design points:
- Simplified microbatch loop
- EMA logic delegated to the model (not duplicated here)
- Uses ``TransformConfig`` from the augmentations module
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Generator
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch import nn, Tensor
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.config import DType
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_world_mesh,
    get_dp_mesh,
    get_dp_process_group,
)
from olmo_core.distributed.utils import get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec, TrainModule
from olmo_core.utils import gc_cuda, get_default_device

from spacenit.ingestion.augmentations import TransformConfig
from spacenit.settings import Config

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpaceNitTrainRunnerConfig(Config):
    """Configuration for building :class:`SpaceNitTrainRunner` instances.

    Args:
        rank_microbatch_size: Micro batch size per rank in instances.
        optim_config: Optimizer configuration.
        transform_config: Data augmentation configuration.
        compile_model: Whether to compile the model with ``torch.compile``.
        dp_config: Data parallel configuration for distributed training.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
        find_unused_parameters: Whether to find unused parameters for DDP.
    """

    optim_config: OptimConfig
    rank_microbatch_size: int

    transform_config: TransformConfig = field(
        default_factory=lambda: TransformConfig(transform_type="dihedral")
    )
    compile_model: bool = False
    dp_config: DataParallelConfig | None = None
    compile_loss: bool = False
    autocast_precision: DType | None = None
    max_grad_norm: float | None = None
    scheduler: Scheduler | None = None
    find_unused_parameters: bool = True
    state_dict_save_opts: dict[str, Any] | None = None
    state_dict_load_opts: dict[str, Any] | None = None

    def prepare_kwargs(self) -> dict[str, Any]:
        """Prepare keyword arguments for the runner constructor."""
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**save_opts)
        if (load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**load_opts)
        return kwargs

    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> SpaceNitTrainRunner:
        """Build the corresponding runner."""
        kwargs = self.prepare_kwargs()
        return SpaceNitTrainRunner(model=model, device=device, **kwargs)


# ---------------------------------------------------------------------------
# Base Runner
# ---------------------------------------------------------------------------


class SpaceNitTrainRunner(TrainModule):
    """Base training runner integrating with olmo-core.

    Handles distributed setup (FSDP/DDP), optimizer construction,
    gradient clipping, learning rate scheduling, and the microbatch
    training loop.  Concrete runners override :meth:`train_batch`.
    """

    def __init__(
        self,
        model: Any,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        find_unused_parameters: bool = True,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.transform = transform_config.build()
        self.device = device or get_default_device()

        # Log parameter counts
        if hasattr(self.model, "encoder"):
            n_enc = sum(p.numel() for p in self.model.encoder.parameters())
            logger.info("Encoder parameters: %d", n_enc)
        if hasattr(self.model, "decoder") and self.model.decoder is not None:
            n_dec = sum(p.numel() for p in self.model.decoder.parameters())
            logger.info("Decoder parameters: %d", n_dec)

        # Distributed setup
        self._dp_config = dp_config
        if dp_config is not None:
            self.world_mesh = build_world_mesh(
                dp=dp_config, device_type=self.device.type
            )
            logger.info(
                "Data parallel world size = %d",
                get_world_size(self.dp_process_group),
            )
        else:
            self.world_mesh = None

        # Compile
        if compile_model and torch.cuda.is_available():
            self.model.apply_compile()
            logger.info("Applied torch.compile() to the model")

        # Shard/replicate
        if dp_config is not None:
            dp_mesh = get_dp_mesh(self.world_mesh)
            if dp_config.name in (DataParallelType.fsdp,):
                param_dtype = (
                    dp_config.param_dtype.as_pt()
                    if dp_config.param_dtype is not None
                    else None
                )
                self.model.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                )
                logger.info("Applied FSDP to the model")
            elif dp_config.name == DataParallelType.ddp:
                self.model.apply_ddp(
                    dp_mesh=dp_mesh,
                    compile_enabled=compile_model,
                    find_unused_parameters=find_unused_parameters,
                )
                logger.info("Applied DDP to the model")
            else:
                raise NotImplementedError(dp_config.name)

        # Optimizer
        logger.info("Building optimizer...")
        self.optimizer: Optimizer = optim_config.build(self.model)
        self.rank_microbatch_size = rank_microbatch_size
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            # Optimizer state may be missing for params that didn't receive grads yet.
            flatten_optimizer_state_dict=True,
            strict=False,
        )

    # -- Properties -----------------------------------------------------------

    @property
    def dp_process_group(self) -> dist.ProcessGroup | None:
        if self.world_mesh is None:
            return None
        return get_dp_process_group(self.world_mesh)

    @property
    def is_fsdp(self) -> bool:
        return self._dp_config is not None and self._dp_config.name in (
            DataParallelType.fsdp,
        )

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        rank_batch_size = self.trainer.global_batch_size // get_world_size(
            self.trainer.dp_process_group
        )
        return EvalBatchSpec(
            rank_batch_size=rank_batch_size,
            batch_size_unit=EvalBatchSizeUnit.instances,
        )

    @property
    def local_rank(self) -> int:
        return self.trainer.data_loader.dp_rank

    @property
    def logits_dtype(self) -> torch.dtype:
        if self.autocast_precision is not None:
            return self.autocast_precision
        elif self._dp_config is not None and self._dp_config.param_dtype is not None:
            return self._dp_config.param_dtype.as_pt()
        else:
            for param in self.model.parameters():
                return param.dtype
        raise RuntimeError("Could not determine logits dtype")

    # -- Lifecycle hooks ------------------------------------------------------

    def on_attach(self) -> None:
        if (
            self.trainer.global_batch_size
            % (
                self.rank_microbatch_size
                * (ws := get_world_size(self.trainer.dp_process_group))
            )
            != 0
        ):
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be "
                f"divisible by micro-batch size ({self.rank_microbatch_size:,d}) "
                f"x DP world size ({ws})"
            )

    # -- State dict -----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts)

    def state_dict_to_load(
        self, metadata: Metadata, optim: bool | None = None
    ) -> dict[str, Any]:
        return self._get_state_dict(self.state_dict_load_opts)

    def state_dict_to_save(self) -> dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        dist_cp_sd.set_model_state_dict(
            self.model, state_dict["model"], options=self.state_dict_load_opts
        )
        gc_cuda()
        dist_cp_sd.set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict["optim"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions
    ) -> dict[str, Any]:
        model_sd = dist_cp_sd.get_model_state_dict(self.model, options=sd_options)
        optim_sd = dist_cp_sd.get_optimizer_state_dict(
            self.model, self.optimizer, options=sd_options
        )
        return {"model": model_sd, "optim": optim_sd}

    # -- Gradient management --------------------------------------------------

    def zero_grads(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def optim_step(self) -> None:
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            if isinstance(self.optimizer, SkipStepOptimizer):
                self.optimizer.latest_grad_norm = grad_norm

        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optimizer.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(
                    f"LR (group {group_idx})", new_lr, namespace="optim"
                )

        self.optimizer.step()
        if isinstance(self.optimizer, SkipStepOptimizer):
            self.trainer.record_metric(
                "step skipped", self.optimizer.step_skipped, namespace="optim"
            )

    def _clip_grad_norm(
        self,
        max_grad_norm: float,
        norm_type: float = 2.0,
        foreach: bool | None = None,
    ) -> Tensor:
        parameters = list(self.model.parameters())
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        torch.nn.utils.clip_grads_with_norm_(
            parameters, max_grad_norm, total_norm, foreach=foreach
        )
        return total_norm

    # -- Context managers -----------------------------------------------------

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            is_last = micro_batch_idx == num_micro_batches - 1
            if isinstance(self.model, FSDPModule):
                self.model.set_is_last_backward(is_last)
            if isinstance(self.model, DDP) and not is_last:
                stack.enter_context(self.model.no_sync())
            yield

    @contextlib.contextmanager
    def _model_forward_context(
        self, no_sync: bool = False
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(
                    torch.autocast(self.device.type, dtype=self.autocast_precision)
                )
            if isinstance(self.model, DDP) and no_sync:
                stack.enter_context(self.model.no_sync())
            yield

    # -- Metrics helpers ------------------------------------------------------

    def log_metrics(
        self,
        metrics: dict[str, Any],
        reduce_type: ReduceType | None = None,
    ) -> None:
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.trainer.record_metric(
                        f"{key}/{sub_key}", sub_value, reduce_type=reduce_type
                    )
            else:
                self.trainer.record_metric(key, value, reduce_type=reduce_type)

    # -- Abstract methods (to be overridden) ----------------------------------

    def eval_batch(
        self, batch: dict[str, Any], labels: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor | None]:
        raise NotImplementedError("eval_batch not implemented")
