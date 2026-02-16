"""Contrastive LatentMIM training runner (OLMoEarth-inspired).

This runner trains a :class:`~spacenit.arch.models.LatentPredictor` using:

- token masking (context vs decoder targets) coming from the DataLoader
- a token-level "patch discrimination" loss on decoder-target tokens
- optional global contrastive loss between two independently augmented views

It is intended to match the high-level training recipe used in
`olmoearth_pretrain` (latent MIM + patch discrimination), adapted to SpaceNit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from spacenit.arch.models import LatentPredictor
from spacenit.ingestion.augmentations import TransformConfig
from spacenit.pipeline.helpers import partition_masked_batch
from spacenit.pipeline.losses import (
    ContrastiveLoss,
    PatchDiscriminationLoss,
    UniformityLoss,
)
from spacenit.pipeline.runners.base_runner import (
    SpaceNitTrainRunner,
    SpaceNitTrainRunnerConfig,
)
from spacenit.structures import MaskedGeoSample, TokenVisibility

logger = getLogger(__name__)


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP for attribute access; keep FSDP wrapped."""
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP

        if isinstance(model, DDP):
            return cast(torch.nn.Module, model.module)
    except Exception:
        pass
    return model


def _build_visibility_mask(
    sample: MaskedGeoSample,
    *,
    sensor_order: list[str],
) -> torch.Tensor:
    """Build a global visibility mask from per-sensor masks.

    The encoder tokenizer prepends a 1-token sensor-type token per sensor. The
    masking policy produces masks only for patch tokens, so we inject a
    VISIBLE_ENCODER entry for each present sensor's type token.

    Returns:
        visibility_mask: ``(B, N_total)`` integer tensor with
        :class:`TokenVisibility` values aligned to the tokenizer's
        output ordering.
    """
    VIS = TokenVisibility.VISIBLE_ENCODER.value
    ABS = TokenVisibility.ABSENT.value

    B = int(sample.timestamps.shape[0])
    parts: list[torch.Tensor] = []

    for label in sensor_order:
        mask_key = f"{label}_mask"
        m = sample[mask_key]
        if m is None:
            continue
        if not isinstance(m, torch.Tensor):
            raise TypeError(f"Expected {mask_key} to be a torch.Tensor, got {type(m)}")
        if m.ndim == 1:
            m = m.unsqueeze(0)
        if m.ndim != 2 or int(m.shape[0]) != B:
            raise ValueError(f"Unexpected mask shape for {mask_key}: {tuple(m.shape)}")

        m = m.to(dtype=torch.long)

        # Sensor-type token is visible only when the sensor has at least one
        # non-ABSENT patch token for that sample. This prevents "phantom"
        # sensor-type tokens for fully-missing modalities from entering the
        # encoder/contrastive pool.
        type_tok = torch.full((B, 1), VIS, device=m.device, dtype=torch.long)
        all_absent = (m == ABS).all(dim=1, keepdim=True)  # (B, 1)
        type_tok = torch.where(all_absent, torch.full_like(type_tok, ABS), type_tok)
        parts.append(type_tok)
        parts.append(m)

    if not parts:
        return sample.timestamps.new_zeros((B, 0), dtype=torch.long)

    return torch.cat(parts, dim=1)  # (B, N_total)


@dataclass
class ContrastiveLatentMIMRunnerConfig(SpaceNitTrainRunnerConfig):
    """Configuration for :class:`ContrastiveLatentMIMRunner`."""

    masking_config: dict[str, Any] = field(default_factory=dict)
    patch_disc_tau: float = 0.1
    patch_disc_pred2unit: bool = False
    patch_disc_weight: float = 1.0

    global_contrastive_temperature: float = 0.1
    global_contrastive_weight: float = 0.1

    uniformity_weight: float = 0.0

    def build(
        self,
        model: LatentPredictor,
        device: torch.device | None = None,
    ) -> "ContrastiveLatentMIMRunner":
        kwargs = self.prepare_kwargs()
        return ContrastiveLatentMIMRunner(model=model, device=device, **kwargs)


class ContrastiveLatentMIMRunner(SpaceNitTrainRunner):
    """LatentMIM-style runner with patch discrimination (+ optional contrastive)."""

    def __init__(
        self,
        model: LatentPredictor,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        masking_config: dict[str, Any] | None = None,
        patch_disc_tau: float = 0.1,
        patch_disc_pred2unit: bool = False,
        patch_disc_weight: float = 1.0,
        global_contrastive_temperature: float = 0.1,
        global_contrastive_weight: float = 0.1,
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

        # Losses
        self.patch_disc = PatchDiscriminationLoss(
            tau=patch_disc_tau, pred2unit=patch_disc_pred2unit,
        )
        self.patch_disc_weight = float(patch_disc_weight)

        self.global_contrastive = ContrastiveLoss(
            initial_temperature=global_contrastive_temperature
        )
        self.global_contrastive_weight = float(global_contrastive_weight)

        self.uniformity = UniformityLoss() if uniformity_weight > 0 else None
        self.uniformity_weight = float(uniformity_weight)

        # For logging/debugging only.
        self.masking_config = masking_config or {}

        # Cache sensor order from the underlying LatentPredictor encoder config.
        base_model = _unwrap_ddp(self.model)
        if not hasattr(base_model, "encoder") or not hasattr(base_model.encoder, "config"):
            raise TypeError("ContrastiveLatentMIMRunner expects a LatentPredictor-like model")
        self._sensor_order = list(getattr(base_model.encoder.config, "sensor_labels", []))

    def _month_indices_from_timestamps(self, ts: torch.Tensor) -> torch.Tensor | None:
        # ts is typically (B, T, 3) with [day, month, year] in last dim.
        try:
            if ts.ndim >= 3 and ts.shape[-1] >= 2:
                month = ts[..., 1].to(dtype=torch.long)
                return month.clamp(min=0, max=11)
        except Exception:
            return None
        return None

    def train_batch(
        self,
        batch: tuple[int, MaskedGeoSample] | tuple[int, MaskedGeoSample, MaskedGeoSample],
        dry_run: bool = False,
    ) -> None:
        if batch is None:
            if dry_run:
                return
            raise TypeError("train_batch received None batch")

        self.model.train()

        # EMA update BEFORE forward pass (matching OLMo-Earth).
        # OLMo-Earth calls update_target_encoder() at the start of train_batch,
        # before any forward passes.  This ensures the target encoder is always
        # one step behind the online encoder, providing a stable prediction
        # target.
        if not dry_run and hasattr(self.model, "step_ema"):
            self.model.step_ema()

        patch_size = batch[0]
        batch_a = batch[1]
        batch_b = batch[2] if len(batch) > 2 else None

        micro_a = partition_masked_batch(batch_a, self.rank_microbatch_size)
        micro_b = (
            partition_masked_batch(batch_b, self.rank_microbatch_size) if batch_b is not None else None
        )
        num_mb = len(micro_a)
        if micro_b is not None and len(micro_b) != num_mb:
            raise ValueError("View A and B microbatch counts do not match")

        total_loss = torch.zeros([], device=self.device)
        total_patch = torch.zeros([], device=self.device)
        total_con = torch.zeros([], device=self.device)
        # Mask diagnostics (helps catch "missing==0" bugs that flatline downstream eval).
        total_abs_ratio = torch.zeros([], device=self.device)
        total_vis_ratio = torch.zeros([], device=self.device)

        for mb_idx in range(num_mb):
            with self._train_microbatch_context(mb_idx, num_mb):
                mb_a = micro_a[mb_idx].transfer_to(self.device)
                mb_b = micro_b[mb_idx].transfer_to(self.device) if micro_b is not None else None

                sensor_a = {k: mb_a[k] for k in mb_a.present_keys if mb_a[k] is not None}
                ts_a = cast(torch.Tensor, mb_a.timestamps)
                month_a = self._month_indices_from_timestamps(ts_a)
                vmask_a = _build_visibility_mask(mb_a, sensor_order=self._sensor_order)

                # Basic visibility stats.
                ABS = TokenVisibility.ABSENT.value
                VIS = TokenVisibility.VISIBLE_ENCODER.value
                denom = float(vmask_a.numel()) if vmask_a.numel() > 0 else 1.0
                abs_ratio = (vmask_a == ABS).sum().float() / denom
                vis_ratio = (vmask_a == VIS).sum().float() / denom
                total_abs_ratio += get_local_tensor((abs_ratio / num_mb).detach())
                total_vis_ratio += get_local_tensor((vis_ratio / num_mb).detach())

                with self._model_forward_context():
                    out_a = self.model(
                        sensor_a,
                        patch_size=patch_size,
                        month_indices=month_a,
                        visibility_mask=vmask_a,
                        contrastive_only=False,
                    )

                # Patch discrimination in raw embedding space (matches OLMo-Earth).
                # The loss compares raw decoder output against raw target encoder
                # output, not projected versions.
                patch_loss_a = self.patch_disc(out_a["predictions"], out_a["targets"])

                patch_loss = patch_loss_a
                con_loss = out_a["predictions"].new_zeros([])

                if mb_b is not None:
                    sensor_b = {k: mb_b[k] for k in mb_b.present_keys if mb_b[k] is not None}
                    ts_b = cast(torch.Tensor, mb_b.timestamps)
                    month_b = self._month_indices_from_timestamps(ts_b)
                    vmask_b = _build_visibility_mask(mb_b, sensor_order=self._sensor_order)

                    with self._model_forward_context():
                        out_b = self.model(
                            sensor_b,
                            patch_size=patch_size,
                            month_indices=month_b,
                            visibility_mask=vmask_b,
                            contrastive_only=False,
                        )

                    patch_loss_b = self.patch_disc(out_b["predictions"], out_b["targets"])
                    patch_loss = 0.5 * (patch_loss_a + patch_loss_b)

                    if self.global_contrastive_weight > 0:
                        # OLMo-Earth: InfoNCE between projected + pooled
                        # encoder outputs of the two views.  Route through
                        # online_proj so the projection head receives
                        # gradients (required for DDP with
                        # find_unused_parameters=False).
                        unwrapped = _unwrap_ddp(self.model)
                        proj_a = F.normalize(
                            unwrapped.online_proj(out_a["encoder_pooled"]),
                            dim=-1,
                        )
                        proj_b = F.normalize(
                            unwrapped.online_proj(out_b["encoder_pooled"]),
                            dim=-1,
                        )
                        con_loss = self.global_contrastive(proj_a, proj_b)

                loss = self.patch_disc_weight * patch_loss + self.global_contrastive_weight * con_loss

                if self.uniformity is not None and self.uniformity_weight > 0:
                    # Encourage spread in pooled encoder representations.
                    pooled = F.normalize(out_a["encoder_pooled"], dim=-1)
                    loss = loss + self.uniformity_weight * self.uniformity(pooled)

                loss = loss / num_mb

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning("NaN/Inf at microbatch %d", mb_idx)
                    continue

                loss.backward()

                total_loss += get_local_tensor(loss.detach())
                total_patch += get_local_tensor((patch_loss / num_mb).detach())
                total_con += get_local_tensor((con_loss / num_mb).detach())

        if not dry_run:
            self.trainer.record_metric("train/total_loss", total_loss, ReduceType.mean)
            self.trainer.record_metric(
                "train/patch_disc_loss", total_patch, ReduceType.mean
            )
            if self.global_contrastive_weight > 0:
                self.trainer.record_metric(
                    "train/global_contrastive_loss", total_con, ReduceType.mean
                )
            self.trainer.record_metric("train/mask_absent_ratio", total_abs_ratio, ReduceType.mean)
            self.trainer.record_metric("train/mask_visible_ratio", total_vis_ratio, ReduceType.mean)

