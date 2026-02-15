"""Objective (loss) functions for SpaceNit training."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from einops import rearrange, repeat
from torch import Tensor

from spacenit.settings import Config
from spacenit.arch.adaptive_vision_encoder import EmbeddingsAndMasks, PoolingType
from spacenit.arch.band_tokenization import TokenizationConfig
from spacenit.structures import MaskedGeoSample, TokenVisibility

logger = logging.getLogger(__name__)


class Objective(ABC):
    """Abstract base class for training objectives (loss functions)."""

    name: str

    @abstractmethod
    def compute(self, predictions: Any, targets: Any, **kwargs: Any) -> Tensor:
        """Compute the objective between predictions and targets."""
        pass

    @staticmethod
    def _expand_and_reciprocate(t: Tensor) -> Tensor:
        """As described in the name.

        >>> _expand_and_reciprocate(torch.tensor([1, 2, 3]))
        tensor([1.0000, 0.5000, 0.5000, 0.3333, 0.3333, 0.3333])
        """
        reciprocals = torch.reciprocal(t.float())
        return torch.repeat_interleave(reciprocals, t)


OBJECTIVE_REGISTRY = ClassRegistry[Objective]()


@OBJECTIVE_REGISTRY.register("global_contrastive")
class GlobalContrastiveLoss(Objective):
    """Contrastive objective across all patches using every sample in a batch.

    Discriminates across patches using all samples in a batch.
    """

    name = "GlobalContrastive"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False):
        """Initialize global contrastive loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.pred2unit = pred2unit

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute global contrastive loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_embeddings_and_masks()
        all_targets = targets.flatten_embeddings_and_masks()[0]

        pred = all_preds[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
        target = all_targets[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape
        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
        count = (all_masks == TokenVisibility.PREDICTED.value).sum(dim=-1)

        labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(
            bs, 1
        )
        loss = F.cross_entropy(
            scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
        ) * (self.tau * 2)

        # emulate averaging across the batch dimension
        loss_multiplier = self._expand_and_reciprocate(count)
        # can't use bs here since this is after the unsqueezing, so bs == 1
        loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
        return loss


@OBJECTIVE_REGISTRY.register("sensor_global_contrastive")
class SensorGlobalContrastiveLoss(Objective):
    """Per-sensor global contrastive objective.

    Discriminates across patches using all samples in a batch, per sensor.
    """

    name = "SensorGlobalContrastive"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False):
        """Initialize per-sensor global contrastive loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.pred2unit = pred2unit

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute per-sensor global contrastive loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        sensor_preds, sensor_masks = predictions.flatten_embeddings_and_masks(
            return_lists=True
        )
        sensor_targets = targets.flatten_embeddings_and_masks(return_lists=True)[0]

        total_loss = 0
        for all_preds, all_masks, all_targets in zip(
            sensor_preds, sensor_masks, sensor_targets
        ):
            pred = all_preds[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
            target = all_targets[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
            bs, nt, _ = pred.shape
            if nt == 0:
                # If no predicted values, skip this sensor
                logger.warning("No predicted values for this sensor")
                continue
            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
            count = (all_masks == TokenVisibility.PREDICTED.value).sum(dim=-1)

            labels = torch.arange(nt, dtype=torch.long, device=pred.device)[
                None
            ].repeat(bs, 1)
            loss = F.cross_entropy(
                scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
            ) * (self.tau * 2)

            # emulate averaging across the batch dimension
            loss_multiplier = self._expand_and_reciprocate(count)
            # can't use bs here since this is after the unsqueezing, so bs == 1
            loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
            total_loss += loss

        return total_loss


@OBJECTIVE_REGISTRY.register("patch_contrastive_new")
class PatchContrastiveLossNew(Objective):
    """Per-sample patch contrastive objective.

    This has lower memory consumption than the legacy patch contrastive loss.
    It does not support global contrastive loss.
    """

    name = "PatchContrastive"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False, weight: float = 1.0):
        """Initialize patch contrastive loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            weight: the weight to apply to this loss
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch contrastive loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_embeddings_and_masks()
        all_targets = targets.flatten_embeddings_and_masks()[0]

        # Samples may have different number of tokens
        pred = all_preds[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
        target = all_targets[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        count = (all_masks == TokenVisibility.PREDICTED.value).sum(dim=-1)
        losses = []
        start = 0
        for c in count:
            end = start + c
            if c == 0:
                # we will occasionally get a sample with no predicted values due to missing data
                logger.warning("No predicted values for this sample")
                continue
            pred_sample = pred[:, start:end, :]
            target_sample = target[:, start:end, :]
            score_sample = (
                torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau
            )
            labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
            loss = F.cross_entropy(
                score_sample.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ) * (self.tau * 2)
            loss = loss.mean()
            losses.append(loss)
            start = end
        loss = torch.stack(losses).mean()
        return self.weight * loss


@OBJECTIVE_REGISTRY.register("sensor_patch_contrastive_new")
class SensorPatchContrastiveLossNew(Objective):
    """Per-sensor per-sample patch contrastive objective.

    This has lower memory consumption than the legacy patch contrastive loss.
    It does not support global contrastive loss.
    """

    name = "SensorPatchContrastive"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        sensor_weights: dict[str, float] | None = None,
    ):
        """Initialize per-sensor patch contrastive loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            weight: the weight to apply to this loss
            sensor_weights: the weights to apply to each sensor
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.sensor_weights = sensor_weights

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute per-sensor patch contrastive loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        sensor_preds, sensor_masks = predictions.flatten_embeddings_and_masks(
            return_lists=True
        )
        sensor_targets = targets.flatten_embeddings_and_masks(return_lists=True)[0]

        # Accumulate to the total loss
        total_loss = 0
        for all_preds, all_masks, all_targets, sensor in zip(
            sensor_preds, sensor_masks, sensor_targets, targets.present_keys
        ):
            # Samples may have different number of tokens
            pred = all_preds[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
            target = all_targets[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
            bs, nt, _ = pred.shape

            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            count = (all_masks == TokenVisibility.PREDICTED.value).sum(dim=-1)
            losses = []
            start = 0
            for c in count:
                end = start + c
                if c == 0:
                    continue
                pred_sample = pred[:, start:end, :]
                target_sample = target[:, start:end, :]
                score_sample = (
                    torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau
                )
                labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
                loss = F.cross_entropy(
                    score_sample.flatten(0, 1),
                    labels.flatten(0, 1),
                    reduction="none",
                ) * (self.tau * 2)
                loss = loss.mean()
                losses.append(loss)
                start = end
            if len(losses) == 0:
                continue
            loss = torch.stack(losses).mean()
            if self.sensor_weights is not None:
                loss = loss * self.sensor_weights[sensor]
            total_loss += loss

        return self.weight * total_loss


@OBJECTIVE_REGISTRY.register("patch_contrastive")
class PatchContrastiveLoss(Objective):
    """Patch contrastive objective with optional cross-sample masking."""

    name = "PatchContrastive"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        mask_other_samples: bool = True,
    ):
        """Initialize patch contrastive loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.mask_other_samples = mask_other_samples

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch contrastive loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_embeddings_and_masks()
        all_targets = targets.flatten_embeddings_and_masks()[0]
        predicted_mask = all_masks == TokenVisibility.PREDICTED.value
        pred = all_preds[predicted_mask].unsqueeze(dim=0)
        target = all_targets[predicted_mask].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
        count = (all_masks == TokenVisibility.PREDICTED.value).sum(dim=-1)
        if self.mask_other_samples:
            logit_mask = torch.full_like(scores, -torch.finfo(scores.dtype).max)
            start = 0
            for c in count:
                end = start + c
                logit_mask[:, start:end, start:end] = 0
                start += c
            scores = scores + logit_mask

        labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(
            bs, 1
        )
        loss = F.cross_entropy(
            scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
        ) * (self.tau * 2)

        # emulate averaging across the batch dimension
        loss_multiplier = self._expand_and_reciprocate(count)
        # can't use bs here since this is after the unsqueezing, so bs == 1
        loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
        return loss


@OBJECTIVE_REGISTRY.register("adjusted_patch_contrastive")
class AdjustedPatchContrastiveLoss(Objective):
    """Adjusted patch contrastive objective with Gaussian-weighted negatives.

    Reference: https://proceedings.neurips.cc/paper_files/paper/2023/file/48aaa5ea741ae8430bd58e25917d267d-Paper-Conference.pdf
    """

    name = "AdjustedPatchContrastive"

    def __init__(
        self,
        tau: float = 0.1,
        mu: float = 0.7,
        sigma: float = 1.0,
        pred2unit: bool = False,
    ):
        """Initialize adjusted patch contrastive loss.

        Args:
            tau: the softmax temperature
            mu: the mean of the Gaussian distribution
            sigma: the standard deviation of the Gaussian distribution
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.mu = mu
        self.sigma = sigma
        self.pred2unit = pred2unit

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute adjusted patch contrastive loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_embeddings_and_masks()
        all_targets = targets.flatten_embeddings_and_masks()[0]

        pred = all_preds[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
        target = all_targets[all_masks == TokenVisibility.PREDICTED.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        count = (all_masks == TokenVisibility.PREDICTED.value).sum(dim=-1)

        losses = []
        start = 0
        for c in count:
            end = start + c
            pred_sample = pred[:, start:end, :]  # (1, c, d)
            target_sample = target[:, start:end, :]  # (1, c, d)

            sim_matrix = torch.einsum(
                "npd,nqd->npq", pred_sample, target_sample
            )  # (1, c, c)

            pos_scores = torch.diagonal(sim_matrix, dim1=-2, dim2=-1)  # (1, c)
            pos_scores = pos_scores / self.tau

            # Mask out diagonal (positives) to get negatives
            mask = ~torch.eye(c, dtype=torch.bool, device=pred.device)
            neg_scores = sim_matrix.masked_select(mask).view(1, c, c - 1)  # (1, c, c-1)
            neg_scores = neg_scores / self.tau

            # Apply Gaussian-based weights to negatives
            weight = (
                1.0
                / (self.sigma * math.sqrt(2 * math.pi))
                * torch.exp(
                    -((neg_scores * self.tau - self.mu) ** 2)
                    / (2 * math.pow(self.sigma, 2))
                )
            )  # (1, c, c-1)
            # Normalize the weights per query
            weight = weight / weight.mean(dim=-1, keepdim=True)
            neg_scores = neg_scores * weight.detach()

            # Reconstruct the sim_matrix
            sim_matrix = torch.zeros(
                1, c, c, device=pred.device, dtype=neg_scores.dtype
            )
            sim_matrix.diagonal(dim1=-2, dim2=-1).copy_(pos_scores)
            sim_matrix.masked_scatter_(mask, neg_scores)

            labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
            loss = F.cross_entropy(
                sim_matrix.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ) * (self.tau * 2)
            loss = loss.mean()
            losses.append(loss)
            start = end

        loss = torch.stack(losses).mean()
        return loss


@OBJECTIVE_REGISTRY.register("absolute_error")
class AbsoluteErrorLoss(Objective):
    """L1 / mean absolute error objective."""

    name = "AbsoluteError"

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute L1 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_embeddings_and_masks()
        all_targets = targets.flatten_embeddings_and_masks()[0]
        pred = all_preds[all_masks == TokenVisibility.PREDICTED.value]
        target = all_targets[all_masks == TokenVisibility.PREDICTED.value]

        return F.l1_loss(pred, target)


@OBJECTIVE_REGISTRY.register("squared_error")
class SquaredErrorLoss(Objective):
    """L2 / mean squared error objective."""

    name = "SquaredError"

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute L2 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_embeddings_and_masks()
        all_targets = targets.flatten_embeddings_and_masks()[0]
        pred = all_preds[all_masks == TokenVisibility.PREDICTED.value]
        target = all_targets[all_masks == TokenVisibility.PREDICTED.value]
        return F.mse_loss(pred, target)


@OBJECTIVE_REGISTRY.register("pixel_reconstruction")
class PixelReconstructionLoss(Objective):
    """Pixel-level reconstruction objective (masked auto-encoding)."""

    name = "PixelReconstruction"

    def __init__(
        self,
        loss_function: str = "MSELoss",
        only_decode: bool = True,
        weight: float = 1.0,
        tokenization_config: TokenizationConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize pixel reconstruction loss.

        Args:
            loss_function: pytorch loss to use
            only_decode: only calculate loss on PREDICTED masked tokens, otherwise all
            weight: the weight to apply to this loss
            tokenization_config: Optional config for custom band groupings
            **kwargs: arguments for pytorch loss constructor
        """
        self.only_decode = only_decode
        self.loss = getattr(torch.nn, loss_function)(reduction="sum", **kwargs)
        self.weight = weight
        self.tokenization_config = tokenization_config or TokenizationConfig()

    # data: [B, H, W, T, C]
    def _flatten_spatiotemporal_data(
        self, data: EmbeddingsAndMasks
    ) -> tuple[Tensor, Tensor]:
        masks = []
        datas = []
        for sensor in data.present_keys:
            pred = getattr(data, sensor)
            if pred is not None:
                mask = getattr(data, data.get_masked_modality_name(sensor))
                for idx, channel_set_idxs in enumerate(
                    self.tokenization_config.group_indices_for(sensor)
                ):
                    bs_mask = mask[..., idx]
                    bs_mask = repeat(
                        bs_mask, "b h w t -> b h w t c", c=len(channel_set_idxs)
                    )
                    bs_mask = rearrange(bs_mask, "b h w t c -> b (h w t c)")
                    masks.append(bs_mask)
                    bs_data = pred[..., channel_set_idxs]
                    bs_data = rearrange(bs_data, "b h w t c -> b (h w t c)")
                    datas.append(bs_data)
        return torch.cat(datas, dim=1), torch.cat(masks, dim=1)

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: MaskedGeoSample, **kwargs: Any
    ) -> Tensor:
        """Compute pixel reconstruction loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        data, masks = self._flatten_spatiotemporal_data(predictions)
        valid_dict = {}
        for sensor in predictions.present_keys:
            if getattr(predictions, sensor) is not None:
                masked_name = predictions.get_masked_modality_name(sensor)
                valid_dict[sensor] = getattr(targets, sensor)
                valid_dict[masked_name] = getattr(targets, masked_name)
        valid_targets = EmbeddingsAndMasks(**valid_dict)
        labels, label_masks = self._flatten_spatiotemporal_data(valid_targets)
        if self.only_decode:
            decode = label_masks == TokenVisibility.PREDICTED.value
        else:
            decode = label_masks != TokenVisibility.ABSENT.value
        data = data * decode
        labels = labels * decode
        return self.weight * self.loss(data, labels) / torch.count_nonzero(decode)


@OBJECTIVE_REGISTRY.register("category")
class CategoryLoss(Objective):
    """Cross-entropy classification objective."""

    name = "Category"

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: EmbeddingsAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute cross entropy between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_embeddings_and_masks()
        all_targets = targets.flatten_embeddings_and_masks()[0]
        pred = all_preds[all_masks == TokenVisibility.PREDICTED.value]
        target = all_targets[all_masks == TokenVisibility.PREDICTED.value]

        return F.cross_entropy(pred, target.squeeze())


@OBJECTIVE_REGISTRY.register("InfoNCE")
class InfoNCEObjective(Objective):
    """InfoNCE contrastive objective."""

    name = "InfoNCE"

    def __init__(self, tau: float = 0.1, weight: float = 1):
        """Initialize InfoNCE objective.

        Args:
            tau: the softmax temperature
            weight: the weight to apply to this loss
        """
        self.tau = tau
        self.weight = weight

    def compute(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> Tensor:
        """Compute InfoNCE between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        predictions = F.normalize(predictions, p=2, dim=-1)
        targets = F.normalize(targets, p=2, dim=-1)
        logits = predictions @ targets.transpose(-2, -1)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(predictions), device=predictions.device)
        return self.weight * F.cross_entropy(logits / self.tau, labels)


@OBJECTIVE_REGISTRY.register("KoLeo")
class KoLeoRegularizer(Objective):
    """KoLeo uniformity regularizer.

    Derives from the Kozachenko-Leonenko differential entropy estimator and
    encourages a uniform span of the features within a batch.

    https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    """

    name = "KoLeo"

    def __init__(
        self,
        weight: float = 0.1,
        mode: str = "instance",
        eps: float = 1e-8,
    ) -> None:
        """Initialize KoLeo regularizer.

        Args:
            weight: a weight to apply to the regularization value. Default value follows Dinov2
            eps: small value to avoid division by zero.
            mode: one of "instance" or "patch" - whether to compute
                nearest neighbours at the instance or patch level
        """
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2, eps=eps)
        if mode not in ["instance", "patch"]:
            raise ValueError(f"Unsupported mode {mode}")
        self.mode = mode
        self.weight = weight

    @staticmethod
    def pairwise_nearest_neighbours(x: torch.Tensor) -> torch.Tensor:
        """Pairwise nearest neighbors for L2-normalized vectors.

        Uses Torch rather than Faiss to remain on GPU.

        Args:
            x: embeddings against which we want to compute nearest neighbours.

        Returns:
            indices: indices of nearest neighbour (i.e. indices[i] will return
            the index for the nearest neighbour of the ith embedding).
        """
        # pairwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, indices = torch.max(dots, dim=1)
        return indices

    def compute(
        self, predictions: EmbeddingsAndMasks, targets: None, **kwargs: Any
    ) -> Tensor:
        """Compute the KoLeo regularization term.

        Args:
            predictions: Model predictions. Unlike other objectives, these are
                _online encoder outputs_, not decoder outputs.
            targets: Unused, and only kept for consistency.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed regularization value.
        """
        if isinstance(predictions, EmbeddingsAndMasks):
            if self.mode == "patch":
                if not isinstance(predictions, EmbeddingsAndMasks):
                    raise ValueError(
                        "predictions must be EmbeddingsAndMasks for patch mode"
                    )
                all_preds, all_masks = predictions.flatten_embeddings_and_masks()
                online_encodings = all_preds[
                    all_masks == TokenVisibility.VISIBLE_ENCODER.value
                ]
            else:
                online_encodings = predictions.pool_unmasked_tokens(
                    PoolingType.MEAN, spatial_pooling=False
                )
        else:
            online_encodings = predictions

        # apply l2 norm
        online_encodings = F.normalize(online_encodings, eps=self.eps, p=2, dim=-1)
        idx_of_nn = self.pairwise_nearest_neighbours(online_encodings)
        distances_to_nn = self.pdist(online_encodings, online_encodings[idx_of_nn])
        return self.weight * -torch.log(distances_to_nn + self.eps).mean()


@dataclass
class ObjectiveConfig(Config):
    """Configuration for training objectives.

    Args:
        objective_config: Objective config in the format of
        e.g.
        {
            "type": "patch_contrastive",
            # rest of init kwargs
        }
    """

    objective_config: dict[str, Any]  # List of objective configs

    def build(self) -> Objective:
        """Build an Objective from the config."""
        objective_key = self.objective_config.pop("type")
        return OBJECTIVE_REGISTRY.get_class(objective_key)(**self.objective_config)
