"""Loss functions for self-supervised pretraining.

Replaces ``objectives.py`` with different implementations:

- :class:`LatentPredictionLoss` -- Smooth L1 loss (Huber) on predicted vs
  target latents (replaces L1/L2 with smooth L1 as default).
- :class:`ContrastiveLoss` -- InfoNCE with learnable temperature and
  dot-product similarity (``torch.mm``) on L2-normalized vectors (replaces
  einsum-based similarity with fixed temperature).
- :class:`ReconstructionLoss` -- pixel reconstruction with configurable
  loss function.
- :class:`UniformityLoss` -- negative log of average pairwise distance
  (replaces KoLeo regularizer with a different uniformity measure from
  "Understanding Contrastive Representation Learning").
- :class:`CompositeLoss` -- weighted sum of multiple losses.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Latent Prediction Loss
# ---------------------------------------------------------------------------


class LatentPredictionLoss(nn.Module):
    """Smooth L1 (Huber) loss between predicted and target latent representations.

    Smooth L1 is less sensitive to outliers than L2 and more stable than
    L1 near zero.  This is different from the original codebase which
    used plain L1 or L2.

    Args:
        beta: Threshold at which the loss transitions from L2 to L1
            behavior.  Smaller values make the loss more like L1.
        normalize: Whether to L2-normalize predictions and targets before
            computing the loss.
    """

    def __init__(self, beta: float = 1.0, normalize: bool = True) -> None:
        super().__init__()
        self.beta = beta
        self.normalize = normalize

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute smooth L1 loss.

        Args:
            predictions: ``(B, N, D)`` predicted representations.
            targets: ``(B, N, D)`` target representations (detached).

        Returns:
            Scalar loss value.
        """
        if self.normalize:
            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)

        return F.smooth_l1_loss(predictions, targets, beta=self.beta)


# ---------------------------------------------------------------------------
# Contrastive Loss (InfoNCE)
# ---------------------------------------------------------------------------


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with learnable temperature.

    Uses ``torch.mm`` (dot product) on L2-normalized vectors for pairwise
    similarity computation.  The temperature parameter is learnable
    (log-parameterized for numerical stability), unlike the original
    fixed temperature.

    Supports both patch-level and global (pooled) contrastive learning.

    Args:
        initial_temperature: Starting temperature value.
        min_temperature: Lower bound for temperature (prevents collapse).
        max_temperature: Upper bound for temperature.
    """

    def __init__(
        self,
        initial_temperature: float = 0.07,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(initial_temperature))
        )
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    @property
    def temperature(self) -> Tensor:
        """Current temperature, clamped to [min, max]."""
        return self.log_temperature.exp().clamp(
            self.min_temperature, self.max_temperature
        )

    def forward(
        self,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor | None = None,
    ) -> Tensor:
        """Compute InfoNCE loss.

        Args:
            anchors: ``(B, D)`` anchor representations (L2-normalized).
            positives: ``(B, D)`` positive representations (L2-normalized).
            negatives: ``(M, D)`` explicit negatives.  If ``None``, uses
                all other samples in the batch as negatives.

        Returns:
            Scalar loss value.
        """
        temp = self.temperature

        # Positive similarities: (B,)
        pos_sim = (anchors * positives).sum(dim=-1) / temp

        if negatives is None:
            # Use all other batch elements as negatives.
            # Similarity matrix: (B, B) via dot product on L2-normalized vectors.
            sim_matrix = torch.mm(anchors, positives.t()) / temp
            # Mask out self-similarity on diagonal
            B = anchors.shape[0]
            labels = torch.arange(B, device=anchors.device)
            loss = F.cross_entropy(sim_matrix, labels)
        else:
            # Explicit negatives
            # neg_sim: (B, M)
            neg_sim = torch.mm(anchors, negatives.t()) / temp
            # Combine: (B, 1+M) where first column is positive
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
            loss = F.cross_entropy(logits, labels)

        return loss


# ---------------------------------------------------------------------------
# Reconstruction Loss
# ---------------------------------------------------------------------------


class ReconstructionLoss(nn.Module):
    """Pixel reconstruction loss with configurable loss function.

    Args:
        loss_type: One of ``"mse"``, ``"l1"``, ``"smooth_l1"``.
        normalize_target: Whether to normalize the target per-patch
            (subtract mean, divide by std) before computing loss.
        beta: Beta parameter for smooth L1 loss.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        normalize_target: bool = True,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.normalize_target = normalize_target
        self.beta = beta

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute reconstruction loss.

        Args:
            predictions: ``(B, C, H, W)`` or ``(B, N, D)`` predicted values.
            targets: Same shape as predictions.
            mask: Optional boolean mask; loss is computed only where
                ``mask == True``.

        Returns:
            Scalar loss value.
        """
        if self.normalize_target:
            # Per-sample normalization
            mean = targets.mean(dim=-1, keepdim=True)
            var = targets.var(dim=-1, keepdim=True, unbiased=False)
            targets = (targets - mean) / (var.sqrt() + 1e-6)

        if self.loss_type == "mse":
            loss = F.mse_loss(predictions, targets, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(predictions, targets, reduction="none")
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(
                predictions, targets, beta=self.beta, reduction="none"
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type!r}")

        if mask is not None:
            # Expand mask to match loss shape if needed
            while mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            loss = loss * mask.float()
            # Mean over masked positions only
            n_masked = mask.float().sum().clamp(min=1.0)
            return loss.sum() / n_masked
        else:
            return loss.mean()


# ---------------------------------------------------------------------------
# Uniformity Loss
# ---------------------------------------------------------------------------


class UniformityLoss(nn.Module):
    """Uniformity regularizer based on average pairwise distance.

    Encourages representations to be uniformly distributed on the
    hypersphere.  Uses the negative log of average pairwise Gaussian
    potential, from "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere" (Wang & Isola, 2020).

    This replaces the KoLeo regularizer from the original codebase with
    a different uniformity measure.

    The loss is:

    .. math::

        \\mathcal{L}_{\\text{uniform}} = \\log \\frac{1}{N^2}
        \\sum_{i,j} e^{-t \\|z_i - z_j\\|^2}

    Args:
        t: Temperature parameter controlling the Gaussian width.
    """

    def __init__(self, t: float = 2.0) -> None:
        super().__init__()
        self.t = t

    def forward(self, embeddings: Tensor) -> Tensor:
        """Compute uniformity loss.

        Args:
            embeddings: ``(B, D)`` L2-normalized embeddings.

        Returns:
            Scalar uniformity loss (lower = more uniform).
        """
        # Pairwise squared distances: (B, B)
        sq_dists = torch.cdist(embeddings, embeddings, p=2).pow(2)

        # Gaussian potential
        potentials = torch.exp(-self.t * sq_dists)

        # Exclude self-pairs (diagonal)
        B = embeddings.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        potentials = potentials[mask]

        # Log average potential
        return potentials.mean().log()


# ---------------------------------------------------------------------------
# Composite Loss
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """Weighted sum of multiple loss functions.

    Args:
        losses: Dictionary mapping loss names to ``(loss_module, weight)``
            pairs.
    """

    def __init__(self, losses: dict[str, tuple[nn.Module, float]]) -> None:
        super().__init__()
        self._loss_names = list(losses.keys())
        self._weights: dict[str, float] = {}
        self.loss_modules = nn.ModuleDict()

        for name, (module, weight) in losses.items():
            self.loss_modules[name] = module
            self._weights[name] = weight

    def forward(self, **loss_inputs: dict) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute weighted sum of losses.

        Args:
            **loss_inputs: Keyword arguments.  Each loss module receives
                the subset of arguments it needs.  The convention is that
                inputs for loss ``"foo"`` are passed as ``foo_predictions``,
                ``foo_targets``, etc.

        Returns:
            Tuple of (total_loss, per_loss_dict) where per_loss_dict maps
            loss names to their individual (unweighted) values.
        """
        total: Tensor | None = None
        individual: dict[str, Tensor] = {}

        for name in self._loss_names:
            module = self.loss_modules[name]
            weight = self._weights[name]

            # Extract inputs for this loss (prefix convention)
            prefix = f"{name}_"
            kwargs = {
                k[len(prefix):]: v
                for k, v in loss_inputs.items()
                if k.startswith(prefix)
            }

            loss_val = module(**kwargs)
            individual[name] = loss_val.detach()

            weighted = weight * loss_val
            total = weighted if total is None else total + weighted

        # If no losses were computed, return a zero tensor (should not
        # happen in practice).
        if total is None:
            total = torch.tensor(0.0)

        return total, individual
