"""Test objective (loss) functions."""

import logging

import torch

from spacenit.arch.adaptive_vision_encoder import EmbeddingsAndMasks
from spacenit.pipeline.objectives import (
    AbsoluteErrorLoss,
    AdjustedPatchContrastiveLoss,
    CategoryLoss,
    InfoNCEObjective,
    PatchContrastiveLoss,
    PatchContrastiveLossNew,
    SquaredErrorLoss,
)
from spacenit.structures import TokenVisibility

logger = logging.getLogger(__name__)


def test_patch_disc_loss() -> None:
    """Just test that it runs as expected."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t))
        * TokenVisibility.PREDICTED.value,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * TokenVisibility.PREDICTED.value,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchContrastiveLoss()
    loss_value = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 0.5


def test_adjusted_patch_disc_loss_comparison() -> None:
    """Compare loss under different mu/sigma configs."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t))
        * TokenVisibility.PREDICTED.value,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * TokenVisibility.PREDICTED.value,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )

    # Loss hard is very sharp focus on the hard negatives, expect higher loss
    loss_easy = AdjustedPatchContrastiveLoss(mu=0.3, sigma=1.0).compute(
        preds, targets
    )
    loss_hard = AdjustedPatchContrastiveLoss(mu=0.9, sigma=0.1).compute(
        preds, targets
    )

    assert loss_hard >= loss_easy or abs(loss_hard - loss_easy) < 1e-3


def test_if_old_and_new_loss_are_the_same() -> None:
    """Test that the old and new patch discrimination loss are the same."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 2
    preds = EmbeddingsAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t))
        * TokenVisibility.PREDICTED.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * TokenVisibility.PREDICTED.value,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t))
        * TokenVisibility.PREDICTED.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * TokenVisibility.PREDICTED.value,
    )
    loss_old = PatchContrastiveLoss()
    loss_new = PatchContrastiveLossNew()
    old_loss = loss_old.compute(preds, targets)
    new_loss = loss_new.compute(preds, targets)
    logger.info(f"old_loss: {old_loss}, new_loss: {new_loss}")
    assert torch.isclose(old_loss, new_loss)


def test_if_old_and_new_loss_are_the_same_uneven_number_of_predicted_tokens() -> None:
    """Test that the old and new patch contrastive loss are the same."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 2

    s2_preds_mask = torch.randint(0, 3, (b, t_h, t_w, t))

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_preds_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * TokenVisibility.PREDICTED.value,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss_old = PatchContrastiveLoss()
    loss_new = PatchContrastiveLossNew()
    old_loss = loss_old.compute(preds, targets)
    new_loss = loss_new.compute(preds, targets)
    logger.info(f"old_loss: {old_loss}, new_loss: {new_loss}")
    assert torch.isclose(old_loss, new_loss)


def test_patch_disc_loss_averaged_over_batch_size() -> None:
    """Test it doesn't scale with batch size."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchContrastiveLoss()
    loss_value = loss.compute(preds, targets)

    # now, use a larger batch size
    b, t_h, t_w, t, d = 8, 4, 4, 2, 2

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss_value_8 = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert torch.isclose(loss_value, loss_value_8)


def test_l1_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = AbsoluteErrorLoss()
    loss_value = loss.compute(preds, targets)
    # MAE should be 1 since preds are 1, targets are 0
    assert loss_value == 1


def test_l2_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = SquaredErrorLoss()
    loss_value = loss.compute(preds, targets)
    # MSE should be 4 since preds are 2, targets are 0
    assert loss_value == 4


def test_cross_entropy_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = EmbeddingsAndMasks(
        sentinel2_l2a=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = EmbeddingsAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, 1), dtype=torch.long),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, 1), dtype=torch.long),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = CategoryLoss()
    loss_value = loss.compute(preds, targets)
    # loss for BCE, prediction of .5 for both classes
    assert torch.isclose(loss_value, -torch.log(torch.tensor(0.5)), 0.0001)


def test_infonce_objective() -> None:
    """Just test that it runs as expected."""
    b, d = 16, 128

    loss = InfoNCEObjective()
    loss_value = loss.compute(torch.ones((b, d)), torch.zeros((b, d)))
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 0.5
    # check the weight
    loss = InfoNCEObjective(weight=0.1)
    w_loss_value = loss.compute(torch.ones((b, d)), torch.zeros((b, d)))
    assert 0.1 * loss_value == w_loss_value
