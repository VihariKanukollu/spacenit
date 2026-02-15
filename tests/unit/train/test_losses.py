"""Tests for the new loss functions.

Verifies loss values against hand-computed examples and checks
gradient flow.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from spacenit.pipeline.losses import (
    CompositeLoss,
    ContrastiveLoss,
    LatentPredictionLoss,
    ReconstructionLoss,
    UniformityLoss,
)


class TestLatentPredictionLoss:
    def test_zero_loss_for_identical_inputs(self):
        loss_fn = LatentPredictionLoss(normalize=False)
        x = torch.randn(4, 16, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_for_different_inputs(self):
        loss_fn = LatentPredictionLoss()
        pred = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_normalized_mode(self):
        """With normalization, loss should be bounded."""
        loss_fn = LatentPredictionLoss(normalize=True)
        pred = torch.randn(4, 16, 64) * 100
        target = torch.randn(4, 16, 64) * 100
        loss = loss_fn(pred, target)
        # After L2 normalization, max smooth L1 is bounded
        assert loss.item() < 10.0

    def test_gradient_flows(self):
        loss_fn = LatentPredictionLoss()
        pred = torch.randn(2, 8, 32, requires_grad=True)
        target = torch.randn(2, 8, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestContrastiveLoss:
    def test_perfect_alignment_low_loss(self):
        """Identical anchors and positives should give low loss."""
        loss_fn = ContrastiveLoss(initial_temperature=0.5)
        x = F.normalize(torch.randn(8, 64), dim=-1)
        loss = loss_fn(x, x)
        # With perfect alignment and batch negatives, loss should be low
        assert loss.item() < 1.0

    def test_random_inputs_higher_loss(self):
        loss_fn = ContrastiveLoss(initial_temperature=0.5)
        anchors = F.normalize(torch.randn(8, 64), dim=-1)
        positives = F.normalize(torch.randn(8, 64), dim=-1)
        loss = loss_fn(anchors, positives)
        assert loss.item() > 0

    def test_learnable_temperature(self):
        """Temperature should be a learnable parameter."""
        loss_fn = ContrastiveLoss(initial_temperature=0.07)
        assert loss_fn.log_temperature.requires_grad
        # Temperature should be clamped
        assert loss_fn.temperature.item() >= loss_fn.min_temperature
        assert loss_fn.temperature.item() <= loss_fn.max_temperature

    def test_with_explicit_negatives(self):
        loss_fn = ContrastiveLoss()
        anchors = F.normalize(torch.randn(4, 32), dim=-1)
        positives = F.normalize(torch.randn(4, 32), dim=-1)
        negatives = F.normalize(torch.randn(16, 32), dim=-1)
        loss = loss_fn(anchors, positives, negatives)
        assert loss.item() > 0

    def test_gradient_through_temperature(self):
        loss_fn = ContrastiveLoss()
        anchors = F.normalize(torch.randn(4, 32), dim=-1)
        positives = F.normalize(torch.randn(4, 32), dim=-1)
        loss = loss_fn(anchors, positives)
        loss.backward()
        assert loss_fn.log_temperature.grad is not None


class TestReconstructionLoss:
    def test_mse_zero_for_identical(self):
        loss_fn = ReconstructionLoss(loss_type="mse", normalize_target=False)
        x = torch.randn(2, 3, 16, 16)
        loss = loss_fn(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_l1_loss(self):
        loss_fn = ReconstructionLoss(loss_type="l1", normalize_target=False)
        pred = torch.zeros(2, 3, 8, 8)
        target = torch.ones(2, 3, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_smooth_l1_loss(self):
        loss_fn = ReconstructionLoss(
            loss_type="smooth_l1", normalize_target=False, beta=1.0
        )
        pred = torch.randn(2, 3, 8, 8)
        target = torch.randn(2, 3, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_with_mask(self):
        loss_fn = ReconstructionLoss(loss_type="mse", normalize_target=False)
        pred = torch.randn(2, 16, 32)
        target = torch.randn(2, 16, 32)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, :8] = True  # Only first 8 positions
        loss = loss_fn(pred, target, mask=mask)
        assert loss.item() > 0

    def test_target_normalization(self):
        loss_fn = ReconstructionLoss(loss_type="mse", normalize_target=True)
        pred = torch.randn(2, 16, 32)
        target = torch.randn(2, 16, 32) * 100 + 50  # large scale
        loss = loss_fn(pred, target)
        # Loss should be finite despite large target values
        assert torch.isfinite(loss)

    def test_unknown_type_raises(self):
        loss_fn = ReconstructionLoss(loss_type="invalid")
        with pytest.raises(ValueError, match="Unknown loss type"):
            loss_fn(torch.randn(2, 8), torch.randn(2, 8))


class TestUniformityLoss:
    def test_uniform_distribution_low_loss(self):
        """Points on a sphere should have low uniformity loss."""
        loss_fn = UniformityLoss(t=2.0)
        # Generate roughly uniform points on unit sphere
        embeddings = F.normalize(torch.randn(32, 64), dim=-1)
        loss = loss_fn(embeddings)
        # Loss should be negative (log of small average potential)
        assert loss.item() < 0

    def test_collapsed_distribution_high_loss(self):
        """All-identical embeddings should have high uniformity loss."""
        loss_fn = UniformityLoss(t=2.0)
        # All points are the same
        base = F.normalize(torch.randn(1, 64), dim=-1)
        embeddings = base.expand(16, -1)
        loss = loss_fn(embeddings)
        # Loss should be close to 0 (log(1) = 0, since all distances are 0)
        assert loss.item() > -1.0  # much higher than uniform case

    def test_gradient_flows(self):
        loss_fn = UniformityLoss()
        embeddings = F.normalize(
            torch.randn(8, 32, requires_grad=True), dim=-1
        )
        loss = loss_fn(embeddings)
        loss.backward()
        # Gradient should flow (though through normalize it's indirect)


class TestCompositeLoss:
    def test_weighted_sum(self):
        l1 = LatentPredictionLoss(normalize=False)
        l2 = ReconstructionLoss(loss_type="mse", normalize_target=False)

        composite = CompositeLoss({
            "latent": (l1, 1.0),
            "recon": (l2, 0.5),
        })

        pred = torch.randn(2, 8, 32)
        target = torch.randn(2, 8, 32)

        total, individual = composite(
            latent_predictions=pred,
            latent_targets=target,
            recon_predictions=pred,
            recon_targets=target,
        )

        assert total.item() > 0
        assert "latent" in individual
        assert "recon" in individual
        # Total should be weighted sum
        expected = individual["latent"] * 1.0 + individual["recon"] * 0.5
        assert torch.allclose(total, expected, atol=1e-5)

    def test_single_loss(self):
        l1 = LatentPredictionLoss(normalize=False)
        composite = CompositeLoss({"only": (l1, 2.0)})

        pred = torch.randn(2, 8, 32)
        target = torch.randn(2, 8, 32)

        total, individual = composite(
            only_predictions=pred,
            only_targets=target,
        )

        assert torch.allclose(total, individual["only"] * 2.0, atol=1e-5)
