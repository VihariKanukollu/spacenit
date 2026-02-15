"""Tests for the new masking module.

Verifies:
- Mask ratios are approximately correct
- Spatial masking produces spatially consistent blocks
- Temporal masking masks entire timesteps
- Composition rules (union, intersection, cascade)
- Strategy dataclasses and build_masking
"""

import pytest
import torch

from spacenit.pipeline.masking import (
    CompositeMasking,
    CrossSensorMasking,
    RandomMasking,
    RangeMasking,
    ScheduledMasking,
    SpatialMasking,
    SpectralMasking,
    TemporalMasking,
    apply_strategy,
    build_masking,
    compose_masks,
    create_mask,
)
from spacenit.structures import TokenVisibility

VIS = TokenVisibility.VISIBLE_ENCODER.value
TGT = TokenVisibility.TARGET_ONLY.value
PRED = TokenVisibility.PREDICTED.value
ABS = TokenVisibility.ABSENT.value


class TestCreateMask:
    def test_random_mask_shape(self):
        mask = create_mask(100, encode_ratio=0.25, structure="random")
        assert mask.shape == (100,)
        assert mask.dtype == torch.long

    def test_random_mask_ratio(self):
        """Visible ratio should be approximately correct."""
        mask = create_mask(1000, encode_ratio=0.3, structure="random")
        visible = (mask == VIS).sum().item()
        # Allow 10% tolerance
        assert abs(visible / 1000 - 0.3) < 0.1

    def test_random_mask_with_decode(self):
        mask = create_mask(
            100, encode_ratio=0.25, decode_ratio=0.25, structure="random"
        )
        n_vis = (mask == VIS).sum().item()
        n_tgt = (mask == TGT).sum().item()
        n_pred = (mask == PRED).sum().item()
        assert n_vis > 0
        assert n_tgt > 0
        assert n_pred > 0
        assert n_vis + n_tgt + n_pred == 100

    def test_at_least_one_visible(self):
        """Even with very low ratio, at least one token should be visible."""
        mask = create_mask(10, encode_ratio=0.01, structure="random")
        assert (mask == VIS).sum().item() >= 1

    def test_spatial_mask(self):
        mask = create_mask(
            64,
            encode_ratio=0.5,
            structure="spatial",
            spatial_shape=(8, 8),
        )
        assert mask.shape == (64,)
        assert (mask == VIS).sum().item() > 0

    def test_temporal_mask(self):
        mask = create_mask(
            120,
            encode_ratio=0.5,
            structure="temporal",
            temporal_length=12,
        )
        assert mask.shape == (120,)
        # Check that entire timesteps are masked consistently
        tokens_per_step = 10
        for t in range(12):
            step_mask = mask[t * tokens_per_step : (t + 1) * tokens_per_step]
            # All tokens in a timestep should have the same visibility
            assert (step_mask == step_mask[0]).all()

    def test_spectral_mask(self):
        mask = create_mask(
            30,
            encode_ratio=0.5,
            structure="spectral",
            num_groups=3,
        )
        assert mask.shape == (30,)
        # Each group of 10 tokens should be uniform
        for g in range(3):
            group_mask = mask[g * 10 : (g + 1) * 10]
            assert (group_mask == group_mask[0]).all()

    def test_unknown_structure_raises(self):
        with pytest.raises(ValueError, match="Unknown masking structure"):
            create_mask(10, 0.5, structure="nonexistent")


class TestComposeMasks:
    def test_union(self):
        """Union: most permissive wins."""
        m1 = torch.tensor([VIS, PRED, PRED, VIS])
        m2 = torch.tensor([PRED, VIS, PRED, PRED])
        result = compose_masks([m1, m2], rule="union")
        assert result[0].item() == VIS  # VIS wins
        assert result[1].item() == VIS  # VIS wins
        assert result[2].item() == PRED  # both PRED
        assert result[3].item() == VIS  # VIS wins

    def test_intersection(self):
        """Intersection: least permissive wins."""
        m1 = torch.tensor([VIS, PRED, VIS, TGT])
        m2 = torch.tensor([VIS, VIS, PRED, PRED])
        result = compose_masks([m1, m2], rule="intersection")
        assert result[0].item() == VIS
        assert result[1].item() == PRED  # PRED > VIS
        assert result[2].item() == PRED
        assert result[3].item() == PRED  # PRED > TGT

    def test_cascade(self):
        """Cascade: sequential restriction."""
        m1 = torch.tensor([VIS, VIS, PRED])
        m2 = torch.tensor([PRED, VIS, VIS])
        result = compose_masks([m1, m2], rule="cascade")
        assert result[0].item() == PRED  # restricted by m2
        assert result[1].item() == VIS  # both VIS
        assert result[2].item() == PRED  # already PRED in m1

    def test_absent_preserved(self):
        """ABSENT should always stay ABSENT regardless of rule."""
        m1 = torch.tensor([ABS, VIS, PRED])
        m2 = torch.tensor([VIS, ABS, VIS])
        for rule in ["union", "intersection", "cascade"]:
            result = compose_masks([m1, m2], rule=rule)
            assert result[0].item() == ABS
            assert result[1].item() == ABS

    def test_single_mask(self):
        m = torch.tensor([VIS, PRED, TGT])
        result = compose_masks([m], rule="union")
        assert torch.equal(result, m)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_masks([], rule="union")


class TestApplyStrategy:
    def test_random_masking(self):
        strategy = RandomMasking(encode_ratio=0.5)
        mask = apply_strategy(strategy, 100)
        assert mask.shape == (100,)
        assert (mask == VIS).sum().item() > 0

    def test_range_masking(self):
        strategy = RangeMasking(min_encode=0.2, max_encode=0.8)
        mask = apply_strategy(strategy, 100)
        visible_ratio = (mask == VIS).sum().item() / 100
        assert 0.1 <= visible_ratio <= 0.9

    def test_scheduled_masking(self):
        strategy = ScheduledMasking(
            initial_encode_ratio=0.1,
            final_encode_ratio=0.9,
            warmup_steps=100,
        )
        # At step 0, should be close to initial
        mask_early = apply_strategy(strategy, 1000, step=0)
        # At step 100, should be close to final
        mask_late = apply_strategy(strategy, 1000, step=100)

        ratio_early = (mask_early == VIS).sum().item() / 1000
        ratio_late = (mask_late == VIS).sum().item() / 1000
        assert ratio_late > ratio_early

    def test_composite_masking(self):
        strategy = CompositeMasking(
            strategies=[
                RandomMasking(encode_ratio=0.3),
                SpatialMasking(encode_ratio=0.7),
            ],
            weights=[0.5, 0.5],
        )
        mask = apply_strategy(
            strategy, 64, spatial_shape=(8, 8)
        )
        assert mask.shape == (64,)

    def test_cross_sensor_masking_min_constraint(self):
        strategy = CrossSensorMasking(
            base_strategy="random",
            base_encode_ratio=0.01,  # very low
            min_encoded=0.2,
        )
        mask = apply_strategy(strategy, 100)
        ratio = (mask == VIS).sum().item() / 100
        assert ratio >= 0.15  # should be bumped up to ~0.2


class TestBuildMasking:
    def test_build_random(self):
        strategy = build_masking({"type": "random", "encode_ratio": 0.4})
        assert isinstance(strategy, RandomMasking)
        assert strategy.encode_ratio == 0.4

    def test_build_spatial(self):
        strategy = build_masking({"type": "spatial", "encode_ratio": 0.5})
        assert isinstance(strategy, SpatialMasking)

    def test_build_composite(self):
        strategy = build_masking({
            "type": "composite",
            "strategies": [
                {"type": "random", "encode_ratio": 0.3},
                {"type": "temporal", "encode_ratio": 0.5},
            ],
            "weights": [0.6, 0.4],
        })
        assert isinstance(strategy, CompositeMasking)
        assert len(strategy.strategies) == 2

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown masking strategy"):
            build_masking({"type": "nonexistent"})
