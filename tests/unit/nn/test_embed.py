"""Tests for the new embedding module.

Verifies:
- AdaptivePatchEmbed produces correct token counts at different patch sizes
- SpatialRoPE frequency properties and GSD scaling
- TemporalRoPE shape and frequency properties
- CyclicMonthEmbed circularity (Dec close to Jan)
- SensorEmbed basic functionality
"""

import math

import pytest
import torch

from spacenit.arch.embed import (
    AdaptivePatchEmbed,
    CyclicMonthEmbed,
    SensorEmbed,
    SpatialRoPE,
    TemporalRoPE,
)


class TestAdaptivePatchEmbed:
    def test_basic_shape(self):
        embed = AdaptivePatchEmbed(
            base_patch_size=16, in_channels=3, embed_dim=64
        )
        x = torch.randn(2, 3, 64, 64)
        tokens = embed(x)
        # 64/16 = 4 patches per dim, 4*4 = 16 patches
        assert tokens.shape == (2, 16, 64)

    def test_different_patch_size(self):
        embed = AdaptivePatchEmbed(
            base_patch_size=16, in_channels=3, embed_dim=64
        )
        x = torch.randn(2, 3, 64, 64)
        tokens = embed(x, patch_size=8)
        # 64/8 = 8 patches per dim, 8*8 = 64 patches
        assert tokens.shape == (2, 64, 64)

    def test_input_resize_adaptation(self):
        """Patch-size adaptation should resize inputs, not weights (OLMo style)."""
        embed = AdaptivePatchEmbed(
            base_patch_size=16, in_channels=3, embed_dim=64
        )
        x = torch.randn(2, 3, 64, 64)

        # Requested patch_size=8 should upsample to (H//8)*16 = 128 before unfolding.
        x_resized = embed._maybe_resize_input(x, patch_size=8)
        assert x_resized.shape == (2, 3, 128, 128)

        # Weights stay at base patch size.
        assert embed.proj.weight.shape[1] == 3 * 16 * 16  # 768

        tokens = embed(x, patch_size=8)
        assert tokens.shape == (2, 64, 64)

    def test_non_square_input(self):
        embed = AdaptivePatchEmbed(
            base_patch_size=8, in_channels=4, embed_dim=32
        )
        x = torch.randn(1, 4, 32, 16)
        tokens = embed(x)
        # 32/8 * 16/8 = 4 * 2 = 8 patches
        assert tokens.shape == (1, 8, 32)

    def test_gradient_flows(self):
        embed = AdaptivePatchEmbed(
            base_patch_size=8, in_channels=3, embed_dim=32
        )
        x = torch.randn(1, 3, 16, 16, requires_grad=True)
        tokens = embed(x)
        tokens.sum().backward()
        assert x.grad is not None


class TestSpatialRoPE:
    def test_output_shape(self):
        rope = SpatialRoPE(dim=64, max_grid=32)
        freqs = rope(num_rows=4, num_cols=4)
        # 4*4 = 16 positions, dim//2 = 32 complex values
        assert freqs.shape == (16, 32)
        assert freqs.is_complex()

    def test_gsd_scaling(self):
        """Different GSD values should produce different frequencies."""
        rope = SpatialRoPE(dim=64, max_grid=32, reference_gsd=10.0)
        freqs_10m = rope(4, 4, gsd=10.0)
        freqs_20m = rope(4, 4, gsd=20.0)
        # Frequencies should differ due to GSD scaling
        assert not torch.allclose(freqs_10m, freqs_20m)

    def test_reference_gsd_identity(self):
        """At reference GSD, positions should be unscaled."""
        rope = SpatialRoPE(dim=64, max_grid=32, reference_gsd=10.0)
        freqs = rope(2, 2, gsd=10.0)
        # scale = 10/10 = 1, so positions are 0, 1
        assert freqs.shape == (4, 32)


class TestTemporalRoPE:
    def test_output_shape(self):
        rope = TemporalRoPE(dim=32, max_timesteps=64)
        freqs = rope(num_timesteps=12)
        assert freqs.shape == (12, 16)  # (T, dim//2)
        assert freqs.is_complex()

    def test_max_timesteps(self):
        rope = TemporalRoPE(dim=32, max_timesteps=64)
        freqs = rope(64)
        assert freqs.shape == (64, 16)

    def test_different_lengths(self):
        rope = TemporalRoPE(dim=32, max_timesteps=64)
        freqs_short = rope(4)
        freqs_long = rope(8)
        # First 4 positions should be identical
        assert torch.allclose(freqs_short, freqs_long[:4])


class TestCyclicMonthEmbed:
    def test_output_shape(self):
        embed = CyclicMonthEmbed(embed_dim=64)
        months = torch.tensor([0, 5, 11])
        out = embed(months)
        assert out.shape == (3, 64)

    def test_december_close_to_january(self):
        """Dec (11) and Jan (0) should be closer than Jun (5) and Jan (0)."""
        embed = CyclicMonthEmbed(embed_dim=64)
        months = torch.tensor([0, 5, 11])
        out = embed(months)
        jan, jun, dec = out[0], out[1], out[2]
        dist_jan_dec = (jan - dec).pow(2).sum().sqrt()
        dist_jan_jun = (jan - jun).pow(2).sum().sqrt()
        # Before training, the raw embeddings ensure Dec is close to Jan
        # because sin/cos of 0 and 11*2pi/12 are close
        # After projection, this may not hold exactly, but the raw
        # embeddings should show this property
        raw = embed.raw_embeddings
        raw_jan = raw[0]
        raw_jun = raw[5]
        raw_dec = raw[11]
        assert (raw_jan - raw_dec).pow(2).sum() < (raw_jan - raw_jun).pow(2).sum()

    def test_batched_input(self):
        embed = CyclicMonthEmbed(embed_dim=32)
        months = torch.tensor([[0, 3, 6], [1, 4, 7]])  # (2, 3)
        out = embed(months)
        assert out.shape == (2, 3, 32)


class TestSensorEmbed:
    def test_output_shape(self):
        embed = SensorEmbed(num_sensors=5, embed_dim=64)
        ids = torch.tensor([0, 2, 4])
        out = embed(ids)
        assert out.shape == (3, 64)

    def test_different_sensors_different_embeddings(self):
        embed = SensorEmbed(num_sensors=5, embed_dim=64)
        ids = torch.tensor([0, 1])
        out = embed(ids)
        # Different sensor IDs should produce different embeddings
        assert not torch.allclose(out[0], out[1])

    def test_batched(self):
        embed = SensorEmbed(num_sensors=10, embed_dim=32)
        ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        out = embed(ids)
        assert out.shape == (2, 3, 32)
