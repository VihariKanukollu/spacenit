"""Tests for the new attention module.

Verifies:
- RMSNorm output shape and normalization properties
- GQA produces correct output shapes with various head configurations
- SwiGLU FFN shapes
- TransformerBlock and TransformerStack forward passes
- RoPE is rotation-equivariant
"""

import math

import pytest
import torch

from spacenit.arch.attention import (
    GroupedQueryAttention,
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
    TransformerStack,
    _build_rope_freqs,
    apply_rotary_emb,
)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_scale(self):
        """After RMSNorm, the RMS of each vector should be close to 1."""
        norm = RMSNorm(128)
        x = torch.randn(4, 16, 128) * 5.0  # large scale
        out = norm(x)
        rms = out.float().pow(2).mean(dim=-1).sqrt()
        # Should be close to 1 (the weight is all ones initially)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_gradient_flows(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestRoPE:
    def test_freqs_shape(self):
        freqs = _build_rope_freqs(dim=64, max_len=128)
        assert freqs.shape == (128, 32)  # (max_len, dim//2)
        assert freqs.is_complex()

    def test_apply_rotary_emb_shape(self):
        x = torch.randn(2, 8, 16, 64)  # (B, heads, N, head_dim)
        freqs = _build_rope_freqs(64, 32)
        out = apply_rotary_emb(x, freqs[:16])
        assert out.shape == x.shape

    def test_rotation_equivariance(self):
        """Verify that RoPE is equivariant: rotating Q and K by the same
        amount should not change their dot product."""
        dim = 32
        freqs = _build_rope_freqs(dim, 64)

        q = torch.randn(1, 1, 8, dim)
        k = torch.randn(1, 1, 8, dim)

        # Dot product without rotation
        dot_orig = (q * k).sum(dim=-1)

        # Apply same rotation
        q_rot = apply_rotary_emb(q, freqs[:8])
        k_rot = apply_rotary_emb(k, freqs[:8])
        dot_rot = (q_rot * k_rot).sum(dim=-1)

        assert torch.allclose(dot_orig, dot_rot, atol=1e-4)


class TestGroupedQueryAttention:
    @pytest.mark.parametrize("num_kv_heads", [None, 4, 2, 1])
    def test_output_shape(self, num_kv_heads):
        dim = 64
        attn = GroupedQueryAttention(dim=dim, num_heads=8, num_kv_heads=num_kv_heads)
        x = torch.randn(2, 16, dim)
        out = attn(x)
        assert out.shape == (2, 16, dim)

    def test_cross_attention(self):
        dim = 64
        attn = GroupedQueryAttention(dim=dim, num_heads=8)
        q = torch.randn(2, 10, dim)
        kv = torch.randn(2, 20, dim)
        out = attn(q, context=kv)
        assert out.shape == (2, 10, dim)

    def test_with_rope(self):
        dim = 64
        attn = GroupedQueryAttention(dim=dim, num_heads=8)
        x = torch.randn(2, 16, dim)
        freqs = _build_rope_freqs(dim // 8, 32)
        out = attn(x, rope_freqs=freqs)
        assert out.shape == (2, 16, dim)

    def test_mqa_is_special_case(self):
        """Multi-query attention (num_kv_heads=1) should work."""
        dim = 64
        attn = GroupedQueryAttention(dim=dim, num_heads=8, num_kv_heads=1)
        x = torch.randn(2, 16, dim)
        out = attn(x)
        assert out.shape == (2, 16, dim)

    def test_gradient_flows(self):
        dim = 64
        attn = GroupedQueryAttention(dim=dim, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 8, dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestSwiGLUFFN:
    def test_output_shape(self):
        ffn = SwiGLUFFN(dim=64)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_hidden_dim_alignment(self):
        """Hidden dim should be rounded to multiple of 8."""
        ffn = SwiGLUFFN(dim=100, expansion=2.5)
        # 100 * 2.5 = 250, rounded to 256 (nearest multiple of 8)
        assert ffn.gate_proj.out_features % 8 == 0

    def test_gradient_flows(self):
        ffn = SwiGLUFFN(dim=32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestTransformerBlock:
    def test_self_attention_only(self):
        block = TransformerBlock(dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_with_cross_attention(self):
        block = TransformerBlock(dim=64, num_heads=4, cross_attention=True)
        x = torch.randn(2, 10, 64)
        ctx = torch.randn(2, 20, 64)
        out = block(x, context=ctx)
        assert out.shape == (2, 10, 64)

    def test_post_norm_residual(self):
        """Verify post-norm: output should differ from input even with
        identity-like initialization."""
        block = TransformerBlock(dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = block(x)
        # Output should not be identical to input
        assert not torch.allclose(out, x, atol=1e-6)


class TestTransformerStack:
    def test_output_shape(self):
        stack = TransformerStack(dim=64, depth=4, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = stack(x)
        assert out.shape == (2, 10, 64)

    def test_with_cross_attention(self):
        stack = TransformerStack(
            dim=64, depth=2, num_heads=4, cross_attention=True
        )
        x = torch.randn(2, 10, 64)
        ctx = torch.randn(2, 20, 64)
        out = stack(x, context=ctx)
        assert out.shape == (2, 10, 64)

    def test_gradient_flows_through_all_layers(self):
        stack = TransformerStack(dim=32, depth=3, num_heads=4)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = stack(x)
        out.sum().backward()
        assert x.grad is not None
        # All layer parameters should have gradients
        for layer in stack.layers:
            for p in layer.parameters():
                assert p.grad is not None
