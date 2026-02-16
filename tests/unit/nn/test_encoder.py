"""Tests for the new encoder module.

Verifies full forward pass shape correctness for:
- MultiSensorTokenizer
- Encoder
- Decoder
"""

import pytest
import torch

from spacenit.arch.encoder import Decoder, Encoder, EncoderConfig, MultiSensorTokenizer
from spacenit.ingestion.sensors import SensorRegistry


def _get_test_sensor_labels():
    """Get a small set of sensor labels for testing."""
    all_labels = SensorRegistry.all_labels()
    # Pick sensors that exist in the registry
    candidates = ["sentinel2_l2a", "sentinel1", "srtm"]
    return [l for l in candidates if l in all_labels][:2] or all_labels[:2]


class TestMultiSensorTokenizer:
    def test_single_sensor(self):
        labels = _get_test_sensor_labels()[:1]
        config = EncoderConfig(
            embed_dim=64,
            depth=2,
            num_heads=4,
            base_patch_size=8,
            sensor_labels=labels,
        )
        tokenizer = MultiSensorTokenizer(config)

        spec = SensorRegistry.get(labels[0])
        C = spec.total_channels
        sensor_data = {labels[0]: torch.randn(2, C, 16, 16)}

        tokens, sensor_ids, spatial_ids, temporal_ids, layout = tokenizer(sensor_data)

        # 16/8 = 2 patches per dim, 2*2 = 4 patches + 1 type token = 5
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == 64
        assert tokens.shape[1] == 5  # 4 patches + 1 sensor type token
        assert sensor_ids.shape == tokens.shape[:2]
        assert spatial_ids.shape == (2, 5, 2)
        assert temporal_ids.shape == (2, 5)
        # Sensor-type token should have spatial_id (-1, -1)
        assert (spatial_ids[:, 0] == -1).all()
        assert len(layout) == 1
        assert layout[0][0] == labels[0]

    def test_multiple_sensors(self):
        labels = _get_test_sensor_labels()
        if len(labels) < 2:
            pytest.skip("Need at least 2 sensors for this test")

        config = EncoderConfig(
            embed_dim=64,
            depth=2,
            num_heads=4,
            base_patch_size=8,
            sensor_labels=labels,
        )
        tokenizer = MultiSensorTokenizer(config)

        sensor_data = {}
        for label in labels:
            spec = SensorRegistry.get(label)
            C = spec.total_channels
            sensor_data[label] = torch.randn(2, C, 16, 16)

        tokens, sensor_ids, spatial_ids, temporal_ids, layout = tokenizer(sensor_data)

        assert tokens.shape[0] == 2
        assert tokens.shape[2] == 64
        # Each sensor: 4 patches + 1 type token = 5, total = 5 * num_sensors
        expected_tokens = 5 * len(labels)
        assert tokens.shape[1] == expected_tokens
        assert len(layout) == len(labels)


class TestEncoder:
    def test_forward_shape(self):
        labels = _get_test_sensor_labels()[:1]
        config = EncoderConfig(
            embed_dim=64,
            depth=2,
            num_heads=4,
            base_patch_size=8,
            sensor_labels=labels,
        )
        encoder = Encoder(config)

        spec = SensorRegistry.get(labels[0])
        C = spec.total_channels
        sensor_data = {labels[0]: torch.randn(2, C, 16, 16)}

        encoded, sensor_ids, spatial_ids, temporal_ids, layout = encoder(sensor_data)

        assert encoded.shape[0] == 2
        assert encoded.shape[2] == 64
        assert sensor_ids.shape == encoded.shape[:2]

    def test_with_masking(self):
        labels = _get_test_sensor_labels()[:1]
        config = EncoderConfig(
            embed_dim=64,
            depth=2,
            num_heads=4,
            base_patch_size=8,
            sensor_labels=labels,
        )
        encoder = Encoder(config)

        spec = SensorRegistry.get(labels[0])
        C = spec.total_channels
        sensor_data = {labels[0]: torch.randn(2, C, 16, 16)}

        # Keep only 3 out of 5 tokens
        mask_indices = torch.tensor([[0, 1, 2], [1, 3, 4]])
        encoded, sensor_ids, spatial_ids, temporal_ids, layout = encoder(
            sensor_data, mask_indices=mask_indices
        )

        assert encoded.shape == (2, 3, 64)
        assert sensor_ids.shape == (2, 3)

    def test_gradient_flows(self):
        labels = _get_test_sensor_labels()[:1]
        config = EncoderConfig(
            embed_dim=32,
            depth=1,
            num_heads=4,
            base_patch_size=8,
            sensor_labels=labels,
        )
        encoder = Encoder(config)

        spec = SensorRegistry.get(labels[0])
        C = spec.total_channels
        x = torch.randn(1, C, 8, 8, requires_grad=True)
        sensor_data = {labels[0]: x}

        encoded, _, _, _, _ = encoder(sensor_data)
        encoded.sum().backward()
        assert x.grad is not None


class TestDecoder:
    def test_forward_shape(self):
        decoder = Decoder(embed_dim=64, depth=2, num_heads=4)
        all_tokens = torch.randn(2, 10, 64)
        # 5 visible, 5 predicted
        vmask = torch.tensor([[0, 0, 0, 0, 0, 2, 2, 2, 2, 2]] * 2)
        decoded = decoder(all_tokens, vmask)
        assert decoded.shape == (2, 10, 64)

    def test_zero_predictions(self):
        decoder = Decoder(embed_dim=64, depth=2, num_heads=4)
        all_tokens = torch.randn(2, 10, 64)
        # All visible, nothing to predict
        vmask = torch.zeros(2, 10, dtype=torch.long)
        decoded = decoder(all_tokens, vmask)
        assert decoded.shape == (2, 10, 64)

    def test_gradient_flows(self):
        decoder = Decoder(embed_dim=32, depth=1, num_heads=4)
        all_tokens = torch.randn(2, 8, 32, requires_grad=True)
        # 4 visible, 4 predicted
        vmask = torch.tensor([[0, 0, 0, 0, 2, 2, 2, 2]] * 2)
        decoded = decoder(all_tokens, vmask)
        decoded.sum().backward()
        assert all_tokens.grad is not None
