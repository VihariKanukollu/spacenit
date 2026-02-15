"""Tests for band tokenization configuration."""

import pytest

from spacenit.ingestion.sensors import SENTINEL2_L2A, SENTINEL1, SensorRegistry
from spacenit.arch.band_tokenization import (
    SensorTokenLayout,
    TokenizationConfig,
)


class TestTokenizationConfig:
    """Tests for configurable band tokenization."""

    def test_default_config_matches_sensor_spec(self) -> None:
        """No custom layouts should return same indices as SensorSpec."""
        config = TokenizationConfig()

        # Sentinel-2 L2A has 3 spectral groups by default
        default_indices = SENTINEL2_L2A.group_indices()
        config_indices = config.group_indices_for(SENTINEL2_L2A.label)

        assert config_indices == default_indices
        assert config.group_count_for(SENTINEL2_L2A.label) == 3

    def test_custom_single_channel_tokenization(self) -> None:
        """Each channel as its own token."""
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL1.label: SensorTokenLayout(
                    channel_groups=[
                        ["vv"],
                        ["vh"],
                    ]
                )
            }
        )

        indices = config.group_indices_for(SENTINEL1.label)
        assert indices == [[0], [1]]
        assert config.group_count_for(SENTINEL1.label) == 2

    def test_custom_grouped_tokenization(self) -> None:
        """Custom grouping of channels."""
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL2_L2A.label: SensorTokenLayout(
                    channel_groups=[
                        # RGB-like group
                        ["B02", "B03", "B04"],
                        # NIR group
                        ["B08", "B8A"],
                        # SWIR group
                        ["B11", "B12"],
                    ]
                )
            }
        )

        indices = config.group_indices_for(SENTINEL2_L2A.label)
        # B02=0, B03=1, B04=2, B08=3, B8A=7, B11=8, B12=9
        assert indices == [[0, 1, 2], [3, 7], [8, 9]]
        assert config.group_count_for(SENTINEL2_L2A.label) == 3

    def test_channel_order_preserved_in_group(self) -> None:
        """Channels within a group maintain requested order, not data order."""
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL2_L2A.label: SensorTokenLayout(
                    channel_groups=[
                        # Request B04 before B02 (reversed from data order)
                        ["B04", "B02"],
                    ]
                )
            }
        )

        indices = config.group_indices_for(SENTINEL2_L2A.label)
        # B04=2, B02=0 (order from config, not data)
        assert indices == [[2, 0]]

    def test_invalid_channel_name_raises(self) -> None:
        """Unknown channel name should raise ValueError."""
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL2_L2A.label: SensorTokenLayout(
                    channel_groups=[
                        ["B02", "INVALID_CHANNEL"],
                    ]
                )
            }
        )

        with pytest.raises(ValueError, match="Channel 'INVALID_CHANNEL' not found"):
            config.group_indices_for(SENTINEL2_L2A.label)

    def test_sensor_without_custom_layout_uses_default(self) -> None:
        """Sensors not in custom_layouts use default spectral groups."""
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL1.label: SensorTokenLayout(
                    channel_groups=[["vv"]]
                )
            }
        )

        # sentinel2_l2a not overridden, should use default
        s2_indices = config.group_indices_for(SENTINEL2_L2A.label)
        assert s2_indices == SENTINEL2_L2A.group_indices()

        # sentinel1 is overridden
        s1_indices = config.group_indices_for(SENTINEL1.label)
        assert s1_indices == [[0]]

    def test_check_consistency_catches_invalid_channel_name(self) -> None:
        """Consistency check should catch invalid channel names."""
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL2_L2A.label: SensorTokenLayout(
                    channel_groups=[
                        ["B02", "INVALID_CHANNEL"],
                    ]
                )
            }
        )

        with pytest.raises(ValueError, match="Channel 'INVALID_CHANNEL' not found"):
            config.check_consistency()

    def test_check_consistency_catches_invalid_sensor_label(self) -> None:
        """Consistency check should catch invalid sensor labels."""
        config = TokenizationConfig(
            custom_layouts={
                "INVALID_SENSOR": SensorTokenLayout(
                    channel_groups=[["B02"]],
                )
            }
        )

        with pytest.raises(ValueError, match="Unknown sensor label in custom_layouts"):
            config.check_consistency()

    def test_channels_per_group_for(self) -> None:
        """Test getting number of channels per group."""
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL2_L2A.label: SensorTokenLayout(
                    channel_groups=[
                        ["B02", "B03", "B04"],
                        ["B08"],
                    ]
                )
            }
        )

        channels_per_group = config.channels_per_group_for(SENTINEL2_L2A.label)
        assert channels_per_group == [3, 1]

        # Default for sentinel1
        s1_channels = config.channels_per_group_for(SENTINEL1.label)
        assert s1_channels == [2]  # sentinel1 has 2 channels in 1 group

    def test_sensor_token_layout_group_count(self) -> None:
        """SensorTokenLayout.group_count property."""
        layout = SensorTokenLayout(
            channel_groups=[
                ["B02", "B03"],
                ["B04"],
                ["B08"],
            ]
        )
        assert layout.group_count == 3

    def test_full_sentinel2_per_channel_tokenization(self) -> None:
        """Test making each Sentinel-2 channel its own token."""
        s2_channels = SENTINEL2_L2A.all_channel_names
        config = TokenizationConfig(
            custom_layouts={
                SENTINEL2_L2A.label: SensorTokenLayout(
                    channel_groups=[[ch] for ch in s2_channels]
                )
            }
        )

        indices = config.group_indices_for(SENTINEL2_L2A.label)
        # Each channel should have its own index
        assert len(indices) == len(s2_channels)
        for i, idx_list in enumerate(indices):
            assert idx_list == [i]

        assert config.group_count_for(SENTINEL2_L2A.label) == len(s2_channels)


class TestTokenizationWithMasking:
    """Tests for tokenization config interaction with the new masking module."""

    def test_masking_strategy_builds_from_config(self) -> None:
        """Masking strategies should build correctly from config dicts."""
        from spacenit.pipeline.masking import RandomMasking, build_masking

        strategy = build_masking({
            "type": "random",
            "encode_ratio": 0.5,
            "decode_ratio": 0.25,
        })
        assert isinstance(strategy, RandomMasking)
        assert strategy.encode_ratio == 0.5
        assert strategy.decode_ratio == 0.25

    def test_masking_respects_group_count(self) -> None:
        """Spectral masking should respect the number of spectral groups."""
        from spacenit.pipeline.masking import create_mask

        # 3 spectral groups, 10 tokens per group = 30 total
        mask = create_mask(
            30,
            encode_ratio=0.5,
            structure="spectral",
            num_groups=3,
        )
        assert mask.shape == (30,)
        # Each group of 10 should be uniformly masked
        for g in range(3):
            group_mask = mask[g * 10 : (g + 1) * 10]
            assert (group_mask == group_mask[0]).all()

    def test_composite_masking_with_spectral(self) -> None:
        """Composite masking should work with spectral sub-strategies."""
        from spacenit.pipeline.masking import CompositeMasking, SpectralMasking, build_masking

        strategy = build_masking({
            "type": "composite",
            "strategies": [
                {"type": "spectral", "encode_ratio": 0.5},
                {"type": "random", "encode_ratio": 0.3},
            ],
            "weights": [0.5, 0.5],
        })
        assert isinstance(strategy, CompositeMasking)
        assert len(strategy.strategies) == 2
