"""Tests for checkpoint loading that run with both minimal and full dependencies.

This directory (tests_minimal_deps/) contains tests that are run twice in CI:
1. With minimal deps only (no olmo-core) -> tests _FallbackConfig path
2. With full deps (with olmo-core) -> tests olmo-core Config path

This verifies model loading works regardless of whether olmo-core is installed.

To run locally:
    # Minimal deps (no olmo-core)
    uv run --group dev pytest -v tests_minimal_deps/

    # Full deps (with olmo-core)
    uv run pytest -v tests_minimal_deps/
"""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from spacenit.settings import (
    HAS_OLMO_CORE,
    Config,
    _FallbackConfig,
)
from spacenit.checkpoint_loader import (
    CONFIG_FILE,
    WEIGHTS_FILE,
    PretrainedModelID,
    load_from_directory,
    load_pretrained,
)
from spacenit.arch.adaptive_vision_encoder import VisionEncoderConfig
from spacenit.arch.adaptive_vision_encoder import LatentPredictorConfig
from spacenit.arch.latent_masked_prediction import LatentMaskedPredictorConfig

# =============================================================================
# Test Helpers
# =============================================================================


def _create_minimal_model_config() -> dict:
    """Create a minimal model config that can be built."""
    encoder_config = VisionEncoderConfig(
        supported_sensor_labels=["sentinel2_l2a", "sentinel1"],
        embed_dim=16,
        tile_patch_size=8,
        head_count=2,
        depth=2,
        ffn_expansion=4.0,
        stochastic_depth_rate=0.1,
        max_sequence_length=12,
    )
    decoder_config = LatentPredictorConfig(
        encoder_embed_dim=16,
        decoder_embed_dim=16,
        depth=2,
        ffn_expansion=4.0,
        head_count=8,
        max_sequence_length=12,
        supported_sensor_labels=["sentinel2_l2a", "sentinel1"],
    )
    model_config = LatentMaskedPredictorConfig(
        encoder_cfg=encoder_config,
        decoder_cfg=decoder_config,
    )
    # Return the structure expected by checkpoint_loader: {"model": <config_dict>}
    return {"model": model_config.as_config_dict()}


def _create_minimal_state_dict() -> dict[str, torch.Tensor]:
    """Create a minimal state dict for testing."""
    return {"dummy_weight": torch.randn(2, 2)}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def encoder_config_dict() -> dict:
    """Create a minimal VisionEncoderConfig as a dict."""
    return {
        "_CLASS_": "spacenit.arch.adaptive_vision_encoder.VisionEncoderConfig",
        "supported_sensor_labels": ["sentinel2_l2a"],
        "embed_dim": 64,
        "head_count": 2,
        "depth": 2,
        "ffn_expansion": 4.0,
        "max_sequence_length": 64,
    }


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write config.json
        config_path = tmpdir_path / CONFIG_FILE
        with open(config_path, "w") as f:
            json.dump(_create_minimal_model_config(), f)

        # Write weights.pth
        weights_path = tmpdir_path / WEIGHTS_FILE
        torch.save(_create_minimal_state_dict(), weights_path)

        yield tmpdir_path


# =============================================================================
# Config Export Tests
# =============================================================================


class TestConfigExport:
    """Tests for the exported Config class behavior."""

    def test_config_is_correct_type(self) -> None:
        """Test that Config is the correct type based on olmo-core availability."""
        if not HAS_OLMO_CORE:
            assert Config is _FallbackConfig
        else:
            from olmo_core.config import Config as OlmoCoreConfig

            assert Config is OlmoCoreConfig

    def test_config_has_from_dict_method(self) -> None:
        """Test that the exported Config has from_dict for model loading."""
        assert hasattr(Config, "from_dict")


# =============================================================================
# VisionEncoderConfig Loading Tests
# =============================================================================


class TestVisionEncoderConfigLoading:
    """Tests for loading VisionEncoderConfig using the exported Config."""

    def test_load_encoder_config_from_dict(self, encoder_config_dict: dict) -> None:
        """Test loading VisionEncoderConfig from a dict."""
        config = VisionEncoderConfig.from_dict(encoder_config_dict)

        assert isinstance(config, VisionEncoderConfig)
        assert config.embed_dim == 64
        assert config.head_count == 2
        assert config.depth == 2
        assert config.supported_sensor_labels == ["sentinel2_l2a"]

    def test_build_encoder_from_config(self, encoder_config_dict: dict) -> None:
        """Test building an Encoder from the loaded config."""
        config = VisionEncoderConfig.from_dict(encoder_config_dict)
        encoder = config.build()

        assert encoder is not None
        assert isinstance(encoder, torch.nn.Module)
        assert len(list(encoder.parameters())) > 0


# =============================================================================
# Model Loading Tests
# =============================================================================


class TestLoadFromDirectory:
    """Tests for load_from_directory."""

    def test_load_with_pathlib_path(self, temp_model_dir: Path) -> None:
        """Test loading model using pathlib.Path."""
        model = load_from_directory(temp_model_dir, load_weights=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_load_with_string(self, temp_model_dir: Path) -> None:
        """Test loading model using string path."""
        model = load_from_directory(str(temp_model_dir), load_weights=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_load_without_weights(self, temp_model_dir: Path) -> None:
        """Test loading model without weights (random init)."""
        model = load_from_directory(temp_model_dir, load_weights=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)


class TestLoadPretrained:
    """Tests for load_pretrained with mocked HuggingFace downloads."""

    def test_load_from_model_id_without_weights(self, temp_model_dir: Path) -> None:
        """Test loading model from PretrainedModelID without weights."""

        def mock_hf_hub_download(repo_id: str, filename: str) -> str:
            return str(temp_model_dir / filename)

        with patch(
            "spacenit.checkpoint_loader.hf_hub_download",
            side_effect=mock_hf_hub_download,
        ):
            model = load_pretrained(
                PretrainedModelID.SPACENIT_V1_NANO, load_weights=False
            )
            assert model is not None
            assert isinstance(model, torch.nn.Module)

    def test_model_id_hf_repo(self) -> None:
        """Test that PretrainedModelID.hf_repo() returns correct format."""
        assert (
            PretrainedModelID.SPACENIT_V1_NANO.hf_repo()
            == "spacenit/SpaceNit-v1-Nano"
        )
        assert (
            PretrainedModelID.SPACENIT_V1_TINY.hf_repo()
            == "spacenit/SpaceNit-v1-Tiny"
        )
        assert (
            PretrainedModelID.SPACENIT_V1_BASE.hf_repo()
            == "spacenit/SpaceNit-v1-Base"
        )
        assert (
            PretrainedModelID.SPACENIT_V1_LARGE.hf_repo()
            == "spacenit/SpaceNit-v1-Large"
        )
