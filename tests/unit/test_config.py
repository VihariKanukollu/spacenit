"""Unit tests for the centralized config module."""

import pytest

from spacenit.settings import (
    HAS_OLMO_CORE,
    Config,
    _FallbackConfig,
    require_olmo_core,
)


class TestOlmoCoreAvailability:
    """Tests for olmo-core availability detection."""

    def test_has_olmo_core_flag_is_bool(self) -> None:
        """Test that HAS_OLMO_CORE is a boolean."""
        assert isinstance(HAS_OLMO_CORE, bool)

    def test_config_is_type(self) -> None:
        """Test that Config is a type/class."""
        assert isinstance(Config, type)


class TestRequireOlmoCore:
    """Tests for the require_olmo_core guard function."""

    def test_require_olmo_core_does_not_raise_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that require_olmo_core doesn't raise when olmo-core is available."""
        monkeypatch.setattr("spacenit.settings.HAS_OLMO_CORE", True)
        # Should not raise
        require_olmo_core("Test operation")

    def test_require_olmo_core_raises_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that require_olmo_core raises ImportError when olmo-core unavailable."""
        monkeypatch.setattr("spacenit.settings.HAS_OLMO_CORE", False)
        with pytest.raises(ImportError, match="needs olmo-core"):
            require_olmo_core("Training")

    def test_require_olmo_core_includes_operation_in_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the operation name is included in the error message."""
        monkeypatch.setattr("spacenit.settings.HAS_OLMO_CORE", False)
        with pytest.raises(ImportError, match="My custom operation"):
            require_olmo_core("My custom operation")


class TestFallbackConfig:
    """Tests for the fallback config implementation."""

    def test_fallback_config_from_dict_simple(self) -> None:
        """Test that _FallbackConfig.from_dict works with simple data."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_FallbackConfig):
            value: int
            name: str

        data = {"value": 42, "name": "test"}
        config = SimpleConfig.from_dict(data)

        assert config.value == 42
        assert config.name == "test"

    def test_fallback_config_as_dict(self) -> None:
        """Test that _FallbackConfig.as_dict works correctly."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_FallbackConfig):
            value: int
            name: str

        config = SimpleConfig(value=42, name="test")
        result = config.as_dict()

        assert result == {"value": 42, "name": "test"}

    def test_fallback_config_as_config_dict(self) -> None:
        """Test that as_config_dict includes class name."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_FallbackConfig):
            value: int

        config = SimpleConfig(value=42)
        result = config.as_config_dict()

        assert "_CLASS_" in result
        assert "SimpleConfig" in result["_CLASS_"]
        assert result["value"] == 42

    def test_fallback_config_build_not_implemented(self) -> None:
        """Test that build() raises NotImplementedError by default."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_FallbackConfig):
            value: int

        config = SimpleConfig(value=42)
        with pytest.raises(NotImplementedError, match="must implement build"):
            config.build()

    def test_fallback_config_import_class(self) -> None:
        """Test that _import_class can resolve fully-qualified class names."""
        # Test resolving a known class
        resolved = _FallbackConfig._import_class(
            "spacenit.structures.TokenVisibility"
        )
        assert resolved is not None

        from spacenit.structures import TokenVisibility

        assert resolved is TokenVisibility

    def test_fallback_config_import_class_raises_for_invalid(self) -> None:
        """Test that _import_class raises for invalid class names."""
        # No dot in name
        with pytest.raises(ValueError, match="dotted path"):
            _FallbackConfig._import_class("InvalidName")

        # Non-existent module
        with pytest.raises(ModuleNotFoundError):
            _FallbackConfig._import_class("nonexistent.module.Class")


class TestConfigCompatibility:
    """Tests for compatibility between olmo-core Config and fallback Config."""

    @pytest.mark.skipif(not HAS_OLMO_CORE, reason="olmo-core not installed")
    def test_config_is_olmo_core_when_available(self) -> None:
        """Test that Config is olmo-core's Config when available."""
        from olmo_core.config import Config as OlmoCoreConfig

        # At runtime (not TYPE_CHECKING), Config should be olmo-core's
        assert Config is OlmoCoreConfig

    def test_fallback_config_has_required_methods(self) -> None:
        """Test that _FallbackConfig has all required methods."""
        assert hasattr(_FallbackConfig, "from_dict")
        assert hasattr(_FallbackConfig, "as_dict")
        assert hasattr(_FallbackConfig, "as_config_dict")
        assert hasattr(_FallbackConfig, "build")
        assert hasattr(_FallbackConfig, "validate")
