"""Unified configuration layer with optional olmo-core backend.

When olmo-core is installed the rich ``Config`` from that library is re-exported
directly.  Otherwise a lightweight dataclass-based ``_FallbackConfig`` is used so
that model configs can still be loaded from JSON for inference without the heavy
training stack.

Typical usage::

    from spacenit.settings import Config, HAS_OLMO_CORE, require_olmo_core

    @dataclass
    class MyModelConfig(Config):
        hidden_dim: int = 256
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from typing import Any, TypeVar

# ---------------------------------------------------------------------------
# Detect whether the full olmo-core library is available
# ---------------------------------------------------------------------------
try:
    from olmo_core.config import Config as _OlmoCoreConfig

    HAS_OLMO_CORE = True
except ImportError:
    HAS_OLMO_CORE = False
    _OlmoCoreConfig = None  # type: ignore[assignment, misc]


def require_olmo_core(feature_label: str = "This feature") -> None:
    """Raise ``ImportError`` when olmo-core is missing.

    Call this at the top of any module that needs the full training stack so
    users get a clear message instead of a cryptic traceback.
    """
    if not HAS_OLMO_CORE:
        raise ImportError(
            f"{feature_label} needs olmo-core.  "
            "Install it with:  pip install spacenit[training]"
        )


# ---------------------------------------------------------------------------
# Lightweight fallback config (inference-only)
# ---------------------------------------------------------------------------
_T = TypeVar("_T", bound="_FallbackConfig")

CLASS_KEY = "_CLASS_"


@dataclass
class _FallbackConfig:
    """Bare-bones config that can deserialise nested JSON into dataclasses.

    It deliberately omits OmegaConf merging, CLI overrides, and YAML I/O --
    those require olmo-core.
    """

    @staticmethod
    def _import_class(fqn: str) -> type:
        """Import a class given its fully-qualified dotted name."""
        if "." not in fqn:
            raise ValueError(f"Expected a dotted path, got '{fqn}'")
        parts = fqn.rsplit(".", 1)
        mod = import_module(parts[0])
        return getattr(mod, parts[1])

    @classmethod
    def _materialise(cls, raw: Any) -> Any:
        """Walk *raw* recursively, turning dicts with ``_CLASS_`` into instances."""
        if isinstance(raw, dict):
            fqn = raw.get(CLASS_KEY)
            materialised = {
                k: cls._materialise(v) for k, v in raw.items() if k != CLASS_KEY
            }
            if fqn is not None:
                target_cls = cls._import_class(fqn)
                if not is_dataclass(target_cls):
                    raise TypeError(f"'{fqn}' is not a dataclass")
                accepted = {f.name for f in fields(target_cls)}
                kwargs = {k: v for k, v in materialised.items() if k in accepted}
                try:
                    return target_cls(**kwargs)
                except TypeError as exc:
                    raise TypeError(f"Cannot instantiate {fqn}: {exc}") from exc
            return materialised

        if isinstance(raw, (list, tuple)):
            items = [cls._materialise(item) for item in raw]
            return type(raw)(items)

        return raw

    @classmethod
    def from_dict(
        cls: type[_T],
        payload: dict[str, Any],
        overrides: list[str] | None = None,
    ) -> _T:
        """Build a config instance from a plain dictionary.

        ``overrides`` is accepted for API parity but ignored without olmo-core.
        """
        if overrides:
            warnings.warn(
                "Overrides are ignored without olmo-core.",
                UserWarning,
                stacklevel=2,
            )

        result = cls._materialise(payload)
        if isinstance(result, cls):
            return result
        if isinstance(result, dict):
            accepted = {f.name for f in fields(cls)}
            kwargs = {k: v for k, v in result.items() if k in accepted}
            return cls(**kwargs)
        raise TypeError(f"Expected dict, got {type(result)}")

    # ---- serialisation helpers ----

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        include_class_name: bool = False,
        json_safe: bool = False,
        recurse: bool = True,
    ) -> dict[str, Any]:
        """Serialise this config back to a plain dictionary."""

        def _walk(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                out: dict[str, Any] = {}
                if include_class_name:
                    out[CLASS_KEY] = (
                        f"{obj.__class__.__module__}.{obj.__class__.__name__}"
                    )
                for fld in fields(obj):
                    if exclude_private_fields and fld.name.startswith("_"):
                        continue
                    val = getattr(obj, fld.name)
                    if exclude_none and val is None:
                        continue
                    out[fld.name] = _walk(val) if recurse else val
                return out
            if isinstance(obj, dict):
                return {k: (_walk(v) if recurse else v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                converted = [_walk(i) if recurse else i for i in obj]
                return converted if json_safe else type(obj)(converted)
            if obj is None or isinstance(obj, (float, int, bool, str)):
                return obj
            return str(obj) if json_safe else obj

        return _walk(self)

    def as_config_dict(self) -> dict[str, Any]:
        """Convenience: JSON-safe serialisation with class names embedded."""
        return self.as_dict(
            exclude_none=True,
            exclude_private_fields=True,
            include_class_name=True,
            json_safe=True,
            recurse=True,
        )

    def validate(self) -> None:
        """Override in subclasses to add constraint checks."""

    def build(self) -> Any:
        """Construct the object described by this config.

        Subclasses must override this.
        """
        raise NotImplementedError("Subclasses must implement build()")


# ---------------------------------------------------------------------------
# Public export: prefer olmo-core when present
# ---------------------------------------------------------------------------
if HAS_OLMO_CORE:
    Config = _OlmoCoreConfig  # type: ignore[assignment,misc]
else:
    Config = _FallbackConfig  # type: ignore[assignment]
    warnings.warn(
        "olmo-core is not installed -- running in inference-only mode.  "
        "For training support: pip install spacenit[training]",
        UserWarning,
        stacklevel=2,
    )

__all__ = ["Config", "HAS_OLMO_CORE", "require_olmo_core"]
