"""Decorator utilities for tagging unstable or in-progress APIs."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

_Fn = TypeVar("_Fn", bound=Callable[..., Any])


def unstable(note: str = "This API is not yet stable") -> Callable[[_Fn], _Fn]:
    """Tag a function or class as *unstable*.

    Unstable APIs may change signature, behaviour, or be removed entirely in
    future releases.  A ``FutureWarning`` is emitted on every call (or
    instantiation, for classes) so that callers are aware.

    Args:
        note: Human-readable context about what makes this unstable.

    Returns:
        A decorator that wraps the target and emits a warning.
    """

    def _wrap(target: _Fn) -> _Fn:
        qualified = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
        banner = f"'{qualified}' is unstable and may change without notice."
        if note:
            banner = f"{banner} {note}"

        # Mark so tests / introspection can detect unstable objects
        setattr(target, "_unstable_", True)

        if isinstance(target, type):
            # For classes: intercept __init__
            orig_init = target.__init__  # type: ignore[misc]

            @functools.wraps(orig_init)
            def _init_shim(self: Any, *a: Any, **kw: Any) -> None:
                warnings.warn(banner, FutureWarning, stacklevel=2)
                orig_init(self, *a, **kw)

            target.__init__ = _init_shim  # type: ignore[method-assign,misc]
            target.__doc__ = (
                f"[UNSTABLE] {banner}\n\n{target.__doc__}"
                if target.__doc__
                else f"[UNSTABLE] {banner}"
            )
            return cast(_Fn, target)

        # For plain functions
        @functools.wraps(target)
        def _fn_shim(*a: Any, **kw: Any) -> Any:
            warnings.warn(banner, FutureWarning, stacklevel=2)
            return target(*a, **kw)

        _fn_shim.__doc__ = (
            f"[UNSTABLE] {banner}\n\n{target.__doc__}"
            if target.__doc__
            else f"[UNSTABLE] {banner}"
        )
        setattr(_fn_shim, "_unstable_", True)
        return cast(_Fn, _fn_shim)

    return _wrap
