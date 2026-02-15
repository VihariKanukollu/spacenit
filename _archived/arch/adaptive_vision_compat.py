"""Backward-compatibility shim for the old ``adaptive_vision_compat`` import path.

All public symbols have moved to :mod:`spacenit.arch.adaptive_vision_encoder`.
This module re-exports everything from there and will be removed in a future
release.

.. deprecated::
    Import directly from ``spacenit.arch.adaptive_vision_encoder`` instead.
"""

import sys
import warnings

import spacenit.arch.adaptive_vision_encoder as adaptive_vision_encoder

from spacenit.arch.adaptive_vision_encoder import *  # noqa: F403

warnings.warn(
    "spacenit.arch.adaptive_vision_compat is deprecated. "
    "Please import from spacenit.arch.adaptive_vision_encoder instead.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = adaptive_vision_encoder
