"""Compatibility import for the saved-model evaluation bridge."""

from __future__ import annotations

import importlib
import sys
import warnings

_CANONICAL_MODULE = f"{__package__}.evaluation.model_bridge"

warnings.warn(
    f"{__package__}.model_eval_bridge is deprecated; import "
    f"{_CANONICAL_MODULE} instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(_CANONICAL_MODULE)
sys.modules[__name__] = _implementation
