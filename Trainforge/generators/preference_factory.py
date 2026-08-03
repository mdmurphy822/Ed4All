"""Compatibility alias for the canonical preference-pair factory."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.preference_factory is deprecated; import "
    "Trainforge.generators.pairs.preference instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.pairs.preference"
)
sys.modules[__name__] = _implementation
