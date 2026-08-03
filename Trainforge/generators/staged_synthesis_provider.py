"""Compatibility alias for the canonical staged-synthesis provider."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.staged_synthesis_provider is deprecated; import "
    "Trainforge.generators.staged.provider instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module("Trainforge.generators.staged.provider")
sys.modules[__name__] = _implementation
