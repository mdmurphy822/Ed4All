"""Compatibility alias for the canonical synthesis implementation."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.synthesis_journal is deprecated; import "
    "Trainforge.synthesis.synthesis_journal instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module("Trainforge.synthesis.synthesis_journal")
sys.modules[__name__] = _implementation
