"""Compatibility alias for the canonical synthesis implementation."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.synthesis_eligibility is deprecated; import "
    "Trainforge.synthesis.synthesis_eligibility instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module("Trainforge.synthesis.synthesis_eligibility")
sys.modules[__name__] = _implementation
