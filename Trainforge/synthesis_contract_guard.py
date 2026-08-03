"""Compatibility alias for the canonical synthesis implementation."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.synthesis_contract_guard is deprecated; import "
    "Trainforge.synthesis.synthesis_contract_guard instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module("Trainforge.synthesis.synthesis_contract_guard")
sys.modules[__name__] = _implementation
