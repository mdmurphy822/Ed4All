"""Compatibility alias for the canonical staged evidence-window contract."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.synthesis_window_contract is deprecated; import "
    "Trainforge.generators.staged.window_contract instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.staged.window_contract"
)
sys.modules[__name__] = _implementation
