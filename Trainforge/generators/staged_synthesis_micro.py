"""Compatibility alias for the canonical staged-synthesis implementation."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.staged_synthesis_micro is deprecated; import "
    "Trainforge.generators.staged.micro instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module("Trainforge.generators.staged.micro")
sys.modules[__name__] = _implementation
