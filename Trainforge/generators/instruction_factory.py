"""Compatibility alias for the canonical instruction-pair factory."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.instruction_factory is deprecated; import "
    "Trainforge.generators.pairs.instruction instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.pairs.instruction"
)
sys.modules[__name__] = _implementation
