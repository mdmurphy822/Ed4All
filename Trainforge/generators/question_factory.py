"""Compatibility alias for the canonical assessment question factory."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.question_factory is deprecated; import "
    "Trainforge.generators.assessment.question_factory instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.assessment.question_factory"
)
sys.modules[__name__] = _implementation
