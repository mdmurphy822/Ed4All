"""Compatibility alias for the canonical assessment generator."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.assessment_generator is deprecated; import "
    "Trainforge.generators.assessment.generator instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.assessment.generator"
)
sys.modules[__name__] = _implementation
