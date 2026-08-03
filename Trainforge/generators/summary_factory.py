"""Compatibility alias for the canonical summary postprocessor."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.summary_factory is deprecated; import "
    "Trainforge.generators.postprocessing.summary_factory instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.postprocessing.summary_factory"
)
sys.modules[__name__] = _implementation
