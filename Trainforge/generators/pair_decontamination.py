"""Compatibility alias for the canonical pair-decontamination helpers."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.pair_decontamination is deprecated; import "
    "Trainforge.generators.postprocessing.pair_decontamination instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.postprocessing.pair_decontamination"
)
sys.modules[__name__] = _implementation
