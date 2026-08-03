"""Compatibility import and module entry point for synthesis generation."""

from __future__ import annotations

import importlib
import runpy
import sys
import warnings

_CANONICAL_MODULE = "Trainforge.synthesis.synthesize_training"

warnings.warn(
    "Trainforge.synthesize_training is deprecated; import "
    "Trainforge.synthesis.synthesize_training instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    runpy.run_module(_CANONICAL_MODULE, run_name="__main__")
else:
    _implementation = importlib.import_module(_CANONICAL_MODULE)
    sys.modules[__name__] = _implementation
