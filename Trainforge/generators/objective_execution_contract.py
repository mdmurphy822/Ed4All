"""Compatibility alias for the canonical staged objective contract."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Trainforge.generators.objective_execution_contract is deprecated; import "
    "Trainforge.generators.staged.objective_contract instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

_implementation = importlib.import_module(
    "Trainforge.generators.staged.objective_contract"
)
sys.modules[__name__] = _implementation
