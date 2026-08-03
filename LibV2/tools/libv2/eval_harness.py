"""Compatibility import and module entry point for the evaluation harness."""

from __future__ import annotations

import importlib
import runpy
import sys
import warnings

_CANONICAL_MODULE = f"{__package__}.evaluation.harness"

warnings.warn(
    f"{__package__}.eval_harness is deprecated; import "
    f"{_CANONICAL_MODULE} instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    runpy.run_module(_CANONICAL_MODULE, run_name="__main__")
else:
    _implementation = importlib.import_module(_CANONICAL_MODULE)
    sys.modules[__name__] = _implementation
