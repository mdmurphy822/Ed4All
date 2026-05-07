"""DEPRECATED: re-exports from :mod:`lib.validators.pair.lo_refs`.

Wave W-D10 T10.1: the validator now lives in the ``pair/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.pair.lo_refs import *  # noqa: F401,F403
from lib.validators.pair.lo_refs import (  # noqa: F401
    PairLearningOutcomeRefsValidator,
)

warnings.warn(
    "lib.validators.pair_lo_refs is deprecated; "
    "use lib.validators.pair.lo_refs",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = ["PairLearningOutcomeRefsValidator"]
