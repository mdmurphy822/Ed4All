"""DEPRECATED: re-exports from :mod:`lib.validators.bloom.structural_enforcement`.

Wave W-D10 T10.1: the validator now lives in the ``bloom/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.bloom.structural_enforcement import *  # noqa: F401,F403
from lib.validators.bloom.structural_enforcement import (  # noqa: F401
    BloomStructuralEnforcementValidator,
    _AUDITED_BLOCK_TYPES,
    _CLAUSE_FLOORS,
    _ANSWER_TOKEN_FLOORS,
    _COMPARISON_MARKERS,
    _JUDGMENT_MARKERS,
    _DESIGN_MARKERS,
    _OPEN_ENDED_QUESTION_TYPES,
)

warnings.warn(
    "lib.validators.bloom_structural_enforcement is deprecated; "
    "use lib.validators.bloom.structural_enforcement",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BloomStructuralEnforcementValidator",
    "_AUDITED_BLOCK_TYPES",
    "_CLAUSE_FLOORS",
    "_ANSWER_TOKEN_FLOORS",
    "_COMPARISON_MARKERS",
    "_JUDGMENT_MARKERS",
    "_DESIGN_MARKERS",
    "_OPEN_ENDED_QUESTION_TYPES",
]
