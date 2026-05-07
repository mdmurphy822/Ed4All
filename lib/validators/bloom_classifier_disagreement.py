"""DEPRECATED: re-exports from :mod:`lib.validators.bloom.classifier_disagreement`.

Wave W-D10 T10.1: the validator now lives in the ``bloom/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.bloom.classifier_disagreement import *  # noqa: F401,F403
from lib.validators.bloom.classifier_disagreement import (  # noqa: F401
    BloomClassifierDisagreementValidator,
    _DISAGREEMENT_CONFIDENCE_FLOOR,
    _DISPERSION_THRESHOLD,
    _AUDITED_BLOCK_TYPES,
    _ISSUE_LIST_CAP,
    _coerce_blocks,
    _extract_text_for_classification,
    _strip_html,
    _block_attr,
)

warnings.warn(
    "lib.validators.bloom_classifier_disagreement is deprecated; "
    "use lib.validators.bloom.classifier_disagreement",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BloomClassifierDisagreementValidator",
    "_DISAGREEMENT_CONFIDENCE_FLOOR",
    "_DISPERSION_THRESHOLD",
    "_AUDITED_BLOCK_TYPES",
    "_ISSUE_LIST_CAP",
    "_coerce_blocks",
    "_extract_text_for_classification",
    "_strip_html",
    "_block_attr",
]
