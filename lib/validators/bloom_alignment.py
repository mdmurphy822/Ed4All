"""DEPRECATED: re-exports from :mod:`lib.validators.bloom.alignment`.

Wave W-D10 T10.0 + T10.1: this is the renamed-then-moved BloomAlignment
validator surface. The validator now lives in the ``bloom/`` subpackage.
Re-exported here for back-compat with downstream code that adopted the
T10.0 transient ``bloom_alignment.py`` name. Will be removed in a future
minor version. Use the new path directly.
"""
import warnings

from lib.validators.bloom.alignment import (  # noqa: F401
    BLOOM_VERBS,
    BloomAlignmentValidator,
    detect_bloom_level,
)

warnings.warn(
    "lib.validators.bloom_alignment is deprecated; "
    "use lib.validators.bloom.alignment",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = ["BLOOM_VERBS", "BloomAlignmentValidator", "detect_bloom_level"]
