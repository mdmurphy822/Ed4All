"""DEPRECATED: re-exports from :mod:`lib.validators.libv2.manifest`.

Wave W-D10 T10.1: the validator now lives in the ``libv2/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.libv2.manifest import (  # noqa: F401
    LibV2ManifestValidator,
    _CONCEPT_GRAPH_SHA256_RE,
    _CHUNKS_SHA256_RE,
    _EXPECTED_SUBDIRS,
)

warnings.warn(
    "lib.validators.libv2_manifest is deprecated; "
    "use lib.validators.libv2.manifest",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LibV2ManifestValidator",
    "_CONCEPT_GRAPH_SHA256_RE",
    "_CHUNKS_SHA256_RE",
    "_EXPECTED_SUBDIRS",
]
