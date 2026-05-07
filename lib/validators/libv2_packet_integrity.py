"""DEPRECATED: re-exports from :mod:`lib.validators.libv2.packet_integrity`.

Wave W-D10 T10.1: the validator now lives in the ``libv2/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.libv2.packet_integrity import *  # noqa: F401,F403
from lib.validators.libv2.packet_integrity import (  # noqa: F401
    PacketIntegrityValidator,
    ValidationIssue,
    ValidationResult,
    ASSESSMENT_CHUNK_TYPE,
    RELAX_ENV_VAR,
)

warnings.warn(
    "lib.validators.libv2_packet_integrity is deprecated; "
    "use lib.validators.libv2.packet_integrity",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PacketIntegrityValidator",
    "ValidationIssue",
    "ValidationResult",
    "ASSESSMENT_CHUNK_TYPE",
    "RELAX_ENV_VAR",
]
