"""DEPRECATED: re-exports from :mod:`lib.validators.libv2.model`.

Wave W-D10 T10.1: the validator now lives in the ``libv2/`` subpackage.
Re-exported here for back-compat with ``config/workflows.yaml``,
external MCP clients, and existing test imports. Will be removed in a
future minor version. Use the new path directly.
"""
import warnings

from lib.validators.libv2.model import LibV2ModelValidator  # noqa: F401

warnings.warn(
    "lib.validators.libv2_model is deprecated; "
    "use lib.validators.libv2.model",
    PendingDeprecationWarning,
    stacklevel=2,
)

__all__ = ["LibV2ModelValidator"]
