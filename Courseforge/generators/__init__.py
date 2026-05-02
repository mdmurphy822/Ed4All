"""Courseforge generators package.

Phase 3 Subtask 12: re-exports the LLM-agnostic provider surface so
callers can ``from Courseforge.generators import ContentGeneratorProvider,
_BaseLLMProvider`` without reaching into the private module names.

OutlineProvider / RewriteProvider re-exports land alongside Subtasks
13-22 once those modules exist; until then this package exposes only
the Phase 1 + Phase 3 Subtask 9 surface.
"""

from Courseforge.generators._base import _BaseLLMProvider
from Courseforge.generators._provider import (
    DEFAULT_PROVIDER,
    ENV_PROVIDER,
    SUPPORTED_PROVIDERS,
    ContentGeneratorProvider,
)

__all__ = [
    "_BaseLLMProvider",
    "ContentGeneratorProvider",
    "DEFAULT_PROVIDER",
    "ENV_PROVIDER",
    "SUPPORTED_PROVIDERS",
]
