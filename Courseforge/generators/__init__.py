"""Courseforge generators package.

Phase 3 Subtask 12: re-exports the LLM-agnostic provider surface so
callers can ``from Courseforge.generators import ContentGeneratorProvider,
_BaseLLMProvider`` without reaching into the private module names.

Outline and rewrite providers live in the focused ``outline`` and ``rewrite``
subpackages. Keeping this package root limited to shared content-generation
primitives avoids loading either specialized provider stack on import.
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
