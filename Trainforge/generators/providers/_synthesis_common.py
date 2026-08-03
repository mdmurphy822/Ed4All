#!/usr/bin/env python3
"""Shared synthesis-provider base symbols.

Neutral home for the synthesis-provider symbols that EVERY provider
(anthropic / local / together / claude_session / curriculum) needs —
relocated out of ``_anthropic_provider`` so the non-anthropic providers
don't hard-import the anthropic module just to reach these definitions.

This module deliberately carries no anthropic / SDK / heavy imports: it
is a small, provider-agnostic base that any synthesis module can import
without pulling in a provider implementation.

Symbols (moved verbatim from ``_anthropic_provider``; values, signatures,
and docstrings unchanged):

- ``SynthesisProviderError`` — typed validation-failure error.
- ``PROMPT_MIN`` / ``PROMPT_MAX`` / ``COMPLETION_MIN`` / ``COMPLETION_MAX``
  — length sentinels shared with the factories.
- ``_KIND_BOUNDS`` — per-``kind`` length-bound lookup table.
- ``_Usage`` — token-usage tally dataclass.

``_anthropic_provider`` re-exports all of these for backward compatibility,
so importers that still point at the anthropic module keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# Length sentinels — match instruction_factory.py / preference_factory.py
# so paraphrased output respects the same downstream gates.
PROMPT_MIN, PROMPT_MAX = 40, 400
COMPLETION_MIN, COMPLETION_MAX = 50, 600

# Per-`kind` length bounds, used by ``_clamp`` to look up [lo, hi] without
# the call site having to thread the bounds through. The ``chosen`` and
# ``rejected`` arms of a preference pair share the completion bounds.
_KIND_BOUNDS: Dict[str, tuple] = {
    "prompt": (PROMPT_MIN, PROMPT_MAX),
    "completion": (COMPLETION_MIN, COMPLETION_MAX),
    "chosen": (COMPLETION_MIN, COMPLETION_MAX),
    "rejected": (COMPLETION_MIN, COMPLETION_MAX),
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SynthesisProviderError(RuntimeError):
    """Typed error raised on synthesis-provider validation failures.

    Short paraphrases raise this error instead of receiving filler text that
    could contaminate training data. The typed error lets the caller's retry
    path request a fresh paraphrase.

    The ``code`` field is a stable string the caller can dispatch on
    (e.g. ``completion_below_minimum``, ``prompt_below_minimum``);
    ``chunk_id`` is optional context for log/audit correlation.
    """

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        chunk_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.chunk_id = chunk_id
        self.details = dict(details or {})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class _Usage:
    """Token-usage tally extracted from the Anthropic response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }


__all__ = [
    "SynthesisProviderError",
    "PROMPT_MIN",
    "PROMPT_MAX",
    "COMPLETION_MIN",
    "COMPLETION_MAX",
    "_KIND_BOUNDS",
    "_Usage",
]
