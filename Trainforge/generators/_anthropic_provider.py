#!/usr/bin/env python3
"""
Anthropic synthesis identity constants (slim module).

Phase 4 — the Anthropic-SDK TRAINING-PAIR synthesis path was removed
entirely: the ``AnthropicSynthesisProvider`` class and its SDK transport
no longer exist, and ``Trainforge/synthesize_training.py::run_synthesis``
now fails closed UNCONDITIONALLY on ``provider="anthropic"`` (no
acknowledgment escape). The training-pair corpus is therefore license-clean
by construction. See ``docs/LICENSING.md`` § "Synthesis providers".

What survives here is ONLY the Anthropic-identity constants
(``DEFAULT_SYNTHESIS_MODEL`` / ``ENV_API_KEY`` / ``ENV_MODEL``) that the
OUT-OF-SCOPE ``_curriculum_provider`` (teaching-role classification) still
imports for its own anthropic backend's model/key resolution. The shared
synthesis-provider symbols (``SynthesisProviderError``, the length
sentinels, ``_KIND_BOUNDS``, ``_Usage``) live in ``_synthesis_common`` —
import them from there, NOT from this module (the legacy re-export shim
was removed with the provider class).
"""

from __future__ import annotations

# Anthropic-identity constants. ``ANTHROPIC_SYNTHESIS_MODEL`` (ENV_MODEL) and
# ``ANTHROPIC_API_KEY`` (ENV_API_KEY) are still consumed by other anthropic
# backends (curriculum-alignment / assessment-generator) for model/key
# resolution; this module is their canonical home.
DEFAULT_SYNTHESIS_MODEL = "claude-sonnet-4-6"
ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_MODEL = "ANTHROPIC_SYNTHESIS_MODEL"


__all__ = [
    "DEFAULT_SYNTHESIS_MODEL",
    "ENV_API_KEY",
    "ENV_MODEL",
]
