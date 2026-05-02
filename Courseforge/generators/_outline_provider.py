#!/usr/bin/env python3
"""Courseforge generators — outline-tier provider (Phase 3 §2.1).

The outline tier emits a structurally-correct JSON outline per
:class:`Courseforge.scripts.blocks.Block`. It is the cheap-and-fast
first pass of the two-pass router (Phase 3 §3.1): a small local model
(default ``qwen2.5:7b-instruct-q4_K_M``) drafts the skeleton (key
claims, section_skeleton, source_refs, structural_warnings); the
rewrite tier (Phase 3 §2.2) then turns that outline into pedagogical
prose.

Constructor surface (per Phase 3 §2.1.1):

- ``provider`` — defaults to ``"local"`` (env ``COURSEFORGE_OUTLINE_PROVIDER``).
- ``model`` — defaults to ``"qwen2.5:7b-instruct-q4_K_M"``
  (env ``COURSEFORGE_OUTLINE_MODEL``).
- ``n_candidates`` — self-consistency candidate count, default ``3``
  (env ``COURSEFORGE_OUTLINE_N_CANDIDATES``).
- ``regen_budget`` — per-block regeneration budget, default ``3``
  (env ``COURSEFORGE_OUTLINE_REGEN_BUDGET``).
- ``grammar_mode`` — ``"gbnf" | "json_schema" | "json_object" | "none"``
  (env ``COURSEFORGE_OUTLINE_GRAMMAR_MODE``); ``None`` autodetects from
  ``provider`` + ``base_url``.
- ``max_tokens`` — defaults to ``1200`` (outline JSON is short).
- ``temperature`` — defaults to ``0.0`` (outline tier is deterministic).

Sibling-of-:class:`Courseforge.generators._provider.ContentGeneratorProvider`,
shares the :class:`Courseforge.generators._base._BaseLLMProvider`
HTTP / dispatch / decision-capture skeleton.

Module-level constants (Phase 3 Subtasks 14, 16, 18, 19):

- ``_OUTLINE_KIND_BOUNDS`` — per-block-type bounds table for
  ``key_claims`` / ``section_skeleton`` / ``summary_chars`` (Subtask 14).
- ``_OUTLINE_SYSTEM_PROMPT`` — ≤80-word system prompt (Subtask 16).
- ``_BLOCK_TYPE_GBNF`` — per-block-type GBNF grammar string for
  llama.cpp / vLLM constrained decoding (Subtask 18).
- ``_BLOCK_TYPE_JSON_SCHEMAS`` — per-block-type Draft 2020-12 schema
  for Ollama 0.5+ / Together / vLLM JSON-schema mode (Subtask 19).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ``blocks.py`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# sibling-of-this-package import dance from ``_provider.py`` so the
# Block / Touch import resolves the same regardless of how this module
# is loaded (CLI, MCP tool, pytest).
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import (  # noqa: E402
    BLOCK_TYPES,
    Block,
    Touch,
)

from Courseforge.generators._base import (  # noqa: E402
    _BaseLLMProvider,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — env vars + defaults
# ---------------------------------------------------------------------------

ENV_PROVIDER = "COURSEFORGE_OUTLINE_PROVIDER"
ENV_MODEL = "COURSEFORGE_OUTLINE_MODEL"
ENV_N_CANDIDATES = "COURSEFORGE_OUTLINE_N_CANDIDATES"
ENV_REGEN_BUDGET = "COURSEFORGE_OUTLINE_REGEN_BUDGET"
ENV_GRAMMAR_MODE = "COURSEFORGE_OUTLINE_GRAMMAR_MODE"

DEFAULT_PROVIDER = "local"
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
DEFAULT_N_CANDIDATES = 3
DEFAULT_REGEN_BUDGET = 3

_DEFAULT_MAX_TOKENS = 1200
_DEFAULT_TEMPERATURE = 0.0

SUPPORTED_PROVIDERS: Tuple[str, ...] = (
    "anthropic",
    "together",
    "local",
    "openai_compatible",
)

# Maximum parse / remediation retries when the outline JSON fails
# Schema validation. Mirrors the analogous knob on the synthesis
# providers in :mod:`Trainforge.generators._local_provider`.
MAX_PARSE_RETRIES = 3


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OutlineProviderError(RuntimeError):
    """Outline-tier dispatch / parse / validation failure.

    Carries an opaque ``code`` field so callers can branch on the
    failure mode without parsing the message string.

    Canonical codes:

    - ``outline_exhausted`` — every parse + remediation retry failed
      Schema validation; the outline tier returned no usable JSON.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Per-block-type bounds, prompts, grammar payloads, schemas
# (filled in by Subtasks 14, 16, 18, 19 below)
# ---------------------------------------------------------------------------

_OUTLINE_KIND_BOUNDS: Dict[str, Dict[str, Tuple[int, int]]] = {}
_OUTLINE_SYSTEM_PROMPT: str = ""
_BLOCK_TYPE_GBNF: Dict[str, str] = {}
_BLOCK_TYPE_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OutlineProvider(_BaseLLMProvider):
    """Outline-tier provider — emits structurally-correct JSON skeletons.

    The outline tier is the first pass of Phase 3's two-pass router.
    It produces the structural skeleton each block needs (block_id,
    block_type, content_type, bloom_level, objective_refs, curies,
    key_claims, section_skeleton, source_refs, structural_warnings)
    in a single JSON object — small enough to fit a 7B-class model's
    constrained-decoding window and cheap enough to run with
    self-consistency at ``n_candidates=3`` per block.

    Public method:

    - ``generate_outline(block, *, source_chunks, objectives) -> Block``
      — single-candidate path; the self-consistency loop is layered
      on top by :class:`Courseforge.router.router.CourseforgeRouter`.
    """

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: List[Dict[str, Any]],
        objectives: List[Dict[str, Any]],
    ) -> Block:
        """Generate a single outline candidate for ``block``.

        Implementation lands in Subtask 20.
        """
        raise NotImplementedError(
            "OutlineProvider.generate_outline lands in Phase 3 Subtask 20"
        )

    def _render_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Render the outline-tier user prompt.

        Implementation lands in Subtask 17.
        """
        raise NotImplementedError(
            "OutlineProvider._render_user_prompt lands in Phase 3 Subtask 17"
        )

    def _build_grammar_payload(self, block_type: str) -> Dict[str, Any]:
        """Return the per-call ``extra_payload`` dict carrying the
        grammar / JSON-schema constraint for the resolved provider.

        Implementation lands in Subtask 18.
        """
        raise NotImplementedError(
            "OutlineProvider._build_grammar_payload lands in Phase 3 Subtask 18"
        )

    def _outline_kind_bounds(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """Return the per-block-type bounds table (Subtask 14)."""
        return _OUTLINE_KIND_BOUNDS

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        """Emit one ``block_outline_call`` decision-capture event.

        Implementation lands in Subtask 20 (rationale interpolation
        runs alongside the dispatch path).
        """
        raise NotImplementedError(
            "OutlineProvider._emit_per_call_decision lands in Phase 3 Subtask 20"
        )


__all__ = [
    "OutlineProvider",
    "OutlineProviderError",
    "ENV_PROVIDER",
    "ENV_MODEL",
    "ENV_N_CANDIDATES",
    "ENV_REGEN_BUDGET",
    "ENV_GRAMMAR_MODE",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "DEFAULT_N_CANDIDATES",
    "DEFAULT_REGEN_BUDGET",
    "SUPPORTED_PROVIDERS",
    "MAX_PARSE_RETRIES",
    "_OUTLINE_KIND_BOUNDS",
    "_OUTLINE_SYSTEM_PROMPT",
    "_BLOCK_TYPE_GBNF",
    "_BLOCK_TYPE_JSON_SCHEMAS",
]
