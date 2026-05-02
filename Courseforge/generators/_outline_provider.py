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

# Per-block-type bounds for the outline tier's structural skeleton.
# Each entry is keyed by ``block_type`` (every value in ``BLOCK_TYPES``)
# and carries (min, max) bounds for three skeleton fields:
#
# - ``key_claims``     — number of factual claims the outline must
#                         enumerate (1-3 for short blocks like
#                         objectives; 1-5 for content blocks).
# - ``section_skeleton`` — number of section headings / subsections
#                         in the outline. ``(0, 0)`` for atomic blocks
#                         (objective, callout, recap) that don't
#                         decompose into sections.
# - ``summary_chars``  — character count for a one-paragraph summary
#                         of the block's content. Mirrors the shape
#                         of ``Trainforge/generators/_local_provider.py
#                         ::DEFAULT_LOCAL_KIND_BOUNDS``.
#
# These values are starting points subject to Phase 4 calibration —
# the bounds are advisory in the system prompt and the grammar payload
# (Subtask 18) does not hard-enforce them at sample time. The Phase 4
# inter-tier validators may tighten or relax them per block_type.
_OUTLINE_KIND_BOUNDS: Dict[str, Dict[str, Tuple[int, int]]] = {
    # Atomic objectives — single claim, no sections.
    "objective": {
        "key_claims": (1, 3),
        "section_skeleton": (0, 0),
        "summary_chars": (40, 200),
    },
    # Concept blocks decompose into 1-3 sections (definition / examples
    # / counter-examples) and carry up to 5 key claims.
    "concept": {
        "key_claims": (1, 5),
        "section_skeleton": (1, 3),
        "summary_chars": (80, 400),
    },
    # Examples are illustrative — minimum claim count, optional section
    # decomposition (worked-step breakdown).
    "example": {
        "key_claims": (1, 3),
        "section_skeleton": (0, 2),
        "summary_chars": (60, 300),
    },
    # Assessment items — stem + answer key + optional rationale section.
    "assessment_item": {
        "key_claims": (1, 2),
        "section_skeleton": (1, 2),
        "summary_chars": (60, 300),
    },
    # Explanations are the long-form pedagogical block; allow more
    # sections + claims.
    "explanation": {
        "key_claims": (2, 6),
        "section_skeleton": (1, 4),
        "summary_chars": (120, 500),
    },
    # Prerequisite sets enumerate prior concepts; sections list each
    # prerequisite cluster.
    "prereq_set": {
        "key_claims": (1, 4),
        "section_skeleton": (1, 3),
        "summary_chars": (60, 300),
    },
    # Activities — instruction set + optional reflection prompt.
    "activity": {
        "key_claims": (1, 4),
        "section_skeleton": (1, 3),
        "summary_chars": (80, 400),
    },
    # Misconceptions — the misconception statement + the correction.
    "misconception": {
        "key_claims": (1, 2),
        "section_skeleton": (1, 2),
        "summary_chars": (60, 300),
    },
    # Atomic callouts — info / warning / success — single claim.
    "callout": {
        "key_claims": (1, 2),
        "section_skeleton": (0, 0),
        "summary_chars": (40, 200),
    },
    # Flip-card grids — N cards × (term, definition).
    "flip_card_grid": {
        "key_claims": (2, 8),
        "section_skeleton": (1, 1),
        "summary_chars": (60, 300),
    },
    # Self-check questions — stem + answer + feedback.
    "self_check_question": {
        "key_claims": (1, 3),
        "section_skeleton": (1, 2),
        "summary_chars": (60, 300),
    },
    # Summary takeaways — bullet list of synthesised claims.
    "summary_takeaway": {
        "key_claims": (2, 5),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
    # Reflection prompts — single claim + the prompt itself.
    "reflection_prompt": {
        "key_claims": (1, 2),
        "section_skeleton": (0, 1),
        "summary_chars": (40, 200),
    },
    # Discussion prompts — opener + branching points.
    "discussion_prompt": {
        "key_claims": (1, 3),
        "section_skeleton": (1, 2),
        "summary_chars": (60, 300),
    },
    # Page chrome — atomic, no claims, no sections.
    "chrome": {
        "key_claims": (0, 1),
        "section_skeleton": (0, 0),
        "summary_chars": (20, 120),
    },
    # Recaps — short summary of prior content.
    "recap": {
        "key_claims": (1, 4),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
}
# Terse outline-tier system prompt. Kept ≤80 words on purpose — the
# 7B-class default model has a small effective instruction-following
# window. Mirrors the terseness of
# ``Trainforge/generators/_local_provider.py
# ::_LOCAL_INSTRUCTION_SYSTEM_PROMPT``.
_OUTLINE_SYSTEM_PROMPT: str = (
    "You are an outline-tier draft generator for Courseforge blocks. "
    "Emit a structurally-correct JSON outline carrying: block_id, "
    "block_type, content_type, bloom_level, objective_refs, curies, "
    "key_claims, section_skeleton, source_refs, structural_warnings. "
    "PRESERVE every CURIE and source_id verbatim from the input. Do "
    "NOT add facts not in the supplied source_chunks. Do NOT generate "
    "prose — generate the structural skeleton only. Output ONLY the "
    "JSON object — no preamble, no markdown, no commentary."
)
# Per-block-type GBNF grammar strings for llama.cpp / vLLM constrained
# decoding. Each grammar accepts a JSON object with at least the
# canonical fields the outline tier emits (block_id, block_type, ...).
#
# Per Phase 3 §2.1.1, these are starting-point grammars subject to
# Phase 4 calibration. The grammars deliberately admit a permissive
# JSON-object surface (mirrors llama.cpp's bundled
# ``grammars/json.gbnf``) rather than a fully-typed shape — the JSON
# Schema validator (Subtask 19) does the strict structural check
# AFTER the model emits, so the GBNF only needs to keep the model
# inside JSON-grammar territory and prevent prose drift.
#
# Authoring per-block-type fully-typed GBNFs (e.g. enforcing
# ``"block_type": "objective"`` as a string literal in-grammar) is
# deferred to Phase 4 — at the 7B-class default model, the JSON-only
# constraint plus a strong system prompt already keeps drift below
# the parse-retry budget on the rdf-shacl-551-2 calibration corpus.
_GENERIC_JSON_GBNF: str = r"""root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ \t\n] ws)?"""

# Lightweight per-block-type GBNF map. Every block_type currently
# maps to the generic JSON grammar; the dict shape exists so a Phase
# 4 author can drop in a tighter per-type grammar without touching
# any call site. The ``_build_grammar_payload`` dispatch reads this
# dict directly.
_BLOCK_TYPE_GBNF: Dict[str, str] = {
    block_type: _GENERIC_JSON_GBNF for block_type in BLOCK_TYPES
}
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

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capture: Optional[Any] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        # Optional dependency injections for tests.
        client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
        # Per-tier knobs (constructor kwargs override env vars).
        n_candidates: Optional[int] = None,
        regen_budget: Optional[int] = None,
        grammar_mode: Optional[str] = None,
    ) -> None:
        # Resolve the model from the per-tier env var BEFORE delegating
        # to ``_BaseLLMProvider.__init__`` so the base only sees a
        # concrete ``model`` value (avoids accidentally falling back to
        # the per-backend baseline when the operator set the per-tier
        # ``COURSEFORGE_OUTLINE_MODEL`` knob).
        resolved_model = (
            model
            or os.environ.get(ENV_MODEL)
            or DEFAULT_MODEL
        )

        super().__init__(
            provider=provider,
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            capture=capture,
            max_tokens=max_tokens,
            temperature=temperature,
            client=client,
            anthropic_client=anthropic_client,
            env_provider_var=ENV_PROVIDER,
            default_provider=DEFAULT_PROVIDER,
            supported_providers=SUPPORTED_PROVIDERS,
            system_prompt=_OUTLINE_SYSTEM_PROMPT,
        )

        # Per-tier knobs not owned by the base.
        self._n_candidates: int = self._resolve_int(
            n_candidates,
            ENV_N_CANDIDATES,
            DEFAULT_N_CANDIDATES,
        )
        self._regen_budget: int = self._resolve_int(
            regen_budget,
            ENV_REGEN_BUDGET,
            DEFAULT_REGEN_BUDGET,
        )
        # ``grammar_mode`` is purely a string knob; ``None`` means
        # autodetect from ``provider`` + ``base_url`` at call time.
        self._grammar_mode: Optional[str] = (
            grammar_mode
            or os.environ.get(ENV_GRAMMAR_MODE)
            or None
        )

    @staticmethod
    def _resolve_int(
        kwarg_value: Optional[int],
        env_var: str,
        default: int,
    ) -> int:
        """Resolve an int knob: kwarg → env var → default."""
        if kwarg_value is not None:
            return int(kwarg_value)
        raw = os.environ.get(env_var)
        if raw is not None and str(raw).strip():
            try:
                return int(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid %s=%r; falling back to default=%d",
                    env_var,
                    raw,
                    default,
                )
        return default

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

    def _render_user_prompt(
        self,
        *,
        block: Block,
        source_chunks: List[Dict[str, Any]],
        objectives: List[Dict[str, Any]],
    ) -> str:
        """Render the outline-tier user prompt for ``block``.

        Sections (in order):

        1. Header: ``Block ID: <id>; Type: <type>``.
        2. Source chunks: id + body, body truncated at 1200 chars each
           so a long chapter doesn't blow the model's context window.
        3. Objectives: id + statement.
        4. Target schema: built from ``_OUTLINE_KIND_BOUNDS[block_type]``
           — informs the model of the structural bounds the JSON
           schema (Subtask 19) hard-enforces.
        5. Per-block-type variations:
           - ``assessment_item``: stem + answer must reference the
             listed objective_refs.
           - ``prereq_set``: list ``prerequisitePages`` explicitly.
        6. Explicit "RESPOND ONLY WITH A JSON OBJECT containing ..."
           closing directive — mirrors the Wave-113 strict-JSON
           hardening.
        """
        block_type = block.block_type
        bounds = _OUTLINE_KIND_BOUNDS.get(block_type, {})

        # Truncate per-chunk body at 1200 chars; mirrors the
        # ``_LOCAL_INSTRUCTION_SYSTEM_PROMPT`` chunk-window heuristic
        # used in :mod:`Trainforge.generators._local_provider`.
        chunk_lines: List[str] = []
        for chunk in source_chunks or []:
            cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
            body = str(chunk.get("body") or chunk.get("text") or "")
            if len(body) > 1200:
                body = body[:1197] + "..."
            chunk_lines.append(f"  - [{cid}] {body}")
        chunks_block = "\n".join(chunk_lines) if chunk_lines else "  (none)"

        objective_lines: List[str] = []
        for obj in objectives or []:
            oid = str(obj.get("id") or obj.get("objective_id") or "")
            stmt = str(obj.get("statement") or obj.get("text") or "")
            objective_lines.append(f"  - {oid}: {stmt}")
        objectives_block = (
            "\n".join(objective_lines) if objective_lines else "  (none)"
        )

        bounds_lines: List[str] = []
        for field_name, (lo, hi) in bounds.items():
            bounds_lines.append(f"  - {field_name}: ({lo}, {hi})")
        bounds_block = (
            "\n".join(bounds_lines) if bounds_lines else "  (no per-type bounds)"
        )

        # Per-block-type variations — appended after the bounds block
        # so the model sees the type-specific contract last (recency
        # bias of the 7B-class default model).
        variation_lines: List[str] = []
        if block_type == "assessment_item":
            variation_lines.append(
                "Assessment item contract: the stem AND the answer "
                "key must reference at least one of the listed "
                "objective_refs verbatim."
            )
        elif block_type == "prereq_set":
            variation_lines.append(
                "Prereq set contract: list every prerequisite page "
                "explicitly under a top-level ``prerequisitePages`` "
                "array; each entry is a string page_id."
            )
        variation_block = "\n".join(variation_lines) if variation_lines else ""

        return (
            f"Block ID: {block.block_id}; Type: {block_type}\n"
            f"Page ID: {block.page_id}\n\n"
            "Source chunks (preserve every source_id verbatim in "
            "source_refs):\n"
            f"{chunks_block}\n\n"
            "Objectives (preserve every objective id verbatim in "
            "objective_refs):\n"
            f"{objectives_block}\n\n"
            "Target structural bounds (per-block-type):\n"
            f"{bounds_block}\n\n"
            f"{variation_block}\n\n"
            "RESPOND ONLY WITH A JSON OBJECT containing: block_id, "
            "block_type, content_type, bloom_level, objective_refs, "
            "curies, key_claims, section_skeleton, source_refs, "
            "structural_warnings. No preamble, no markdown, no "
            "commentary."
        )

    def _build_grammar_payload(self, block_type: str) -> Dict[str, Any]:
        """Return the per-call ``extra_payload`` dict.

        The returned dict is merged into the OpenAI-compatible POST
        body just before the wire-call by Subtask 21's extension to
        :meth:`_BaseLLMProvider._dispatch_call`. Dispatch on
        ``(self._provider, self._base_url, self._grammar_mode)``
        per Phase 3 §2.1.1:

        - ``mode=="gbnf"`` OR
          (``provider in {"local","openai_compatible"}`` AND
           ``base_url`` looks like llama.cpp / lmstudio) →
          ``{"grammar": <gbnf-string>}``.
        - ``mode=="json_schema"`` → full Ollama 0.5+ JSON-Schema dict
          via ``{"format": <schema_dict>}``.
        - ``provider=="together"`` → strict OpenAI-style
          ``{"response_format": {"type": "json_schema", ...}}``.
        - vLLM (detected by base_url) →
          ``{"extra_body": {"guided_json": <schema_dict>}}``.
        - Anthropic / unrecognised → ``{}`` (rely on Wave-113
          ``json_mode=True`` on the OA client).
        """
        schema = _BLOCK_TYPE_JSON_SCHEMAS.get(block_type)
        gbnf = _BLOCK_TYPE_GBNF.get(block_type)
        base_url = (self._base_url or "").lower()
        mode = (self._grammar_mode or "").lower() or None
        provider = self._provider

        # Explicit mode wins.
        if mode == "gbnf":
            if gbnf:
                return {"grammar": gbnf}
            return {}
        if mode == "json_schema":
            if schema is not None:
                return {"format": schema}
            return {}
        if mode == "json_object":
            # Wave-113 OA-style ``json_object`` — already injected by
            # the OpenAICompatibleClient when ``json_mode=True``; no
            # additional payload needed.
            return {}
        if mode == "none":
            return {}

        # Auto-detect path.
        if provider == "together":
            if schema is not None:
                return {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"OutlineBlock_{block_type}",
                            "schema": schema,
                            "strict": True,
                        },
                    }
                }
            return {}

        if provider in {"local", "openai_compatible"}:
            # llama.cpp / LM Studio expose a ``grammar`` payload field;
            # detect by base_url substring (llama.cpp default is
            # :8080, LM Studio :1234, but the URL substring is the
            # canonical signal).
            if any(
                marker in base_url
                for marker in ("llama", "lmstudio", "lm-studio")
            ):
                if gbnf:
                    return {"grammar": gbnf}
                return {}
            # vLLM exposes ``guided_json`` under ``extra_body``.
            if "vllm" in base_url:
                if schema is not None:
                    return {"extra_body": {"guided_json": schema}}
                return {}
            # Default for ``local`` (Ollama) — fall back to the
            # GBNF grammar payload. Ollama's older flag is
            # ``format: "json"`` (Wave-113 default on the OA client),
            # so the ``grammar`` field is silently ignored on
            # legacy Ollama and consumed by llama.cpp-compatible
            # servers; either way the JSON-mode-only path remains
            # the wire-level fallback.
            if gbnf:
                return {"grammar": gbnf}
            return {}

        # Anthropic and any other unrecognised backend — let the
        # Wave-113 ``json_mode`` carry the constraint.
        return {}

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
