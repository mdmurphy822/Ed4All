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
import re
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
from MCP.hardening.error_classifier import (  # noqa: E402
    ErrorClass,
    classify_error,
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

# Worker W6: per-block transient-retry budget for dispatch-side
# failures (Ollama 503 / connection reset / read timeout). Transient
# retries do NOT advance ``MAX_PARSE_RETRIES`` so a flaky local server
# can't burn the parse budget before any parse attempt completes.
# Permanent errors (auth failure, bad request) re-raise immediately.
# UNKNOWN-class errors fall through to the legacy parse-retry path so
# semantic regressions don't change behavior on unclassified errors.
_TRANSIENT_RETRY_BUDGET = 3


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
    - ``outline_transient_exhausted`` — Worker W6: the transient-retry
      budget (``_TRANSIENT_RETRY_BUDGET``) was exhausted on dispatch-
      side failures (Ollama 503 / connection reset / read timeout)
      without any parse attempt completing. Distinct from
      ``outline_exhausted`` so the router can branch on the failure
      mode.
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
    # Plan §3.4 / §1.4: bumped key_claims from (1, 2) to (1, 4). The
    # canonical RDF-triple shape (subject, predicate, object) is a
    # three-tuple; the previous (1, 2) cap forced a 7B-class model to
    # synthesise a single compressed claim it demonstrably can't
    # produce on its own, so the model emitted ["subject", "predicate",
    # "object"] and tripped the maxItems gate. (1, 4) admits the
    # natural three-tuple and matches the concept-block bound.
    "assessment_item": {
        "key_claims": (1, 4),  # plan §3.4: bumped from (1, 2)
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
    "JSON object — no preamble, no markdown, no commentary. "
    # Plan §3.1: bloom_level enum directive — closes the "bloom_level: 2"
    # numeric-tier drift the 7B-class default model emits when it
    # infers Bloom Level 2 = "Understand" from the canonical six-level
    # taxonomy and writes the tier number rather than the string label.
    "bloom_level MUST be one of: remember, understand, apply, analyze, "
    "evaluate, create. Use the lowercase string label, not a numeric "
    "tier. "
    # Plan §3.3: empty-CURIE permission directive — closes the
    # "invented CURIE prefix" / "full IRI as CURIE" failure modes the
    # model emits when faced with an empty source-side CURIE list and
    # a pattern-bearing required array.
    "curies MUST be either the empty list [] when no CURIE tokens are "
    "in the source chunks, or a list of strict prefix:local CURIE "
    "strings (e.g. rdf:type, sh:NodeShape). NEVER emit a full IRI as "
    "a CURIE value. NEVER invent a CURIE prefix from a chunk slug."
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
# ---------------------------------------------------------------------------
# Per-block-type JSON Schema map (Subtask 19).
# Each value is a Draft 2020-12 schema requiring the canonical outline
# fields (block_id, block_type, content_type, bloom_level,
# objective_refs, curies, key_claims, section_skeleton, source_refs,
# structural_warnings) plus per-block-type extras (e.g. assessment_item
# requires stem + answer_key; prereq_set requires prerequisitePages).
# ``additionalProperties: false`` keeps the model from drifting into
# fabricated fields.
# ---------------------------------------------------------------------------

_BLOOM_LEVEL_ENUM: List[str] = [
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]

# CURIE pattern mirrors the canonical SHACL/RDF surface form check
# used elsewhere in the project (e.g. lib/ontology/* prefix maps).
_CURIE_PATTERN: str = r"^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$"


def _build_block_outline_schema(
    block_type: str,
    *,
    extra_required: Optional[List[str]] = None,
    extra_properties: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Construct the per-block-type JSON Schema payload."""
    bounds = _OUTLINE_KIND_BOUNDS.get(block_type, {})
    key_claim_min, key_claim_max = bounds.get("key_claims", (0, 32))
    section_min, section_max = bounds.get("section_skeleton", (0, 16))

    properties: Dict[str, Dict[str, Any]] = {
        "block_id": {"type": "string", "minLength": 1},
        "block_type": {"const": block_type},
        "content_type": {"type": "string", "minLength": 1},
        "bloom_level": {"type": "string", "enum": _BLOOM_LEVEL_ENUM},
        "objective_refs": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "curies": {
            "type": "array",
            "items": {"type": "string", "pattern": _CURIE_PATTERN},
        },
        "key_claims": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": key_claim_min,
            "maxItems": key_claim_max,
        },
        "section_skeleton": {
            "type": "array",
            "items": {"type": "object"},
            "minItems": section_min,
            "maxItems": section_max,
        },
        "source_refs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sourceId": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "minLength": 1},
                },
                "required": ["sourceId", "role"],
            },
        },
        "structural_warnings": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    }
    required: List[str] = [
        "block_id",
        "block_type",
        "content_type",
        "bloom_level",
        "objective_refs",
        "curies",
        "key_claims",
        "section_skeleton",
        "source_refs",
        "structural_warnings",
    ]
    if extra_properties:
        properties.update(extra_properties)
    if extra_required:
        required.extend(extra_required)

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_BLOCK_TYPE_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {}
for _bt in BLOCK_TYPES:
    if _bt == "assessment_item":
        _BLOCK_TYPE_JSON_SCHEMAS[_bt] = _build_block_outline_schema(
            _bt,
            extra_required=["stem", "answer_key"],
            extra_properties={
                "stem": {"type": "string", "minLength": 1},
                "answer_key": {"type": "string", "minLength": 1},
            },
        )
    elif _bt == "prereq_set":
        _BLOCK_TYPE_JSON_SCHEMAS[_bt] = _build_block_outline_schema(
            _bt,
            extra_required=["prerequisitePages"],
            extra_properties={
                "prerequisitePages": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                }
            },
        )
    else:
        _BLOCK_TYPE_JSON_SCHEMAS[_bt] = _build_block_outline_schema(_bt)


# ---------------------------------------------------------------------------
# Per-error-pattern retry-directive table (plan §3.6).
# ---------------------------------------------------------------------------
#
# When the lenient JSON parse + Draft 2020-12 validator rejects a
# candidate, the schema-fix retry message echoes the validator's
# terse error string. Pre-§3.6 the message was bare; §3.6 layers a
# small per-pattern directive table on top so the model sees the
# canonical fix-it instruction next to the validator output instead
# of having to infer the remediation from the message alone. Keys
# are compiled regexes matched against the validator's
# ``ValidationError.message``; values are imperative directives
# (~120 chars) the user prompt appends after the schema dump on
# the retry attempt.
#
# Patterns covered (mirrors plan §1 failure classes):
#   - bloom_level enum drift (numeric tier vs string label)
#   - CURIE pattern violation (full IRI / invented prefix / missing colon)
#   - key_claims maxItems exceeded (model emitted a list shape the
#     bound rejects)
#   - generic enum-vs-int (any "is not of type 'string'" error)
#
# Adding a new pattern: append a tuple ``(re.compile(...), directive)``
# below; the retry helper picks the FIRST matching pattern (highest-
# precedence rule first). Keep the directive ≤200 chars to bound
# the suffix size.

_RETRY_DIRECTIVE_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (
        re.compile(r"is not one of \['remember'"),
        "bloom_level MUST be the lowercase string label, not a numeric "
        "tier or capitalised form. Use exactly one of: remember, "
        "understand, apply, analyze, evaluate, create.",
    ),
    (
        re.compile(r"does not match '\^\[a-z\]"),
        "CURIE pattern requires strict prefix:local form (e.g. "
        "rdf:type, sh:NodeShape). If no CURIEs are present in the "
        "source chunks, emit 'curies': []. NEVER emit a full IRI "
        "(no slashes, no '#' characters) and NEVER invent a CURIE "
        "prefix from a chunk slug.",
    ),
    (
        re.compile(r" is too long$"),
        "key_claims is a flat array of short prose statements "
        "(≤30 words each). Compress list-shaped data (e.g. "
        "['subject', 'predicate', 'object']) into a single claim "
        "string ('An RDF triple has three components: subject, "
        "predicate, object.') rather than emitting one claim per "
        "list element.",
    ),
    (
        re.compile(r"is not of type 'string'"),
        "Every enum-typed field MUST be a JSON string (quoted), "
        "not a number or bare token. Wrap numeric tier or boolean "
        "values in their canonical string label.",
    ),
]


def _match_retry_directive(last_error: str) -> Optional[str]:
    """Return the directive matching ``last_error``'s validator pattern.

    Walks :data:`_RETRY_DIRECTIVE_PATTERNS` in declaration order and
    returns the first matching directive. Returns ``None`` when no
    pattern matches — the caller falls back to the bare validator
    error echo.
    """
    if not last_error:
        return None
    for pattern, directive in _RETRY_DIRECTIVE_PATTERNS:
        if pattern.search(last_error):
            return directive
    return None


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
        #
        # Phase 3a env-var-first contract (Subtask 24): the resolution
        # chain here is ``kwargs.get("model") or os.environ.get(ENV_MODEL)
        # or DEFAULT_MODEL`` — the per-call kwarg wins outright (highest
        # priority), the env var beats the hardcoded default, and the
        # hardcoded default fires only when both are unset. Acceptance
        # test: ``test_phase3a_env_var_overrides_hardcoded_default`` in
        # ``Courseforge/router/tests/test_router.py``.
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
        remediation_suffix: Optional[str] = None,
    ) -> Block:
        """Generate a single outline candidate for ``block``.

        Single-candidate path — the self-consistency loop is layered
        on top by :class:`Courseforge.router.router.CourseforgeRouter`
        (Phase 3 Subtask 37). Steps:

        1. Build the user prompt via :meth:`_render_user_prompt`.
        2. Build the per-block-type ``extra_payload`` via
           :meth:`_build_grammar_payload`.
        3. Dispatch up to ``MAX_PARSE_RETRIES`` times via
           :meth:`_BaseLLMProvider._dispatch_call`, applying
           :meth:`OpenAICompatibleClient._extract_json_lenient` to the
           response and validating against
           :data:`_BLOCK_TYPE_JSON_SCHEMAS[block.block_type]`. On
           parse / Schema-validation failure, append a remediation
           hint to the user prompt and retry.
        4. On exhaustion, raise
           :class:`OutlineProviderError(code="outline_exhausted")`.
        5. On success, return a new :class:`Block` via
           :func:`dataclasses.replace` carrying the parsed outline
           dict as ``content`` plus a ``Touch(tier="outline",
           purpose="draft", ...)`` entry on ``touched_by``.

        Phase 3.5 Subtask 18: when ``remediation_suffix`` is non-None,
        the rendered user prompt is augmented with a per-failure
        remediation block before dispatch. The suffix is built by the
        :func:`Courseforge.router.remediation._append_remediation_for_gates`
        helper from the prior validator-chain failures so the
        re-rolled candidate sees what went wrong on the previous
        attempt and the directive to fix it. ``None`` is the default
        so the legacy single-candidate path keeps emitting byte-stable
        prompts.
        """
        if block is None:
            raise ValueError("OutlineProvider.generate_outline: block required")
        if block.block_type not in BLOCK_TYPES:
            raise ValueError(
                f"OutlineProvider.generate_outline: unknown block_type "
                f"{block.block_type!r}"
            )

        # Lazy-import the lenient JSON parser to avoid pulling
        # OpenAICompatibleClient in test environments that stub the
        # base class. ``_extract_json_lenient`` is a staticmethod so we
        # don't need a client instance.
        from Trainforge.generators._openai_compatible_client import (
            OpenAICompatibleClient,
        )
        import jsonschema  # type: ignore[import-untyped]

        schema = _BLOCK_TYPE_JSON_SCHEMAS.get(block.block_type)
        extra_payload = self._build_grammar_payload(block.block_type)

        base_user_prompt = self._render_user_prompt(
            block=block,
            source_chunks=source_chunks,
            objectives=objectives,
            remediation_suffix=remediation_suffix,
        )

        last_error: Optional[str] = None
        last_raw: str = ""
        parsed: Optional[Dict[str, Any]] = None
        total_retries = 0
        # Worker W6: transient retries (Ollama 503 / connection reset /
        # read timeout) are counted separately from MAX_PARSE_RETRIES so
        # they do NOT burn the parse budget. Permanent errors re-raise
        # immediately. UNKNOWN-class errors preserve legacy semantics
        # (advance the parse retry loop).
        transient_retries = 0
        attempt = 0

        while attempt < MAX_PARSE_RETRIES:
            user_prompt = base_user_prompt
            if attempt > 0 and last_error:
                schema_hint = (
                    json.dumps(schema, sort_keys=True) if schema else "{}"
                )
                # Plan §3.6: pull the per-pattern directive matching the
                # validator's last_error message and append it after the
                # schema dump so the model sees the canonical fix-it
                # instruction, not just the terse validator string.
                directive = _match_retry_directive(last_error)
                directive_block = (
                    f"\nFix-it directive: {directive}" if directive else ""
                )
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    "Your previous output failed JSON Schema validation: "
                    f"{last_error}"
                    f"{directive_block}\n"
                    "Return ONLY a JSON object matching this schema:\n"
                    f"{schema_hint}"
                )
            try:
                raw_text, retry_count = self._dispatch_call(
                    user_prompt,
                    extra_payload=extra_payload or None,
                )
            except Exception as exc:
                # Worker W6: classify the dispatch-side failure so a
                # transient (Ollama 503 / connection reset / read
                # timeout) doesn't burn the parse-retry budget. Permanent
                # errors (auth failure, bad request) surface immediately;
                # UNKNOWN-class errors preserve the legacy parse-retry
                # path so semantic regressions don't shift behavior on
                # unclassified errors.
                classified = classify_error(exc, task_id=block.block_id)
                if classified.error_class is ErrorClass.TRANSIENT:
                    if transient_retries < _TRANSIENT_RETRY_BUDGET:
                        transient_retries += 1
                        # Do NOT advance attempt — re-dispatch under the
                        # same parse-retry slot.
                        continue
                    raise OutlineProviderError(
                        f"Outline tier exhausted transient-retry budget "
                        f"({_TRANSIENT_RETRY_BUDGET}) for block "
                        f"{block.block_id!r} (last_error={exc!r})",
                        code="outline_transient_exhausted",
                    ) from exc
                if classified.error_class is ErrorClass.PERMANENT:
                    # Re-raise immediately — no retry on permanent
                    # errors (validation_error, missing_input,
                    # 401/403/404, etc.).
                    raise
                # UNKNOWN / POISON_PILL → fall through to legacy
                # parse-retry path. POISON_PILL is treated like UNKNOWN
                # at the per-call site; batch-level poison-pill detection
                # is the orchestrator's responsibility.
                last_error = (
                    f"dispatch failure ({classified.error_class.value}): "
                    f"{exc}"
                )
                last_raw = ""
                attempt += 1
                continue

            total_retries += int(retry_count)
            last_raw = raw_text

            candidate = OpenAICompatibleClient._extract_json_lenient(raw_text)
            if candidate is None:
                last_error = "lenient JSON parse returned None"
                attempt += 1
                continue
            if schema is not None:
                try:
                    jsonschema.Draft202012Validator(schema).validate(candidate)
                except jsonschema.ValidationError as exc:
                    # Truncate the validation message so the
                    # remediation hint stays inside the model's
                    # context window.
                    last_error = str(exc.message)[:300]
                    attempt += 1
                    continue

            parsed = candidate
            break

        # Emit the per-call decision-capture event regardless of
        # outcome so the audit trail captures every dispatch.
        self._emit_per_call_decision(
            raw_text=last_raw,
            retry_count=total_retries,
            block_id=block.block_id,
            block_type=block.block_type,
            page_id=block.page_id,
            success=parsed is not None,
            attempts=attempt + 1 if parsed is not None else MAX_PARSE_RETRIES,
            last_error=last_error,
        )

        if parsed is None:
            raise OutlineProviderError(
                f"Outline tier exhausted {MAX_PARSE_RETRIES} attempts for "
                f"block {block.block_id!r} (last_error={last_error!r})",
                code="outline_exhausted",
            )

        # Construct the touch + new Block. Provider must be one of the
        # ``_TOUCH_PROVIDERS`` set in ``blocks.py`` — we map our
        # provider tag onto that set (``openai_compatible`` collapses
        # to ``local`` for the audit trail since both go through the
        # same OA client). Anthropic / together / local map 1:1.
        touch_provider = self._provider
        if touch_provider == "openai_compatible":
            touch_provider = "local"

        touch = Touch(
            model=self._model,
            provider=touch_provider,
            tier="outline",
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision_capture_id=self._last_capture_id(),
            purpose="draft",
        )
        new_block = dataclasses.replace(block, content=parsed)
        return new_block.with_touch(touch)

    def _render_user_prompt(
        self,
        *,
        block: Block,
        source_chunks: List[Dict[str, Any]],
        objectives: List[Dict[str, Any]],
        remediation_suffix: Optional[str] = None,
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
        7. Phase 3.5 Subtask 18: ``remediation_suffix`` (when non-None)
           is appended after the closing directive. Built upstream by
           the router's self-consistency loop from the prior
           validator-chain failures via
           :func:`Courseforge.router.remediation._append_remediation_for_gates`
           so the re-rolled candidate sees what went wrong and the
           directive to fix it.
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
        # Plan §3.1: emit the canonical bloom_level allowed-set on
        # every bounds block so the user prompt enumerates the same
        # enum the JSON Schema enforces. Recency-bias of the 7B-class
        # default model means a bottom-of-bounds reminder noticeably
        # lifts attempt-1 pass rate.
        bounds_lines.append(
            "  - bloom_level allowed values: remember | understand | "
            "apply | analyze | evaluate | create"
        )
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

        out = (
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
        # Phase 3.5 Subtask 18: append the remediation suffix when
        # supplied. The suffix is the canonical
        # _append_remediation_for_gates output (header + per-failure
        # blocks); we only need to glue it on with two newlines so the
        # closing JSON directive above stays distinct from the
        # remediation context.
        if remediation_suffix:
            out += "\n\n" + remediation_suffix
        return out

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
            # Plan §3.2: default for ``local`` (Ollama) flipped to
            # the Ollama 0.5+ JSON-Schema engagement path. Most
            # current local deployments are Ollama 0.5+, which
            # honours ``format: <schema_dict>`` for full schema-
            # constrained decoding. Operators on older Ollama or
            # llama.cpp / LM Studio override by setting
            # ``COURSEFORGE_OUTLINE_GRAMMAR_MODE=gbnf`` or by adding
            # ``lmstudio`` to the base_url. The pre-§3.2 default
            # was ``{"grammar": <gbnf>}``, which Ollama silently
            # ignored — leaving the per-block-type schema enforcement
            # purely post-hoc against the lenient JSON parser.
            if schema is not None:
                return {"format": schema}
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

        Rationale interpolates per-call signals (block_id, block_type,
        page_id, provider, model, output character count, retry
        count, attempts, success/failure, last_error) per the project's
        LLM call-site instrumentation contract (≥20 chars, dynamic
        signals).
        """
        block_id = call_context.get("block_id", "")
        block_type = call_context.get("block_type", "")
        page_id = call_context.get("page_id", "")
        success = bool(call_context.get("success", False))
        attempts = int(call_context.get("attempts", 0))
        last_error = call_context.get("last_error")
        char_count = len(raw_text or "")

        decision = (
            f"outline_call:{block_type}:{block_id}:"
            f"{'success' if success else 'failed'}"
        )
        rationale_parts = [
            f"block_id={block_id}",
            f"block_type={block_type}",
            f"page_id={page_id}",
            f"provider={self._provider}",
            f"model={self._model}",
            f"output_chars={char_count}",
            f"retry_count={retry_count}",
            f"attempts={attempts}",
            f"success={success}",
        ]
        if last_error:
            # Truncate the last_error to keep the rationale below the
            # decision-capture validator's soft length cap.
            rationale_parts.append(f"last_error={str(last_error)[:120]}")
        rationale = "; ".join(rationale_parts)

        self._emit_decision(
            decision_type="block_outline_call",
            decision=decision,
            rationale=rationale,
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
    "_RETRY_DIRECTIVE_PATTERNS",
    "_match_retry_directive",
]
