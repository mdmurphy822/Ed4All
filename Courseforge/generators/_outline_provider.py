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
import math
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
    _default_supported_providers,
)
from Trainforge.generators._openai_compatible_client import (  # noqa: E402
    ENV_REQUEST_TIMEOUT as _OA_ENV_REQUEST_TIMEOUT,
)
from MCP.hardening.error_classifier import (  # noqa: E402
    ErrorClass,
    classify_error,
)
from lib.validators.content_type import (  # noqa: E402
    get_valid_chunk_types,
)
from lib.ontology.bloom import (  # noqa: E402
    BLOOM_LEVELS as _BLOOM_LEVELS,
)
from lib.retrieval._prompts import (  # noqa: E402
    estimate_tokens as _estimate_tokens,
    resolve_num_ctx as _resolve_num_ctx,
)
from lib.llm.truncation_guard import (  # noqa: E402
    check_prompt_not_truncated,
)
from lib.retrieval.answer_backend import (  # noqa: E402
    PromptTruncatedError as _PromptTruncatedError,
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
# outline-overflow-fix-2026-07: cap on the number of source chunks rendered
# into a single outline-tier user prompt + the input-truncation tripwire.
ENV_MAX_CHUNKS = "COURSEFORGE_OUTLINE_MAX_CHUNKS"
ENV_TRUNCATION_TRIPWIRE = "COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE"

DEFAULT_PROVIDER = "local"
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
DEFAULT_N_CANDIDATES = 3
DEFAULT_REGEN_BUDGET = 3
#: Default head-K cap on source chunks rendered per outline prompt. The
#: median block resolves ~20 chunks (median chunk section ~24k chars); an
#: un-capped render blows a small served window and the local server silently
#: head-truncates the prompt, so the model sees only the objectives / closing
#: TAIL. 8 keeps the grounding tight enough to fit a 4k-class window while
#: preserving the top-of-list (already relevance/citation-ordered) chunks.
DEFAULT_MAX_CHUNKS = 8

#: Truthy / falsey tokens (case-insensitive) for the boolean env resolvers.
#: Mirrors ``Courseforge/generators/_rewrite_fit_window.py``.
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})

_DEFAULT_MAX_TOKENS = 1200
_DEFAULT_TEMPERATURE = 0.0


def _resolve_outline_max_chunks(
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Resolve ``COURSEFORGE_OUTLINE_MAX_CHUNKS`` (default 8).

    Head-K cap on the source chunks rendered into a single outline-tier
    user prompt. Parse-with-fallback: garbage / non-positive / unset →
    :data:`DEFAULT_MAX_CHUNKS` (a misconfigured cap must never silently
    disable the guard, mirroring ``resolve_rewrite_num_ctx``).
    """
    raw = (env or os.environ).get(ENV_MAX_CHUNKS)
    if not raw or not str(raw).strip():
        return DEFAULT_MAX_CHUNKS
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHUNKS
    return val if val > 0 else DEFAULT_MAX_CHUNKS


def _resolve_outline_truncation_tripwire(
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Resolve ``COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE`` (default ON).

    The tripwire is output-neutral (it only fail-closes on a detected
    silent head-truncation), so it ships ON by default; an explicit falsey
    ``0/false/no/off`` (case-insensitive) is the escape hatch. Unset /
    garbage / truthy → on. Mirrors the rewrite tier's
    ``resolve_truncation_tripwire``.
    """
    raw = (env or os.environ).get(ENV_TRUNCATION_TRIPWIRE)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY

# Per-request HTTP timeout (seconds) for the outline tier's
# OpenAI-compatible backends (local / together). The outline tier is a
# 7B-class local model emitting structured JSON; while its output is
# short, a cold model load + constrained-decoding pass can still exceed
# the OpenAICompatibleClient 60s default. We pass an explicit generous
# timeout, sourced from ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` when set,
# else 300.0 — matching the rewrite tier so one knob drives both content-
# generation tiers. Resolution / precedence (high → low): explicit
# per-call ``timeout`` kwarg > ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` >
# this default.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0


def _resolve_request_timeout() -> float:
    """Resolve the outline-tier per-request timeout default.

    Reads ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (the cross-cutting
    content-generation timeout knob); a missing / unparseable /
    non-positive value falls back to
    :data:`_DEFAULT_REQUEST_TIMEOUT_SECONDS` (300.0) — never the bare
    60s client default.
    """
    raw = os.environ.get(_OA_ENV_REQUEST_TIMEOUT)
    if not raw or not str(raw).strip():
        return _DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_REQUEST_TIMEOUT_SECONDS
    if not math.isfinite(parsed) or parsed <= 0:
        return _DEFAULT_REQUEST_TIMEOUT_SECONDS
    return parsed

# Legacy alias for the "non-Ollama / non-Together OpenAI-compatible server"
# tag. It collapses to ``local`` at constructor entry (mirroring the router's
# ``_get_outline_provider`` collapse) so a standalone
# ``OutlineProvider(provider="openai_compatible")`` behaves identically to a
# router-mediated construction. Without the collapse, the value would reach
# the base's registry else-branch and raise ``UnknownEndpoint`` (it is NOT a
# registry row name — the alias only works because the router collapses it).
_OPENAI_COMPATIBLE_ALIAS = "openai_compatible"


def _supported_providers() -> Tuple[str, ...]:
    """Registry-superset provider allow-list for the outline tier.

    The base's registry-derived default (``anthropic`` SDK transport +
    every ``kind: openai_compatible`` row in ``config/endpoints.yaml`` —
    ``local`` / ``together`` / ``nvidia`` / ``nvidia-deepseek`` / ``groq`` /
    ``fireworks`` / ``deepseek`` / …) PLUS the legacy ``openai_compatible``
    alias (collapsed to ``local`` at constructor entry). Adding a provider
    is a registry-entry change, NOT a subclass. The base's
    ``_default_supported_providers`` already falls back to the legacy trio
    on a registry read failure, so this composition never crashes at import.
    """
    base = _default_supported_providers()
    extras = (_OPENAI_COMPATIBLE_ALIAS,)
    seen: set = set()
    out: List[str] = []
    for name in tuple(base) + extras:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


SUPPORTED_PROVIDERS: Tuple[str, ...] = _supported_providers()


def _touch_provenance(provider: str) -> str:
    """Collapse a resolved provider name to its registry Touch provenance.

    Mirrors ``CourseforgeRouter._collapse_to_touch_provider`` WITHOUT the
    router import (avoids a cycle): a registry openai-compatible seat
    (e.g. ``groq`` / ``fireworks`` / ``nvidia-deepseek``) stamps its
    declared ``provenance_provider`` (``together`` / ``nvidia`` / …) so
    ``Touch.provider`` stays inside the closed provenance set
    (``Courseforge/scripts/blocks.py::_TOUCH_PROVIDERS``). The legacy
    ``openai_compatible`` alias collapses to ``local``; an unknown /
    non-registry name passes through unchanged so Touch validation
    surfaces the bad value rather than this helper masking it.
    """
    if provider == _OPENAI_COMPATIBLE_ALIAS:
        return "local"
    try:
        from lib.llm.endpoints import load_endpoint_registry  # noqa: PLC0415

        row = load_endpoint_registry().get(provider)
        if row is not None:
            return str(row.get("provenance_provider", provider))
    except Exception:  # noqa: BLE001 — defensive; never crash on registry I/O
        pass
    return provider

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
    # Examples are illustrative — optional section decomposition
    # (worked-step breakdown). CB5b (content-block-quality-2026-06
    # §CB5b): bumped key_claims from (1, 3) to (2, 4) so a CONCRETE
    # worked instance has room for problem + step(s) + answer claims
    # (the abstract-rule failure mode emitted only 2 generic claims and
    # the rewrite tier rendered a stub). (2, 4) leaves headroom for the
    # 3-part problem/steps/answer decomposition without forcing it.
    "example": {
        "key_claims": (2, 4),
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
        # 2026-06-21 content-gap fix: section_skeleton floor 1 -> 0 (a
        # section-less activity prompt — e.g. "Evaluate the following
        # expressions: …" — must SHIP rather than be dropped on the
        # empty-required-array failure after the regen budget; key_claims
        # min stays 1, never lowered to 0).
        "section_skeleton": (0, 3),
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
        # 2026-06-21 content-gap fix: key_claims floor 2 -> 1 (a single
        # grounded takeaway must SHIP rather than be dropped after the
        # regen budget; the retry directive still pushes for richer
        # output, this only stops dropping a valid block).
        "key_claims": (1, 5),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
    # Reflection prompts — single claim + the prompt itself.
    "reflection_prompt": {
        "key_claims": (1, 2),
        "section_skeleton": (0, 1),
        "summary_chars": (40, 200),
    },
    # B15 Resources — a curated list of further-reading links; each
    # resource is one claim, no prose body (cycle-1 B15 completion).
    "resources": {
        "key_claims": (1, 8),
        "section_skeleton": (0, 1),
        "summary_chars": (40, 240),
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
    # Wave-2 block-variety additions.
    # Application / case scenario — short setup + apply prompt.
    "scenario": {
        # 2026-06-21 content-gap fix: key_claims floor 2 -> 1 (a single
        # grounded scenario claim must SHIP rather than be dropped after
        # the regen budget; the retry directive still pushes for richer
        # output, this only stops dropping a valid block).
        "key_claims": (1, 4),
        "section_skeleton": (0, 1),
        "summary_chars": (80, 400),
    },
    # Practice problem with a no-JS reveal solution.
    "problem": {
        "key_claims": (1, 3),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
    # Single vocabulary term + definition — atomic.
    "vocab_card": {
        "key_claims": (1, 2),
        "section_skeleton": (0, 0),
        "summary_chars": (40, 200),
    },
    # Single highlighted formula + variable gloss — atomic.
    "formula": {
        "key_claims": (1, 2),
        "section_skeleton": (0, 0),
        "summary_chars": (30, 160),
    },
    # Checklist of actionable steps / criteria.
    "checklist": {
        # 2026-06-21 content-gap fix: key_claims floor 2 -> 1 (a single
        # grounded checklist item must SHIP rather than be dropped after
        # the regen budget; the retry directive still pushes for richer
        # output, this only stops dropping a valid block).
        "key_claims": (1, 6),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
    # Issue I6 instruction-palette-v2 additions.
    # Comparison / tabular block — one claim per row's salient fact, no
    # section decomposition (the rows ARE the structure).
    "table": {
        "key_claims": (2, 8),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
    # Acronym / mnemonic block — one claim per letter→term mapping; atomic
    # (the <dl> rows are the structure, no sub-sections).
    "acronym": {
        "key_claims": (2, 8),
        "section_skeleton": (0, 0),
        "summary_chars": (40, 240),
    },
    # Key-idea aside — a single emphasized principle, atomic.
    "key_idea": {
        "key_claims": (1, 3),
        "section_skeleton": (0, 0),
        "summary_chars": (40, 240),
    },
    # IB5 framework-aligned pedagogical block types.
    # Hook / activation — an attention prompt + forward transition; no new
    # content (a single activation claim), atomic.
    "hook": {
        "key_claims": (1, 2),
        "section_skeleton": (0, 0),
        "summary_chars": (40, 240),
    },
    # Multimedia — a time-based artifact + its a11y stack; one claim per
    # key segment the media teaches.
    "multimedia": {
        "key_claims": (1, 4),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
    # Worked example (faded) — a fully-worked procedure: one claim per
    # subgoal/step plus problem + answer; section_skeleton holds the steps.
    "worked_example": {
        "key_claims": (2, 6),
        "section_skeleton": (1, 3),
        "summary_chars": (80, 400),
    },
    # Diagram / visual model — one claim per relationship the diagram
    # dual-codes; the data-table rows are the structure (no sub-sections).
    "diagram": {
        "key_claims": (2, 6),
        "section_skeleton": (0, 1),
        "summary_chars": (60, 300),
    },
    # FR-INT-02 — B08 first-class Guided Practice — faded-scaffold practice
    # items that follow a worked example; mirrors activity/problem bounds (one
    # claim per practice item, optional section decomposition for the steps).
    "guided_practice": {
        "key_claims": (1, 4),
        "section_skeleton": (0, 3),
        "summary_chars": (80, 400),
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
    "a CURIE value. NEVER invent a CURIE prefix from a chunk slug. "
    # Wave 1.5 W1.5.B: per-claim source attribution directive. The
    # outline-tier schema's ``key_claims`` field admits a back-compat
    # ``oneOf`` (legacy List[str] vs structured List[{claim,
    # source_chunk_ids[]}]) for existing fixtures, but new authoring
    # under this prompt MUST emit the structured shape so Wave 2 W2.F
    # NLI scoring can fan out per-claim instead of against the union of
    # block-level source_refs[].
    "key_claims MUST be a list of objects, each "
    "{\"claim\": \"<short prose statement>\", "
    "\"source_chunk_ids\": [\"<chunk_id>\", ...]}. The "
    "source_chunk_ids array carries the IDs of the supplied "
    "source_chunks the claim is derived from — at least 1, more when "
    "the claim synthesizes across chunks. Every chunk_id MUST appear "
    "in the block's top-level source_refs[]. PROHIBITED: emitting "
    "key_claims as a flat array of strings; that is the legacy shape "
    "and the new schema rejects it for new authoring."
    # Objective-echo fix (2026-07): claims are facts FROM the source, not a
    # restatement of the learning objective. On thin / exercise-grounded
    # pages the 7B lazily echoes the CO statement verbatim into key_claims
    # ("Identify the place value of each digit …"), which is vacuous for
    # grounding, NLI scoring, and downstream assessment/training synthesis.
    " "
    "Each key_claims[] entry MUST be a factual, teachable assertion "
    "EXTRACTED from the source chunks — a specific fact, definition, "
    "relationship, formula, or worked result. A claim MUST NEVER be a "
    "restatement or paraphrase of a learning objective itself: the "
    "objective states what the learner will DO, whereas a claim states "
    "what is TRUE according to the source (e.g. the objective 'Identify "
    "the place value of each digit in a number' yields the CLAIM 'The "
    "place value of the 4 in 51,493 is thousands.')."
    # Wave 1.7 W1.7.B: behavioral-outcome / Bloom-floor directive.
    # The objectives list in the user prompt surfaces the declared
    # Bloom triple `[Bloom: {level}, verb: {verb}]` per objective; the
    # outline-tier `bloom_level` MUST be at or above the declared
    # level so the rewrite tier authors prose at the correct cognitive
    # demand. Pairs with the symmetric rewrite-tier
    # `MUST teach the BEHAVIORAL OUTCOME` directive in
    # `_REWRITE_SYSTEM_PROMPT`.
    " "
    "`bloom_level` MUST be at or above the declared Bloom level of "
    "the objective(s) listed in `objective_refs`. A block whose "
    "`objective_refs` cites a `create`-level objective MUST NOT emit "
    "`bloom_level: remember` or `bloom_level: understand` — the "
    "block's pedagogy must climb to the objective's level. When a "
    "single page carries multiple `objective_refs`, distribute the "
    "Bloom levels across blocks (e.g. one `concept` at `understand` "
    "scaffolding the foundation, one `example` at `apply`, one "
    "`assessment_item` at the full declared level)."
    # CB5b (content-block-quality-2026-06 §CB5b): example-block
    # concrete-worked-instance directive. The rewrite tier renders an
    # `example` block from its `key_claims` (it does NOT consult the
    # section_skeleton), so an `example` block whose claims are the
    # abstract rule ("To divide fractions, multiply by the reciprocal")
    # hands the rewriter nothing concrete to author and it ships a stub.
    # Mint the CONCRETE worked instance at outline time instead. Stays
    # consistent with the "skeleton only, short prose statements"
    # framing — each claim is still a short statement, it just carries
    # the specific instance (problem / step / answer) rather than the
    # general rule.
    " "
    "For an `example` block, the `key_claims` MUST capture a CONCRETE "
    "WORKED INSTANCE drawn from a worked `Example` or `TRY IT` item in "
    "the cited source chunks — include the SPECIFIC problem (the actual "
    "numbers / expression), the intermediate steps, and the final "
    "answer. Do NOT state only the general rule or formula. If the "
    "source chunk for this block contains a worked example, use ITS "
    "numbers; an example block must give the rewriter a complete worked "
    "instance to render, not just the rule."
    # Wave5-W27 propagation (from Wave4-W27 `ffe517d` content-generator.md).
    # The outline tier emits structured JSON, not HTML — heading hierarchy
    # (HEADING_SKIP) applies at the rewrite tier, not here. But the two
    # downstream HTML attributes the rewrite tier stamps
    # (`data-cf-source-ids`, `data-cf-objective-id`) are derived from the
    # outline block's `source_refs[]` and `objective_refs[]` — so the
    # outline tier MUST populate both lists for the rewrite-tier stamping
    # to fire correctly. The post-rewrite EMPTY_SOURCE_REFS gate
    # (Wave4-I10 `ccd6374`) is now fail-closed CRITICAL, so an outline
    # block that ships with no `source_refs[]` propagates a missing
    # `data-cf-source-ids` attribute and trips the gate downstream.
    " "
    "Wave-27 source-grounding contract: `source_refs[]` MUST be "
    "populated with the source-chunk IDs supplied in the user prompt's "
    "Source chunks section. Each entry is a "
    "`{\"sourceId\": \"semantik:<slug>#<block_id>\", \"role\": \"<role>\"}` "
    "object. Empty list (`[]`) is permitted ONLY for boilerplate / "
    "navigational / template-chrome blocks with no SemantiK source "
    "grounding (Wave-27 carve-out). The rewrite tier stamps "
    "`data-cf-source-ids` on the rendered HTML from this list; missing "
    "entries surface at the post-rewrite EMPTY_SOURCE_REFS gate as a "
    "fail-closed critical violation."
    " "
    "Wave-27 objective-grounding contract: `objective_refs[]` MUST be "
    "populated with the canonical TO-NN / CO-NN learning-objective IDs "
    "supplied in the user prompt's Objectives section. The pattern is "
    "`^[A-Z]{2,}-\\d{2,}$` (e.g. `TO-01`, `CO-03`). NEVER invent or "
    "abbreviate an LO ID. The rewrite tier stamps `data-cf-objective-id` "
    "from this list onto the rendered HTML; the chunker's "
    "`learning_outcome_refs[]` field reads the stamp at extraction "
    "time, so a missing or invented LO ID silently breaks RAG retrieval "
    "by learning objective."
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
# the parse-retry budget on the RDF/SHACL calibration corpus.
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

# Sourced from the single-source-of-truth canonical tuple in
# ``lib/ontology/bloom.py`` (``BLOOM_LEVELS``) rather than a hardcoded
# duplicate, so the outline-schema enum stays in lockstep with the
# ontology and the numeric-tier bloom repair (``_repair_outline_bloom_level``)
# can index ``_BLOOM_LEVELS`` directly. Order is the canonical
# 1=remember … 6=create progression.
_BLOOM_LEVEL_ENUM: List[str] = list(_BLOOM_LEVELS)

# Bloom-diversity fix: level → 0-based rank for the FLOOR comparison.
# Sourced from the canonical ``_BLOOM_LEVELS`` tuple (NOT a hardcoded
# ordering), so it stays in lockstep with ``lib/ontology/bloom.py`` if the
# taxonomy is ever re-ordered. Higher rank = higher cognitive demand.
_BLOOM_LEVEL_RANK: Dict[str, int] = {
    level: idx for idx, level in enumerate(_BLOOM_LEVELS)
}


def _max_bloom_level(*levels: Optional[str]) -> Optional[str]:
    """Return the highest-ranked valid Bloom level among ``levels``.

    Uses the canonical ``_BLOOM_LEVELS`` ordering (via ``_BLOOM_LEVEL_RANK``)
    so the comparison stays in lockstep with the ontology. Unknown / ``None``
    / empty inputs are ignored (never raised on). Returns ``None`` when no
    input is a valid canonical level — the caller then leaves the field
    untouched (fail-closed; no fabrication of an arbitrary level).
    """
    best: Optional[str] = None
    best_rank = -1
    for level in levels:
        if not level:
            continue
        rank = _BLOOM_LEVEL_RANK.get(str(level).strip().lower())
        if rank is None:
            continue
        if rank > best_rank:
            best_rank = rank
            best = _BLOOM_LEVELS[rank]
    return best

# content_type enum mirrors the canonical chunk-type taxonomy
# (schemas/taxonomies/content_type.json::$defs.ChunkType) — the SAME
# vocabulary ``BlockContentTypeValidator`` enforces at the inter-tier
# seam. Pinning it into the outline JSON Schema lets Ollama
# structured-output decoding force a canonical value at sample time.
# Without the enum the field was free-form ``{"type": "string"}`` and
# Qwen drifted off-vocabulary (observed 2026-05-15 on Qwen-7B: emitted
# 'text', 'definition_and_example', 'place value and rounding' — every
# one of which fails the inter-tier content_type gate).
_CONTENT_TYPE_ENUM: List[str] = sorted(get_valid_chunk_types())

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
        "content_type": {"type": "string", "enum": _CONTENT_TYPE_ENUM},
        "bloom_level": {"type": "string", "enum": _BLOOM_LEVEL_ENUM},
        "objective_refs": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "curies": {
            "type": "array",
            "items": {"type": "string", "pattern": _CURIE_PATTERN},
        },
        # Wave 1.5 W1.5.A: back-compat oneOf admitting both shapes.
        # - legacy: List[str] (every existing fixture + every existing
        #   corpus emit; preserved so the schema bump is non-breaking).
        # - structured: List[{claim, source_chunk_ids[]}] (new authoring;
        #   per-claim attribution drives Wave 2 W2.F NLI scoring).
        # Mixed-shape arrays (one string + one object) are rejected by
        # ``oneOf`` semantics — the desired contract per plan §6.1.
        # Each arm preserves the per-block-type ``minItems`` /
        # ``maxItems`` bounds resolved above so the bound enforcement
        # is shape-symmetric.
        "key_claims": {
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": key_claim_min,
                    "maxItems": key_claim_max,
                },
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string", "minLength": 1},
                            "source_chunk_ids": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "minItems": 1,
                            },
                        },
                        "required": ["claim", "source_chunk_ids"],
                        "additionalProperties": False,
                    },
                    "minItems": key_claim_min,
                    "maxItems": key_claim_max,
                },
            ],
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
        # Worker W7: distractors[] (>=2) + correct_answer_index (>=0)
        # required alongside stem + answer_key. Mirrors
        # ``schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.AssessmentItem``;
        # the canonical shape lives there and ``lib.validators.
        # assessment_item_payload.BlockAssessmentItemPayloadValidator``
        # gates the same fields end-to-end. Pre-W7 the outline tier
        # required only stem + answer_key, so a model could ship a
        # "valid" assessment_item with one distractor or none and
        # ship it; the new fields close that regression class. The
        # per-distractor ``misconception_ref`` is OPTIONAL and is a
        # forward-compat slot — the chunk -> misconception linkage
        # that would populate it is deferred to a separate plan
        # post-W9. When present, the pattern is enforced at the
        # validator (and at the JSON-LD emit's $defs.AssessmentItem).
        _BLOCK_TYPE_JSON_SCHEMAS[_bt] = _build_block_outline_schema(
            _bt,
            extra_required=[
                "stem",
                "answer_key",
                "distractors",
                "correct_answer_index",
            ],
            extra_properties={
                "stem": {"type": "string", "minLength": 1},
                "answer_key": {"type": "string", "minLength": 1},
                "distractors": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "required": ["text"],
                        "properties": {
                            "text": {"type": "string", "minLength": 1},
                            "misconception_ref": {
                                "type": "string",
                                "pattern": r"^[A-Z]{2,}-\d{2,}#m\d+$",
                            },
                        },
                    },
                },
                "correct_answer_index": {
                    "type": "integer",
                    "minimum": 0,
                },
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
    # improvement-map Step 3 (concern #2): key_claims minItems recovery
    # directive. ``key_claims`` is a ``oneOf`` (legacy ``List[str]`` vs
    # structured ``List[{claim, source_chunk_ids}]``), so when the 7B
    # emits exactly ONE well-formed structured claim against a block
    # whose bound demands ≥2 (e.g. ``explanation``: ``key_claims=(2,6)``),
    # jsonschema's ``best_match`` reports the TOP-LEVEL ``oneOf`` message
    # ("... is not valid under any of the given schemas") which MASKS the
    # real minItems cause. The dispatch loop now walks ``exc.context`` and
    # appends the structured arm's "too short" sub-error to ``last_error``
    # so THIS pattern matches FIRST (before the W1.5.B ``oneOf`` catch-all
    # directly below), steering the model toward DECOMPOSING the
    # explanation into ≥{min} distinct claims rather than re-emitting the
    # same single object across all retries (the misfire this fixes). The
    # ``{min}`` placeholder is interpolated by ``_match_retry_directive``
    # from the block's per-type ``_OUTLINE_KIND_BOUNDS`` lower bound.
    # The pattern is scoped to a ``key_claims``-tagged "too short"
    # marker (the surfacing step in ``generate_outline`` prefixes the
    # surfaced sub-error with ``key_claims:``) so it does NOT steal the
    # W7 ``assessment_item`` ``distractors ... is too short`` directive
    # below. Placed FIRST so first-match wins.
    (
        re.compile(
            r"key_claims:.*(too short|too few items|minItems)",
            re.IGNORECASE,
        ),
        "key_claims has too FEW entries — emit at least {min} distinct "
        "claim objects, each {{claim, source_chunk_ids}}, by decomposing "
        "the explanation into separate factual statements (e.g. split a "
        "process into its steps). Do NOT fabricate; if the source "
        "supports one idea, break it into its constituent sub-claims.",
    ),
    # Wave 1.5 W1.5.B: per-claim source attribution recovery directive.
    # Fires when the model emits the legacy ``List[str]`` shape under
    # the new ``oneOf`` schema — the validator surfaces "is not valid
    # under any of the given schemas" as the canonical error. The
    # directive points the model at the new shape contract on the next
    # parse-retry. Placed AFTER the minItems directive above so a masked
    # minItems cause (now surfaced) takes precedence; this remains the
    # catch-all for a genuine flat-string emit.
    (
        re.compile(r"is not valid under any of the given schemas"),
        "key_claims MUST be a list of objects each containing "
        "{\"claim\": \"<text>\", \"source_chunk_ids\": [\"<chunk_id>\"]}. "
        "Do NOT emit key_claims as flat strings. Every source_chunk_id "
        "MUST also appear in the block's source_refs[].",
    ),
    # Wave 1.7 W1.7.B: BLOCK_OBJECTIVE_BLOOM_UNDERMET retry directive.
    # The validator class itself lands in W1.7.C; this tuple prepares
    # the prompt-side recovery surface so when W1.7.C wires the
    # validator into the inter-tier seam, the regenerate-loop already
    # has a directive to splice into the next outline-tier prompt.
    # Placed immediately AFTER the W1.5.B `oneOf` directive so the
    # most-specific patterns retain priority under
    # `_match_retry_directive`'s first-match semantics.
    (
        re.compile(r"BLOCK_OBJECTIVE_BLOOM_UNDERMET"),
        "The block's bloom_level is below the declared Bloom level "
        "of its objective_refs. Re-emit with bloom_level at or above "
        "the objective's declared level. Adjust prose to scaffold up "
        "to the higher cognitive demand.",
    ),
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
    # Worker W7 + assessment-item-descriptor fix (2026-06): assessment_item
    # Blocks must carry the four dedicated fields ``stem`` / ``answer_key`` /
    # ``distractors[]`` / ``correct_answer_index`` as TOP-LEVEL keys with REAL
    # VALUES. The validator surfaces the missing-key / too-few-items errors as
    # ``'stem' is a required property`` / ``'answer_key' is a required
    # property`` / ``'distractors' is a required property`` /
    # ``'correct_answer_index' is a required property`` / ``[...] is too
    # short`` depending on which constraint trips first. The 2026-06
    # investigation (7B + 14B) showed the model pours the question into
    # ``key_claims`` / ``section_skeleton`` and OMITS the dedicated fields,
    # OR emits a LIST of field-TYPE DESCRIPTOR objects
    # (``[{"type": "stem"}, {"type": "distractors"}, ...]``) instead of real
    # values — so the directive now names all four fields AND forbids the
    # descriptor-list shape, telling the model to emit the actual stem text /
    # option texts / answer value. One pattern catches every failure mode via
    # a non-greedy alternation match.
    (
        re.compile(
            r"'(stem|answer_key|distractors|correct_answer_index)' is a "
            r"required property"
            r"|distractors.* is too short"
        ),
        "Block of type 'assessment_item' MUST emit four TOP-LEVEL fields "
        "with REAL VALUES (never field-type descriptors like "
        "{\"type\": \"stem\"}, and never inside key_claims / "
        "section_skeleton): `stem` (the ACTUAL question text the learner "
        "reads), `answer_key` (the ACTUAL correct answer VALUE), "
        "`distractors` (a list of at least 2 objects, each {\"text\": "
        "\"<an ACTUAL wrong-answer option>\"}), and `correct_answer_index` "
        "(a 0-based integer). Replace any {\"type\": ...} descriptor object "
        "with the real stem text, real option texts, and real answer value.",
    ),
    # prereq_set Blocks must carry a non-empty `prerequisitePages` array
    # (the outline schema marks it required, minItems:1). A 7B-class model
    # routinely emits it (or `key_claims`) as `[]`, which trips the
    # jsonschema minItems error — surfaced as "'prerequisitePages' is a
    # required property", "[] is too short", or "[] should be non-empty"
    # depending on Draft-validator version. Without a matching directive the
    # model blindly re-rolls the same empty array and exhausts the budget.
    (
        re.compile(
            r"'prerequisitePages' is a required property"
            r"|prerequisitePages.* is too short"
            r"|\[\] is too short"
            r"|\[\] should be non-empty"
            r"|should be non-empty"
        ),
        "A required list came back empty (`[]`). EVERY required array must "
        "have at least one item: `key_claims` (>=1 grounded claim) and, for "
        "a 'prereq_set' block, `prerequisitePages` (>=1 string, each naming "
        "a PRIOR topic/skill the learner needs before this block — drawn "
        "from the source's stated prerequisites, e.g. 'list the factors of "
        "a whole number'; never restate this block's own objective as a "
        "prerequisite). Never emit an empty array for a required field.",
    ),
]


def _block_source_chunk_ids(
    source_chunks: List[Dict[str, Any]],
) -> List[str]:
    """Return the ordered, de-duplicated id set the block was given.

    The outline tier's chunk-id universe is exactly the ``source_chunks``
    list the router supplies to :meth:`OutlineProvider.generate_outline`.
    Each chunk carries its id under ``id`` (canonical) or ``chunk_id``
    (legacy); we honour both, mirroring :meth:`_render_user_prompt`'s
    ``cid = chunk.get("id") or chunk.get("chunk_id")`` extraction so the
    repair's id universe is byte-identical to the universe the model saw
    in the prompt.
    """
    seen: set[str] = set()
    ids: List[str] = []
    for chunk in source_chunks or []:
        if not isinstance(chunk, dict):
            continue
        cid = chunk.get("id") or chunk.get("chunk_id")
        if isinstance(cid, str) and cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _repair_claim_grounding(
    candidate: Dict[str, Any],
    *,
    valid_ids: List[str],
) -> Dict[str, Any]:
    """Repair empty / unresolvable ``key_claims[].source_chunk_ids``.

    Wave 2 validated-id-fallback grounding repair (ports the smoke
    harness ``_carry_grounding`` semantics into the production outline
    path). The 7B outline tier is unreliable on the per-claim
    ``source_chunk_ids`` field: it routinely emits an empty array, or a
    prose string ("Apply divisibility tests …") instead of a real chunk
    id, which then fails the strict outline schema
    (``minItems: 1`` + ``items.minLength: 1``) and exhausts the
    parse-retry budget — failing the whole block. This pass runs BEFORE
    the schema-validation rejection so the previously-failing case is
    POPULATED with real ids rather than the schema being relaxed.

    For every structured ``key_claims`` entry (``{claim, source_chunk_ids}``):

    1. Filter ``source_chunk_ids`` to the ids that name a real chunk the
       block was given (``valid_ids``) — drops the model's prose-string
       hallucinations and any fabricated ids.
    2. When a claim is left with zero valid ids, fall back to the
       block's FULL ``valid_ids`` set — the chunks the block was
       genuinely synthesized from, so legitimate block-level provenance
       (NOT fabrication; the downstream NLI scorer judges grounding
       against those real premises).

    No-ops (returns the candidate unchanged) when:
    - ``valid_ids`` is empty (block has no source chunks → fail-closed
      stays in force; we never invent provenance), or
    - ``key_claims`` is absent / not the structured object shape (legacy
      flat-string arrays are left for the ``oneOf`` legacy arm).

    Returns ``(candidate, repaired, n_cited, n_valid, n_fallback)`` via a
    small dict so the caller can fold the repair signals into the
    per-call decision capture.
    """
    repair_meta: Dict[str, Any] = {
        "repaired": False,
        "n_cited": 0,
        "n_valid": 0,
        "n_fallback_claims": 0,
    }
    if not valid_ids:
        # No source chunks supplied → never fabricate provenance; the
        # existing fail-closed schema rejection still fires downstream.
        candidate["_grounding_repair"] = repair_meta
        return candidate

    valid_set = set(valid_ids)
    key_claims = candidate.get("key_claims")
    if not isinstance(key_claims, list):
        candidate["_grounding_repair"] = repair_meta
        return candidate

    n_cited = 0
    n_valid = 0
    n_fallback = 0
    repaired = False
    for claim in key_claims:
        if not isinstance(claim, dict):
            # Legacy flat-string arm — leave untouched for the oneOf.
            continue
        cited = claim.get("source_chunk_ids")
        cited_list = cited if isinstance(cited, list) else []
        # Keep only ids the block was genuinely given; preserve order +
        # de-dup so the repaired array round-trips the strict schema.
        seen: set[str] = set()
        kept: List[str] = []
        for cid in cited_list:
            n_cited += 1
            if isinstance(cid, str) and cid in valid_set and cid not in seen:
                seen.add(cid)
                kept.append(cid)
                n_valid += 1
        if not kept:
            # Model cited nothing resolvable → ground in the block's own
            # source chunks (legitimate block-level provenance).
            kept = list(valid_ids)
            n_fallback += 1
        if kept != cited_list:
            claim["source_chunk_ids"] = kept
            repaired = True

    repair_meta.update(
        repaired=repaired,
        n_cited=n_cited,
        n_valid=n_valid,
        n_fallback_claims=n_fallback,
    )
    candidate["_grounding_repair"] = repair_meta
    return candidate


# Objective-echo repair (2026-07). A key_claim is an OBJECTIVE ECHO when its
# normalized text is a near-verbatim match of ANY learning-objective statement
# available to the call. Near-verbatim = normalized exact match OR
# token-Jaccard >= this threshold (catches word-order shuffles / trivial
# paraphrases of the CO statement). Such claims are vacuous for grounding, NLI
# scoring, and downstream assessment/training synthesis, so they are dropped.
_OBJECTIVE_ECHO_JACCARD_THRESHOLD: float = 0.85

_OBJECTIVE_ECHO_WARNING: str = "OBJECTIVE_ECHO_CLAIMS"

# Word-token splitter shared by the echo normalizer (alphanumerics only, so
# punctuation / commas in "51,493" don't skew the token-Jaccard).
_ECHO_TOKEN_RE: "re.Pattern[str]" = re.compile(r"[a-z0-9]+")


def _normalize_echo_text(text: Any) -> str:
    """Normalize claim / objective text for the echo comparison.

    Lowercase, collapse internal whitespace, strip a single trailing period,
    and trim — so "Identify the place value of each digit in a given number."
    and a word-order shuffle of the same sentence normalize to comparable
    forms. Non-string inputs coerce to ``""``.
    """
    low = re.sub(r"\s+", " ", str(text or "").strip().lower()).strip()
    return low[:-1].strip() if low.endswith(".") else low


def _echo_tokens(normalized: str) -> frozenset[str]:
    return frozenset(_ECHO_TOKEN_RE.findall(normalized))


def _is_objective_echo(
    claim_text: str,
    *,
    objective_norms: List[str],
    objective_token_sets: List[frozenset[str]],
) -> bool:
    """True when ``claim_text`` near-verbatim-matches any objective statement.

    Near-verbatim = normalized exact match OR token-Jaccard >=
    :data:`_OBJECTIVE_ECHO_JACCARD_THRESHOLD` against ANY objective. A claim
    that merely SHARES objective vocabulary but asserts a concrete fact
    ("The place value of 4 in 51,493 is hundreds") stays well below the
    Jaccard floor and survives.
    """
    norm = _normalize_echo_text(claim_text)
    if not norm:
        return False
    if norm in objective_norms:
        return True
    claim_tokens = _echo_tokens(norm)
    if not claim_tokens:
        return False
    for obj_tokens in objective_token_sets:
        if not obj_tokens:
            continue
        union = claim_tokens | obj_tokens
        if not union:
            continue
        jaccard = len(claim_tokens & obj_tokens) / len(union)
        if jaccard >= _OBJECTIVE_ECHO_JACCARD_THRESHOLD:
            return True
    return False


def _claim_text_of(claim: Any) -> str:
    """Extract the prose text from a structured ``{claim, ...}`` object or a
    legacy flat-string claim."""
    if isinstance(claim, dict):
        return str(claim.get("claim") or "")
    if isinstance(claim, str):
        return claim
    return ""


def _drop_objective_echo_claims(
    candidate: Dict[str, Any],
    *,
    block_type: str,
    objectives: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop ``key_claims`` that are near-verbatim restatements of an objective.

    A 7B outline tier lazily echoes a learning-objective statement verbatim
    into ``key_claims`` on thin / exercise-grounded pages (measured: 7% of
    outline-tier claims on a real full-book run). An objective-echo claim is
    vacuous for grounding, NLI scoring, and downstream assessment/training
    synthesis, so it is dropped here (BEFORE the strict validator; same
    pre-validation point as the sibling ``_repair_*`` helpers).

    Semantics:

    - ``block_type == "objective"`` is EXEMPT (restating the objective is that
      block's whole job) — no-op.
    - No-op when ``key_claims`` is absent / not a list / empty, or when no
      objective statements are in scope.
    - Handles BOTH the structured ``[{claim, source_chunk_ids}]`` arm and the
      legacy flat-string arm (echo detection reads the claim TEXT of each).
    - NEVER drops a block to zero claims: if filtering would empty the array,
      the highest-information surviving claim is kept — the longest non-echo
      when one exists; if ALL claims are echoes, the FIRST claim is kept and a
      structural_warnings entry (``OBJECTIVE_ECHO_CLAIMS``) is appended INSTEAD
      of dropping.

    No fabrication: this pass only DROPS echo claims; it never mints a claim.
    Stashes its signals under the transient ``_objective_echo_repair`` key
    (popped by the caller before validation, mirroring ``_grounding_repair``).
    """
    repair_meta: Dict[str, Any] = {
        "repaired": False,
        "n_dropped": 0,
        "n_objective_echo_warned": 0,
    }
    if block_type == "objective":
        candidate["_objective_echo_repair"] = repair_meta
        return candidate

    key_claims = candidate.get("key_claims")
    if not isinstance(key_claims, list) or not key_claims:
        candidate["_objective_echo_repair"] = repair_meta
        return candidate

    objective_norms: List[str] = []
    objective_token_sets: List[frozenset[str]] = []
    for obj in objectives or []:
        stmt = (obj or {}).get("statement") or (obj or {}).get("text") or ""
        norm = _normalize_echo_text(stmt)
        if not norm:
            continue
        objective_norms.append(norm)
        objective_token_sets.append(_echo_tokens(norm))
    if not objective_norms:
        candidate["_objective_echo_repair"] = repair_meta
        return candidate

    non_echoes: List[Any] = []
    echoes: List[Any] = []
    for claim in key_claims:
        text = _claim_text_of(claim)
        if text and _is_objective_echo(
            text,
            objective_norms=objective_norms,
            objective_token_sets=objective_token_sets,
        ):
            echoes.append(claim)
        else:
            non_echoes.append(claim)

    if not echoes:
        candidate["_objective_echo_repair"] = repair_meta
        return candidate

    n_dropped = 0
    n_warned = 0
    if non_echoes:
        # Some real claims survive → drop every echo.
        kept = non_echoes
        n_dropped = len(echoes)
    else:
        # ALL claims are echoes → never empty the block. Keep the FIRST claim
        # and flag the whole block instead of dropping to zero.
        kept = [key_claims[0]]
        n_warned = 1
        warnings = candidate.get("structural_warnings")
        if not isinstance(warnings, list):
            warnings = []
        if _OBJECTIVE_ECHO_WARNING not in warnings:
            warnings = warnings + [_OBJECTIVE_ECHO_WARNING]
        candidate["structural_warnings"] = warnings

    candidate["key_claims"] = kept
    repair_meta.update(
        repaired=True,
        n_dropped=n_dropped,
        n_objective_echo_warned=n_warned,
    )
    candidate["_objective_echo_repair"] = repair_meta
    return candidate


# Compiled once: the per-item CURIE surface-form check (mirrors the
# schema's ``curies.items.pattern``). Used by ``_repair_outline_curies``
# to drop URL-CURIEs / malformed entries before the strict validator
# sees them.
_CURIE_PATTERN_RE: "re.Pattern[str]" = re.compile(_CURIE_PATTERN)


def _repair_outline_key_claims_shape(
    candidate: Dict[str, Any],
    *,
    valid_ids: List[str],
) -> Dict[str, Any]:
    """Coerce a MIXED ``key_claims`` array to the single all-object arm.

    The outline schema's ``key_claims`` is a back-compat ``oneOf``: EITHER
    an all-string array OR an all-object ``[{claim, source_chunk_ids[]}]``
    array. A 7B model routinely emits a MIXED array — structured claim
    objects PLUS a stray bare string (observed live: an object-claim list
    with a bare chunk-id string appended). A mixed array satisfies NEITHER
    ``oneOf`` arm, so the strict Draft 2020-12 validator rejects it
    (``'<chunk-id>' is not of type 'object'``) and the parse-retry budget
    is exhausted → the block escalates with no prose.

    This pass runs BEFORE the strict validator (same pre-validation point
    as :func:`_repair_claim_grounding`, which it must run BEFORE so the
    grounding repair sees a consistently-object-shaped array). It coerces
    a mixed array to the all-object arm — the shape the system prompt
    mandates and the shape Wave 1.5 W1.5.C per-claim attribution + Wave 2
    W2.F NLI scoring consume:

    - Object items (``{claim, source_chunk_ids}``) are kept verbatim.
    - A bare string that names a real chunk the block was given
      (``valid_ids``) is a stray source-reference, NOT a claim — it is
      DROPPED (the per-claim ``source_chunk_ids`` field is where chunk
      ids belong; a top-level bare chunk-id is malformed model output).
    - A bare string that is NOT a chunk-id is a genuine prose claim
      emitted under the legacy flat arm; it is WRAPPED into an object
      ``{claim: <text>, source_chunk_ids: []}``. The empty
      ``source_chunk_ids`` is then POPULATED by the downstream
      :func:`_repair_claim_grounding` fallback (legitimate block-level
      provenance) — never fabricated here.

    No-ops (returns the candidate untouched) when:

    - ``key_claims`` is absent / not a list, OR
    - the array is HOMOGENEOUS (all-string or all-object) — a clean
      single-arm array already validates; we never disturb it.

    No fabrication: this pass only DROPS stray chunk-id strings and
    RESHAPES genuine prose strings into the canonical object envelope.
    It never invents a claim. If coercion leaves the array below the
    per-block-type ``minItems`` bound, that block legitimately fails
    the strict validator downstream (fail-closed) — correct.

    Stashes its signals under the transient ``_key_claims_shape_repair``
    key (the caller pops it before validation + before the Block content
    lands, mirroring the ``_grounding_repair`` convention).
    """
    repair_meta: Dict[str, Any] = {
        "repaired": False,
        "n_dropped_chunk_id_strings": 0,
        "n_wrapped_prose_strings": 0,
    }
    key_claims = candidate.get("key_claims")
    if not isinstance(key_claims, list) or not key_claims:
        candidate["_key_claims_shape_repair"] = repair_meta
        return candidate

    has_object = any(isinstance(c, dict) for c in key_claims)
    has_string = any(isinstance(c, str) for c in key_claims)
    # Only a MIXED array (object items + string items) trips the oneOf;
    # a homogeneous array already matches one arm — leave it untouched.
    if not (has_object and has_string):
        candidate["_key_claims_shape_repair"] = repair_meta
        return candidate

    valid_set = set(valid_ids)
    n_dropped = 0
    n_wrapped = 0
    repaired_claims: List[Any] = []
    for c in key_claims:
        if isinstance(c, dict):
            repaired_claims.append(c)
            continue
        if isinstance(c, str):
            if c in valid_set:
                # Stray chunk-id string masquerading as a claim → drop.
                n_dropped += 1
                continue
            text = c.strip()
            if not text:
                # Empty / whitespace-only string is neither a claim nor a
                # chunk-id → drop (never wrap into an empty-claim object,
                # which the schema's ``claim.minLength: 1`` would reject).
                n_dropped += 1
                continue
            # Genuine prose claim under the legacy flat arm → wrap into
            # the object envelope. source_chunk_ids left empty; the
            # downstream grounding repair populates it from valid_ids.
            repaired_claims.append({"claim": text, "source_chunk_ids": []})
            n_wrapped += 1
            continue
        # Any other type (number / null / list) — drop; the schema would
        # have rejected it under both arms anyway.
        n_dropped += 1

    candidate["key_claims"] = repaired_claims
    repair_meta.update(
        repaired=True,
        n_dropped_chunk_id_strings=n_dropped,
        n_wrapped_prose_strings=n_wrapped,
    )
    candidate["_key_claims_shape_repair"] = repair_meta
    return candidate


def _repair_outline_curies(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``curies`` array entries that violate ``_CURIE_PATTERN``.

    The outline schema requires every ``curies`` item to match
    ``_CURIE_PATTERN`` (``^[a-z][a-z0-9]*:[A-Za-z0-9_-]+$``). A 7B model
    emits URL-CURIEs (observed live: ``schema:https://example.com/vocab``
    — contains ``/`` and ``.``) which violate the pattern, so the strict
    validator rejects the whole payload
    (``'schema:https://...' does not match '^[a-z]...'``) and the
    parse-retry budget is exhausted → the block escalates with no prose.

    This pass runs BEFORE the strict validator (same pre-validation point
    as :func:`_repair_claim_grounding`). It filters ``curies`` to only the
    entries matching ``_CURIE_PATTERN``, dropping malformed / URL-CURIE
    entries. ``curies`` carries no ``minItems`` bound for any block type
    (verified against ``_build_block_outline_schema``), so emptying the
    array is schema-valid — and the downstream ``_mint_outline_curies``
    backstop (run at the phase handler) still mints per-course domain
    CURIEs onto an empty list, so dropping garbage here never starves a
    prose block of grounding.

    No fabrication: this pass only DROPS malformed entries; it never
    invents or rewrites a CURIE. A URL-CURIE is discarded outright (we do
    not attempt to salvage a prefix from it — that would be fabrication).

    No-ops when ``curies`` is absent / not a list. Stashes its signals
    under the transient ``_curie_shape_repair`` key (popped by the caller
    before validation, mirroring the ``_grounding_repair`` convention).
    """
    repair_meta: Dict[str, Any] = {
        "repaired": False,
        "n_dropped_malformed": 0,
        "n_kept": 0,
    }
    curies = candidate.get("curies")
    if not isinstance(curies, list):
        candidate["_curie_shape_repair"] = repair_meta
        return candidate

    kept: List[str] = []
    n_dropped = 0
    for c in curies:
        if isinstance(c, str) and _CURIE_PATTERN_RE.match(c):
            kept.append(c)
        else:
            n_dropped += 1

    if n_dropped:
        candidate["curies"] = kept
        repair_meta.update(
            repaired=True,
            n_dropped_malformed=n_dropped,
            n_kept=len(kept),
        )
    else:
        repair_meta["n_kept"] = len(kept)
    candidate["_curie_shape_repair"] = repair_meta
    return candidate


def _repair_outline_source_refs(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce malformed ``source_refs`` items to the canonical shape.

    The outline schema requires every ``source_refs`` item be an object
    ``{"sourceId": str(minLength 1), "role": str(minLength 1)}`` with BOTH
    keys required. A 7B model routinely emits the WRONG primitive shape
    here — a bare chunk-id string per item
    (``source_refs: ["semantik:slug#chunk_a", ...]``) or an object that names a
    ``sourceId`` but drops the required ``role`` — so the strict validator
    rejects the whole payload and the parse-retry budget is exhausted → the
    block escalates with no prose.

    This pass runs BEFORE the strict validator (same pre-validation point as
    the other repairs). Per item:

    - A bare non-empty string is WRAPPED to
      ``{"sourceId": <string>, "role": "primary"}``. ``"primary"`` is a
      defensible default role, NOT fabricated provenance — the ``sourceId``
      is the model's OWN cited id, carried through verbatim; only the
      missing role primitive is supplied (the model already asserted the
      grounding, it merely omitted the role enum).
    - An object carrying a usable ``sourceId`` but a missing / empty /
      non-string ``role`` gets ``role="primary"`` set.
    - An item with no usable ``sourceId`` (empty / whitespace-only string,
      ``None``, a number, an object with no non-empty ``sourceId``) is
      DROPPED. ``source_refs`` carries NO ``minItems`` bound for any block
      type (verified against :func:`_build_block_outline_schema`), so an
      emptied array is schema-valid; downstream source-grounding gates
      catch a prose block that legitimately lost all provenance.

    No fabrication: no ``sourceId`` is ever invented — the only value
    supplied is the deterministic ``role="primary"`` default for an id the
    model already cited.

    No-ops when ``source_refs`` is absent / not a list. Stashes its signals
    under the transient ``_source_refs_shape_repair`` key (popped by the
    caller before validation, mirroring the ``_grounding_repair`` convention).
    """
    repair_meta: Dict[str, Any] = {
        "repaired": False,
        "n_wrapped_bare_strings": 0,
        "n_role_backfilled": 0,
        "n_dropped_malformed": 0,
        "n_kept": 0,
    }
    source_refs = candidate.get("source_refs")
    if not isinstance(source_refs, list):
        candidate["_source_refs_shape_repair"] = repair_meta
        return candidate

    repaired_refs: List[Dict[str, str]] = []
    n_wrapped = 0
    n_role = 0
    n_dropped = 0
    for ref in source_refs:
        if isinstance(ref, str):
            source_id = ref.strip()
            if not source_id:
                # Empty / whitespace-only string — no usable id → drop.
                n_dropped += 1
                continue
            # Bare cited id under the legacy flat arm → wrap with the
            # default role. The id is the model's own; role is the only
            # supplied primitive.
            repaired_refs.append({"sourceId": source_id, "role": "primary"})
            n_wrapped += 1
            continue
        if isinstance(ref, dict):
            raw_id = ref.get("sourceId")
            source_id = raw_id.strip() if isinstance(raw_id, str) else ""
            if not source_id:
                # Object with no usable sourceId — nothing to anchor → drop.
                n_dropped += 1
                continue
            raw_role = ref.get("role")
            role = raw_role.strip() if isinstance(raw_role, str) else ""
            if not role:
                role = "primary"
                n_role += 1
            repaired_refs.append({"sourceId": source_id, "role": role})
            continue
        # Any other type (number / null / list) — no usable sourceId → drop.
        n_dropped += 1

    if n_wrapped or n_role or n_dropped:
        candidate["source_refs"] = repaired_refs
        repair_meta.update(
            repaired=True,
            n_wrapped_bare_strings=n_wrapped,
            n_role_backfilled=n_role,
            n_dropped_malformed=n_dropped,
            n_kept=len(repaired_refs),
        )
    else:
        repair_meta["n_kept"] = len(repaired_refs)
    candidate["_source_refs_shape_repair"] = repair_meta
    return candidate


# A page-id-shaped garbage token the 7B sometimes emits for a prereq page
# instead of a real prior-topic name (observed live: ``'p#factors_x_0'`` —
# a `{namespace}#{slug}` chunk-id fragment, NOT a prerequisite topic). Used
# by :func:`_repair_prereq_pages` to strip such entries before backfilling.
_PREREQ_PAGE_GARBAGE_RE: "re.Pattern[str]" = re.compile(r"^[a-z]+#")

# Stop-words dropped from extracted prerequisite noun-phrases so a phrase like
# "list the factors of a whole number" yields "factors of a whole number"
# (the leading imperative verb + article are not part of the topic name).
_PREREQ_LEADING_VERBS: "frozenset[str]" = frozenset(
    {"list", "find", "know", "identify", "recall", "compute", "determine",
     "calculate", "understand", "recognize", "state", "define", "explain"}
)


def _extract_prereq_phrases_from_source(
    source_chunks: List[Dict[str, Any]],
) -> List[str]:
    """Pull prerequisite topic phrases out of an explicit Prerequisites sentence.

    Many textbook sections name their prerequisites in prose — the clean
    fractions source literally says::

        Prerequisites: before simplifying fractions a learner must know how to
        list the factors of a whole number and how to find the greatest common
        factor (GCF) of two numbers.

    This scrapes the clause AFTER a ``Prerequisites``/``Prerequisite`` marker
    and splits it on ``how to`` / ``and`` / ``;`` / ``,`` boundaries into
    discrete topic phrases, then trims a leading imperative verb + article
    (``list the factors …`` → ``factors of a whole number``). The result is
    grounded entirely in the source text — no fabrication; an absent marker
    yields an empty list and the caller falls back to ``key_terms``.
    """
    phrases: List[str] = []
    seen: set[str] = set()
    for chunk in source_chunks or []:
        if not isinstance(chunk, dict):
            continue
        body = str(chunk.get("body") or chunk.get("text") or "")
        if not body:
            continue
        # Find the prerequisites clause: text after the marker up to the next
        # sentence-terminating period that ends the clause (newline or '. ').
        m = re.search(
            r"prerequisites?\s*[:\-]?\s*(.+?)(?:\n|\. (?=[A-Z])|$)",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            continue
        clause = m.group(1)
        # Strip the common "before <topic> a learner must [know/be able to]"
        # framing preamble so only the enumerated topics survive the split.
        clause = re.sub(
            r"^\s*before\b.*?\b(?:must|need to|should)\b"
            r"(?:\s+(?:know|be able to))?\s*(?:how to\s+)?",
            "",
            clause,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Normalize the recurring "how to" connector (and "and how to") to a
        # plain list delimiter so each enumerated topic splits cleanly.
        clause = re.sub(r"\b(?:and\s+)?how to\b", "||", clause, flags=re.IGNORECASE)
        # Split into candidate topic phrases on the natural list delimiters.
        for raw in re.split(r"\|\||\s+and\s+|;|,", clause):
            phrase = raw.strip().rstrip(".").strip()
            if not phrase:
                continue
            # Drop any leading imperative verb + article so the phrase names
            # the TOPIC, not the instruction ("list the factors …" →
            # "factors …").
            tokens = phrase.split()
            while tokens and tokens[0].lower() in _PREREQ_LEADING_VERBS:
                tokens = tokens[1:]
            while tokens and tokens[0].lower() in ("the", "a", "an"):
                tokens = tokens[1:]
            phrase = " ".join(tokens).strip()
            # Require ≥2 tokens so stray framing fragments don't slip through.
            if not phrase or len(phrase.split()) < 2:
                continue
            # Drop any residual framing fragment (a phrase still carrying the
            # "learner must" / "a learner" preamble is not a topic name).
            if re.search(r"\blearner\b|\bmust\b|\bbefore\b", phrase, re.IGNORECASE):
                continue
            low = phrase.lower()
            if low in seen:
                continue
            seen.add(low)
            phrases.append(phrase)
    return phrases


def _repair_prereq_pages(
    candidate: Dict[str, Any],
    *,
    block_type: str,
    source_chunks: List[Dict[str, Any]],
    key_terms: Tuple[str, ...],
    objectives: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Backfill / clean an empty or garbage ``prereq_set`` ``prerequisitePages``.

    The ``prereq_set`` outline schema marks ``prerequisitePages`` REQUIRED with
    ``minItems: 1`` (every entry a non-empty string). The local 7B is
    unreliable here: investigation across 5 trials showed the model emits
    ``prerequisitePages`` as ``None`` (missing) or ``[]`` (empty) on most early
    attempts — ``key_claims`` is always populated, so the empty array is
    EXCLUSIVELY ``prerequisitePages`` — and only stochastically recovers,
    tripping the strict validator (``'[] should be non-empty'``) and exhausting
    the parse-retry budget → the block degrades to a thin stub. Even on success
    the 7B sometimes emits page-id-shaped garbage (``'p#factors_x_0'``) instead
    of a real prior-topic name.

    This pass runs BEFORE the strict validator (same pre-validation point as
    the other ``_repair_*`` helpers) so the previously-failing payload is
    POPULATED rather than the schema being relaxed. Two deterministic steps,
    grounded only in the block's own source / key_terms (no fabrication):

    1. **Strip garbage** — drop any existing entry that is empty / non-string
       or page-id-shaped (matches :data:`_PREREQ_PAGE_GARBAGE_RE`, e.g.
       ``'p#factors_x_0'``). Order + dedup preserved on the survivors.
    2. **Backfill when empty** — if (1) leaves the array empty/missing, fill it
       from available signal, best-first:
         a. Prerequisite topic phrases scraped from an explicit
            ``Prerequisites:`` sentence in the source chunks
            (:func:`_extract_prereq_phrases_from_source`) — the source
            literally names them, so this is grounded extraction.
         b. Fallback to the block's ``key_terms`` (the terms the page is
            built around — a defensible prior-topic proxy).
       NEVER backfilled with the block's own ``objectives`` statement text
       (a known-bad 7B output — the objective is what the page TEACHES, not a
       prerequisite). If neither source signal exists, the array is LEFT
       empty and the strict validator still fails closed (no fabrication).

    No-ops for any non-``prereq_set`` block, or when ``prerequisitePages`` is
    already a non-empty list of clean (non-garbage) strings. Stashes its
    signals under the transient ``_prereq_pages_repair`` key (popped by the
    caller before validation, mirroring the ``_grounding_repair`` convention).
    """
    repair_meta: Dict[str, Any] = {
        "repaired": False,
        "n_garbage_stripped": 0,
        "backfill_source": None,
        "n_backfilled": 0,
    }
    if block_type != "prereq_set":
        candidate["_prereq_pages_repair"] = repair_meta
        return candidate

    raw = candidate.get("prerequisitePages")
    raw_list = raw if isinstance(raw, list) else []

    # Step 1: keep only clean, non-garbage, non-empty string entries.
    seen: set[str] = set()
    kept: List[str] = []
    n_garbage = 0
    for entry in raw_list:
        if not isinstance(entry, str):
            n_garbage += 1
            continue
        val = entry.strip()
        if not val or _PREREQ_PAGE_GARBAGE_RE.match(val):
            n_garbage += 1
            continue
        if val.lower() in seen:
            continue
        seen.add(val.lower())
        kept.append(val)

    backfill_source: Optional[str] = None
    n_backfilled = 0
    if not kept:
        # Step 2: backfill from grounded signal. NEVER from the objective
        # statement (a known-bad output — that is what the page teaches).
        objective_texts = {
            str((o or {}).get("statement") or (o or {}).get("text") or "")
            .strip()
            .lower()
            for o in (objectives or [])
        }
        candidates: List[str] = _extract_prereq_phrases_from_source(source_chunks)
        if candidates:
            backfill_source = "source_prerequisites"
        else:
            candidates = [t.strip() for t in (key_terms or ()) if str(t).strip()]
            if candidates:
                backfill_source = "key_terms"
        fill_seen: set[str] = set()
        for cand in candidates:
            low = cand.lower()
            if not cand or low in fill_seen or low in objective_texts:
                continue
            fill_seen.add(low)
            kept.append(cand)
            n_backfilled += 1

    if kept != raw_list:
        candidate["prerequisitePages"] = kept
        repair_meta.update(
            repaired=True,
            n_garbage_stripped=n_garbage,
            backfill_source=backfill_source,
            n_backfilled=n_backfilled,
        )
    candidate["_prereq_pages_repair"] = repair_meta
    return candidate


def _repair_assessment_item_payload(
    candidate: Dict[str, Any],
    *,
    block_type: str,
) -> Dict[str, Any]:
    """Reconcile the assessment_item ``distractors`` / ``answer_key`` /
    ``correct_answer_index`` trio so it satisfies the downstream validators.

    The outline PROMPT (``§ assessment_item branch``) instructs the 7B to emit
    ``distractors[]`` as WRONG-only options plus a SEPARATE ``answer_key`` (the
    correct value) and a ``correct_answer_index``. But BOTH downstream
    validators read the trio the OTHER way: the correct answer must LIVE IN
    ``distractors[]`` at ``correct_answer_index`` — i.e.
    ``distractors[correct_answer_index]["text"] == answer_key`` with the index
    in ``[0, len(distractors))``. ``assessment_item_payload``'s
    ``ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE`` and
    ``assessment_retrieval_grounding`` (which indexes
    ``distractors[correct_answer_index]["text"]``) both depend on it. A model
    that follows the prompt literally (wrong-only distractors + an index
    pointing PAST them at the answer, e.g. 2 distractors + ``index=2``)
    therefore fails a CRITICAL gate and stops the whole
    ``inter_tier_validation`` phase on a single block.

    This pass reconciles the two interpretations deterministically and WITHOUT
    fabrication — it uses only the model's own ``answer_key`` + ``distractors``
    text — by ENFORCING the invariant
    ``distractors[correct_answer_index]["text"] == answer_key`` whenever
    ``answer_key`` is present. Two failure shapes this catches (BOTH observed
    on a real 7B run):

    * **Out-of-range index** — wrong-only distractors + an index pointing PAST
      them at the separate answer (2 distractors + ``index=2``). Fails the
      gate's range check outright.
    * **In-range index on the WRONG option** — wrong-only distractors + an
      index that lands on a wrong distractor by luck (the correct answer is in
      ``answer_key``, absent from ``distractors``). Passes the gate's range
      check but marks a WRONG answer correct — an incoherent quiz.

    The reconciliation:

    * If ``answer_key`` already appears among the distractor texts
      (whitespace-normalized compare, so a ``"a + b"`` / ``"a+b"`` formatting
      variant is NOT duplicated), point ``correct_answer_index`` at that entry.
      Idempotent on an already-correct block (index unchanged).
    * Otherwise INSERT ``answer_key`` into ``distractors`` at the clamped
      ``correct_answer_index`` (the model's intended slot in the combined
      option list, ``[0, len(distractors)]``) and set the index there —
      yielding the invariant the rewrite tier renders as
      ``<li data-cf-distractor-index="N" data-cf-correct="true">``.

    No-op for every non-assessment_item block, and when ``distractors`` is
    absent / the wrong shape (the strict validator below still fails closed on
    a genuinely empty/short payload). Stashes its signals under the transient
    ``_assessment_item_payload_repair`` key (popped by the caller).
    """
    repair_meta: Dict[str, Any] = {"repaired": False, "mode": None}
    if block_type != "assessment_item":
        candidate["_assessment_item_payload_repair"] = repair_meta
        return candidate

    distractors = candidate.get("distractors")
    if not isinstance(distractors, list) or not distractors:
        candidate["_assessment_item_payload_repair"] = repair_meta
        return candidate

    cai = candidate.get("correct_answer_index")
    cai_int = cai if isinstance(cai, int) and not isinstance(cai, bool) else None

    def _norm(s: str) -> str:
        return " ".join(s.split()).replace(" ", "").lower()

    answer_key = candidate.get("answer_key")
    if isinstance(answer_key, str) and answer_key.strip():
        ak = answer_key.strip()
        ak_norm = _norm(ak)
        texts_norm = [
            _norm(e["text"])
            for e in distractors
            if isinstance(e, dict) and isinstance(e.get("text"), str)
        ]
        if ak_norm in texts_norm:
            # Correct answer already a distractor → point the index at it
            # (no-op when the model already aligned them).
            idx = texts_norm.index(ak_norm)
            if cai_int != idx:
                candidate["correct_answer_index"] = idx
                repair_meta.update(repaired=True, mode="reindex_to_existing")
        else:
            # Answer absent from distractors → insert it at the model's
            # intended (clamped) slot so the invariant holds.
            pos = (
                cai_int
                if (cai_int is not None and 0 <= cai_int <= len(distractors))
                else len(distractors)
            )
            distractors.insert(pos, {"text": ak})
            candidate["distractors"] = distractors
            candidate["correct_answer_index"] = pos
            repair_meta.update(
                repaired=True, mode="insert_answer_key", inserted_at=pos
            )
    else:
        # No usable answer_key — clamp an out-of-range index into the existing
        # range so the range check passes (the similarity / grounding
        # validators still gate the substance).
        if cai_int is None or cai_int < 0 or cai_int >= len(distractors):
            candidate["correct_answer_index"] = 0
            repair_meta.update(repaired=True, mode="clamp_index_no_answer_key")

    candidate["_assessment_item_payload_repair"] = repair_meta
    return candidate


def _repair_outline_bloom_level(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a numeric / mis-cased ``bloom_level`` to the canonical enum.

    The outline schema pins ``bloom_level`` to :data:`_BLOOM_LEVEL_ENUM`
    (the six canonical lowercase Bloom levels, byte-identical to
    :data:`lib.ontology.bloom.BLOOM_LEVELS`). A 7B model sometimes emits a
    numeric tier (``bloom_level: 2`` / ``"2"``) or a capitalized /
    whitespace-padded form (``"Apply"``), all of which the strict validator
    rejects.

    This pass runs BEFORE the strict validator. Two deterministic
    normalizations of the model's OWN output (no fabrication):

    - A numeric tier 1..6 (int or numeric string) maps to
      ``BLOOM_LEVELS[tier - 1]`` — the canonical 1=remember … 6=create
      ordering sourced from the single-source-of-truth tuple in
      ``lib/ontology/bloom.py`` (NOT a hardcoded map; reusing the canonical
      loader keeps this in lockstep if the ontology ever re-orders).
    - A string that case-insensitively / whitespace-insensitively matches a
      canonical enum value is normalized to that value (``"Apply  "`` →
      ``"apply"``).

    Any value that cannot be mapped to a valid enum member is LEFT
    UNCHANGED — the strict validator then fails closed (fail-closed; no
    fabrication of an arbitrary level).

    No-ops when ``bloom_level`` is absent. Stashes its signals under the
    transient ``_bloom_level_repair`` key (popped by the caller before
    validation, mirroring the ``_grounding_repair`` convention).
    """
    repair_meta: Dict[str, Any] = {
        "repaired": False,
        "original": None,
        "normalized": None,
    }
    if "bloom_level" not in candidate:
        candidate["_bloom_level_repair"] = repair_meta
        return candidate

    original = candidate.get("bloom_level")
    repair_meta["original"] = original
    normalized: Optional[str] = None

    # Numeric tier (int or numeric string) → canonical level by 1-based
    # index into the single-source-of-truth ``BLOOM_LEVELS`` tuple.
    tier: Optional[int] = None
    if isinstance(original, bool):
        # bool is an int subclass — never treat True/False as a tier.
        tier = None
    elif isinstance(original, int):
        tier = original
    elif isinstance(original, str):
        stripped = original.strip()
        if stripped.isdigit():
            tier = int(stripped)
        else:
            # Case-/whitespace-insensitive enum match.
            lowered = stripped.lower()
            for level in _BLOOM_LEVEL_ENUM:
                if lowered == level:
                    normalized = level
                    break

    if tier is not None and 1 <= tier <= len(_BLOOM_LEVELS):
        normalized = _BLOOM_LEVELS[tier - 1]

    if normalized is not None and normalized != original:
        candidate["bloom_level"] = normalized
        repair_meta.update(repaired=True, normalized=normalized)
    else:
        repair_meta["normalized"] = normalized

    candidate["_bloom_level_repair"] = repair_meta
    return candidate


def _surface_key_claims_min_items(
    exc: "jsonschema.ValidationError",
) -> Optional[str]:
    """Surface a masked ``key_claims`` minItems sub-error from a ``oneOf``.

    ``key_claims`` is a ``oneOf`` (legacy ``List[str]`` vs structured
    ``List[{claim, source_chunk_ids}]``). When the 7B emits a single
    well-formed structured claim against a block whose bound demands ≥2,
    jsonschema's ``best_match`` raises the TOP-LEVEL ``oneOf`` error
    ("... is not valid under any of the given schemas"), masking the
    real minItems cause. jsonschema attaches every per-arm sub-error to
    ``exc.context``; this walks them and returns the message of the most
    relevant minItems / "too short" / "too few items" sub-error on the
    STRUCTURED arm (a `key_claims` sub-error whose `validator == 'minItems'`
    or whose message reads "too short"), prefixed with ``key_claims:`` so
    the directive matcher's key_claims-scoped pattern fires.

    Returns ``None`` when the raised error is not the ``key_claims``
    top-level ``oneOf`` or no matching sub-error exists — message-building
    only; never alters validation logic.
    """
    # Only act on the top-level key_claims oneOf rejection. ``exc.path``
    # is a deque; its first element is the failing property name.
    top_path = list(getattr(exc, "absolute_path", []) or getattr(exc, "path", []))
    is_key_claims = bool(top_path) and top_path[0] == "key_claims"
    is_oneof = "is not valid under any of the given schemas" in str(exc.message)
    if not (is_key_claims and is_oneof):
        return None
    best: Optional["jsonschema.ValidationError"] = None
    for sub in getattr(exc, "context", None) or []:
        validator = getattr(sub, "validator", None)
        message = str(getattr(sub, "message", ""))
        is_min = validator == "minItems" or "too short" in message.lower() or (
            "too few items" in message.lower()
        )
        if is_min:
            # Prefer the FIRST minItems sub-error on the structured arm;
            # any minItems miss is on the structured (object-list) arm
            # since the legacy str arm has no minItems constraint.
            best = sub
            break
    if best is None:
        return None
    return f"key_claims: {str(best.message)}"


def _match_retry_directive(
    last_error: str, block_type: Optional[str] = None
) -> Optional[str]:
    """Return the directive matching ``last_error``'s validator pattern.

    Walks :data:`_RETRY_DIRECTIVE_PATTERNS` in declaration order and
    returns the first matching directive. Returns ``None`` when no
    pattern matches — the caller falls back to the bare validator
    error echo.

    ``block_type`` (optional) lets the matched directive interpolate the
    per-block-type ``key_claims`` minItems lower bound from
    :data:`_OUTLINE_KIND_BOUNDS` (e.g. ``explanation`` → ``2``). The
    minItems directive carries a ``{min}`` placeholder; we substitute it
    with the resolved bound (falling back to ``2`` — the smallest bound
    that can trip a minItems miss — when the block_type is unknown or
    carries no ``key_claims`` bound). Directives with no ``{min}`` token
    are returned unchanged, so the substitution is a no-op for every
    pre-existing pattern.
    """
    if not last_error:
        return None
    for pattern, directive in _RETRY_DIRECTIVE_PATTERNS:
        if pattern.search(last_error):
            if "{min}" in directive:
                bounds = _OUTLINE_KIND_BOUNDS.get(block_type or "", {})
                key_claims_bound = bounds.get("key_claims")
                min_required = (
                    key_claims_bound[0]
                    if isinstance(key_claims_bound, tuple)
                    and key_claims_bound
                    else 2
                )
                # ``{{...}}`` escapes in the directive collapse to literal
                # braces for the ``{claim, source_chunk_ids}`` shape hint;
                # only the bare ``{min}`` token interpolates.
                return directive.format(min=min_required)
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
        # Per-request HTTP timeout (seconds). Resolution chain (high →
        # low): this kwarg > ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` env
        # var > :data:`_DEFAULT_REQUEST_TIMEOUT_SECONDS` (300.0). Only
        # applies to the OpenAI-compatible backends (local / together).
        timeout: Optional[float] = None,
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

        # Resolve + collapse the legacy ``openai_compatible`` alias to
        # ``local`` at constructor entry so a standalone construction
        # behaves exactly like a router-mediated one (the router collapses
        # it in ``_get_outline_provider``). Without this, the alias would
        # reach the base's registry else-branch and raise ``UnknownEndpoint``
        # (it is NOT a registry row name). Every other value is a registry
        # endpoint the base constructs generically; we pass the collapsed
        # ``resolved_provider`` (not the raw kwarg) and do NOT pin a narrow
        # ``supported_providers`` — the base's registry-derived default
        # governs, so adding a provider stays a ``config/endpoints.yaml``
        # registry-entry change, never a subclass.
        resolved_provider = (
            provider
            or os.environ.get(ENV_PROVIDER)
            or DEFAULT_PROVIDER
        ).lower()
        if resolved_provider == _OPENAI_COMPATIBLE_ALIAS:
            resolved_provider = "local"

        # Per-call kwarg wins; otherwise source the generous default
        # from ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (fallback 300.0).
        resolved_timeout: float = (
            float(timeout) if timeout is not None
            else _resolve_request_timeout()
        )

        super().__init__(
            provider=resolved_provider,
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            capture=capture,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=resolved_timeout,
            client=client,
            anthropic_client=anthropic_client,
            env_provider_var=ENV_PROVIDER,
            default_provider=DEFAULT_PROVIDER,
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

        # outline-overflow-fix-2026-07: read the chunk-count cap + the
        # input-truncation tripwire ONCE at construction (mirrors the
        # rewrite tier's ``resolve_rewrite_num_ctx`` / ``resolve_truncation_
        # tripwire`` in ``RewriteProvider.__init__``). ``_outline_num_ctx``
        # is the served-window number the tripwire's message + ratio read;
        # it reuses the generic content-gen serving-window resolver
        # (``ED4ALL_ANSWER_NUM_CTX``, default 4096) since the outline tier
        # ships no dedicated window env.
        self._max_chunks: int = _resolve_outline_max_chunks()
        self._truncation_tripwire: bool = _resolve_outline_truncation_tripwire()
        self._outline_num_ctx: int = _resolve_num_ctx()

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

    def _dispatch_call(
        self,
        user_prompt: str,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """outline-overflow-fix-2026-07: usage-bearing dispatch override.

        Returns a 3-tuple ``(text, retry_count, usage)`` (instead of the
        base's 2-tuple) so :meth:`generate_outline` can read the server-
        reported ``usage.prompt_tokens`` for the input-truncation tripwire.
        Delegates to the base's :meth:`_dispatch_call_with_usage`; the usage
        dict is EMPTY on the Anthropic path (no per-request token tally) so
        the tripwire fail-OPENs there. Mirrors the rewrite tier's
        ``RewriteProvider._dispatch_call`` 3-tuple contract. The seam name
        is preserved so callers/tests that patch ``_dispatch_call`` stay on
        the dispatch path.
        """
        return self._dispatch_call_with_usage(
            user_prompt, extra_payload=extra_payload
        )

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

        # Wave 2 validated-id-fallback grounding repair: the block's
        # chunk-id universe is exactly the supplied ``source_chunks``
        # list (same extraction the user prompt enumerates). The repair
        # filters each ``key_claims[].source_chunk_ids`` to this set and
        # falls back to it when the model cited nothing resolvable —
        # never fabricating an id outside the block's own source chunks.
        valid_chunk_ids = _block_source_chunk_ids(source_chunks)

        last_error: Optional[str] = None
        last_raw: str = ""
        parsed: Optional[Dict[str, Any]] = None
        grounding_repair: Optional[Dict[str, Any]] = None
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
                directive = _match_retry_directive(
                    last_error, block.block_type
                )
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
                # outline-overflow-fix-2026-07: the outline provider's
                # ``_dispatch_call`` override returns a 3-tuple carrying the
                # server-reported ``usage`` so the input-truncation tripwire
                # can read ``usage.prompt_tokens``. The seam name is kept so
                # the existing transient-retry tests (which patch
                # ``_dispatch_call``) stay on the path.
                raw_text, retry_count, usage = self._dispatch_call(
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

            # outline-overflow-fix-2026-07: input-truncation tripwire.
            # Compare the server-reported ``usage.prompt_tokens`` against
            # the local 2.5-char/token estimate of system + user prompt. A
            # large shortfall means the served window silently dropped the
            # prompt HEAD (the block header + source chunks), so the model
            # authored from the surviving TAIL — a wrong-topic outline the
            # downstream gates then re-stamp green. HARD, NON-RETRYABLE:
            # re-dispatching the same over-window prompt re-truncates, so we
            # fail the call LOUDLY instead of returning a silent-success
            # stub. Gated by ``COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE``
            # (default ON); fail-OPEN when usage is absent/zero (Ollama may
            # omit it) or empty (Anthropic returns no per-request tally).
            # Mirrors the rewrite tier's ``_check_truncation``.
            if self._truncation_tripwire and isinstance(usage, dict):
                estimated_prompt_tokens = (
                    _estimate_tokens(self._system_prompt)
                    + _estimate_tokens(user_prompt)
                )
                try:
                    check_prompt_not_truncated(
                        usage.get("prompt_tokens"),
                        estimated_prompt_tokens,
                        model_id=self._model,
                        num_ctx=self._outline_num_ctx,
                    )
                except _PromptTruncatedError as exc:
                    logger.warning(
                        "OutlineProvider: input prompt truncated for block "
                        "%r (num_ctx=%d) — failing the call: %s",
                        block.block_id,
                        self._outline_num_ctx,
                        exc,
                    )
                    self._emit_per_call_decision(
                        raw_text=last_raw,
                        retry_count=total_retries,
                        block_id=block.block_id,
                        block_type=block.block_type,
                        page_id=block.page_id,
                        success=False,
                        attempts=attempt + 1,
                        last_error=(
                            f"input_prompt_truncated: estimated "
                            f"~{estimated_prompt_tokens} prompt tokens but "
                            f"server reported "
                            f"{usage.get('prompt_tokens')!r} "
                            f"(num_ctx={self._outline_num_ctx})"
                        ),
                    )
                    raise OutlineProviderError(
                        f"Outline tier detected silent input-prompt "
                        f"truncation for block {block.block_id!r}: the "
                        f"served window (num_ctx={self._outline_num_ctx}) "
                        f"dropped the prompt HEAD, so the outline is "
                        f"ungrounded. Raise the server window / "
                        f"ED4ALL_ANSWER_NUM_CTX, lower "
                        f"COURSEFORGE_OUTLINE_MAX_CHUNKS, or disable the "
                        f"guard via COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE=off.",
                        code="outline_input_truncated",
                    ) from exc

            candidate = OpenAICompatibleClient._extract_json_lenient(raw_text)
            if candidate is None:
                last_error = "lenient JSON parse returned None"
                attempt += 1
                continue
            # Malformed-shape repairs — all run BEFORE schema validation
            # so the previously-failing payload is NORMALIZED rather than
            # the schema being relaxed. Each stashes its signals under a
            # transient ``_*_repair`` key, popped off before the candidate
            # is validated (the strict schema forbids unknown keys) and
            # before it lands as the Block content. Ordering matters:
            #
            # 1. ``_repair_outline_key_claims_shape`` first — coerces a
            #    MIXED ``key_claims`` array (object claims + a stray bare
            #    chunk-id string, the live 7B failure) to the all-object
            #    arm, so the grounding repair below sees a consistently
            #    object-shaped array.
            # 2. ``_repair_outline_curies`` — drops URL-CURIEs / malformed
            #    entries (the live ``schema:https://...`` failure) that
            #    violate the per-item CURIE pattern. ``curies`` is not
            #    ``minItems``-bound, so emptying it is schema-valid; the
            #    ``_mint_outline_curies`` phase-handler backstop still
            #    mints per-course domain CURIEs onto the empty list.
            # 3. ``_repair_claim_grounding`` last — filters/populates each
            #    ``key_claims[].source_chunk_ids`` against the block's real
            #    chunk-id universe (now that the array is object-shaped).
            #
            # None of the three fabricate content; they only DROP/COERCE
            # malformed model output. A block left below ``minItems`` after
            # coercion still fails the strict validator (fail-closed).
            candidate = _repair_outline_key_claims_shape(
                candidate, valid_ids=valid_chunk_ids
            )
            key_claims_shape_repair = candidate.pop(
                "_key_claims_shape_repair", None
            )
            candidate = _repair_outline_curies(candidate)
            curie_shape_repair = candidate.pop("_curie_shape_repair", None)
            candidate = _repair_claim_grounding(
                candidate, valid_ids=valid_chunk_ids
            )
            repair_meta = candidate.pop("_grounding_repair", None)
            # 3b. ``_drop_objective_echo_claims`` — drop any key_claim whose
            #    normalized text near-verbatim matches ANY objective statement
            #    (the 7B's lazy "restate the CO into a claim" failure on thin /
            #    exercise-grounded pages). Runs AFTER grounding repair (claims
            #    are in final shape) and is EXEMPT for objective blocks. Never
            #    empties the block: keeps the highest-information survivor, or
            #    the first claim + an OBJECTIVE_ECHO_CLAIMS warning when every
            #    claim is an echo. No fabrication (drop-only).
            candidate = _drop_objective_echo_claims(
                candidate,
                block_type=block.block_type,
                objectives=objectives,
            )
            objective_echo_repair = candidate.pop(
                "_objective_echo_repair", None
            )
            if (
                isinstance(objective_echo_repair, dict)
                and objective_echo_repair.get("repaired")
            ):
                logger.warning(
                    "outline objective-echo repair fired for block %r "
                    "(type=%s): dropped %d echo claim(s), warned %d block(s) "
                    "(all-echo → kept 1 + OBJECTIVE_ECHO_CLAIMS)",
                    block.block_id,
                    block.block_type,
                    int(objective_echo_repair.get("n_dropped", 0)),
                    int(objective_echo_repair.get("n_objective_echo_warned", 0)),
                )
            # 4. ``_repair_outline_source_refs`` — wraps bare cited
            #    sourceId strings (the live 7B failure) into the canonical
            #    {sourceId, role} object and backfills a missing ``role``;
            #    drops items with no usable sourceId. ``source_refs`` has no
            #    minItems bound for prose types, so an emptied array stays
            #    schema-valid. No sourceId is fabricated.
            candidate = _repair_outline_source_refs(candidate)
            source_refs_shape_repair = candidate.pop(
                "_source_refs_shape_repair", None
            )
            # 5. ``_repair_outline_bloom_level`` — maps a numeric tier 1..6
            #    to the canonical enum and normalizes a mis-cased string;
            #    leaves an unmappable value untouched (fail-closed).
            candidate = _repair_outline_bloom_level(candidate)
            bloom_level_repair = candidate.pop("_bloom_level_repair", None)
            # 5b. ``_repair_prereq_pages`` — for ``prereq_set`` blocks only,
            #    strip page-id-shaped garbage entries and BACKFILL an empty /
            #    missing ``prerequisitePages`` (the 7B's exclusive empty-array
            #    failure here — key_claims is always populated) from grounded
            #    signal: prerequisite topic phrases scraped from an explicit
            #    "Prerequisites:" sentence in the source, else the block's
            #    key_terms. Never the objective statement (known-bad output).
            #    No-op for every other block type. Makes the model RELIABLY
            #    satisfy the required+minItems:1 schema instead of re-rolling.
            candidate = _repair_prereq_pages(
                candidate,
                block_type=block.block_type,
                source_chunks=source_chunks,
                key_terms=block.key_terms,
                objectives=objectives,
            )
            prereq_pages_repair = candidate.pop("_prereq_pages_repair", None)
            # 5c. ``_repair_assessment_item_payload`` — for ``assessment_item``
            #    blocks only, reconcile the distractors/answer_key/index trio so
            #    the CORRECT answer lives IN distractors[] at correct_answer_index
            #    (what BOTH downstream validators require), when the 7B followed
            #    the prompt literally (wrong-only distractors + an index pointing
            #    past them at the separate answer_key). Surgical: fires only on a
            #    genuinely out-of-range index; no fabrication (uses the model's
            #    own answer_key + distractor text). No-op for every other type.
            candidate = _repair_assessment_item_payload(
                candidate,
                block_type=block.block_type,
            )
            assessment_item_payload_repair = candidate.pop(
                "_assessment_item_payload_repair", None
            )
            # 6. Bloom-diversity FLOOR (the real fix). The §3.1/§3.3 system
            #    + per-block directives alone won't reliably move a 7B model
            #    off the lazy "understand" floor (observed 70 understand /
            #    19 apply / 1 remember skew). After the LLM emits and after
            #    the numeric/case repair above, LIFT bloom_level to
            #    max(emitted, target, objective-declared) using the
            #    canonical ``_BLOOM_LEVELS`` ordering. The target LIFTS a
            #    lazy emit but NEVER lowers below the existing ≥-objective
            #    rule (this COMPOSES with, does not replace, the system
            #    prompt's "MUST be at or above the declared Bloom level"
            #    directive). Only fires when a valid bloom_level survives
            #    the repair (the strict validator below still fails closed
            #    on an unmappable value); never fabricates a level when no
            #    valid input exists.
            bloom_floor_meta: Optional[Dict[str, Any]] = None
            emitted_bloom = candidate.get("bloom_level")
            objective_declared_floor = _max_bloom_level(
                *(str((o or {}).get("bloom_level") or "") for o in (objectives or []))
            )
            lifted_bloom = _max_bloom_level(
                emitted_bloom if isinstance(emitted_bloom, str) else None,
                block.target_bloom,
                objective_declared_floor,
            )
            if lifted_bloom is not None and lifted_bloom != emitted_bloom:
                bloom_floor_meta = {
                    "emitted": emitted_bloom,
                    "target": block.target_bloom,
                    "objective_declared_floor": objective_declared_floor,
                    "lifted_to": lifted_bloom,
                }
                candidate["bloom_level"] = lifted_bloom
            if isinstance(repair_meta, dict):
                # Fold the shape-repair signals into the grounding-repair
                # meta dict so the per-call decision capture records every
                # normalization that fired this attempt.
                if key_claims_shape_repair is not None:
                    repair_meta["key_claims_shape_repair"] = (
                        key_claims_shape_repair
                    )
                if curie_shape_repair is not None:
                    repair_meta["curie_shape_repair"] = curie_shape_repair
                if source_refs_shape_repair is not None:
                    repair_meta["source_refs_shape_repair"] = (
                        source_refs_shape_repair
                    )
                if bloom_level_repair is not None:
                    repair_meta["bloom_level_repair"] = bloom_level_repair
                if prereq_pages_repair is not None:
                    repair_meta["prereq_pages_repair"] = prereq_pages_repair
                if assessment_item_payload_repair is not None:
                    repair_meta["assessment_item_payload_repair"] = (
                        assessment_item_payload_repair
                    )
                if bloom_floor_meta is not None:
                    repair_meta["bloom_floor"] = bloom_floor_meta
                if objective_echo_repair is not None:
                    repair_meta["objective_echo_repair"] = (
                        objective_echo_repair
                    )
            if schema is not None:
                try:
                    jsonschema.Draft202012Validator(schema).validate(candidate)
                except jsonschema.ValidationError as exc:
                    # improvement-map Step 3 (concern #2): when the raised
                    # error is the masking top-level ``key_claims``
                    # ``oneOf`` rejection, walk ``exc.context`` for the
                    # structured arm's real minItems sub-error and append
                    # it so both the directive matcher AND the model see
                    # the true cause (otherwise the bare oneOf message
                    # matches the W1.5.B "emit objects not flat strings"
                    # directive, which the model already satisfies → it
                    # re-emits the same single object every retry and
                    # collapses the block). Message-building only.
                    surfaced = _surface_key_claims_min_items(exc)
                    # Truncate the validation message so the remediation
                    # hint stays inside the model's context window. When a
                    # sub-error is surfaced, place it FIRST so the true
                    # cause survives truncation (the bare oneOf message
                    # echoes the full rejected array and can blow the
                    # 300-char budget on its own).
                    if surfaced is not None:
                        last_error = f"{surfaced} | {str(exc.message)}"[:300]
                    else:
                        last_error = str(exc.message)[:300]
                    attempt += 1
                    continue

            parsed = candidate
            grounding_repair = repair_meta
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
            grounding_repair=grounding_repair,
        )

        if parsed is None:
            raise OutlineProviderError(
                f"Outline tier exhausted {MAX_PARSE_RETRIES} attempts for "
                f"block {block.block_id!r} (last_error={last_error!r})",
                code="outline_exhausted",
            )

        # Construct the touch + new Block. Provider must be one of the
        # ``_TOUCH_PROVIDERS`` set in ``blocks.py`` — we map our resolved
        # provider tag onto its registry-declared ``provenance_provider``
        # (a ``groq`` / ``fireworks`` seat stamps ``together``; the legacy
        # ``openai_compatible`` alias collapses to ``local``). Anthropic /
        # together / local / nvidia map 1:1.
        touch_provider = _touch_provenance(self._provider)

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

        # outline-overflow-fix-2026-07: head-K cap on the number of source
        # chunks rendered per block BEFORE the per-chunk char cap. An
        # un-capped render (median ~20 chunks/block, chunk sections up to
        # ~46k chars) blows a small served window so the local server
        # silently head-truncates the prompt and the model sees only the
        # objectives / closing TAIL — the outline-tier silent-truncation
        # defect. ``source_chunks`` arrives already relevance/citation
        # ordered on this lane (no ranking to reuse), so a deterministic
        # head-K preserves the top-of-list grounding. ``_max_chunks <= 0``
        # is guarded out by the resolver (never disables the cap).
        capped_chunks = list(source_chunks or [])
        if self._max_chunks > 0:
            capped_chunks = capped_chunks[: self._max_chunks]

        # Truncate per-chunk body at 1200 chars; mirrors the
        # ``_LOCAL_INSTRUCTION_SYSTEM_PROMPT`` chunk-window heuristic
        # used in :mod:`Trainforge.generators._local_provider`.
        chunk_lines: List[str] = []
        for chunk in capped_chunks:
            cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
            body = str(chunk.get("body") or chunk.get("text") or "")
            if len(body) > 1200:
                body = body[:1197] + "..."
            chunk_lines.append(f"  - [{cid}] {body}")
        chunks_block = "\n".join(chunk_lines) if chunk_lines else "  (none)"

        # Wave 1.7 W1.7.B: surface the Bloom triple
        # `[Bloom: {level}, verb: {verb}]` inline next to each
        # objective render so the outline-tier model has the declared
        # cognitive demand pinned next to the behavioral outcome it
        # must shape the block around. Symmetric with the rewrite-tier
        # `_format_objectives` widening. Falls back to the legacy
        # `- {oid}: {stmt}` shape when both Bloom fields are absent
        # (back-compat with legacy fixtures).
        objective_lines: List[str] = []
        for obj in objectives or []:
            oid = str(obj.get("id") or obj.get("objective_id") or "")
            stmt = str(obj.get("statement") or obj.get("text") or "")
            bloom_level = str(obj.get("bloom_level") or "")
            bloom_verb = str(obj.get("bloom_verb") or "")
            if bloom_level or bloom_verb:
                objective_lines.append(
                    f"  - {oid} [Bloom: {bloom_level}, verb: {bloom_verb}]: "
                    f"{stmt}"
                )
            else:
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
        # Mirror the bloom_level treatment for content_type: the JSON
        # Schema enum constrains the field at decode time, but the 7B
        # model's recency bias means a bottom-of-bounds reminder of the
        # canonical vocabulary noticeably lifts attempt-1 pass rate.
        bounds_lines.append(
            "  - content_type allowed values: "
            + " | ".join(_CONTENT_TYPE_ENUM)
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
            # assessment-item-descriptor fix (2026-06): the 7B AND the 14B
            # routinely OMIT the four dedicated assessment fields (`stem`,
            # `answer_key`, `distractors`, `correct_answer_index`) — they pour
            # the question into `key_claims` / `section_skeleton` (which the
            # global prompt heavily emphasises) and never emit the dedicated
            # fields, so the strict schema rejects with "'stem' is a required
            # property" and the budget exhausts. A second observed failure mode
            # is emitting a LIST of field-TYPE DESCRIPTOR objects
            # (`[{"type": "stem"}, {"type": "distractors"}, ...]`) in place of
            # the real values. This contract names the four fields explicitly,
            # in the recency-biased per-type variation block, and forbids the
            # descriptor-list shape. Symmetric with the new closing-directive
            # field list (assessment_item branch) + the
            # `_RETRY_DIRECTIVE_PATTERNS` assessment-extras directive.
            variation_lines.append(
                "REQUIRED assessment_item fields — emit ALL FOUR as TOP-LEVEL "
                "keys with REAL VALUES (NOT inside key_claims / "
                "section_skeleton, and NEVER as field-type descriptors like "
                "{\"type\": \"stem\"}):\n"
                "  - \"stem\": the ACTUAL question text the learner reads "
                "(a complete prose question, e.g. \"What is 3/4 + 1/8?\").\n"
                "  - \"answer_key\": the ACTUAL correct answer VALUE "
                "(e.g. \"7/8\"), not a description of it.\n"
                "  - \"distractors\": a JSON array of AT LEAST 2 objects, each "
                "{\"text\": \"<an ACTUAL wrong-answer option the learner could "
                "pick, e.g. 4/12>\"}. Every option is a real answer string, "
                "never {\"type\": \"distractor\"}.\n"
                "  - \"correct_answer_index\": a 0-based integer naming which "
                "distractor (if the correct answer is also listed) or the "
                "ordinal of the correct option.\n"
                "Do NOT emit a list of {\"type\": ...} objects for any of "
                "these — emit the real stem text, real option texts, and the "
                "real answer value."
            )
        elif block_type == "prereq_set":
            variation_lines.append(
                "Prereq set contract: list every prerequisite page "
                "explicitly under a top-level ``prerequisitePages`` "
                "array; each entry is a string page_id."
            )
        elif block_type == "example":
            # CB5b (content-block-quality-2026-06 §CB5b): per-block
            # concrete-worked-instance contract. Mirrors the global
            # system-prompt directive but with the recency bias of the
            # type-specific variation block so the 7B-class model mints
            # the actual problem / steps / answer rather than restating
            # the rule. The rewrite tier renders this block from its
            # key_claims (not the section_skeleton), so abstract-rule
            # claims yield a stub example.
            variation_lines.append(
                "Example block contract: the key_claims MUST capture a "
                "CONCRETE WORKED INSTANCE from a worked Example / TRY IT "
                "item in the cited source chunks — the SPECIFIC problem "
                "(the actual numbers / expression), the intermediate "
                "steps, and the final answer. Do NOT emit only the "
                "general rule or formula; an abstract claim like \"To "
                "divide fractions, multiply by the reciprocal\" gives "
                "the rewriter nothing concrete to render. Use the "
                "numbers from the source chunk's worked example."
            )
        # Bloom-diversity fix: per-block target-Bloom directive. The GLOBAL
        # system prompt only pins the ≥-objective FLOOR; this names the
        # DETERMINISTIC per-template target so the model authors at the
        # right cognitive demand instead of defaulting to the lazy
        # "understand" floor (the directive alone won't reliably move a 7B,
        # which is why the floor is also enforced post-emit — but it lifts
        # attempt-1 quality and keeps the rewrite-tier prose on-target).
        # Appended last so the type-specific contract has recency.
        if block.target_bloom:
            variation_lines.append(
                f"Author this block at bloom_level={block.target_bloom} "
                "(unless an objective_ref declares a HIGHER Bloom level, in "
                "which case use that higher level)."
            )
        variation_block = "\n".join(variation_lines) if variation_lines else ""

        # assessment-item-descriptor fix (2026-06): name the four required
        # assessment-extra fields in the closing field enumeration too, so the
        # model is reminded of them at the recency-biased tail of the prompt
        # (not only in the per-type variation block above). Other block types
        # keep the byte-identical 10-field enumeration.
        closing_fields = (
            "block_id, block_type, content_type, bloom_level, objective_refs, "
            "curies, key_claims, section_skeleton, source_refs, "
            "structural_warnings"
        )
        if block_type == "assessment_item":
            closing_fields += (
                ", stem (real question text), answer_key (real answer value), "
                "distractors (array of >=2 {\"text\": ...} options with real "
                "option texts), correct_answer_index (0-based integer)"
            )

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
            f"RESPOND ONLY WITH A JSON OBJECT containing: {closing_fields}. "
            "No preamble, no markdown, no "
            "commentary. "
            # Wave 1.5 W1.5.B: per-claim source attribution closing
            # clause. Names the supplied source_chunks list as the
            # universe ``source_chunk_ids`` may draw from, so the
            # outline-tier model doesn't fabricate chunk IDs absent
            # from the prompt. Symmetric with the system-prompt
            # directive that mandates the structured ``key_claims``
            # shape; without this clause the model has the shape but
            # no reminder of the chunk-id universe bound.
            "For each key_claims[] entry, source_chunk_ids MUST be a "
            "non-empty subset of the chunk IDs listed in the \"Source "
            "chunks\" section above. A claim that synthesizes "
            "information from N chunks carries N IDs; a single-chunk "
            "claim carries 1 ID. "
            # Wave5-W27 propagation: pin the source_refs[] + objective_refs[]
            # contract inline alongside the per-claim citation map so the
            # outline tier surfaces both lists in every block emit. The
            # rewrite tier reads source_refs[] for the HTML
            # `data-cf-source-ids` stamp (Wave4-I10 EMPTY_SOURCE_REFS
            # critical gate) and objective_refs[] for the HTML
            # `data-cf-objective-id` stamp.
            "Wave-27 stamping contract: the top-level source_refs[] "
            "array MUST be populated with every chunk_id that "
            "contributed to the block (or [] for boilerplate). The "
            "top-level objective_refs[] array MUST cite the canonical "
            "TO-NN / CO-NN learning_outcome_refs IDs from the "
            "Objectives section above — pattern ^[A-Z]{2,}-\\d{2,}$. "
            "These lists drive the rewrite tier's HTML "
            "data-cf-source-ids and data-cf-objective-id attribute "
            "stamping."
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
        grounding_repair = call_context.get("grounding_repair") or {}
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
        # Wave 2 validated-id-fallback grounding repair signals. Folded
        # into the existing per-call event (the canonical capture at this
        # call site) so an audit can replay whether the per-claim
        # citation repair fired, how many cited ids were real vs garbage,
        # and how many claims fell back to block-level provenance — per
        # the LLM call-site instrumentation contract (dynamic signals).
        if grounding_repair:
            rationale_parts.append(
                "grounding_repair="
                f"repaired={bool(grounding_repair.get('repaired'))},"
                f"n_cited={int(grounding_repair.get('n_cited', 0))},"
                f"n_valid={int(grounding_repair.get('n_valid', 0))},"
                f"n_fallback_claims="
                f"{int(grounding_repair.get('n_fallback_claims', 0))}"
            )
        echo_repair = (
            grounding_repair.get("objective_echo_repair")
            if grounding_repair
            else None
        )
        if isinstance(echo_repair, dict) and echo_repair.get("repaired"):
            rationale_parts.append(
                "objective_echo="
                f"dropped={int(echo_repair.get('n_dropped', 0))},"
                f"warned={int(echo_repair.get('n_objective_echo_warned', 0))}"
            )
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
    "ENV_MAX_CHUNKS",
    "ENV_TRUNCATION_TRIPWIRE",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "DEFAULT_N_CANDIDATES",
    "DEFAULT_REGEN_BUDGET",
    "DEFAULT_MAX_CHUNKS",
    "_resolve_outline_max_chunks",
    "_resolve_outline_truncation_tripwire",
    "SUPPORTED_PROVIDERS",
    "MAX_PARSE_RETRIES",
    "_OUTLINE_KIND_BOUNDS",
    "_OUTLINE_SYSTEM_PROMPT",
    "_BLOCK_TYPE_GBNF",
    "_BLOCK_TYPE_JSON_SCHEMAS",
    "_RETRY_DIRECTIVE_PATTERNS",
    "_match_retry_directive",
    "_surface_key_claims_min_items",
    "_build_block_outline_schema",
    "_block_source_chunk_ids",
    "_repair_claim_grounding",
    "_drop_objective_echo_claims",
    "_repair_prereq_pages",
    "_extract_prereq_phrases_from_source",
]
