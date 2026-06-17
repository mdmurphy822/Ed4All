#!/usr/bin/env python3
"""Courseforge rewrite-tier provider — pedagogical-depth HTML synthesis.

Phase 3 Subtasks 22-26. Sibling to
:class:`Courseforge.generators._provider.ContentGeneratorProvider`
(Phase 1) and :class:`Courseforge.generators._outline_provider.OutlineProvider`
(Phase 3 Subtasks 13-20). All three subclass
:class:`Courseforge.generators._base._BaseLLMProvider` so the HTTP
plumbing, decision-capture surface, and per-backend env-var resolution
stay in one place.

Tier responsibility:

- The outline tier (smaller, cheaper model — typically a 7B-class local
  Qwen) emits a structurally-correct outline dict per block (key claims,
  CURIEs to preserve, source refs, objective refs).
- The rewrite tier (larger, pedagogically-adept model — Anthropic
  Sonnet by default) consumes that outline dict and authors the rendered
  HTML body. The rewrite tier MUST preserve every CURIE the outline
  declared verbatim — drift would silently break the corpus's CURIE
  anchoring contract (root ``CLAUDE.md`` § Wave 135 + Wave 137 family
  completeness).

Operator selects the rewrite-tier backend via
``COURSEFORGE_REWRITE_PROVIDER`` (defaults to ``anthropic``) and the
rewrite-tier model via ``COURSEFORGE_REWRITE_MODEL`` (defaults to
``claude-sonnet-4-6``). The shared HTTP plumbing reuses the synthesis-
pipeline env vars (``ANTHROPIC_API_KEY`` / ``TOGETHER_API_KEY`` /
``LOCAL_SYNTHESIS_*``) so a single Ollama / Together / Anthropic
credentials surface serves both task surfaces.

Default config:

- ``max_tokens=2400`` (the rewrite tier authors a single block's HTML
  body, not a whole page; 2400 is the empirically-derived Pattern-22
  per-block budget).
- ``temperature=0.4`` (light authorial variation while keeping
  determinism viable for cache-keyed reruns; mirrors Phase 1's
  ContentGeneratorProvider default).

Public surface:

- :meth:`RewriteProvider.generate_rewrite` — the entry point the router
  calls; consumes a Block whose ``content`` is the outline-tier dict and
  returns a Block whose ``content`` is the rendered HTML body plus a
  cumulative ``Touch(tier="rewrite", purpose="pedagogical_depth", ...)``.

CURIE-preservation gate (Subtask 26): the rewrite tier asserts every
CURIE present in the input outline's ``content["curies"]`` survives into
the emitted HTML verbatim. On miss, the gate appends a remediation
turn naming the dropped CURIEs and retries up to ``MAX_PARSE_RETRIES``.
On exhaustion the call raises :class:`RewriteProviderError` with
``code="rewrite_curie_drop"`` so the router escalates upstream rather
than silently shipping CURIE-stripped HTML.

Direct port of the
:meth:`Trainforge.generators._local_provider.LocalSynthesisProvider._missing_preserve_tokens`
+ ``_append_preserve_remediation`` pattern (`Trainforge/generators/_local_provider.py:548-583`),
adapted to Block.content's outline-dict shape (the Trainforge precedent
operates on flat instruction / preference dicts).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Courseforge.generators._base import _BaseLLMProvider
from Trainforge.generators._openai_compatible_client import (  # noqa: E402
    ENV_REQUEST_TIMEOUT as _OA_ENV_REQUEST_TIMEOUT,
)
from MCP.hardening.error_classifier import (  # noqa: E402
    ErrorClass,
    classify_error,
)
# Phase 3.5 Subtask 3: the generalized preserve-token helpers live in
# ``Courseforge/router/remediation.py`` so the Phase 3.5 router-side
# remediation injection (Subtasks 18-22) and the rewrite-tier CURIE-
# preservation gate share the same canonical implementation. The
# rewrite tier passes ``in_keys=("body",)`` (default) so the dict-
# branch the helpers expose is unused here — Block.content arrives as
# an HTML string for the gate's check.
from Courseforge.router.remediation import (  # noqa: E402
    _append_preserve_remediation,
    _missing_preserve_tokens,
)
# Single source of truth for the rewrite-tier required-attribute table.
# The same table the post-rewrite gate enforces (`rewrite_html_shape`)
# is enumerated in the prompt so the model sees the contract instead of
# inferring it from prose. Drift between the prompt and the gate is the
# regression class this fixes — sharing the constant prevents it.
from lib.validators.rewrite_html_shape import REQUIRED_ATTRS  # noqa: E402
# CURIE extraction — single source of truth (lib.ontology). Used by the
# force-injection idempotency check so it mirrors the str-path
# validator's extraction exactly.
from lib.ontology.curie_extraction import (  # noqa: E402
    extract_curies as _extract_curies,
)

# Phase 2 Subtask 35: ``blocks.py`` lives at
# ``Courseforge/scripts/blocks.py``; mirror the import bridge from
# ``_provider.py`` so ``from blocks import Block`` resolves the same
# regardless of how this module is loaded.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block, Touch  # noqa: E402  (Phase 2 intermediate format)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_PROVIDER = "COURSEFORGE_REWRITE_PROVIDER"
ENV_MODEL = "COURSEFORGE_REWRITE_MODEL"

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"

# Per-backend defaults the rewrite tier passes through to the base. The
# rewrite tier prefers a larger / pedagogically-adept model than the
# outline tier even on the same backend, so the per-backend defaults
# differ from Phase 1's :class:`ContentGeneratorProvider`.
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
DEFAULT_MODEL_TOGETHER = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_MODEL_LOCAL = "qwen2.5:14b-instruct-q4_K_M"

_DEFAULT_MAX_TOKENS = 2400
_DEFAULT_TEMPERATURE = 0.4

# Per-request HTTP timeout (seconds) for the rewrite tier's
# OpenAI-compatible backends (local / together). The rewrite tier
# authors multi-paragraph pedagogical prose; on a local 7B server that
# routinely exceeds the OpenAICompatibleClient 60s default (made worse
# by CURIE-preservation re-generation that re-calls the model). We pass
# an explicit generous timeout, sourced from
# ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` when set, else 300.0. This
# mirrors the TEXTBOOK_SYNTHESIS_TIMEOUT_SECONDS posture (also 300s for
# long-context local synthesis). Resolution / precedence (high → low):
# explicit per-call ``timeout`` kwarg > ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS``
# > this default.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0


def _resolve_request_timeout() -> float:
    """Resolve the rewrite-tier per-request timeout default.

    Reads ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (the cross-cutting
    content-generation timeout knob); a missing / unparseable /
    non-positive value falls back to
    :data:`_DEFAULT_REQUEST_TIMEOUT_SECONDS` (300.0) — never the bare
    60s client default, so 7B prose generation isn't capped at 60s.
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

SUPPORTED_PROVIDERS = (
    "anthropic",
    "together",
    "local",
    "openai_compatible",
    # Wave6: in-session subagent dispatch — the rewrite tier dispatches
    # through ``MCP/orchestrator/local_dispatcher.py::LocalDispatcher``
    # to the ``content-generator`` Claude Code subagent (Wave4b frontmatter
    # pin: ``model: sonnet``) so a Claude Max session can drive the
    # rewrite tier without an ANTHROPIC_API_KEY. Direct port of the
    # Trainforge pattern at ``Trainforge/generators/_claude_session_provider.py``.
    "claude_session",
)

# Wave6: agent-type reused intentionally. The ``content-generator`` agent
# file at ``Courseforge/agents/content-generator.md`` already carries:
#   * Wave4b YAML frontmatter pinning ``model: sonnet``
#   * Wave4-W27 MANDATORY directives for HEADING_SKIP avoidance, source-
#     ID stamping, and objective-ID stamping
# That is exactly the contract the rewrite tier needs — no new agent
# spec to mint.
_CLAUDE_SESSION_AGENT_TYPE = "content-generator"
_CLAUDE_SESSION_TASK_NAME = "rewrite_block"

# Wave6: dispatcher prerequisite message. Standalone scripts that don't
# run inside the workflow runner can't dispatch to a subagent — the
# message mirrors ``Trainforge/generators/_claude_session_provider.py::_NO_DISPATCHER_MSG``.
_NO_DISPATCHER_MSG = (
    "RewriteProvider(provider='claude_session') requires a LocalDispatcher; "
    "CourseforgeRouter must run inside the workflow runner or MCP tool "
    "(both inject one). Standalone CLI invocation has no Claude Code "
    "session to dispatch to."
)

# Subtask 26: bounded remediation retries for the CURIE-preservation
# gate. Direct port of the Trainforge precedent
# (``_local_provider.py:540`` :: ``MAX_PARSE_RETRIES``).
MAX_PARSE_RETRIES = 2

# Worker W6: per-block transient-retry budget for dispatch-side
# failures (Ollama 503 / connection reset / read timeout). Transient
# retries do NOT advance ``MAX_PARSE_RETRIES`` so a flaky local server
# can't burn the parse budget before any parse attempt completes.
# Permanent errors (auth failure, bad request) re-raise immediately.
# UNKNOWN-class errors fall through to the legacy parse-retry path so
# semantic regressions don't change behavior on unclassified errors.
_TRANSIENT_RETRY_BUDGET = 3


# ---------------------------------------------------------------------------
# System prompt — Pattern-22 prevention contract + tier-specific contract.
# ---------------------------------------------------------------------------

# Pattern-22 prevention contract — verbatim port of Phase 1's
# ``Courseforge/generators/_provider.py::_SYSTEM_PROMPT`` (`:90-103`)
# so the rewrite-tier authoring constraints stay byte-stable with
# Phase 1's content-generator. Plus the rewrite-tier-specific paragraph
# the plan calls for (Subtask 22): preserve CURIEs / facts / refs;
# rewrite for pedagogical depth, scaffolding, examples, voice; never
# add facts not in the outline's ``key_claims`` or in the source
# chunks.
_REWRITE_SYSTEM_PROMPT = (
    "You are a Courseforge content-generator authoring a single page of "
    "accessible course HTML. Always follow Pattern 22 prevention: "
    "produce substantive educational depth (theoretical foundation "
    "before examples; progressive complexity; learning-objective "
    "alignment). Use only the official Courseforge color palette "
    "(#2c5aa0 primary blue, #1a3d6e secondary, #28a745 success, "
    "#ffc107 warning, #dc3545 danger, #f8f9fa light gray, #e0e0e0 "
    "border, #333333 text). Align content to the OSCQR rubric and the "
    "page's stated learning objectives. Ground every claim in the "
    "supplied source material when present. Emit ONLY the rendered "
    "HTML body for the page — no preamble, no markdown fences, no "
    "explanation, no commentary."
    "\n\n"
    "Outline is structurally correct but generated by a smaller model. "
    "PRESERVE: factual claims (verbatim), CURIEs (verbatim), objective "
    "refs, source refs. REWRITE: for pedagogical depth, scaffolding, "
    "examples, voice. DO NOT add facts not in the outline's key_claims "
    "or in the source chunks."
    "\n\n"
    # Wave 1.7 W1.7.B: behavioral-outcome directive. Each block declares
    # one or more `objective_refs`; the rewrite-tier prose MUST teach
    # the behavioral outcome at or above the declared Bloom level. The
    # objective render in the user prompt now surfaces the Bloom triple
    # `[Bloom: {level}, verb: {verb}]` so the model sees the declared
    # cognitive demand inline.
    "Each block declares one or more `objective_refs`. The block's "
    "prose MUST teach the BEHAVIORAL OUTCOME described in those "
    "objectives' statements, at or above the declared Bloom level. A "
    "block whose objective is at `apply` or `analyze` MUST contain "
    "scaffolded reasoning, worked examples, or guided practice — not "
    "definitions alone. A block whose objective is at `evaluate` or "
    "`create` MUST contain comparison / synthesis / construction "
    "prose. The objective's `bloom_verb` (e.g. `develop`, `evaluate`, "
    "`construct`) SHOULD appear at least once in the block's prose or "
    "assessment stems; synonyms within the same Bloom level satisfy "
    "this requirement."
    "\n\n"
    "When the outline carries per-claim source attribution "
    "(`key_claims[].source_chunk_ids[]`), the rendered HTML SHOULD "
    "group the prose for each claim near a `<cite>` or "
    "`data-cf-source-ids` reference scoped to that claim's chunks. "
    "Block-level `source_refs` remains the gate-enforced superset; "
    "the per-claim attribution is for finer-grained pedagogical "
    "context."
    "\n\n"
    # CB1: pedagogical-depth directive. The outline tier produces a
    # structurally-correct but pedagogically-thin draft; the rewrite
    # tier MUST add instructional depth (rationale, verification,
    # second cases, expert tips) WITHOUT inventing any new material —
    # the NLI grounding gates fail-close on fabrication, so every
    # added sentence must be a better EXPLANATION of supplied source
    # material, never new facts/numbers/examples.
    "PEDAGOGICAL DEPTH (grounded). Add instructional depth per block "
    "type, drawing ONLY on material already in the supplied source "
    "chunks and the outline's `key_claims`:"
    "\n\n"
    "- concept / explanation blocks: include a brief CONCEPTUAL "
    "RATIONALE — explain *why* it works or *what it means*, not just "
    "the procedure or the definition. The learner should come away "
    "understanding the underlying reason, not just the steps. ADD the "
    "rationale ALONGSIDE the source's concrete worked detail — do NOT "
    "replace specific worked steps or the source's numbers with "
    "abstract generalities (keep `-32/56 -> -4/7`, not only `a/b`)."
    "\n\n"
    "- example / worked blocks: REUSE THE EXACT NUMBERS AND WORKED "
    "VALUES from the source's own worked examples (its `Example`, "
    "`TRY IT`, and solution items) VERBATIM — do NOT substitute, "
    "round, simplify-to-nicer, or invent your own example numbers "
    "when the source already gives a worked example for that "
    "operation. Walk the SOURCE's example through its steps. Only "
    "when the source states an operation as a general formula with NO "
    "accompanying worked numbers may you supply ONE short illustrative "
    "example, and it MUST be arithmetically correct and consistent "
    "with its own heading (e.g. do not title an example "
    "\"common denominator\" then use unlike denominators). Do NOT pad "
    "a block with extra invented examples to add length — one correct, "
    "source-grounded worked example beats two invented ones. An example "
    "block MUST contain a COMPLETE worked solution: state the problem, "
    "then show EVERY intermediate step through to the FINAL ANSWER, then "
    "a VERIFICATION / CHECK line (confirm the result, check a sign or "
    "identity, or substitute the answer back). The VERIFICATION / CHECK "
    "line is MANDATORY ON EVERY WORKED EXAMPLE — exactly as the matching "
    "explanation block carries it — never omit it even when the example "
    "block sits beside an explanation block that already verifies; each "
    "worked example closes with its OWN check line. NEVER stop at the "
    "problem statement — a bare 'Find the sum of …' with no solution steps "
    "is unacceptable. When the source gives a worked example for the "
    "operation, walk THAT example; otherwise supply ONE arithmetically "
    "correct illustrative example and solve it in full."
    "\n\n"
    "- include at least ONE EXPERT TIP or technique where natural — a "
    "shortcut, a heuristic, or a common-pitfall warning — provided it "
    "is supported by the source material."
    "\n\n"
    "- self_check questions: each question MUST have EXACTLY ONE "
    "correct option, with plausible but unambiguously incorrect "
    "distractors. NEVER author a question where two or more options "
    "are simultaneously true — exactly one option is correct, every "
    "other option is wrong."
    "\n\n"
    # Defect 1 — self-check must NOT reveal its answer inline. A 7B run
    # emitted "Simplify 20/30. The GCF is 10 … 20/30 simplifies to 2/3"
    # — the worked solution sat in the visible body, so the learner read
    # the answer before attempting. The answer/solution MUST live behind a
    # reveal so self-assessment actually happens.
    "- self_check_question blocks: NEVER REVEAL THE ANSWER INLINE. The "
    "visible body of a `<div class=\"self-check\">` MUST contain ONLY the "
    "question (the prompt the learner attempts) — NOT the answer, NOT the "
    "worked solution, NOT the GCF / intermediate steps that give it away. "
    "Put the answer and any worked solution inside a HIDDEN / REVEAL "
    "element WITHIN the `.self-check` div so the learner attempts the "
    "question BEFORE seeing the solution — use "
    "`<details><summary>Show answer</summary> … solution … </details>`. "
    "A self-check that prints \"… simplifies to 2/3\" in its visible body "
    "DEFEATS formative assessment and is unacceptable; the answer goes "
    "behind the reveal."
    "\n\n"
    # Defect 2 — assessment items must carry a marked answer key and
    # value-DISTINCT distractors. A 7B run emitted "Simplify 20/30. 2/3
    # 4/6 5/7" with no correct-answer marker, and 4/6 EQUALS 2/3 in value
    # (an equivalent form is not a distractor — it is a second correct
    # answer). Strengthen the existing correct-answer-marker contract.
    "- assessment_item blocks: ALWAYS EMIT AN ANSWER KEY AND VALUE-DISTINCT "
    "DISTRACTORS. (a) CLEARLY MARK THE CORRECT ANSWER — every assessment "
    "item MUST identify which option is correct (a `<strong>`/checkmark "
    "marker on the option, a `data-cf-correct=\"true\"` attribute, or an "
    "explicit \"Correct answer: …\" line). An item with no marked correct "
    "answer is unacceptable. (b) Provide 3-4 options where EVERY DISTRACTOR "
    "IS PLAUSIBLY WRONG AND DISTINCT IN VALUE from the correct answer — "
    "NEVER use an option that is an EQUIVALENT FORM of the correct answer "
    "(e.g. for \"simplify 20/30 = 2/3\" NEVER use 4/6, 6/9, 8/12, or any "
    "fraction that reduces to 2/3 — those EQUAL the correct answer and are "
    "not distractors). Each distractor must differ in actual numeric / "
    "semantic VALUE, not merely in surface form. (c) Include a BRIEF "
    "PER-DISTRACTOR RATIONALE naming the misconception the distractor "
    "represents (e.g. \"5/7 — subtracted instead of dividing by the GCF\"), "
    "so the wrong options target real learner errors. Distractor values + "
    "the misconception each represents must be grounded in the source's own "
    "worked material — never invent a number to fill an option."
    "\n\n"
    # Canonical-markup contract — the answer key + distractors MUST be
    # emitted in the EXACT structure the W7 payload gate + the distractor-
    # plausibility gate parse. A prior run emitted the answer key in
    # NON-canonical markup (`<ul><li data-cf-correct="true">2/3</li>…`)
    # which carries the right INTENT but is INVISIBLE to the validators:
    # they scan for `<li data-cf-distractor-index="N">` siblings
    # (`lib/validators/assessment_item_payload.py::_DATA_CF_DISTRACTOR_INDEX_LI_RE`
    # + `lib/validators/distractor_plausibility.py`), and the correct
    # answer is read from a `data-cf-correct="true"` flag ON that <li>
    # (`lib/validators/assessment_retrieval_grounding.py::_LI_CORRECT_RE`).
    # Pin the exact shape so the 7B emits parseable MCQ markup.
    "- assessment_item OPTION MARKUP — emit the options as an `<ol>` (or "
    "`<ul>`) where EACH option is a `<li data-cf-distractor-index=\"N\">` "
    "sibling and N is the option's 0-based index (0, 1, 2, 3 in order). "
    "The CORRECT option carries an additional `data-cf-correct=\"true\"` "
    "attribute on its own `<li>`. Each option's body is the option text "
    "followed by its brief rationale, joined with an em dash: "
    "`<li data-cf-distractor-index=\"N\">option text — rationale</li>`. "
    "There MUST be at least 2 (preferably 3-4) such "
    "`<li data-cf-distractor-index>` siblings with contiguous indices "
    "from 0, and EXACTLY ONE must carry `data-cf-correct=\"true\"`. "
    "CONCRETE EXAMPLE (simplify 20/30): "
    "`<ol>"
    "<li data-cf-distractor-index=\"0\" data-cf-correct=\"true\">2/3 — "
    "correct: 20 and 30 share GCF 10, 20÷10=2, 30÷10=3.</li>"
    "<li data-cf-distractor-index=\"1\">5/7 — subtracted the GCF instead "
    "of dividing by it.</li>"
    "<li data-cf-distractor-index=\"2\">3/4 — divided by an incorrect "
    "common factor.</li>"
    "<li data-cf-distractor-index=\"3\">10/15 — divided by 2 only and "
    "stopped before fully reducing.</li>"
    "</ol>`. This markup is REQUIRED — a `<strong>` marker or a bare "
    "\"Correct answer: …\" line WITHOUT the "
    "`<li data-cf-distractor-index=\"N\">` siblings fails the W7 payload "
    "gate (it sees zero options). The `data-cf-correct=\"true\"` flag, "
    "NOT prose, is how the correct answer is identified."
    "\n\n"
    # Defect 3 — concept blocks presenting a taxonomy must use structure,
    # not one prose paragraph. A 7B run wrote a single defining paragraph
    # where the Sonnet baseline used a TYPES TABLE (proper/improper/mixed
    # with examples) and covered the sub-types.
    "- concept blocks presenting a TAXONOMY / CATEGORIES / MULTIPLE TYPES: "
    "use a STRUCTURED `<table>` or `<ul>` — NOT a single prose paragraph. "
    "When the source distinguishes sub-types (e.g. proper / improper / "
    "mixed fractions), present them in a `<table>` (type, definition, "
    "example column) or a `<ul>` with one item per type, and COVER EVERY "
    "relevant sub-type the source supports — do not collapse a multi-type "
    "concept into one undifferentiated paragraph. The structure + sub-type "
    "rows must use only the types, definitions, and examples the source "
    "actually supplies (no invented categories)."
    "\n\n"
    "CITATION HYGIENE: NEVER write raw source-chunk identifiers or "
    "bracketed chunk tokens in visible, learner-facing prose — e.g. "
    "NEVER write `[openstax_..._chunk_00043]`, `[chunk_12]`, or any "
    "bracketed `[..._chunk_NN]` token in the rendered text. Source "
    "attribution lives ONLY in the `data-cf-source-ids` attribute and "
    "`<cite>` elements, never in the prose the learner reads."
    "\n\n"
    "HARD CONSTRAINT — DEPTH IS GROUNDED, NEVER FABRICATED. Every "
    "rationale, verification line, worked example, second case, and "
    "expert tip you add MUST use ONLY the facts, numbers, and worked "
    "values present in the supplied source chunks or the outline's "
    "`key_claims`. NEVER invent an example, a number, a formula, a "
    "result, or a fact to manufacture depth. This includes NAMED "
    "TECHNICAL TERMS, entities, compounds, mechanisms, or vocabulary "
    "that the source never mentions: even if a term is factually "
    "correct (e.g. naming an intermediate molecule, a sub-process, or "
    "an alternate name the source omits), do NOT introduce it — an "
    "out-of-source term reads as authoritative but is ungrounded and "
    "fails the entailment gate. Stay strictly within the source's own "
    "vocabulary and named concepts. The NLI grounding gates "
    "FAIL CLOSED on fabrication — an ungrounded sentence does not just "
    "lower quality, it BLOCKS the block. Depth comes from EXPLAINING "
    "the supplied source material more clearly (the why behind the "
    "what, a check that the source's own numbers confirm, a second "
    "case the source already presents), NOT from adding new material. "
    "If the source does not supply enough to add a rationale, a "
    "second example, or a tip, OMIT it rather than inventing it."
    "\n\n"
    "Every block MUST carry the per-block-type ``data-cf-*`` "
    "attributes enumerated in the user prompt's `Required attributes` "
    "line (the post-rewrite gate fails closed when any are missing). "
    "Use the supplied `Block id` value verbatim as the "
    "`data-cf-block-id` attribute — do not invent or reformat it."
    "\n\n"
    "Use the supplied `Objectives:` IDs verbatim as the "
    "`data-cf-objective-id` attribute value — do not invent, "
    "reformat, or substitute objective IDs. The valid IDs are "
    "exactly the leading tokens (e.g. `TO-01`, `CO-03`) of each "
    "`- {oid}` line in the `Objectives:` block of the user prompt. "
    "Multiple LOs use a comma-separated list "
    "(`data-cf-objective-id=\"TO-01,CO-02\"`)."
    "\n\n"
    # Wave5-W27 propagation (from Wave4-W27 `ffe517d` content-generator.md +
    # Wave4-I10 `ccd6374` EMPTY_SOURCE_REFS critical). The rewrite tier
    # emits the canonical published HTML, so the three Wave-27 MANDATORY
    # directives that originally targeted the single-pass `content-generator`
    # subagent prompt MUST apply here — the two-pass path bypasses the
    # subagent entirely, so without these directives the gates fail closed.
    "MANDATORY OUTPUT CONTRACT (Wave-27, gate-enforced):"
    "\n\n"
    "1. HEADING HIERARCHY (HEADING_SKIP critical gate at "
    "`lib/validators/content.py::ContentStructureValidator`): every "
    "emitted HTML body MUST follow strict h1 → h2 → h3 → h4 "
    "progression. Each heading level descends by AT MOST one level. "
    "NEVER skip a level — no `<h1>` → `<h3>`, no `<h2>` → `<h4>`, "
    "no `<h2>` → `<h5>`. Skipping levels trips the HEADING_SKIP gate "
    "and fail-closes IMSCC packaging."
    "\n\n"
    "2. SOURCE-ID STAMPING (EMPTY_SOURCE_REFS critical gate, "
    "Wave4-I10 fail-closed): every `<section>`, content wrapper "
    "(`.flip-card`, `.self-check`, `.activity-card`, "
    "`.discussion-prompt`), and heading-bearing block MUST carry "
    "`data-cf-source-ids=\"<comma-separated source IDs>\"`. The "
    "value is derived from the outline block's `source_refs[]` "
    "array supplied in the user prompt — list each `sourceId` "
    "comma-separated. Empty string `data-cf-source-ids=\"\"` is "
    "permitted ONLY for boilerplate / navigation blocks per the "
    "Wave-27 carve-out, BUT THE ATTRIBUTE MUST BE PRESENT. NEVER "
    "emit `data-cf-source-ids` on `<p>`, `<li>`, or `<tr>` children "
    "— scope stays at the section / component-wrapper level "
    "(Wave-9 decision P2)."
    "\n\n"
    "3. OBJECTIVE-ID STAMPING (chunker `learning_outcome_refs[]` "
    "back-fill avoidance): every content block addressing a specific "
    "Learning Objective MUST carry "
    "`data-cf-objective-id=\"<TO-NN or CO-NN>\"`. The value is "
    "derived from the outline block's `objective_refs[]` array — "
    "pattern `^[A-Z]{2,}-\\d{2,}$` (e.g. `TO-01`, `CO-03`). Multiple "
    "LOs use a comma-separated list "
    "(`data-cf-objective-id=\"TO-01,CO-02\"`). Missing stamps force "
    "the chunker to fall back to text-scan heuristics — less reliable "
    "than explicit attribute stamping, and breaks RAG-by-LO retrieval."
    "\n\n"
    "Every `<` and `>` you write outside a real HTML tag MUST be "
    "escaped as `&lt;` and `&gt;`. This applies to schematic / "
    "placeholder text — RDF triples, URI patterns, generic "
    "`<thing>` slots, BNF-style productions, anything where the "
    "angle brackets are illustrative, not structural. PROHIBITED — "
    "the parser treats these as unclosed elements and fails the "
    "post-rewrite shape gate: bare `<subject>`, bare `<predicate>`, "
    "bare `<object>`, bare `<URI>`, bare `<value>`, bare `<term>`, "
    "bare `<placeholder>`. REQUIRED instead — write either "
    "`<code>subject</code> <code>predicate</code> <code>object</code>` "
    "or `&lt;subject&gt; &lt;predicate&gt; &lt;object&gt;`. The same "
    "rule applies to URI brackets in RDF / SPARQL / Turtle examples: "
    "write `&lt;http://example.org/x&gt;` or "
    "`<code>&lt;http://example.org/x&gt;</code>`, never bare "
    "`<http://example.org/x>`."
)


# ---------------------------------------------------------------------------
# Block-type → HTML output contract map (Subtask 24).
# ---------------------------------------------------------------------------
#
# Per-block-type HTML attribute contracts the rewrite tier must follow
# when authoring the rendered body. Mirrors ``Block.to_html_attrs``
# (`Courseforge/scripts/blocks.py:336-465`) so the rewrite output is
# downstream-extractable by the same priority chain Trainforge's
# ``process_course._extract_section_metadata`` walks.

_BLOCK_TYPE_OUTPUT_CONTRACTS: Dict[str, str] = {
    "objective": (
        "Emit a `<li>` carrying `data-cf-objective-id` (canonical "
        "TO-NN / CO-NN), `data-cf-bloom-level`, `data-cf-bloom-verb`, "
        "and `data-cf-cognitive-domain` attributes. The objective "
        "statement is the `<li>`'s text content."
    ),
    "concept": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h2>` "
        "or `<h3>` heading carrying `data-cf-content-type` + "
        "`data-cf-bloom-range` + `data-cf-key-terms`, followed by "
        "explanatory paragraphs. When the concept presents a taxonomy / "
        "categories / multiple sub-types, present them in a structured "
        "`<table>` or `<ul>` (one row/item per sub-type with its example) "
        "rather than a single prose paragraph, covering every sub-type the "
        "source supports."
    ),
    "example": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h3>` "
        "and `<p>` paragraphs presenting a worked example. Carry "
        "`data-cf-content-type=\"example\"` on the heading."
    ),
    "explanation": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h2>` "
        "or `<h3>` and explanatory paragraphs. Carry "
        "`data-cf-content-type=\"explanation\"` on the heading."
    ),
    "summary_takeaway": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h3>` "
        "and a short `<ul>` of takeaways. Carry "
        "`data-cf-content-type=\"summary\"` on the heading."
    ),
    "callout": (
        "Emit a `<div class=\"callout callout-{kind}\">` carrying "
        "`data-cf-component=\"callout\"` + `data-cf-purpose` + "
        "`data-cf-content-type=\"callout\"`. "
        "SCOPE — a callout is ONE FOCUSED, CONCISE HIGHLIGHT: a single "
        "key tip, warning, caution, or note (typically 1-3 sentences, at "
        "most ONE short illustrative instance). It is NOT A FULL LESSON — "
        "NO MULTI-EXAMPLE sequences, no step-by-step worked solutions, and "
        "do NOT cover several cases or sub-types in a callout. That "
        "instructional content belongs in concept / explanation / example "
        "blocks, never here. Keep the highlight grounded in the source "
        "material (no fabricated facts) — surface the single most "
        "important alert about the surrounding content, not a mini-lesson."
    ),
    "flip_card_grid": (
        "Emit a `<div class=\"flip-card-grid\">` whose children are "
        "per-card `<div class=\"flip-card\">` elements carrying "
        "`data-cf-component=\"flip-card\"`, "
        "`data-cf-purpose=\"term-definition\"`, "
        "`data-cf-teaching-role`, and `data-cf-term`. Emit ONE card per "
        "DISTINCT key term (front = the term, back = its definition drawn "
        "from the source) — cover each supplied key term exactly once; "
        "NEVER repeat a term or restate the same sentence across cards, and "
        "NEVER fill cards with a generic procedure paragraph instead of "
        "term/definition pairs. EACH CARD FRONT MUST BE ONE OF THE BLOCK'S "
        "SUPPLIED `key_terms` (the DOMAIN VOCABULARY of the chapter — e.g. "
        "`variable`, `coefficient`, `like terms`), or a domain term "
        "explicitly DEFINED in the source; NEVER a PEDAGOGY / STRUCTURAL "
        "META-WORD. Card fronts are DRAWN FROM THE DOMAIN VOCABULARY, NOT "
        "from the most-frequent capitalized SOURCE HEADINGS. DENY as a card "
        "front any of these pedagogy/structural meta-words: `example`, "
        "`exercise`, `problem`, `try it`, `solution`, `practice`, `note`, "
        "`activity`, `summary` — they are NOT domain key terms. The card "
        "back is that term's DEFINITION drawn from the source. Cover the "
        "supplied key terms."
    ),
    "self_check_question": (
        "Emit a `<div class=\"self-check\">` carrying "
        "`data-cf-component=\"self-check\"`, "
        "`data-cf-purpose=\"formative-assessment\"`, "
        "`data-cf-bloom-level`, `data-cf-objective-ref`, and "
        "`data-cf-source-ids` / `data-cf-source-primary`. The visible body "
        "is the QUESTION ONLY — put the answer / worked solution behind a "
        "reveal INSIDE the div "
        "(`<details><summary>Show answer</summary> … </details>`) so the "
        "learner attempts before seeing it; NEVER reveal the answer inline."
    ),
    "activity": (
        "Emit a `<div class=\"activity-card\">` carrying "
        "`data-cf-component=\"activity\"`, `data-cf-purpose=\"practice\"`, "
        "`data-cf-bloom-level`, `data-cf-objective-ref`, and "
        "`data-cf-source-ids`. EVERY PRACTICE ITEM MUST EXERCISE THE SAME "
        "OPERATION / SKILL named in the activity's instruction line, using "
        "ONLY values and operations drawn from the source — NEVER list items "
        "of a DIFFERENT type than the instruction states (e.g. do NOT list "
        "integer-arithmetic expressions like \"-2 ÷ 3\" or \"6 - 3(5)\" under "
        "a \"Simplify the following fractions\" instruction — those do not "
        "exercise fraction simplification). The instruction line and its "
        "practice items must be INTERNALLY CONSISTENT: each item is a "
        "concrete instance of the exact skill the instruction names. "
        "EACH PRACTICE ITEM MUST STATE THE ACTUAL PROBLEM / TASK IN FULL "
        "(the expression to simplify, the equation to solve, the question "
        "to answer) — a learner reads the item and knows exactly what to "
        "do. NEVER emit BARE EXERCISE / REFERENCE NUMBERS (e.g. "
        "\"83, 84, 85\") or cite SOURCE EXERCISE INDICES as items — a "
        "textbook exercise number is meaningless to a learner; write out "
        "the actual problem instead."
    ),
    "misconception": (
        "Emit a `<section>` whose JSON-LD entry carries the "
        "misconception ID (mc_[0-9a-f]{16}) plus correction. The HTML "
        "body need not carry data-cf-* attributes — JSON-LD is the "
        "authoritative shape for misconception blocks."
    ),
    "assessment_item": (
        "Emit a `<div class=\"assessment-item\">` carrying the question "
        "stem in a `<p>`, then the options as an `<ol>` whose children are "
        "`<li data-cf-distractor-index=\"N\">` siblings (N = 0,1,2,3 in "
        "order; at least 2, preferably 3-4, contiguous from 0). The CORRECT "
        "option additionally carries `data-cf-correct=\"true\"` on its own "
        "`<li>` (EXACTLY ONE option) — this attribute, not prose, marks the "
        "answer. Each option body is `option text — rationale`, the "
        "rationale naming the misconception the distractor represents. "
        "Every distractor must be DISTINCT IN VALUE from the correct answer "
        "(never an equivalent form, e.g. never 4/6 for a 2/3 answer). "
        "EACH DISTRACTOR'S VALUE MUST BE THE ACTUAL RESULT A STUDENT WHO MADE "
        "THE NAMED MISCONCEPTION WOULD COMPUTE — the rationale names the error "
        "AND the value must be the ARITHMETIC CONSEQUENCE of that error, not "
        "an arbitrary wrong value. Work the named mistake through to its real "
        "result (e.g. a distractor whose rationale is \"multiplied the "
        "numerators and denominators instead of using the reciprocal\" must "
        "show the value that erroneous multiplication actually yields — never "
        "a value the misconception would not produce). "
        "EXAMPLE: `<li data-cf-distractor-index=\"1\">5/7 — subtracted the "
        "GCF instead of dividing by it.</li>`. "
        "Assessment items in IMSCC live in QTI XML downstream; the "
        "HTML emit here is the authoring fixture."
    ),
    "prereq_set": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h2>` "
        "or `<h3>` and an `<ol>` of prerequisite topic refs. Each `<ol>` "
        "item NAMES A PRIOR FOUNDATIONAL SKILL OR TOPIC the learner needs "
        "BEFORE this content — a simpler, earlier capability this content "
        "ASSUMES (e.g. for adding integers: \"understand the number line\", "
        "\"compute absolute value\") — drawn from the source's stated "
        "prerequisites when present. NEVER list the current chapter's OWN "
        "learning objectives as prerequisites, NEVER emit a raw "
        "`CO-NN` / `TO-NN` objective id as a prerequisite, and NEVER restate "
        "this block's own objective. A prerequisite is a PRIOR skill the "
        "learner brings IN, not an OUTCOME this content produces. If the "
        "source states no explicit prerequisites, list the FOUNDATIONAL "
        "CONCEPTS this content builds on (the simpler skills it assumes), "
        "NOT its outcomes."
    ),
    "reflection_prompt": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h3>` and one or more `<p>` reflection prompts. Each `<p>` is an "
        "OPEN-ENDED QUESTION addressed to the learner about THEIR OWN "
        "thinking, experience, or confidence (e.g. 'When have you needed a "
        "fraction in simplest form? How will you check your own work for the "
        "subtract-instead-of-divide mistake?') — it ends in a question mark "
        "and invites self-reflection. A reflection prompt is NOT an "
        "explanation, definition, worked example, or graded assessment — "
        "never re-teach the content or pose a right/wrong evaluate-this task."
    ),
    "discussion_prompt": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h3>` and one or more `<p>` discussion prompts. Each `<p>` is an "
        "OPEN-ENDED QUESTION posed to a GROUP to debate or compare views "
        "(e.g. 'Why does dividing by the GCF preserve a fraction's value "
        "while subtracting does not? Share a real-world situation where "
        "simplest form matters.') — it ends in a question mark and has no "
        "single correct answer. A discussion prompt is NOT an explanation, "
        "definition, worked example, or quiz item — never re-teach the "
        "content as expository prose; pose the question instead."
    ),
    "chrome": (
        "Emit page chrome (header / footer / nav). Carry "
        "`data-cf-role=\"template-chrome\"` on the wrapper."
    ),
    "recap": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h2>` or `<h3>` and a BRIEF recap of the prior week's key terms — "
        "a short `<ul>` of key terms with one-line reminders, or one or two "
        "summary `<p>` paragraphs. A recap is NOT an assessment: never emit "
        "multiple-choice options, `<li data-cf-distractor-index>`, answer "
        "keys, or a fresh worked example — it only restates already-taught "
        "terms."
    ),
}


def _block_type_output_contract(block_type: str) -> str:
    """Return the per-block-type HTML attribute contract paragraph.

    Falls back to a generic instruction when the block_type has no
    entry in the table — defensive only; ``Block.__post_init__``
    already validates the set.
    """
    return _BLOCK_TYPE_OUTPUT_CONTRACTS.get(
        block_type,
        (
            f"Emit the rendered HTML body for a block of type "
            f"{block_type!r}. Carry `data-cf-source-ids` on the top "
            f"wrapper to attribute the source chunks."
        ),
    )


def _required_attrs_directive(block_type: str, block_id: str) -> str:
    """Return the gate-enforced required-attribute directive line.

    Reads the canonical ``REQUIRED_ATTRS`` table from
    :mod:`lib.validators.rewrite_html_shape` so the prompt enumerates
    the same attributes the post-rewrite gate enforces. The block_id
    is interpolated as the ``data-cf-block-id`` value the model must
    use verbatim — the gate doesn't validate the value, but the
    downstream Trainforge consumer cross-references the JSON-LD
    ``blocks[]`` projection by block_id, so an invented id silently
    breaks chunk extraction.

    Empty REQUIRED_ATTRS entry (block_type not in the table) returns an
    empty string — defensive only; every BLOCK_TYPES value has an entry.
    """
    required = REQUIRED_ATTRS.get(block_type, ())
    if not required:
        return ""
    attr_list = ", ".join(f"`{a}`" for a in required)
    return (
        "Required attributes (gate-enforced; block fails post-rewrite "
        "validation when any are missing): "
        f"{attr_list}. "
        f"Use `data-cf-block-id=\"{block_id}\"` verbatim — do not "
        "invent or reformat the block_id."
    )


# HTML5 void elements (WHATWG HTML Living Standard § 12.1.2) — never
# carry a closing tag. Used by ``_escape_orphan_placeholder_tags`` to
# skip elements that legitimately appear without a closer.
_HTML5_VOID_ELEMENTS: frozenset = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
})

# Bare-opener pattern: ``<word>`` with no attributes, where ``word`` is
# alphanumeric (with optional ``:`` for CURIE-shaped placeholders and
# ``-`` / ``_`` for typical tag names). The "no attributes" constraint
# keeps the sanitizer conservative — real attribute-bearing elements
# like ``<section data-cf-block-id="...">`` don't match and pass
# through untouched.
_BARE_OPENER_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_:-]*)>")


def _escape_orphan_placeholder_tags(html: str) -> str:
    """Escape orphan-opener placeholder tags as HTML entities.

    Walks the input for bare opener patterns ``<word>`` (no
    attributes). For each match, looks for a corresponding ``</word>``
    closer in the rest of the string; when no closer exists AND the
    tag isn't an HTML5 void element, the opener is rewritten as
    ``&lt;word&gt;``. Real HTML element pairs
    (``<section>...</section>``) pass through unchanged because the
    closer is found.

    Closes the Qwen-7B-Q4 failure mode where the rewrite tier emits
    RDF triples as bare ``<subject> <predicate> <object>`` placeholders
    that ``html.parser`` sees as unclosed elements (firing
    ``REWRITE_HTML_PARSE_FAIL`` critical at the post-rewrite shape
    gate). The conservative regex shape (no attrs) means the
    sanitizer cannot mis-escape a real attribute-bearing element.
    """
    out: List[str] = []
    last_end = 0
    for m in _BARE_OPENER_RE.finditer(html):
        out.append(html[last_end : m.start()])
        tag = m.group(1)
        if tag.lower() in _HTML5_VOID_ELEMENTS:
            out.append(m.group(0))
        else:
            closer_re = re.compile(
                r"</" + re.escape(tag) + r"\s*>",
                re.IGNORECASE,
            )
            if closer_re.search(html, m.end()):
                out.append(m.group(0))
            else:
                out.append(f"&lt;{tag}&gt;")
        last_end = m.end()
    out.append(html[last_end:])
    return "".join(out)


def _safe_json_dumps(content: Any) -> str:
    """Serialize ``Block.content`` to a JSON string for the prompt.

    ``Block.content`` is ``Union[str, Dict[str, Any]]``. Strings pass
    through unchanged so the legacy Phase 1 path that emits a
    ``content=html`` Block still renders sensibly through the rewrite
    tier. Dicts are serialised with ``ensure_ascii=False`` so CURIEs
    (``sh:NodeShape``, ``rdfs:subClassOf``, …) survive verbatim —
    critical for the Subtask 26 CURIE-preservation gate.
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(
            content, ensure_ascii=False, sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        # Defensive — a non-serialisable content payload would prevent
        # the rewrite tier from even seeing the outline. Surface a
        # readable repr instead so postmortem still has the data.
        logger.warning("Outline payload not JSON-serialisable: %s", exc)
        return repr(content)


def _format_per_claim_citations(content: Any) -> str:
    """Render the per-claim source-attribution map for the rewrite prompt.

    Wave 1.5 W1.5.D: surface the structured ``key_claims[].source_chunk_ids[]``
    attribution as a distinct, prompt-level signal so the rewrite-tier
    author has the citation map salient (instead of buried inside the
    serialised outline JSON blob). The block is sandwiched between the
    "Outline (...)" and "Source chunks (cite via source_refs)" sections
    in both the regular and escalated user prompts.

    Branches:

    - ``content`` is not a dict → returns the single-line placeholder
      ``"(none — outline content not a dict)"`` (legacy Phase 1 path
      where the rewrite tier consumes a content=html Block directly).
    - ``content["key_claims"]`` missing or empty → returns ``"(no
      claims)"``.
    - structured shape (each claim is a dict with ``claim`` +
      ``source_chunk_ids[]``) → renders one line per claim with the
      claim text truncated at 80 chars and a comma-separated chunk_id
      list.
    - legacy List[str] shape (each claim is a bare string) → renders
      the claim with a ``(legacy shape — no per-claim citation)``
      annotation so the rewrite tier knows the citation map is absent
      for that claim.

    Per-claim text is truncated at 80 chars (with ``"..."`` ellipsis
    above that bound) to keep the prompt-size growth bounded — the
    plan §6.3 prompt-size risk budget caps the total addition at
    < 500 chars on a 5-claim block.
    """
    if not isinstance(content, dict):
        return "(none — outline content not a dict)"
    claims = content.get("key_claims") or []
    lines: List[str] = []
    for idx, c in enumerate(claims, start=1):
        if isinstance(c, dict):
            claim_text = c.get("claim", "")
            chunk_ids = c.get("source_chunk_ids") or []
            ids_str = ", ".join(chunk_ids) if chunk_ids else "(none)"
            short = claim_text if len(claim_text) <= 80 else claim_text[:77] + "..."
            lines.append(f"  - claim {idx}: \"{short}\" cites chunk(s) [{ids_str}]")
        elif isinstance(c, str):
            short = c if len(c) <= 80 else c[:77] + "..."
            lines.append(
                f"  - claim {idx}: \"{short}\" (legacy shape — no per-claim citation)"
            )
    return "\n".join(lines) if lines else "(no claims)"


def _format_source_chunks(chunks: Sequence[Any]) -> str:
    """Format the source-chunk list into a readable prompt block.

    Accepts either dict shape (``{"chunk_id": ..., "text": ...}``) or
    a chunk-like object exposing ``chunk_id`` / ``text`` attributes.
    Empty input renders ``"(none)"``.
    """
    if not chunks:
        return "(none)"
    parts: List[str] = []
    for c in chunks:
        if isinstance(c, dict):
            cid = c.get("chunk_id") or c.get("id") or "<unknown>"
            text = c.get("text") or c.get("content") or ""
        else:
            cid = (
                getattr(c, "chunk_id", None)
                or getattr(c, "id", None)
                or "<unknown>"
            )
            text = getattr(c, "text", "") or getattr(c, "content", "")
        parts.append(f"- [{cid}] {text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Escalation-marker context map (Subtask 25).
# ---------------------------------------------------------------------------
#
# Maps each ``escalation_marker`` value to a short context paragraph the
# rewrite tier prepends to the escalated prompt. Markers come from
# ``Courseforge/scripts/blocks.py::_ESCALATION_MARKERS`` (the canonical
# set the dataclass validates against) plus ``outline_skipped_by_policy``
# which the router emits when ``BlockProviderSpec.escalate_immediately``
# pre-fires the escalation without an outline call.

_ESCALATION_MARKER_CONTEXT: Dict[str, str] = {
    "outline_budget_exhausted": (
        "The outline contains a partial draft you MAY reference, but "
        "the outline tier could not refine it further within budget."
    ),
    "structural_unfixable": (
        "The outline tier's emit was structurally invalid against the "
        "block's JSON schema; treat the outline as untrustworthy "
        "context only and synthesise from the source chunks."
    ),
    "validator_consensus_fail": (
        "The outline contained semantic violations the deterministic "
        "validators flagged; rewrite from the source chunks rather "
        "than the outline draft."
    ),
    "outline_skipped_by_policy": (
        "No outline was generated (router short-circuited per "
        "BlockProviderSpec.escalate_immediately). Create the block "
        "from scratch using the supplied source chunks and objectives."
    ),
    # Wave 1.5 W1.5.C: per-claim attribution unfixable. Fires when the
    # outline-tier regen budget exhausted purely on per-claim source-
    # attribution misses (no block-level structural miss).
    "per_claim_attribution_unfixable": (
        "outline tier could not consistently attribute claims to "
        "specific source chunks; treat the per-claim citation map "
        "as advisory rather than authoritative; preserve block-level "
        "`source_refs[]` grounding instead."
    ),
    # Wave 1.7 W1.7.D: block-objective delivery unfixable. Fires when
    # the rewrite-tier regen budget exhausted purely on Wave-1.7
    # block-objective delivery misses (no upstream structural miss).
    # The rewrite-tier prompt-builder treats the prior prose as
    # best-effort; the surviving block ships with
    # ``objective_alignment[*].status="unverifiable"`` so a postmortem
    # reader sees the gate failure even though the rendered block
    # ships.
    "block_objective_undelivered": (
        "Block could not be brought into pedagogical alignment with "
        "its declared objective_refs after the regen budget. Treat "
        "the existing prose as best-effort; preserve the structural "
        "grounding signals (source_refs, observed_bloom) but flag "
        "objective_alignment[*].status as unverifiable so a postmortem "
        "reader sees the gate failure even though the rendered block "
        "ships."
    ),
}


def _extract_outline_curies(content: Any) -> List[str]:
    """Return the list of CURIEs the outline declared for preservation.

    The outline tier's emit shape (per Subtask 17 contract) carries a
    ``curies`` key whose value is a list of CURIE strings. When the
    block's content is a string (legacy / Phase 1 path), no outline
    CURIE list exists — return an empty list.

    Used both by :meth:`RewriteProvider._render_escalated_user_prompt`
    (to surface the preserve list in the escalated prompt) and by
    :meth:`RewriteProvider.generate_rewrite` (to enforce the
    Subtask 26 CURIE-preservation gate).
    """
    if not isinstance(content, dict):
        return []
    raw = content.get("curies", []) or []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(c) for c in raw if c]


# ---------------------------------------------------------------------------
# CURIE-preservation gate helpers (Subtask 26 → Phase 3.5 Subtask 3).
# ---------------------------------------------------------------------------
#
# The generalized helpers live in ``Courseforge/router/remediation.py``
# (Phase 3.5 Subtask 3) so the router-side remediation injection
# (Subtasks 18-22) and the rewrite-tier CURIE-preservation gate share
# one canonical implementation. The thin wrappers below preserve the
# rewrite-tier-specific call signatures (positional args, no
# ``in_keys`` kwarg) so the existing call sites in
# :meth:`RewriteProvider.generate_rewrite` and the existing
# ``test_rewrite_provider.py`` regression suite remain byte-stable
# across the move. The Trainforge precedent
# (``Trainforge/generators/_local_provider.py:548-583``) is the same
# function the new module ports; the rewrite tier consumes the
# string-content branch of the generalised signature.


# ---------------------------------------------------------------------------
# Plan §3.5: contextual CURIE-preservation gate.
# ---------------------------------------------------------------------------
#
# Pre-§3.5 the gate did a bare substring match: any verbatim
# occurrence of a CURIE token in the HTML body counted as preserved.
# That contract is structurally satisfied by token-stuffing — the
# rewrite tier learned to splice ``<span vocab="rdf:RDF">`` mid-
# sentence or invent fake-triple examples that the substring matcher
# accepts but a learner would never read as pedagogically natural.
#
# §3.5 replaces the substring match with a positional check. A CURIE
# counts as "preserved in pedagogical context" when:
#
# 1. It appears as text content inside a ``<code>`` / ``<kbd>`` /
#    ``<samp>`` element (definitional / sample-code voice).
# 2. It is the value of a ``<span data-cf-term="<local>">`` whose
#    local-name part matches the CURIE (the canonical Courseforge
#    inline-term pattern).
# 3. It appears inside a ≥40-char window of prose containing one of
#    the pedagogical-voice anchors: ``prefix`` / ``vocabulary`` /
#    ``namespace`` / ``triple`` or any canonical Bloom verb.
#
# Token-stuffing patterns rejected: CURIE in an attribute value
# (``vocab="rdf:RDF"``), spliced into a non-pedagogical sentence,
# or stuffed into a fabricated triple example without the surrounding
# pedagogical-voice anchor.

# Tags whose text content counts as a "definitional voice" anchor.
_PEDAGOGICAL_VOICE_TAGS: Tuple[str, ...] = ("code", "kbd", "samp")

# Sentence-window anchor terms for the pedagogical-prose path.
_PEDAGOGICAL_VOICE_ANCHORS: Tuple[str, ...] = (
    "prefix",
    "vocabulary",
    "namespace",
    "triple",
)

# Minimum prose-window size around a CURIE occurrence for the
# sentence-context check. Below this floor a stuffed CURIE in a
# fragment "sentence" doesn't count.
_PEDAGOGICAL_VOICE_WINDOW_CHARS: int = 40


def _curie_in_pedagogical_context(html: str, curie: str) -> bool:
    """Return True when ``curie`` appears in pedagogical context in ``html``.

    Pedagogical context = the CURIE token appears in at least one of
    three positional shapes:

    - inside the text content of a ``<code>`` / ``<kbd>`` / ``<samp>``
      element (definitional / sample-code voice);
    - as the (local-name part of a) ``<span data-cf-term="...">``
      attribute pair (the canonical Courseforge inline-term pattern);
    - in a ≥40-char prose window containing one of the pedagogical-
      voice anchor terms (prefix / vocabulary / namespace / triple)
      OR any canonical Bloom verb.

    Returns False when the only occurrences are token-stuffed in tag
    attributes, fabricated triple examples without surrounding
    pedagogical voice, or absent entirely. Plan §3.5 contract.
    """
    if not html or not curie:
        return False

    # Case 1: text content inside a definitional-voice tag. We
    # substring-match the open / close pair around the CURIE; this
    # accepts both ``<code>rdf:type</code>`` and ``<code class="x">
    # rdf:type ...</code>``.
    for tag in _PEDAGOGICAL_VOICE_TAGS:
        pattern = re.compile(
            rf"<{tag}\b[^>]*>([^<]*?){re.escape(curie)}([^<]*?)</{tag}>",
            re.IGNORECASE,
        )
        if pattern.search(html):
            return True

    # Case 2: ``<span data-cf-term="...">`` whose local part matches.
    # The canonical Courseforge pattern stamps `data-cf-term=<local>`
    # where <local> is the CURIE's right-of-colon part (lowercase
    # slugged), so the CURIE token itself usually appears as the
    # span's text content — we accept either the local-match attr
    # form OR the verbatim CURIE text inside the span.
    local = curie.split(":", 1)[1] if ":" in curie else curie
    span_pattern = re.compile(
        rf'<span\b[^>]*data-cf-term=(?:"|\')(?:{re.escape(local)}|'
        rf'{re.escape(curie)})(?:"|\')[^>]*>([^<]*?)</span>',
        re.IGNORECASE,
    )
    if span_pattern.search(html):
        return True

    # Case 3: prose-window anchor. Walk every CURIE occurrence and
    # check the surrounding window for an anchor term or Bloom verb.
    # Skip occurrences inside attribute values: a ``"<curie>"`` shape
    # adjacent to ``=`` indicates an attribute value, not prose.
    blooms_verbs = _flat_bloom_verbs()
    anchor_set = set(_PEDAGOGICAL_VOICE_ANCHORS) | blooms_verbs

    start = 0
    while True:
        idx = html.find(curie, start)
        if idx == -1:
            break
        start = idx + len(curie)
        # Skip attribute-value occurrences. An attribute value sits
        # inside quotes adjacent to an ``=``; we walk backwards to
        # the most recent quote / angle bracket and reject when the
        # immediate preceding non-space character is ``=``.
        if _looks_like_attribute_value(html, idx):
            continue
        window_start = max(0, idx - _PEDAGOGICAL_VOICE_WINDOW_CHARS)
        window_end = min(
            len(html), idx + len(curie) + _PEDAGOGICAL_VOICE_WINDOW_CHARS
        )
        window = html[window_start:window_end].lower()
        if any(anchor in window for anchor in anchor_set):
            return True
    return False


def _flat_bloom_verbs() -> set[str]:
    """Return the union of canonical Bloom verbs across every level.

    Lazy-loaded once per process via the canonical helper at
    :func:`lib.ontology.learning_objectives.BLOOMS_VERBS`. Cached on
    the module-level frozenset so the helper stays fast in the
    rewrite-tier hot path.
    """
    cached = getattr(_flat_bloom_verbs, "_cache", None)
    if cached is not None:
        return cached
    try:
        from lib.ontology.learning_objectives import BLOOMS_VERBS

        flat: set[str] = set()
        for verbs in BLOOMS_VERBS.values():
            flat.update(verbs)
    except Exception:  # pragma: no cover — defensive
        flat = set()
    _flat_bloom_verbs._cache = flat  # type: ignore[attr-defined]
    return flat


def _looks_like_attribute_value(html: str, idx: int) -> bool:
    """Return True when ``html[idx:]`` looks like an HTML attribute value.

    Heuristic: walk backward from ``idx`` to find the most recent
    ``"`` or ``'`` quote character; if the character just before that
    quote is ``=``, the substring is inside an attribute value.
    """
    # Walk back at most 200 chars (generous bound for any single
    # attribute value).
    bound = max(0, idx - 200)
    region = html[bound:idx]
    quote_pos = max(region.rfind('"'), region.rfind("'"))
    if quote_pos == -1:
        return False
    # Walk back from the quote to skip any whitespace then check for ``=``.
    j = bound + quote_pos - 1
    while j >= 0 and html[j].isspace():
        j -= 1
    return j >= 0 and html[j] == "="


def _missing_preserve_curies(
    html_response: str, outline_curies: Sequence[str]
) -> List[str]:
    """Return CURIEs that don't appear in pedagogical context in the HTML.

    Plan §3.5: the pre-§3.5 substring-only contract permitted token-
    stuffing patterns that satisfied the gate structurally but
    flouted pedagogical voice. This helper now wraps
    :func:`_curie_in_pedagogical_context` so the gate accepts only
    the three positional contexts the new contract recognises (code-
    voice text content / data-cf-term span / prose-window anchor).

    Empty input returns an empty list. Empty CURIE list returns the
    empty list (nothing to enforce).
    """
    if not outline_curies:
        return []
    html = html_response or ""
    missing: List[str] = []
    for curie in outline_curies:
        if not curie:
            continue
        if not _curie_in_pedagogical_context(html, curie):
            missing.append(curie)
    return missing


# Plan §3.5 rephrased remediation directive. Pre-§3.5 wording was
# "Rewrite the response so each of those tokens appears VERBATIM",
# which the model satisfied by token-stuffing. The new directive
# explicitly names the three permitted positional contexts AND
# forbids the two stuffing patterns the introspection run exposed.
_CURIE_PEDAGOGICAL_DIRECTIVE: str = (
    "Each preserved CURIE must appear in pedagogical voice — inside "
    "<code>, as a definitional <span data-cf-term=...>, or in a "
    "sentence introducing the prefix/vocabulary/namespace. Do NOT "
    "stuff CURIEs into attribute values or invented triple examples."
)


def _append_curie_remediation(
    user_prompt: str, missing_curies: Sequence[str]
) -> str:
    """Append a contextual-CURIE remediation directive to ``user_prompt``.

    Plan §3.5: the legacy implementation re-used the Trainforge
    "did not include the required" phrase, which was structurally
    correct but didn't tell the model HOW to include the tokens —
    encouraging the token-stuffing failure mode the audit surfaced.
    The new directive names the three pedagogical-context shapes the
    gate accepts and explicitly forbids attribute-value / fake-triple
    stuffing.

    The base "did not include the required" phrase from the
    Trainforge precedent is preserved as the opener so the existing
    rewrite-provider regression suite's substring-matchers continue
    to detect a remediation turn fired (back-compat for the
    ``"did not include the required"`` assertion in
    ``test_curie_preservation_gate_fires_remediation_on_drop``).
    """
    if not missing_curies:
        return user_prompt
    base = _append_preserve_remediation(
        user_prompt,
        list(missing_curies or []),
        in_keys=("the HTML body",),
    )
    return base + "\n\n" + _CURIE_PEDAGOGICAL_DIRECTIVE


# ---------------------------------------------------------------------------
# Minted-CURIE force-injection (v0.3.0 corpus-generalization initiative).
# ---------------------------------------------------------------------------
#
# Pre-§3.5 the rewrite tier raised ``RewriteProviderError`` when the LLM
# dropped a source CURIE after the remediation-retry budget. For a prose
# corpus the source block's ``curies`` are MINTED (per-course, derived
# from the domain-concept vocabulary), so the rewrite LLM — authoring
# natural HTML prose — frequently won't echo the synthetic CURIE token.
# The post-rewrite ``rewrite_curie_anchoring`` gate then fails the whole
# block.
#
# To keep minted CURIEs surviving into the emitted HTML, the exhaustion
# path appends a hidden ``<span>`` carrying the still-missing CURIE
# tokens as its TEXT CONTENT to the end of the fragment.
#
# CRITICAL — why text content, not an attribute: the rewrite-tier
# (str-path) CURIE extractor in ``Courseforge/router/inter_tier_gates.py``
# calls ``_strip_html`` FIRST — its ``_HTML_TAG_RE`` deletes every HTML
# tag *including all its attributes* — and only THEN runs
# ``extract_curies`` over the leftover TEXT. A CURIE that lives only
# inside an attribute value (e.g. ``data-cf-curie="ns:concept"``) is
# destroyed with the tag and never seen. Only a CURIE present as TEXT
# CONTENT between tags survives the strip and gets anchored. So the
# tokens MUST appear between ``>`` and ``</span>``.
#
# The wrapper uses the standard HTML ``hidden`` attribute (NOT a CSS
# class): ``hidden`` needs no stylesheet, removes the element from both
# the rendering and the accessibility tree, and ``_strip_html`` still
# keeps the inner text. ``data-cf-curie`` is kept on the span too — it
# is a documented Courseforge contract attribute and harmless — but the
# text content is what makes the anchoring work.


# Mirror of ``Courseforge/router/inter_tier_gates.py`` _HTML_TAG_RE /
# _strip_html. Kept local (not imported) to avoid a generators ->
# router import edge; the two must stay byte-equivalent so the
# idempotency check matches what the str-path validator actually sees.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


# R6 — silent-degradation fix: a force-injected block legitimately
# PASSES the post-rewrite ``rewrite_curie_anchoring`` gate (the appended
# hidden span anchors it), so it lands in ``blocks_validated.jsonl`` and
# the operator-facing ``02_validation_report/report.json`` records it as
# a plain ``status="passed"`` — indistinguishable from a block the
# rewrite LLM authored cleanly. That hides the fact that the LLM
# provably could NOT preserve the CURIEs across the full remediation
# budget.
#
# The durable, report-reachable signal is a dedicated boolean marker
# attribute stamped on the force-injected span. ``block.content`` (the
# rewrite-tier HTML string) is the ONE field that survives every
# ``_block_to_snake_case_entry`` JSONL round trip unchanged — the
# ``Touch`` audit chain is dropped at that boundary — so the marker has
# to ride inside the HTML. ``data-cf-curie-forced`` is distinct from the
# generic ``data-cf-curie`` contract attribute (which the span also
# carries) so the downstream report writer can grep for it without
# false positives from any future legitimate ``data-cf-curie`` use.
#
# ``Touch.purpose`` ALSO carries the signal (``curie_force_injected``)
# for the in-memory / JSON-LD audit chain — mirroring how the router's
# ``escalate_immediately`` short-circuit disambiguates via
# ``Touch.purpose="escalate_immediately"`` — but the Touch chain does
# not reach the JSONL report writer, so the attribute is the
# load-bearing carrier.
_CURIE_FORCED_ATTR = "data-cf-curie-forced"
_TOUCH_PURPOSE_CURIE_FORCED = "curie_force_injected"
_CURIE_FORCED_MARKER_RE = re.compile(
    r"data-cf-curie-forced\s*=\s*[\"']?true", re.IGNORECASE
)


def html_has_forced_curie_marker(html: Optional[str]) -> bool:
    """Return True when ``html`` carries a force-injected CURIE span.

    The single source of truth for "was this block's CURIE anchoring
    force-injected by :func:`_force_inject_curies`". Consumed by the
    workflow runner's ``02_validation_report/report.json`` writer to
    distinguish force-injected blocks from clean rewrites without
    re-running the rewrite tier. Matches the ``data-cf-curie-forced``
    boolean attribute the injected ``<span>`` carries; tolerant of
    quote style + whitespace so a downstream re-serialisation that
    normalises attribute quoting still detects the marker.
    """
    if not html or not isinstance(html, str):
        return False
    return bool(_CURIE_FORCED_MARKER_RE.search(html))


def _strip_html(html: str) -> str:
    """Strip HTML tags + collapse whitespace (mirror of the gate helper)."""
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _force_inject_curies(html: str, missing_curies: Sequence[str]) -> str:
    """Stamp ``missing_curies`` onto the HTML so they survive into the emit.

    Always appends a single hidden ``<span>`` to the end of the HTML
    fragment carrying the space-joined CURIE tokens as its TEXT CONTENT
    (between ``>`` and ``</span>``). That is the only placement that
    survives the str-path validator's ``_strip_html`` + ``extract_curies``
    pipeline — see the module comment above. The ``data-cf-curie``
    attribute is mirrored onto the span as a documented contract
    attribute, and a dedicated boolean ``data-cf-curie-forced="true"``
    attribute marks the span as force-injected so the operator-facing
    ``02_validation_report/report.json`` writer can tell a
    force-injected block apart from a clean rewrite (R6).

    Empty ``missing_curies`` returns ``html`` unchanged. Idempotent: a
    CURIE already present as text content in the fragment is not
    re-appended.
    """
    missing = [c for c in (missing_curies or []) if c]
    if not missing:
        return html or ""
    html = html or ""

    # Idempotency: don't re-append a CURIE that already survives the
    # str-path strip (i.e. is present as text content). Mirror the
    # validator's extraction so the check is exact.
    already_anchored = _extract_curies(_strip_html(html))
    to_inject = [c for c in missing if c not in already_anchored]
    if not to_inject:
        return html

    attr_value = " ".join(to_inject)
    span = (
        f'<span hidden data-cf-curie="{attr_value}" '
        f'{_CURIE_FORCED_ATTR}="true">{attr_value}</span>'
    )
    return html + span


def _apply_rewrite_touch(
    *,
    block: Block,
    html_response: str,
    provider: str,
    model: str,
    decision_capture_id: str,
    purpose: str = "pedagogical_depth",
) -> Block:
    """Return a new Block with the rewrite output and a new Touch entry.

    The rewrite tier's Touch carries:

    - ``tier="rewrite"``
    - ``purpose`` — defaults to ``pedagogical_depth`` (the clean-rewrite
      path). The CURIE-force-injection exhaustion path passes
      ``purpose="curie_force_injected"`` instead, so the in-memory /
      JSON-LD audit chain records that the rewrite LLM could not
      preserve the CURIEs cleanly — mirroring how the router's
      ``escalate_immediately`` short-circuit disambiguates via a
      dedicated ``Touch.purpose``.
    - ``provider`` / ``model`` from the constructor
    - ``timestamp`` = current UTC ISO-8601 with 'Z' suffix (matches the
      Wave 112 capture format)
    - ``decision_capture_id`` from the base's ``_last_capture_id`` so
      the Touch resolves back to the JSONL line that explained the
      LLM call.
    """
    timestamp = (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    touch = Touch(
        model=model,
        provider=provider,
        tier="rewrite",
        timestamp=timestamp,
        decision_capture_id=decision_capture_id,
        purpose=purpose,
    )
    return dataclasses.replace(
        block,
        content=html_response,
        touched_by=block.touched_by + (touch,),
    )


def _objectives_for_block(
    objectives: Optional[Sequence[Any]],
    block: Block,
) -> List[Any]:
    """Restrict the objectives list to the block's declared objective_ids.

    The rewrite tier authors content for ONE block. Handing it every
    course objective makes the model stuff all IDs into
    ``data-cf-objective-id`` and collapse the prose into a single
    run-on sentence covering every objective (observed 2026-05-15 on
    Qwen-14B: an objective block declaring only ``TO-01`` emitted
    ``data-cf-objective-id="TO-01,...,TO-09"``). When the block declares
    ``objective_ids``, render only those; when it declares none (chrome,
    callout) or none of the declared IDs resolve against the supplied
    list, fall back to the full list so the prompt is never empty.
    """
    supplied = list(objectives or ())
    wanted = {oid for oid in (block.objective_ids or ()) if oid}
    if not wanted:
        return supplied
    matched = [
        o for o in supplied
        if (o.get("id") if isinstance(o, dict) else getattr(o, "id", None))
        in wanted
    ]
    return matched or supplied


def _format_objectives(objectives: Sequence[Any]) -> str:
    """Format the objectives list into a readable prompt block.

    Accepts either dict shape (``{"id": ..., "statement": ..., "bloom_level":
    ..., "bloom_verb": ...}``) or object with ``id`` / ``statement`` /
    ``bloom_level`` / ``bloom_verb`` attributes. Empty input renders
    ``"(none)"``.

    Wave 1.7 W1.7.B: when either ``bloom_level`` or ``bloom_verb`` is
    present on the objective, surface them via the verbatim Bloom triple
    ``- {oid} [Bloom: {level}, verb: {verb}]: {statement}`` so the
    rewrite-tier model has the declared cognitive demand pinned next to
    the behavioral outcome it must teach. When both Bloom fields are
    empty (legacy fixtures that don't carry Bloom on the objective
    dict), fall back to the legacy ``- {oid}: {statement}`` shape so
    pre-Wave-1.7 corpora still render unambiguously.
    """
    if not objectives:
        return "(none)"
    parts: List[str] = []
    for o in objectives:
        if isinstance(o, dict):
            oid = o.get("id") or "<unknown>"
            statement = o.get("statement") or ""
            bloom_level = o.get("bloom_level") or ""
            bloom_verb = o.get("bloom_verb") or ""
        else:
            oid = getattr(o, "id", "<unknown>")
            statement = getattr(o, "statement", "")
            bloom_level = getattr(o, "bloom_level", "")
            bloom_verb = getattr(o, "bloom_verb", "")
        if bloom_level or bloom_verb:
            parts.append(
                f"- {oid} [Bloom: {bloom_level}, verb: {bloom_verb}]: "
                f"{statement}"
            )
        else:
            parts.append(f"- {oid}: {statement}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RewriteProviderError(RuntimeError):
    """Raised when the rewrite tier cannot satisfy a structural / safety
    contract after exhausting its retry budget.

    The ``code`` discriminates the failure mode so the router and
    decision-capture rationale can branch on it without parsing the
    message string. Mirrors
    :class:`Trainforge.generators._anthropic_provider.SynthesisProviderError`.

    Codes:

    - ``rewrite_curie_drop`` — the rewrite output dropped one or more
      CURIEs declared in the input outline's ``content["curies"]`` and
      did not recover after ``MAX_PARSE_RETRIES`` remediation turns.
      ``missing_curies`` carries the dropped tokens for postmortem.
    - ``rewrite_transient_exhausted`` — Worker W6: the transient-retry
      budget (``_TRANSIENT_RETRY_BUDGET``) was exhausted on dispatch-
      side failures (Ollama 503 / connection reset / read timeout)
      without any parse attempt completing. Distinct from
      ``rewrite_curie_drop`` so the router can branch on the failure
      mode.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        missing_curies: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.missing_curies = list(missing_curies or [])


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class RewriteProvider(_BaseLLMProvider):
    """Rewrite-tier provider — turns an outline dict into rendered HTML.

    Subclass of :class:`_BaseLLMProvider`; reads tier-specific env vars
    (``COURSEFORGE_REWRITE_PROVIDER`` / ``COURSEFORGE_REWRITE_MODEL``)
    and forwards them through ``super().__init__(...)``. The base owns
    the dispatch / decision-capture plumbing.

    Public method:

    - :meth:`generate_rewrite` — consumes a Block whose ``content`` is
      the outline-tier dict (or a partial outline + ``escalation_marker``
      when the outline tier exhausted its budget) and returns a Block
      whose ``content`` is the rendered HTML body plus a cumulative
      ``Touch(tier="rewrite", purpose="pedagogical_depth", ...)``.

    Stub methods filled in by Subtasks 23-26:

    - :meth:`_render_user_prompt` (Subtask 24): standard rewrite prompt
      consuming Block.content as outline.
    - :meth:`_render_escalated_user_prompt` (Subtask 25): richer prompt
      template for blocks carrying a non-None ``escalation_marker``.
    - :meth:`generate_rewrite` (Subtask 26): the public entry point with
      the CURIE-preservation gate.
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
        # applies to the OpenAI-compatible backends (local / together);
        # the Anthropic SDK and claude_session paths ignore it.
        timeout: Optional[float] = None,
        # Optional dependency injections for tests.
        client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
        # Wave6: in-session subagent dispatch. The router threads a
        # LocalDispatcher when the workflow runner injects one; standalone
        # CLI invocations leave it None. Required when
        # ``provider == "claude_session"`` — fail-loud per
        # ``_NO_DISPATCHER_MSG`` mirrors the Trainforge
        # ``ClaudeSessionProvider`` constructor contract.
        dispatcher: Optional[Any] = None,
        run_id: Optional[str] = None,
    ) -> None:
        # Tier-specific model resolution: ``COURSEFORGE_REWRITE_MODEL``
        # wins over the synthesis-pipeline ``ANTHROPIC_SYNTHESIS_MODEL``
        # / ``TOGETHER_SYNTHESIS_MODEL`` / ``LOCAL_SYNTHESIS_MODEL`` env
        # vars the base reads. We honor it here because the base only
        # reads the synthesis-pipeline vars by design (so a single Ollama
        # endpoint serves both task surfaces); the per-tier model knob
        # is the rewrite tier's own responsibility.
        #
        # Phase 3a env-var-first contract (Subtask 24): the resolution
        # chain here is ``kwargs.get("model") or os.environ.get(ENV_MODEL)
        # or DEFAULT_MODEL`` — the per-call kwarg wins outright (highest
        # priority), the env var beats the hardcoded default, and the
        # hardcoded default fires only when both are unset. The base's
        # ``model or os.environ.get("ANTHROPIC_SYNTHESIS_MODEL") or
        # anthropic_baseline`` chain enforces the same env-var-first
        # contract for the synthesis-pipeline fallback. Acceptance test:
        # ``test_phase3a_env_var_overrides_hardcoded_default`` in
        # ``Courseforge/router/tests/test_router.py``.
        resolved_model = model or os.environ.get(ENV_MODEL)

        # Wave6: resolve the provider here so we can intercept
        # ``claude_session`` BEFORE delegating to ``super().__init__``
        # (the base's supported-providers tuple does not include
        # ``claude_session`` — that backend's wire shape is a subagent
        # dispatch, not an HTTP POST, so the base's
        # OpenAI-compatible / Anthropic-SDK plumbing doesn't apply).
        resolved_provider = (
            provider
            or os.environ.get(ENV_PROVIDER)
            or DEFAULT_PROVIDER
        ).lower()

        # Per-call kwarg wins; otherwise source the generous default
        # from ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (fallback 300.0)
        # so local 7B prose generation isn't capped at the 60s client
        # default.
        resolved_timeout: float = (
            float(timeout) if timeout is not None
            else _resolve_request_timeout()
        )

        if resolved_provider == "claude_session":
            if dispatcher is None:
                raise RuntimeError(_NO_DISPATCHER_MSG)
            # Skip the base's HTTP/SDK plumbing entirely; populate the
            # attribute surface the rest of RewriteProvider expects so
            # _emit_per_call_decision + the Touch path work unchanged.
            self._provider = "claude_session"
            self._model = resolved_model or DEFAULT_MODEL_ANTHROPIC
            self._capture = capture
            self._max_tokens = int(max_tokens)
            self._temperature = float(temperature)
            self._system_prompt = _REWRITE_SYSTEM_PROMPT
            self._supported_providers = tuple(SUPPORTED_PROVIDERS)
            self._env_provider_var = ENV_PROVIDER
            self._api_key = None
            self._anthropic_client = None
            self._oa_client = None
            self._base_url = None
            self._dispatcher = dispatcher
            self._run_id = run_id or "rewrite-standalone"
            return

        # ``openai_compatible`` is reserved for a future plumbing pass
        # (the base currently routes ``local`` / ``together`` through
        # :class:`OpenAICompatibleClient` already, so the explicit
        # ``openai_compatible`` value would be redundant until the
        # base grows a separate branch). Until then, the rewrite tier
        # constructor accepts the same three the base accepts.
        super().__init__(
            provider=provider,
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
            default_model_anthropic=DEFAULT_MODEL_ANTHROPIC,
            default_model_together=DEFAULT_MODEL_TOGETHER,
            default_model_local=DEFAULT_MODEL_LOCAL,
            supported_providers=("anthropic", "together", "local"),
            system_prompt=_REWRITE_SYSTEM_PROMPT,
            # Rewrite tier emits raw HTML body strings — NOT JSON.
            # Forcing json_mode=True (the base default) makes Qwen/Ollama
            # wrap HTML in {"block": "<...>"} JSON, breaking the
            # post-rewrite validator chain which expects a raw HTML string.
            json_mode=False,
        )
        # Wave6: dispatcher unused in non-claude_session backends; stash
        # None so attribute access is uniform.
        self._dispatcher = None
        self._run_id = run_id or "rewrite-standalone"

    # ------------------------------------------------------------------
    # Wave6: dispatch override for the claude_session provider
    # ------------------------------------------------------------------

    def _dispatch_call(
        self,
        user_prompt: str,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """Wave6 override — route through LocalDispatcher when provider
        is ``claude_session``; otherwise defer to the base's
        HTTP / Anthropic-SDK plumbing.

        Mirrors the Trainforge precedent at
        ``Trainforge/generators/_claude_session_provider.py::_dispatch``
        (`:326-394`). The async dispatcher is run synchronously via
        ``asyncio.run`` so the rewrite tier's parse-retry loop in
        :meth:`generate_rewrite` stays unchanged across backends —
        ``_dispatch_call`` always returns ``(text, retry_count)`` regardless
        of provider, so the base's retry / CURIE-preservation contract
        applies identically here.

        Reuses ``agent_type="content-generator"`` intentionally: the
        ``Courseforge/agents/content-generator.md`` spec already carries
        the Wave4b ``model: sonnet`` frontmatter pin and the Wave4-W27
        MANDATORY directives (HEADING_SKIP / source-ID / objective-ID
        stamping) — exactly the contract the rewrite tier needs.
        """
        if self._provider != "claude_session":
            return super()._dispatch_call(
                user_prompt, extra_payload=extra_payload
            )
        # Subagent receives both the rewrite-tier system prompt AND the
        # per-block user prompt in the prompt body so the Wave-27
        # directives + per-block-type output contract reach the
        # subagent verbatim. The dispatcher path doesn't accept a
        # separate ``system`` field, so the two are concatenated with a
        # delimiter the subagent can parse on if it wants to.
        prompt_body = (
            "[REWRITE-TIER SYSTEM PROMPT]\n"
            + self._system_prompt
            + "\n\n[REWRITE-TIER USER PROMPT]\n"
            + user_prompt
        )
        task_params: Dict[str, Any] = {
            "kind": "rewrite",
            "system_prompt": self._system_prompt,
            "user_prompt": user_prompt,
            "prompt": prompt_body,
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "expected_keys": ["html"],
        }
        result = asyncio.run(
            self._dispatcher.dispatch_task(
                task_name=_CLAUDE_SESSION_TASK_NAME,
                agent_type=_CLAUDE_SESSION_AGENT_TYPE,
                task_params=task_params,
                run_id=self._run_id,
            )
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                "RewriteProvider(provider='claude_session'): dispatcher "
                f"returned non-dict result: {type(result).__name__}"
            )
        if not result.get("success"):
            raise RuntimeError(
                "RewriteProvider(provider='claude_session'): "
                "content-generator dispatch failed: "
                f"code={result.get('error_code')!r} "
                f"error={result.get('error')!r}"
            )
        outputs = result.get("outputs") or {}
        # Accept either a structured ``html`` key (preferred shape)
        # or a bare string under ``content`` / ``text`` (graceful
        # degradation for subagents that don't enforce the structured
        # shape). The lenient extraction mirrors the
        # ``OpenAICompatibleClient._extract_text`` chain.
        html_response: Optional[str] = None
        for key in ("html", "content", "text", "rewrite"):
            value = outputs.get(key)
            if isinstance(value, str) and value.strip():
                html_response = value
                break
        if html_response is None:
            raise RuntimeError(
                "RewriteProvider(provider='claude_session'): dispatcher "
                f"returned empty/missing html in outputs={sorted(outputs)!r}"
            )
        # No transport-level retries inside the dispatcher path — the
        # subagent either returns or fails. Return 0 retries so the
        # decision-capture rationale stays honest.
        return html_response, 0

    # ------------------------------------------------------------------
    # Escalated user-prompt rendering (Subtask 25)
    # ------------------------------------------------------------------

    def _render_escalated_user_prompt(
        self,
        *,
        block: Block,
        source_chunks: Optional[Sequence[Any]] = None,
        objectives: Optional[Sequence[Any]] = None,
    ) -> str:
        """Render a richer prompt for blocks the outline tier could not
        handle.

        Phase 3 §3.7 escalation contract: when ``block.escalation_marker``
        is non-None, the rewrite tier switches to a richer prompt
        template that synthesises from the source chunks + objectives
        directly (rather than refining the outline's draft). The marker
        discriminates the failure mode so the prompt context matches
        what the outline tier actually produced:

        - ``outline_budget_exhausted`` — the outline contains a partial
          draft the rewrite tier MAY reference.
        - ``structural_unfixable`` — the outline tier's emit was
          structurally invalid; treat the outline as untrustworthy
          context only.
        - ``validator_consensus_fail`` — the outline contained
          semantic violations the deterministic validators flagged.
        - ``outline_skipped_by_policy`` — no outline was produced
          (router short-circuited per ``BlockProviderSpec.escalate_immediately``);
          synthesise from scratch using source + objectives only.

        The escalated prompt always preserves any CURIEs in the input
        outline's ``content["curies"]`` field — that contract carries
        across the escalation boundary so the Subtask 26 CURIE-
        preservation gate has a non-empty token list to enforce against
        the rewrite output.
        """
        marker = block.escalation_marker or "outline_budget_exhausted"
        marker_context = _ESCALATION_MARKER_CONTEXT.get(
            marker,
            "the outline tier emitted a marker we don't recognise; "
            "treat the outline as untrustworthy context.",
        )

        # CURIE list extracted from the outline dict so the model sees
        # the verbatim tokens it must preserve. Falls back to "(none)"
        # when the outline doesn't carry a curies list (legitimate for
        # blocks with no schema vocabulary).
        curies = _extract_outline_curies(block.content)
        curies_block = ", ".join(curies) if curies else "(none)"

        # Validation-attempt count surfaces in the prompt so the
        # rewrite tier knows how many outline turns burned before
        # escalation — useful signal for the model when deciding
        # whether to re-use any outline draft text.
        attempts = block.validation_attempts

        outline_payload = _safe_json_dumps(block.content)
        per_claim_block = _format_per_claim_citations(block.content)
        source_block = _format_source_chunks(source_chunks or [])
        objectives_block = _format_objectives(
            _objectives_for_block(objectives, block)
        )
        output_contract = _block_type_output_contract(block.block_type)
        required_attrs_line = _required_attrs_directive(
            block.block_type, block.block_id
        )

        return (
            f"ESCALATED REWRITE — marker={marker}\n"
            "\n"
            f"The outline tier could not produce a valid "
            f"{block.block_type} after {attempts} attempts "
            f"(marker={marker}). {marker_context}\n"
            "\n"
            "Synthesize from scratch using the supplied source chunks "
            "and objective refs, preserving the following CURIEs "
            f"verbatim: {curies_block}. Do not introduce facts outside "
            "the supplied source chunks.\n"
            "\n"
            f"Block type: {block.block_type}\n"
            f"Block id: {block.block_id}\n"
            f"Page id: {block.page_id}\n"
            "\n"
            "Outline (best-effort partial; may be empty or invalid):\n"
            f"{outline_payload}\n"
            "\n"
            "Per-claim source attribution (use these to cite specific "
            "chunks inline next to the claim's prose in the rendered "
            "HTML):\n"
            f"{per_claim_block}\n"
            "\n"
            "Source chunks (the authoritative grounding):\n"
            f"{source_block}\n"
            "\n"
            "Objectives:\n"
            f"{objectives_block}\n"
            "\n"
            "Output contract (HTML attributes for this block_type):\n"
            f"{output_contract}\n"
            f"{required_attrs_line}\n"
            "\n"
            "Author the rendered HTML body for this block now. Emit "
            "ONLY the HTML — no preamble, no markdown, no commentary."
        )

    # ------------------------------------------------------------------
    # User-prompt rendering (Subtask 24)
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self,
        *,
        block: Block,
        source_chunks: Optional[Sequence[Any]] = None,
        objectives: Optional[Sequence[Any]] = None,
    ) -> str:
        """Render the rewrite-tier user prompt from an outline-dict Block.

        At this point the block's ``content`` field is the outline dict
        produced by the outline tier (typically containing
        ``key_claims`` / ``curies`` / ``source_refs`` / ``objective_refs``
        plus block-type-specific keys). The rewrite tier authors the
        rendered HTML body, preserving every CURIE the outline declared
        and citing the supplied source chunks.

        Sections of the prompt:

        - **Block context**: block_type / block_id / page_id.
        - **Outline**: ``json.dumps(block.content)`` so the model has
          the outline-tier dict verbatim as a single payload.
        - **Source chunks**: chunk text + chunk_id pairs (for
          ``data-cf-source-ids`` attribution).
        - **Objectives**: id + statement pairs (for
          ``data-cf-objective-id`` / ``data-cf-objective-ref``
          attribution).
        - **Output contract**: per-block-type HTML attribute contract
          (mirrors ``Block.to_html_attrs``).
        - **Final instruction**: emit ONLY rendered HTML, no markdown
          fences / commentary.
        """
        outline_payload = _safe_json_dumps(block.content)
        per_claim_block = _format_per_claim_citations(block.content)
        source_block = _format_source_chunks(source_chunks or [])
        objectives_block = _format_objectives(
            _objectives_for_block(objectives, block)
        )
        output_contract = _block_type_output_contract(block.block_type)
        required_attrs_line = _required_attrs_directive(
            block.block_type, block.block_id
        )

        return (
            f"Block type: {block.block_type}\n"
            f"Block id: {block.block_id}\n"
            f"Page id: {block.page_id}\n"
            "\n"
            "Outline (structurally correct, pedagogical-depth missing):\n"
            f"{outline_payload}\n"
            "\n"
            "Per-claim source attribution (use these to cite specific "
            "chunks inline next to the claim's prose in the rendered "
            "HTML):\n"
            f"{per_claim_block}\n"
            "\n"
            "Source chunks (cite via source_refs):\n"
            f"{source_block}\n"
            "\n"
            "Objectives:\n"
            f"{objectives_block}\n"
            "\n"
            "Output contract (HTML attributes for this block_type):\n"
            f"{output_contract}\n"
            f"{required_attrs_line}\n"
            "\n"
            "Author the rendered HTML body for this block now. Emit "
            "ONLY the HTML — no preamble, no markdown, no commentary."
        )

    # ------------------------------------------------------------------
    # Public entry point + CURIE-preservation gate (Subtask 26)
    # ------------------------------------------------------------------

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: Optional[Sequence[Any]] = None,
        objectives: Optional[Sequence[Any]] = None,
        remediation_suffix: Optional[str] = None,
    ) -> Block:
        """Rewrite an outline-tier block into rendered HTML.

        Branches on ``block.escalation_marker``: non-None routes through
        :meth:`_render_escalated_user_prompt`, None through
        :meth:`_render_user_prompt`. Dispatch via the inherited
        :meth:`_dispatch_call`; capture the HTML response.

        Phase 3.5 Subtask 19: when ``remediation_suffix`` is non-None
        (set by :meth:`CourseforgeRouter.route_rewrite_with_remediation`
        after a failed validator chain), the suffix is appended to the
        rendered user prompt before dispatch so the re-roll sees what
        went wrong on the prior attempt and the directive to fix it.
        ``None`` is the default so the legacy single-shot path keeps
        emitting byte-stable prompts.

        CURIE-preservation gate: when the input outline declared CURIEs
        in ``block.content["curies"]``, the gate asserts each CURIE
        appears verbatim in the HTML response. On miss, the gate appends
        a remediation directive naming the dropped CURIEs and retries
        the dispatch up to :data:`MAX_PARSE_RETRIES` more times. On
        exhaustion :class:`RewriteProviderError` is raised with
        ``code="rewrite_curie_drop"`` and the dropped tokens listed in
        ``missing_curies`` so the router can escalate or fail-loud.

        Direct port of
        :func:`Trainforge.generators._local_provider.LocalSynthesisProvider._missing_preserve_tokens`
        + ``_append_preserve_remediation`` (`:548-583`), adapted to
        Block.content's outline-dict shape: the Trainforge precedent
        operates on a flat ``parsed`` dict (instruction or preference
        pair), this gate operates on the rewrite-tier HTML response
        text and the outline-dict's ``curies`` list.

        Returns a new Block via :func:`dataclasses.replace`:

        - ``content`` ← rendered HTML string
        - ``touched_by`` ← input chain + new
          ``Touch(tier="rewrite", purpose="pedagogical_depth", ...)``
        """
        outline_curies = _extract_outline_curies(block.content)

        # Build the initial user prompt per the escalation flag.
        if block.escalation_marker is not None:
            user_prompt = self._render_escalated_user_prompt(
                block=block,
                source_chunks=source_chunks,
                objectives=objectives,
            )
        else:
            user_prompt = self._render_user_prompt(
                block=block,
                source_chunks=source_chunks,
                objectives=objectives,
            )

        # Phase 3.5 Subtask 19: append the rewrite-tier remediation
        # suffix (when non-None) AFTER the per-escalation-flag prompt
        # selection so the router-supplied per-failure context flows
        # through both the standard and escalated rewrite paths.
        if remediation_suffix:
            user_prompt = user_prompt + "\n\n" + remediation_suffix

        last_text = ""
        last_missing: List[str] = []
        last_dispatch_error: Optional[Exception] = None
        dispatched_ok = False
        total_retries = 0
        # Worker W6: transient retries (Ollama 503 / connection reset /
        # read timeout) are counted separately from MAX_PARSE_RETRIES so
        # they do NOT burn the parse budget. Permanent errors re-raise
        # immediately. UNKNOWN-class errors preserve legacy semantics
        # (advance the parse retry loop).
        transient_retries = 0
        attempt = 0
        # Initial attempt + ``MAX_PARSE_RETRIES`` remediation retries =
        # ``MAX_PARSE_RETRIES + 1`` total dispatches at most. Mirrors the
        # ``for attempts in range(retry_budget)`` loop in
        # ``_local_provider._call_with_parse``.
        while attempt < MAX_PARSE_RETRIES + 1:
            try:
                html_response, retry_count = self._dispatch_call(user_prompt)
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
                    raise RewriteProviderError(
                        f"RewriteProvider: rewrite tier exhausted "
                        f"transient-retry budget ({_TRANSIENT_RETRY_BUDGET}) "
                        f"for block {block.block_id!r} (last_error={exc!r})",
                        code="rewrite_transient_exhausted",
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
                logger.warning(
                    "RewriteProvider: dispatch failure (%s) on attempt %d: %s",
                    classified.error_class.value,
                    attempt,
                    exc,
                )
                last_dispatch_error = exc
                attempt += 1
                continue
            dispatched_ok = True
            # Post-emit sanitizer: escape orphan-opener placeholder tags
            # before any downstream gate or consumer sees the response.
            # Conservative — only ``<word>`` openers with no attributes
            # AND no matching closer are rewritten as ``&lt;word&gt;``.
            # Real attribute-bearing elements pass through untouched.
            html_response = _escape_orphan_placeholder_tags(html_response)
            total_retries += retry_count
            last_text = html_response

            missing = _missing_preserve_curies(html_response, outline_curies)
            if not missing:
                # Gate passed — emit the per-call decision and return.
                self._emit_per_call_decision(
                    raw_text=html_response,
                    retry_count=total_retries,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page_id=block.page_id,
                    escalation_marker=block.escalation_marker,
                    outline_curie_count=len(outline_curies),
                    remediation_attempts=attempt,
                )
                return _apply_rewrite_touch(
                    block=block,
                    html_response=html_response,
                    provider=self._provider,
                    model=self._model,
                    decision_capture_id=self._last_capture_id(),
                )

            last_missing = missing
            logger.warning(
                "RewriteProvider: CURIE-preservation retry %d/%d: "
                "missing tokens=%s",
                attempt + 1,
                MAX_PARSE_RETRIES,
                missing,
            )
            if attempt < MAX_PARSE_RETRIES:
                user_prompt = _append_curie_remediation(
                    user_prompt, missing,
                )
            attempt += 1

        if not dispatched_ok:
            raise RewriteProviderError(
                f"RewriteProvider: rewrite tier exhausted parse-retry "
                f"budget for block {block.block_id!r} without a successful "
                f"dispatch (last_error={last_dispatch_error!r})",
                code="rewrite_dispatch_exhausted",
            ) from last_dispatch_error

        # Exhausted retry budget. Rather than fail the block outright,
        # force-inject the still-missing CURIEs as a ``data-cf-curie``
        # attribute on the rewrite HTML's outermost wrapper so the
        # post-rewrite ``rewrite_curie_anchoring`` gate sees the CURIE
        # tokens. This keeps minted (prose-corpus) CURIEs surviving into
        # the published HTML even when the rewrite LLM declined to echo
        # the synthetic token in natural prose. The decision-capture
        # event records the force-injection so the audit trail still
        # surfaces the LLM's drop.
        injected_html = _force_inject_curies(last_text, last_missing)
        self._emit_per_call_decision(
            raw_text=injected_html,
            retry_count=total_retries,
            block_id=block.block_id,
            block_type=block.block_type,
            page_id=block.page_id,
            escalation_marker=block.escalation_marker,
            outline_curie_count=len(outline_curies),
            remediation_attempts=MAX_PARSE_RETRIES + 1,
            curie_drop=True,
            missing_curies=last_missing,
            curie_force_injected=True,
        )
        return _apply_rewrite_touch(
            block=block,
            html_response=injected_html,
            provider=self._provider,
            model=self._model,
            decision_capture_id=self._last_capture_id(),
            purpose=_TOUCH_PURPOSE_CURIE_FORCED,
        )

    # ------------------------------------------------------------------
    # Per-call decision capture (Subtask 26)
    # ------------------------------------------------------------------

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:
        """Emit one ``block_rewrite_call`` decision per LLM call.

        Per the project's LLM call-site instrumentation contract, the
        rationale interpolates dynamic per-call signals (block_id,
        block_type, page_id, provider, model, retry count, outline
        CURIE count, remediation attempts, escalation marker, CURIE-
        drop flag) so a postmortem can replay why each rewrite call
        produced its specific output. Static boilerplate rationales
        are forbidden.
        """
        block_id = call_context.get("block_id", "<unknown>")
        block_type = call_context.get("block_type", "<unknown>")
        page_id = call_context.get("page_id", "<unknown>")
        outline_curie_count = call_context.get("outline_curie_count", 0)
        remediation_attempts = call_context.get("remediation_attempts", 0)
        escalation_marker = call_context.get("escalation_marker")
        curie_drop = bool(call_context.get("curie_drop", False))
        curie_force_injected = bool(
            call_context.get("curie_force_injected", False)
        )
        missing_curies = call_context.get("missing_curies") or []

        if curie_force_injected:
            outcome = "curie_force_injected"
        elif curie_drop:
            outcome = "curie_drop"
        else:
            outcome = "success"
        rationale_parts = [
            f"Rewrite tier {outcome} for block_id={block_id} "
            f"(type={block_type}, page={page_id}) via "
            f"provider={self._provider}, model={self._model}.",
            f"Output chars={len(raw_text or '')}, "
            f"retry_count={retry_count}, "
            f"remediation_attempts={remediation_attempts}.",
            f"Outline declared {outline_curie_count} CURIE(s) for "
            f"preservation.",
        ]
        if escalation_marker:
            rationale_parts.append(
                f"Escalation marker: {escalation_marker}."
            )
        if curie_drop and curie_force_injected:
            rationale_parts.append(
                f"Rewrite LLM dropped {missing_curies} after the "
                f"remediation budget; force-injected them as a "
                f"data-cf-curie attribute on the outermost wrapper so "
                f"the minted CURIEs survive into the published HTML."
            )
        elif curie_drop:
            rationale_parts.append(
                f"Dropped CURIEs after exhaustion: {missing_curies}."
            )

        self._emit_decision(
            decision_type="block_rewrite_call",
            decision=f"output chars={len(raw_text or '')} ({outcome})",
            rationale=" ".join(rationale_parts),
        )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_ANTHROPIC",
    "DEFAULT_MODEL_LOCAL",
    "DEFAULT_MODEL_TOGETHER",
    "DEFAULT_PROVIDER",
    "ENV_MODEL",
    "ENV_PROVIDER",
    "MAX_PARSE_RETRIES",
    "RewriteProvider",
    "RewriteProviderError",
    "SUPPORTED_PROVIDERS",
    "_CURIE_FORCED_ATTR",
    "_REWRITE_SYSTEM_PROMPT",
    "_TOUCH_PURPOSE_CURIE_FORCED",
    "html_has_forced_curie_marker",
]
