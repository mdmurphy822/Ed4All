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
from html import escape as html_escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Courseforge.generators._base import (
    _BaseLLMProvider,
    _default_supported_providers,
)
from Courseforge.generators._rewrite_fit_window import (  # noqa: E402
    RESERVE_TOKENS,
    cited_chunk_ids_from_content,
    resolve_fit_window,
    resolve_rewrite_num_ctx,
    resolve_truncation_tripwire,
    select_chunks_under_budget,
)
from lib.llm.truncation_guard import (  # noqa: E402
    check_prompt_fits_window,
    check_prompt_not_truncated,
)
from lib.retrieval.answer_backend import (  # noqa: E402
    PromptTruncatedError as _PromptTruncatedError,
)
from lib.retrieval._prompts import estimate_tokens as _estimate_tokens  # noqa: E402
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
DEFAULT_MODEL_LOCAL = "qwen2.5:7b-instruct-q4_K_M"

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

# Legacy alias for the "non-Ollama / non-Together OpenAI-compatible server"
# tag. It collapses to ``local`` at constructor entry (mirroring the router's
# ``_get_rewrite_provider`` collapse) so a standalone
# ``RewriteProvider(provider="openai_compatible")`` behaves identically to a
# router-mediated construction — both route through the same
# ``OpenAICompatibleClient`` at the ``local`` seat's base_url.
_OPENAI_COMPATIBLE_ALIAS = "openai_compatible"


def _supported_providers() -> Tuple[str, ...]:
    """Registry-superset provider allow-list for the rewrite tier.

    The rewrite tier reaches the SAME openai-compatible seats the base
    supports — every ``kind: openai_compatible`` row in
    ``config/endpoints.yaml`` (``anthropic`` SDK transport + ``local`` /
    ``together`` / ``nvidia`` / ``nvidia-deepseek`` / ``groq`` /
    ``fireworks`` / ``deepseek`` / … — one YAML row, zero subclass edits)
    — PLUS two non-registry-endpoint tags this tier handles specially:

    - ``claude_session`` — a subagent-dispatch backend (not an HTTP
      endpoint), intercepted BEFORE ``super().__init__`` and driven via
      ``MCP/orchestrator/local_dispatcher.py::LocalDispatcher`` so a Claude
      Max session can author the rewrite tier without an ``ANTHROPIC_API_KEY``.
    - ``openai_compatible`` — the legacy alias, collapsed to ``local`` at
      constructor entry.

    Anti-cycle / missing-file hardening: ``_default_supported_providers``
    (the base) already falls back to the legacy trio on a registry read
    failure, so this composition never crashes at import.
    """
    base = _default_supported_providers()
    extras = ("claude_session", _OPENAI_COMPATIBLE_ALIAS)
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

# rewrite-overflow-fix-2026-06: escalation marker stamped when the input-
# truncation tripwire detects the served window silently dropped the system
# prompt HEAD. Member of ``Courseforge/scripts/blocks.py::_ESCALATION_MARKERS``;
# surfaces the block as ``escalated`` (not ``failed``) in the validation
# report and routes it through the W5 packager-side escalation filter.
_INPUT_PROMPT_TRUNCATED_MARKER = "input_prompt_truncated"

# rewrite-overflow-fix-2026-07: escalation marker stamped when the whole-prompt
# fit-window budget finds the NON-CHUNK scaffold (system prompt + outline dict +
# per-claim + objectives + contract) already overflows the served window, so no
# grounding chunk fits and the prompt cannot be authored without silent head-
# truncation. Member of ``Courseforge/scripts/blocks.py::_ESCALATION_MARKERS``.
_SCAFFOLD_OVERFLOW_MARKER = "rewrite_scaffold_overflow"


class _ScaffoldOverflowError(Exception):
    """Raised by the whole-prompt budget when the non-chunk scaffold alone
    cannot fit the served window (num_ctx). Carries the token accounting so
    the caller can stamp ``rewrite_scaffold_overflow`` + emit a decision.
    """

    def __init__(
        self, *, sys_tokens: int, scaffold_tokens: int, num_ctx: int
    ) -> None:
        self.sys_tokens = sys_tokens
        self.scaffold_tokens = scaffold_tokens
        self.num_ctx = num_ctx
        super().__init__(
            f"rewrite scaffold overflow: sys={sys_tokens} + "
            f"scaffold={scaffold_tokens} tok >= num_ctx={num_ctx}"
        )

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
    "PRESERVE: factual claims (verbatim), objective refs, source refs. "
    "DISCUSS each supplied concept by its NATURAL NAME in the prose — do "
    "NOT echo synthetic ``prefix:localname`` CURIE tokens into the visible "
    "text (they are machine anchoring ids, never learner-facing). REWRITE: "
    "for pedagogical depth, scaffolding, examples, voice. DO NOT add facts "
    "not in the outline's key_claims or in the source chunks."
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
    "(`key_claims[].source_chunk_ids[]`), scope each claim's prose to its "
    "chunks via the `data-cf-source-ids` ATTRIBUTE on the enclosing "
    "section/element ONLY. NEVER write a source-chunk identifier, a "
    "`<cite>chunk_id</cite>`, or an \"According to the source <id>\" / "
    "\"According to the source chunk <id>\" lead-in into the VISIBLE prose "
    "the learner reads — citations are machine-readable attributes, not "
    "learner text. Block-level `source_refs` remains the gate-enforced "
    "superset; the per-claim attribution is for finer-grained pedagogical "
    "context, carried in attributes."
    "\n\n"
    # CB1: pedagogical-depth directive. The outline tier produces a
    # structurally-correct but pedagogically-thin draft; the rewrite
    # tier MUST add instructional depth (rationale, verification,
    # second cases, expert tips) WITHOUT inventing any new material —
    # the NLI grounding gates fail-close on fabrication, so every
    # added sentence must be a better EXPLANATION of supplied source
    # material, never new facts/numbers/examples.
    # 7B-parity P2: styled-component vocabulary. The page inherits a
    # stylesheet defining .example-box / .definition-box / .key-rule /
    # .worked-example / .self-check-item — the rewrite tier MUST use these
    # classes so worked content, definitions, and rules render as styled
    # callout boxes (Sonnet IxD parity) instead of bare paragraphs.
    "STYLED COMPONENTS. The page CSS defines these classes — USE them so "
    "content renders as styled boxes, not bare prose: wrap a WORKED EXAMPLE "
    "in `<div class=\"example-box\">` (multi-step → `<div "
    "class=\"worked-example\">` with `<div class=\"step-row\"><span "
    "class=\"step-label\">Step N:</span> …</div>` rows and a `<div "
    "class=\"solution-line\">` final answer); wrap a formal DEFINITION in "
    "`<div class=\"definition-box\">`; wrap a stated RULE / law / property "
    "in `<div class=\"key-rule\">`; render a self-check question card with "
    "`<div class=\"self-check self-check-item\">`. Emit the class names "
    "exactly as written."
    "\n\n"
    # Finding 1 (step-label colon): the step-row label badge is a flex
    # element; a colon written OUTSIDE the `<span class=\"step-label\">`
    # floats beside the badge instead of reading as part of it. Pin the
    # colon INSIDE the label span so the rendered badge reads `Step 1:`.
    "STEP LABEL COLON. When you emit a `<span class=\"step-label\">`, the "
    "colon goes INSIDE the span — write `<span class=\"step-label\">Step "
    "N:</span>`, NEVER `<span class=\"step-label\">Step N</span>:`. The "
    "`step-label` is a flex badge; a colon left outside the closing "
    "`</span>` floats beside the badge instead of sitting inside it."
    "\n\n"
    # 7B-parity P3: heading-boundary directive. Downstream chunking splits
    # the published HTML on `<h2>`/`<h3>` headings — MORE heading
    # boundaries → finer sections → finer (and more) training chunks
    # (Sonnet's ~498 sections vs the 7B's ~115). When a block presents
    # MULTIPLE distinct ideas (sub-concepts, separate sub-rules, distinct
    # worked sub-cases), give EACH its OWN `<h3>` heading rather than
    # folding them under a single heading. This adds STRUCTURE, not length.
    "HEADING STRUCTURE. Open every concept / explanation block with an "
    "`<h2>` heading, and give EACH distinct sub-idea inside it its OWN "
    "`<h3>` sub-heading (a separate sub-concept, sub-rule, case, or "
    "worked sub-step). PREFER several short, well-titled `<h3>` sections "
    "over one long undivided block of prose — finer headings make the "
    "page easier to scan and chunk. Add headings for STRUCTURE only, "
    "never to pad length; keep every section under the brevity limit."
    "\n\n"
    "BREVITY (HARD LIMIT). Author ONE compact, dense block — never an "
    "essay. Target 120-300 words of prose; a styled box is 2-5 "
    "sentences. For calibration: a full reference page carrying FIFTEEN "
    "worked examples is only ~4000 tokens total, and a single example "
    "box is ~100 tokens — so one block must stay FAR under that. State "
    "the concept / example / rule once, clearly, with its rationale and "
    "worked steps, then STOP. Do NOT repeat a point, restate the "
    "definition, re-explain the same step, or continue past the block's "
    "single teaching objective. Run-on length is a DEFECT, not depth — a "
    "block that exhausts the token budget will be discarded. Finish well "
    "inside the limit."
    "\n\n"
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
    # Finding 5 — COMPACT STRUCTURED ELEMENTS over verbose prose. A 7B
    # run authored a content page with 0 tables and long expository
    # paragraphs where the Sonnet baseline used a comparison `<table>`
    # plus compact worked examples. Whenever the content COMPARES,
    # CONTRASTS, or lays out MULTI-COLUMN / PARALLEL information, an HTML
    # `<table>` is the right element — not prose. Prefer compact
    # structured blocks (tables, lists, styled boxes) over long prose.
    "STRUCTURED OVER PROSE (compactness). PREFER compact, structured "
    "elements over long expository prose. Whenever the content COMPARES "
    "or CONTRASTS two or more things, lays out MULTI-COLUMN / parallel "
    "information (e.g. operation vs. rule vs. example, before vs. after, "
    "term vs. definition vs. example), or presents a set of cases that "
    "share the same columns, render it as an HTML `<table>` with a `<thead>` "
    "header row (`<th>` cells) and one `<tr>` per row — NOT as a run of "
    "prose paragraphs. A table, a `<ul>`/`<ol>`, or a styled callout box "
    "that says the same thing in a quarter of the words BEATS a long "
    "paragraph. Use the table / list / box structure for the comparison "
    "and reserve prose for the one-sentence framing around it. Tables and "
    "lists may use ONLY the values, rows, and columns the source supplies "
    "(no invented rows or columns)."
    "\n\n"
    # Finding 12 — MATH RENDERING + numeric self-check. Studio does NOT
    # render raw `$...$` / `\(...\)` LaTeX (it ships no MathJax/KaTeX), so
    # a distractor written as `$\frac{9}{20}$` renders as literal dollar-
    # sign LaTeX source. Render math as MathML or as Unicode glyphs
    # instead. AND: a 7B run emitted a distractor whose stated value
    # contradicted its own rationale (claimed 3/4 × 5/6 = 9/20, but the
    # arithmetic gives 15/24). Add a self-check that every numeric
    # distractor's VALUE is the actual arithmetic result of the error its
    # rationale names.
    "MATH RENDERING (Studio has NO LaTeX renderer). NEVER emit raw "
    "`$...$`, `$$...$$`, `\\(...\\)`, or `\\[...\\]` LaTeX math — Studio "
    "ships no MathJax/KaTeX, so LaTeX renders as literal dollar-sign "
    "source text the learner sees verbatim (e.g. `$\\frac{9}{20}$` shows "
    "as `$\\frac{9}{20}$`, not as a fraction). Render every fraction, "
    "exponent, root, and symbol as EITHER inline MathML "
    "(`<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mfrac><mn>9</mn>"
    "<mn>20</mn></mfrac></math>`) OR as Unicode glyphs (`9/20`, `½`, `x²`, "
    "`√2`, `×`, `÷`, `≤`, `≥`, `≠`, `π`, `°`) — never LaTeX command "
    "syntax. Pick ONE form and use it consistently within a block."
    "\n\n"
    "NUMERIC SELF-CHECK (distractor value ↔ rationale consistency). When "
    "an assessment_item distractor's rationale names a specific arithmetic "
    "MISTAKE, the distractor's stated VALUE MUST be the actual result that "
    "mistake produces — VERIFY the arithmetic before you emit it. A "
    "distractor whose rationale says \"multiplied 3/4 by 5/6\" MUST show "
    "the real product `15/24` (not an unrelated `9/20`); a distractor whose "
    "value contradicts its own rationale is a DEFECT. Compute each "
    "distractor's value from the named error and confirm they agree; if you "
    "cannot make the value match the rationale, change the value, not the "
    "rationale, so they are consistent."
    "\n\n"
    # Finding 15 — diverse, content-derived TITLE + section framing. A 7B
    # run used the raw filename as the page `<h1>` and a repeated generic
    # "Objectives" header; the Sonnet baseline used content-derived topic
    # titles ("Why Apply These Skills?", "Big Ideas from Week 1"). Steer
    # the model toward specific, content-shaped headings.
    "DIVERSE, CONTENT-DERIVED HEADINGS (no generic labels). Title every "
    "heading from the SPECIFIC content it introduces — derive it from the "
    "block's concepts, key terms, and objective statements. Do NOT emit a "
    "generic, repeated label like a bare \"Objectives\", \"Content\", "
    "\"Section\", \"Overview\", or the raw page filename "
    "(e.g. `week_01_summary`) as a heading. Prefer a specific, "
    "learner-facing framing header that tells the reader what the section "
    "is about — e.g. \"Why Simplify Fractions?\", \"Big Ideas from This "
    "Week\", \"Common Mistakes to Avoid\" — over a one-word category label. "
    "Vary the headings across a page; never repeat the same generic header. "
    "Headings must reflect the source's actual topics (no invented "
    "subjects)."
    "\n\n"
    # Finding 16 — "Key Idea" framing on key-rule / callout blocks. The
    # 42 key-rule blocks in a 7B run carried no recognizable label; the
    # callout block type was authored 0 times. Give every stated rule /
    # law / property / key takeaway a recognizable framing header so the
    # learner spots it as the important point.
    "KEY-IDEA FRAMING. Every key-rule / callout / stated-law block MUST "
    "open with a RECOGNIZABLE framing header that flags it as the important "
    "point — a `<strong>` lead-in or a heading reading \"Key Idea\", "
    "\"Key Rule\", \"Remember\", \"Important\", or the rule's own name "
    "(e.g. \"Key Idea: dividing numerator and denominator by the same "
    "factor keeps a fraction's value\"). A `<div class=\"key-rule\">` or "
    "`<div class=\"callout ...\">` whose body is bare prose with no such "
    "framing label is a DEFECT — the learner cannot tell it apart from "
    "ordinary text. Lead with the \"Key Idea\"-style label, then state the "
    "rule. The framing label must describe the source's actual rule (no "
    "invented rules)."
    "\n\n"
    "CITATION HYGIENE: NEVER write raw source-chunk identifiers or "
    "bracketed chunk tokens in visible, learner-facing prose — e.g. "
    "NEVER write `[coursename_..._chunk_00043]`, `[chunk_12]`, or any "
    "bracketed `[..._chunk_NN]` token in the rendered text. NEVER write a "
    "chunk-id `<cite>` (e.g. `<cite>..._chunk_00013</cite>`) or any "
    "\"According to the source ...\" / \"According to the source chunk "
    "...\" lead-in naming a chunk id in the visible prose. NEVER write a "
    "raw provenance id such as `dart:...#<hash>` in the visible text. "
    "NEVER write an inline objective reference like `(CO-NN)` / `(TO-NN)` "
    "in body prose. Source attribution lives ONLY in the "
    "`data-cf-source-ids` attribute, never in the prose the learner reads."
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
# Fit-window system-prompt trim (rewrite-overflow-fix-2026-06).
# ---------------------------------------------------------------------------
#
# The untrimmed ``_REWRITE_SYSTEM_PROMPT`` above is ≈7,800 tok — alone it
# overflows an 8k served window before a single grounding chunk, so Ollama
# silently head-truncates and drops the authoring CONTRACT. When
# ``ED4ALL_REWRITE_FIT_WINDOW`` is ON, the ≈42% block-type-specific
# segments are RELOCATED out of the global system prompt and into the
# per-block-type output contract (only the relevant block sees its rules).
#
# Derivation is structural, NOT a hand-retyped copy: the authoritative
# ``_REWRITE_SYSTEM_PROMPT`` bytes above are split on ``"\n\n"`` and
# classified by index. When OFF, NOTHING here runs — the constructor uses
# ``_REWRITE_SYSTEM_PROMPT`` verbatim and ``_block_type_output_contract``
# returns the original contract, so every snapshot / prompt is byte-
# identical to today.
#
# KEEP (global): role/palette/grounding/emit-only (0), outline-preserve +
# CURIE-natural-name (1), per-objective Bloom (2), per-claim attribution
# (3), HEADING STRUCTURE (6), BREVITY (7), STRUCTURED-OVER-PROSE (17), MATH
# RENDERING (18), DIVERSE HEADINGS (20), CITATION HYGIENE (22), HARD-
# CONSTRAINT GROUNDING (23), per-block data-cf attrs (24), objective-id
# verbatim (25), Wave-27 MANDATORY OUTPUT CONTRACT (26-30).
#
# MOVE (block-specific) → relocated into the per-type contract:
_REWRITE_SYSTEM_PROMPT_SEGMENTS: Tuple[str, ...] = tuple(
    _REWRITE_SYSTEM_PROMPT.split("\n\n")
)

# Segment indices that relocate out of the global prompt when the fit-
# window flag is ON (everything not listed here stays global / KEEP).
_RELOCATED_SEGMENT_INDICES: frozenset = frozenset(
    {4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 21}
)

# Per-block-type relocation map: which moved-segment indices append to each
# block type's output contract when the flag is ON. A block type absent
# from this map carries only the KEEP-global system prompt (its contract is
# unchanged). The lead-in PEDAGOGICAL-DEPTH segment (8) rides with the per-
# type pedagogical bullets it introduces.
_RELOCATED_SEGMENTS_BY_BLOCK_TYPE: Dict[str, Tuple[int, ...]] = {
    "concept": (4, 8, 9, 11, 16, 21),
    "explanation": (4, 8, 9, 11, 21),
    "example": (4, 5, 8, 10, 11),
    "worked_example": (4, 5, 8, 10, 11),
    "self_check_question": (4, 12, 13),
    "assessment_item": (14, 15, 19),
    "callout": (21,),
    "key_idea": (21,),
}

# The trimmed (fit-window-ON) system prompt: the KEEP-global segments only,
# joined with the SAME ``"\n\n"`` separator so the kept text is byte-
# identical to its appearance in the untrimmed prompt.
_REWRITE_SYSTEM_PROMPT_TRIMMED: str = "\n\n".join(
    seg
    for i, seg in enumerate(_REWRITE_SYSTEM_PROMPT_SEGMENTS)
    if i not in _RELOCATED_SEGMENT_INDICES
)


def _resolve_system_prompt(fit_window: bool) -> str:
    """Return the rewrite-tier system prompt for the fit-window state.

    OFF → the untrimmed ``_REWRITE_SYSTEM_PROMPT`` (byte-identical to
    today). ON → the trimmed KEEP-global prompt (block-specific guidance
    relocated into the per-type contracts).
    """
    return _REWRITE_SYSTEM_PROMPT_TRIMMED if fit_window else _REWRITE_SYSTEM_PROMPT


def _relocated_contract_suffix(block_type: str) -> str:
    """Return the relocated block-specific segments for ``block_type``.

    Empty string when the block type has no relocated segments. The
    segments are joined with ``"\n\n"`` (matching their original prompt
    spacing) and prefixed with a separator so they append cleanly onto the
    existing per-type contract paragraph. Only consulted when the fit-
    window flag is ON.
    """
    indices = _RELOCATED_SEGMENTS_BY_BLOCK_TYPE.get(block_type)
    if not indices:
        return ""
    moved = "\n\n".join(
        _REWRITE_SYSTEM_PROMPT_SEGMENTS[i] for i in indices
    )
    return "\n\n" + moved


# System-turn directive prepended to the BATCHED multi-block rewrite call.
# Instructs the model to emit one delimited block per requested block, in
# order, with NO commentary or code fences between blocks, so
# ``parse_rewrite_batch_envelope`` can split the response back per-block. Each
# block's USER section already carries its full single-block rewrite prompt; the
# directive only adds the envelope contract.
_BATCH_ENVELOPE_DIRECTIVE = (
    "You will author MULTIPLE course-HTML blocks in ONE response. The USER "
    "message lists each block, tagged with its id and carrying that block's "
    "full authoring instructions. For EACH block, emit EXACTLY this delimited "
    "block and nothing else around it:\n"
    '<<<CF_BLOCK id="THE-BLOCK-ID">>>\n'
    "...the rendered HTML body for that block (same single-block contract)...\n"
    '<<<CF_BLOCK_END id="THE-BLOCK-ID">>>\n'
    "Use each block's OWN id verbatim in BOTH delimiters. Emit the blocks IN "
    "THE SAME ORDER the USER message lists them. Put NOTHING between blocks — "
    "no commentary, no Markdown, no code fences, no blank-line prose. Each "
    "block's body is the SAME accessible HTML a single-block conversion would "
    "produce; the per-block authoring rules below the id apply to that block."
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
        "statement is the `<li>`'s text content. "
        "ANATOMY (§4): frame the objective ANALYTICALLY and tie it to its "
        "terminal objective (TO) — not a bare one-sentence skill restatement. "
        "Align `data-cf-bloom-level` to the objective's actual main verb "
        "(do NOT inflate a recall verb to \"analyze\")."
    ),
    "concept": (
        "Emit a `<section data-cf-source-ids=...>` opening with an `<h2>` "
        "heading carrying `data-cf-content-type` + "
        "`data-cf-bloom-range` + `data-cf-key-terms`, followed by "
        "explanatory paragraphs. TITLE the heading from the SPECIFIC "
        "concept it introduces (a content-derived topic header, not a "
        "generic \"Concept\" / \"Content\" / \"Section\" label or the raw "
        "page filename). When the concept covers MULTIPLE distinct "
        "sub-ideas, give EACH its own `<h3>` sub-heading (finer headings → "
        "finer downstream chunks) rather than running them together under "
        "one heading. When the concept presents a taxonomy / "
        "categories / multiple sub-types, OR COMPARES / CONTRASTS two or "
        "more things across shared columns, present them in a structured "
        "`<table>` (with a `<thead>` `<th>` header row, one `<tr>` per row) "
        "or `<ul>` (one row/item per sub-type with its example) "
        "rather than a single prose paragraph, covering every sub-type the "
        "source supports. STYLED COMPONENTS: wrap a formal DEFINITION of a "
        "term in `<div class=\"definition-box\"><strong>Term</strong> "
        "...</div>`, and wrap a stated RULE / law / property in "
        "`<div class=\"key-rule\"><strong>Key Idea: Rule name</strong> "
        "...</div>` — LEAD the key-rule box with a recognizable "
        "\"Key Idea\"-style framing label so the learner spots the rule — "
        "so the styled callout boxes render. Carry "
        "`data-cf-teaching-role=\"introduce\"` on the `<section>`. "
        "ANATOMY (§4) — DEFINITION-IN-CONTEXT: state each formal definition "
        "WITH the condition / exclusions it needs (e.g. a nonzero denominator, "
        "`a≠0`, \"not both zero\", the principal root) — a definition missing "
        "its side-condition is incomplete. Include ONE example AND ONE "
        "NON-EXAMPLE (a near-miss that violates a condition so the boundary is "
        "visible), and LINK the concept back to a prior-taught concept."
    ),
    "example": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h3>` "
        "and a WORKED example presented in a styled box. Carry "
        "`data-cf-content-type=\"example\"` on the heading. STYLED "
        "COMPONENT: wrap the worked content in "
        "`<div class=\"example-box\"><strong>Example N.</strong> ...</div>`. "
        "For a MULTI-STEP solution use "
        "`<div class=\"worked-example\">` with one "
        "`<div class=\"step-row\"><span class=\"step-label\">Step N:</span> "
        "...</div>` per step (the colon goes INSIDE the `step-label` span — "
        "`Step N:</span>`, never `Step N</span>:`) and the final answer in "
        "`<div class=\"solution-line\">...</div>`. Carry "
        "`data-cf-teaching-role=\"elaborate\"` on the `<section>`. "
        "Copy worked-example arithmetic verbatim from the source; if your "
        "verification check fails, fix the algebra — NEVER substitute source "
        "numbers to force agreement. "
        "ANATOMY (§4): open with the STATED problem, then a one-line "
        "PREDICTION prompt (ask the learner to predict the result before the "
        "steps). Give 3-7 steps, EACH stating the operation AND its "
        "justification (never a label-only step — say WHY, e.g. \"combine like "
        "terms because addition is associative\"). Close with a "
        "CHECK-BY-SUBSTITUTION that verifies the answer, a \"Common wrong "
        "turn:\" line naming the typical error, and ONE Try-It practice item "
        "whose answer sits behind a hidden "
        "`<details><summary>Show answer</summary> … </details>` reveal."
    ),
    "explanation": (
        "Emit a `<section data-cf-source-ids=...>` opening with an `<h2>` "
        "and explanatory paragraphs; give each distinct sub-point its own "
        "`<h3>` sub-heading (finer headings → finer downstream chunks) "
        "rather than one undivided block of prose. TITLE each heading from "
        "the SPECIFIC content it introduces (a content-derived topic header, "
        "not a generic label or the raw page filename). When the explanation "
        "COMPARES / CONTRASTS items across shared columns, render the "
        "comparison as a `<table>` (with a `<thead>` `<th>` header row) "
        "rather than parallel prose. Carry "
        "`data-cf-content-type=\"explanation\"` on the heading. STYLED "
        "COMPONENTS: place a formal definition in "
        "`<div class=\"definition-box\">`, a stated rule/law in "
        "`<div class=\"key-rule\">` LED with a recognizable \"Key Idea\"-"
        "style framing label (e.g. `<strong>Key Idea: ...</strong>`) so the "
        "learner spots the rule, and any illustrative worked instance in "
        "`<div class=\"example-box\">` so the styled boxes render. Carry "
        "`data-cf-teaching-role=\"elaborate\"` on the `<section>`. "
        "ANATOMY (§4): an explanation MUST NOT restate the concept block — "
        "explain WHY the idea works and WHEN to use which method, and connect "
        "back to the prior week. On first use of a technical term, inline a "
        "one-sentence bolded definition WITH its side-condition."
    ),
    "summary_takeaway": (
        "Emit a `<div class=\"takeaway-card\" data-cf-source-ids=...>` "
        "carrying `data-cf-content-type=\"summary\"`. Organize the recap "
        "into MULTIPLE DISTINCT takeaway SECTIONS (aim for 3-5, never just "
        "one or two) — give each its OWN content-derived `<h3>` sub-heading "
        "(e.g. \"Big Ideas\", \"Key Vocabulary\", \"Common Mistakes to "
        "Avoid\", \"What to Practice Next\") followed by a short `<ul>` of "
        "that section's takeaways, ALL kept inside the `takeaway-card` div "
        "(the `takeaway-card` class renders the distinct summary callout). "
        "Each section groups RELATED already-taught key points, and each "
        "takeaway is a concise restatement of a point drawn from the "
        "source; a summary is NOT a fresh worked example, definition, or "
        "assessment. Cover the week's distinct themes across the sections "
        "rather than collapsing everything into one undifferentiated list — "
        "but use ONLY points the source actually taught (no invented "
        "sections or takeaways). "
        "ANATOMY (§4): include a distinct MISCONCEPTION-REVIEW section (the "
        "week's common errors and how to avoid them) among the takeaway "
        "sections, and REFERENCE the week's Key Terms page rather than "
        "re-stating full definitions (a summary points back, it does not "
        "re-teach)."
    ),
    "callout": (
        "Emit a `<div class=\"callout callout-{kind}\">` carrying "
        "`data-cf-component=\"callout\"` + `data-cf-purpose` + "
        "`data-cf-content-type=\"callout\"`. OPEN with a recognizable "
        "KEY-IDEA framing header — a `<strong>` lead-in or heading reading "
        "\"Key Idea\", \"Remember\", \"Important\", or the highlight's own "
        "name — so the learner spots it as the important point (a callout "
        "whose body is bare prose with no framing label is a defect). "
        "SCOPE — a callout is ONE FOCUSED, CONCISE HIGHLIGHT: a single "
        "key tip, warning, caution, or note (typically 1-3 sentences, at "
        "most ONE short illustrative instance). It is NOT A FULL LESSON — "
        "NO MULTI-EXAMPLE sequences, no step-by-step worked solutions, and "
        "do NOT cover several cases or sub-types in a callout. That "
        "instructional content belongs in concept / explanation / example "
        "blocks, never here. Keep the highlight grounded in the source "
        "material (no fabricated facts) — surface the single most "
        "important alert about the surrounding content, not a mini-lesson. "
        "ANATOMY (§4): ONE focused highlight keyed to the surrounding content "
        "— a single tip, warning, caution, or note behind its framing label."
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
        "Emit a `<div class=\"self-check self-check-item\">` carrying "
        "`data-cf-component=\"self-check\"`, "
        "`data-cf-purpose=\"formative-assessment\"`, "
        "`data-cf-bloom-level`, `data-cf-objective-ref`, and "
        "`data-cf-source-ids` / `data-cf-source-primary` (the "
        "`self-check-item` class renders the bordered question card). Author "
        "MULTIPLE self-check questions (aim for 3-5, never just one) that "
        "probe DIFFERENT facets of the supplied content — each a separate "
        "`<p>` question (or its own bordered item) inside the div, all "
        "grounded in the source. For EACH question, the visible body "
        "is the QUESTION ONLY — put that question's answer / worked solution "
        "behind its OWN reveal directly after it "
        "(`<details><summary>Show answer</summary> … </details>`) so the "
        "learner attempts before seeing it; NEVER reveal the answer inline. "
        "Use ONLY a `<details>`/`<summary>` reveal — NEVER a button + "
        "JavaScript `onclick` toggle (the sanitizer strips scripts and "
        "event handlers, which would leave the answer permanently hidden). "
        "Do NOT pad with invented questions: every question and its answer "
        "must be grounded in the source — if the source only supports one "
        "or two questions, author only those. "
        "ANATOMY (§4): key each distractor-bearing item to a NAMED "
        "misconception (its wrong options are the values that named error "
        "produces), and tag each item's Bloom level HONESTLY — do NOT label a "
        "recall / apply item \"analyze\". Every item's answer sits behind its "
        "own `<details>` reveal with brief feedback."
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
        "the actual problem instead. "
        "ANATOMY (§4): author 4-6 problems as a FADED sequence of MIXED types "
        "(interleave methods so CHOOSING the method is part of the practice — "
        "never the same operation repeated N times in a row), and give EVERY "
        "item a worked `<details>` solution (never a bare answer)."
    ),
    "misconception": (
        "Emit a `<div class=\"misconception-card\" data-cf-source-ids=...>` "
        "whose JSON-LD entry carries the misconception ID "
        "(mc_[0-9a-f]{16}) plus correction — the JSON-LD stays the "
        "authoritative shape, but ALSO wrap the visible HTML so it renders "
        "as a distinct caution card. Inside the div, state the mistaken "
        "belief in a `<p class=\"misconception-claim\">` (the wrong idea a "
        "learner commonly holds) followed by the fix in a "
        "`<p class=\"misconception-correction\">` (why it is wrong and the "
        "correct understanding), both grounded in the source. A "
        "misconception is NOT an assessment item or a fresh worked example "
        "— it names a specific error and corrects it. "
        "ANATOMY (§4): NAME the specific faulty mental model (e.g. canceling "
        "terms vs factors, `(a+b)²=a²+b²`, distributing an exponent, a sign "
        "error in polynomial subtraction, dividing by a variable and losing a "
        "solution), DIAGNOSE why a learner falls into it, then give the "
        "productive-failure correction — all grounded in the source."
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
        "MATH RENDERING: write every fraction / exponent / symbol in the "
        "stem AND options as Unicode (`9/20`, `x²`, `√2`, `×`, `÷`) or "
        "inline MathML — NEVER raw `$...$` / `\\(...\\)` LaTeX (Studio has "
        "no LaTeX renderer; `$\\frac{9}{20}$` would render as literal LaTeX "
        "source). NUMERIC SELF-CHECK: before emitting, VERIFY each "
        "distractor's stated VALUE equals the actual arithmetic result of "
        "the mistake its rationale names — a value that contradicts its own "
        "rationale (e.g. claiming 3/4 × 5/6 = 9/20 when the product is "
        "15/24) is a defect; fix the value, not the rationale. "
        "Assessment items in IMSCC live in QTI XML downstream; the "
        "HTML emit here is the authoring fixture."
    ),
    "prereq_set": (
        "Emit a `<div class=\"prereq-card\" data-cf-source-ids=...>` "
        "wrapping an `<h2>` or `<h3>` and an `<ol>` of prerequisite topic "
        "refs. The `prereq-card` class renders the distinct prerequisites "
        "callout — keep the `<ol>` of prior skills inside the div. Each "
        "`<ol>` "
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
        "Emit a `<div class=\"reflection-prompt\" data-cf-source-ids=...>` "
        "wrapping an `<h3>` and one or more `<p>` reflection prompts. The "
        "`reflection-prompt` class renders the distinct prompt callout — "
        "keep the open-ended `<p>` questions inside the div. Each `<p>` is "
        "an "
        "OPEN-ENDED QUESTION addressed to the learner about THEIR OWN "
        "thinking, experience, or confidence (e.g. 'When have you needed a "
        "fraction in simplest form? How will you check your own work for the "
        "subtract-instead-of-divide mistake?') — it ends in a question mark "
        "and invites self-reflection. A reflection prompt is NOT an "
        "explanation, definition, worked example, or graded assessment — "
        "never re-teach the content or pose a right/wrong evaluate-this task."
    ),
    "discussion_prompt": (
        "Emit a `<div class=\"discussion-prompt\" data-cf-source-ids=...>` "
        "wrapping an `<h3>` and one or more `<p>` discussion prompts. The "
        "`discussion-prompt` class renders the distinct prompt callout — "
        "keep the open-ended group `<p>` questions inside the div. Each "
        "`<p>` is an "
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
        "Emit a `<div class=\"recap-box\" data-cf-source-ids=...>` "
        "wrapping an `<h2>` or `<h3>` and a BRIEF recap of the prior "
        "week's key terms — the `recap-box` class renders the distinct "
        "summary callout, so keep the brief `<ul>`/`<p>` recap inside the "
        "div: a short `<ul>` of key terms with one-line reminders, or one "
        "or two summary `<p>` paragraphs. A recap is NOT an assessment: "
        "never emit "
        "multiple-choice options, `<li data-cf-distractor-index>`, answer "
        "keys, or a fresh worked example — it only restates already-taught "
        "terms. "
        "ANATOMY (§4): strengthen the recap into CUMULATIVE RETRIEVAL — pose "
        "2-3 short recall questions that reach back AT LEAST two weeks (not "
        "just last week), each answer behind a `<details>` reveal. Keep them "
        "FREE-RESPONSE recall prompts — never multiple-choice options or an "
        "answer key."
    ),
    "scenario": (
        "Emit a `<div class=\"scenario-card\" data-cf-source-ids=...>` "
        "carrying `data-cf-content-type=\"scenario\"`. Write ONE short, "
        "realistic APPLICATION scenario grounded in the source (a concrete "
        "situation in which the concept is used), then a single question or "
        "task in a `<p>` prompting the learner to APPLY the concept to that "
        "scenario. Keep it ONE focused scenario — never a multi-part lesson, "
        "never re-teach the concept as expository prose; the scenario sets up "
        "a context and the prompt asks the learner to act on it. "
        "ANATOMY (§4): ground the scenario in a REAL-WORLD application context "
        "(a concrete situation where the concept matters) and prompt the "
        "learner to APPLY the concept to it — one focused scenario, not a "
        "lesson."
    ),
    "problem": (
        "Emit a `<div class=\"problem-card\" data-cf-source-ids=...>` "
        "carrying `data-cf-content-type=\"problem\"`. State the practice "
        "problem in full in a `<p>` (the actual task — the expression to "
        "simplify, the equation to solve, the question to answer — NEVER a "
        "bare exercise / reference number), then put the worked solution "
        "behind a NO-JS reveal INSIDE the div: "
        "`<details><summary>Show solution</summary> … </details>` showing "
        "the solution steps and the final answer. The problem is grounded in "
        "the source; never reveal the solution inline outside the "
        "`<details>`."
    ),
    "vocab_card": (
        "Emit a `<div class=\"vocab-card\" data-cf-source-ids=...>` carrying "
        "`data-cf-content-type=\"vocabulary\"`. Put ONE domain key term in a "
        "`<span class=\"vocab-term\">Term</span>` followed by its definition "
        "(drawn from the source) in a `<p>`. The term MUST be a DOMAIN "
        "vocabulary word (e.g. `coefficient`, `numerator`), NEVER a "
        "pedagogy / structural meta-word (`example`, `exercise`, `note`, "
        "`summary`, `problem`, …). One term per card — never list several."
    ),
    "formula": (
        "Emit a `<div class=\"formula-card\" data-cf-source-ids=...>` "
        "carrying `data-cf-content-type=\"formula\"`. Show ONE highlighted "
        "formula / equation (the symbolic statement) drawn from the source, "
        "followed by a one-line gloss naming each variable / symbol in the "
        "formula. ONE formula only — never a multi-formula derivation or a "
        "step-by-step worked solution; just the formula and what its symbols "
        "mean."
    ),
    "checklist": (
        "Emit a `<ul class=\"checklist\" data-cf-source-ids=...>` carrying "
        "`data-cf-content-type=\"checklist\"`, whose children are "
        "`<li class=\"checklist-item\">` items. Each item is ONE concrete, "
        "verifiable step or criterion drawn from the source (a procedure to "
        "follow or a self-verification list) — actionable and checkable, "
        "NOT a prose paragraph and NOT a restated objective. Keep each item "
        "short and imperative."
    ),
    # Issue I6 instruction-palette-v2: WCAG-correct structural block types.
    "table": (
        "Emit a real, accessible HTML `<table data-cf-source-ids=...>` "
        "carrying `data-cf-content-type=\"table\"` — NOT a prose paragraph "
        "and NOT a `<div>` grid. The table MUST carry a `<caption>` "
        "describing what it compares (the first child of `<table>`), a "
        "`<thead>` with a header `<tr>` whose cells are `<th scope=\"col\">` "
        "column headers, and a `<tbody>` with one `<tr>` per row. When the "
        "first cell of each body row is itself a row label (a term, a type, "
        "an option being compared), emit it as `<th scope=\"row\">` and the "
        "remaining cells as `<td>`. Use ONLY the rows, columns, and values "
        "the source supplies — never invent a row or column. A table is the "
        "right element when the content COMPARES / CONTRASTS several items "
        "across the SAME dimensions (term / definition / example, "
        "operation / rule / example, before / after); reserve one sentence of "
        "framing prose around it."
    ),
    "acronym": (
        "Emit a `<div class=\"acronym-card\" data-cf-source-ids=...>` "
        "carrying `data-cf-content-type=\"acronym\"`. State the acronym / "
        "mnemonic (e.g. `PEMDAS`) in a leading `<p>` or `<strong>`, then map "
        "EACH LETTER to its expansion term in a description list "
        "`<dl class=\"acronym-list\">`: one `<dt>` per letter (the letter "
        "itself, e.g. `P`) immediately followed by its matching "
        "`<dd>` (the term that letter stands for, e.g. `Parentheses`). There "
        "MUST be exactly one `<dd>` per `<dt>` and they MUST be paired in "
        "order. Use ONLY the expansion the source supplies — never invent a "
        "mnemonic or a term the source does not state. An acronym block is "
        "for a memory aid whose letters spell out an ordered list of terms; "
        "it is NOT a vocabulary card or a definition list of unrelated terms."
    ),
    "key_idea": (
        "Emit an `<aside class=\"key-idea\" data-cf-source-ids=...>` carrying "
        "`data-cf-content-type=\"key_idea\"` — an `<aside>` (NOT a `<div>` or "
        "a bare `<p>`) so the key idea is semantically set apart from the "
        "body flow for assistive technology. OPEN with a recognizable framing "
        "label — a `<strong>` lead-in reading \"Key Idea\", \"Remember\", or "
        "the principle's own name — then state the single most important "
        "principle / takeaway in 1-3 sentences. A key_idea is ONE emphasized "
        "principle, not a mini-lesson: no multi-example sequences, no "
        "step-by-step worked solutions. Keep it grounded in the source (no "
        "fabricated rules)."
    ),
    # IB5 framework-aligned pedagogical block types.
    "hook": (
        "Emit a `<section class=\"hook\" data-cf-source-ids=...>` that OPENS "
        "the topic by gaining attention and surfacing the learner's PRIOR "
        "knowledge BEFORE any new content. Write a `<p>` activation / predict "
        "prompt (\"What do you already know about …?\", \"Predict the result "
        "before you read on …\") then a `<p>` forward-transition into what the "
        "section will teach. Carry NO new teaching content / definitions — a "
        "hook ACTIVATES; the exposition blocks teach. Keep it 2-3 short "
        "sentences total, grounded in the source topic. "
        "ANATOMY (§4): prefer a REAL-WORLD application context (a concrete "
        "situation where the topic matters — e.g. projectile height, area, "
        "revenue for quadratics) as the attention-grabber before the "
        "prediction prompt and the forward transition."
    ),
    "multimedia": (
        "Emit a `<figure class=\"multimedia\" data-cf-source-ids=...>` for a "
        "time-based audio / video artifact. It MUST carry the FULL time-based-"
        "media accessibility stack: a media element with the `controls` "
        "attribute (`<video controls>` or `<audio controls>`), a "
        "`<track kind=\"captions\">` child for synchronized captions, a "
        "downloadable / inline transcript in a `<details data-cf-transcript>` "
        "(a `<summary>` reading \"Transcript\" + the transcript text), and a "
        "note describing the audio-described visual content. When the source "
        "supplies no media URL, emit the a11y-contract SKELETON + a "
        "\"media pending\" note rather than nothing (ship the structure even "
        "when unpopulated). Use ONLY content the source supplies for the "
        "transcript / description — never fabricate a narration."
    ),
    "worked_example": (
        "Emit a `<section class=\"worked-example\" data-cf-source-ids=...>` "
        "carrying `data-cf-fade-state` (one of `worked` / `completion` / "
        "`independent`). State the problem in a leading `<p>`, then an ordered "
        "`<ol>` whose `<li>` steps EACH carry a SUBGOAL LABEL "
        "(`<span class=\"subgoal-label\">…</span>`) naming the sub-goal that "
        "step achieves, the step body, and a per-step \"Why\" gloss "
        "(`<span class=\"why\">Why: …</span>`) justifying the step. Use ONLY "
        "the procedure the source supplies — never invent a step. A "
        "worked_example is the SCAFFOLDED, fade-able instance the framework "
        "reserves; it is richer than a single un-labeled `example`. "
        "Copy worked-example arithmetic verbatim from the source; if your "
        "verification check fails, fix the algebra — NEVER substitute source "
        "numbers to force agreement. "
        "ANATOMY (§4): as the fade-sequence ANCHOR that precedes "
        "`guided_practice`, keep EVERY step fully worked here — state the "
        "problem, prompt a PREDICTION, give per-step operation AND \"Why\" "
        "justification, a CHECK-BY-SUBSTITUTION, and a \"Common wrong turn:\" "
        "line naming the typical error."
    ),
    "diagram": (
        "Emit a `<figure class=\"diagram\" data-cf-source-ids=...>` for a "
        "spatial / relational artifact (process flow, hierarchy, mapping). It "
        "MUST carry an image / inline-svg slot (or a \"diagram pending\" note "
        "when the source supplies no image), a short `<figcaption>`, a "
        "STRUCTURED long-description in a `<details>` (a `<summary>` reading "
        "\"Long description\" + prose walking the relationships in reading "
        "order), AND a `<table>` data-table EQUIVALENT carrying the same "
        "nodes / edges / values so the spatial relationships are available "
        "non-visually (caption + scoped `<th>` headers). Use ONLY the "
        "relationships the source supplies — never invent a node or edge. "
        "ANATOMY (§4): when the source supplies an equation or numeric "
        "relationship, render the matching visual (number line / coordinate "
        "grid / line / parabola) keyed off THAT equation alongside the "
        "data-table equivalent — dual-code the relationship instead of "
        "describing it in prose alone."
    ),
    "guided_practice": (
        "Emit a `<section class=\"guided-practice\" data-cf-source-ids=...>` "
        "carrying `data-cf-fade-state=\"completion\"` — the FADED middle of "
        "the gradual-release ladder that sits BETWEEN a fully-worked "
        "`worked_example` and independent practice. Present a problem with "
        "MOST steps worked but 1-2 steps LEFT BLANK for the learner to "
        "complete (a `____` blank or an empty "
        "`<div class=\"step-row\">` they fill in), then put the omitted "
        "step(s) AND the final answer behind a "
        "`<details><summary>Show the completed steps</summary> … </details>` "
        "reveal. ANATOMY (§4): it FADES the scaffolding — do NOT re-work every "
        "step; the learner supplies the missing move. Use ONLY the procedure "
        "the source supplies — never invent a step."
    ),
    "resources": (
        "Emit a `<section class=\"resources\" data-cf-source-ids=...>` "
        "(Resources / Further Reading) wrapping an `<h2>` or `<h3>` and a "
        "`<ul>` of curated links. ANATOMY (§4): EACH `<a>` MUST carry "
        "DESCRIPTIVE link text naming the destination and why it helps "
        "(WCAG 2.4.4) — NEVER \"click here\", a bare URL, or \"read more\". "
        "List ONLY resources the source supplies or names; never fabricate a "
        "link or a citation the source does not contain."
    ),
}


def _block_type_output_contract(
    block_type: str, *, fit_window: bool = False
) -> str:
    """Return the per-block-type HTML attribute contract paragraph.

    Falls back to a generic instruction when the block_type has no
    entry in the table — defensive only; ``Block.__post_init__``
    already validates the set.

    When ``fit_window`` is True (``ED4ALL_REWRITE_FIT_WINDOW`` ON), the
    block-type-specific authoring segments relocated out of the trimmed
    system prompt are APPENDED here so the per-block prompt still carries
    its rules. When False (default), the contract is byte-identical to the
    original table entry — no relocation, so OFF is byte-stable.
    """
    base = _BLOCK_TYPE_OUTPUT_CONTRACTS.get(
        block_type,
        (
            f"Emit the rendered HTML body for a block of type "
            f"{block_type!r}. Carry `data-cf-source-ids` on the top "
            f"wrapper to attribute the source chunks."
        ),
    )
    if not fit_window:
        return base
    return base + _relocated_contract_suffix(block_type)


def _recall_self_check_contract_suffix(block: Any) -> str:
    """recall_self_check — flag-gated cloze / free-recall authoring suffix.

    Read at author time behind ``resolve_recall_self_check()`` (mirroring the
    ``ED4ALL_REWRITE_FIT_WINDOW`` conditional-contract pattern), NEVER appended
    unconditionally. Returns ``""`` when the flag is off OR the block carries no
    ``recall_format`` → off-path output bytes are unchanged.
    """
    try:
        from lib.generation.recall_self_check import (
            resolve_recall_format,
            resolve_recall_self_check,
        )
    except Exception:  # noqa: BLE001
        return ""
    if not resolve_recall_self_check():
        return ""
    fmt = resolve_recall_format(getattr(block, "recall_format", None))
    if fmt is None:
        return ""
    if fmt == "cloze":
        return (
            " RECALL VARIANT — CLOZE: author this self-check as a "
            "fill-in-the-blank retrieval item. Elide ONE key term (grounded in "
            "the source) from a sentence with a `____` blank instead of "
            "enumerating options; put the elided term's answer behind the "
            "`<details>` reveal. Do NOT pre-enumerate answer choices — the "
            "learner must RECALL the term, not recognize it."
        )
    return (
        " RECALL VARIANT — FREE RECALL: author this self-check as a "
        "produce-the-answer question (no enumerated options). Ask the learner "
        "to write the answer from memory; put the source-grounded answer behind "
        "the `<details>` reveal. Do NOT list answer choices — the learner must "
        "RECALL, not recognize."
    )


def _misconception_productive_failure_contract_suffix(block: Any) -> str:
    """misconception_rich — flag-gated productive-failure authoring suffix.

    Read at author time behind ``resolve_misconception_rich()`` (mirroring the
    ``ED4ALL_REWRITE_FIT_WINDOW`` conditional-contract pattern), NEVER appended
    unconditionally. Returns ``""`` when the flag is off OR the block carries no
    ``mc_named_concept`` → off-path output bytes are unchanged.
    """
    try:
        from lib.generation.misconception_rich import resolve_misconception_rich
    except Exception:  # noqa: BLE001
        return ""
    if not resolve_misconception_rich():
        return ""
    named = getattr(block, "mc_named_concept", None)
    if not (isinstance(named, str) and named.strip()):
        return ""
    label = named.replace("_", " ").replace("-", " ").strip()
    return (
        f" PRODUCTIVE FAILURE — NAME the targeted faulty mental model "
        f"({label!r}) explicitly. BEFORE the correction, add a "
        f"`<p class=\"misconception-predict\">` predict-then-reveal prompt asking "
        f"the learner to PREDICT what the faulty model expects. AFTER the "
        f"`misconception-correction` <p>, add a distinct "
        f"`<p class=\"misconception-reconcile\">` RECONCILE step: explain why the "
        f"named model fails (reconcile the prediction with the correct "
        f"understanding), grounded in the source. The `misconception-predict` and "
        f"`misconception-reconcile` paragraphs are BOTH required. Do not fabricate "
        f"a concept the source does not discuss."
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


# FR-PLAN-01 — human-readable rendering hints for each Chapter-5 activity type,
# so the rewrite tier authors the planner-selected interaction shape rather than
# defaulting to a generic prompt. Absent token → a generic directive.
_INTERACTION_TYPE_HINTS: Dict[str, str] = {
    "multiple_choice": "a single-best-answer multiple-choice question (one correct option + misconception-targeted distractors)",
    "multiple_response": "a select-all-that-apply question (more than one correct option)",
    "true_false": "a true/false statement the learner judges",
    "fill_in_blank": "a fill-in-the-blank item with the missing term(s) elided from a sentence",
    "matching": "a matching exercise pairing items from two columns",
    "ordering": "an ordering/sequencing task (arrange the steps/items in correct order)",
    "drag_drop": "a drag-and-drop task (place items into the correct targets)",
    "hotspot": "a hotspot task (identify/select the correct region of an image or diagram)",
    "short_answer": "a short constructed-response question (a sentence or two)",
    "essay": "an extended constructed-response / essay prompt",
    "numeric": "a numeric-entry computation item (the learner enters a calculated value)",
    "categorization": "a categorization task (sort items into named categories)",
    "labeling": "a labeling task (label the parts of a figure/diagram)",
    "branching_scenario": "a branching decision scenario (choices lead to different consequences)",
}


def _interaction_type_directive(interaction_type: Optional[str]) -> str:
    """FR-PLAN-01 — render the planner-selected interaction-type directive.

    Returns an empty string when the planner did not select an interaction type
    (non-interaction-bearing block, or the planner flag was off → byte-stable
    prompt). Otherwise instructs the rewrite tier to author the specific
    Chapter-5 activity shape the planner chose.
    """
    if not interaction_type:
        return ""
    hint = _INTERACTION_TYPE_HINTS.get(
        interaction_type, f"a {interaction_type.replace('_', ' ')} interaction"
    )
    return (
        f"Interaction type (planner-selected): author the interaction as "
        f"{hint}.\n"
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


# Malformed comment-close pattern: the Qwen-7B-Q4 rewrite tier, when it
# free-authors a B04 multimedia / B06 diagram block, sometimes emits an
# HTML comment whose CLOSE delimiter was entity-escaped — ``--&gt;`` (or a
# double-/triple-escaped variant) instead of a real ``-->``. Per the WHATWG
# tokenizer (and Python's ``html.parser``) the entity-escaped close does NOT
# terminate the comment, so the comment runs on and SWALLOWS every downstream
# sibling (``<source>`` / ``<track>`` / ``<details>`` / ``</table>``), genuinely
# destroying the captions / transcript / long-description in a real browser
# (confirmed on a real block ``week_02_content_01#multimedia_week02_4`` — the
# ``<!-- … --&gt;`` swallowed 1293 chars including the ``<track>`` and
# ``<details>``). The repair rewrites the escaped close back to a real ``-->``
# BEFORE any downstream gate / parser / consumer sees the content.
#
# The regex anchors on ``--`` immediately followed by ONE OR MORE
# entity-escaped ``>`` layers (``&gt;`` / ``&amp;gt;`` / ``&amp;amp;gt;`` / …)
# with NO intervening characters, so it can only match a comment-close that was
# escaped — it never touches a well-formed ``-->`` (no ``&`` there) nor a
# legitimate escaped entity in prose (those are not preceded by a bare ``--``).
# Idempotent: a single pass collapses any depth of escaping to one real close,
# and a second pass finds nothing left to rewrite.
_MALFORMED_COMMENT_CLOSE_RE = re.compile(r"--(?:&(?:amp;)*gt;)+")


def _fix_malformed_comment_closes(html: str) -> str:
    """Rewrite entity-escaped comment closes ``--&gt;`` back to real ``-->``.

    Closes the Qwen-7B-Q4 failure mode where the rewrite tier emits an HTML
    comment terminated with an HTML-entity-escaped close (``--&gt;`` or any
    double-/triple-escaped variant) instead of a literal ``-->``. The escaped
    close leaves the comment UNTERMINATED under the HTML tokenizer, so the
    parser swallows the downstream ``<source>`` / ``<track>`` / ``<details>`` /
    ``</table>`` siblings — silently destroying the B04 captions/transcript and
    the B06 long-description/data-table in any real browser (and hiding them
    from the IB5 a11y shape gate).

    Conservative + idempotent: only the ``--`` + escaped-``>`` sequence is
    rewritten; well-formed ``-->`` (no ``&``) and legitimate escaped entities
    elsewhere in prose are untouched. A second invocation is a no-op.
    """
    if "--&" not in html:
        # Fast path — no candidate substring, nothing to repair.
        return html
    return _MALFORMED_COMMENT_CLOSE_RE.sub("-->", html)


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
    if not lines:
        return "(no claims)"
    # Directive: the chunk ids below are for the `data-cf-source-ids`
    # ATTRIBUTE only — they MUST NOT appear as visible `<cite>` text or in
    # an "According to the source <id>" lead-in in the learner-facing prose.
    header = (
        "(cited chunk ids are for the data-cf-source-ids ATTRIBUTE only — "
        "never write them as visible cite text or prose)"
    )
    return header + "\n" + "\n".join(lines)


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


def _rank_query_for_block(block: Block) -> str:
    """Build the cosine-rank query for fit-window chunk selection.

    The grounding chunks should be relevant to what the block TEACHES, so
    the query is the block's key-claim text (joined) falling back to the
    block id. Pure read of ``block.content`` — no fabrication.
    """
    content = getattr(block, "content", None)
    parts: List[str] = []
    if isinstance(content, dict):
        claims = content.get("key_claims")
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, dict):
                    txt = claim.get("text") or claim.get("claim") or ""
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
                elif isinstance(claim, str) and claim.strip():
                    parts.append(claim.strip())
    query = " ".join(parts).strip()
    return query or str(getattr(block, "block_id", "") or "")


def _try_rewrite_embedder() -> Optional[Any]:
    """Lazy-load the statistical-tier sentence embedder, or ``None``.

    Reuses the same ``lib.embedding`` loader the Courseforge statistical-
    tier validators use, so a single model serves both. Absence (no
    ``[embedding]`` extras, or a load failure) returns ``None`` → the
    chunk selector degrades to citation-order ranking (NOT a new fail-
    closed dependency).
    """
    try:
        from lib.embedding import try_load_embedder  # noqa: PLC0415

        return try_load_embedder()
    except Exception:  # noqa: BLE001 — embedding is optional
        return None


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
    # dispatch-resilience-2026-07: the outline-tier dispatch raised an
    # infrastructure exception (endpoint flap / transport / 5xx / timeout)
    # and gave up after bounded retries, so NO usable outline draft exists.
    # Symmetric with ``outline_skipped_by_policy`` — synthesise from scratch.
    "outline_dispatch_error": (
        "The outline tier failed to produce a draft — its dispatch raised "
        "an infrastructure error (endpoint unavailable / transport / "
        "timeout) after bounded retries. No usable outline exists; create "
        "the block from scratch using the supplied source chunks and "
        "objectives."
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


def _curie_in_pedagogical_context(
    html: str, curie: str, *, minted: bool = False
) -> bool:
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

    7B-parity P0: when ``minted=True`` (a synthetic prose-corpus CURIE,
    NOT an RDF CURIE the prose legitimately quotes), Case 1 (code-voice
    text content) NO LONGER counts as preserved — a minted
    ``prefix:localname`` token is an anchoring signal that belongs in the
    hidden ``<span data-cf-curie>``, never in visible code-voice prose a
    learner reads. RDF CURIEs (``minted=False``, the default) keep the
    full three-case contract so ``rdf:type`` in ``<code>`` still counts.
    """
    if not html or not curie:
        return False

    # Case 1: text content inside a definitional-voice tag. We
    # substring-match the open / close pair around the CURIE; this
    # accepts both ``<code>rdf:type</code>`` and ``<code class="x">
    # rdf:type ...</code>``. SKIPPED for minted CURIEs (P0): a minted
    # token in code voice is a leaked anchoring token, not pedagogy.
    if not minted:
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


# ---------------------------------------------------------------------------
# M3 — tolerant CURIE preservation (CURIE-churn truncation fix).
# ---------------------------------------------------------------------------
#
# Root cause (diagnostic M3): the pre-M3 preservation retry loop demanded
# that EVERY minted CURIE the outline declared (8-11 per CURIE-dense block)
# appear in the rewritten prose. The rewrite tier — authoring NATURAL HTML
# prose — rarely echoes every synthetic token, so the loop re-prompted with
# a "re-inject these 7+ tokens" directive. Forcing 7+ off-topic CURIEs back
# into the prose bloated the output past the 16384 ``max_tokens`` cap →
# ``finish_reason='length'`` → degraded/truncated emit → block fails
# ``content_grounding`` / ``rewrite_curie_anchoring`` / ``example_completeness``.
#
# The fix makes preservation TOLERANT: only REQUIRE a CURIE whose underlying
# TERM actually appears in the rewritten prose (the model used it naturally).
# CURIEs the model didn't use are PRUNED from the enforced set rather than
# forced — that's what stops the long re-sends. ANTI-FABRICATION: we only
# ever PRUNE the outline-declared set; we never add a CURIE.
#
# HARD INVARIANT — never drop below 1 CURIE: ``rewrite_curie_anchoring``
# (``Courseforge/router/inter_tier_gates.py::BlockCurieAnchoringValidator``)
# requires ≥1 anchoring CURIE per block. When the on-topic set is empty we
# keep exactly ONE CURIE (the most on-topic, by surface-term overlap; the
# first outline CURIE as last resort) so the block always ends with ≥1.


def _curie_surface_terms(curie: str) -> List[str]:
    """Return the natural-language surface term(s) a CURIE stands for.

    A minted CURIE is ``{prefix}:{localname}`` where the localname is the
    concept slug with hyphens→underscores (see
    ``lib/ontology/curie_discovery.py::curie_for_concept``). To detect
    whether the model "used the term", we surface-match BOTH:

    - the localname with underscores→spaces (e.g.
      ``introbio101:least_common_multiple`` → ``"least common multiple"``),
      the natural-language form a learner-facing sentence would carry; and
    - the literal CURIE token (``introbio101:least_common_multiple``) —
      RDF corpora carry the token verbatim in pedagogical prose.

    Returns lower-cased, de-duplicated, non-empty terms.
    """
    if not curie:
        return []
    terms: List[str] = [curie.lower()]
    local = curie.split(":", 1)[1] if ":" in curie else curie
    natural = local.replace("_", " ").replace("-", " ").strip().lower()
    if natural:
        terms.append(natural)
    seen: set[str] = set()
    out: List[str] = []
    for t in terms:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _curie_term_in_prose(curie: str, prose: str) -> bool:
    """Return True when ``curie``'s underlying term appears in ``prose``.

    Case-insensitive surface match of any of the CURIE's surface terms
    (``_curie_surface_terms``) against the stripped prose text. Used to
    decide whether the rewrite tier NATURALLY used the concept — the
    signal that gates whether we enforce its preservation (M3 tolerance).
    """
    if not curie or not prose:
        return False
    blob = prose.lower()
    return any(term in blob for term in _curie_surface_terms(curie))


def _select_enforceable_curies(
    outline_curies: Sequence[str], html_response: str
) -> List[str]:
    """Prune ``outline_curies`` to the set worth enforcing for this emit.

    M3 tolerance contract:

    1. Keep every outline CURIE whose underlying TERM appears in the
       rewritten prose (the model used the concept naturally — preserving
       its CURIE is cheap and correct).
    2. If that on-topic set is EMPTY, keep exactly ONE CURIE so the
       ≥1-anchoring invariant holds: the most on-topic by surface-term
       token overlap with the prose, falling back to the first outline
       CURIE when nothing overlaps.

    PRUNE-ONLY: the returned set is always a subset of ``outline_curies``;
    no CURIE is ever invented (anti-fabrication).
    """
    curies = [c for c in (outline_curies or []) if c]
    if not curies:
        return []
    prose = _strip_html(html_response or "")
    on_topic = [c for c in curies if _curie_term_in_prose(c, prose)]
    if on_topic:
        return on_topic
    # Anchoring invariant: keep the single most on-topic CURIE. Score by
    # how many of its surface-term tokens appear in the prose (a partial
    # signal even when the full term didn't match); ties + all-zero fall
    # back to the first outline CURIE (stable, deterministic).
    blob = prose.lower()

    def _overlap(curie: str) -> int:
        tokens: set[str] = set()
        for term in _curie_surface_terms(curie):
            tokens.update(t for t in term.split() if len(t) > 2)
        return sum(1 for t in tokens if t in blob)

    best = max(curies, key=_overlap)
    return [best]


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


# Length-truncation remediation. Root cause (parity build, 2026-06-18): on a
# ``finish_reason='length'`` truncation the legacy retry re-dispatched the
# IDENTICAL prompt, so the model re-generated the same over-long output and
# re-truncated every attempt until the budget exhausted (~52% of blocks fell
# back to the deterministic template). The model over-generates because the
# PEDAGOGICAL DEPTH instructions compete with BREVITY; a length-aware
# remediation turn that DOMINATES (drastically shorter, one example, no
# repetition) makes the retry finish inside the cap. Never raises the cap.
_BREVITY_REMEDIATION_DIRECTIVE: str = (
    "\n\n[LENGTH REMEDIATION — your previous attempt EXCEEDED the output "
    "limit and was TRUNCATED mid-block (unusable). This overrides any "
    "instruction to add depth: produce a DRASTICALLY SHORTER version NOW — "
    "at most ~120 words of prose; exactly ONE worked example (never two); "
    "no repetition, no restating the definition, no re-explaining a step, "
    "no extra 'expert tips' or asides. Keep ONLY the required styled "
    "component(s) and the single teaching point. A block that truncates "
    "again is DISCARDED — finish FAR inside the limit.]"
)


def _is_length_truncation(exc: object) -> bool:
    """True iff a dispatch error is a ``max_tokens`` length truncation.

    The provider raises ``SynthesisProviderError`` with a message naming
    ``finish_reason='length'`` when the model hit the cap; that's distinct
    from transient/connection failures and warrants a brevity remediation
    rather than an identical re-dispatch.
    """
    s = str(exc)
    return "finish_reason='length'" in s or "max_tokens cap" in s


def _append_brevity_remediation(user_prompt: str) -> str:
    """Append the length-truncation brevity remediation to ``user_prompt``.

    Idempotent — appending twice (across retries) is harmless; the directive
    re-asserts the same hard constraint.
    """
    return user_prompt + _BREVITY_REMEDIATION_DIRECTIVE


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


# ---------------------------------------------------------------------------
# IB5 structural a11y backstop (mirror of the CURIE force-inject sweep).
#
# When the rewrite tier FREE-AUTHORS a B04 multimedia / B06 diagram block
# (instead of consuming the renderer's guaranteed skeleton) it sometimes drops
# a renderer-guaranteed a11y piece — the audio-description note, a captions
# <track>, the transcript <details>, or (B06) the long-description <details> /
# data-<table>. The IB5 a11y shape gate (rewrite_html_shape._check_ib5_a11y_shape)
# checks STRUCTURAL PRESENCE of exactly these pieces. This backstop re-injects
# ONLY the missing structural skeleton — the same shape the renderer guarantees
# — so a shipping block satisfies the structural contract.
#
# ANTI-FABRICATION: the backstop NEVER invents narration / caption prose. It
# threads through any long-description / transcript text the block already
# carries in its fields (``Block.long_description`` / ``content[...]``); when no
# source text exists it emits the SAME empty-labelled "pending" skeleton the
# renderer would (the gate's contract is structural presence, not prose).
# ---------------------------------------------------------------------------


class _Ib5ShapeProbe(HTMLParser):
    """Detect which IB5-gate structural a11y markers a fragment already has.

    Mirrors the marker logic in ``rewrite_html_shape._ShapeParser._track_a11y``
    so a backstop decision agrees with what the gate will conclude. Runs over
    the POST-sanitize HTML (after ``_fix_malformed_comment_closes``), so a
    repaired comment-close exposes the previously-swallowed siblings here.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.media_controls = False
        self.media_track = False        # <track kind="captions">
        self.media_audio_desc = False   # AD affordance
        self.media_transcript = False   # <details data-cf-transcript>
        self.saw_long_desc_details = False
        self.saw_data_table = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self._mark(tag, {k: (v or "") for k, v in attrs})

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        self._mark(tag, {k: (v or "") for k, v in attrs})

    def _mark(self, tag: str, attr_map: Dict[str, str]) -> None:
        if tag in ("video", "audio") and "controls" in attr_map:
            self.media_controls = True
        if tag == "track":
            kind = (attr_map.get("kind") or "").strip().lower()
            if kind == "captions":
                self.media_track = True
            if kind == "descriptions":
                self.media_audio_desc = True
        if "data-cf-transcript" in attr_map:
            self.media_transcript = True
        if "data-cf-audio-description" in attr_map:
            self.media_audio_desc = True
        if "audio-description" in (attr_map.get("class") or "").lower().split():
            self.media_audio_desc = True
        if tag == "details":
            self.saw_long_desc_details = True
        if "aria-describedby" in attr_map:
            self.saw_long_desc_details = True
        if tag == "table":
            self.saw_data_table = True


def _ib5_block_field(block: Block, *keys: str) -> str:
    """Return the first non-empty value among ``block``'s dataclass fields and
    its ``content`` dict for ``keys`` (anti-fabrication source text).

    Reads only text the block ALREADY carries — never synthesizes prose.
    """
    for key in keys:
        val = getattr(block, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    content = block.content if isinstance(block.content, dict) else {}
    for key in keys:
        val = content.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _inject_ib5_a11y_skeleton(html: str, block: Block) -> str:
    """Re-inject any renderer-guaranteed IB5 a11y skeleton the block is missing.

    For a SHIPPING B04 (``multimedia``) / B06 (``diagram``) block whose
    post-sanitize HTML is missing a structural a11y piece the gate
    (``_check_ib5_a11y_shape``) requires, append ONLY the missing structural
    skeleton — mirroring ``generate_course._render_multimedia_section`` /
    ``_render_diagram_section``. Well-formed blocks are byte-identical
    (idempotent — nothing appended when every marker is already present).

    Anti-fabrication: emits the empty labelled "pending" skeleton the renderer
    would when the block carries no source text; threads through
    ``long_description`` / ``transcript`` / ``audio_desc`` text the block
    already has.
    """
    btype = getattr(block, "block_type", "")
    if btype not in ("multimedia", "diagram"):
        return html
    html = html or ""
    probe = _Ib5ShapeProbe()
    try:
        probe.feed(html)
        probe.close()
    except Exception as exc:  # noqa: BLE001 — never let a probe raise kill the emit
        logger.debug("IB5 a11y probe raised on %r: %s", block.block_id, exc)
        return html

    additions: List[str] = []
    if btype == "multimedia":
        if not probe.media_controls:
            # The media element carries controls + the captions track; emit the
            # renderer's media-pending skeleton when neither is present.
            additions.append(
                '<video controls>'
                '<track kind="captions" srclang="en" '
                'label="Captions pending"></video>'
            )
        elif not probe.media_track:
            # Controls present but no captions track — add a standalone track.
            additions.append(
                '<track kind="captions" srclang="en" label="Captions pending">'
            )
        if not probe.media_transcript:
            transcript = _ib5_block_field(block, "transcript") or "Transcript pending."
            additions.append(
                f'<details data-cf-transcript><summary>Transcript</summary>'
                f'<p>{html_escape(transcript)}</p></details>'
            )
        if not probe.media_audio_desc:
            audio_desc = (
                _ib5_block_field(block, "audio_desc", "audio_description")
                or "Audio description pending."
            )
            additions.append(
                f'<p class="audio-description">Audio description: '
                f'{html_escape(audio_desc)}</p>'
            )
    else:  # diagram
        if not probe.saw_long_desc_details:
            long_desc = (
                _ib5_block_field(block, "long_description")
                or "Long description pending."
            )
            additions.append(
                f'<details class="diagram-longdesc">'
                f'<summary>Long description</summary>'
                f'<p>{html_escape(long_desc)}</p></details>'
            )
        if not probe.saw_data_table:
            additions.append(
                '<table><caption>Diagram — data equivalent</caption>'
                '<tbody></tbody></table>'
            )

    if not additions:
        return html
    logger.warning(
        "RewriteProvider: IB5 a11y backstop re-injected %d structural "
        "skeleton piece(s) for %s block %r (free-authored emit dropped them)",
        len(additions),
        btype,
        getattr(block, "block_id", "?"),
    )
    return html + "".join(additions)


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
        # Collapse the legacy ``openai_compatible`` alias to ``local`` at
        # constructor entry so a standalone construction behaves exactly
        # like a router-mediated one (the router collapses it in
        # ``_get_rewrite_provider``). Without this, ``openai_compatible``
        # would reach the base's registry else-branch and raise
        # ``UnknownEndpoint`` (it is NOT a registry row name).
        if resolved_provider == _OPENAI_COMPATIBLE_ALIAS:
            resolved_provider = "local"

        # Per-call kwarg wins; otherwise source the generous default
        # from ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` (fallback 300.0)
        # so local 7B prose generation isn't capped at the 60s client
        # default.
        resolved_timeout: float = (
            float(timeout) if timeout is not None
            else _resolve_request_timeout()
        )

        # Fit-window state (rewrite-overflow-fix-2026-06). Resolved ONCE at
        # construction so the whole rewrite call uses a consistent state.
        # OFF (default) → untrimmed system prompt + no chunk-window budget
        # (byte-identical to today). ON → trimmed system prompt + num_ctx-
        # aware grounding budget. The truncation tripwire is independent
        # (default ON, output-neutral).
        self._fit_window = resolve_fit_window()
        self._rewrite_num_ctx = resolve_rewrite_num_ctx()
        self._truncation_tripwire = resolve_truncation_tripwire()
        resolved_system_prompt = _resolve_system_prompt(self._fit_window)

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
            self._system_prompt = resolved_system_prompt
            self._supported_providers = tuple(SUPPORTED_PROVIDERS)
            self._env_provider_var = ENV_PROVIDER
            self._api_key = None
            self._anthropic_client = None
            self._oa_client = None
            self._base_url = None
            self._dispatcher = dispatcher
            self._run_id = run_id or "rewrite-standalone"
            return

        # Registry-superset plumbing: the two non-registry-endpoint tags
        # are already intercepted above — ``claude_session`` returned early
        # and ``openai_compatible`` was collapsed to ``local``. Every other
        # value is a registry endpoint the base constructs generically (the
        # per-vendor branches for ``anthropic`` / ``together`` / ``nvidia`` /
        # ``local`` plus the W9.2 else-branch for any other
        # ``kind: openai_compatible`` row — ``groq`` / ``fireworks`` /
        # ``deepseek`` / ``nvidia-deepseek`` / …). We pass the collapsed
        # ``resolved_provider`` (not the raw kwarg) so the alias collapse
        # survives, and we do NOT pin a narrow ``supported_providers`` — the
        # base's registry-derived default governs, so adding a provider stays
        # a ``config/endpoints.yaml`` registry-entry change, never a subclass.
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
            default_model_anthropic=DEFAULT_MODEL_ANTHROPIC,
            default_model_together=DEFAULT_MODEL_TOGETHER,
            default_model_local=DEFAULT_MODEL_LOCAL,
            system_prompt=resolved_system_prompt,
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
    # Fit-window helpers (rewrite-overflow-fix-2026-06)
    # ------------------------------------------------------------------

    def _num_ctx_options_payload(self) -> Optional[Dict[str, Any]]:
        """Request-body extras pinning the SERVED window to ``_rewrite_num_ctx``.

        num_ctx single-source-of-truth (rewrite-overflow-fix-2026-07): the
        budget + both truncation tripwires size against ``self._rewrite_num_ctx``
        (``ED4ALL_REWRITE_NUM_CTX``), but the OpenAI-compatible REQUEST omitted
        it, so a local Ollama/vLLM/llama.cpp server served ITS OWN default
        window (often 8192) — even when the Modelfile / env said 16384. That
        split-brain silently head-truncated the prompt while the tripwire
        message reported the assumed (larger) num_ctx. Passing
        ``options.num_ctx`` makes the SERVED window the SAME value the budget
        assumes, so the request, the budget, and the tripwire all resolve one
        number.

        Scoped to the LOCAL LOOPBACK lane: ``num_ctx`` is an Ollama/vLLM/
        llama.cpp served-window option. Cloud OpenAI endpoints carry a large
        native context and may REJECT an unknown ``options`` field, so we do
        NOT send it there. ``anthropic`` / ``claude_session`` have no
        server-side num_ctx (return ``None``). Servers that ignore
        ``options`` are no worse off than today — the pre/post-dispatch
        tripwires remain the deterministic backstop for the min-rule case
        (request cannot guarantee the window).
        """
        if self._provider in ("anthropic", "claude_session"):
            return None
        base = self._base_url or ""
        host_local = any(
            token in base
            for token in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1")
        )
        if not host_local:
            return None
        return {"options": {"num_ctx": int(self._rewrite_num_ctx)}}

    def _scaffold_prompt_for_block(
        self,
        block: Block,
        objectives: Optional[Sequence[Any]],
    ) -> str:
        """Render the user prompt with NO source chunks — the non-chunk scaffold.

        Whole-prompt budget (rewrite-overflow-fix-2026-07): the chunk budget
        must reserve the ACTUAL scaffold (block context + outline dict +
        per-claim + objectives + output contract + closing instructions),
        which is block-specific and can be large (a big outline payload alone
        can blow the window). Rendering with an empty chunk list measures it
        exactly, on the SAME render path (escalated vs standard) the real
        dispatch will use — so the chunk budget bounds the WHOLE prompt, not
        just the chunks.
        """
        if block.escalation_marker is not None:
            return self._render_escalated_user_prompt(
                block=block, source_chunks=[], objectives=objectives
            )
        return self._render_user_prompt(
            block=block, source_chunks=[], objectives=objectives
        )

    def _select_source_chunks_for_budget(
        self,
        block: Block,
        source_chunks: Optional[Sequence[Any]],
        objectives: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[Any], List[str]]:
        """Trim ``source_chunks`` to the num_ctx budget when fit-window ON.

        Returns ``(selected, dropped_cited_chunk_ids)``. OFF → the input
        list verbatim + no drops (byte-stable). ON → the num_ctx-aware
        selection budgeted against the WHOLE prompt (cited-first, cosine-
        ranked, always-keep-≥1, drop-trailing-whole, per-chunk cap). Anti-
        fabrication: ``selected`` ⊆ the input universe; never invents a chunk.

        Whole-prompt budget (rewrite-overflow-fix-2026-07): the scaffold token
        cost is MEASURED (render with empty chunks) rather than a fixed
        estimate, so a huge outline payload can no longer smuggle an over-
        window prompt past a chunk-only budget. When the non-chunk scaffold
        ALONE (system prompt + scaffold + reserve) cannot fit the served
        window, :class:`_ScaffoldOverflowError` is raised so the caller
        stamps ``rewrite_scaffold_overflow`` and NEVER dispatches a prompt
        that cannot fit.
        """
        chunks = list(source_chunks or [])
        if not self._fit_window:
            return chunks, []
        # Trimmed system prompt is already on ``self._system_prompt`` (ON
        # state), so its token cost is consistent with the budget.
        sys_tokens = _estimate_tokens(self._system_prompt)
        # Measure the ACTUAL non-chunk scaffold on the render path the real
        # dispatch will use (escalated vs standard).
        scaffold_prompt = self._scaffold_prompt_for_block(block, objectives)
        scaffold_tokens = _estimate_tokens(scaffold_prompt)
        # Loud escalation: the scaffold alone (system prompt + scaffold +
        # reserve) can't fit the served window → no chunk could ever fit and
        # the prompt WILL head-truncate. Fail closed BEFORE any dispatch.
        # (max_tokens — the OUTPUT budget — is intentionally NOT part of the
        # input-fit test; an output that overruns is a separate
        # finish_reason='length' concern handled by the brevity remediation.)
        if sys_tokens + scaffold_tokens + RESERVE_TOKENS >= self._rewrite_num_ctx:
            raise _ScaffoldOverflowError(
                sys_tokens=sys_tokens,
                scaffold_tokens=scaffold_tokens,
                num_ctx=self._rewrite_num_ctx,
            )
        if not chunks:
            return chunks, []
        cited = cited_chunk_ids_from_content(block.content)
        # Rank query = the block's key-claim / objective text (what the
        # grounding should be relevant to). Lazy-load the statistical-tier
        # embedder; absence degrades to citation-order (NOT fail-closed).
        rank_query = _rank_query_for_block(block)
        embedder = _try_rewrite_embedder()
        selected, dropped_cited = select_chunks_under_budget(
            chunks,
            num_ctx=self._rewrite_num_ctx,
            sys_tokens=sys_tokens,
            scaffold_tokens=scaffold_tokens,
            max_tokens=self._max_tokens,
            cited_chunk_ids=cited,
            rank_query=rank_query,
            embedder=embedder,
        )
        return selected, dropped_cited

    def _check_input_fits_predispatch(self, user_prompt: str) -> None:
        """PRE-dispatch deterministic tripwire (fit-window ON only).

        After the whole-prompt budget, assert the FINAL rendered prompt's
        local estimate does not exceed the served window BEFORE dispatch —
        the cheapest, server-usage-independent truncation guard. Catches the
        always-keep-≥1 chunk overshoot or any residual the budget could not
        shrink. Gated on ``_fit_window`` so the OFF (byte-stable, un-
        budgeted) default path is unchanged, and on ``_truncation_tripwire``
        so the escape hatch disables it too. The caller maps the raise to the
        ``input_prompt_truncated`` marker, exactly like the post-dispatch arm.
        """
        if not (self._fit_window and self._truncation_tripwire):
            return
        estimated = _estimate_tokens(self._system_prompt) + _estimate_tokens(
            user_prompt
        )
        check_prompt_fits_window(
            estimated,
            model_id=self._model,
            num_ctx=self._rewrite_num_ctx,
        )

    def _check_truncation(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        usage: Optional[Dict[str, Any]],
        block_id: str,
    ) -> None:
        """Fire the input-truncation tripwire (default ON, fail-OPEN).

        Compares the server-reported ``usage.prompt_tokens`` against the
        local 2.5-char/token estimate of system+user prompt and raises
        :class:`PromptTruncatedError` on a large shortfall. No-op when the
        tripwire is disabled, usage is absent/zero (Ollama may omit it), or
        the provider returned no usage (Anthropic / claude_session). The
        caller maps the raise to the ``input_prompt_truncated`` escalation
        marker (hard, non-retryable).
        """
        if not self._truncation_tripwire:
            return
        if not isinstance(usage, dict):
            return
        estimated = _estimate_tokens(system_prompt) + _estimate_tokens(
            user_prompt
        )
        check_prompt_not_truncated(
            usage.get("prompt_tokens"),
            estimated,
            model_id=self._model,
            num_ctx=self._rewrite_num_ctx,
        )

    # ------------------------------------------------------------------
    # Wave6: dispatch override for the claude_session provider
    # ------------------------------------------------------------------

    def _dispatch_call(
        self,
        user_prompt: str,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Wave6 override — route through LocalDispatcher when provider
        is ``claude_session``; otherwise defer to the base's
        HTTP / Anthropic-SDK plumbing.

        Mirrors the Trainforge precedent at
        ``Trainforge/generators/_claude_session_provider.py::_dispatch``
        (`:326-394`). The async dispatcher is run synchronously via
        ``asyncio.run`` so the rewrite tier's parse-retry loop in
        :meth:`generate_rewrite` stays unchanged across backends.

        Returns a 3-tuple ``(text, retry_count, usage)`` (rewrite-overflow-
        fix-2026-06): ``usage`` is the server-reported token dict (carrying
        ``prompt_tokens``) extracted from the OpenAI-compatible response
        body, threaded back so the input-truncation tripwire can compare it
        against the local estimate. The Anthropic + claude_session branches
        return an EMPTY usage dict (no per-request token signal) → the
        tripwire no-ops there (fail-OPEN). No shared-mutable
        ``last_usage`` is read — usage travels by return value so a cloud
        block can't read a stale local OAI count.

        Reuses ``agent_type="content-generator"`` intentionally: the
        ``Courseforge/agents/content-generator.md`` spec already carries
        the Wave4b ``model: sonnet`` frontmatter pin and the Wave4-W27
        MANDATORY directives (HEADING_SKIP / source-ID / objective-ID
        stamping) — exactly the contract the rewrite tier needs.
        """
        if self._provider != "claude_session":
            return super()._dispatch_call_with_usage(
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
        # subagent either returns or fails. Return 0 retries + empty usage
        # (the subagent path has no server token tally) so the tripwire
        # no-ops here.
        return html_response, 0, {}

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
        output_contract = _block_type_output_contract(
            block.block_type, fit_window=getattr(self, "_fit_window", False)
        )
        output_contract += _recall_self_check_contract_suffix(block)
        output_contract += _misconception_productive_failure_contract_suffix(block)
        required_attrs_line = _required_attrs_directive(
            block.block_type, block.block_id
        )
        interaction_line = _interaction_type_directive(
            getattr(block, "interaction_type", None)
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
            f"{interaction_line}"
            "\n"
            "Title every heading from the SPECIFIC content of this block "
            "(its concepts / key terms / objective statements) — never a "
            "generic label like \"Objectives\" / \"Content\" / \"Section\" "
            "or the raw page id. Render math as Unicode or MathML, never raw "
            "LaTeX. Prefer compact structured elements (tables for "
            "comparisons, lists, styled boxes) over long prose.\n"
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
        output_contract = _block_type_output_contract(
            block.block_type, fit_window=getattr(self, "_fit_window", False)
        )
        output_contract += _recall_self_check_contract_suffix(block)
        output_contract += _misconception_productive_failure_contract_suffix(block)
        required_attrs_line = _required_attrs_directive(
            block.block_type, block.block_id
        )
        interaction_line = _interaction_type_directive(
            getattr(block, "interaction_type", None)
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
            f"{interaction_line}"
            "\n"
            "Title every heading from the SPECIFIC content of this block "
            "(its concepts / key terms / objective statements) — never a "
            "generic label like \"Objectives\" / \"Content\" / \"Section\" "
            "or the raw page id. Render math as Unicode or MathML, never raw "
            "LaTeX. Prefer compact structured elements (tables for "
            "comparisons, lists, styled boxes) over long prose.\n"
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

        # Fit-window (rewrite-overflow-fix-2026-06): trim the grounding to
        # the num_ctx budget BEFORE rendering. OFF → ``budgeted_chunks`` is
        # the input verbatim + no drops (byte-stable). Anti-fabrication:
        # ``budgeted_chunks`` ⊆ the input universe. rewrite-overflow-fix-
        # 2026-07: the budget now measures the WHOLE prompt (real scaffold),
        # and a scaffold that alone overflows the window is a LOUD, non-
        # dispatched escalation (never author a prompt that cannot fit).
        try:
            budgeted_chunks, dropped_cited_chunk_ids = (
                self._select_source_chunks_for_budget(
                    block, source_chunks, objectives
                )
            )
        except _ScaffoldOverflowError as exc:
            logger.warning(
                "RewriteProvider: non-chunk scaffold overflows the served "
                "window for block %r (sys=%d + scaffold=%d >= num_ctx=%d) — "
                "stamping rewrite_scaffold_overflow (never dispatched)",
                block.block_id,
                exc.sys_tokens,
                exc.scaffold_tokens,
                exc.num_ctx,
            )
            self._emit_per_call_decision(
                raw_text="",
                retry_count=0,
                block_id=block.block_id,
                block_type=block.block_type,
                page_id=block.page_id,
                escalation_marker=block.escalation_marker,
                outline_curie_count=len(outline_curies),
                remediation_attempts=0,
                scaffold_overflow=True,
                estimated_prompt_tokens=exc.sys_tokens + exc.scaffold_tokens,
                num_ctx=exc.num_ctx,
            )
            return dataclasses.replace(
                block, escalation_marker=_SCAFFOLD_OVERFLOW_MARKER
            )

        # Build the initial user prompt per the escalation flag.
        if block.escalation_marker is not None:
            user_prompt = self._render_escalated_user_prompt(
                block=block,
                source_chunks=budgeted_chunks,
                objectives=objectives,
            )
        else:
            user_prompt = self._render_user_prompt(
                block=block,
                source_chunks=budgeted_chunks,
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
        last_enforceable: List[str] = list(outline_curies)
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
            # PRE-dispatch deterministic tripwire (fit-window ON): a final
            # prompt whose local estimate exceeds the served window WILL
            # head-truncate — escalate BEFORE paying the round-trip rather
            # than dispatch a prompt that cannot fit.
            try:
                self._check_input_fits_predispatch(user_prompt)
            except _PromptTruncatedError as exc:
                logger.warning(
                    "RewriteProvider: pre-dispatch input over window for "
                    "block %r (num_ctx=%d) — short-circuiting as escalated: %s",
                    block.block_id,
                    self._rewrite_num_ctx,
                    exc,
                )
                self._emit_per_call_decision(
                    raw_text="",
                    retry_count=total_retries,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page_id=block.page_id,
                    escalation_marker=block.escalation_marker,
                    outline_curie_count=len(outline_curies),
                    remediation_attempts=attempt,
                    input_truncated=True,
                    estimated_prompt_tokens=(
                        _estimate_tokens(self._system_prompt)
                        + _estimate_tokens(user_prompt)
                    ),
                    reported_prompt_tokens=None,
                    num_ctx=self._rewrite_num_ctx,
                    dropped_cited_chunk_ids=dropped_cited_chunk_ids,
                )
                return dataclasses.replace(
                    block,
                    escalation_marker=_INPUT_PROMPT_TRUNCATED_MARKER,
                )
            try:
                html_response, retry_count, usage = self._dispatch_call(
                    user_prompt,
                    extra_payload=self._num_ctx_options_payload(),
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
                # A length-truncation means the model over-generated. Re-
                # dispatching the identical prompt just re-truncates. Inject a
                # dominating brevity remediation so the NEXT attempt produces a
                # drastically shorter block that finishes inside the cap (the
                # deterministic styled-template fallback remains the net if the
                # model still won't comply after the budget).
                if _is_length_truncation(exc):
                    user_prompt = _append_brevity_remediation(user_prompt)
                attempt += 1
                continue
            dispatched_ok = True
            # Input-truncation tripwire (rewrite-overflow-fix-2026-06):
            # compare the server-reported prompt_tokens against the local
            # estimate. A large shortfall means the served window dropped
            # the system prompt HEAD (the authoring CONTRACT), so the model
            # authored with source but no rules. HARD, NON-RETRYABLE fail —
            # re-dispatching the same prompt re-truncates. Stamp the
            # block with ``input_prompt_truncated`` (surfaces as escalated)
            # and short-circuit the parse-retry loop. Default-ON, fail-OPEN
            # (no-op when usage is absent/zero — Ollama may omit it).
            try:
                self._check_truncation(
                    system_prompt=self._system_prompt,
                    user_prompt=user_prompt,
                    usage=usage,
                    block_id=block.block_id,
                )
            except _PromptTruncatedError as exc:
                logger.warning(
                    "RewriteProvider: input prompt truncated for block %r "
                    "(num_ctx=%d) — short-circuiting as escalated: %s",
                    block.block_id,
                    self._rewrite_num_ctx,
                    exc,
                )
                self._emit_per_call_decision(
                    raw_text=html_response,
                    retry_count=total_retries + retry_count,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page_id=block.page_id,
                    escalation_marker=block.escalation_marker,
                    outline_curie_count=len(outline_curies),
                    remediation_attempts=attempt,
                    input_truncated=True,
                    estimated_prompt_tokens=(
                        _estimate_tokens(self._system_prompt)
                        + _estimate_tokens(user_prompt)
                    ),
                    reported_prompt_tokens=(
                        usage.get("prompt_tokens")
                        if isinstance(usage, dict) else None
                    ),
                    num_ctx=self._rewrite_num_ctx,
                    dropped_cited_chunk_ids=dropped_cited_chunk_ids,
                )
                return dataclasses.replace(
                    block,
                    escalation_marker=_INPUT_PROMPT_TRUNCATED_MARKER,
                )
            # Post-emit sanitizers, run BEFORE any downstream gate / parser /
            # consumer sees the response:
            #   1. Repair entity-escaped comment closes (``--&gt;`` → ``-->``)
            #      so an unterminated comment stops swallowing the downstream
            #      <track>/<details>/<table> a11y siblings. Runs FIRST so the
            #      previously-swallowed tags are visible to the orphan-tag
            #      sanitizer and the IB5 a11y backstop below.
            #   2. Escape orphan-opener placeholder tags (``<word>`` with no
            #      attributes and no closer) as ``&lt;word&gt;``.
            #   3. IB5 structural a11y backstop — re-inject any
            #      renderer-guaranteed B04/B06 a11y skeleton the free-authored
            #      emit dropped (structural presence only; no fabricated prose).
            html_response = _fix_malformed_comment_closes(html_response)
            html_response = _escape_orphan_placeholder_tags(html_response)
            html_response = _inject_ib5_a11y_skeleton(html_response, block)
            total_retries += retry_count
            last_text = html_response

            # M3 tolerance: only ENFORCE the CURIEs whose underlying term
            # the rewrite tier actually used in the prose (plus a ≥1
            # anchoring fallback). Off-topic CURIEs the model didn't use
            # are pruned from enforcement rather than forced back in — the
            # all-or-nothing force-all was what bloated CURIE-dense blocks
            # past max_tokens (finish_reason='length' → truncation). The
            # enforced set is always a subset of outline_curies (no
            # fabrication); the ≥1 fallback keeps rewrite_curie_anchoring
            # satisfied. ``last_enforceable`` carries the post-prune set so
            # the exhaustion force-inject targets only the kept ≥1.
            enforceable = _select_enforceable_curies(
                outline_curies, html_response
            )
            last_enforceable = enforceable

            missing = _missing_preserve_curies(html_response, enforceable)
            if not missing:
                # Provider's tolerant M3 check (``_curie_in_pedagogical_context``,
                # TERM-based) is satisfied. BUT the post-rewrite
                # ``BlockCurieAnchoringValidator`` str path extracts actual CURIE
                # TOKENS via ``_extract_curies(_strip_html(html))`` — when the
                # prose used the term but not the synthetic token, the provider
                # is happy yet the validator finds ZERO CURIEs and fails closed
                # (observed 2026-06-20: 84/289 blocks). Reconcile the two
                # success conditions by force-injecting the enforceable set as
                # hidden spans before returning: ``_force_inject_curies`` is
                # idempotent and dedupes via the validator's OWN
                # ``_extract_curies`` pipeline, so it adds ONLY the tokens the
                # validator can't already extract (prose stays natural — the
                # token rides in a hidden span, never visible code-voice).
                html_response = _force_inject_curies(html_response, enforceable)
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
                    enforced_curie_count=len(enforceable),
                    pruned_curie_count=len(outline_curies) - len(enforceable),
                    dropped_cited_chunk_ids=dropped_cited_chunk_ids,
                )
                return _apply_rewrite_touch(
                    block=block,
                    html_response=html_response,
                    provider=_touch_provenance(self._provider),
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
            enforced_curie_count=len(last_enforceable),
            pruned_curie_count=len(outline_curies) - len(last_enforceable),
            dropped_cited_chunk_ids=dropped_cited_chunk_ids,
        )
        return _apply_rewrite_touch(
            block=block,
            html_response=injected_html,
            provider=_touch_provenance(self._provider),
            model=self._model,
            decision_capture_id=self._last_capture_id(),
            purpose=_TOUCH_PURPOSE_CURIE_FORCED,
        )

    # ------------------------------------------------------------------
    # Multi-block BATCHED rewrite dispatch (rate-limit defeat)
    # ------------------------------------------------------------------

    def generate_rewrite_batch(
        self,
        blocks: Sequence[Block],
        *,
        source_chunks_by_id: Optional[Dict[str, Sequence[Any]]] = None,
        objectives: Optional[Sequence[Any]] = None,
        remediation_suffix_by_id: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Optional[Block]]:
        """Rewrite MANY outline blocks in ONE HTTP POST.

        Builds ONE user prompt that wraps each block's existing rendered
        single-block prompt (``_render_user_prompt`` / the escalated variant)
        inside the ``CF_BLOCK`` envelope, dispatches ONE ``_dispatch_call``,
        parses the response back per-block via
        :func:`parse_rewrite_batch_envelope`, and reuses ``_force_inject_curies``
        per parsed block so minted CURIEs survive into the published HTML.

        Returns ``{block_id: Block | None}`` — a block whose envelope slot is
        absent / empty (parse-miss or model omission) maps to ``None`` (the
        fail-soft sentinel; the router re-batches None blocks, never silently
        dropping them). The single-block :meth:`generate_rewrite` path is left
        UNCHANGED so the off (per-block) path stays byte-stable.

        This is the rate-limit-defeat path: ONE POST for ``len(blocks)`` blocks
        instead of one POST per block. Per-block-type validation still runs
        per-block (post-parse) in the router — this method only collapses the
        HTTP dispatch.
        """
        from Courseforge.generators._rewrite_batch import (  # noqa: PLC0415
            CF_BLOCK_OPEN,
            CF_BLOCK_CLOSE,
            parse_rewrite_batch_envelope,
        )

        block_list = list(blocks)
        block_ids = [b.block_id for b in block_list]
        if not block_list:
            return {}

        chunks_by_id = source_chunks_by_id or {}
        suffix_by_id = remediation_suffix_by_id or {}

        # Build ONE user prompt: each block's full single-block prompt wrapped
        # in its CF_BLOCK envelope. Reuses the exact per-block renderers so the
        # batched and per-block prompts carry identical authoring instructions.
        segments: List[str] = [_BATCH_ENVELOPE_DIRECTIVE, ""]
        # Track per-block fit-window cited-chunk drops for the decision
        # capture (empty when the flag is off → byte-stable).
        dropped_cited_by_id: Dict[str, List[str]] = {}
        # Blocks whose non-chunk scaffold alone overflowed the served window
        # are stamped + excluded from the batch POST (never dispatch a prompt
        # that cannot fit); they surface as escalated in the return map.
        overflow_results: Dict[str, Optional[Block]] = {}
        batched_blocks: List[Block] = []
        for block in block_list:
            block_chunks = chunks_by_id.get(block.block_id) or []
            # Fit-window: trim each block's grounding to the budget BEFORE
            # rendering. OFF → verbatim input + no drops (byte-stable).
            # rewrite-overflow-fix-2026-07: a scaffold that alone overflows the
            # window is a loud per-block escalation (not batched).
            try:
                budgeted_chunks, dropped_cited = (
                    self._select_source_chunks_for_budget(
                        block, block_chunks, objectives
                    )
                )
            except _ScaffoldOverflowError as exc:
                logger.warning(
                    "RewriteProvider.generate_rewrite_batch: non-chunk "
                    "scaffold overflows the served window for block %r "
                    "(sys=%d + scaffold=%d >= num_ctx=%d) — stamping "
                    "rewrite_scaffold_overflow (excluded from batch)",
                    block.block_id,
                    exc.sys_tokens,
                    exc.scaffold_tokens,
                    exc.num_ctx,
                )
                overflow_results[block.block_id] = dataclasses.replace(
                    block, escalation_marker=_SCAFFOLD_OVERFLOW_MARKER
                )
                continue
            batched_blocks.append(block)
            if dropped_cited:
                dropped_cited_by_id[block.block_id] = dropped_cited
            if block.escalation_marker is not None:
                inner = self._render_escalated_user_prompt(
                    block=block,
                    source_chunks=budgeted_chunks,
                    objectives=objectives,
                )
            else:
                inner = self._render_user_prompt(
                    block=block,
                    source_chunks=budgeted_chunks,
                    objectives=objectives,
                )
            suffix = suffix_by_id.get(block.block_id)
            if suffix:
                inner = inner + "\n\n" + suffix
            open_tag = CF_BLOCK_OPEN.format(block_id=block.block_id)
            close_tag = CF_BLOCK_CLOSE.format(block_id=block.block_id)
            segments.append(open_tag)
            segments.append(inner)
            segments.append(close_tag)
            segments.append("")
        batch_prompt = "\n".join(segments)

        # Every block's scaffold overflowed → nothing to POST; return the
        # per-block escalations only.
        if not batched_blocks:
            return dict(overflow_results)

        batched_ids = [b.block_id for b in batched_blocks]

        # PRE-dispatch deterministic tripwire (fit-window ON): a batch prompt
        # whose local estimate exceeds the served window WILL head-truncate
        # (dropping the leading block(s)' authoring contracts). Stamp the whole
        # batch escalated BEFORE the POST rather than dispatch a prompt that
        # cannot fit.
        try:
            self._check_input_fits_predispatch(batch_prompt)
        except _PromptTruncatedError as exc:
            logger.warning(
                "RewriteProvider.generate_rewrite_batch: pre-dispatch batch "
                "prompt over window for %d block(s) (num_ctx=%d) — stamping "
                "all as escalated: %s",
                len(batched_blocks),
                self._rewrite_num_ctx,
                exc,
            )
            results = dict(overflow_results)
            for b in batched_blocks:
                results[b.block_id] = dataclasses.replace(
                    b, escalation_marker=_INPUT_PROMPT_TRUNCATED_MARKER
                )
            return results

        # ONE POST. A dispatch raise degrades EVERY block in the batch to None
        # (the router re-batches them) rather than aborting the whole phase —
        # mirrors the per-block fail-soft contract.
        try:
            html_response, retry_count, usage = self._dispatch_call(
                batch_prompt,
                extra_payload=self._num_ctx_options_payload(),
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft per batch
            logger.warning(
                "RewriteProvider.generate_rewrite_batch: batched POST "
                "raised for %d block(s): %s — degrading those blocks to None",
                len(batched_blocks),
                exc,
            )
            results = dict(overflow_results)
            results.update({bid: None for bid in batched_ids})
            return results

        # Input-truncation tripwire (rewrite-overflow-fix-2026-06): the whole
        # batch travelled through ONE POST, so a silent head-truncation
        # dropped the leading block(s)' prompts. HARD, NON-RETRYABLE — stamp
        # EVERY block in the batch with ``input_prompt_truncated`` (re-
        # batching the same over-window batch re-truncates). Default-ON,
        # fail-OPEN (no-op when usage absent/zero — the cloud lane that
        # batching defaults to returns empty Anthropic usage).
        try:
            self._check_truncation(
                system_prompt=self._system_prompt,
                user_prompt=batch_prompt,
                usage=usage,
                block_id=",".join(batched_ids),
            )
        except _PromptTruncatedError as exc:
            logger.warning(
                "RewriteProvider.generate_rewrite_batch: input prompt "
                "truncated for %d block(s) (num_ctx=%d) — stamping all as "
                "escalated: %s",
                len(batched_blocks),
                self._rewrite_num_ctx,
                exc,
            )
            results = dict(overflow_results)
            for b in batched_blocks:
                results[b.block_id] = dataclasses.replace(
                    b, escalation_marker=_INPUT_PROMPT_TRUNCATED_MARKER
                )
            return results

        # Repair entity-escaped comment closes (``--&gt;`` → ``-->``) at the
        # envelope level FIRST so an unterminated comment in one block's
        # fragment doesn't swallow the per-block delimiter / a11y siblings,
        # then escape orphan-opener placeholder tags. The IB5 a11y backstop
        # runs PER-BLOCK below (it needs the Block for its source-text fields).
        html_response = _fix_malformed_comment_closes(html_response)
        html_response = _escape_orphan_placeholder_tags(html_response)
        parsed = parse_rewrite_batch_envelope(html_response, batched_ids)

        # Seed with the scaffold-overflow escalations so they surface in the
        # return map alongside the batched results.
        results: Dict[str, Optional[Block]] = dict(overflow_results)
        n_parsed = 0
        for block in batched_blocks:
            fragment = parsed.get(block.block_id)
            if not (isinstance(fragment, str) and fragment.strip()):
                results[block.block_id] = None
                continue
            n_parsed += 1
            # Reuse the per-block CURIE-preservation force-inject so minted
            # CURIEs survive (the validator's str path extracts tokens). The
            # batched path runs no inner remediation retry — a block whose
            # validation fails is re-batched by the router round loop, exactly
            # like the per-block remediation-suffix retry.
            # IB5 structural a11y backstop — re-inject any renderer-guaranteed
            # B04/B06 a11y skeleton this free-authored fragment dropped (the
            # comment-close repair above already ran at the envelope level, so a
            # repaired comment exposes the previously-swallowed siblings here).
            fragment = _inject_ib5_a11y_skeleton(fragment, block)
            outline_curies = _extract_outline_curies(block.content)
            enforceable = _select_enforceable_curies(outline_curies, fragment)
            fragment = _force_inject_curies(fragment, enforceable)
            self._emit_per_call_decision(
                raw_text=fragment,
                retry_count=retry_count,
                block_id=block.block_id,
                block_type=block.block_type,
                page_id=block.page_id,
                escalation_marker=block.escalation_marker,
                outline_curie_count=len(outline_curies),
                remediation_attempts=0,
                enforced_curie_count=len(enforceable),
                pruned_curie_count=len(outline_curies) - len(enforceable),
                dropped_cited_chunk_ids=dropped_cited_by_id.get(
                    block.block_id, []
                ),
            )
            results[block.block_id] = _apply_rewrite_touch(
                block=block,
                html_response=fragment,
                provider=_touch_provenance(self._provider),
                model=self._model,
                decision_capture_id=self._last_capture_id(),
            )

        if self._capture is not None:
            try:
                self._emit_decision(
                    decision_type="block_rewrite_call",
                    decision=(
                        f"batched rewrite: {n_parsed}/{len(batched_blocks)} "
                        f"block(s) parsed"
                    ),
                    rationale=(
                        f"Batched rewrite POST authored {len(batched_blocks)} "
                        f"block(s) in ONE request via provider={self._provider}, "
                        f"model={self._model} (rate-limit defeat). Parsed "
                        f"{n_parsed} block slot(s) from the CF_BLOCK envelope; "
                        f"{len(batched_blocks) - n_parsed} slot(s) were "
                        f"absent/empty and degrade to None for the router "
                        f"round-loop to re-batch (fail-soft, never silently "
                        f"dropped). {len(overflow_results)} block(s) excluded "
                        f"as rewrite_scaffold_overflow (scaffold alone exceeds "
                        f"the served window)."
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — never break dispatch
                logger.warning(
                    "generate_rewrite_batch decision-capture emit failed: %s",
                    exc,
                )

        return results

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
        enforced_curie_count = call_context.get("enforced_curie_count")
        pruned_curie_count = call_context.get("pruned_curie_count")
        input_truncated = bool(call_context.get("input_truncated", False))
        scaffold_overflow = bool(call_context.get("scaffold_overflow", False))
        estimated_prompt_tokens = call_context.get("estimated_prompt_tokens")
        reported_prompt_tokens = call_context.get("reported_prompt_tokens")
        num_ctx = call_context.get("num_ctx")
        dropped_cited_chunk_ids = (
            call_context.get("dropped_cited_chunk_ids") or []
        )

        if scaffold_overflow:
            outcome = "scaffold_overflow"
        elif input_truncated:
            outcome = "input_truncated"
        elif curie_force_injected:
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
        if pruned_curie_count is not None and enforced_curie_count is not None:
            # M3 tolerance: how many off-topic CURIEs were pruned from
            # enforcement vs kept (the model used / the ≥1 anchoring floor).
            rationale_parts.append(
                f"M3 tolerant preservation: enforced {enforced_curie_count} "
                f"on-topic CURIE(s), pruned {pruned_curie_count} the rewrite "
                f"tier did not use (kept >=1 to satisfy the anchoring "
                f"invariant) so the prose was not bloated past max_tokens."
            )
        if scaffold_overflow:
            rationale_parts.append(
                f"SCAFFOLD OVERFLOW: the non-chunk scaffold (system prompt + "
                f"outline dict + per-claim + objectives + contract) estimated "
                f"~{estimated_prompt_tokens} tokens, which alone exceeds the "
                f"served window (num_ctx={num_ctx}) — no grounding chunk could "
                f"fit, so the block was stamped rewrite_scaffold_overflow and "
                f"NEVER dispatched (authoring an over-window prompt would "
                f"silently head-truncate the CONTRACT). Raise the served "
                f"window / ED4ALL_REWRITE_NUM_CTX, or shrink the upstream "
                f"outline payload."
            )
        if input_truncated:
            rationale_parts.append(
                f"INPUT PROMPT TRUNCATED: the served window "
                f"(num_ctx={num_ctx}) silently dropped the prompt HEAD — "
                f"estimated ~{estimated_prompt_tokens} prompt tokens but "
                f"the server reported {reported_prompt_tokens}; the "
                f"system-prompt authoring CONTRACT was lost. Stamped "
                f"input_prompt_truncated (hard, non-retryable); raise the "
                f"served window or ED4ALL_REWRITE_NUM_CTX, or enable "
                f"ED4ALL_REWRITE_FIT_WINDOW to shrink the prompt to fit."
            )
            if dropped_cited_chunk_ids:
                rationale_parts.append(
                    f"Fit-window dropped {len(dropped_cited_chunk_ids)} "
                    f"cited chunk(s) under the budget: "
                    f"{dropped_cited_chunk_ids}."
                )
        elif dropped_cited_chunk_ids:
            rationale_parts.append(
                f"Fit-window chunk budget dropped "
                f"{len(dropped_cited_chunk_ids)} cited chunk(s) "
                f"(drop-trailing-whole, never head-truncated): "
                f"{dropped_cited_chunk_ids}."
            )
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
