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

import dataclasses
import datetime as _dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Courseforge.generators._base import _BaseLLMProvider
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

SUPPORTED_PROVIDERS = ("anthropic", "together", "local", "openai_compatible")

# Subtask 26: bounded remediation retries for the CURIE-preservation
# gate. Direct port of the Trainforge precedent
# (``_local_provider.py:540`` :: ``MAX_PARSE_RETRIES``).
MAX_PARSE_RETRIES = 2


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
        "explanatory paragraphs."
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
        "`data-cf-content-type=\"callout\"`."
    ),
    "flip_card_grid": (
        "Emit a `<div class=\"flip-card-grid\">` whose children are "
        "per-card `<div class=\"flip-card\">` elements carrying "
        "`data-cf-component=\"flip-card\"`, "
        "`data-cf-purpose=\"term-definition\"`, "
        "`data-cf-teaching-role`, and `data-cf-term`."
    ),
    "self_check_question": (
        "Emit a `<div class=\"self-check\">` carrying "
        "`data-cf-component=\"self-check\"`, "
        "`data-cf-purpose=\"formative-assessment\"`, "
        "`data-cf-bloom-level`, `data-cf-objective-ref`, and "
        "`data-cf-source-ids` / `data-cf-source-primary`."
    ),
    "activity": (
        "Emit a `<div class=\"activity-card\">` carrying "
        "`data-cf-component=\"activity\"`, `data-cf-purpose=\"practice\"`, "
        "`data-cf-bloom-level`, `data-cf-objective-ref`, and "
        "`data-cf-source-ids`."
    ),
    "misconception": (
        "Emit a `<section>` whose JSON-LD entry carries the "
        "misconception ID (mc_[0-9a-f]{16}) plus correction. The HTML "
        "body need not carry data-cf-* attributes — JSON-LD is the "
        "authoritative shape for misconception blocks."
    ),
    "assessment_item": (
        "Emit a `<div class=\"assessment-item\">` carrying the "
        "question stem, options, and correct-answer marker. "
        "Assessment items in IMSCC live in QTI XML downstream; the "
        "HTML emit here is the authoring fixture."
    ),
    "prereq_set": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an `<h2>` "
        "or `<h3>` and an `<ol>` of prerequisite topic refs."
    ),
    "reflection_prompt": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h3>` and one or more `<p>` reflection prompts."
    ),
    "discussion_prompt": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h3>` and one or more `<p>` discussion prompts."
    ),
    "chrome": (
        "Emit page chrome (header / footer / nav). Carry "
        "`data-cf-role=\"template-chrome\"` on the wrapper."
    ),
    "recap": (
        "Emit a `<section data-cf-source-ids=...>` wrapping an "
        "`<h2>` or `<h3>` and a recap of the prior week's key terms."
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


def _apply_rewrite_touch(
    *,
    block: Block,
    html_response: str,
    provider: str,
    model: str,
    decision_capture_id: str,
) -> Block:
    """Return a new Block with the rewrite output and a new Touch entry.

    The rewrite tier's Touch carries:

    - ``tier="rewrite"``
    - ``purpose="pedagogical_depth"``
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
        purpose="pedagogical_depth",
    )
    return dataclasses.replace(
        block,
        content=html_response,
        touched_by=block.touched_by + (touch,),
    )


def _format_objectives(objectives: Sequence[Any]) -> str:
    """Format the objectives list into a readable prompt block.

    Accepts either dict shape (``{"id": ..., "statement": ...}``) or
    object with ``id`` + ``statement`` attributes. Empty input renders
    ``"(none)"``.
    """
    if not objectives:
        return "(none)"
    parts: List[str] = []
    for o in objectives:
        if isinstance(o, dict):
            oid = o.get("id") or "<unknown>"
            statement = o.get("statement") or ""
        else:
            oid = getattr(o, "id", "<unknown>")
            statement = getattr(o, "statement", "")
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
        # Optional dependency injections for tests.
        client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
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
        import os
        resolved_model = model or os.environ.get(ENV_MODEL)

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
            client=client,
            anthropic_client=anthropic_client,
            env_provider_var=ENV_PROVIDER,
            default_provider=DEFAULT_PROVIDER,
            default_model_anthropic=DEFAULT_MODEL_ANTHROPIC,
            default_model_together=DEFAULT_MODEL_TOGETHER,
            default_model_local=DEFAULT_MODEL_LOCAL,
            supported_providers=("anthropic", "together", "local"),
            system_prompt=_REWRITE_SYSTEM_PROMPT,
        )

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
        source_block = _format_source_chunks(source_chunks or [])
        objectives_block = _format_objectives(objectives or [])
        output_contract = _block_type_output_contract(block.block_type)

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
            "Source chunks (the authoritative grounding):\n"
            f"{source_block}\n"
            "\n"
            "Objectives:\n"
            f"{objectives_block}\n"
            "\n"
            "Output contract (HTML attributes for this block_type):\n"
            f"{output_contract}\n"
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
        source_block = _format_source_chunks(source_chunks or [])
        objectives_block = _format_objectives(objectives or [])
        output_contract = _block_type_output_contract(block.block_type)

        return (
            f"Block type: {block.block_type}\n"
            f"Block id: {block.block_id}\n"
            f"Page id: {block.page_id}\n"
            "\n"
            "Outline (structurally correct, pedagogical-depth missing):\n"
            f"{outline_payload}\n"
            "\n"
            "Source chunks (cite via source_refs):\n"
            f"{source_block}\n"
            "\n"
            "Objectives:\n"
            f"{objectives_block}\n"
            "\n"
            "Output contract (HTML attributes for this block_type):\n"
            f"{output_contract}\n"
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
        total_retries = 0
        # Initial attempt + ``MAX_PARSE_RETRIES`` remediation retries =
        # ``MAX_PARSE_RETRIES + 1`` total dispatches at most. Mirrors the
        # ``for attempts in range(retry_budget)`` loop in
        # ``_local_provider._call_with_parse``.
        for attempt in range(MAX_PARSE_RETRIES + 1):
            html_response, retry_count = self._dispatch_call(user_prompt)
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

        # Exhausted retry budget. Emit the failure decision so the
        # audit trail captures the drop, then raise.
        self._emit_per_call_decision(
            raw_text=last_text,
            retry_count=total_retries,
            block_id=block.block_id,
            block_type=block.block_type,
            page_id=block.page_id,
            escalation_marker=block.escalation_marker,
            outline_curie_count=len(outline_curies),
            remediation_attempts=MAX_PARSE_RETRIES + 1,
            curie_drop=True,
            missing_curies=last_missing,
        )
        raise RewriteProviderError(
            f"RewriteProvider: rewrite output dropped CURIEs after "
            f"{MAX_PARSE_RETRIES + 1} attempts. Missing: {last_missing}; "
            f"tail of last response: {last_text[-500:]!r}",
            code="rewrite_curie_drop",
            missing_curies=last_missing,
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
        missing_curies = call_context.get("missing_curies") or []

        outcome = "curie_drop" if curie_drop else "success"
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
        if curie_drop:
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
    "_REWRITE_SYSTEM_PROMPT",
]
