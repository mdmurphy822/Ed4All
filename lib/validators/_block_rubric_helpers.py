"""Shared block-reading helpers for the IB6 quality-rubric surface.

The IB6 validators (cognitive-load ceiling, anatomy slot-presence,
interaction-feedback, knowledge-type chip, the rubric scorer, the QA
checklist) all operate over the canonical ``Courseforge.scripts.blocks.Block``
dataclass OR the outline-tier block dict. They share a handful of pure,
shape-tolerant accessors so each validator does not re-implement the
``getattr``-or-``[key]`` dance + the HTML-text strip + the framework B-code
lookup. Mirrors the SoT-loader posture of the sibling helpers in
``lib.validators.alignment.verb_triple`` (which owns the same ``_block_attr``).

NO new dependencies: stdlib + ``lib.ontology.framework_blocks`` (B-code map).
Pure functions, tolerant of missing / unresolvable inputs (return ``None`` /
``""`` / ``()`` rather than raising) so a gate wiring these behind a flag
degrades cleanly on legacy / thin block shapes.
"""
from __future__ import annotations

import os
import re
from typing import Any, List, Mapping, Optional, Tuple

from lib.ontology.framework_blocks import framework_block_for

__all__ = [
    "BLOCK_QUALITY_RUBRIC_ENV",
    "block_quality_rubric_enabled",
    "BLOCK_QUALITY_SHADOW_ENV",
    "block_quality_shadow_enabled",
    "block_quality_scoring_active",
    "BLOCK_BODY_CHAR_CEILING_ENV",
    "resolve_body_char_ceiling",
    "INTERACTIVE_FRAMEWORK_BLOCKS",
    "block_attr",
    "block_type_of",
    "framework_block_of",
    "is_interactive_block",
    "strip_html_text",
    "body_text_of",
    "count_idea_chunks",
]

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# IB6 keystone flag. Default OFF → the whole IB6 scoring / rollup / chip surface
# is inert and snapshots stay byte-identical (read each call so tests toggle).
BLOCK_QUALITY_RUBRIC_ENV = "ED4ALL_BLOCK_QUALITY_RUBRIC"

# W8.8 SHADOW-COLLECT flag. Default OFF. Chicken-and-egg fix: the IB6 gates are
# default-OFF (keystone rubric flag), so they never COMPUTE and never generate the
# fire-rate data the calibration harness needs to justify a critical-flip. When this
# flag is truthy (and the keystone rubric flag is off), the IB6 validators run their
# MEASUREMENT path — computing + recording the per-block signals (metadata + warning
# GateIssues + decision captures the calibration harness reads) — WITHOUT any verdict
# change: these gates are warning-day-1 (severity: warning, on_fail: warn), so they
# already never block, and shadow adds no critical issue. The emit-side render (the
# ``ED4ALL_BLOCK_QUALITY_RUBRIC``-gated chip / rubric-field HTML) stays OFF, so product
# bytes / snapshots remain byte-identical — only validator telemetry accrues.
BLOCK_QUALITY_SHADOW_ENV = "ED4ALL_BLOCK_QUALITY_SHADOW"

# IB6.4 D2 body ceiling (chars). Default 200 ("~200char single idea" target).
BLOCK_BODY_CHAR_CEILING_ENV = "ED4ALL_BLOCK_BODY_CHAR_CEILING"
_DEFAULT_BODY_CHAR_CEILING = 200

# IB6.4 per-block-TYPE body-char budget (FIX 2 — type-blind-ceiling fix;
# FIX 3 — per-type granularity calibration, cal2 cohort).
#
# The single global 200-char ceiling was structurally unsatisfiable for the
# exposition / answer-bearing block types: across two real corpora only 1 of 99
# non-exempt blocks measured <=200 chars (measured medians 794 and 1216 on
# two calibration corpora). The
# 200 target is the right ATOMIC single-idea budget (key_idea / callout / vocab
# card / formula — a one-line idea), but exposition that legitimately develops
# ONE idea across a worked example, a diagram long-description, a scenario, or a
# multi-step problem needs a higher budget, and a STRUCTURAL aggregator (a
# prereq set, a recap, a checklist, an acronym table) is a roll-up of several
# entries by construction, so it needs a higher budget still.
#
# FIX 2 raised only 6 exposition types to 1000 and left every OTHER non-exempt
# type pinned at the 200 atomic default. On the cal2 cohort (a 66-block
# calibration-corpus rewrite) the 7B authored reasonable-length content for
# many of those still-pinned types — objective 1599 / acronym 1267 /
# prereq_set 836 / hook 547 / problem 897 / reflection_prompt 1003 / recap 473
# / checklist 784 / activity 1623 / explanation 2879 — so 37 non-exempt blocks
# (50 incl. the orthogonal chunk axis) tripped the char ceiling, pinning the
# gate at its 50-issue cap with mostly FALSE positives (a 547-char hook is not
# an "everything-block"; it is a normally-sized activation prompt).
#
# FIX 3 extends the table to a realistic per-type ceiling reflecting each
# type's INTENDED granularity, grounded in the cal2 cohort's measured p50-p75
# bodies (so the table does NOT just blanket-raise to never fire — genuine
# over-stuffers still trip it):
#
#   * ATOMIC (~200) — a single one-line idea: vocab_card / callout / formula /
#     key_idea. (cal2 key_idea ran 266-516 → still trips the char axis, which
#     is the intended catch: a key_idea is meant to be one line.)
#   * DEVELOPED (~1000) — one idea developed across a few sentences / steps:
#     concept / example / worked_example / self_check_question / diagram /
#     scenario (the FIX 2 six) PLUS explanation / problem / activity /
#     guided_practice / misconception / hook / multimedia. (cal2 p50-p75 for
#     these sits 397-937; the genuine over-stuffers — concept 1839,
#     explanation 2879 — still trip it.)
#   * AGGREGATING (~1200) — a structural roll-up of several entries by
#     construction: prereq_set / recap / checklist / reflection_prompt /
#     acronym / table / discussion_prompt / resources / flip_card_grid. (cal2
#     p75 836/473/784/1165 sits under 1200; the genuine over-stuffer —
#     acronym 2058 — still trips it.)
#
# ``objective`` and ``summary_takeaway`` are NOT in the table: they are
# EXEMPT from the body-overflow check entirely in
# ``content.py::_COGNITIVE_LOAD_EXEMPT_TYPES`` (the validator skips them before
# resolving a ceiling), so adding a ceiling row for them would be dead config.
#
# The block_catalog.yaml carries no char-budget field, so this constant table
# is the single source of truth. The orthogonal ">4-idea-chunk" axis
# (content.py::_BLOCK_IDEA_CHUNK_CEILING) is UNTOUCHED — it remains the real
# "everything-block" over-stuffing catcher regardless of the per-type char
# budget; the char ceiling is the softer per-type signal.
_ATOMIC_BODY_CHAR_CEILING = _DEFAULT_BODY_CHAR_CEILING  # 200
_EXPOSITION_BODY_CHAR_CEILING = 1000
_AGGREGATING_BODY_CHAR_CEILING = 1200
_BLOCK_BODY_CHAR_CEILING_BY_TYPE: dict = {
    # ATOMIC single-idea micro-blocks — keep the 200 atomic budget.
    "vocab_card": _ATOMIC_BODY_CHAR_CEILING,
    "callout": _ATOMIC_BODY_CHAR_CEILING,
    "formula": _ATOMIC_BODY_CHAR_CEILING,
    "key_idea": _ATOMIC_BODY_CHAR_CEILING,
    # DEVELOPED single-idea exposition / answer-bearing types — ~1000.
    "concept": _EXPOSITION_BODY_CHAR_CEILING,
    "example": _EXPOSITION_BODY_CHAR_CEILING,
    "worked_example": _EXPOSITION_BODY_CHAR_CEILING,
    "self_check_question": _EXPOSITION_BODY_CHAR_CEILING,
    "diagram": _EXPOSITION_BODY_CHAR_CEILING,
    "scenario": _EXPOSITION_BODY_CHAR_CEILING,
    "explanation": _EXPOSITION_BODY_CHAR_CEILING,
    "problem": _EXPOSITION_BODY_CHAR_CEILING,
    "activity": _EXPOSITION_BODY_CHAR_CEILING,
    "guided_practice": _EXPOSITION_BODY_CHAR_CEILING,
    "misconception": _EXPOSITION_BODY_CHAR_CEILING,
    "hook": _EXPOSITION_BODY_CHAR_CEILING,
    "multimedia": _EXPOSITION_BODY_CHAR_CEILING,
    # AGGREGATING / structural roll-ups — ~1200.
    "prereq_set": _AGGREGATING_BODY_CHAR_CEILING,
    "recap": _AGGREGATING_BODY_CHAR_CEILING,
    "checklist": _AGGREGATING_BODY_CHAR_CEILING,
    "reflection_prompt": _AGGREGATING_BODY_CHAR_CEILING,
    "acronym": _AGGREGATING_BODY_CHAR_CEILING,
    "table": _AGGREGATING_BODY_CHAR_CEILING,
    "discussion_prompt": _AGGREGATING_BODY_CHAR_CEILING,
    "resources": _AGGREGATING_BODY_CHAR_CEILING,
    "flip_card_grid": _AGGREGATING_BODY_CHAR_CEILING,
}

# The framework's interactive block codes — B07 Knowledge-Check, B08 Guided
# Practice, B10 Discussion, B14 Graded Assessment. These are the codes the
# Feedback / interaction-presence gates apply to (the plan's B07/B08/B10/B14).
INTERACTIVE_FRAMEWORK_BLOCKS: frozenset = frozenset({"B07", "B08", "B10", "B14"})


def block_quality_rubric_enabled() -> bool:
    """True iff ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is truthy (read each call)."""
    return os.environ.get(BLOCK_QUALITY_RUBRIC_ENV, "").strip().lower() in _TRUTHY


def block_quality_shadow_enabled() -> bool:
    """True iff ``ED4ALL_BLOCK_QUALITY_SHADOW`` is truthy (W8.8, read each call).

    Shadow-collect is measurement-only: the IB6 gates compute + record their signals
    without gating (they are warning-day-1, so no verdict changes) so calibration
    fire-rate data can accrue while the keystone rubric flag — and its emit-side
    rendering — stays off. Parse-with-fallback: garbage / unset → False.
    """
    return os.environ.get(BLOCK_QUALITY_SHADOW_ENV, "").strip().lower() in _TRUTHY


def block_quality_scoring_active() -> bool:
    """True iff the IB6 scoring/measurement path should run (W8.8).

    ``ED4ALL_BLOCK_QUALITY_RUBRIC`` (the keystone: scoring + emit + rollup) OR
    ``ED4ALL_BLOCK_QUALITY_SHADOW`` (measurement only, no emit, no verdict change).
    Default (both unset) → False → byte-identical to the pre-W8.8 disabled no-op.
    """
    return block_quality_rubric_enabled() or block_quality_shadow_enabled()


def resolve_body_char_ceiling(
    override: Optional[int] = None, block_type: Optional[str] = None
) -> int:
    """Resolve the D2 body char ceiling per block type.

    Precedence (high → low):
      1. explicit positive ``override`` arg (caller-pinned), then
      2. ``ED4ALL_BLOCK_BODY_CHAR_CEILING`` env — a GLOBAL override that still
         wins over the per-type default (preserves the historical env
         semantics: an env of 50 makes even an exposition concept overflow),
         then
      3. the per-block-TYPE budget (``_BLOCK_BODY_CHAR_CEILING_BY_TYPE``) —
         ~200 atomic micro-blocks, ~1000 developed exposition / answer-bearing
         types, ~1200 aggregating / structural roll-ups, then
      4. the 200-char atomic single-idea default (for any block_type not in
         the table).

    Garbage / non-positive values fall back to the next tier
    (parse-with-fallback, mirroring ``ED4ALL_ANSWER_NUM_CTX``).
    """
    if isinstance(override, int) and override > 0:
        return override
    raw = os.environ.get(BLOCK_BODY_CHAR_CEILING_ENV, "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    if isinstance(block_type, str):
        bt = block_type.strip().lower()
        if bt in _BLOCK_BODY_CHAR_CEILING_BY_TYPE:
            return _BLOCK_BODY_CHAR_CEILING_BY_TYPE[bt]
    return _DEFAULT_BODY_CHAR_CEILING


def block_attr(block: Any, key: str) -> Any:
    """Read ``block.<key>`` (dataclass) OR ``block[<key>]`` (dict).

    Mirrors ``lib.validators.alignment.verb_triple._block_attr`` so the IB6
    helpers work on both the frozen ``Block`` and the outline-tier dict.
    """
    if isinstance(block, Mapping):
        return block.get(key)
    if hasattr(block, key):
        return getattr(block, key)
    return None


def block_type_of(block: Any) -> str:
    """Resolve the lowercase ``block_type`` of a block (``""`` when absent)."""
    bt = block_attr(block, "block_type")
    if isinstance(bt, str):
        return bt.strip().lower()
    return ""


def framework_block_of(block: Any) -> Optional[str]:
    """Resolve the canonical B-code for a block via the IB2 catalog map.

    ``None`` for the non-pedagogical ``chrome`` scaffolding and any unknown
    block_type not in the catalog.
    """
    bt = block_type_of(block)
    if not bt:
        return None
    return framework_block_for(bt)


def is_interactive_block(block: Any) -> bool:
    """True iff the block's framework B-code is one of B07/B08/B10/B14."""
    return framework_block_of(block) in INTERACTIVE_FRAMEWORK_BLOCKS


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# IB6.4 escaped-provenance scrub (FIX 2 — escaped-tag measurement bug).
#
# Source-provenance markup is sometimes ENTITY-ESCAPED into a block's body
# (``&lt;aside data-cf-source-ids='dart:slug#b1'&gt;…&lt;/aside&gt;``). The
# plain ``_TAG_RE`` only strips literal ``<...>`` tags, so an escaped run is
# counted as visible body text and inflates the measured length (observed: a
# a real ``key_idea`` 667 → 2843 chars). We TARGET escaped-tag runs only — we do
# NOT blanket-unescape every entity, because a legitimately-escaped ``&amp;``
# or a math ``&lt;`` ("x &lt; 5") is real content and must keep its length.
#
# Rule:
#   1. Drop an escaped ``aside`` provenance element WHOLE — opening tag,
#      inner text, and closing tag — since the inner text is the provenance
#      payload, not body prose.
#   2. Drop any other escaped tag (``&lt;tag …&gt;`` / ``&lt;/tag&gt;``) whose
#      attributes carry a provenance token (``data-cf-source-ids`` /
#      ``data-cf-`` / ``data-dart-`` / a ``dart:`` CURIE), leaving its inner
#      text (which, for a span wrapper, IS body content).
# A token like "x &lt; 5" has no tag-name + attr shape, so it never matches.
_ESCAPED_ASIDE_PROVENANCE_RE = re.compile(
    r"&lt;\s*aside\b[^&]*?&gt;.*?&lt;\s*/\s*aside\s*&gt;",
    re.IGNORECASE | re.DOTALL,
)
_ESCAPED_PROVENANCE_TAG_RE = re.compile(
    r"&lt;\s*/?\s*[A-Za-z][\w-]*\b"  # an escaped opening/closing tag name …
    r"[^&]*?"                          # … its attributes (no entity inside) …
    r"(?:data-cf-source-ids|data-cf-|data-dart-|dart:)"  # … a provenance token
    r"[^&]*?&gt;",
    re.IGNORECASE,
)


def _scrub_escaped_provenance(s: str) -> str:
    """Remove entity-escaped provenance markup runs so they don't inflate body.

    Targeted (NOT a blanket ``html.unescape``): only escaped ``aside``
    provenance elements + escaped tags carrying a ``data-cf-*`` / ``data-dart-``
    / ``dart:`` token are dropped, preserving legitimately-escaped content
    entities (``&amp;``, math ``&lt;``).
    """
    if "&lt;" not in s:
        return s
    s = _ESCAPED_ASIDE_PROVENANCE_RE.sub(" ", s)
    s = _ESCAPED_PROVENANCE_TAG_RE.sub(" ", s)
    return s


def strip_html_text(s: Any) -> str:
    """Strip HTML tags + collapse whitespace to visible text.

    Mirrors ``lib.validators._assessment_helpers.placeholders._strip_html_text``
    but additionally collapses internal whitespace so the visible-character
    count is stable across markup-induced whitespace. Returns ``""`` for
    non-string / empty input.
    """
    if not isinstance(s, str) or not s:
        return ""
    no_tags = _TAG_RE.sub(" ", s)
    return _WS_RE.sub(" ", no_tags).strip()


def body_text_of(block: Any) -> str:
    """Resolve the BODY-slot visible text of a block.

    Resolution: the IB1 ``body`` anatomy slot does NOT exist as a separate
    field (``content`` IS the canonical body per IB1.2), so we prefer an
    explicit ``body`` key when one is present (outline-tier dict), then fall
    back to ``content``. A structured ``content`` payload (dict) contributes
    its stem / prompt / question / text / body strings. HTML is stripped.
    """
    parts: List[str] = []
    body = block_attr(block, "body")
    if isinstance(body, str) and body.strip():
        parts.append(body)
    else:
        content = block_attr(block, "content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
        elif isinstance(content, Mapping):
            for key in ("stem", "prompt", "question", "text", "body", "html"):
                val = content.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val)
    # FIX 2: drop entity-escaped provenance markup BEFORE stripping literal
    # tags so an escaped ``&lt;aside data-cf-source-ids=…&gt;…&lt;/aside&gt;``
    # run does not inflate the measured body length. Scoped to body_text_of so
    # strip_html_text's contract (used by interaction_feedback) is unchanged.
    joined = _scrub_escaped_provenance("\n".join(parts))
    return strip_html_text(joined)


_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
_BLOCK_ELEMENT_RE = re.compile(r"<(?:p|li|h[1-6]|tr|blockquote)\b", re.IGNORECASE)


def count_idea_chunks(block: Any) -> int:
    """Count the idea-chunks in a block's body (framework "~4 chunks" axis).

    Prefers an HTML block-element count (``<p>`` / ``<li>`` / ``<h*>`` / ``<tr>``
    / ``<blockquote>``) when the raw body carries markup — these are the
    framework's discrete idea-chunks. Falls back to a sentence count on the
    stripped text when no block elements are present (plain-prose body).
    Returns ``0`` for an empty body.
    """
    raw = block_attr(block, "body")
    if not (isinstance(raw, str) and raw.strip()):
        content = block_attr(block, "content")
        raw = content if isinstance(content, str) else ""
    if isinstance(raw, str) and raw.strip():
        elements = len(_BLOCK_ELEMENT_RE.findall(raw))
        if elements > 0:
            return elements
    text = body_text_of(block)
    if not text:
        return 0
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    return max(1, len(sentences))
