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

# IB6.4 D2 body ceiling (chars). Default 200 ("~200char single idea" target).
BLOCK_BODY_CHAR_CEILING_ENV = "ED4ALL_BLOCK_BODY_CHAR_CEILING"
_DEFAULT_BODY_CHAR_CEILING = 200

# IB6.4 per-block-TYPE body-char budget (FIX 2 — type-blind-ceiling fix).
#
# The single global 200-char ceiling was structurally unsatisfiable for the
# exposition / answer-bearing block types: across two real corpora only 1 of 99
# non-exempt blocks measured <=200 chars (alg median body 794, demo 1216). The
# 200 target is the right ATOMIC single-idea budget (key_idea / callout / vocab
# card / a one-line check), but exposition that legitimately develops one idea
# across a worked example, a diagram long-description, or a scenario needs a
# higher budget. The higher-budget set below is the gap-map's "exposition /
# answer-bearing" types; the ~1000 value sits at p50-p75 of the measured bodies
# (alg median 794 / demo 1216). The block_catalog.yaml carries no char-budget
# field, so this constant table is the single source of truth.
#
# The orthogonal ">4-idea-chunk" axis (content.py::_BLOCK_IDEA_CHUNK_CEILING)
# is UNTOUCHED — it still catches genuine multi-idea "everything-blocks"
# regardless of the per-type char budget.
_EXPOSITION_BODY_CHAR_CEILING = 1000
_BLOCK_BODY_CHAR_CEILING_BY_TYPE: dict = {
    "concept": _EXPOSITION_BODY_CHAR_CEILING,
    "example": _EXPOSITION_BODY_CHAR_CEILING,
    "worked_example": _EXPOSITION_BODY_CHAR_CEILING,
    "self_check_question": _EXPOSITION_BODY_CHAR_CEILING,
    "diagram": _EXPOSITION_BODY_CHAR_CEILING,
    "scenario": _EXPOSITION_BODY_CHAR_CEILING,
}

# The framework's interactive block codes — B07 Knowledge-Check, B08 Guided
# Practice, B10 Discussion, B14 Graded Assessment. These are the codes the
# Feedback / interaction-presence gates apply to (the plan's B07/B08/B10/B14).
INTERACTIVE_FRAMEWORK_BLOCKS: frozenset = frozenset({"B07", "B08", "B10", "B14"})


def block_quality_rubric_enabled() -> bool:
    """True iff ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is truthy (read each call)."""
    return os.environ.get(BLOCK_QUALITY_RUBRIC_ENV, "").strip().lower() in _TRUTHY


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
      3. the per-block-TYPE budget (``_BLOCK_BODY_CHAR_CEILING_BY_TYPE``) for
         the exposition / answer-bearing types, then
      4. the 200-char atomic single-idea default.

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
# demo ``key_idea`` 667 → 2843 chars). We TARGET escaped-tag runs only — we do
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
