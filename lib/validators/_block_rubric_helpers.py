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

# The framework's interactive block codes — B07 Knowledge-Check, B08 Guided
# Practice, B10 Discussion, B14 Graded Assessment. These are the codes the
# Feedback / interaction-presence gates apply to (the plan's B07/B08/B10/B14).
INTERACTIVE_FRAMEWORK_BLOCKS: frozenset = frozenset({"B07", "B08", "B10", "B14"})


def block_quality_rubric_enabled() -> bool:
    """True iff ``ED4ALL_BLOCK_QUALITY_RUBRIC`` is truthy (read each call)."""
    return os.environ.get(BLOCK_QUALITY_RUBRIC_ENV, "").strip().lower() in _TRUTHY


def resolve_body_char_ceiling(override: Optional[int] = None) -> int:
    """Resolve the D2 body char ceiling (arg > env > 200).

    Garbage / non-positive values fall back to the 200-char default
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
    return strip_html_text("\n".join(parts))


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
