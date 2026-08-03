"""Placeholder regex patterns + question-type / TOC heuristics.

W-D7 T7.4: extracted from :mod:`lib.validators.assessment` so the
placeholder-detection regex set + the TOC-fragment heuristics live in
one auditable file. Cross-file consumers (``training_pair_promotion``,
``assessment_objective_alignment``) keep importing from
``lib.validators.assessment`` via the canonical's re-exports.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.4.
"""
from __future__ import annotations

import re
from typing import Any, Dict

__all__ = [
    "ASSESSMENT_PLACEHOLDER_PATTERNS",
    "_TOC_THREE_INTS_RE",
    "_CHAPTER_HEADING_RE",
    "_INTEGER_TOKEN_RE",
    "_strip_html_text",
    "_looks_like_toc_fragment",
    "_normalize_question_type",
]


ASSESSMENT_PLACEHOLDER_PATTERNS = [
    re.compile(r"Correct answer based on content", re.IGNORECASE),
    re.compile(r"Plausible distractor [A-C]", re.IGNORECASE),
    re.compile(r"Statement about .+ content\.", re.IGNORECASE),
    re.compile(r"The key concept from .+ is _______", re.IGNORECASE),
    re.compile(r"the concept from (?:LO-|INT|[A-Z]{2,})", re.IGNORECASE),
    re.compile(r"^Briefly \w+ the key points from ", re.IGNORECASE),
    re.compile(r"concepts from .+ and provide examples\.", re.IGNORECASE),
    re.compile(r"^concept term$", re.IGNORECASE),
    re.compile(r"Review content for objective ", re.IGNORECASE),
    re.compile(r"This statement is accurate based on ", re.IGNORECASE),
    re.compile(r"The correct term is found in .+ content", re.IGNORECASE),
    re.compile(r"A complete response should address all aspects of ", re.IGNORECASE),
    re.compile(r"Your response should cover the main concepts from ", re.IGNORECASE),
]

# TOC fragment: three standalone integers inline ("1.1 Something 14 1.7 ...").
# RETAINED as a re-export only (lib.validators.assessment and
# lib.validators._assessment_helpers re-export it). It is NO LONGER the
# discriminator -- it matched any string holding three integers, so every
# arithmetic correct answer ("5 + 6 + 2 + 5 = 18", "42 = 7 x 6, and 6 is a
# counting number") tripped TOC_FRAGMENT_ANSWER on a quantitative corpus.
# See _looks_like_toc_fragment for the structural rule that replaced it.
_TOC_THREE_INTS_RE = re.compile(r"\b\d+\b\s+\S+.*\b\d+\b.*\b\d+\b", re.DOTALL)
_CHAPTER_HEADING_RE = re.compile(r"\b\d+\.\d+\b")
_INTEGER_TOKEN_RE = re.compile(r"\b\d+\b")

# One table-of-contents entry: a number label, a *title run*, a page number.
#   "3 The Calvin Cycle 42"  /  "1.1 Structural changes in the economy 14"
# The title run is what discriminates a TOC from a computation: it starts
# with a capitalised word and carries no digits, because a TOC is prose
# titles interleaved with page numbers. An equation's inter-numeral text is
# operators and lowercase connectives, so it cannot form an entry.
_TOC_ENTRY_RE = re.compile(
    r"\b\d{1,3}(?:\.\d{1,3})*\.?\s+"   # entry label: 3 / 1.1 / 2.4.1
    r"[A-Z][^\d]{2,80}?\s+"            # title run: capitalised, digit-free
    r"\b\d{1,4}\b"                     # trailing page number
)
# A dotted *section label* ("1.1 Calvin"), as opposed to a bare decimal
# quantity ("3.14 cm") -- the trailing capital is the whole point.
_TOC_SECTION_LABEL_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\b\s+[A-Z]")
# Arithmetic / relational markers. A table of contents never holds an
# equation, so their presence rules a TOC fragment out outright.
_EQUATION_MARKER_RE = re.compile(r"[=+×÷≠≤≥→]|\d\s*/\s*\d")

# A genuine TOC fragment is *made of* entries; a sentence that happens to
# contain one is not. Entries must cover this share of the text.
_TOC_ENTRY_COVERAGE = 0.6
_TOC_MIN_ENTRIES = 2


def _normalize_question_type(q: Dict[str, Any]) -> str:
    """Resolve question_type, tolerating QTI-parsed ``type`` field.

    W4.B Lesson 1 (pair-schema field-name divergence):
    ``Trainforge/parsers/qti_parser.py:24`` uses ``type:`` while
    ``Trainforge/generators/assessment/generator.py:101`` uses
    ``question_type:``. Mirrors W4.B's ``lo_refs`` resolution chain
    (``pair.get("lo_refs") or pair.get("learning_outcome_refs")``).
    Generator surface always emits ``question_type``; QTI surface
    always emits ``type``. Day-1 only generator-emit reaches this
    validator; this helper is defense-in-depth against a future
    call site that routes QTI-parsed dicts directly.
    """
    if not isinstance(q, dict):
        return ""
    return str(q.get("question_type") or q.get("type") or "").lower()


def _strip_html_text(s: str) -> str:
    """Helper: strip HTML tags and normalize whitespace."""
    if not s:
        return ""
    return re.sub(r"<[^>]+>", "", s).strip()


def _looks_like_toc_fragment(answer_text: str) -> bool:
    """Return True if answer_text looks like a raw TOC fragment.

    A TOC fragment is *structural*: repeated ``<label> <Title> <page>``
    entries, i.e. prose titles interleaved with page numbers. Detection
    keys on that shape rather than on digit density, so a digit-dense
    computational answer is not mistaken for a contents page.

    Fires when, and only when, the text holds no equation markers AND
    either:
      - it is built from >= 2 TOC entries covering >= 60% of its
        characters, OR
      - it is > 500 chars with >= 3 integers and >= 2 dotted *section
        labels* (``1.1 Calvin``, not the decimal quantity ``3.14 cm``).

    The equation-marker veto is what keeps arithmetic out: a table of
    contents never contains ``=``, ``+``, ``x``, ``/`` between digits or
    an arrow, whereas a worked numeric answer nearly always does.
    """
    if not answer_text:
        return False
    text = _strip_html_text(answer_text)
    if _EQUATION_MARKER_RE.search(text):
        return False

    entries = list(_TOC_ENTRY_RE.finditer(text))
    if len(entries) >= _TOC_MIN_ENTRIES and text:
        covered = sum(m.end() - m.start() for m in entries)
        if covered / len(text) >= _TOC_ENTRY_COVERAGE:
            return True

    if len(text) > 500:
        int_count = len(_INTEGER_TOKEN_RE.findall(text))
        label_count = len(_TOC_SECTION_LABEL_RE.findall(text))
        if int_count >= 3 and label_count >= 2:
            return True
    return False
