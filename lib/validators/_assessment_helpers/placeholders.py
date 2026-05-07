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

# TOC fragment: three standalone integers inline ("1.1 Something 14 1.7 ...")
_TOC_THREE_INTS_RE = re.compile(r"\b\d+\b\s+\S+.*\b\d+\b.*\b\d+\b", re.DOTALL)
_CHAPTER_HEADING_RE = re.compile(r"\b\d+\.\d+\b")
_INTEGER_TOKEN_RE = re.compile(r"\b\d+\b")


def _normalize_question_type(q: Dict[str, Any]) -> str:
    """Resolve question_type, tolerating QTI-parsed ``type`` field.

    W4.B Lesson 1 (pair-schema field-name divergence):
    ``Trainforge/parsers/qti_parser.py:24`` uses ``type:`` while
    ``Trainforge/generators/assessment_generator.py:101`` uses
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

    Matches when the string contains either:
      - Three standalone integers inline (page numbers), OR
      - Is > 500 chars AND has >= 3 integers AND >= 2 dotted-numeric
        headings like ``1.1`` / ``4.2``.
    """
    if not answer_text:
        return False
    text = _strip_html_text(answer_text)
    if _TOC_THREE_INTS_RE.search(text):
        return True
    if len(text) > 500:
        int_count = len(_INTEGER_TOKEN_RE.findall(text))
        heading_count = len(_CHAPTER_HEADING_RE.findall(text))
        if int_count >= 3 and heading_count >= 2:
            return True
    return False
