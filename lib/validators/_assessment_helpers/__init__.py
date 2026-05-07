"""W-D7 T7.4 — extracted helpers for :mod:`lib.validators.assessment`.

Public surface re-exported here so the canonical ``assessment`` module
can ``from lib.validators._assessment_helpers import *`` once instead
of cherry-picking. Cross-file consumers continue to import from
``lib.validators.assessment`` directly; this package is intentionally
private (leading underscore in the name).

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.4.
"""
from __future__ import annotations

from lib.validators._assessment_helpers.placeholders import (
    ASSESSMENT_PLACEHOLDER_PATTERNS,
    _CHAPTER_HEADING_RE,
    _INTEGER_TOKEN_RE,
    _TOC_THREE_INTS_RE,
    _looks_like_toc_fragment,
    _normalize_question_type,
    _strip_html_text,
)
from lib.validators._assessment_helpers.thresholds import (
    CORRECT_ANSWER_DIVERSITY_THRESHOLD,
    DISTRACTOR_TEMPLATE_MAX_RATIO,
    STEM_DIVERSITY_THRESHOLD,
    _DEFAULT_QUESTION_TYPE_THRESHOLDS,
    _PER_QUESTION_TYPE_THRESHOLDS,
    _resolve_per_type_thresholds,
    _thresholds_for_type,
)

__all__ = [
    "ASSESSMENT_PLACEHOLDER_PATTERNS",
    "CORRECT_ANSWER_DIVERSITY_THRESHOLD",
    "DISTRACTOR_TEMPLATE_MAX_RATIO",
    "STEM_DIVERSITY_THRESHOLD",
    "_CHAPTER_HEADING_RE",
    "_DEFAULT_QUESTION_TYPE_THRESHOLDS",
    "_INTEGER_TOKEN_RE",
    "_PER_QUESTION_TYPE_THRESHOLDS",
    "_TOC_THREE_INTS_RE",
    "_looks_like_toc_fragment",
    "_normalize_question_type",
    "_resolve_per_type_thresholds",
    "_strip_html_text",
    "_thresholds_for_type",
]
