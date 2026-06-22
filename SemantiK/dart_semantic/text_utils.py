"""Shared text-processing helpers used across ingesters and data builders.

Centralized here so every site that matches OCR text against HTML
ground-truth tags uses the same normalization rules. Divergent
implementations would cause silent training-label drift.
"""
from __future__ import annotations

import re


_NON_ALNUM_WHITESPACE = re.compile(r"[^a-z0-9\s]")


def normalize_for_match(s: str) -> str:
    """Lowercase + strip non-alphanumeric/whitespace to prepare for token
    comparison. Preserves word boundaries by replacing punctuation with
    spaces rather than removing."""
    return _NON_ALNUM_WHITESPACE.sub(" ", s.lower())


def jaccard_overlap(a: str, b: str) -> float:
    """Jaccard similarity over whitespace-split tokens after normalization.

    Returns 0.0 when either side has no tokens — callers should treat
    that as 'no confident match' and reject.
    """
    ta = set(normalize_for_match(a).split())
    tb = set(normalize_for_match(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
