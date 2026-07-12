"""Running-header / page-number furniture sanitizer for textbook titles.

Motivation
==========

A scanned-textbook conversion pass extracts the document ``<title>`` / ``<h1>``
from OCR'd page imagery. On a running-header page the OCR fuses the chapter
title with the page's running header + page number + the first content word,
producing titles like::

    "Chapter 1 Foundations 83 ✓ Solution"
    "Chapter 3 Math Models 401 Question"

The clean chapter title is the prefix up to the stray page number
(``"Chapter 1 Foundations"`` / ``"Chapter 3 Math Models"``). That furniture
then propagates into TWO surfaces that poison downstream synthesis:

* the ``textbook_structure.json`` chapter ``headingText`` (noisy chapter node),
* every chunk's ``source.module_title`` / ``source.lesson_title`` provenance
  (144/185 chunks on a real 3-chapter scan carried the running-header title).

This module is a PURE, domain-agnostic sanitizer: it removes a trailing
page-number token (2-4 bare digits) plus any trailing OCR furniture words that
follow it, but ONLY when doing so leaves a plausible title behind. It never
touches a clean title (``"Chapter 2 Solving Linear Equations and
Inequalities"`` is returned verbatim — no bare page-number token to strip).

Contract
========

:func:`sanitize_running_header_title` is a pure ``str -> str`` function. It is
inert (byte-identical) unless the caller has opted in via
``SEMANTIK_TITLE_SANITIZE`` (checked by the call site through
:func:`title_sanitize_enabled`, NOT inside the pure function, so the function
stays testable without env mutation).

Conservative design (never mangle a real title):

* Only a STANDALONE 2-4 digit integer token counts as a page number. Section
  decimals (``1.2``), 1-digit chapter numbers, and 5+ digit numbers are never
  treated as page furniture.
* The page-number token must be preceded by >= ``_MIN_TITLE_WORDS`` alphabetic
  title words, and the surviving prefix must still carry >= ``_MIN_TITLE_WORDS``
  alphabetic words — otherwise the original is returned unchanged.
* The tail after the page number must be SHORT (<= ``_MAX_TAIL_WORDS`` words);
  a long tail signals the "number" is real content (e.g. a title that
  genuinely embeds a year + a subtitle), so the original is kept.
"""

from __future__ import annotations

import os
import re
from typing import Optional

__all__ = [
    "FLAG_ENV",
    "title_sanitize_enabled",
    "sanitize_running_header_title",
]

FLAG_ENV = "SEMANTIK_TITLE_SANITIZE"

#: Minimum alphabetic word tokens that must precede the page-number token AND
#: survive in the cleaned prefix. Below this the "title" is too thin to trust
#: the strip, so the original is returned unchanged.
_MIN_TITLE_WORDS = 2

#: Maximum word tokens allowed in the tail AFTER the page-number token. A long
#: tail means the number is probably real content, not a page-number furniture
#: seam — keep the original.
_MAX_TAIL_WORDS = 4

#: A standalone page-number token: exactly 2-4 digits, whole-token (not part of
#: a decimal / ISBN / longer number). ``\d{2,4}`` bounded by non-digit / dot.
_PAGE_NUM_RE = re.compile(r"(?<![\d.])\d{2,4}(?![\d.])")


def title_sanitize_enabled(env: Optional[dict] = None) -> bool:
    """Whether ``SEMANTIK_TITLE_SANITIZE`` is truthy (parse-with-fallback)."""
    src = env if env is not None else os.environ
    raw = src.get(FLAG_ENV)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _alpha_word_count(text: str) -> int:
    return sum(1 for tok in text.split() if any(c.isalpha() for c in tok))


def sanitize_running_header_title(title: Optional[str]) -> str:
    """Strip trailing running-header / page-number furniture from ``title``.

    Pure + deterministic. Returns the cleaned title, or the input verbatim when
    no unambiguous page-number seam is found (see module docstring for the
    conservative guards). ``None`` / empty -> ``""`` / input echoed.

    Examples::

        "Chapter 1 Foundations 83 ✓ Solution"  -> "Chapter 1 Foundations"
        "Chapter 3 Math Models 401 Question"   -> "Chapter 3 Math Models"
        "Chapter 2 Solving Linear Equations and Inequalities"  (unchanged)
        "1.2 Add Whole Numbers"                (unchanged — 1.2 is a decimal)
    """
    if not title:
        return title or ""
    stripped = title.strip()
    if not stripped:
        return stripped

    tokens = stripped.split()
    # Scan left-to-right for the FIRST standalone 2-4 digit page-number token
    # that sits after enough title words and leaves a short tail.
    for i, tok in enumerate(tokens):
        # Token must be EXACTLY a bare 2-4 digit number (strip surrounding
        # punctuation the split may have left attached, e.g. "83.").
        core = tok.strip(".,:;–-—")
        if not _PAGE_NUM_RE.fullmatch(core):
            continue
        prefix_tokens = tokens[:i]
        tail_tokens = tokens[i + 1:]
        prefix = " ".join(prefix_tokens)
        # Guard 1: enough real title words precede the number, AND survive it.
        if _alpha_word_count(prefix) < _MIN_TITLE_WORDS:
            continue
        # Guard 2: the tail after the page number must be short (furniture),
        # else the number is probably part of a real longer title.
        if len(tail_tokens) > _MAX_TAIL_WORDS:
            continue
        return prefix.strip()

    return stripped
