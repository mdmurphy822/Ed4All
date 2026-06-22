"""Heading hierarchy normalization (Plans/04 §1.1).

Rules:

  1. **Promote-the-first**: the first heading in document order becomes
     ``<h1>`` regardless of its raw level.
  2. **Demote-forward**: a heading that would skip a level (e.g.
     ``h1`` → ``h3``) is demoted to ``last + 1``.
  3. **Never-skip**: contiguous heading levels — emitted level may
     never exceed the previously-emitted level by more than 1.

The algorithm mirrors ``dart_semantic/ir.py:normalize_heading_levels``
but operates on a flat list of integers (extracted from the per-region
candidate HTML's ``<hN>`` tag) rather than an IR tree.

Example::

    >>> normalize_heading_levels([3, 4])
    [1, 2]
    >>> normalize_heading_levels([1, 4])
    [1, 2]
    >>> normalize_heading_levels([1, 2, 5])
    [1, 2, 3]
    >>> normalize_heading_levels([])
    []

Heading IDs (WCAG 2.4.5 anchor support):

    >>> assign_heading_ids(["Intro", "Methods", "Intro"])
    ['intro', 'methods', 'intro-2']
    >>> assign_heading_ids(["", "  ?? "])
    ['section-1', 'section-2']
"""
from __future__ import annotations

import re


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """ASCII kebab-case slug; empty if input has no alphanumerics."""
    if not text:
        return ""
    # Drop non-ASCII (emoji, smart quotes, etc.) before lowercasing.
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_RE.sub("-", ascii_text).strip("-")
    return slug


def assign_heading_ids(heading_texts: list[str]) -> list[str]:
    """Generate stable, deduped heading IDs in document order.

    Empty/punctuation-only headings fall back to ``section-{N}`` where
    N is the 1-based ordinal of the heading. Collisions get ``-2``,
    ``-3``, ... suffixes by first-occurrence order.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for i, text in enumerate(heading_texts, start=1):
        base = slugify(text) or f"section-{i}"
        count = seen.get(base, 0)
        if count == 0:
            ident = base
        else:
            ident = f"{base}-{count + 1}"
        seen[base] = count + 1
        # Guard: a synthesized "{base}-N" can collide with an explicit
        # heading text that already produced that exact slug earlier.
        while ident in seen and ident != base:
            count += 1
            ident = f"{base}-{count + 1}"
            seen[base] = count + 1
        seen.setdefault(ident, 1)
        out.append(ident)
    return out


def normalize_heading_levels(heading_levels: list[int]) -> list[int]:
    """Take raw heading levels and return promote/demote/never-skip output.

    The output list has the same length as the input.
    """
    if not heading_levels:
        return []
    out: list[int] = []
    last = 0  # 0 = before any heading
    for raw in heading_levels:
        if last == 0:
            out.append(1)
            last = 1
            continue
        target = max(1, min(6, int(raw)))
        if target > last + 1:
            target = last + 1
        out.append(target)
        last = target
    return out


__all__ = ["assign_heading_ids", "normalize_heading_levels", "slugify"]
