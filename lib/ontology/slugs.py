"""Canonical slug function. Single source of truth for concept/key-term slugification.

Consolidates per-REC-ID-03 what were previously three independent implementations:

- ``Courseforge/scripts/generate_course.py::_slugify`` (the canonical reference)
- ``Trainforge/process_course.py::normalize_tag`` (wrapped; display-truncation
  + LibV2 alpha-first rule layered on top)
- ``Trainforge/rag/inference_rules/is_a_from_key_terms.py::_slugify`` (wrapped;
  SC-reference canonicalization + punctuation-to-space preprocessing layered on
  top)

The canonical algorithm matches Courseforge's ``_slugify`` byte-for-byte:

    1. Lowercase the input.
    2. Delete any character that is not ``[a-z0-9\\s-]`` (alnum, whitespace,
       hyphen). Note this DELETES — it does not replace with a separator. So
       ``"2.2"`` → ``"22"`` (digits fuse) rather than ``"2-2"``.
    3. Collapse runs of whitespace to a single hyphen.
    4. Strip leading/trailing hyphens.

The fuse-on-delete behavior is preserved because downstream LibV2 slugs already
depend on it (e.g. tag URLs and concept-graph node IDs emitted by Courseforge
through the existing ``_slugify``).

Rationale for NOT collapsing runs of hyphens: Courseforge's reference
implementation does not; preserving that matches the existing corpus exactly.
The ``is_a_from_key_terms`` caller that DID collapse multi-hyphens wraps
``canonical_slug`` and applies its own collapse after.

See ``plans/kg-quality-review-2026-04/worker-q-subplan.md`` for the byte-by-byte
migration trace and regression-test design.
"""

from __future__ import annotations

import re

__all__ = ["canonical_slug"]


# Pre-compiled character classes.
# Matches Courseforge ``_slugify``'s regex exactly: any character that is not
# alphanumeric, whitespace, or hyphen. These are DELETED (replaced with "").
_SLUG_STRIP_DISALLOWED = re.compile(r"[^a-z0-9\s-]")

# Whitespace runs → single hyphen.
_SLUG_WS_COLLAPSE = re.compile(r"\s+")


def canonical_slug(text: str) -> str:
    """Return the canonical kebab-case slug for ``text``.

    Byte-for-byte equivalent to the historical ``Courseforge.scripts.
    generate_course._slugify``:

        >>> canonical_slug("Cognitive Load Theory")
        'cognitive-load-theory'
        >>> canonical_slug("WCAG 2.2 AA")
        'wcag-22-aa'
        >>> canonical_slug("")
        ''
        >>> canonical_slug("!!!")
        ''
        >>> canonical_slug("-foo-bar-")
        'foo-bar'
        >>> canonical_slug("--a--b--")
        'a--b'

    Args:
        text: Free text to slugify. Falsy input (``""``, ``None``) returns
            ``""`` without raising.

    Returns:
        A lowercase kebab-case slug. May be empty if the input contained no
        alphanumeric content (or only disallowed characters that all get
        stripped).
    """
    if not text:
        return ""
    lowered = text.lower()
    stripped = _SLUG_STRIP_DISALLOWED.sub("", lowered)
    hyphenated = _SLUG_WS_COLLAPSE.sub("-", stripped)
    return hyphenated.strip("-")
