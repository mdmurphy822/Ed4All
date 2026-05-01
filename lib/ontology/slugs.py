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

__all__ = ["canonical_slug", "strip_lo_ref_suffix", "deslugify_concept"]


# Pre-compiled character classes.
# Matches Courseforge ``_slugify``'s regex exactly: any character that is not
# alphanumeric, whitespace, or hyphen. These are DELETED (replaced with "").
_SLUG_STRIP_DISALLOWED = re.compile(r"[^a-z0-9\s-]")

# Whitespace runs → single hyphen.
_SLUG_WS_COLLAPSE = re.compile(r"\s+")

# Wave 130d: trailing learning-objective-ref suffix on concept-tag slugs.
# Concept tags built from ``CO-NN`` / ``TO-NN`` LO refs land in chunks
# as ``property-paths-co-15`` / ``subqueries-to-03``; the pair-render
# pipeline used to ``.replace("-", " ")`` those slugs and bleed
# ``property paths co 15`` artifact tokens into prompt text. The
# pattern requires a trailing numeric LO code (1-3 digits), so
# legitimate slugs like ``map-to-existing-vocabularies``,
# ``pattern-to-remember``, ``attach-to-the-focus`` are untouched.
_LO_REF_SUFFIX_RE = re.compile(r"-(co|to)-\d{1,3}$", re.IGNORECASE)


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


def strip_lo_ref_suffix(slug: str) -> str:
    """Strip a trailing learning-objective-ref suffix from a concept slug.

    Concept tags built from ``CO-NN`` / ``TO-NN`` learning-objective refs
    land in chunks as ``property-paths-co-15`` / ``subqueries-to-03``; this
    helper removes the LO-ref suffix so downstream consumers (concept graph,
    prereq recap, audit script, deslugify) see the clean concept slug.

        >>> strip_lo_ref_suffix("property-paths-co-15")
        'property-paths'
        >>> strip_lo_ref_suffix("subqueries-to-03")
        'subqueries'
        >>> strip_lo_ref_suffix("map-to-existing-vocabularies")
        'map-to-existing-vocabularies'
        >>> strip_lo_ref_suffix("pattern-to-remember")
        'pattern-to-remember'
        >>> strip_lo_ref_suffix("Property-Paths-CO-15")
        'Property-Paths'
        >>> strip_lo_ref_suffix("")
        ''

    The pattern is anchored to end-of-string and requires a numeric
    suffix (1-3 digits), so legitimate ``-(co|to)-`` substrings without
    a trailing LO code are untouched.

    Args:
        slug: Concept slug; may carry a trailing ``-(co|to)-NN`` suffix.
            Falsy input (``""``, ``None``) returns ``""`` without raising.

    Returns:
        The slug with any trailing LO-ref suffix removed.
    """
    return _LO_REF_SUFFIX_RE.sub("", slug or "")


def deslugify_concept(slug: str) -> str:
    """Render a concept slug as human-readable text for prompt templates.

    Strips an LO-ref suffix (Wave 130d) before the hyphen-to-space
    transform so concept tags built from ``CO-NN`` / ``TO-NN``
    learning-objective refs don't bleed into prompts as ``co 15`` /
    ``to 03`` artifact tokens.

        >>> deslugify_concept("property-paths-co-15")
        'property paths'
        >>> deslugify_concept("subqueries-to-03")
        'subqueries'
        >>> deslugify_concept("rdf_type")
        'rdf type'
        >>> deslugify_concept("")
        ''

    Args:
        slug: Concept slug; may carry a trailing ``-(co|to)-NN`` suffix
            and/or hyphens / underscores. Falsy input returns ``""``.

    Returns:
        A human-readable phrase suitable for filling into prompt
        templates (e.g. ``"Explain {topic}"``).
    """
    cleaned = strip_lo_ref_suffix(slug)
    return cleaned.replace("-", " ").replace("_", " ")
