"""Shared objective-filler lexicon (objective-synthesis fix W2, Defect B + E).

A learning-objective statement can be grammatically valid yet carry NO teachable
skill once the Bloom verb and generic scaffolding are removed — "Apply various
techniques to solve real-world problems" reduces to nothing nameable. The
DOMAIN-AGNOSTIC "filler" vocabulary that produces this vacuity (hedge quantifiers,
empty nouns, evaluative adjectives) lives in
``schemas/taxonomies/objective_filler_lexicon.json`` as PROFILES (publisher /
subject vocabulary is DATA, not code, so a new corpus is onboarded by adding a
profile rather than editing Python) and is loaded here through the
one documented entry point ``lib.ontology.taxonomy.load_taxonomy``.

This module is the SINGLE shared helper the plan requires both Defect B and
Defect E to import (so the same filler subtraction is applied in exactly one
place):

* **Defect E** — ``lib/objectives/objective_dedup.py::_skill_signature`` subtracts
  :func:`filler_tokens` from a cluster representative's keyphrase signature before
  the cross-window lexical-dedup Jaccard, so "understand various key concepts" and
  "understand the important concepts" collapse on their (empty) real residual
  rather than being kept apart by filler noise.
* **Defect B** — ``lib/validators/objective_specificity.py`` subtracts
  :func:`filler_tokens` from every CO statement's content-residual (V1 vacuity,
  V3 source-anchoring recall) and keys its V2 generic-object check on
  :func:`vague_object_regexes`.

Placement rationale (no import cycle): this module imports ONLY
``lib.ontology.taxonomy`` (stdlib + JSON). ``objective_dedup`` and the validator
both already sit above ``lib.ontology`` in the import graph, so both can import
from here freely. Nothing here is gated by an env flag — it is inert vocabulary +
pure helpers; the behavior-changing wiring lives at the two call sites behind
their own flags (``ED4ALL_OBJECTIVE_DEDUP_LEXICAL`` / ``ED4ALL_OBJECTIVE_SPECIFICITY``).

Graceful degrade: if the lexicon file is missing or malformed the loaders return
EMPTY collections (filler subtraction becomes a no-op; the validator's V1/V3 fall
back to bloom+stopword residual only) rather than raising — a missing lexicon must
never break the dedup pass or the gate.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "TAXONOMY_NAME",
    "filler_tokens",
    "vague_object_regexes",
    "has_vague_object",
]

#: Taxonomy basename loaded via :func:`lib.ontology.taxonomy.load_taxonomy`.
TAXONOMY_NAME = "objective_filler_lexicon"


def _select_profiles(lex: dict, profile_spec: Optional[str]) -> List[str]:
    """Resolve a ``+``-joined profile spec to known profile keys, in order.

    ``None`` / ``"*"`` / empty → EVERY profile (the default union — filler
    over-match is safe because subtracting a token can only REVEAL a real skill,
    never hide one). Unknown keys are skipped; an all-unknown spec falls back to
    the full union so a caller never gets a silently-empty lexicon.
    """
    known = list((lex.get("profiles") or {}).keys())
    if profile_spec is None or not str(profile_spec).strip() or profile_spec == "*":
        return known
    keys = [k.strip() for k in str(profile_spec).split("+") if k.strip()]
    keys = [k for k in keys if k in known]
    return keys or known


@lru_cache(maxsize=None)
def filler_tokens(profile_spec: Optional[str] = None) -> frozenset:
    """Return the union of DOMAIN-AGNOSTIC filler tokens for ``profile_spec``.

    Lowercase single tokens (``various`` / ``techniques`` / ``concepts`` / …).
    Cached per spec. Default (``None``) unions every profile. A missing or
    malformed lexicon → EMPTY frozenset (subtraction becomes a no-op — the
    filler pass must never break its callers).
    """
    try:
        from lib.ontology.taxonomy import load_taxonomy  # noqa: PLC0415

        lex = load_taxonomy(TAXONOMY_NAME)
    except Exception as exc:  # noqa: BLE001 — missing/malformed lexicon degrades
        logger.debug(
            "objective_filler_lexicon: load failed (%s); filler subtraction "
            "disabled (empty token set).",
            exc,
        )
        return frozenset()

    profiles = lex.get("profiles") or {}
    out: set = set()
    for key in _select_profiles(lex, profile_spec):
        prof = profiles.get(key) or {}
        for tok in prof.get("filler_tokens") or []:
            norm = str(tok).strip().lower()
            if norm:
                out.add(norm)
    return frozenset(out)


def _phrase_to_regex(phrase: str) -> re.Pattern:
    """Compile a vague-object phrase to a case-insensitive regex.

    Word-internal whitespace becomes ``\\s+`` so a phrase still matches across an
    OCR line-break / doubled space; a hyphen matches an optional hyphen-or-space
    ("real-world" also matches "real world"). Deliberately NOT word-boundary
    anchored — over-match is safe (V2 only fires when the content residual is ALSO
    thin, so a phrase hit alone never flags a specific objective).
    """
    raw = str(phrase).strip()
    if not raw:
        return re.compile(r"(?!x)x")  # never-matches sentinel
    # Split on whitespace and hyphens; join with a flexible separator.
    tokens = [re.escape(tok) for tok in re.split(r"[\s\-]+", raw) if tok]
    if not tokens:
        return re.compile(r"(?!x)x")
    return re.compile(r"[\s\-]+".join(tokens), re.IGNORECASE)


@lru_cache(maxsize=None)
def vague_object_regexes(profile_spec: Optional[str] = None) -> Tuple[re.Pattern, ...]:
    """Return the compiled vague-object phrase regexes for ``profile_spec``.

    Cached per spec. Default unions every profile. Missing / malformed lexicon →
    EMPTY tuple (the V2 generic-object check becomes a no-op).
    """
    try:
        from lib.ontology.taxonomy import load_taxonomy  # noqa: PLC0415

        lex = load_taxonomy(TAXONOMY_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "objective_filler_lexicon: load failed (%s); vague-object check "
            "disabled (empty pattern set).",
            exc,
        )
        return ()

    profiles = lex.get("profiles") or {}
    seen: set = set()
    regexes: List[re.Pattern] = []
    for key in _select_profiles(lex, profile_spec):
        prof = profiles.get(key) or {}
        for phrase in prof.get("vague_object_patterns") or []:
            norm = str(phrase).strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            regexes.append(_phrase_to_regex(phrase))
    return tuple(regexes)


def has_vague_object(text: str, profile_spec: Optional[str] = None) -> bool:
    """Whether ``text`` contains ANY compiled vague-object phrase."""
    if not text:
        return False
    return any(rgx.search(text) for rgx in vague_object_regexes(profile_spec))
