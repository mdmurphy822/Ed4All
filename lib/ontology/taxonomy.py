"""Subject-classification taxonomy loader (REC-TAX-01).

Loads ``schemas/taxonomies/taxonomy.json`` — the authoritative STEM / ARTS
hierarchy consolidated in Wave 1 (Worker S migration) — and exposes:

    load_taxonomy()          -> Dict  (raw JSON, cached)
    get_valid_divisions()    -> Set[str]
    get_valid_domains(...)   -> Set[str]
    get_valid_subdomains(...)-> Set[str]
    get_valid_topics(...)    -> Set[str]
    validate_classification(cls_dict) -> List[str]

Mirrors the pattern in ``lib/ontology/bloom.py`` (Wave 1 REC-BL-01):
``@lru_cache`` single-shot loader, frozen project-root schema path, defensive
copies on every getter so callers may mutate without polluting the cache.

The traversal logic is lifted from ``LibV2/tools/libv2/concept_vocabulary.py``
(``ConceptVocabulary._load_taxonomy`` at L112-136) but preserves the nested
hierarchy rather than flattening to a canonical-tag set. LibV2's flat view
is concerned with concept-tag normalization; this module's structured view
is concerned with classification-block validation at Courseforge emit time
and Trainforge consume time.

Downstream consumers:
    Courseforge/scripts/rendering/generate_course.py ``generate_course`` — validates
        the course-level classification dict before writing
        ``course_metadata.json`` or any per-page JSON-LD. Fail-closed.
    Trainforge/pipeline/process_course.py — loads the stub and merges it with CLI
        flag overrides. CLI flags, when present, override individual fields
        of the stub (not the whole block).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "load_taxonomy",
    "get_valid_divisions",
    "get_valid_domains",
    "get_valid_subdomains",
    "get_valid_topics",
    "validate_classification",
    # SemantiK pedagogical lexicon (Wave #22).
    "load_semantik_lexicon",
    "resolve_lexicon_profile",
    "get_lexicon_openers",
    "get_lexicon_apparatus_names",
    "get_lexicon_interior_apparatus_names",
    "get_lexicon_numbered_apparatus_names",
    "get_lexicon_apparatus_whitelist",
    "get_lexicon_confusables",
    "get_lexicon_subclasses",
    "DEFAULT_LEXICON_PROFILE",
    "LEGACY_PROFILE_ALIASES",
    "LEXICON_PROFILE_ENV",
    "strip_leading_ordinal",
]


# ---------------------------------------------------------------------------
# Shared shape utility — leading-ordinal strip (SemantiK structure-fidelity,
# Package 3). A printed section banner "1.4 Exercises" / "10.3 Review
# Exercises" carries a leading ``N`` / ``N.M`` / ``N.M.K`` ordinal that
# defeats an exact/prefix apparatus-name match. This single helper strips that
# ordinal so the apparatus/EOC lexicon matchers in the extractor
# (``_is_eoc_section_heading``), the SemantiK heading classifier
# (``lib/semantik/heading_classifier.py``), and the collapse re-segmenter
# (``lib/semantic_structure_extractor/resegment.py``) all see the bare name.
# Domain-agnostic pure-shape rule (no vocabulary) — the vocabulary stays in
# ``schemas/taxonomies/semantik_lexicon.json``.
# ---------------------------------------------------------------------------
import re as _re

_LEADING_ORDINAL_RE = _re.compile(r"^\s*\d+(?:\.\d+)*\s+")


def strip_leading_ordinal(text: str) -> str:
    """Strip a leading ``N`` / ``N.M`` / ``N.M.K`` ordinal + trailing space.

    "1.4 Exercises" -> "Exercises"; "10.3 Review Exercises" -> "Review
    Exercises". A heading with no leading ordinal ("Add and Subtract
    Integers") is returned unchanged. The trailing-space requirement means a
    bare decimal with no following word ("1.4") is left alone (nothing after
    the ordinal to keep), and an interior number ("Section 2 Foo") is not
    touched (anchored ``^``). Idempotent on already-stripped text.
    """
    if not text:
        return text
    return _LEADING_ORDINAL_RE.sub("", text, count=1)


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_TAXONOMY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "taxonomy.json"
)


_TAXONOMIES_DIR = _TAXONOMY_SCHEMA_PATH.parent


@lru_cache(maxsize=None)
def load_taxonomy(name: Optional[str] = None) -> Dict:
    """Load and cache a JSON taxonomy from ``schemas/taxonomies/``.

    Two modes:

    * ``name is None`` (default) — the canonical subject-classification
      taxonomy ``taxonomy.json``, validated for its ``divisions`` root. This
      is the historical, backward-compatible behavior every existing caller
      relies on.
    * ``name`` given — a GENERIC read of ``schemas/taxonomies/<name>.json``
      (raw JSON, no shape assumptions), so profile-organized lexicons
      (e.g. ``exercise_apparatus_lexicon``) are loadable through the one
      documented entry point (root CLAUDE.md § Canonical Helpers). ``name``
      must be a bare basename — a path separator raises ``ValueError``.

    Raises:
        FileNotFoundError: the requested taxonomy file is missing.
        ValueError: an invalid ``name``, or (default mode) a malformed
            ``divisions`` root.
    """
    if name is not None:
        if not str(name).strip() or "/" in str(name) or "\\" in str(name):
            raise ValueError(f"Invalid taxonomy name: {name!r}")
        stem = str(name).strip()
        if stem.endswith(".json"):
            stem = stem[: -len(".json")]
        path = _TAXONOMIES_DIR / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy not found at {path}.")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    if not _TAXONOMY_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Taxonomy schema not found at {_TAXONOMY_SCHEMA_PATH}. "
            "Expected canonical copy from Wave 1 (Worker S)."
        )
    with open(_TAXONOMY_SCHEMA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if "divisions" not in data or not isinstance(data["divisions"], dict):
        raise ValueError(
            f"Malformed taxonomy schema at {_TAXONOMY_SCHEMA_PATH}: "
            "missing or non-dict 'divisions' root."
        )
    return data


# ---------------------------------------------------------------------------
# Public getters
# ---------------------------------------------------------------------------


def get_valid_divisions() -> Set[str]:
    """Return the set of valid division slugs, e.g. ``{'STEM', 'ARTS'}``.

    Defensive copy — mutation does not pollute the cache.
    """
    return set(load_taxonomy().get("divisions", {}).keys())


def get_valid_domains(division: str) -> Set[str]:
    """Return the set of valid ``primary_domain`` slugs under ``division``.

    Returns the empty set when ``division`` is not recognized (callers that
    want fail-closed behavior use :func:`validate_classification`).
    """
    divisions = load_taxonomy().get("divisions", {})
    division_data = divisions.get(division)
    if not isinstance(division_data, dict):
        return set()
    return set(division_data.get("domains", {}).keys())


def get_valid_subdomains(division: str, domain: str) -> Set[str]:
    """Return the set of valid subdomain slugs under ``(division, domain)``.

    Returns the empty set when either level is unrecognized.
    """
    divisions = load_taxonomy().get("divisions", {})
    division_data = divisions.get(division)
    if not isinstance(division_data, dict):
        return set()
    domain_data = division_data.get("domains", {}).get(domain)
    if not isinstance(domain_data, dict):
        return set()
    return set(domain_data.get("subdomains", {}).keys())


def get_valid_topics(division: str, domain: str, subdomain: str) -> Set[str]:
    """Return the set of valid topic slugs under a full path.

    Returns the empty set when any level is unrecognized. Topics in
    ``taxonomy.json`` are stored as lists (not dicts with children), so the
    list contents are the authoritative slugs.
    """
    divisions = load_taxonomy().get("divisions", {})
    division_data = divisions.get(division)
    if not isinstance(division_data, dict):
        return set()
    domain_data = division_data.get("domains", {}).get(domain)
    if not isinstance(domain_data, dict):
        return set()
    subdomain_data = domain_data.get("subdomains", {}).get(subdomain)
    if not isinstance(subdomain_data, dict):
        return set()
    topics = subdomain_data.get("topics", [])
    if not isinstance(topics, list):
        return set()
    return set(topics)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SemantiK pedagogical lexicon (Wave #22 — tiers 1-2 semantic enrichment).
#
# The opener / apparatus / confusable vocabularies the SemantiK adapter seam
# consumes live in ``schemas/taxonomies/semantik_lexicon.json`` as PROFILES so a
# new corpus is onboarded by a lexicon entry, never a code edit (lexicon
# profiles, not corpus-specific code). Profile selection is env-driven
# (``SEMANTIK_LEXICON_PROFILE``); the default is the ``generic-academic`` base
# overlaid by the ``open-textbook`` overlay. The loader is a single-shot ``lru_cache``
# read (mirrors :func:`load_taxonomy`); the per-profile getters take a resolved
# profile spec so a caller can cache the merged view at module import.
# ---------------------------------------------------------------------------

_SEMANTIK_LEXICON_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "semantik_lexicon.json"
)

#: Env var selecting the active lexicon profile spec (``+``-joined profile keys,
#: merged left-to-right, later profile overlays earlier). Default combines the
#: generic-academic base with the open-textbook overlay so a scanned open-textbook
#: scan corpus (TRY IT / BE PREPARED / trvit) keeps working out of the box.
LEXICON_PROFILE_ENV = "SEMANTIK_LEXICON_PROFILE"
DEFAULT_LEXICON_PROFILE = "generic-academic+open-textbook"

#: legacy profile-key aliases (pre-rename compat) — old publisher-named keys
#: map silently onto the neutral keys wherever a ``+``-separated spec segment
#: is resolved, so pre-rename operator env settings keep working.
LEGACY_PROFILE_ALIASES = {
    "openstax": "open-textbook",
    "openstax-shaped": "textbook-shaped",
}


@lru_cache(maxsize=1)
def load_semantik_lexicon() -> Dict:
    """Load and cache ``schemas/taxonomies/semantik_lexicon.json``.

    Raises:
        FileNotFoundError: the lexicon schema is missing.
        ValueError: the schema shape is invalid (missing ``profiles`` root).
    """
    if not _SEMANTIK_LEXICON_PATH.exists():
        raise FileNotFoundError(
            f"SemantiK lexicon not found at {_SEMANTIK_LEXICON_PATH}. "
            "Expected the Wave #22 pedagogical-lexicon taxonomy."
        )
    with open(_SEMANTIK_LEXICON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if "profiles" not in data or not isinstance(data["profiles"], dict):
        raise ValueError(
            f"Malformed SemantiK lexicon at {_SEMANTIK_LEXICON_PATH}: "
            "missing or non-dict 'profiles' root."
        )
    return data


def resolve_lexicon_profile(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve the active lexicon profile spec (``SEMANTIK_LEXICON_PROFILE``).

    Returns the raw ``+``-joined spec string (default
    :data:`DEFAULT_LEXICON_PROFILE`). An unknown / empty value falls back to the
    default rather than yielding an empty vocabulary.
    """
    src = env if env is not None else os.environ
    val = (src.get(LEXICON_PROFILE_ENV) or "").strip()
    return val or DEFAULT_LEXICON_PROFILE


def _profile_keys(profile_spec: str) -> List[str]:
    """Split a ``+``-joined profile spec into known profile keys (in order).

    Each segment is normalized through :data:`LEGACY_PROFILE_ALIASES` so a
    pre-rename spec resolves to the same lexicon. Unknown profile keys are
    skipped (defensive — a typo yields the remaining known profiles rather
    than a hard failure). An empty result falls back to the default spec's
    keys so a caller never gets a silently-empty vocabulary.
    """
    lex = load_semantik_lexicon()
    known = lex.get("profiles", {})
    keys = [k.strip() for k in (profile_spec or "").split("+") if k.strip()]
    keys = [LEGACY_PROFILE_ALIASES.get(k, k) for k in keys]
    keys = [k for k in keys if k in known]
    if not keys:
        keys = [
            k.strip()
            for k in DEFAULT_LEXICON_PROFILE.split("+")
            if k.strip() in known
        ]
    return keys


def _merged_profile_field(profile_spec: str, field: str) -> List:
    """Concatenate one list-valued field across the merged profiles, in order."""
    lex = load_semantik_lexicon()
    profiles = lex.get("profiles", {})
    out: List = []
    for key in _profile_keys(profile_spec):
        out.extend(profiles.get(key, {}).get(field, []) or [])
    return out


def get_lexicon_openers(
    profile_spec: Optional[str] = None,
) -> Tuple[Dict, ...]:
    """Return the merged opener specs for ``profile_spec``, ordered by ``order``.

    Each entry is a dict ``{role, pattern, display, numbered, interior_split,
    association_role, order}``. Ordering by the explicit ``order`` key
    reproduces the canonical ``opener_classifier._OPENERS`` sequence so the
    refactor is behavior-preserving. ``profile_spec`` defaults to the resolved
    env profile.
    """
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    openers = _merged_profile_field(spec, "openers")
    return tuple(sorted(openers, key=lambda o: o.get("order", 1_000)))


def get_lexicon_apparatus_names(
    profile_spec: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return the merged apparatus-section display names for ``profile_spec``.

    Concatenation order across merged profiles reproduces
    ``heading_classifier.APPARATUS_HEADING_NAMES``.
    """
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    return tuple(
        s["display"]
        for s in _merged_profile_field(spec, "apparatus_sections")
        if s.get("display")
    )


def get_lexicon_interior_apparatus_names(
    profile_spec: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return the ALL-CAPS interior-banner apparatus names for ``profile_spec``.

    The ``interior_banner``-flagged subset (uppercased) reproduces
    ``heading_classifier._INTERIOR_APPARATUS_NAMES``.
    """
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    return tuple(
        s["display"].upper()
        for s in _merged_profile_field(spec, "apparatus_sections")
        if s.get("display") and s.get("interior_banner")
    )


def get_lexicon_numbered_apparatus_names(
    profile_spec: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return the BARE apparatus words that are apparatus ONLY when numbered.

    Merged ``numbered_apparatus_names`` (a flat list of display strings, e.g.
    ``"Exercises"``) across the active profiles. Consumed ONLY by
    ``heading_classifier.is_numbered_apparatus_heading`` (the ordinal-prefixed
    banner demotion) — never by the standalone ``is_apparatus_heading``, so a
    bare unnumbered ``"Exercises"`` is NOT demoted (preserving the W1 builder's
    deliberate choice to keep bare "Exercises" out of ``apparatus_sections``).
    Empty for profiles that declare no such list.
    """
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    return tuple(
        str(s).strip()
        for s in _merged_profile_field(spec, "numbered_apparatus_names")
        if str(s).strip()
    )


def get_lexicon_apparatus_whitelist(
    profile_spec: Optional[str] = None,
) -> frozenset:
    """Return the merged short-heading apparatus whitelist for ``profile_spec``."""
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    return frozenset(
        str(w).strip().lower()
        for w in _merged_profile_field(spec, "apparatus_whitelist")
        if str(w).strip()
    )


def get_lexicon_confusables(
    profile_spec: Optional[str] = None,
) -> Tuple[Dict, ...]:
    """Return the merged OCR-confusable ``{pattern, canonical}`` entries."""
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    return tuple(_merged_profile_field(spec, "confusables"))


def get_lexicon_subclasses(
    profile_spec: Optional[str] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Return the merged composite-unit SUBCLASS seed vocabulary (Build #23).

    ``subclasses`` is a per-profile ``{unit_type: [label, ...]}`` dict (the
    Tier-3 seed vocabulary the model-assisted subclassifier picks from). Unlike
    the other lexicon fields (flat list concatenation), this is a dict-of-lists,
    so the merge is per-KEY: for each unit type present in any merged profile,
    the label lists are concatenated left-to-right and de-duplicated first-seen
    (so an overlay profile can EXTEND ``worked_example``'s seeds without
    displacing the generic-academic base). Returns a plain dict mapping each
    unit type to an ordered de-duplicated tuple of kebab-case seed labels.
    ``profile_spec`` defaults to the resolved env profile.
    """
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    lex = load_semantik_lexicon()
    profiles = lex.get("profiles", {})
    merged: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}
    for key in _profile_keys(spec):
        sub = profiles.get(key, {}).get("subclasses", {}) or {}
        if not isinstance(sub, dict):
            continue
        for unit_type, labels in sub.items():
            bucket = merged.setdefault(unit_type, [])
            seen_set = seen.setdefault(unit_type, set())
            for label in labels or []:
                lab = str(label).strip()
                if lab and lab not in seen_set:
                    seen_set.add(lab)
                    bucket.append(lab)
    return {k: tuple(v) for k, v in merged.items()}


def get_lexicon_subclass_glosses(
    profile_spec: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Return per-label GLOSSES for the subclass seed vocabulary.

    The lexicon ``subclasses`` value per unit type is either the legacy
    ``[label, ...]`` list (no glosses → empty strings) or a
    ``{label: gloss}`` mapping (the gloss is a one-line discriminative
    definition rendered into the subclassifier prompt). Merge mirrors
    :func:`get_lexicon_subclasses` — per unit type, left-to-right across
    the profile spec, first profile to define a label's gloss wins.
    """
    spec = profile_spec if profile_spec is not None else resolve_lexicon_profile()
    lex = load_semantik_lexicon()
    profiles = lex.get("profiles", {})
    merged: Dict[str, Dict[str, str]] = {}
    for key in _profile_keys(spec):
        sub = profiles.get(key, {}).get("subclasses", {}) or {}
        if not isinstance(sub, dict):
            continue
        for unit_type, labels in sub.items():
            bucket = merged.setdefault(unit_type, {})
            if isinstance(labels, dict):
                items = [(str(k).strip(), str(v or "").strip()) for k, v in labels.items()]
            else:
                items = [(str(label).strip(), "") for label in labels or []]
            for lab, gloss in items:
                if lab and lab not in bucket:
                    bucket[lab] = gloss
    return merged


def validate_classification(classification: Optional[Dict]) -> List[str]:
    """Validate a classification block against the authoritative taxonomy.

    Returns the empty list on success. A non-empty list contains one
    human-readable error message per violation. Callers use the list
    length as a pass/fail signal and the messages for logging.

    Checks:
        * ``division`` present and recognized.
        * ``primary_domain`` present and belongs to the declared division.
        * Every ``subdomains[]`` entry belongs to the declared domain.
        * Every ``topics[]`` entry belongs to at least one declared
          subdomain (or, when ``subdomains`` is empty, to any subdomain
          under the domain).

    ``classification`` == ``None`` or empty dict returns an error list so
    callers can treat "empty classification" as an explicit failure when
    they expect a populated block. Callers that want to skip validation
    for the absent-classification case should check before calling.
    """
    errors: List[str] = []

    if classification is None:
        return ["classification is None"]
    if not isinstance(classification, dict):
        return [f"classification must be a dict, got {type(classification).__name__}"]
    if not classification:
        return ["classification is empty"]

    # Division
    division = classification.get("division")
    if not division:
        errors.append("classification missing required field 'division'")
        # Can't validate further without a division.
        return errors

    valid_divisions = get_valid_divisions()
    if division not in valid_divisions:
        errors.append(
            f"division '{division}' not in valid divisions "
            f"{sorted(valid_divisions)}"
        )
        return errors  # Downstream checks require a valid division.

    # Primary domain
    primary_domain = classification.get("primary_domain")
    if not primary_domain:
        errors.append(
            "classification missing required field 'primary_domain'"
        )
        return errors

    valid_domains = get_valid_domains(division)
    if primary_domain not in valid_domains:
        errors.append(
            f"primary_domain '{primary_domain}' not valid under division "
            f"'{division}' (valid: {sorted(valid_domains)})"
        )
        return errors

    # Subdomains (optional)
    subdomains = classification.get("subdomains", []) or []
    if not isinstance(subdomains, list):
        errors.append(
            f"subdomains must be a list, got {type(subdomains).__name__}"
        )
        subdomains = []
    else:
        valid_subs = get_valid_subdomains(division, primary_domain)
        for sd in subdomains:
            if sd not in valid_subs:
                errors.append(
                    f"subdomain '{sd}' not valid under "
                    f"{division}/{primary_domain} (valid: {sorted(valid_subs)})"
                )

    # Topics (optional). Accept a topic if it belongs to any declared
    # subdomain, OR (when no subdomains were declared) to any subdomain
    # under the domain.
    topics = classification.get("topics", []) or []
    if not isinstance(topics, list):
        errors.append(
            f"topics must be a list, got {type(topics).__name__}"
        )
        topics = []
    else:
        candidate_subs = (
            [s for s in subdomains if s in get_valid_subdomains(division, primary_domain)]
            if subdomains
            else list(get_valid_subdomains(division, primary_domain))
        )
        allowed_topics: Set[str] = set()
        for sd in candidate_subs:
            allowed_topics |= get_valid_topics(division, primary_domain, sd)
        for topic in topics:
            if topic not in allowed_topics:
                errors.append(
                    f"topic '{topic}' not valid under "
                    f"{division}/{primary_domain} "
                    f"(subdomains={candidate_subs or '[]'})"
                )

    return errors
