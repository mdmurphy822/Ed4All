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
    Courseforge/scripts/generate_course.py ``generate_course`` — validates
        the course-level classification dict before writing
        ``course_metadata.json`` or any per-page JSON-LD. Fail-closed.
    Trainforge/process_course.py — loads the stub and merges it with CLI
        flag overrides. CLI flags, when present, override individual fields
        of the stub (not the whole block).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

__all__ = [
    "load_taxonomy",
    "get_valid_divisions",
    "get_valid_domains",
    "get_valid_subdomains",
    "get_valid_topics",
    "validate_classification",
]


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_TAXONOMY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "taxonomy.json"
)


@lru_cache(maxsize=1)
def load_taxonomy() -> Dict:
    """Load and cache ``schemas/taxonomies/taxonomy.json``.

    Raises:
        FileNotFoundError: schema is missing (Wave 1 Worker S migration
            should have published it).
        ValueError: schema shape is invalid (missing ``divisions`` root).
    """
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
