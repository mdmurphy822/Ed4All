"""Bloom's Taxonomy loader.

Loads `schemas/taxonomies/bloom_verbs.json` (the authoritative 60-verb /
6-level canonical list, structured as `properties.{level}.default` arrays
of `{verb, usage_context, example_template}` dicts) and exposes multiple
shapes over that single source of truth:

    BLOOM_LEVELS        -- immutable tuple of the six level names in
                           pedagogical order (low -> high complexity).
    BloomVerb           -- frozen dataclass capturing a single verb entry.
    get_verbs()         -- Dict[str, Set[str]]    (for validators that
                           treat verb presence as membership).
    get_verbs_list()    -- Dict[str, List[str]]   (for parsers that iterate
                           and need deterministic order).
    get_verb_objects()  -- Dict[str, List[BloomVerb]] (richest view; used
                           by callers that need usage_context / templates).
    get_all_verbs()     -- Set[str]               (flat union; for regex
                           alternation construction).
    detect_bloom_level  -- (text) -> Tuple[Optional[str], Optional[str]]
                           Canonical detector. Returns (level, verb) of
                           the first match, trying longer verbs first and
                           preferring higher-order levels on ties.

The schema is read once at import time. All getters return fresh copies so
callers may freely mutate without polluting the cache.

See `plans/kg-quality-review-2026-04/worker-h-subplan.md` for the migration
design and the 7 callsites that load from this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

__all__ = [
    "BLOOM_LEVELS",
    "BloomVerb",
    "get_verbs",
    "get_verbs_list",
    "get_verb_objects",
    "get_all_verbs",
    "detect_bloom_level",
    # Wave 48: schema-sourced cognitive domain
    "COGNITIVE_DOMAINS",
    "cognitive_domain_enum",
    "bloom_to_cognitive_domain",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOOM_LEVELS: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)


@dataclass(frozen=True)
class BloomVerb:
    """A single action verb with its pedagogical usage metadata.

    Fields:
        verb:             the action verb itself, lowercase.
        usage_context:    brief description of when to use this verb.
        example_template: template string for generating an objective stem
                          (uses curly-brace placeholders like {concept}).
    """

    verb: str
    usage_context: str
    example_template: str


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_BLOOM_VERBS_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "bloom_verbs.json"
)


@lru_cache(maxsize=1)
def _load_raw() -> Dict[str, List[Dict[str, str]]]:
    """Read the bloom_verbs schema and return a {level: [verb_dict, ...]} map.

    Only the per-level `default` arrays are extracted; the schema's `$defs`
    metadata is ignored here (it's for validators, not loaders).
    """
    if not _BLOOM_VERBS_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Bloom verbs schema not found at {_BLOOM_VERBS_SCHEMA_PATH}. "
            "Expected canonical copy from Worker F (Wave 1.1)."
        )

    with open(_BLOOM_VERBS_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    properties = schema.get("properties", {})
    out: Dict[str, List[Dict[str, str]]] = {}
    for level in BLOOM_LEVELS:
        level_prop = properties.get(level)
        if not level_prop or "default" not in level_prop:
            raise ValueError(
                f"Malformed bloom_verbs schema: missing properties.{level}.default "
                f"at {_BLOOM_VERBS_SCHEMA_PATH}"
            )
        out[level] = list(level_prop["default"])
    return out


@lru_cache(maxsize=1)
def _build_verb_objects() -> Dict[str, Tuple[BloomVerb, ...]]:
    """Build immutable (per-level) tuples of BloomVerb objects.

    Cached so the dataclass construction happens once per process. Getter
    functions return defensive copies.
    """
    raw = _load_raw()
    return {
        level: tuple(
            BloomVerb(
                verb=entry["verb"],
                usage_context=entry["usage_context"],
                example_template=entry["example_template"],
            )
            for entry in raw[level]
        )
        for level in BLOOM_LEVELS
    }


# ---------------------------------------------------------------------------
# Public getters
# ---------------------------------------------------------------------------

def get_verbs() -> Dict[str, Set[str]]:
    """Return `Dict[str, Set[str]]` — one set of verb strings per level.

    Shape used by `lib/validators/bloom.py`.
    """
    built = _build_verb_objects()
    return {level: {v.verb for v in built[level]} for level in BLOOM_LEVELS}


def get_verbs_list() -> Dict[str, List[str]]:
    """Return `Dict[str, List[str]]` — ordered list of verb strings per level.

    Shape used by `Trainforge/parsers/html_content_parser.py`,
    `Courseforge/scripts/generate_course.py`, and several others.
    """
    built = _build_verb_objects()
    return {level: [v.verb for v in built[level]] for level in BLOOM_LEVELS}


def get_verb_objects() -> Dict[str, List[BloomVerb]]:
    """Return `Dict[str, List[BloomVerb]]` — richest view (defensive copy).

    Shape used by `Courseforge/scripts/textbook-objective-generator/
    bloom_taxonomy_mapper.py`, which rekeys to its local `BloomLevel` enum.
    """
    built = _build_verb_objects()
    return {level: list(built[level]) for level in BLOOM_LEVELS}


def get_all_verbs() -> Set[str]:
    """Return a flat set of every verb across all six levels.

    Suitable for constructing a regex alternation (see
    `Trainforge/rag/libv2_bridge.py`). Caller is responsible for sorting
    by length if longest-first matching is required.
    """
    built = _build_verb_objects()
    out: Set[str] = set()
    for level in BLOOM_LEVELS:
        for v in built[level]:
            out.add(v.verb)
    return out


# ---------------------------------------------------------------------------
# Canonical detector
# ---------------------------------------------------------------------------

# Pre-compute: (verb, level) pairs sorted longest verb first, with higher-
# order levels winning ties. This matches the priority used historically
# by `lib/validators/bloom.py::detect_bloom_level` (higher levels first)
# and by the "longest match first" principle needed for regex alternation.
_LEVEL_PRIORITY = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}


@lru_cache(maxsize=1)
def _detection_order() -> Tuple[Tuple[str, str], ...]:
    """Build the (verb, level) iteration order for detect_bloom_level.

    Sort key:
      1. Verb length descending (longest multi-word verbs first, even
         though the current canonical list has none — defensive for
         future additions like "distinguish between").
      2. Level priority descending (create > evaluate > ... > remember)
         so that if a verb appears at multiple levels the higher-order
         level wins. (The current canonical has no duplicate verbs
         across levels, verified 2026-04.)
      3. Verb alphabetical for stable ordering.
    """
    built = _build_verb_objects()
    pairs: List[Tuple[str, str]] = []
    for level in BLOOM_LEVELS:
        for v in built[level]:
            pairs.append((v.verb, level))
    pairs.sort(
        key=lambda p: (-len(p[0]), -_LEVEL_PRIORITY[p[1]], p[0])
    )
    return tuple(pairs)


def detect_bloom_level(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Detect the Bloom's level and verb from free text.

    Lowercases the input, strips, and searches for each canonical verb as
    a whole word (`\\b{verb}\\b`). Returns `(level, verb)` on first match
    or `(None, None)` if no verb is found.

    Iteration order is longest-verb-first with higher-level ties winning,
    preserving the historical behavior of per-site detectors while
    consolidating them under a single canonical implementation.

    Examples:
        >>> detect_bloom_level("design a system to handle high load")
        ('create', 'design')
        >>> detect_bloom_level("list the steps of photosynthesis")
        ('remember', 'list')
        >>> detect_bloom_level("no verbs here whatsoever")
        (None, None)
    """
    if not text:
        return (None, None)
    lowered = text.lower().strip()
    for verb, level in _detection_order():
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            return (level, verb)
    return (None, None)


# ---------------------------------------------------------------------------
# Cognitive domain (Wave 48: schema-sourced)
# ---------------------------------------------------------------------------
#
# The Bloom-level → cognitive-domain mapping used to be duplicated as a
# hardcoded dict at two callsites (``Courseforge/scripts/generate_course.py``
# ``BLOOM_TO_DOMAIN`` and ``MCP/tools/_content_gen_helpers.py``
# ``_render_objectives_section``'s ``domain_map`` local). Wave 48 promotes
# the mapping to ``schemas/taxonomies/cognitive_domain.json`` and routes both
# callsites through :func:`bloom_to_cognitive_domain` so the two copies can
# no longer drift.

#: The authoritative 4-value cognitive-domain enum, frozen as a tuple for
#: hashability / immutability. Must stay in sync with the schema enum
#: (asserted at import time below).
COGNITIVE_DOMAINS: Tuple[str, ...] = (
    "factual",
    "conceptual",
    "procedural",
    "metacognitive",
)

#: Fallback for unknown bloom levels — preserves pre-Wave-48 behavior at
#: both callsites.
_COGNITIVE_DOMAIN_FALLBACK: str = "conceptual"

_COGNITIVE_DOMAIN_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "cognitive_domain.json"
)


@lru_cache(maxsize=1)
def _load_cognitive_domain_schema() -> Dict:
    """Read ``schemas/taxonomies/cognitive_domain.json`` once and cache."""
    if not _COGNITIVE_DOMAIN_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Cognitive-domain schema not found at "
            f"{_COGNITIVE_DOMAIN_SCHEMA_PATH}. Expected canonical copy from "
            "Wave 48."
        )
    with open(_COGNITIVE_DOMAIN_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_bloom_to_domain_map() -> Dict[str, str]:
    """Extract the ``bloom_level_to_domain`` mapping from the schema.

    The schema publishes the mapping as the ``default`` on the
    ``properties.bloom_level_to_domain`` node, mirroring the shape used by
    ``bloom_verbs.json`` (per-level ``default`` arrays).
    """
    schema = _load_cognitive_domain_schema()
    props = schema.get("properties") or {}
    node = props.get("bloom_level_to_domain") or {}
    default = node.get("default")
    if not isinstance(default, dict) or not default:
        raise ValueError(
            "Malformed cognitive_domain.json: missing "
            "properties.bloom_level_to_domain.default mapping."
        )
    # Sanity: every bloom level must map to a value in the enum.
    enum_set = set(COGNITIVE_DOMAINS)
    for level in BLOOM_LEVELS:
        if level not in default:
            raise ValueError(
                f"cognitive_domain.json bloom_level_to_domain missing level "
                f"{level!r}"
            )
        if default[level] not in enum_set:
            raise ValueError(
                f"cognitive_domain.json bloom_level_to_domain[{level!r}] = "
                f"{default[level]!r} not in enum {sorted(enum_set)}"
            )
    return dict(default)


def cognitive_domain_enum() -> Tuple[str, ...]:
    """Return the canonical 4-value cognitive-domain tuple.

    Suitable for validator enum checks and for tests that want to assert
    the emit-side ``data-cf-cognitive-domain`` value is canonical.
    """
    return COGNITIVE_DOMAINS


def bloom_to_cognitive_domain(bloom_level: Optional[str]) -> str:
    """Return the cognitive knowledge-domain for a Bloom's cognitive-process
    level.

    Falls back to ``"conceptual"`` for unknown / missing levels, matching
    pre-Wave-48 behavior at both migrated callsites (``generate_course.py``
    objective emit + JSON-LD, ``_content_gen_helpers._render_objectives_section``
    objective emit).

    Examples:
        >>> bloom_to_cognitive_domain("remember")
        'factual'
        >>> bloom_to_cognitive_domain("create")
        'procedural'
        >>> bloom_to_cognitive_domain("bogus")
        'conceptual'
        >>> bloom_to_cognitive_domain(None)
        'conceptual'
    """
    if not bloom_level:
        return _COGNITIVE_DOMAIN_FALLBACK
    mapping = _load_bloom_to_domain_map()
    return mapping.get(bloom_level, _COGNITIVE_DOMAIN_FALLBACK)


# Import-time sanity: schema enum must match the module's expected tuple.
# Runs once per process (``_load_cognitive_domain_schema`` is cached) so the
# cost is a single JSON read.
def _assert_cognitive_domain_enum_matches_schema() -> None:
    schema = _load_cognitive_domain_schema()
    defs = schema.get("$defs") or {}
    node = defs.get("CognitiveDomain") or {}
    schema_enum = tuple(node.get("enum") or ())
    if schema_enum != COGNITIVE_DOMAINS:
        raise RuntimeError(
            "cognitive_domain.json $defs.CognitiveDomain.enum drift: "
            f"schema={schema_enum!r} expected={COGNITIVE_DOMAINS!r}"
        )


_assert_cognitive_domain_enum_matches_schema()
