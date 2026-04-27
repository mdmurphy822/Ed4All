"""Concept-graph edge-type slug → RDF predicate registry (Phase 2.1).

The Trainforge concept-graph emit
(``Trainforge/rag/typed_edge_inference.py``) carries 9 edge types as
opaque slugs in the JSON-LD payload. The slug enum lives in
``schemas/knowledge/concept_graph_semantic.schema.json`` (``edges[].type``).
This module is the single source of truth for the slug → IRI mapping
that lifts those slugs to actual RDF predicates with declared
``rdfs:domain`` / ``rdfs:range`` semantics.

Two slugs reuse pre-existing W3C predicates rather than mint new ones:

* ``is-a``      → ``rdfs:subClassOf`` — genuine class subsumption; an
  RDFS reasoner gets inheritance entailment for free.
* ``related-to`` → ``skos:related`` — symmetric associative tie between
  SKOS concepts; the lowest-precedence typed edge in Trainforge
  (dropped when ``is-a`` or ``prerequisite`` covers the same pair).

The remaining 7 slugs are minted in the ``ed4all:`` namespace at
``https://ed4all.dev/ns/courseforge/v1#`` and declared in
``schemas/context/courseforge_v1.vocabulary.ttl`` with full RDFS
semantics (label, comment grounded in the emitting Trainforge inference
rule, domain, range).

**Synchronization invariant.** This registry, the JSON enum, and the
Turtle vocabulary file MUST stay in sync — the test suite enforces it:

* ``schemas/tests/test_courseforge_vocabulary.py`` asserts every slug in
  the JSON enum has a key here, every minted ed4all: predicate parses
  out of the Turtle file, and the slug↔IRI round-trip is symmetric.

When adding a new edge type:

1. Add the slug to the JSON enum (``edges[].type`` in
   ``concept_graph_semantic.schema.json``).
2. Mint or pick a canonical IRI. Reuse W3C/SKOS/Schema.org predicates
   over minting new ones whenever the semantics align.
3. Add the predicate declaration to ``courseforge_v1.vocabulary.ttl``
   with rdfs:domain + rdfs:range pointing at declared ed4all: classes.
4. Add the slug → IRI binding to ``SLUG_TO_IRI`` below.
5. Run ``pytest schemas/tests/test_courseforge_vocabulary.py`` — drift
   between the four locations fails the test suite.

This phase does NOT yet alter the runtime emission in
``typed_edge_inference.py`` — that's deferred to a flag-gated change in
a later phase (Phase 2.4 in the enrichment plan,
``TRAINFORGE_RDF_ALIGNED_EDGE_TYPES``). For now the registry is the
declarative bridge; consumers wanting RDF can resolve the slug here.
"""

from __future__ import annotations

from typing import Dict, Final, Optional

# Pre-existing W3C / SKOS predicate IRIs (reused, not minted).
_RDFS_SUBCLASS_OF: Final[str] = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
_SKOS_RELATED: Final[str] = "http://www.w3.org/2004/02/skos/core#related"

# ed4all: namespace base — must match the @prefix declaration in
# schemas/context/courseforge_v1.vocabulary.ttl line 41.
_ED4ALL_NS: Final[str] = "https://ed4all.dev/ns/courseforge/v1#"


def _ed4all(local: str) -> str:
    return _ED4ALL_NS + local


SLUG_TO_IRI: Final[Dict[str, str]] = {
    # Phase 2.1 — concept_graph_semantic.schema.json::edges[].type enum.
    "is-a": _RDFS_SUBCLASS_OF,
    "related-to": _SKOS_RELATED,
    "prerequisite": _ed4all("hasPrerequisite"),
    "defined-by": _ed4all("isDefinedBy"),
    "derived-from-objective": _ed4all("isDerivedFromObjective"),
    "exemplifies": _ed4all("exemplifiedBy"),
    "misconception-of": _ed4all("isMisconceptionOf"),
    "assesses": _ed4all("assessesObjective"),
    "targets-concept": _ed4all("targetsConcept"),
    # Phase 2.6 — pedagogy_graph_builder.py emits 13 distinct
    # ``relation_type`` slugs. Four overlap with Phase 2.1
    # (``assesses``, ``exemplifies``, plus ``derived_from_objective`` /
    # ``prerequisite_of`` underscore variants resolved via Phase 2.7's
    # normalizer back to their hyphenated Phase 2.1 keys). The nine
    # below close the gap so every pedagogy slug resolves to a real
    # predicate IRI rather than the JSON-LD ``@vocab`` fallback at
    # https://ed4all.io/vocab/edge-type#<slug>. Keys use the schema's
    # emit form (underscores, no ``_of`` suffix on these nine because
    # none of the pedagogy slugs minted in 2.6 happen to use it) — the
    # Phase 2.7 normalizer in ``edge_slug_normalizer.py`` handles
    # hyphen↔underscore drift on lookup.
    "teaches": _ed4all("teaches"),
    "belongs_to_module": _ed4all("belongsToModule"),
    "supports_outcome": _ed4all("supportsOutcome"),
    "at_bloom_level": _ed4all("atBloomLevel"),
    "follows": _ed4all("follows"),
    "concept_supports_outcome": _ed4all("conceptSupportsOutcome"),
    "assessment_validates_outcome": _ed4all("assessmentValidatesOutcome"),
    "chunk_at_difficulty": _ed4all("chunkAtDifficulty"),
    "interferes_with": _ed4all("interferesWith"),
}
"""Slug → full IRI string. Source of truth for resolving a JSON-emit
edge type to its RDF predicate. Keys mirror two emit surfaces:

* ``schemas/knowledge/concept_graph_semantic.schema.json::edges[].type``
  for the Phase 2.1 entries (hyphenated, no ``_of`` suffix).
* The 13 ``relation_type`` values emitted by
  ``Trainforge/pedagogy_graph_builder.py`` for the Phase 2.6 entries
  (underscored). The four pedagogy slugs that overlap with Phase 2.1
  — ``assesses``, ``exemplifies``, ``derived_from_objective``,
  ``prerequisite_of`` — are resolved by Phase 2.7's slug normalizer
  to the corresponding hyphenated Phase 2.1 keys."""


IRI_TO_SLUG: Final[Dict[str, str]] = {iri: slug for slug, iri in SLUG_TO_IRI.items()}
"""Inverse of ``SLUG_TO_IRI`` for round-trip parsing — when a graph
consumer encounters one of our predicate IRIs and needs the canonical
slug back (e.g., for evidence-discriminator routing keyed off the
JSON ``type`` field). The ``SLUG_TO_IRI`` mapping is bijective by
construction (no two slugs share an IRI), so this inversion is exact."""


def lookup_iri(slug: str) -> Optional[str]:
    """Resolve a slug to its IRI, tolerant of emitter convention drift.

    Use this instead of ``SLUG_TO_IRI[slug]`` when slug origin is
    uncertain (e.g., parsing ``pedagogy_graph.json``, whose emitter
    uses underscored / ``_of``-suffixed slugs while this registry holds
    the hyphenated canonical form). The function:

    1. Tries the slug verbatim against ``SLUG_TO_IRI``.
    2. On miss, normalizes via
       :func:`lib.ontology.edge_slug_normalizer.normalize_edge_slug`
       and tries again.
    3. Returns ``None`` if still unresolved.

    Phase 2.7 normalizer details — including the ``_of`` strip rule and
    the safety guard that prevents fabricating registry entries — live
    in ``lib/ontology/edge_slug_normalizer.py``.
    """
    direct = SLUG_TO_IRI.get(slug)
    if direct is not None:
        return direct

    # Lazy import to avoid a circular dependency: the normalizer
    # imports SLUG_TO_IRI from this module at call time.
    from lib.ontology.edge_slug_normalizer import normalize_edge_slug

    normalized = normalize_edge_slug(slug)
    if normalized != slug:
        return SLUG_TO_IRI.get(normalized)
    return None


__all__ = ["SLUG_TO_IRI", "IRI_TO_SLUG", "lookup_iri"]
