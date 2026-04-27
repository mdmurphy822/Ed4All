"""RDF-backed slug aliasing — Phase 2.2 + Phase 2.5 of the RDF/SHACL plan.

This module is the runtime entry point for canonicalizing concept-graph
slugs against the equivalence classes declared in
``schemas/context/aliases.ttl``. It replaces the hand-rolled
``KNOWN_EQUIVALENT_ALIASES`` dict in
``lib.ontology.concept_classifier`` with a single source of truth that
is itself an RDF graph — so the same Turtle file feeds the runtime
helper, the SHACL validation pipeline (Phase 4), and downstream
RDF tooling that wants to import our vocabulary.

Why ``owl:equivalentProperty`` / ``owl:equivalentClass`` (not ``owl:sameAs``)?
-----------------------------------------------------------------------------
Per the LibV2 corpus answers Q9 (`q_20260426_205704_f9dc2230`) and
Q11 (`q_20260426_205705_7c3f954f`), ``owl:sameAs`` is reserved for
INDIVIDUAL identity: it asserts that two IRIs denote the same
real-world thing, and every triple about one is entailed about the
other in both directions. Using sameAs for a surface-form synonym
relation would be type-incorrect (we are aliasing CLASS or PREDICATE
IRIs, not individuals) and would risk the "sameAs explosion" — under
OWL 2 RL reasoning, chained sameAs assertions multiply triple
closures combinatorially and can balloon a tractable graph into an
unworkable one. ``owl:equivalentProperty`` and ``owl:equivalentClass``
are the structurally correct vocabularies: symmetric, transitive,
type-respecting, and bounded in scope.

Two equivalence predicates are walked
-------------------------------------
The Turtle file mixes two flavors of equivalence in a single graph:

1. **Slug-canonicalization aliases** (Phase 2.2) — pairs entirely
   inside the ``ed4all:`` namespace at ``https://ed4all.io/vocab/``,
   e.g. ``ed4all:rdfxml owl:equivalentProperty ed4all:rdf-xml``. Both
   sides are typed as ``owl:DatatypeProperty`` so they participate in
   the predicate equivalence relation.
2. **Cross-namespace vocabulary bridges** (Phase 2.5) — pairs that
   bridge the same local name across the two production namespaces:
   ``ed4all:`` (https://ed4all.io/vocab/, used by every JSON-LD
   ``@context`` file) and ``cf:`` (https://ed4all.dev/ns/courseforge/v1#,
   used by the formal RDFS/OWL vocabulary in
   ``schemas/context/courseforge_v1.vocabulary.ttl``). Class bridges
   use ``owl:equivalentClass``; predicate bridges use
   ``owl:equivalentProperty``.

Both equivalence predicates are walked unconditionally — the closure
collapses class-typed pairs and predicate-typed pairs into a single
union-find structure. This is sound because the two relations never
intersect in practice (a given local name is either a class or a
predicate, not both), and unioning two disjoint relations yields a
correct equivalence-class closure across both. Mixing the two in one
union-find pass also keeps the implementation symmetric: the IRI's
local-name fragment is the only key the canonical-slug API exposes,
so callers don't need to know whether a slug originally named a class
or a predicate.

Closure-walk algorithm
----------------------
The Turtle file declares a sparse set of pairwise equivalences. To
answer ``canonicalize(slug)`` we need the full transitive+symmetric
closure of the equivalence relation. Implementation: union-find
(disjoint-set) with path compression. For each ``owl:equivalentProperty``
or ``owl:equivalentClass`` triple ``(a, b)`` we union the two local
slugs into the same class; at the end we walk every class and pick a
representative — preferring the member annotated ``ed4all:isCanonical
true`` if one exists, else falling back to lexicographic order for
determinism. The resulting ``Dict[str, str]`` (slug → canonical) is
cached as a module-level singleton so repeated ``canonicalize`` calls
are O(1).

Cross-namespace bridges work because the slug extractor accepts IRIs
from BOTH the ``ed4all:`` and ``cf:`` namespaces and projects them to
the same local-name slug. So ``ed4all:Concept`` and ``cf:Concept`` both
extract the slug ``Concept``, get unioned by the bridge axiom, and
canonicalize to the same representative regardless of which side the
input came from. IRIs outside both namespaces are skipped — they would
be cross-vocabulary equivalences (e.g., to schema.org or FOAF) and are
out of scope for the slug-canonicalization layer.

Union-find was chosen over BFS/fixed-point/SPARQL property paths
because:

- BFS would re-walk the relation per query; we'd cache per slug but
  union-find amortizes better and produces the full closure in a
  single pass.
- A fixed-point loop over the triple set has worst-case O(N^2) and
  needs explicit termination logic.
- SPARQL property paths (``?x owl:equivalentProperty+ ?y``) would work
  but introduce a runtime dependency on rdflib's path-evaluation
  performance, which has historically been uneven.

Cycles in the input graph (e.g. ``A ↔ B``, ``B ↔ A``) are absorbed
naturally by union-find — they collapse to a single equivalence class
without any special handling. Self-loops are no-ops.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TTL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "schemas"
    / "context"
    / "aliases.ttl"
)

# IRI namespace prefixes that alias terms can live under. Slugs are the
# local-name fragment after one of these prefixes.
#
# - _ED4ALL_NS: the JSON-LD-side namespace shared by all four
#   schemas/context/*_v1.jsonld files and by every Phase 2.2 alias
#   triple in aliases.ttl. Primary surface for slug canonicalization.
# - _CF_NS: the RDFS/OWL-side namespace declared in
#   schemas/context/courseforge_v1.vocabulary.ttl. Phase 2.5 bridges
#   add `ed4all:<local> owl:equivalent{Class,Property} cf:<local>`
#   axioms; the parser must accept IRIs from this namespace and
#   project them to the same local slug so the union-find collapses
#   the cross-namespace pair.
_ED4ALL_NS = "https://ed4all.io/vocab/"
_CF_NS = "https://ed4all.dev/ns/courseforge/v1#"

# Namespaces whose IRIs participate in slug canonicalization. Order is
# irrelevant — a local name is the same string regardless of which
# namespace prefix it was extracted from.
_BRIDGED_NAMESPACES = (_ED4ALL_NS, _CF_NS)

# IRI of the annotation predicate flagging the canonical representative.
_IS_CANONICAL_IRI = _ED4ALL_NS + "isCanonical"

# IRIs of the two equivalence predicates we walk. Both are symmetric
# and transitive; unioning them in a single pass yields the correct
# closure across the slug-alias and cross-namespace-bridge relations.
_OWL_EQUIVALENT_PROPERTY_IRI = "http://www.w3.org/2002/07/owl#equivalentProperty"
_OWL_EQUIVALENT_CLASS_IRI = "http://www.w3.org/2002/07/owl#equivalentClass"


# ---------------------------------------------------------------------------
# Module-level cache — built lazily on first canonicalize() call.
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# Union-find primitives.
# ---------------------------------------------------------------------------


def _find(parent: Dict[str, str], slug: str) -> str:
    """Find the root of ``slug`` with path compression."""
    root = slug
    while parent[root] != root:
        root = parent[root]
    # Path compression — point every visited node directly at the root
    # so future ``_find`` calls are O(1) amortized.
    cursor = slug
    while parent[cursor] != root:
        parent[cursor], cursor = root, parent[cursor]
    return root


def _union(parent: Dict[str, str], a: str, b: str) -> None:
    """Union the equivalence classes containing ``a`` and ``b``."""
    if a not in parent:
        parent[a] = a
    if b not in parent:
        parent[b] = b
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        # Deterministic merge — smaller-string wins as the parent so
        # the build is reproducible across Python sessions. The final
        # canonical pick at the end overrides this anyway.
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b


# ---------------------------------------------------------------------------
# Turtle parsing.
# ---------------------------------------------------------------------------


def _slug_from_iri(iri: str) -> Optional[str]:
    """Extract the slug fragment from an ``ed4all:`` or ``cf:`` IRI.

    Cross-namespace bridges (Phase 2.5) declare equivalences between
    IRIs sharing a local name across the two production namespaces:
    ``https://ed4all.io/vocab/<name>`` (JSON-LD surface) and
    ``https://ed4all.dev/ns/courseforge/v1#<name>`` (formal RDFS/OWL
    vocabulary). Returning the bare ``<name>`` for either form lets
    the union-find collapse the pair into a single equivalence class.

    Returns ``None`` for IRIs outside both bridged namespaces — those
    would be cross-vocabulary equivalences (schema.org, FOAF, etc.)
    and are out of scope for the slug-canonicalization layer.
    """
    for prefix in _BRIDGED_NAMESPACES:
        if iri.startswith(prefix):
            return iri[len(prefix):]
    return None


def _parse_aliases_ttl(
    ttl_path: Path,
) -> Tuple[List[Tuple[str, str]], Iterable[str]]:
    """Parse the aliases Turtle and return ``(equivalence_pairs, canonical_slugs)``.

    Uses rdflib for robust Turtle parsing — we're not in a position to
    rewrite the parser by hand and Turtle's syntax (prefixes, semicolons,
    blank-line continuations) is non-trivial. rdflib is already a
    transitive dependency through pyshacl in the dev extras.
    """
    # rdflib import is local to keep the helper importable in
    # environments where rdflib isn't installed (e.g. lightweight
    # production containers running only the Trainforge subset). The
    # caller catches ImportError and falls back gracefully.
    from rdflib import Graph, URIRef
    from rdflib.namespace import OWL, RDFS  # noqa: F401  (kept for symmetry)

    graph = Graph()
    graph.parse(ttl_path, format="turtle")

    equivalence_pairs: List[Tuple[str, str]] = []
    canonical_slugs: set[str] = set()

    is_canonical_predicate = URIRef(_IS_CANONICAL_IRI)

    # Collect every equivalence triple — both `owl:equivalentProperty`
    # (Phase 2.2 slug aliases AND Phase 2.5 cross-namespace predicate
    # bridges) and `owl:equivalentClass` (Phase 2.5 cross-namespace
    # class bridges). Both relations are symmetric and transitive; we
    # union them into a single equivalence-class structure because the
    # slug API exposes only the local-name fragment, which is the same
    # string regardless of whether the original IRI named a class or a
    # predicate.
    #
    # Triples involving IRIs outside the bridged namespaces are ignored
    # — those would be cross-vocabulary equivalences (e.g. to schema.org
    # or FOAF) and are out of scope for the slug-canonicalization layer.
    for equiv_iri in (_OWL_EQUIVALENT_PROPERTY_IRI, _OWL_EQUIVALENT_CLASS_IRI):
        equiv_predicate = URIRef(equiv_iri)
        for subject, _, obj in graph.triples((None, equiv_predicate, None)):
            s_slug = _slug_from_iri(str(subject))
            o_slug = _slug_from_iri(str(obj))
            if s_slug is not None and o_slug is not None:
                equivalence_pairs.append((s_slug, o_slug))

    # Collect every IRI flagged as the canonical representative.
    for subject, _, obj in graph.triples((None, is_canonical_predicate, None)):
        # rdflib parses ``true`` as a Literal with python value True.
        if bool(obj):
            slug = _slug_from_iri(str(subject))
            if slug is not None:
                canonical_slugs.add(slug)

    return equivalence_pairs, canonical_slugs


# ---------------------------------------------------------------------------
# Closure walk — union-find driver.
# ---------------------------------------------------------------------------


def _build_canonicalization_map(
    equivalence_pairs: List[Tuple[str, str]],
    canonical_slugs: Iterable[str],
) -> Dict[str, str]:
    """Walk the equivalence-pair list and produce a slug→canonical map."""
    canonical_set = set(canonical_slugs)

    parent: Dict[str, str] = {}
    for a, b in equivalence_pairs:
        _union(parent, a, b)

    # Group every slug by its root — these are the equivalence classes.
    classes: Dict[str, List[str]] = {}
    for slug in parent:
        root = _find(parent, slug)
        classes.setdefault(root, []).append(slug)

    # For each class, pick the canonical representative.
    result: Dict[str, str] = {}
    for members in classes.values():
        # Prefer a member explicitly flagged ed4all:isCanonical true.
        canonical_members = [m for m in members if m in canonical_set]
        if canonical_members:
            # If multiple are flagged canonical (shouldn't happen, but
            # defensive), pick the lexicographically smallest for
            # determinism.
            chosen = sorted(canonical_members)[0]
        else:
            # Fallback: lexicographic order so the build is
            # reproducible. A test will catch the missing isCanonical
            # annotation but we don't fail closed at runtime — the
            # result is still self-consistent.
            chosen = sorted(members)[0]
        for member in members:
            result[member] = chosen

    return result


# ---------------------------------------------------------------------------
# Cache management.
# ---------------------------------------------------------------------------


def _load_cache() -> Dict[str, str]:
    """Load (or rebuild) the slug→canonical map. Thread-safe."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:
            return _CACHE
        try:
            equivalence_pairs, canonical_slugs = _parse_aliases_ttl(_TTL_PATH)
            _CACHE = _build_canonicalization_map(
                equivalence_pairs, canonical_slugs
            )
        except Exception:
            # Fail-open: if the Turtle file is missing or rdflib isn't
            # installed, the caller (concept_classifier) falls back to
            # the legacy KNOWN_EQUIVALENT_ALIASES dict. We log nothing
            # here because the caller decides the policy.
            _CACHE = {}
        return _CACHE


def reload_cache() -> None:
    """Clear the cached canonicalization map.

    Test-only — production code never invalidates the cache because the
    Turtle file is read once at process start. The unit tests under
    ``lib/ontology/tests/test_aliases.py`` use this to force a fresh
    parse after monkeypatching the TTL path.
    """
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def canonicalize(slug: str) -> str:
    """Return the canonical slug for ``slug``.

    Pass-through for slugs not declared in ``schemas/context/aliases.ttl``
    — keeps the function safe for arbitrary concept-graph node IDs.

    Lookup precedence:

    1. **Literal lookup** — try ``slug`` exactly as given. Phase 2.5
       cross-namespace bridges (``ed4all:Concept ↔ cf:Concept``,
       ``ed4all:hasPrerequisite ↔ cf:hasPrerequisite``, etc.) extract
       CamelCase local names from the IRIs, so the bridged keys land
       in the cache in their original case. A literal lookup catches
       these without forcing the caller to know whether their slug
       came from a Phase-2.2 alias family (typically lowercased) or a
       Phase-2.5 bridge (typically CamelCase).

    2. **Case-insensitive fallback** — if the literal miss, try
       ``slug.lower()``. This preserves parity with the legacy
       ``concept_classifier.KNOWN_EQUIVALENT_ALIASES.get(slug.lower(),
       slug)`` contract: callers like ``canonicalize_alias("RDFXML")``
       still resolve to ``rdf-xml``, and ``canonicalize_alias("Turtle")``
       still resolves to ``turtle``.

    3. **Pass-through** — neither lookup hit; return ``slug`` unchanged
       (mixed case preserved on miss, matching the legacy contract).
    """
    if not slug:
        return slug
    cache = _load_cache()
    if slug in cache:
        return cache[slug]
    return cache.get(slug.lower(), slug)


__all__ = ["canonicalize", "reload_cache"]
