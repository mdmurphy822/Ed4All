"""Edge-slug normalizer (Phase 2.7) — bridge two emitters' slug conventions.

The Ed4All concept-graph surface ships with two edge-emitting code
paths that disagree about slug spelling:

* ``Trainforge/rag/typed_edge_inference.py`` (emits
  ``concept_graph_semantic.json``) — uses **hyphens** with no
  ``_of`` / ``-of`` suffix: ``is-a``, ``defined-by``,
  ``derived-from-objective``, ``targets-concept``.
* ``Trainforge/pedagogy_graph_builder.py`` (emits
  ``pedagogy_graph.json``) — uses **underscores** and a directional
  ``_of`` suffix on directional ties: ``prerequisite_of``,
  ``derived_from_objective``, ``belongs_to_module``,
  ``targets_concept``.

The same logical predicate can therefore appear under two different
surface spellings depending on which emitter produced the JSON, and the
canonical registry (``lib.ontology.edge_predicates.SLUG_TO_IRI``) only
holds the hyphenated form. Worker F's pedagogy round-trip test bridged
this for four known slugs by hand-rolling a per-key alias dict; the
canonical fix is a tiny, declarative normalizer.

Why a normalizer beats forcing one emitter to change:

1. **Backward compatibility.** Existing pedagogy_graph.json archives in
   LibV2 corpora (``LibV2/courses/*/graph/pedagogy_graph.json``) were
   emitted with the ``_of`` convention. Rewriting the emitter would
   silently invalidate those archives until they are regenerated.
2. **Each convention is locally defensible.** Underscored slugs match
   the relation_type field-naming used elsewhere in
   ``pedagogy_graph_builder.py``; the ``_of`` suffix encodes
   directionality, which is information the purely-hyphenated
   convention drops. A normalizer preserves both surfaces while still
   collapsing them to one canonical form at lookup time.
3. **Single point of change for future emitters.** Any future emitter
   that follows either convention (or invents a third) only needs the
   normalization rules updated here, not a per-call-site rewrite.

The normalizer is intentionally **not** a slug-rewriter: it never
fabricates registry entries. If a normalized slug fails to resolve in
``SLUG_TO_IRI``, the original input is returned unchanged so callers
can fall back to whatever pre-existing handling (e.g., the JSON-LD
``@vocab`` fallback) the call site relies on. Phase 2.6 will mint the
9 currently-unminted pedagogy predicates; once that lands, more
normalized inputs will resolve, but the normalizer's behavior on
already-resolving slugs stays unchanged.

Normalization algorithm:

1. Lowercase the input.
2. Replace internal ``_`` with ``-``.
3. Strip a single trailing ``-of`` (post step 2; equivalently a
   trailing ``_of`` in the raw input).
4. If the normalized form is in ``SLUG_TO_IRI``, return it.
5. Otherwise, return the **original** input slug unchanged.

The ``-of`` strip happens after the ``_`` → ``-`` substitution so a
trailing ``_of`` in raw input is handled by the same rule.

Idempotent on every canonical key in ``SLUG_TO_IRI``: the registry
keys are already lowercase and hyphenated; the only one ending in
``-of`` is ``misconception-of``, and the post-strip form
``misconception`` is NOT in the registry, so the safety guard returns
the original input unchanged. See
``lib/ontology/tests/test_edge_slug_normalizer.py`` for the
round-trip proof.
"""

from __future__ import annotations

__all__ = ["normalize_edge_slug"]


def normalize_edge_slug(slug: str) -> str:
    """Return the registry-canonical form of ``slug`` if one exists.

    Maps emit-time slug variants (underscored, ``_of``-suffixed) to the
    hyphenated keys held in ``lib.ontology.edge_predicates.SLUG_TO_IRI``.
    Idempotent on already-canonical input. Returns the input unchanged
    when no matching registry key exists, so callers can fall back to
    whatever they did before (no silent fabrication of registry
    entries).

    Examples
    --------
    >>> normalize_edge_slug("prerequisite_of")
    'prerequisite'
    >>> normalize_edge_slug("derived_from_objective")
    'derived-from-objective'
    >>> normalize_edge_slug("is-a")
    'is-a'
    >>> normalize_edge_slug("random_unknown_slug")
    'random_unknown_slug'
    """
    # Lazy import to keep the dependency direction one-way: callers in
    # edge_predicates.py may eventually import this module, so importing
    # SLUG_TO_IRI at module top-level would form a cycle.
    from lib.ontology.edge_predicates import SLUG_TO_IRI

    if not isinstance(slug, str) or not slug:
        return slug

    candidate = slug.lower().replace("_", "-")
    if candidate.endswith("-of"):
        candidate = candidate[: -len("-of")]

    if candidate in SLUG_TO_IRI:
        return candidate

    return slug
