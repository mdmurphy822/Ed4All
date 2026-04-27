"""Phase 2.7 tests: edge-slug normalizer + ``lookup_iri`` shim.

The Ed4All concept-graph surface is fed by two emitters that disagree
about slug spelling — ``Trainforge/rag/typed_edge_inference.py`` emits
hyphenated slugs (``is-a``, ``derived-from-objective``) and
``Trainforge/pedagogy_graph_builder.py`` emits underscored slugs with
a ``_of`` suffix on directional ties (``prerequisite_of``,
``derived_from_objective``). The Phase 2.7 normalizer
(``lib.ontology.edge_slug_normalizer.normalize_edge_slug``) collapses
both surfaces to the registry-canonical form held in
``lib.ontology.edge_predicates.SLUG_TO_IRI``.

Property coverage:

* **Idempotency** — every canonical key in ``SLUG_TO_IRI`` round-trips
  through the normalizer unchanged.
* **Bridging** — the underscored / ``_of``-suffixed variants emitted
  by ``pedagogy_graph_builder.py`` map to their hyphenated canonical
  forms.
* **Pass-through** — unknown slugs (no matching registry entry after
  normalization) are returned unchanged. Critical: the normalizer
  must NOT fabricate registry entries.
* **``lookup_iri`` shim** — every ``SLUG_TO_IRI`` key resolves;
  underscored / ``_of`` variants of registered keys also resolve;
  unknown returns ``None``.
* **Pedagogy-graph parity** — every ``relation_type`` in the
  rdf-shacl-551-2 fixture either resolves to an IRI or sits in an
  explicit "expected unminted" set. Phase 2.6 will mint those 9
  predicates; when it lands, this test breaks meaningfully because the
  unminted set will start shrinking and ``EXPECTED_UNMINTED`` will go
  out of date.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.ontology.edge_predicates import IRI_TO_SLUG, SLUG_TO_IRI, lookup_iri
from lib.ontology.edge_slug_normalizer import normalize_edge_slug


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("canonical_slug", sorted(SLUG_TO_IRI.keys()))
def test_normalizer_idempotent_on_canonical_keys(canonical_slug: str) -> None:
    """Every registry key must round-trip through the normalizer unchanged."""
    assert normalize_edge_slug(canonical_slug) == canonical_slug


def test_normalizer_idempotent_double_apply() -> None:
    """Applying the normalizer twice must equal applying it once."""
    samples = [
        "prerequisite_of",
        "derived_from_objective",
        "targets_concept",
        "is-a",
        "misconception-of",
        "random_unknown_slug",
    ]
    for s in samples:
        once = normalize_edge_slug(s)
        twice = normalize_edge_slug(once)
        assert once == twice, f"normalizer not idempotent on {s!r}"


# ---------------------------------------------------------------------------
# Bridging — emit-time variants → canonical forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "emit_slug,expected_canonical",
    [
        ("prerequisite_of", "prerequisite"),
        ("derived_from_objective", "derived-from-objective"),
        ("targets_concept", "targets-concept"),
        # Mixed-case input must lowercase first.
        ("Prerequisite_Of", "prerequisite"),
        ("DERIVED_FROM_OBJECTIVE", "derived-from-objective"),
    ],
)
def test_normalizer_bridges_underscore_variants(
    emit_slug: str, expected_canonical: str
) -> None:
    assert normalize_edge_slug(emit_slug) == expected_canonical


# ---------------------------------------------------------------------------
# Pass-through — never fabricate registry entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unknown_slug",
    [
        "random_unknown_slug",
        "completely-made-up",
        "totally-not-a-real-edge",
        # Only-trailing-of inputs whose post-strip form isn't registered:
        # "misconception" is NOT a registry key, so "misconception-of"
        # must pass through unchanged (and resolve via the original
        # form, which IS a key — but that's lookup_iri's job, not the
        # normalizer's).
    ],
)
def test_normalizer_passthrough_for_unknown(unknown_slug: str) -> None:
    assert normalize_edge_slug(unknown_slug) == unknown_slug


def test_misconception_of_normalizes_to_self() -> None:
    """``misconception-of`` is itself a registry key; stripping ``-of``
    yields ``misconception`` which is NOT registered, so the safety
    guard must return the original input so the canonical key still
    resolves at the lookup_iri layer."""
    assert normalize_edge_slug("misconception-of") == "misconception-of"
    assert "misconception-of" in SLUG_TO_IRI
    assert "misconception" not in SLUG_TO_IRI


def test_normalizer_handles_empty_and_non_string() -> None:
    """Defensive: empty string returns itself; non-strings return as-is."""
    assert normalize_edge_slug("") == ""
    # Non-string inputs are returned unchanged (defensive guard).
    assert normalize_edge_slug(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# lookup_iri shim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("canonical_slug", sorted(SLUG_TO_IRI.keys()))
def test_lookup_iri_resolves_every_canonical_key(canonical_slug: str) -> None:
    iri = lookup_iri(canonical_slug)
    assert iri is not None
    assert iri == SLUG_TO_IRI[canonical_slug]


@pytest.mark.parametrize(
    "emit_slug,expected_canonical",
    [
        ("prerequisite_of", "prerequisite"),
        ("derived_from_objective", "derived-from-objective"),
        ("targets_concept", "targets-concept"),
    ],
)
def test_lookup_iri_resolves_underscore_variants(
    emit_slug: str, expected_canonical: str
) -> None:
    iri = lookup_iri(emit_slug)
    assert iri is not None
    assert iri == SLUG_TO_IRI[expected_canonical]
    # Inverse direction is sane too.
    assert IRI_TO_SLUG[iri] == expected_canonical


def test_lookup_iri_returns_none_for_unknown() -> None:
    assert lookup_iri("totally_not_a_slug") is None
    assert lookup_iri("completely-made-up") is None
    assert lookup_iri("") is None


# ---------------------------------------------------------------------------
# Pedagogy-graph parity (rdf-shacl-551-2 fixture)
# ---------------------------------------------------------------------------


# Phase 2.6 has landed: all 9 pedagogy-graph relation_types that
# previously needed minting are now in SLUG_TO_IRI. The expected-
# unminted set is therefore empty. If a future emitter introduces a
# new relation_type that does not resolve, this set is the place to
# enumerate it (with a code-comment justification) so the parity
# tests below still pass while the registry catches up.
EXPECTED_UNMINTED: frozenset[str] = frozenset()

PEDAGOGY_GRAPH_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "LibV2"
    / "courses"
    / "rdf-shacl-551-2"
    / "graph"
    / "pedagogy_graph.json"
)


@pytest.fixture(scope="module")
def pedagogy_relation_types() -> frozenset[str]:
    if not PEDAGOGY_GRAPH_FIXTURE.is_file():
        pytest.skip(f"pedagogy_graph fixture not present: {PEDAGOGY_GRAPH_FIXTURE}")
    data = json.loads(PEDAGOGY_GRAPH_FIXTURE.read_text(encoding="utf-8"))
    edges = data.get("edges", [])
    return frozenset(
        e.get("relation_type") for e in edges if e.get("relation_type")
    )


def test_pedagogy_relation_types_either_resolve_or_are_expected_unminted(
    pedagogy_relation_types: frozenset[str],
) -> None:
    """Every relation_type in the fixture must either resolve via
    ``lookup_iri`` or be in the explicit ``EXPECTED_UNMINTED`` set.

    This is the breakage signal for Phase 2.6: as predicates get
    minted, ``EXPECTED_UNMINTED`` must shrink to match. If a relation
    is neither resolved nor declared expected-unminted, that's drift
    and must fail loudly.
    """
    unresolved: set[str] = set()
    for rt in pedagogy_relation_types:
        iri = lookup_iri(rt)
        if iri is None:
            unresolved.add(rt)

    unexpected_unresolved = unresolved - EXPECTED_UNMINTED
    assert not unexpected_unresolved, (
        f"Pedagogy relation_types failed to resolve and are NOT in the "
        f"expected-unminted set: {sorted(unexpected_unresolved)}. "
        f"Either Phase 2.6 minted them (update EXPECTED_UNMINTED) or a "
        f"new emitter convention slipped in."
    )

    # Inverse direction: anything we declared "expected unminted" but
    # which now resolves means Phase 2.6 partially landed; force the
    # set to be tightened.
    over_declared = EXPECTED_UNMINTED - unresolved
    # Filter out slugs that don't appear in this fixture at all — the
    # set is global to the test, not fixture-scoped, so absence is OK.
    over_declared_present = over_declared & pedagogy_relation_types
    assert not over_declared_present, (
        f"Slugs declared expected-unminted but now resolving: "
        f"{sorted(over_declared_present)}. Tighten EXPECTED_UNMINTED."
    )


def test_pedagogy_unminted_set_matches_phase_2_6_plan(
    pedagogy_relation_types: frozenset[str],
) -> None:
    """Lock the unminted set to exactly the 9 slugs Phase 2.6 plans to
    mint. If the fixture grows a new unminted relation_type, this
    test catches it before downstream SHACL gates see drift."""
    actually_unresolved = {
        rt for rt in pedagogy_relation_types if lookup_iri(rt) is None
    }
    assert actually_unresolved == set(EXPECTED_UNMINTED), (
        f"Pedagogy unresolved-set drift. Got {sorted(actually_unresolved)}, "
        f"expected {sorted(EXPECTED_UNMINTED)}."
    )
