"""Phase 2.2 tests: RDF-backed slug aliasing.

Pins the contract of :mod:`lib.ontology.aliases` and verifies parity
with the legacy :data:`lib.ontology.concept_classifier.KNOWN_EQUIVALENT_ALIASES`
transition cache.

Property-style coverage:

- **Symmetry** — every term in an equivalence class canonicalizes to
  the same canonical form regardless of which surface form the call
  starts from.
- **Transitivity** — if ``A ↔ B`` and ``B ↔ C`` are declared in the
  Turtle file, then ``canonicalize(A) == canonicalize(C)`` even though
  no direct ``A ↔ C`` triple exists. (rdfs:subPropertyOf for
  equivalentProperty is symmetric+transitive per RDF Semantics.)
- **Pass-through** — unknown slugs return themselves unchanged.
- **Parity** — every entry in the legacy dict resolves to the same
  canonical via the Turtle path, so the dict can be removed once this
  test stays green across corpora.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.ontology import aliases
from lib.ontology.concept_classifier import (
    KNOWN_EQUIVALENT_ALIASES,
    canonicalize_alias,
)


# Skip the whole module if rdflib is unavailable — the Turtle path is
# unreachable in that case and the legacy dict still services callers.
rdflib = pytest.importorskip("rdflib")


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Reset the module-level cache before each test.

    Tests that monkeypatch the TTL path or otherwise force a re-parse
    rely on the cache being empty at entry.
    """
    aliases.reload_cache()
    yield
    aliases.reload_cache()


# ---------------------------------------------------------------------------
# Sanity: file actually exists and parses.
# ---------------------------------------------------------------------------


def test_aliases_ttl_file_exists():
    ttl_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "schemas"
        / "context"
        / "aliases.ttl"
    )
    assert ttl_path.exists(), f"Expected aliases TTL at {ttl_path}"


def test_module_loads_some_aliases():
    # If the cache loads cleanly and the Turtle file has at least one
    # equivalence pair, calling canonicalize on a known surface variant
    # must return a different slug than the input.
    assert aliases.canonicalize("rdfxml") == "rdf-xml"


# ---------------------------------------------------------------------------
# Symmetry — round-trip from any class member lands at the same canonical.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "members",
    [
        ("rdfxml", "rdf-xml"),
        ("jsonld", "json-ld"),
        ("ntriples", "n-triples"),
        ("nquads", "n-quads"),
        ("ttl", "turtle"),
        ("rdf-schema", "rdfs"),
        ("web-ontology-language", "owl"),
        ("shapes-constraint-language", "shacl"),
        ("owlsameas", "owl-sameas", "sameas", "same-as"),
    ],
)
def test_symmetry_all_members_canonicalize_alike(members):
    canonicalized = {aliases.canonicalize(m) for m in members}
    assert (
        len(canonicalized) == 1
    ), f"Class {members} split into multiple canonicals: {canonicalized}"


# ---------------------------------------------------------------------------
# Transitivity — the same-as family has 4 surface forms, so the union-find
# closure must collapse them to a single class even though only pairwise
# triples are declared.
# ---------------------------------------------------------------------------


def test_transitivity_three_member_class_resolves():
    # The same-as family: owlsameas, owl-sameas, sameas all declared as
    # owl:equivalentProperty same-as. By transitivity owlsameas and
    # sameas must both canonicalize to same-as without a direct
    # owlsameas <-> sameas declaration.
    assert aliases.canonicalize("owlsameas") == "same-as"
    assert aliases.canonicalize("sameas") == "same-as"
    # And the two non-canonical members agree with each other.
    assert aliases.canonicalize("owlsameas") == aliases.canonicalize("sameas")


def test_transitivity_synthetic_chain(tmp_path, monkeypatch):
    """Inject a synthetic 3-link chain (A ↔ B ↔ C) and verify closure.

    Guards against a regression where the closure walk only handles
    direct pairs. The Turtle file currently has no 3-deep chain (the
    same-as family is a 1-hop star), so this test seeds one explicitly.
    """
    synthetic_ttl = tmp_path / "chain.ttl"
    synthetic_ttl.write_text(
        """
        @prefix owl:    <http://www.w3.org/2002/07/owl#> .
        @prefix ed4all: <https://ed4all.io/vocab/> .

        ed4all:alpha   owl:equivalentProperty ed4all:beta .
        ed4all:beta    owl:equivalentProperty ed4all:gamma .
        ed4all:alpha   <https://ed4all.io/vocab/isCanonical> true .
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(aliases, "_TTL_PATH", synthetic_ttl)
    aliases.reload_cache()

    assert aliases.canonicalize("alpha") == "alpha"
    assert aliases.canonicalize("beta") == "alpha"
    assert aliases.canonicalize("gamma") == "alpha"


# ---------------------------------------------------------------------------
# Pass-through — unknown slug returns itself.
# ---------------------------------------------------------------------------


def test_unknown_slug_passes_through():
    assert aliases.canonicalize("rdf-graph") == "rdf-graph"
    assert aliases.canonicalize("blank-node") == "blank-node"
    assert aliases.canonicalize("sparql-select") == "sparql-select"


def test_empty_input_passes_through():
    assert aliases.canonicalize("") == ""
    # The function is documented as ``-> str``, so we don't test None
    # — that's the caller's contract to enforce.


def test_case_insensitive_lookup():
    # The legacy dict lowercased the input; we keep that behavior so
    # call-site contracts don't break.
    assert aliases.canonicalize("RDFXML") == "rdf-xml"
    assert aliases.canonicalize("Turtle") == "turtle"  # canonical = "turtle"


# ---------------------------------------------------------------------------
# Parity — every dict entry resolves identically via the Turtle path.
#
# When this test passes for the entire dict, the dict can be removed
# from concept_classifier (Phase 2.2 follow-up).
# ---------------------------------------------------------------------------


def test_known_aliases_dict_parity():
    """Every entry in KNOWN_EQUIVALENT_ALIASES maps to the same canonical
    via the new Turtle path."""
    mismatches = []
    for surface, expected_canonical in KNOWN_EQUIVALENT_ALIASES.items():
        actual = aliases.canonicalize(surface)
        if actual != expected_canonical:
            mismatches.append(
                f"  {surface!r}: dict={expected_canonical!r}, ttl={actual!r}"
            )
    assert not mismatches, (
        "Turtle path disagrees with the legacy dict for these entries:\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# Integration — concept_classifier.canonicalize_alias still honors the
# original contract after the refactor.
# ---------------------------------------------------------------------------


def test_canonicalize_alias_external_surface_unchanged():
    # All the assertions from the original test_concept_classifier.py
    # still hold — canonicalize_alias is the external surface that
    # downstream callers (LibV2, Trainforge, the wave76 cleanup script)
    # depend on.
    assert canonicalize_alias("rdfxml") == "rdf-xml"
    assert canonicalize_alias("ttl") == "turtle"
    assert canonicalize_alias("rdf-schema") == "rdfs"
    assert canonicalize_alias("web-ontology-language") == "owl"
    assert canonicalize_alias("shapes-constraint-language") == "shacl"
    assert canonicalize_alias("owlsameas") == "same-as"
    # Pass-through preserved.
    assert canonicalize_alias("blank-node") == "blank-node"
    assert canonicalize_alias("") == ""


def test_canonicalize_alias_falls_back_to_dict_when_ttl_empty(monkeypatch):
    """If the Turtle path returns the slug unchanged, the dict serves."""
    # Force the cache empty by pointing _TTL_PATH at a nonexistent file
    # and reloading. The aliases module fails open (returns {}), so
    # canonicalize() always returns the slug unchanged. The dict
    # fallback in canonicalize_alias must then take over.
    monkeypatch.setattr(aliases, "_TTL_PATH", Path("/nonexistent.ttl"))
    aliases.reload_cache()

    # Sanity: the Turtle path is now a no-op.
    assert aliases.canonicalize("ttl") == "ttl"

    # But the dict fallback still works.
    assert canonicalize_alias("ttl") == "turtle"
    assert canonicalize_alias("rdfxml") == "rdf-xml"


# ---------------------------------------------------------------------------
# Phase 2.5 — Cross-namespace vocabulary bridges.
#
# Bridges are declared in aliases.ttl between IRIs in two namespaces:
#   * ed4all: = https://ed4all.io/vocab/        (JSON-LD @context surface)
#   * cf:     = https://ed4all.dev/ns/courseforge/v1#  (RDFS/OWL surface)
# When both IRIs share the same local name (Concept, hasPrerequisite,
# bloomLevel, etc.), the bridge axiom unions them in the closure walk
# so a canonicalize() call returns a single deterministic slug
# regardless of which side the input came from. The slug extractor in
# lib/ontology/aliases.py::_slug_from_iri accepts both namespaces and
# projects each IRI to its bare local name; the union-find driver
# walks both owl:equivalentProperty AND owl:equivalentClass so the
# class bridges (Phase 2.5) and the predicate bridges (Phase 2.2 +
# Phase 2.5) collapse into the same equivalence-class structure.
# ---------------------------------------------------------------------------


def test_cross_namespace_class_bridge():
    """``ed4all:Concept`` and ``cf:Concept`` resolve to the same canonical.

    Both IRIs share the local name ``Concept``. The
    ``owl:equivalentClass`` bridge in aliases.ttl unions them; the
    closure walker then assigns both a single canonical slug. Because
    no member of this two-element class is annotated
    ``ed4all:isCanonical true``, the lex-tiebreaker picks the smaller
    string — both members ARE the same string ``Concept``, so that's
    the canonical regardless. The load-bearing assertion is that the
    bridge actually hit the cache (i.e., ``Concept`` is present as a
    cache key) — verifying with a second class-bridge entry that the
    parser walked owl:equivalentClass at all.
    """
    cache = aliases._load_cache()

    # Every parallel-class local name from the bridge axioms must land
    # in the closure cache. If the parser only walked
    # owl:equivalentProperty (not owl:equivalentClass), or only
    # accepted ed4all: IRIs (not cf:), these keys would be absent.
    for local_name in (
        "Concept",
        "LearningObjective",
        "Misconception",
        "Chunk",
        "TargetedConcept",
    ):
        assert local_name in cache, (
            f"Cross-namespace class bridge for {local_name!r} missing "
            f"from canonicalization cache — parser likely skipped "
            f"owl:equivalentClass or rejected the cf: namespace."
        )

    # Symmetry: ed4all-side and cf-side inputs canonicalize to the
    # same slug. The two IRIs share a local name, so this collapses
    # to "the same input produces the same output" — the meaningful
    # content of the assertion is that the bridge WAS exercised by
    # virtue of the local name appearing as a cache key.
    assert aliases.canonicalize("Concept") == aliases.canonicalize("Concept")
    assert aliases.canonicalize("Concept") in {"Concept", "concept"}


def test_cross_namespace_property_bridge():
    """ed4all and cf predicate IRIs sharing a local name canonicalize alike."""
    cache = aliases._load_cache()

    # Spot-check the 13 predicate bridges declared in aliases.ttl.
    bridged_predicates = (
        "bloomLevel",
        "bloomVerb",
        "cognitiveDomain",
        "hierarchyLevel",
        "targetsConcept",
        "hasMisconception",
        "hasPrerequisite",
        "isDefinedBy",
        "isDerivedFromObjective",
        "exemplifiedBy",
        "isMisconceptionOf",
        "assessesObjective",
        "correction",
    )
    for local_name in bridged_predicates:
        assert local_name in cache, (
            f"Cross-namespace property bridge for {local_name!r} missing "
            f"from canonicalization cache — closure walk did not pick "
            f"up the owl:equivalentProperty axiom."
        )
        # Idempotence — calling canonicalize twice yields the same
        # output. This guards against a regression where the closure
        # depends on caller iteration order.
        first = aliases.canonicalize(local_name)
        second = aliases.canonicalize(local_name)
        assert first == second


def test_cross_namespace_synthetic_class_bridge(tmp_path, monkeypatch):
    """Inject a synthetic class bridge with DISTINCT local names per side.

    Validates that the parser actually unions across the two
    namespaces — using distinct local names removes the "same input,
    same output" tautology that the natural aliases.ttl bridges
    collapse to. We declare ``ed4all:fooBar owl:equivalentClass
    cf:bazQux`` with ``ed4all:fooBar`` annotated canonical, and
    require that ``canonicalize("bazQux")`` returns ``"fooBar"``.
    """
    synthetic_ttl = tmp_path / "synthetic_class_bridge.ttl"
    synthetic_ttl.write_text(
        """
        @prefix owl:    <http://www.w3.org/2002/07/owl#> .
        @prefix ed4all: <https://ed4all.io/vocab/> .
        @prefix cf:     <https://ed4all.dev/ns/courseforge/v1#> .

        ed4all:fooBar  owl:equivalentClass cf:bazQux .
        ed4all:fooBar  <https://ed4all.io/vocab/isCanonical> true .
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(aliases, "_TTL_PATH", synthetic_ttl)
    aliases.reload_cache()

    # Both local names must collapse to the canonical-flagged side.
    assert aliases.canonicalize("fooBar") == "fooBar"
    assert aliases.canonicalize("bazQux") == "fooBar"


def test_cross_namespace_synthetic_property_bridge(tmp_path, monkeypatch):
    """Inject a synthetic predicate bridge with DISTINCT local names.

    Same shape as the synthetic class-bridge test but uses
    ``owl:equivalentProperty`` to confirm that the predicate-bridge
    code path also unions across the ed4all/cf split.
    """
    synthetic_ttl = tmp_path / "synthetic_property_bridge.ttl"
    synthetic_ttl.write_text(
        """
        @prefix owl:    <http://www.w3.org/2002/07/owl#> .
        @prefix ed4all: <https://ed4all.io/vocab/> .
        @prefix cf:     <https://ed4all.dev/ns/courseforge/v1#> .

        ed4all:hasFoo  owl:equivalentProperty cf:hasBar .
        cf:hasBar      <https://ed4all.io/vocab/isCanonical> true .
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(aliases, "_TTL_PATH", synthetic_ttl)
    aliases.reload_cache()

    # Canonical-flagged side wins regardless of which namespace it
    # lives in — the isCanonical annotation works the same on cf:
    # IRIs as on ed4all: IRIs.
    assert aliases.canonicalize("hasBar") == "hasBar"
    assert aliases.canonicalize("hasFoo") == "hasBar"


def test_existing_alias_families_unchanged():
    """Parity test: every entry in the legacy KNOWN_EQUIVALENT_ALIASES
    dict resolves identically through the post-bridge Turtle path.

    Phase 2.5 added owl:equivalentClass walking and cf: namespace
    acceptance. This test guards against a regression where those
    additions would interfere with Phase 2.2's slug-alias families
    (the lowercased pairs like ``rdfxml`` ↔ ``rdf-xml``). The aliases
    families and the cross-namespace bridges live in the same
    aliases.ttl file and are walked in the same union-find pass, so
    a parser bug that mishandles one would silently corrupt the
    other.

    This is a stricter sibling of ``test_known_aliases_dict_parity``
    above — that test established the baseline; this re-asserts it
    after the Phase 2.5 changes landed, so reviewers see an explicit
    "no regression" gate that names the right wave.
    """
    mismatches = []
    for surface, expected_canonical in KNOWN_EQUIVALENT_ALIASES.items():
        actual = aliases.canonicalize(surface)
        if actual != expected_canonical:
            mismatches.append(
                f"  {surface!r}: dict={expected_canonical!r}, ttl={actual!r}"
            )
    assert not mismatches, (
        "Phase 2.5 cross-namespace bridges broke parity with the legacy "
        "KNOWN_EQUIVALENT_ALIASES dict for these entries:\n"
        + "\n".join(mismatches)
    )
