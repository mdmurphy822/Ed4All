"""Tests for OWL property characteristics on cf:hasPrerequisite / cf:isPrerequisiteOf.

The vocabulary file (``schemas/context/courseforge_v1.vocabulary.ttl``)
declares ``cf:hasPrerequisite`` and its inverse ``cf:isPrerequisiteOf``
as ``owl:TransitiveProperty`` paired via ``owl:inverseOf``. With those
declarations an OWL-RL reasoner derives the closure of a prerequisite
chain (A → B → C ⊨ A → C) and the inverse direction without SPARQL
property paths — every consumer querying prereq chains gets the
closure for free.

Sequencing-style edges (``cf:follows`` between modules) intentionally
remain non-transitive — only adjacent ties are emitted by design and
chained closure would entail false sequencing relationships.

Coverage:

* The transitive declaration triple lands for both predicates.
* The bidirectional ``owl:inverseOf`` pair is in the graph.
* When an OWL-RL reasoner is available (``rdflib`` ships
  ``owlrl``), a small 3-node chain entails the transitive closure.
  Otherwise the test falls back to the declaration-only checks
  (still meaningful — the closure is a downstream consumer
  responsibility).
* Negative: ``cf:follows`` must NOT be declared transitive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

rdflib = pytest.importorskip(
    "rdflib", reason="rdflib is required for OWL property characteristic checks."
)
from rdflib import Graph, Namespace, URIRef  # noqa: E402
from rdflib.namespace import OWL, RDF  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = PROJECT_ROOT / "schemas" / "context" / "courseforge_v1.vocabulary.ttl"

ED4ALL = Namespace("https://ed4all.dev/ns/courseforge/v1#")


@pytest.fixture(scope="module")
def vocab_graph() -> Graph:
    g = Graph()
    g.parse(VOCAB_PATH, format="turtle")
    return g


# ---------------------------------------------------------------------------
# 1. Declaration triples
# ---------------------------------------------------------------------------


def test_has_prerequisite_is_object_property(vocab_graph: Graph) -> None:
    assert (ED4ALL.hasPrerequisite, RDF.type, OWL.ObjectProperty) in vocab_graph


def test_has_prerequisite_is_transitive(vocab_graph: Graph) -> None:
    """Transitive characteristic lets an OWL-RL reasoner chain prereqs
    without SPARQL `+` property paths."""
    assert (ED4ALL.hasPrerequisite, RDF.type, OWL.TransitiveProperty) in vocab_graph, (
        "cf:hasPrerequisite must be declared owl:TransitiveProperty so chained "
        "prerequisite closure is derivable under OWL-RL reasoning."
    )


def test_is_prerequisite_of_is_object_property(vocab_graph: Graph) -> None:
    assert (ED4ALL.isPrerequisiteOf, RDF.type, OWL.ObjectProperty) in vocab_graph, (
        "cf:isPrerequisiteOf must be declared as the canonical inverse "
        "property of cf:hasPrerequisite."
    )


def test_is_prerequisite_of_is_transitive(vocab_graph: Graph) -> None:
    assert (ED4ALL.isPrerequisiteOf, RDF.type, OWL.TransitiveProperty) in vocab_graph


def test_inverse_of_pair_holds_in_both_directions(vocab_graph: Graph) -> None:
    """owl:inverseOf is symmetric in OWL semantics, but we author both
    directions so neither side depends on a reasoner for the simple
    inversion lookup."""
    assert (
        ED4ALL.hasPrerequisite,
        OWL.inverseOf,
        ED4ALL.isPrerequisiteOf,
    ) in vocab_graph, (
        "cf:hasPrerequisite owl:inverseOf cf:isPrerequisiteOf must be authored."
    )
    assert (
        ED4ALL.isPrerequisiteOf,
        OWL.inverseOf,
        ED4ALL.hasPrerequisite,
    ) in vocab_graph, (
        "cf:isPrerequisiteOf owl:inverseOf cf:hasPrerequisite must be authored."
    )


# ---------------------------------------------------------------------------
# 2. Negative: cf:follows must NOT be transitive
# ---------------------------------------------------------------------------


def test_follows_is_not_transitive(vocab_graph: Graph) -> None:
    """cf:follows captures adjacent-module sequencing only — making it
    transitive would entail week_01 follows week_03 from week_01 follows
    week_02 + week_02 follows week_03, which contradicts the design
    invariant in the vocabulary's rdfs:comment for cf:follows."""
    assert (ED4ALL.follows, RDF.type, OWL.TransitiveProperty) not in vocab_graph, (
        "cf:follows must NOT be declared owl:TransitiveProperty (sequencing "
        "is inherently directional, only adjacent ties are emitted by "
        "design — see vocabulary.ttl rdfs:comment on cf:follows)."
    )


# ---------------------------------------------------------------------------
# 3. End-to-end: build a 3-node chain and check transitive closure
# ---------------------------------------------------------------------------


def test_three_node_chain_yields_transitive_closure(vocab_graph: Graph) -> None:
    """Build A --hasPrerequisite--> B --hasPrerequisite--> C in a fresh
    graph, merge in the vocabulary, run OWL-RL if available, and assert
    that A --hasPrerequisite--> C is entailed.

    When ``owlrl`` (the OWL-RL reasoner that ships with rdflib's
    extras) isn't importable, falls back to the declaration-only
    assertion that the transitive characteristic IS declared — the
    closure step is a downstream consumer responsibility under that
    fallback.
    """
    base = "https://example.test/concept/"
    a, b, c = URIRef(f"{base}A"), URIRef(f"{base}B"), URIRef(f"{base}C")
    has_prereq = ED4ALL.hasPrerequisite

    data = Graph()
    # Bring the vocabulary in so the OWL reasoner sees the
    # owl:TransitiveProperty declaration.
    data.parse(VOCAB_PATH, format="turtle")
    data.add((a, has_prereq, b))
    data.add((b, has_prereq, c))

    try:
        import owlrl  # type: ignore[import-not-found]
    except ImportError:
        # No reasoner — declaration-only check is still meaningful.
        assert (has_prereq, RDF.type, OWL.TransitiveProperty) in data
        return

    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(data)

    assert (a, has_prereq, c) in data, (
        "OWL-RL closure failed to derive A hasPrerequisite C from the "
        "two-step chain. The owl:TransitiveProperty declaration on "
        "cf:hasPrerequisite isn't taking effect."
    )


def test_inverse_closure_under_owl_reasoning(vocab_graph: Graph) -> None:
    """When an OWL reasoner is available, A --hasPrerequisite--> B
    must entail B --isPrerequisiteOf--> A via the owl:inverseOf
    declaration. Falls back to the inverse-pair declaration check when
    no reasoner is importable."""
    base = "https://example.test/concept/"
    a, b = URIRef(f"{base}A"), URIRef(f"{base}B")

    data = Graph()
    data.parse(VOCAB_PATH, format="turtle")
    data.add((a, ED4ALL.hasPrerequisite, b))

    try:
        import owlrl  # type: ignore[import-not-found]
    except ImportError:
        assert (
            ED4ALL.hasPrerequisite,
            OWL.inverseOf,
            ED4ALL.isPrerequisiteOf,
        ) in data
        return

    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(data)

    assert (b, ED4ALL.isPrerequisiteOf, a) in data, (
        "OWL-RL closure failed to derive the inverse triple — the "
        "owl:inverseOf declaration isn't taking effect."
    )
