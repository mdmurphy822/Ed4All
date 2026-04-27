"""sh:condition gating on the Phase 5 SHACL-AF rules.

The Wave 85 ``sh:SPARQLRule`` prototype in
``schemas/context/courseforge_v1.shacl-rules.ttl`` was unconditional —
the SPARQL CONSTRUCT body fired against every focus node selected by
``sh:targetClass``, even when the input data was shape-incomplete (no
``ed4all:occurrence`` triples). This test pins the post-improvement
behavior:

* The ``defined_by_from_first_mention`` rule carries an
  ``sh:condition`` requiring at least one ``ed4all:occurrence`` per
  focus node.
* When the focus node is shape-incomplete (zero occurrences),
  pyshacl skips it: NO ``ed4all:isDefinedBy`` triple is derived.
* When the focus node has at least one occurrence, the rule fires
  exactly as before — the behavior under the canonical case is
  unchanged.

Falls back to a declaration-only check when pyshacl/rdflib aren't
importable (the same convention the existing
``test_shacl_rules_defined_by.py`` uses).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Trainforge.rag import shacl_rule_runner

# Skip the whole module when pyshacl/rdflib aren't importable. Phase 5
# is opt-in; callers without the optional dep stack run the Python rule
# path which carries its own coverage.
pyshacl = pytest.importorskip("pyshacl")
rdflib = pytest.importorskip("rdflib")

from rdflib import Graph, Literal, Namespace, URIRef  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = PROJECT_ROOT / "schemas" / "context" / "courseforge_v1.shacl-rules.ttl"

ED4ALL = Namespace("https://ed4all.dev/ns/courseforge/v1#")
SH = Namespace("http://www.w3.org/ns/shacl#")
CONCEPT_BASE = "https://ed4all.io/concept/"
CHUNK_BASE = "https://ed4all.io/chunk/"


# ---------------------------------------------------------------------------
# 1. Declaration: the rule carries an authored sh:condition
# ---------------------------------------------------------------------------


def test_defined_by_rule_declares_sh_condition() -> None:
    """Parse the rules TTL and confirm the rule node carries an
    ``sh:condition`` sub-shape requiring ``ed4all:occurrence
    sh:minCount 1``."""
    g = Graph()
    g.parse(RULES_PATH, format="turtle")

    rule_holder = ED4ALL.DefinedByRule
    rules = list(g.objects(rule_holder, SH.rule))
    assert rules, "DefinedByRule must declare at least one sh:rule"

    rule = rules[0]
    conditions = list(g.objects(rule, SH.condition))
    assert conditions, (
        "Phase 5 sh:SPARQLRule must declare sh:condition so pyshacl "
        "skips shape-incomplete focus nodes (zero ed4all:occurrence)."
    )

    # The condition's property block must require minCount >= 1 on
    # ed4all:occurrence — the rule's known minimum input expectation.
    cond = conditions[0]
    found_min_count = False
    for prop in g.objects(cond, SH.property):
        paths = set(g.objects(prop, SH.path))
        min_counts = list(g.objects(prop, SH.minCount))
        if ED4ALL.occurrence in paths and min_counts:
            for mc in min_counts:
                # rdflib parses xsd:integer literals as Python ints.
                if int(mc) >= 1:
                    found_min_count = True
                    break
        if found_min_count:
            break

    assert found_min_count, (
        "sh:condition must require ed4all:occurrence sh:minCount 1 — "
        "the rule's canonical input precondition."
    )


# ---------------------------------------------------------------------------
# 2. Behavior: shape-incomplete focus node → rule does NOT fire
# ---------------------------------------------------------------------------


def _build_data_graph_with_occurrences(slug: str, occurrences: list) -> Graph:
    """Mirror Trainforge.rag.shacl_rule_runner._build_data_graph for a
    single concept. Lets the test materialize the exact slice the
    runner would feed pyshacl."""
    g = Graph()
    g.bind("ed4all", ED4ALL)
    subj = URIRef(f"{CONCEPT_BASE}{slug}")
    g.add((subj, RDF.type, ED4ALL.Concept))
    for occ in occurrences:
        g.add((subj, ED4ALL.occurrence, Literal(occ)))
    return g


def _run_rules(data_graph: Graph) -> Graph:
    shapes_graph = Graph()
    shapes_graph.parse(RULES_PATH, format="turtle")

    pyshacl.validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        advanced=True,
        inplace=True,
        abort_on_first=False,
        meta_shacl=False,
        js=False,
        debug=False,
    )
    return data_graph


def test_no_occurrences_means_no_derived_triple() -> None:
    """A Concept with zero ed4all:occurrence triples is a shape-
    incomplete focus node. With sh:condition gating, pyshacl skips it
    and no ed4all:isDefinedBy triple is derived."""
    data = _build_data_graph_with_occurrences("alpha", occurrences=[])

    triples_before = set(data.triples((None, ED4ALL.isDefinedBy, None)))
    assert not triples_before, "fixture must start with no isDefinedBy triples"

    _run_rules(data)

    derived = list(data.triples((None, ED4ALL.isDefinedBy, None)))
    assert derived == [], (
        "sh:condition must skip the focus node when ed4all:occurrence "
        "is empty — got "
        f"{[(str(s), str(o)) for s, _, o in derived]} instead."
    )


def test_one_occurrence_makes_the_rule_fire() -> None:
    """The complement: with a single occurrence present, the rule
    fires and emits exactly one ed4all:isDefinedBy triple pointing at
    the lex-min occurrence (which is the only one)."""
    data = _build_data_graph_with_occurrences("alpha", occurrences=["c_00001"])

    _run_rules(data)

    derived = list(data.triples((None, ED4ALL.isDefinedBy, None)))
    assert len(derived) == 1, (
        f"Expected exactly one ed4all:isDefinedBy triple after the rule "
        f"fires; got {len(derived)}: "
        f"{[(str(s), str(o)) for s, _, o in derived]}"
    )
    s, _, o = derived[0]
    assert str(s) == f"{CONCEPT_BASE}alpha"
    # The CONSTRUCT body's ?firstChunk is bound to MIN(?occ); the
    # value is the literal we added, "c_00001".
    assert str(o) == "c_00001"


def test_adding_occurrence_changes_outcome_within_same_test() -> None:
    """Builds a node with no occurrences, runs the rule (expect zero
    derived triples), adds a single occurrence in a fresh data graph,
    re-runs (expect one derived triple). Pins the round-trip
    behavior of the gate."""
    # First run: empty occurrences → no derived triple.
    empty_data = _build_data_graph_with_occurrences("beta", occurrences=[])
    _run_rules(empty_data)
    assert list(empty_data.triples((None, ED4ALL.isDefinedBy, None))) == []

    # Second run on a separate graph: one occurrence → one derived
    # triple. We use a fresh graph because pyshacl's inplace=True
    # mutates the input; reusing empty_data would conflate the runs.
    populated_data = _build_data_graph_with_occurrences(
        "beta", occurrences=["c_00050"]
    )
    _run_rules(populated_data)
    derived = list(populated_data.triples((None, ED4ALL.isDefinedBy, None)))
    assert len(derived) == 1
    assert str(derived[0][2]) == "c_00050"


# ---------------------------------------------------------------------------
# 3. End-to-end via the runner: empty occurrences → empty edge list
# ---------------------------------------------------------------------------


def test_runner_emits_no_edges_for_concepts_without_occurrences(monkeypatch) -> None:
    """The orchestrator-facing surface stays equivalent to the Python
    rule when the gate fires (both produce an empty edge list for a
    concept with no occurrences). Pins the runner-level contract."""
    monkeypatch.setattr(shacl_rule_runner, "USE_SHACL_RULES", True)

    concept_graph = {
        "kind": "concept",
        "nodes": [
            # Node with empty occurrences[] — gate must skip.
            {"id": "lonely", "label": "Lonely", "frequency": 0,
             "occurrences": []},
            # Node with no occurrences key at all (legacy) — also skipped.
            {"id": "legacy", "label": "Legacy", "frequency": 0},
        ],
    }
    edges = shacl_rule_runner.shacl_defined_by_edges(
        chunks=[], course=None, concept_graph=concept_graph
    )
    assert edges == [], (
        f"Runner must emit no edges for concepts with empty/missing "
        f"occurrences[]; got {edges}"
    )


def test_runner_still_emits_for_normal_concepts(monkeypatch) -> None:
    """The gate must NOT regress the canonical case: a concept with
    occurrences still produces an edge."""
    monkeypatch.setattr(shacl_rule_runner, "USE_SHACL_RULES", True)

    concept_graph = {
        "kind": "concept",
        "nodes": [
            {"id": "real", "label": "Real", "frequency": 1,
             "occurrences": ["c_00099"]},
        ],
    }
    edges = shacl_rule_runner.shacl_defined_by_edges(
        chunks=[], course=None, concept_graph=concept_graph
    )
    assert len(edges) == 1, edges
    assert edges[0]["source"] == "real"
    assert edges[0]["target"] == "c_00099"
