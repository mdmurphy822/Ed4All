"""Phase 5 equivalence tests — Python `defined_by_from_first_mention`
versus SHACL-AF `:DefinedByRule` in
``schemas/context/courseforge_v1.shacl-rules.ttl``.

The contract pinned here:

* On any concept graph that the Python rule can process, the SHACL
  runner produces the SAME edge list (modulo deterministic
  ``(source, target)`` ordering, which both paths sort by anyway).
* Concepts with no ``occurrences[]`` produce zero edges on both
  paths.
* Concepts with multiple occurrences produce one edge whose target
  equals ``sorted(occurrences)[0]`` on both paths.
* When ``TRAINFORGE_USE_SHACL_RULES`` is unset (or false), the
  orchestrator dispatches the Python rule and the SHACL runner is
  silent — no SHACL evaluation, no IRI minting, no rdflib import on
  the hot path.

Without these tests the Phase 5 plan can't claim equivalence; with
them the plan can flip the flag default in a later wave once the rest
of the inference rules port over.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest

from Trainforge.rag.inference_rules import defined_by_from_first_mention as py_rule
from Trainforge.rag import shacl_rule_runner

# Skip the whole module when pyshacl/rdflib aren't importable. Phase 5
# is opt-in; callers without the optional dep stack get the Python rule
# path and that path has its own coverage already.
pyshacl = pytest.importorskip("pyshacl")
rdflib = pytest.importorskip("rdflib")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _force_shacl_on(monkeypatch):
    """Pin TRAINFORGE_USE_SHACL_RULES=true for the duration of a test.

    The runner captures the env var at import time, so monkeypatching
    the module attribute is the canonical way to flip it inside a test
    without reloading the module (which would also reset other module
    state).
    """
    monkeypatch.setattr(shacl_rule_runner, "USE_SHACL_RULES", True)


def _build_concept_graph(*nodes: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a tuple of node dicts in the concept-graph envelope shape
    that both the Python rule and the SHACL runner accept.
    """
    return {"kind": "concept", "nodes": list(nodes)}


def _normalize_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip stamps that the orchestrator (not the rule) injects.

    Both paths produce the same shape *before* the orchestrator's
    ``_stamp_provenance`` runs. The Python rule emits no ``run_id`` /
    ``created_at``; the SHACL runner is the same. Sorting is already
    canonical inside both paths but we re-sort defensively.
    """
    return sorted(edges, key=lambda e: (e["source"], e["target"]))


# ---------------------------------------------------------------------------
# Test 1 — basic equivalence on a non-trivial fixture
# ---------------------------------------------------------------------------


def test_python_and_shacl_produce_identical_edges(monkeypatch):
    """The full equivalence assertion: same input → same output."""
    _force_shacl_on(monkeypatch)

    cg = _build_concept_graph(
        {"id": "alpha", "label": "Alpha", "frequency": 3,
         "occurrences": ["c_00001", "c_00003", "c_00007"]},
        {"id": "beta", "label": "Beta", "frequency": 2,
         "occurrences": ["c_00002", "c_00005"]},
        {"id": "gamma", "label": "Gamma", "frequency": 1,
         "occurrences": ["c_00010"]},
    )

    py_edges = _normalize_edges(py_rule.infer([], None, cg))
    shacl_edges = _normalize_edges(
        shacl_rule_runner.shacl_defined_by_edges([], None, cg)
    )

    assert py_edges == shacl_edges, (py_edges, shacl_edges)
    assert len(py_edges) == 3
    # Verify the targets are the lex-min of each occurrences list.
    by_source = {e["source"]: e for e in py_edges}
    assert by_source["alpha"]["target"] == "c_00001"
    assert by_source["beta"]["target"] == "c_00002"
    assert by_source["gamma"]["target"] == "c_00010"


# ---------------------------------------------------------------------------
# Test 2 — concept with no occurrences produces no edge on either path
# ---------------------------------------------------------------------------


def test_no_occurrences_produces_no_edge(monkeypatch):
    _force_shacl_on(monkeypatch)

    cg = _build_concept_graph(
        {"id": "lonely", "label": "Lonely", "frequency": 0},
        {"id": "empty", "label": "Empty", "frequency": 0,
         "occurrences": []},
        # Mix in a regular concept so we know SHACL fires at all.
        {"id": "real", "label": "Real", "frequency": 1,
         "occurrences": ["c_00099"]},
    )

    py_edges = _normalize_edges(py_rule.infer([], None, cg))
    shacl_edges = _normalize_edges(
        shacl_rule_runner.shacl_defined_by_edges([], None, cg)
    )

    assert py_edges == shacl_edges
    # Only "real" should produce an edge; "lonely" / "empty" are silent.
    assert [e["source"] for e in py_edges] == ["real"]


# ---------------------------------------------------------------------------
# Test 3 — multiple occurrences pick the same first-mention on both paths
# ---------------------------------------------------------------------------


def test_multiple_occurrences_same_first_mention(monkeypatch):
    """Defensively unsorted occurrences[] — the Python rule re-sorts at
    emit time; the SHACL rule's MIN(?occ) over the multi-valued
    property arrives at the same lex-min. Both must agree."""
    _force_shacl_on(monkeypatch)

    cg = _build_concept_graph(
        # Note the deliberately unsorted occurrences arrays.
        {"id": "mixed", "label": "Mixed", "frequency": 4,
         "occurrences": ["c_00050", "c_00010", "c_00030", "c_00020"]},
        {"id": "duplicates", "label": "Dup", "frequency": 3,
         # Even with non-unique entries (which shouldn't normally
         # happen but the input shape doesn't forbid it), both rules
         # must agree on the lex-min.
         "occurrences": ["c_00100", "c_00099", "c_00100"]},
    )

    py_edges = _normalize_edges(py_rule.infer([], None, cg))
    shacl_edges = _normalize_edges(
        shacl_rule_runner.shacl_defined_by_edges([], None, cg)
    )

    assert py_edges == shacl_edges
    by_source = {e["source"]: e for e in py_edges}
    assert by_source["mixed"]["target"] == "c_00010"
    assert by_source["duplicates"]["target"] == "c_00099"


# ---------------------------------------------------------------------------
# Test 4 — flag-off behavior: orchestrator uses Python rule, SHACL silent
# ---------------------------------------------------------------------------


def test_flag_off_uses_python_rule_orchestrator():
    """When TRAINFORGE_USE_SHACL_RULES is unset, the orchestrator must
    dispatch the Python rule. The SHACL runner must return [] when
    called directly (the safe-fallthrough contract of the runner)."""
    # Make sure we're observing the module-level default. We do NOT
    # monkeypatch USE_SHACL_RULES here — that's the whole point.
    assert shacl_rule_runner.USE_SHACL_RULES is False, (
        "Phase 5 default must remain off; another test failed to clean up."
    )

    cg = _build_concept_graph(
        {"id": "alpha", "label": "Alpha", "frequency": 1,
         "occurrences": ["c_00001"]},
    )

    # The runner returns [] when the flag is off (silent fallthrough).
    shacl_edges = shacl_rule_runner.shacl_defined_by_edges([], None, cg)
    assert shacl_edges == []

    # The Python rule still works as usual.
    py_edges = py_rule.infer([], None, cg)
    assert len(py_edges) == 1
    assert py_edges[0]["target"] == "c_00001"

    # And the orchestrator-level dispatch picks the Python rule when
    # the flag is off. We verify by importing the orchestrator and
    # asserting that, with the flag off, build_semantic_graph emits
    # the same defined-by edges the Python rule produces directly.
    from Trainforge.rag import typed_edge_inference

    # Defensive: reload not needed because USE_SHACL_RULES is already
    # off — but re-importing the runner from typed_edge_inference's
    # namespace lets us assert the module-level reference is to the
    # same object.
    assert typed_edge_inference._shacl_runner is shacl_rule_runner

    semantic = typed_edge_inference.build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph=cg,
    )
    defined_by_edges = [
        e for e in semantic["edges"] if e["type"] == "defined-by"
    ]
    assert len(defined_by_edges) == 1
    assert defined_by_edges[0]["source"] == "alpha"
    assert defined_by_edges[0]["target"] == "c_00001"
    # And the rule provenance carries the Python rule's name (which is
    # the same name the SHACL path borrows from the module — both
    # paths agree on identity, which is the whole point).
    assert defined_by_edges[0]["provenance"]["rule"] == py_rule.RULE_NAME


# ---------------------------------------------------------------------------
# Test 5 — flag-on orchestrator equivalence
# ---------------------------------------------------------------------------


def test_flag_on_orchestrator_emits_equivalent_edges(monkeypatch):
    """End-to-end check: with the flag flipped, the orchestrator must
    produce the same defined-by edges as the Python rule path would
    have produced on the same input. This is the operational
    equivalence test (the prior tests pin the rule-call equivalence)."""
    from Trainforge.rag import typed_edge_inference

    cg = _build_concept_graph(
        {"id": "alpha", "label": "Alpha", "frequency": 2,
         "occurrences": ["c_00010", "c_00002"]},
        {"id": "beta", "label": "Beta", "frequency": 1,
         "occurrences": ["c_00005"]},
    )

    # Capture the Python-path output first, BEFORE flipping the flag,
    # so the comparison is deterministic.
    semantic_py = typed_edge_inference.build_semantic_graph(
        chunks=[], course=None, concept_graph=cg,
    )
    py_defined_by = sorted(
        (e for e in semantic_py["edges"] if e["type"] == "defined-by"),
        key=lambda e: (e["source"], e["target"]),
    )

    # Flip flag and re-run the orchestrator.
    _force_shacl_on(monkeypatch)
    semantic_shacl = typed_edge_inference.build_semantic_graph(
        chunks=[], course=None, concept_graph=cg,
    )
    shacl_defined_by = sorted(
        (e for e in semantic_shacl["edges"] if e["type"] == "defined-by"),
        key=lambda e: (e["source"], e["target"]),
    )

    # Strip orchestrator-injected stamps (run_id / created_at) so we
    # compare only the rule-emitted shape. The orchestrator stamps
    # both paths identically, so the only differences would come from
    # the rule emit itself — which is what this test pins.
    def _strip(edges):
        out = []
        for e in edges:
            stripped = {k: v for k, v in e.items() if k not in ("run_id", "created_at")}
            out.append(stripped)
        return out

    assert _strip(py_defined_by) == _strip(shacl_defined_by)


# ---------------------------------------------------------------------------
# Test 6 — sanity: the SHACL rule uses the canonical ed4all:isDefinedBy IRI
# ---------------------------------------------------------------------------


def test_shacl_rule_uses_canonical_predicate():
    """The runner must reuse the Phase 2.1 ed4all:isDefinedBy predicate
    (declared in courseforge_v1.vocabulary.ttl) rather than mint a
    parallel one. We verify by parsing the rules TTL and asserting
    that ed4all:isDefinedBy appears in the CONSTRUCT body and that
    the rule's targetClass is ed4all:Concept."""
    rules_path = shacl_rule_runner.SHACL_RULES_PATH
    body = rules_path.read_text(encoding="utf-8")
    assert "ed4all:isDefinedBy" in body
    assert "ed4all:Concept" in body
    assert "sh:targetClass ed4all:Concept" in body
