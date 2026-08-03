"""W3.1 wiring regression — ``prerequisite_from_definition_mention`` dispatch.

The keystone producer (``lib/generation/prerequisite_from_definition_mention.py``)
is unit-tested in isolation. This suite pins its INTEGRATION into
``Trainforge.rag.typed_edge_inference.build_semantic_graph`` — the production
concept-graph dispatch that folds ``rule_versions`` into
``graph_build_hash`` / the downstream ``concept_graph_sha256``.

Two guarantees:

* Flag ON — a concept defined in ``TO_a``'s chunk and assumed-undefined in a
  different ``TO_b``'s chunk yields a federation ``TO_b --prerequisite--> TO_a``
  edge in the graph, stamped ``edge_kind="inferred"`` and carrying the rule's
  provenance; ``rule_versions`` gains the rule at its integer version.

* Flag OFF (default) — the rule is ABSENT from ``rule_versions``, produces no
  edge, and the serialized graph (hence ``concept_graph_sha256``) is
  byte-identical whether or not the new ``terminal_objectives`` /
  ``decision_capture`` inputs are threaded through. This is the byte-identity
  contract: registering the rule unconditionally would add a ``rule_versions``
  key even on a zero-edge run and change the hash.

Synthetic in-memory fixtures only — no torch / embeddings / NLI / model load,
no LibV2 mutation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from lib.generation.prerequisite_from_definition_mention import (
    RULE_NAME,
    RULE_VERSION,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph

_ENV = "TRAINFORGE_PREREQ_DEFINITION_MENTION"
_NOW = datetime(2026, 4, 20, tzinfo=timezone.utc)


def _inputs():
    """A concept defined in TO-01's chunk, assumed-undefined in TO-02's.

    ``occurrences[0]`` (ascending chunk-ID sort) is ``chunk_01`` → the
    definition chunk, owned by TO-01. ``chunk_02`` (owned by TO-02) mentions the
    same concept but does not define it → TO-02 depends on TO-01.
    """
    chunks = [
        {
            "id": "chunk_01",
            "source": {"course_id": "TST_905", "module_id": "m", "lesson_id": "l1"},
            "learning_outcome_refs": ["to-01"],
            "concept_tags": ["photosynthesis"],
            "chunk_type": "definition",
        },
        {
            "id": "chunk_02",
            "source": {"course_id": "TST_905", "module_id": "m", "lesson_id": "l2"},
            "learning_outcome_refs": ["to-02"],
            "concept_tags": ["photosynthesis"],
            "chunk_type": "explanation",
        },
    ]
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {
                "id": "photosynthesis",
                "label": "photosynthesis",
                "frequency": 2,
                "occurrences": ["chunk_01", "chunk_02"],
            },
            {
                "id": "chloroplast",
                "label": "chloroplast",
                "frequency": 1,
                "occurrences": ["chunk_01"],
            },
        ],
    }
    terminal_objectives = [
        {"id": "TO-01", "statement": "Explain photosynthesis."},
        {"id": "TO-02", "statement": "Apply photosynthesis to crop yield."},
    ]
    return chunks, concept_graph, terminal_objectives


def _graph_bytes(graph) -> bytes:
    """Mirror the downstream ``concept_graph_sha256`` serialization."""
    return json.dumps(graph, indent=2, ensure_ascii=False).encode("utf-8")


def _sha(graph) -> str:
    return hashlib.sha256(_graph_bytes(graph)).hexdigest()


def _prereq_edges(graph):
    return [
        e
        for e in graph.get("edges", [])
        if isinstance(e, dict)
        and (e.get("provenance") or {}).get("rule") == RULE_NAME
    ]


def test_flag_on_emits_to_prerequisite_edge_inferred(monkeypatch):
    monkeypatch.setenv(_ENV, "true")
    chunks, concept_graph, terminal_objectives = _inputs()

    graph = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=_NOW,
        terminal_objectives=terminal_objectives,
    )

    edges = _prereq_edges(graph)
    assert len(edges) == 1, f"expected exactly one definition-mention edge: {edges}"
    edge = edges[0]
    # source = dependent (assuming TO), target = prerequisite (defining TO).
    assert edge["source"] == "TO-02"
    assert edge["target"] == "TO-01"
    assert edge["type"] == "prerequisite"
    assert edge["edge_kind"] == "inferred"

    # The rule is folded into rule_versions at its integer version.
    assert graph["rule_versions"].get(RULE_NAME) == RULE_VERSION


def test_flag_off_absent_from_rule_versions_and_no_edge(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    chunks, concept_graph, terminal_objectives = _inputs()

    graph = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=_NOW,
        terminal_objectives=terminal_objectives,
    )

    assert RULE_NAME not in graph["rule_versions"]
    assert _prereq_edges(graph) == []


def test_flag_off_byte_identical_ignores_terminal_objectives(monkeypatch):
    """The OFF path must fully ignore the new ``terminal_objectives`` input so
    ``concept_graph_sha256`` is byte-identical to a legacy call that never
    threaded it (the pre-change behavior).

    NB: the pipeline (``_run_concept_extraction``) withholds the phase
    ``decision_capture`` entirely when the flag is off — passing one would
    stamp its ``run_id`` on every node/edge (a pre-existing
    ``build_semantic_graph`` behavior, unrelated to this rule), so the OFF-path
    contract is specifically: no capture, and ``terminal_objectives`` ignored.
    """
    monkeypatch.delenv(_ENV, raising=False)
    chunks, concept_graph, terminal_objectives = _inputs()

    graph_plain = build_semantic_graph(chunks, None, concept_graph, now=_NOW)
    graph_with_terminals = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=_NOW,
        terminal_objectives=terminal_objectives,
    )

    assert _sha(graph_plain) == _sha(graph_with_terminals)
    assert RULE_NAME not in graph_plain["rule_versions"]


def test_flag_toggles_the_only_difference(monkeypatch):
    """Flag ON vs OFF differ ONLY by the rule's edge + rule_versions key."""
    chunks, concept_graph, terminal_objectives = _inputs()

    monkeypatch.delenv(_ENV, raising=False)
    graph_off = build_semantic_graph(
        chunks, None, concept_graph, now=_NOW,
        terminal_objectives=terminal_objectives,
    )
    monkeypatch.setenv(_ENV, "1")
    graph_on = build_semantic_graph(
        chunks, None, concept_graph, now=_NOW,
        terminal_objectives=terminal_objectives,
    )

    off_versions = set(graph_off["rule_versions"])
    on_versions = set(graph_on["rule_versions"])
    assert on_versions - off_versions == {RULE_NAME}
    assert len(_prereq_edges(graph_on)) == 1
    assert _prereq_edges(graph_off) == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
