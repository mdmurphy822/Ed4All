"""Wave 11 — TRAINFORGE_SOURCE_PROVENANCE flag gates evidence-arm emission.

Contract locked by this suite:

- **Flag OFF (default)**: the 5 chunk-anchored evidence arms
  (``IsAEvidence``, ``ExemplifiesEvidence``, ``DerivedFromObjectiveEvidence``,
  ``DefinedByEvidence``, ``AssessesEvidence``) DO NOT carry
  ``source_references[]`` — output matches the pre-Wave-11 shape, even when
  chunks carry Wave-10 ``source.source_references[]``.
- **Flag ON**: each rule copies the originating chunk's
  ``source.source_references[]`` into the evidence arm. For chunk-anchored
  rules the originating chunk is the one stamped in the evidence. For
  ``AssessesEvidence`` the originating chunk is the one referenced by
  ``source_chunk_id`` on the question.
- **Legacy corpora (flag on, chunks carry no refs)**: evidence arms omit
  ``source_references`` — absence = unknown, per the additive discipline.
- **Abstract arms** (``PrerequisiteEvidence``, ``RelatedEvidence``,
  ``MisconceptionOfEvidence``) never emit ``source_references`` regardless
  of flag state (P4 deferral).

Flag toggling is implemented via monkeypatching the module-level
``SOURCE_PROVENANCE`` constant on each rule module (captured at import
time, mirroring the ``SCOPE_CONCEPT_IDS`` flag pattern).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.rag.inference_rules import (
    assesses_from_question_lo,
    defined_by_from_first_mention,
    derived_from_lo_ref,
    exemplifies_from_example_chunks,
    is_a_from_key_terms,
)

# --------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------- #


SAMPLE_REFS = [
    {"sourceId": "dart:science_of_learning#s5_p2", "role": "primary"},
    {"sourceId": "dart:science_of_learning#s6_p1", "role": "contributing"},
]


def _chunk_with_refs(chunk_id: str, **extras: Any) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "source": {
            "course_id": "SAMPLE_101",
            "module_id": "m",
            "lesson_id": "l",
            "source_references": [dict(r) for r in SAMPLE_REFS],
        },
        **extras,
    }


def _chunk_no_refs(chunk_id: str, **extras: Any) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "source": {
            "course_id": "SAMPLE_101",
            "module_id": "m",
            "lesson_id": "l",
        },
        **extras,
    }


@pytest.fixture
def flag_on(monkeypatch):
    """Enable TRAINFORGE_SOURCE_PROVENANCE for all 5 rule modules."""
    for mod in (
        is_a_from_key_terms,
        exemplifies_from_example_chunks,
        derived_from_lo_ref,
        defined_by_from_first_mention,
        assesses_from_question_lo,
    ):
        monkeypatch.setattr(mod, "SOURCE_PROVENANCE", True)
    yield


@pytest.fixture
def flag_off(monkeypatch):
    """Disable TRAINFORGE_SOURCE_PROVENANCE for all 5 rule modules."""
    for mod in (
        is_a_from_key_terms,
        exemplifies_from_example_chunks,
        derived_from_lo_ref,
        defined_by_from_first_mention,
        assesses_from_question_lo,
    ):
        monkeypatch.setattr(mod, "SOURCE_PROVENANCE", False)
    yield


def _graph(node_ids, *, occurrences_by_id=None):
    occurrences_by_id = occurrences_by_id or {}
    nodes = []
    for nid in node_ids:
        n = {"id": nid, "label": nid, "frequency": 2}
        if nid in occurrences_by_id:
            n["occurrences"] = list(occurrences_by_id[nid])
        nodes.append(n)
    return {"kind": "concept", "nodes": nodes, "edges": []}


# --------------------------------------------------------------------- #
# IsA rule
# --------------------------------------------------------------------- #


def _is_a_chunks(chunk_builder):
    return [
        chunk_builder(
            "chunk_is_a",
            key_terms=[
                {
                    "term": "cognitive load",
                    "definition": "Cognitive load is a type of mental effort.",
                }
            ],
        )
    ]


def test_is_a_flag_off_omits_source_references(flag_off):
    chunks = _is_a_chunks(_chunk_with_refs)
    graph = _graph(["cognitive-load", "mental-effort"])
    edges = is_a_from_key_terms.infer(chunks, None, graph)
    assert edges, "rule must still emit the is-a edge"
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_is_a_flag_on_copies_refs_from_originating_chunk(flag_on):
    chunks = _is_a_chunks(_chunk_with_refs)
    graph = _graph(["cognitive-load", "mental-effort"])
    edges = is_a_from_key_terms.infer(chunks, None, graph)
    assert edges
    for edge in edges:
        refs = edge["provenance"]["evidence"].get("source_references")
        assert refs == SAMPLE_REFS, (
            "is_a evidence must carry chunk's source_references"
        )


def test_is_a_flag_on_legacy_chunk_omits_refs(flag_on):
    """Flag on + chunk carries no refs → evidence has no source_references."""
    chunks = _is_a_chunks(_chunk_no_refs)
    graph = _graph(["cognitive-load", "mental-effort"])
    edges = is_a_from_key_terms.infer(chunks, None, graph)
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_is_a_rule_version_bumped_to_2():
    assert is_a_from_key_terms.RULE_VERSION == 2


def test_is_a_refs_are_deep_copied_not_shared(flag_on):
    """Mutating the evidence must not leak back to the chunk's source."""
    chunks = _is_a_chunks(_chunk_with_refs)
    original_refs = list(chunks[0]["source"]["source_references"])
    graph = _graph(["cognitive-load", "mental-effort"])
    edges = is_a_from_key_terms.infer(chunks, None, graph)
    # Mutate emitted evidence
    for edge in edges:
        edge["provenance"]["evidence"]["source_references"].append(
            {"sourceId": "dart:x#y", "role": "primary"}
        )
    # Originating chunk's refs untouched
    assert chunks[0]["source"]["source_references"] == original_refs


# --------------------------------------------------------------------- #
# Exemplifies rule
# --------------------------------------------------------------------- #


def _exemplifies_chunks(chunk_builder):
    return [
        chunk_builder(
            "chunk_ex_01",
            chunk_type="example",
            concept_tags=["cognitive-load"],
        )
    ]


def test_exemplifies_flag_off_omits_source_references(flag_off):
    chunks = _exemplifies_chunks(_chunk_with_refs)
    graph = _graph(["cognitive-load"])
    edges = exemplifies_from_example_chunks.infer(chunks, None, graph)
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_exemplifies_flag_on_copies_refs(flag_on):
    chunks = _exemplifies_chunks(_chunk_with_refs)
    graph = _graph(["cognitive-load"])
    edges = exemplifies_from_example_chunks.infer(chunks, None, graph)
    assert edges
    for edge in edges:
        assert edge["provenance"]["evidence"]["source_references"] == SAMPLE_REFS


def test_exemplifies_flag_on_legacy_chunk_omits_refs(flag_on):
    chunks = _exemplifies_chunks(_chunk_no_refs)
    graph = _graph(["cognitive-load"])
    edges = exemplifies_from_example_chunks.infer(chunks, None, graph)
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_exemplifies_rule_version_bumped_to_2():
    assert exemplifies_from_example_chunks.RULE_VERSION == 2


# --------------------------------------------------------------------- #
# DerivedFromObjective rule
# --------------------------------------------------------------------- #


def _derived_chunks(chunk_builder):
    return [chunk_builder("chunk_derived", learning_outcome_refs=["to-01"])]


def test_derived_flag_off_omits_source_references(flag_off):
    edges = derived_from_lo_ref.infer(
        _derived_chunks(_chunk_with_refs), None, {"nodes": []}
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_derived_flag_on_copies_refs(flag_on):
    edges = derived_from_lo_ref.infer(
        _derived_chunks(_chunk_with_refs), None, {"nodes": []}
    )
    assert edges
    for edge in edges:
        assert edge["provenance"]["evidence"]["source_references"] == SAMPLE_REFS


def test_derived_flag_on_legacy_chunk_omits_refs(flag_on):
    edges = derived_from_lo_ref.infer(
        _derived_chunks(_chunk_no_refs), None, {"nodes": []}
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_derived_rule_version_bumped_to_2():
    assert derived_from_lo_ref.RULE_VERSION == 2


# --------------------------------------------------------------------- #
# DefinedBy rule (uses chunks list for flag-on lookup)
# --------------------------------------------------------------------- #


def test_defined_by_flag_off_omits_source_references(flag_off):
    """Flag off — chunks list may be anything (rule doesn't use it)."""
    graph = _graph(
        ["concept-x"], occurrences_by_id={"concept-x": ["chunk_1"]}
    )
    edges = defined_by_from_first_mention.infer(
        [_chunk_with_refs("chunk_1")], None, graph
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_defined_by_flag_on_copies_refs_from_first_mention(flag_on):
    graph = _graph(
        ["concept-x"], occurrences_by_id={"concept-x": ["chunk_1"]}
    )
    edges = defined_by_from_first_mention.infer(
        [_chunk_with_refs("chunk_1")], None, graph
    )
    assert edges
    for edge in edges:
        assert edge["provenance"]["evidence"]["source_references"] == SAMPLE_REFS


def test_defined_by_flag_on_no_chunks_list_omits_refs(flag_on):
    """Flag on but chunks=None (orchestrator didn't provide) → no refs."""
    graph = _graph(
        ["concept-x"], occurrences_by_id={"concept-x": ["chunk_1"]}
    )
    edges = defined_by_from_first_mention.infer(None, None, graph)
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_defined_by_flag_on_chunk_missing_from_list_omits_refs(flag_on):
    """first_chunk isn't in the chunks list → no refs."""
    graph = _graph(
        ["concept-x"], occurrences_by_id={"concept-x": ["chunk_1"]}
    )
    edges = defined_by_from_first_mention.infer(
        [_chunk_with_refs("chunk_other")], None, graph
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_defined_by_flag_on_legacy_chunk_omits_refs(flag_on):
    graph = _graph(
        ["concept-x"], occurrences_by_id={"concept-x": ["chunk_1"]}
    )
    edges = defined_by_from_first_mention.infer(
        [_chunk_no_refs("chunk_1")], None, graph
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_defined_by_rule_version_bumped_to_2():
    assert defined_by_from_first_mention.RULE_VERSION == 2


# --------------------------------------------------------------------- #
# Assesses rule (resolves chunk via source_chunk_id on the question)
# --------------------------------------------------------------------- #


def test_assesses_flag_off_omits_source_references(flag_off):
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_1"}
    ]
    edges = assesses_from_question_lo.infer(
        [_chunk_with_refs("chunk_1")],
        None,
        {"nodes": []},
        questions=questions,
    )
    assert edges
    for edge in edges:
        ev = edge["provenance"]["evidence"]
        assert "source_references" not in ev
        # Legacy source_chunk_id still present
        assert ev.get("source_chunk_id") == "chunk_1"


def test_assesses_flag_on_copies_refs_from_source_chunk(flag_on):
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_1"}
    ]
    edges = assesses_from_question_lo.infer(
        [_chunk_with_refs("chunk_1")],
        None,
        {"nodes": []},
        questions=questions,
    )
    assert edges
    for edge in edges:
        ev = edge["provenance"]["evidence"]
        assert ev.get("source_references") == SAMPLE_REFS


def test_assesses_flag_on_no_source_chunk_id_no_refs(flag_on):
    """Question without source_chunk_id → can't resolve chunk → no refs."""
    questions = [{"id": "q-001", "objective_id": "to-01"}]
    edges = assesses_from_question_lo.infer(
        [_chunk_with_refs("chunk_1")],
        None,
        {"nodes": []},
        questions=questions,
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_assesses_flag_on_chunk_not_found_omits_refs(flag_on):
    """source_chunk_id points at a chunk that doesn't exist → no refs."""
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_missing"}
    ]
    edges = assesses_from_question_lo.infer(
        [_chunk_with_refs("chunk_1")],
        None,
        {"nodes": []},
        questions=questions,
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_assesses_flag_on_legacy_chunk_omits_refs(flag_on):
    """source_chunk_id resolves to a pre-Wave-10 chunk → no refs."""
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_1"}
    ]
    edges = assesses_from_question_lo.infer(
        [_chunk_no_refs("chunk_1")],
        None,
        {"nodes": []},
        questions=questions,
    )
    assert edges
    for edge in edges:
        assert "source_references" not in edge["provenance"]["evidence"]


def test_assesses_rule_version_bumped_to_2():
    assert assesses_from_question_lo.RULE_VERSION == 2


# --------------------------------------------------------------------- #
# End-to-end: build_semantic_graph
# --------------------------------------------------------------------- #


def test_build_semantic_graph_flag_off_no_evidence_refs(flag_off):
    """Running the full orchestrator with flag off yields no evidence refs on
    any of the 5 chunk-anchored edge types."""
    from datetime import datetime, timezone

    from Trainforge.rag.typed_edge_inference import build_semantic_graph

    chunks = [
        _chunk_with_refs(
            "chunk_01",
            key_terms=[{
                "term": "cognitive load",
                "definition": "Cognitive load is a type of mental effort.",
            }],
            learning_outcome_refs=["to-01"],
            concept_tags=["cognitive-load"],
            chunk_type="example",
        ),
    ]
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {
                "id": "cognitive-load",
                "label": "cognitive-load",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
            {
                "id": "mental-effort",
                "label": "mental-effort",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
        ],
    }
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_01"},
    ]
    artifact = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        questions=questions,
    )
    for edge in artifact["edges"]:
        ev = edge.get("provenance", {}).get("evidence") or {}
        assert "source_references" not in ev, (
            f"{edge['type']} emitted source_references with flag off: {ev}"
        )


def test_build_semantic_graph_flag_on_evidence_refs_present(flag_on):
    """Flag on → all 5 chunk-anchored rules emit source_references in the
    evidence where the originating chunk carries refs."""
    from datetime import datetime, timezone

    from Trainforge.rag.typed_edge_inference import build_semantic_graph

    chunks = [
        _chunk_with_refs(
            "chunk_01",
            key_terms=[{
                "term": "cognitive load",
                "definition": "Cognitive load is a type of mental effort.",
            }],
            learning_outcome_refs=["to-01"],
            concept_tags=["cognitive-load"],
            chunk_type="example",
        ),
    ]
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {
                "id": "cognitive-load",
                "label": "cognitive-load",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
            {
                "id": "mental-effort",
                "label": "mental-effort",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
        ],
    }
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_01"},
    ]
    artifact = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        questions=questions,
    )

    # Collect observed edge types that carry evidence refs
    with_refs_types = set()
    without_refs_but_chunk_anchored = set()
    chunk_anchored_edge_types = {
        # ``is_a_from_key_terms`` emits ``broader-than`` when both
        # endpoints are cf:Concept instances (the canonical case here:
        # cognitive-load and mental-effort are both concept-graph
        # nodes). ``is-a`` is kept in the set for forward-compat with
        # any future class-level subsumption emit.
        "is-a", "broader-than",
        "exemplifies", "derived-from-objective",
        "defined-by", "assesses",
    }
    for edge in artifact["edges"]:
        if edge["type"] not in chunk_anchored_edge_types:
            continue
        ev = edge.get("provenance", {}).get("evidence") or {}
        if "source_references" in ev:
            with_refs_types.add(edge["type"])
            assert ev["source_references"] == SAMPLE_REFS
        else:
            without_refs_but_chunk_anchored.add(edge["type"])

    # Every chunk-anchored edge emitted by this fixture should carry refs —
    # the chunk is known to carry source_references.
    assert not without_refs_but_chunk_anchored, (
        f"chunk-anchored edges missing refs: {without_refs_but_chunk_anchored}"
    )
    # At a minimum, is-a + derived-from-objective + defined-by + exemplifies
    # + assesses all fire in this fixture.
    assert with_refs_types, "No chunk-anchored edge emitted with refs"


def test_build_semantic_graph_flag_on_legacy_corpus_no_refs(flag_on):
    """Flag on but chunks have no Wave-10 refs → evidence arms omit refs
    (absence = unknown, back-compat)."""
    from datetime import datetime, timezone

    from Trainforge.rag.typed_edge_inference import build_semantic_graph

    chunks = [
        _chunk_no_refs(
            "chunk_01",
            key_terms=[{
                "term": "cognitive load",
                "definition": "Cognitive load is a type of mental effort.",
            }],
            learning_outcome_refs=["to-01"],
            concept_tags=["cognitive-load"],
            chunk_type="example",
        ),
    ]
    concept_graph = {
        "kind": "concept",
        "nodes": [
            {
                "id": "cognitive-load",
                "label": "cognitive-load",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
            {
                "id": "mental-effort",
                "label": "mental-effort",
                "frequency": 2,
                "occurrences": ["chunk_01"],
            },
        ],
    }
    questions = [
        {"id": "q-001", "objective_id": "to-01", "source_chunk_id": "chunk_01"},
    ]
    artifact = build_semantic_graph(
        chunks,
        None,
        concept_graph,
        now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        questions=questions,
    )
    for edge in artifact["edges"]:
        ev = edge.get("provenance", {}).get("evidence") or {}
        assert "source_references" not in ev, (
            f"{edge['type']} emitted refs on legacy corpus: {ev}"
        )


# --------------------------------------------------------------------- #
# Abstract arms never emit source_references regardless of flag
# --------------------------------------------------------------------- #


def test_prerequisite_evidence_never_carries_source_references(flag_on):
    """PrerequisiteEvidence is not touched by Wave 11 — flag ON or OFF, no refs."""
    from Trainforge.rag.inference_rules import prerequisite_from_lo_order
    chunks = [
        _chunk_with_refs(
            "c1",
            concept_tags=["concept-a"],
            learning_outcome_refs=["to-01"],
        ),
        _chunk_with_refs(
            "c2",
            concept_tags=["concept-b"],
            learning_outcome_refs=["to-02"],
        ),
    ]
    course = {
        "learning_outcomes": [
            {"id": "to-01", "statement": "x"},
            {"id": "to-02", "statement": "y"},
        ]
    }
    graph = _graph(["concept-a", "concept-b"])
    edges = prerequisite_from_lo_order.infer(chunks, course, graph)
    for edge in edges:
        ev = edge.get("provenance", {}).get("evidence") or {}
        assert "source_references" not in ev


def test_prerequisite_rule_version_not_bumped():
    """P4: PrerequisiteEvidence is untouched by Wave 11 — version stays at 1."""
    from Trainforge.rag.inference_rules import prerequisite_from_lo_order
    assert prerequisite_from_lo_order.RULE_VERSION == 1


def test_related_rule_version_not_bumped():
    from Trainforge.rag.inference_rules import related_from_cooccurrence
    assert related_from_cooccurrence.RULE_VERSION == 1


def test_misconception_rule_version_not_bumped():
    from Trainforge.rag.inference_rules import (
        misconception_of_from_misconception_ref,
    )
    assert misconception_of_from_misconception_ref.RULE_VERSION == 1
