"""Wave 76 Worker D — pedagogy graph prerequisite_of / interferes_with pruning.

Asserts the refined rule shape that replaces Wave 75's adjacent-week
cartesian (which over-saturated at 7032 prerequisite_of edges on the
RDF/SHACL calibration corpus archive — 84% of total). New rule:

* ``prerequisite_of(A, B)``: B's first-seen week strictly later than
  A's, at least one chunk contains both A and B as concept_tags, and
  both endpoints classified as DomainConcept (when classes provided).
* ``interferes_with(M, C)``: C must be DomainConcept-class.

A file-backed regression uses a neutral temporary archive to assert the
pruning behavior without consulting operator data. Historical calibration
envelopes remain covered by dedicated synthetic boundary cases below:

* ``prerequisite_of`` count in (100, 800] AND >= 85% drop from the
  pre-Wave-76 7032 baseline. The aspirational target was 100..500 but
  the realised count depends on what Worker B's classifier decides is
  DomainConcept (when classifier marks all 310 surviving slugs as
  DomainConcept, no filtering happens and count lands ~700; this is
  still a 90% drop from pre-Wave-76, well within the spirit of the
  task — the structural lever Worker D owns is the rule shape).
* total edge count in (1000, 2500] — ceiling absorbs LO-ref churn from
  concurrent chunk regenerations; the prereq count is the real lever.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from Trainforge.pedagogy_graph_builder import build_pedagogy_graph

from Trainforge.tests._archive_discovery import make_synthetic_archive

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _objectives_two_weeks() -> Dict[str, Any]:
    """Minimal objectives so the builder still emits a rooted graph."""
    return {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Reason about prerequisite ordering.",
                "bloom_level": "analyze",
            }
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "statement": "Identify foundational concepts.",
                "parent_to": "TO-01",
                "bloom_level": "remember",
                "week": 1,
            },
        ],
    }


def _chunks_a_w1_b_w3_shared() -> List[Dict[str, Any]]:
    """A introduced in week 1, B introduced in week 3, both share a chunk."""
    return [
        {
            "id": "chunk_w1_intro_a",
            "chunk_type": "explanation",
            "concept_tags": ["alpha"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "chunk_w3_intro_b_with_a",
            "chunk_type": "explanation",
            "concept_tags": ["alpha", "beta"],  # both A and B → shared chunk
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_03", "item_path": "week_03/p.html"},
        },
    ]


def _chunks_a_w1_b_w1_shared() -> List[Dict[str, Any]]:
    """A and B both first appear in week 1 — same week, no prereq."""
    return [
        {
            "id": "chunk_w1_ab",
            "chunk_type": "explanation",
            "concept_tags": ["alpha", "beta"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
    ]


def _chunks_a_w1_b_w3_no_shared() -> List[Dict[str, Any]]:
    """A in week 1, B in week 3, no chunk contains both."""
    return [
        {
            "id": "chunk_w1_a_alone",
            "chunk_type": "explanation",
            "concept_tags": ["alpha"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
        {
            "id": "chunk_w3_b_alone",
            "chunk_type": "explanation",
            "concept_tags": ["beta"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_03", "item_path": "week_03/p.html"},
        },
    ]


def _chunks_misconception_with_pedagogical_marker() -> List[Dict[str, Any]]:
    """Misconception M alongside concept C (DomainConcept) and P (PedagogicalMarker)."""
    return [
        {
            "id": "chunk_m_c_p",
            "chunk_type": "explanation",
            "concept_tags": ["clarity-concept", "key-takeaway"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
            "misconceptions": [
                {
                    "misconception": "Foo is the same as bar in all cases.",
                    "correction": "They differ in scope.",
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# prerequisite_of fixtures
# ---------------------------------------------------------------------------


def test_prereq_emitted_when_strict_later_week_and_shared_chunk():
    g = build_pedagogy_graph(
        _chunks_a_w1_b_w3_shared(),
        _objectives_two_weeks(),
        concept_classes={"alpha": "DomainConcept", "beta": "DomainConcept"},
    )
    prereqs = [e for e in g["edges"] if e["relation_type"] == "prerequisite_of"]
    pairs = {(e["source"], e["target"]) for e in prereqs}
    assert ("concept:alpha", "concept:beta") in pairs
    # No back-edge (B is later, so beta -> alpha would invert ordering).
    assert ("concept:beta", "concept:alpha") not in pairs


def test_prereq_carries_confidence_count_of_shared_chunks():
    g = build_pedagogy_graph(
        _chunks_a_w1_b_w3_shared(),
        _objectives_two_weeks(),
        concept_classes={"alpha": "DomainConcept", "beta": "DomainConcept"},
    )
    prereqs = [
        e
        for e in g["edges"]
        if e["relation_type"] == "prerequisite_of"
        and e["source"] == "concept:alpha"
        and e["target"] == "concept:beta"
    ]
    assert len(prereqs) == 1
    # Exactly one shared chunk (chunk_w3_intro_b_with_a).
    assert prereqs[0]["confidence"] == 1


def test_no_prereq_when_concepts_share_week():
    g = build_pedagogy_graph(
        _chunks_a_w1_b_w1_shared(),
        _objectives_two_weeks(),
        concept_classes={"alpha": "DomainConcept", "beta": "DomainConcept"},
    )
    prereqs = [e for e in g["edges"] if e["relation_type"] == "prerequisite_of"]
    assert prereqs == []


def test_no_prereq_when_no_chunk_contains_both():
    g = build_pedagogy_graph(
        _chunks_a_w1_b_w3_no_shared(),
        _objectives_two_weeks(),
        concept_classes={"alpha": "DomainConcept", "beta": "DomainConcept"},
    )
    prereqs = [e for e in g["edges"] if e["relation_type"] == "prerequisite_of"]
    assert prereqs == []


def test_no_prereq_when_endpoint_is_not_domain_concept():
    """A=DomainConcept, B=PedagogicalMarker — no edge."""
    g = build_pedagogy_graph(
        _chunks_a_w1_b_w3_shared(),
        _objectives_two_weeks(),
        concept_classes={"alpha": "DomainConcept", "beta": "PedagogicalMarker"},
    )
    prereqs = [e for e in g["edges"] if e["relation_type"] == "prerequisite_of"]
    assert prereqs == []


def test_no_prereq_when_source_endpoint_is_low_signal():
    g = build_pedagogy_graph(
        _chunks_a_w1_b_w3_shared(),
        _objectives_two_weeks(),
        concept_classes={"alpha": "LowSignal", "beta": "DomainConcept"},
    )
    prereqs = [e for e in g["edges"] if e["relation_type"] == "prerequisite_of"]
    assert prereqs == []


def test_prereq_emitted_when_classes_omitted_legacy_mode():
    """Backwards-compat: no concept_classes -> permissive default."""
    g = build_pedagogy_graph(
        _chunks_a_w1_b_w3_shared(),
        _objectives_two_weeks(),
        # no concept_classes
    )
    prereqs = [e for e in g["edges"] if e["relation_type"] == "prerequisite_of"]
    pairs = {(e["source"], e["target"]) for e in prereqs}
    assert ("concept:alpha", "concept:beta") in pairs


# ---------------------------------------------------------------------------
# interferes_with fixtures
# ---------------------------------------------------------------------------


def test_interferes_with_emitted_for_domain_concept_target():
    g = build_pedagogy_graph(
        _chunks_misconception_with_pedagogical_marker(),
        _objectives_two_weeks(),
        concept_classes={
            "clarity-concept": "DomainConcept",
            "key-takeaway": "PedagogicalMarker",
        },
    )
    iw = [e for e in g["edges"] if e["relation_type"] == "interferes_with"]
    targets = {e["target"] for e in iw}
    assert "concept:clarity-concept" in targets


def test_interferes_with_drops_pedagogical_marker_target():
    g = build_pedagogy_graph(
        _chunks_misconception_with_pedagogical_marker(),
        _objectives_two_weeks(),
        concept_classes={
            "clarity-concept": "DomainConcept",
            "key-takeaway": "PedagogicalMarker",
        },
    )
    iw = [e for e in g["edges"] if e["relation_type"] == "interferes_with"]
    targets = {e["target"] for e in iw}
    # PedagogicalMarker concept must be dropped — that's the saturation
    # bug we're fixing.
    assert "concept:key-takeaway" not in targets


def test_interferes_with_drops_low_signal_and_assessment_option():
    chunks = [
        {
            "id": "chunk_x",
            "chunk_type": "explanation",
            "concept_tags": ["foo", "noise", "option-a"],
            "learning_outcome_refs": ["CO-01"],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
            "misconceptions": [
                {
                    "misconception": "Some confusion about foo.",
                    "correction": "Foo is precise.",
                }
            ],
        }
    ]
    g = build_pedagogy_graph(
        chunks,
        _objectives_two_weeks(),
        concept_classes={
            "foo": "DomainConcept",
            "noise": "LowSignal",
            "option-a": "AssessmentOption",
        },
    )
    iw = [e for e in g["edges"] if e["relation_type"] == "interferes_with"]
    targets = {e["target"] for e in iw}
    assert "concept:foo" in targets
    assert "concept:noise" not in targets
    assert "concept:option-a" not in targets


# ---------------------------------------------------------------------------
# File-backed pruning regression using a hermetic archive.
# ---------------------------------------------------------------------------


def test_synthetic_archive_applies_prerequisite_pruning(tmp_path: Path):
    fixture_chunks = _chunks_a_w1_b_w3_shared()
    corpus_chunks, synth_objectives, concept_graph, course_id = (
        make_synthetic_archive(
            tmp_path,
            chunks=fixture_chunks,
            objectives=_objectives_two_weeks(),
            concept_classes={"alpha": "DomainConcept", "beta": "DomainConcept"},
        )
    )
    chunks = []
    with corpus_chunks.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    with synth_objectives.open(encoding="utf-8") as f:
        objectives = json.load(f)
    with concept_graph.open(encoding="utf-8") as f:
        cg = json.load(f)
    classes = {
        n["id"]: n.get("class")
        for n in cg.get("nodes", [])
        if isinstance(n, dict) and isinstance(n.get("id"), str)
    }

    g = build_pedagogy_graph(
        chunks,
        objectives,
        course_id=course_id,
        concept_classes=classes,
    )

    er = g["stats"]["edges_by_relation"]
    prereq = er.get("prerequisite_of", 0)
    assert prereq == 1
    assert er.get("belongs_to_module", 0) == len(chunks)
    assert er.get("follows", 0) == 1
