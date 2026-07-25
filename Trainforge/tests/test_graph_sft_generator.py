"""Regression net for ``Trainforge/generators/graph_sft_generator.py`` (SFT S5).

Covers the concept-graph->SFT contract:
  * required DecisionCapture (None raises) + one event per family batch;
  * relation-QA / prereq study-path / concept-verbalization families emit;
  * consensus filter — contradicted/retracted edges never yield a pair;
  * graph-frame coverage — every surviving node + edge yields >= 1 pair;
  * navigation cap <= 2% of the emitted total;
  * schema-bounded text + full per-pair provenance;
  * determinism.

Offline / deterministic — no network, no model, no course slugs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.graph_sft_generator import (  # noqa: E402
    generate_graph_sft_pairs,
)


class _RecordingCapture:
    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)


def _graph() -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": "linear-equation", "label": "Linear Equation", "frequency": 5,
             "occurrences": ["c_00001", "c_00002"]},
            {"id": "variable", "label": "Variable", "frequency": 8,
             "occurrences": ["c_00003"]},
            {"id": "slope", "label": "Slope", "frequency": 3, "occurrences": []},
            {"id": "bad-concept", "label": "Bad Concept", "frequency": 1},
        ],
        "edges": [
            {"source": "linear-equation", "target": "variable", "type": "prerequisite",
             "confidence": 0.8, "edge_status": "supported"},
            {"source": "slope", "target": "linear-equation", "type": "related-to",
             "confidence": 0.6, "edge_status": "confirmed"},
            # Contradicted / retracted -> must be excluded by the consensus filter.
            {"source": "linear-equation", "target": "bad-concept", "type": "is-a",
             "confidence": 0.4, "edge_status": "contradicted"},
            {"source": "variable", "target": "bad-concept", "type": "related-to",
             "confidence": 0.4, "edge_status": "retracted"},
        ],
    }


def test_requires_capture():
    with pytest.raises(ValueError):
        list(generate_graph_sft_pairs(_graph(), None))


def test_families_emit_and_batch_events():
    cap = _RecordingCapture()
    pairs = list(generate_graph_sft_pairs(_graph(), cap, seed=7))
    fams = {p["pair_format"] for p in pairs}
    assert "relation_qa" in fams
    assert "prereq_study_path" in fams
    assert "concept_verbalization" in fams
    # One batch decision per family (relation_qa, prereq, verbalization).
    dtypes = [d["decision_type"] for d in cap.decisions]
    assert dtypes.count("kg_metadata_generation") >= 3
    assert all(len(d["rationale"]) >= 20 for d in cap.decisions)


def test_consensus_filter_excludes_contradicted_and_retracted():
    cap = _RecordingCapture()
    pairs = list(generate_graph_sft_pairs(_graph(), cap))
    # No pair may reference the bad-concept node (only reachable via the
    # contradicted/retracted edges) in a relation_qa pair.
    rel_pairs = [p for p in pairs if p["pair_format"] == "relation_qa"]
    for p in rel_pairs:
        assert "Bad Concept" not in p["completion"]


def test_graph_frame_covers_every_surviving_edge_and_node():
    cap = _RecordingCapture()
    pairs = list(generate_graph_sft_pairs(_graph(), cap))
    # 2 surviving edges -> 2 relation_qa pairs.
    rel = [p for p in pairs if p["pair_format"] == "relation_qa"]
    assert len(rel) == 2
    # Every node yields a concept_verbalization pair.
    verbal = [p for p in pairs if p["pair_format"] == "concept_verbalization"]
    assert len(verbal) == 4


def test_prereq_study_path_present():
    cap = _RecordingCapture()
    pairs = list(generate_graph_sft_pairs(_graph(), cap))
    prereq = [p for p in pairs if p["pair_format"] == "prereq_study_path"]
    assert len(prereq) == 1
    assert "prerequisite" in prereq[0]["completion"].lower()


def test_schema_bounds_and_provenance():
    cap = _RecordingCapture()
    pairs = list(generate_graph_sft_pairs(_graph(), cap))
    assert pairs
    for p in pairs:
        assert 40 <= len(p["prompt"]) <= 400
        assert 50 <= len(p["completion"]) <= 600
        assert p["provider"] == "local"
        assert p["generation_method"] == "deterministic_template"
        assert p["seat_license"]
        assert p["holdout_safe"] is True
        assert p["decontam_checked"] is False
        assert "edge_consensus_status" in p["verifier_results"]
        assert p["template_id"].startswith("graph_sft.")


def test_navigation_cap_two_percent():
    cap = _RecordingCapture()
    nav = [(f"concept-{i}", f"Week {i}") for i in range(50)]
    pairs = list(generate_graph_sft_pairs(_graph(), cap, navigation_items=nav))
    nav_pairs = [p for p in pairs if p["pair_format"] == "navigation"]
    non_nav = len(pairs) - len(nav_pairs)
    # cap = floor(non_nav * 0.02); with ~7 non-nav pairs -> 0 navigation.
    assert len(nav_pairs) <= max(1, int(non_nav * 0.02)) or len(nav_pairs) == 0


def test_max_pairs_cap_and_determinism():
    cap1 = _RecordingCapture()
    cap2 = _RecordingCapture()
    p1 = list(generate_graph_sft_pairs(_graph(), cap1, max_pairs=3, seed=11))
    p2 = list(generate_graph_sft_pairs(_graph(), cap2, max_pairs=3, seed=11))
    assert len(p1) == 3
    assert [x["prompt"] for x in p1] == [x["prompt"] for x in p2]
    assert [x["seed"] for x in p1] == [x["seed"] for x in p2]


# --------------------------------------------------------------------- #
# Identifier leakage (SFT corpus poisoning)                              #
# --------------------------------------------------------------------- #
#
# A concept graph also holds evidence/bookkeeping nodes whose ``label`` is
# just their own opaque id (``class: "Chunk"`` / ``ComponentObjective`` /
# ``Outcome``). Verbalizing those emitted pairs like
#   "explain how 'Complete Factorization' relates to '<course>_chunk_00633'"
# which teach the adapter to answer with internal identifiers. On a real
# course this reached 86% of the emitted corpus.

import re  # noqa: E402

_CHUNK_ID_RE = re.compile(r"chunk[_\s-]?\d{2,}", re.I)


def _graph_with_id_only_nodes() -> Dict[str, Any]:
    """Graph mixing real concepts with id-labelled bookkeeping nodes."""
    return {
        "nodes": [
            {"id": "complete-factorization", "label": "Complete Factorization",
             "class": "DomainConcept", "occurrences": ["c_00001"]},
            {"id": "binomial", "label": "Binomial", "class": "DomainConcept",
             "occurrences": ["c_00002"]},
            # label == id, no human name available.
            {"id": "course_scan_chunk_00633", "label": "course_scan_chunk_00633",
             "class": "Chunk"},
            # Compound id: the digit run is followed by "_", not a word
            # boundary — a \b-anchored pattern misses this shape.
            {"id": "q_course_scan_chunk_00190_co-01",
             "label": "q_course_scan_chunk_00190_co-01", "class": "Chunk"},
            {"id": "co-01", "label": "co-01", "class": "ComponentObjective"},
            {"id": "to-02", "label": "to-02", "class": "Outcome"},
        ],
        "edges": [
            {"source": "complete-factorization", "target": "course_scan_chunk_00633",
             "type": "defined-by", "edge_status": "supported"},
            {"source": "binomial", "target": "q_course_scan_chunk_00190_co-01",
             "type": "assesses", "edge_status": "supported"},
            {"source": "complete-factorization", "target": "co-01",
             "type": "derived-from-objective", "edge_status": "supported"},
            {"source": "binomial", "target": "to-02",
             "type": "derived-from-objective", "edge_status": "supported"},
            # The one genuine concept-to-concept relation.
            {"source": "binomial", "target": "complete-factorization",
             "type": "prerequisite", "edge_status": "supported"},
        ],
    }


def test_no_pair_contains_a_raw_chunk_identifier():
    pairs = list(generate_graph_sft_pairs(
        _graph_with_id_only_nodes(), _RecordingCapture()))
    offenders = [
        p for p in pairs
        if _CHUNK_ID_RE.search(p["prompt"]) or _CHUNK_ID_RE.search(p["completion"])
    ]
    assert not offenders, (
        "chunk identifiers leaked into training text: "
        f"{[(p['prompt'][:80], p['completion'][:80]) for p in offenders[:3]]}"
    )


def test_id_only_labels_are_never_verbalized():
    """co-01 / to-02 / chunk ids must not appear as quoted concept names."""
    pairs = list(generate_graph_sft_pairs(
        _graph_with_id_only_nodes(), _RecordingCapture()))
    blob = " ".join(p["prompt"] + " " + p["completion"] for p in pairs)
    for opaque in ("co-01", "to-02", "course_scan_chunk_00633",
                   "q_course_scan_chunk_00190_co-01"):
        assert f"'{opaque}'" not in blob, f"{opaque!r} verbalized as a concept"


def test_provenance_edges_never_become_relation_pairs():
    """derived-from-objective records where a node CAME FROM; it is not a
    relation between two topics and must not be verbalized."""
    pairs = list(generate_graph_sft_pairs(
        _graph_with_id_only_nodes(), _RecordingCapture()))
    blob = " ".join(p["completion"] for p in pairs)
    assert "is derived from the objective" not in blob


def test_real_concepts_still_emit_pairs():
    """The filter must not empty the corpus — genuine concept-to-concept
    edges and nameable nodes still produce pairs."""
    pairs = list(generate_graph_sft_pairs(
        _graph_with_id_only_nodes(), _RecordingCapture()))
    assert pairs, "filter removed every pair"
    blob = " ".join(p["prompt"] + " " + p["completion"] for p in pairs)
    assert "Complete Factorization" in blob
    assert "Binomial" in blob
