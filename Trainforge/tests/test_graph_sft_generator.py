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
