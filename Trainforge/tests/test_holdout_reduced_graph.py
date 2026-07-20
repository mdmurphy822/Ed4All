"""Regression net for the holdout-reduced-graph design rule (SFT S4).

Live-defect fix: ``kg_metadata_generator`` (and every graph->pair generator)
must consume the holdout-REDUCED graph so a WITHHELD edge — the exact edges the
Tier-2 eval holds out as ground truth — can never appear in a training pair.

Covers:
  * ``_load_withheld_edge_index`` parses ``eval/holdout_split.json``;
  * ``_reduce_graph_by_holdout`` drops the withheld edge (pedagogy + concept,
    across the ``relation_type`` / ``type`` key + naming normalization);
  * PROOF: a withheld edge never surfaces in a kg_metadata pair;
  * PROOF: a withheld edge never surfaces in a concept-graph->SFT pair;
  * empty / absent holdout split => byte-identical (no reduction).

Offline / deterministic — no network, no model, no course slugs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import (  # noqa: E402
    _load_withheld_edge_index,
    _normalize_holdout_rel,
    _reduce_graph_by_holdout,
)
from Trainforge.generators.kg_metadata_generator import (  # noqa: E402
    generate_kg_metadata_pairs,
)
from Trainforge.generators.graph_sft_generator import (  # noqa: E402
    generate_graph_sft_pairs,
)


class _RecordingCapture:
    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)


def _write_holdout(course_dir: Path, withheld: List[Dict[str, Any]]) -> None:
    (course_dir / "eval").mkdir(parents=True, exist_ok=True)
    (course_dir / "eval" / "holdout_split.json").write_text(
        json.dumps({"withheld_edges": withheld}), encoding="utf-8",
    )


def test_normalize_holdout_rel():
    assert _normalize_holdout_rel("prerequisite_of") == "prerequisite"
    assert _normalize_holdout_rel("related_to") == "related-to"
    assert _normalize_holdout_rel("PREREQUISITE") == "prerequisite"


def test_load_index_absent_is_empty(tmp_path):
    assert _load_withheld_edge_index(tmp_path) == set()


def test_reduce_drops_withheld_pedagogy_edge():
    graph = {
        "edges": [
            {"source": "c1", "target": "concept_a", "relation_type": "assesses"},
            {"source": "c2", "target": "concept_b", "relation_type": "assesses"},
        ],
    }
    idx = {("c1", "concept_a", "assesses")}
    reduced, removed = _reduce_graph_by_holdout(graph, idx)
    assert removed == 1
    assert {(e["source"], e["target"]) for e in reduced["edges"]} == {("c2", "concept_b")}
    # Input graph untouched (shallow copy).
    assert len(graph["edges"]) == 2


def test_reduce_matches_concept_type_key_and_naming():
    # Withheld carries relation_type="prerequisite_of"; concept edge uses
    # type="prerequisite" — normalization must make them match.
    graph = {"edges": [{"source": "a", "target": "b", "type": "prerequisite"}]}
    idx = {("a", "b", "prerequisite")}
    reduced, removed = _reduce_graph_by_holdout(graph, idx)
    assert removed == 1
    assert reduced["edges"] == []


def test_empty_index_is_byte_identical():
    graph = {"edges": [{"source": "a", "target": "b", "type": "is-a"}]}
    reduced, removed = _reduce_graph_by_holdout(graph, set())
    assert removed == 0
    assert reduced is graph


def test_withheld_edge_never_in_kg_metadata_pair(tmp_path):
    ped = {
        "nodes": [],
        "edges": [
            {"source": "c_00001", "target": "concept_alpha", "relation_type": "assesses"},
            {"source": "c_00002", "target": "concept_beta", "relation_type": "assesses"},
            {"source": "c_00003", "target": "concept_gamma", "relation_type": "assesses"},
        ],
    }
    _write_holdout(tmp_path, [
        {"source": "c_00001", "target": "concept_alpha", "relation_type": "assesses"},
    ])
    idx = _load_withheld_edge_index(tmp_path)
    reduced, removed = _reduce_graph_by_holdout(ped, idx)
    assert removed == 1

    cap = _RecordingCapture()
    pairs, _ = generate_kg_metadata_pairs(reduced, capture=cap, max_pairs=100, seed=3)
    # The positive pair for the withheld edge must be gone. The generator's
    # positive completion literally names 'source' -[rel]-> 'target'; assert
    # no pair carries the withheld (source, target) as a POSITIVE.
    for p in pairs:
        is_withheld_target = (
            p.get("chunk_id") == "c_00001"
            and p.get("kg_metadata_target") == "concept_alpha"
            and p.get("kg_metadata_polarity") == "yes"
        )
        assert not is_withheld_target


def test_withheld_edge_never_in_graph_sft_pair(tmp_path):
    graph = {
        "nodes": [
            {"id": "concept-a", "label": "Concept A", "occurrences": ["c1"]},
            {"id": "concept-b", "label": "Concept B", "occurrences": ["c2"]},
            {"id": "concept-c", "label": "Concept C", "occurrences": ["c3"]},
        ],
        "edges": [
            {"source": "concept-a", "target": "concept-b", "type": "related-to",
             "edge_status": "supported"},
            {"source": "concept-b", "target": "concept-c", "type": "related-to",
             "edge_status": "supported"},
        ],
    }
    # Withhold the a->b edge via a pedagogy-style relation_type spelling.
    _write_holdout(tmp_path, [
        {"source": "concept-a", "target": "concept-b", "relation_type": "related_to"},
    ])
    idx = _load_withheld_edge_index(tmp_path)
    reduced, removed = _reduce_graph_by_holdout(graph, idx)
    assert removed == 1

    cap = _RecordingCapture()
    pairs = list(generate_graph_sft_pairs(reduced, cap))
    rel_pairs = [p for p in pairs if p["pair_format"] == "relation_qa"]
    # No relation_qa pair may verbalize the withheld a<->b relation.
    for p in rel_pairs:
        text = p["completion"]
        assert not ("Concept A" in text and "Concept B" in text)
