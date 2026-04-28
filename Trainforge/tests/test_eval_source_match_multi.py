"""Wave 108 / Phase B: source_match must accept multi-chunk ground truth.

A fact can be supported by multiple chunks. A model that cites ANY
ground-truth chunk is correct; today's evaluator only credits the
single chunk that happened to anchor the held-out edge."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.source_match import SourceMatchEvaluator


def test_multi_citation_ground_truth_accepts_any_match(tmp_path: Path) -> None:
    """A probe carrying ground_truth_chunk_ids=[A,B,C] credits the model
    when it cites any of those three. The cited set [B] suffices."""
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "withheld_edges": [
            {
                "source": "chunk_001",
                "target": "concept_rdfs",
                "relation_type": "teaches",
                "ground_truth_chunk_ids": ["chunk_001", "chunk_002", "chunk_005"],
            },
        ],
    }), encoding="utf-8")

    model = lambda _prompt: "RDFS describes vocabulary [chunk_005]."
    result = SourceMatchEvaluator(split, model).evaluate()
    assert result["source_match_rate"] == 1.0
    per_q = result["per_question"][0]
    assert per_q["score"] == 1.0
    assert "chunk_005" in per_q["cited_chunk_ids"]


def test_legacy_single_ground_truth_still_works(tmp_path: Path) -> None:
    """A withheld_edge without ground_truth_chunk_ids falls back to
    edge.source (legacy behaviour); citing the source counts."""
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "withheld_edges": [
            {
                "source": "chunk_007",
                "target": "concept_shacl",
                "relation_type": "teaches",
            },
        ],
    }), encoding="utf-8")

    model = lambda _prompt: "SHACL validates RDF graphs [chunk_007]."
    result = SourceMatchEvaluator(split, model).evaluate()
    assert result["source_match_rate"] == 1.0


def test_no_match_when_cited_chunk_outside_ground_truth_set(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "withheld_edges": [
            {
                "source": "chunk_001",
                "target": "concept_rdfs",
                "relation_type": "teaches",
                "ground_truth_chunk_ids": ["chunk_001", "chunk_002"],
            },
        ],
    }), encoding="utf-8")

    model = lambda _prompt: "RDFS [chunk_999]."
    result = SourceMatchEvaluator(split, model).evaluate()
    assert result["source_match_rate"] == 0.0
