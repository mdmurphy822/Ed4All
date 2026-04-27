"""Wave 92 — DisambiguationEvaluator tests.

Synthetic interferes_with edges + chunk-level corrections; mocked
model_callable. Asserts the heuristic scorer differentiates good
disambiguations (correction text + distinguishing language) from
weak ones (no correction language).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.disambiguation import DisambiguationEvaluator  # noqa: E402


def _build_course(tmp_path: Path) -> Path:
    course = tmp_path / "tst-101"
    (course / "graph").mkdir(parents=True)
    (course / "corpus").mkdir(parents=True)

    nodes = [
        {
            "id": "mc_001", "class": "Misconception",
            "label": "RDF triples are SQL rows",
            "statement": "An RDF triple is the same as a row in a relational table.",
        },
        {"id": "concept_rdf", "class": "Concept", "label": "RDF triple"},
    ]
    edges = [
        {
            "source": "mc_001",
            "target": "concept_rdf",
            "relation_type": "interferes_with",
        },
    ]
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}), encoding="utf-8",
    )
    chunk = {
        "id": "c_001",
        "misconceptions": [
            {
                "misconception": (
                    "An RDF triple is the same as a row in a relational table."
                ),
                "correction": (
                    "Triples are not rows; every triple is a first-class fact "
                    "and the schema is open-world unlike fixed columns."
                ),
            }
        ],
    }
    (course / "corpus" / "chunks.jsonl").write_text(
        json.dumps(chunk) + "\n", encoding="utf-8",
    )
    return course


def test_strong_disambiguation_passes(tmp_path):
    course = _build_course(tmp_path)
    scorer = DisambiguationEvaluator(
        course_path=course,
        model_callable=lambda p: (
            "Actually that's incorrect. Triples are first-class facts, "
            "rather than fixed columns; the schema is open-world."
        ),
    )
    out = scorer.evaluate()
    assert out["passed"] == 1
    assert out["pass_rate"] == 1.0


def test_no_distinguishing_signal_fails(tmp_path):
    course = _build_course(tmp_path)
    scorer = DisambiguationEvaluator(
        course_path=course,
        model_callable=lambda p: "Triples and rows are similar in some ways.",
    )
    out = scorer.evaluate()
    assert out["pass_rate"] == 0.0


def test_distinguishing_signal_without_correction_anchor_fails(tmp_path):
    course = _build_course(tmp_path)
    scorer = DisambiguationEvaluator(
        course_path=course,
        model_callable=lambda p: "Actually no, they differ. The misconception is wrong.",
    )
    out = scorer.evaluate()
    # Has signal but no overlap with correction tokens → fail.
    assert out["pass_rate"] == 0.0


def test_max_pairs_caps_run(tmp_path):
    course = _build_course(tmp_path)
    scorer = DisambiguationEvaluator(
        course_path=course,
        model_callable=lambda p: "rather",
        max_pairs=0,
    )
    out = scorer.evaluate()
    assert out["total"] == 0
    assert out["pass_rate"] == 0.0


def test_per_pair_carries_correction_anchors(tmp_path):
    course = _build_course(tmp_path)
    scorer = DisambiguationEvaluator(
        course_path=course,
        model_callable=lambda p: "rather first-class facts open-world",
    )
    out = scorer.evaluate()
    assert out["per_pair"]
    entry = out["per_pair"][0]
    assert "correction_anchors" in entry
    assert entry["correction_anchors"]
