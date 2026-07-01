"""Worker W4.6 — :class:`IntelligenceLevelAggregator` tests.

Coverage matrix:

1. All-axes-present course scores 4/5 (FAQ always absent).
2. Empty course scores 0/5.
3. Per-axis evidence counters populated correctly.
4. assessment_density floor drives the density verdict.
5. FAQ axis always absent + present in the axis list.
6. Schema validation.
7. Decision capture — one event per build.
8. Flag resolver — parse-with-fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.aggregators.intelligence_level import (
    MAX_SCORE,
    SCHEMA_VERSION,
    IntelligenceLevelAggregator,
    resolve_intelligence_rubric,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "aggregators" / "intelligence_level.schema.json"
)


def _write_graph(course_dir: Path, graph: Dict[str, Any]) -> None:
    gdir = course_dir / "graph"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "concept_graph_semantic.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )


def _write_assessments(course_dir: Path, n: int) -> None:
    (course_dir / "assessments.json").write_text(
        json.dumps({"questions": [{"question_id": f"q-{i}"} for i in range(n)]}),
        encoding="utf-8",
    )


def _write_objectives(course_dir: Path, n: int) -> None:
    (course_dir / "objectives.json").write_text(
        json.dumps({
            "learning_outcomes": [
                {"id": f"CO-{i:02d}", "statement": "x"} for i in range(n)
            ]
        }),
        encoding="utf-8",
    )


def _edge(etype: str, rule: str = "r") -> Dict[str, Any]:
    return {
        "source": "a",
        "target": "b",
        "type": etype,
        "provenance": {"rule": rule, "rule_version": 1},
    }


class TestScoring:
    def test_rich_course_scores_four_of_five(self, tmp_path):
        # glossary (is-a), concept_graph (nodes+edges), assessment_density
        # (2 questions / 2 objectives = 1.0 >= 0.5), prereq_cross_links.
        _write_graph(tmp_path, {
            "nodes": [{"id": "a", "class": "DomainConcept"}],
            "edges": [
                _edge("is-a"),
                _edge("prerequisite", rule="prerequisite_from_definition_mention"),
            ],
        })
        _write_assessments(tmp_path, 2)
        _write_objectives(tmp_path, 2)
        report = IntelligenceLevelAggregator(libv2_course_path=tmp_path).build()
        assert report["score"] == 4
        assert report["max_score"] == MAX_SCORE
        present = set(report["summary"]["present_axes"])
        assert present == {
            "key_terms_glossary", "concept_graph",
            "assessment_density", "prereq_cross_links",
        }
        assert report["summary"]["absent_axes"] == ["faq"]

    def test_empty_course_scores_zero(self, tmp_path):
        report = IntelligenceLevelAggregator(libv2_course_path=tmp_path).build()
        assert report["score"] == 0
        assert report["summary"]["present_axes"] == []
        assert len(report["axes"]) == 5

    def test_density_below_floor_absent(self, tmp_path):
        # 1 question / 10 objectives = 0.1 < 0.5 -> absent.
        _write_assessments(tmp_path, 1)
        _write_objectives(tmp_path, 10)
        report = IntelligenceLevelAggregator(libv2_course_path=tmp_path).build()
        density_axis = next(
            a for a in report["axes"] if a["axis"] == "assessment_density"
        )
        assert density_axis["present"] is False
        assert density_axis["evidence"]["density"] == 0.1

    def test_content_derived_prereq_evidence(self, tmp_path):
        _write_graph(tmp_path, {
            "nodes": [{"id": "a", "class": "DomainConcept"}],
            "edges": [
                _edge("prerequisite", rule="prerequisite_from_definition_mention"),
                _edge("prerequisite", rule="prerequisite_from_lo_order"),
            ],
        })
        report = IntelligenceLevelAggregator(libv2_course_path=tmp_path).build()
        prereq_axis = next(
            a for a in report["axes"] if a["axis"] == "prereq_cross_links"
        )
        assert prereq_axis["present"] is True
        assert prereq_axis["evidence"]["prerequisite_edges"] == 2
        assert prereq_axis["evidence"]["content_derived_prerequisite_edges"] == 1

    def test_self_check_chunks_count_toward_density(self, tmp_path):
        (tmp_path / "imscc_chunks").mkdir(parents=True)
        (tmp_path / "imscc_chunks" / "chunks.jsonl").write_text(
            "\n".join(json.dumps({"id": f"c{i}", "chunk_type": "self_check"})
                      for i in range(3)),
            encoding="utf-8",
        )
        _write_objectives(tmp_path, 2)
        report = IntelligenceLevelAggregator(libv2_course_path=tmp_path).build()
        density_axis = next(
            a for a in report["axes"] if a["axis"] == "assessment_density"
        )
        assert density_axis["evidence"]["self_check_chunks"] == 3
        assert density_axis["present"] is True

    def test_faq_always_absent(self, tmp_path):
        report = IntelligenceLevelAggregator(libv2_course_path=tmp_path).build()
        faq = next(a for a in report["axes"] if a["axis"] == "faq")
        assert faq["present"] is False


class TestSchemaValidation:
    def test_emitted_report_validates_against_schema(self, tmp_path):
        pytest.importorskip("jsonschema")
        from jsonschema import Draft202012Validator

        _write_graph(tmp_path, {
            "nodes": [{"id": "a", "class": "DomainConcept"}],
            "edges": [_edge("is-a"), _edge("prerequisite")],
        })
        _write_assessments(tmp_path, 5)
        _write_objectives(tmp_path, 4)
        out_path = tmp_path / "intelligence_level_report.json"
        IntelligenceLevelAggregator(
            course_code="BIO_201", run_id="WF-9",
            libv2_course_path=tmp_path,
        ).write(out_path)
        report = json.loads(out_path.read_text(encoding="utf-8"))

        with SCHEMA_PATH.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(report), key=lambda e: list(e.path)
        )
        assert errors == [], "; ".join(
            f"{list(e.path)}: {e.message}" for e in errors
        )
        assert report["schema_version"] == SCHEMA_VERSION


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class TestDecisionCapture:
    def test_one_event_per_build(self, tmp_path):
        capture = _FakeCapture()
        IntelligenceLevelAggregator(
            libv2_course_path=tmp_path, decision_capture=capture,
        ).build()
        assert len(capture.events) == 1
        evt = capture.events[0]
        assert evt["decision_type"] == "intelligence_level_scored"
        assert len(evt["rationale"]) >= 20


class TestFlagResolver:
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("On", True), ("yes", True),
        ("0", False), ("", False), ("garbage", False), (None, False),
    ])
    def test_parse_with_fallback(self, val, expected, monkeypatch):
        monkeypatch.delenv("ED4ALL_INTELLIGENCE_RUBRIC", raising=False)
        assert resolve_intelligence_rubric(val) is expected

    def test_reads_env_when_value_none(self, monkeypatch):
        monkeypatch.setenv("ED4ALL_INTELLIGENCE_RUBRIC", "1")
        assert resolve_intelligence_rubric() is True
