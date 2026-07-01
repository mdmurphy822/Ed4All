"""Worker W4.1 — :class:`ConceptCoverageAggregator` tests.

Coverage matrix:

1. Surface tallying — each edge type maps to its axis; occurrences drive
   the ``explained`` fallback.
2. Fully-covered concept — all five surfaces touch one node.
3. Uncovered concept — a minted node touched by nothing.
4. Endpoint filtering — Chunk / Outcome nodes are excluded (concept
   vocabulary only).
5. Content-derived prereq edges surfaced in the summary.
6. Missing graph — empty report, no crash.
7. Schema validation — emitted concept_coverage.json validates.
8. Decision capture — one ``concept_coverage_aggregated`` event per build.
9. Flag resolver — parse-with-fallback.
10. write() emits deterministic sorted JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lib.aggregators.concept_coverage import (
    SCHEMA_VERSION,
    ConceptCoverageAggregator,
    resolve_concept_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "aggregators" / "concept_coverage.schema.json"
)


def _write_graph(course_dir: Path, graph: Dict[str, Any], *, subdir: str = "graph") -> Path:
    gdir = course_dir / subdir
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / "concept_graph_semantic.json"
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return path


def _edge(source: str, target: str, etype: str, rule: str = "some_rule") -> Dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "type": etype,
        "provenance": {"rule": rule, "rule_version": 1},
    }


class TestSurfaceTallying:
    def test_each_edge_type_maps_to_its_axis(self, tmp_path):
        graph = {
            "kind": "concept_semantic",
            "generated_at": "2026-06-30T00:00:00Z",
            "nodes": [
                {"id": "photosynthesis", "label": "Photosynthesis",
                 "class": "DomainConcept", "occurrences": ["c-1"]},
                {"id": "chlorophyll", "label": "Chlorophyll",
                 "class": "DomainConcept"},
            ],
            "edges": [
                _edge("c-2", "photosynthesis", "defined-by"),
                _edge("photosynthesis", "chlorophyll", "is-a"),
                _edge("TO-01", "photosynthesis", "targets-concept"),
                _edge("c-3", "photosynthesis", "exemplifies"),
                _edge("photosynthesis", "chlorophyll", "prerequisite"),
            ],
        }
        _write_graph(tmp_path, graph)
        agg = ConceptCoverageAggregator(libv2_course_path=tmp_path)
        report = agg.build()
        by_id = {c["concept_id"]: c for c in report["concepts"]}

        photo = by_id["photosynthesis"]
        # explained via both occurrences AND defined-by edge.
        assert photo["surfaces"]["explained"] is True
        assert photo["surfaces"]["defined_in_glossary"] is True
        assert photo["surfaces"]["assessed"] is True
        assert photo["surfaces"]["demonstrated"] is True
        assert photo["surfaces"]["prereq_scaffolded"] is True
        assert photo["surface_count"] == 5
        assert photo["touching_chunks"] == ["c-1"]

        # chlorophyll: no occurrences, only is-a + prerequisite endpoints.
        chl = by_id["chlorophyll"]
        assert chl["surfaces"]["explained"] is False
        assert chl["surfaces"]["defined_in_glossary"] is True
        assert chl["surfaces"]["prereq_scaffolded"] is True
        assert chl["surface_count"] == 2

    def test_summary_counts_and_histogram(self, tmp_path):
        graph = {
            "nodes": [
                {"id": "a", "class": "DomainConcept", "occurrences": ["c1"]},
                {"id": "b", "class": "DomainConcept"},
            ],
            "edges": [
                _edge("c2", "a", "defined-by"),
            ],
        }
        _write_graph(tmp_path, graph)
        report = ConceptCoverageAggregator(libv2_course_path=tmp_path).build()
        s = report["summary"]
        assert s["total_concepts"] == 2
        assert s["concepts_explained"] == 1
        assert s["uncovered_concepts"] == ["b"]
        assert s["surface_count_histogram"]["0"] == 1
        assert s["surface_count_histogram"]["1"] == 1


class TestEndpointFiltering:
    def test_non_concept_nodes_excluded(self, tmp_path):
        graph = {
            "nodes": [
                {"id": "concept-x", "class": "DomainConcept"},
                {"id": "chunk-1", "class": "Chunk"},
                {"id": "TO-01", "class": "Outcome"},
                {"id": "classless-node"},  # classless == concept vocab
            ],
            "edges": [],
        }
        _write_graph(tmp_path, graph)
        report = ConceptCoverageAggregator(libv2_course_path=tmp_path).build()
        ids = {c["concept_id"] for c in report["concepts"]}
        assert ids == {"concept-x", "classless-node"}


class TestContentDerivedPrereq:
    def test_content_derived_prereq_edges_counted(self, tmp_path):
        graph = {
            "nodes": [
                {"id": "a", "class": "DomainConcept"},
                {"id": "b", "class": "DomainConcept"},
            ],
            "edges": [
                _edge("a", "b", "prerequisite",
                      rule="prerequisite_from_definition_mention"),
                _edge("a", "b", "prerequisite",
                      rule="prerequisite_from_lo_order"),
            ],
        }
        _write_graph(tmp_path, graph)
        report = ConceptCoverageAggregator(libv2_course_path=tmp_path).build()
        assert report["summary"]["content_derived_prereq_edges"] == 1


class TestMissingGraph:
    def test_no_graph_yields_empty_report(self, tmp_path):
        report = ConceptCoverageAggregator(libv2_course_path=tmp_path).build()
        assert report["concepts"] == []
        assert report["summary"]["total_concepts"] == 0
        assert report["schema_version"] == SCHEMA_VERSION

    def test_concept_graph_subdir_layout(self, tmp_path):
        graph = {"nodes": [{"id": "z", "class": "DomainConcept"}], "edges": []}
        _write_graph(tmp_path, graph, subdir="concept_graph")
        report = ConceptCoverageAggregator(libv2_course_path=tmp_path).build()
        assert [c["concept_id"] for c in report["concepts"]] == ["z"]


class TestSchemaValidation:
    def test_emitted_report_validates_against_schema(self, tmp_path):
        pytest.importorskip("jsonschema")
        from jsonschema import Draft202012Validator

        graph = {
            "nodes": [
                {"id": "a", "label": "A", "class": "DomainConcept",
                 "occurrences": ["c1"]},
                {"id": "b", "class": "DomainConcept"},
            ],
            "edges": [
                _edge("c2", "a", "defined-by"),
                _edge("a", "b", "prerequisite",
                      rule="prerequisite_from_definition_mention"),
            ],
        }
        _write_graph(tmp_path, graph)
        out_path = tmp_path / "concept_coverage.json"
        ConceptCoverageAggregator(
            course_code="PHYS_101", run_id="WF-1",
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


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class TestDecisionCapture:
    def test_one_event_per_build(self, tmp_path):
        graph = {"nodes": [{"id": "a", "class": "DomainConcept"}], "edges": []}
        _write_graph(tmp_path, graph)
        capture = _FakeCapture()
        ConceptCoverageAggregator(
            libv2_course_path=tmp_path, decision_capture=capture,
        ).build()
        assert len(capture.events) == 1
        evt = capture.events[0]
        assert evt["decision_type"] == "concept_coverage_aggregated"
        assert len(evt["rationale"]) >= 20


class TestFlagResolver:
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("", False), ("nonsense", False), (None, False),
    ])
    def test_parse_with_fallback(self, val, expected, monkeypatch):
        monkeypatch.delenv("ED4ALL_CONCEPT_COVERAGE", raising=False)
        assert resolve_concept_coverage(val) is expected

    def test_reads_env_when_value_none(self, monkeypatch):
        monkeypatch.setenv("ED4ALL_CONCEPT_COVERAGE", "on")
        assert resolve_concept_coverage() is True
        monkeypatch.delenv("ED4ALL_CONCEPT_COVERAGE", raising=False)
        assert resolve_concept_coverage() is False
