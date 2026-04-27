"""Tests for ``Trainforge/rag/kg_quality_report.py``.

Pure aggregation tests — no SHACL evaluation pass is invoked. Synthetic
ValidationReport-shaped objects are constructed via a thin
``_StubResult`` dataclass that mirrors the canonical SHACL result
fields (``severity``, ``source_shape``, ``focus_node``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

# Add repo root for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Trainforge.rag.kg_quality_report import (  # noqa: E402
    DEFAULT_REQUIRED_PREDICATES,
    KGQualityReporter,
    RULE_GRAPH_IRI_PREFIX,
)


# ---------------------------------------------------------------------- #
# Synthetic ValidationReport infrastructure
# ---------------------------------------------------------------------- #


@dataclass
class _StubResult:
    """Mirrors the fields the reporter reads off each SHACL result."""

    severity: str
    source_shape: Optional[str] = None
    focus_node: Optional[str] = None
    message: str = ""


@dataclass
class _StubReport:
    """Mirrors pyshacl's ValidationReport surface for test purposes.

    The reporter walks ``report.results`` for severity / source_shape /
    focus_node — exactly what we expose here.
    """

    results: List[_StubResult] = field(default_factory=list)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _write_concept_graph(path: Path, *, node_count: int,
                         edge_count: int = 0,
                         missing_label_count: int = 0) -> None:
    """Build a synthetic concept_graph.json with N nodes, optionally
    missing the ``label`` predicate on the first ``missing_label_count``.
    """
    nodes = []
    for i in range(node_count):
        node = {"id": f"concept-{i:03d}"}
        if i >= missing_label_count:
            node["label"] = f"Concept {i:03d}"
        nodes.append(node)

    edges = []
    for j in range(edge_count):
        edges.append({
            "source": f"concept-{j % node_count:03d}",
            "target": f"concept-{(j + 1) % node_count:03d}",
            "type": "related-to",
        })
    path.write_text(
        json.dumps({"kind": "concept", "nodes": nodes, "edges": edges}),
        encoding="utf-8",
    )


def _write_semantic_graph(
    path: Path,
    *,
    rule_edges: dict,
    rule_versions: Optional[dict] = None,
) -> None:
    """Build a synthetic concept_graph_semantic.json from a
    ``{rule_name: edge_count}`` mapping.
    """
    edges = []
    for rule, count in rule_edges.items():
        for k in range(count):
            edges.append({
                "source": f"concept-a-{k}",
                "target": f"concept-b-{k}",
                "type": "related-to",
                "provenance": {
                    "rule": rule,
                    "rule_version": (rule_versions or {}).get(rule, 1),
                },
            })
    payload = {
        "kind": "concept_semantic",
        "generated_at": "2026-04-26T00:00:00+00:00",
        "rule_versions": rule_versions or {r: 1 for r in rule_edges},
        "nodes": [],
        "edges": edges,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def graphs(tmp_path: Path):
    """Return a (concept_path, semantic_path) tuple under tmp_path."""
    concept = tmp_path / "concept_graph.json"
    semantic = tmp_path / "concept_graph_semantic.json"
    return concept, semantic


@pytest.fixture()
def reporter(tmp_path: Path) -> KGQualityReporter:
    return KGQualityReporter(
        course_slug="test-course",
        run_id="test-run-001",
        output_dir=tmp_path / "report_out",
    )


# ---------------------------------------------------------------------- #
# Completeness
# ---------------------------------------------------------------------- #


def test_completeness_perfect(graphs, reporter: KGQualityReporter):
    """Every node carries every required predicate -> score 1.0."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=10)
    _write_semantic_graph(semantic, rule_edges={})
    report = reporter.compute(concept, semantic, _StubReport())
    completeness = report["dimensions"]["completeness"]
    assert completeness["score"] == 1.0
    assert completeness["denominator"] == 10
    assert completeness["numerator"] == 10
    assert completeness["required_predicates"] == DEFAULT_REQUIRED_PREDICATES


def test_completeness_with_violations(graphs, reporter: KGQualityReporter):
    """1 of 10 nodes missing the ``label`` predicate -> score 0.9."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=10, missing_label_count=1)
    _write_semantic_graph(semantic, rule_edges={})
    report = reporter.compute(concept, semantic, _StubReport())
    completeness = report["dimensions"]["completeness"]
    assert completeness["score"] == 0.9
    assert completeness["denominator"] == 10
    assert completeness["numerator"] == 9


# ---------------------------------------------------------------------- #
# Consistency / accuracy
# ---------------------------------------------------------------------- #


def test_consistency_violation_counting(graphs, reporter: KGQualityReporter):
    """5 critical violations + 100 nodes -> consistency = 0.95."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=100)
    _write_semantic_graph(semantic, rule_edges={})
    results = [
        _StubResult(severity="critical", source_shape="urn:shape:A",
                    focus_node=f"node-{i}")
        for i in range(5)
    ]
    report = reporter.compute(concept, semantic, _StubReport(results=results))
    consistency = report["dimensions"]["consistency"]
    assert consistency["score"] == 0.95
    assert consistency["violation_count"] == 5
    assert consistency["total_focus_nodes"] == 100


def test_accuracy_uses_warning_count(graphs, reporter: KGQualityReporter):
    """20 warnings + 100 nodes -> accuracy = 0.8."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=100)
    _write_semantic_graph(semantic, rule_edges={})
    results = [
        _StubResult(severity="warning", source_shape="urn:shape:Range",
                    focus_node=f"node-{i}")
        for i in range(20)
    ]
    report = reporter.compute(concept, semantic, _StubReport(results=results))
    accuracy = report["dimensions"]["accuracy"]
    assert accuracy["score"] == 0.8
    assert accuracy["warning_count"] == 20


def test_severity_iri_normalization(graphs, reporter: KGQualityReporter):
    """SHACL IRI severities should normalize to critical/warning."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=10)
    _write_semantic_graph(semantic, rule_edges={})
    results = [
        _StubResult(severity="http://www.w3.org/ns/shacl#Violation",
                    source_shape="urn:shape:A", focus_node="n1"),
        _StubResult(severity="http://www.w3.org/ns/shacl#Warning",
                    source_shape="urn:shape:B", focus_node="n2"),
    ]
    report = reporter.compute(concept, semantic, _StubReport(results=results))
    assert report["dimensions"]["consistency"]["violation_count"] == 1
    assert report["dimensions"]["accuracy"]["warning_count"] == 1


# ---------------------------------------------------------------------- #
# Coverage (asserted vs derived)
# ---------------------------------------------------------------------- #


def test_coverage_derived_triple_diff(graphs, reporter: KGQualityReporter):
    """10 asserted edges + 50 derived edges -> coverage = 10/60 ≈ 0.1667."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=20, edge_count=10)
    _write_semantic_graph(
        semantic,
        rule_edges={"is_a_from_key_terms": 30, "related_from_cooccurrence": 20},
        rule_versions={"is_a_from_key_terms": 1,
                       "related_from_cooccurrence": 2},
    )
    report = reporter.compute(concept, semantic, _StubReport())
    coverage = report["dimensions"]["coverage"]
    assert coverage["asserted_count"] == 10
    assert coverage["derived_count"] == 50
    assert coverage["score"] == round(10 / 60, 4)


def test_rule_outputs_iri_scheme_matches_named_graph_writer(
    graphs, reporter: KGQualityReporter,
):
    """rule_iri must follow https://ed4all.io/run/<run_id>/rule/<rule>."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=5, edge_count=0)
    _write_semantic_graph(
        semantic,
        rule_edges={"is_a_from_key_terms": 3},
        rule_versions={"is_a_from_key_terms": 1},
    )
    report = reporter.compute(concept, semantic, _StubReport())
    rule_outputs = report["rule_outputs"]
    assert len(rule_outputs) == 1
    iri = rule_outputs[0]["rule_iri"]
    assert iri.startswith(RULE_GRAPH_IRI_PREFIX)
    assert iri == "https://ed4all.io/run/test-run-001/rule/is_a_from_key_terms"
    assert rule_outputs[0]["edge_count"] == 3
    assert rule_outputs[0]["rule_version"] == 1


def test_coverage_no_edges_handles_zero_division(
    graphs, reporter: KGQualityReporter,
):
    """No asserted + no derived -> coverage 1.0 (vacuously true)."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=5, edge_count=0)
    _write_semantic_graph(semantic, rule_edges={})
    report = reporter.compute(concept, semantic, _StubReport())
    assert report["dimensions"]["coverage"]["score"] == 1.0


# ---------------------------------------------------------------------- #
# Per-shape rollup
# ---------------------------------------------------------------------- #


def test_per_shape_rollup(graphs, reporter: KGQualityReporter):
    """3 distinct source_shape IRIs -> 3 entries with correct counts."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=20)
    _write_semantic_graph(semantic, rule_edges={})
    results = [
        _StubResult(severity="critical", source_shape="urn:shape:A",
                    focus_node="n1"),
        _StubResult(severity="critical", source_shape="urn:shape:A",
                    focus_node="n2"),
        _StubResult(severity="warning", source_shape="urn:shape:B",
                    focus_node="n3"),
        _StubResult(severity="critical", source_shape="urn:shape:C",
                    focus_node="n4"),
        _StubResult(severity="warning", source_shape="urn:shape:C",
                    focus_node="n5"),
    ]
    report = reporter.compute(concept, semantic, _StubReport(results=results))
    per_shape = report["per_shape"]
    assert len(per_shape) == 3
    by_iri = {row["shape_iri"]: row for row in per_shape}
    assert by_iri["urn:shape:A"]["violations"] == 2
    assert by_iri["urn:shape:A"]["warnings"] == 0
    assert by_iri["urn:shape:A"]["focus_nodes"] == 2
    assert by_iri["urn:shape:B"]["violations"] == 0
    assert by_iri["urn:shape:B"]["warnings"] == 1
    assert by_iri["urn:shape:C"]["violations"] == 1
    assert by_iri["urn:shape:C"]["warnings"] == 1
    assert by_iri["urn:shape:C"]["focus_nodes"] == 2


# ---------------------------------------------------------------------- #
# End-to-end write / read
# ---------------------------------------------------------------------- #


def test_writes_json_to_disk(graphs, reporter: KGQualityReporter):
    """End-to-end: synthesize -> write -> read -> all 4 dimensions present."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=10, edge_count=5)
    _write_semantic_graph(
        semantic,
        rule_edges={"is_a_from_key_terms": 3},
        rule_versions={"is_a_from_key_terms": 1},
    )
    results = [
        _StubResult(severity="critical", source_shape="urn:shape:A",
                    focus_node="n1"),
    ]
    report = reporter.compute(concept, semantic, _StubReport(results=results))
    out_path = reporter.write(report)

    assert out_path.exists()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["course_slug"] == "test-course"
    assert parsed["run_id"] == "test-run-001"
    assert "generated_at" in parsed
    dims = parsed["dimensions"]
    assert set(dims.keys()) == {
        "completeness", "consistency", "accuracy", "coverage"
    }
    for dim in dims.values():
        assert "score" in dim
        assert "metric" in dim
    assert isinstance(parsed["per_shape"], list)
    assert isinstance(parsed["rule_outputs"], list)


def test_accepts_dict_validation_report(graphs, reporter: KGQualityReporter):
    """Validation report passed as a plain dict with results key."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=5)
    _write_semantic_graph(semantic, rule_edges={})
    report_dict = {"results": [
        {"severity": "critical", "source_shape": "urn:s:A", "focus_node": "n1"}
    ]}
    report = reporter.compute(concept, semantic, report_dict)
    assert report["dimensions"]["consistency"]["violation_count"] == 1
    assert report["per_shape"][0]["shape_iri"] == "urn:s:A"


def test_accepts_list_validation_report(graphs, reporter: KGQualityReporter):
    """Validation report passed as a bare list of result objects."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=5)
    _write_semantic_graph(semantic, rule_edges={})
    results = [_StubResult(severity="warning", source_shape="urn:s:B",
                           focus_node="n1")]
    report = reporter.compute(concept, semantic, results)
    assert report["dimensions"]["accuracy"]["warning_count"] == 1


def test_pedagogy_graph_path_recorded(graphs, reporter: KGQualityReporter,
                                       tmp_path: Path):
    """Optional pedagogy_graph path is preserved in the report."""
    concept, semantic = graphs
    _write_concept_graph(concept, node_count=3)
    _write_semantic_graph(semantic, rule_edges={})
    pedagogy = tmp_path / "pedagogy_graph.json"
    pedagogy.write_text("{}", encoding="utf-8")
    report = reporter.compute(
        concept, semantic, _StubReport(),
        pedagogy_graph=pedagogy,
    )
    assert report["pedagogy_graph_path"] == str(pedagogy)
