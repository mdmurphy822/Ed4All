"""Tests for ``lib.validators.libv2_packet_integrity`` (Wave 75 Worker D).

Builds tiny fixture archives that exercise each rule in the
``PacketIntegrityValidator`` catalog independently, then runs the
validator on the real ``LibV2/courses/rdf-shacl-550-rdf-shacl-550``
archive as a regression baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.validators.libv2_packet_integrity import (
    PacketIntegrityValidator,
    RULE_SEVERITY,
    SCAFFOLDING_CLASSES,
    ValidationResult,
)


# ---------------------------------------------------------------------- #
# Helpers — build a minimal valid archive on disk
# ---------------------------------------------------------------------- #


def _write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _baseline_archive(tmp_path: Path) -> Path:
    """Build a 1-TO + 1-CO + 2-chunk fixture that passes every rule."""
    root = tmp_path / "course"
    root.mkdir(parents=True)

    # Objectives — 1 TO, 1 CO; CO points at TO.
    _write_json(
        root / "objectives.json",
        {
            "terminal_outcomes": [
                {"id": "to-01", "statement": "Understand widgets."},
            ],
            "component_objectives": [
                {
                    "id": "co-01",
                    "statement": "Identify widget kinds.",
                    "parent_terminal": "to-01",
                },
            ],
        },
    )

    # course.json — present so the manifest gate isn't grumpy if used
    _write_json(
        root / "course.json",
        {
            "course_code": "FX_101",
            "title": "Fixtures 101",
            "learning_outcomes": [
                {
                    "id": "to-01",
                    "statement": "Understand widgets.",
                    "hierarchy_level": "terminal",
                },
                {
                    "id": "co-01",
                    "statement": "Identify widget kinds.",
                    "hierarchy_level": "chapter",
                },
            ],
        },
    )

    # Chunks — one explanation (teaches both to-01 and co-01) + one
    # assessment that covers both objectives directly. Wave 78 strict
    # coverage rules require each TO + CO to have a teaching chunk
    # AND an assessment chunk (with TO rollup); referencing both
    # objectives explicitly satisfies that fixture-side.
    _write_jsonl(
        root / "corpus" / "chunks.jsonl",
        [
            {
                "id": "chunk-001",
                "chunk_type": "explanation",
                "text": "A widget is a widget. The widget-kind concept is foundational.",
                "concept_tags": ["widget-kind"],
                "learning_outcome_refs": ["to-01", "co-01"],
            },
            {
                "id": "chunk-002",
                "chunk_type": "assessment_item",
                "text": "Q: What is a widget?",
                "concept_tags": [],
                "learning_outcome_refs": ["to-01", "co-01"],
            },
        ],
    )

    # Concept graph — one DomainConcept node that's referenced in
    # chunk-001 above. One self-edge so the graph isn't empty.
    _write_json(
        root / "graph" / "concept_graph.json",
        {
            "kind": "concept_graph",
            "nodes": [
                {"id": "widget-kind", "label": "Widget Kind", "class": "DomainConcept"},
            ],
            "edges": [],
        },
    )

    # Concept graph semantic — one edge that targets the DomainConcept
    # (so it's NOT a scaffolding-as-assessed violation).
    _write_json(
        root / "graph" / "concept_graph_semantic.json",
        {
            "kind": "concept_graph_semantic",
            "nodes": [
                {"id": "widget-kind", "label": "Widget Kind", "class": "DomainConcept"},
            ],
            "edges": [
                {
                    "source": "co-01",
                    "target": "widget-kind",
                    "type": "derived-from-objective",
                },
            ],
        },
    )

    # Pedagogy graph — minimal, well-formed (every edge resolves).
    _write_json(
        root / "graph" / "pedagogy_graph.json",
        {
            "kind": "pedagogy_graph",
            "nodes": [
                {"id": "TO-01", "class": "TerminalOutcome", "label": "Understand widgets."},
                {"id": "CO-01", "class": "ComponentObjective", "label": "Identify widget kinds."},
            ],
            "edges": [
                {"source": "CO-01", "target": "TO-01", "relation_type": "supports_outcome"},
            ],
        },
    )

    # Quality dir — placeholder so the report can be written.
    (root / "quality").mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------- #
# 1. Baseline fixture: every rule passes
# ---------------------------------------------------------------------- #


def test_baseline_archive_passes_every_rule(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    result = PacketIntegrityValidator().validate(root)

    assert isinstance(result, ValidationResult)
    # Wave 78 added 3 rules: every_objective_has_teaching,
    # every_objective_has_assessment, edge_endpoint_typing.
    assert result.rules_run == 12
    assert result.rules_passed == 12
    assert result.rules_failed == 0
    assert result.issues == []
    assert result.summary["chunk_count"] == 2
    assert result.summary["terminal_outcome_count"] == 1
    assert result.summary["component_outcome_count"] == 1
    assert result.summary["objectives_source"] == "objectives.json"


# ---------------------------------------------------------------------- #
# 2. UNRESOLVED_OBJECTIVE_REF
# ---------------------------------------------------------------------- #


def test_unresolved_objective_ref_fires(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    # Append a chunk referencing a nonexistent CO.
    chunks_path = root / "corpus" / "chunks.jsonl"
    existing = chunks_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
    existing.append(
        json.dumps(
            {
                "id": "chunk-bad",
                "chunk_type": "explanation",
                "text": "An orphan reference.",
                "concept_tags": [],
                "learning_outcome_refs": ["co-99"],
            }
        )
    )
    chunks_path.write_text("\n".join(existing) + "\n", encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    codes = [i.issue_code for i in result.issues]
    assert "UNRESOLVED_OBJECTIVE_REF" in codes
    rule_failed = next(
        i for i in result.issues if i.issue_code == "UNRESOLVED_OBJECTIVE_REF"
    )
    assert rule_failed.severity == "critical"
    assert rule_failed.context["ref"] == "co-99"


# ---------------------------------------------------------------------- #
# 3. ORPHAN_COMPONENT_OBJECTIVE
# ---------------------------------------------------------------------- #


def test_orphan_component_objective_fires_no_parent(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    obj = json.loads((root / "objectives.json").read_text(encoding="utf-8"))
    obj["component_objectives"][0]["parent_terminal"] = ""
    (root / "objectives.json").write_text(json.dumps(obj), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    codes = [i.issue_code for i in result.issues]
    assert "ORPHAN_COMPONENT_OBJECTIVE" in codes


def test_orphan_component_objective_fires_unknown_parent(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    obj = json.loads((root / "objectives.json").read_text(encoding="utf-8"))
    obj["component_objectives"][0]["parent_terminal"] = "to-99"
    (root / "objectives.json").write_text(json.dumps(obj), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    issues = [
        i for i in result.issues if i.issue_code == "ORPHAN_COMPONENT_OBJECTIVE"
    ]
    assert issues, "expected ORPHAN_COMPONENT_OBJECTIVE for unknown parent"
    assert issues[0].context["parent_terminal"] == "to-99"


# ---------------------------------------------------------------------- #
# 4. UNANCHORED_ASSESSMENT
# ---------------------------------------------------------------------- #


def test_unanchored_assessment_fires(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    # Drop refs from the assessment chunk.
    items = [
        json.loads(line)
        for line in (root / "corpus" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for item in items:
        if item["id"] == "chunk-002":
            item["learning_outcome_refs"] = []
    _write_jsonl(root / "corpus" / "chunks.jsonl", items)

    result = PacketIntegrityValidator().validate(root)
    codes = [i.issue_code for i in result.issues]
    assert "UNANCHORED_ASSESSMENT" in codes


# ---------------------------------------------------------------------- #
# 5. UNCOVERED_TERMINAL_OUTCOME
# ---------------------------------------------------------------------- #


def test_uncovered_terminal_outcome_fires(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    # Add a TO that has zero chunks referencing it (or a CO under it).
    obj = json.loads((root / "objectives.json").read_text(encoding="utf-8"))
    obj["terminal_outcomes"].append(
        {"id": "to-02", "statement": "Lonely outcome."}
    )
    (root / "objectives.json").write_text(json.dumps(obj), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    issues = [
        i for i in result.issues if i.issue_code == "UNCOVERED_TERMINAL_OUTCOME"
    ]
    assert issues, "expected UNCOVERED_TERMINAL_OUTCOME for to-02"
    assert issues[0].context["to_id"] == "to-02"


# ---------------------------------------------------------------------- #
# 6. ORPHAN_DOMAIN_CONCEPT
# ---------------------------------------------------------------------- #


def test_orphan_domain_concept_fires(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    cg_path = root / "graph" / "concept_graph.json"
    cg = json.loads(cg_path.read_text(encoding="utf-8"))
    cg["nodes"].append(
        {
            "id": "phantom-concept",
            "label": "Phantom Concept",
            "class": "DomainConcept",
        }
    )
    cg_path.write_text(json.dumps(cg), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    codes = [i.issue_code for i in result.issues]
    assert "ORPHAN_DOMAIN_CONCEPT" in codes


# ---------------------------------------------------------------------- #
# 7. DANGLING_EDGE
# ---------------------------------------------------------------------- #


def test_dangling_edge_fires_in_concept_graph(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    cg_path = root / "graph" / "concept_graph.json"
    cg = json.loads(cg_path.read_text(encoding="utf-8"))
    cg["edges"].append(
        {"source": "widget-kind", "target": "ghost-node", "weight": 1}
    )
    cg_path.write_text(json.dumps(cg), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    issues = [i for i in result.issues if i.issue_code == "DANGLING_EDGE"]
    assert issues
    assert any(i.context["graph"] == "concept_graph" for i in issues)


def test_dangling_edge_fires_in_pedagogy_graph(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text(encoding="utf-8"))
    pg["edges"].append(
        {"source": "TO-01", "target": "GHOST-NODE", "relation_type": "x"}
    )
    pg_path.write_text(json.dumps(pg), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    issues = [
        i
        for i in result.issues
        if i.issue_code == "DANGLING_EDGE" and i.context["graph"] == "pedagogy_graph"
    ]
    assert issues


# ---------------------------------------------------------------------- #
# 8. MALFORMED_COMMA_REF
# ---------------------------------------------------------------------- #


def test_malformed_comma_ref_fires(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    items = [
        json.loads(line)
        for line in (root / "corpus" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items[0]["learning_outcome_refs"] = ["co-01,co-02"]
    _write_jsonl(root / "corpus" / "chunks.jsonl", items)

    result = PacketIntegrityValidator().validate(root)
    codes = [i.issue_code for i in result.issues]
    assert "MALFORMED_COMMA_REF" in codes


# ---------------------------------------------------------------------- #
# 9. SCAFFOLDING_AS_ASSESSED
# ---------------------------------------------------------------------- #


def test_scaffolding_as_assessed_fires(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    # Add a PedagogicalMarker node + an edge whose target is that node.
    cg_path = root / "graph" / "concept_graph.json"
    cg = json.loads(cg_path.read_text(encoding="utf-8"))
    cg["nodes"].append(
        {
            "id": "key-takeaway",
            "label": "Key Takeaway",
            "class": "PedagogicalMarker",
        }
    )
    cg_path.write_text(json.dumps(cg), encoding="utf-8")

    sem_path = root / "graph" / "concept_graph_semantic.json"
    sem = json.loads(sem_path.read_text(encoding="utf-8"))
    sem["edges"].append(
        {
            "source": "co-01",
            "target": "key-takeaway",
            "type": "derived-from-objective",
        }
    )
    sem_path.write_text(json.dumps(sem), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    codes = [i.issue_code for i in result.issues]
    assert "SCAFFOLDING_AS_ASSESSED" in codes
    issue = next(i for i in result.issues if i.issue_code == "SCAFFOLDING_AS_ASSESSED")
    assert issue.context["target"] == "key-takeaway"
    assert issue.context["target_class"] in SCAFFOLDING_CLASSES


# ---------------------------------------------------------------------- #
# 10. DUPLICATE_CHUNK_ID
# ---------------------------------------------------------------------- #


def test_duplicate_chunk_id_fires(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    # Append a duplicate of chunk-001.
    chunks_path = root / "corpus" / "chunks.jsonl"
    existing = chunks_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
    existing.append(
        json.dumps(
            {
                "id": "chunk-001",
                "chunk_type": "explanation",
                "text": "Duplicate.",
                "concept_tags": [],
                "learning_outcome_refs": ["co-01"],
            }
        )
    )
    chunks_path.write_text("\n".join(existing) + "\n", encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    codes = [i.issue_code for i in result.issues]
    assert "DUPLICATE_CHUNK_ID" in codes


# ---------------------------------------------------------------------- #
# Course.json fallback path
# ---------------------------------------------------------------------- #


def test_course_json_fallback_when_objectives_missing(tmp_path: Path):
    root = _baseline_archive(tmp_path)
    # Remove objectives.json so the loader falls back to course.json.
    (root / "objectives.json").unlink()
    # Pad course.json with a parent-terminal-bearing chapter LO so the
    # fallback path itself doesn't trip co_has_parent. (course.json
    # schema doesn't formally carry parent_terminal, but the loader
    # tolerates it being present for degraded-archive coverage.)
    course = json.loads((root / "course.json").read_text(encoding="utf-8"))
    for lo in course["learning_outcomes"]:
        if lo.get("hierarchy_level") == "chapter":
            lo["parent_terminal"] = "to-01"
    (root / "course.json").write_text(json.dumps(course), encoding="utf-8")

    result = PacketIntegrityValidator().validate(root)
    assert result.summary["objectives_source"].startswith("course.json")
    # The fallback parses correctly and refs_resolve still passes.
    codes = [i.issue_code for i in result.issues]
    assert "UNRESOLVED_OBJECTIVE_REF" not in codes
    # Source switched cleanly.
    assert result.summary["terminal_outcome_count"] == 1
    assert result.summary["component_outcome_count"] == 1


# ---------------------------------------------------------------------- #
# Severity table sanity
# ---------------------------------------------------------------------- #


def test_rule_severity_catalog_matches_spec():
    expected_critical = {
        "unique_chunk_ids",
        "refs_resolve",
        "co_has_parent",
        "no_comma_refs",
        "graph_edges_resolve",
    }
    expected_warning = {
        "assessment_has_objective",
        "to_has_teaching_and_assessment",
        "domain_concept_has_chunk",
        "scaffolding_not_assessed",
        # Wave 78 — coverage + typing rules default to warning;
        # promoted to critical via --strict-coverage / --strict-typing.
        "every_objective_has_teaching",
        "every_objective_has_assessment",
        "edge_endpoint_typing",
    }
    actual_critical = {k for k, v in RULE_SEVERITY.items() if v == "critical"}
    actual_warning = {k for k, v in RULE_SEVERITY.items() if v == "warning"}
    assert actual_critical == expected_critical
    assert actual_warning == expected_warning


# ---------------------------------------------------------------------- #
# Sanity baseline on the real archive
# ---------------------------------------------------------------------- #

REAL_ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "LibV2"
    / "courses"
    / "rdf-shacl-550-rdf-shacl-550"
)


@pytest.mark.skipif(
    not REAL_ARCHIVE.exists(),
    reason="rdf-shacl-550 archive not present in this checkout",
)
def test_real_archive_has_zero_critical_after_workers_a_b_c():
    """Regression baseline.

    After Workers A (objectives.json), B (concept node typing), and C
    (pedagogy graph) land, the rdf-shacl-550 archive should have 0
    critical issues. Warnings are allowed and reported for honesty.
    """
    result = PacketIntegrityValidator().validate(REAL_ARCHIVE)
    assert (
        result.critical_count == 0
    ), (
        f"Expected 0 critical issues on rdf-shacl-550 baseline, got "
        f"{result.critical_count}: "
        f"{[i.issue_code for i in result.issues if i.severity == 'critical']}"
    )
    # Sanity: every rule actually executed.
    assert result.rules_run == 12
