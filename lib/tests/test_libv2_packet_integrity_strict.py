"""Tests for Wave 78 strict-mode packet integrity rules.

Covers:

* The three new rules (``every_objective_has_teaching``,
  ``every_objective_has_assessment``, ``edge_endpoint_typing``) on
  fixtures that pass / fail each rule independently.
* Strict-mode flag precedence: ``--strict-coverage`` /
  ``--strict-typing`` / ``--strict`` (which implies both).
* The ``LIBV2_RELAX_PACKET_INTEGRITY=true`` escape hatch.
* The CLI exit code under strict mode.
* The dual-interface dispatch (Path → ValidationResult; dict →
  GateResult) including the gate-config ``strict: true`` path.
* End-to-end workflow gate behaviour: a phase that runs the
  ``packet_integrity_strict`` gate fails when the archive is
  broken.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.validators.libv2_packet_integrity import (
    COVERAGE_RULES,
    EDGE_TYPING_CONTRACT,
    PacketIntegrityValidator,
    RELAX_ENV_VAR,
    TYPING_RULES,
    ValidationResult,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _ok_archive(tmp_path: Path) -> Path:
    """Build a fixture archive that passes every Wave 78 rule.

    Two TOs + one CO under TO-01. Both TOs have direct teaching +
    assessment chunks so coverage rules + rollup all pass.
    """
    root = tmp_path / "course"
    root.mkdir(parents=True)

    _write_json(
        root / "objectives.json",
        {
            "terminal_outcomes": [
                {"id": "to-01", "statement": "Understand widgets."},
                {"id": "to-02", "statement": "Build widgets."},
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
    _write_json(
        root / "course.json",
        {
            "course_code": "WAVE_78",
            "title": "Wave78 fixtures",
            "learning_outcomes": [],
        },
    )

    _write_jsonl(
        root / "corpus" / "chunks.jsonl",
        [
            {
                "id": "chunk-001",
                "chunk_type": "explanation",
                "text": "A widget is a widget. The widget-kind concept matters.",
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
            {
                "id": "chunk-003",
                "chunk_type": "explanation",
                "text": "Building widgets requires assembly.",
                "concept_tags": ["widget-kind"],
                "learning_outcome_refs": ["to-02"],
            },
            {
                "id": "chunk-004",
                "chunk_type": "assessment_item",
                "text": "Q: How do you build a widget?",
                "concept_tags": [],
                "learning_outcome_refs": ["to-02"],
            },
        ],
    )

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
    _write_json(
        root / "graph" / "concept_graph_semantic.json",
        {
            "kind": "concept_graph_semantic",
            "nodes": [
                {"id": "widget-kind", "label": "Widget Kind", "class": "DomainConcept"},
            ],
            "edges": [],
        },
    )
    # Pedagogy graph — properly typed: every edge endpoint matches the
    # contract in EDGE_TYPING_CONTRACT.
    _write_json(
        root / "graph" / "pedagogy_graph.json",
        {
            "kind": "pedagogy_graph",
            "nodes": [
                {"id": "TO-01", "class": "Outcome", "label": "Understand widgets."},
                {"id": "TO-02", "class": "Outcome", "label": "Build widgets."},
                {"id": "CO-01", "class": "ComponentObjective", "label": "Identify widget kinds."},
                {"id": "chunk-001", "class": "Chunk", "label": "explanation"},
                {"id": "chunk-002", "class": "Chunk", "label": "assessment"},
                {"id": "module:wk1", "class": "Module", "label": "week 1"},
                {"id": "module:wk2", "class": "Module", "label": "week 2"},
                {"id": "concept:widget-kind", "class": "Concept", "label": "Widget Kind"},
            ],
            "edges": [
                {"source": "CO-01", "target": "TO-01", "relation_type": "supports_outcome"},
                {"source": "chunk-001", "target": "module:wk1", "relation_type": "belongs_to_module"},
                {"source": "module:wk1", "target": "module:wk2", "relation_type": "follows"},
                {"source": "chunk-001", "target": "concept:widget-kind", "relation_type": "exemplifies"},
                {"source": "concept:widget-kind", "target": "concept:widget-kind", "relation_type": "prerequisite_of"},
            ],
        },
    )
    (root / "quality").mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------- #
# 1. Baseline: every Wave 78 rule passes on the OK fixture
# ---------------------------------------------------------------------- #


def test_ok_fixture_no_critical_no_warning(tmp_path: Path):
    """A well-formed archive yields zero issues even under --strict."""
    root = _ok_archive(tmp_path)
    v = PacketIntegrityValidator(strict_coverage=True, strict_typing=True)
    result = v.validate(root)
    assert isinstance(result, ValidationResult)
    assert result.critical_count == 0
    assert result.warning_count == 0
    # All 12 rules ran (Wave 75: 9 + Wave 78: 3).
    assert result.rules_run == 12
    assert result.rules_passed == 12


# ---------------------------------------------------------------------- #
# 2. every_objective_has_teaching
# ---------------------------------------------------------------------- #


def _strip_co01_from_all_chunks(root: Path) -> None:
    """Helper: drop ``co-01`` from every chunk's learning_outcome_refs."""
    chunks_path = root / "corpus" / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    for c in chunks:
        c["learning_outcome_refs"] = [
            r for r in (c.get("learning_outcome_refs") or []) if r != "co-01"
        ]
    _write_jsonl(chunks_path, chunks)


def test_co_without_teaching_fires_warning_default(tmp_path: Path):
    """Default mode: missing teaching for a CO is a warning, not critical."""
    root = _ok_archive(tmp_path)
    _strip_co01_from_all_chunks(root)

    # Also drop any teaches edge to co-01 (none in fixture).
    result = PacketIntegrityValidator().validate(root)
    teaching_issues = [
        i for i in result.issues if i.issue_code == "OBJECTIVE_NO_TEACHING_CHUNK"
    ]
    assert teaching_issues, "expected OBJECTIVE_NO_TEACHING_CHUNK on co-01"
    assert teaching_issues[0].context["objective_id"] == "co-01"
    # Default mode: warning.
    assert teaching_issues[0].severity == "warning"


def test_co_without_teaching_fires_critical_under_strict_coverage(tmp_path: Path):
    root = _ok_archive(tmp_path)
    _strip_co01_from_all_chunks(root)

    result = PacketIntegrityValidator(strict_coverage=True).validate(root)
    teaching_issues = [
        i for i in result.issues if i.issue_code == "OBJECTIVE_NO_TEACHING_CHUNK"
    ]
    assert teaching_issues
    assert teaching_issues[0].severity == "critical"


def test_teaches_edge_satisfies_objective_teaching_coverage(tmp_path: Path):
    """A pedagogy_graph 'teaches' edge counts as teaching coverage."""
    root = _ok_archive(tmp_path)
    _strip_co01_from_all_chunks(root)

    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text())
    pg["edges"].append(
        {"source": "chunk-001", "target": "co-01", "relation_type": "teaches"}
    )
    pg_path.write_text(json.dumps(pg))

    result = PacketIntegrityValidator(strict_coverage=True).validate(root)
    teaching_issues = [
        i for i in result.issues if i.issue_code == "OBJECTIVE_NO_TEACHING_CHUNK"
    ]
    assert not teaching_issues, "teaches edge should satisfy coverage"


# ---------------------------------------------------------------------- #
# 3. every_objective_has_assessment
# ---------------------------------------------------------------------- #


def test_co_assessment_rolls_up_to_to(tmp_path: Path):
    """A CO with assessment satisfies its parent TO via rollup."""
    root = _ok_archive(tmp_path)
    chunks_path = root / "corpus" / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    # Make the assessment chunk reference only co-01 (not to-01) — TO
    # rollup should still cover to-01 via co-01.
    for c in chunks:
        if c["id"] == "chunk-002":
            c["learning_outcome_refs"] = ["co-01"]
    _write_jsonl(chunks_path, chunks)

    result = PacketIntegrityValidator(strict_coverage=True).validate(root)
    assess_issues = [
        i for i in result.issues if i.issue_code == "OBJECTIVE_NO_ASSESSMENT"
    ]
    # to-01 is rolled up via co-01; to-02 has its own assessment;
    # co-01 is directly covered. Zero issues.
    assert not assess_issues, [i.context for i in assess_issues]


def test_co_without_assessment_fires_critical_under_strict(tmp_path: Path):
    root = _ok_archive(tmp_path)
    chunks_path = root / "corpus" / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    # Add a third CO under TO-01 with no chunk + no assesses edge.
    obj = json.loads((root / "objectives.json").read_text())
    obj["component_objectives"].append(
        {
            "id": "co-02",
            "statement": "Lonely CO.",
            "parent_terminal": "to-01",
        },
    )
    (root / "objectives.json").write_text(json.dumps(obj))
    # Also need to provide teaching coverage to avoid double-firing
    # (we want this test to focus on assessment).
    chunks.append(
        {
            "id": "chunk-005",
            "chunk_type": "explanation",
            "text": "Some text covering CO-02.",
            "concept_tags": [],
            "learning_outcome_refs": ["co-02"],
        }
    )
    _write_jsonl(chunks_path, chunks)

    result = PacketIntegrityValidator(strict_coverage=True).validate(root)
    assess_issues = [
        i for i in result.issues if i.issue_code == "OBJECTIVE_NO_ASSESSMENT"
    ]
    assert assess_issues
    assert any(i.context["objective_id"] == "co-02" for i in assess_issues)
    assert all(i.severity == "critical" for i in assess_issues)


def test_assesses_edge_satisfies_objective_assessment(tmp_path: Path):
    root = _ok_archive(tmp_path)
    obj = json.loads((root / "objectives.json").read_text())
    obj["component_objectives"].append(
        {"id": "co-02", "statement": "Edge-only.", "parent_terminal": "to-01"}
    )
    (root / "objectives.json").write_text(json.dumps(obj))
    # Provide teaching for co-02 via a chunk, assessment via an edge.
    chunks_path = root / "corpus" / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    chunks.append(
        {
            "id": "chunk-005",
            "chunk_type": "explanation",
            "text": "Teaches CO-02.",
            "concept_tags": [],
            "learning_outcome_refs": ["co-02"],
        }
    )
    _write_jsonl(chunks_path, chunks)
    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text())
    pg["edges"].append(
        {"source": "chunk-005", "target": "co-02", "relation_type": "assesses"}
    )
    pg_path.write_text(json.dumps(pg))

    result = PacketIntegrityValidator(strict_coverage=True).validate(root)
    assess_issues = [
        i for i in result.issues if i.issue_code == "OBJECTIVE_NO_ASSESSMENT"
        and i.context.get("objective_id") == "co-02"
    ]
    assert not assess_issues


# ---------------------------------------------------------------------- #
# 4. edge_endpoint_typing
# ---------------------------------------------------------------------- #


def test_assesses_from_misconception_to_outcome_fires_typing(tmp_path: Path):
    """An 'assesses' edge with a Misconception source and Outcome target
    violates the typed-endpoint contract — the spec example."""
    root = _ok_archive(tmp_path)
    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text())
    pg["nodes"].append(
        {"id": "mc:bad", "class": "Misconception", "label": "Bad."}
    )
    pg["edges"].append(
        # assesses requires Chunk → Outcome | ComponentObjective.
        # Misconception → Outcome is illegal.
        {"source": "mc:bad", "target": "TO-01", "relation_type": "assesses"}
    )
    pg_path.write_text(json.dumps(pg))

    # Default — warning.
    result = PacketIntegrityValidator().validate(root)
    typing_issues = [
        i for i in result.issues if i.issue_code == "EDGE_ENDPOINT_TYPE_MISMATCH"
    ]
    assert typing_issues
    assert all(i.severity == "warning" for i in typing_issues)
    issue = typing_issues[0]
    assert issue.context["edge_type"] == "assesses"
    assert issue.context["source_class"] == "Misconception"

    # Strict typing — critical.
    result = PacketIntegrityValidator(strict_typing=True).validate(root)
    typing_issues = [
        i for i in result.issues if i.issue_code == "EDGE_ENDPOINT_TYPE_MISMATCH"
    ]
    assert typing_issues
    assert any(i.severity == "critical" for i in typing_issues)


def test_typing_rule_skips_unknown_relation_types(tmp_path: Path):
    """Edges with relations not in the contract are silently skipped."""
    root = _ok_archive(tmp_path)
    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text())
    pg["edges"].append(
        # 'magically' isn't in EDGE_TYPING_CONTRACT — should be ignored.
        {"source": "mc:nope", "target": "TO-01", "relation_type": "magically"}
    )
    pg_path.write_text(json.dumps(pg))

    result = PacketIntegrityValidator(strict_typing=True).validate(root)
    # The edge is dangling so it'll fire DANGLING_EDGE; that's fine.
    typing_issues = [
        i for i in result.issues if i.issue_code == "EDGE_ENDPOINT_TYPE_MISMATCH"
    ]
    assert not typing_issues


# ---------------------------------------------------------------------- #
# 5. --strict implies both
# ---------------------------------------------------------------------- #


def test_strict_flag_implies_both_strict_modes(tmp_path: Path):
    """Using ``strict=True`` (e.g., CLI --strict / gate config) flips
    both coverage and typing to critical."""
    root = _ok_archive(tmp_path)

    # Break both: drop CO-01 teaching coverage AND add a typing
    # violation.
    _strip_co01_from_all_chunks(root)

    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text())
    pg["nodes"].append({"id": "mc:bad", "class": "Misconception", "label": "x"})
    pg["edges"].append(
        {"source": "mc:bad", "target": "TO-01", "relation_type": "assesses"}
    )
    pg_path.write_text(json.dumps(pg))

    # Gate-shape with strict=True merges into both granular flags.
    v = PacketIntegrityValidator()
    gate_result = v.validate({"course_dir": str(root), "strict": True})
    codes = {i.code for i in gate_result.issues}
    assert "OBJECTIVE_NO_TEACHING_CHUNK" in codes
    assert "EDGE_ENDPOINT_TYPE_MISMATCH" in codes
    crit_codes = {i.code for i in gate_result.issues if i.severity == "critical"}
    assert "OBJECTIVE_NO_TEACHING_CHUNK" in crit_codes
    assert "EDGE_ENDPOINT_TYPE_MISMATCH" in crit_codes
    assert gate_result.passed is False


# ---------------------------------------------------------------------- #
# 6. LIBV2_RELAX_PACKET_INTEGRITY=true downgrades all to warnings
# ---------------------------------------------------------------------- #


def test_relax_env_var_downgrades_to_warning(tmp_path: Path, monkeypatch):
    """Even with --strict, the env override forces warning severity."""
    root = _ok_archive(tmp_path)

    # Break CO-01 teaching + add a typing violation.
    _strip_co01_from_all_chunks(root)

    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text())
    pg["nodes"].append({"id": "mc:bad", "class": "Misconception", "label": "x"})
    pg["edges"].append(
        {"source": "mc:bad", "target": "TO-01", "relation_type": "assesses"}
    )
    pg_path.write_text(json.dumps(pg))

    monkeypatch.setenv(RELAX_ENV_VAR, "true")
    result = PacketIntegrityValidator(
        strict_coverage=True, strict_typing=True
    ).validate(root)
    targets = {"OBJECTIVE_NO_TEACHING_CHUNK", "EDGE_ENDPOINT_TYPE_MISMATCH"}
    relevant = [i for i in result.issues if i.issue_code in targets]
    assert relevant, "expected the broken-fixture issues to fire"
    assert all(i.severity == "warning" for i in relevant), (
        "LIBV2_RELAX_PACKET_INTEGRITY=true must downgrade gated rules"
    )


# ---------------------------------------------------------------------- #
# 7. Dual-interface dispatch
# ---------------------------------------------------------------------- #


def test_path_input_returns_validation_result(tmp_path: Path):
    root = _ok_archive(tmp_path)
    res = PacketIntegrityValidator().validate(root)
    assert isinstance(res, ValidationResult)


def test_dict_input_returns_gate_result(tmp_path: Path):
    from MCP.hardening.validation_gates import GateResult

    root = _ok_archive(tmp_path)
    res = PacketIntegrityValidator().validate({"course_dir": str(root)})
    assert isinstance(res, GateResult)
    assert res.passed is True


def test_dict_input_without_archive_inputs_fails_critical():
    from MCP.hardening.validation_gates import GateResult

    res = PacketIntegrityValidator().validate({})
    assert isinstance(res, GateResult)
    assert res.passed is False
    assert any(i.code == "MISSING_ARCHIVE_INPUTS" for i in res.issues)


def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        PacketIntegrityValidator().validate(12345)


# ---------------------------------------------------------------------- #
# 8. Workflow gate test — config: { strict: true } end-to-end
# ---------------------------------------------------------------------- #


def test_packet_integrity_strict_gate_blocks_broken_archive(tmp_path: Path):
    """Simulate the libv2_archival workflow gate.

    Wires the validator + ``GateConfig(config={'strict': True})``
    through ``ValidationGateManager.run_gate`` with a deliberately
    broken archive (typing violation + missing CO teaching). The gate
    must fail closed.
    """
    from MCP.hardening.validation_gates import (
        GateConfig,
        GateSeverity,
        ValidationGateManager,
    )

    root = _ok_archive(tmp_path)

    # Break: drop CO-01 teaching + add Misconception → Outcome
    # 'assesses' edge.
    _strip_co01_from_all_chunks(root)
    pg_path = root / "graph" / "pedagogy_graph.json"
    pg = json.loads(pg_path.read_text())
    pg["nodes"].append({"id": "mc:bad", "class": "Misconception", "label": "x"})
    pg["edges"].append(
        {"source": "mc:bad", "target": "TO-01", "relation_type": "assesses"}
    )
    pg_path.write_text(json.dumps(pg))

    gate = GateConfig(
        gate_id="packet_integrity_strict",
        validator_path=(
            "lib.validators.libv2_packet_integrity.PacketIntegrityValidator"
        ),
        severity=GateSeverity.CRITICAL,
        threshold={"max_critical_issues": 0},
        config={"strict": True},
    )
    manager = ValidationGateManager()
    result = manager.run_gate(
        gate, {"course_dir": str(root), "manifest_path": str(root / "manifest.json")}
    )
    assert result.passed is False
    codes = {i.code if hasattr(i, "code") else i["code"] for i in result.issues}
    assert "OBJECTIVE_NO_TEACHING_CHUNK" in codes
    assert "EDGE_ENDPOINT_TYPE_MISMATCH" in codes


def test_packet_integrity_strict_gate_passes_clean_archive(tmp_path: Path):
    from MCP.hardening.validation_gates import (
        GateConfig,
        GateSeverity,
        ValidationGateManager,
    )

    root = _ok_archive(tmp_path)
    gate = GateConfig(
        gate_id="packet_integrity_strict",
        validator_path=(
            "lib.validators.libv2_packet_integrity.PacketIntegrityValidator"
        ),
        severity=GateSeverity.CRITICAL,
        threshold={"max_critical_issues": 0},
        config={"strict": True},
    )
    manager = ValidationGateManager()
    result = manager.run_gate(gate, {"course_dir": str(root)})
    assert result.passed is True
    assert result.critical_count == 0


# ---------------------------------------------------------------------- #
# 9. Severity-resolution helper
# ---------------------------------------------------------------------- #


def test_resolve_severity_default_warning():
    v = PacketIntegrityValidator()
    for rule in COVERAGE_RULES | TYPING_RULES:
        assert v._resolve_severity(rule) == "warning"


def test_resolve_severity_strict_coverage_promotes():
    v = PacketIntegrityValidator(strict_coverage=True)
    for rule in COVERAGE_RULES:
        assert v._resolve_severity(rule) == "critical"
    for rule in TYPING_RULES:
        # strict_coverage doesn't touch typing rules.
        assert v._resolve_severity(rule) == "warning"


def test_resolve_severity_strict_typing_promotes():
    v = PacketIntegrityValidator(strict_typing=True)
    for rule in TYPING_RULES:
        assert v._resolve_severity(rule) == "critical"
    for rule in COVERAGE_RULES:
        assert v._resolve_severity(rule) == "warning"


def test_edge_typing_contract_covers_spec():
    expected_relations = {
        "teaches",
        "assesses",
        "practices",
        "exemplifies",
        "supports_outcome",
        "interferes_with",
        "prerequisite_of",
        "belongs_to_module",
        "at_bloom_level",
        "follows",
    }
    assert set(EDGE_TYPING_CONTRACT.keys()) == expected_relations
