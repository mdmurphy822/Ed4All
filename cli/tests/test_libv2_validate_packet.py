"""Tests for ``ed4all libv2 validate-packet`` (Wave 75 Worker D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.commands.libv2_validate_packet import libv2_group


# ---------------------------------------------------------------------- #
# Fixture archive helpers (parallel to test_libv2_packet_integrity.py
# but kept local so the CLI tests don't depend on the validator tests).
# ---------------------------------------------------------------------- #


def _write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_passing_archive(courses_root: Path, slug: str) -> Path:
    """Build a minimal archive that passes every rule (Wave 75 + Wave 78)."""
    root = courses_root / slug
    root.mkdir(parents=True)
    _write_json(
        root / "objectives.json",
        {
            "terminal_outcomes": [
                {"id": "to-01", "statement": "Understand widgets."}
            ],
            "component_objectives": [
                {
                    "id": "co-01",
                    "statement": "Identify widgets.",
                    "parent_terminal": "to-01",
                }
            ],
        },
    )
    _write_json(
        root / "course.json",
        {
            "course_code": "FX_101",
            "title": "Fixtures",
            "learning_outcomes": [
                {"id": "to-01", "statement": "x", "hierarchy_level": "terminal"},
            ],
        },
    )
    # Wave 78: chunks must cover both TO + CO with teaching AND
    # assessment so every_objective_has_{teaching,assessment} pass.
    _write_jsonl(
        root / "corpus" / "chunks.jsonl",
        [
            {
                "id": "c1",
                "chunk_type": "explanation",
                "text": "Widget kind explained.",
                "concept_tags": ["widget-kind"],
                "learning_outcome_refs": ["to-01", "co-01"],
            },
            {
                "id": "c2",
                "chunk_type": "assessment_item",
                "text": "Q",
                "learning_outcome_refs": ["to-01", "co-01"],
            },
        ],
    )
    _write_json(
        root / "graph" / "concept_graph.json",
        {
            "nodes": [
                {"id": "widget-kind", "label": "Widget Kind", "class": "DomainConcept"}
            ],
            "edges": [],
        },
    )
    _write_json(
        root / "graph" / "concept_graph_semantic.json",
        {"nodes": [{"id": "widget-kind", "class": "DomainConcept"}], "edges": []},
    )
    _write_json(
        root / "graph" / "pedagogy_graph.json",
        {
            "nodes": [{"id": "TO-01", "class": "Outcome"}],
            "edges": [],
        },
    )
    return root


def _build_archive_with_warning_only(courses_root: Path, slug: str) -> Path:
    """Build an archive that has warnings but no critical issues.

    Achieved by adding an assessment_item chunk with no
    learning_outcome_refs (warning: UNANCHORED_ASSESSMENT) without
    breaking any critical rule.
    """
    root = _build_passing_archive(courses_root, slug)
    # Append an unanchored assessment chunk.
    chunks_path = root / "corpus" / "chunks.jsonl"
    existing = chunks_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
    existing.append(
        json.dumps(
            {
                "id": "c3",
                "chunk_type": "assessment_item",
                "text": "Unanchored Q",
                "learning_outcome_refs": [],
            }
        )
    )
    chunks_path.write_text("\n".join(existing) + "\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------- #
# CLI tests
# ---------------------------------------------------------------------- #


def test_cli_help_lists_validate_packet():
    runner = CliRunner()
    result = runner.invoke(libv2_group, ["validate-packet", "--help"])
    assert result.exit_code == 0
    assert "--slug" in result.output
    assert "--strict" in result.output
    assert "--format" in result.output


def test_cli_text_format_passes_clean_archive(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_passing_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "demo-course",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 0, result.output
    # Wave 78 added 3 rules → 12 total.
    assert "12 run, 12 passed, 0 failed" in result.output
    assert "critical=0" in result.output


def test_cli_json_format_emits_valid_json_and_writes_quality_file(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    archive = _build_passing_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "demo-course",
            "--format",
            "json",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Wave 78 added 3 rules → 12 total.
    assert payload["rules_run"] == 12
    assert payload["rules_passed"] == 12
    assert payload["critical_count"] == 0

    # Quality file persisted to <archive>/quality/graph_validation_report.json
    quality_file = archive / "quality" / "graph_validation_report.json"
    assert quality_file.exists()
    persisted = json.loads(quality_file.read_text(encoding="utf-8"))
    assert persisted["rules_run"] == 12


def test_cli_returns_zero_with_warnings_when_strict_unset(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_archive_with_warning_only(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "demo-course",
            "--courses-root",
            str(courses_root),
        ],
    )
    # No critical issues → exit 0 even though warnings exist.
    assert result.exit_code == 0, result.output
    assert "warning=" in result.output
    assert "critical=0" in result.output


def test_cli_returns_nonzero_with_warnings_when_strict_set(tmp_path: Path):
    """Wave 78: --strict promotes coverage + typing rules to critical.

    Pre-Wave-78 ``--strict`` treated *every* warning as critical.
    Wave 78 narrows the semantics: only the coverage rules
    (every_objective_has_teaching / every_objective_has_assessment /
    to_has_teaching_and_assessment / domain_concept_has_chunk) and
    the typing rule (edge_endpoint_typing) get promoted. Other
    warning-severity rules (UNANCHORED_ASSESSMENT, etc.) stay
    warnings under --strict.

    This test exercises a coverage gap: an archive whose CO has no
    teaching coverage. Without --strict it's a warning; with
    --strict it's critical and the CLI exits 1.
    """
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    archive = _build_passing_archive(courses_root, "demo-course")
    # Strip co-01 from every chunk's refs → coverage gap.
    chunks_path = archive / "corpus" / "chunks.jsonl"
    items = [
        json.loads(line)
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    for it in items:
        it["learning_outcome_refs"] = [
            r for r in (it.get("learning_outcome_refs") or []) if r != "co-01"
        ]
    _write_jsonl(chunks_path, items)

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "demo-course",
            "--strict",
            "--courses-root",
            str(courses_root),
        ],
    )
    # --strict promotes coverage rules → critical → exit 1.
    assert result.exit_code == 1, result.output


def test_cli_strict_coverage_flag_promotes_only_coverage_rules(tmp_path: Path):
    """Wave 78: --strict-coverage only escalates coverage rules."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    archive = _build_passing_archive(courses_root, "demo-course")
    # Coverage gap: drop co-01 refs.
    chunks_path = archive / "corpus" / "chunks.jsonl"
    items = [
        json.loads(line)
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    for it in items:
        it["learning_outcome_refs"] = [
            r for r in (it.get("learning_outcome_refs") or []) if r != "co-01"
        ]
    _write_jsonl(chunks_path, items)

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "demo-course",
            "--strict-coverage",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 1, result.output


def test_cli_strict_typing_does_not_escalate_coverage_warnings(tmp_path: Path):
    """Wave 78: --strict-typing alone does not promote coverage gaps."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    archive = _build_passing_archive(courses_root, "demo-course")
    # Coverage gap only — no typing violation.
    chunks_path = archive / "corpus" / "chunks.jsonl"
    items = [
        json.loads(line)
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    for it in items:
        it["learning_outcome_refs"] = [
            r for r in (it.get("learning_outcome_refs") or []) if r != "co-01"
        ]
    _write_jsonl(chunks_path, items)

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "demo-course",
            "--strict-typing",
            "--courses-root",
            str(courses_root),
        ],
    )
    # Coverage gaps remain warnings under --strict-typing; exit 0.
    assert result.exit_code == 0, result.output


def test_cli_returns_nonzero_on_critical(tmp_path: Path):
    """Critical rule failure → exit 1 even without --strict."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    archive = _build_passing_archive(courses_root, "demo-course")
    # Inject a duplicate chunk id (DUPLICATE_CHUNK_ID — critical).
    chunks_path = archive / "corpus" / "chunks.jsonl"
    existing = chunks_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
    existing.append(json.dumps({"id": "c1", "chunk_type": "explanation"}))
    chunks_path.write_text("\n".join(existing) + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "demo-course",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 1, result.output


def test_cli_missing_slug_directory_reports_error(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    # Don't create the slug dir.

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "validate-packet",
            "--slug",
            "nonexistent-slug",
            "--courses-root",
            str(courses_root),
        ],
    )
    # Validator reports ARCHIVE_NOT_FOUND as critical → exit 1.
    assert result.exit_code == 1
    assert "ARCHIVE_NOT_FOUND" in result.output
