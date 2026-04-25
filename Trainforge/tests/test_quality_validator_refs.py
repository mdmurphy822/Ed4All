"""Wave 76 — quality validator broken-refs regression.

Pre-Wave-76 ``CourseProcessor._build_valid_outcome_ids`` only read
``terminal_objectives`` / ``chapter_objectives`` from the loaded
objectives dict. Wave 75 Worker A's emit uses ``terminal_outcomes`` /
``component_objectives``, so the resulting set was empty and EVERY chunk
ref was flagged as broken (rdf-shacl-550 reported 312 broken_refs where
311 were valid IDs and 1 was a genuinely malformed comma-joined string).

This test locks in:
  - Both objective-file schemas resolve.
  - Comparison is case-insensitive (chunk emits ``CO-01``, objectives
    emit ``co-01`` — must resolve).
  - Empty / null / whitespace-only refs are skipped (not flagged).
  - Comma-joined refs survive as a SINGLE broken entry (preserved
    verbatim so a reviewer can see the malformed shape; the split logic
    is Wave 75 Worker A's responsibility, not this validator's).
"""

from __future__ import annotations

from typing import Set

import pytest

from Trainforge.process_course import CourseProcessor


# ---------------------------------------------------------------------------
# _collect_broken_refs (static method)
# ---------------------------------------------------------------------------


def test_resolves_co_and_to_refs_against_lowercase_set() -> None:
    valid: Set[str] = {"co-01", "to-02", "co-29"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": ["co-01", "to-02"]},
        {"id": "chunk_002", "learning_outcome_refs": ["co-29"]},
        {"id": "chunk_003", "learning_outcome_refs": ["co-99"]},  # unknown
    ]
    broken = CourseProcessor._collect_broken_refs(chunks, valid)
    assert len(broken) == 1
    assert broken[0]["chunk_id"] == "chunk_003"
    assert broken[0]["ref"] == "co-99"


def test_case_insensitive_ref_resolution() -> None:
    """Chunk emits uppercase, objectives emit lowercase — must resolve."""
    valid: Set[str] = {"co-01", "to-02"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": ["CO-01", "TO-02"]},
    ]
    broken = CourseProcessor._collect_broken_refs(chunks, valid)
    assert broken == []


def test_empty_string_ref_not_flagged_as_broken() -> None:
    valid: Set[str] = {"co-01"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": ["", "co-01", "  "]},
    ]
    broken = CourseProcessor._collect_broken_refs(chunks, valid)
    assert broken == []


def test_none_ref_not_flagged_as_broken() -> None:
    valid: Set[str] = {"co-01"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": [None, "co-01", None]},
    ]
    broken = CourseProcessor._collect_broken_refs(chunks, valid)
    assert broken == []


def test_comma_joined_ref_flagged_as_single_broken_entry() -> None:
    """The legitimate bad case from Wave 75: a chunk emitted
    ``"co-01,co-02,co-03"`` as ONE string element. The Wave 75 Worker A
    fix splits these at chunk-emit time; the validator's job here is
    just to report the malformed string as broken if it slips through.
    """
    valid: Set[str] = {"co-01", "co-02", "co-03"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": ["co-01,co-02,co-03"]},
    ]
    broken = CourseProcessor._collect_broken_refs(chunks, valid)
    assert len(broken) == 1
    assert broken[0]["ref"] == "co-01,co-02,co-03"


def test_total_count_matches_external_review_scenario() -> None:
    """Mirror the scenario from the rdf-shacl-550 external review:
    chunk has 3 refs, 2 resolve, 1 unknown → broken=1, not 3.
    """
    valid: Set[str] = {"co-01", "to-02"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": ["co-01", "to-02", "co-99"]},
    ]
    broken = CourseProcessor._collect_broken_refs(chunks, valid)
    assert len(broken) == 1


# ---------------------------------------------------------------------------
# _resolving_lo_coverage — case-insensitive
# ---------------------------------------------------------------------------


def test_resolving_coverage_case_insensitive() -> None:
    valid: Set[str] = {"co-01", "to-02"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": ["CO-01"]},
        {"id": "chunk_002", "learning_outcome_refs": ["TO-02"]},
        {"id": "chunk_003", "learning_outcome_refs": []},
    ]
    coverage = CourseProcessor._resolving_lo_coverage(chunks, valid)
    # 2 of 3 chunks have at least one resolving ref
    assert abs(coverage - (2 / 3)) < 1e-9


def test_resolving_coverage_skips_empty_and_none() -> None:
    valid: Set[str] = {"co-01"}
    chunks = [
        {"id": "chunk_001", "learning_outcome_refs": [None, "", "co-01"]},
        {"id": "chunk_002", "learning_outcome_refs": [None, ""]},  # no resolvable ref
    ]
    coverage = CourseProcessor._resolving_lo_coverage(chunks, valid)
    assert abs(coverage - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _build_valid_outcome_ids via load_objectives — both schemas
# ---------------------------------------------------------------------------


def test_load_objectives_wave75_schema(tmp_path) -> None:
    """Wave 75 emit: terminal_outcomes + component_objectives (flat shape)."""
    import json

    from Trainforge.process_course import load_objectives

    objectives_path = tmp_path / "objectives.json"
    objectives_path.write_text(json.dumps({
        "schema_version": "wave75",
        "course_code": "TEST_101",
        "terminal_outcomes": [
            {"id": "to-01", "statement": "T1", "bloom_level": "analyze"},
            {"id": "to-02", "statement": "T2", "bloom_level": "apply"},
        ],
        "component_objectives": [
            {"id": "co-01", "statement": "C1", "parent_terminal": "to-01",
             "bloom_level": "remember", "week": 1},
            {"id": "co-02", "statement": "C2", "parent_terminal": "to-01",
             "bloom_level": "understand", "week": 2},
        ],
    }))
    loaded = load_objectives(objectives_path)
    assert len(loaded["terminal_objectives"]) == 2
    assert len(loaded["chapter_objectives"]) == 2
    # Bloom map should pick up the flat-shape week field
    assert "remember" in loaded["week_bloom_map"].get(1, [])
    assert "understand" in loaded["week_bloom_map"].get(2, [])


def test_load_objectives_legacy_schema(tmp_path) -> None:
    """Pre-Wave-75: terminal_objectives + chapter_objectives (nested)."""
    import json

    from Trainforge.process_course import load_objectives

    objectives_path = tmp_path / "objectives.json"
    objectives_path.write_text(json.dumps({
        "terminal_objectives": [
            {"id": "to-01", "statement": "T1"},
        ],
        "chapter_objectives": [
            {"chapter": "Week 1: Intro", "objectives": [
                {"id": "co-01", "statement": "C1", "bloom_level": "remember"},
            ]},
        ],
    }))
    loaded = load_objectives(objectives_path)
    assert len(loaded["terminal_objectives"]) == 1
    assert len(loaded["chapter_objectives"]) == 1


def test_build_valid_outcome_ids_wave75_schema(tmp_path) -> None:
    """End-to-end: a CourseProcessor pointed at a Wave-75-shaped
    objectives file resolves both terminals AND components.
    """
    import json

    from Trainforge.process_course import CourseProcessor, load_objectives

    objectives_path = tmp_path / "objectives.json"
    objectives_path.write_text(json.dumps({
        "schema_version": "wave75",
        "terminal_outcomes": [{"id": "to-01", "statement": "T"}],
        "component_objectives": [
            {"id": "co-01", "statement": "C1", "parent_terminal": "to-01"},
            {"id": "co-02", "statement": "C2", "parent_terminal": "to-01"},
        ],
    }))

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Build a CourseProcessor without invoking its full IMSCC init.
    cp = CourseProcessor.__new__(CourseProcessor)
    cp.objectives = load_objectives(objectives_path)
    cp.output_dir = output_dir

    ids = cp._build_valid_outcome_ids()
    assert ids == {"to-01", "co-01", "co-02"}


def test_build_valid_outcome_ids_falls_back_to_course_json(tmp_path) -> None:
    """No objectives file → fall back to course.json::learning_outcomes."""
    import json

    from Trainforge.process_course import CourseProcessor

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "course.json").write_text(json.dumps({
        "course_code": "TEST_101",
        "title": "Test",
        "learning_outcomes": [
            {"id": "TO-01", "statement": "T", "hierarchy_level": "terminal"},
            {"id": "co-01", "statement": "C", "hierarchy_level": "chapter"},
        ],
    }))

    cp = CourseProcessor.__new__(CourseProcessor)
    cp.objectives = {}  # no objectives loaded
    cp.output_dir = output_dir

    ids = cp._build_valid_outcome_ids()
    # Lowercased on emit
    assert ids == {"to-01", "co-01"}
