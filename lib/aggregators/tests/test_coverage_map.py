"""Worker W3.E — :class:`CoverageMapAggregator` tests.

Coverage matrix (per the W3.E plan):

1. Full coverage — every objective has chunks + questions + training_pairs.
2. Orphan objectives — objectives with no chunks/questions/pairs.
3. Orphan chunks — chunks with empty learning_outcome_refs[].
4. Orphan questions — questions with empty objective_id.
5. Partial inputs — missing assessments.json.
6. Partial inputs — missing training_specs/.
7. Mixed objective forms — Courseforge form + LibV2-archive form.
8. answers_grounded_in_chunks — chunk-set intersection semantics.
9. Schema validation — emitted coverage_map.json validates against schema.
10. Decision capture — one ``coverage_map_aggregated`` event per build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lib.aggregators.coverage_map import (
    SCHEMA_VERSION,
    CoverageMapAggregator,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "aggregators" / "coverage_map.schema.json"
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _write_objectives(
    path: Path,
    objectives: List[Dict[str, Any]],
    *,
    shape: str = "courseforge",
) -> Path:
    """Write a synthesized_objectives.json in either the Courseforge or
    LibV2-archive form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if shape == "courseforge":
        # terminal_objectives / chapter_objectives form. Distribute
        # entries evenly: TO-* go into terminal, CO-* go into chapter.
        terminal = [o for o in objectives if o.get("id", "").startswith("TO")]
        chapter = [o for o in objectives if o.get("id", "").startswith("CO")]
        payload: Dict[str, Any] = {
            "terminal_objectives": terminal,
            "chapter_objectives": chapter,
        }
    elif shape == "libv2":
        # terminal_outcomes / component_objectives form.
        terminal = [o for o in objectives if o.get("id", "").startswith("TO")]
        component = [o for o in objectives if o.get("id", "").startswith("CO")]
        payload = {
            "terminal_outcomes": terminal,
            "component_objectives": component,
        }
    elif shape == "flat":
        payload = {"learning_outcomes": objectives}
    else:
        raise ValueError(f"unknown shape: {shape}")
    path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    return path


def _write_chunks(
    path: Path, chunks: List[Dict[str, Any]],
) -> Path:
    """Write chunks to a chunks.jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_assessments(
    path: Path, questions: List[Dict[str, Any]],
) -> Path:
    """Write a single-document assessments.json with the questions list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "course_code": "TEST",
        "questions": questions,
        "objectives_targeted": sorted({
            q.get("objective_id") for q in questions if q.get("objective_id")
        }),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _write_pairs(
    path: Path, pairs: List[Dict[str, Any]],
) -> Path:
    """Write pairs to a {instruction,preference}_pairs.jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(p, ensure_ascii=False) for p in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_layout(
    tmp_path: Path,
    *,
    objectives: Optional[List[Dict[str, Any]]] = None,
    objectives_shape: str = "courseforge",
    chunks: Optional[List[Dict[str, Any]]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    instruction_pairs: Optional[List[Dict[str, Any]]] = None,
    preference_pairs: Optional[List[Dict[str, Any]]] = None,
    write_assessments: bool = True,
    write_training_specs: bool = True,
) -> Dict[str, Path]:
    """Build a synthetic LibV2 course + Trainforge dir layout under tmp_path."""
    course_dir = tmp_path / "libv2_course"
    trainforge_dir = tmp_path / "trainforge"
    course_dir.mkdir(parents=True, exist_ok=True)
    trainforge_dir.mkdir(parents=True, exist_ok=True)

    objectives_path: Optional[Path] = None
    if objectives is not None:
        objectives_path = _write_objectives(
            course_dir / "objectives.json",
            objectives,
            shape=objectives_shape,
        )
    if chunks is not None:
        _write_chunks(course_dir / "imscc_chunks" / "chunks.jsonl", chunks)
    if questions is not None and write_assessments:
        _write_assessments(trainforge_dir / "assessments.json", questions)
    if instruction_pairs is not None and write_training_specs:
        _write_pairs(
            trainforge_dir / "training_specs" / "instruction_pairs.jsonl",
            instruction_pairs,
        )
    if preference_pairs is not None and write_training_specs:
        _write_pairs(
            trainforge_dir / "training_specs" / "preference_pairs.jsonl",
            preference_pairs,
        )
    return {
        "course_dir": course_dir,
        "trainforge_dir": trainforge_dir,
        "objectives_path": objectives_path,
    }


def _build_aggregator(
    layout: Dict[str, Path],
    *,
    course_code: str = "TEST",
    run_id: str = "WF-TEST",
    decision_capture: Optional[Any] = None,
) -> CoverageMapAggregator:
    return CoverageMapAggregator(
        phase_outputs={},
        course_code=course_code,
        run_id=run_id,
        libv2_course_path=layout["course_dir"],
        trainforge_dir=layout["trainforge_dir"],
        decision_capture=decision_capture,
    )


# --------------------------------------------------------------------------
# Test 1 — full coverage scenario
# --------------------------------------------------------------------------


class TestFullCoverage:
    def test_every_objective_has_chunks_questions_and_pairs(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "Define X."},
            {"id": "TO-02", "statement": "Apply X."},
            {"id": "CO-01", "statement": "Compare X and Y."},
        ]
        chunks = [
            {
                "id": "chunk_001",
                "learning_outcome_refs": ["TO-01"],
            },
            {
                "id": "chunk_002",
                "learning_outcome_refs": ["TO-02"],
            },
            {
                "id": "chunk_003",
                "learning_outcome_refs": ["CO-01"],
            },
        ]
        questions = [
            {
                "question_id": "q-001",
                "objective_id": "TO-01",
                "source_chunks": ["chunk_001"],
            },
            {
                "question_id": "q-002",
                "objective_id": "TO-02",
                "source_chunks": ["chunk_002"],
            },
            {
                "question_id": "q-003",
                "objective_id": "CO-01",
                "source_chunks": ["chunk_003"],
            },
        ]
        instr = [
            {"id": "p-001", "lo_refs": ["TO-01"]},
            {"id": "p-002", "lo_refs": ["TO-02"]},
            {"id": "p-003", "lo_refs": ["CO-01"]},
        ]
        pref = [
            {"id": "pref-001", "lo_refs": ["TO-01"]},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=questions,
            instruction_pairs=instr,
            preference_pairs=pref,
        )
        agg = _build_aggregator(layout)
        report = agg.build()

        summary = report["summary"]
        assert summary["total_objectives"] == 3
        assert summary["objectives_with_chunks"] == 3
        assert summary["objectives_with_questions"] == 3
        assert summary["objectives_with_training_pairs"] == 3
        assert summary["orphan_objectives"] == []
        assert summary["orphan_chunks"] == []
        assert summary["orphan_questions"] == []

        # Check per-objective links.
        rows = {row["objective_id"]: row for row in report["objectives"]}
        assert rows["TO-01"]["chunks"] == ["chunk_001"]
        assert rows["TO-01"]["questions"] == ["q-001"]
        assert rows["TO-01"]["answers_grounded_in_chunks"] == ["q-001"]
        # Both pairs cite TO-01.
        assert sorted(rows["TO-01"]["training_pairs"]) == ["p-001", "pref-001"]


# --------------------------------------------------------------------------
# Test 2 — orphan objectives
# --------------------------------------------------------------------------


class TestOrphanObjectives:
    def test_two_orphan_objectives_listed_in_summary(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "covered"},
            {"id": "TO-02", "statement": "orphan A"},
            {"id": "TO-03", "statement": "orphan B"},
        ]
        chunks = [
            {"id": "c-1", "learning_outcome_refs": ["TO-01"]},
        ]
        questions = [
            {
                "question_id": "q-1",
                "objective_id": "TO-01",
                "source_chunks": [],
            },
        ]
        instr = [
            {"id": "p-1", "lo_refs": ["TO-01"]},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=questions,
            instruction_pairs=instr,
            preference_pairs=[],
        )
        agg = _build_aggregator(layout)
        report = agg.build()

        summary = report["summary"]
        assert summary["total_objectives"] == 3
        assert summary["objectives_with_chunks"] == 1
        assert summary["objectives_with_questions"] == 1
        assert summary["objectives_with_training_pairs"] == 1
        assert summary["orphan_objectives"] == ["TO-02", "TO-03"]


# --------------------------------------------------------------------------
# Test 3 — orphan chunks
# --------------------------------------------------------------------------


class TestOrphanChunks:
    def test_chunks_with_empty_lo_refs_are_orphan(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "an objective"},
        ]
        chunks = [
            {"id": "c-bound", "learning_outcome_refs": ["TO-01"]},
            {"id": "c-empty", "learning_outcome_refs": []},
            # A chunk that cites a non-extant objective is also orphan.
            {"id": "c-bogus", "learning_outcome_refs": ["TO-99"]},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=[],
            instruction_pairs=[],
            preference_pairs=[],
        )
        agg = _build_aggregator(layout)
        report = agg.build()
        orphan_chunks = report["summary"]["orphan_chunks"]
        assert "c-empty" in orphan_chunks
        assert "c-bogus" in orphan_chunks
        assert "c-bound" not in orphan_chunks


# --------------------------------------------------------------------------
# Test 4 — orphan questions
# --------------------------------------------------------------------------


class TestOrphanQuestions:
    def test_questions_with_empty_objective_id_are_orphan(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "..."},
        ]
        chunks = [
            {"id": "c-1", "learning_outcome_refs": ["TO-01"]},
        ]
        questions = [
            {"question_id": "q-bound", "objective_id": "TO-01"},
            {"question_id": "q-empty", "objective_id": ""},
            {"question_id": "q-bogus", "objective_id": "TO-99"},
            # A question with no objective_id field at all -> orphan.
            {"question_id": "q-missing"},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=questions,
            instruction_pairs=[],
            preference_pairs=[],
        )
        agg = _build_aggregator(layout)
        report = agg.build()
        orphan_questions = report["summary"]["orphan_questions"]
        assert "q-empty" in orphan_questions
        assert "q-bogus" in orphan_questions
        assert "q-missing" in orphan_questions
        assert "q-bound" not in orphan_questions


# --------------------------------------------------------------------------
# Test 5 — partial inputs (missing assessments.json)
# --------------------------------------------------------------------------


class TestMissingAssessments:
    def test_missing_assessments_yields_empty_question_arrays(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "..."},
            {"id": "TO-02", "statement": "..."},
        ]
        chunks = [
            {"id": "c-1", "learning_outcome_refs": ["TO-01"]},
            {"id": "c-2", "learning_outcome_refs": ["TO-02"]},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=None,  # do not write assessments.json
            instruction_pairs=[{"id": "p-1", "lo_refs": ["TO-01"]}],
            preference_pairs=[],
        )
        agg = _build_aggregator(layout)
        # Should not crash.
        report = agg.build()
        assert report["schema_version"] == SCHEMA_VERSION
        for row in report["objectives"]:
            assert row["questions"] == []
            assert row["answers_grounded_in_chunks"] == []
        assert report["summary"]["objectives_with_questions"] == 0


# --------------------------------------------------------------------------
# Test 6 — partial inputs (missing training_specs/)
# --------------------------------------------------------------------------


class TestMissingTrainingSpecs:
    def test_missing_pairs_yields_empty_training_pair_arrays(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "..."},
        ]
        chunks = [
            {"id": "c-1", "learning_outcome_refs": ["TO-01"]},
        ]
        questions = [
            {
                "question_id": "q-1",
                "objective_id": "TO-01",
                "source_chunks": ["c-1"],
            },
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=questions,
            instruction_pairs=None,  # do not write training_specs files
            preference_pairs=None,
        )
        agg = _build_aggregator(layout)
        report = agg.build()
        for row in report["objectives"]:
            assert row["training_pairs"] == []
        assert report["summary"]["objectives_with_training_pairs"] == 0


# --------------------------------------------------------------------------
# Test 7 — mixed objective forms (Courseforge + LibV2-archive)
# --------------------------------------------------------------------------


class TestObjectiveShapes:
    def test_courseforge_form_loads(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "term"},
            {"id": "CO-01", "statement": "comp"},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            objectives_shape="courseforge",
            chunks=[],
            questions=[],
        )
        agg = _build_aggregator(layout)
        report = agg.build()
        ids = {row["objective_id"] for row in report["objectives"]}
        assert ids == {"TO-01", "CO-01"}

    def test_libv2_form_loads(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "term"},
            {"id": "CO-01", "statement": "comp"},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            objectives_shape="libv2",
            chunks=[],
            questions=[],
        )
        agg = _build_aggregator(layout)
        report = agg.build()
        ids = {row["objective_id"] for row in report["objectives"]}
        assert ids == {"TO-01", "CO-01"}

    def test_flat_form_loads(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "term"},
            {"id": "CO-01", "statement": "comp"},
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            objectives_shape="flat",
            chunks=[],
            questions=[],
        )
        agg = _build_aggregator(layout)
        report = agg.build()
        ids = {row["objective_id"] for row in report["objectives"]}
        assert ids == {"TO-01", "CO-01"}


# --------------------------------------------------------------------------
# Test 8 — answers_grounded_in_chunks intersection semantics
# --------------------------------------------------------------------------


class TestAnswersGroundedInChunks:
    def test_question_not_grounded_when_source_chunks_dont_intersect(
        self, tmp_path,
    ):
        objectives = [
            {"id": "TO-01", "statement": "..."},
        ]
        chunks = [
            # chunk_001 is the only one citing TO-01.
            {"id": "chunk_001", "learning_outcome_refs": ["TO-01"]},
            # chunk_002 doesn't cite any objective (orphan chunk),
            # but the question below still cites it as a source.
            {"id": "chunk_002", "learning_outcome_refs": []},
        ]
        questions = [
            # Question cites TO-01 structurally but cites only chunk_002
            # — chunk_002 doesn't roll up to TO-01, so the question is
            # NOT in answers_grounded_in_chunks.
            {
                "question_id": "q-structural",
                "objective_id": "TO-01",
                "source_chunks": ["chunk_002"],
            },
            # Question cites TO-01 AND chunk_001 (which itself cites TO-01)
            # — IS in answers_grounded_in_chunks.
            {
                "question_id": "q-grounded",
                "objective_id": "TO-01",
                "source_chunks": ["chunk_001"],
            },
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=questions,
            instruction_pairs=[],
        )
        agg = _build_aggregator(layout)
        report = agg.build()
        rows = {row["objective_id"]: row for row in report["objectives"]}
        # Both questions appear in questions[] (structural citation).
        assert sorted(rows["TO-01"]["questions"]) == [
            "q-grounded", "q-structural",
        ]
        # Only the grounded one in answers_grounded_in_chunks.
        assert rows["TO-01"]["answers_grounded_in_chunks"] == ["q-grounded"]


# --------------------------------------------------------------------------
# Test 9 — schema validation
# --------------------------------------------------------------------------


class TestSchemaValidation:
    def test_emitted_coverage_map_validates_against_schema(self, tmp_path):
        jsonschema = pytest.importorskip("jsonschema")
        from jsonschema import Draft202012Validator

        objectives = [
            {"id": "TO-01", "statement": "..."},
            {"id": "TO-02", "statement": "..."},
        ]
        chunks = [
            {"id": "c-1", "learning_outcome_refs": ["TO-01"]},
            {"id": "c-2", "learning_outcome_refs": []},  # orphan
        ]
        questions = [
            {
                "question_id": "q-1",
                "objective_id": "TO-01",
                "source_chunks": ["c-1"],
            },
            {
                "question_id": "q-orphan",
                "objective_id": "",
                "source_chunks": [],
            },
        ]
        instr = [{"id": "p-1", "lo_refs": ["TO-01"]}]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=questions,
            instruction_pairs=instr,
            preference_pairs=[],
        )
        agg = _build_aggregator(layout)
        # Use write() so we exercise the actual on-disk emit path.
        out_path = layout["course_dir"] / "coverage_map.json"
        agg.write(out_path)
        report = json.loads(out_path.read_text(encoding="utf-8"))

        with SCHEMA_PATH.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(report), key=lambda e: list(e.path)
        )
        assert errors == [], (
            "coverage_map.json failed schema validation: "
            + "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
        )


# --------------------------------------------------------------------------
# Test 10 — decision capture
# --------------------------------------------------------------------------


class _FakeCapture:
    """Minimal stub mirroring the DecisionCapture.log_decision signature."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class TestDecisionCapture:
    def test_one_coverage_map_aggregated_event_per_build(self, tmp_path):
        objectives = [
            {"id": "TO-01", "statement": "covered"},
            {"id": "TO-02", "statement": "orphan"},
        ]
        chunks = [
            {"id": "c-1", "learning_outcome_refs": ["TO-01"]},
        ]
        questions = [
            {
                "question_id": "q-1",
                "objective_id": "TO-01",
                "source_chunks": ["c-1"],
            },
        ]
        layout = _build_layout(
            tmp_path,
            objectives=objectives,
            chunks=chunks,
            questions=questions,
            instruction_pairs=[{"id": "p-1", "lo_refs": ["TO-01"]}],
            preference_pairs=[],
        )
        capture = _FakeCapture()
        agg = _build_aggregator(layout, decision_capture=capture)
        report = agg.build()

        # Exactly one event.
        assert len(capture.events) == 1
        evt = capture.events[0]
        assert evt["decision_type"] == "coverage_map_aggregated"
        # Rationale clears the 20-char floor.
        assert isinstance(evt["rationale"], str)
        assert len(evt["rationale"]) >= 20
        # Rationale interpolates the six summary counts.
        rationale = evt["rationale"]
        for token in (
            "total_objectives=",
            "objectives_with_chunks=",
            "objectives_with_questions=",
            "objectives_with_training_pairs=",
            "orphan_objectives=",
            "orphan_chunks=",
            "orphan_questions=",
        ):
            assert token in rationale, (
                f"rationale missing {token!r}: {rationale}"
            )
        # Metrics carry the same counts.
        metrics = evt["metrics"]
        assert metrics["total_objectives"] == 2
        assert metrics["objectives_with_chunks"] == 1
        assert metrics["objectives_with_questions"] == 1
        assert metrics["objectives_with_training_pairs"] == 1
        assert metrics["orphan_objective_count"] == 1


# --------------------------------------------------------------------------
# Bonus — empty-fixture invariants
# --------------------------------------------------------------------------


class TestEmptyInputs:
    def test_no_inputs_yields_minimal_report(self, tmp_path):
        agg = CoverageMapAggregator(
            phase_outputs={},
            course_code="EMPTY",
            run_id="WF-EMPTY",
            libv2_course_path=tmp_path / "nonexistent",
            trainforge_dir=tmp_path / "nonexistent2",
        )
        report = agg.build()
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["objectives"] == []
        summary = report["summary"]
        assert summary["total_objectives"] == 0
        assert summary["orphan_objectives"] == []
        assert summary["orphan_chunks"] == []
        assert summary["orphan_questions"] == []
