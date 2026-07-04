"""W10 Phase 2 — unit tests for the assessment-synthesis emit helpers.

These tests exercise the PURE helpers behind the ``run_assessment_synthesis``
phase handler WITHOUT an LLM call, a pipeline run, or a course slug:

* the ``AssessmentData`` / discussion / assignment → XML + ``manifest.json``
  emit round-trips the packager's discovery (the manifest matches the pinned
  ``06_assessments/manifest.json`` contract byte-for-shape);
* the two new decision-event enum members validate against the canonical
  schema;
* anti-fabrication — an assessment item whose ``objective_id`` is outside the
  real synthesized-objectives universe is DROPPED, never emitted.

Run:

    source .venv/bin/activate && \
      ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
      python -m pytest MCP/tests/test_assessment_synthesis.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from MCP.tools.pipeline_tools import (
    _assessment_manifest_entry,
    _build_assessment_artifacts_from_data,
    _filter_questions_to_objective_universe,
    _safe_assessment_basename,
    _write_assessment_artifacts,
)
from Trainforge.generators.assessment_generator import (
    AssessmentData,
    QuestionData,
)


# ---------------------------------------------------------------------------
# Fixtures — built in-test, NO course slug, NO LLM.
# ---------------------------------------------------------------------------


def _mc_question(qid: str, objective_id: str) -> QuestionData:
    return QuestionData(
        question_id=qid,
        question_type="multiple_choice",
        stem=f"Which statement best describes the concept in {objective_id}?",
        bloom_level="understand",
        objective_id=objective_id,
        choices=[
            {"id": "A", "text": "The correct definition", "is_correct": True},
            {"id": "B", "text": "A plausible distractor", "is_correct": False},
            {"id": "C", "text": "Another distractor", "is_correct": False},
        ],
        correct_answer="A",
        source_chunks=["chunk_001"],
    )


def _fixture_assessment(week: int = 3) -> AssessmentData:
    a = AssessmentData(
        assessment_id="ASM-DEMO-001",
        title="Week 3 Quiz",
        course_code="DEMO",
        questions=[
            _mc_question("q-001", "TO-01"),
            _mc_question("q-002", "CO-01"),
        ],
        objectives_targeted=["TO-01", "CO-01"],
        bloom_levels=["understand"],
    )
    # The emit helper reads an optional ``week`` attribute for clean nav.
    setattr(a, "week", week)
    return a


# ---------------------------------------------------------------------------
# 1. Basename safety (anti path-traversal).
# ---------------------------------------------------------------------------


def test_safe_basename_strips_path_traversal():
    assert _safe_assessment_basename("../../etc/passwd", fallback="x") == "etc_passwd"
    assert _safe_assessment_basename("Week 3 Quiz!", fallback="x") == "week_3_quiz"
    assert _safe_assessment_basename("", fallback="quiz") == "quiz"
    assert _safe_assessment_basename("///", fallback="quiz") == "quiz"
    # No path separator ever survives.
    out = _safe_assessment_basename("a/b\\c", fallback="x")
    assert "/" not in out and "\\" not in out


# ---------------------------------------------------------------------------
# 2. Manifest entry shape matches the pinned contract.
# ---------------------------------------------------------------------------


def test_manifest_entry_pinned_shape():
    entry = _assessment_manifest_entry(
        basename="week_03_quiz",
        kind="qti",
        title="Week 3 Quiz",
        week=3,
        identifier="RES_week_03_quiz",
    )
    assert entry == {
        "file": "week_03_quiz.xml",
        "type": "qti",
        "title": "Week 3 Quiz",
        "identifier": "RES_week_03_quiz",
        "week": 3,
    }


def test_manifest_entry_week_omitted_when_none():
    entry = _assessment_manifest_entry(
        basename="x", kind="discussion", title="X", week=None, identifier="RES_x"
    )
    assert "week" not in entry


# ---------------------------------------------------------------------------
# 3. Full emit round-trips the packager's manifest contract.
# ---------------------------------------------------------------------------


def test_emit_writes_xml_and_manifest(tmp_path: Path):
    out_dir = tmp_path / "06_assessments"
    artifacts = _build_assessment_artifacts_from_data(
        assessments=[_fixture_assessment(week=3)],
        discussions=[
            {
                "discussion_id": "DISC-TO-01",
                "objective_id": "TO-01",
                "title": "Week 1 Discussion",
                "text": "Discuss how the concept applies in a real scenario.",
                "week": 1,
            }
        ],
        assignments=[
            {
                "assignment_id": "ASGN-CO-01",
                "objective_id": "CO-01",
                "title": "Week 1 Assignment",
                "text": "Complete a short task demonstrating the objective.",
                "points": 10,
                "week": 1,
            }
        ],
    )
    manifest = _write_assessment_artifacts(out_dir, artifacts)

    # ---- manifest matches the pinned contract byte-for-shape -------------
    assert manifest["schema_version"] == "v1"
    assert isinstance(manifest["assessments"], list)
    assert len(manifest["assessments"]) == 3
    types = {e["type"] for e in manifest["assessments"]}
    assert types == {"qti", "discussion", "assignment"}

    for entry in manifest["assessments"]:
        # ``file`` is a basename under 06_assessments/, no path traversal.
        assert entry["file"].endswith(".xml")
        assert "/" not in entry["file"] and ".." not in entry["file"]
        # The referenced file actually exists on disk.
        assert (out_dir / entry["file"]).is_file()
        # type/title/identifier present.
        assert entry["type"] in {"qti", "discussion", "assignment"}
        assert entry["title"]
        assert entry["identifier"]

    # ---- on-disk manifest.json parses + equals the returned dict ---------
    on_disk = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_emit_roundtrips_packager_discovery(tmp_path: Path):
    """The packager's own discovery classifies every emitted artifact."""
    from Courseforge.scripts.package_multifile_imscc import _discover_assessments

    # The packager joins ``content_dir / "06_assessments"``, so pass the
    # PARENT of our output dir as the content_dir.
    content_dir = tmp_path
    out_dir = content_dir / "06_assessments"
    artifacts = _build_assessment_artifacts_from_data(
        assessments=[_fixture_assessment(week=2)],
        discussions=[
            {"objective_id": "TO-01", "title": "Disc", "text": "Discuss.", "week": 2}
        ],
        assignments=[
            {"objective_id": "CO-01", "title": "Task", "text": "Do it.", "week": 2}
        ],
    )
    _write_assessment_artifacts(out_dir, artifacts)

    discovered, authored = _discover_assessments(content_dir)
    assert authored == 3
    assert len(discovered) == 3
    kinds = sorted(a.kind for a in discovered)
    assert kinds == ["assignment", "discussion", "qti"]


def test_emit_empty_inputs_clean_empty_manifest(tmp_path: Path):
    out_dir = tmp_path / "06_assessments"
    manifest = _write_assessment_artifacts(out_dir, [])
    assert manifest == {"schema_version": "v1", "assessments": []}
    assert (out_dir / "manifest.json").is_file()


def test_emit_decollides_duplicate_basenames(tmp_path: Path):
    out_dir = tmp_path / "06_assessments"
    # Two discussions both resolving to the same week → same basename stem.
    artifacts = _build_assessment_artifacts_from_data(
        assessments=[],
        discussions=[
            {"objective_id": "TO-01", "title": "A", "text": "x", "week": 1},
            {"objective_id": "TO-02", "title": "B", "text": "y", "week": 1},
        ],
        assignments=[],
    )
    manifest = _write_assessment_artifacts(out_dir, artifacts)
    files = [e["file"] for e in manifest["assessments"]]
    assert len(files) == len(set(files)), "duplicate basenames must de-collide"
    for f in files:
        assert (out_dir / f).is_file()


# ---------------------------------------------------------------------------
# 4. Anti-fabrication — phantom objective items are dropped.
# ---------------------------------------------------------------------------


def test_phantom_objective_questions_dropped():
    valid = {"TO-01", "CO-01"}
    questions = [
        _mc_question("q-1", "TO-01"),   # kept
        _mc_question("q-2", "CO-99"),   # phantom → dropped
        _mc_question("q-3", "CO-01"),   # kept
    ]
    kept, dropped = _filter_questions_to_objective_universe(questions, valid)
    assert {q.objective_id for q in kept} == {"TO-01", "CO-01"}
    assert [q.objective_id for q in dropped] == ["CO-99"]


def test_phantom_filter_accepts_dict_questions():
    valid = {"TO-01"}
    questions = [
        {"objective_id": "TO-01", "stem": "real"},
        {"objective_id": "X", "stem": "phantom"},
    ]
    kept, dropped = _filter_questions_to_objective_universe(questions, valid)
    assert len(kept) == 1 and len(dropped) == 1
    assert kept[0]["objective_id"] == "TO-01"


def test_emitted_quiz_only_carries_grounded_objectives(tmp_path: Path):
    """End-to-end: an assessment with a phantom item emits only the kept item."""
    valid = {"TO-01", "CO-01"}
    a = _fixture_assessment(week=4)
    a.questions.append(_mc_question("q-phantom", "CO-77"))
    kept, dropped = _filter_questions_to_objective_universe(a.questions, valid)
    a.questions = kept
    assert dropped, "the phantom item must have been identified"

    out_dir = tmp_path / "06_assessments"
    artifacts = _build_assessment_artifacts_from_data([a], [], [])
    _write_assessment_artifacts(out_dir, artifacts)

    # Parse the emitted QTI and confirm no item references a phantom objective.
    qti_files = list(out_dir.glob("*.xml"))
    assert len(qti_files) == 1
    text = qti_files[0].read_text(encoding="utf-8")
    assert "CO-77" not in text
    assert "TO-01" in text and "CO-01" in text


# ---------------------------------------------------------------------------
# 5. New decision-event enum members validate against the schema.
# ---------------------------------------------------------------------------


def _load_decision_schema() -> dict:
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "schemas" / "events" / "decision_event.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_new_enum_members_present_in_schema():
    schema = _load_decision_schema()
    enum = schema["properties"]["decision_type"]["enum"]
    assert "assignment_prompt_synthesis" in enum
    assert "discussion_prompt_synthesis" in enum
    # Alphabetised-insert sanity for the two new members: each must sit
    # between its lexical neighbours (the whole enum predates this change
    # and carries a few historical out-of-order entries, so we check the
    # LOCAL ordering of our inserts, not the global list).
    i_asg = enum.index("assignment_prompt_synthesis")
    assert enum[i_asg - 1] == "assessment_template_skip"
    assert enum[i_asg + 1] == "base_model_eval_call"
    i_disc = enum.index("discussion_prompt_synthesis")
    # "difficulty_calibration" (TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD enum member)
    # sorts alphabetically between deterministic_pair_audit_stamp and
    # discussion_prompt_synthesis, so it is the immediate predecessor here.
    assert enum[i_disc - 1] == "difficulty_calibration"
    assert enum[i_disc + 1] == "distractor_generation"


def test_new_enum_members_validate_a_capture_event():
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_decision_schema()
    base = {
        "run_id": "RUN_20260620_000000_test",
        "timestamp": "2026-06-20T00:00:00Z",
        "operation": "assessment_synthesis",
        "decision": "assignment:CO-01:week_01",
        "rationale": (
            "objective_id=CO-01; source_chunk_ids=['chunk_001']; "
            "provider=local; model=qwen2.5:14b; n_grounded_chunks=1"
        ),
    }
    for dtype in ("assignment_prompt_synthesis", "discussion_prompt_synthesis"):
        event = dict(base, decision_type=dtype)
        # Must not raise.
        jsonschema.validate(instance=event, schema=schema)
