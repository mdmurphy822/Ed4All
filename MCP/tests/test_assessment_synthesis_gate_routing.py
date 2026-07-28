"""Regression coverage for assessment_synthesis gate input routing.

The phase handler emits ``manifest_path``/``assessments_dir`` while its source
chunks remain in ``phase_outputs.chunking.semantik_chunks_path``.  These tests
exercise the same phase-output envelope as a live workflow and then invoke the
real validators, preventing a structured router skip or a false
NO_DISCUSSION_ASSIGNMENT_ITEMS no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

from MCP.core.workflow_runner import _get_phase_output_keys
from MCP.hardening.gate_input_routing import (
    _build_assessment_quality,
    default_router,
)
from lib.validators.assessment_objective_alignment import (
    AssessmentObjectiveAlignmentValidator,
)
from lib.validators.discussion_assignment_grounding import (
    DiscussionAssignmentGroundingValidator,
)


ALIGNMENT_PATH = (
    "lib.validators.assessment_objective_alignment."
    "AssessmentObjectiveAlignmentValidator"
)
GROUNDING_PATH = (
    "lib.validators.discussion_assignment_grounding."
    "DiscussionAssignmentGroundingValidator"
)


def _representative_outputs(tmp_path: Path) -> tuple[dict, Path, Path]:
    assessment_dir = tmp_path / "export" / "06_assessments"
    assessment_dir.mkdir(parents=True)
    manifest_path = assessment_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "assessments": [
                    {
                        "file": "week_01_quiz.xml",
                        "type": "qti",
                        "title": "Week 1 Quiz",
                        "identifier": "RES_week_01_quiz",
                    },
                    {
                        "file": "week_01_discussion.xml",
                        "type": "discussion",
                        "title": "Week 1 Discussion",
                        "identifier": "RES_week_01_discussion",
                    },
                    {
                        "file": "week_01_assignment.xml",
                        "type": "assignment",
                        "title": "Week 1 Assignment",
                        "identifier": "RES_week_01_assignment",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (assessment_dir / "week_01_quiz.xml").write_text(
        "<questestinterop><assessment><section>"
        '<item ident="Q-1" title="TO-01" />'
        "</section></assessment></questestinterop>",
        encoding="utf-8",
    )
    (assessment_dir / "week_01_discussion.xml").write_text(
        "<topic><title>TO-01 discussion</title><text>Prompt</text></topic>",
        encoding="utf-8",
    )
    (assessment_dir / "week_01_assignment.xml").write_text(
        "<assignment><title>TO-01 assignment</title><text>Prompt</text></assignment>",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "semantik_chunks" / "chunks.jsonl"
    chunks_path.parent.mkdir()
    chunks_path.write_text(
        json.dumps(
            {
                "id": "chunk-1",
                "text": "Grounded source.",
                # SemantiK source chunks may legitimately carry no objective
                # refs; the documented W5.E resolution surface unions the
                # canonical synthesized objective ids.
                "learning_outcome_refs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    objectives_path = tmp_path / "synthesized_objectives.json"
    objectives_path.write_text(
        json.dumps(
            {
                "terminal_objectives": [{"id": "TO-01"}],
                "chapter_objectives": [],
            }
        ),
        encoding="utf-8",
    )
    outputs = {
        # An unrelated generic output_path must never shadow the manifest.
        "semantik_conversion": {"output_path": str(tmp_path / "source.html")},
        "chunking": {"semantik_chunks_path": str(chunks_path)},
        "course_planning": {
            "synthesized_objectives_path": str(objectives_path)
        },
        "assessment_synthesis": {
            "assessments_dir": str(assessment_dir),
            "manifest_path": str(manifest_path),
            "assessment_count": 2,
            "discussion_count": 1,
            "assignment_count": 1,
        },
    }
    return outputs, manifest_path, chunks_path


def test_declared_outputs_match_assessment_handler_envelope() -> None:
    keys = set(_get_phase_output_keys("assessment_synthesis"))
    assert {"assessments_dir", "manifest_path"} <= keys


def test_alignment_router_uses_manifest_and_upstream_semantik_chunks(
    tmp_path: Path,
) -> None:
    outputs, manifest_path, chunks_path = _representative_outputs(tmp_path)
    inputs, missing = default_router().build(ALIGNMENT_PATH, outputs, {})

    assert missing == []
    assert inputs["assessments_path"] == str(manifest_path)
    assert inputs["chunks_path"] == str(chunks_path)

    # Prove the real validator executes instead of receiving an executor-level
    # GATE_SKIPPED_MISSING_INPUTS waiver.  The product manifest carries XML
    # resources rather than embedded quiz questions, so NO_QUESTIONS is the
    # validator's current honest result.
    result = AssessmentObjectiveAlignmentValidator().validate(inputs)
    assert result.validator_version != "skipped"
    assert result.passed is True
    assert all(i.code != "GATE_SKIPPED_MISSING_INPUTS" for i in result.issues)


def test_grounding_router_and_validator_see_manifest_xml_items(
    tmp_path: Path,
) -> None:
    outputs, manifest_path, chunks_path = _representative_outputs(tmp_path)
    inputs, missing = default_router().build(GROUNDING_PATH, outputs, {})

    assert missing == []
    assert inputs["assessments_path"] == str(manifest_path)
    assert inputs["chunks_path"] == str(chunks_path)

    result = DiscussionAssignmentGroundingValidator().validate(inputs)
    codes = {issue.code for issue in result.issues}
    assert result.validator_version != "skipped"
    assert "NO_DISCUSSION_ASSIGNMENT_ITEMS" not in codes
    assert "GROUNDING_INPUTS_UNAVAILABLE" not in codes
    assert "DISCUSSION_UNGROUNDED" not in codes
    assert "ASSIGNMENT_UNGROUNDED" not in codes


def test_strict_product_coverage_fails_when_one_canonical_objective_is_missing(
    tmp_path: Path,
) -> None:
    outputs, manifest_path, chunks_path = _representative_outputs(tmp_path)
    objectives_path = Path(
        outputs["course_planning"]["synthesized_objectives_path"]
    )
    objectives_path.write_text(
        json.dumps(
            {
                "terminal_objectives": [{"id": "TO-01"}],
                "chapter_objectives": [
                    {"id": "CO-01"},
                    {"id": "CO-02"},
                ],
            }
        ),
        encoding="utf-8",
    )
    # Two of three canonical objectives appear in QTI. Source chunks need not
    # carry objective refs because W5.E resolves through the canonical union.
    quiz = manifest_path.parent / "week_01_quiz.xml"
    quiz.write_text(
        "<questestinterop><assessment><section>"
        '<item ident="Q-1" title="TO-01" />'
        '<item ident="Q-2" title="CO-01" />'
        "</section></assessment></questestinterop>",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assessments"].append(
        {"file": quiz.name, "type": "qti", "title": "Quiz"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    inputs, missing = default_router().build(ALIGNMENT_PATH, outputs, {})
    assert missing == []
    result = AssessmentObjectiveAlignmentValidator().validate(inputs)
    codes = {issue.code for issue in result.issues}
    assert result.passed is False
    assert "ASSESSMENT_OBJECTIVE_COVERAGE_INCOMPLETE" in codes
    assert "CHUNK_OBJECTIVE_COVERAGE_INCOMPLETE" not in codes


def test_trainforge_alignment_routes_canonical_universe_and_fails_phantom_ref(
    tmp_path: Path,
) -> None:
    """The routed validator must fail a question absent from both live
    chunks and the canonical objective universe."""
    assessments = tmp_path / "assessments.json"
    assessments.write_text(
        json.dumps(
            {
                "questions": [
                    {"question_id": "Q-phantom", "objective_id": "TO-99"}
                ]
            }
        ),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps({"id": "chunk-1", "learning_outcome_refs": ["TO-01"]})
        + "\n",
        encoding="utf-8",
    )
    objectives = tmp_path / "synthesized_objectives.json"
    objectives.write_text(
        json.dumps(
            {
                "terminal_objectives": [{"id": "TO-01"}],
                "chapter_objectives": [],
            }
        ),
        encoding="utf-8",
    )
    outputs = {
        "course_planning": {"synthesized_objectives_path": str(objectives)},
        "chunking": {"semantik_chunks_path": str(chunks)},
        "trainforge_assessment": {"assessments_path": str(assessments)},
    }

    inputs, missing = default_router().build(ALIGNMENT_PATH, outputs, {})
    assert missing == []
    assert inputs["chunks_path"] == str(chunks)
    assert inputs["synthesized_objectives_path"] == str(objectives)

    result = AssessmentObjectiveAlignmentValidator().validate(inputs)
    assert result.passed is False
    assert "PHANTOM_OBJECTIVE_REFS" in {issue.code for issue in result.issues}


def test_assessment_quality_flattens_nested_product_questions(
    tmp_path: Path,
) -> None:
    product = tmp_path / "assessment_items.json"
    product.write_text(
        json.dumps({
            "assessments": [
                {"questions": [
                    {"question_id": "Q-1", "objective_id": "CO-01"},
                    {"question_id": "Q-2", "objective_id": "CO-02"},
                ]},
                {"questions": [
                    {"question_id": "Q-3", "objective_id": "CO-03"},
                ]},
            ]
        }),
        encoding="utf-8",
    )

    inputs, missing = _build_assessment_quality(
        {"assessment_synthesis": {"assessments_path": str(product)}},
        {},
    )

    assert missing == []
    assert [q["question_id"] for q in inputs["assessment_data"]["questions"]] == [
        "Q-1", "Q-2", "Q-3",
    ]


def test_assessment_quality_prefers_trainforge_phase_local_json(
    tmp_path: Path,
) -> None:
    product = tmp_path / "assessment_items.json"
    product.write_text(
        json.dumps({"assessments": [{"questions": [
            {"question_id": f"PRODUCT-{i}"} for i in range(4)
        ]}]}),
        encoding="utf-8",
    )
    trainforge = tmp_path / "assessments.json"
    trainforge.write_text(
        json.dumps({"questions": [
            {"question_id": f"TRAINFORGE-{i}"} for i in range(2)
        ]}),
        encoding="utf-8",
    )

    inputs, missing = _build_assessment_quality(
        {
            "assessment_synthesis": {"assessments_path": str(product)},
            "trainforge_assessment": {"assessments_path": str(trainforge)},
        },
        {},
    )

    assert missing == []
    assert inputs == {"assessment_path": str(trainforge)}
