"""Tests for AssessmentObjectiveAlignmentValidator (Wave 24 scope 7).

The validator fails loudly when any assessment question's objective_id
is not present in any chunk's learning_outcome_refs. This guards
against the pre-Wave-24 failure mode (disjoint LO naming schemes) from
resurfacing silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.validators.assessment_objective_alignment import (
    AssessmentObjectiveAlignmentValidator,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_chunks(path: Path, chunks: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def test_all_aligned_passes(tmp_path):
    """Every question's objective_id appears in chunk refs → pass."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
            {"question_id": "Q2", "objective_id": "CO-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
        {"id": "c2", "learning_outcome_refs": ["co-01", "to-02"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert result.passed
    assert result.score == 1.0


def test_phantom_refs_fail_loudly(tmp_path):
    """Questions referencing non-existent objective_ids → critical fail."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "PHYS_101_OBJ_1"},
            {"question_id": "Q2", "objective_id": "PHYS_101_OBJ_2"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01", "co-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    # PHANTOM_OBJECTIVE_REFS code should surface.
    codes = {i.code for i in result.issues}
    assert "PHANTOM_OBJECTIVE_REFS" in codes


def test_empty_questions_list_is_skipped(tmp_path):
    """No questions in payload → warning-only, not critical."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {"questions": []})
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    # Passes with a warning.
    assert result.passed


def test_missing_inputs_error(tmp_path):
    """Missing required inputs → critical fail with descriptive code."""
    result = AssessmentObjectiveAlignmentValidator().validate({})
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "MISSING_ASSESSMENTS_PATH" in codes


def test_case_insensitive_match(tmp_path):
    """TO-01 in question matches to-01 in chunks (Trainforge lowercases)."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert result.passed


def test_empty_chunkset_fails_closed(tmp_path):
    """C4 audit fix: zero learning_outcome_refs across chunks file
    indicates an upstream chunking failure. Fail closed with the
    named code rather than vacuously passing.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    # File exists, parses cleanly, but no chunk carries refs.
    _write_chunks(chunks, [
        {"id": "c1"},
        {"id": "c2", "learning_outcome_refs": []},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "ASSESSMENT_ALIGNMENT_NO_CHUNKS" in codes
    crit = [i for i in result.issues if i.code == "ASSESSMENT_ALIGNMENT_NO_CHUNKS"]
    assert crit and crit[0].severity == "critical"
    # Operator hint must reference the upstream phase + the chunks path.
    msg = crit[0].message
    assert str(chunks) in msg
    assert "imscc_chunking" in msg or "trainforge_assessment" in msg


def test_completely_empty_chunks_file_fails_closed(tmp_path):
    """An empty chunks.jsonl file (zero lines) also fails closed."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    chunks.write_text("", encoding="utf-8")
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "ASSESSMENT_ALIGNMENT_NO_CHUNKS" in codes


def test_question_missing_objective_id_fails(tmp_path):
    """Question with no objective_id → critical."""
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1"},  # no objective_id
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "QUESTION_MISSING_OBJECTIVE" in codes


# --------------------------------------------------------------------- #
# Wave 5 W5.E: synthesized-objectives union resolution surface.
# --------------------------------------------------------------------- #


class _StubCapture:
    """Minimal DecisionCapture stub used by the W5.E rationale assertions."""

    def __init__(self) -> None:
        self.calls: list = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.calls.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "kwargs": kw,
        })


def test_assessment_alignment_resolves_objectives_via_synthesized_objectives(
    tmp_path,
):
    """W5.E: chunks have empty learning_outcome_refs but synthesized
    objectives carry the assessment's objective_id → union resolution
    surface lets the validator pass.

    Pre-W5.E: every chunk's learning_outcome_refs is empty, so the
    chunks-only resolution surface vacuously phantoms every question →
    ASSESSMENT_ALIGNMENT_NO_CHUNKS critical fail.
    Post-W5.E: synthesized_objectives_path contributes ``TO-05`` to the
    resolution set; the union is non-empty and contains the assessment
    objective → pass.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    synthesized = tmp_path / "synthesized_objectives.json"

    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-05"},
        ],
    })
    # Every chunk has empty refs (the failure mode W5.E unblocks).
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": []},
        {"id": "c2", "learning_outcome_refs": []},
    ])
    # Synthesized objectives carry the TO-05 the assessment references.
    _write_json(synthesized, {
        "terminal_objectives": [
            {"id": "TO-05", "statement": "Demonstrate the W5.E union surface."},
        ],
        "chapter_objectives": [],
    })

    # Pre-W5.E behaviour: no synthesized_objectives_path → critical
    # fail with ASSESSMENT_ALIGNMENT_NO_CHUNKS.
    pre = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not pre.passed
    pre_codes = {i.code for i in pre.issues}
    assert "ASSESSMENT_ALIGNMENT_NO_CHUNKS" in pre_codes

    # Post-W5.E behaviour: synthesized_objectives_path provided →
    # passes because TO-05 is in the union resolution surface.
    capture = _StubCapture()
    post = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "synthesized_objectives_path": str(synthesized),
        "decision_capture": capture,
    })
    assert post.passed, [i.code for i in post.issues]
    assert post.score == 1.0

    # Decision rationale carries the W5.E
    # ``resolved_via_synthesized_objectives`` flag.
    assert len(capture.calls) == 1
    rationale = capture.calls[0]["rationale"]
    assert "resolved_via_synthesized_objectives=True" in rationale
    metrics = capture.calls[0]["kwargs"].get("metrics") or {}
    assert metrics.get("resolved_via_synthesized_objectives") is True


def test_assessment_alignment_resolves_via_libv2_archive_form_objectives(
    tmp_path,
):
    """W5.E: synthesized objectives in LibV2 archive form
    (``terminal_outcomes`` + ``component_objectives``) also feed the
    union resolution surface.

    The canonical loader (``_flatten_objectives``) handles both shapes;
    this test pins parity for the LibV2-archived corpus path so a
    rebuild from a LibV2 export doesn't regress.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    synthesized = tmp_path / "synthesized_objectives.json"

    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-09"},
            {"question_id": "Q2", "objective_id": "CO-03"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": []},
    ])
    # LibV2 archive form: terminal_outcomes + grouped component_objectives.
    _write_json(synthesized, {
        "terminal_outcomes": [
            {"id": "TO-09", "statement": "Top-level archived outcome."},
        ],
        "component_objectives": [
            {
                "chapter": "ch3",
                "objectives": [
                    {"id": "CO-03", "statement": "Component-level archived objective."},
                ],
            },
        ],
    })

    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "synthesized_objectives_path": str(synthesized),
    })
    assert result.passed, [i.code for i in result.issues]


def test_assessment_alignment_back_compat_no_synthesized_objectives_path(
    tmp_path,
):
    """W5.E back-compat: when synthesized_objectives_path is NOT
    provided, validator behaviour is byte-identical to pre-W5.E.

    Re-runs the pre-W5.E happy path + the pre-W5.E phantom path and
    asserts both produce the same passed/score/codes as the existing
    tests above. Guards against accidental coupling.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"

    # Happy path mirror of test_all_aligned_passes.
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    happy = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert happy.passed
    assert happy.score == 1.0
    # No SYNTHESIZED_OBJECTIVES_NOT_FOUND warning when the path was
    # never provided (back-compat invariant).
    happy_codes = {i.code for i in happy.issues}
    assert "SYNTHESIZED_OBJECTIVES_NOT_FOUND" not in happy_codes

    # Phantom path mirror of test_phantom_refs_fail_loudly.
    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "PHANTOM_OBJ_1"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    phantom = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
    })
    assert not phantom.passed
    phantom_codes = {i.code for i in phantom.issues}
    assert "PHANTOM_OBJECTIVE_REFS" in phantom_codes


def test_assessment_alignment_handles_missing_synthesized_objectives_file(
    tmp_path,
):
    """W5.E graceful-degrade: synthesized_objectives_path provided but
    the file does not exist → validator emits a warning-severity
    SYNTHESIZED_OBJECTIVES_NOT_FOUND issue and falls back to the
    chunks-only resolution surface (NOT a critical fail).
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    missing = tmp_path / "does_not_exist" / "synthesized_objectives.json"

    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])

    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "synthesized_objectives_path": str(missing),
    })
    # Chunks-only fallback: TO-01 is present in chunks → still passes.
    assert result.passed
    # Warning surfaced for operators.
    warning_codes = {
        i.code for i in result.issues if i.severity == "warning"
    }
    assert "SYNTHESIZED_OBJECTIVES_NOT_FOUND" in warning_codes
    # No critical issues introduced.
    critical_codes = {
        i.code for i in result.issues if i.severity == "critical"
    }
    assert not critical_codes


# --------------------------------------------------------------------- #
# Wave 6 W6.E: per-question-type metrics on the alignment decision.
# --------------------------------------------------------------------- #


def test_decision_metrics_carry_question_type(tmp_path):
    """W6.E: every per-question alignment decision metrics dict carries
    ``question_type`` AND the rationale string interpolates the value.

    Pair an assessment q with ``question_type="multiple_choice"`` and a
    phantom objective_id; assert (a) the captured event's
    ``metrics["question_type"] == "multiple_choice"`` and (b) the
    rationale string contains ``"multiple_choice"`` so post-hoc replay
    can segment phantom-objective failures by question shape.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            {
                "question_id": "Q1",
                "objective_id": "PHANTOM_OBJ_1",
                "question_type": "multiple_choice",
            },
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    capture = _StubCapture()
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert not result.passed
    assert len(capture.calls) == 1
    call = capture.calls[0]
    metrics = call["kwargs"].get("metrics") or {}
    assert metrics.get("question_type") == "multiple_choice"
    assert "multiple_choice" in call["rationale"]


def test_decision_metrics_question_type_empty_for_legacy_q(tmp_path):
    """W6.E defensive contract: a question with NO ``question_type``
    field (legacy / malformed) still stamps ``question_type=""`` in
    the metrics dict — empty string, NOT a missing key. Keeps the
    metrics dict shape invariant across all per-question events so
    post-hoc replay can rely on the field always being present.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            # Legacy q: no question_type (and no QTI ``type`` field).
            {"question_id": "Q1", "objective_id": "TO-01"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    capture = _StubCapture()
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert result.passed
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["kwargs"].get("metrics") or {}
    # Key MUST be present; value MUST be the empty string (not None).
    assert "question_type" in metrics
    assert metrics["question_type"] == ""
    assert metrics["question_type"] is not None


def test_decision_metrics_question_type_normalizes_qti_field(tmp_path):
    """W6.E + W4.B field-name resolution chain inheritance: a question
    carrying ``type="multiple_choice"`` (QTI shape, NOT the generator's
    ``question_type``) still resolves to ``"multiple_choice"`` in the
    metrics dict because ``_normalize_question_type`` falls back to
    the QTI ``type`` field when ``question_type`` is absent.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    _write_json(assessments, {
        "questions": [
            # QTI-parsed shape: ``type`` not ``question_type``.
            {"question_id": "Q1", "objective_id": "TO-01", "type": "multiple_choice"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    capture = _StubCapture()
    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "decision_capture": capture,
    })
    assert result.passed
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["kwargs"].get("metrics") or {}
    assert metrics.get("question_type") == "multiple_choice"


def test_assessment_alignment_union_surface_does_not_mask_real_phantoms(
    tmp_path,
):
    """W5.E invariant: an assessment objective_id that is in NEITHER
    chunks-side refs NOR the synthesized objectives' id set still
    triggers the PHANTOM_OBJECTIVE_REFS critical. The union widens the
    resolution surface; it does not silently accept references the
    union doesn't contain.
    """
    assessments = tmp_path / "assessments.json"
    chunks = tmp_path / "chunks.jsonl"
    synthesized = tmp_path / "synthesized_objectives.json"

    _write_json(assessments, {
        "questions": [
            {"question_id": "Q1", "objective_id": "TO-01"},
            {"question_id": "Q2", "objective_id": "PHANTOM_NOT_ANYWHERE"},
        ],
    })
    _write_chunks(chunks, [
        {"id": "c1", "learning_outcome_refs": ["to-01"]},
    ])
    _write_json(synthesized, {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Already resolvable via chunks."},
            {"id": "TO-02", "statement": "Resolvable only via synthesized."},
        ],
        "chapter_objectives": [],
    })

    result = AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "synthesized_objectives_path": str(synthesized),
    })
    assert not result.passed
    codes = {i.code for i in result.issues}
    assert "PHANTOM_OBJECTIVE_REFS" in codes
