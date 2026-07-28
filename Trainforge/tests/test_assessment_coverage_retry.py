from __future__ import annotations

from collections import Counter

import pytest

from Trainforge.generators.assessment_generator import (
    AssessmentGenerator,
    QuestionData,
    SkippedItem,
)


def _question(objective_id: str, bloom_level: str) -> QuestionData:
    return QuestionData(
        question_id=f"q-{objective_id}",
        question_type="multiple_choice",
        stem="<p>Which grounded statement is accurate?</p>",
        bloom_level=bloom_level,
        objective_id=objective_id,
        choices=[
            {"text": "<p>Correct grounded statement.</p>", "is_correct": True},
            {"text": "<p>Incorrect grounded statement A.</p>", "is_correct": False},
            {"text": "<p>Incorrect grounded statement B.</p>", "is_correct": False},
        ],
        source_chunks=["chunk-1"],
    )


def test_coverage_mode_retries_only_uncovered_objectives(monkeypatch):
    """A rejected first attempt must not become duplicate covered-objective work."""
    generator = AssessmentGenerator(capture=None, check_leaks=False)
    attempts: Counter[str] = Counter()

    def generate_question(*, objective_id, bloom_level, source_chunks):
        attempts[objective_id] += 1
        if objective_id == "CO-02" and attempts[objective_id] == 1:
            return SkippedItem(
                question_id="skip-co-02",
                question_type="multiple_choice",
                bloom_level=bloom_level,
                objective_id=objective_id,
                reason="apparatus_contaminated_content",
            )
        return _question(objective_id, bloom_level)

    monkeypatch.setattr(generator, "_generate_question", generate_question)
    monkeypatch.setattr(generator, "_apparatus_guard", lambda result: result)

    assessment = generator.generate(
        course_code="DEMO",
        objective_ids=["CO-01", "CO-02", "CO-03"],
        bloom_levels=["remember", "understand"],
        question_count=3,
        source_chunks=[{"id": "chunk-1", "text": "Grounded source."}],
        ensure_objective_coverage=True,
    )

    assert {question.objective_id for question in assessment.questions} == {
        "CO-01",
        "CO-02",
        "CO-03",
    }
    assert attempts == Counter({"CO-02": 2, "CO-01": 1, "CO-03": 1})


def test_coverage_mode_repairs_objective_removed_by_leak_filter(monkeypatch):
    generator = AssessmentGenerator(capture=None, check_leaks=True)
    attempts: Counter[str] = Counter()

    def generate_question(*, objective_id, bloom_level, source_chunks):
        attempts[objective_id] += 1
        question = _question(objective_id, bloom_level)
        question.question_id = f"q-{objective_id}-{attempts[objective_id]}"
        if objective_id == "CO-02" and attempts[objective_id] == 1:
            question.stem = "<p>Correct grounded statement.</p>"
        return question

    monkeypatch.setattr(generator, "_generate_question", generate_question)
    monkeypatch.setattr(generator, "_apparatus_guard", lambda result: result)

    assessment = generator.generate(
        course_code="DEMO",
        objective_ids=["CO-01", "CO-02"],
        bloom_levels=["remember", "understand"],
        question_count=2,
        source_chunks=[{"id": "chunk-1", "text": "Grounded source."}],
        ensure_objective_coverage=True,
    )

    assert {question.objective_id for question in assessment.questions} == {
        "CO-01",
        "CO-02",
    }
    assert attempts["CO-02"] == 2


def test_coverage_retry_rotates_grounded_source_pool(monkeypatch):
    generator = AssessmentGenerator(capture=None, check_leaks=False)

    def generate_question(*, objective_id, bloom_level, source_chunks):
        if source_chunks[0]["id"] == "bad-front":
            return SkippedItem(
                question_id="skip-front",
                question_type="multiple_choice",
                bloom_level=bloom_level,
                objective_id=objective_id,
                reason="apparatus_contaminated_content",
            )
        return _question(objective_id, bloom_level)

    monkeypatch.setattr(generator, "_generate_question", generate_question)
    monkeypatch.setattr(generator, "_apparatus_guard", lambda result: result)

    assessment = generator.generate(
        course_code="DEMO",
        objective_ids=["CO-01"],
        bloom_levels=["remember"],
        question_count=1,
        source_chunks=[
            {"id": "bad-front", "text": "Contaminated apparatus."},
            {"id": "grounded-next", "text": "Grounded prose."},
        ],
        ensure_objective_coverage=True,
    )

    assert [q.objective_id for q in assessment.questions] == ["CO-01"]


def test_diversified_trim_preserves_tail_objective_after_two_tier_overshoot(
    monkeypatch,
):
    generator = AssessmentGenerator(capture=None, check_leaks=False)

    def build_by_subtype(
        subtype, question_id, objective_id, bloom_level, source_chunks
    ):
        first = _question(objective_id, bloom_level)
        first.question_id = f"{question_id}-answer"
        first.item_subtype = subtype
        if objective_id == "CO-01":
            reason = _question(objective_id, bloom_level)
            reason.question_id = f"{question_id}-reason"
            reason.item_subtype = "two_tier_reason"
            return [first, reason]
        return first

    monkeypatch.setattr(generator, "_build_by_subtype", build_by_subtype)
    monkeypatch.setattr(generator, "_apparatus_guard", lambda result: result)

    assessment = generator.generate_diversified(
        course_code="DEMO",
        objective_ids=["CO-01", "CO-02", "CO-03"],
        bloom_levels=["remember"],
        question_count=3,
        source_chunks=[{"id": "chunk-1", "text": "Grounded source."}],
        ensure_objective_coverage=True,
    )

    assert len(assessment.questions) == 3
    assert {q.objective_id for q in assessment.questions} == {
        "CO-01",
        "CO-02",
        "CO-03",
    }


def test_diversified_retries_objective_missed_by_coverage_and_count_fill(
    monkeypatch,
):
    generator = AssessmentGenerator(capture=None, check_leaks=False)
    observed_co3_chunk_heads = []

    def build_by_subtype(
        subtype, question_id, objective_id, bloom_level, source_chunks
    ):
        if objective_id == "CO-03":
            observed_co3_chunk_heads.append(source_chunks[0]["id"])
        if objective_id == "CO-03" and source_chunks[0]["id"] == "bad-front":
            return SkippedItem(
                question_id=question_id,
                question_type="multiple_choice",
                bloom_level=bloom_level,
                objective_id=objective_id,
                reason="apparatus_contaminated_content",
            )
        question = _question(objective_id, bloom_level)
        question.question_id = question_id
        question.item_subtype = subtype
        if objective_id == "CO-01":
            reason = _question(objective_id, bloom_level)
            reason.question_id = f"{question_id}-reason"
            reason.item_subtype = "two_tier_reason"
            return [question, reason]
        return question

    monkeypatch.setattr(generator, "_build_by_subtype", build_by_subtype)
    monkeypatch.setattr(generator, "_apparatus_guard", lambda result: result)
    monkeypatch.setattr(
        generator,
        "build_mc_single",
        lambda question_id, objective_id, bloom_level, source_chunks: (
            SkippedItem(
                question_id=question_id,
                question_type="multiple_choice",
                bloom_level=bloom_level,
                objective_id=objective_id,
                reason="bad_scoped_front",
            )
            if source_chunks[0]["id"] == "bad-front"
            else _question(objective_id, bloom_level)
        ),
    )

    assessment = generator.generate_diversified(
        course_code="DEMO",
        objective_ids=["CO-01", "CO-02", "CO-03"],
        bloom_levels=["remember"],
        question_count=3,
        source_chunks=[
            {"id": "global-only", "text": "Unrelated global prose."},
        ],
        chunks_by_objective={
            "CO-01": [{"id": "co1", "text": "CO 1 prose."}],
            "CO-02": [{"id": "co2", "text": "CO 2 prose."}],
            "CO-03": [
                {"id": "bad-front", "text": "Contaminated apparatus."},
                {"id": "grounded-next", "text": "Grounded CO 3 prose."},
            ],
        },
        ensure_objective_coverage=True,
    )

    assert len(assessment.questions) == 3
    assert {q.objective_id for q in assessment.questions} == {
        "CO-01",
        "CO-02",
        "CO-03",
    }
    assert "global-only" not in observed_co3_chunk_heads
    assert observed_co3_chunk_heads[-1] == "grounded-next"


def test_diversified_all_scoped_retries_fail_loudly_with_reasons(monkeypatch):
    generator = AssessmentGenerator(capture=None, check_leaks=False)

    def always_skip(
        subtype, question_id, objective_id, bloom_level, source_chunks
    ):
        return SkippedItem(
            question_id=question_id,
            question_type="multiple_choice",
            bloom_level=bloom_level,
            objective_id=objective_id,
            reason=f"no_grounded_{source_chunks[0]['id']}",
        )

    monkeypatch.setattr(generator, "_build_by_subtype", always_skip)
    monkeypatch.setattr(
        generator,
        "build_mc_single",
        lambda question_id, objective_id, bloom_level, source_chunks: (
            always_skip(
                "mc_single",
                question_id,
                objective_id,
                bloom_level,
                source_chunks,
            )
        ),
    )
    monkeypatch.setattr(generator, "_apparatus_guard", lambda result: result)

    with pytest.raises(RuntimeError) as exc_info:
        generator.generate_diversified(
            course_code="DEMO",
            objective_ids=["CO-09"],
            bloom_levels=["remember"],
            question_count=1,
            source_chunks=[{"id": "global", "text": "Global prose."}],
            chunks_by_objective={
                "CO-09": [{"id": "scoped", "text": "Scoped prose."}],
            },
            ensure_objective_coverage=True,
        )

    message = str(exc_info.value)
    assert "CO-09" in message
    assert "no_grounded_scoped" in message
