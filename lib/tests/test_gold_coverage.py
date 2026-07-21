"""retrieval-answer-eval-set P0 — tests for lib/retrieval/gold_coverage.py.

Synthetic union-corpus fixture (source + course populations across weeks) +
a synthetic gold set; asserts the §1.3 coverage matrix and its warning codes.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.retrieval.gold_coverage import (
    build_coverage_report,
    write_coverage_report,
    _population_of_chunk,
    _week_of_item_path,
)


def _chunks():
    return {
        # course-page chunks (week_NN/ item_paths)
        "c_course_w1": {"id": "c_course_w1", "text": "Week 1 generated course page about fractions.",
                        "source": {"item_path": "week_01/lesson.html"},
                        "learning_outcome_refs": ["to-01", "co-01"]},
        "c_course_w2": {"id": "c_course_w2", "text": "Week 2 generated course page about decimals.",
                        "source": {"item_path": "week_02/lesson.html"},
                        "learning_outcome_refs": ["co-02"]},
        # source (original document) chunk: carries source_references
        "c_source_a": {"id": "c_source_a", "text": "Original textbook says fractions need a common denominator.",
                       "source": {"item_path": "textbook.html",
                                  "source_references": [{"sourceId": "semantik:textbook#abc"}]},
                       "learning_outcome_refs": ["co-01"]},
        # source by flat-html heuristic (no source_references, flat *.html)
        "c_source_b": {"id": "c_source_b", "text": "Original textbook defines an irrational number.",
                       "source": {"item_path": "textbook.html"},
                       "learning_outcome_refs": ["co-03"]},
        # a chapter objective with zero question coverage but chunk support
        "c_course_w3": {"id": "c_course_w3", "text": "Week 3 page on integers.",
                        "source": {"item_path": "week_03/lesson.html"},
                        "learning_outcome_refs": ["co-99"]},
    }


def _gold():
    return {
        "schema_version": "1.1",
        "course_slug": "demo-union",
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": "0" * 64},
        "questions": [
            # course-population, week 1, difficulty easy, TO-01
            {"question_id": "gq-demo-union-0001", "question_type": "factual_recall",
             "question_text": "What is in week 1?", "difficulty": "easy",
             "relevant_passages": [
                 {"chunk_id": "c_course_w1", "relevance": "primary",
                  "anchor": {"item_path": "week_01/lesson.html", "text_quote": "x" * 41}}]},
            # source-population, difficulty medium
            {"question_id": "gq-demo-union-0002", "question_type": "factual_recall",
             "question_text": "What do fractions need?", "difficulty": "medium",
             "relevant_passages": [
                 {"chunk_id": "c_source_a", "relevance": "primary",
                  "anchor": {"item_path": "textbook.html", "text_quote": "y" * 41}}]},
            # both-population synthesis: spans a source chunk + a course chunk
            {"question_id": "gq-demo-union-0003", "question_type": "conceptual_synthesis",
             "question_text": "How does the textbook connect to week 2?", "difficulty": "hard",
             "relevant_passages": [
                 {"chunk_id": "c_source_b", "relevance": "primary",
                  "anchor": {"item_path": "textbook.html", "text_quote": "z" * 41}},
                 {"chunk_id": "c_course_w2", "relevance": "supporting",
                  "anchor": {"item_path": "week_02/lesson.html", "text_quote": "w" * 41}}]},
        ],
    }


def test_population_helper():
    chunks = _chunks()
    assert _population_of_chunk(chunks["c_course_w1"]) == "course"
    assert _population_of_chunk(chunks["c_source_a"]) == "source"
    assert _population_of_chunk(chunks["c_source_b"]) == "source"


def test_week_helper():
    assert _week_of_item_path("week_01/lesson.html") == "01"
    assert _week_of_item_path("week_12/x.html") == "12"
    assert _week_of_item_path("textbook.html") is None


def test_matrix_counts():
    report = build_coverage_report(_gold(), _chunks(), is_union=True)
    assert report.total_questions == 3
    assert report.by_type == {"factual_recall": 2, "conceptual_synthesis": 1}
    assert report.by_difficulty == {"easy": 1, "medium": 1, "hard": 1}
    # populations: q1 course, q2 source, q3 both
    assert report.by_population.get("course") == 1
    assert report.by_population.get("source") == 1
    assert report.by_population.get("both") == 1


def test_week_underfilled_warning():
    report = build_coverage_report(_gold(), _chunks(), is_union=True)
    codes = {w.code for w in report.warnings}
    # weeks 1,2,3 each have < 3 questions -> underfilled
    assert "COVERAGE_WEEK_UNDERFILLED" in codes


def test_population_underfilled_warning():
    report = build_coverage_report(_gold(), _chunks(), is_union=True)
    codes = {w.code for w in report.warnings}
    # source(1)<10, course(1)<25, both(1)<5 all underfilled
    assert "COVERAGE_POPULATION_UNDERFILLED" in codes
    pop_warns = [w for w in report.warnings if w.code == "COVERAGE_POPULATION_UNDERFILLED"]
    assert len(pop_warns) == 3


def test_objective_warnings():
    report = build_coverage_report(_gold(), _chunks(), is_union=True)
    codes = {w.code for w in report.warnings}
    # to-01 covered by 1 q (< 2) -> underfilled
    assert "COVERAGE_TO_UNDERFILLED" in codes
    # co-99 has chunk support but no question -> uncovered
    assert "COVERAGE_CO_UNCOVERED" in codes


def test_non_union_skips_population_warnings():
    report = build_coverage_report(_gold(), _chunks(), is_union=False)
    codes = {w.code for w in report.warnings}
    assert "COVERAGE_POPULATION_UNDERFILLED" not in codes
    assert report.by_population == {}


def test_difficulty_skew_only_at_scale():
    # < 25 questions: no difficulty skew warning even though split is 1/1/1.
    report = build_coverage_report(_gold(), _chunks(), is_union=True)
    assert "COVERAGE_DIFFICULTY_SKEWED" not in {w.code for w in report.warnings}


def test_write_coverage_report(tmp_path: Path):
    course = tmp_path / "demo-union"
    chunks_dir = course / "corpus"
    chunks_dir.mkdir(parents=True)
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in _chunks().values():
            fh.write(json.dumps(c) + "\n")
    report, path = write_coverage_report(course, _gold(), is_union=True)
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["course_slug"] == "demo-union"
    assert on_disk["matrix"]["by_type"]["factual_recall"] == 2
    assert "warnings" in on_disk
