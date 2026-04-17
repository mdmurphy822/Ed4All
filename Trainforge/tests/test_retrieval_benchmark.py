"""Tests for Trainforge/rag/retrieval_benchmark.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIBV2_TOOLS = PROJECT_ROOT / "LibV2" / "tools"
if str(LIBV2_TOOLS) not in sys.path:
    sys.path.insert(0, str(LIBV2_TOOLS))

from Trainforge.rag.retrieval_benchmark import (
    build_question_set,
    recall_at_k,
    run_benchmark,
    write_benchmark,
)


FIXTURE = PROJECT_ROOT / "Trainforge" / "tests" / "fixtures" / "mini_course_summaries"


def test_recall_at_k_basic_math():
    """Arithmetic sanity: 1 of 2 relevant in top-k => recall == 0.5."""
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], ["a", "b"], 5) == pytest.approx(1.0)
    assert recall_at_k(["x", "y", "z"], ["a", "b"], 3) == pytest.approx(0.0)
    # Empty relevant set => 0.0 by convention
    assert recall_at_k(["a"], [], 5) == pytest.approx(0.0)


def test_build_question_set_from_fixture():
    """Question set derivation uses course.json LOs and chunk LO refs."""
    chunks = [
        json.loads(line) for line in (FIXTURE / "chunks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    course = json.loads((FIXTURE / "course.json").read_text())

    questions = build_question_set(chunks, course)

    assert len(questions) == 2
    by_lo = {q["lo_id"]: q for q in questions}
    assert set(by_lo.keys()) == {"lo-01", "lo-02"}
    # lo-01 is tagged on c001 and c005
    assert set(by_lo["lo-01"]["relevant_chunk_ids"]) == {"c001", "c005"}
    # lo-02 is tagged on c002, c003, c004, c006
    assert set(by_lo["lo-02"]["relevant_chunk_ids"]) == {"c002", "c003", "c004", "c006"}


def test_run_benchmark_reports_recall_on_fixture():
    """End-to-end: run the benchmark and assert the shape + that both
    text and summary variants report a recall@5 number.
    """
    results = run_benchmark(
        FIXTURE / "chunks.jsonl",
        FIXTURE / "course.json",
        k_values=(1, 5, 10),
    )

    assert results["chunk_count"] == 6
    assert results["question_count"] == 2
    assert set(results["fields_compared"]) >= {"text", "summary"}

    text_scores = results["variants"]["text"]
    summary_scores = results["variants"]["summary"]

    # recall@5 should exist and be in [0, 1] for both variants.
    assert 0.0 <= text_scores["recall@5"] <= 1.0
    assert 0.0 <= summary_scores["recall@5"] <= 1.0
    # Also recall@1 and recall@10 exist.
    for v in ("text", "summary"):
        for k in (1, 5, 10):
            key = f"recall@{k}"
            assert key in results["variants"][v]
            assert 0.0 <= results["variants"][v][key] <= 1.0


def test_run_benchmark_computes_recall_at_5_correctly(tmp_path):
    """Handcrafted micro-fixture: retrieve should surface the relevant
    chunk in top-5, giving recall@5 == 1.0 for an exact-term query.
    """
    chunks = [
        {
            "id": "x1",
            "text": "The photopyroelectric effect measures thermal diffusivity.",
            "summary": "Photopyroelectric measurement covers thermal diffusivity calibration.",
            "learning_outcome_refs": ["lo-a"],
        },
        {
            "id": "x2",
            "text": "Cognitive load theory describes working memory limits.",
            "summary": "Cognitive load theory describes working memory limits.",
            "learning_outcome_refs": ["lo-b"],
        },
        {
            "id": "x3",
            "text": "Neutral filler about unrelated topics for noise.",
            "summary": "Neutral filler about unrelated topics for noise.",
            "learning_outcome_refs": [],
        },
    ]
    course = {
        "learning_outcomes": [
            {"id": "lo-a", "statement": "photopyroelectric thermal diffusivity"},
            {"id": "lo-b", "statement": "cognitive load working memory"},
        ]
    }
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    course_path = tmp_path / "course.json"
    course_path.write_text(json.dumps(course))

    results = run_benchmark(chunks_path, course_path, k_values=(1, 5))

    # Both LOs have exactly one relevant chunk, and that chunk contains
    # the query terms verbatim, so text-variant recall@5 should be 1.0.
    assert results["variants"]["text"]["recall@5"] == pytest.approx(1.0)
    # Summary variant also has the terms and should also hit 1.0.
    assert results["variants"]["summary"]["recall@5"] == pytest.approx(1.0)


def test_write_benchmark_creates_quality_artifact(tmp_path):
    """write_benchmark should create quality/retrieval_benchmark.json at the
    expected path and the JSON should round-trip.
    """
    # Build a minimal output-dir layout mimicking Trainforge/output/<slug>/.
    out = tmp_path / "course_out"
    corpus = out / "corpus"
    corpus.mkdir(parents=True)
    # Copy the fixture files into the expected layout.
    (corpus / "chunks.jsonl").write_text((FIXTURE / "chunks.jsonl").read_text())
    (out / "course.json").write_text((FIXTURE / "course.json").read_text())

    out_path, results = write_benchmark(out)

    assert out_path == out / "quality" / "retrieval_benchmark.json"
    assert out_path.exists()

    on_disk = json.loads(out_path.read_text())
    assert on_disk == results
    assert "variants" in on_disk
    assert "text" in on_disk["variants"]
    assert "summary" in on_disk["variants"]


def test_question_set_excludes_degenerate_queries():
    """LOs with zero matching chunks should be included in the questions
    list (as a record) but not count toward the mean recall.
    """
    chunks = [
        {"id": "c1", "text": "alpha", "summary": "alpha", "learning_outcome_refs": ["lo-x"]},
    ]
    course = {
        "learning_outcomes": [
            {"id": "lo-x", "statement": "alpha"},
            {"id": "lo-y", "statement": "beta that is not tagged on any chunk"},
        ]
    }

    qs = build_question_set(chunks, course)
    by_id = {q["lo_id"]: q for q in qs}
    assert by_id["lo-x"]["relevant_chunk_ids"] == ["c1"]
    assert by_id["lo-y"]["relevant_chunk_ids"] == []
