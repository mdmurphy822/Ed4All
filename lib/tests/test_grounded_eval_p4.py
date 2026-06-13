"""P4 integration tests for grounded_eval.py — the additive v1.1 per-question
scoring fields (key_point_coverage, part_coverage, citation_population) and the
headline rollups, exercised end-to-end on a hermetic v1.1 gold set.

CI-safe: builds a tiny course layout in tmp_path (corpus chunkset + a v1.1 gold
set carrying expected_key_points / parts / expected_citation_population) and
drives run_grounded_eval with an injected fake answer_fn (no LLM, no real slug).
Asserts the report JSON round-trips and carries the P4 fields + the §4 gold-pin
block additions (schema_version + question_count + authored_at).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lib.retrieval._text import normalize_ws
from lib.retrieval.grounded_eval import EVAL_SCHEMA_VERSION, run_grounded_eval
from lib.utils import sha256_file


# ===========================================================================
# Hermetic v1.1 course fixture (union corpus: source + course populations)
# ===========================================================================

_KP_TEXT = "A vector store is a database that indexes high-dimensional embedding vectors"
_CHUNK_SOURCE = (
    "A vector store is a database that indexes high-dimensional embedding "
    "vectors for nearest-neighbor retrieval, per the original textbook."
)
_CHUNK_COURSE = (
    "Retrieval quality is commonly measured with recall at k and mean "
    "reciprocal rank across the gold queries."
)


def _csha(text: str) -> str:
    return hashlib.sha256(normalize_ws(text).lower().encode("utf-8")).hexdigest()


@pytest.fixture
def v1_1_course(tmp_path, monkeypatch):
    slug = "syn-union-101"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        # source population: carries source_references[] + a flat *.html item_path
        {"id": "src1", "text": _CHUNK_SOURCE,
         "source": {"item_path": "textbook.html",
                    "source_references": [{"block_id": "b1", "pdf_pages": [3]}]},
         "learning_outcome_refs": ["to-01"]},
        # course population: a generated week page
        {"id": "crs1", "text": _CHUNK_COURSE,
         "source": {"item_path": "week_02/measure.html"},
         "learning_outcome_refs": ["co-01"]},
    ]
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    eval_dir = course_dir / "retrieval_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    gold = {
        "schema_version": "1.1",
        "course_slug": slug,
        "chunkset": {
            "kind": "corpus",
            "chunks_path": "corpus/chunks.jsonl",
            "chunks_sha256": sha256_file(chunks_path),
        },
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": False,
        "questions": [
            {
                "question_id": "gq-syn-union-101-0001",
                "question_text": "What does a vector store index?",
                "question_type": "factual_recall",
                "expected_citation_population": "source",
                "expected_key_points": [
                    _KP_TEXT,
                    "It is used for nearest-neighbor retrieval",
                ],
                "relevant_passages": [
                    {"chunk_id": "src1", "relevance": "primary",
                     "anchor": {"item_path": "textbook.html",
                                "text_quote": _KP_TEXT,
                                "content_sha256": _csha(_CHUNK_SOURCE)}},
                ],
                "authoring": {"method": "manual", "author": "@t",
                              "reviewed_by": "PENDING_REVIEW", "status": "seed"},
            },
            {
                "question_id": "gq-syn-union-101-0002",
                "question_text": "Describe vector stores and how quality is measured.",
                "question_type": "multi_part",
                "expected_citation_population": "both",
                "parts": [
                    {"part_id": "a",
                     "part_text": "A vector store is a database that indexes high-dimensional embedding vectors",
                     "covered": True, "relevant_passage_refs": ["src1"]},
                    {"part_id": "b",
                     "part_text": "the runtime latency budget of the index server",
                     "covered": False,
                     "absence_note": "The corpus does not discuss latency budgets; "
                                     "dry-run confirmed the absence."},
                ],
                "relevant_passages": [
                    {"chunk_id": "src1", "relevance": "primary",
                     "anchor": {"item_path": "textbook.html", "text_quote": _KP_TEXT}},
                    {"chunk_id": "crs1", "relevance": "supporting",
                     "anchor": {"item_path": "week_02/measure.html",
                                "text_quote": "Retrieval quality is commonly measured with recall at k"}},
                ],
                "authoring": {"method": "manual", "author": "@t",
                              "reviewed_by": "PENDING_REVIEW", "status": "seed"},
            },
        ],
    }
    (eval_dir / "gold_set.json").write_text(json.dumps(gold), encoding="utf-8")
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug, course_dir


# ===========================================================================
# Fake pipeline
# ===========================================================================

class _FakeCite:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.anchor_status = "resolved_exact"
        self.page_label = "P"
        self.text_quote = "q"

    def to_dict(self):
        return {"chunk_id": self.chunk_id, "anchor_status": self.anchor_status,
                "page_label": self.page_label, "text_quote": self.text_quote}


class _FakeAnswer:
    def __init__(self, *, answer_text, citations):
        self.status = "answered"
        self.answer_text = answer_text
        self.citations = citations
        self.groundedness = None
        self.latency_ms = 5.0
        self.model_id = "fake"
        self.prompt_version = "ws3.v2"
        self.confidence = {"policy_version": "ws3.v0-uncalibrated"}


def _answer_fn(_repo, _slug, query, **kwargs):
    if query.startswith("What does a vector store index"):
        # Cites the source chunk; answer contains both key points + a disclaimer
        # about the unrelated part (irrelevant to q1, harmless).
        return _FakeAnswer(
            answer_text=(
                "A vector store is a database that indexes high-dimensional "
                "embedding vectors. It is used for nearest-neighbor retrieval."
            ),
            citations=[_FakeCite("src1")],
        )
    # multi_part question: cites both populations; answers part a + flags part b.
    return _FakeAnswer(
        answer_text=(
            "A vector store is a database that indexes high-dimensional embedding "
            "vectors. The course does not discuss the latency budget of the index "
            "server anywhere."
        ),
        citations=[_FakeCite("src1"), _FakeCite("crs1")],
    )


# ===========================================================================
# Tests
# ===========================================================================

def test_eval_schema_bumped_to_1_1():
    # 1.2 added the groundedness scorer-v2 surface; 1.3 (additive) added the
    # refusal-safety axis: answered-probe groundedness rolled into
    # headline.refusal + a top-level probe_results key. 1.4 (additive) added
    # headline.citation_precision_preadd + per-row cited_chunk_ids/cited_pages.
    # The report schema_version moves with it.
    assert EVAL_SCHEMA_VERSION == "1.4"


def test_gold_pin_block_carries_section4_fields(v1_1_course):
    repo_root, slug, _ = v1_1_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_answer_fn,
        with_groundedness=False, write=False,
    )
    pin = report["gold"]
    assert pin["schema_version"] == "1.1"
    assert pin["question_count"] == 2
    assert pin["authored_at"] == "2026-06-11T00:00:00Z"
    # v1.0 fields preserved.
    assert pin["chunks_sha256"]
    assert pin["chunkset_kind"] == "corpus"


def test_key_point_coverage_per_question_and_headline(v1_1_course):
    repo_root, slug, _ = v1_1_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_answer_fn,
        with_groundedness=False, write=False,
    )
    q1 = report["questions"][0]
    assert "key_point_coverage" in q1
    kp = q1["key_point_coverage"]
    assert kp["total"] == 2
    assert kp["covered"] == 2  # both points present in the answer
    # headline rollup: only q1 carries expected_key_points → 1 question scored.
    hk = report["headline"]["key_point_coverage"]
    assert hk["questions_scored"] == 1
    assert hk["total_key_points"] == 2
    assert hk["covered_key_points"] == 2
    assert hk["coverage_rate"] == 1.0


def test_part_coverage_per_question_and_flagging(v1_1_course):
    repo_root, slug, _ = v1_1_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_answer_fn,
        with_groundedness=False, write=False,
    )
    q2 = report["questions"][1]
    assert "part_coverage" in q2
    pc = q2["part_coverage"]
    assert pc["n_covered_parts"] == 1
    assert pc["n_answered"] == 1            # part a answered
    assert pc["n_uncovered_parts"] == 1
    assert pc["n_correctly_flagged"] == 1   # part b absence acknowledged
    hp = report["headline"]["part_coverage"]
    assert hp["questions_scored"] == 1
    assert hp["answered_rate"] == 1.0
    assert hp["correctly_flagged_rate"] == 1.0
    assert "_diagnostic" in hp


def test_population_breakdown_and_expected_satisfaction(v1_1_course):
    repo_root, slug, _ = v1_1_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_answer_fn,
        with_groundedness=False, write=False,
    )
    q1 = report["questions"][0]
    cp1 = q1["citation_population"]
    assert cp1["cited_source"] == 1
    assert cp1["cited_course"] == 0
    assert cp1["expected_population"] == "source"
    assert cp1["expected_satisfied"] is True

    q2 = report["questions"][1]
    cp2 = q2["citation_population"]
    assert cp2["cited_source"] == 1
    assert cp2["cited_course"] == 1
    assert cp2["expected_population"] == "both"
    assert cp2["expected_satisfied"] is True

    # headline per-population citation precision + satisfaction rollup
    by_pop = report["headline"]["citation_precision_by_population"]
    assert by_pop["emitted_source"] == 2  # src1 cited in both questions
    assert by_pop["emitted_course"] == 1  # crs1 cited once
    # src1 is gold-relevant in both; crs1 is gold-relevant (supporting) in q2.
    assert by_pop["source"] == 1.0
    assert by_pop["course"] == 1.0
    sat = report["headline"]["expected_population_satisfaction"]
    assert sat["checked"] == 2
    assert sat["satisfied"] == 2
    assert sat["rate"] == 1.0


def test_report_json_round_trips(v1_1_course):
    repo_root, slug, course_dir = v1_1_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_answer_fn,
        with_groundedness=False, write=True,
    )
    written = Path(report["_written"]["report_path"])
    doc = json.loads(written.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "1.4"
    assert doc["gold"]["question_count"] == 2
    assert "key_point_coverage" in doc["questions"][0]
    assert "part_coverage" in doc["questions"][1]
    assert "citation_population" in doc["questions"][0]
    # schema-1.4 additive per-row ids persisted through the JSON round-trip.
    assert "cited_chunk_ids" in doc["questions"][0]
    assert "cited_pages" in doc["questions"][0]
    # headline P4 blocks present
    for key in ("citation_precision_by_population", "expected_population_satisfaction",
                "key_point_coverage", "part_coverage"):
        assert key in doc["headline"]
    # schema-1.4 additive headline: pre-add precision present, == precision when
    # NLI-ADD is off (this fixture run).
    assert "citation_precision_preadd" in doc["headline"]
    assert (
        doc["headline"]["citation_precision_preadd"]
        == doc["headline"]["citation_precision"]
    )


# ===========================================================================
# Part-coverage aggregation regression: real-shape gold (terse-prompt
# part_text + per-part key_points) must NOT score answered_parts==0; and a
# zero-multipart gold must report rates honestly (null/0, never a fake 0.0).
# ===========================================================================

_MP_CHUNK_A = "isolate the absolute value then split into two cases for x"
_MP_CHUNK_B = "draw the V-shaped graph opening upward from the vertex"


def _write_multipart_course(tmp_path, monkeypatch, *, questions):
    """Build a hermetic two-chunk course carrying ``questions`` and return
    (repo_root, slug). Minimal scaffolding mirroring :func:`v1_1_course`. The
    two chunk bodies carry the answer content for the two multi_part parts."""
    slug = "syn-mp-202"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        {"id": "c1", "text": f"To solve an absolute value equation, {_MP_CHUNK_A}.",
         "source": {"item_path": "textbook.html",
                    "source_references": [{"block_id": "b1", "pdf_pages": [1]}]},
         "learning_outcome_refs": ["to-01"]},
        {"id": "c2", "text": f"To graph it, {_MP_CHUNK_B}.",
         "source": {"item_path": "textbook.html",
                    "source_references": [{"block_id": "b2", "pdf_pages": [2]}]},
         "learning_outcome_refs": ["to-01"]},
    ]
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    eval_dir = course_dir / "retrieval_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    gold = {
        "schema_version": "1.1",
        "course_slug": slug,
        "chunkset": {"kind": "corpus", "chunks_path": "corpus/chunks.jsonl",
                     "chunks_sha256": sha256_file(chunks_path)},
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": False,
        "questions": questions,
    }
    (eval_dir / "gold_set.json").write_text(json.dumps(gold), encoding="utf-8")
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug


def test_part_coverage_live_shape_terse_prompt_with_key_points(tmp_path, monkeypatch):
    """Regression for the inert part-coverage scorer: real gold authors a terse
    sub-question PROMPT as part_text and the answer-content claims as per-part
    key_points. The answer addresses part a and silently drops part b — the
    scorer must report answered_parts==1 (NOT 0)."""
    q = {
        "question_id": "gq-syn-mp-202-0001",
        "question_text": "Solve the absolute value equation and graph the shaded ray.",
        "question_type": "multi_part",
        "expected_citation_population": "any",
        "parts": [
            {"part_id": "a", "part_text": "Solve for x.", "covered": True,
             "relevant_passage_refs": ["c1"]},
            {"part_id": "b", "part_text": "Graph the shaded ray.", "covered": True,
             "relevant_passage_refs": ["c2"]},
        ],
        "relevant_passages": [
            {"chunk_id": "c1", "relevance": "primary",
             "anchor": {"item_path": "textbook.html", "text_quote": _MP_CHUNK_A}},
            {"chunk_id": "c2", "relevance": "supporting",
             "anchor": {"item_path": "textbook.html", "text_quote": _MP_CHUNK_B}},
        ],
        "authoring": {"method": "manual", "author": "@t",
                      "reviewed_by": "PENDING_REVIEW", "status": "seed"},
    }
    repo_root, slug = _write_multipart_course(tmp_path, monkeypatch, questions=[q])

    def _fn(_repo, _slug, _query, **_kw):
        # Answer addresses part a (chunk c1 body) and silently drops part b.
        return _FakeAnswer(
            answer_text=f"To solve, {_MP_CHUNK_A}.",
            citations=[_FakeCite("c1")],
        )

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=False, write=False,
    )
    pc = report["questions"][0]["part_coverage"]
    assert pc["n_covered_parts"] == 2
    assert pc["n_answered"] == 1          # part b silently dropped — the bug we fixed
    hp = report["headline"]["part_coverage"]
    assert hp["questions_scored"] == 1
    assert hp["covered_parts"] == 2
    assert hp["answered_parts"] == 1
    assert hp["answered_rate"] == 0.5     # honest: 1 of 2 covered parts answered


def test_part_coverage_zero_multipart_rates_honest(tmp_path, monkeypatch):
    """No multi_part questions → covered/answered/uncovered all 0, both rates
    null (denominator 0), never a fabricated 0.0."""
    q = {
        "question_id": "gq-syn-mp-202-0002",
        "question_text": "What does isolating do?",
        "question_type": "factual_recall",
        "expected_citation_population": "any",
        "expected_key_points": ["isolate the absolute value", "split into two cases for x"],
        "relevant_passages": [
            {"chunk_id": "c1", "relevance": "primary",
             "anchor": {"item_path": "textbook.html",
                        "text_quote": "isolate the absolute value then split into two cases for x"}},
        ],
        "authoring": {"method": "manual", "author": "@t",
                      "reviewed_by": "PENDING_REVIEW", "status": "seed"},
    }
    repo_root, slug = _write_multipart_course(tmp_path, monkeypatch, questions=[q])

    def _fn(_repo, _slug, _query, **_kw):
        return _FakeAnswer(
            answer_text="To solve, isolate the absolute value first.",
            citations=[_FakeCite("c1")],
        )

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=False, write=False,
    )
    hp = report["headline"]["part_coverage"]
    assert hp["questions_scored"] == 0
    assert hp["covered_parts"] == 0
    assert hp["answered_parts"] == 0
    assert hp["uncovered_parts"] == 0
    assert hp["correctly_flagged_parts"] == 0
    assert hp["answered_rate"] is None          # denominator 0 → null, not 0.0
    assert hp["correctly_flagged_rate"] is None
    # no per-question part_coverage block on a non-multi_part question
    assert "part_coverage" not in report["questions"][0]
