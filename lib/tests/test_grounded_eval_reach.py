"""Tests for E2 reachability: prior_turns threading + the library-wide slice.

Two structurally-unreachable paths get harness coverage here:
  * ``prior_turns`` gold items now drive ``answer_course_question(prior_turns=)``
    (the ED4ALL_ANSWER_MULTITURN seam). A spy pipeline proves the flat text
    sequence reaches the call, and the report's ``headline.multiturn`` counts it.
  * a library-wide slice calls ``answer_library_question`` over a cross-course
    set (skipped cleanly on a single course).

CI-safe: pipelines are injected (no model / network). The gold set is patched in
a tmp copy to carry prior_turns (the fixture is single-turn v1.0).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lib.retrieval.grounded_eval import run_grounded_eval
from lib.retrieval.grounded_eval_library import (
    LIBRARY_SLICE_SCHEMA_VERSION,
    run_library_eval,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)

_GOLD_MAP = {
    "What does a vector store index?": "mini_alpha_chunk_001",
    "How is retrieval quality commonly measured?": "mini_alpha_chunk_003",
    "Where does the course cover chunking strategies?": "mini_beta_chunk_005",
}

_MULTITURN_QUERY = "What does a vector store index?"
_PRIOR_TURNS = [
    {"role": "user", "text": "Tell me about vector stores."},
    {"role": "assistant", "text": "A vector store holds embeddings."},
]


class _FakeCitation:
    def __init__(self, chunk_id, course_slug=None):
        self.chunk_id = chunk_id
        self.anchor_status = "resolved_exact"
        self.page_label = "P"
        self.text_quote = "q"
        self.course_slug = course_slug

    def to_dict(self):
        return {"chunk_id": self.chunk_id, "anchor_status": self.anchor_status,
                "page_label": self.page_label, "text_quote": self.text_quote,
                "course_slug": self.course_slug}


class _FakeAnswer:
    def __init__(self, status, citations):
        self.status = status
        self.answer_text = "A." if citations else None
        self.citations = citations
        self.groundedness = None
        self.latency_ms = 1.0
        self.model_id = "fake"
        self.prompt_version = "v"
        self.confidence = {"policy_version": "p"}


@pytest.fixture
def multiturn_course(tmp_path, monkeypatch):
    """Fixture copy whose first gold question carries prior_turns (v1.2)."""
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    shutil.copytree(FIXTURE, course_dir)
    gold_path = course_dir / "retrieval_eval" / "gold_set.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["schema_version"] = "1.2"
    for q in gold["questions"]:
        if q.get("question_text") == _MULTITURN_QUERY:
            q["prior_turns"] = _PRIOR_TURNS
            q["learner_intent"] = "multi_turn_followup"
            q["expected_behavior"] = "answer_grounded"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug, course_dir


# ===========================================================================
# prior_turns threading
# ===========================================================================

def test_prior_turns_reaches_pipeline(multiturn_course):
    repo_root, slug, _ = multiturn_course
    seen = {}

    def _spy(repo_root, course_slug, query, **kwargs):
        seen[query] = kwargs.get("prior_turns", "__ABSENT__")
        cid = _GOLD_MAP.get(query)
        if cid is None:
            return _FakeAnswer("refused_low_confidence", [])
        return _FakeAnswer("answered", [_FakeCitation(cid)])

    run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_spy,
        with_groundedness=False, write=False,
    )
    # The multiturn question passed the flat text sequence of its prior turns.
    assert seen[_MULTITURN_QUERY] == [
        "Tell me about vector stores.",
        "A vector store holds embeddings.",
    ]
    # Single-turn questions passed prior_turns=None (byte-identical off path).
    other = [q for q in _GOLD_MAP if q != _MULTITURN_QUERY][0]
    assert seen[other] is None


def test_headline_multiturn_counts(multiturn_course):
    repo_root, slug, _ = multiturn_course

    def _fn(repo_root, course_slug, query, **kwargs):
        cid = _GOLD_MAP.get(query)
        if cid is None:
            return _FakeAnswer("refused_low_confidence", [])
        return _FakeAnswer("answered", [_FakeCitation(cid)])

    report = run_grounded_eval(
        repo_root, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=False, write=False,
    )
    mt = report["headline"]["multiturn"]
    assert mt["gold_with_prior_turns"] == 1
    assert mt["answered_with_prior_turns"] == 1
    assert mt["prior_turns_total"] == 2
    assert mt["reachable"] is True


def test_headline_multiturn_zero_on_single_turn_fixture(tmp_path, monkeypatch):
    """The unmodified v1.0 fixture (no prior_turns) → all-zero multiturn block."""
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    shutil.copytree(FIXTURE, libv2_root / "courses" / slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    def _fn(repo_root, course_slug, query, **kwargs):
        cid = _GOLD_MAP.get(query)
        return _FakeAnswer("answered", [_FakeCitation(cid)]) if cid else \
            _FakeAnswer("refused_low_confidence", [])

    report = run_grounded_eval(
        tmp_path, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=False, write=False,
    )
    mt = report["headline"]["multiturn"]
    assert mt["gold_with_prior_turns"] == 0
    assert mt["reachable"] is False


# ===========================================================================
# library-wide slice (run_library_eval unit)
# ===========================================================================

_LIB_QUESTIONS = [
    {"question_id": "lq1", "question_text": "cross-course question one"},
    {"question_id": "lq2", "question_text": "cross-course question two"},
]


def test_library_eval_skips_on_single_course(tmp_path):
    section = run_library_eval(
        tmp_path, "course-a", _LIB_QUESTIONS,
        course_slugs=["course-a"],  # single course → nothing to union
        answer_fn=lambda *a, **k: _FakeAnswer("answered", []),
    )
    assert section["skipped"] is True
    assert section["reason"] == "single_course"
    assert section["resolved_course_count"] == 1
    assert section["schema_version"] == LIBRARY_SLICE_SCHEMA_VERSION


def test_library_eval_runs_over_multi_course(tmp_path):
    def _lib_fn(repo_root, home_slug, query, **kwargs):
        # Assert the library-wide contract: called with the resolved course set
        # in library-wide mode.
        assert kwargs.get("library_wide") is True
        assert kwargs.get("course_slugs") == ["course-a", "course-b"]
        if query == "cross-course question one":
            # A genuinely cross-course answer (citations from 2 courses).
            return _FakeAnswer("answered", [
                _FakeCitation("c1", course_slug="course-a"),
                _FakeCitation("c2", course_slug="course-b"),
            ])
        return _FakeAnswer("refused_not_in_course", [])

    section = run_library_eval(
        tmp_path, "course-a", _LIB_QUESTIONS,
        course_slugs=["course-a", "course-b"], answer_fn=_lib_fn,
    )
    assert section["skipped"] is False
    assert section["n_questions"] == 2
    assert section["answered_count"] == 1
    assert section["refused_count"] == 1
    assert section["answer_rate"] == pytest.approx(0.5)
    # One answer drew citations from >1 course.
    assert section["cross_course_answer_count"] == 1
    assert section["cross_course_rate"] == pytest.approx(1.0)
    rows = {r["question_id"]: r for r in section["questions"]}
    assert rows["lq1"]["citation_courses"] == ["course-a", "course-b"]


def test_library_eval_pipeline_unavailable_skips(tmp_path, monkeypatch):
    """No injected answer_fn + unimportable pipeline → skip, never crash."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "lib.retrieval.library_wide" or name.endswith("library_wide"):
            raise ImportError("simulated: library_wide absent")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    section = run_library_eval(
        tmp_path, "course-a", _LIB_QUESTIONS, course_slugs=["course-a", "course-b"],
    )
    assert section["skipped"] is True
    assert section["reason"] == "pipeline_unavailable"


# ===========================================================================
# library-wide slice wired into run_grounded_eval
# ===========================================================================

def test_library_wide_section_absent_by_default(tmp_path, monkeypatch):
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    shutil.copytree(FIXTURE, libv2_root / "courses" / slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    def _fn(repo_root, course_slug, query, **kwargs):
        cid = _GOLD_MAP.get(query)
        return _FakeAnswer("answered", [_FakeCitation(cid)]) if cid else \
            _FakeAnswer("refused_low_confidence", [])

    report = run_grounded_eval(
        tmp_path, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=False, write=False,
    )
    # No library_questions passed → the optional section is ABSENT (additive).
    assert "library_wide" not in report


def test_library_wide_section_present_when_requested(tmp_path, monkeypatch):
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    shutil.copytree(FIXTURE, libv2_root / "courses" / slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    def _fn(repo_root, course_slug, query, **kwargs):
        cid = _GOLD_MAP.get(query)
        return _FakeAnswer("answered", [_FakeCitation(cid)]) if cid else \
            _FakeAnswer("refused_low_confidence", [])

    def _lib_fn(repo_root, home_slug, query, **kwargs):
        return _FakeAnswer("answered", [_FakeCitation("c1", course_slug="course-b")])

    report = run_grounded_eval(
        tmp_path, slug, engine="lexical", answer_fn=_fn,
        with_groundedness=False, write=False,
        library_questions=_LIB_QUESTIONS,
        library_answer_fn=_lib_fn,
        library_course_slugs=[slug, "course-b"],
    )
    assert "library_wide" in report
    assert report["library_wide"]["skipped"] is False
    assert report["library_wide"]["n_questions"] == 2
