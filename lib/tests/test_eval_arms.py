"""Tests for the three-arm eval scorecard (``lib.retrieval.eval_arms``).

CI-safe: a fake local-model client + fake retriever (and the injected
``answer_fn`` for the grounded arm) are used throughout, so no network / no
model / no real LLM is ever touched. The deterministic key-point machinery
(``score_key_point_coverage``) is the real one — that's the whole point of the
honest cross-arm comparison.

Two gold-set strategies:
  * synthetic in-memory gold (``gold=`` injection) for the per-arm scoring
    tests — lets us pin expected_key_points + relevant_passages precisely;
  * the real mini-course fixture (copied into a tmp LibV2 root) for the
    scorecard-assembly tests that exercise ``_load_verified_gold``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lib.retrieval.eval_arms import (
    ALL_ARMS,
    ARM_BASE,
    ARM_GROUNDED,
    ARM_RETRIEVAL,
    SCORECARD_SCHEMA_VERSION,
    format_scorecard_table,
    run_base_arm,
    run_retrieval_arm,
    run_scorecard,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)


# ===========================================================================
# Synthetic gold + fakes
# ===========================================================================

def _synthetic_gold():
    """A 2-question gold carrying expected_key_points + a primary passage."""
    return {
        "schema_version": "1.1",
        "questions": [
            {
                "question_id": "gq-0001",
                "question_text": "What does a vector store index?",
                "question_type": "short_answer",
                "expected_key_points": [
                    "high dimensional embedding vectors",
                    "similarity search",
                ],
                "relevant_passages": [
                    {"chunk_id": "chunk_a", "relevance": "primary"},
                ],
            },
            {
                "question_id": "gq-0002",
                "question_text": "How is retrieval quality measured?",
                "question_type": "short_answer",
                "expected_key_points": ["recall at k", "mean reciprocal rank"],
                "relevant_passages": [
                    {"chunk_id": "chunk_b", "relevance": "primary"},
                ],
            },
        ],
    }


class _FakeClient:
    """Duck-types ``chat_completion`` + ``model``. Maps question text → answer
    text; an entry of ``RAISE`` raises to exercise per-question isolation."""

    def __init__(self, answers, *, model="fake-qwen-7b"):
        self._answers = answers
        self.model = model
        self.calls = []

    def chat_completion(self, messages, *, max_tokens=800, temperature=0.4):
        user = messages[-1]["content"]
        self.calls.append(user)
        out = self._answers.get(user)
        if out == "RAISE":
            raise RuntimeError("simulated backend hiccup")
        return out if out is not None else ""


class _Result:
    """Duck-types the live retrieval-result shape (``chunk_id`` + ``text``)."""

    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text


class _RecordingCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


# ===========================================================================
# BASE arm
# ===========================================================================

def test_base_arm_scores_key_points_with_shared_scorer():
    gold = _synthetic_gold()
    client = _FakeClient(
        {
            # First answer covers BOTH key points verbatim → 2/2.
            "What does a vector store index?": (
                "A vector store indexes high dimensional embedding vectors and "
                "supports similarity search over them."
            ),
            # Second answer covers ONE key point → 1/2.
            "How is retrieval quality measured?": (
                "Retrieval quality is often measured by recall at k."
            ),
        }
    )
    result = run_base_arm(Path("/unused"), "course-x", client=client, gold=gold)

    assert result["arm"] == ARM_BASE
    assert result["retrieval"] is False
    assert result["refusal_scaffolding"] is False
    assert result["citations"] == "n/a"
    assert result["answers_everything"] is True
    assert result["declined"] == 0
    assert result["answered"] == 2
    assert result["errored"] == 0
    # 2 + 1 of 4 total key points covered.
    kp = result["key_point_coverage"]
    assert kp["total_key_points"] == 4
    assert kp["covered_key_points"] == 3
    assert kp["coverage_rate"] == pytest.approx(0.75)
    # No system grounding / passages were injected — base prompt only.
    assert all("vector store indexes" not in c for c in client.calls[1:2])


def test_base_arm_error_isolation_does_not_kill_arm():
    gold = _synthetic_gold()
    client = _FakeClient(
        {
            "What does a vector store index?": "RAISE",  # one question explodes
            "How is retrieval quality measured?": (
                "Measured by recall at k and mean reciprocal rank."
            ),
        }
    )
    result = run_base_arm(Path("/unused"), "course-x", client=client, gold=gold)
    # The arm still completed: one errored, one answered + scored.
    assert result["errored"] == 1
    assert result["answered"] == 1
    rows = {r["question_id"]: r for r in result["questions"]}
    assert rows["gq-0001"]["answered"] is False
    assert "error" in rows["gq-0001"]
    assert rows["gq-0002"]["answered"] is True
    # Only the surviving question's key points are in the denominator.
    assert result["key_point_coverage"]["total_key_points"] == 2
    assert result["key_point_coverage"]["covered_key_points"] == 2


def test_base_arm_emits_decision_capture_per_question():
    gold = _synthetic_gold()
    client = _FakeClient(
        {
            "What does a vector store index?": "embedding vectors",
            "How is retrieval quality measured?": "recall at k",
        }
    )
    cap = _RecordingCapture()
    run_base_arm(
        Path("/unused"), "course-x", client=client, gold=gold, capture=cap
    )
    base_events = [
        e for e in cap.events if e["decision_type"] == "base_model_eval_call"
    ]
    # One capture per scored question.
    assert len(base_events) == 2
    for e in base_events:
        # Rationale interpolates dynamic per-call signals (not boilerplate).
        assert len(e["rationale"]) >= 20
        assert "course-x" in e["rationale"]
        assert "fake-qwen-7b" in e["rationale"]
    # Each question id appears in exactly one decision string.
    decided = " ".join(e["decision"] for e in base_events)
    assert "gq-0001" in decided and "gq-0002" in decided


# ===========================================================================
# RETRIEVAL arm
# ===========================================================================

def _retrieve_fn_factory(by_query):
    def _fn(_root, _slug, query, *, engine, limit):
        return by_query.get(query, [])[:limit]

    return _fn


def test_retrieval_arm_extractive_ceiling_and_hits():
    gold = _synthetic_gold()
    by_query = {
        # Q1: primary chunk_a is top-1; passages cover BOTH key points.
        "What does a vector store index?": [
            _Result(
                "chunk_a",
                "A vector store indexes high dimensional embedding vectors "
                "for similarity search.",
            ),
            _Result("chunk_z", "unrelated passage"),
        ],
        # Q2: primary chunk_b is present but NOT top-1; passages cover ONE
        # key point.
        "How is retrieval quality measured?": [
            _Result("chunk_y", "Some passage about recall at k only."),
            _Result("chunk_b", "Mean reciprocal rank is one measure."),
        ],
    }
    result = run_retrieval_arm(
        Path("/unused"),
        "course-x",
        engine="semantic",
        limit=8,
        retrieve_fn=_retrieve_fn_factory(by_query),
        gold=gold,
    )
    assert result["arm"] == ARM_RETRIEVAL
    assert result["retrieval"] is True
    assert result["model_id"] is None  # no LLM

    # Extractive ceiling: Q1 covers 2/2, Q2 covers 2/2 (recall at k + MRR both
    # appear across the concatenated passages) → 4/4.
    kp = result["key_point_coverage"]
    assert kp["total_key_points"] == 4
    assert kp["covered_key_points"] == 4

    hit = result["primary_relevant_hit"]
    assert hit["questions"] == 2
    # Both primaries present in top-k.
    assert hit["hit_at_k"] == 2
    assert hit["hit_at_k_rate"] == pytest.approx(1.0)
    # Only Q1's primary is top-1.
    assert hit["hit_top1"] == 1
    assert hit["hit_top1_rate"] == pytest.approx(0.5)


def test_retrieval_arm_error_isolation():
    gold = _synthetic_gold()

    def _fn(_root, _slug, query, *, engine, limit):
        if query == "What does a vector store index?":
            raise RuntimeError("index unavailable for this query")
        return [_Result("chunk_b", "recall at k and mean reciprocal rank")]

    result = run_retrieval_arm(
        Path("/unused"), "course-x", retrieve_fn=_fn, gold=gold
    )
    rows = {r["question_id"]: r for r in result["questions"]}
    assert "error" in rows["gq-0001"]
    assert rows["gq-0001"]["hit_at_k"] is False
    # The other question still scored.
    assert rows["gq-0002"]["hit_at_k"] is True


# ===========================================================================
# Scorecard assembly — uses the real fixture for _load_verified_gold
# ===========================================================================

@pytest.fixture
def libv2_course(tmp_path, monkeypatch):
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    shutil.copytree(FIXTURE, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug, course_dir


# --- a fake grounded pipeline, shaped like answer_course_question -----------

class _FakeCitation:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.anchor_status = "resolved_exact"
        self.page_label = "P"
        self.text_quote = "q"

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "anchor_status": self.anchor_status,
            "page_label": self.page_label,
            "text_quote": self.text_quote,
        }


class _FakeAnswer:
    def __init__(self, *, status, answer_text, citations):
        self.status = status
        self.answer_text = answer_text
        self.citations = citations
        self.groundedness = None
        self.latency_ms = 10.0
        self.model_id = "fake-grounded-model"
        self.prompt_version = "ws3.v1"
        self.confidence = {"policy_version": "ws3.v0"}


def _fake_grounded_fn():
    answers = {
        "What does a vector store index?": "mini_alpha_chunk_001",
        "How is retrieval quality commonly measured?": "mini_alpha_chunk_003",
        "Where does the course cover chunking strategies?": "mini_beta_chunk_005",
    }

    def _fn(repo_root, course_slug, query, *, engine="lexical", limit=8,
            client=None, refusal_policy=None, with_groundedness=False,
            capture=None, **kwargs):
        cid = answers.get(query)
        if cid is None:
            return _FakeAnswer(
                status="refused_low_confidence", answer_text=None, citations=[]
            )
        return _FakeAnswer(
            status="answered",
            answer_text=f"Answer about {cid}.",
            citations=[_FakeCitation(cid)],
        )

    return _fn


def _fake_retrieve_for_fixture():
    # Map fixture gold question text → a top-k list whose top-1 is the gold
    # primary chunk (so hit_top1_rate == 1.0).
    by_query = {
        "What does a vector store index?": [_Result("mini_alpha_chunk_001", "x")],
        "How is retrieval quality commonly measured?": [
            _Result("mini_alpha_chunk_003", "y")
        ],
        "Where does the course cover chunking strategies?": [
            _Result("mini_beta_chunk_005", "z")
        ],
    }
    return _retrieve_fn_factory(by_query)


def test_scorecard_default_is_grounded_only(libv2_course):
    """Default arms=(grounded,) — only the grounded arm runs; the grounded
    report is still written (staleness test contract)."""
    repo_root, slug, course_dir = libv2_course
    scorecard = run_scorecard(
        repo_root, slug, engine="lexical", write=True,
        grounded_kwargs={"answer_fn": _fake_grounded_fn()},
    )
    assert scorecard["schema_version"] == SCORECARD_SCHEMA_VERSION
    assert scorecard["arms_run"] == [ARM_GROUNDED]
    assert set(scorecard["arms"]) == {ARM_GROUNDED}
    # Grounded arm still wrote its own grounded_answer_eval report.
    eval_dir = course_dir / "retrieval_eval"
    assert list(eval_dir.glob("grounded_answer_eval_*.json"))
    # Scorecard file also written.
    assert list(eval_dir.glob("eval_scorecard_*.json"))
    # Comparison carries only the grounded column.
    assert set(scorecard["comparison"]) == {ARM_GROUNDED}


def test_scorecard_three_arms_side_by_side(libv2_course):
    repo_root, slug, course_dir = libv2_course
    base_client = _FakeClient(
        {
            "What does a vector store index?": "an answer",
            "How is retrieval quality commonly measured?": "another",
            "Where does the course cover chunking strategies?": "third",
        },
        model="fake-base-7b",
    )
    scorecard = run_scorecard(
        repo_root, slug, engine="lexical",
        arms=[ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED],
        base_client=base_client,
        retrieve_fn=_fake_retrieve_for_fixture(),
        grounded_kwargs={"answer_fn": _fake_grounded_fn()},
        write=True,
    )
    assert scorecard["arms_run"] == [ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED]
    # Per-arm blocks all present + carry their identity.
    assert scorecard["arms"][ARM_BASE]["model_id"] == "fake-base-7b"
    assert scorecard["arms"][ARM_RETRIEVAL]["retrieval"] is True
    assert scorecard["arms"][ARM_GROUNDED]["model_id"] == "fake-grounded-model"

    # Comparison block: all three columns on the shared axes.
    comp = scorecard["comparison"]
    assert set(comp) == {ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED}
    for arm in (ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED):
        assert "key_point_coverage_rate" in comp[arm]
        assert "latency_ms" in comp[arm]
    # Base answers everything → declined 0; retrieval has no answered notion.
    assert comp[ARM_BASE]["declined"] == 0
    assert comp[ARM_RETRIEVAL]["answered"] is None
    # Retrieval hit@k surfaced in its comparison column (fixture: all top-1).
    assert comp[ARM_RETRIEVAL]["primary_relevant_hit_at_k_rate"] == pytest.approx(
        1.0
    )

    # Grounded-only axes stay inside the grounded ARM block, NOT in comparison.
    assert "refusal" in scorecard["arms"][ARM_GROUNDED]["headline"]
    assert "refusal" not in comp[ARM_GROUNDED]

    # Aligned table renders all three columns.
    table = format_scorecard_table(scorecard)
    assert "BASE (qwen)" in table and "RETRIEVAL" in table and "GROUNDED" in table
    assert "key_point_coverage" in table


def test_scorecard_two_arms_base_and_retrieval(libv2_course):
    """A 2-arm scorecard (no grounded) assembles cleanly + writes no grounded
    report."""
    repo_root, slug, course_dir = libv2_course
    base_client = _FakeClient(
        {
            "What does a vector store index?": "a",
            "How is retrieval quality commonly measured?": "b",
            "Where does the course cover chunking strategies?": "c",
        }
    )
    scorecard = run_scorecard(
        repo_root, slug, engine="lexical",
        arms=[ARM_BASE, ARM_RETRIEVAL],
        base_client=base_client,
        retrieve_fn=_fake_retrieve_for_fixture(),
        write=True,
    )
    assert scorecard["arms_run"] == [ARM_BASE, ARM_RETRIEVAL]
    assert ARM_GROUNDED not in scorecard["arms"]
    # No grounded arm → no grounded_answer_eval report written by this run.
    eval_dir = course_dir / "retrieval_eval"
    assert not list(eval_dir.glob("grounded_answer_eval_*.json"))
    # Scorecard still written.
    assert list(eval_dir.glob("eval_scorecard_*.json"))


def test_scorecard_single_base_arm(libv2_course):
    repo_root, slug, _ = libv2_course
    base_client = _FakeClient({})  # all questions → "" (answered, 0 coverage)
    scorecard = run_scorecard(
        repo_root, slug, arms=[ARM_BASE], base_client=base_client, write=False
    )
    assert scorecard["arms_run"] == [ARM_BASE]
    assert set(scorecard["comparison"]) == {ARM_BASE}


def test_scorecard_unknown_arm_raises(libv2_course):
    repo_root, slug, _ = libv2_course
    with pytest.raises(ValueError) as exc:
        run_scorecard(repo_root, slug, arms=["base", "bogus"], write=False)
    assert "bogus" in str(exc.value)


def test_scorecard_critical_gold_issue_raises(libv2_course):
    """A tampered (sha-mismatched) gold set fails closed BEFORE any arm runs."""
    repo_root, slug, course_dir = libv2_course
    chunks = course_dir / "dart_chunks" / "chunks.jsonl"
    chunks.write_text(chunks.read_text() + '\n{"id":"x","text":"tamper"}\n')
    with pytest.raises(RuntimeError) as exc:
        run_scorecard(
            repo_root, slug, arms=[ARM_BASE],
            base_client=_FakeClient({}), write=False,
        )
    assert "critical" in str(exc.value).lower()


def test_all_arms_constant():
    assert ALL_ARMS == (ARM_BASE, ARM_RETRIEVAL, ARM_GROUNDED)
