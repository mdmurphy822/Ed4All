"""Pipeline-level tests for the pre-retrieval augmentation arms (W5.1/2/6/7).

Deterministic + CI-safe: no model, no server, no network, no torch. Retrieval
runs the REAL lexical BM25 path over the committed mini-course fixture (reused
via the ``mini_libv2`` fixture + ``FakeAnswerClient`` / ``SpyCapture`` doubles
from the existing pipeline test); the LLM call site is the fake client.

Focus: (a) every arm is byte-identical-when-off (no new capture rows, same
answer); (b) each arm emits its DecisionCapture row when on; (c) the merge arms
never turn an answered result into a refusal (verdict-safety).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pytest

import lib.retrieval.grounded_answer as ga
from lib.retrieval.grounded_answer import (
    STATUS_ANSWERED,
    answer_course_question,
)
from lib.retrieval.answer_composer import ComposedAnswer
from lib.retrieval import query_augment as qa

# Reuse the committed fixtures + doubles from the sibling pipeline test.
from lib.tests.test_grounded_answer_pipeline import (  # noqa: F401
    COURSE_SLUG,
    _PERMISSIVE_LEXICAL,
    mini_libv2,
)
from lib.tests.test_answer_composer import FakeAnswerClient, SpyCapture


def _envelope(answer, citations, not_in_course=False):
    return json.dumps(
        {"answer": answer, "citations": citations, "not_in_course": not_in_course}
    )


_ANSWER = "A vector store indexes embedding vectors."
_CID = "mini_alpha_chunk_001"
_QUERY = "What does a vector store index?"


def _augment_types(spy):
    aug = {
        qa.DECISION_TYPE_INTENT_ROUTE,
        qa.DECISION_TYPE_MULTITURN,
        qa.DECISION_TYPE_DECOMPOSE,
        qa.DECISION_TYPE_HYDE,
    }
    return [e for e in spy.events if e["decision_type"] in aug]


# --------------------------------------------------------------------------- #
# Byte-identical-when-off
# --------------------------------------------------------------------------- #
def test_all_arms_off_no_augment_captures(mini_libv2: Path, monkeypatch):
    for env in (qa.ENV_INTENT_ROUTE, qa.ENV_MULTITURN, qa.ENV_DECOMPOSE, qa.ENV_HYDE):
        monkeypatch.delenv(env, raising=False)
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == _ANSWER
    assert result.citations[0].chunk_id == _CID
    assert _augment_types(spy) == []


# --------------------------------------------------------------------------- #
# W5.1 intent-route bias
# --------------------------------------------------------------------------- #
def test_intent_route_on_emits_capture_and_still_answers(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(qa.ENV_INTENT_ROUTE, "1")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "show me an example of a vector store",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    rows = [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_INTENT_ROUTE]
    assert len(rows) == 1
    assert len(rows[0]["rationale"]) >= 20
    assert "intent_class=" in rows[0]["rationale"]


# --------------------------------------------------------------------------- #
# W5.2 multi-turn — prior_turns default empty is byte-identical
# --------------------------------------------------------------------------- #
def test_multiturn_no_prior_turns_is_noop(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(qa.ENV_MULTITURN, "1")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    # No prior_turns → the rewrite hook is skipped entirely (no capture).
    assert [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_MULTITURN] == []


@dataclass(frozen=True)
class _FakeRaw:
    """Minimal duck-type of a LibV2 RetrievalResult for the mocked boundary."""

    chunk_id: str
    score: float
    text: str = ""
    source: Dict[str, Any] = field(default_factory=dict)


def test_multiturn_offtopic_rewrite_retains_original_top_passage(monkeypatch):
    """W5.2 verdict-safety — an OFF-TOPIC prior-turn rewrite must NOT evict the
    original-query top passage from the retrieved set.

    The additive rewrite appends unrelated salient terms; the rewritten query
    here retrieves an OFF-TOPIC set WITHOUT the good chunk (simulating the
    worse-retrieval failure mode). The union with the ORIGINAL query keeps the
    original-query top-scoring passage in the set that reaches compose /
    confidence — so the refusal verdict can only improve or stay equal, never
    worsen. Retrieval boundary is mocked; no model / index / torch loaded, and
    no LibV2 course is touched.
    """
    for env in (qa.ENV_INTENT_ROUTE, qa.ENV_DECOMPOSE, qa.ENV_HYDE):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(qa.ENV_MULTITURN, "1")

    query = "what is it?"  # short, self-contained follow-up
    prior_turns = ["Tell me about Photosynthesis and Chloroplasts"]  # off-topic

    good = _FakeRaw("good_original_top", score=9.0, text="Original-query best hit.")
    off_a = _FakeRaw("offtopic_a", score=8.5)
    off_b = _FakeRaw("offtopic_b", score=7.0)

    def _fake_retrieve(libv2_root, course_slug, q, *, engine, limit):
        # ORIGINAL query surfaces the good chunk on top; the rewritten
        # (off-topic-enriched) query ranks unrelated chunks and drops it.
        if q == query:
            return [good, off_b]
        return [off_a, off_b]

    seen: Dict[str, Any] = {}

    def _fake_compose(q, passages, **kwargs):
        seen["compose_query"] = q
        seen["ids"] = [p.chunk_id for p in passages]
        return ComposedAnswer(
            answer_text=None,
            cited_chunk_ids=[],
            not_in_course=True,  # short-circuit before the citation gate (no FS)
            model_id="fake",
            prompt_version="v0",
            attempts=1,
            latency_ms=0.0,
            raw_response_len=0,
        )

    monkeypatch.setattr(ga, "_retrieve", _fake_retrieve)
    monkeypatch.setattr(ga, "compose_answer", _fake_compose)

    spy = SpyCapture()
    answer_course_question(
        Path("/nonexistent-root"), COURSE_SLUG, query,
        client=object(), refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
        chunkset_kind="dart", prior_turns=prior_turns,
    )

    # The rewrite actually fired (so the union path was exercised)...
    rows = [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_MULTITURN]
    assert rows and rows[0]["decision"] == "multiturn:rewritten"
    # ...and the ORIGINAL-query top passage survived into the retrieved set even
    # though the rewritten query dropped it → the verdict cannot worsen.
    assert "good_original_top" in seen["ids"]
    # Compose still runs on the ORIGINAL query (rewrite is retrieval-only).
    assert seen["compose_query"] == query


def test_multiturn_followup_rewrites_and_answers(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(qa.ENV_MULTITURN, "1")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, "what does it index?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
        prior_turns=["Tell me about Vector Stores and Embeddings"],
    )
    assert result.status == STATUS_ANSWERED
    # The stored query is the ORIGINAL (rewrite is retrieval-only).
    assert result.query == "what does it index?"
    rows = [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_MULTITURN]
    assert len(rows) == 1
    assert rows[0]["decision"] == "multiturn:rewritten"


# --------------------------------------------------------------------------- #
# W5.6 decompose
# --------------------------------------------------------------------------- #
def test_decompose_multipart_emits_capture_and_answers(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(qa.ENV_DECOMPOSE, "1")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "What does a vector store index? What is an embedding?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    rows = [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_DECOMPOSE]
    assert len(rows) == 1
    assert rows[0]["decision"] == "decompose:merged"


def test_decompose_single_part_noop(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(qa.ENV_DECOMPOSE, "1")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    rows = [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_DECOMPOSE]
    assert rows and "noop" in rows[0]["decision"]


# --------------------------------------------------------------------------- #
# W5.7 HyDE — the generation consumes one client call, then compose consumes the
# next. The hypothetical never becomes a citation (only real chunks are cited).
# --------------------------------------------------------------------------- #
def test_hyde_on_emits_capture_and_answers(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(qa.ENV_HYDE, "1")
    spy = SpyCapture()
    # 1st response = HyDE hypothetical text (plain), 2nd = compose envelope.
    client = FakeAnswerClient([
        "A vector store indexes embedding vectors for similarity search.",
        _envelope(_ANSWER, [_CID]),
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    # Every cited chunk is a REAL retrieved chunk (the hypothetical is not cited).
    assert all(c.chunk_id == _CID for c in result.citations)
    rows = [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_HYDE]
    assert len(rows) == 1
    assert rows[0]["decision"].startswith("hyde:")


def test_hyde_generation_failure_is_fail_open(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(qa.ENV_HYDE, "1")
    spy = SpyCapture()
    # HyDE gen raises (1st scripted response is an Exception); compose then
    # succeeds on the next. The answer must still resolve (fail-open).
    client = FakeAnswerClient([
        RuntimeError("gen transport down"),
        _envelope(_ANSWER, [_CID]),
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    rows = [e for e in spy.events
            if e["decision_type"] == qa.DECISION_TYPE_HYDE]
    assert rows and "noop" in rows[0]["decision"]
