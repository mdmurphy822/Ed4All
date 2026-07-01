"""Unit tests for the pre-retrieval query/retrieval augmentation helpers (W5.1/2/6/7).

Pure + CI-safe: no model, no server, no network, no torch. Exercises the flag
resolvers, the verdict-safe merge, the deterministic multi-turn rewrite, the
intent-bias reorder, and the decompose/HyDE closures with fake passages + a fake
retrieve/gen closure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from lib.retrieval import query_augment as qa


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakePassage:
    chunk_id: str
    score: float
    text: str = ""
    source: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeResult:
    chunk_id: str
    score: float = 0.0
    source: Dict[str, Any] = field(default_factory=dict)
    chunk_type: str | None = None
    concept_tags: list | None = None
    learning_outcome_refs: list | None = None
    bloom_level: str | None = None


class SpyCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, decision_type, decision, rationale, **kwargs):
        self.events.append(
            {"decision_type": decision_type, "decision": decision,
             "rationale": rationale, **kwargs}
        )


# --------------------------------------------------------------------------- #
# Flag resolvers (parse-with-fallback, default OFF)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "resolver,env",
    [
        (qa.intent_route_enabled, qa.ENV_INTENT_ROUTE),
        (qa.multiturn_enabled, qa.ENV_MULTITURN),
        (qa.decompose_enabled, qa.ENV_DECOMPOSE),
        (qa.hyde_enabled, qa.ENV_HYDE),
    ],
)
def test_flags_default_off_and_parse(resolver, env, monkeypatch):
    monkeypatch.delenv(env, raising=False)
    assert resolver() is False
    monkeypatch.setenv(env, "garbage")
    assert resolver() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv(env, truthy)
        assert resolver() is True


# --------------------------------------------------------------------------- #
# W5 verdict-safe merge
# --------------------------------------------------------------------------- #
def test_merge_retains_flat_top_scorer_and_dedups():
    flat = [FakePassage("a", 9.0), FakePassage("b", 1.0)]
    extra = [[FakePassage("c", 5.0), FakePassage("a", 2.0)]]
    merged = qa.merge_passage_candidates(flat, extra, limit=2)
    ids = [p.chunk_id for p in merged]
    # Flat top-scorer 'a' (9.0) retained as global max; dedup keeps the
    # higher-score instance (9.0 > 2.0). Top-2 by score = [a, c].
    assert ids == ["a", "c"]
    assert merged[0].score == 9.0


def test_merge_keeps_at_least_one_and_new_ids_added():
    flat = [FakePassage("a", 3.0)]
    extra = [[FakePassage("z", 4.0)]]
    merged = qa.merge_passage_candidates(flat, extra, limit=1)
    # limit=1 → single highest scorer; flat 'a' would be dropped but 'z' (4.0)
    # is a strictly higher scorer, so n_above_floor cannot decrease.
    assert len(merged) == 1
    assert merged[0].chunk_id == "z"


def test_merge_empty_flat_uses_extra():
    merged = qa.merge_passage_candidates([], [[FakePassage("x", 1.0)]], limit=5)
    assert [p.chunk_id for p in merged] == ["x"]


# --------------------------------------------------------------------------- #
# W5.2 multi-turn rewrite (deterministic, retrieval-only)
# --------------------------------------------------------------------------- #
def test_multiturn_noop_without_history():
    spy = SpyCapture()
    out = qa.rewrite_query_with_context("How does it work?", None, capture=spy)
    assert out == "How does it work?"
    assert spy.events[0]["decision_type"] == qa.DECISION_TYPE_MULTITURN
    assert "noop" in spy.events[0]["decision"]


def test_multiturn_noop_when_not_followup():
    # A long, self-contained query with no dangling pronoun is not a follow-up.
    q = "What are the core properties of a SHACL NodeShape declaration"
    out = qa.rewrite_query_with_context(q, ["We discussed RDF graphs."], capture=None)
    assert out == q


def test_multiturn_rewrites_followup_additively():
    spy = SpyCapture()
    out = qa.rewrite_query_with_context(
        "How does it work?",
        ["Tell me about Vector Stores and Embeddings"],
        capture=spy,
    )
    assert out.startswith("How does it work?")
    assert "Vector Stores" in out  # salient prior-turn phrase appended
    assert out != "How does it work?"
    ev = spy.events[0]
    assert ev["decision"] == "multiturn:rewritten"
    assert len(ev["rationale"]) >= 20


def test_multiturn_deterministic():
    a = qa.rewrite_query_with_context("what about it?", ["Cognitive Load Theory basics"])
    b = qa.rewrite_query_with_context("what about it?", ["Cognitive Load Theory basics"])
    assert a == b


# --------------------------------------------------------------------------- #
# W5.1 intent-route bias (stable reorder, no set change)
# --------------------------------------------------------------------------- #
def test_intent_bias_no_set_change_and_boosts_facet_match():
    # Query names an objective id + a chunk-type word → facets extracted.
    passages = [
        FakePassage("p1", 5.0, source={"chunk_type": "explanation"}),
        FakePassage("p2", 4.0, source={"chunk_type": "example"}),
        FakePassage("p3", 3.0, source={"chunk_type": "explanation"}),
    ]
    spy = SpyCapture()
    out = qa.bias_passages_by_intent(
        "show me an example", passages, capture=spy,
    )
    # Same set (no drop / add), just reordered — p2 (example) floats forward.
    assert {p.chunk_id for p in out} == {"p1", "p2", "p3"}
    assert out[0].chunk_id == "p2"
    assert spy.events[0]["decision_type"] == qa.DECISION_TYPE_INTENT_ROUTE


def test_intent_bias_no_facets_preserves_order():
    passages = [FakePassage("p1", 5.0), FakePassage("p2", 4.0)]
    out = qa.bias_passages_by_intent("define a derivative", passages, capture=None)
    assert [p.chunk_id for p in out] == ["p1", "p2"]


# --------------------------------------------------------------------------- #
# enrich_passage_facets
# --------------------------------------------------------------------------- #
def test_enrich_folds_result_facets_into_source():
    p = FakePassage("c", 1.0, source={"item_path": "a.html"})
    r = FakeResult("c", chunk_type="example", learning_outcome_refs=["to-01"])
    ep = qa.enrich_passage_facets(p, r)
    assert ep.source["chunk_type"] == "example"
    assert ep.source["learning_outcome_refs"] == ["to-01"]
    assert ep.source["item_path"] == "a.html"  # preserved
    # No facets → identity (same object returned).
    r2 = FakeResult("c")
    assert qa.enrich_passage_facets(p, r2) is p


# --------------------------------------------------------------------------- #
# W5.6 decompose_and_merge
# --------------------------------------------------------------------------- #
def test_decompose_single_part_is_noop():
    flat = [FakePassage("a", 1.0)]
    spy = SpyCapture()

    def _retrieve(q, n):  # pragma: no cover - must not be called
        raise AssertionError("single-part must not retrieve")

    out = qa.decompose_and_merge(
        "What is a derivative", flat, _retrieve, limit=5, capture=spy,
    )
    assert [p.chunk_id for p in out] == ["a"]
    assert "noop" in spy.events[0]["decision"]


def test_decompose_multipart_merges_subquery_hits():
    flat = [FakePassage("a", 9.0)]
    calls: List[str] = []

    def _retrieve(q, n):
        calls.append(q)
        return [FakePassage(f"hit_{len(calls)}", 2.0)]

    spy = SpyCapture()
    out = qa.decompose_and_merge(
        "What is a rectangle? What is a circle?", flat, _retrieve,
        limit=5, capture=spy,
    )
    ids = {p.chunk_id for p in out}
    assert "a" in ids and len(ids) > 1  # flat retained + sub-query hits added
    assert out[0].chunk_id == "a"  # flat top-scorer still first (verdict-safe)
    assert len(calls) >= 2
    assert spy.events[0]["decision"] == "decompose:merged"


def test_decompose_per_part_failure_is_fail_open():
    flat = [FakePassage("a", 9.0)]

    def _retrieve(q, n):
        raise RuntimeError("boom")

    out = qa.decompose_and_merge(
        "What is X? What is Y?", flat, _retrieve, limit=5, capture=None,
    )
    assert [p.chunk_id for p in out] == ["a"]  # merge over flat only


# --------------------------------------------------------------------------- #
# W5.7 HyDE
# --------------------------------------------------------------------------- #
def test_hyde_unions_hypothetical_retrieval():
    flat = [FakePassage("a", 9.0)]
    spy = SpyCapture()

    def _retrieve(text, n):
        assert "hypo" in text
        return [FakePassage("hyde_hit", 3.0)]

    out = qa.hyde_augment(
        "q", flat, _retrieve, lambda q: "hypo passage", limit=5, capture=spy,
    )
    ids = {p.chunk_id for p in out}
    assert ids == {"a", "hyde_hit"}
    assert out[0].chunk_id == "a"  # flat top-scorer retained
    ev = spy.events[0]
    assert ev["decision"] == "hyde:merged"
    assert "never cited" in ev["rationale"]


def test_hyde_generation_failure_fail_open():
    flat = [FakePassage("a", 9.0)]
    spy = SpyCapture()

    def _bad_gen(q):
        raise RuntimeError("gen down")

    out = qa.hyde_augment(
        "q", flat, lambda t, n: [], _bad_gen, limit=5, capture=spy,
    )
    assert [p.chunk_id for p in out] == ["a"]
    assert spy.events[0]["decision"] == "hyde:noop:generation_failed"


def test_hyde_empty_hypothetical_noop():
    flat = [FakePassage("a", 9.0)]
    out = qa.hyde_augment("q", flat, lambda t, n: [], lambda q: "  ", limit=5)
    assert [p.chunk_id for p in out] == ["a"]


def test_generate_hypothetical_uses_chat_completion():
    class C:
        def __init__(self):
            self.calls = []

        def chat_completion(self, messages, *, max_tokens, temperature, **kw):
            self.calls.append((messages, max_tokens, temperature))
            return "a hypothetical answer"

    c = C()
    out = qa.generate_hypothetical_answer(c, "what is X?")
    assert out == "a hypothetical answer"
    assert c.calls[0][2] == 0.0  # deterministic temperature
