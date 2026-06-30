"""Unit tests for the cross-encoder reranker (lib/retrieval/reranker.py).

Deterministic + CI-safe: the ``fake`` registry provider scores via
``sha256(query+text)`` so reorder/trim/anti-fabrication run with zero weights
and zero network. A tier-2 real-model smoke (BAAI/bge-reranker-base) is
discover-or-skip — skipped unless the [embedding] extra weights are present.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from lib.retrieval.reranker import (
    DEFAULT_CANDIDATE_POOL,
    DECISION_TYPE_RERANK,
    RerankerBackendUnavailable,
    RerankerClient,
    _fake_score,
    allow_fake_enabled,
    build_reranker_client,
    maybe_rerank,
    rerank_retrieved_passages,
    resolve_candidate_pool,
    resolve_reranker_provider,
    reranker_configured,
)


@dataclass(frozen=True)
class _Passage:
    """Minimal duck-type of RetrievedPassage for reorder tests."""

    chunk_id: str
    text: str
    score: float


class _SpyCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, decision_type, decision, rationale, **kwargs):
        self.events.append(
            {"decision_type": decision_type, "decision": decision,
             "rationale": rationale, **kwargs}
        )


def _fake_resolved():
    return resolve_reranker_provider("fake")


def _fake_client():
    return build_reranker_client(_fake_resolved())


# --------------------------------------------------------------------------- #
# Resolver / registry
# --------------------------------------------------------------------------- #


def test_resolve_unknown_provider_raises_listing():
    with pytest.raises(ValueError) as exc:
        resolve_reranker_provider("bogus-not-registered")
    msg = str(exc.value)
    assert "Unknown reranker provider" in msg
    assert "st-cross-encoder" in msg  # lists registered providers
    assert "fake" in msg


def test_resolver_chain(monkeypatch):
    monkeypatch.delenv("ED4ALL_RERANK_MODEL", raising=False)
    # Registry default for the real provider.
    r = resolve_reranker_provider("st-cross-encoder")
    assert r.model_id == "BAAI/bge-reranker-base"
    assert r.license == "MIT"
    assert r.kind == "cross-encoder"
    assert r.device == "cpu"
    # Env override.
    monkeypatch.setenv("ED4ALL_RERANK_MODEL", "BAAI/bge-reranker-large")
    assert resolve_reranker_provider("st-cross-encoder").model_id == "BAAI/bge-reranker-large"
    # Explicit arg wins over env.
    assert (
        resolve_reranker_provider("st-cross-encoder", "BAAI/bge-reranker-v2-m3").model_id
        == "BAAI/bge-reranker-v2-m3"
    )


def test_candidate_pool_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_RERANK_CANDIDATE_POOL", raising=False)
    assert resolve_candidate_pool() == DEFAULT_CANDIDATE_POOL
    monkeypatch.setenv("ED4ALL_RERANK_CANDIDATE_POOL", "50")
    assert resolve_candidate_pool() == 50
    # Garbage / non-positive → default.
    monkeypatch.setenv("ED4ALL_RERANK_CANDIDATE_POOL", "not-an-int")
    assert resolve_candidate_pool() == DEFAULT_CANDIDATE_POOL
    monkeypatch.setenv("ED4ALL_RERANK_CANDIDATE_POOL", "-3")
    assert resolve_candidate_pool() == DEFAULT_CANDIDATE_POOL
    # Explicit arg wins.
    assert resolve_candidate_pool(7) == 7


def test_batch_size_parse_with_fallback(monkeypatch):
    monkeypatch.setenv("ED4ALL_RERANK_BATCH_SIZE", "garbage")
    assert resolve_reranker_provider("st-cross-encoder").batch_size == 16
    monkeypatch.setenv("ED4ALL_RERANK_BATCH_SIZE", "4")
    assert resolve_reranker_provider("st-cross-encoder").batch_size == 4


def test_device_override(monkeypatch):
    monkeypatch.setenv("ED4ALL_RERANK_DEVICE", "cuda")
    assert resolve_reranker_provider("st-cross-encoder").device == "cuda"


def test_reranker_configured(monkeypatch):
    monkeypatch.delenv("ED4ALL_RERANK_PROVIDER", raising=False)
    assert reranker_configured() is None
    monkeypatch.setenv("ED4ALL_RERANK_PROVIDER", "   ")
    assert reranker_configured() is None  # whitespace → off
    monkeypatch.setenv("ED4ALL_RERANK_PROVIDER", "fake")
    assert reranker_configured() == "fake"


# --------------------------------------------------------------------------- #
# Fake scorer determinism
# --------------------------------------------------------------------------- #


def test_fake_reranker_deterministic():
    a = _fake_score("what is x?", "passage about x")
    b = _fake_score("what is x?", "passage about x")
    assert a == b
    assert 0.0 <= a < 1.0
    # Different inputs (almost surely) differ.
    assert _fake_score("q1", "t") != _fake_score("q2", "t")


# --------------------------------------------------------------------------- #
# Reorder + trim + anti-fabrication
# --------------------------------------------------------------------------- #


def test_rerank_reorders_and_trims():
    client = _fake_client()
    # Build 5 passages, then find which text the fake scores HIGHEST for a query.
    query = "the reranker target query"
    texts = [f"candidate passage number {i}" for i in range(5)]
    scores = client.score(query, texts)
    best_idx = max(range(5), key=lambda i: scores[i])
    passages = [
        _Passage(chunk_id=f"c{i}", text=texts[i], score=1.0 - i * 0.01)
        for i in range(5)
    ]
    # Deliberately put the cross-encoder-best passage LAST in first-stage order.
    passages = [p for i, p in enumerate(passages) if i != best_idx] + [passages[best_idx]]
    out = rerank_retrieved_passages(query, passages, top_k=3, client=client)
    assert len(out) == 3  # trimmed to top_k
    assert out[0].chunk_id == f"c{best_idx}"  # best moved to rank 1


def test_anti_fabrication_subset():
    client = _fake_client()
    passages = [_Passage(chunk_id=f"c{i}", text=f"text {i}", score=0.5) for i in range(4)]
    out = rerank_retrieved_passages("q", passages, top_k=2, client=client)
    in_ids = {p.chunk_id for p in passages}
    out_ids = [p.chunk_id for p in out]
    assert set(out_ids).issubset(in_ids)  # strict subset, no invented ids
    assert len(out_ids) == len(set(out_ids))  # no dupes
    assert all(p in passages for p in out)  # same frozen objects, never fabricated


def test_always_keep_at_least_one():
    client = _fake_client()
    passages = [_Passage(chunk_id="c0", text="only", score=0.5)]
    # top_k=0 must still keep >=1.
    out = rerank_retrieved_passages("q", passages, top_k=0, client=client)
    assert len(out) == 1
    assert out[0].chunk_id == "c0"
    # Empty input → empty output (no crash).
    assert rerank_retrieved_passages("q", [], top_k=5, client=client) == []


def test_native_score_preserved():
    client = _fake_client()
    passages = [_Passage(chunk_id=f"c{i}", text=f"text {i}", score=0.9 - i * 0.1) for i in range(4)]
    score_by_id = {p.chunk_id: p.score for p in passages}
    out = rerank_retrieved_passages("q", passages, top_k=4, client=client)
    # Only ORDER changes; each passage keeps its first-stage native score.
    for p in out:
        assert p.score == score_by_id[p.chunk_id]


# --------------------------------------------------------------------------- #
# Fail-open
# --------------------------------------------------------------------------- #


class _BoomClient(RerankerClient):
    def score(self, query, texts):  # noqa: D401
        raise RerankerBackendUnavailable("simulated backend down")


def test_fail_open_on_backend_unavailable(monkeypatch, caplog):
    monkeypatch.setenv("ED4ALL_RERANK_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_RERANK_ALLOW_FAKE", "1")
    passages = [_Passage(chunk_id=f"c{i}", text=f"t{i}", score=0.5) for i in range(3)]
    boom = _BoomClient(_fake_resolved())
    out = maybe_rerank("q", passages, top_k=2, client=boom)
    # Fail-open: original order returned unchanged, no raise.
    assert [p.chunk_id for p in out] == ["c0", "c1", "c2"]


def test_fake_refused_without_allow_flag(monkeypatch):
    monkeypatch.setenv("ED4ALL_RERANK_PROVIDER", "fake")
    monkeypatch.delenv("ED4ALL_RERANK_ALLOW_FAKE", raising=False)
    passages = [_Passage(chunk_id=f"c{i}", text=f"t{i}", score=0.5) for i in range(3)]
    # Anti-poisoning: fake reranker refused on a production read path → fail-open,
    # original order preserved (never silently reorders a real answer).
    out = maybe_rerank("q", passages, top_k=3)
    assert [p.chunk_id for p in out] == ["c0", "c1", "c2"]


def test_maybe_rerank_off_is_noop(monkeypatch):
    monkeypatch.delenv("ED4ALL_RERANK_PROVIDER", raising=False)
    passages = [_Passage(chunk_id=f"c{i}", text=f"t{i}", score=0.5) for i in range(3)]
    out = maybe_rerank("q", passages, top_k=2)
    # Unset → strict no-op: same objects, same order, NOT trimmed.
    assert out == passages


def test_score_count_mismatch_fails_open(monkeypatch):
    monkeypatch.setenv("ED4ALL_RERANK_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_RERANK_ALLOW_FAKE", "1")

    class _ShortClient(RerankerClient):
        def score(self, query, texts):
            return [0.5]  # wrong length

    passages = [_Passage(chunk_id=f"c{i}", text=f"t{i}", score=0.5) for i in range(3)]
    out = maybe_rerank("q", passages, top_k=3, client=_ShortClient(_fake_resolved()))
    assert [p.chunk_id for p in out] == ["c0", "c1", "c2"]


# --------------------------------------------------------------------------- #
# Decision capture
# --------------------------------------------------------------------------- #


def test_rerank_emits_capture():
    client = _fake_client()
    spy = _SpyCapture()
    passages = [_Passage(chunk_id=f"c{i}", text=f"text {i}", score=0.5) for i in range(5)]
    rerank_retrieved_passages(
        "q", passages, top_k=3, client=client, capture=spy,
        course_slug="demo", query_sha="deadbeef",
    )
    rows = [e for e in spy.events if e["decision_type"] == DECISION_TYPE_RERANK]
    assert len(rows) == 1
    rationale = rows[0]["rationale"]
    assert len(rationale) >= 20
    assert "provider=fake" in rationale
    assert "candidate_pool=5" in rationale
    assert "top_k=3" in rationale
    assert "rank_churn=" in rationale
    assert "fallback_used=False" in rationale


# --------------------------------------------------------------------------- #
# Tier-2 real-model smoke (discover-or-skip; never network)
# --------------------------------------------------------------------------- #


def test_real_cross_encoder_smoke():
    pytest.importorskip("sentence_transformers")
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    resolved = resolve_reranker_provider("st-cross-encoder")
    client = build_reranker_client(resolved)
    try:
        scores = client.score("what is a cell?", ["a cell is the unit of life", "unrelated"])
    except RerankerBackendUnavailable:
        pytest.skip("bge-reranker-base weights not cached locally")
    assert len(scores) == 2
    # The on-topic passage should out-score the unrelated one.
    assert scores[0] > scores[1]
