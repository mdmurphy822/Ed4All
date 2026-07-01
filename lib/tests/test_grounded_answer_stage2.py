"""Stage-2 post-retrieval arm tests: W5.3 graph-expand, W5.4 completeness
re-retrieve, W5.5 confidence-graded hedge tier.

Deterministic + CI-safe: no model, no server, no network, no torch. The LLM call
site is the ``FakeAnswerClient`` double; retrieval runs the REAL lexical BM25 path
over the committed mini-course fixture (reused via ``mini_libv2``).

Focus: (a) byte-identical-when-off (no capture rows, same answer/status);
(b) each arm emits its DecisionCapture row when on; (c) verdict-safety — no arm
turns an answered result into a refusal (and the hedge never downgrades a
confident answer to a refusal).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval import graph_expand as gx
from lib.retrieval import grounded_answer as ga
from lib.retrieval.answer_composer import RetrievedPassage
from lib.retrieval.grounded_answer import (
    STATUS_ANSWERED,
    STATUS_ANSWERED_WITH_WARNINGS,
    STATUS_REFUSED_LOW_CONFIDENCE,
    answer_course_question,
)
from lib.retrieval.refusal import ConfidenceVerdict, RefusalPolicy

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


def _passage(cid: str, score: float, text: str = "x") -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=cid, text=text, score=score, engine="lexical",
        item_path="", section_heading=None, module_id=None, source={},
    )


# --------------------------------------------------------------------------- #
# W5.3 graph-expand — pure module unit tests
# --------------------------------------------------------------------------- #
def _write_graph(course_dir: Path, nodes, edges) -> None:
    gdir = course_dir / "graph"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "concept_graph_semantic.json").write_text(
        json.dumps({"kind": "concept_semantic", "generated_at": "t",
                    "nodes": nodes, "edges": edges}),
        encoding="utf-8",
    )


def test_build_index_occurrences_and_chunk_nodes():
    graph = {
        "nodes": [
            {"id": "vec", "class": "DomainConcept",
             "occurrences": ["c1", "c2"]},
            {"id": "c9", "class": "Chunk"},
        ],
        "edges": [{"source": "vec", "target": "c9", "type": "related-to"}],
    }
    chunk_to_nodes, node_to_chunks, adjacency = gx._build_index(graph)
    assert node_to_chunks["vec"] == {"c1", "c2"}
    assert chunk_to_nodes["c1"] == {"vec"}
    # Chunk-class node id is bound to itself.
    assert node_to_chunks["c9"] == {"c9"}
    assert adjacency["vec"] == {"c9"} and adjacency["c9"] == {"vec"}


def test_graph_reachable_excludes_seeds_and_caps():
    graph = {
        "nodes": [{"id": "vec", "class": "DomainConcept",
                   "occurrences": ["c1", "c2", "c3"]}],
        "edges": [],
    }
    c2n, n2c, adj = gx._build_index(graph)
    out = gx.graph_reachable_chunk_ids(["c1"], c2n, n2c, adj, max_add=5)
    assert "c1" not in out  # seed excluded
    assert set(out) == {"c2", "c3"}
    # cap honoured
    assert len(gx.graph_reachable_chunk_ids(["c1"], c2n, n2c, adj, max_add=1)) == 1


def test_graph_reachable_no_seed_match_is_empty():
    graph = {"nodes": [{"id": "vec", "occurrences": ["c1"]}], "edges": []}
    c2n, n2c, adj = gx._build_index(graph)
    assert gx.graph_reachable_chunk_ids(["zzz"], c2n, n2c, adj, max_add=5) == []


def test_expand_appends_real_neighbor_verdict_safe(mini_libv2: Path):
    course_dir = mini_libv2 / "courses" / COURSE_SLUG
    # vector-store concept co-occurs the retrieved seed (001) with 002.
    _write_graph(
        course_dir,
        nodes=[{"id": "vector-store", "class": "DomainConcept",
                "occurrences": ["mini_alpha_chunk_001", "mini_alpha_chunk_002"]}],
        edges=[],
    )
    flat = [_passage("mini_alpha_chunk_001", 3.0)]
    spy = SpyCapture()
    out = gx.expand_passages_via_graph(
        flat, course_dir, "dart", engine="lexical", capture=spy,
        course_slug=COURSE_SLUG, query_sha="deadbeef",
    )
    # Neighbor appended, real text loaded from the chunkset, score 0.0.
    assert [p.chunk_id for p in out] == [
        "mini_alpha_chunk_001", "mini_alpha_chunk_002"
    ]
    neighbor = out[1]
    assert neighbor.score == 0.0
    assert neighbor.text.startswith("FAISS")
    # Original passage untouched (order + score preserved → verdict-safe).
    assert out[0].chunk_id == "mini_alpha_chunk_001" and out[0].score == 3.0
    rows = [e for e in spy.events
            if e["decision_type"] == gx.DECISION_TYPE_GRAPH_EXPAND]
    assert len(rows) == 1 and rows[0]["decision"] == "graph_expand:expanded"
    assert len(rows[0]["rationale"]) >= 20


def test_expand_no_graph_is_fail_open(mini_libv2: Path):
    course_dir = mini_libv2 / "courses" / COURSE_SLUG  # no graph written
    flat = [_passage("mini_alpha_chunk_001", 3.0)]
    spy = SpyCapture()
    out = gx.expand_passages_via_graph(
        flat, course_dir, "dart", engine="lexical", capture=spy,
        course_slug=COURSE_SLUG, query_sha="x",
    )
    assert [p.chunk_id for p in out] == ["mini_alpha_chunk_001"]
    rows = [e for e in spy.events
            if e["decision_type"] == gx.DECISION_TYPE_GRAPH_EXPAND]
    assert rows and "noop" in rows[0]["decision"]


def test_expand_skips_unloadable_ids(mini_libv2: Path):
    course_dir = mini_libv2 / "courses" / COURSE_SLUG
    _write_graph(
        course_dir,
        nodes=[{"id": "vec", "class": "DomainConcept",
                "occurrences": ["mini_alpha_chunk_001", "ghost_chunk_999"]}],
        edges=[],
    )
    flat = [_passage("mini_alpha_chunk_001", 3.0)]
    out = gx.expand_passages_via_graph(
        flat, course_dir, "dart", engine="lexical",
    )
    # ghost id has no chunkset record → skipped (anti-fabrication), no crash.
    assert [p.chunk_id for p in out] == ["mini_alpha_chunk_001"]


# --------------------------------------------------------------------------- #
# W5.3 graph-expand — pipeline wiring (off vs on)
# --------------------------------------------------------------------------- #
def test_graph_expand_off_no_capture(mini_libv2: Path, monkeypatch):
    monkeypatch.delenv(gx.ENV_GRAPH_EXPAND, raising=False)
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    assert [e for e in spy.events
            if e["decision_type"] == gx.DECISION_TYPE_GRAPH_EXPAND] == []


def test_graph_expand_on_emits_capture_and_still_answers(mini_libv2: Path, monkeypatch):
    course_dir = mini_libv2 / "courses" / COURSE_SLUG
    _write_graph(
        course_dir,
        nodes=[{"id": "vector-store", "class": "DomainConcept",
                "occurrences": ["mini_alpha_chunk_001", "mini_alpha_chunk_002"]}],
        edges=[],
    )
    monkeypatch.setenv(gx.ENV_GRAPH_EXPAND, "1")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    # limit=1 so retrieval returns only the seed; the neighbor is genuinely new.
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY, limit=1,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    rows = [e for e in spy.events
            if e["decision_type"] == gx.DECISION_TYPE_GRAPH_EXPAND]
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# W5.5 hedge tier — pure helper tests
# --------------------------------------------------------------------------- #
def _verdict(confident: bool, top: float, floor: float) -> ConfidenceVerdict:
    policy = RefusalPolicy(
        engine="lexical", min_top_score=floor, score_floor=0.0,
        min_passages_above_floor=1, policy_version="t",
    )
    return ConfidenceVerdict(
        confident=confident, signals={"top_score": top}, policy=policy
    )


def test_hedge_active_only_in_borderline_band():
    # floor=1.0, margin=0.15 → band [1.0, 1.15)
    assert ga._hedge_active(_verdict(True, 1.05, 1.0), 0.15) is True
    # comfortably above the band → confident, no hedge
    assert ga._hedge_active(_verdict(True, 2.0, 1.0), 0.15) is False
    # a NON-confident verdict never hedges (it refuses upstream)
    assert ga._hedge_active(_verdict(False, 0.5, 1.0), 0.15) is False
    # a permissive floor of 0.0 → empty band → never hedge
    assert ga._hedge_active(_verdict(True, 0.0, 0.0), 0.15) is False


def test_hedge_answer_adds_caveat_and_warning_never_refuses():
    text, warns, status = ga._hedge_answer(
        "raw answer", ["w0"], STATUS_ANSWERED, active=True
    )
    assert text.startswith("Note:")
    assert "raw answer" in text
    assert ga._HEDGE_WARNING in warns and "w0" in warns
    assert status == STATUS_ANSWERED_WITH_WARNINGS


def test_hedge_answer_inactive_is_identity():
    text, warns, status = ga._hedge_answer(
        "raw", ["w0"], STATUS_ANSWERED, active=False
    )
    assert text == "raw" and warns == ["w0"] and status == STATUS_ANSWERED


# --------------------------------------------------------------------------- #
# W5.5 hedge tier — pipeline wiring
# --------------------------------------------------------------------------- #
def _borderline_policy(mini_libv2: Path):
    # Resolve the fixture's actual top score, then pin a floor just below it so
    # the query is CONFIDENT but lands inside the hedge band.
    from lib.retrieval.grounded_answer import _retrieve, _libv2_root
    raw = _retrieve(_libv2_root(mini_libv2), COURSE_SLUG, _QUERY,
                    engine="lexical", limit=8)
    top = max(float(getattr(r, "score", 0.0) or 0.0) for r in raw)
    floor = top / 1.05  # top sits ~5% above floor → inside a 15% band
    return RefusalPolicy(
        engine="lexical", min_top_score=floor, score_floor=0.0,
        min_passages_above_floor=1, policy_version="t-borderline",
    ), top


def test_hedge_off_is_byte_identical(mini_libv2: Path, monkeypatch):
    monkeypatch.delenv(ga.ENV_HEDGE_TIER, raising=False)
    policy, _ = _borderline_policy(mini_libv2)
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=policy, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == _ANSWER  # no caveat prepended
    assert ga._HEDGE_WARNING not in result.warnings
    assert [e for e in spy.events
            if e["decision_type"] == ga.DECISION_TYPE_HEDGE] == []


def test_hedge_on_borderline_hedges_answer(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(ga.ENV_HEDGE_TIER, "1")
    policy, _ = _borderline_policy(mini_libv2)
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=policy, capture=spy,
    )
    # Still ANSWERED (with warnings) — never a refusal.
    assert result.status == STATUS_ANSWERED_WITH_WARNINGS
    assert result.answer_text.startswith("Note:")
    assert _ANSWER in result.answer_text
    assert ga._HEDGE_WARNING in result.warnings
    assert result.confidence.get("hedged") is True
    rows = [e for e in spy.events
            if e["decision_type"] == ga.DECISION_TYPE_HEDGE]
    assert len(rows) == 1 and rows[0]["decision"] == "hedge:hedged"


def test_hedge_on_confident_answer_not_hedged(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(ga.ENV_HEDGE_TIER, "1")
    spy = SpyCapture()
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    # Permissive floor 0.0 → comfortably confident, never in the hedge band.
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    assert result.status == STATUS_ANSWERED
    assert result.answer_text == _ANSWER
    rows = [e for e in spy.events
            if e["decision_type"] == ga.DECISION_TYPE_HEDGE]
    assert len(rows) == 1 and rows[0]["decision"] == "hedge:confident"


def test_hedge_never_rescues_a_hard_refusal(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv(ga.ENV_HEDGE_TIER, "1")
    strict = RefusalPolicy(
        engine="lexical", min_top_score=100.0, score_floor=0.5,
        min_passages_above_floor=1, policy_version="t-strict",
    )
    client = FakeAnswerClient([_envelope(_ANSWER, [_CID])])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG, _QUERY,
        client=client, refusal_policy=strict,
    )
    # A hard refusal stays a refusal — the hedge band never reaches this path.
    assert result.status == STATUS_REFUSED_LOW_CONFIDENCE


# --------------------------------------------------------------------------- #
# W5.4 completeness re-retrieve
# --------------------------------------------------------------------------- #
def test_completeness_reretrieve_flag_default_off():
    assert ga._resolve_completeness_reretrieve() is False


def test_completeness_reretrieve_on_still_answers(mini_libv2: Path, monkeypatch):
    monkeypatch.setenv("ED4ALL_ANSWER_COMPLETENESS_RERETRIEVE", "1")
    spy = SpyCapture()
    # Multi-part query; first compose answers only the first part, the re-ask
    # (2nd scripted response) covers the second. Both cite a real chunk.
    client = FakeAnswerClient([
        _envelope("A vector store indexes embedding vectors.", [_CID]),
        _envelope("FAISS performs similarity search.",
                  ["mini_alpha_chunk_002"]),
    ])
    result = answer_course_question(
        mini_libv2, COURSE_SLUG,
        "What does a vector store index? What does FAISS do?",
        client=client, refusal_policy=_PERMISSIVE_LEXICAL, capture=spy,
    )
    # Never regressed into a refusal.
    assert result.status in (STATUS_ANSWERED, STATUS_ANSWERED_WITH_WARNINGS)
    rows = [e for e in spy.events
            if e["decision_type"] == ga.DECISION_TYPE_COMPLETENESS_RECHECK]
    assert rows  # the recheck ran and emitted its capture
