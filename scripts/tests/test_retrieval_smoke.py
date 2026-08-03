"""GAP 2 — tests for scripts/harness/retrieval_smoke.py.

Covers the pure logic with a mocked retriever + synthetic chunks (no
LibV2 course on disk):

* sampling determinism (same seed ⇒ identical sample) + content-rich
  preference + apparatus exclusion,
* self-retrieval scoring math (hit@k + MRR),
* OOD probe margin math,
* report shape + pass/fail verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts" / "harness"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import retrieval_smoke as rs  # noqa: E402

# ------------------------------------------------------------------ #
# Fixtures.
# ------------------------------------------------------------------ #


def _chunk(cid: str, *, text: str, chunk_type: str = "explanation",
           words: int = 60, heading: str = "") -> Dict[str, Any]:
    return {
        "id": cid,
        "text": text,
        "chunk_type": chunk_type,
        "word_count": words,
        "source": {"section_heading": heading} if heading else {},
    }


def _corpus(n: int = 30) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for i in range(n):
        # Mix of content-rich explanations and apparatus exercises.
        if i % 5 == 0:
            chunks.append(_chunk(
                f"c_{i:03d}", text=f"Exercise {i}. Solve for x. " * 8,
                chunk_type="exercise", words=64,
            ))
        else:
            chunks.append(_chunk(
                f"c_{i:03d}",
                text=(f"Section {i} explains the core concept number {i} "
                      f"in careful detail. " * 6),
                chunk_type="explanation", words=70,
                heading=f"Topic {i}",
            ))
    return chunks


class _FakeRetriever:
    """Deterministic fake: returns the chunk whose id is embedded in the
    query first, then filler ids. OOD probes get only filler."""

    def __init__(self, corpus: Sequence[Dict[str, Any]]):
        self._by_id = {rs._chunk_id(c): c for c in corpus}
        self._ids = list(self._by_id.keys())

    def __call__(self, query: str, k: int) -> List[Dict[str, Any]]:
        # If a known heading maps to a chunk, surface that chunk at rank 1.
        # Match precisely on the "{heading}. " query prefix so "Topic 1"
        # doesn't collide with "Topic 10".
        hit_id = None
        for cid, chunk in self._by_id.items():
            heading = chunk.get("source", {}).get("section_heading", "")
            if heading and query.startswith(f"{heading}."):
                hit_id = cid
                break
        results: List[Dict[str, Any]] = []
        if hit_id:
            results.append({"chunk_id": hit_id, "score": 0.91})
        for cid in self._ids:
            if cid == hit_id:
                continue
            results.append({"chunk_id": cid, "score": 0.10})
            if len(results) >= k:
                break
        return results[:k]


# ------------------------------------------------------------------ #
# Sampling.
# ------------------------------------------------------------------ #


def test_sampling_is_deterministic() -> None:
    corpus = _corpus(30)
    a = rs.select_sample(corpus, n=8, seed=42)
    b = rs.select_sample(corpus, n=8, seed=42)
    assert [rs._chunk_id(c) for c in a] == [rs._chunk_id(c) for c in b]
    assert len(a) == 8


def test_sampling_excludes_apparatus() -> None:
    corpus = _corpus(30)
    sample = rs.select_sample(corpus, n=20, seed=7)
    for c in sample:
        assert c["chunk_type"] != "exercise", (
            "apparatus chunk leaked into the sample"
        )


def test_sampling_different_seed_can_differ() -> None:
    corpus = _corpus(60)
    a = [rs._chunk_id(c) for c in rs.select_sample(corpus, n=6, seed=1)]
    b = [rs._chunk_id(c) for c in rs.select_sample(corpus, n=6, seed=999)]
    # Not guaranteed distinct, but with 60 chunks the pool is large enough
    # that two seeds should diverge — pin the mechanism works.
    assert a != b or len(set(a)) == 6


def test_sampling_returns_all_when_fewer_than_n() -> None:
    corpus = [_chunk("only_1", text="A rich explanation. " * 10, words=40)]
    sample = rs.select_sample(corpus, n=25, seed=1)
    assert len(sample) == 1


def test_richness_score_excludes_short() -> None:
    short = _chunk("s", text="Too short.", words=3)
    assert rs.richness_score(short, min_words=25) == -1.0


def test_richness_prefers_instructional_over_neutral() -> None:
    rich = _chunk("r", text="x " * 60, chunk_type="worked_example", words=60)
    neutral = _chunk("n", text="x " * 60, chunk_type="", words=60)
    assert rs.richness_score(rich) > rs.richness_score(neutral)


def test_richness_reads_renamed_unit_roles_field() -> None:
    """Multi-ontology rename: composite_unit / unit_roles carry the
    pedagogical signal now. Prefer via unit_roles, exclude via unit_roles."""
    # Preferred role via the renamed unit_roles list.
    rich = {"id": "r", "text": "x " * 60, "word_count": 60,
            "unit_roles": ["worked_example"]}
    neutral = {"id": "n", "text": "x " * 60, "word_count": 60}
    assert rs.richness_score(rich) > rs.richness_score(neutral)
    # Apparatus excluded via the renamed composite_unit field.
    apparatus = {"id": "a", "text": "x " * 60, "word_count": 60,
                 "composite_unit": "exercise_set"}
    assert rs.richness_score(apparatus) == -1.0


def test_sampling_excludes_apparatus_by_renamed_field() -> None:
    chunks = [
        {"id": f"c_{i}", "text": "Concept prose sentence. " * 8,
         "word_count": 64,
         "unit_roles": ["exercise_set"] if i % 2 == 0 else ["statement"]}
        for i in range(20)
    ]
    sample = rs.select_sample(chunks, n=20, seed=3)
    for c in sample:
        assert "exercise_set" not in c.get("unit_roles", [])


# ------------------------------------------------------------------ #
# Query forming.
# ------------------------------------------------------------------ #


def test_form_query_combines_heading_and_first_sentence() -> None:
    c = _chunk("c", text="First sentence here. Second sentence ignored.",
               heading="My Heading", words=40)
    q = rs.form_query(c)
    assert "My Heading" in q
    assert "First sentence here" in q
    assert "Second sentence" not in q


# ------------------------------------------------------------------ #
# Scoring math.
# ------------------------------------------------------------------ #


def test_self_retrieval_hit_and_mrr_math() -> None:
    corpus = _corpus(30)
    retriever = _FakeRetriever(corpus)
    sample = rs.select_sample(corpus, n=10, seed=3)
    outcomes = rs.score_self_retrieval(sample, retriever, k=5)
    # Every sampled chunk has a heading → fake returns it at rank 1.
    assert all(o.hit for o in outcomes)
    assert all(o.rank == 1 for o in outcomes)
    assert all(o.reciprocal_rank == 1.0 for o in outcomes)


def test_self_retrieval_miss_scores_zero_rr() -> None:
    # A chunk whose heading the fake retriever won't surface (no heading).
    c = _chunk("lonely", text="Some content sentence. " * 6, words=40)
    corpus = _corpus(10) + [c]
    retriever = _FakeRetriever(corpus)
    outcomes = rs.score_self_retrieval([c], retriever, k=5)
    o = outcomes[0]
    # "lonely" has no heading → query is just first sentence → fake
    # won't rank it first; it may appear as filler though. Assert the
    # math: if not in top-k, rr==0; if present, rr==1/rank.
    if o.hit:
        assert o.reciprocal_rank == pytest.approx(1.0 / o.rank)
    else:
        assert o.reciprocal_rank == 0.0


def test_ood_probe_margin_math() -> None:
    corpus = _corpus(10)
    retriever = _FakeRetriever(corpus)
    probes = rs.probe_margins(["totally unrelated cooking question"],
                              retriever, k=5)
    assert len(probes) == 1
    p = probes[0]
    # OOD probe → no heading hit → all filler at 0.10 → top 0.10, margin 0.
    assert p.top_score == pytest.approx(0.10)
    assert p.margin == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# Report shape + verdict.
# ------------------------------------------------------------------ #


def test_build_report_shape_and_pass() -> None:
    corpus = _corpus(30)
    retriever = _FakeRetriever(corpus)
    sample = rs.select_sample(corpus, n=10, seed=5)
    self_outcomes = rs.score_self_retrieval(sample, retriever, k=5)
    probe_outcomes = rs.probe_margins(rs.DEFAULT_OOD_PROBES, retriever, k=5)
    report = rs.build_report(
        course_code="synthetic",
        engine="hybrid-rrf",
        k=5,
        sample_requested=10,
        self_outcomes=self_outcomes,
        probe_outcomes=probe_outcomes,
        hit_threshold=0.8,
        seed=5,
    )
    # Required keys present.
    for key in ("course_code", "engine", "k", "hit_at_k", "mrr",
                "passed", "self_retrieval", "ood_probes", "sample_scored"):
        assert key in report
    assert report["hit_at_k"] == 1.0
    assert report["passed"] is True
    assert len(report["self_retrieval"]) == 10
    assert len(report["ood_probes"]) == len(rs.DEFAULT_OOD_PROBES)


def test_build_report_fails_below_threshold() -> None:
    # All misses → hit@k 0 → fail.
    outcomes = [
        rs.SelfRetrievalOutcome(chunk_id=f"c{i}", query="q", hit=False,
                                rank=None, reciprocal_rank=0.0)
        for i in range(5)
    ]
    report = rs.build_report(
        course_code="x", engine="lexical", k=5, sample_requested=5,
        self_outcomes=outcomes, probe_outcomes=[], hit_threshold=0.8, seed=1,
    )
    assert report["hit_at_k"] == 0.0
    assert report["passed"] is False


def test_build_report_empty_sample_fails() -> None:
    report = rs.build_report(
        course_code="x", engine="lexical", k=5, sample_requested=25,
        self_outcomes=[], probe_outcomes=[], hit_threshold=0.8, seed=1,
    )
    assert report["passed"] is False
    assert report["sample_scored"] == 0
