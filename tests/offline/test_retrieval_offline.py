"""WS1.1 offline assertion: the lexical retrieval pipeline does zero network I/O.

Every test runs the real retrieval path (BM25 over archived ``chunks.jsonl``,
multi-query single-course fusion, citation anchoring, gold-set loading) under
the ``offline_guard`` socket guard. Any socket use anywhere in the
import-to-answer path raises ``NetworkBlockedError`` and fails the test loudly.

Heavy imports are kept INSIDE the guard scope so a future regression that adds
a network-touching import to ``retriever.py`` (an HF-hub check, a telemetry
ping) is caught here, not silently tolerated because the import already ran at
collection time.

Tests 3/4 depend on Executor A/B modules (``lib.retrieval.citation_anchor`` /
``lib.retrieval.gold_set``). They ``importorskip`` those modules so this suite
can land before A/B finish and activate automatically as the modules appear.
The tier-2 real-corpus test skips when the in-tree LibV2 corpora are absent
(clean checkout / CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Resolve the in-tree LibV2 root for the tier-2 real-corpus test without
# importing lib.paths at module scope (keep the offline path minimal).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gold_questions(course_dir: Path) -> list[str]:
    gold = json.loads((course_dir / "retrieval_eval" / "gold_set.json").read_text())
    return [q["question_text"] for q in gold["questions"]]


def test_bm25_retrieval_end_to_end_offline(mini_course: Path, offline_guard):
    """BM25 over the mini-course fixture answers every gold query — offline."""
    from LibV2.tools.libv2.retriever import LazyBM25

    chunks = _load_jsonl(mini_course / "semantik_chunks" / "chunks.jsonl")
    bm25 = LazyBM25(chunks)
    questions = _gold_questions(mini_course)
    assert questions, "fixture gold set must carry questions"
    for q in questions:
        results = bm25.search(q, limit=5)
        assert len(results) >= 1, f"no offline BM25 result for query: {q!r}"


def test_multi_retriever_offline(mini_course: Path, offline_guard):
    """Single-course MultiQueryRetriever.retrieve runs fully offline."""
    from LibV2.tools.libv2.multi_retriever import MultiQueryRetriever

    libv2_root = mini_course.parents[1]  # .../LibV2
    retriever = MultiQueryRetriever(
        repo_root=libv2_root, course_slug="mini-retrieval-101"
    )
    # decompose=False keeps it to a single in-process BM25 pass (no fan-out
    # threads opening anything); course-scoped so it resolves the semantik chunkset.
    result = retriever.retrieve(
        "chunking strategies",
        limit=5,
        decompose=False,
        course_slug="mini-retrieval-101",
    )
    assert result.result_count >= 1


def test_citation_anchor_offline(mini_course: Path, offline_guard):
    """``anchor_report`` over the fixture runs offline (citation-back is part of
    the query path and must be offline too)."""
    anchor_mod = pytest.importorskip("lib.retrieval.citation_anchor")

    chunks_path = mini_course / "semantik_chunks" / "chunks.jsonl"
    report = anchor_mod.anchor_report(
        chunks_path, mini_course, chunkset_kind="semantik"
    )
    assert report["total_chunks"] >= 1
    # The fixture deliberately includes one fabricated-span chunk, so the
    # anchoring rate need not be 1.0; we only assert the report computed and
    # found real source pages (no offline failure mid-resolution).
    assert report["status_counts"].get("source_page_missing", 0) == 0


def test_gold_set_loader_offline(mini_course: Path, offline_guard):
    """The gold-set loader's verify mode (opens chunkset, recomputes sha,
    checks containment) runs offline."""
    gold_mod = pytest.importorskip("lib.retrieval.gold_set")

    _gold, issues = gold_mod.load_gold_set(mini_course, verify=True)
    critical = [i for i in issues if getattr(i, "severity", "") == "critical"]
    assert not critical, f"offline gold-set verify produced critical issues: {critical}"


def _discover_real_corpus_chunkset() -> Path | None:
    """Find any in-tree real corpus chunkset, newest-first.

    Tier-2 tests discover slugs dynamically and skip when absent (no hardcoded
    course slugs in tracked code). Probes the canonical chunkset locations
    under each LibV2 course dir and returns the first non-empty chunks.jsonl.
    """
    courses_root = _REPO_ROOT / "LibV2" / "courses"
    if not courses_root.is_dir():
        return None
    # Mirrors the production chunkset resolver precedence (imscc -> semantik ->
    # legacy dart -> legacy corpus); dart_chunks/ + corpus/ stay as legacy
    # dual-read coverage for un-migrated on-disk corpora.
    rel_paths = (
        "imscc_chunks/chunks.jsonl",
        "semantik_chunks/chunks.jsonl",
        "dart_chunks/chunks.jsonl",
        "corpus/chunks.jsonl",
    )
    for course_dir in sorted(courses_root.iterdir(), reverse=True):
        if not course_dir.is_dir():
            continue
        for rel in rel_paths:
            chunks_path = course_dir / rel
            if chunks_path.exists() and chunks_path.stat().st_size > 0:
                return chunks_path
    return None


def test_real_corpus_retrieval_offline(offline_guard):
    """Tier-2: BM25 over an in-tree real corpus runs offline (skip if absent).

    Proves the real archived pipeline is offline-clean today, not just the
    synthetic fixture. Discovers any archived course chunkset dynamically and
    skips cleanly on a clean checkout / CI where no real corpus is present.
    """
    chunks_path = _discover_real_corpus_chunkset()
    if chunks_path is None:
        pytest.skip("no in-tree real corpus chunkset found under LibV2/courses/")

    from LibV2.tools.libv2.retriever import LazyBM25

    chunks = _load_jsonl(chunks_path)
    if not chunks:
        pytest.skip(f"real corpus chunkset empty: {chunks_path}")
    bm25 = LazyBM25(chunks)
    # Query the corpus with a generic token drawn from its own first chunk so
    # the assertion is corpus-agnostic (no domain-specific query hardcoded).
    first_text = str(chunks[0].get("text", "")).strip()
    query = " ".join(first_text.split()[:5]) or "the"
    results = bm25.search(query, limit=10)
    assert len(results) >= 1
