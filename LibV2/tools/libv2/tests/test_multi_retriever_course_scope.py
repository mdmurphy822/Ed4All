"""Regression tests for ``MultiQueryRetriever`` course scoping.

Root cause covered here: ``MultiQueryRetriever`` previously called the base
``retrieve_chunks`` WITHOUT a ``course_slug``, which forces a master-catalog
search. On-disk courses absent from ``master_catalog.json`` (e.g. DART-only
corpora whose chunks live in ``dart_chunks/``) were therefore unreachable —
``retrieve_with_decomposition`` returned zero results for queries that the
single-course BM25 path resolves fine.

The fix threads a ``course_slug`` scope through the retriever so it routes
through the same file-aware ``resolve_imscc_chunks_path`` resolver
(``imscc_chunks/`` → ``dart_chunks/`` → legacy ``corpus/``) the single path
uses. These tests assert:

* a ``dart_chunks/``-only course returns results when scoped (decompose
  on AND off);
* a legacy ``corpus/``-only course does not regress;
* the ``decompose=False`` branch returns the retrieved results (it used to
  drop them and always return an empty list).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from LibV2.tools.libv2.multi_retriever import MultiQueryRetriever


def _make_course(
    tmp_path: Path,
    slug: str,
    chunks: List[dict],
    *,
    chunkset_dir: str = "dart_chunks",
) -> Path:
    """Write a minimal course dir with chunks.jsonl in ``chunkset_dir``.

    ``chunkset_dir`` defaults to ``dart_chunks`` so the fixture mirrors the
    real DART-only corpora that exposed this bug; pass ``corpus`` to model a
    legacy archive.
    """
    course_dir = tmp_path / "courses" / slug
    (course_dir / chunkset_dir).mkdir(parents=True)
    with open(course_dir / chunkset_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    with open(course_dir / "manifest.json", "w") as f:
        json.dump({"classification": {"primary_domain": "test-domain"}}, f)
    return course_dir


def _algebra_chunks() -> List[dict]:
    return [
        {
            "id": "c1",
            "chunk_type": "definition",
            "text": (
                "A linear equation is an equation that graphs as a straight "
                "line. Solving a linear equation isolates the variable."
            ),
            "concept_tags": ["linear-equation"],
            "source": {"section_heading": "Linear Equations"},
        },
        {
            "id": "c2",
            "chunk_type": "example",
            "text": (
                "Example: solve the linear equation 2x + 3 = 7 by "
                "subtracting 3 then dividing by 2 to get x = 2."
            ),
            "concept_tags": ["linear-equation", "example"],
            "source": {"section_heading": "Solving Linear Equations"},
        },
        {
            "id": "c3",
            "chunk_type": "explanation",
            "text": (
                "The slope-intercept form of a linear equation is y = mx + b "
                "where m is the slope of the line."
            ),
            "concept_tags": ["linear-equation", "slope"],
            "source": {"section_heading": "Slope of a Linear Equation"},
        },
        # Distractor chunks on unrelated topics. These keep "linear" and
        # "equation" discriminative (BM25 IDF collapses to ~0 for a term that
        # appears in EVERY chunk), so the exact-phrase query still scores
        # above the relevance floor — mirroring a real multi-topic corpus.
        {
            "id": "d1",
            "chunk_type": "definition",
            "text": (
                "Photosynthesis converts sunlight, water, and carbon dioxide "
                "into glucose and oxygen inside plant chloroplasts."
            ),
            "concept_tags": ["photosynthesis"],
            "source": {"section_heading": "Photosynthesis"},
        },
        {
            "id": "d2",
            "chunk_type": "explanation",
            "text": (
                "The French Revolution began in 1789 and reshaped European "
                "political institutions over the following decade."
            ),
            "concept_tags": ["french-revolution"],
            "source": {"section_heading": "The French Revolution"},
        },
    ]


class TestDartChunksOnlyCourse:
    """A course whose chunks live only in ``dart_chunks/`` (no catalog entry)."""

    def test_decompose_returns_results(self, tmp_path):
        _make_course(tmp_path, "alg-dart", _algebra_chunks(), chunkset_dir="dart_chunks")
        retriever = MultiQueryRetriever(
            repo_root=tmp_path, course_slug="alg-dart",
        )
        fusion, decomposed = retriever.retrieve_with_decomposition(
            query="linear equation", limit=5,
        )
        assert fusion.result_count > 0
        assert decomposed.sub_queries  # decomposition actually fired

    def test_decompose_false_returns_results(self, tmp_path):
        """The non-decomposed branch must surface results, not an empty list
        (the pre-fix branch dropped ``results`` on the floor)."""
        _make_course(tmp_path, "alg-dart", _algebra_chunks(), chunkset_dir="dart_chunks")
        retriever = MultiQueryRetriever(
            repo_root=tmp_path, course_slug="alg-dart",
        )
        fusion = retriever.retrieve(
            query="linear equation", limit=5, decompose=False,
        )
        assert fusion.result_count > 0
        assert fusion.query_coverage.get("original", 0) > 0

    def test_per_call_slug_overrides_instance_default(self, tmp_path):
        _make_course(tmp_path, "alg-dart", _algebra_chunks(), chunkset_dir="dart_chunks")
        # No instance-level scope; supply it per call.
        retriever = MultiQueryRetriever(repo_root=tmp_path)
        fusion, _ = retriever.retrieve_with_decomposition(
            query="linear equation", limit=5, course_slug="alg-dart",
        )
        assert fusion.result_count > 0


class TestLegacyCorpusCourseNoRegression:
    """A legacy ``corpus/``-only course must still resolve."""

    def test_corpus_course_returns_results(self, tmp_path):
        _make_course(tmp_path, "alg-legacy", _algebra_chunks(), chunkset_dir="corpus")
        retriever = MultiQueryRetriever(
            repo_root=tmp_path, course_slug="alg-legacy",
        )
        fusion, _ = retriever.retrieve_with_decomposition(
            query="linear equation", limit=5,
        )
        assert fusion.result_count > 0

    def test_corpus_course_decompose_false(self, tmp_path):
        _make_course(tmp_path, "alg-legacy", _algebra_chunks(), chunkset_dir="corpus")
        retriever = MultiQueryRetriever(
            repo_root=tmp_path, course_slug="alg-legacy",
        )
        fusion = retriever.retrieve(
            query="linear equation", limit=5, decompose=False,
        )
        assert fusion.result_count > 0
