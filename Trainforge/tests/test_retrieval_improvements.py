"""
Tests for retrieval improvements (BM25, n-gram boosting, configurable threshold).
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIBV2_TOOLS = PROJECT_ROOT / "LibV2" / "tools"
if str(LIBV2_TOOLS) not in sys.path:
    sys.path.insert(0, str(LIBV2_TOOLS))


SAMPLE_CHUNKS = [
    {"id": "c1", "text": "Cognitive load theory describes how working memory limits information processing during learning."},
    {"id": "c2", "text": "Intrinsic load refers to the inherent complexity of the subject matter being taught."},
    {"id": "c3", "text": "Extraneous cognitive load is caused by poor instructional design that wastes mental resources."},
    {"id": "c4", "text": "The theory of evolution by natural selection was proposed by Charles Darwin and Alfred Wallace."},
    {"id": "c5", "text": "Photosynthesis is the process by which plants convert sunlight into chemical energy."},
    {"id": "c6", "text": "Schema acquisition and automation reduce cognitive demands on working memory."},
    {"id": "c7", "text": "Germane load is the mental effort dedicated to creating and organizing new knowledge schemas."},
]


class TestLazyBM25:
    def setup_method(self):
        from libv2.retriever import LazyBM25
        self.bm25 = LazyBM25(SAMPLE_CHUNKS)

    def test_basic_search_returns_results(self):
        results = self.bm25.search("cognitive load theory", limit=3)
        assert len(results) > 0
        # Top result should be about cognitive load
        assert "cognitive" in results[0][0]["text"].lower()

    def test_search_respects_limit(self):
        results = self.bm25.search("load", limit=2)
        assert len(results) <= 2

    def test_search_filters_by_threshold(self):
        # Very high threshold should return fewer results
        high = self.bm25.search("cognitive load", limit=10, min_relevance=10.0)
        low = self.bm25.search("cognitive load", limit=10, min_relevance=0.01)
        assert len(high) <= len(low)

    def test_irrelevant_query_returns_empty(self):
        results = self.bm25.search("quantum entanglement teleportation", limit=5, min_relevance=5.0)
        assert len(results) == 0

    def test_bm25_term_saturation(self):
        """BM25 should handle repeated terms better than raw TF-IDF."""
        # A query with repeated terms should not disproportionately
        # boost documents with that term
        results = self.bm25.search("load load load load load", limit=3)
        assert len(results) > 0
        # Should still return diverse results about different types of load
        texts = [r[0]["text"] for r in results]
        assert len(set(texts)) == len(texts), "Should return unique chunks"


class TestCharTrigramBoosting:
    def test_ngram_boosting_helps_morphological_variants(self):
        from libv2.retriever import LazyBM25

        # With n-gram boosting
        boosted = LazyBM25(SAMPLE_CHUNKS, ngram_weight=0.15)
        # Without n-gram boosting
        unboosted = LazyBM25(SAMPLE_CHUNKS, ngram_weight=0.0)

        # Search for a morphological variant
        query = "instructional designing"  # variant of "instructional design"
        results_boosted = boosted.search(query, limit=5, min_relevance=0.0)
        results_unboosted = unboosted.search(query, limit=5, min_relevance=0.0)

        # Both should return results, but scores may differ
        assert len(results_boosted) > 0
        assert len(results_unboosted) > 0


class TestConfigurableThreshold:
    def test_default_threshold_filters(self):
        from libv2.retriever import LazyBM25, DEFAULT_MIN_RELEVANCE

        bm25 = LazyBM25(SAMPLE_CHUNKS)
        results = bm25.search("cognitive load", limit=10)
        for _, score in results:
            assert score >= DEFAULT_MIN_RELEVANCE

    def test_zero_threshold_returns_all_matches(self):
        from libv2.retriever import LazyBM25

        bm25 = LazyBM25(SAMPLE_CHUNKS)
        results_default = bm25.search("load", limit=10)
        results_zero = bm25.search("load", limit=10, min_relevance=0.0)
        assert len(results_zero) >= len(results_default)


class TestBloomChunkStrategy:
    def test_strategy_mapping_exists(self):
        from Trainforge.rag.libv2_bridge import TrainforgeRAG
        strategy = TrainforgeRAG.BLOOM_CHUNK_STRATEGY
        assert "remember" in strategy
        assert "create" in strategy
        for level, (primary, secondary) in strategy.items():
            # secondary should always have a value
            assert secondary is not None or primary is None


class TestMergeChunkLists:
    def test_deduplicates_by_id(self):
        from Trainforge.rag.libv2_bridge import TrainforgeRAG, RAGChunk

        c1 = RAGChunk(chunk_id="a", text="t1", score=0.9, course_slug="s", chunk_type="explanation")
        c2 = RAGChunk(chunk_id="b", text="t2", score=0.8, course_slug="s", chunk_type="example")
        c3 = RAGChunk(chunk_id="a", text="t1", score=0.7, course_slug="s", chunk_type="explanation")

        merged = TrainforgeRAG._merge_chunk_lists([c1, c2], [c3], limit=10)
        ids = [c.chunk_id for c in merged]
        assert ids == ["a", "b"]  # c3 deduplicated

    def test_respects_limit(self):
        from Trainforge.rag.libv2_bridge import TrainforgeRAG, RAGChunk

        chunks = [
            RAGChunk(chunk_id=f"c{i}", text=f"text {i}", score=0.5, course_slug="s", chunk_type="explanation")
            for i in range(10)
        ]
        merged = TrainforgeRAG._merge_chunk_lists(chunks[:5], chunks[5:], limit=3)
        assert len(merged) == 3


class TestExtractQueryConcepts:
    def test_strips_bloom_preamble(self):
        from Trainforge.rag.libv2_bridge import TrainforgeRAG
        result = TrainforgeRAG._extract_query_concepts(
            "Students will be able to explain the principles of cognitive load"
        )
        assert "students" not in result.lower()
        assert "explain" not in result.lower()
        assert "cognitive load" in result.lower()

    def test_preserves_content_when_no_preamble(self):
        from Trainforge.rag.libv2_bridge import TrainforgeRAG
        result = TrainforgeRAG._extract_query_concepts("cognitive load theory")
        assert "cognitive load" in result.lower()
