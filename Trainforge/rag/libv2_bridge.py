"""
LibV2 RAG Bridge for Trainforge

Provides retrieval interface between Trainforge assessment generation
and the LibV2 RAG system. Uses LibV2's streaming BM25 retrieval
with multi-query decomposition, dual chunk-type Bloom mapping,
and multi-strategy fallback for assessment-optimized retrieval.
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Determine project root dynamically
_RAG_DIR = Path(__file__).resolve().parent
_TRAINFORGE_ROOT = _RAG_DIR.parent
_PROJECT_ROOT = _TRAINFORGE_ROOT.parent

# Add Ed4All lib to path (must be done before importing lib modules)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Add LibV2 tools to path using centralized lib.paths
try:
    from lib.paths import LIBV2_TOOLS_PATH
except ImportError:
    # Fallback if lib.paths not available
    LIBV2_TOOLS_PATH = Path(os.environ.get("ED4ALL_ROOT", _PROJECT_ROOT)) / "LibV2" / "tools"

if str(LIBV2_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(LIBV2_TOOLS_PATH))

from lib.libv2_storage import LIBV2_COURSES, LIBV2_ROOT  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class RAGChunk:
    """Normalized chunk representation for Trainforge."""
    chunk_id: str
    text: str
    score: float
    course_slug: str
    chunk_type: str
    source: Dict[str, Any] = field(default_factory=dict)
    concept_tags: List[str] = field(default_factory=list)
    difficulty: Optional[str] = None
    tokens_estimate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "course_slug": self.course_slug,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "concept_tags": self.concept_tags,
            "difficulty": self.difficulty,
            "tokens_estimate": self.tokens_estimate,
        }


@dataclass
class RetrievalMetrics:
    """Metrics from a retrieval operation (renamed from RetrievalMetrics to avoid conflict with lib.trainforge_capture.RetrievalMetrics)."""
    query: str
    chunks_retrieved: int
    chunks_used: int
    retrieval_latency_ms: float
    sub_queries: List[str] = field(default_factory=list)
    was_decomposed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_used": self.chunks_used,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "sub_queries": self.sub_queries,
            "was_decomposed": self.was_decomposed,
        }


class TrainforgeRAG:
    """
    RAG interface for Trainforge using LibV2.

    Provides methods for:
    - Single query retrieval with TF-IDF ranking
    - Multi-query retrieval with query decomposition
    - Chunk filtering by type, difficulty, concept tags
    - Course-specific and cross-course retrieval

    Usage:
        rag = TrainforgeRAG("int-101")
        chunks = rag.retrieve("What is instructional design?", top_k=10)
        for chunk in chunks:
            print(chunk.text, chunk.score)
    """

    def __init__(self, course_slug: str):
        """
        Initialize RAG interface for a course.

        Args:
            course_slug: Course slug (e.g., "int-101")
        """
        self.course_slug = course_slug
        self.course_path = LIBV2_COURSES / course_slug
        self.corpus_path = self.course_path / "corpus"
        self.chunks_path = self.corpus_path / "chunks.jsonl"

        # Lazy-loaded retrievers
        self._retriever = None
        self._multi_retriever = None
        self._retrieve_func = None  # Set by _ensure_retriever()

        # Verify course exists
        if not self.corpus_path.exists():
            logger.warning(f"Course corpus not found: {self.corpus_path}")

    @property
    def has_corpus(self) -> bool:
        """Check if course has a corpus."""
        return self.chunks_path.exists()

    def _ensure_retriever(self):
        """Lazy-load the LibV2 retriever."""
        if self._retriever is None:
            try:
                from libv2.retriever import retrieve_chunks
                self._retrieve_func = retrieve_chunks
            except ImportError as e:
                logger.error(f"Failed to import LibV2 retriever: {e}")
                raise ImportError(
                    "LibV2 retriever not available. "
                    "Ensure LibV2/tools/libv2/ is in PYTHONPATH"
                ) from e

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        chunk_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        concept_tags: Optional[List[str]] = None,
    ) -> Tuple[List[RAGChunk], RetrievalMetrics]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            chunk_type: Filter by type (explanation, example, assessment, etc.)
            difficulty: Filter by difficulty (introductory, intermediate, advanced)
            concept_tags: Filter by concept tags (any match)

        Returns:
            Tuple of (chunks, metrics)
        """
        start_time = time.time()

        self._ensure_retriever()

        # Call LibV2 retriever
        results = self._retrieve_func(
            repo_root=LIBV2_ROOT,
            query=query,
            course_slug=self.course_slug,
            chunk_type=chunk_type,
            difficulty=difficulty,
            concept_tags=concept_tags,
            limit=top_k,
        )

        # Convert to RAGChunk format
        # Handle both RetrievalResult (score) and FusedResult (fused_score)
        chunks = [
            RAGChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                score=getattr(r, 'score', getattr(r, 'fused_score', 0.0)),
                course_slug=getattr(r, 'course_slug', ''),
                chunk_type=getattr(r, 'chunk_type', ''),
                source=getattr(r, 'source', {}),
                concept_tags=getattr(r, 'concept_tags', []),
                difficulty=getattr(r, 'difficulty', None),
                tokens_estimate=getattr(r, 'tokens_estimate', 0),
            )
            for r in results
        ]

        elapsed_ms = (time.time() - start_time) * 1000
        metrics = RetrievalMetrics(
            query=query,
            chunks_retrieved=len(chunks),
            chunks_used=len(chunks),  # Will be updated by caller
            retrieval_latency_ms=elapsed_ms,
        )

        return chunks, metrics

    def multi_query_retrieve(
        self,
        query: str,
        top_k: int = 10,
        chunk_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        auto_decompose: bool = True,
    ) -> Tuple[List[RAGChunk], RetrievalMetrics]:
        """
        Multi-query retrieval with optional query decomposition.

        For complex queries (comparisons, multi-concept), decomposes
        into sub-queries and fuses results using RRF.

        Args:
            query: Search query
            top_k: Number of results to return
            chunk_type: Filter by type
            difficulty: Filter by difficulty
            auto_decompose: Whether to decompose complex queries

        Returns:
            Tuple of (chunks, metrics)
        """
        start_time = time.time()

        try:
            from libv2.multi_retriever import MultiQueryRetriever
            from libv2.query_decomposer import QueryDecomposer
        except ImportError as e:
            logger.warning(f"Multi-retriever not available: {e}, falling back to single query")
            return self.retrieve(query, top_k, chunk_type, difficulty)

        # Attempt query decomposition
        sub_queries = []
        if auto_decompose:
            try:
                decomposer = QueryDecomposer()
                decomposition = decomposer.decompose(query)
                if decomposition.sub_queries:
                    sub_queries = [sq.text for sq in decomposition.sub_queries]
            except Exception as e:
                logger.warning(f"Query decomposition failed: {e}")

        was_decomposed = bool(sub_queries)

        if was_decomposed:
            # Multi-query retrieval with RRF fusion
            retriever = MultiQueryRetriever(repo_root=LIBV2_ROOT)
            fusion_result = retriever.retrieve(
                query=query,
                limit=top_k,
                decompose=True,  # Let retriever handle decomposition
                chunk_type=chunk_type,
                difficulty=difficulty,
            )
            results = fusion_result.results
        else:
            # Fall back to single query
            self._ensure_retriever()
            results = self._retrieve_func(
                repo_root=LIBV2_ROOT,
                query=query,
                course_slug=self.course_slug,
                chunk_type=chunk_type,
                difficulty=difficulty,
                limit=top_k,
            )

        # Convert to RAGChunk format
        # Handle both RetrievalResult (score) and FusedResult (fused_score)
        chunks = [
            RAGChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                score=getattr(r, 'score', getattr(r, 'fused_score', 0.0)),
                course_slug=getattr(r, 'course_slug', ''),
                chunk_type=getattr(r, 'chunk_type', ''),
                source=getattr(r, 'source', {}),
                concept_tags=getattr(r, 'concept_tags', []),
                difficulty=getattr(r, 'difficulty', None),
                tokens_estimate=getattr(r, 'tokens_estimate', 0),
            )
            for r in results
        ]

        elapsed_ms = (time.time() - start_time) * 1000
        metrics = RetrievalMetrics(
            query=query,
            chunks_retrieved=len(chunks),
            chunks_used=len(chunks),
            retrieval_latency_ms=elapsed_ms,
            sub_queries=sub_queries,
            was_decomposed=was_decomposed,
        )

        return chunks, metrics

    # Bloom level -> (primary chunk type, secondary chunk type)
    BLOOM_CHUNK_STRATEGY = {
        "remember": ("explanation", "example"),
        "understand": ("explanation", "example"),
        "apply": ("example", "explanation"),
        "analyze": ("example", "explanation"),
        "evaluate": (None, "example"),      # unfiltered primary
        "create": (None, "explanation"),     # unfiltered primary
    }

    def retrieve_for_objective(
        self,
        objective_text: str,
        bloom_level: str,
        top_k: int = 10,
    ) -> Tuple[List[RAGChunk], RetrievalMetrics]:
        """
        Retrieve chunks relevant to a learning objective using dual
        chunk-type retrieval.

        Uses both primary and secondary chunk types for the given Bloom
        level, ensuring assessments have both conceptual and applied
        content available.

        Args:
            objective_text: The learning objective text
            bloom_level: Target Bloom's taxonomy level
            top_k: Number of results

        Returns:
            Tuple of (chunks, metrics)
        """
        primary_type, secondary_type = self.BLOOM_CHUNK_STRATEGY.get(
            bloom_level.lower(), (None, None)
        )

        # Primary retrieval with preferred chunk type
        primary_chunks, primary_metrics = self.multi_query_retrieve(
            query=objective_text,
            top_k=top_k,
            chunk_type=primary_type,
            auto_decompose=True,
        )

        # Secondary retrieval with complementary chunk type
        secondary_k = max(3, top_k // 2)
        secondary_chunks, _ = self.retrieve(
            query=objective_text,
            top_k=secondary_k,
            chunk_type=secondary_type,
        )

        # Merge: primary first, then secondary (deduplicated)
        merged = self._merge_chunk_lists(primary_chunks, secondary_chunks, top_k)

        metrics = RetrievalMetrics(
            query=primary_metrics.query,
            chunks_retrieved=len(merged),
            chunks_used=len(merged),
            retrieval_latency_ms=primary_metrics.retrieval_latency_ms,
            sub_queries=primary_metrics.sub_queries,
            was_decomposed=primary_metrics.was_decomposed,
        )

        return merged, metrics

    def retrieve_with_fallback(
        self,
        objective_text: str,
        bloom_level: str,
        top_k: int = 10,
        min_chunks: int = 3,
    ) -> Tuple[List[RAGChunk], RetrievalMetrics]:
        """
        Multi-strategy retrieval with fallback chain.

        Tries increasingly relaxed strategies until enough chunks are found:
        1. Bloom-aware dual chunk-type retrieval
        2. Concept-only query without chunk type filter
        3. Cross-course retrieval

        Args:
            objective_text: The learning objective text
            bloom_level: Target Bloom's taxonomy level
            top_k: Number of results desired
            min_chunks: Minimum chunks before triggering fallback

        Returns:
            Tuple of (chunks, metrics)
        """
        # Strategy 1: Full objective text + Bloom-aware chunk types
        chunks, metrics = self.retrieve_for_objective(
            objective_text, bloom_level, top_k
        )
        if len(chunks) >= min_chunks:
            return chunks, metrics

        logger.info(
            "Fallback: primary retrieval returned %d chunks (need %d), "
            "trying concept extraction",
            len(chunks), min_chunks,
        )

        # Strategy 2: Extract key concepts and retry without type filter
        concepts = self._extract_query_concepts(objective_text)
        if concepts and concepts != objective_text:
            more_chunks, _ = self.retrieve(
                query=concepts, top_k=top_k, chunk_type=None
            )
            chunks = self._merge_chunk_lists(chunks, more_chunks, top_k)
            if len(chunks) >= min_chunks:
                return chunks, metrics

        logger.info(
            "Fallback: concept retrieval returned %d total, trying cross-course",
            len(chunks),
        )

        # Strategy 3: Cross-course retrieval
        cross_rag = CrossCourseRAG()
        cross_chunks, _ = cross_rag.retrieve(
            query=objective_text, top_k=top_k, sample_per_course=3
        )
        chunks = self._merge_chunk_lists(chunks, cross_chunks, top_k)

        return chunks, metrics

    @staticmethod
    def _merge_chunk_lists(
        primary: List[RAGChunk],
        secondary: List[RAGChunk],
        limit: int,
    ) -> List[RAGChunk]:
        """Merge two chunk lists, deduplicating by chunk_id, capped at limit."""
        seen_ids: set = set()
        merged: List[RAGChunk] = []
        for chunk in primary:
            if len(merged) >= limit:
                break
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                merged.append(chunk)
        for chunk in secondary:
            if len(merged) >= limit:
                break
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                merged.append(chunk)
        return merged

    @staticmethod
    def _extract_query_concepts(text: str) -> str:
        """Extract key concept words from objective text.

        Strips common objective preamble (verbs, articles) to focus
        on the domain-specific content words.
        """
        # Remove common Bloom preamble verbs
        preamble = (
            r"^(?:students?\s+(?:will|should|can)\s+(?:be\s+able\s+to\s+)?)?",
        )
        cleaned = re.sub(preamble[0], "", text, flags=re.IGNORECASE).strip()

        # Remove leading Bloom verbs
        bloom_verbs = (
            "define|list|recall|identify|explain|describe|summarize|"
            "apply|demonstrate|use|solve|analyze|compare|contrast|"
            "evaluate|judge|justify|create|design|develop"
        )
        cleaned = re.sub(
            rf"^(?:{bloom_verbs})\s+", "", cleaned, flags=re.IGNORECASE
        ).strip()

        return cleaned if cleaned else text

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the course corpus."""
        if not self.has_corpus:
            return {"exists": False, "chunk_count": 0}

        chunk_count = 0
        chunk_types = {}
        difficulties = {}
        total_tokens = 0

        with open(self.chunks_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    chunk_count += 1

                    ct = chunk.get("chunk_type", "unknown")
                    chunk_types[ct] = chunk_types.get(ct, 0) + 1

                    diff = chunk.get("difficulty", "unspecified")
                    difficulties[diff] = difficulties.get(diff, 0) + 1

                    total_tokens += chunk.get("tokens_estimate", 0)
                except json.JSONDecodeError:
                    continue

        return {
            "exists": True,
            "chunk_count": chunk_count,
            "chunk_types": chunk_types,
            "difficulties": difficulties,
            "total_tokens_estimate": total_tokens,
        }


class CrossCourseRAG:
    """
    Cross-course retrieval for finding content across multiple courses.

    Useful for:
    - Finding examples from related courses
    - Cross-referencing concepts
    - Building comprehensive assessments
    """

    def __init__(self, domain: Optional[str] = None):
        """
        Initialize cross-course retriever.

        Args:
            domain: Optional domain filter (e.g., "pedagogy", "physics")
        """
        self.domain = domain

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        sample_per_course: int = 5,
        chunk_type: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Tuple[List[RAGChunk], RetrievalMetrics]:
        """
        Retrieve chunks across multiple courses.

        Args:
            query: Search query
            top_k: Total results to return
            sample_per_course: Max chunks per course
            chunk_type: Filter by type
            difficulty: Filter by difficulty

        Returns:
            Tuple of (chunks, metrics)
        """
        start_time = time.time()

        try:
            from libv2.retriever import retrieve_chunks
        except ImportError as e:
            raise ImportError(f"LibV2 retriever not available: {e}") from e

        results = retrieve_chunks(
            repo_root=LIBV2_ROOT,
            query=query,
            domain=self.domain,
            chunk_type=chunk_type,
            difficulty=difficulty,
            limit=top_k,
            sample_per_course=sample_per_course,
        )

        chunks = [
            RAGChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                course_slug=r.course_slug,
                chunk_type=r.chunk_type,
                source=r.source,
                concept_tags=r.concept_tags,
                difficulty=r.difficulty,
                tokens_estimate=r.tokens_estimate,
            )
            for r in results
        ]

        elapsed_ms = (time.time() - start_time) * 1000
        metrics = RetrievalMetrics(
            query=query,
            chunks_retrieved=len(chunks),
            chunks_used=len(chunks),
            retrieval_latency_ms=elapsed_ms,
        )

        return chunks, metrics


def get_rag_for_course(course_slug: str) -> TrainforgeRAG:
    """Get RAG interface for a specific course."""
    return TrainforgeRAG(course_slug)


def get_cross_course_rag(domain: Optional[str] = None) -> CrossCourseRAG:
    """Get cross-course RAG interface."""
    return CrossCourseRAG(domain)


__all__ = [
    'TrainforgeRAG',
    'CrossCourseRAG',
    'RAGChunk',
    'RetrievalMetrics',
    'get_rag_for_course',
    'get_cross_course_rag',
]
