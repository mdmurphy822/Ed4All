"""
LibV2 RAG Bridge for Trainforge

Provides retrieval interface between Trainforge assessment generation
and the LibV2 RAG system. Uses LibV2's streaming TF-IDF retrieval
with multi-query decomposition for complex educational queries.
"""

import json
import logging
import os
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

from lib.libv2_storage import LIBV2_ROOT, LIBV2_COURSES, LibV2Storage

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

    def retrieve_for_objective(
        self,
        objective_text: str,
        bloom_level: str,
        top_k: int = 10,
    ) -> Tuple[List[RAGChunk], RetrievalMetrics]:
        """
        Retrieve chunks relevant to a learning objective.

        Optimized for assessment generation by considering
        the learning objective and Bloom's level.

        Args:
            objective_text: The learning objective text
            bloom_level: Target Bloom's taxonomy level
            top_k: Number of results

        Returns:
            Tuple of (chunks, metrics)
        """
        # Build query that incorporates objective and Bloom level
        # Lower Bloom levels (remember, understand) need definitions/explanations
        # Higher levels (apply, analyze, evaluate, create) need examples/applications
        chunk_type = None
        if bloom_level.lower() in ["remember", "understand"]:
            chunk_type = "explanation"
        elif bloom_level.lower() in ["apply", "analyze"]:
            chunk_type = "example"
        elif bloom_level.lower() in ["evaluate", "create"]:
            # For higher order thinking, get mixed content
            chunk_type = None

        return self.multi_query_retrieve(
            query=objective_text,
            top_k=top_k,
            chunk_type=chunk_type,
            auto_decompose=True,
        )

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the course corpus."""
        if not self.has_corpus:
            return {"exists": False, "chunk_count": 0}

        chunk_count = 0
        chunk_types = {}
        difficulties = {}
        total_tokens = 0

        with open(self.chunks_path, 'r') as f:
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
            raise ImportError(f"LibV2 retriever not available: {e}")

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
