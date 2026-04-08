"""
Trainforge RAG Query Interface

Provides query-only access to LibV2 RAG corpus. Trainforge does NOT
store or index content - it only retrieves from LibV2 for assessment
generation.

Usage:
    from Trainforge.rag import TrainforgeRAG, get_rag_for_course

    # Query-based retrieval for a specific course
    rag = get_rag_for_course("int-101")
    chunks, metrics = rag.retrieve("What is instructional design?", top_k=10)

    # Retrieve for a specific learning objective
    chunks, metrics = rag.retrieve_for_objective(
        objective_text="Explain the principles of UDL",
        bloom_level="understand"
    )

    # Cross-course retrieval
    from Trainforge.rag import get_cross_course_rag
    cross_rag = get_cross_course_rag(domain="pedagogy")
    chunks, metrics = cross_rag.retrieve("assessment strategies", top_k=20)
"""

from .libv2_bridge import (
    CrossCourseRAG,
    RAGChunk,
    RetrievalMetrics,
    TrainforgeRAG,
    get_cross_course_rag,
    get_rag_for_course,
)

__all__ = [
    'TrainforgeRAG',
    'CrossCourseRAG',
    'RAGChunk',
    'RetrievalMetrics',
    'get_rag_for_course',
    'get_cross_course_rag',
]
