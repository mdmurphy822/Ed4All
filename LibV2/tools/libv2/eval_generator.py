"""Evaluation set generator for LibV2.

Generates evaluation query sets by sampling representative chunks
and creating queries from their content.

Usage:
    eval_set = generate_eval_set(course_dir, num_queries=50)
"""

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .eval_harness import EvalQuery, EvalSet

logger = logging.getLogger(__name__)


@dataclass
class ChunkSample:
    """A sampled chunk for eval set generation."""
    chunk_id: str
    text: str
    chunk_type: str
    difficulty: Optional[str]
    concept_tags: List[str]
    module_title: Optional[str]
    lesson_title: Optional[str]


def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from text for query generation.

    Simple extraction based on patterns:
    - Capitalized phrases
    - Technical terms (hyphenated)
    - Quoted terms
    """
    phrases = []

    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', text)
    phrases.extend(quoted[:2])

    # Extract capitalized phrases (likely proper nouns/concepts)
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    phrases.extend(caps[:2])

    # Extract hyphenated technical terms
    hyphenated = re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', text.lower())
    phrases.extend(hyphenated[:2])

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p in phrases:
        p_lower = p.lower()
        if p_lower not in seen:
            seen.add(p_lower)
            unique.append(p)

    return unique[:max_phrases]


def generate_query_from_chunk(chunk: ChunkSample) -> Tuple[str, str]:
    """
    Generate a query and intent description from a chunk.

    Returns:
        Tuple of (query_text, intent_description)
    """
    text = chunk.text

    # Strategy 1: Use concept tags if available
    if chunk.concept_tags and len(chunk.concept_tags) >= 2:
        tags = random.sample(chunk.concept_tags, min(2, len(chunk.concept_tags)))
        query = " ".join(t.replace("-", " ") for t in tags)
        intent = f"Find content about {' and '.join(tags)}"
        return query, intent

    # Strategy 2: Extract key phrases from text
    phrases = extract_key_phrases(text)
    if phrases:
        query = " ".join(phrases[:2])
        intent = f"Find content mentioning {query}"
        return query, intent

    # Strategy 3: Use first sentence keywords
    first_sentence = text.split(".")[0] if "." in text else text[:200]
    words = first_sentence.lower().split()
    # Filter out common words
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "this", "that", "in", "on", "at", "to", "for", "of"}
    keywords = [w for w in words if w not in stop and len(w) > 3][:4]

    if keywords:
        query = " ".join(keywords)
        intent = f"Find content about {query}"
        return query, intent

    # Fallback: Use module/lesson titles
    if chunk.lesson_title:
        query = chunk.lesson_title.lower()
        intent = f"Find content from lesson: {chunk.lesson_title}"
        return query, intent

    if chunk.module_title:
        query = chunk.module_title.lower()
        intent = f"Find content from module: {chunk.module_title}"
        return query, intent

    # Last resort
    return "content overview", "General content search"


def load_chunks(course_dir: Path) -> List[ChunkSample]:
    """Load chunks from a course directory."""
    chunks_path = course_dir / "corpus" / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found at {chunks_path}")

    with open(chunks_path) as f:
        raw_chunks = json.load(f)

    if not isinstance(raw_chunks, list):
        raise ValueError("chunks.json must be a list")

    samples = []
    for chunk in raw_chunks:
        source = chunk.get("source", {})
        samples.append(ChunkSample(
            chunk_id=chunk.get("id", ""),
            text=chunk.get("text", ""),
            chunk_type=chunk.get("chunk_type", "unknown"),
            difficulty=chunk.get("difficulty"),
            concept_tags=chunk.get("concept_tags", []),
            module_title=source.get("module_title"),
            lesson_title=source.get("lesson_title"),
        ))

    return samples


def stratified_sample(
    chunks: List[ChunkSample],
    num_samples: int,
) -> List[ChunkSample]:
    """
    Sample chunks with stratification by chunk_type and difficulty.

    Ensures diverse representation across the corpus.
    """
    if len(chunks) <= num_samples:
        return chunks

    # Group by chunk_type
    by_type: Dict[str, List[ChunkSample]] = {}
    for chunk in chunks:
        t = chunk.chunk_type or "unknown"
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(chunk)

    # Calculate proportional allocation
    type_counts = {t: len(cs) for t, cs in by_type.items()}
    total = sum(type_counts.values())

    samples = []
    remaining = num_samples

    # Allocate proportionally
    for t, cs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        allocation = max(1, int(num_samples * len(cs) / total))
        allocation = min(allocation, remaining, len(cs))

        if allocation > 0:
            selected = random.sample(cs, allocation)
            samples.extend(selected)
            remaining -= allocation

        if remaining <= 0:
            break

    # Fill remaining with random samples if needed
    if remaining > 0 and len(chunks) > len(samples):
        sampled_ids = {s.chunk_id for s in samples}
        available = [c for c in chunks if c.chunk_id not in sampled_ids]
        additional = random.sample(available, min(remaining, len(available)))
        samples.extend(additional)

    return samples


def generate_eval_set(
    course_dir: Path,
    num_queries: int = 50,
    min_text_length: int = 100,
    include_difficult_queries: bool = True,
) -> EvalSet:
    """
    Generate an evaluation set for a course.

    Args:
        course_dir: Path to course directory
        num_queries: Number of evaluation queries to generate
        min_text_length: Minimum chunk text length to consider
        include_difficult_queries: Include queries targeting specific difficulties

    Returns:
        EvalSet ready for evaluation
    """
    course_slug = course_dir.name

    # Load and filter chunks
    all_chunks = load_chunks(course_dir)
    logger.info(f"Loaded {len(all_chunks)} chunks from {course_slug}")

    # Filter out very short chunks
    viable_chunks = [c for c in all_chunks if len(c.text) >= min_text_length]
    logger.info(f"Filtered to {len(viable_chunks)} viable chunks")

    if len(viable_chunks) < num_queries:
        logger.warning(
            f"Only {len(viable_chunks)} viable chunks available, "
            f"requested {num_queries} queries"
        )
        num_queries = len(viable_chunks)

    # Stratified sampling
    sampled = stratified_sample(viable_chunks, num_queries)
    logger.info(f"Sampled {len(sampled)} chunks for eval set")

    # Generate queries
    queries = []
    for i, chunk in enumerate(sampled):
        query_text, intent = generate_query_from_chunk(chunk)

        query = EvalQuery(
            query_id=f"q_{i+1:03d}",
            query_text=query_text,
            expected_chunk_ids=[chunk.chunk_id],
            chunk_type=chunk.chunk_type if include_difficult_queries else None,
            difficulty=chunk.difficulty if include_difficult_queries else None,
            notes=intent,
        )
        queries.append(query)

    # Log statistics
    type_dist = Counter(c.chunk_type for c in sampled)
    logger.info(f"Chunk type distribution: {dict(type_dist)}")

    return EvalSet(
        course_slug=course_slug,
        created_timestamp=datetime.now().isoformat(),
        queries=queries,
        description=f"Auto-generated eval set with {len(queries)} queries",
        version="1.0",
    )


def save_eval_set(
    eval_set: EvalSet,
    course_dir: Path,
) -> Path:
    """
    Save evaluation set to course quality directory.

    Args:
        eval_set: EvalSet to save
        course_dir: Path to course directory

    Returns:
        Path to saved eval set
    """
    quality_dir = course_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    eval_set_path = quality_dir / "eval_set.json"

    with open(eval_set_path, "w") as f:
        json.dump(eval_set.to_dict(), f, indent=2)

    logger.info(f"Saved eval set to {eval_set_path}")
    return eval_set_path


def generate_and_save_eval_set(
    course_dir: Path,
    num_queries: int = 50,
) -> Tuple[EvalSet, Path]:
    """
    Generate and save evaluation set for a course.

    Args:
        course_dir: Path to course directory
        num_queries: Number of queries to generate

    Returns:
        Tuple of (EvalSet, path to saved file)
    """
    eval_set = generate_eval_set(course_dir, num_queries)
    path = save_eval_set(eval_set, course_dir)
    return eval_set, path


def augment_eval_set(
    eval_set_path: Path,
    additional_queries: List[Dict],
) -> EvalSet:
    """
    Augment an existing eval set with manually curated queries.

    Args:
        eval_set_path: Path to existing eval_set.json
        additional_queries: List of query dicts to add

    Returns:
        Updated EvalSet
    """
    with open(eval_set_path) as f:
        data = json.load(f)

    eval_set = EvalSet.from_dict(data)

    # Get next query ID
    existing_ids = {q.query_id for q in eval_set.queries}
    next_id = len(eval_set.queries) + 1

    for q_data in additional_queries:
        query_id = q_data.get("query_id", f"q_{next_id:03d}")
        while query_id in existing_ids:
            next_id += 1
            query_id = f"q_{next_id:03d}"

        query = EvalQuery(
            query_id=query_id,
            query_text=q_data["query_text"],
            expected_chunk_ids=q_data["expected_chunk_ids"],
            notes=q_data.get("notes", "Manually added"),
        )
        eval_set.queries.append(query)
        existing_ids.add(query_id)
        next_id += 1

    # Update description
    eval_set.description = (
        f"Eval set with {len(eval_set.queries)} queries "
        f"(includes manually curated queries)"
    )

    return eval_set


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python eval_generator.py <course_dir> [num_queries]")
        sys.exit(1)

    course_dir = Path(sys.argv[1])
    num_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    try:
        eval_set, path = generate_and_save_eval_set(course_dir, num_queries)

        print(f"\nGenerated Eval Set for {eval_set.course_slug}")
        print("=" * 50)
        print(f"Total queries: {len(eval_set.queries)}")
        print(f"Saved to: {path}")

        # Show sample queries
        print("\nSample queries:")
        for q in eval_set.queries[:5]:
            print(f"  [{q.query_id}] {q.query_text}")
            print(f"    Expected: {q.expected_chunk_ids[0]}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
