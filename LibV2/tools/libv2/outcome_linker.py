"""
Learning Outcome Linker for LibV2.

Links learning outcomes from Courseforge to LibV2 chunks using TF-IDF similarity.

Pipeline:
    Courseforge objectives → outcome_linker → LibV2 chunks with learning_outcome_refs
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LearningOutcome:
    """A learning outcome extracted from Courseforge."""
    objective_id: str
    statement: str
    bloom_level: str
    key_concepts: List[str] = field(default_factory=list)
    hierarchy_level: str = "course"  # course, chapter, section
    source_reference: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.objective_id,
            "statement": self.statement,
            "bloom_level": self.bloom_level,
            "key_concepts": self.key_concepts,
            "hierarchy_level": self.hierarchy_level,
        }


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Lowercase and extract words
    words = re.findall(r'\b[a-z][a-z0-9]+\b', text.lower())
    return words


class SimpleTFIDF:
    """Simple TF-IDF implementation for outcome matching."""

    def __init__(self, documents: List[str]):
        """
        Initialize TF-IDF index.

        Args:
            documents: List of document texts to index
        """
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_tokens: List[List[str]] = []
        self.df: Counter = Counter()  # Document frequency
        self.doc_tfidf: List[Dict[str, float]] = []

        self._build_index()

    def _build_index(self) -> None:
        """Build TF-IDF index for all documents."""
        # Tokenize documents
        for doc in self.documents:
            tokens = tokenize(doc)
            self.doc_tokens.append(tokens)
            # Update document frequency (count unique terms per doc)
            self.df.update(set(tokens))

        # Compute TF-IDF for each document
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            total_terms = len(tokens) if tokens else 1
            tfidf = {}

            for term, count in tf.items():
                # TF: term frequency (normalized)
                tf_val = count / total_terms
                # IDF: inverse document frequency
                idf_val = math.log((self.doc_count + 1) / (self.df[term] + 1)) + 1
                tfidf[term] = tf_val * idf_val

            self.doc_tfidf.append(tfidf)

    def search(self, query: str, limit: int = 5) -> List[Tuple[int, float]]:
        """
        Search for documents similar to query.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of (doc_index, similarity_score) tuples, sorted by score descending
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Compute query TF-IDF
        query_tf = Counter(query_tokens)
        total_terms = len(query_tokens)
        query_tfidf = {}

        for term, count in query_tf.items():
            tf_val = count / total_terms
            idf_val = math.log((self.doc_count + 1) / (self.df.get(term, 0) + 1)) + 1
            query_tfidf[term] = tf_val * idf_val

        # Compute cosine similarity with each document
        scores = []
        query_norm = math.sqrt(sum(v * v for v in query_tfidf.values()))

        for doc_idx, doc_tfidf in enumerate(self.doc_tfidf):
            if not doc_tfidf:
                scores.append((doc_idx, 0.0))
                continue

            # Dot product
            dot = sum(query_tfidf.get(term, 0) * weight
                     for term, weight in doc_tfidf.items())

            # Document norm
            doc_norm = math.sqrt(sum(v * v for v in doc_tfidf.values()))

            # Cosine similarity
            if query_norm > 0 and doc_norm > 0:
                similarity = dot / (query_norm * doc_norm)
            else:
                similarity = 0.0

            scores.append((doc_idx, similarity))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])
        return scores[:limit]


def load_courseforge_objectives(objectives_path: Path) -> List[LearningOutcome]:
    """
    Load learning objectives from Courseforge JSON file.

    Args:
        objectives_path: Path to learning_objectives.json

    Returns:
        List of LearningOutcome objects
    """
    with open(objectives_path) as f:
        doc = json.load(f)

    outcomes = []

    # Course-level objectives
    for obj in doc.get("courseObjectives", []):
        outcomes.append(LearningOutcome(
            objective_id=obj["objectiveId"],
            statement=obj["statement"],
            bloom_level=obj.get("bloomLevel", "understand"),
            key_concepts=obj.get("keyConcepts", []),
            hierarchy_level="course",
        ))

    # Chapter and section objectives
    for chapter in doc.get("chapters", []):
        # Chapter objectives
        for obj in chapter.get("chapterObjectives", []):
            outcomes.append(LearningOutcome(
                objective_id=obj["objectiveId"],
                statement=obj["statement"],
                bloom_level=obj.get("bloomLevel", "understand"),
                key_concepts=obj.get("keyConcepts", []),
                hierarchy_level="chapter",
            ))

        # Section objectives
        for section in chapter.get("sections", []):
            for obj in section.get("sectionObjectives", []):
                outcomes.append(LearningOutcome(
                    objective_id=obj["objectiveId"],
                    statement=obj["statement"],
                    bloom_level=obj.get("bloomLevel", "understand"),
                    key_concepts=obj.get("keyConcepts", []),
                    hierarchy_level="section",
                ))

            # Subsection objectives
            for subsection in section.get("subsections", []):
                for obj in subsection.get("subsectionObjectives", []):
                    outcomes.append(LearningOutcome(
                        objective_id=obj["objectiveId"],
                        statement=obj["statement"],
                        bloom_level=obj.get("bloomLevel", "understand"),
                        key_concepts=obj.get("keyConcepts", []),
                        hierarchy_level="subsection",
                    ))

    return outcomes


def link_chunks_to_outcomes(
    chunks: List[Dict[str, Any]],
    outcomes: List[LearningOutcome],
    similarity_threshold: float = 0.15,
    max_outcomes_per_chunk: int = 3,
) -> List[Dict[str, Any]]:
    """
    Link each chunk to relevant learning outcomes using TF-IDF similarity.

    Args:
        chunks: List of chunk dictionaries
        outcomes: List of LearningOutcome objects
        similarity_threshold: Minimum similarity score to link
        max_outcomes_per_chunk: Maximum outcomes to link per chunk

    Returns:
        Updated chunks with learning_outcome_refs populated
    """
    if not outcomes:
        logger.warning("No outcomes provided, chunks will not be linked")
        return chunks

    # Build outcome search texts (statement + concepts)
    outcome_texts = [
        o.statement + " " + " ".join(o.key_concepts)
        for o in outcomes
    ]

    # Build TF-IDF index on outcomes
    index = SimpleTFIDF(outcome_texts)

    linked_count = 0
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        # Search for matching outcomes
        matches = index.search(chunk_text, limit=max_outcomes_per_chunk * 2)

        # Filter by threshold and limit
        outcome_refs = []
        for outcome_idx, score in matches:
            if score >= similarity_threshold:
                outcome_refs.append(outcomes[outcome_idx].objective_id)
                if len(outcome_refs) >= max_outcomes_per_chunk:
                    break

        chunk["learning_outcome_refs"] = outcome_refs
        if outcome_refs:
            linked_count += 1

    logger.info(f"Linked {linked_count}/{len(chunks)} chunks to outcomes")
    return chunks


def populate_course_outcomes(
    course_dir: Path,
    objectives_path: Path,
) -> Dict[str, Any]:
    """
    Populate course.json with learning outcomes from Courseforge.

    Args:
        course_dir: Path to LibV2 course directory
        objectives_path: Path to Courseforge learning_objectives.json

    Returns:
        Updated course data
    """
    outcomes = load_courseforge_objectives(objectives_path)

    course_json_path = course_dir / "course.json"

    if course_json_path.exists():
        with open(course_json_path) as f:
            course = json.load(f)
    else:
        course = {}

    # Add course-level outcomes
    course["learning_outcomes"] = [
        o.to_dict()
        for o in outcomes
        if o.hierarchy_level == "course"
    ]

    # Add all outcomes for reference
    course["all_learning_outcomes"] = [o.to_dict() for o in outcomes]

    return course


def link_course_outcomes(
    course_dir: Path,
    objectives_path: Path,
    similarity_threshold: float = 0.15,
) -> Dict[str, int]:
    """
    Full outcome linking for a course.

    1. Loads outcomes from Courseforge
    2. Updates course.json with outcomes
    3. Links chunks to outcomes

    Args:
        course_dir: Path to LibV2 course directory
        objectives_path: Path to Courseforge learning_objectives.json
        similarity_threshold: Minimum similarity for linking

    Returns:
        Statistics about the linking
    """
    # Load outcomes
    outcomes = load_courseforge_objectives(objectives_path)
    logger.info(f"Loaded {len(outcomes)} learning outcomes")

    # Update course.json
    course = populate_course_outcomes(course_dir, objectives_path)
    course_json_path = course_dir / "course.json"
    with open(course_json_path, "w") as f:
        json.dump(course, f, indent=2)

    # Load and link chunks
    chunks_path = course_dir / "corpus" / "chunks.json"
    if not chunks_path.exists():
        logger.error(f"chunks.json not found at {chunks_path}")
        return {"outcomes_loaded": len(outcomes), "chunks_linked": 0}

    with open(chunks_path) as f:
        chunks = json.load(f)

    linked_chunks = link_chunks_to_outcomes(
        chunks, outcomes, similarity_threshold
    )

    # Write updated chunks
    with open(chunks_path, "w") as f:
        json.dump(linked_chunks, f, indent=2)

    # Calculate statistics
    chunks_with_refs = sum(1 for c in linked_chunks if c.get("learning_outcome_refs"))
    coverage = chunks_with_refs / len(chunks) if chunks else 0

    stats = {
        "outcomes_loaded": len(outcomes),
        "course_level_outcomes": len(course.get("learning_outcomes", [])),
        "total_chunks": len(chunks),
        "chunks_linked": chunks_with_refs,
        "coverage_percent": round(coverage * 100, 1),
    }

    logger.info(f"Outcome linking complete: {stats}")
    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python outcome_linker.py <course_dir> <objectives_path>")
        sys.exit(1)

    course_dir = Path(sys.argv[1])
    objectives_path = Path(sys.argv[2])

    stats = link_course_outcomes(course_dir, objectives_path)
    print(json.dumps(stats, indent=2))
