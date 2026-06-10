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


# Defaults for the precision-controlled linker. Kept as module constants so
# callers / tests can reference the canonical values.
DEFAULT_MAX_OUTCOMES_PER_CHUNK = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.20
# An outcome linked to more than this fraction of all chunks is treated as
# "too broad" and pruned down to the chunks where it is a top-ranked match.
BROAD_OUTCOME_FREQUENCY = 0.40
# When an outcome is pruned for being too broad, keep it only on chunks where
# it ranks at or above this position (1 = top-1 only, 2 = top-2).
BROAD_OUTCOME_KEEP_RANK = 2
# Extra margin above the threshold below which a zero-concept-overlap link is
# considered marginal and dropped.
MARGINAL_MARGIN = 0.05


def _concept_overlap(outcome: LearningOutcome, chunk: Dict[str, Any]) -> bool:
    """Return True if the outcome's key_concepts share ≥1 token with the
    chunk's concept_tags (token-level, case-insensitive)."""
    chunk_tags = chunk.get("concept_tags") or []
    if not outcome.key_concepts or not chunk_tags:
        return False

    def _tokens(values: List[str]) -> set:
        toks: set = set()
        for v in values:
            toks.update(tokenize(str(v)))
        return toks

    outcome_tokens = _tokens(outcome.key_concepts)
    tag_tokens = _tokens(chunk_tags)
    return bool(outcome_tokens & tag_tokens)


def link_chunks_to_outcomes(
    chunks: List[Dict[str, Any]],
    outcomes: List[LearningOutcome],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_outcomes_per_chunk: int = DEFAULT_MAX_OUTCOMES_PER_CHUNK,
    broad_outcome_frequency: float = BROAD_OUTCOME_FREQUENCY,
    broad_outcome_keep_rank: int = BROAD_OUTCOME_KEEP_RANK,
    use_concept_gate: bool = True,
) -> List[Dict[str, Any]]:
    """
    Link each chunk to relevant learning outcomes using TF-IDF similarity,
    with precision controls to prevent broad objectives from over-linking.

    Precision controls (all default-valued, signature stays backward
    compatible):

    1. Per-chunk top-K cap — keep only the ``max_outcomes_per_chunk`` highest-
       scoring outcomes per chunk, ranked by score descending.
    2. Raised floor — ``similarity_threshold`` defaults to 0.20 (overridable).
    3. Global-frequency anti-signal — after scoring all chunks, any outcome
       linked to more than ``broad_outcome_frequency`` of chunks is too broad;
       it is kept only on chunks where it ranks within
       ``broad_outcome_keep_rank`` (its top-1/top-2 match), and dropped from
       the marginal attachments.
    4. Key-concept overlap gate — when ``use_concept_gate`` and an outcome's
       ``key_concepts`` share ≥1 token with the chunk's ``concept_tags``, the
       link is kept even at a lower score; with zero overlap AND a marginal
       score (< threshold + 0.05), the link is dropped.

    Args:
        chunks: List of chunk dictionaries
        outcomes: List of LearningOutcome objects
        similarity_threshold: Minimum similarity score to link
        max_outcomes_per_chunk: Maximum outcomes to link per chunk
        broad_outcome_frequency: Fraction-of-chunks ceiling above which an
            outcome is pruned to its top-ranked chunks
        broad_outcome_keep_rank: Rank cutoff retained when pruning a broad
            outcome (1 = top-1 only, 2 = top-2)
        use_concept_gate: Enable the key-concept overlap boost/gate

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

    # Pass 1: score every chunk against every outcome and build the per-chunk
    # ranked candidate list (post top-K cap + concept gate). We also track, for
    # each outcome, the rank at which it appears in each chunk so the global-
    # frequency pruner can decide which attachments to keep.
    #
    # candidates[chunk_pos] = list of (outcome_idx, score, rank) kept for chunk
    # outcome_chunk_rank[outcome_idx] = {chunk_pos: rank}
    candidates: Dict[int, List[Tuple[int, float, int]]] = {}
    outcome_chunk_rank: Dict[int, Dict[int, int]] = {}

    for chunk_pos, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            chunk["learning_outcome_refs"] = []
            continue

        # Rank ALL outcomes for this chunk so we know true top-1/top-2 even for
        # links that the per-chunk cap or threshold later drops.
        ranked = index.search(chunk_text, limit=len(outcomes))

        kept: List[Tuple[int, float, int]] = []
        for rank, (outcome_idx, score) in enumerate(ranked, start=1):
            outcome = outcomes[outcome_idx]
            overlap = _concept_overlap(outcome, chunk) if use_concept_gate else False

            # Concept-overlap boost: a shared concept token between the
            # outcome's key_concepts and the chunk's concept_tags is strong
            # topical evidence, so keep the link even well below the floor
            # (down to any positive TF-IDF similarity). The per-chunk top-K cap
            # and the global-frequency pruner still bound how many such links
            # survive.
            if overlap:
                if score <= 0.0:
                    continue
            else:
                if score < similarity_threshold:
                    continue
                # Zero overlap + marginal score -> drop (concept gate).
                if use_concept_gate and score < similarity_threshold + MARGINAL_MARGIN:
                    continue

            kept.append((outcome_idx, score, rank))
            outcome_chunk_rank.setdefault(outcome_idx, {})[chunk_pos] = rank
            if len(kept) >= max_outcomes_per_chunk:
                break

        candidates[chunk_pos] = kept

    # Pass 2: global-frequency anti-signal. Identify outcomes attached to more
    # than `broad_outcome_frequency` of all chunks; for those, keep only the
    # attachments where the outcome ranks within `broad_outcome_keep_rank`.
    total_chunks = len(chunks) if chunks else 1
    broad_threshold_count = broad_outcome_frequency * total_chunks

    drop_for_outcome: Dict[int, set] = {}
    for outcome_idx, chunk_ranks in outcome_chunk_rank.items():
        if len(chunk_ranks) > broad_threshold_count:
            drop = {
                cpos
                for cpos, rank in chunk_ranks.items()
                if rank > broad_outcome_keep_rank
            }
            if drop:
                drop_for_outcome[outcome_idx] = drop
                logger.info(
                    "Outcome %s over-linked (%d/%d chunks); pruning %d marginal "
                    "attachments below rank %d",
                    outcomes[outcome_idx].objective_id,
                    len(chunk_ranks),
                    total_chunks,
                    len(drop),
                    broad_outcome_keep_rank,
                )

    # Pass 3: materialize refs onto chunks, applying the broad-outcome drops.
    linked_count = 0
    for chunk_pos, chunk in enumerate(chunks):
        kept = candidates.get(chunk_pos, [])
        outcome_refs = [
            outcomes[outcome_idx].objective_id
            for (outcome_idx, _score, _rank) in kept
            if chunk_pos not in drop_for_outcome.get(outcome_idx, ())
        ]
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
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
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
    # Phase 7c: prefer imscc_chunks/, fall back to legacy corpus/.
    from lib.libv2_storage import resolve_imscc_chunks_path
    chunks_path = resolve_imscc_chunks_path(course_dir, "chunks.json")
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
