"""Concept vocabulary governance for LibV2.

Provides controlled vocabulary validation and normalization for concept tags.
Prevents vocabulary explosion and ensures tag consistency.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Stopwords that should never appear in concept tags
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these',
    'those', 'it', 'its', 'as', 'if', 'when', 'where', 'how', 'what',
    'which', 'who', 'whom', 'why', 'can', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
    'there', 'then', 'once', 'any', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'while', 'your', 'you', 'we', 'our', 'they', 'their',
    'students', 'learners', 'users', 'people', 'things', 'example',
    'examples', 'understanding', 'using', 'used', 'use',
}

# Pattern for valid concept tags: lowercase, hyphenated, 1-4 words
VALID_TAG_PATTERN = re.compile(r'^[a-z][a-z0-9]*(-[a-z0-9]+){0,3}$')


def _resolve_project_schemas_dir(repo_root: Path) -> Path:
    """Resolve the project-root /schemas/ directory from an arbitrary repo_root.

    Ontology data (previously under ``LibV2/ontology/``) is now unified under
    ``<project-root>/schemas/taxonomies/``. This helper resolves the location
    regardless of whether the caller supplied a LibV2 directory or the
    project root as ``repo_root``.
    """
    try:
        from lib.paths import SCHEMAS_PATH  # type: ignore
        if SCHEMAS_PATH.exists():
            return SCHEMAS_PATH
    except Exception:
        pass
    if (repo_root / "courses").exists() and (repo_root.parent / "schemas").exists():
        return repo_root.parent / "schemas"
    return repo_root / "schemas"

# Patterns for clearly invalid content
INVALID_PATTERNS = [
    re.compile(r'^#+\s'),          # Markdown headers
    re.compile(r'^\*+'),           # Markdown bold/italic
    re.compile(r'^`'),             # Markdown code
    re.compile(r'^<[^>]+>'),       # HTML tags
    re.compile(r'^\d+$'),          # Numbers only
    re.compile(r'^[a-z]:\w+'),     # OOXML tokens like a:lnB
    re.compile(r'^p:\w+'),         # PowerPoint OOXML
    re.compile(r'^\w+:\w+:'),      # Namespaced tokens
]


@dataclass
class NormalizationResult:
    """Result of normalizing a concept tag."""
    original: str
    normalized: Optional[str]
    is_valid: bool
    reason: Optional[str] = None


@dataclass
class VocabularyAnalysis:
    """Analysis results for a corpus's concept vocabulary."""
    total_tags: int
    unique_tags: int
    valid_tags: int
    invalid_tags: int
    in_taxonomy: int
    not_in_taxonomy: int
    format_violations: List[Tuple[str, str]]  # (tag, reason)
    top_tags: List[Tuple[str, int]]  # (tag, count)
    top_invalid: List[Tuple[str, str, int]]  # (tag, reason, count)


class ConceptVocabulary:
    """Controlled vocabulary for concept tags.

    Loads canonical terms from taxonomy and provides normalization/validation.
    """

    def __init__(self, taxonomy_path: Optional[Path] = None):
        """
        Initialize vocabulary from taxonomy.

        Args:
            taxonomy_path: Path to taxonomy.json. If None, uses empty vocabulary.
        """
        self.canonical_terms: Set[str] = set()
        self.aliases: Dict[str, str] = {}
        self.taxonomy_path = taxonomy_path

        if taxonomy_path and taxonomy_path.exists():
            self._load_taxonomy(taxonomy_path)

    def _load_taxonomy(self, taxonomy_path: Path) -> None:
        """Load canonical terms from taxonomy.json."""
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

        divisions = taxonomy.get("divisions", {})

        for _division_name, division_data in divisions.items():
            domains = division_data.get("domains", {})

            for domain_key, domain_data in domains.items():
                # Add domain as canonical term
                self.canonical_terms.add(domain_key)

                subdomains = domain_data.get("subdomains", {})
                for subdomain_key, subdomain_data in subdomains.items():
                    # Add subdomain as canonical term
                    self.canonical_terms.add(subdomain_key)

                    # Add topics as canonical terms
                    topics = subdomain_data.get("topics", [])
                    for topic in topics:
                        self.canonical_terms.add(topic)

        logger.info(f"Loaded {len(self.canonical_terms)} canonical terms from taxonomy")

    def add_alias(self, alias: str, canonical: str) -> None:
        """Add an alias that maps to a canonical term."""
        self.aliases[alias.lower()] = canonical

    def normalize(self, tag: str) -> NormalizationResult:
        """
        Normalize and validate a concept tag.

        Args:
            tag: Raw concept tag to normalize

        Returns:
            NormalizationResult with normalized tag and validity
        """
        original = tag

        # Check for clearly invalid patterns first
        for pattern in INVALID_PATTERNS:
            if pattern.match(tag):
                return NormalizationResult(
                    original=original,
                    normalized=None,
                    is_valid=False,
                    reason="invalid_pattern"
                )

        # Strip and lowercase
        tag = tag.strip().lower()

        # Empty check
        if not tag:
            return NormalizationResult(
                original=original,
                normalized=None,
                is_valid=False,
                reason="empty"
            )

        # Check if it's pure stopwords
        words = tag.replace('-', ' ').split()
        if all(w in STOPWORDS for w in words):
            return NormalizationResult(
                original=original,
                normalized=None,
                is_valid=False,
                reason="stopwords_only"
            )

        # Normalize: replace spaces/underscores with hyphens
        tag = re.sub(r'[\s_]+', '-', tag)

        # Remove non-alphanumeric except hyphens
        tag = re.sub(r'[^a-z0-9-]', '', tag)

        # Collapse multiple hyphens
        tag = re.sub(r'-+', '-', tag)

        # Strip leading/trailing hyphens
        tag = tag.strip('-')

        # Check format validity
        if not VALID_TAG_PATTERN.match(tag):
            return NormalizationResult(
                original=original,
                normalized=tag if tag else None,
                is_valid=False,
                reason="invalid_format"
            )

        # Check word count (1-4 words)
        word_count = len(tag.split('-'))
        if word_count > 4:
            return NormalizationResult(
                original=original,
                normalized=tag,
                is_valid=False,
                reason="too_many_words"
            )

        # Check for alias
        if tag in self.aliases:
            tag = self.aliases[tag]

        # Check if in taxonomy (optional - tags not in taxonomy can still be valid)
        in_taxonomy = tag in self.canonical_terms

        return NormalizationResult(
            original=original,
            normalized=tag,
            is_valid=True,
            reason="in_taxonomy" if in_taxonomy else "valid_format"
        )

    def is_valid(self, tag: str) -> bool:
        """Quick check if a tag is valid."""
        return self.normalize(tag).is_valid

    def analyze_corpus(self, chunks_path: Path) -> VocabularyAnalysis:
        """
        Analyze concept vocabulary usage in a corpus.

        Args:
            chunks_path: Path to chunks.json

        Returns:
            VocabularyAnalysis with statistics
        """
        with open(chunks_path) as f:
            chunks = json.load(f)

        if not isinstance(chunks, list):
            raise ValueError("chunks.json must be a list")

        tag_counts: Counter = Counter()
        invalid_tags: Dict[str, Tuple[str, int]] = {}  # tag -> (reason, count)
        valid_tags_set: Set[str] = set()
        in_taxonomy_set: Set[str] = set()

        for chunk in chunks:
            tags = chunk.get("concept_tags", [])
            for tag in tags:
                tag_counts[tag] += 1
                result = self.normalize(tag)

                if result.is_valid:
                    valid_tags_set.add(tag)
                    if result.reason == "in_taxonomy":
                        in_taxonomy_set.add(tag)
                else:
                    if tag not in invalid_tags:
                        invalid_tags[tag] = (result.reason or "unknown", 0)
                    reason, count = invalid_tags[tag]
                    invalid_tags[tag] = (reason, count + 1)

        # Build format violations list
        format_violations = [
            (tag, reason) for tag, (reason, _) in invalid_tags.items()
        ]

        # Top tags by frequency
        top_tags = tag_counts.most_common(20)

        # Top invalid tags
        top_invalid = sorted(
            [(tag, reason, count) for tag, (reason, count) in invalid_tags.items()],
            key=lambda x: -x[2]
        )[:20]

        return VocabularyAnalysis(
            total_tags=sum(tag_counts.values()),
            unique_tags=len(tag_counts),
            valid_tags=len(valid_tags_set),
            invalid_tags=len(invalid_tags),
            in_taxonomy=len(in_taxonomy_set),
            not_in_taxonomy=len(valid_tags_set) - len(in_taxonomy_set),
            format_violations=format_violations,
            top_tags=top_tags,
            top_invalid=top_invalid,
        )

    def clean_chunk_tags(
        self,
        chunk: Dict,
        remove_invalid: bool = True,
    ) -> Tuple[Dict, List[str]]:
        """
        Clean concept tags in a chunk.

        Args:
            chunk: Chunk dictionary
            remove_invalid: If True, remove invalid tags. If False, normalize them.

        Returns:
            Tuple of (updated chunk, list of removed tags)
        """
        tags = chunk.get("concept_tags", [])
        cleaned_tags = []
        removed_tags = []

        for tag in tags:
            result = self.normalize(tag)

            if result.is_valid and result.normalized:
                if result.normalized not in cleaned_tags:
                    cleaned_tags.append(result.normalized)
            elif not remove_invalid and result.normalized:
                # Keep normalized version even if not fully valid
                if result.normalized not in cleaned_tags:
                    cleaned_tags.append(result.normalized)
            else:
                removed_tags.append(tag)

        chunk["concept_tags"] = cleaned_tags
        return chunk, removed_tags


def clean_guardrails_allowed_topics(
    guardrails_path: Path,
    vocab: ConceptVocabulary,
) -> Dict:
    """
    Clean allowed_topics in guardrails using vocabulary.

    Args:
        guardrails_path: Path to guardrails.json
        vocab: ConceptVocabulary instance

    Returns:
        Updated guardrails dict
    """
    with open(guardrails_path) as f:
        guardrails = json.load(f)

    allowed_topics = guardrails.get("allowed_topics", [])
    cleaned_topics = []

    for topic in allowed_topics:
        result = vocab.normalize(topic)
        if result.is_valid and result.normalized:
            if result.normalized not in cleaned_topics:
                cleaned_topics.append(result.normalized)

    guardrails["allowed_topics"] = cleaned_topics

    logger.info(
        f"Cleaned guardrails: {len(allowed_topics)} -> {len(cleaned_topics)} topics"
    )

    return guardrails


def analyze_course_concepts(
    course_dir: Path,
    repo_root: Path,
) -> VocabularyAnalysis:
    """
    Analyze concept vocabulary for a course.

    Args:
        course_dir: Path to course directory
        repo_root: Path to repository root (for taxonomy)

    Returns:
        VocabularyAnalysis
    """
    taxonomy_path = _resolve_project_schemas_dir(repo_root) / "taxonomies" / "taxonomy.json"
    vocab = ConceptVocabulary(taxonomy_path)

    chunks_path = course_dir / "corpus" / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found at {chunks_path}")

    return vocab.analyze_corpus(chunks_path)


def clean_course_concepts(
    course_dir: Path,
    repo_root: Path,
    remove_invalid: bool = True,
    clean_guardrails: bool = True,
) -> Dict[str, int]:
    """
    Clean concept tags in a course.

    Args:
        course_dir: Path to course directory
        repo_root: Path to repository root
        remove_invalid: If True, remove invalid tags
        clean_guardrails: If True, also clean guardrails.json

    Returns:
        Statistics about the cleaning
    """
    taxonomy_path = _resolve_project_schemas_dir(repo_root) / "taxonomies" / "taxonomy.json"
    vocab = ConceptVocabulary(taxonomy_path)

    # Clean chunks
    chunks_path = course_dir / "corpus" / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found at {chunks_path}")

    with open(chunks_path) as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must be a list")

    total_removed = 0
    chunks_modified = 0

    for chunk in chunks:
        chunk, removed = vocab.clean_chunk_tags(chunk, remove_invalid)

        if removed:
            total_removed += len(removed)
            chunks_modified += 1

    # Write cleaned chunks
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)

    stats = {
        "chunks_modified": chunks_modified,
        "tags_removed": total_removed,
    }

    # Clean guardrails if requested
    if clean_guardrails:
        guardrails_path = course_dir / "pedagogy" / "guardrails.json"
        if guardrails_path.exists():
            original_guardrails = json.load(open(guardrails_path))
            original_topic_count = len(original_guardrails.get("allowed_topics", []))

            cleaned_guardrails = clean_guardrails_allowed_topics(guardrails_path, vocab)
            new_topic_count = len(cleaned_guardrails.get("allowed_topics", []))

            with open(guardrails_path, "w") as f:
                json.dump(cleaned_guardrails, f, indent=2)

            stats["guardrails_topics_removed"] = original_topic_count - new_topic_count

    logger.info(f"Cleaned course concepts: {stats}")
    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python concept_vocabulary.py <course_dir> <repo_root>")
        sys.exit(1)

    course_dir = Path(sys.argv[1])
    repo_root = Path(sys.argv[2])

    analysis = analyze_course_concepts(course_dir, repo_root)

    print("\nConcept Vocabulary Analysis")
    print("===========================")
    print(f"Total tags: {analysis.total_tags}")
    print(f"Unique tags: {analysis.unique_tags}")
    print(f"Valid tags: {analysis.valid_tags}")
    print(f"Invalid tags: {analysis.invalid_tags}")
    print(f"In taxonomy: {analysis.in_taxonomy}")
    print(f"Not in taxonomy: {analysis.not_in_taxonomy}")

    if analysis.top_invalid:
        print("\nTop Invalid Tags:")
        for tag, reason, count in analysis.top_invalid[:10]:
            print(f"  {tag}: {reason} ({count}x)")
