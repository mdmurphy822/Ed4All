"""Streaming retrieval module for LibV2.

This module provides sample-based querying without loading the entire corpus.
It filters courses by metadata first, then streams chunks line-by-line from
chunks.jsonl files, applying filters and ranking with BM25 + character
n-gram boosting for improved semantic resilience.
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .catalog import load_master_catalog, search_catalog
from .models.catalog import CatalogEntry
from .retrieval_scoring import (
    BoostContributions,
    chunk_type_intent_prior,
    combine_bm25_with_boosts,
    compute_tag_idf,
    concept_graph_overlap_boost,
    detect_query_intent,
    extract_query_concepts,
    load_concept_graph_node_ids,
    load_course_outcomes,
    load_pedagogy_model,
    load_targets_concept_edges,
    lo_match_boost,
    prereq_coverage_boost,
    resolve_method_preset,
    tag_idf_overlap_score,
    targets_concept_boost,
)

_WEEK_NUM_RE = re.compile(r"week[_\-\s]?(\d+)", re.IGNORECASE)


def _parse_week_num(module_id: Optional[str]) -> Optional[int]:
    """Pull the integer week number out of a ``week_NN_*`` style module id."""
    if not module_id:
        return None
    match = _WEEK_NUM_RE.search(module_id)
    return int(match.group(1)) if match else None


@dataclass
class ChunkFilter:
    """Filter criteria for chunks.

    Fields up through ``bloom_level`` are the pre-v4 schema and preserved as-is
    for back-compat with existing callers.  The ``teaching_role`` /
    ``content_type_label`` / ``module_id`` / ``week_num`` fields were added in
    Worker J to expose v4 chunk metadata as filter axes.

    REC-VOC-03 Phase 2 (Worker T): when ``TRAINFORGE_ENFORCE_CONTENT_TYPE=true``,
    ``content_type_label`` is validated against the ChunkType enum from
    ``schemas/taxonomies/content_type.json``. Flag off: accept any string
    (backward-compat).
    """
    chunk_type: Optional[str] = None
    difficulty: Optional[str] = None
    concept_tags: Optional[list[str]] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    learning_outcome_refs: Optional[list[str]] = None
    bloom_level: Optional[str] = None
    # v4 additions (Worker J)
    teaching_role: Optional[str] = None
    content_type_label: Optional[str] = None
    module_id: Optional[str] = None
    week_num: Optional[int] = None
    # Wave 70 additions — RDF-aligned filter axes.
    # ``cognitive_domain`` is expected on chunks directly (factual /
    # conceptual / procedural / metacognitive per
    # schemas/context/courseforge_v1.vocabulary.ttl). Dependent on Wave 69
    # extension that lands the predicate on chunk emit.
    # ``hierarchy_level`` is NOT on chunks; it's resolved via the chunk's
    # ``learning_outcome_refs[]`` against the course's outcomes list.
    # Value space: "terminal" or "chapter" (matches LO hierarchyLevel).
    cognitive_domain: Optional[str] = None
    hierarchy_level: Optional[str] = None

    def __post_init__(self) -> None:
        # REC-VOC-03 Phase 2 (Worker T): opt-in content_type enforcement.
        # Import inside the method so module-import time stays free of the
        # lib.validators dependency (keeps LibV2 CLI startup cheap when the
        # flag is off, which is the default).
        if self.content_type_label is not None:
            from lib.validators.content_type import assert_chunk_type

            assert_chunk_type(
                self.content_type_label,
                context="ChunkFilter.content_type_label",
            )

    def as_applied_dict(self) -> dict:
        """Return only the fields that are actively constraining results.
        Used by the rationale payload so readers see which filters fired."""
        out: dict = {}
        for name, value in self.__dict__.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and not value:
                continue
            out[name] = value
        return out


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata.

    The ``rationale`` field is populated only when ``retrieve_chunks`` was
    called with ``include_rationale=True``.  All existing fields are preserved
    for back-compat with production callers in Trainforge/rag/libv2_bridge.py.
    """
    chunk_id: str
    text: str
    score: float
    course_slug: str
    domain: str
    chunk_type: str
    difficulty: Optional[str]
    concept_tags: list[str]
    source: dict
    tokens_estimate: int = 0
    learning_outcome_refs: list[str] = field(default_factory=list)
    bloom_level: Optional[str] = None
    rationale: Optional[dict] = None  # Worker J — None when include_rationale=False

    def to_dict(self) -> dict:
        base = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "course_slug": self.course_slug,
            "domain": self.domain,
            "chunk_type": self.chunk_type,
            "difficulty": self.difficulty,
            "concept_tags": self.concept_tags,
            "source": self.source,
            "tokens_estimate": self.tokens_estimate,
            "learning_outcome_refs": self.learning_outcome_refs,
            "bloom_level": self.bloom_level,
        }
        # Back-compat: only emit `rationale` key when populated.  Legacy
        # consumers serialising results to JSON get byte-identical output.
        if self.rationale is not None:
            base["rationale"] = self.rationale
        return base

    def to_jsonld(
        self,
        context_url: str = "https://ed4all.dev/ns/courseforge/v1",
    ) -> dict:
        """Wave 70 — RDF-compatible JSON-LD projection.

        Additive wrapper over :meth:`to_dict` — the legacy dict shape is
        untouched. See :mod:`LibV2.tools.libv2.jsonld_emit` for the full
        predicate alignment table.
        """
        # Local import so the (rarely-used) emit path doesn't get pulled
        # into every `from .retriever import ...` call.
        from .jsonld_emit import retrieval_result_to_jsonld

        return retrieval_result_to_jsonld(self, context_url=context_url)


# Retrieval scoring utilities

# Default minimum relevance threshold.
# Results below this threshold are filtered. If no results meet the
# threshold, an empty list is returned rather than low-quality fallbacks.
# This ensures downstream consumers (Trainforge) only receive content
# with sufficient relevance for high-quality question generation.
DEFAULT_MIN_RELEVANCE = 0.5

STOP_WORDS = {
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
}

# Backward compatibility alias
MIN_RELEVANCE_THRESHOLD = DEFAULT_MIN_RELEVANCE


# ``structured_tokens=True`` preserves hyphenated slugs (aria-labelledby,
# skip-link, focus-indicator) and WCAG SC references (sc-1.4.3, wcag-2.2)
# as single tokens instead of splitting them into their alphanumeric parts.
# Ordering matters: SC refs are matched first (more specific), then hyphenated
# slugs, then bare alphanumeric tokens pick up the remainder.
_STRUCTURED_TOKEN_RE = re.compile(
    r"sc-\d+(?:\.\d+){1,2}"              # SC refs: sc-1.4.3, sc-2.4.7
    r"|wcag-\d+(?:\.\d+)?"                # WCAG version tokens: wcag-2.2
    r"|[a-z][a-z0-9]*(?:-[a-z0-9]+)+"     # hyphenated slugs: aria-labelledby
    r"|[a-z0-9]+"                         # bare alphanumeric fallback
)

# Wave 84: split CamelCase boundaries before lowercasing so RDF/SHACL URI
# forms like ``NodeShape`` / ``PropertyShape`` / ``subClassOf`` produce
# tokens that align with the prose form. The probe-set diagnosis showed
# chunk_00147 ("Node Shapes and Property Shapes" — the definition chunk)
# losing to chunk_00175 ("Logical Composition with sh:and, sh:or") for
# the query "Define a SHACL NodeShape" because:
#   - chunk body says "node shape" (tokens: node, shape)
#   - URI form "sh:NodeShape" tokenizes to "sh", "nodeshape"
#   - query "NodeShape" tokenizes to "nodeshape"
# The query never matches the prose body; URI count alone drives ranking.
# Solution: emit BOTH the joined form (``nodeshape``) and the split forms
# (``node``, ``shape``) so a query in either style matches text in either
# style. Pre-existing URI-exact probes still hit because the joined form
# is preserved.
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _expand_camel_case(text: str) -> str:
    """Insert spaces at CamelCase boundaries while preserving the original.

    Returns the input with both the original (joined) form and the split
    form, separated by spaces, so downstream tokenization picks up each
    component as its own token AND the original camelCase word survives
    (lowercased) as a joined token via the bare-alphanumeric fallback.

    Examples:
        ``"NodeShape"`` → ``"NodeShape Node Shape"``
        ``"sh:PropertyShape"`` → ``"sh:PropertyShape sh:Property Shape"``
        ``"subClassOf entailment"`` → ``"subClassOf sub Class Of entailment"``

    No-op for plain lowercase text and for tokens with no CamelCase
    boundaries (no double-emission cost).
    """
    if not text:
        return text
    expanded = _CAMEL_BOUNDARY_RE.sub(" ", text)
    if expanded == text:
        return text
    return f"{text} {expanded}"


def tokenize(text: str, *, structured_tokens: bool = True) -> list[str]:
    """Tokenize ``text`` for BM25 indexing and query matching.

    When ``structured_tokens`` is True (default), hyphenated slugs and SC refs
    are preserved as single tokens so querying ``"aria-labelledby"`` matches
    a chunk tagged with the same slug instead of leaking into generic ``aria``
    and ``labelledby`` tokens.  Set False to reproduce the pre-Worker-J
    tokenization (used by back-compat regression tests).

    Wave 84: CamelCase boundaries are split BEFORE lowercasing so URI forms
    like ``NodeShape`` produce both ``nodeshape`` (joined) AND ``node`` +
    ``shape`` (split). See ``_expand_camel_case`` for the rationale.
    """
    if structured_tokens:
        text = _expand_camel_case(text)
    text = text.lower()
    if structured_tokens:
        tokens = _STRUCTURED_TOKEN_RE.findall(text)
    else:
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# Rewrites "SC 1.4.3" and "WCAG 2.2" (common query forms) into hyphenated
# slugs that align with concept_tags emitted by Trainforge.  Applied before
# tokenization so the structured-token regex picks them up as single tokens.
_SC_QUERY_NORMALIZE_RE = re.compile(r"\b(sc|wcag)\s+(\d+(?:\.\d+){0,2})\b", re.IGNORECASE)


def _normalize_structured_refs(query: str) -> str:
    """Fold space-separated SC/WCAG references into hyphenated form."""
    if not query:
        return query
    return _SC_QUERY_NORMALIZE_RE.sub(lambda m: f"{m.group(1).lower()}-{m.group(2)}", query)


def _canonicalize_query(query: str) -> str:
    """Apply WCAG SC canonicalization so query tokens match the same
    normalized form that Trainforge used when tagging chunk metadata.
    Silently no-ops when the Trainforge helper isn't importable (keeps
    retriever usable in repos that don't ship Trainforge)."""
    # Always normalize "SC 1.4.3" → "sc-1.4.3" first so structured tokens fire.
    normalized = _normalize_structured_refs(query or "")
    try:
        from Trainforge.rag.wcag_canonical_names import canonicalize_sc_references
    except Exception:
        return normalized
    try:
        return canonicalize_sc_references(normalized)
    except Exception:
        return normalized


def _char_trigrams(text: str) -> set[str]:
    """Extract character trigrams from text for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    trigrams: set[str] = set()
    for word in text.split():
        if len(word) >= 3:
            for i in range(len(word) - 2):
                trigrams.add(word[i:i + 3])
    return trigrams


class LazyBM25:
    """BM25 index with character n-gram boosting.

    Replaces the previous LazyTFIDF implementation with Okapi BM25
    for better term saturation and document length normalization.
    Character trigram overlap provides lightweight fuzzy matching
    for morphological variants and partial word matches.

    Args:
        chunks: List of chunk dicts with "text" field.
        k1: BM25 term frequency saturation parameter (default 1.5).
        b: BM25 document length normalization (0=none, 1=full, default 0.75).
        ngram_weight: Weight for character trigram score blending (default 0.15).
    """

    def __init__(
        self,
        chunks: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
        ngram_weight: float = 0.15,
        use_retrieval_text: bool = True,
        structured_tokens: bool = True,
    ):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.ngram_weight = ngram_weight
        # When use_retrieval_text is True, chunks that carry a non-empty
        # `retrieval_text` (v4 schema addition, = summary + key_terms) are
        # indexed against that shorter, higher-signal string instead of
        # the full chunk text.  v3 chunks without retrieval_text fall back
        # to chunk.text and behave identically to pre-Worker-J.
        self.use_retrieval_text = use_retrieval_text
        self.structured_tokens = structured_tokens
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.doc_tokens: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self._build_index()

    # Wave 84: section_heading injection. The probe diagnosis showed the
    # NodeShape definition chunk (heading "Node Shapes and Property Shapes")
    # losing to a sh:and/or/not logical-composition chunk because the
    # heading — the most signal-dense string in any chunk — was never
    # indexed. Headings live at chunk.source.section_heading; they are
    # injected at the front of the indexed string with 3× repetition so
    # heading terms get a soft boost without overwhelming body text.
    # Headings are capped at HEADING_TOKEN_CAP words to prevent very long
    # headings on short chunks from dominating term frequency.
    HEADING_REPETITIONS = 3
    HEADING_TOKEN_CAP = 8

    def _heading_for_indexing(self, chunk: dict) -> str:
        """Return the section heading repeated for indexing, or empty.

        Reads from ``chunk.source.section_heading`` (v4 schema) with a
        fallback to ``chunk.section_heading`` for any back-compat path.
        Caps heading length so a verbose heading doesn't dominate the
        chunk's term-frequency distribution.
        """
        source = chunk.get("source") or {}
        heading = source.get("section_heading") or chunk.get("section_heading") or ""
        if not heading or not isinstance(heading, str):
            return ""
        words = heading.split()
        if len(words) > self.HEADING_TOKEN_CAP:
            words = words[: self.HEADING_TOKEN_CAP]
        capped = " ".join(words)
        return " ".join([capped] * self.HEADING_REPETITIONS)

    def _doc_text_for_indexing(self, chunk: dict) -> str:
        """Return the string to index for a chunk.  Prefers retrieval_text
        when the chunk carries one and ``use_retrieval_text`` is on.

        Wave 84: prepends ``section_heading`` (capped + repeated) so the
        heading contributes term frequency to BM25. Headings are the most
        signal-dense string per chunk; pre-Wave-84 they were invisible to
        retrieval, which let same-topic chunks with denser body keyword
        repetition outscore the actual definition chunk for the topic.
        """
        heading_block = self._heading_for_indexing(chunk)
        if self.use_retrieval_text:
            rt = chunk.get("retrieval_text")
            if rt:
                return f"{heading_block} {rt}".strip() if heading_block else str(rt)
        body = chunk.get("text", "")
        return f"{heading_block} {body}".strip() if heading_block else body

    def _build_index(self):
        """Build BM25 index from chunks."""
        total_length = 0

        for chunk in self.chunks:
            text = self._doc_text_for_indexing(chunk)
            tokens = tokenize(text, structured_tokens=self.structured_tokens)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            for token in set(tokens):
                self.doc_freq[token] += 1

        n = len(self.chunks)
        self.avgdl = total_length / n if n > 0 else 1.0

    def _idf(self, term: str) -> float:
        """BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_relevance: Optional[float] = None,
        return_components: bool = False,
    ) -> list[tuple]:
        """Search using BM25 scoring with optional n-gram boosting.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            min_relevance: Minimum score threshold (default: DEFAULT_MIN_RELEVANCE).
            return_components: When True, return 4-tuples
                ``(chunk, blended_score, bm25_score, ngram_score)`` for each
                result so callers (retrieve_chunks rationale) can separate
                the BM25 contribution from the n-gram contribution.  Default
                False keeps the pre-Worker-J 2-tuple shape for back-compat.

        Returns:
            List of (chunk, score) tuples sorted by descending score, or 4-tuples
            when return_components=True.
        """
        if min_relevance is None:
            min_relevance = DEFAULT_MIN_RELEVANCE

        # Canonicalise SC refs so "Contrast Minimum" and "Contrast (Minimum)"
        # tokenize identically to the chunk side (Worker J).
        query_canonical = _canonicalize_query(query)
        query_tokens = tokenize(query_canonical, structured_tokens=self.structured_tokens)
        if not query_tokens:
            return []

        # BM25 scoring
        scored: list[tuple[dict, float]] = []
        for i, chunk in enumerate(self.chunks):
            doc_len = self.doc_lengths[i]
            tf_counts = Counter(self.doc_tokens[i])

            bm25_score = 0.0
            for term in query_tokens:
                if term not in tf_counts:
                    continue
                tf = tf_counts[term]
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                bm25_score += idf * (numerator / denominator)

            if bm25_score > 0:
                scored.append((chunk, bm25_score))

        if not scored:
            return []

        # Character trigram boosting for fuzzy matching
        with_components: list[tuple[dict, float, float, float]] = []
        if self.ngram_weight > 0:
            query_trigrams = _char_trigrams(query_canonical)
        else:
            query_trigrams = set()

        for chunk, bm25_score in scored:
            jaccard = 0.0
            if query_trigrams:
                # Use the same indexed text for trigram matching so summary-
                # indexed chunks are compared fairly.
                chunk_text = self._doc_text_for_indexing(chunk)
                chunk_trigrams = _char_trigrams(chunk_text[:500])
                if chunk_trigrams:
                    intersection = len(query_trigrams & chunk_trigrams)
                    union = len(query_trigrams | chunk_trigrams)
                    jaccard = intersection / union if union > 0 else 0.0

            if self.ngram_weight > 0 and query_trigrams:
                blended = (
                    (1 - self.ngram_weight) * bm25_score
                    + self.ngram_weight * jaccard * bm25_score
                )
            else:
                blended = bm25_score

            ngram_component = self.ngram_weight * jaccard * bm25_score
            with_components.append((chunk, blended, bm25_score, ngram_component))

        with_components.sort(key=lambda x: x[1], reverse=True)

        # Apply relevance threshold — no fallback.
        above_threshold = [
            t for t in with_components if t[1] >= min_relevance
        ][:limit]

        if return_components:
            return above_threshold
        return [(c, s) for c, s, _, _ in above_threshold]


# Backward compatibility alias
LazyTFIDF = LazyBM25


def _matches_filter(
    chunk: dict,
    chunk_filter: ChunkFilter,
    outcomes_by_id: Optional[dict] = None,
) -> bool:
    """Check if a chunk matches the filter criteria.

    ``outcomes_by_id`` is an optional ``{lo_id_lower: outcome_dict}`` map
    used to resolve the Wave 70 ``hierarchy_level`` filter — the
    hierarchy lives on the LO, not the chunk, so we fan out via
    ``learning_outcome_refs[]``. Callers that don't pass the map (e.g.
    direct unit-test invocation) get no hierarchy_level coverage, which
    matches the intent: the predicate is only meaningful when resolved
    against the course outcomes.
    """
    if chunk_filter.chunk_type:
        if chunk.get("chunk_type") != chunk_filter.chunk_type:
            return False

    if chunk_filter.difficulty:
        if chunk.get("difficulty") != chunk_filter.difficulty:
            return False

    if chunk_filter.concept_tags:
        chunk_tags = set(chunk.get("concept_tags", []))
        filter_tags = set(chunk_filter.concept_tags)
        if not filter_tags & chunk_tags:  # No overlap
            return False

    if chunk_filter.min_tokens:
        if chunk.get("tokens_estimate", 0) < chunk_filter.min_tokens:
            return False

    if chunk_filter.max_tokens:
        if chunk.get("tokens_estimate", 0) > chunk_filter.max_tokens:
            return False

    if chunk_filter.learning_outcome_refs:
        chunk_refs = set(chunk.get("learning_outcome_refs", []))
        filter_refs = set(chunk_filter.learning_outcome_refs)
        if not filter_refs & chunk_refs:  # No overlap
            return False

    if chunk_filter.bloom_level:
        # Match bloom level from linked outcomes or chunk metadata
        chunk_bloom = chunk.get("bloom_level")
        if chunk_bloom != chunk_filter.bloom_level:
            return False

    # v4 additions (Worker J)
    if chunk_filter.teaching_role:
        if chunk.get("teaching_role") != chunk_filter.teaching_role:
            return False

    if chunk_filter.content_type_label:
        if chunk.get("content_type_label") != chunk_filter.content_type_label:
            return False

    if chunk_filter.module_id:
        chunk_module = (chunk.get("source") or {}).get("module_id")
        if chunk_module != chunk_filter.module_id:
            return False

    if chunk_filter.week_num is not None:
        chunk_module = (chunk.get("source") or {}).get("module_id")
        chunk_week = _parse_week_num(chunk_module)
        if chunk_week != chunk_filter.week_num:
            return False

    # Wave 70 — RDF-aligned filters.
    if chunk_filter.cognitive_domain:
        # Expected on the chunk directly (Wave 60 → Wave 69 emit). Match
        # case-insensitively to be kind to corpora with mixed case.
        chunk_cd = chunk.get("cognitive_domain")
        if not chunk_cd or str(chunk_cd).lower() != str(chunk_filter.cognitive_domain).lower():
            return False

    if chunk_filter.hierarchy_level:
        target = str(chunk_filter.hierarchy_level).lower()
        refs = [str(r).lower() for r in chunk.get("learning_outcome_refs", []) if r]
        if not refs:
            return False
        if not outcomes_by_id:
            # No lookup table available — we can't resolve the LO's
            # hierarchyLevel, so be conservative and reject rather than
            # let through a chunk we can't attest about.
            return False
        matched = False
        for ref in refs:
            lo = outcomes_by_id.get(ref)
            if not lo:
                continue
            lo_level = str(lo.get("hierarchy_level", "")).lower()
            if lo_level == target:
                matched = True
                break
        if not matched:
            return False

    return True


def _build_outcomes_lookup(course_dir: Path) -> dict:
    """Build a ``{lo_id_lower: outcome_dict}`` map for hierarchy_level lookup.

    Consumes the same ``course.json`` shape that
    ``retrieval_scoring.load_course_outcomes`` reads. Returns an empty
    dict when the course has no outcomes file — callers should handle
    that case the same as "no match".
    """
    path = course_dir / "course.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    los = data.get("learning_outcomes") or data.get("outcomes") or []
    out: dict = {}
    for lo in los:
        if not isinstance(lo, dict):
            continue
        lo_id = lo.get("id")
        if lo_id:
            out[str(lo_id).lower()] = lo
    return out


def stream_chunks_from_course(
    course_dir: Path,
    course_slug: str,
    domain: str,
    chunk_filter: Optional[ChunkFilter] = None,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    """Stream chunks line-by-line from chunks.jsonl, applying filters.

    Yields chunks one at a time without loading the entire file.
    """
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    if not chunks_path.exists():
        return

    # Wave 70: build the outcomes lookup lazily — only the
    # hierarchy_level filter needs it, so courses without course.json
    # aren't penalized.
    outcomes_by_id: Optional[dict] = None
    if chunk_filter and chunk_filter.hierarchy_level:
        outcomes_by_id = _build_outcomes_lookup(course_dir)

    count = 0
    with open(chunks_path) as f:
        for line in f:
            if limit and count >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Add course metadata
            chunk["_course_slug"] = course_slug
            chunk["_domain"] = domain

            # Apply filter
            if chunk_filter and not _matches_filter(
                chunk, chunk_filter, outcomes_by_id=outcomes_by_id,
            ):
                continue

            count += 1
            yield chunk


def _collect_filtered_chunks(
    courses: list[CatalogEntry],
    repo_root: Path,
    chunk_filter: Optional[ChunkFilter],
    budget: int,
    per_course_budget: Optional[int] = None,
) -> list[dict]:
    """Collect chunks from filtered courses until budget is reached.

    Uses round-robin across courses to get diverse results.
    """
    collected = []
    course_iterators = {}
    course_counts = {}

    # Initialize iterators for each course
    for entry in courses:
        course_dir = repo_root / "courses" / entry.slug
        chunks_path = course_dir / "corpus" / "chunks.jsonl"
        if chunks_path.exists():
            course_iterators[entry.slug] = {
                "iterator": stream_chunks_from_course(
                    course_dir, entry.slug, entry.primary_domain, chunk_filter
                ),
                "domain": entry.primary_domain,
            }
            course_counts[entry.slug] = 0

    # Round-robin collection
    while len(collected) < budget and course_iterators:
        exhausted = []
        for slug, info in list(course_iterators.items()):
            if len(collected) >= budget:
                break

            if per_course_budget and course_counts[slug] >= per_course_budget:
                exhausted.append(slug)
                continue

            try:
                chunk = next(info["iterator"])
                collected.append(chunk)
                course_counts[slug] += 1
            except StopIteration:
                exhausted.append(slug)

        for slug in exhausted:
            del course_iterators[slug]

    return collected


def retrieve_chunks(
    repo_root: Path,
    query: str,
    domain: Optional[str] = None,
    division: Optional[str] = None,
    subdomain: Optional[str] = None,
    course_slug: Optional[str] = None,
    chunk_type: Optional[str] = None,
    difficulty: Optional[str] = None,
    concept_tags: Optional[list[str]] = None,
    learning_outcome_refs: Optional[list[str]] = None,
    bloom_level: Optional[str] = None,
    # v4 filters (Worker J)
    teaching_role: Optional[str] = None,
    content_type_label: Optional[str] = None,
    module_id: Optional[str] = None,
    week_num: Optional[int] = None,
    # Wave 70 RDF-aligned filters
    cognitive_domain: Optional[str] = None,
    hierarchy_level: Optional[str] = None,
    # Worker J additions — back-compat by default
    include_rationale: bool = False,
    metadata_scoring: bool = True,
    use_concept_graph_boost: bool = True,
    use_lo_match_boost: bool = True,
    prefer_self_contained: bool = False,  # prereq boost, off by default (niche)
    use_targets_concept_boost: bool = True,  # Wave 71: typed LO→concept edges
    query_bloom_level: Optional[str] = None,  # Wave 71: detected query Bloom for edge-level match bonus
    # Wave 84 — additive signals (default off so existing callers see no
    # behavior change). Opt in via ``method`` preset or explicit booleans.
    use_tag_idf_boost: bool = False,
    use_chunk_type_intent_boost: bool = False,
    method: Optional[str] = None,  # preset selector — overrides individual flags above
    lo_filter: Optional[list[str]] = None,
    boost_weights: Optional[dict] = None,
    use_retrieval_text: bool = True,
    structured_tokens: bool = True,
    limit: int = 10,
    sample_per_course: Optional[int] = None,
    min_relevance: Optional[float] = None,
) -> list[RetrievalResult]:
    """Retrieve chunks matching query and filters.

    Back-compat contract:  when ``include_rationale=False`` (the default) the
    returned ``RetrievalResult.to_dict()`` output is byte-identical to the
    pre-Worker-J shape — production callers (Trainforge/rag/libv2_bridge.py)
    are unaffected.  Opt into the rationale payload and metadata-aware
    scoring explicitly.
    """
    # Wave 84: ``method`` preset overrides individual boost flags so
    # callers can A/B retrieval configurations without juggling 6 booleans.
    # Unknown method names raise ValueError (fail loudly on typos).
    if method is not None:
        preset = resolve_method_preset(method)
        if "metadata_scoring" in preset:
            metadata_scoring = bool(preset["metadata_scoring"])
        if "use_concept_graph_boost" in preset:
            use_concept_graph_boost = bool(preset["use_concept_graph_boost"])
        if "use_lo_match_boost" in preset:
            use_lo_match_boost = bool(preset["use_lo_match_boost"])
        if "use_targets_concept_boost" in preset:
            use_targets_concept_boost = bool(preset["use_targets_concept_boost"])
        if "use_tag_idf_boost" in preset:
            use_tag_idf_boost = bool(preset["use_tag_idf_boost"])
        if "use_chunk_type_intent_boost" in preset:
            use_chunk_type_intent_boost = bool(preset["use_chunk_type_intent_boost"])

    # Phase 1: Filter courses by metadata
    if course_slug:
        course_dir = repo_root / "courses" / course_slug
        if not course_dir.exists():
            return []
        manifest_path = course_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            domain_info = manifest.get("classification", {}).get("primary_domain", "unknown")
        else:
            domain_info = "unknown"
        courses = [CatalogEntry(
            slug=course_slug, title="", division="", primary_domain=domain_info,
        )]
    else:
        catalog = load_master_catalog(repo_root)
        if catalog is None:
            return []
        courses = search_catalog(
            catalog, division=division, domain=domain, subdomain=subdomain,
        )
        if not courses:
            return []

    # Phase 2: Stream and filter chunks
    chunk_filter = ChunkFilter(
        chunk_type=chunk_type,
        difficulty=difficulty,
        concept_tags=concept_tags,
        learning_outcome_refs=learning_outcome_refs,
        bloom_level=bloom_level,
        teaching_role=teaching_role,
        content_type_label=content_type_label,
        module_id=module_id,
        week_num=week_num,
        cognitive_domain=cognitive_domain,
        hierarchy_level=hierarchy_level,
    )

    # Wave 84: previously ``candidate_budget = max(limit * 10, 100)`` —
    # the budget capped at 100 by default for limit=10, so
    # ``_collect_filtered_chunks`` returned the FIRST 100 chunks in
    # file/stream order (which mirrors module/week order), then BM25
    # scored only that subset. For any course with > 100 chunks, gold
    # chunks past the first 100 (later weeks/modules) were invisible
    # to retrieval — a candidate-collection bug, not a tokenization
    # issue. The audit's rdf-shacl-551-2 probe revealed it as a
    # 60% Hit@10 ceiling (6/15 queries' gold chunks lived past chunk_00100).
    #
    # Fix: for single-course queries (``course_slug`` set) the budget
    # is effectively the whole course (10K hard cap, well above any
    # realistic course size). For multi-course queries we still bound
    # so cross-corpus searches don't index a million chunks; bumped
    # the floor from 100 to 1000 so even small limits get a representative
    # pool. ``sample_per_course`` still bounds per-course collection
    # for diversity-shaping cross-corpus queries.
    if course_slug:
        candidate_budget = 10000
    else:
        candidate_budget = max(limit * 10, 1000)
    candidates = _collect_filtered_chunks(
        courses=courses,
        repo_root=repo_root,
        chunk_filter=chunk_filter,
        budget=candidate_budget,
        per_course_budget=sample_per_course,
    )
    if not candidates:
        return []

    # Phase 3: Rank with BM25 + n-gram boosting (+ optional metadata boosts)
    index = LazyBM25(
        candidates,
        use_retrieval_text=use_retrieval_text,
        structured_tokens=structured_tokens,
    )
    scored_with_components = index.search(
        query, limit=candidate_budget, min_relevance=min_relevance, return_components=True,
    )

    # Per-course metadata (loaded once per unique slug seen in candidates)
    graph_nodes_by_slug: dict[str, set[str]] = {}
    outcomes_by_slug: dict[str, list[dict]] = {}
    pedagogy_by_slug: dict[str, dict] = {}
    # Wave 71: typed LO→concept edges from concept_graph_semantic.json.
    # Shape: {lo_id_lower: [(concept_id_lower, bloom_level_lower|None), ...]}
    targets_concept_by_slug: dict[str, dict] = {}

    def _metadata_for(slug: str):
        if slug not in graph_nodes_by_slug:
            cd = repo_root / "courses" / slug
            graph_nodes_by_slug[slug] = load_concept_graph_node_ids(cd)
            outcomes_by_slug[slug] = load_course_outcomes(cd)
            pedagogy_by_slug[slug] = load_pedagogy_model(cd)
            targets_concept_by_slug[slug] = load_targets_concept_edges(cd)
        return (
            graph_nodes_by_slug[slug],
            outcomes_by_slug[slug],
            pedagogy_by_slug[slug],
            targets_concept_by_slug[slug],
        )

    # Assemble results.  Apply metadata boosts AFTER BM25 so their effect is
    # multiplicative, bounded by MAX_TOTAL_BOOST, and attributable per-boost
    # in the rationale payload.
    q_tokens_lower = _lower_tokens_for_rationale(query)

    # Wave 84: pre-compute IDF over the candidate pool's tag distribution.
    # Doing this once over candidates (not the global corpus) keeps the
    # signal sensitive to the active filter — e.g. when narrowed to a
    # single course/week, "rdf" becomes high-IDF if it's rare in that
    # subset. Empty when no boost requested (cheap short-circuit).
    tag_idf_weights: dict[str, float] = (
        compute_tag_idf(c.get("concept_tags", []) for c, _, _, _ in scored_with_components)
        if metadata_scoring and use_tag_idf_boost
        else {}
    )
    query_intents = (
        detect_query_intent(query)
        if metadata_scoring and use_chunk_type_intent_boost
        else set()
    )

    results: list[tuple[RetrievalResult, float]] = []
    for chunk, blended, bm25_score, ngram_score in scored_with_components:
        slug = chunk.get("_course_slug", "")
        graph_nodes, course_outcomes, pedagogy_model, targets_concept_edges = (
            _metadata_for(slug)
        )

        contributions = BoostContributions()
        # Compute query concepts once — the concept-graph overlap boost and
        # the Wave 71 targets-concept boost both consume them, so caching
        # avoids re-tokenizing the query per chunk.
        q_concepts = (
            extract_query_concepts(_canonicalize_query(query), graph_nodes)
            if metadata_scoring and graph_nodes
            else set()
        )
        if metadata_scoring and use_concept_graph_boost:
            contributions.concept_graph_overlap = concept_graph_overlap_boost(chunk, q_concepts)
        if metadata_scoring and use_lo_match_boost:
            contributions.lo_match = lo_match_boost(
                chunk, query, course_outcomes, explicit_lo_filter=lo_filter,
            )
        if metadata_scoring and prefer_self_contained:
            contributions.prereq_coverage = prereq_coverage_boost(chunk, pedagogy_model)
        # Wave 71: reward chunks whose referenced LOs explicitly target a
        # query concept via the typed concept graph, with a Bloom-level
        # match bonus when the caller supplies query_bloom_level.
        if metadata_scoring and use_targets_concept_boost and targets_concept_edges:
            contributions.targets_concept = targets_concept_boost(
                chunk,
                q_concepts,
                targets_concept_edges,
                query_bloom_level=query_bloom_level,
            )
        # Wave 84: separately-fused IDF-weighted tag overlap.
        if metadata_scoring and use_tag_idf_boost and tag_idf_weights:
            contributions.tag_idf_overlap = tag_idf_overlap_score(
                chunk, q_tokens_lower, tag_idf_weights,
            )
        # Wave 84: chunk_type intent prior (define / explain / example / ...).
        if metadata_scoring and use_chunk_type_intent_boost and query_intents:
            contributions.chunk_type_intent = chunk_type_intent_prior(
                chunk, query_intents,
            )

        final_score, capped_boost = combine_bm25_with_boosts(
            blended, contributions, weights=boost_weights,
        )

        result = RetrievalResult(
            chunk_id=chunk.get("id", ""),
            text=chunk.get("text", ""),
            score=final_score,
            course_slug=slug,
            domain=chunk.get("_domain", ""),
            chunk_type=chunk.get("chunk_type", ""),
            difficulty=chunk.get("difficulty"),
            concept_tags=chunk.get("concept_tags", []),
            source=chunk.get("source", {}),
            tokens_estimate=chunk.get("tokens_estimate", 0),
            learning_outcome_refs=chunk.get("learning_outcome_refs", []),
            bloom_level=chunk.get("bloom_level"),
        )

        if include_rationale:
            chunk_tags_lower = {str(t).lower() for t in chunk.get("concept_tags", [])}
            # Include bigram-matched graph concepts, not just whole-token matches
            # (so ``color-contrast`` surfaces when the query is "color contrast").
            q_concepts_for_rationale = extract_query_concepts(
                _canonicalize_query(query), graph_nodes or set(),
            ) if graph_nodes else set()
            matched_concept_tags = sorted(
                (chunk_tags_lower & q_tokens_lower) | (chunk_tags_lower & q_concepts_for_rationale)
            )
            matched_lo_refs = _rationale_matched_lo_refs(
                chunk, query, course_outcomes, lo_filter,
            )
            matched_key_terms = _rationale_matched_key_terms(chunk, q_tokens_lower)
            result.rationale = {
                "bm25_score": round(bm25_score, 4),
                "ngram_score": round(ngram_score, 4),
                "metadata_boost": round(capped_boost, 4),
                "final_score": round(final_score, 4),
                "matched_concept_tags": matched_concept_tags,
                "matched_lo_refs": matched_lo_refs,
                "matched_key_terms": matched_key_terms,
                "applied_filters": chunk_filter.as_applied_dict(),
                "boost_contributions": contributions.to_dict(),
            }

        results.append((result, final_score))

    # Re-sort by the final (boost-adjusted) score.
    results.sort(key=lambda t: t[1], reverse=True)

    # Re-apply the min-relevance floor against the final score so boosts can
    # rescue a borderline chunk or, in the prereq-violation case, correctly
    # push one below the floor.
    threshold = DEFAULT_MIN_RELEVANCE if min_relevance is None else min_relevance
    filtered = [r for r, s in results if s >= threshold][:limit]
    return filtered


def _lower_tokens_for_rationale(text: str) -> set:
    """Lowercase, structured-token set of query words for rationale matching."""
    canonical = _canonicalize_query(text)
    return set(tokenize(canonical, structured_tokens=True))


def _rationale_matched_lo_refs(
    chunk: dict,
    query: str,
    course_outcomes: list,
    lo_filter: Optional[list[str]],
) -> list[str]:
    """Which of the chunk's LO refs were implicated by the query?  Matches
    either an explicit LO filter or by fuzzy statement overlap."""
    chunk_refs = [str(r).lower() for r in chunk.get("learning_outcome_refs", []) if r]
    if not chunk_refs:
        return []
    matched: set[str] = set()
    if lo_filter:
        matched |= {str(x).lower() for x in lo_filter if x} & set(chunk_refs)
    # Explicit id tokens in query (co-03, to-01)
    for ref in chunk_refs:
        if ref in query.lower():
            matched.add(ref)
    # Statement fuzzy overlap
    q_tokens = _lower_tokens_for_rationale(query)
    if q_tokens:
        for outcome in course_outcomes:
            oid = str(outcome.get("id", "")).lower()
            if oid not in chunk_refs:
                continue
            stmt_tokens = set(tokenize(str(outcome.get("statement") or outcome.get("text") or ""),
                                        structured_tokens=True))
            if stmt_tokens and len(q_tokens & stmt_tokens) / max(1, len(q_tokens | stmt_tokens)) >= 0.4:
                matched.add(oid)
    return sorted(matched)


def _rationale_matched_key_terms(chunk: dict, q_tokens_lower: set) -> list[dict]:
    """Return key_terms whose ``term`` slug-form appears in the query tokens.
    Keeps the payload small — only the matches, not the whole key_terms list."""
    matches: list[dict] = []
    for kt in chunk.get("key_terms") or []:
        if not isinstance(kt, dict):
            continue
        term = str(kt.get("term") or "").strip()
        if not term:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
        if slug in q_tokens_lower or term.lower() in q_tokens_lower:
            matches.append({"term": term, "definition": str(kt.get("definition") or "")})
    return matches
