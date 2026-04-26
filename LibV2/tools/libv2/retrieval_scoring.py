"""Metadata-aware retrieval score boosts for the LibV2 reference retriever.

These boosts exploit the v4 chunk metadata (concept_tags, learning_outcome_refs,
prereq_concepts) and the per-course graphs (concept_graph.json,
pedagogy_model.json) to differentiate LibV2's reference retrieval from generic
RAG.  Each boost is a pure function returning a float in [0, 1] (or negative
for the prereq-violation penalty case).

The combine helper applies the boosts multiplicatively with a cap so metadata
cannot dominate BM25 on off-topic text.

Scope (per docs/architecture/ADR-002-retrieval-scope.md): these are *reference*
boosts, not a replacement for a real reranker.  Consumers who need more than
this should build their own ranker on top of the chunk schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

DEFAULT_BOOST_WEIGHTS: Dict[str, float] = {
    "concept_graph_overlap": 0.3,
    "lo_match": 0.3,
    "prereq_coverage": 0.2,
    # Wave 71: Bloom-qualified LO→concept edges from Wave 66's typed graph.
    # Boosts chunks whose referenced LOs explicitly target the query's
    # concepts (with a bonus when the Bloom level matches too).
    "targets_concept": 0.25,
}

# Caps the multiplicative effect of all enabled boosts combined.
# final = bm25 * (1 + min(MAX_TOTAL_BOOST, weighted_sum))
MAX_TOTAL_BOOST = 0.5


_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _lower_tokens(text: str) -> Set[str]:
    """Lowercase word-and-hyphen-slug tokens.  Used by boost helpers to match
    query tokens against concept_tag slugs and LO statement words."""
    if not text:
        return set()
    return set(_WORD_RE.findall(text.lower()))


# ---------------------------------------------------------------------------
# Concept-graph overlap boost
# ---------------------------------------------------------------------------

def concept_graph_overlap_boost(
    chunk: Mapping[str, Any],
    query_concepts: Iterable[str],
) -> float:
    """Jaccard of chunk.concept_tags and query-derived concepts.

    `query_concepts` should be the subset of query tokens that matched any
    node id in the course's concept graph (resolved upstream so this function
    stays pure and cheap).  An empty intersection yields 0.0.
    """
    q = {str(c).lower() for c in query_concepts if c}
    if not q:
        return 0.0
    tags = {str(t).lower() for t in chunk.get("concept_tags", []) if t}
    if not tags:
        return 0.0
    inter = q & tags
    union = q | tags
    return len(inter) / len(union) if union else 0.0


def extract_query_concepts(query: str, graph_node_ids: Set[str]) -> Set[str]:
    """Find the subset of query tokens (plus adjacent-bigram slugs) that
    appear as node ids in ``graph_node_ids``.

    A query like ``"color contrast body text"`` won't match the node
    ``color-contrast`` via single-token lookup, so we also try the
    2-gram ``color-contrast`` and ``contrast-body`` forms.  Bigrams are
    cheap and dramatically lift recall against chunks whose concept_tags
    are hyphenated compounds.

    Each candidate token is also passed through
    :func:`lib.ontology.concept_classifier.canonicalize_alias` so query
    surface forms like ``ttl`` / ``rdfxml`` resolve to the same canonical
    slug (``turtle`` / ``rdf-xml``) the chunk-emit path applies. Without
    this, the emit→query asymmetry causes graph-assisted retrieval to
    miss chunks for any non-canonical surface form.
    """
    if not graph_node_ids:
        return set()
    # Word-level tokens for bigram synthesis (ordered list, not a set)
    words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w]
    q_tokens: Set[str] = set(words)
    # Also keep any hyphenated tokens the user typed whole
    q_tokens |= set(_WORD_RE.findall((query or "").lower()))
    for i in range(len(words) - 1):
        q_tokens.add(f"{words[i]}-{words[i + 1]}")
    # Optional alias canonicalization: mirrors the emit-side normalization in
    # ``Trainforge.process_course._extract_concept_tags``. Soft import keeps
    # the retriever usable in repos that don't ship the lib/ontology layer.
    try:
        from lib.ontology.concept_classifier import canonicalize_alias
    except Exception:
        canonical: Set[str] = set()
    else:
        canonical = {canonicalize_alias(t) for t in q_tokens}
    return (q_tokens | canonical) & graph_node_ids


def load_concept_graph_node_ids(course_dir: Path) -> Set[str]:
    """Load node ids from concept_graph.json; falls back to semantic graph
    when the plain one is missing.  Returns empty set if neither exists."""
    for name in ("concept_graph.json", "concept_graph_semantic.json"):
        path = course_dir / "graph" / name
        if not path.exists():
            continue
        try:
            with open(path) as f:
                graph = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        return {str(n.get("id", "")).lower() for n in graph.get("nodes", []) if n.get("id")}
    return set()


# ---------------------------------------------------------------------------
# Wave 71: targets-concept edge boost (Bloom-qualified LO→concept)
# ---------------------------------------------------------------------------
#
# The Wave 66 Trainforge inference rule `targets_concept_from_lo` materializes
# Wave 57 Courseforge `targetedConcepts[]` payloads as first-class typed edges
# in `concept_graph_semantic.json`:
#
#     {"source": "<lo_id>", "target": "<concept_id>",
#      "type": "targets-concept",
#      "provenance": {"evidence": {"bloom_level": "apply", ...}}}
#
# This lets retrieval answer a pedagogically precise question the untyped
# graph couldn't: "does this chunk's LO explicitly target the concepts in
# the user's query, and at what cognitive demand?"
#
# The pre-Wave-71 `concept_graph_overlap_boost` only compared chunk.concept_tags
# against query tokens — concept membership, not the LO→concept relationship.
# This boost reads the typed graph, so a chunk whose LO targets the concept at
# the query's Bloom level scores higher than a chunk that happens to mention
# the concept without the explicit LO binding.


def load_targets_concept_edges(
    course_dir: Path,
) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """Load targets-concept edges from ``graph/concept_graph_semantic.json``.

    Returns a ``{lo_id_lower: [(concept_id_lower, bloom_level_lower), ...]}``
    map. Empty dict when the typed graph is absent (pre-Wave-66 corpora) or
    has no targets-concept edges (corpus built from LOs without Wave 57
    ``targetedConcepts[]``).

    LO IDs and concept slugs are lowercased here so callers can match
    case-insensitively against chunk.learning_outcome_refs / query tokens
    without re-normalizing on every lookup.
    """
    path = course_dir / "graph" / "concept_graph_semantic.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            graph = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    edges = graph.get("edges") if isinstance(graph, dict) else None
    if not isinstance(edges, list):
        return {}
    out: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("type") != "targets-concept":
            continue
        lo = edge.get("source")
        concept = edge.get("target")
        if not isinstance(lo, str) or not isinstance(concept, str):
            continue
        lo_key = lo.lower()
        concept_key = concept.lower()
        bloom = None
        provenance = edge.get("provenance") or {}
        evidence = provenance.get("evidence") if isinstance(provenance, dict) else None
        if isinstance(evidence, dict):
            raw_bloom = evidence.get("bloom_level")
            if isinstance(raw_bloom, str) and raw_bloom:
                bloom = raw_bloom.lower()
        out.setdefault(lo_key, []).append((concept_key, bloom))
    return out


def targets_concept_boost(
    chunk: Mapping[str, Any],
    query_concepts: Iterable[str],
    targets_by_lo: Mapping[str, List[Tuple[str, Optional[str]]]],
    *,
    query_bloom_level: Optional[str] = None,
    bloom_match_bonus: float = 0.2,
) -> float:
    """Score a chunk by how explicitly its LOs target the query's concepts.

    Algorithm:
      1. Collect the chunk's LO refs (case-insensitive).
      2. Union the targets-concept edges across those LOs: every concept
         those LOs target (with per-edge Bloom level).
      3. Intersect with the query's concept set.
      4. Base score = Jaccard(query_concepts ∩ chunk_targeted_concepts,
                              query_concepts ∪ chunk_targeted_concepts).
      5. Bloom-match bonus: if ``query_bloom_level`` is set AND at least
         one intersecting concept's edge has matching ``bloom_level``,
         multiply the base score by ``(1 + bloom_match_bonus)`` (capped at 1.0).

    Returns 0.0 when:
      * The chunk has no LO refs.
      * None of the chunk's LOs carry targets-concept edges (loader empty).
      * The query has no concepts.
      * The intersection is empty (chunk's LOs don't target any query concept).

    The score is bounded in [0.0, 1.0] so it composes cleanly with the
    existing boost mix in ``combine_bm25_with_boosts``.
    """
    q = {str(c).lower() for c in query_concepts if c}
    if not q:
        return 0.0
    if not targets_by_lo:
        return 0.0
    lo_refs = chunk.get("learning_outcome_refs") or []
    if not isinstance(lo_refs, list):
        return 0.0
    lo_keys = {str(lo).lower() for lo in lo_refs if lo}
    if not lo_keys:
        return 0.0

    # Build (concept, bloom_level) set this chunk's LOs target.
    targeted: Dict[str, Set[Optional[str]]] = {}
    for lo in lo_keys:
        for concept, bloom in targets_by_lo.get(lo, ()):
            targeted.setdefault(concept, set()).add(bloom)
    if not targeted:
        return 0.0

    targeted_concepts = set(targeted.keys())
    inter = q & targeted_concepts
    if not inter:
        return 0.0
    union = q | targeted_concepts
    base = len(inter) / len(union) if union else 0.0

    if query_bloom_level:
        qb = query_bloom_level.lower()
        for concept in inter:
            if qb in targeted[concept]:
                return min(1.0, base * (1.0 + bloom_match_bonus))
    return base


# ---------------------------------------------------------------------------
# Learning-outcome match boost
# ---------------------------------------------------------------------------

_OUTCOME_ID_RE = re.compile(r"\b([a-z]{2})-(\d{2,3})\b", re.IGNORECASE)


def _extract_explicit_lo_ids(query: str) -> Set[str]:
    """Pull any explicit outcome id tokens like ``co-03`` or ``to-01`` out of
    the query.  Case-insensitive."""
    return {m.group(0).lower() for m in _OUTCOME_ID_RE.finditer(query or "")}


def lo_match_boost(
    chunk: Mapping[str, Any],
    query: str,
    course_outcomes: Sequence[Mapping[str, Any]],
    explicit_lo_filter: Optional[Sequence[str]] = None,
    statement_overlap_threshold: float = 0.4,
) -> float:
    """Boost for chunks whose learning_outcome_refs overlap a caller-declared
    LO list, OR whose referenced LOs' statements are similar to the query.

    - If ``explicit_lo_filter`` is non-empty and intersects the chunk's refs,
      return 1.0 — the caller has told us exactly which LOs to target.
    - Else, look for any LO whose statement has ≥ ``statement_overlap_threshold``
      Jaccard overlap with the query tokens AND which appears in the chunk's
      ``learning_outcome_refs``.  Return 0.7 on the first match, else 0.0.
    """
    chunk_refs = {str(r).lower() for r in chunk.get("learning_outcome_refs", []) if r}
    if not chunk_refs:
        return 0.0

    if explicit_lo_filter:
        filt = {str(x).lower() for x in explicit_lo_filter if x}
        if filt & chunk_refs:
            return 1.0

    # Implicit: match by LO id appearing in the query text (co-03)
    implicit_ids = _extract_explicit_lo_ids(query or "")
    if implicit_ids & chunk_refs:
        return 1.0

    # Statement-based fuzzy match
    q_tokens = _lower_tokens(query)
    if not q_tokens or not course_outcomes:
        return 0.0

    for outcome in course_outcomes:
        oid = str(outcome.get("id", "")).lower()
        if oid not in chunk_refs:
            continue
        stmt = str(outcome.get("statement") or outcome.get("text") or "")
        s_tokens = _lower_tokens(stmt)
        if not s_tokens:
            continue
        inter = q_tokens & s_tokens
        union = q_tokens | s_tokens
        jac = len(inter) / len(union) if union else 0.0
        if jac >= statement_overlap_threshold:
            return 0.7

    return 0.0


def load_course_outcomes(course_dir: Path) -> List[Dict[str, Any]]:
    """Return the flat list of outcomes (terminal + chapter) from course.json.
    Returns an empty list when course.json is absent or malformed."""
    path = course_dir / "course.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    # course.json in v4 packages uses a flat "learning_outcomes" list
    los = data.get("learning_outcomes") or data.get("outcomes") or []
    return [lo for lo in los if isinstance(lo, dict)]


# ---------------------------------------------------------------------------
# Prerequisite coverage boost
# ---------------------------------------------------------------------------

def prereq_coverage_boost(
    chunk: Mapping[str, Any],
    pedagogy_model: Mapping[str, Any],
) -> float:
    """Score a chunk by whether its ``prereq_concepts`` are all earlier-defined
    in ``pedagogy_model.prerequisite_chain``, or appear in the
    ``prerequisite_violations`` list.

    Returns 0.7 when the chunk's prereqs are covered (self-contained enough
    to retrieve standalone), -0.5 when any of them shows up in the violations
    list, 0.0 otherwise.  A chunk with no declared prereqs scores 0.0 — the
    boost is deliberately silent about chunks the pipeline didn't tag.
    """
    prereqs = [str(p).lower() for p in chunk.get("prereq_concepts", []) if p]
    if not prereqs:
        return 0.0

    chain = pedagogy_model.get("prerequisite_chain") or []
    covered: Set[str] = set()
    for entry in chain:
        if isinstance(entry, dict):
            concept = str(entry.get("concept") or entry.get("id") or "").lower()
            if concept:
                covered.add(concept)
        elif isinstance(entry, str):
            covered.add(entry.lower())

    violations = pedagogy_model.get("prerequisite_violations") or []
    violating: Set[str] = set()
    for entry in violations:
        if isinstance(entry, dict):
            concept = str(entry.get("concept") or entry.get("id") or "").lower()
            if concept:
                violating.add(concept)
        elif isinstance(entry, str):
            violating.add(entry.lower())

    if violating & set(prereqs):
        return -0.5
    if all(p in covered for p in prereqs):
        return 0.7
    return 0.0


def load_pedagogy_model(course_dir: Path) -> Dict[str, Any]:
    """Load pedagogy_model.json; returns empty dict if absent/malformed."""
    path = course_dir / "pedagogy" / "pedagogy_model.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

@dataclass
class BoostContributions:
    """Per-boost scores recorded for rationale output."""
    concept_graph_overlap: float = 0.0
    lo_match: float = 0.0
    prereq_coverage: float = 0.0
    targets_concept: float = 0.0  # Wave 71

    def to_dict(self) -> Dict[str, float]:
        return {
            "concept_graph_overlap": round(self.concept_graph_overlap, 4),
            "lo_match": round(self.lo_match, 4),
            "prereq_coverage": round(self.prereq_coverage, 4),
            "targets_concept": round(self.targets_concept, 4),
        }


def combine_bm25_with_boosts(
    bm25_score: float,
    contributions: BoostContributions,
    weights: Optional[Mapping[str, float]] = None,
    max_total_boost: float = MAX_TOTAL_BOOST,
) -> Tuple[float, float]:
    """Apply the boost cap and return (final_score, capped_boost).

    ``capped_boost`` is the additive multiplier actually applied — useful for
    including in the rationale payload so consumers see how much metadata
    lifted the score.
    """
    w = dict(DEFAULT_BOOST_WEIGHTS)
    if weights:
        w.update(weights)
    raw = (
        contributions.concept_graph_overlap * w.get("concept_graph_overlap", 0.0)
        + contributions.lo_match * w.get("lo_match", 0.0)
        + contributions.prereq_coverage * w.get("prereq_coverage", 0.0)
        + contributions.targets_concept * w.get("targets_concept", 0.0)  # Wave 71
    )
    # Negative penalties from prereq violations can reduce the score but not below 0.
    capped = max(-max_total_boost, min(max_total_boost, raw))
    final = bm25_score * (1.0 + capped)
    if final < 0.0:
        final = 0.0
    return final, capped
