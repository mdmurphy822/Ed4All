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
    """Find the subset of query tokens (and hyphenated sub-slugs) that appear
    as node ids in ``graph_node_ids``.

    Handles the common case where the graph node is ``aria-labelledby`` and
    the query contains the literal phrase.  Each hyphenated slug is tested
    whole, and each hyphen-split substring is also tried so that a query
    like "focus management" still matches a ``focus-indicator`` node when
    the shared sub-token ``focus`` is a node id (some courses carry short
    node ids; this is a recall optimization, not an exact-match guarantee).
    """
    if not graph_node_ids:
        return set()
    q_tokens = _lower_tokens(query)
    return q_tokens & graph_node_ids


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

    def to_dict(self) -> Dict[str, float]:
        return {
            "concept_graph_overlap": round(self.concept_graph_overlap, 4),
            "lo_match": round(self.lo_match, 4),
            "prereq_coverage": round(self.prereq_coverage, 4),
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
    )
    # Negative penalties from prereq violations can reduce the score but not below 0.
    capped = max(-max_total_boost, min(max_total_boost, raw))
    final = bm25_score * (1.0 + capped)
    if final < 0.0:
        final = 0.0
    return final, capped
