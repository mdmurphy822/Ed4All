"""Query intent router (Wave 78 Worker C).

Wave 77 β shipped ``ed4all libv2 query`` for *user-driven* faceted
filtering — the user already knows what facets they want and the CLI
just applies them. ChatGPT's review identified that the package needs
an *intent-routed* retrieval surface where natural-language queries
dispatch to the right backend automatically:

* exact graph lookup for objective queries (``which chunks assess
  to-04?``);
* graph traversal for prerequisites / misconceptions (``what's a
  prerequisite for SHACL?``);
* hybrid concept-graph similarity + chunk text BM25 for open-ended
  concept questions (``how does sh:minCount work?``).

This module is the engine. It performs heuristic — not LLM —
classification (so it has no API key dependency, no latency tail, and
no opaque model decision) over a small set of canonical intent classes:

1. ``objective_lookup`` — query mentions a ``to-NN`` or ``co-NN`` ID.
   Routes to chunk_query filtered by the extracted outcome (with TO→CO
   rollup) plus an ``assesses`` / ``teaches`` edge bias diagnostic.
2. ``prerequisite_query`` — query contains ``before|prerequisite|...``
   markers. Routes to a direct pedagogy_graph ``prerequisite_of`` walk
   from the extracted concept.
3. ``misconception_query`` — query contains ``misconception|confuse|
   ...`` markers. Routes to ``MCP/tools/tutoring_tools.match_misconception``.
4. ``assessment_query`` — query mentions ``quiz|assessment|test|
   question|...``. Routes to chunk_query filtered to chunk_type ∈
   {assessment_item, exercise} plus the residual text as substring.
5. ``faceted_query`` — query carries explicit structural cues (week
   number, ``examples of ...``, bloom verb, chunk-type word) but no
   objective / prereq / misconception markers. Routes to chunk_query
   with the extracted facets.
6. ``concept_query`` (default fallback) — concept-graph similarity +
   chunk text BM25 hybrid over the residual query.

Two public entry points::

    classify_intent(query) -> {intent_class, confidence, extracted_entities, route}
    dispatch(query, slug, top_k=5) -> {intent_class, results, source_path,
                                        confidence, entities}

The dispatcher reuses Wave 77 β's ``LibV2/tools/chunk_query`` and Wave
77's ``MCP/tools/tutoring_tools`` for the heavy lifting; this module
only owns *intent classification + entity extraction + result envelope
shaping*.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from LibV2.tools.chunk_query import (
    BLOOM_LEVELS,
    CHUNK_TYPES,
    QueryFilter,
    UnknownSlugError,
    query_chunks,
)


__all__ = [
    "INTENT_CLASSES",
    "classify_intent",
    "dispatch",
    "extract_entities",
]


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

INTENT_CLASSES: Tuple[str, ...] = (
    "objective_lookup",
    "prerequisite_query",
    "misconception_query",
    "assessment_query",
    "faceted_query",
    "concept_query",
)


# Order matters here — both ``misconception`` triggers (``confuse``,
# ``misunderstand``, ``mistake``, ``common error``, ``wrong about``)
# and prereq triggers are handled before the assessment / faceted
# fallback so the most-specific intent wins.
_PREREQ_MARKERS = re.compile(
    r"(?i)\b("
    r"prerequisite|prereq|"
    r"before\s+(?:i|you|we|they|one|a\s|the\s|learning|studying|approach)"
    r"|need\s+to\s+know|prior\s+knowledge|comes\s+first|"
    r"depends\s+on|build\s+on"
    r")\b"
)
_MISCONCEPTION_MARKERS = re.compile(
    r"(?i)\b("
    r"misconceptions?|misunderstandings?|misunderstood|"
    r"confuse[ds]?|confusing|"
    r"mistakes?|common\s+errors?|"
    r"wrong\s+about|wrongly|incorrectly\s+(?:think|believe)"
    r")\b"
)
_ASSESSMENT_MARKERS = re.compile(
    r"(?i)\b("
    r"quiz|quizzes|assessment|assessments|test|tests|"
    r"question|questions|practice\s+problem|exercise|exercises"
    r")\b"
)
_OBJECTIVE_ID_RE = re.compile(r"(?i)\b(to|co)-(\d{2,})\b")
_WEEK_RE = re.compile(r"(?i)\bweek\s+(\d{1,2})\b")
_CHUNK_TYPE_WORDS = re.compile(
    r"(?i)\b(example|examples|exercise|exercises|overview|summary|"
    r"assessment|assessments|quiz|quizzes|practice)\b"
)

# Map from chunk-type-word matches → canonical chunk_type enum value.
_CHUNK_TYPE_WORD_MAP: Dict[str, str] = {
    "example": "example",
    "examples": "example",
    "exercise": "exercise",
    "exercises": "exercise",
    "overview": "overview",
    "summary": "summary",
    "assessment": "assessment_item",
    "assessments": "assessment_item",
    "quiz": "assessment_item",
    "quizzes": "assessment_item",
    "practice": "exercise",
}


# Lazy-loaded once: bloom verbs (lower-cased) → bloom level.
_BLOOM_VERB_MAP: Optional[Dict[str, str]] = None


def _bloom_verb_map() -> Dict[str, str]:
    """Lazily load the canonical bloom verb → level map."""
    global _BLOOM_VERB_MAP
    if _BLOOM_VERB_MAP is not None:
        return _BLOOM_VERB_MAP
    try:
        from lib.ontology.bloom import get_verbs_list
    except Exception:
        _BLOOM_VERB_MAP = {}
        return _BLOOM_VERB_MAP
    out: Dict[str, str] = {}
    for level, verbs in get_verbs_list().items():
        for v in verbs:
            out[v.lower()] = level
    _BLOOM_VERB_MAP = out
    return out


# --------------------------------------------------------------------------- #
# Entity extraction                                                           #
# --------------------------------------------------------------------------- #


def _extract_objective_ids(query: str) -> List[str]:
    """All ``to-NN`` / ``co-NN`` ids in the query, lower-cased and deduped."""
    out: List[str] = []
    seen: Set[str] = set()
    for match in _OBJECTIVE_ID_RE.finditer(query or ""):
        prefix = match.group(1).lower()
        digits = match.group(2)
        ident = f"{prefix}-{digits}"
        if ident not in seen:
            seen.add(ident)
            out.append(ident)
    return out


def _extract_weeks(query: str) -> List[int]:
    """All ``week N`` numbers in the query."""
    out: List[int] = []
    seen: Set[int] = set()
    for match in _WEEK_RE.finditer(query or ""):
        try:
            wk = int(match.group(1))
        except ValueError:  # pragma: no cover
            continue
        if wk not in seen:
            seen.add(wk)
            out.append(wk)
    return out


def _extract_bloom_verbs(query: str) -> List[Tuple[str, str]]:
    """Return ``[(verb, level), ...]`` for every bloom verb mention."""
    verb_map = _bloom_verb_map()
    if not verb_map:
        return []
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    # Tokenize on alpha-only boundaries (split hyphens too, so
    # "apply-level" -> ["apply", "level"]) for case-insensitive lookup.
    for match in re.finditer(r"[A-Za-z]+", query or ""):
        token = match.group(0).lower()
        if token in verb_map and token not in seen:
            seen.add(token)
            out.append((token, verb_map[token]))
    return out


def _extract_chunk_type_words(query: str) -> List[str]:
    """Return canonical chunk_type values for every chunk-type-word mention."""
    out: List[str] = []
    seen: Set[str] = set()
    for match in _CHUNK_TYPE_WORDS.finditer(query or ""):
        word = match.group(1).lower()
        canonical = _CHUNK_TYPE_WORD_MAP.get(word)
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


# Minimal stoplist for the "substantive residual" heuristic — only the
# very-high-frequency function words that don't disambiguate a
# substring filter. Domain terms ("RDF", "SHACL", "Turtle") never
# appear here on purpose.
_RESIDUAL_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "so", "if", "then", "than",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "what", "which", "who", "how", "why", "when", "where",
    "show", "me", "us", "give", "list", "find", "all", "some", "any",
    "do", "does", "did", "have", "has", "had",
    "chunks", "chunk", "content",
})


def _residual_text(query: str) -> str:
    """Return the query stripped of structural cue words + IDs.

    Used as the BM25 / substring text for the routed backend so we
    don't double-match on the cue itself (e.g. ``week 7`` shouldn't
    bleed into the substring filter).
    """
    text = query or ""
    text = _OBJECTIVE_ID_RE.sub(" ", text)
    text = _WEEK_RE.sub(" ", text)
    text = _PREREQ_MARKERS.sub(" ", text)
    text = _MISCONCEPTION_MARKERS.sub(" ", text)
    text = _ASSESSMENT_MARKERS.sub(" ", text)
    text = _CHUNK_TYPE_WORDS.sub(" ", text)
    # Collapse whitespace.
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(text: str) -> List[str]:
    """Return the alpha-numeric tokens of ``text`` minus question
    scaffolding. Used to decide whether a residual is "substantive"
    enough to apply as a substring filter (small leftover stop-only
    fragments like "Which chunks ?" should not filter at all).
    """
    if not text:
        return []
    out = []
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_:\-]*", text):
        if tok.lower() in _RESIDUAL_STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        out.append(tok)
    return out


def _substantive_residual(residual: str) -> Optional[str]:
    """Return ``residual`` if it carries content tokens; ``None`` else.

    A residual that's just question scaffolding ("Which chunks ?",
    "What is a for ?") would zero-out the substring filter; better to
    skip the filter entirely than wipe the result set.
    """
    if not residual or not residual.strip():
        return None
    tokens = _content_tokens(residual)
    if not tokens:
        return None
    # Prefer the longest contiguous content fragment so we still match
    # multi-word phrases ("SHACL validation", "RDF triples").
    return residual.strip()


def extract_entities(query: str) -> Dict[str, Any]:
    """Return all heuristic entities found in ``query``.

    Shape::

        {
            "objective_ids": ["to-04", ...],
            "weeks": [7, ...],
            "bloom_verbs": [("apply", "apply"), ...],
            "chunk_types": ["exercise", ...],
            "has_prereq_marker": bool,
            "has_misconception_marker": bool,
            "has_assessment_marker": bool,
            "residual_text": <query stripped of cue words>,
        }
    """
    return {
        "objective_ids": _extract_objective_ids(query),
        "weeks": _extract_weeks(query),
        "bloom_verbs": _extract_bloom_verbs(query),
        "chunk_types": _extract_chunk_type_words(query),
        "has_prereq_marker": bool(_PREREQ_MARKERS.search(query or "")),
        "has_misconception_marker": bool(_MISCONCEPTION_MARKERS.search(query or "")),
        "has_assessment_marker": bool(_ASSESSMENT_MARKERS.search(query or "")),
        "residual_text": _residual_text(query),
    }


# --------------------------------------------------------------------------- #
# Intent classification                                                       #
# --------------------------------------------------------------------------- #


def _classify(entities: Dict[str, Any]) -> Tuple[str, float, str]:
    """Return ``(intent_class, confidence, route)`` from entity envelope.

    Precedence order (most-specific wins):

    1. ``objective_lookup`` if any objective ID is present.
    2. ``misconception_query`` if a misconception marker is present.
    3. ``prerequisite_query`` if a prereq marker is present.
    4. ``assessment_query`` if an assessment marker is present.
    5. ``faceted_query`` if a structural cue (week / bloom / chunk-type)
       is present without any of the above.
    6. ``concept_query`` (default fallback).

    Confidence is a coarse, monotone score in ``[0.4, 1.0]`` derived
    from how many disambiguating signals fired. It's a sanity proxy,
    not a calibrated probability — callers can use it to threshold
    "ambiguous" results.
    """
    obj_ids = entities["objective_ids"]
    has_pre = entities["has_prereq_marker"]
    has_mc = entities["has_misconception_marker"]
    has_as = entities["has_assessment_marker"]
    weeks = entities["weeks"]
    bloom = entities["bloom_verbs"]
    ctypes = entities["chunk_types"]

    # 1. Objective lookup — strongest signal; an ID is an explicit ask.
    if obj_ids:
        confidence = 0.95 if len(obj_ids) == 1 else 0.85
        return (
            "objective_lookup",
            confidence,
            "chunk_query.outcome+pedagogy_graph(assesses|teaches)",
        )

    # 2. Misconception — explicit corrective framing.
    if has_mc:
        return (
            "misconception_query",
            0.9,
            "tutoring_tools.match_misconception",
        )

    # 3. Prerequisite — graph walk over prerequisite_of edges.
    if has_pre:
        return (
            "prerequisite_query",
            0.85,
            "pedagogy_graph.prerequisite_of",
        )

    # 4. Faceted — explicit structural cues (week / bloom / chunk-type)
    #    take precedence over a bare assessment marker because a week
    #    number or a bloom verb is a much stronger filter than the
    #    "assessments" cue alone. ("Show me apply-level exercises for
    #    week 7" -> faceted, not assessment_query.)
    if weeks or bloom or ctypes:
        # Confidence scales with the number of orthogonal facets.
        n_facets = (1 if weeks else 0) + (1 if bloom else 0) + (1 if ctypes else 0)
        confidence = 0.55 + 0.15 * min(n_facets, 3)
        return (
            "faceted_query",
            confidence,
            "chunk_query.facets",
        )

    # 5. Assessment — bare assessment marker without any structural cue.
    #    Routes to chunk_query with chunk_type bias to
    #    assessment_item / exercise. Reached only when the query is
    #    "give me a quiz / questions / practice problems" with no
    #    week / bloom / chunk-type-word qualifier.
    if has_as:
        return (
            "assessment_query",
            0.8,
            "chunk_query.chunk_type=assessment_item|exercise",
        )

    # 6. Concept — open-ended fallback. Confidence is intentionally
    #    low to flag "we're guessing" to auditing callers.
    return (
        "concept_query",
        0.5,
        "concept_graph.bm25_hybrid",
    )


def classify_intent(query: str) -> Dict[str, Any]:
    """Return ``{intent_class, confidence, extracted_entities, route}``.

    Pure function — no archive access, no I/O. Use :func:`dispatch` to
    actually run the routed retrieval.
    """
    entities = extract_entities(query or "")
    intent, confidence, route = _classify(entities)
    return {
        "intent_class": intent,
        "confidence": float(confidence),
        "extracted_entities": entities,
        "route": route,
    }


# --------------------------------------------------------------------------- #
# Backend dispatch helpers                                                    #
# --------------------------------------------------------------------------- #


def _course_dir(slug: str, courses_root: Optional[Path] = None) -> Path:
    """Resolve ``slug`` to a populated LibV2 course dir.

    Mirrors ``MCP/tools/tutoring_tools._course_dir`` — preferring the
    populated ``{slug}-{slug}`` form when both exist.
    """
    if courses_root is None:
        from lib.paths import LIBV2_COURSES

        courses_root = LIBV2_COURSES
    candidates = [courses_root / slug, courses_root / f"{slug}-{slug}"]
    for c in candidates:
        if c.is_dir() and (c / "corpus" / "chunks.jsonl").is_file():
            return c
    for c in candidates:
        if c.is_dir():
            return c
    return courses_root / slug


def _load_pedagogy_graph(slug: str, courses_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load the pedagogy_graph.json for ``slug`` (or empty stub)."""
    course = _course_dir(slug, courses_root=courses_root)
    path = course / "graph" / "pedagogy_graph.json"
    if not path.is_file():
        return {"nodes": [], "edges": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"nodes": [], "edges": []}


def _load_concept_graph(slug: str, courses_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load the concept_graph.json for ``slug`` (or empty stub)."""
    course = _course_dir(slug, courses_root=courses_root)
    path = course / "graph" / "concept_graph.json"
    if not path.is_file():
        return {"nodes": [], "edges": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"nodes": [], "edges": []}


def _normalize_concept_slug(target: str) -> str:
    """Strip ``concept:`` prefix; lowercase."""
    if not target:
        return ""
    if target.startswith("concept:"):
        target = target.split(":", 1)[1]
    return target.lower()


def _slugify_token(text: str) -> str:
    """Coarse hyphen-slug for matching against concept graph node ids."""
    if not text:
        return ""
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return s


def _candidate_concepts(query: str, slug: str, courses_root: Optional[Path] = None) -> List[str]:
    """Return concept-graph node ids that the query mentions.

    Strategy: build the set of concept-graph node ids (DomainConcept
    only, post-Wave-76), then for each id, check if its slug or label
    appears as a substring in the lowercased query. Returns matched
    ids sorted by descending node ``frequency`` (most central first)
    so the dispatcher's first-match rule picks the most-anchored
    concept.
    """
    graph = _load_concept_graph(slug, courses_root=courses_root)
    nodes = graph.get("nodes") or []
    q_lc = (query or "").lower()
    if not q_lc.strip():
        return []
    matches: List[Tuple[str, int]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if (n.get("class") or "DomainConcept") != "DomainConcept":
            continue
        cid = (n.get("id") or "").lower()
        label = (n.get("label") or "").lower()
        if not cid:
            continue
        # Match on the slug-id (hyphenated) by replacing hyphens with
        # space — so "rdf-graph" matches "rdf graph" in the query.
        cid_words = cid.replace("-", " ")
        if cid in q_lc or label in q_lc or cid_words in q_lc:
            matches.append((cid, int(n.get("frequency") or 0)))
    matches.sort(key=lambda t: (-t[1], t[0]))
    return [m[0] for m in matches]


# --------------------------------------------------------------------------- #
# Per-intent dispatchers                                                      #
# --------------------------------------------------------------------------- #


def _dispatch_objective(
    slug: str,
    entities: Dict[str, Any],
    top_k: int,
    courses_root: Optional[Path],
) -> List[Dict[str, Any]]:
    """Route ``objective_lookup`` to chunk_query filtered by the IDs.

    Bias the chunk_type filter toward ``assessment_item`` and
    ``exercise`` *only* if the query also has an assessment marker;
    otherwise return chunks tagged with the outcome (the full
    ``assesses`` ∪ ``teaches`` set, since ``learning_outcome_refs``
    on a chunk covers both).
    """
    objective_ids = entities["objective_ids"]
    if not objective_ids:
        return []
    chunk_types: Optional[Sequence[str]] = None
    if entities["has_assessment_marker"]:
        chunk_types = ["assessment_item", "exercise"]
    # Don't apply residual_text as a substring filter — the objective
    # ID is already the canonical filter, and the residual is usually
    # just question scaffolding ("Which chunks assess ?") that would
    # filter out every match.
    qf = QueryFilter(
        outcomes=objective_ids,
        chunk_types=chunk_types,
        limit=top_k,
        sort_key="chunk_id",
    )
    try:
        result = query_chunks(slug, qf, courses_root=courses_root)
    except UnknownSlugError:
        return []
    return list(result.chunks)


def _dispatch_prerequisite(
    slug: str,
    entities: Dict[str, Any],
    top_k: int,
    courses_root: Optional[Path],
) -> List[Dict[str, Any]]:
    """Route ``prerequisite_query`` to a pedagogy_graph walk.

    Resolution rule: pick the candidate concept (from the residual
    text) that has the largest set of ``prerequisite_of`` edges
    pointing *into* it; for that target, return the source concepts
    of those edges. Each result carries ``{concept, target,
    confidence, source_node}``.

    If no concept can be resolved, returns ``[]`` so the caller can
    surface "couldn't pin a concept" cleanly.
    """
    candidates = _candidate_concepts(entities["residual_text"] or "", slug, courses_root)
    graph = _load_pedagogy_graph(slug, courses_root=courses_root)
    edges = [
        e for e in (graph.get("edges") or [])
        if isinstance(e, dict) and e.get("relation_type") == "prerequisite_of"
    ]
    if not edges:
        return []

    # Build target -> [edges] index keyed by lower-cased slug.
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        target = _normalize_concept_slug(e.get("target") or "")
        if not target:
            continue
        by_target.setdefault(target, []).append(e)

    # Pick the candidate with the most prereq edges into it.
    target_choice: Optional[str] = None
    if candidates:
        candidates_norm = [_normalize_concept_slug(c) for c in candidates]
        ranked = [
            (c, len(by_target.get(c, [])))
            for c in candidates_norm
            if c in by_target
        ]
        if ranked:
            ranked.sort(key=lambda t: (-t[1], t[0]))
            target_choice = ranked[0][0]

    if target_choice is None:
        return []

    out: List[Dict[str, Any]] = []
    for e in by_target.get(target_choice, []):
        source = _normalize_concept_slug(e.get("source") or "")
        out.append({
            "concept": source,
            "target": target_choice,
            "confidence": e.get("confidence"),
            "relation": "prerequisite_of",
        })
    # Stable: sort by source slug for determinism.
    out.sort(key=lambda r: r["concept"])
    return out[: max(0, top_k)]


def _dispatch_misconception(
    slug: str,
    entities: Dict[str, Any],
    top_k: int,
    courses_root: Optional[Path],  # unused — tutoring_tools resolves on its own
) -> List[Dict[str, Any]]:
    """Route ``misconception_query`` to the tutoring_tools matcher.

    The residual text (cue words stripped) is what we score against
    the misconception statements. Empty residuals fall back to the
    raw query so a bare "misconceptions about RDF" still finds RDF
    misconceptions.
    """
    from MCP.tools.tutoring_tools import match_misconception

    text = entities["residual_text"] or ""
    if not text.strip():
        # Whole query was cue words — pass it through anyway, BM25 /
        # Jaccard will handle the cue-only case by returning empty.
        text = entities.get("_raw_query") or ""
    return list(match_misconception(slug, text, top_k=top_k))


def _dispatch_assessment(
    slug: str,
    entities: Dict[str, Any],
    top_k: int,
    courses_root: Optional[Path],
) -> List[Dict[str, Any]]:
    """Route ``assessment_query`` to chunk_query with chunk_type bias."""
    chunk_types: List[str] = list(entities["chunk_types"]) or [
        "assessment_item",
        "exercise",
    ]
    bloom_levels: Optional[List[str]] = None
    if entities["bloom_verbs"]:
        bloom_levels = sorted({lvl for _v, lvl in entities["bloom_verbs"]})
    week_min: Optional[int] = None
    week_max: Optional[int] = None
    if entities["weeks"]:
        wks = sorted(entities["weeks"])
        week_min, week_max = wks[0], wks[-1]
    # Skip residual_text — for assessment queries the structural
    # filters (chunk_type / bloom / week) are the canonical signal,
    # and the residual is usually scaffolding ("Show me a quiz on...").
    qf = QueryFilter(
        chunk_types=chunk_types,
        bloom_levels=bloom_levels,
        week_min=week_min,
        week_max=week_max,
        limit=top_k,
        sort_key="chunk_id",
    )
    try:
        result = query_chunks(slug, qf, courses_root=courses_root)
    except UnknownSlugError:
        return []
    return list(result.chunks)


def _dispatch_faceted(
    slug: str,
    entities: Dict[str, Any],
    top_k: int,
    courses_root: Optional[Path],
) -> List[Dict[str, Any]]:
    """Route ``faceted_query`` to chunk_query with all extracted facets."""
    chunk_types = list(entities["chunk_types"]) or None
    bloom_levels: Optional[List[str]] = None
    if entities["bloom_verbs"]:
        bloom_levels = sorted({lvl for _v, lvl in entities["bloom_verbs"]})
    week_min: Optional[int] = None
    week_max: Optional[int] = None
    if entities["weeks"]:
        wks = sorted(entities["weeks"])
        week_min, week_max = wks[0], wks[-1]
    # Skip residual_text — facets are the canonical filter, and the
    # residual is mostly scaffolding ("Show me ... for ..."). If the
    # residual carries strong content tokens, the concept_query path
    # would have been a better routing choice anyway.
    qf = QueryFilter(
        chunk_types=chunk_types,
        bloom_levels=bloom_levels,
        week_min=week_min,
        week_max=week_max,
        limit=top_k,
        sort_key="week",
    )
    try:
        result = query_chunks(slug, qf, courses_root=courses_root)
    except UnknownSlugError:
        return []
    return list(result.chunks)


# Tokenizer and lightweight BM25 for the concept-query fallback. We
# keep it dependency-free (no rank_bm25 import) so this module loads
# on a vanilla Python install.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:\-]*")


def _bm25_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_score(
    query_tokens: List[str],
    docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Tiny BM25Okapi implementation. Returns a normalized [0, 1] score
    per doc; higher = more relevant. Empty inputs -> all-zero."""
    n = len(docs)
    if n == 0 or not query_tokens:
        return [0.0] * n
    # Document frequencies.
    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    avgdl = sum(len(d) for d in docs) / max(1, n)
    import math

    raw: List[float] = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            n_t = df.get(term, 0)
            # +1 smoothing avoids divide-by-zero on rare terms.
            idf = math.log((n - n_t + 0.5) / (n_t + 0.5) + 1.0)
            denom = f + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
            score += idf * (f * (k1 + 1)) / denom
        raw.append(score)
    mx = max(raw) if raw else 0.0
    if mx <= 0:
        return [0.0] * n
    return [r / mx for r in raw]


def _dispatch_concept(
    slug: str,
    entities: Dict[str, Any],
    top_k: int,
    courses_root: Optional[Path],
) -> List[Dict[str, Any]]:
    """Route ``concept_query`` to a chunk-text BM25 over the residual text.

    True concept-graph similarity is out of scope here; the heavy
    lifting lives in ``LibV2/tools/libv2/retrieval_scoring.py`` and is
    architecturally too large for an intent-router fallback. The
    BM25-over-chunk-text path covers the common case (open-ended
    concept question) at low cost and parallels the "hybrid" envelope
    the design contract calls for.
    """
    text = entities["residual_text"] or entities.get("_raw_query") or ""
    if not text.strip():
        return []
    # Load all chunks via chunk_query (no filters) so we don't reach
    # into the JSONL directly — keeps the call tree single-pathed.
    qf = QueryFilter(sort_key="chunk_id")
    try:
        result = query_chunks(slug, qf, courses_root=courses_root)
    except UnknownSlugError:
        return []
    chunks = list(result.chunks)
    if not chunks:
        return []
    docs = [_bm25_tokenize(c.get("text") or "") for c in chunks]
    q_tokens = _bm25_tokenize(text)
    scores = _bm25_score(q_tokens, docs)
    ranked = sorted(
        (
            {**c, "score": float(s)}
            for c, s in zip(chunks, scores)
            if s > 0
        ),
        key=lambda r: r["score"],
        reverse=True,
    )
    return ranked[: max(0, top_k)]


# --------------------------------------------------------------------------- #
# Public dispatcher                                                           #
# --------------------------------------------------------------------------- #


_DISPATCH_TABLE = {
    "objective_lookup": _dispatch_objective,
    "prerequisite_query": _dispatch_prerequisite,
    "misconception_query": _dispatch_misconception,
    "assessment_query": _dispatch_assessment,
    "faceted_query": _dispatch_faceted,
    "concept_query": _dispatch_concept,
}


def dispatch(
    query: str,
    slug: str,
    *,
    top_k: int = 5,
    courses_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Classify ``query`` and dispatch to the routed retrieval primitive.

    Returns::

        {
            "query": <input>,
            "slug": <input>,
            "intent_class": <one of INTENT_CLASSES>,
            "confidence": float,
            "route": <human-readable backend descriptor>,
            "entities": <extract_entities() output>,
            "results": [...],   # backend-specific shape
            "source_path": <route>,
        }

    Backends fail-soft on unknown slugs / missing graphs (they return
    ``[]``), so the caller can detect "intent classified, but the
    archive isn't there" without exception handling.
    """
    classification = classify_intent(query or "")
    entities = dict(classification["extracted_entities"])
    # Carry the raw query so the misconception / concept dispatchers
    # can fall back when the residual text is empty.
    entities["_raw_query"] = query or ""

    intent = classification["intent_class"]
    fn = _DISPATCH_TABLE.get(intent)
    if fn is None:  # pragma: no cover - guarded by INTENT_CLASSES
        results: List[Dict[str, Any]] = []
    else:
        results = fn(slug, entities, top_k, courses_root)

    # Drop the internal _raw_query before exposing to caller.
    entities.pop("_raw_query", None)

    return {
        "query": query or "",
        "slug": slug,
        "intent_class": intent,
        "confidence": classification["confidence"],
        "route": classification["route"],
        "source_path": classification["route"],
        "entities": entities,
        "results": results,
    }
