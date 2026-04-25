"""Misconception-aware tutoring tools (Wave 77).

Surfaces the editorial misconception/correction pairs anchored in
LibV2-archived chunks so downstream consumers (diagnostic loops,
pre-generation guardrails, pedagogy inventory) can match against them
directly. BM25 over chunk text cannot reach the diagnostic intent
because the corrective framing isn't in the indexable surface form;
it's in the structured ``chunk.misconceptions[]`` envelope and the
pedagogy graph's ``interferes_with`` edges (DomainConcept-only after
Wave 76 pruning).

Three public entry points:

* :func:`match_misconception` — student utterance vs. misconception
  statements (cosine / TF-IDF / BM25 / Jaccard, in graceful-degrade
  order). Returns top-k matches with chunk_id + source_references.
* :func:`preemptive_misconception_guardrails` — given a target concept
  slug, returns every misconception whose ``interferes_with`` edge in
  the pedagogy graph points at that concept. Use to instruct an LLM
  "do not commit these errors when explaining {concept}".
* :func:`cluster_misconceptions` — KMeans (or greedy fallback) over
  the misconception statements to produce a semantic inventory of the
  corpus's misconceptions for human review.

Each return envelope carries ``backend`` so callers can tell which
similarity tier was used. Heavy ML libs (sentence-transformers,
scikit-learn, rank_bm25) are imported lazily inside the helpers — the
module imports cleanly on a vanilla Python install.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lib.paths import LIBV2_COURSES

__all__ = [
    "match_misconception",
    "preemptive_misconception_guardrails",
    "cluster_misconceptions",
    "load_misconception_index",
]


# ---------------------------------------------------------------------- #
# Tokenization + Jaccard fallback
# ---------------------------------------------------------------------- #


# Stoplist for the Jaccard fallback. Kept tiny + obviously non-content
# so we don't accidentally over-remove technical terms (e.g. "is" appears
# as a copula in many misconceptions like "An RDF triple is like a row").
_STOPLIST = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
        "and", "or", "but", "not", "no", "so", "if", "then", "than", "that",
        "this", "these", "those", "it", "its", "they", "them", "their",
        "i", "you", "we", "he", "she", "his", "her", "our", "your",
        "do", "does", "did", "have", "has", "had", "can", "could", "should",
        "would", "may", "might", "must", "will", "shall",
        "about", "into", "over", "under", "between", "through",
        "all", "some", "any", "each", "every", "what", "which", "who",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


def _tokenize(text: str, *, drop_stopwords: bool = True) -> List[str]:
    """Lowercase alpha-num token extraction with optional stoplist filter."""
    if not text:
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPLIST and len(t) > 1]
    return toks


def _jaccard(a: List[str], b: List[str]) -> float:
    """Token-set Jaccard. Empty either side -> 0.0."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------- #
# Backend selection (graceful degradation chain)
# ---------------------------------------------------------------------- #


def _select_backend() -> str:
    """Probe optional dependencies and return the highest-fidelity tier
    available. Imports happen here only — never at module top — so a
    vanilla Python install always loads this module.

    Order: embedding -> tfidf -> bm25 -> jaccard.
    """
    try:
        import sentence_transformers  # noqa: F401

        return "embedding"
    except Exception:
        pass
    try:
        import sklearn  # noqa: F401

        return "tfidf"
    except Exception:
        pass
    try:
        import rank_bm25  # noqa: F401

        return "bm25"
    except Exception:
        pass
    return "jaccard"


# ---------------------------------------------------------------------- #
# Per-(slug, mtime) misconception index cache
# ---------------------------------------------------------------------- #


_INDEX_CACHE: Dict[Tuple[str, float], "MisconceptionIndex"] = {}


class MisconceptionIndex:
    """Loaded misconceptions for one LibV2 course.

    Walks ``LibV2/courses/{slug}/corpus/chunks.jsonl`` and the pedagogy
    graph at ``LibV2/courses/{slug}/graph/pedagogy_graph.json`` to build:

    * ``items``: list of canonical misconception records with statement,
      correction, originating chunk_id, source_references, concept_tags.
    * ``concept_to_mc_keys``: concept_slug -> set of statement-keys for
      O(1) ``preemptive_misconception_guardrails`` lookup.
    """

    def __init__(self, slug: str, items: List[Dict[str, Any]],
                 concept_to_mc_keys: Dict[str, set]):
        self.slug = slug
        self.items = items
        self.concept_to_mc_keys = concept_to_mc_keys

    def __len__(self) -> int:
        return len(self.items)


def _course_dir(slug: str) -> Path:
    """Resolve a course slug to a LibV2 course directory.

    Tries the slug directly first, then ``{slug}-{slug}`` (the doubled
    form that ``Trainforge/process_course.py`` emits when course_id ==
    course_slug, e.g. ``rdf-shacl-550-rdf-shacl-550``).

    Preference rule: a candidate is "good" only if it actually carries
    a populated archive (``corpus/chunks.jsonl`` exists). The repo
    sometimes contains an empty scaffold dir at the bare-slug path
    alongside the populated doubled-slug dir; without this check we
    would route to the empty scaffold and silently return zero items.
    """
    candidates = [
        LIBV2_COURSES / slug,
        LIBV2_COURSES / f"{slug}-{slug}",
    ]
    # Prefer any candidate that has a populated chunks.jsonl
    for c in candidates:
        if c.is_dir() and (c / "corpus" / "chunks.jsonl").is_file():
            return c
    # Fall back to the first existing dir (preserves error visibility).
    for c in candidates:
        if c.is_dir():
            return c
    return LIBV2_COURSES / slug


def _normalize_concept_slug(target: str) -> str:
    """Turn ``concept:rdf-graph`` (graph node id) into ``rdf-graph``."""
    if not target:
        return ""
    if target.startswith("concept:"):
        return target.split(":", 1)[1]
    return target


def load_misconception_index(slug: str) -> MisconceptionIndex:
    """Load (and cache by mtime) the misconception index for ``slug``.

    Cache key is ``(slug, max(mtime of chunks.jsonl, mtime of
    pedagogy_graph.json))`` so an updated archive invalidates cleanly.
    Missing files -> empty index, never an exception. (The CLI surfaces
    "no misconceptions found" as a clear empty-result envelope.)
    """
    course = _course_dir(slug)
    chunks_path = course / "corpus" / "chunks.jsonl"
    graph_path = course / "graph" / "pedagogy_graph.json"

    mtime = 0.0
    for p in (chunks_path, graph_path):
        if p.exists():
            mtime = max(mtime, p.stat().st_mtime)

    cache_key = (slug, mtime)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    items: List[Dict[str, Any]] = []
    seen_keys: set = set()
    if chunks_path.exists():
        import json

        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mcs = chunk.get("misconceptions") or []
                if not mcs:
                    continue
                chunk_id = chunk.get("id")
                source = chunk.get("source") or {}
                source_refs = list(source.get("source_references") or [])
                concept_tags = list(chunk.get("concept_tags") or [])
                for mc in mcs:
                    if not isinstance(mc, dict):
                        continue
                    statement = (mc.get("misconception") or "").strip()
                    correction = (mc.get("correction") or "").strip()
                    if not statement:
                        continue
                    # Dedupe on statement+correction pair: same pair anchored
                    # in multiple chunks counts once but we keep the FIRST
                    # chunk_id (the chunk where it first appeared) plus that
                    # chunk's source_references so downstream consumers can
                    # cite a canonical origin.
                    key = (statement, correction)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    items.append({
                        "misconception": statement,
                        "correction": correction,
                        "chunk_id": chunk_id,
                        "source_references": source_refs,
                        "concept_tags": concept_tags,
                    })

    # Pedagogy graph: map concept_slug -> set of misconception statements
    # via ``interferes_with`` edges. Wave 76 Worker D pruned these to
    # DomainConcept-only targets, so this lookup yields high-signal
    # concepts only (no rhetorical-question or reading-budget noise).
    concept_to_mc_keys: Dict[str, set] = defaultdict(set)
    if graph_path.exists():
        import json

        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            graph = {}
        nodes = {n.get("id"): n for n in (graph.get("nodes") or []) if isinstance(n, dict)}
        for edge in (graph.get("edges") or []):
            if not isinstance(edge, dict):
                continue
            if edge.get("relation_type") != "interferes_with":
                continue
            mc_node_id = edge.get("source")
            target = edge.get("target") or ""
            concept_slug = _normalize_concept_slug(target)
            if not concept_slug or not mc_node_id:
                continue
            mc_node = nodes.get(mc_node_id) or {}
            statement = (mc_node.get("statement")
                         or mc_node.get("label") or "").strip()
            if statement:
                concept_to_mc_keys[concept_slug].add(statement)

    index = MisconceptionIndex(
        slug=slug,
        items=items,
        concept_to_mc_keys=dict(concept_to_mc_keys),
    )
    _INDEX_CACHE[cache_key] = index
    return index


# ---------------------------------------------------------------------- #
# Public tools
# ---------------------------------------------------------------------- #


def _score_jaccard(query: str, statements: List[str]) -> List[float]:
    """Token-set Jaccard between query and each statement."""
    q_tokens = _tokenize(query)
    return [_jaccard(q_tokens, _tokenize(s)) for s in statements]


def _score_bm25(query: str, statements: List[str]) -> List[float]:
    """rank_bm25 BM25Okapi scoring (lazy import). Falls through to
    Jaccard on import failure (defensive even though _select_backend
    already gated us)."""
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return _score_jaccard(query, statements)
    tokens = [_tokenize(s) for s in statements]
    if not any(tokens):
        return [0.0] * len(statements)
    bm25 = BM25Okapi(tokens)
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [0.0] * len(statements)
    raw = list(bm25.get_scores(q_tokens))
    # Normalize to [0, 1] so the score is comparable across queries.
    mx = max(raw) if raw else 0.0
    if mx <= 0:
        return [0.0] * len(raw)
    return [r / mx for r in raw]


def _score_tfidf(query: str, statements: List[str]) -> List[float]:
    """sklearn TF-IDF cosine. Lazy import; jaccard fallback on failure."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return _score_jaccard(query, statements)
    if not statements:
        return []
    docs = [query] + statements
    try:
        vec = TfidfVectorizer(stop_words="english", lowercase=True)
        mat = vec.fit_transform(docs)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        return [float(s) for s in sims]
    except Exception:
        return _score_jaccard(query, statements)


def _score_embedding(query: str, statements: List[str]) -> List[float]:
    """sentence-transformers cosine using all-MiniLM-L6-v2. Lazy load."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return _score_tfidf(query, statements)
    if not statements:
        return []
    try:
        # Cache on first call.
        global _ST_MODEL
        try:
            model = _ST_MODEL  # type: ignore[name-defined]
        except NameError:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            _ST_MODEL = model  # type: ignore[assignment]
        q_emb = model.encode([query], convert_to_tensor=True)
        s_emb = model.encode(statements, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, s_emb).cpu().numpy().ravel()
        return [float(s) for s in sims]
    except Exception:
        return _score_tfidf(query, statements)


def _score(backend: str, query: str, statements: List[str]) -> List[float]:
    """Dispatch scoring to the requested backend with safe fallbacks."""
    if backend == "embedding":
        return _score_embedding(query, statements)
    if backend == "tfidf":
        return _score_tfidf(query, statements)
    if backend == "bm25":
        return _score_bm25(query, statements)
    return _score_jaccard(query, statements)


def match_misconception(
    slug: str,
    student_text: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Match free-form student input against the corpus's misconceptions.

    Returns a list of ``{misconception, correction, chunk_id,
    source_references, concept_tags, score, backend}`` records sorted
    by descending similarity. Only items with score > 0 are returned.

    Empty ``student_text`` or a slug with no misconceptions yields ``[]``.
    """
    if not student_text or not student_text.strip():
        return []
    index = load_misconception_index(slug)
    if not index.items:
        return []
    backend = _select_backend()
    statements = [it["misconception"] for it in index.items]
    # Score against the misconception STATEMENT (the wrong-belief frame
    # is what a struggling student's text resembles, not the correction).
    scores = _score(backend, student_text, statements)
    ranked: List[Dict[str, Any]] = []
    for it, sc in zip(index.items, scores):
        if sc <= 0:
            continue
        ranked.append({
            "misconception": it["misconception"],
            "correction": it["correction"],
            "chunk_id": it["chunk_id"],
            "source_references": it["source_references"],
            "concept_tags": it["concept_tags"],
            "score": float(sc),
            "backend": backend,
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[: max(0, top_k)]


def preemptive_misconception_guardrails(
    slug: str,
    concept_slug: str,
) -> List[Dict[str, Any]]:
    """Return misconceptions whose ``interferes_with`` target is
    ``concept_slug``. Use to instruct an LLM "do not commit these
    errors when explaining {concept}".

    Resolution: pedagogy graph edges (``interferes_with`` relation_type)
    map mc_<hash> nodes -> ``concept:<slug>`` targets. Wave 76 Worker D
    pruned these to DomainConcept targets only, so the result is high-
    signal. The mc-node statement is matched back to its corpus chunk
    by exact statement text so we can carry chunk_id + source_refs.

    Empty list if the concept has no recorded interferers, including
    unknown concepts. ``concept:rdf-graph`` and ``rdf-graph`` are both
    accepted (the ``concept:`` prefix is stripped).
    """
    if not concept_slug:
        return []
    index = load_misconception_index(slug)
    target = _normalize_concept_slug(concept_slug.strip())
    statements = index.concept_to_mc_keys.get(target, set())
    if not statements:
        return []
    # Match each statement back to its chunk record so we can attach
    # the canonical chunk_id + source_references. A statement that has
    # an interferes_with edge but no corpus chunk anchor (rare; would
    # only happen if pedagogy graph drifted from chunks.jsonl) is still
    # surfaced — chunk_id falls back to None.
    by_statement = {it["misconception"]: it for it in index.items}
    out: List[Dict[str, Any]] = []
    for s in sorted(statements):
        item = by_statement.get(s)
        if item is not None:
            out.append({
                "misconception": item["misconception"],
                "correction": item["correction"],
                "chunk_id": item["chunk_id"],
                "source_references": item["source_references"],
                "concept_tags": item["concept_tags"],
                "concept_slug": target,
            })
        else:
            out.append({
                "misconception": s,
                "correction": "",
                "chunk_id": None,
                "source_references": [],
                "concept_tags": [],
                "concept_slug": target,
            })
    return out


# ---------------------------------------------------------------------- #
# Clustering
# ---------------------------------------------------------------------- #


def _cluster_kmeans(
    statements: List[str], n_clusters: int
) -> Tuple[List[int], Optional[List[List[float]]]]:
    """KMeans over TF-IDF vectors. Returns (labels, vectors-as-list).
    Lazy sklearn import; raises ImportError on failure so callers can
    fall back to greedy."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    mat = vec.fit_transform(statements)
    n_clusters = max(1, min(n_clusters, len(statements)))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(mat).tolist()
    # Return dense vectors for centroid-distance label-picking.
    vectors = mat.toarray().tolist()
    return labels, vectors


def _cluster_greedy(
    statements: List[str], n_clusters: int
) -> List[int]:
    """Greedy clustering by Jaccard threshold. Picks ``n_clusters``
    seeds (the longest statements, tie-broken by index), then assigns
    every statement to the nearest seed by Jaccard similarity. Used
    when sklearn isn't available — coarse, but bounded + deterministic.
    """
    n = len(statements)
    if n == 0:
        return []
    n_clusters = max(1, min(n_clusters, n))
    # Seeds: pick the n_clusters longest statements (more tokens =>
    # higher chance of distinct semantic anchors than a random pick).
    seed_indices = sorted(
        range(n), key=lambda i: (-len(statements[i]), i)
    )[:n_clusters]
    seed_tokens = [_tokenize(statements[i]) for i in seed_indices]
    labels: List[int] = []
    for stmt in statements:
        toks = _tokenize(stmt)
        best, best_score = 0, -1.0
        for ci, st in enumerate(seed_tokens):
            sc = _jaccard(toks, st)
            if sc > best_score:
                best_score = sc
                best = ci
        labels.append(best)
    return labels


def cluster_misconceptions(
    slug: str,
    n_clusters: int = 8,
) -> List[Dict[str, Any]]:
    """Cluster the corpus's misconceptions into semantic groups.

    Returns a list of cluster dicts with shape::

        {
            "label": <statement closest to centroid>,
            "members": [<statement>, ...],
            "size": <int>,
            "canonical_correction": <longest correction across members>,
            "backend": <"kmeans"|"greedy">,
        }

    Empty list if the corpus has no misconceptions.
    """
    index = load_misconception_index(slug)
    if not index.items:
        return []
    statements = [it["misconception"] for it in index.items]
    by_stmt = {it["misconception"]: it for it in index.items}

    backend_name = "kmeans"
    labels: List[int]
    vectors: Optional[List[List[float]]] = None
    try:
        labels, vectors = _cluster_kmeans(statements, n_clusters)
    except Exception:
        labels = _cluster_greedy(statements, n_clusters)
        backend_name = "greedy"

    # Group members by cluster label.
    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        grouped[lab].append(idx)

    clusters: List[Dict[str, Any]] = []
    for lab in sorted(grouped.keys()):
        member_indices = grouped[lab]
        members = [statements[i] for i in member_indices]
        # Pick label = member closest to the cluster centroid in the
        # tf-idf vector space (kmeans path) or longest statement
        # (greedy path).
        if vectors is not None and member_indices:
            # Compute cluster centroid + each member's distance to it.
            dim = len(vectors[0])
            centroid = [0.0] * dim
            for i in member_indices:
                v = vectors[i]
                for d in range(dim):
                    centroid[d] += v[d]
            inv_n = 1.0 / len(member_indices)
            centroid = [c * inv_n for c in centroid]
            best_i, best_dist = member_indices[0], float("inf")
            for i in member_indices:
                v = vectors[i]
                dist = sum((v[d] - centroid[d]) ** 2 for d in range(dim))
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            label_text = statements[best_i]
        else:
            label_text = max(members, key=len)

        # Canonical correction = longest correction across members.
        corrections = [by_stmt[m]["correction"] for m in members
                       if by_stmt[m]["correction"]]
        canonical_correction = max(corrections, key=len) if corrections else ""

        clusters.append({
            "label": label_text,
            "members": members,
            "size": len(members),
            "canonical_correction": canonical_correction,
            "backend": backend_name,
        })

    return clusters
