"""W2 §4.2 — Pass D: embedding-dedup of near-duplicate candidate objectives.

Pass B partitions a chapter's chunks into disjoint windows (overlap = 0), so a
concept split across a window boundary produces near-duplicate candidate
objectives BY DESIGN. Pass D collapses them: embed every surviving (Pass-C)
candidate's statement, single-link cluster by cosine, and pick the best-grounded
representative per cluster (unioning every cluster member's ``source_chunk_ids``
onto the representative so a merged CO keeps all its supporting chunks).

N is small (tens to low-hundreds of objectives) so an O(N²) cosine matrix is fine
— no ANN index needed. The threshold ships ADVISORY (``_DEFAULT_DEDUP_THRESHOLD``
= 0.88, env-overridable) and the result carries ``max_pairwise_cosine`` /
``near_dup_pairs`` so a downstream calibration harness can pin it on real data
(Risk R6; W2 ships the MEASUREMENT fields, not the calibration).

Graceful degrade: ``build_embedding_client`` raises ``EmbeddingBackendUnavailable``
(extras absent) AND not strict → pass-through (every candidate its own cluster,
``available=False``, no collapse). Mirrors Pass C's NLI-absent pass-through.
``provider="fake"`` is refused unless ``allow_fake`` (the anti-poisoning gate
already in ``lib/embedding/providers.py``).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lib.embedding._math import cosine_similarity
from lib.embedding.providers import (
    EmbeddingBackendUnavailable,
    build_embedding_client,
)

logger = logging.getLogger(__name__)

#: Cosine ≥ this clusters two candidate objectives. ADVISORY starting point —
#: Risk R6 mandates measuring ``max_pairwise_cosine`` on a real run before
#: pinning; env-overridable via ``ED4ALL_OBJECTIVE_DEDUP_THRESHOLD``.
_DEFAULT_DEDUP_THRESHOLD = 0.88

#: Env override for the dedup threshold (cross-cutting ``ED4ALL_*``, root-owned).
ENV_DEDUP_THRESHOLD = "ED4ALL_OBJECTIVE_DEDUP_THRESHOLD"

#: Strict-mode flag bridge (mirrors the NLI/embedding strict contract). When
#: truthy, an absent embedding backend RAISES rather than passing through.
ENV_REQUIRE_EMBEDDINGS = "TRAINFORGE_REQUIRE_EMBEDDINGS"

#: WS1 — cosine clustering threshold for BOTTOM-UP TO derivation. Deliberately
#: LOWER than the dedup threshold (0.88): dedup collapses near-identical
#: restatements, whereas TO-clustering groups RELATED COs into coarse themes
#: (~8–12 clusters). Roadmap range 0.45–0.55. Env-overridable.
_DEFAULT_TO_CLUSTER_THRESHOLD = 0.50
ENV_TO_CLUSTER_THRESHOLD = "ED4ALL_TO_CLUSTER_THRESHOLD"

#: WS1.1 — TARGET-K Ward agglomerative clustering supersedes WS1's single-link
#: cosine-threshold TO clustering. A real 7B run proved single-link has NO good
#: operating point (1 mega-cluster at ≤0.70 → dozens of singletons at ≥0.80);
#: calibration on the real 68-CO embeddings proved Ward linkage (euclidean) on
#: L2-normalized vectors at K≈12 gives balanced clusters. ``ED4ALL_TO_CLUSTER_K``
#: pins a FIXED K (0 = auto); auto derives K from the CO count via
#: ``ED4ALL_TO_COS_PER_CLUSTER`` (≈ COs per cluster, default 6), clamped [3, 15].
_DEFAULT_TO_CLUSTER_K = 0  # 0 = auto (derive from n)
ENV_TO_CLUSTER_K = "ED4ALL_TO_CLUSTER_K"
_DEFAULT_TO_COS_PER_CLUSTER = 6
ENV_TO_COS_PER_CLUSTER = "ED4ALL_TO_COS_PER_CLUSTER"
#: Auto-K is clamped into this band (mirrors WS1's _clamp_cluster_count target).
_TO_CLUSTER_K_LO = 3
_TO_CLUSTER_K_HI = 15

#: Fix 1A — cap on cited chunks per merged objective. When near-duplicate
#: candidates merge, every member's ``source_chunk_ids`` is unioned; with no cap
#: a merged CO accreted 25+ chunks (most off-topic). Rank the union by cosine to
#: the rep's STATEMENT and keep the top-K. ALWAYS keeps ≥1 (never zero
#: provenance). Env-overridable; default 5.
_DEFAULT_MAX_CHUNKS_PER_OBJECTIVE = 5
ENV_MAX_CHUNKS_PER_OBJECTIVE = "ED4ALL_OBJECTIVE_MAX_CHUNKS_PER_OBJECTIVE"

#: Fix 1A — relevance floor for a unioned chunk. A chunk whose cosine to the
#: rep statement is below this is dropped (even within the top-K) — diffuse
#: provenance is worse than thin provenance ("no source rather than a misleading
#: one"). Subject to the ALWAYS-keep-≥1 contract. Env-overridable; default 0.30.
_DEFAULT_CHUNK_RELEVANCE_FLOOR = 0.30
ENV_CHUNK_RELEVANCE_FLOOR = "ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR"

#: I3 PRONG A — distinct-skill SPLIT. Default OFF (opt-in). When on, a
#: post-clustering pass re-surfaces a DISTINCT named skill that single-link
#: transitively chained into a broader cluster (e.g. "order of operations /
#: PEMDAS" chains into a "simplify expressions" cluster, loses its statement,
#: and demotes to supporting evidence). The 0.88 same-signature collapse for
#: exact restatements is preserved; only clusters spanning ≥2 DISTINCT skill
#: signatures are split. ANTI-FABRICATION: the split only re-surfaces an
#: already-synthesized, already-grounded candidate statement — never invents one.
_DEFAULT_DISTINCT_SKILL_SPLIT = False
ENV_DISTINCT_SKILL_SPLIT = "ED4ALL_OBJECTIVE_DISTINCT_SKILL_SPLIT"

#: Defect E — CROSS-WINDOW lexical-dedup SECOND PASS. Pass B partitions a
#: chapter into disjoint windows, so a concept restated across a window boundary
#: yields near-duplicate COs. The single-link cosine pass at the 0.88 dedup
#: threshold catches near-identical embeddings, but a RESTATEMENT at ~0.80 cosine
#: (different phrasing of the same skill) slips through — a real scan-corpus run left
#: ~35-50 residual dupes, one 50-member cosine chain. This opt-in second pass runs
#: AFTER ``cluster_by_cosine`` and BEFORE the PRONG-A distinct-skill split (so a
#: wrongly-chained cosine cluster can still split), merging two clusters iff their
#: centroid cosine >= ``ED4ALL_OBJECTIVE_DEDUP_LEXICAL_COSINE`` (0.78) AND their
#: best-grounded members' skill-signature Jaccard >=
#: ``ED4ALL_OBJECTIVE_DEDUP_LEXICAL_JACCARD`` (0.60) AND they share a Bloom band.
#: **Complete linkage** (every cross pair must pass both floors) structurally
#: prevents 50-member chains. Default OFF → byte-identical DedupResult.
_DEFAULT_DEDUP_LEXICAL = False
ENV_DEDUP_LEXICAL = "ED4ALL_OBJECTIVE_DEDUP_LEXICAL"

#: Defect E — centroid-cosine floor for a lexical merge edge. Deliberately BELOW
#: the 0.88 single-link dedup threshold (this pass exists to catch the sub-0.88
#: restatements the cosine pass missed) but high enough that only genuinely
#: related clusters are candidates. Env-overridable; default 0.78.
_DEFAULT_DEDUP_LEXICAL_COSINE = 0.78
ENV_DEDUP_LEXICAL_COSINE = "ED4ALL_OBJECTIVE_DEDUP_LEXICAL_COSINE"

#: Defect E — skill-signature Jaccard floor for a lexical merge edge. Deliberately
#: ABOVE PRONG-A's distinctness band (<0.34): two clusters merge only when their
#: named-skill signatures OVERLAP heavily (same skill, different words), the exact
#: inverse of the split. Env-overridable; default 0.60.
_DEFAULT_DEDUP_LEXICAL_JACCARD = 0.60
ENV_DEDUP_LEXICAL_JACCARD = "ED4ALL_OBJECTIVE_DEDUP_LEXICAL_JACCARD"

#: P1 (clustering rework) — post-cluster GUARDS for bottom-up TO derivation.
#: ``cluster_to_target_k`` runs Ward at a HARD target K with no guards, which on
#: a real ~85-CO run produced (1) a semantic-OUTLIER CO getting its OWN
#: singleton cluster → a wholesale-hallucinated TO; (2) tiny 1-2-CO "runt"
#: clusters that should fold into a sibling. The CONSOLIDATE pass
#: (outlier-absorb + undersize-merge) folds every sub-``min_size`` cluster into
#: its nearest-centroid neighbor. Master gate; default OFF so every current run
#: stays byte-identical/reproducible — the operator opts in. Mirrors the
#: ``_DEFAULT_DISTINCT_SKILL_SPLIT`` opt-in posture exactly.
#:
#: NOTE (backstop supersedes this for size-1 clusters): the "a genuinely
#: distinct singleton REMAINS standing" philosophy below is SUPERSEDED for
#: SIZE-1 clusters by the unconditional :func:`dissolve_singletons` backstop —
#: a lone CO is never a valid terminal objective (a TO aggregates >=2 COs), so
#: it always folds into its nearest neighbor regardless of this floor. The
#: absorb floor here still governs MULTI-member runts (2..min_size-1).
_DEFAULT_TO_CLUSTER_GUARDS = False
ENV_TO_CLUSTER_GUARDS = "ED4ALL_TO_CLUSTER_GUARDS"

#: P1 — clusters smaller than this are merge/absorb candidates for the
#: CONSOLIDATE pass. Env-overridable; default 3. Only consulted when the
#: consolidation is enabled (``ED4ALL_TO_CLUSTER_GUARDS`` on or kwarg passed).
_DEFAULT_TO_OUTLIER_MIN_SIZE = 3
ENV_TO_OUTLIER_MIN_SIZE = "ED4ALL_TO_OUTLIER_MIN_SIZE"

#: P1 — absolute "has a clear home" floor for the OUTLIER-absorb decision. A
#: sub-``min_size`` cluster is absorbed into its nearest-centroid neighbor only
#: when that neighbor's centroid cosine is >= this floor; below it the cluster
#: may remain standing (a genuinely distinct competency with no good home).
#: Env-overridable; default 0.20 (deliberately low — a tiny cluster that is
#: closest to SOME sibling almost always belongs there). SUPERSEDED for SIZE-1
#: clusters by the unconditional :func:`dissolve_singletons` backstop (a lone CO
#: is never a valid TO); this floor now governs only MULTI-member runts.
_DEFAULT_TO_OUTLIER_ABSORB_FLOOR = 0.20
ENV_TO_OUTLIER_ABSORB_FLOOR = "ED4ALL_TO_OUTLIER_ABSORB_FLOOR"

#: P1 — NEAR-DUPLICATE-TO merge. Ward at a hard K over-splits a cosine-dense
#: theme (e.g. a number-line theme split into 3 TOs whose centroids are ~0.85
#: similar → near-duplicate TOs). Two clusters whose centroid cosine >= this
#: floor are UNIONED (transitively, via union-find — like ``cluster_by_cosine``).
#: This is the DETERMINISTIC post-cluster merge the P1 plan prefers over
#: non-deterministic silhouette K. Gated behind ``ED4ALL_TO_MERGE_NEAR_DUP``;
#: default OFF so DEFAULT behavior stays byte-identical/reproducible.
_DEFAULT_TO_MERGE_NEAR_DUP = False
ENV_TO_MERGE_NEAR_DUP = "ED4ALL_TO_MERGE_NEAR_DUP"
_DEFAULT_TO_MERGE_COSINE = 0.85
ENV_TO_MERGE_COSINE = "ED4ALL_TO_MERGE_COSINE"

#: Anti-hallucinated-TO backstop — OPT-OUT (default OFF → singletons ARE
#: dissolved = fix ON). A terminal objective is definitionally an AGGREGATE of
#: >=2 chapter objectives, so a size-1 cluster must never author a standalone
#: TO: a lone semantic-outlier CO would otherwise be promoted into a
#: wholesale course-wide terminal competency that content generation weaves
#: through many pages. Unlike the opt-in CONSOLIDATE guard (which DELIBERATELY
#: leaves a genuinely-distinct singleton standing when no neighbor clears the
#: absorb floor), :func:`dissolve_singletons` runs UNCONDITIONALLY and with NO
#: floor — a lone CO always folds UNDER its nearest neighbor (the concept it
#: exemplifies). Truthy (``1``/``true``/``yes``/``on``) restores the legacy
#: keep-singleton behavior (a byte-stable escape hatch).
_DEFAULT_TO_ALLOW_SINGLETON_TO = False
ENV_TO_ALLOW_SINGLETON_TO = "ED4ALL_TO_ALLOW_SINGLETON_TO"

#: Tiny-course floor for :func:`dissolve_singletons`. Folding halts before the
#: surviving cluster count would drop below this, so a genuinely tiny course
#: (few themes, mostly singletons) is not collapsed into one mega-TO. Default
#: 3; garbage / < 1 → 3 (parse-with-fallback).
_DEFAULT_TO_MIN_CLUSTERS = 3
ENV_TO_MIN_CLUSTERS = "ED4ALL_TO_MIN_CLUSTERS"


@dataclass(frozen=True)
class DedupResult:
    """Outcome of the Pass-D dedup over a grounded candidate pool."""

    canonical: List[Dict[str, Any]] = field(default_factory=list)
    clusters: List[List[int]] = field(default_factory=list)
    max_pairwise_cosine: float = 0.0
    near_dup_pairs: int = 0
    available: bool = False
    #: Fix 1A — total chunk_ids pruned across all merged objectives (calibration
    #: signal surfaced onto ``grounding_signals``). 0 when prune didn't run
    #: (legacy callers without ``chunks_by_id``).
    pruned_chunk_total: int = 0
    #: Fix 1A — the resolved per-objective cap (top-K) used for this run.
    max_chunks_per_objective: int = 0
    #: I3 PRONG A — number of clusters re-partitioned by the distinct-skill
    #: split (0 when the split is off or no cluster spanned ≥2 skills).
    clusters_split_for_distinct_skill: int = 0
    #: I3 PRONG A — total distinct named-skill groups after the split (== number
    #: of canonical COs the split produced; == len(clusters) when split off).
    distinct_skill_count: int = 0
    #: Defect E — number of successful cross-window lexical MERGE operations (each
    #: unions two cosine clusters). 0 when the lexical pass is off / no-op.
    lexical_merge_pairs: int = 0
    #: Defect E — number of INPUT cosine clusters absorbed by the lexical pass
    #: (== clusters_before - clusters_after). 0 when off / no-op.
    lexical_clusters_merged: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical": [dict(c) for c in self.canonical],
            "clusters": [list(g) for g in self.clusters],
            "max_pairwise_cosine": round(float(self.max_pairwise_cosine), 6),
            "near_dup_pairs": self.near_dup_pairs,
            "available": self.available,
            "pruned_chunk_total": self.pruned_chunk_total,
            "max_chunks_per_objective": self.max_chunks_per_objective,
            "clusters_split_for_distinct_skill": (
                self.clusters_split_for_distinct_skill
            ),
            "distinct_skill_count": self.distinct_skill_count,
            "lexical_merge_pairs": self.lexical_merge_pairs,
            "lexical_clusters_merged": self.lexical_clusters_merged,
        }


def resolve_dedup_threshold(threshold: Optional[float] = None) -> float:
    """Resolve the dedup threshold: explicit arg → env → default.

    Out-of-range / garbage env values fall back to the default (a misconfigured
    threshold must never silently disable clustering).
    """
    if threshold is not None:
        return float(threshold)
    raw = os.environ.get(ENV_DEDUP_THRESHOLD)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_DEDUP_THRESHOLD
        if 0.0 < val <= 1.0:
            return val
    return _DEFAULT_DEDUP_THRESHOLD


def resolve_to_cluster_threshold(threshold: Optional[float] = None) -> float:
    """Resolve the WS1 TO-cluster threshold: explicit arg → env → default.

    Out-of-range / garbage env values fall back to the default (a misconfigured
    threshold must never silently disable clustering). Mirrors
    :func:`resolve_dedup_threshold`'s parse-with-fallback posture; accepts
    ``0.0 < val <= 1.0``.
    """
    if threshold is not None:
        return float(threshold)
    raw = os.environ.get(ENV_TO_CLUSTER_THRESHOLD)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_TO_CLUSTER_THRESHOLD
        if 0.0 < val <= 1.0:
            return val
    return _DEFAULT_TO_CLUSTER_THRESHOLD


def resolve_to_cos_per_cluster(value: Optional[int] = None) -> int:
    """Resolve the WS1.1 target COs-per-cluster divisor: explicit → env → default.

    Garbage / non-positive env values fall back to the default (a misconfigured
    divisor must never crash or produce a zero/negative auto-K). Mirrors the
    parse-with-fallback posture of the other ``resolve_*`` helpers.
    """
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get(ENV_TO_COS_PER_CLUSTER)
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_TO_COS_PER_CLUSTER
        if val >= 1:
            return val
    return _DEFAULT_TO_COS_PER_CLUSTER


def resolve_to_cluster_k(n: int, explicit: Optional[int] = None) -> int:
    """Resolve the WS1.1 TARGET-K for Ward TO clustering.

    Resolution (high → low precedence):
      explicit arg  >  ``ED4ALL_TO_CLUSTER_K`` (fixed K; 0 = auto)  >  auto

    Auto-K = ``max(3, min(15, ceil(n / cos_per_cluster)))`` where
    ``cos_per_cluster`` comes from :func:`resolve_to_cos_per_cluster`. The final
    K is always clamped to ``[3, 15]`` AND to ``n`` (you can never ask for more
    clusters than points). Garbage env values fall back to auto. ``n <= 0`` → 1.
    """
    import math

    if n <= 0:
        return 1

    def _clamp(k: int) -> int:
        return max(1, min(k, n))

    if explicit is not None:
        return _clamp(max(1, int(explicit)))

    raw = os.environ.get(ENV_TO_CLUSTER_K)
    if raw:
        try:
            fixed = int(raw)
        except (TypeError, ValueError):
            fixed = _DEFAULT_TO_CLUSTER_K
        if fixed >= 1:
            return _clamp(fixed)
        # fixed <= 0 → fall through to auto.

    cos_per_cluster = resolve_to_cos_per_cluster()
    auto = max(
        _TO_CLUSTER_K_LO,
        min(_TO_CLUSTER_K_HI, math.ceil(n / cos_per_cluster)),
    )
    return _clamp(auto)


def cluster_to_target_k(
    vecs: List[List[float]], k: int, *, linkage: str = "ward"
) -> List[List[int]]:
    """WS1.1 — TARGET-K Ward agglomerative clustering of CO statement vectors.

    Supersedes WS1's single-link cosine-threshold TO clustering, which a real
    7B run proved collapses cosine-dense CO embeddings into 1 catch-all + a
    singleton (no good threshold operating point). Ward linkage (euclidean) on
    L2-normalized vectors at a target K gives balanced clusters.

    L2-normalizes the vectors internally (so ward/euclidean ≈ spherical /
    cosine-like geometry), then runs ``AgglomerativeClustering(n_clusters=
    min(k, n), linkage=linkage)`` and returns clusters as lists of ORIGINAL
    indices, ordered by minimum member index, each member list ascending —
    MATCHING :func:`cluster_by_cosine`'s ordering contract so downstream stays
    deterministic.

    Guards: ``n == 0`` → ``[]``; ``n == 1`` → ``[[0]]``; ``k < 1`` → treated as
    1. GRACEFUL FALLBACK: if sklearn cannot be imported, fall back to
    :func:`cluster_by_cosine` at :func:`resolve_to_cluster_threshold` (the
    no-sklearn single-link path) and log a warning — never crash.
    """
    import math

    n = len(vecs)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
    except Exception as exc:  # noqa: BLE001 — ImportError or any load failure
        logger.warning(
            "WS1.1 cluster_to_target_k: sklearn unavailable (%s); falling back "
            "to single-link cluster_by_cosine at the TO-cluster threshold.",
            exc,
        )
        clusters, _max, _near = cluster_by_cosine(
            vecs, resolve_to_cluster_threshold()
        )
        return clusters

    k_eff = max(1, int(k))
    k_eff = min(k_eff, n)

    # L2-normalize so euclidean ward ≈ spherical (cosine-like) geometry.
    normalized: List[List[float]] = []
    for v in vecs:
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            normalized.append([x / norm for x in v])
        else:
            normalized.append(list(v))

    labels = AgglomerativeClustering(
        n_clusters=k_eff, linkage=linkage
    ).fit_predict(normalized)

    clusters_by_label: Dict[Any, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters_by_label.setdefault(lab, []).append(idx)
    # Each member list ascending; clusters ordered by minimum member index —
    # matches cluster_by_cosine's deterministic ordering contract.
    ordered = sorted(
        (sorted(members) for members in clusters_by_label.values()),
        key=lambda g: min(g),
    )
    return ordered


def cluster_by_cosine(
    vecs: List[List[float]], threshold: float
) -> Tuple[List[List[int]], float, int]:
    """Single-link (agglomerative) cluster by cosine ≥ ``threshold``.

    Returns ``(ordered_clusters, max_pairwise_cosine, near_dup_pairs)``. Clusters
    are ordered by minimum member index; members are ascending. O(N²);
    ``n == 0`` → ``([], 0.0, 0)``; ``n == 1`` → ``([[0]], 0.0, 0)``. Lower-index
    -as-root union keeps cluster order stable (preserves the deterministic
    on-disk shape the dedup snapshot tests assert).

    This is the shared graph-clustering core extracted from
    :func:`dedup_candidates` (WS1 §1). Both the dedup pass (at the 0.88 dedup
    threshold) and the bottom-up TO derivation (at the lower TO-cluster
    threshold) call it.
    """
    n = len(vecs)
    if n == 0:
        return [], 0.0, 0

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            # Keep the lower index as the root so cluster order follows course
            # order (the rep is chosen by the caller; ties → course order).
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    max_cos = 0.0
    near_dup_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            cos = cosine_similarity(vecs[i], vecs[j])
            if cos > max_cos:
                max_cos = cos
            if cos >= threshold:
                near_dup_pairs += 1
                _union(i, j)

    # Collect clusters keyed by root, preserving member (course) order.
    clusters_by_root: Dict[int, List[int]] = {}
    for i in range(n):
        root = _find(i)
        clusters_by_root.setdefault(root, []).append(i)
    # Order clusters by their minimum member index (course order of the rep).
    ordered_clusters = sorted(clusters_by_root.values(), key=lambda g: min(g))
    return ordered_clusters, max_cos, near_dup_pairs


def resolve_max_chunks_per_objective(value: Optional[int] = None) -> int:
    """Resolve the per-objective chunk cap: explicit arg → env → default.

    Out-of-range / garbage env values fall back to the default (a misconfigured
    cap must never silently disable the prune or drop to zero). Mirrors
    :func:`resolve_dedup_threshold`'s parse-with-fallback posture.
    """
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get(ENV_MAX_CHUNKS_PER_OBJECTIVE)
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_MAX_CHUNKS_PER_OBJECTIVE
        if val >= 1:
            return val
    return _DEFAULT_MAX_CHUNKS_PER_OBJECTIVE


def resolve_chunk_relevance_floor(floor: Optional[float] = None) -> float:
    """Resolve the chunk relevance floor: explicit arg → env → default.

    Out-of-range / garbage env values fall back to the default. A floor of 0.0
    disables the relevance drop (cap still applies); >1.0 / negative / non-float
    → default.
    """
    if floor is not None:
        return float(floor)
    raw = os.environ.get(ENV_CHUNK_RELEVANCE_FLOOR)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_CHUNK_RELEVANCE_FLOOR
        if 0.0 <= val <= 1.0:
            return val
    return _DEFAULT_CHUNK_RELEVANCE_FLOOR


def resolve_distinct_skill_split(enabled: Optional[bool] = None) -> bool:
    """Resolve the I3 PRONG-A distinct-skill split toggle: arg → env → default.

    Default OFF (opt-in). Truthy env values (``1``/``true``/``yes``/``on``)
    enable; anything else (incl. garbage) → the default. Mirrors the
    parse-with-fallback posture of the other ``resolve_*`` helpers.
    """
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_DISTINCT_SKILL_SPLIT, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_DISTINCT_SKILL_SPLIT


def resolve_dedup_lexical(enabled: Optional[bool] = None) -> bool:
    """Resolve the Defect-E lexical-merge master gate: arg → env → default.

    Default OFF (opt-in). Truthy env values (``1``/``true``/``yes``/``on``)
    enable the cross-window lexical second pass; anything else (incl. garbage) →
    the default. Mirrors :func:`resolve_distinct_skill_split`.
    """
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_DEDUP_LEXICAL, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_DEDUP_LEXICAL


def resolve_dedup_lexical_cosine(floor: Optional[float] = None) -> float:
    """Resolve the Defect-E lexical-merge centroid-cosine floor: arg → env → default.

    Out-of-range / garbage env values fall back to the default (a misconfigured
    floor must never silently collapse all clusters). Accepts ``0.0 < val <= 1.0``.
    """
    if floor is not None:
        return float(floor)
    raw = os.environ.get(ENV_DEDUP_LEXICAL_COSINE)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_DEDUP_LEXICAL_COSINE
        if 0.0 < val <= 1.0:
            return val
    return _DEFAULT_DEDUP_LEXICAL_COSINE


def resolve_dedup_lexical_jaccard(floor: Optional[float] = None) -> float:
    """Resolve the Defect-E lexical-merge skill-signature Jaccard floor.

    Out-of-range / garbage env values fall back to the default. Accepts
    ``0.0 < val <= 1.0``. Mirrors :func:`resolve_dedup_lexical_cosine`.
    """
    if floor is not None:
        return float(floor)
    raw = os.environ.get(ENV_DEDUP_LEXICAL_JACCARD)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_DEDUP_LEXICAL_JACCARD
        if 0.0 < val <= 1.0:
            return val
    return _DEFAULT_DEDUP_LEXICAL_JACCARD


#: Tokens that carry no teachable-skill signal — stripped from the keyphrase
#: signature so "simplify expressions" and "simplify the expression" collapse
#: but "order operations" stays distinct from "simplify expressions".
_SKILL_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with",
        "by", "from", "as", "at", "is", "are", "be", "that", "this", "these",
        "those", "it", "its", "their", "your", "you", "we", "they", "given",
        "using", "use", "able", "students", "student", "will", "can", "each",
        "into", "out", "up", "down", "than", "then", "when", "which", "what",
        "how", "why", "where", "expression", "expressions", "problem",
        "problems", "value", "values", "number", "numbers", "real", "world",
        "following", "appropriate", "correctly", "clearly", "properly",
    }
)


def _skill_keyphrase_tokens(statement: str) -> frozenset:
    """Deterministic content-token signature of a statement's NAMED skill.

    Lowercases, strips punctuation, drops Bloom verbs (the action) + generic
    stopwords (the scaffolding), and returns the residual content tokens — the
    skill's *object*. Two statements teaching the SAME skill share most of
    these tokens; two distinct skills ("order operations" vs "simplify
    expression" vs "associative property") do not. CHEAP + embedding-OPTIONAL
    (pure string ops) so the split is never a new fail-closed dependency.
    """
    import re as _re

    try:
        from lib.ontology.bloom import get_all_verbs

        verbs = {v.lower() for v in get_all_verbs()}
    except Exception:  # noqa: BLE001 — bloom load must never break the split
        verbs = set()

    toks = _re.findall(r"[a-z]+", str(statement).lower())
    out = {
        t
        for t in toks
        if len(t) > 2 and t not in _SKILL_STOPWORDS and t not in verbs
    }
    return frozenset(out)


def _filler_tokens() -> frozenset:
    """Shared Defect-B/E filler-token set (empty when the lexicon is absent).

    One indirection through :mod:`lib.objectives.filler_lexicon` so BOTH the
    Defect-E ``_skill_signature`` subtraction here and the Defect-B
    ``ObjectiveSpecificityValidator`` residual subtract the SAME domain-agnostic
    filler vocabulary (plan: "one shared helper used by both B and E"). Never
    raises — a missing lexicon degrades to an empty set (no subtraction).
    """
    try:
        from lib.objectives.filler_lexicon import filler_tokens

        return filler_tokens()
    except Exception:  # noqa: BLE001 — filler lexicon must never break the split
        return frozenset()


def _skill_signature(candidate: Dict[str, Any]) -> Optional[frozenset]:
    """Return a candidate's distinct-skill signature, or None if empty.

    Combines the statement keyphrase tokens with any ``concept_tags`` /
    ``domain_concept_vocabulary`` the candidate carries (those are explicit
    named-concept signals), then SUBTRACTS the shared Defect-B/E filler lexicon
    (:func:`_filler_tokens`) so domain-agnostic scaffolding ("various", "key",
    "concepts", "techniques") never inflates a signature or props two vacuous
    restatements apart. An empty residual → ``None`` (no nameable skill; such a
    candidate never forces a split and is a natural lexical-merge magnet).
    """
    sig: set = set(_skill_keyphrase_tokens(_statement(candidate)))
    for key in ("concept_tags", "domain_concepts", "domain_concept_vocabulary"):
        raw = candidate.get(key)
        if isinstance(raw, list):
            for tag in raw:
                tok = str(tag).strip().lower()
                if tok:
                    sig.add(tok)
    sig -= _filler_tokens()
    return frozenset(sig) if sig else None


def _signatures_distinct(a: frozenset, b: frozenset, *, min_jaccard: float = 0.34) -> bool:
    """True when two skill signatures name DISTINCT skills.

    Jaccard overlap below ``min_jaccard`` → distinct. Exact restatements share
    nearly all tokens (Jaccard → 1.0) and stay merged; "order operations" vs
    "simplify associative property" share ~0 and split. The floor is
    deliberately low (precision over recall — only split clearly-distinct
    skills, never near-restatements). Two empty signatures are NOT distinct.
    """
    if not a or not b:
        return False
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return False
    return (inter / union) < min_jaccard


def split_clusters_for_distinct_skills(
    clusters: List[List[int]],
    grounded: List[Dict[str, Any]],
) -> Tuple[List[List[int]], int, int]:
    """I3 PRONG A — split clusters spanning ≥2 DISTINCT named skills.

    Single-link cosine clustering TRANSITIVELY chains distinct-but-related
    skills (cos(A,B)≥t ∧ cos(B,C)≥t ⇒ A,B,C one cluster even if cos(A,C) is
    low + A,C are distinct skills). The best-grounded rep keeps ONE statement;
    the rest's statements are discarded. This pass re-partitions each cluster
    by skill signature so every distinct named skill keeps its OWN
    representative.

    Algorithm (deterministic, embedding-FREE): within a cluster, greedily
    assign members to skill-groups — a member joins an existing group iff its
    signature is NOT distinct from the group's seed signature; otherwise it
    seeds a new group. Members with no nameable signature attach to the first
    group (they don't force a split). Returns
    ``(new_clusters, clusters_split, distinct_skill_count)`` where
    ``new_clusters`` preserves the deterministic min-index ordering contract.

    ANTI-FABRICATION: this only RE-PARTITIONS existing candidate indices — it
    never adds, removes, or invents a member.
    """
    new_clusters: List[List[int]] = []
    clusters_split = 0
    distinct_skill_count = 0

    for cluster in clusters:
        if len(cluster) < 2:
            new_clusters.append(list(cluster))
            distinct_skill_count += 1 if cluster else 0
            continue

        # Build per-member signatures (None = no nameable skill).
        sigs = {idx: _skill_signature(grounded[idx]) for idx in cluster}

        # Greedy skill-grouping: each group has a seed signature.
        groups: List[Dict[str, Any]] = []  # {"seed": frozenset, "members": [int]}
        for idx in cluster:  # cluster is ascending → deterministic
            sig = sigs[idx]
            if sig is None:
                # No nameable skill → attach to the first group (or seed one).
                if groups:
                    groups[0]["members"].append(idx)
                else:
                    groups.append({"seed": frozenset(), "members": [idx]})
                continue
            placed = False
            for grp in groups:
                seed = grp["seed"]
                if seed and not _signatures_distinct(sig, seed):
                    grp["members"].append(idx)
                    placed = True
                    break
            if not placed:
                groups.append({"seed": sig, "members": [idx]})

        if len(groups) > 1:
            clusters_split += 1
        distinct_skill_count += len(groups)

        for grp in groups:
            new_clusters.append(sorted(grp["members"]))

    # Re-impose the deterministic ordering contract (by min member index).
    new_clusters.sort(key=lambda g: min(g) if g else 0)
    return new_clusters, clusters_split, distinct_skill_count


# ---------------------------------------------------------------------------
# Defect E — CROSS-WINDOW lexical-dedup SECOND PASS (complete-linkage merge).
# ---------------------------------------------------------------------------

#: Coarse Bloom bands the lexical merge respects (a merge never crosses a band).
#: remember/understand share the lower-order recall band; apply is its own band;
#: analyze/evaluate/create share the higher-order band. An unknown level maps to
#: its own ``"unknown"`` band (only merges with another unknown), so a
#: level-less restatement pair can still merge but never bleeds a lower-order CO
#: into a higher-order cluster.
_BLOOM_BAND_BY_LEVEL: Dict[str, str] = {
    "remember": "lower",
    "understand": "lower",
    "apply": "apply",
    "analyze": "higher",
    "evaluate": "higher",
    "create": "higher",
}


def _bloom_band(candidate: Dict[str, Any]) -> str:
    """Return a candidate's coarse Bloom band ("lower"|"apply"|"higher"|"unknown").

    Prefers the candidate's DECLARED ``bloom_level`` (real field, no imputation);
    falls back to detecting the level from the statement's main verb via
    ``lib.ontology.bloom.detect_bloom_level``. An undetectable level → "unknown"
    (only band-equal to another unknown).
    """
    level = str(candidate.get("bloom_level") or "").strip().lower()
    if level not in _BLOOM_BAND_BY_LEVEL:
        try:
            from lib.ontology.bloom import detect_bloom_level

            detected, _verb = detect_bloom_level(_statement(candidate))
            level = str(detected or "").strip().lower()
        except Exception:  # noqa: BLE001 — bloom load must never break the merge
            level = ""
    return _BLOOM_BAND_BY_LEVEL.get(level, "unknown")


def _best_grounded_member(cluster: List[int], grounded: List[Dict[str, Any]]) -> int:
    """Index of a cluster's best-grounded member (highest entailment; ties → longer
    statement; ties → lowest index). Matches ``dedup_candidates``' rep-selection so
    the lexical signature/band are computed over the SAME member that becomes the
    canonical representative.
    """
    def _key(idx: int) -> Any:
        return (
            _entailment_score(grounded[idx]),
            len(_statement(grounded[idx])),
            -idx,
        )

    return max(cluster, key=_key)


def _jaccard(a: Optional[frozenset], b: Optional[frozenset]) -> float:
    """Jaccard overlap of two skill signatures; two empties → 0.0 (never merge)."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def merge_clusters_lexical(
    clusters: List[List[int]],
    grounded: List[Dict[str, Any]],
    vecs: List[List[float]],
    *,
    cosine_floor: Optional[float] = None,
    jaccard_floor: Optional[float] = None,
) -> Tuple[List[List[int]], int, int]:
    """Defect E — COMPLETE-LINKAGE lexical merge of near-restatement clusters.

    Runs after ``cluster_by_cosine`` and BEFORE the PRONG-A distinct-skill split.
    A merge EDGE between two clusters exists iff ALL three hold:

    * centroid cosine >= ``cosine_floor`` (:func:`resolve_dedup_lexical_cosine`),
    * best-grounded-member skill-signature Jaccard >= ``jaccard_floor``
      (:func:`resolve_dedup_lexical_jaccard`), computed via :func:`_skill_signature`
      (which subtracts the shared filler lexicon), and
    * the two best-grounded members share a Bloom band (:func:`_bloom_band`).

    Groups are agglomerated GREEDILY in descending ``(jaccard, cosine)`` edge
    order; a group-merge executes ONLY when EVERY cross pair between the two
    groups is itself an edge (**complete linkage**) — this structurally prevents
    a wrongly-chained mega-cluster (the 50-member chain the single-link pass
    left). Ties break to the lower minimum cluster index (deterministic).

    Returns ``(new_clusters, merge_ops, clusters_absorbed, merged_counts)`` where
    ``new_clusters`` unions each group's member indices and re-imposes the
    deterministic min-index ordering contract, ``merge_ops`` counts executed
    union operations, ``clusters_absorbed`` == ``clusters_before -
    clusters_after``, and ``merged_counts`` is a list PARALLEL to
    ``new_clusters`` giving the number of INPUT cosine-clusters folded into each
    output cluster (1 = untouched; >1 = lexically merged — the stamp source for
    ``dedup_lexical_merged_count``). PARTITION INVARIANT: the flattened output is
    a permutation of the flattened input — never adds / drops / invents a CO.
    """
    cfloor = resolve_dedup_lexical_cosine(cosine_floor)
    jfloor = resolve_dedup_lexical_jaccard(jaccard_floor)

    work = [list(c) for c in clusters if c]
    nc = len(work)
    if nc <= 1:
        return _reorder_clusters(work), 0, 0, [1] * len(work)

    # Per-cluster centroid / best-grounded signature / Bloom band.
    centroids = [_cluster_centroid(vecs, c) for c in work]
    reps = [_best_grounded_member(c, grounded) for c in work]
    sigs = [_skill_signature(grounded[r]) for r in reps]
    bands = [_bloom_band(grounded[r]) for r in reps]

    # Edge matrix: qualifying (i, j) with their (jaccard, cosine).
    def _edge_ok(i: int, j: int) -> Tuple[bool, float, float]:
        if bands[i] != bands[j]:
            return False, 0.0, 0.0
        jac = _jaccard(sigs[i], sigs[j])
        if jac < jfloor:
            return False, jac, 0.0
        cos = cosine_similarity(centroids[i], centroids[j])
        if cos < cfloor:
            return False, jac, cos
        return True, jac, cos

    edges: List[Tuple[float, float, int, int]] = []
    ok_pair: Dict[Tuple[int, int], bool] = {}
    for i in range(nc):
        for j in range(i + 1, nc):
            ok, jac, cos = _edge_ok(i, j)
            ok_pair[(i, j)] = ok
            if ok:
                edges.append((jac, cos, i, j))

    # Descending (jaccard, cosine); ties → lower (i, j) for determinism.
    edges.sort(key=lambda e: (-e[0], -e[1], e[2], e[3]))

    # Greedy complete-linkage agglomeration over cluster indices.
    groups: List[List[int]] = [[i] for i in range(nc)]
    group_of: List[int] = list(range(nc))
    merge_ops = 0

    def _pair_ok(a: int, b: int) -> bool:
        return ok_pair.get((a, b) if a < b else (b, a), False)

    for _jac, _cos, i, j in edges:
        gi, gj = group_of[i], group_of[j]
        if gi == gj:
            continue
        members_i, members_j = groups[gi], groups[gj]
        # Complete linkage: EVERY cross pair must be an edge.
        if all(_pair_ok(a, b) for a in members_i for b in members_j):
            # Merge gj into gi; keep the lower group id (lower min-index) as home.
            keep, drop = (gi, gj) if min(members_i) <= min(members_j) else (gj, gi)
            groups[keep].extend(groups[drop])
            for m in groups[drop]:
                group_of[m] = keep
            groups[drop] = []
            merge_ops += 1

    # Union each surviving group's member indices into output clusters, tracking
    # how many INPUT clusters composed each (for the merged-count stamp). Key by
    # the output cluster's minimum member index — stable under _reorder_clusters.
    unmerged: List[List[int]] = []
    count_by_minidx: Dict[int, int] = {}
    for grp in groups:
        if not grp:
            continue
        merged: List[int] = []
        for ci in grp:
            merged.extend(work[ci])
        unmerged.append(merged)
        count_by_minidx[min(merged)] = len(grp)

    ordered = _reorder_clusters(unmerged)
    merged_counts = [count_by_minidx.get(min(c), 1) for c in ordered]
    clusters_absorbed = nc - len(ordered)
    return ordered, merge_ops, clusters_absorbed, merged_counts


def _emit_lexical_merge_capture(
    capture: Any,
    *,
    merge_ops: int,
    clusters_absorbed: int,
    clusters_before: int,
    cosine_floor: float,
    jaccard_floor: float,
) -> None:
    """Best-effort ``objective_lexical_merge`` decision-capture (never raises)."""
    try:
        capture.log_decision(
            decision_type="objective_lexical_merge",
            decision=(
                f"lexically merged {clusters_absorbed} cross-window "
                f"restatement cluster(s) via {merge_ops} complete-linkage "
                f"union(s) ({clusters_before} -> {clusters_before - clusters_absorbed})"
            ),
            rationale=(
                f"Defect E: single-link cosine dedup (0.88) left cross-window "
                f"restatements at sub-threshold cosine; a complete-linkage second "
                f"pass merged clusters whose centroid cosine >= {cosine_floor:.2f} "
                f"AND best-grounded skill-signature Jaccard >= {jaccard_floor:.2f} "
                f"(filler-subtracted) AND share a Bloom band. Complete linkage "
                f"(every cross pair must qualify) structurally prevents the "
                f"50-member cosine chain. Anti-fabrication: unions existing "
                f"candidate indices only — no statement invented."
            ),
            alternatives_considered=[
                "single-link only (leaves ~35-50 residual cross-window dupes)",
                "single-link lexical merge (risks a re-chained mega-cluster)",
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("objective_lexical_merge capture failed (%s); continuing", exc)


# ---------------------------------------------------------------------------
# P1 (clustering rework) — post-cluster GUARDS for bottom-up TO derivation.
# All operate on cluster INDEX-lists + their vectors (PRE-id-mint), so
# downstream learning_outcome_refs continuity is preserved — they only
# RE-PARTITION existing CO indices, never add / remove / invent a CO.
# ---------------------------------------------------------------------------


def resolve_to_cluster_guards(enabled: Optional[bool] = None) -> bool:
    """Resolve the P1 cluster-consolidation master gate: arg → env → default.

    Default OFF (opt-in). Truthy env values (``1``/``true``/``yes``/``on``)
    enable the outlier-absorb + undersize-merge consolidation; anything else
    (incl. garbage) → the default. Mirrors :func:`resolve_distinct_skill_split`.
    """
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_TO_CLUSTER_GUARDS, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_TO_CLUSTER_GUARDS


def resolve_to_outlier_min_size(value: Optional[int] = None) -> int:
    """Resolve the consolidate min-cluster-size: explicit arg → env → default.

    Garbage / non-positive env values fall back to the default (a misconfigured
    min-size must never crash or fold everything). Mirrors
    :func:`resolve_to_cos_per_cluster`'s parse-with-fallback posture.
    """
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get(ENV_TO_OUTLIER_MIN_SIZE)
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_TO_OUTLIER_MIN_SIZE
        if val >= 1:
            return val
    return _DEFAULT_TO_OUTLIER_MIN_SIZE


def resolve_to_outlier_absorb_floor(floor: Optional[float] = None) -> float:
    """Resolve the outlier-absorb "has a clear home" floor: arg → env → default.

    Out-of-range / garbage env values fall back to the default. A floor of 0.0
    absorbs any tiny cluster into its nearest neighbor unconditionally.
    """
    if floor is not None:
        return float(floor)
    raw = os.environ.get(ENV_TO_OUTLIER_ABSORB_FLOOR)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_TO_OUTLIER_ABSORB_FLOOR
        if 0.0 <= val <= 1.0:
            return val
    return _DEFAULT_TO_OUTLIER_ABSORB_FLOOR


def resolve_to_merge_near_dup(enabled: Optional[bool] = None) -> bool:
    """Resolve the near-duplicate-TO merge gate: arg → env → default.

    Default OFF (opt-in). Truthy env values (``1``/``true``/``yes``/``on``)
    enable; anything else (incl. garbage) → the default. Mirrors
    :func:`resolve_distinct_skill_split`.
    """
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_TO_MERGE_NEAR_DUP, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_TO_MERGE_NEAR_DUP


def resolve_to_merge_cosine(floor: Optional[float] = None) -> float:
    """Resolve the near-duplicate-TO centroid-cosine merge floor: arg → env → default.

    Out-of-range / garbage env values fall back to the default (a misconfigured
    floor must never silently collapse all clusters). Accepts ``0.0 < val <= 1.0``.
    """
    if floor is not None:
        return float(floor)
    raw = os.environ.get(ENV_TO_MERGE_COSINE)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_TO_MERGE_COSINE
        if 0.0 < val <= 1.0:
            return val
    return _DEFAULT_TO_MERGE_COSINE


def resolve_to_allow_singleton_to(enabled: Optional[bool] = None) -> bool:
    """Resolve the singleton-TO opt-OUT gate: arg → env → default.

    Default OFF (``False``) → singletons ARE dissolved (the fix is ON). Truthy
    env values (``1``/``true``/``yes``/``on``) → ``True`` → the legacy
    keep-singleton behavior. Anything else (incl. garbage / unset) → the
    default. Mirrors :func:`resolve_to_cluster_guards`' parse-with-fallback.
    """
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_TO_ALLOW_SINGLETON_TO, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_TO_ALLOW_SINGLETON_TO


def resolve_to_min_clusters(value: Optional[int] = None) -> int:
    """Resolve the dissolve tiny-course floor: explicit arg → env → default.

    Garbage / non-positive env values fall back to the default (a misconfigured
    floor must never crash or collapse a course to one TO). Mirrors
    :func:`resolve_to_outlier_min_size`'s parse-with-fallback posture.
    """
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get(ENV_TO_MIN_CLUSTERS)
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_TO_MIN_CLUSTERS
        if val >= 1:
            return val
    return _DEFAULT_TO_MIN_CLUSTERS


def _cluster_centroid(
    vecs: List[List[float]], member_idxs: List[int]
) -> List[float]:
    """Mean of a cluster's member vectors. Empty cluster → ``[]``.

    NumPy-free (pure-Python) so it shares the embedding-optional posture of the
    rest of the module. Assumes all member vectors share a dimension (they come
    from one ``encode_batch`` call); a zero-length member list returns ``[]``.
    """
    if not member_idxs:
        return []
    dim = len(vecs[member_idxs[0]])
    if dim == 0:
        return []
    acc = [0.0] * dim
    for idx in member_idxs:
        v = vecs[idx]
        for d in range(dim):
            acc[d] += v[d]
    n = float(len(member_idxs))
    return [x / n for x in acc]


def _mean_intra_cluster_cosine(
    vecs: List[List[float]], member_idxs: List[int]
) -> float:
    """Mean pairwise cosine among a cluster's members (cohesion).

    Singleton / empty cluster → ``1.0`` (perfectly cohesive by convention — a
    single point has no internal disagreement). Local private re-implementation
    of the equivalent helper in ``pipeline_tools.py`` (which must NOT be imported
    cross-module); NumPy-free, O(m²) over the cluster's small membership.
    """
    m = len(member_idxs)
    if m <= 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            total += cosine_similarity(vecs[member_idxs[i]], vecs[member_idxs[j]])
            pairs += 1
    return total / pairs if pairs else 1.0


def _reorder_clusters(clusters: List[List[int]]) -> List[List[int]]:
    """Re-impose the deterministic ordering contract on a partition.

    Each (non-empty) member list ascending; clusters ordered by minimum member
    index — MATCHING :func:`cluster_to_target_k` / :func:`cluster_by_cosine`.
    Empty clusters are dropped (a merge can empty a source cluster).
    """
    cleaned = [sorted(c) for c in clusters if c]
    cleaned.sort(key=lambda g: min(g))
    return cleaned


def _nearest_centroid_neighbor(
    target_members: List[int],
    clusters: List[List[int]],
    vecs: List[List[float]],
    *,
    skip_index: int,
) -> Tuple[int, float]:
    """Return ``(index_in_clusters, centroid_cosine)`` of ``target``'s nearest
    OTHER cluster by centroid cosine, or ``(-1, -1.0)`` when none exists.

    ``skip_index`` is the position of the target cluster itself (never its own
    neighbor). Deterministic: ties on cosine break to the LOWER cluster index.
    """
    target_centroid = _cluster_centroid(vecs, target_members)
    best_idx = -1
    best_cos = -1.0
    for ci, members in enumerate(clusters):
        if ci == skip_index or not members:
            continue
        cos = cosine_similarity(target_centroid, _cluster_centroid(vecs, members))
        if cos > best_cos:
            best_cos = cos
            best_idx = ci
    return best_idx, best_cos


def consolidate_small_clusters(
    clusters: List[List[int]],
    vecs: List[List[float]],
    *,
    min_size: Optional[int] = None,
    absorb_floor: Optional[float] = None,
) -> Tuple[List[List[int]], int, int]:
    """P1 — fold every sub-``min_size`` cluster into its nearest-centroid neighbor.

    Subsumes BOTH guards from the P1 plan: the OUTLIER guard (a semantic-outlier
    CO in its own singleton that actually belongs to a sibling) and the
    MIN-CLUSTER-SIZE merge (1-2-CO runts). A sub-``min_size`` cluster is absorbed
    into whichever surviving cluster its centroid is closest to, SMALLEST-FIRST,
    PROVIDED that neighbor's centroid cosine is >= ``absorb_floor`` (it HAS a
    clear home). A tiny cluster with NO neighbor above the very-low floor REMAINS
    standing (a genuinely distinct competency). Iterates until all clusters are
    >= ``min_size`` OR no further absorption is possible (no eligible neighbor /
    one cluster left).

    NOTE: the "REMAINS standing" behavior is SUPERSEDED for SIZE-1 clusters by
    the unconditional :func:`dissolve_singletons` backstop (a lone CO is never a
    valid terminal objective, so it always folds into its nearest neighbor). The
    ``absorb_floor`` here still governs MULTI-member runts (2..min_size-1).

    Returns ``(new_clusters, outliers_absorbed, undersize_merged)`` where
    ``outliers_absorbed`` counts absorbed SINGLETONS (the outlier-guard arm) and
    ``undersize_merged`` counts absorbed multi-member runts (2..min_size-1). Both
    arms run in the same pass; the split counts let a caller attribute each.

    PARTITION INVARIANT: the flattened output indices are a permutation of the
    flattened input indices — never drops, adds, or invents a CO.
    """
    resolved_min = resolve_to_outlier_min_size(min_size)
    resolved_floor = resolve_to_outlier_absorb_floor(absorb_floor)

    # Work on a mutable copy of the partition.
    work: List[List[int]] = [list(c) for c in clusters if c]
    outliers_absorbed = 0
    undersize_merged = 0

    while True:
        if len(work) <= 1:
            break
        # Pick the smallest sub-min-size cluster (smallest-first; ties → lowest
        # min-index for determinism) that HAS an eligible home.
        candidates = [
            i for i, c in enumerate(work) if 0 < len(c) < resolved_min
        ]
        if not candidates:
            break
        candidates.sort(key=lambda i: (len(work[i]), min(work[i])))

        absorbed_any = False
        for ci in candidates:
            neighbor_idx, neighbor_cos = _nearest_centroid_neighbor(
                work[ci], work, vecs, skip_index=ci
            )
            if neighbor_idx < 0 or neighbor_cos < resolved_floor:
                continue  # no clear home → leave this one standing (for now)
            size_before = len(work[ci])
            work[neighbor_idx].extend(work[ci])
            work[ci] = []
            if size_before <= 1:
                outliers_absorbed += 1
            else:
                undersize_merged += 1
            # Drop the emptied cluster and restart the scan (indices shift).
            work = [c for c in work if c]
            absorbed_any = True
            break
        if not absorbed_any:
            break

    return _reorder_clusters(work), outliers_absorbed, undersize_merged


def dissolve_singletons(
    clusters: List[List[int]],
    vecs: List[List[float]],
    *,
    min_clusters: Optional[int] = None,
) -> Tuple[List[List[int]], int]:
    """Fold every remaining size-1 cluster into its NEAREST-centroid neighbor.

    The anti-hallucinated-TO backstop. A terminal objective is definitionally
    an AGGREGATE of >=2 chapter objectives, so a size-1 cluster must never
    author a standalone TO — a lone semantic-outlier CO would otherwise be
    promoted into a wholesale course-wide terminal competency. Unlike
    :func:`consolidate_small_clusters` (which DELIBERATELY leaves a
    genuinely-distinct singleton standing when no neighbor clears the absorb
    floor), this runs with NO floor: a lone CO ALWAYS folds UNDER the concept it
    exemplifies — its nearest embedding neighbor. Folds SMALLEST-min-index-first,
    one at a time, deterministically (reusing :func:`_nearest_centroid_neighbor`
    with ties broken to the lower cluster index).

    The ONLY thing that halts a fold is the ``min_clusters`` tiny-course floor:
    folding stops before the surviving cluster count would drop BELOW it, so a
    genuinely tiny course (few themes, mostly singletons) is not collapsed into
    one mega-TO.

    Returns ``(new_clusters, n_dissolved)``. PARTITION INVARIANT: the flattened
    output indices are a permutation of the flattened input indices — never
    drops, adds, or invents a CO. (Essentially
    :func:`consolidate_small_clusters` with ``min_size=2`` + ``absorb_floor=0.0``
    PLUS the new ``min_clusters`` tiny-course protection.)
    """
    resolved_min = resolve_to_min_clusters(min_clusters)

    work: List[List[int]] = [list(c) for c in clusters if c]
    n_dissolved = 0

    while True:
        # Halt before a fold would drop the surviving count below the floor
        # (and never fold with only one cluster left).
        if len(work) <= 1 or len(work) <= resolved_min:
            break
        singletons = [i for i, c in enumerate(work) if len(c) == 1]
        if not singletons:
            break
        # Smallest-first is trivially satisfied (all size 1); order by min member
        # index for deterministic selection.
        singletons.sort(key=lambda i: min(work[i]))
        ci = singletons[0]
        neighbor_idx, _neighbor_cos = _nearest_centroid_neighbor(
            work[ci], work, vecs, skip_index=ci
        )
        if neighbor_idx < 0:
            break  # no neighbor (shouldn't happen with >1 cluster) → give up
        work[neighbor_idx].extend(work[ci])
        work[ci] = []
        # Drop the emptied cluster and restart the scan (indices shift).
        work = [c for c in work if c]
        n_dissolved += 1

    return _reorder_clusters(work), n_dissolved


def absorb_outlier_clusters(
    clusters: List[List[int]],
    vecs: List[List[float]],
    *,
    min_size: Optional[int] = None,
    absorb_floor: Optional[float] = None,
) -> Tuple[List[List[int]], int]:
    """P1 OUTLIER GUARD — absorb sub-``min_size`` clusters into their nearest home.

    Thin wrapper over :func:`consolidate_small_clusters` that reports only the
    total absorbed count (singletons + runts). A singleton outlier whose best
    cross-cluster centroid cosine is >= ``absorb_floor`` is folded into that
    neighbor; an outlier with NO neighbor above the floor remains (a genuinely
    distinct competency). Returns ``(new_clusters, n_absorbed)``. PARTITION
    INVARIANT preserved (see :func:`consolidate_small_clusters`).

    NOTE: for SIZE-1 clusters this floor-gated "remains standing" behavior is
    SUPERSEDED by the unconditional :func:`dissolve_singletons` backstop (a lone
    CO is never a valid terminal objective); the floor governs multi-member runts.
    """
    new_clusters, outliers, runts = consolidate_small_clusters(
        clusters, vecs, min_size=min_size, absorb_floor=absorb_floor
    )
    return new_clusters, outliers + runts


def merge_undersize_clusters(
    clusters: List[List[int]],
    vecs: List[List[float]],
    *,
    min_size: Optional[int] = None,
) -> Tuple[List[List[int]], int]:
    """P1 MIN-CLUSTER-SIZE MERGE — fold every sub-``min_size`` runt into a sibling.

    Smallest-first, until all clusters are >= ``min_size`` OR only one cluster
    remains. Implemented over :func:`consolidate_small_clusters` with the
    absorb-floor disabled (``0.0``) so EVERY undersize cluster finds a home (the
    pure runt-merge semantics: there is always a nearest neighbor when >1 cluster
    remains). Returns ``(new_clusters, n_merged)``. PARTITION INVARIANT preserved.
    """
    new_clusters, outliers, runts = consolidate_small_clusters(
        clusters, vecs, min_size=min_size, absorb_floor=0.0
    )
    return new_clusters, outliers + runts


def merge_near_duplicate_clusters(
    clusters: List[List[int]],
    vecs: List[List[float]],
    *,
    merge_floor: Optional[float] = None,
) -> Tuple[List[List[int]], int]:
    """P1 NEAR-DUPLICATE-TO MERGE — union clusters with near-identical centroids.

    Ward at a hard K over-splits a cosine-dense theme into 2-3 near-duplicate
    TOs. Compute each cluster's centroid; any two clusters whose centroid cosine
    >= ``merge_floor`` are UNIONED (transitively, via union-find over the
    centroid-cosine graph — exactly like :func:`cluster_by_cosine` does over the
    point graph). Returns ``(new_clusters, n_merged_pairs)`` where
    ``n_merged_pairs`` counts the centroid pairs at/above the floor that drove a
    union (the deterministic "merge edges" count). PARTITION INVARIANT preserved.
    """
    resolved_floor = resolve_to_merge_cosine(merge_floor)
    work = [list(c) for c in clusters if c]
    nc = len(work)
    if nc <= 1:
        return _reorder_clusters(work), 0

    centroids = [_cluster_centroid(vecs, c) for c in work]

    parent = list(range(nc))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            # Lower index as root → cluster order follows course order.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    merged_pairs = 0
    for i in range(nc):
        for j in range(i + 1, nc):
            cos = cosine_similarity(centroids[i], centroids[j])
            if cos >= resolved_floor:
                merged_pairs += 1
                _union(i, j)

    # Collect merged clusters keyed by root, unioning their member indices.
    by_root: Dict[int, List[int]] = {}
    for ci in range(nc):
        root = _find(ci)
        by_root.setdefault(root, []).extend(work[ci])

    return _reorder_clusters(list(by_root.values())), merged_pairs


def apply_cluster_guards(
    clusters: List[List[int]],
    vecs: List[List[float]],
    *,
    min_size: Optional[int] = None,
    merge_floor: Optional[float] = None,
    absorb_floor: Optional[float] = None,
    enable_consolidate: Optional[bool] = None,
    enable_merge: Optional[bool] = None,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """P1 ORCHESTRATOR — apply the post-cluster guards (when enabled).

    Runs (when each gate is on): consolidate-small (outlier-absorb +
    undersize-merge) → near-dup-merge, then re-imposes the deterministic
    min-index ordering contract :func:`cluster_to_target_k` returns. Returns
    ``(new_clusters, signals)`` where ``signals`` always carries at least:
    ``outliers_absorbed``, ``undersize_merged``, ``near_dup_clusters_merged``,
    ``clusters_before``, ``clusters_after``.

    When ALL gates are OFF (no kwargs, no env) this returns the input clusters
    UNCHANGED (re-ordered to the contract) and ALL-ZERO signals — the byte-stable
    default. PARTITION INVARIANT preserved in every branch.
    """
    do_consolidate = resolve_to_cluster_guards(enable_consolidate)
    do_merge = resolve_to_merge_near_dup(enable_merge)

    clusters_before = len([c for c in clusters if c])
    work = [list(c) for c in clusters if c]
    outliers_absorbed = 0
    undersize_merged = 0
    near_dup_merged = 0

    if do_consolidate:
        work, outliers_absorbed, undersize_merged = consolidate_small_clusters(
            work, vecs, min_size=min_size, absorb_floor=absorb_floor
        )

    if do_merge:
        work, near_dup_merged = merge_near_duplicate_clusters(
            work, vecs, merge_floor=merge_floor
        )

    ordered = _reorder_clusters(work)
    signals = {
        "outliers_absorbed": outliers_absorbed,
        "undersize_merged": undersize_merged,
        "near_dup_clusters_merged": near_dup_merged,
        "clusters_before": clusters_before,
        "clusters_after": len(ordered),
    }
    return ordered, signals


def _require_strict() -> bool:
    return str(os.environ.get(ENV_REQUIRE_EMBEDDINGS, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _statement(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("statement") or candidate.get("text") or "").strip()


def _source_chunk_ids(candidate: Dict[str, Any]) -> List[str]:
    raw = candidate.get("source_chunk_ids")
    if isinstance(raw, list):
        return [str(c) for c in raw if c]
    return []


def _entailment_score(candidate: Dict[str, Any]) -> float:
    val = candidate.get("entailment_score")
    try:
        return float(val) if val is not None else -1.0
    except (TypeError, ValueError):
        return -1.0


def _passthrough(grounded: List[Dict[str, Any]]) -> DedupResult:
    return DedupResult(
        canonical=[dict(c) for c in grounded],
        clusters=[[i] for i in range(len(grounded))],
        max_pairwise_cosine=0.0,
        near_dup_pairs=0,
        available=False,
    )


def _chunk_text(rec: Any) -> str:
    if not isinstance(rec, dict):
        return ""
    return str(rec.get("text") or rec.get("body") or "").strip()


def _prune_union_chunks(
    *,
    rep_statement_vec: List[float],
    union_ids: List[str],
    chunks_by_id: Dict[str, Any],
    client: Any,
    cap: int,
    floor: float,
) -> Tuple[List[str], int]:
    """Fix 1A — rank a merged objective's unioned chunks, keep the top-K relevant.

    Rank ``union_ids`` by cosine(``rep_statement_vec``, chunk_text_vec)
    descending, drop any below ``floor``, then keep at most ``cap``. ALWAYS keep
    ≥1 (the single strongest) even if it's below the floor — a merged CO must
    never lose all provenance. The returned set is a SUBSET of ``union_ids``
    (anti-fabrication). Returns ``(kept_ids, n_dropped)``.

    Graceful degrade: a chunk with no resolvable text is treated as relevance
    ``-1.0`` (ranked last) but still eligible as the keep-≥1 fallback only if
    nothing else resolves. If the embed of chunk texts fails entirely, returns
    the full union unchanged (no crash, no drop).
    """
    # Resolve chunk texts in union order; remember which ids had real text.
    texts: List[str] = []
    has_text: List[bool] = []
    for cid in union_ids:
        t = _chunk_text(chunks_by_id.get(cid))
        texts.append(t if t else " ")  # encode a non-empty placeholder
        has_text.append(bool(t))

    try:
        chunk_vecs = client.encode_batch(texts)
    except EmbeddingBackendUnavailable:
        return union_ids, 0
    except Exception:  # noqa: BLE001 — any embed failure degrades to full union
        logger.warning(
            "objective dedup: embedding union chunk texts failed; keeping the "
            "full chunk union for this objective (no Fix-1A prune)."
        )
        return union_ids, 0

    scored: List[Tuple[str, float]] = []
    for cid, vec, real in zip(union_ids, chunk_vecs, has_text):
        cos = cosine_similarity(rep_statement_vec, vec) if real else -1.0
        scored.append((cid, float(cos)))

    # Stable sort by relevance descending (ties keep union/course order).
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
    ranked = [scored[i] for i in order]

    # Apply the floor, then the cap. ALWAYS keep ≥1 (the strongest), even if it
    # is below the floor (keep-≥1 supersedes the floor).
    above_floor = [cid for cid, cos in ranked if cos >= floor]
    if not above_floor:
        kept = [ranked[0][0]]  # strongest single survivor
    else:
        kept = above_floor[:cap]

    # Re-project ``kept`` onto the original union order for stable on-disk shape.
    kept_set = set(kept)
    kept_ordered = [cid for cid in union_ids if cid in kept_set]
    n_dropped = len(union_ids) - len(kept_ordered)
    return kept_ordered, n_dropped


def _emit_prune_capture(
    capture: Any,
    *,
    rep_id: str,
    statement: str,
    union_ids: List[str],
    kept_ids: List[str],
    cap: int,
    floor: float,
) -> None:
    """Best-effort ``objective_chunk_prune`` decision-capture (never raises)."""
    try:
        kept_set = set(kept_ids)
        dropped = [cid for cid in union_ids if cid not in kept_set]
        capture.log_decision(
            decision_type="objective_chunk_prune",
            decision=(
                f"pruned {len(dropped)} off-topic chunk(s) from merged "
                f"objective {rep_id or '(unminted)'} "
                f"({len(union_ids)} -> {len(kept_ids)})"
            ),
            rationale=(
                f"Fix 1A: merged objective '{statement[:60]}' unioned "
                f"{len(union_ids)} cited chunks; ranked by cosine to the rep "
                f"statement and kept top-{cap} above floor {floor:.2f} "
                f"(kept={kept_ids}, dropped={dropped}) to avoid diffuse / "
                f"misleading provenance (no source beats a misleading one)."
            ),
            alternatives_considered=[
                "keep full union (legacy; produced 25-chunk grab-bags)",
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug("objective_chunk_prune capture failed (%s); continuing", exc)


def _emit_split_capture(
    capture: Any,
    *,
    clusters_split: int,
    distinct_skill_count: int,
    pre_split_clusters: int,
    threshold: float,
) -> None:
    """Best-effort ``objective_distinct_skill_split`` capture (never raises)."""
    try:
        capture.log_decision(
            decision_type="objective_distinct_skill_split",
            decision=(
                f"split {clusters_split} cluster(s) into {distinct_skill_count} "
                f"distinct-skill group(s)"
            ),
            rationale=(
                f"I3 PRONG A: single-link cosine clustering at threshold "
                f"{threshold:.2f} transitively chained distinct named skills "
                f"into shared clusters ({pre_split_clusters} near-dup pair(s)); "
                f"re-partitioned {clusters_split} multi-skill cluster(s) by "
                f"verb-object/keyphrase + concept-tag signature so each distinct "
                f"named skill keeps its OWN representative statement + grounding "
                f"(total {distinct_skill_count} distinct skills). "
                f"Anti-fabrication: re-partition of already-synthesized "
                f"candidate statements only — no statement invented."
            ),
            alternatives_considered=[
                "keep single-link clusters (discards chained distinct skills' "
                "statements; their chunks demote to supporting evidence)",
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "objective_distinct_skill_split capture failed (%s); continuing", exc
        )


def dedup_candidates(
    grounded: List[Dict[str, Any]],
    *,
    embed: Optional[Any] = None,
    threshold: Optional[float] = None,
    allow_fake: bool = False,
    chunks_by_id: Optional[Dict[str, Any]] = None,
    max_chunks_per_objective: Optional[int] = None,
    chunk_relevance_floor: Optional[float] = None,
    distinct_skill_split: Optional[bool] = None,
    dedup_lexical: Optional[bool] = None,
    capture: Optional[Any] = None,
) -> DedupResult:
    """Collapse near-duplicate ``grounded`` candidates into a canonical CO set.

    Args:
        grounded: Pass-C survivors (each has ``entailment_score`` +
            ``source_chunk_ids``).
        embed: injection seam; default builds an :class:`EmbeddingClient` via
            ``build_embedding_client()``. A ``provider="fake"`` client is refused
            unless ``allow_fake``.
        threshold: cosine clustering threshold; resolved via
            :func:`resolve_dedup_threshold` when None.
        allow_fake: permit a fake-provider client (CI / tests only; mirrors the
            anti-poisoning gate).
        chunks_by_id: ``chunk_id -> chunk record`` (must carry ``text``/``body``).
            Fix 1A — required to rank a merged objective's unioned chunks by
            relevance to the rep statement and PRUNE off-topic ones. ``None`` /
            missing text → graceful degrade to the legacy full-union (no prune,
            logged once).
        max_chunks_per_objective: Fix 1A cap (top-K by cosine); resolved via
            :func:`resolve_max_chunks_per_objective` when None.
        chunk_relevance_floor: Fix 1A relevance floor; resolved via
            :func:`resolve_chunk_relevance_floor` when None.
        capture: optional :class:`DecisionCapture` for the per-objective
            ``objective_chunk_prune`` event (best-effort; never raises).

    Returns a :class:`DedupResult`. Embeddings absent + not strict → pass-through
    (every candidate its own cluster, ``available=False``).
    """
    if not grounded:
        return DedupResult(
            canonical=[], clusters=[], max_pairwise_cosine=0.0,
            near_dup_pairs=0, available=True,
        )

    resolved_threshold = resolve_dedup_threshold(threshold)
    resolved_cap = resolve_max_chunks_per_objective(max_chunks_per_objective)
    resolved_floor = resolve_chunk_relevance_floor(chunk_relevance_floor)

    # Resolve the embedding client (injection seam first).
    client = embed
    if client is None:
        try:
            client = build_embedding_client(offline=True)
        except EmbeddingBackendUnavailable:
            if _require_strict():
                raise
            return _passthrough(grounded)
        except Exception:  # noqa: BLE001 — any backend resolution failure degrades
            if _require_strict():
                raise
            return _passthrough(grounded)

    # Anti-poisoning gate: refuse a fake-provider client unless allowed.
    resolved = getattr(client, "resolved", None)
    kind = getattr(resolved, "kind", None)
    if kind == "fake" and not allow_fake:
        if _require_strict():
            raise EmbeddingBackendUnavailable(
                "objective dedup refused a fake-provider embedding client "
                "(set ED4ALL_EMBEDDING_ALLOW_FAKE or pass allow_fake=True)."
            )
        return _passthrough(grounded)

    texts = [_statement(c) for c in grounded]
    try:
        vecs = client.encode_batch(texts)
    except EmbeddingBackendUnavailable:
        if _require_strict():
            raise
        return _passthrough(grounded)
    except Exception:  # noqa: BLE001
        if _require_strict():
            raise
        return _passthrough(grounded)

    n = len(grounded)
    # Single-link clustering via the shared graph-clustering core (WS1 §1
    # extraction; behavior-preserving — the dedup pass keeps the 0.88 dedup
    # threshold, NOT the lower WS1 TO-cluster threshold).
    ordered_clusters, max_cos, near_dup_pairs = cluster_by_cosine(
        vecs, resolved_threshold
    )

    # Defect E — CROSS-WINDOW lexical-dedup SECOND PASS (opt-in). Runs BEFORE the
    # PRONG-A split so a wrongly-chained cosine cluster can still split. Merges
    # two cosine clusters iff centroid cosine >= floor AND best-grounded
    # skill-signature Jaccard >= floor AND same Bloom band, via COMPLETE linkage
    # (every cross pair must qualify → no re-chained mega-cluster). Default OFF →
    # byte-identical clusters. The per-cluster merged-count map is keyed by the
    # merged cluster's min member index (stable under the PRONG-A split for
    # high-Jaccard merged clusters — which the <0.34-distinctness split never
    # touches — so the stamp lands on the right rep).
    lexical_merge_pairs = 0
    lexical_clusters_merged = 0
    lexical_count_by_minidx: Dict[int, int] = {}
    if resolve_dedup_lexical(dedup_lexical) and ordered_clusters:
        clusters_before_lex = len(ordered_clusters)
        (
            ordered_clusters,
            lexical_merge_pairs,
            lexical_clusters_merged,
            _merged_counts,
        ) = merge_clusters_lexical(ordered_clusters, grounded, vecs)
        for cl, cnt in zip(ordered_clusters, _merged_counts):
            if cnt > 1 and cl:
                lexical_count_by_minidx[min(cl)] = cnt
        if lexical_merge_pairs and capture is not None:
            _emit_lexical_merge_capture(
                capture,
                merge_ops=lexical_merge_pairs,
                clusters_absorbed=lexical_clusters_merged,
                clusters_before=clusters_before_lex,
                cosine_floor=resolve_dedup_lexical_cosine(),
                jaccard_floor=resolve_dedup_lexical_jaccard(),
            )

    # I3 PRONG A — distinct-skill SPLIT (opt-in). Single-link transitively
    # chains distinct-but-related skills into one cluster, so the best-grounded
    # rep keeps ONE statement and the rest's NAMED skills (e.g. "order of
    # operations") are discarded (their chunks demote to supporting evidence).
    # Re-partition each cluster by skill signature so every distinct named skill
    # keeps its OWN representative + grounding. The 0.88 collapse for
    # same-signature restatements is preserved; ANTI-FABRICATION — the split
    # only re-partitions existing candidate indices, never invents a statement.
    split_enabled = resolve_distinct_skill_split(distinct_skill_split)
    clusters_split = 0
    distinct_skill_count = len(ordered_clusters)
    if split_enabled and ordered_clusters:
        ordered_clusters, clusters_split, distinct_skill_count = (
            split_clusters_for_distinct_skills(ordered_clusters, grounded)
        )
        if clusters_split and capture is not None:
            _emit_split_capture(
                capture,
                clusters_split=clusters_split,
                distinct_skill_count=distinct_skill_count,
                pre_split_clusters=near_dup_pairs,
                threshold=resolved_threshold,
            )

    # Fix 1A — decide ONCE whether the relevance-prune can run. It needs chunk
    # TEXT to embed; absent ``chunks_by_id`` (legacy callers) → graceful degrade
    # to the legacy full-union (logged once, no crash). Mirrors the
    # embedding-optional posture of the statistical-tier validators.
    prune_enabled = bool(chunks_by_id)
    if not prune_enabled and any(
        len(_source_chunk_ids(grounded[i])) > 0 for i in range(n)
    ):
        logger.warning(
            "objective dedup: chunks_by_id not provided; skipping the Fix-1A "
            "relevance-prune (merged objectives keep the full chunk union)."
        )

    canonical: List[Dict[str, Any]] = []
    pruned_chunk_total = 0
    for cluster in ordered_clusters:
        # Best-grounded representative: highest entailment_score; tie → longer
        # statement; tie → first by course order (min index).
        def _rep_key(idx: int) -> Any:
            return (
                _entailment_score(grounded[idx]),
                len(_statement(grounded[idx])),
                -idx,  # earlier index wins on a full tie
            )

        rep_idx = max(cluster, key=_rep_key)
        rep = dict(grounded[rep_idx])
        # Union every member's source_chunk_ids onto the representative.
        union_ids: List[str] = []
        for member_idx in cluster:
            for cid in _source_chunk_ids(grounded[member_idx]):
                if cid not in union_ids:
                    union_ids.append(cid)

        # Fix 1A — cap + relevance-prune the union. Rank by cosine(rep_statement,
        # chunk_text) descending, keep top-K, drop below the floor. ALWAYS keep
        # ≥1 (the single strongest) so a merged CO never loses all provenance.
        # ANTI-FABRICATION: the kept set is a SUBSET of ``union_ids`` — we only
        # ever drop, never add a chunk no cluster member cited.
        kept_ids = union_ids
        if prune_enabled and len(union_ids) > 1:
            kept_ids, n_dropped = _prune_union_chunks(
                rep_statement_vec=vecs[rep_idx],
                union_ids=union_ids,
                chunks_by_id=chunks_by_id or {},
                client=client,
                cap=resolved_cap,
                floor=resolved_floor,
            )
            # PIN the representative's NLI-entailing chunk (Pass-C
            # ``entailing_chunk_id``): the rep's ``grounded=True`` +
            # ``entailment_score`` were EARNED by that chunk, but the prune
            # above ranks by EMBEDDING cosine — a different, weaker metric —
            # so the one entailing chunk can lose the relevance race to
            # topically-similar non-entailing neighbors (TOC lists, heading
            # chunks). Shipping the CO without it fails the downstream
            # objective_entailment gate (measured 2026-07-04 on a real run: 10 COs).
            # The cap widens by <= 1, mirroring citation_reselect's
            # protected-original widening. ANTI-FABRICATION: the pin is
            # already ⊆ union_ids (the rep cited it), so we only re-keep,
            # never add a chunk no cluster member cited.
            _pin = str(rep.get("entailing_chunk_id") or "").strip()
            if _pin and _pin in union_ids and _pin not in kept_ids:
                kept_ids = list(kept_ids) + [_pin]
                n_dropped = max(0, n_dropped - 1)
            pruned_chunk_total += n_dropped
            if n_dropped and capture is not None:
                _emit_prune_capture(
                    capture,
                    rep_id=str(rep.get("id") or rep.get("co_id") or ""),
                    statement=_statement(rep),
                    union_ids=union_ids,
                    kept_ids=kept_ids,
                    cap=resolved_cap,
                    floor=resolved_floor,
                )

        if kept_ids:
            rep["source_chunk_ids"] = list(kept_ids)
            rep["source_refs"] = [
                {
                    "ref": str(rep.get("chapter_id") or ""),
                    "chunk_ids": list(kept_ids),
                }
            ]
        if len(cluster) > 1:
            rep["dedup_merged_count"] = len(cluster)
        # Defect E — stamp the number of INPUT cosine-clusters the lexical pass
        # folded into this rep's cluster (>1 only when a lexical merge landed).
        if cluster:
            _lex_cnt = lexical_count_by_minidx.get(min(cluster), 1)
            if _lex_cnt > 1:
                rep["dedup_lexical_merged_count"] = _lex_cnt
        canonical.append(rep)

    return DedupResult(
        canonical=canonical,
        clusters=ordered_clusters,
        max_pairwise_cosine=max_cos,
        near_dup_pairs=near_dup_pairs,
        available=True,
        pruned_chunk_total=pruned_chunk_total,
        max_chunks_per_objective=resolved_cap,
        clusters_split_for_distinct_skill=clusters_split,
        distinct_skill_count=distinct_skill_count,
        lexical_merge_pairs=lexical_merge_pairs,
        lexical_clusters_merged=lexical_clusters_merged,
    )


__all__ = [
    "DedupResult",
    "dedup_candidates",
    "cluster_by_cosine",
    "cluster_to_target_k",
    "split_clusters_for_distinct_skills",
    "merge_clusters_lexical",
    "consolidate_small_clusters",
    "absorb_outlier_clusters",
    "merge_undersize_clusters",
    "merge_near_duplicate_clusters",
    "apply_cluster_guards",
    "resolve_distinct_skill_split",
    "resolve_dedup_lexical",
    "resolve_dedup_lexical_cosine",
    "resolve_dedup_lexical_jaccard",
    "resolve_to_cluster_guards",
    "resolve_to_outlier_min_size",
    "resolve_to_outlier_absorb_floor",
    "resolve_to_merge_near_dup",
    "resolve_to_merge_cosine",
    "resolve_dedup_threshold",
    "resolve_to_cluster_threshold",
    "resolve_to_cluster_k",
    "resolve_to_cos_per_cluster",
    "resolve_max_chunks_per_objective",
    "resolve_chunk_relevance_floor",
    "_DEFAULT_DEDUP_THRESHOLD",
    "_DEFAULT_TO_CLUSTER_THRESHOLD",
    "_DEFAULT_TO_CLUSTER_K",
    "_DEFAULT_TO_COS_PER_CLUSTER",
    "_DEFAULT_MAX_CHUNKS_PER_OBJECTIVE",
    "_DEFAULT_CHUNK_RELEVANCE_FLOOR",
    "_DEFAULT_DISTINCT_SKILL_SPLIT",
    "_DEFAULT_DEDUP_LEXICAL",
    "_DEFAULT_DEDUP_LEXICAL_COSINE",
    "_DEFAULT_DEDUP_LEXICAL_JACCARD",
    "_DEFAULT_TO_CLUSTER_GUARDS",
    "_DEFAULT_TO_OUTLIER_MIN_SIZE",
    "_DEFAULT_TO_OUTLIER_ABSORB_FLOOR",
    "_DEFAULT_TO_MERGE_NEAR_DUP",
    "_DEFAULT_TO_MERGE_COSINE",
    "ENV_DISTINCT_SKILL_SPLIT",
    "ENV_DEDUP_LEXICAL",
    "ENV_DEDUP_LEXICAL_COSINE",
    "ENV_DEDUP_LEXICAL_JACCARD",
    "ENV_TO_CLUSTER_GUARDS",
    "ENV_TO_OUTLIER_MIN_SIZE",
    "ENV_TO_OUTLIER_ABSORB_FLOOR",
    "ENV_TO_MERGE_NEAR_DUP",
    "ENV_TO_MERGE_COSINE",
    "ENV_DEDUP_THRESHOLD",
    "ENV_TO_CLUSTER_THRESHOLD",
    "ENV_TO_CLUSTER_K",
    "ENV_TO_COS_PER_CLUSTER",
    "ENV_MAX_CHUNKS_PER_OBJECTIVE",
    "ENV_CHUNK_RELEVANCE_FLOOR",
]
