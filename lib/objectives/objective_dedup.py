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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical": [dict(c) for c in self.canonical],
            "clusters": [list(g) for g in self.clusters],
            "max_pairwise_cosine": round(float(self.max_pairwise_cosine), 6),
            "near_dup_pairs": self.near_dup_pairs,
            "available": self.available,
            "pruned_chunk_total": self.pruned_chunk_total,
            "max_chunks_per_objective": self.max_chunks_per_objective,
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


def dedup_candidates(
    grounded: List[Dict[str, Any]],
    *,
    embed: Optional[Any] = None,
    threshold: Optional[float] = None,
    allow_fake: bool = False,
    chunks_by_id: Optional[Dict[str, Any]] = None,
    max_chunks_per_objective: Optional[int] = None,
    chunk_relevance_floor: Optional[float] = None,
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
        canonical.append(rep)

    return DedupResult(
        canonical=canonical,
        clusters=ordered_clusters,
        max_pairwise_cosine=max_cos,
        near_dup_pairs=near_dup_pairs,
        available=True,
        pruned_chunk_total=pruned_chunk_total,
        max_chunks_per_objective=resolved_cap,
    )


__all__ = [
    "DedupResult",
    "dedup_candidates",
    "cluster_by_cosine",
    "cluster_to_target_k",
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
    "ENV_DEDUP_THRESHOLD",
    "ENV_TO_CLUSTER_THRESHOLD",
    "ENV_TO_CLUSTER_K",
    "ENV_TO_COS_PER_CLUSTER",
    "ENV_MAX_CHUNKS_PER_OBJECTIVE",
    "ENV_CHUNK_RELEVANCE_FLOOR",
]
