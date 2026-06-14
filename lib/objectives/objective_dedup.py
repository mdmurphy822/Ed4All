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

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lib.embedding._math import cosine_similarity
from lib.embedding.providers import (
    EmbeddingBackendUnavailable,
    build_embedding_client,
)

#: Cosine ≥ this clusters two candidate objectives. ADVISORY starting point —
#: Risk R6 mandates measuring ``max_pairwise_cosine`` on a real run before
#: pinning; env-overridable via ``ED4ALL_OBJECTIVE_DEDUP_THRESHOLD``.
_DEFAULT_DEDUP_THRESHOLD = 0.88

#: Env override for the dedup threshold (cross-cutting ``ED4ALL_*``, root-owned).
ENV_DEDUP_THRESHOLD = "ED4ALL_OBJECTIVE_DEDUP_THRESHOLD"

#: Strict-mode flag bridge (mirrors the NLI/embedding strict contract). When
#: truthy, an absent embedding backend RAISES rather than passing through.
ENV_REQUIRE_EMBEDDINGS = "TRAINFORGE_REQUIRE_EMBEDDINGS"


@dataclass(frozen=True)
class DedupResult:
    """Outcome of the Pass-D dedup over a grounded candidate pool."""

    canonical: List[Dict[str, Any]] = field(default_factory=list)
    clusters: List[List[int]] = field(default_factory=list)
    max_pairwise_cosine: float = 0.0
    near_dup_pairs: int = 0
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical": [dict(c) for c in self.canonical],
            "clusters": [list(g) for g in self.clusters],
            "max_pairwise_cosine": round(float(self.max_pairwise_cosine), 6),
            "near_dup_pairs": self.near_dup_pairs,
            "available": self.available,
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


def dedup_candidates(
    grounded: List[Dict[str, Any]],
    *,
    embed: Optional[Any] = None,
    threshold: Optional[float] = None,
    allow_fake: bool = False,
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

    Returns a :class:`DedupResult`. Embeddings absent + not strict → pass-through
    (every candidate its own cluster, ``available=False``).
    """
    if not grounded:
        return DedupResult(
            canonical=[], clusters=[], max_pairwise_cosine=0.0,
            near_dup_pairs=0, available=True,
        )

    resolved_threshold = resolve_dedup_threshold(threshold)

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
    # Single-link clustering via union-find over pairs cosine ≥ threshold.
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
            # order (the rep is chosen below by grounding, ties → course order).
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
            if cos >= resolved_threshold:
                near_dup_pairs += 1
                _union(i, j)

    # Collect clusters keyed by root, preserving member (course) order.
    clusters_by_root: Dict[int, List[int]] = {}
    for i in range(n):
        root = _find(i)
        clusters_by_root.setdefault(root, []).append(i)
    # Order clusters by their minimum member index (course order of the rep).
    ordered_clusters = sorted(
        clusters_by_root.values(), key=lambda g: min(g)
    )

    canonical: List[Dict[str, Any]] = []
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
        if union_ids:
            rep["source_chunk_ids"] = union_ids
            rep["source_refs"] = [
                {
                    "ref": str(rep.get("chapter_id") or ""),
                    "chunk_ids": list(union_ids),
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
    )


__all__ = [
    "DedupResult",
    "dedup_candidates",
    "resolve_dedup_threshold",
    "_DEFAULT_DEDUP_THRESHOLD",
    "ENV_DEDUP_THRESHOLD",
]
