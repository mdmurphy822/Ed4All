"""I3 PRONG B — source-richness BACKFILL of dedup-discarded grounded COs.

Single-link cosine dedup keeps ONE representative statement per cluster and
DISCARDS the rest's statements (only their ``source_chunk_ids`` survive, unioned
onto the rep). When a discarded candidate named a DISTINCT teachable skill that
the kept rep does NOT name, that skill silently vanishes from the objectives set
and the chunk it cited ends up cited by no canonical CO — content generation
never authors a page for it.

This pass is the BACKFILL cure. It retains the PRE-DEDUP grounded candidate pool
and, for every CONTENT-BEARING chunk left uncited by the post-dedup/split
canonical set, PROMOTES the best-grounded DISCARDED pre-dedup candidate that
cites that chunk into a new canonical CO.

ANTI-FABRICATION (hard contract):
  * PROMOTION ONLY — no re-synthesis. A promoted CO is a verbatim copy of an
    already-synthesized, already-grounded pre-dedup candidate.
  * The promoted CO's ``source_chunk_ids`` are a STRICT SUBSET of the promoted
    pre-dedup candidate's ``source_chunk_ids`` (we never widen provenance).
  * A candidate is eligible only if it NAMES a distinct skill the canonical set
    does not already name (skill-signature check, reused from PRONG A) — we
    never promote a near-restatement of an existing CO.

COVERAGE TARGET (locked decision): content-bearing chunks carrying a distinct
teachable skill ONLY. Figures, exercises, assessment items, and front-matter
chunks get NO CO (fabricating a CO for a non-instructional chunk is worse than
the gap). The content-bearing filter is deterministic (chunk_type + word floor).

Default OFF (opt-in via ``ED4ALL_OBJECTIVE_SOURCE_BACKFILL``) — default-off is
byte-identical to today.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: I3 PRONG B — source-richness backfill toggle. Default OFF (opt-in).
_DEFAULT_SOURCE_BACKFILL = False
ENV_SOURCE_BACKFILL = "ED4ALL_OBJECTIVE_SOURCE_BACKFILL"

#: Coverage target — minimum fraction of content-bearing chunks that must be
#: cited by the canonical CO set. The backfill promotes candidates until this is
#: met (or it runs out of eligible discarded candidates). Default 1.0 (cover
#: every content-bearing chunk that has an eligible discarded candidate). A
#: lower value caps how aggressively the pass promotes.
_DEFAULT_COVERAGE_TARGET = 1.0
ENV_COVERAGE_TARGET = "ED4ALL_OBJECTIVE_BACKFILL_COVERAGE_TARGET"

#: Content-bearing chunk_types — only these carry a distinct teachable skill.
#: Exercises / assessment_items / figures / front-matter are excluded (no CO).
_CONTENT_BEARING_CHUNK_TYPES = frozenset({"explanation", "concept", "content"})

#: Minimum word_count for a chunk to be content-bearing (a sub-floor stub is a
#: caption / heading / answer-key fragment, not teachable prose).
_MIN_CONTENT_WORD_COUNT = 40


@dataclass
class BackfillResult:
    """Outcome of the I3 PRONG-B source-richness backfill."""

    canonical: List[Dict[str, Any]] = field(default_factory=list)
    uncovered_before: int = 0
    uncovered_after: int = 0
    cos_promoted: int = 0
    named_skills_added: List[str] = field(default_factory=list)
    content_bearing_chunks: int = 0
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uncovered_before": self.uncovered_before,
            "uncovered_after": self.uncovered_after,
            "cos_promoted": self.cos_promoted,
            "named_skills_added": list(self.named_skills_added),
            "content_bearing_chunks": self.content_bearing_chunks,
            "available": self.available,
        }


def resolve_source_backfill(enabled: Optional[bool] = None) -> bool:
    """Resolve the PRONG-B backfill toggle: arg → env → default (OFF)."""
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_SOURCE_BACKFILL, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_SOURCE_BACKFILL


def resolve_coverage_target(target: Optional[float] = None) -> float:
    """Resolve the coverage target: arg → env → default. Clamped to [0,1]."""
    if target is not None:
        try:
            return max(0.0, min(1.0, float(target)))
        except (TypeError, ValueError):
            return _DEFAULT_COVERAGE_TARGET
    raw = os.environ.get(ENV_COVERAGE_TARGET)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_COVERAGE_TARGET
        if 0.0 <= val <= 1.0:
            return val
    return _DEFAULT_COVERAGE_TARGET


def _chunk_id(rec: Any) -> str:
    if not isinstance(rec, dict):
        return ""
    return str(rec.get("id") or rec.get("chunk_id") or "")


def _chunk_word_count(rec: Dict[str, Any]) -> int:
    wc = rec.get("word_count")
    if isinstance(wc, (int, float)):
        return int(wc)
    text = str(rec.get("text") or rec.get("body") or "")
    return len(text.split())


def is_content_bearing(rec: Any) -> bool:
    """Deterministic content-bearing filter (no LLM, no embeddings).

    A chunk is content-bearing iff its ``chunk_type`` is teachable prose
    (explanation/concept/content) AND it clears the word-count floor. Figures,
    exercises, assessment items, and front-matter stubs are NOT content-bearing
    — fabricating an objective for them is worse than the coverage gap.
    """
    if not isinstance(rec, dict):
        return False
    ctype = str(rec.get("chunk_type") or "").strip().lower()
    if ctype and ctype not in _CONTENT_BEARING_CHUNK_TYPES:
        return False
    if not ctype:
        # No type → fall back to the word floor alone (legacy chunks).
        return _chunk_word_count(rec) >= _MIN_CONTENT_WORD_COUNT
    return _chunk_word_count(rec) >= _MIN_CONTENT_WORD_COUNT


def _source_chunk_ids(candidate: Dict[str, Any]) -> List[str]:
    raw = candidate.get("source_chunk_ids")
    if isinstance(raw, list):
        return [str(c) for c in raw if c]
    return []


def _statement(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("statement") or candidate.get("text") or "").strip()


def _entailment(candidate: Dict[str, Any]) -> float:
    val = candidate.get("entailment_score")
    try:
        return float(val) if val is not None else -1.0
    except (TypeError, ValueError):
        return -1.0


def backfill_uncovered_chunks(
    *,
    canonical: List[Dict[str, Any]],
    pre_dedup_candidates: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    enabled: Optional[bool] = None,
    coverage_target: Optional[float] = None,
    capture: Optional[Any] = None,
) -> BackfillResult:
    """Promote dedup-discarded grounded candidates to cover uncited chunks.

    Args:
        canonical: the post-dedup/split canonical CO set (mutated by appending
            promoted COs).
        pre_dedup_candidates: the FULL grounded candidate pool BEFORE dedup
            (the discarded-statement source for promotion).
        chunks_by_id: ``chunk_id -> chunk record`` (carries ``chunk_type`` /
            ``word_count`` / ``text``) for the content-bearing filter.
        enabled / coverage_target: resolved via the ``resolve_*`` helpers.
        capture: optional DecisionCapture for one ``objective_source_backfill``
            event (best-effort).

    Returns a :class:`BackfillResult`. Default-off / empty inputs → no-op
    (``canonical`` unchanged, ``available=False``).
    """
    from lib.objectives.objective_dedup import _skill_signature

    if not resolve_source_backfill(enabled):
        return BackfillResult(canonical=canonical, available=False)
    if not pre_dedup_candidates or not chunks_by_id:
        return BackfillResult(canonical=canonical, available=False)

    target = resolve_coverage_target(coverage_target)

    # Content-bearing chunk universe.
    content_chunk_ids = {
        cid
        for cid, rec in chunks_by_id.items()
        if is_content_bearing(rec)
    }
    n_content = len(content_chunk_ids)
    if n_content == 0:
        return BackfillResult(canonical=canonical, available=True)

    # Chunks the current canonical set already cites.
    def _covered_ids() -> set:
        out: set = set()
        for co in canonical:
            for cid in _source_chunk_ids(co):
                out.add(cid)
        return out

    covered = _covered_ids()
    uncovered = content_chunk_ids - covered
    uncovered_before = len(uncovered)

    # Existing canonical skill signatures (never promote a near-restatement).
    canonical_sigs: List[frozenset] = []
    for co in canonical:
        sig = _skill_signature(co)
        if sig:
            canonical_sigs.append(sig)

    from lib.objectives.objective_dedup import _signatures_distinct

    def _names_distinct_skill(cand: Dict[str, Any]) -> bool:
        sig = _skill_signature(cand)
        if not sig:
            return False
        # Distinct from EVERY existing canonical skill.
        for existing in canonical_sigs:
            if not _signatures_distinct(sig, existing):
                return False
        return True

    # Index discarded candidates by the content-bearing chunks they cite, so we
    # can find a promotion source for each uncovered chunk. Best-grounded first.
    promoted: List[Dict[str, Any]] = []
    named_skills: List[str] = []

    def _coverage_ratio() -> float:
        return (n_content - len(uncovered)) / n_content if n_content else 1.0

    # Order uncovered chunks deterministically (stable id sort).
    for cid in sorted(uncovered):
        if _coverage_ratio() >= target:
            break
        if cid not in uncovered:
            continue  # already covered by a prior promotion's union
        # Eligible discarded candidates that cite this chunk AND name a distinct
        # skill not already in the canonical set. Best-grounded → longest stmt.
        eligible = [
            c
            for c in pre_dedup_candidates
            if cid in _source_chunk_ids(c) and _names_distinct_skill(c)
        ]
        if not eligible:
            continue
        eligible.sort(key=lambda c: (_entailment(c), len(_statement(c))), reverse=True)
        best = eligible[0]

        # ANTI-FABRICATION: the promoted CO's source_chunk_ids are a STRICT
        # SUBSET of the promoted candidate's — keep only its content-bearing
        # cited chunks (never widen). Always retains the triggering chunk.
        cand_ids = _source_chunk_ids(best)
        kept_ids = [
            c for c in cand_ids
            if c in content_chunk_ids or c == cid
        ] or [cid]
        # Guarantee subset of the candidate's ids (anti-fabrication invariant).
        kept_ids = [c for c in kept_ids if c in cand_ids] or (
            [cid] if cid in cand_ids else []
        )
        if not kept_ids:
            continue

        promoted_co = dict(best)
        promoted_co["source_chunk_ids"] = list(kept_ids)
        promoted_co["source_refs"] = [
            {
                "ref": str(best.get("chapter_id") or ""),
                "chunk_ids": list(kept_ids),
            }
        ]
        promoted_co["backfilled"] = True
        promoted.append(promoted_co)
        canonical.append(promoted_co)

        # Register its skill signature so later promotions don't duplicate it.
        sig = _skill_signature(promoted_co)
        if sig:
            canonical_sigs.append(sig)
        named_skills.append(_statement(promoted_co)[:80])

        # Mark its kept content-bearing chunks as covered.
        for kid in kept_ids:
            uncovered.discard(kid)

    uncovered_after = len(uncovered)
    result = BackfillResult(
        canonical=canonical,
        uncovered_before=uncovered_before,
        uncovered_after=uncovered_after,
        cos_promoted=len(promoted),
        named_skills_added=named_skills,
        content_bearing_chunks=n_content,
        available=True,
    )
    if capture is not None and promoted:
        _emit_backfill_capture(capture, result=result, target=target)
    return result


def _emit_backfill_capture(
    capture: Any, *, result: BackfillResult, target: float
) -> None:
    """Best-effort ``objective_source_backfill`` capture (never raises)."""
    try:
        capture.log_decision(
            decision_type="objective_source_backfill",
            decision=(
                f"promoted {result.cos_promoted} discarded candidate(s) to "
                f"cover {result.uncovered_before - result.uncovered_after} "
                f"uncited content chunk(s)"
            ),
            rationale=(
                f"I3 PRONG B: dedup discarded distinct-skill candidate "
                f"statements (their chunks demoted to supporting evidence), "
                f"leaving {result.uncovered_before}/{result.content_bearing_chunks} "
                f"content-bearing chunk(s) cited by NO canonical CO. Promoted "
                f"the best-grounded discarded PRE-DEDUP candidate citing each "
                f"(coverage_target={target:.2f}); uncovered "
                f"{result.uncovered_before} -> {result.uncovered_after}. "
                f"Named skills added: {result.named_skills_added[:5]}. "
                f"Anti-fabrication: promotion only (no re-synthesis); each "
                f"promoted CO's source_chunk_ids subset the candidate's."
            ),
            alternatives_considered=[
                "leave content chunks uncovered (skill vanishes from course)",
                "re-synthesize a CO (fabrication — rejected)",
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "objective_source_backfill capture failed (%s); continuing", exc
        )


__all__ = [
    "BackfillResult",
    "backfill_uncovered_chunks",
    "is_content_bearing",
    "resolve_source_backfill",
    "resolve_coverage_target",
    "_DEFAULT_SOURCE_BACKFILL",
    "_DEFAULT_COVERAGE_TARGET",
    "ENV_SOURCE_BACKFILL",
    "ENV_COVERAGE_TARGET",
]
