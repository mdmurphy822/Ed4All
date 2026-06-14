"""W2 §4.1 — Pass C: the in-synthesis NLI-grounding FILTER.

This is the **IN-SYNTHESIS filter**, NOT a workflow validation gate. It runs
INSIDE ``_plan_course_structure`` (between the Pass-B MAP collect and the Pass-D
dedup), mutates the candidate pool (drops ungrounded candidates before they
become COs), and reuses
:func:`lib.retrieval.groundedness.score_groundedness` as a LIBRARY call. It emits
no GateResult.

**W4 owns the GATE-AS-VALIDATOR** — a ``course_planning``-phase validator that
scores the FINAL ``synthesized_objectives.json`` (post-mint) and emits a
GateResult. Both share :func:`score_groundedness` + the same
:data:`_DEFAULT_ENTAILMENT_FLOOR` (re-exported from ``lib.objectives``) +
``thresholds_provenance`` so "grounded" means the same thing in both places.
**W2 must NOT add the gate; W4 must NOT mutate the candidate pool.**

Per-candidate algorithm (one objective statement = one NLI "claim"):

1. Resolve cited chunk_ids → chunk records. No cited chunk resolves → ungrounded,
   reason ``no_resolved_chunk``.
2. Score the statement against the cited passages via ``score_groundedness``.
3. Read the single ``ClaimVerdict.entailment`` (robust to the sentence-splitter
   dropping a terse objective: ``scored_count == 0`` → ungrounded, reason
   ``unscorable_statement`` — never fabricate grounded).
4. ``grounded = entailment >= entailment_floor``.

Re-ground rescue (deterministic, one-shot, default ON): for an ungrounded
candidate, re-score against ALL chunks in the candidate's chapter pool (not just
cited); if some non-cited chapter chunk entails ≥ floor, ADOPT that chunk_id and
mark ``grounded=True, reground=True``. Never crosses a chapter boundary, never a
second LLM call.

Graceful degrade: ``score_groundedness`` returns ``available=False`` (NLI extras
absent) AND ``require=False`` → Pass C is a PASS-THROUGH (every candidate keeps
``grounded=None``, ``entailment_score=None``, nothing dropped). ``require=True``
(bridged from ``TRAINFORGE_REQUIRE_EMBEDDINGS``) re-raises (score_groundedness
already raises). NLI-absent ≠ silently-grounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lib.objectives import _DEFAULT_ENTAILMENT_FLOOR
from lib.retrieval.groundedness import (
    THRESHOLDS_PROVENANCE_DEFAULTS,
    score_groundedness,
)

#: Deterministic re-ground rescue is cheap (no LLM) so it ships ON by default;
#: a module const rather than an env flag (the rescue is always safe + bounded
#: to the chapter chunk pool).
_OBJECTIVE_REGROUND = True


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of the Pass-C filter over a candidate pool."""

    grounded: List[Dict[str, Any]] = field(default_factory=list)
    ungrounded: List[Dict[str, Any]] = field(default_factory=list)
    available: bool = False
    scored_count: int = 0
    dropped_count: int = 0
    reground_count: int = 0
    thresholds_provenance: str = THRESHOLDS_PROVENANCE_DEFAULTS
    entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded": [dict(c) for c in self.grounded],
            "ungrounded": [dict(c) for c in self.ungrounded],
            "available": self.available,
            "scored_count": self.scored_count,
            "dropped_count": self.dropped_count,
            "reground_count": self.reground_count,
            "thresholds_provenance": self.thresholds_provenance,
            "entailment_floor": self.entailment_floor,
        }


def _candidate_statement(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("statement") or candidate.get("text") or "").strip()


def _candidate_source_chunk_ids(candidate: Dict[str, Any]) -> List[str]:
    raw = candidate.get("source_chunk_ids")
    if isinstance(raw, list):
        return [str(c) for c in raw if c]
    return []


def _candidate_chapter_id(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("chapter_id") or "")


def _single_entailment(report: Any) -> Optional[float]:
    """Read the single claim's entailment from a GroundednessReport.

    Returns ``None`` when nothing was scored (terse statement dropped by the
    sentence-splitter) so the caller can treat it as ``unscorable_statement``
    rather than fabricating a grounded verdict.
    """
    if getattr(report, "scored_count", 0) <= 0:
        return None
    claims = getattr(report, "claims", None) or []
    # The objective statement is one sentence; take the max entailment over any
    # scored claim it produced (robust if the splitter yields >1 fragment).
    entailments = [
        float(getattr(c, "entailment", 0.0))
        for c in claims
        if getattr(c, "verdict", "") != "computational_unverified"
    ]
    if not entailments:
        return None
    return max(entailments)


def _passages_from_ids(
    chunk_ids: List[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    passages: List[Dict[str, Any]] = []
    for cid in chunk_ids:
        rec = chunks_by_id.get(cid)
        if not isinstance(rec, dict):
            continue
        text = str(rec.get("text") or rec.get("body") or "")
        if text.strip():
            passages.append({"chunk_id": cid, "text": text})
    return passages


def _chapter_chunk_pool(
    chapter_id: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
    cited_ids: List[str],
) -> List[Dict[str, Any]]:
    """All chunks of ``chapter_id`` NOT already cited (the rescue pool).

    A chunk belongs to the chapter when its record carries a matching
    ``chapter_id`` field; absent that linkage the pool is empty (the rescue
    simply doesn't fire — never cross-chapter).
    """
    cited = set(cited_ids)
    pool: List[Dict[str, Any]] = []
    for cid, rec in chunks_by_id.items():
        if cid in cited or not isinstance(rec, dict):
            continue
        if str(rec.get("chapter_id") or "") != chapter_id or not chapter_id:
            continue
        text = str(rec.get("text") or rec.get("body") or "")
        if text.strip():
            pool.append({"chunk_id": cid, "text": text})
    return pool


def ground_candidates(
    candidates: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    nli: Optional[Any] = None,
    entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
    require: bool = False,
    reground: bool = _OBJECTIVE_REGROUND,
) -> GroundingResult:
    """Filter ``candidates`` by NLI groundedness against their cited chunks.

    Args:
        candidates: Pass-B output; each carries ``statement`` + ``source_chunk_ids``.
        chunks_by_id: ``chunk_id -> chunk record`` (must carry ``text``; an
            optional ``chapter_id`` enables the re-ground rescue pool).
        nli: injection seam for CI fakes (a stub with ``score_batch(pairs=...)``).
        entailment_floor: grounded iff per-claim entailment ≥ this.
        require: bridged from ``TRAINFORGE_REQUIRE_EMBEDDINGS``; True re-raises on
            NLI-absent (no fabricated scores).
        reground: enable the deterministic adjacent-chunk rescue (default ON).

    Returns a :class:`GroundingResult`. On NLI-absent + ``require=False`` every
    candidate passes through with ``grounded=None`` / ``entailment_score=None``.
    """
    # Probe NLI availability ONCE with a cheap call so the pass-through decision
    # is made before we iterate. score_groundedness handles strict-mode raising.
    if candidates:
        probe = score_groundedness(
            "availability probe sentence with several content tokens",
            [{"chunk_id": "_probe", "text": "availability probe premise text"}],
            nli=nli,
            entailment_floor=entailment_floor,
        )
        if not getattr(probe, "available", False):
            # NLI extras absent (require=False reached here; require=True would
            # have raised inside score_groundedness). Pass-through, no drops.
            passthrough = []
            for cand in candidates:
                c = dict(cand)
                c["grounded"] = None
                c["entailment_score"] = None
                passthrough.append(c)
            return GroundingResult(
                grounded=passthrough,
                ungrounded=[],
                available=False,
                scored_count=0,
                dropped_count=0,
                reground_count=0,
                entailment_floor=float(entailment_floor),
            )

    grounded: List[Dict[str, Any]] = []
    ungrounded: List[Dict[str, Any]] = []
    scored_count = 0
    reground_count = 0

    for cand in candidates:
        c = dict(cand)
        statement = _candidate_statement(c)
        cited_ids = _candidate_source_chunk_ids(c)
        cited_passages = _passages_from_ids(cited_ids, chunks_by_id)

        if not statement:
            c["grounded"] = False
            c["entailment_score"] = 0.0
            c["grounding_reason"] = "empty_statement"
            ungrounded.append(c)
            continue

        if not cited_passages:
            # No cited chunk resolves — try the rescue directly against the
            # chapter pool before giving up.
            entailment = None
            best_chunk_id = None
        else:
            report = score_groundedness(
                statement,
                cited_passages,
                nli=nli,
                entailment_floor=entailment_floor,
            )
            entailment = _single_entailment(report)
            best_chunk_id = None
            claims = getattr(report, "claims", None) or []
            if claims:
                best_chunk_id = getattr(claims[0], "best_chunk_id", None)
            if entailment is not None:
                scored_count += 1

        if entailment is not None and entailment >= entailment_floor:
            c["grounded"] = True
            c["entailment_score"] = round(float(entailment), 6)
            c["grounding_reason"] = "cited_chunk_entailed"
            grounded.append(c)
            continue

        # --- Deterministic re-ground rescue. -------------------------------
        rescued = False
        if reground:
            pool = _chapter_chunk_pool(
                _candidate_chapter_id(c), chunks_by_id, cited_ids
            )
            if pool:
                rescue_report = score_groundedness(
                    statement,
                    pool,
                    nli=nli,
                    entailment_floor=entailment_floor,
                )
                rescue_ent = _single_entailment(rescue_report)
                if rescue_ent is not None:
                    if entailment is None:
                        scored_count += 1
                    if rescue_ent >= entailment_floor:
                        # Adopt the entailing adjacent chunk's id.
                        adopt_id = None
                        rc = getattr(rescue_report, "claims", None) or []
                        if rc:
                            adopt_id = getattr(rc[0], "best_chunk_id", None)
                        new_ids = list(cited_ids)
                        if adopt_id and adopt_id not in new_ids:
                            new_ids.append(adopt_id)
                        c["source_chunk_ids"] = new_ids
                        # Keep on-disk source_refs back-compat in sync.
                        _resync_source_refs(c, new_ids)
                        c["grounded"] = True
                        c["entailment_score"] = round(float(rescue_ent), 6)
                        c["grounding_reason"] = "reground_adjacent_chunk"
                        c["reground"] = True
                        reground_count += 1
                        grounded.append(c)
                        rescued = True

        if not rescued:
            c["grounded"] = False
            if entailment is None and not cited_passages:
                c["entailment_score"] = 0.0
                c["grounding_reason"] = "no_resolved_chunk"
            elif entailment is None:
                c["entailment_score"] = None
                c["grounding_reason"] = "unscorable_statement"
            else:
                c["entailment_score"] = round(float(entailment), 6)
                c["grounding_reason"] = "below_entailment_floor"
            ungrounded.append(c)

    return GroundingResult(
        grounded=grounded,
        ungrounded=ungrounded,
        available=True,
        scored_count=scored_count,
        dropped_count=len(ungrounded),
        reground_count=reground_count,
        entailment_floor=float(entailment_floor),
    )


def _resync_source_refs(candidate: Dict[str, Any], chunk_ids: List[str]) -> None:
    """Rebuild ``source_refs[].chunk_ids`` from the surviving ``source_chunk_ids``.

    Preserves the on-disk back-compat shape ``[{ref: chapter_id, chunk_ids: [...]}]``
    that ``ObjectiveSourceRefValidator`` reads.
    """
    chapter_id = _candidate_chapter_id(candidate) or candidate.get(
        "chapter_id", ""
    )
    candidate["source_refs"] = [
        {"ref": chapter_id, "chunk_ids": list(chunk_ids)}
    ]


__all__ = [
    "GroundingResult",
    "ground_candidates",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_OBJECTIVE_REGROUND",
]
