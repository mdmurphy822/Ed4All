"""Groundedness scoring harness — per-claim NLI over (cited passage, answer
sentence) pairs (D6).

This module measures how well a grounded answer's sentences are *entailed* by
the passages it cited. It mirrors the per-claim loop of
:class:`lib.validators.pair.claim_support.PairClaimSupportValidator` —
sentence-split → content-token filter → NLI per ``(passage_text premise,
answer_sentence hypothesis)`` → verdict — but does NOT import that validator
class (it is pair-record-shaped and gate-runner-coupled). The shared knobs
(sentence-split regex, content-token regex, minimum-token floor, and the
entailment / contradiction floors) are imported from
``lib.validators.pair._claim_support_thresholds`` so the floors that scored a
report are auditable and a future Q&A-specific recalibration changes one kwarg,
not the report shape.

Groundedness is **advisory**: the floors were calibrated on training pairs, not
Q&A claims, so this harness emits warnings / counts, never a block. The honest
gap is recorded per-report: ``thresholds`` carries the floors actually used and
``thresholds_provenance`` pins their origin (``"claim_support_defaults"``).

Graceful degrade is the loader contract: when the singleton DeBERTa NLI model
is unavailable (missing ``transformers`` / ``torch`` extras or a load failure),
:func:`score_groundedness` returns ``GroundednessReport(available=False, ...)``
— no fabricated scores — UNLESS ``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` flips the
consumer into strict mode (the repo's established NLI strict-mode flag, per
``claim_support``), in which case the missing model raises ``RuntimeError``
naming the extras.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.validators.pair._claim_support_thresholds import (
    _CONTENT_TOKEN_RE,
    _DEFAULT_CONTRADICTION_FLOOR,
    _DEFAULT_ENTAILMENT_FLOOR,
    _MIN_SENTENCE_TOKENS,
    _SENTENCE_SPLIT_RE,
)

#: Verdict literals (mirrors claim_support's per-claim outcomes).
VERDICT_ENTAILED = "entailed"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_CONTRADICTED = "contradicted"

#: Audit pin recorded on every report so a post-hoc reader knows which floors
#: scored a run (the floors are training-pair-calibrated — § 0.4 of the plan).
THRESHOLDS_PROVENANCE_DEFAULTS = "claim_support_defaults"

#: The repo's established NLI / embedding strict-mode flag. When truthy, a
#: missing NLI model raises rather than degrading to ``available=False``.
ENV_REQUIRE_EMBEDDINGS = "TRAINFORGE_REQUIRE_EMBEDDINGS"


@dataclass(frozen=True)
class ClaimVerdict:
    """Per-answer-sentence NLI verdict against the cited passages."""

    claim_text: str
    verdict: str  # VERDICT_ENTAILED | VERDICT_UNSUPPORTED | VERDICT_CONTRADICTED
    entailment: float  # max entailment over the cited passages
    contradiction: float  # contradiction of the argmax-entailment passage
    best_chunk_id: Optional[str]  # argmax-entailment passage's chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "entailment": round(self.entailment, 6),
            "contradiction": round(self.contradiction, 6),
            "best_chunk_id": self.best_chunk_id,
        }


@dataclass(frozen=True)
class GroundednessReport:
    """Aggregate groundedness over one answer's claims (D6 report shape)."""

    available: bool  # False ⇒ NLI deps absent (honest degrade)
    claims: List[ClaimVerdict] = field(default_factory=list)
    groundedness_rate: float = 0.0  # entailed / scored claims (0.0 when none)
    unsupported_count: int = 0
    contradicted_count: int = 0
    scored_count: int = 0
    thresholds: Dict[str, float] = field(default_factory=dict)
    thresholds_provenance: str = THRESHOLDS_PROVENANCE_DEFAULTS
    nli_model_revision: Optional[str] = None
    reason: Optional[str] = None  # populated when available=False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "claims": [c.to_dict() for c in self.claims],
            "groundedness_rate": round(self.groundedness_rate, 6),
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "scored_count": self.scored_count,
            "thresholds": dict(self.thresholds),
            "thresholds_provenance": self.thresholds_provenance,
            "nli_model_revision": self.nli_model_revision,
            "reason": self.reason,
        }


def split_claims(answer_text: str) -> List[str]:
    """Split an answer into pedagogical claim sentences worth scoring.

    Uses the canonical ``_SENTENCE_SPLIT_RE`` split and the
    ``_MIN_SENTENCE_TOKENS`` content-token floor (both imported from
    ``_claim_support_thresholds``) so groundedness counts sentences the same
    way the pair-claim-support validator does. A sentence is kept only when it
    has at least ``_MIN_SENTENCE_TOKENS`` content tokens (``_CONTENT_TOKEN_RE``
    matches alphabetic tokens of length ≥ 2), filtering out fragments,
    headings, and bare references.
    """
    if not answer_text:
        return []
    claims: List[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(answer_text.strip()):
        sentence = raw.strip()
        if not sentence:
            continue
        if len(_CONTENT_TOKEN_RE.findall(sentence)) < _MIN_SENTENCE_TOKENS:
            continue
        claims.append(sentence)
    return claims


def _require_strict() -> bool:
    """True when ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is set to a truthy value."""
    return str(os.environ.get(ENV_REQUIRE_EMBEDDINGS, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_nli(nli: Optional[Any]) -> Optional[Any]:
    """Resolve the NLI classifier — injected instance or the singleton.

    The injection seam (``nli=...``) keeps CI hermetic: tests pass a
    deterministic fake. When ``nli is None`` we ask the process-singleton
    :class:`lib.classifiers.nli_classifier.NliClassifier` for the real model,
    which returns ``None`` on missing extras / load failure (graceful degrade).
    The import is lazy so the ~750 MB DeBERTa stack is never imported on the
    default (advisory-off) query path.
    """
    if nli is not None:
        return nli
    try:
        from lib.classifiers.nli_classifier import NliClassifier
    except Exception:  # noqa: BLE001 — missing module ⇒ degrade
        return None
    return NliClassifier.get_or_load()


def _passage_text(passage: Any) -> str:
    text = getattr(passage, "text", None)
    if text is None and isinstance(passage, dict):
        text = passage.get("text")
    return str(text or "")


def _passage_chunk_id(passage: Any) -> Optional[str]:
    cid = getattr(passage, "chunk_id", None)
    if cid is None and isinstance(passage, dict):
        cid = passage.get("chunk_id")
    return str(cid) if cid else None


def _model_revision(nli: Any) -> Optional[str]:
    """Best-effort read of the NLI model's pinned revision for provenance."""
    return getattr(nli, "_revision", None)


def score_groundedness(
    answer_text: str,
    cited_passages: Sequence[Any],
    *,
    nli: Optional[Any] = None,
    entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
    contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
) -> GroundednessReport:
    """Score each answer sentence against the cited passages via NLI.

    Premise = cited passage text; hypothesis = answer sentence. Each claim's
    entailment is the **max** over the cited passages (an answer sentence is
    entailed if any one cited passage supports it); its contradiction is read
    from the argmax-entailment passage. Verdict:

      * ``entailment >= entailment_floor`` → ``entailed``
      * else ``contradiction >= contradiction_floor`` → ``contradicted``
      * else → ``unsupported``

    ``groundedness_rate`` is ``entailed / scored_count`` (0.0 when nothing is
    scored — never fabricated as 1.0). All ``(passage, claim)`` pairs are
    batched through ``score_batch`` (one forward pass per 8 pairs); k passages ×
    m claims stays small (≤ 8 × ~10).

    Degrade: ``nli`` resolves to ``None`` → ``GroundednessReport(available=
    False, ...)`` with a ``reason``, UNLESS ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is
    truthy, then a ``RuntimeError`` naming the extras is raised.
    """
    thresholds = {
        "entailment_floor": float(entailment_floor),
        "contradiction_floor": float(contradiction_floor),
    }

    resolved = _resolve_nli(nli)
    if resolved is None:
        if _require_strict():
            raise RuntimeError(
                "Groundedness scoring requires the NLI model "
                "(MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) but it is "
                "unavailable; install the embedding/NLI extras "
                "(transformers + torch) or unset "
                f"{ENV_REQUIRE_EMBEDDINGS}. No groundedness scores were "
                "fabricated."
            )
        return GroundednessReport(
            available=False,
            claims=[],
            groundedness_rate=0.0,
            unsupported_count=0,
            contradicted_count=0,
            scored_count=0,
            thresholds=thresholds,
            thresholds_provenance=THRESHOLDS_PROVENANCE_DEFAULTS,
            nli_model_revision=None,
            reason="nli_unavailable",
        )

    claims = split_claims(answer_text)
    passages = [
        p
        for p in cited_passages
        if _passage_text(p).strip()
    ]

    if not claims or not passages:
        # NLI is available but there is nothing to score (empty answer or no
        # cited passages). available=True; rate is 0.0 (honest — no entailed
        # claims), scored_count=0.
        return GroundednessReport(
            available=True,
            claims=[],
            groundedness_rate=0.0,
            unsupported_count=0,
            contradicted_count=0,
            scored_count=0,
            thresholds=thresholds,
            thresholds_provenance=THRESHOLDS_PROVENANCE_DEFAULTS,
            nli_model_revision=_model_revision(resolved),
            reason=None if claims else "no_scorable_claims",
        )

    # Build the full (premise, hypothesis) grid: every cited passage × every
    # claim. Track (claim_index, passage_index) so we can fold max-entailment
    # back per claim after one batched forward pass.
    pairs: List[Tuple[str, str]] = []
    index_map: List[Tuple[int, int]] = []
    for ci, claim in enumerate(claims):
        for pi, passage in enumerate(passages):
            pairs.append((_passage_text(passage), claim))
            index_map.append((ci, pi))

    scores = resolved.score_batch(pairs=pairs)

    # Fold per-claim: argmax entailment over the claim's passage scores.
    # best[ci] = (entailment, contradiction, passage_index)
    best: Dict[int, Tuple[float, float, int]] = {}
    for (ci, pi), score in zip(index_map, scores):
        ent = float(getattr(score, "entailment", 0.0))
        con = float(getattr(score, "contradiction", 0.0))
        cur = best.get(ci)
        if cur is None or ent > cur[0]:
            best[ci] = (ent, con, pi)

    verdicts: List[ClaimVerdict] = []
    entailed_count = 0
    unsupported_count = 0
    contradicted_count = 0
    for ci, claim in enumerate(claims):
        ent, con, pi = best.get(ci, (0.0, 0.0, -1))
        if ent >= entailment_floor:
            verdict = VERDICT_ENTAILED
            entailed_count += 1
        elif con >= contradiction_floor:
            verdict = VERDICT_CONTRADICTED
            contradicted_count += 1
        else:
            verdict = VERDICT_UNSUPPORTED
            unsupported_count += 1
        best_chunk_id = (
            _passage_chunk_id(passages[pi]) if 0 <= pi < len(passages) else None
        )
        verdicts.append(
            ClaimVerdict(
                claim_text=claim,
                verdict=verdict,
                entailment=ent,
                contradiction=con,
                best_chunk_id=best_chunk_id,
            )
        )

    scored_count = len(verdicts)
    rate = (entailed_count / scored_count) if scored_count else 0.0

    return GroundednessReport(
        available=True,
        claims=verdicts,
        groundedness_rate=rate,
        unsupported_count=unsupported_count,
        contradicted_count=contradicted_count,
        scored_count=scored_count,
        thresholds=thresholds,
        thresholds_provenance=THRESHOLDS_PROVENANCE_DEFAULTS,
        nli_model_revision=_model_revision(resolved),
        reason=None,
    )


__all__ = [
    "ClaimVerdict",
    "GroundednessReport",
    "split_claims",
    "score_groundedness",
    "VERDICT_ENTAILED",
    "VERDICT_UNSUPPORTED",
    "VERDICT_CONTRADICTED",
    "THRESHOLDS_PROVENANCE_DEFAULTS",
    "ENV_REQUIRE_EMBEDDINGS",
]
