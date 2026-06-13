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
import re
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
#: v2: a claim that is novel arithmetic / a derived numeric result. A textbook
#: premise can never *entail* a computation the model performed, so NLI verdicts
#: on these are noise (the 2026-06-12 audit found 5/18 non-entailed claims were
#: correct-but-unentailable computations, and NLI even called a verified-correct
#: result "contradicted" at 0.90). These are EXEMPTED from NLI and from the
#: groundedness denominator; auto-verification of the arithmetic is future work.
VERDICT_COMPUTATIONAL = "computational_unverified"

#: Scorer surface version stamped on every report (v1 = whole-chunk-premise grid
#: only; v2 = artifact filtering + computational exemption + windowed rescue +
#: wider evidence pool + cited-flag). Additive — never changes existing keys.
SCORER_VERSION = "2"

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
    verdict: str  # VERDICT_ENTAILED | _UNSUPPORTED | _CONTRADICTED | _COMPUTATIONAL
    entailment: float  # max entailment over the evidence pool
    contradiction: float  # contradiction of the argmax-entailment premise
    best_chunk_id: Optional[str]  # argmax-entailment passage's chunk_id
    #: v2 additive — True when the verdict came from the stage-2 windowed rescue
    #: pass (sentence-window premise) rather than the stage-1 whole-passage grid.
    windowed: bool = False
    #: v2 additive — whether ``best_chunk_id`` is one of the citations the model
    #: actually emitted. ``None`` when ``cited_chunk_ids`` was not supplied (the
    #: caller did not ask for the cited/uncited split).
    best_chunk_cited: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "entailment": round(self.entailment, 6),
            "contradiction": round(self.contradiction, 6),
            "best_chunk_id": self.best_chunk_id,
            # v2 additive keys (existing keys above unchanged).
            "windowed": self.windowed,
            "best_chunk_cited": self.best_chunk_cited,
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
    #: Torch device the NLI model scored on (``"cpu"`` / ``"cuda*"``).
    #: Recorded for provenance — GPU softmax is non-associative so a
    #: cuda-scored run can differ ~1e-6 from a cpu pin (verdict thresholds
    #: are robust). Mirrors the embedding manifest recording its device.
    nli_device: Optional[str] = None
    reason: Optional[str] = None  # populated when available=False
    #: v2 additive — claims routed to the computational exemption (NLI-skipped,
    #: excluded from ``scored_count`` / ``groundedness_rate`` / the un/contra
    #: counts).
    computational_count: int = 0
    #: v2 additive — structural non-claims dropped by :func:`_split_claims_for_scoring`.
    filtered_count: int = 0
    #: v2 additive — scorer surface version (see :data:`SCORER_VERSION`).
    scorer_version: str = SCORER_VERSION

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
            "nli_device": self.nli_device,
            "reason": self.reason,
            # v2 additive keys (existing keys above unchanged).
            "computational_count": self.computational_count,
            "filtered_count": self.filtered_count,
            "scorer_version": self.scorer_version,
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

    Public, backward-compatible, and reused by ``citation_attribution`` — its
    output is the canonical claim unit there, so the v2 structural-artifact
    filter (:func:`_is_structural_artifact`) is applied only in the
    groundedness wrapper (:func:`_split_claims_for_scoring`), NOT here, to keep
    attribution's claim set unchanged.
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


# --------------------------------------------------------------------------- #
# v2 claim-shape helpers (2026-06-12 audit response)
# --------------------------------------------------------------------------- #

#: Matched dict/JSON/bracket literal envelope: stripped sentence opens with one
#: of ``{ [`` and closes with the matching bracket. Catches claim-splitter
#: artifacts like ``{'lower_bound': 25, 'upper_bound': 28}`` that the audit
#: flagged as non-claims.
_BRACKET_PAIRS = {"{": "}", "[": "]"}

#: Enumeration stub — a sentence that is just a label ending in a colon,
#: optionally trailed by a bare enumerator (``1.`` / ``2)``). Whitespace is
#: normalized before matching so ``Steps :  1.`` is caught too.
_ENUM_STUB_RE = re.compile(r":\s*(?:\d+[.)])?$")

#: Derived-numeric / equation signal for the computational exemption. Any of:
#:   * an ``=`` with a digit on either side (within a few chars),
#:   * digit-operator-digit (``3 + 10``, ``25*4``, ``2^3``),
#:   * a result-announcing phrase (``the answer is`` / ``equals`` / ``simplified
#:     form is``) immediately followed by digits.
_DIGIT_OP_RE = re.compile(r"\d\s*[-+×*/÷^]\s*\d")
_EQ_DIGIT_RE = re.compile(r"(?:\d\s*=)|(?:=\s*[-+]?\s*\d)")
_RESULT_PHRASE_RE = re.compile(
    r"(?:the answer is|equals|simplified form is|simplifies to|"
    r"the (?:result|value|solution) is)\s*[:=]?\s*[-+(]?\s*\d",
    re.IGNORECASE,
)


def _is_structural_artifact(sentence: str) -> bool:
    """True when ``sentence`` is a structural non-claim worth dropping.

    Conservative (the audit's claim-splitter artifacts were only 2/39): a
    sentence is dropped only if it is (a) a fully-bracketed dict/JSON/list
    literal, or (b) an enumeration stub (ends in ``:`` optionally followed by a
    bare enumerator). The ``_MIN_SENTENCE_TOKENS`` floor in
    :func:`split_claims` already removes most short fragments; this catches the
    two shapes that survive it because they carry digits/punctuation.
    """
    stripped = sentence.strip()
    if not stripped:
        return False
    # (a) bracket literal: opens with { or [ and ends with its matching close.
    #     Tolerate a single trailing sentence terminator the splitter left on
    #     (``{...}.`` — the splitter keeps the terminator attached to the
    #     preceding sentence).
    body = stripped[:-1] if stripped[-1] in ".!?" else stripped
    body = body.strip()
    if body:
        close = _BRACKET_PAIRS.get(body[0])
        if close is not None and body.endswith(close):
            return True
    # (b) enumeration stub on the whitespace-normalized sentence.
    normalized = " ".join(stripped.split())
    if _ENUM_STUB_RE.search(normalized):
        return True
    return False


def _is_computational(sentence: str) -> bool:
    """True when ``sentence`` asserts a derived numeric result / equation.

    Such claims are exempted from NLI: a textbook premise cannot entail a
    computation the model just performed (see :data:`VERDICT_COMPUTATIONAL`).
    """
    if not sentence:
        return False
    if _DIGIT_OP_RE.search(sentence):
        return True
    if _EQ_DIGIT_RE.search(sentence):
        return True
    if _RESULT_PHRASE_RE.search(sentence):
        return True
    return False


def _split_claims_for_scoring(answer_text: str) -> Tuple[List[str], int]:
    """Claim list for scoring + count of structural artifacts dropped.

    Wraps :func:`split_claims` (so attribution's claim unit is untouched) and
    drops :func:`_is_structural_artifact` sentences, returning the surviving
    claims and how many were filtered. Computational claims are NOT dropped
    here — they survive as claims and are routed to the exemption verdict in
    :func:`score_groundedness`.
    """
    kept: List[str] = []
    filtered = 0
    for claim in split_claims(answer_text):
        if _is_structural_artifact(claim):
            filtered += 1
            continue
        kept.append(claim)
    return kept, filtered


def _sentence_windows(
    text: str, *, window: int = 3, stride: int = 2, min_chars: int = 20
) -> List[str]:
    """Sentence-window premises for the stage-2 windowed rescue pass.

    Splits ``text`` on the canonical ``_SENTENCE_SPLIT_RE`` and slides a
    ``window``-sentence window with the given ``stride``, joining each window
    back into a premise string. Windows under ``min_chars`` (post-strip) are
    skipped (too short to entail anything). Glossary-style multi-topic chunks
    defeat whole-chunk-premise NLI; a tight 3-sentence window restores the
    local support signal the audit found buried in the cited chunk.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []
    windows: List[str] = []
    for i in range(0, len(sentences), stride):
        chunk = " ".join(sentences[i : i + window])
        if len(chunk) >= min_chars:
            windows.append(chunk)
        if i + window >= len(sentences):
            break
    return windows


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


def _model_device(nli: Any) -> Optional[str]:
    """Best-effort read of the NLI model's torch device for provenance.

    Mirrors :func:`_model_revision`. ``None`` for any scorer that doesn't
    expose a ``device`` attribute (defensive against injected stubs).
    """
    return getattr(nli, "device", None)


#: Stage-2 (windowed rescue) pair cap. Keeps runtime bounded on large evidence
#: pools — over this many windows we take the first N per passage (round-robin
#: by passage order). 512 windows × ~10 unrescued claims is a sane upper bound.
_MAX_STAGE2_WINDOWS = 512


def score_groundedness(
    answer_text: str,
    cited_passages: Sequence[Any],
    *,
    nli: Optional[Any] = None,
    entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
    contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
    cited_chunk_ids: Optional[set] = None,
) -> GroundednessReport:
    """Score each answer sentence against the evidence pool via NLI (v2).

    Premise = evidence-pool passage text; hypothesis = answer sentence. v2
    pipeline:

      1. **Artifact filtering** — structural non-claims (dict/JSON/bracket
         literals, enumeration stubs) are dropped pre-scoring;
         ``filtered_count`` records how many.
      2. **Computational exemption** — claims asserting a derived numeric
         result / equation are NOT sent through NLI (a textbook premise cannot
         entail novel arithmetic); they get ``VERDICT_COMPUTATIONAL`` and are
         EXCLUDED from ``scored_count`` / ``groundedness_rate`` / the
         un/contra counts. ``computational_count`` records them.
      3. **Stage-1 whole-passage grid** — every pool passage × every scorable
         claim; per-claim entailment is the **max** over passages, contradiction
         is read from the argmax-entailment passage.
      4. **Stage-2 windowed rescue** — for claims still below
         ``entailment_floor``, re-score against 3-sentence windows (stride 2) of
         each passage and take the max. Glossary-style multi-topic chunks defeat
         a whole-chunk premise; a tight window restores the local support
         signal. A rescued claim records ``windowed=True`` and reads its
         contradiction from the windowed argmax-entailment window.

    Verdict (for scored, non-computational claims):

      * ``entailment >= entailment_floor`` → ``entailed``
      * else ``contradiction >= contradiction_floor`` → ``contradicted``
      * else → ``unsupported``

    ``groundedness_rate`` is ``entailed / scored_count`` (0.0 when nothing is
    scored — never fabricated as 1.0). When ``cited_chunk_ids`` is supplied each
    verdict's ``best_chunk_cited`` flags whether its supporting chunk was one the
    model actually cited (the pool is wider than the citations so a corpus
    supporter the model under-cited can still entail a claim); ``None`` when the
    kwarg is omitted.

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

    # Stage 0: split, drop structural artifacts, and partition out
    # computational claims (NLI-exempt). ``order`` preserves the answer's
    # claim order so the emitted verdict list reads top-to-bottom.
    all_claims, filtered_count = _split_claims_for_scoring(answer_text)
    order: List[Tuple[str, str]] = []  # (kind, claim) — kind in {"comp","score"}
    scorable: List[str] = []
    for claim in all_claims:
        if _is_computational(claim):
            order.append(("comp", claim))
        else:
            order.append(("score", claim))
            scorable.append(claim)
    computational_count = sum(1 for kind, _ in order if kind == "comp")

    passages = [p for p in cited_passages if _passage_text(p).strip()]

    if not scorable or not passages:
        # NLI is available but there are no SCORABLE claims (empty answer, only
        # computational/artifact claims) or no evidence passages. Still emit any
        # computational verdicts (excluded from the denominator) so the report
        # is honest about what it saw. rate is 0.0 (no entailed scored claims).
        comp_verdicts = [
            ClaimVerdict(
                claim_text=claim,
                verdict=VERDICT_COMPUTATIONAL,
                entailment=0.0,
                contradiction=0.0,
                best_chunk_id=None,
                windowed=False,
                best_chunk_cited=None,
            )
            for kind, claim in order
            if kind == "comp"
        ]
        return GroundednessReport(
            available=True,
            claims=comp_verdicts,
            groundedness_rate=0.0,
            unsupported_count=0,
            contradicted_count=0,
            scored_count=0,
            thresholds=thresholds,
            thresholds_provenance=THRESHOLDS_PROVENANCE_DEFAULTS,
            nli_model_revision=_model_revision(resolved),
            nli_device=_model_device(resolved),
            reason=None if all_claims else "no_scorable_claims",
            computational_count=computational_count,
            filtered_count=filtered_count,
        )

    # Stage 1: full (premise, hypothesis) grid — every pool passage × every
    # scorable claim. Track (claim_index, passage_index) so we can fold
    # max-entailment back per claim after one batched forward pass.
    pairs: List[Tuple[str, str]] = []
    index_map: List[Tuple[int, int]] = []
    for ci, claim in enumerate(scorable):
        for pi, passage in enumerate(passages):
            pairs.append((_passage_text(passage), claim))
            index_map.append((ci, pi))

    scores = resolved.score_batch(pairs=pairs)

    # Fold per-claim: argmax entailment over the claim's passage scores.
    # best[ci] = (entailment, contradiction, passage_index, windowed_bool)
    best: Dict[int, Tuple[float, float, int, bool]] = {}
    for (ci, pi), score in zip(index_map, scores):
        ent = float(getattr(score, "entailment", 0.0))
        con = float(getattr(score, "contradiction", 0.0))
        cur = best.get(ci)
        if cur is None or ent > cur[0]:
            best[ci] = (ent, con, pi, False)

    # Stage 2: windowed rescue for claims still below the entailment floor. For
    # each such claim, re-score against 3-sentence windows (stride 2) of each
    # passage; the winning window's entailment/contradiction + its passage's
    # chunk_id replace the stage-1 result when it clears the floor. Capped at
    # ``_MAX_STAGE2_WINDOWS`` premises (first-N per passage) to bound runtime.
    unrescued = [ci for ci in range(len(scorable)) if best.get(ci, (0.0,))[0] < entailment_floor]
    if unrescued:
        window_premises: List[str] = []
        window_passage_idx: List[int] = []
        for pi, passage in enumerate(passages):
            for win in _sentence_windows(_passage_text(passage)):
                window_premises.append(win)
                window_passage_idx.append(pi)
                if len(window_premises) >= _MAX_STAGE2_WINDOWS:
                    break
            if len(window_premises) >= _MAX_STAGE2_WINDOWS:
                break
        if window_premises:
            s2_pairs: List[Tuple[str, str]] = []
            s2_map: List[Tuple[int, int]] = []  # (claim_index, window_index)
            for ci in unrescued:
                claim = scorable[ci]
                for wi, premise in enumerate(window_premises):
                    s2_pairs.append((premise, claim))
                    s2_map.append((ci, wi))
            s2_scores = resolved.score_batch(pairs=s2_pairs)
            # Argmax windowed entailment per rescued claim.
            s2_best: Dict[int, Tuple[float, float, int]] = {}
            for (ci, wi), score in zip(s2_map, s2_scores):
                ent = float(getattr(score, "entailment", 0.0))
                con = float(getattr(score, "contradiction", 0.0))
                cur = s2_best.get(ci)
                if cur is None or ent > cur[0]:
                    s2_best[ci] = (ent, con, wi)
            for ci, (ent, con, wi) in s2_best.items():
                if ent >= entailment_floor:
                    best[ci] = (ent, con, window_passage_idx[wi], True)

    # Build the scored verdicts (in scorable order).
    scored_verdicts: Dict[int, ClaimVerdict] = {}
    entailed_count = 0
    unsupported_count = 0
    contradicted_count = 0
    for ci, claim in enumerate(scorable):
        ent, con, pi, windowed = best.get(ci, (0.0, 0.0, -1, False))
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
        best_chunk_cited: Optional[bool] = None
        if cited_chunk_ids is not None:
            best_chunk_cited = bool(best_chunk_id) and best_chunk_id in cited_chunk_ids
        scored_verdicts[ci] = ClaimVerdict(
            claim_text=claim,
            verdict=verdict,
            entailment=ent,
            contradiction=con,
            best_chunk_id=best_chunk_id,
            windowed=windowed,
            best_chunk_cited=best_chunk_cited,
        )

    # Reassemble in the original answer order, interleaving computational
    # verdicts back into place.
    verdicts: List[ClaimVerdict] = []
    score_cursor = 0
    for kind, claim in order:
        if kind == "comp":
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim,
                    verdict=VERDICT_COMPUTATIONAL,
                    entailment=0.0,
                    contradiction=0.0,
                    best_chunk_id=None,
                    windowed=False,
                    best_chunk_cited=None,
                )
            )
        else:
            verdicts.append(scored_verdicts[score_cursor])
            score_cursor += 1

    scored_count = len(scorable)
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
        nli_device=_model_device(resolved),
        reason=None,
        computational_count=computational_count,
        filtered_count=filtered_count,
    )


__all__ = [
    "ClaimVerdict",
    "GroundednessReport",
    "split_claims",
    "score_groundedness",
    "VERDICT_ENTAILED",
    "VERDICT_UNSUPPORTED",
    "VERDICT_CONTRADICTED",
    "VERDICT_COMPUTATIONAL",
    "SCORER_VERSION",
    "THRESHOLDS_PROVENANCE_DEFAULTS",
    "ENV_REQUIRE_EMBEDDINGS",
]
