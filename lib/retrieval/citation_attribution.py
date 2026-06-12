"""Claim-level citation attribution for the grounded-answer path (2026-06).

Pure, deterministic, stdlib-only. Given an answer's text and the passages the
model could have cited (the gate-eligible retrieved set), this module decides,
per (claim, passage), whether the passage *supports* the claim — and rolls that
up into an :class:`AttributionReport` that drives TWO answer-composition-time
refinements without ever changing an answer verdict:

  1. **Prune** a model-cited citation that supports zero claims (claim-less).
     The motivating case (2026-06-11): "what is rag?" answered with 2 citations
     where every sentence traced to citation #2 (definitional) and citation #1
     (a RAG-evaluation chunk that rode in on the lexical leg) supported nothing.

  2. **Add** an UNCITED gate-eligible passage when it is the clear best
     supporter of some claim and out-supports the kept citations. The second
     motivating case (2026-06-11): "what is the five step problem solving
     strategy ..." where the definitional supporter was in the model's passage
     set (rank 7/8) and the answer's steps clearly derived from it, but the
     model cited the three top-ranked passages (which merely *mention* the
     strategy) and not the supporter. Attribution therefore runs over ALL
     gate-eligible passages, not just the model-cited ones.

Philosophy (mirrors ``plans/claimless-citation-pruning-2026-06.md`` §2.1):

* The claim unit is :func:`groundedness.split_claims` (reused, NOT
  re-implemented) so attribution counts claims exactly the way
  ``unsupported_claim_rate`` does.
* Both sides are projected through the citation anchor's entity-symmetric,
  whitespace-collapsed, lowercased normalization (``_normalize_for_match``
  reused) so a chunk that kept raw HTML entities compares equal to an answer
  whose entities the model already decoded.
* A passage **supports** a claim iff its phrase-level shingle containment
  (``_text.shingle_containment``, size 4 — claims are ~10-25 tokens; the
  anchor's 8 scores 0 on any light paraphrase) is at or above ``min_overlap``,
  OR its content-token coverage is at or above a fixed secondary floor (catches
  reordered / lightly-inflected paraphrase the shingle arm misses). Threshold
  rule, NOT argmax: a claim supported by two passages keeps both.

Addition is held to a deliberately HIGHER, shingle-only bar than pruning (see
:data:`ADD_MIN_SHINGLE`): an added citation must (a) clear the high shingle
floor on some claim — token-coverage paraphrase evidence alone never qualifies
an addition — AND (b) out-support the strongest kept citation on that claim, so
additions are precision-first (the failure cost of a spurious added citation is
a fabricated-looking source, higher than the cost of leaving a weak supporter
uncredited).

No LLM, no network, no torch import (the ``_claim_support_thresholds`` regexes
are constants-only). Independently testable; reusable by the offline eval.
"""

from __future__ import annotations

import html as _html
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from lib.retrieval._text import normalize_ws, shingle_containment
from lib.retrieval.groundedness import split_claims
from lib.validators.pair._claim_support_thresholds import _CONTENT_TOKEN_RE

__all__ = [
    "ENV_CITATION_PRUNE",
    "ENV_PRUNE_MIN_OVERLAP",
    "ENV_ADD_MIN_SHINGLE",
    "ENV_NLI_ADD",
    "MODE_OFF",
    "MODE_SHADOW",
    "MODE_ON",
    "DEFAULT_MODE",
    "DEFAULT_NLI_ADD_MODE",
    "DEFAULT_MIN_OVERLAP",
    "TOKEN_COVERAGE_FLOOR",
    "ADD_MIN_SHINGLE",
    "ATTRIBUTION_SHINGLE_SIZE",
    "MAX_ADDED_CITATIONS",
    "NLI_ADD_ENTAILMENT_FLOOR",
    "NLI_ADD_TOKEN_COVERAGE_FLOOR",
    "NLI_ADD_MAX_ADDED_CITATIONS",
    "CitationSupport",
    "AttributionReport",
    "resolve_prune_mode",
    "resolve_min_overlap",
    "resolve_add_min_shingle",
    "resolve_nli_add_mode",
    "claim_numeric_literals",
    "claim_numerics_present",
    "passage_token_coverage",
    "attribute_citations",
]


# --------------------------------------------------------------------------- #
# Flag + knob names / defaults (plan §2.7)
# --------------------------------------------------------------------------- #

#: Three-valued rollout flag governing BOTH prune and add.
ENV_CITATION_PRUNE = "ED4ALL_ANSWER_CITATION_PRUNE"
#: Float knob → the support ``min_overlap`` used for the PRUNE decision.
ENV_PRUNE_MIN_OVERLAP = "ED4ALL_ANSWER_PRUNE_MIN_OVERLAP"
#: Float knob → the shingle floor used for the ADD decision (see
#: :data:`ADD_MIN_SHINGLE` for the provenance of the knob promotion).
ENV_ADD_MIN_SHINGLE = "ED4ALL_ANSWER_ADD_MIN_SHINGLE"
#: Three-valued (off|shadow|on) rollout flag for the NLI-based citation-ADD arm
#: (the entailment-driven successor to the shingle ADD arm, which the 2026-06-12
#: under-citing investigation measured unsalvageable — paraphrase answers never
#: quote, so the shingle bar fires zero adds even at 0.10). See
#: :func:`resolve_nli_add_mode` and :data:`DEFAULT_NLI_ADD_MODE`.
ENV_NLI_ADD = "ED4ALL_ANSWER_NLI_ADD"

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ON = "on"
#: Code default is ``shadow`` (per plan §2.7): compute attribution, emit the
#: capture + warnings + excerpts, but mutate nothing. The deployable posture
#: (docker-compose) sets ``on`` explicitly.
DEFAULT_MODE = MODE_SHADOW

#: Code default for the NLI-ADD arm is ``off`` — deliberately NOT ``shadow``.
#: Unlike the lexical attribution arm (pure stdlib, zero marginal cost), the
#: NLI-ADD criterion requires the ~750 MB DeBERTa NLI model; defaulting to
#: ``shadow`` would drag that load onto the default answer path. The deployable
#: posture (docker-compose) sets ``shadow`` explicitly so eval/aggregation can
#: measure would-adds where the model is already resident; ``on`` ships dark.
DEFAULT_NLI_ADD_MODE = MODE_OFF

#: THE prune-side calibration knob. Deliberately low: the failure cost of
#: over-pruning a legitimate paraphrase citation is higher than that of keeping
#: a marginal one — precision of the PRUNE decision over its recall.
DEFAULT_MIN_OVERLAP = 0.25

#: Fixed secondary support floor: content-token coverage of the claim by the
#: passage. Catches reordered / lightly-inflected paraphrase the shingle arm
#: misses. Used by the PRUNE arm only (additions are shingle-only).
TOKEN_COVERAGE_FLOOR = 0.80

#: The ADD bar (DEFAULT). An addition is a high-precision, rarely-fired
#: refinement — its bar should not move with the prune knob, which calibrates a
#: different (keep-vs-drop) decision. 0.50 is twice the prune default; combined
#: with the relative "out-support the kept citations" gate (§2.2) it keeps
#: additions conservative.
#:
#: PROVENANCE (single-course union-corpus calibration basis, 2026-06-12): the
#: P5 attribution-calibration pass over the frozen gold set (77 gold questions,
#: lexical attribution arm, bge-large + 7B) found the median cited-citation
#: shingle sits at 0.000 — a full 0.500 BELOW this ADD bar — so additions
#: essentially never fire on this corpus. That measurement WARRANTS a tunable
#: knob (so an operator on a paraphrase-light corpus can lower the bar to let
#: additions fire) without changing the conservative default: the artifact
#: argues for the KNOB, NOT for a lower default (lowering it globally would
#: re-clutter the source list the prune arm just cleaned). The knob is therefore
#: :data:`ENV_ADD_MIN_SHINGLE` (``ED4ALL_ANSWER_ADD_MIN_SHINGLE``), resolved by
#: :func:`resolve_add_min_shingle`; the default below is unchanged.
ADD_MIN_SHINGLE = 0.50

#: Shingle size for the claim-vs-passage phrase arm. 4 (not the anchor's 8)
#: because claims are short (~10-25 tokens); 8-shingles score 0 on any light
#: paraphrase.
ATTRIBUTION_SHINGLE_SIZE = 4

#: Cap on additions per answer (the addition is a high-precision tail; an
#: unbounded add would re-clutter the source list the prune arm just cleaned).
MAX_ADDED_CITATIONS = 2


# --------------------------------------------------------------------------- #
# NLI-ADD composite criterion (2026-06-12 under-citing investigation)
# --------------------------------------------------------------------------- #
#
# The shingle ADD arm above is measured unsalvageable on paraphrase-heavy
# answers (median cited shingle 0.000; zero adds even at bar 0.10 — see
# /tmp/under_citing_findings.json and :data:`ADD_MIN_SHINGLE`). Meanwhile
# entailed-uncited claims are real (55 on the clean run) and some are
# zero-citation answers that deserve an honest source. A hand-judged sample
# found raw NLI ADD at ent>=0.70 fabricates provenance ~43% of the time, but a
# COMPOSITE criterion separated 4/4 genuine adds from 3/3 false adds:
#
#   1. NLI entailment >= NLI_ADD_ENTAILMENT_FLOOR (windowed, scorer-v2
#      semantics from ``groundedness.score_groundedness``), AND
#   2. claim<->chunk content-token coverage >= NLI_ADD_TOKEN_COVERAGE_FLOOR
#      (reuses :func:`_token_coverage`), AND
#   3. every numeric literal in the claim text appears in the chunk text — NLI
#      is blind to numbers (it entailed "the price of one pen is $1.50" at 0.90
#      against a chunk with no $1.50), AND
#   4. the chunk anchors/resolves like any added citation must (enforced at the
#      call site in ``grounded_answer``), AND
#   5. <= NLI_ADD_MAX_ADDED_CITATIONS adds per answer.

#: NLI entailment floor for the ADD criterion. The hand-judged genuine adds all
#: sat at or above 0.735; 0.75 cleanly separates them from the 0.76-0.77 false
#: positives ONLY in combination with the coverage + numeric legs (NLI alone
#: cannot — the false adds entail at 0.76/0.77/0.90).
NLI_ADD_ENTAILMENT_FLOOR = 0.75

#: Content-token coverage floor (claim tokens present in the chunk). The
#: 2026-06-12 hand sample: genuine adds covered 0.667-1.0; the false adds that
#: NLI passed sat at 0.40-0.545. 0.65 separates them. Reuses the same
#: ``_content_tokens`` machinery the lexical arm's :data:`TOKEN_COVERAGE_FLOOR`
#: uses (so math/symbol fragments exert no support pressure).
NLI_ADD_TOKEN_COVERAGE_FLOOR = 0.65

#: Cap on NLI-driven additions per answer (mirrors :data:`MAX_ADDED_CITATIONS`).
NLI_ADD_MAX_ADDED_CITATIONS = 2


# --------------------------------------------------------------------------- #
# Flag / knob resolution
# --------------------------------------------------------------------------- #


def resolve_prune_mode(explicit: Optional[str] = None) -> str:
    """Resolve the three-valued prune/add mode.

    Precedence: explicit arg > ``ED4ALL_ANSWER_CITATION_PRUNE`` env >
    :data:`DEFAULT_MODE`. An unrecognized value falls back to the default (a
    typo must never silently disable the audit trail by leaving it neither on
    nor shadow — it lands on the documented ``shadow`` default).
    """
    raw = explicit if explicit is not None else os.environ.get(ENV_CITATION_PRUNE)
    if raw is None:
        return DEFAULT_MODE
    val = str(raw).strip().lower()
    if val in (MODE_OFF, MODE_SHADOW, MODE_ON):
        return val
    return DEFAULT_MODE


def resolve_min_overlap(explicit: Optional[float] = None) -> float:
    """Resolve the prune-side ``min_overlap`` knob.

    Precedence: explicit arg > ``ED4ALL_ANSWER_PRUNE_MIN_OVERLAP`` env >
    :data:`DEFAULT_MIN_OVERLAP`. Garbage / out-of-range (not in ``[0, 1]``)
    values fall back to the default (a misconfigured knob must never disable
    support scoring).
    """
    if explicit is not None:
        cand = explicit
    else:
        raw = os.environ.get(ENV_PRUNE_MIN_OVERLAP)
        if raw is None or str(raw).strip() == "":
            return DEFAULT_MIN_OVERLAP
        try:
            cand = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_MIN_OVERLAP
    try:
        cand = float(cand)
    except (TypeError, ValueError):
        return DEFAULT_MIN_OVERLAP
    if 0.0 <= cand <= 1.0:
        return cand
    return DEFAULT_MIN_OVERLAP


def resolve_add_min_shingle(explicit: Optional[float] = None) -> float:
    """Resolve the ADD-side shingle floor knob.

    Mirrors :func:`resolve_min_overlap`. Precedence: explicit arg >
    ``ED4ALL_ANSWER_ADD_MIN_SHINGLE`` env > :data:`ADD_MIN_SHINGLE`. Garbage /
    out-of-range (not in ``[0, 1]``) values fall back to the default (a
    misconfigured knob must never disable the ADD bar — an addition is a
    precision-first refinement and the conservative default is the safe value).

    PROVENANCE (single-course union-corpus calibration basis, 2026-06-12): the
    knob exists because the P5 attribution-calibration measured a median cited
    shingle of 0.000 (0.500 below the bar), warranting tunability without moving
    the conservative default — see :data:`ADD_MIN_SHINGLE`.
    """
    if explicit is not None:
        cand = explicit
    else:
        raw = os.environ.get(ENV_ADD_MIN_SHINGLE)
        if raw is None or str(raw).strip() == "":
            return ADD_MIN_SHINGLE
        try:
            cand = float(raw)
        except (TypeError, ValueError):
            return ADD_MIN_SHINGLE
    try:
        cand = float(cand)
    except (TypeError, ValueError):
        return ADD_MIN_SHINGLE
    if 0.0 <= cand <= 1.0:
        return cand
    return ADD_MIN_SHINGLE


def resolve_nli_add_mode(explicit: Optional[str] = None) -> str:
    """Resolve the three-valued NLI-ADD mode (off|shadow|on).

    Mirrors :func:`resolve_prune_mode` EXCEPT the default is :data:`MODE_OFF`,
    not ``shadow`` (the NLI model is a ~750 MB lazy load; the audit trail must
    not drag it onto the default answer path — see :data:`DEFAULT_NLI_ADD_MODE`).
    Precedence: explicit arg > ``ED4ALL_ANSWER_NLI_ADD`` env >
    :data:`DEFAULT_NLI_ADD_MODE`. An unrecognized / garbage value falls back to
    the default (a typo lands on the safe ``off``, never silently ``on``).
    """
    raw = explicit if explicit is not None else os.environ.get(ENV_NLI_ADD)
    if raw is None:
        return DEFAULT_NLI_ADD_MODE
    val = str(raw).strip().lower()
    if val in (MODE_OFF, MODE_SHADOW, MODE_ON):
        return val
    return DEFAULT_NLI_ADD_MODE


# --------------------------------------------------------------------------- #
# Numeric-literal guard (NLI is blind to numbers — composite-criterion leg 3)
# --------------------------------------------------------------------------- #

#: A numeric literal in claim/chunk text: an optional leading currency symbol,
#: a digit run that may carry thousands-separator commas, an optional decimal
#: fraction, and an optional trailing percent sign. Matches ``$1.50``, ``27.78``,
#: ``1,234``, ``50%``, ``3``. Comparison is on the *normalized* token (commas
#: stripped, currency/percent affixes dropped) so ``$1.50`` in a claim matches a
#: bare ``1.50`` in the chunk, while ``27.78`` does NOT match ``27.8``.
_NUMERIC_LITERAL_RE = re.compile(r"[$£€]?\d[\d,]*(?:\.\d+)?%?")


def _normalize_numeric(token: str) -> str:
    """Strip currency/percent affixes and thousands commas → comparable form.

    ``$1.50`` → ``1.50``; ``1,234`` → ``1234``; ``50%`` → ``50``; ``27.78`` →
    ``27.78`` (NOT equal to ``27.8`` — decimals are compared literally, never
    rounded). Returns ``""`` when nothing numeric survives.
    """
    body = token.strip().lstrip("$£€").rstrip("%")
    body = body.replace(",", "")
    return body


def claim_numeric_literals(claim_text: Optional[str]) -> List[str]:
    """Normalized numeric literals appearing in ``claim_text`` (order-preserving).

    Empty list when the claim carries no numerics (the leg-3 check then passes
    vacuously — there is nothing for the chunk to fail to contain).
    """
    out: List[str] = []
    for raw in _NUMERIC_LITERAL_RE.findall(claim_text or ""):
        norm = _normalize_numeric(raw)
        if norm:
            out.append(norm)
    return out


def claim_numerics_present(
    claim_text: Optional[str], chunk_text: Optional[str]
) -> bool:
    """True iff EVERY numeric literal in the claim also appears in the chunk.

    NLI is blind to numbers (the audit found it entailed "the price of one pen
    is $1.50" at 0.90 against a chunk with no ``$1.50``), so an NLI-ADD candidate
    must independently clear this guard. A claim with no numerics passes
    vacuously. Comparison is on the normalized form (commas stripped,
    currency/percent affixes dropped) so ``$1.50`` matches a bare ``1.50``;
    ``27.78`` does NOT match ``27.8`` (decimals are literal, never rounded).
    """
    claim_nums = claim_numeric_literals(claim_text)
    if not claim_nums:
        return True
    chunk_nums = set(claim_numeric_literals(chunk_text))
    return all(n in chunk_nums for n in claim_nums)


def passage_token_coverage(claim_text: Optional[str], passage_text: Optional[str]) -> float:
    """Public wrapper over :func:`_token_coverage` (normalizes both sides first).

    Lets the NLI-ADD criterion in ``grounded_answer`` reuse the EXACT
    content-token coverage machinery the lexical attribution arm uses, so the two
    arms never diverge on what "coverage" means.
    """
    return _token_coverage(_normalize(claim_text or ""), _normalize(passage_text or ""))


# --------------------------------------------------------------------------- #
# Normalization (reuses the anchor's projection so the two never diverge)
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    """Entity-symmetric, whitespace-collapsed projection (anchor-equivalent).

    Mirrors ``citation_anchor._normalize_for_match`` (``html.unescape`` +
    whitespace collapse). ``shingle_containment`` lowercases internally; the
    token-coverage arm lowercases its tokens, so this stays case-preserving and
    both arms agree on casing.
    """
    return normalize_ws(_html.unescape(text or ""))


def _content_tokens(text: str) -> List[str]:
    """Lowercased content tokens per the groundedness ``_CONTENT_TOKEN_RE``.

    ``_CONTENT_TOKEN_RE`` matches alphabetic tokens of length >= 2 — the same
    definition the pair-claim-support validator and groundedness use, so
    math/symbol-dense fragments exert no support pressure (consistent with how
    the eval ignores them).
    """
    return [t.lower() for t in _CONTENT_TOKEN_RE.findall(text or "")]


# --------------------------------------------------------------------------- #
# Per-(claim, passage) support scoring
# --------------------------------------------------------------------------- #


def _shingle_support(claim_norm: str, passage_norm: str) -> float:
    """Phrase-level shingle containment of the claim in the passage."""
    return shingle_containment(
        claim_norm, passage_norm, shingle_size=ATTRIBUTION_SHINGLE_SIZE
    )


def _token_coverage(claim_norm: str, passage_norm: str) -> float:
    """Fraction of the claim's content tokens present in the passage."""
    claim_tokens = _content_tokens(claim_norm)
    if not claim_tokens:
        return 0.0
    passage_tokens = set(_content_tokens(passage_norm))
    hit = sum(1 for t in claim_tokens if t in passage_tokens)
    return hit / len(claim_tokens)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _passage_sentences(passage_text: str) -> List[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(passage_text or "") if s.strip()]


def _best_supporting_sentence(
    claim_norm: str, passage_text: str
) -> Optional[str]:
    """The passage sentence sharing the most content tokens with the claim.

    Returns the original-cased sentence (whitespace-normalized), never
    fabricated: ``None`` when the passage has no sentence overlapping the claim.
    Used to populate ``supporting_excerpt`` (the UX defect fix, plan §2.6).
    """
    claim_tokens = set(_content_tokens(claim_norm))
    if not claim_tokens:
        return None
    best: Optional[str] = None
    best_overlap = 0
    for sentence in _passage_sentences(passage_text):
        sent_tokens = set(_content_tokens(sentence))
        overlap = len(claim_tokens & sent_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = sentence.strip()
    if best_overlap == 0:
        return None
    return normalize_ws(best)


# --------------------------------------------------------------------------- #
# Report shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CitationSupport:
    """Attribution outcome for ONE passage against the answer's claims.

    ``rank`` is the passage's retrieval rank (0-based) over the gate-eligible
    set — the ordering tiebreaker (plan extension: final citations sort by
    ``(supported_claim_count desc, rank asc)``). ``best_shingle`` /
    ``best_coverage`` are the maxima over the claims this passage supports (or
    over ALL claims when it supports none — the prune-side audit signal).
    ``best_claim_index`` + ``supporting_excerpt`` describe the strongest
    attributed (claim, passage-sentence) pair.
    """

    chunk_id: str
    rank: int
    cited: bool  # was this passage in the model's citation list?
    supported_claim_indexes: List[int] = field(default_factory=list)
    best_shingle: float = 0.0
    best_coverage: float = 0.0
    best_claim_index: Optional[int] = None
    supporting_excerpt: Optional[str] = None

    @property
    def supported_claim_count(self) -> int:
        return len(self.supported_claim_indexes)

    @property
    def is_claimless(self) -> bool:
        return not self.supported_claim_indexes


@dataclass(frozen=True)
class AttributionReport:
    """Per-answer attribution over the gate-eligible passages.

    ``supports`` is keyed by chunk_id. ``n_claims`` is the count of scorable
    claims (``split_claims``); ``claim_supporters`` maps each claim index to the
    chunk_ids that support it (records ALL supporters so the excerpt picker never
    collapses shared support, plan risk §3).
    """

    n_claims: int
    min_overlap: float
    supports: Dict[str, CitationSupport] = field(default_factory=dict)
    claim_supporters: Dict[int, List[str]] = field(default_factory=dict)

    def cited_claimless_ids(self) -> List[str]:
        """Model-cited chunk_ids that support zero claims (prune candidates)."""
        return [
            cid
            for cid, s in self.supports.items()
            if s.cited and s.is_claimless
        ]

    def cited_supported_ids(self) -> List[str]:
        """Model-cited chunk_ids that support >= 1 claim (always kept)."""
        return [
            cid for cid, s in self.supports.items() if s.cited and not s.is_claimless
        ]


# --------------------------------------------------------------------------- #
# THE attribution pass
# --------------------------------------------------------------------------- #


def attribute_citations(
    answer_text: Optional[str],
    passages: Sequence[object],
    cited_ids: Sequence[str],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> AttributionReport:
    """Attribute the answer's claims across ALL gate-eligible ``passages``.

    ``passages`` are duck-typed (``.chunk_id`` + ``.text``) in retrieval-rank
    order; ``cited_ids`` is the model's citation list (a subset). A passage
    supports a claim iff ``shingle4 >= min_overlap`` OR
    ``token_coverage >= TOKEN_COVERAGE_FLOOR``. No claims (empty/short answer) →
    every passage scores zero claims (the caller treats that as the
    "no scorable claims" no-op).
    """
    claims = split_claims(answer_text or "")
    claim_norms = [_normalize(c) for c in claims]
    cited_set = set(cited_ids)

    supports: Dict[str, CitationSupport] = {}
    claim_supporters: Dict[int, List[str]] = {i: [] for i in range(len(claims))}

    for rank, passage in enumerate(passages):
        chunk_id = str(getattr(passage, "chunk_id", "") or "")
        if not chunk_id or chunk_id in supports:
            # First occurrence wins the rank; a duplicate id (shouldn't happen
            # in a gate-eligible set) is ignored rather than overwriting.
            continue
        passage_text = str(getattr(passage, "text", "") or "")
        passage_norm = _normalize(passage_text)

        supported: List[int] = []
        best_shingle = 0.0
        best_coverage = 0.0
        best_claim_index: Optional[int] = None
        best_metric = -1.0

        for ci, claim_norm in enumerate(claim_norms):
            shingle = _shingle_support(claim_norm, passage_norm)
            coverage = _token_coverage(claim_norm, passage_norm)
            if shingle > best_shingle:
                best_shingle = shingle
            if coverage > best_coverage:
                best_coverage = coverage
            is_support = shingle >= min_overlap or coverage >= TOKEN_COVERAGE_FLOOR
            if is_support:
                supported.append(ci)
                claim_supporters[ci].append(chunk_id)
            # Track the strongest attributed claim for the excerpt (prefer the
            # claim with the highest shingle, tie-broken by coverage). Only
            # claims this passage SUPPORTS are excerpt-eligible.
            if is_support:
                metric = shingle + coverage / 1000.0
                if metric > best_metric:
                    best_metric = metric
                    best_claim_index = ci

        excerpt = (
            _best_supporting_sentence(claim_norms[best_claim_index], passage_text)
            if best_claim_index is not None
            else None
        )
        supports[chunk_id] = CitationSupport(
            chunk_id=chunk_id,
            rank=rank,
            cited=chunk_id in cited_set,
            supported_claim_indexes=supported,
            best_shingle=round(best_shingle, 6),
            best_coverage=round(best_coverage, 6),
            best_claim_index=best_claim_index,
            supporting_excerpt=excerpt,
        )

    return AttributionReport(
        n_claims=len(claims),
        min_overlap=float(min_overlap),
        supports=supports,
        claim_supporters=claim_supporters,
    )
