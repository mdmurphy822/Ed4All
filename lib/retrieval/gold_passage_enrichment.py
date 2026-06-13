"""Gold passage-enrichment PROPOSAL builder + fail-closed promote (2026-06).

Companion to ``plans/gold-enrichment-2026-06.md`` §2. The frozen gold set pins
essentially ONE ``relevant_passages`` entry per question (113/125 questions
carry exactly one passage), so every additional valid citation the tutor emits
scores as a ``citation_precision`` MISS even though the metric already credits
ANY pinned passage (``relevance`` primary OR supporting) and the v1.1 schema
already supports a multi-entry list. This module PROPOSES the genuinely-
supporting-but-unpinned passages so the relevant set becomes honest — a
DATA-authoring change, never a schema or scorer change.

The one genuinely-new piece is the **chunk-vs-expected_key_points support
scorer** (:func:`chunk_supports_question`): investigation confirmed no existing
pass scores a candidate chunk against a question's ``expected_key_points``. It
REUSES the exact primitives the eval already trusts to decide "a claim is
supported by a passage" — :func:`citation_attribution._shingle_support`
(4-shingle containment) and :func:`citation_attribution._token_coverage`,
projected through the anchor-equivalent :func:`citation_attribution._normalize`,
at the SAME floors (:data:`DEFAULT_MIN_OVERLAP` ``0.25`` shingle OR
:data:`TOKEN_COVERAGE_FLOOR` ``0.80`` token-coverage). When an NLI classifier is
injected (groundedness enabled), a chunk ALSO supports a key point if the chunk
entails it at the SAME :data:`NLI_ADD_ENTAILMENT_FLOOR` ``0.70`` the
groundedness/NLI-ADD arms use — so "supporting" here means exactly what the
groundedness scorer means by "supported". NO new thresholds are invented.

The premise/hypothesis orientation is the INVERSE of
:func:`answer_scoring.score_key_point_coverage`: there the ANSWER is the premise
and the key point the hypothesis ("does the answer support this point?"); here
the CANDIDATE CHUNK is the premise and the key point the hypothesis ("does this
chunk support this point?"). A chunk SUPPORTS the question iff it supports >= 1
expected key point.

Process (deterministic, operator-reviewed — proposal-then-promote, never a
one-shot mutation of a frozen set):

  1. **Candidate sourcing** (:func:`gather_candidates`) — per gold question, the
     candidate pool is retrieval top-k UNION the chunks the tutor actually cited
     (read from a per-question ``cited_chunk_ids`` map when an eval persisted
     it, else retrieval only). These are the reachable passages — never the
     whole corpus (plan risk §5: "don't pin chunks the tutor would never
     retrieve").
  2. **Scoring + acceptance** (:func:`propose_enrichment_for_question`) — score
     each candidate; KEEP those that support >= 1 key point AND are not the
     existing pinned primary AND are not already pinned (any relevance) AND are
     not a near-duplicate of an already-pinned passage (reuses
     :func:`gold_authoring.prescreen_candidate`'s ``NEAR_DUPLICATE`` shingle
     >= 0.80 check semantics).
  3. **Quote/anchor construction** — pick the verbatim >= 40-char sentence from
     the chunk that carries the supported key point, verify with
     :func:`gold_set._quote_in_chunk`, reject as ambiguous when
     :func:`gold_set._quote_chunk_match_count` > 3 (mirrors
     ``prescreen_candidate``'s ``QUOTE_AMBIGUOUS``), and build the passage entry
     with the canonical :func:`gold_authoring._anchor_for` — never reimplemented.
  4. **Emit a PROPOSAL** (:func:`generate_enrichment_proposal`) — write
     ``retrieval_eval/gold_passage_enrichment_proposal.json`` (per question:
     existing_primary + proposed_additional_passages[] each carrying its support
     score / supported key point / quote / rationale). NEVER mutates
     ``gold_set.json``.
  5. **Promote, fail-closed** (:func:`promote_enrichment`) — a SEPARATE explicit
     operator step. Appends ONLY operator-accepted ``supporting`` passages to
     the matching questions' ``relevant_passages``, honors a frozen-set unfreeze
     gate (mirrors ``gold-repin --unfreeze-for-repin``), then re-validates
     :func:`gold_set.validate_gold_set` FAIL-CLOSED (unknown chunk id /
     quote-not-in-chunk / content-sha mismatch all block the write) and re-pins
     the chunkset sha. NEVER touches/replaces the single existing primary.

ACCEPT CONVENTION (promote)
---------------------------
A proposed passage is accepted when its ``status`` is edited from ``"proposed"``
to ``"accepted"``. ``promote_enrichment`` applies exactly the passages that are
``status: accepted``; everything else is left in the proposal untouched. One
lifecycle field, one source of truth — mirroring ``gold_authoring``'s
``authoring.status`` accept convention.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.retrieval._text import shingle_containment
from lib.retrieval.citation_attribution import (
    DEFAULT_MIN_OVERLAP,
    NLI_ADD_ENTAILMENT_FLOOR,
    TOKEN_COVERAGE_FLOOR,
    _normalize,
    _shingle_support,
    _token_coverage,
)
from lib.retrieval.gold_authoring import _anchor_for
from lib.retrieval.gold_coverage import _population_of_chunk
from lib.retrieval.gold_set import (
    GOLD_SET_FILENAME,
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    _quote_chunk_match_count,
    _quote_in_chunk,
    critical_issues,
    validate_gold_set,
)
from lib.utils import sha256_file

GOLD_PASSAGE_ENRICHMENT_FILENAME = "gold_passage_enrichment_proposal.json"

# Minimum verbatim quote length (mirrors gold_authoring._MIN_QUOTE_CHARS /
# the schema's anchor.text_quote minLength). A proposed supporting passage that
# can't yield a >= 40-char verbatim quote is not anchorable and is skipped.
_MIN_QUOTE_CHARS = 40
# Quote-ambiguity cap (mirrors gold_set._AMBIGUOUS_QUOTE_MAX_CHUNKS /
# gold_authoring._AMBIGUOUS_QUOTE_MAX_CHUNKS): a quote contained in > 3 chunks
# inflates eval credit, so reject it rather than pin an ambiguous anchor.
_AMBIGUOUS_QUOTE_MAX_CHUNKS = 3
# Near-duplicate threshold (mirrors gold_authoring._NEAR_DUP_CONTAINMENT /
# _NEAR_DUP_SHINGLE_SIZE): a candidate whose text is >= 0.80 shingle-contained
# in an already-pinned passage's chunk is a duplicate, not new support.
_NEAR_DUP_CONTAINMENT = 0.80
_NEAR_DUP_SHINGLE_SIZE = 4
# Default retrieval candidate pool depth (the chunks the tutor sees; plan §2.2).
_DEFAULT_TOP_K = 10

# Proposed-passage lifecycle status values (the promote accept convention).
_STATUS_PROPOSED = "proposed"
_STATUS_ACCEPTED = "accepted"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")


# --------------------------------------------------------------------------- #
# The one genuinely-new piece: chunk-vs-expected_key_points support scorer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KeyPointSupport:
    """How one candidate chunk supports ONE expected key point.

    ``supported`` is the OR of the deterministic shingle/coverage arms and (when
    an NLI classifier is injected) the entailment arm — at the SAME floors the
    eval trusts. ``arm`` names the arm that fired (``shingle`` /
    ``token_coverage`` / ``nli`` / ``none``).
    """

    key_point: str
    supported: bool
    shingle: float = 0.0
    coverage: float = 0.0
    entailment: Optional[float] = None
    arm: str = "none"


@dataclass(frozen=True)
class ChunkSupport:
    """A candidate chunk's support verdict for one question.

    ``supports`` is True iff the chunk supports >= 1 expected key point.
    ``best`` is the strongest (highest shingle, then coverage) supported
    key-point verdict — the one whose quote should anchor the proposed passage.
    """

    chunk_id: str
    supports: bool
    per_key_point: List[KeyPointSupport] = field(default_factory=list)
    best: Optional[KeyPointSupport] = None

    @property
    def supported_key_point_count(self) -> int:
        return sum(1 for kp in self.per_key_point if kp.supported)


def _nli_entailment(nli: Any, premise: str, hypothesis: str) -> Optional[float]:
    """Best-effort single-pair entailment via the injected NLI classifier.

    Returns ``None`` (degrade) when the classifier can't score the pair — the
    deterministic arms still decide the verdict, never a fabricated 0.0. Mirrors
    :func:`answer_scoring._nli_entailment`.
    """
    try:
        scores = nli.score_batch(pairs=[(premise, hypothesis)])
    except Exception:  # noqa: BLE001 — NLI failure degrades to deterministic arms
        return None
    if not scores:
        return None
    return float(getattr(scores[0], "entailment", 0.0) or 0.0)


def chunk_supports_question(
    chunk_text: Optional[str],
    key_points: Sequence[str],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    token_coverage_floor: float = TOKEN_COVERAGE_FLOOR,
    nli: Optional[Any] = None,
    entailment_floor: float = NLI_ADD_ENTAILMENT_FLOOR,
) -> ChunkSupport:
    """Score a candidate chunk against a question's ``expected_key_points``.

    The chunk text is the PREMISE, each key point the HYPOTHESIS ("does this
    chunk support this expected point?") — the INVERSE of
    :func:`answer_scoring.score_key_point_coverage`. A key point is supported iff
    (deterministic arms, REUSED from ``citation_attribution`` at the SAME floors
    the eval trusts):

      * ``shingle4(key_point, chunk) >= min_overlap``  OR
      * ``token_coverage(key_point, chunk) >= token_coverage_floor``

    OR, when ``nli`` is supplied (groundedness enabled), the chunk entails the
    key point at ``entailment >= entailment_floor``
    (:data:`NLI_ADD_ENTAILMENT_FLOOR`, the SAME floor the NLI-ADD / groundedness
    arms use). The chunk SUPPORTS the question iff it supports >= 1 key point.

    No key points (the v1.0 case) → ``supports=False`` (nothing to support
    against — enrichment requires expected_key_points to score honestly).
    """
    points = [str(k) for k in (key_points or []) if str(k).strip()]
    chunk_norm = _normalize(chunk_text or "")
    nli_used = nli is not None

    per_kp: List[KeyPointSupport] = []
    best: Optional[KeyPointSupport] = None
    best_metric = -1.0
    for kp in points:
        kp_norm = _normalize(kp)
        shingle = _shingle_support(kp_norm, chunk_norm)
        coverage = _token_coverage(kp_norm, chunk_norm)

        arm = "none"
        supported = False
        if shingle >= min_overlap:
            supported, arm = True, "shingle"
        elif coverage >= token_coverage_floor:
            supported, arm = True, "token_coverage"

        entailment: Optional[float] = None
        if nli_used:
            # Premise = chunk, hypothesis = key point. Only consult NLI when the
            # deterministic arms did not already cover it (cheaper + can only
            # flip false->true, never the reverse — mirrors answer_scoring).
            ent = _nli_entailment(nli, chunk_text or "", kp)
            if ent is not None:
                entailment = ent
                if not supported and ent >= float(entailment_floor):
                    supported, arm = True, "nli"

        verdict = KeyPointSupport(
            key_point=kp,
            supported=supported,
            shingle=round(shingle, 6),
            coverage=round(coverage, 6),
            entailment=None if entailment is None else round(entailment, 6),
            arm=arm,
        )
        per_kp.append(verdict)
        if supported:
            # Strongest supported key point (highest shingle, tie-broken by
            # coverage) carries the anchor quote.
            metric = shingle + coverage / 1000.0
            if metric > best_metric:
                best_metric = metric
                best = verdict

    return ChunkSupport(
        chunk_id="",  # set by the caller (the scorer is chunk-text-only here)
        supports=best is not None,
        per_key_point=per_kp,
        best=best,
    )


# --------------------------------------------------------------------------- #
# Quote / anchor construction (reuses gold_set + gold_authoring primitives)
# --------------------------------------------------------------------------- #


def _content_tokens(text: str) -> List[str]:
    """Lowercased alphabetic content tokens (len >= 2). Local helper so the
    quote-ranking overlap measures the same token shape the support arms do."""
    return [t.lower() for t in _CONTENT_TOKEN_RE.findall(text or "")]


def _candidate_quotes(chunk_text: str, key_point: str) -> List[str]:
    """Verbatim >= 40-char substrings of ``chunk_text`` that may carry the key
    point, strongest first.

    Picks original-cased SENTENCES (so the quote is verbatim-contained and reads
    naturally), ranked by content-token overlap with the key point. Never
    fabricates: returns only true substrings of the chunk text. The whole-chunk
    text is appended as a last-resort quote so a single-sentence chunk still
    anchors.
    """
    kp_tokens = set(_content_tokens(key_point))
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(chunk_text or "") if s.strip()]
    scored: List[Tuple[int, int, str]] = []
    for idx, sent in enumerate(sentences):
        if len(sent) < _MIN_QUOTE_CHARS:
            continue
        overlap = len(kp_tokens & set(_content_tokens(sent)))
        # Higher overlap first; stable by source order on ties (idx ASC).
        scored.append((-overlap, idx, sent))
    scored.sort()
    quotes = [s for _, _, s in scored]
    whole = (chunk_text or "").strip()
    if len(whole) >= _MIN_QUOTE_CHARS and whole not in quotes:
        quotes.append(whole)
    return quotes


def _pick_anchor_quote(
    chunk: Dict[str, Any],
    key_point: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Pick a verbatim >= 40-char quote from ``chunk`` carrying ``key_point``.

    Verifies normalized containment via :func:`gold_set._quote_in_chunk` and
    rejects an ambiguous quote (contained in > 3 chunks of the pinned set, via
    :func:`gold_set._quote_chunk_match_count`) — mirrors
    ``gold_authoring.prescreen_candidate``'s QUOTE_TOO_SHORT / QUOTE_NOT_IN_CHUNK
    / QUOTE_AMBIGUOUS screen. Returns the first quote that passes, or ``None``.
    """
    chunk_text = str(chunk.get("text") or "")
    for quote in _candidate_quotes(chunk_text, key_point):
        if len(quote.strip()) < _MIN_QUOTE_CHARS:
            continue
        if not _quote_in_chunk(quote, chunk):
            continue
        if _quote_chunk_match_count(quote, chunks_by_id) > _AMBIGUOUS_QUOTE_MAX_CHUNKS:
            continue
        return quote
    return None


# --------------------------------------------------------------------------- #
# Candidate sourcing (retrieval top-k UNION eval-observed citations)
# --------------------------------------------------------------------------- #


def gather_candidates(
    question: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    retrieve_fn: Optional[Any] = None,
    libv2_root: Optional[Path] = None,
    course_slug: Optional[str] = None,
    engine: str = "lexical",
    top_k: int = _DEFAULT_TOP_K,
    cited_chunk_ids: Sequence[str] = (),
) -> List[str]:
    """Build the candidate-chunk pool for one gold question (plan §2.2).

    The pool = retrieval top-k (the chunks the tutor SEES) UNION the chunks the
    tutor ACTUALLY cited on this question (``cited_chunk_ids``, read from a
    persisted per-question map when available, else empty). NEVER the whole
    corpus — only reachable passages (plan risk §5). Ids absent from the pinned
    chunkset are dropped (can't score / anchor them). Returns chunk ids in a
    deterministic order (retrieval rank first, then any cited-only ids in sorted
    order). When ``retrieve_fn`` is ``None`` the pool is the cited ids only (the
    eval-observed-citations arm alone).
    """
    qtext = str(question.get("question_text") or "")
    ordered: List[str] = []
    seen: set = set()

    if retrieve_fn is not None and libv2_root is not None and qtext:
        try:
            results = retrieve_fn(
                libv2_root, qtext, course_slug=course_slug,
                engine=engine, limit=top_k,
            )
        except Exception:  # noqa: BLE001 — a retrieval failure degrades to cited-only
            results = []
        for r in results or []:
            cid = str(getattr(r, "chunk_id", "") or "")
            if cid and cid in chunks_by_id and cid not in seen:
                seen.add(cid)
                ordered.append(cid)

    for cid in sorted({str(c) for c in cited_chunk_ids if str(c)}):
        if cid in chunks_by_id and cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    return ordered


# --------------------------------------------------------------------------- #
# Per-question proposal
# --------------------------------------------------------------------------- #


@dataclass
class ProposedPassage:
    """One proposed ADDITIONAL supporting passage for a gold question."""

    chunk_id: str
    relevance: str  # always "supporting" — primary is never proposed/replaced
    anchor: Dict[str, Any]
    supported_key_point: str
    support_arm: str
    support_shingle: float
    support_coverage: float
    support_entailment: Optional[float]
    page_label: str
    population: str
    rationale: str
    status: str = _STATUS_PROPOSED

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "relevance": self.relevance,
            "anchor": self.anchor,
            "supported_key_point": self.supported_key_point,
            "support": {
                "arm": self.support_arm,
                "shingle": round(self.support_shingle, 6),
                "coverage": round(self.support_coverage, 6),
            },
            "page_label": self.page_label,
            "population": self.population,
            "rationale": self.rationale,
            "status": self.status,
        }
        if self.support_entailment is not None:
            out["support"]["entailment"] = round(self.support_entailment, 6)
        return out


@dataclass
class QuestionEnrichment:
    """All proposed additional passages for one gold question."""

    question_id: str
    question_text: str
    existing_primary: Optional[Dict[str, Any]]
    proposed_additional_passages: List[ProposedPassage] = field(default_factory=list)
    n_candidates_scored: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "existing_primary": self.existing_primary,
            "n_candidates_scored": self.n_candidates_scored,
            "n_proposed": len(self.proposed_additional_passages),
            "proposed_additional_passages": [
                p.to_dict() for p in self.proposed_additional_passages
            ],
        }


def _existing_primary(question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The single relevance:primary passage (the one enrichment NEVER touches)."""
    passages = question.get("relevant_passages") or []
    for p in passages:
        if isinstance(p, dict) and p.get("relevance") == "primary":
            return p
    return passages[0] if passages and isinstance(passages[0], dict) else None


def _pinned_chunk_ids(question: Dict[str, Any]) -> set:
    """Every chunk id already pinned on the question (any relevance)."""
    out: set = set()
    for p in question.get("relevant_passages") or []:
        if isinstance(p, dict) and isinstance(p.get("chunk_id"), str):
            out.add(p["chunk_id"])
    return out


def _is_near_duplicate(candidate_text: str, pinned_texts: Sequence[str]) -> bool:
    """True when the candidate chunk text is >= 0.80 shingle-contained in any
    already-pinned passage's chunk text (mirrors prescreen_candidate's
    NEAR_DUPLICATE) — already-pinned support, not NEW support."""
    for pinned in pinned_texts:
        if not pinned:
            continue
        if shingle_containment(
            candidate_text, pinned, shingle_size=_NEAR_DUP_SHINGLE_SIZE
        ) >= _NEAR_DUP_CONTAINMENT:
            return True
    return False


def _page_label(chunk: Dict[str, Any]) -> str:
    """Best-effort human page label (section_heading / item_path) for review."""
    source = chunk.get("source") or {}
    for key in ("section_heading", "module_title", "item_path"):
        val = source.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def propose_enrichment_for_question(
    question: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    candidate_ids: Sequence[str],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    token_coverage_floor: float = TOKEN_COVERAGE_FLOOR,
    nli: Optional[Any] = None,
    entailment_floor: float = NLI_ADD_ENTAILMENT_FLOOR,
) -> QuestionEnrichment:
    """Score ``candidate_ids`` against the question's expected_key_points and
    propose the genuinely-supporting-but-unpinned ones as ADDITIONAL supporting
    passages.

    Acceptance (each clause REUSED, no new thresholds): a candidate is proposed
    iff it (1) supports >= 1 expected key point (:func:`chunk_supports_question`
    at the eval's floors), (2) is not already pinned on this question (primary or
    supporting — the existing primary is NEVER touched/replaced), (3) is not a
    near-duplicate of an already-pinned passage's chunk text, and (4) yields a
    verbatim >= 40-char, non-ambiguous anchor quote carrying the supported key
    point. Pure + deterministic for a fixed (question, chunks, candidate_ids).
    """
    qid = str(question.get("question_id") or "")
    qtext = str(question.get("question_text") or "")
    key_points = question.get("expected_key_points") or []
    primary = _existing_primary(question)
    pinned_ids = _pinned_chunk_ids(question)
    pinned_texts = [
        str((chunks_by_id.get(cid) or {}).get("text") or "")
        for cid in pinned_ids
        if cid in chunks_by_id
    ]

    enrichment = QuestionEnrichment(
        question_id=qid,
        question_text=qtext,
        existing_primary=primary,
        n_candidates_scored=0,
    )

    # No expected_key_points → nothing to score honestly against (the scorer
    # requires hypotheses); leave the question un-enriched.
    if not [str(k) for k in key_points if str(k).strip()]:
        return enrichment

    seen_proposed: set = set()
    for cid in candidate_ids:
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        # (2) skip already-pinned chunks (this is where the primary is excluded).
        if cid in pinned_ids or cid in seen_proposed:
            continue
        enrichment.n_candidates_scored += 1

        support = chunk_supports_question(
            str(chunk.get("text") or ""),
            key_points,
            min_overlap=min_overlap,
            token_coverage_floor=token_coverage_floor,
            nli=nli,
            entailment_floor=entailment_floor,
        )
        if not support.supports or support.best is None:
            continue
        # (3) near-duplicate of an already-pinned passage → not new support.
        if _is_near_duplicate(str(chunk.get("text") or ""), pinned_texts):
            continue
        # (4) verbatim, non-ambiguous anchor quote for the supported key point.
        quote = _pick_anchor_quote(chunk, support.best.key_point, chunks_by_id)
        if quote is None:
            continue

        anchor = _anchor_for(chunk, quote)
        rationale = (
            f"chunk {cid} supports expected key point "
            f"{support.best.key_point!r} via arm={support.best.arm} "
            f"(shingle={support.best.shingle:.3f} >= {min_overlap} OR "
            f"coverage={support.best.coverage:.3f} >= {token_coverage_floor}"
            + (
                f" OR nli={support.best.entailment:.3f} >= {entailment_floor}"
                if support.best.entailment is not None else ""
            )
            + f"); {support.supported_key_point_count} of "
            f"{len(support.per_key_point)} key point(s) supported. "
            f"Proposed as an ADDITIONAL supporting passage — the existing primary "
            f"is untouched. Reuses the eval's citation-attribution support floors "
            f"(no new threshold)."
        )
        enrichment.proposed_additional_passages.append(
            ProposedPassage(
                chunk_id=cid,
                relevance="supporting",
                anchor=anchor,
                supported_key_point=support.best.key_point,
                support_arm=support.best.arm,
                support_shingle=support.best.shingle,
                support_coverage=support.best.coverage,
                support_entailment=support.best.entailment,
                page_label=_page_label(chunk),
                population=_population_of_chunk(chunk),
                rationale=rationale,
            )
        )
        seen_proposed.add(cid)

    return enrichment


# --------------------------------------------------------------------------- #
# Proposal doc + I/O entry
# --------------------------------------------------------------------------- #


def build_enrichment_proposals(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    retrieve_fn: Optional[Any] = None,
    libv2_root: Optional[Path] = None,
    engine: str = "lexical",
    top_k: int = _DEFAULT_TOP_K,
    cited_by_question: Optional[Dict[str, Sequence[str]]] = None,
    nli: Optional[Any] = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    token_coverage_floor: float = TOKEN_COVERAGE_FLOOR,
    entailment_floor: float = NLI_ADD_ENTAILMENT_FLOOR,
) -> List[QuestionEnrichment]:
    """Build a per-question enrichment proposal over every gold question.

    Questions whose candidate pool yields no genuinely-supporting-but-unpinned
    chunk produce an entry with an empty ``proposed_additional_passages`` (so the
    operator sees the question WAS scored, not silently skipped). Pure relative
    to ``retrieve_fn`` (deterministic when ``retrieve_fn`` is deterministic /
    absent)."""
    course_slug = gold.get("course_slug")
    cited_map = cited_by_question or {}
    out: List[QuestionEnrichment] = []
    for q in gold.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id")
        if not isinstance(qid, str):
            continue
        candidate_ids = gather_candidates(
            q, chunks_by_id,
            retrieve_fn=retrieve_fn, libv2_root=libv2_root,
            course_slug=course_slug, engine=engine, top_k=top_k,
            cited_chunk_ids=cited_map.get(qid, ()),
        )
        enrichment = propose_enrichment_for_question(
            q, chunks_by_id, candidate_ids,
            min_overlap=min_overlap,
            token_coverage_floor=token_coverage_floor,
            nli=nli,
            entailment_floor=entailment_floor,
        )
        out.append(enrichment)
    return out


def build_enrichment_doc(
    course_slug: str,
    enrichments: List[QuestionEnrichment],
    *,
    method: str,
) -> Dict[str, Any]:
    """Assemble the ``gold_passage_enrichment_proposal.json`` wrapper doc."""
    n_with_proposals = sum(1 for e in enrichments if e.proposed_additional_passages)
    n_proposed = sum(len(e.proposed_additional_passages) for e in enrichments)
    return {
        "schema_version": "1.0",
        "course_slug": course_slug,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_operator_note": (
            "PROPOSAL ONLY — this artifact is never auto-applied to gold_set.json "
            "(the frozen set is the canonical eval pin). Each proposed passage is "
            "an ADDITIONAL relevance:supporting entry; the single existing primary "
            "is NEVER touched or replaced. Review each proposal, set its status to "
            "'accepted' for the ones to apply, then run a separate explicit "
            "`gold-enrich-passages --promote` step (fail-closed validate + frozen "
            "unfreeze gate)."
        ),
        "method": method,
        "support_floors": {
            "min_overlap_shingle": DEFAULT_MIN_OVERLAP,
            "token_coverage_floor": TOKEN_COVERAGE_FLOOR,
            "nli_entailment_floor": NLI_ADD_ENTAILMENT_FLOOR,
            "_note": "reused verbatim from citation_attribution — no new thresholds.",
        },
        "n_questions": len(enrichments),
        "n_questions_with_proposals": n_with_proposals,
        "n_proposed_passages": n_proposed,
        "questions": [e.to_dict() for e in enrichments],
    }


def generate_enrichment_proposal(
    course_dir: Path,
    *,
    retrieve_fn: Optional[Any] = None,
    libv2_root: Optional[Path] = None,
    engine: str = "lexical",
    top_k: int = _DEFAULT_TOP_K,
    cited_by_question: Optional[Dict[str, Sequence[str]]] = None,
    nli: Optional[Any] = None,
    write: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end: load the gold set + its pinned chunkset, build per-question
    enrichment proposals, and (unless ``write=False``) write
    ``retrieval_eval/gold_passage_enrichment_proposal.json``. Returns
    ``(doc, written_path)``.

    NEVER mutates ``gold_set.json``. Raises ``FileNotFoundError`` when the gold
    set / pinned chunkset is absent.
    """
    from lib.retrieval.gold_set import load_gold_set

    course_dir = Path(course_dir)
    gold, _ = load_gold_set(course_dir, verify=False)
    if not gold:
        raise FileNotFoundError(f"no gold set found under {course_dir}.")
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path")
    if not chunks_rel:
        raise FileNotFoundError(
            f"no chunkset pin in gold set at {course_dir}; cannot source candidates."
        )
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise FileNotFoundError(f"pinned chunkset not found at {chunks_path}.")
    chunks_by_id = _load_chunks_by_id(chunks_path)

    enrichments = build_enrichment_proposals(
        gold, chunks_by_id,
        retrieve_fn=retrieve_fn,
        libv2_root=libv2_root if libv2_root is not None else course_dir,
        engine=engine, top_k=top_k,
        cited_by_question=cited_by_question, nli=nli,
    )
    method = "retrieval+attribution" if retrieve_fn is not None else "cited+attribution"
    if nli is not None:
        method += "+nli"
    doc = build_enrichment_doc(
        gold.get("course_slug", course_dir.name), enrichments, method=method
    )

    out_path: Optional[Path] = None
    if write:
        out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / GOLD_PASSAGE_ENRICHMENT_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path


# --------------------------------------------------------------------------- #
# Promote (fail-closed) — a SEPARATE explicit operator step
# --------------------------------------------------------------------------- #


class GoldEnrichError(Exception):
    """Raised on a fail-closed enrichment-promotion refusal."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass
class EnrichPromoteReport:
    """Outcome of an enrichment-promotion pass."""

    course_slug: str
    generated_at: str = ""
    applied: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    frozen: bool = False

    @property
    def n_applied(self) -> int:
        return len(self.applied)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_slug": self.course_slug,
            "generated_at": self.generated_at,
            "applied": self.applied,
            "applied_count": self.n_applied,
            "skipped": self.skipped,
            "frozen": self.frozen,
        }


def _accepted_passage(passage: Dict[str, Any]) -> bool:
    """ACCEPT CONVENTION: status == 'accepted' (operator-edited from
    'proposed')."""
    return str(passage.get("status") or "").strip().lower() == _STATUS_ACCEPTED


def apply_enrichment_into_gold(
    gold: Dict[str, Any],
    proposal_doc: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    chunks_sha256: str,
    *,
    course_slug: str,
) -> Tuple[Dict[str, Any], EnrichPromoteReport]:
    """Append operator-ACCEPTED supporting passages into a gold doc (pure).

    Appends each ``status: accepted`` proposed passage (``relevance:supporting``)
    to the matching question's ``relevant_passages`` — NEVER touching/replacing
    the existing primary, NEVER adding a duplicate chunk id already pinned on the
    question — then RE-VALIDATES the merged doc FAIL-CLOSED
    (:func:`gold_set.validate_gold_set`). A critical issue refuses the WHOLE
    promotion (anti-silent-degradation). The chunkset sha pin is refreshed.
    Returns ``(new_gold, report)``; the input is never mutated (deep copy).
    """
    gold = json.loads(json.dumps(gold))
    report = EnrichPromoteReport(
        course_slug=course_slug,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        frozen=bool(gold.get("frozen")),
    )
    by_id: Dict[str, Dict[str, Any]] = {}
    for q in gold.get("questions") or []:
        if isinstance(q, dict) and isinstance(q.get("question_id"), str):
            by_id[q["question_id"]] = q

    for entry in proposal_doc.get("questions") or []:
        if not isinstance(entry, dict):
            continue
        qid = entry.get("question_id")
        question = by_id.get(qid) if isinstance(qid, str) else None
        for passage in entry.get("proposed_additional_passages") or []:
            if not isinstance(passage, dict):
                continue
            label = {"question_id": qid, "chunk_id": passage.get("chunk_id")}
            if not _accepted_passage(passage):
                report.skipped.append({**label, "reason": "not_accepted"})
                continue
            if question is None:
                report.skipped.append({**label, "reason": "unknown_question_id"})
                continue
            cid = passage.get("chunk_id")
            if not isinstance(cid, str) or not cid:
                report.skipped.append({**label, "reason": "missing_chunk_id"})
                continue
            already = _pinned_chunk_ids(question)
            if cid in already:
                report.skipped.append({**label, "reason": "already_pinned"})
                continue
            anchor = passage.get("anchor")
            if not isinstance(anchor, dict):
                report.skipped.append({**label, "reason": "missing_anchor"})
                continue
            # ADD-ONLY: append a supporting passage; never mutate the primary.
            question.setdefault("relevant_passages", []).append(
                {
                    "chunk_id": cid,
                    "relevance": "supporting",
                    "anchor": dict(anchor),
                }
            )
            report.applied.append({**label, "relevance": "supporting"})

    # Refresh the chunkset sha pin (records the live bytes the additions resolve
    # against).
    if isinstance(gold.get("chunkset"), dict):
        gold["chunkset"]["chunks_sha256"] = chunks_sha256

    # Re-validate fail-closed; a critical issue refuses the WHOLE promotion.
    issues = validate_gold_set(gold, chunks_by_id, chunks_sha256)
    crit = critical_issues(issues)
    if crit:
        raise GoldEnrichError(
            "GOLD_ENRICH_VALIDATION_FAILED",
            "; ".join(f"[{i.code}] {i.message}" for i in crit[:5]),
        )
    return gold, report


def promote_enrichment(
    course_dir: Path,
    *,
    unfreeze_for_enrich: bool = False,
    dry_run: bool = False,
) -> Tuple[EnrichPromoteReport, Optional[Path]]:
    """I/O entry: apply operator-accepted passages from
    ``gold_passage_enrichment_proposal.json`` into ``gold_set.json``,
    re-validate FAIL-CLOSED, and (unless ``dry_run``) write the updated gold set
    + a promote-report sidecar.

    Refuses a frozen gold set unless ``unfreeze_for_enrich=True`` (mirrors
    ``gold-repin --unfreeze-for-repin``): a frozen set is the canonical eval pin,
    so adding passages silently would break comparability. Raises
    :class:`GoldEnrichError` on a fail-closed refusal.
    """
    course_dir = Path(course_dir)
    eval_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
    gold_path = eval_dir / GOLD_SET_FILENAME
    proposal_path = eval_dir / GOLD_PASSAGE_ENRICHMENT_FILENAME
    if not gold_path.is_file():
        raise GoldEnrichError("GOLD_SET_NOT_FOUND", f"no gold set at {gold_path}")
    if not proposal_path.is_file():
        raise GoldEnrichError(
            "GOLD_ENRICH_PROPOSAL_NOT_FOUND",
            f"no enrichment proposal at {proposal_path}; run "
            f"`gold-enrich-passages` (proposal) first.",
        )
    try:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        proposal_doc = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldEnrichError("INVALID_JSON", str(exc)) from exc

    if gold.get("frozen") and not unfreeze_for_enrich:
        raise GoldEnrichError(
            "GOLD_SET_FROZEN",
            "gold set is frozen; pass unfreeze_for_enrich=True "
            "(`--unfreeze-for-enrich`) to apply passage enrichment.",
        )

    course_slug = gold.get("course_slug") or course_dir.name
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path")
    if not chunks_rel:
        raise GoldEnrichError("NO_CHUNKSET_PIN", "gold set has no chunkset pin.")
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise GoldEnrichError(
            "GOLD_SET_CHUNKSET_NOT_FOUND",
            f"pinned chunkset not found at {chunks_path}.",
        )
    chunks_sha = sha256_file(chunks_path)
    chunks_by_id = _load_chunks_by_id(chunks_path)

    new_gold, report = apply_enrichment_into_gold(
        gold, proposal_doc, chunks_by_id, chunks_sha, course_slug=course_slug,
    )

    if dry_run:
        return report, None
    gold_path.write_text(json.dumps(new_gold, indent=2) + "\n", encoding="utf-8")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = eval_dir / f"gold_passage_enrichment_report_{ts}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report, gold_path


__all__ = [
    "GOLD_PASSAGE_ENRICHMENT_FILENAME",
    "KeyPointSupport",
    "ChunkSupport",
    "ProposedPassage",
    "QuestionEnrichment",
    "EnrichPromoteReport",
    "GoldEnrichError",
    "chunk_supports_question",
    "gather_candidates",
    "propose_enrichment_for_question",
    "build_enrichment_proposals",
    "build_enrichment_doc",
    "generate_enrichment_proposal",
    "apply_enrichment_into_gold",
    "promote_enrichment",
]
