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

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

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

#: W1.8 opt-in gate for the computational-sentence numeric-grounding check.
#: Default OFF => the historical exemption is byte-identical (a computational
#: claim is skipped from NLI + excluded from the denominator, and NO
#: ``computational_numeric_check`` block is written). When truthy, the scorer
#: ADDITIONALLY verifies that each computational sentence's numeric literals
#: appear in a cited chunk and records the ungrounded ones as a WARNING-ONLY
#: diagnostic that never changes ``groundedness_rate`` / ``scored_count`` /
#: any verdict (a textbook premise still cannot ENTAIL a computation, so the
#: exemption from the NLI denominator is preserved — this only surfaces
#: fabricated numeric literals that the exemption otherwise hides).
ENV_GROUNDEDNESS_COMPUTATIONAL = "ED4ALL_GROUNDEDNESS_COMPUTATIONAL"


def resolve_groundedness_computational(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_GROUNDEDNESS_COMPUTATIONAL`` is set to a truthy value.

    Parse-with-fallback: only the explicit truthy tokens enable the check;
    anything else (unset / falsey / garbage) leaves it OFF so the historical
    computational exemption is byte-identical.
    """
    src = env if env is not None else os.environ
    return str(src.get(ENV_GROUNDEDNESS_COMPUTATIONAL, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


#: Opt-in stage-1 redesign (``ED4ALL_GROUNDEDNESS_S1_TOPK=<K>``): instead of
#: the whole-passage grid (every ~700-word chunk premise is silently TRUNCATED
#: at the NLI's 512-token window AND costs an O(L²) max-length forward —
#: measured 28-45 pairs/s on uniform-long drains), score each claim against
#: its top-K sentence-WINDOWS selected by embedding cosine (windows via the
#: same ``_sentence_windows`` stage-2 rescue uses; MiniLM embeddings, one
#: batched encode per call). Short premises ≈15× cheaper per pair; K bounds
#: the per-claim pair count. Claims still below the floor take a WIDENED
#: preselected rescue (next ``_S1_TOPK_RESCUE_MULT``×K windows by the SAME
#: cached cosine scores) as the safety net for cosine misses. NOT
#: verdict-bitwise-identical to the legacy grid: windows can see text the
#: 512-token truncation dropped (recovering real support) and premise
#: composition changes low-order scores — adopt only after an A/B against a
#: serial-baseline sidecar quantifies the verdict diffs. ``0`` / unset /
#: garbage → legacy grid, byte-identical.
ENV_GROUNDEDNESS_S1_TOPK = "ED4ALL_GROUNDEDNESS_S1_TOPK"

#: Per-passage cap on stage-1 windows (a pathological mega-chunk cannot
#: explode the window pool; 32 windows ≈ 96 sentences at stride 2).
_S1_TOPK_MAX_WINDOWS_PER_PASSAGE = 32

#: Rescue widening multiplier: below-floor claims re-score against the next
#: ``mult × K`` cosine-ranked windows beyond the stage-1 top-K.
_S1_TOPK_RESCUE_MULT = 4


#: Serializes the top-K preselect's ``encode_batch`` calls: SentenceTransformer
#: is not concurrency-certified, and the cross-block gate path
#: (``ED4ALL_NLI_CROSSBLOCK``) calls ``score_groundedness`` from 16 worker
#: threads. NLI forwards dominate the gate's wall-clock, so serializing the
#: ~100ms embed step costs little.
_S1_EMBED_LOCK = __import__("threading").Lock()

#: Process-lifetime window-vector memo for the top-K preselect, keyed by
#: sha256(window_text) → UNIT-NORMALIZED float32 vector. The provenance
#: resolver expands section-level refs to ALL section chunks, so co-located
#: blocks share (nearly) the same ~10k-window premise pool — without this
#: memo every block re-embeds that pool from scratch (``encode_batch``
#: deliberately bypasses the disk EmbeddingCache; measured as a recurring
#: 416-batch MiniLM encode per block = the dominant per-block cost). Unique
#: windows are bounded by the corpus (~chunks × windows-per-chunk ≈ 10-20k
#: rows ≈ tens of MB). Guarded by ``_S1_EMBED_LOCK``.
_S1_WIN_VEC_CACHE: Dict[str, Any] = {}


def resolve_groundedness_s1_topk(env: Optional[Dict[str, str]] = None) -> int:
    """Resolved K for the top-K windowed stage-1 (0 = off, the default).

    Parse-with-fallback: unset / empty / garbage / non-positive → ``0`` (legacy
    whole-passage grid, byte-identical).
    """
    src = env if env is not None else os.environ
    raw = str(src.get(ENV_GROUNDEDNESS_S1_TOPK, "")).strip()
    if not raw:
        return 0
    try:
        iv = int(raw)
    except (TypeError, ValueError):
        return 0
    return iv if iv > 0 else 0


#: Opt-in FRONTIER stage-1 (``ED4ALL_GROUNDEDNESS_FRONTIER``): instead of the
#: whole-passage grid (every claim × every ~700-word passage in one ~2.3M-pair
#: forward-pass storm — the ~12-hour block-prose-entailment monster), score each
#: claim against its passage pool in a deterministic PRIORITY ORDER (direct
#: citations → lexical anchor → cosine, see :mod:`lib.retrieval.candidate_order`)
#: and RETIRE a claim the instant a premise entails it. Premises are the SAME
#: whole-passage texts as the legacy grid (not windows), so the verdict rule is
#: unchanged and parity is exact:
#:   * an entailed claim retires early — legacy fires iff max-ent ≥ floor, the
#:     frontier retires iff *some* ent ≥ floor: identical verdict (only the
#:     recorded best-passage metadata may differ — first-clearing vs argmax);
#:   * a non-entailed claim exhausts its WHOLE pool, so its argmax (ent, con)
#:     (with the legacy passage-index tiebreak) is bitwise-identical to the grid
#:     ⇒ identical contradicted / unsupported verdict + identical stage-2 rescue.
#: Value: only the entailing prefix of each claim's pool is scored (early-exit),
#: while the dispatcher still coalesces every round's contributions across
#: claims / blocks / threads into one batched forward pass. Unset / falsey /
#: garbage → legacy grid, byte-identical. When BOTH this and
#: ``ED4ALL_GROUNDEDNESS_S1_TOPK`` are set, FRONTIER wins and top-K is ignored
#: (one-time warning) — the two are mutually-exclusive stage-1 redesigns.
ENV_GROUNDEDNESS_FRONTIER = "ED4ALL_GROUNDEDNESS_FRONTIER"

#: Per-round frontier width: each still-active claim contributes its next
#: ``W`` candidate passages, and ALL active claims' contributions are combined
#: into ONE ``score_batch`` call so the dispatcher coalesces across claims. A
#: wider ``W`` means fewer rounds (less scheduling overhead) but scores more
#: past a claim's first entailing premise before it can retire; ``4`` balances
#: the two. Env ``ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH``.
ENV_GROUNDEDNESS_FRONTIER_WIDTH = "ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH"

#: Default per-round frontier width (see :data:`ENV_GROUNDEDNESS_FRONTIER_WIDTH`).
_FRONTIER_WIDTH_DEFAULT = 4

#: One-time guard so the FRONTIER-supersedes-top-K warning logs once per process.
_FRONTIER_TOPK_WARNED = [False]


def resolve_groundedness_frontier(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_GROUNDEDNESS_FRONTIER`` is set to a truthy value.

    Parse-with-fallback: only the explicit truthy tokens enable the frontier
    stage-1; anything else (unset / falsey / garbage) leaves it OFF so the
    legacy whole-passage grid is byte-identical.
    """
    src = env if env is not None else os.environ
    return str(src.get(ENV_GROUNDEDNESS_FRONTIER, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_groundedness_frontier_width(env: Optional[Dict[str, str]] = None) -> int:
    """Resolved per-round frontier width (default :data:`_FRONTIER_WIDTH_DEFAULT`).

    Parse-with-fallback: unset / empty / garbage / non-positive → the module
    default so the frontier always makes forward progress.
    """
    src = env if env is not None else os.environ
    raw = str(src.get(ENV_GROUNDEDNESS_FRONTIER_WIDTH, "")).strip()
    if not raw:
        return _FRONTIER_WIDTH_DEFAULT
    try:
        iv = int(raw)
    except (TypeError, ValueError):
        return _FRONTIER_WIDTH_DEFAULT
    return iv if iv > 0 else _FRONTIER_WIDTH_DEFAULT


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
    #: W1.8 additive — the computational-sentence numeric-grounding diagnostic.
    #: ``None`` (and OMITTED from :meth:`to_dict`) unless
    #: ``ED4ALL_GROUNDEDNESS_COMPUTATIONAL`` is on AND ≥1 computational claim was
    #: seen, so the report is byte-identical on the default (flag-off) path.
    #: WARNING-ONLY — never feeds any verdict / rate / count.
    computational_numeric_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
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
        # W1.8 additive: present ONLY when the opt-in numeric check produced a
        # block (flag on + ≥1 computational claim). Omitted otherwise so the
        # default-off report stays byte-identical.
        if self.computational_numeric_check is not None:
            out["computational_numeric_check"] = self.computational_numeric_check
        return out


# --------------------------------------------------------------------------- #
# Abbreviation guard for the sentence splitter (2026-07-01 CO-13 fix)
# --------------------------------------------------------------------------- #
# The canonical ``_SENTENCE_SPLIT_RE`` (``(?<=[.!?])\s+``) splits after ANY
# period+whitespace, so an abbreviation mid-sentence beheads the claim:
# "…in both U.S. and metric systems…" → fragment 1 "…both U.S." + fragment 2
# "and metric systems to solve real-world problems." — and NLI then scored the
# beheaded fragment CONTRADICTED against a perfectly-supporting chunk (the
# observed OBJECTIVE_CONTRADICTED false failure on CO-13). The guard below
# re-joins fragments the regex wrongly split at a known abbreviation.
#
# Conservative three-tier rule (merge precision over recall — a wrong merge
# fuses two real sentences into one hypothesis, so ambiguous enders only merge
# when the next fragment could not plausibly start a new sentence):
#   * ALWAYS-merge: titles/latinisms that essentially never end a sentence
#     (e.g., i.e., vs., Dr., Mr., Mrs., Ms., Prof., St.).
#   * NUMERIC-merge: label abbreviations (Fig., No., Eq.) merge only when the
#     next fragment starts with a digit ("Fig. 3", "No. 5").
#   * AMBIGUOUS-merge: enders that CAN legitimately terminate a sentence
#     (dotted uppercase pairs like U.S. / U.K., and etc.) merge only when the
#     next fragment starts lowercase or with a digit ("U.S. and metric…"
#     merges; "…in the U.S. The metric system…" stays split).

_ABBREV_ALWAYS_MERGE_RE = re.compile(
    r"\b(?:e\.g|i\.e|vs|Dr|Mr|Mrs|Ms|Prof|St)\.$"
)
_ABBREV_NUMERIC_MERGE_RE = re.compile(r"\b(?:Fig|No|Eq)\.$")
_ABBREV_AMBIGUOUS_MERGE_RE = re.compile(r"(?:\b[A-Z]\.[A-Z]\.|\betc\.)$")


def _should_merge_abbrev_split(prev: str, nxt: str) -> bool:
    """True when ``prev``/``nxt`` were wrongly split at an abbreviation."""
    if not prev or not nxt:
        return False
    if _ABBREV_ALWAYS_MERGE_RE.search(prev):
        return True
    lead = nxt[0]
    if _ABBREV_NUMERIC_MERGE_RE.search(prev) and lead.isdigit():
        return True
    if _ABBREV_AMBIGUOUS_MERGE_RE.search(prev) and (
        lead.islower() or lead.isdigit()
    ):
        return True
    return False


def _merge_abbreviation_splits(parts: List[str]) -> List[str]:
    """Re-join sentence fragments wrongly split at a known abbreviation.

    Iterates left-to-right, folding a fragment into its predecessor when
    :func:`_should_merge_abbrev_split` fires; chained abbreviations
    ("e.g. U.S. and …") merge naturally because each fold re-checks the
    accumulated predecessor's tail.
    """
    merged: List[str] = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if merged and _should_merge_abbrev_split(merged[-1], piece):
            merged[-1] = merged[-1] + " " + piece
            continue
        merged.append(piece)
    return merged


def split_claims(answer_text: str) -> List[str]:
    """Split an answer into pedagogical claim sentences worth scoring.

    Uses the canonical ``_SENTENCE_SPLIT_RE`` split and the
    ``_MIN_SENTENCE_TOKENS`` content-token floor (both imported from
    ``_claim_support_thresholds``) so groundedness counts sentences the same
    way the pair-claim-support validator does. A sentence is kept only when it
    has at least ``_MIN_SENTENCE_TOKENS`` content tokens (``_CONTENT_TOKEN_RE``
    matches alphabetic tokens of length ≥ 2), filtering out fragments,
    headings, and bare references.

    Fragments the regex wrongly split at a known abbreviation (U.S., e.g.,
    Dr., Fig. 3, …) are re-joined by :func:`_merge_abbreviation_splits` BEFORE
    the token floor, so an abbreviation mid-sentence no longer beheads a claim
    into an NLI-contradictable fragment (the CO-13 false-CONTRADICTED bug).

    Public, backward-compatible, and reused by ``citation_attribution`` — its
    output is the canonical claim unit there, so the v2 structural-artifact
    filter (:func:`_is_structural_artifact`) is applied only in the
    groundedness wrapper (:func:`_split_claims_for_scoring`), NOT here, to keep
    attribution's claim set unchanged.
    """
    if not answer_text:
        return []
    claims: List[str] = []
    parts = _merge_abbreviation_splits(
        _SENTENCE_SPLIT_RE.split(answer_text.strip())
    )
    for raw in parts:
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


#: A bare numeric literal (integer or decimal). Thousands separators are stripped
#: before matching (``1,000`` → ``1000``) so ``25,000`` and ``25000`` compare
#: equal. Sign/operator context is irrelevant to the membership test.
_NUMERIC_LITERAL_RE = re.compile(r"\d+(?:\.\d+)?")


def _numeric_literals(text: str) -> List[str]:
    """Ordered list of normalized numeric literals in ``text``.

    Commas are removed first (thousands separators) so ``1,024`` matches
    ``1024`` in a source chunk. Returns the raw string forms (e.g. ``"3"``,
    ``"3.14"``) so the ungrounded set is human-readable in the report.
    """
    if not text:
        return []
    return _NUMERIC_LITERAL_RE.findall(text.replace(",", ""))


def _compute_computational_numeric_check(
    comp_claims: List[str],
    passages: Sequence[Any],
    cited_chunk_ids: Optional[set],
) -> Optional[Dict[str, Any]]:
    """W1.8 numeric-grounding diagnostic for computational sentences.

    For each computational claim (exempted from NLI), extract its numeric
    literals and flag the ones NOT present in any CITED chunk's text. When
    ``cited_chunk_ids`` is supplied the pool is restricted to those passages
    (the model's actual citations); otherwise the whole evidence pool is used.

    WARNING-ONLY: the returned block is pure diagnostic — the caller never lets
    it change a verdict / rate / count. Returns ``None`` when there are no
    computational claims (nothing to check → no block, keeps the report
    byte-identical to the pre-W1.8 shape on computation-free answers).
    """
    if not comp_claims:
        return None
    if cited_chunk_ids is not None:
        pool = [p for p in passages if _passage_chunk_id(p) in cited_chunk_ids]
    else:
        pool = list(passages)
    corpus_numerics: set = set()
    for passage in pool:
        corpus_numerics.update(_numeric_literals(_passage_text(passage)))

    claim_rows: List[Dict[str, Any]] = []
    claims_with_ungrounded = 0
    ungrounded_total = 0
    for claim in comp_claims:
        nums = _numeric_literals(claim)
        ungrounded = [n for n in nums if n not in corpus_numerics]
        if ungrounded:
            claims_with_ungrounded += 1
            ungrounded_total += len(ungrounded)
        claim_rows.append(
            {
                "claim_text": claim,
                "numeric_literals": nums,
                "ungrounded_numeric_literals": ungrounded,
                "grounded": not ungrounded,
            }
        )
    return {
        "enabled": True,
        "cited_pool_size": len(pool),
        "computational_claims_checked": len(comp_claims),
        "claims_with_ungrounded_numeric": claims_with_ungrounded,
        "ungrounded_numeric_count": ungrounded_total,
        "claims": claim_rows,
    }


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


def _frontier_embed(
    passage_texts: List[str], claim_texts: List[str]
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """Best-effort MiniLM vectors for the frontier's cosine tiebreaker.

    Returns ``(claim_vecs, passage_vecs)`` (unit-normalized float32), or
    ``(None, None)`` when the embedder / numpy is unavailable — the frontier
    ordering then degrades to lexical-only, which is correct (cosine is only a
    tiebreaker and NEVER excludes a candidate, so verdicts are unaffected).
    Passage vectors are memoized in the shared ``_S1_WIN_VEC_CACHE`` (keyed by
    ``sha256(text)`` — a generic text→vec memo) so co-located blocks that share
    the provenance-expanded pool re-use the encode; ``encode_batch`` runs under
    ``_S1_EMBED_LOCK`` (SentenceTransformer is not concurrency-certified).
    """
    try:
        import hashlib as _hl

        import numpy as _np

        from lib.embedding.sentence_embedder import try_load_embedder as _tle

        emb = _tle()
        if emb is None or not hasattr(emb, "encode_batch"):
            return None, None
        shas = [_hl.sha256(t.encode("utf-8")).hexdigest() for t in passage_texts]
        with _S1_EMBED_LOCK:
            miss = [i for i, h in enumerate(shas) if h not in _S1_WIN_VEC_CACHE]
            if miss:
                uniq: Dict[str, int] = {}
                uniq_texts: List[str] = []
                for i in miss:
                    if shas[i] not in uniq:
                        uniq[shas[i]] = len(uniq_texts)
                        uniq_texts.append(passage_texts[i])
                mv = _np.asarray(emb.encode_batch(uniq_texts), dtype=_np.float32)
                mv = mv / (_np.linalg.norm(mv, axis=1, keepdims=True) + 1e-8)
                for h, row in zip(uniq.keys(), mv):
                    _S1_WIN_VEC_CACHE[h] = row
            pvecs = [_S1_WIN_VEC_CACHE[h] for h in shas]
            cv = _np.asarray(emb.encode_batch(claim_texts), dtype=_np.float32)
        cv = cv / (_np.linalg.norm(cv, axis=1, keepdims=True) + 1e-8)
        return [row for row in cv], pvecs
    except Exception as exc:  # noqa: BLE001 — cosine order is optional
        logger.debug("frontier embed unavailable (%s); lexical-only order.", exc)
        return None, None


def _frontier_stage1_best(
    scorable: List[str],
    passages: Sequence[Any],
    resolved: Any,
    entailment_floor: float,
    *,
    direct_cited_ids: Optional[set],
    width: int,
    stop_check: Optional[Any] = None,
) -> Dict[int, Tuple[float, float, int, bool]]:
    """Frontier stage-1 → the per-claim ``best`` fold (drop-in for the grid fold).

    Each round every still-active claim contributes its next ``width`` candidate
    passages (in :func:`candidate_order.order_passages_for_claim` priority), ALL
    rounds' contributions batch into one ``score_batch`` (the dispatcher
    coalesces across claims / blocks / threads), and a claim RETIRES the instant
    a scored premise reaches ``entailment_floor``. The recorded ``best[ci]`` uses
    the SAME (max-entailment, then min-passage-index) tiebreak as the legacy grid
    fold, so a non-retiring claim — which scores its WHOLE pool — yields a
    ``best`` bitwise-identical to the grid. Retired claims stop early (their
    metadata may pick a different but still-entailing premise; verdict is
    identical). ``best[ci]`` = ``(entailment, contradiction, passage_index,
    windowed=False)``.
    """
    from lib.retrieval.candidate_order import order_passages_for_claim

    passage_texts = [_passage_text(p) for p in passages]
    passage_chunk_ids = [_passage_chunk_id(p) for p in passages]

    claim_vecs, passage_vecs = _frontier_embed(passage_texts, scorable)

    orders: List[List[int]] = []
    for ci, claim in enumerate(scorable):
        cvec = claim_vecs[ci] if claim_vecs is not None else None
        orders.append(
            order_passages_for_claim(
                claim,
                passage_texts,
                passage_chunk_ids,
                direct_cited_ids=direct_cited_ids,
                claim_vec=cvec,
                passage_vecs=passage_vecs,
            )
        )

    cursors = [0] * len(scorable)
    retired: set = set()
    best: Dict[int, Tuple[float, float, int, bool]] = {}
    active = [ci for ci in range(len(scorable)) if orders[ci]]

    while active:
        if stop_check is not None:
            stop_check()
        batch_pairs: List[Tuple[str, str]] = []
        batch_map: List[Tuple[int, int]] = []
        for ci in active:
            order = orders[ci]
            start = cursors[ci]
            contributed = order[start : start + width]
            cursors[ci] = start + len(contributed)
            for pi in contributed:
                batch_pairs.append((passage_texts[pi], scorable[ci]))
                batch_map.append((ci, pi))
        if not batch_pairs:
            break
        scores = resolved.score_batch(pairs=batch_pairs)
        for (ci, pi), score in zip(batch_map, scores):
            ent = float(getattr(score, "entailment", 0.0))
            con = float(getattr(score, "contradiction", 0.0))
            cur = best.get(ci)
            # Legacy grid tiebreak: keep the max-entailment premise, and on an
            # exact entailment tie keep the LOWER passage index (the grid folds
            # passages in ascending index with a strict ``>``). This makes a
            # fully-scored (non-retiring) claim's (ent, con, pi) identical to
            # the grid regardless of the frontier's visit order.
            if cur is None or ent > cur[0] or (ent == cur[0] and pi < cur[2]):
                best[ci] = (ent, con, pi, False)
            if ent >= entailment_floor:
                retired.add(ci)
        active = [
            ci
            for ci in active
            if ci not in retired and cursors[ci] < len(orders[ci])
        ]

    return best


def score_groundedness(
    answer_text: str,
    cited_passages: Sequence[Any],
    *,
    nli: Optional[Any] = None,
    entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
    contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
    cited_chunk_ids: Optional[set] = None,
    capture: Optional[Any] = None,
    split_claims: bool = True,
) -> GroundednessReport:
    """Score each answer sentence against the evidence pool via NLI (v2).

    ``split_claims=False`` (2026-07-01, ObjectiveEntailmentValidator) scores
    the ENTIRE ``answer_text`` as ONE hypothesis — no sentence split, no
    structural-artifact filter. Callers whose unit of meaning is a single
    statement (an LO statement is documented as "a single hypothesis") use
    this so a sentence-splitter artifact (e.g. the abbreviation period in
    "U.S.") can never behead the statement into an NLI-contradictable
    fragment. Default ``True`` preserves the grounded-answer path byte-for-
    byte.

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

    W1.8 (opt-in ``ED4ALL_GROUNDEDNESS_COMPUTATIONAL``, default OFF): computational
    claims stay NLI-exempt + out of the denominator exactly as before, but the
    report ALSO gains a ``computational_numeric_check`` block flagging any numeric
    literal in a computational sentence that is absent from a cited chunk. It is
    WARNING-ONLY — it never changes a verdict / rate / count — and is omitted
    entirely when the flag is off (byte-identical to the historical exemption).
    When ``capture`` is supplied AND the check ran on ≥1 computational claim, one
    ``groundedness_computational_check`` decision is logged.

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
    # ``split_claims=False`` → the whole answer_text is the single claim
    # (single-hypothesis callers, e.g. per-LO entailment).
    if split_claims:
        all_claims, filtered_count = _split_claims_for_scoring(answer_text)
    else:
        _single = (answer_text or "").strip()
        all_claims, filtered_count = ([_single] if _single else []), 0
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

    # W1.8 (opt-in) — verify each computational sentence's numeric literals are
    # present in a cited chunk. Default OFF ⇒ ``comp_check`` stays None and the
    # report is byte-identical to the historical exemption. WARNING-ONLY: this
    # block never feeds a verdict / rate / count below.
    comp_check: Optional[Dict[str, Any]] = None
    if resolve_groundedness_computational():
        comp_claims = [claim for kind, claim in order if kind == "comp"]
        comp_check = _compute_computational_numeric_check(
            comp_claims, passages, cited_chunk_ids
        )
        if comp_check is not None and capture is not None:
            try:
                capture.log_decision(
                    decision_type="groundedness_computational_check",
                    decision=(
                        f"checked {comp_check['computational_claims_checked']} "
                        f"computational claim(s); "
                        f"{comp_check['claims_with_ungrounded_numeric']} carried "
                        f"ungrounded numeric literals"
                    ),
                    rationale=(
                        "ED4ALL_GROUNDEDNESS_COMPUTATIONAL on: verified the "
                        f"numeric literals of "
                        f"{comp_check['computational_claims_checked']} "
                        f"NLI-exempt computational sentence(s) against "
                        f"{comp_check['cited_pool_size']} cited chunk(s); flagged "
                        f"{comp_check['ungrounded_numeric_count']} ungrounded "
                        "numeric literal(s) as a warning (answer verdict + "
                        "groundedness_rate unchanged)."
                    ),
                    is_default=False,
                )
            except Exception:  # noqa: BLE001 — capture must never abort scoring
                pass

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
            computational_numeric_check=comp_check,
        )

    # Stage 1. Two shapes:
    #
    # * LEGACY (default): full (premise, hypothesis) grid — every pool passage
    #   × every scorable claim, whole-chunk premises (truncated at the NLI's
    #   512-token window).
    # * TOP-K WINDOWED (``ED4ALL_GROUNDEDNESS_S1_TOPK=<K>``): each claim vs its
    #   top-K cosine-ranked sentence windows — see ``ENV_GROUNDEDNESS_S1_TOPK``.
    #   Degrades to LEGACY when the embedder / numpy is unavailable (loud via
    #   the debug log, never a silent verdict fabrication).
    # FRONTIER (``ED4ALL_GROUNDEDNESS_FRONTIER``) is a mutually-exclusive
    # stage-1 redesign that supersedes top-K when both are set. It forces
    # ``s1_topk = 0`` (legacy stage-2 rescue path) and computes ``best`` via
    # the priority-ordered early-retiring frontier below.
    frontier = resolve_groundedness_frontier()
    if (
        frontier
        and resolve_groundedness_s1_topk() > 0
        and not _FRONTIER_TOPK_WARNED[0]
    ):
        _FRONTIER_TOPK_WARNED[0] = True
        logger.warning(
            "ED4ALL_GROUNDEDNESS_FRONTIER supersedes ED4ALL_GROUNDEDNESS_S1_TOPK; "
            "top-K ignored this run."
        )
    s1_topk = 0 if frontier else resolve_groundedness_s1_topk()
    s1_sims = None  # (claims × windows) cosine matrix when top-K is active
    s1_win_texts: List[str] = []
    s1_win_passage_idx: List[int] = []
    if s1_topk > 0:
        try:
            import numpy as _np

            from lib.embedding.sentence_embedder import (
                _TUNE_BATCH_SIZE as _s1_tune_bs,
            )
            from lib.embedding.sentence_embedder import (
                resolve_embed_batch_tune as _s1_res_tune,
            )
            from lib.embedding.sentence_embedder import (
                resolve_embed_persist_cache as _s1_res_persist,
            )
            from lib.embedding.sentence_embedder import (
                try_load_embedder as _s1_tle,
            )

            # Builder-C incremental-embed knobs (default off → legacy call):
            # ED4ALL_EMBED_BATCH_TUNE adds batch_size/length_sort;
            # ED4ALL_EMBED_PERSIST_CACHE routes through the disk-persisted
            # encode_batch_cached when the embedder exposes it.
            _s1_kw = (
                {"batch_size": _s1_tune_bs, "length_sort": True}
                if _s1_res_tune()
                else {}
            )
            _s1_persist = _s1_res_persist()

            def _s1_encode(_txts: List[str]) -> Any:
                if _s1_persist:
                    _cached = getattr(_s1_emb, "encode_batch_cached", None)
                    if callable(_cached):
                        return _cached(_txts, **_s1_kw)
                return _s1_emb.encode_batch(_txts, **_s1_kw)

            _s1_emb = _s1_tle()
            if _s1_emb is not None and hasattr(_s1_emb, "encode_batch"):
                for pi, passage in enumerate(passages):
                    p_text = _passage_text(passage)
                    wins = _sentence_windows(p_text)
                    if not wins and p_text.strip():
                        wins = [p_text.strip()]
                    for win in wins[:_S1_TOPK_MAX_WINDOWS_PER_PASSAGE]:
                        s1_win_texts.append(win)
                        s1_win_passage_idx.append(pi)
                if s1_win_texts:
                    import hashlib as _hl

                    shas = [
                        _hl.sha256(t.encode("utf-8")).hexdigest()
                        for t in s1_win_texts
                    ]
                    with _S1_EMBED_LOCK:
                        miss = [
                            i for i, h in enumerate(shas)
                            if h not in _S1_WIN_VEC_CACHE
                        ]
                        if miss:
                            # Dedup within the call (duplicate window texts
                            # from overlapping passages encode once).
                            uniq: Dict[str, int] = {}
                            uniq_texts: List[str] = []
                            for i in miss:
                                if shas[i] not in uniq:
                                    uniq[shas[i]] = len(uniq_texts)
                                    uniq_texts.append(s1_win_texts[i])
                            mv = _np.asarray(
                                _s1_encode(uniq_texts),
                                dtype=_np.float32,
                            )
                            mv = mv / (
                                _np.linalg.norm(mv, axis=1, keepdims=True) + 1e-8
                            )
                            for h, row in zip(uniq.keys(), mv):
                                _S1_WIN_VEC_CACHE[h] = row
                        wv = _np.stack([_S1_WIN_VEC_CACHE[h] for h in shas])
                        cv = _np.asarray(
                            _s1_encode(scorable), dtype=_np.float32
                        )
                    cv = cv / (_np.linalg.norm(cv, axis=1, keepdims=True) + 1e-8)
                    s1_sims = cv @ wv.T
        except Exception as exc:  # noqa: BLE001 — degrade to the legacy grid
            logger.debug("S1 top-K preselect unavailable (%s); legacy grid.", exc)
            s1_sims = None

    # best[ci] = (entailment, contradiction, passage_index, windowed_bool)
    best: Dict[int, Tuple[float, float, int, bool]] = {}
    if frontier:
        # FRONTIER stage-1: priority-ordered, early-retiring scoring. The fold
        # is identical to the grid for every non-retiring claim (whole pool
        # scored, legacy passage-index tiebreak), so contradicted / unsupported
        # verdicts + the stage-2 rescue set are bitwise-identical; entailed
        # claims retire early. See ``_frontier_stage1_best`` /
        # ``ENV_GROUNDEDNESS_FRONTIER``.
        best = _frontier_stage1_best(
            scorable,
            passages,
            resolved,
            entailment_floor,
            direct_cited_ids=cited_chunk_ids,
            width=resolve_groundedness_frontier_width(),
        )
    else:
        pairs: List[Tuple[str, str]] = []
        index_map: List[Tuple[int, int]] = []
        if s1_sims is not None:
            # Top-K windowed stage 1: claim × its K best windows by cosine.
            import numpy as _np

            k = min(s1_topk, len(s1_win_texts))
            s1_rank = _np.argsort(-s1_sims, axis=1)
            for ci, claim in enumerate(scorable):
                for wi in s1_rank[ci, :k]:
                    pairs.append((s1_win_texts[int(wi)], claim))
                    index_map.append((ci, s1_win_passage_idx[int(wi)]))
        else:
            for ci, claim in enumerate(scorable):
                for pi, passage in enumerate(passages):
                    pairs.append((_passage_text(passage), claim))
                    index_map.append((ci, pi))

        scores = resolved.score_batch(pairs=pairs)

        # Fold per-claim: argmax entailment over the claim's passage scores.
        for (ci, pi), score in zip(index_map, scores):
            ent = float(getattr(score, "entailment", 0.0))
            con = float(getattr(score, "contradiction", 0.0))
            cur = best.get(ci)
            if cur is None or ent > cur[0]:
                best[ci] = (ent, con, pi, False)

    # Stage 2: windowed rescue for claims still below the entailment floor.
    unrescued = [ci for ci in range(len(scorable)) if best.get(ci, (0.0,))[0] < entailment_floor]
    if unrescued and s1_sims is not None:
        # Top-K variant rescue: widen to the next ``_S1_TOPK_RESCUE_MULT × K``
        # cosine-ranked windows per claim (same cached similarity matrix — no
        # extra embeds). Catches stage-1 cosine misses without a full window
        # scan; a window that entails replaces the stage-1 result exactly like
        # the legacy rescue (``windowed=True``).
        import numpy as _np

        k = min(s1_topk, len(s1_win_texts))
        k_hi = min(k + _S1_TOPK_RESCUE_MULT * s1_topk, len(s1_win_texts))
        if k_hi > k:
            s1_rank = _np.argsort(-s1_sims, axis=1)
            s2_pairs: List[Tuple[str, str]] = []
            s2_map: List[Tuple[int, int]] = []  # (claim_index, window_index)
            for ci in unrescued:
                claim = scorable[ci]
                for wi in s1_rank[ci, k:k_hi]:
                    s2_pairs.append((s1_win_texts[int(wi)], claim))
                    s2_map.append((ci, int(wi)))
            if s2_pairs:
                s2_scores = resolved.score_batch(pairs=s2_pairs)
                s2_best: Dict[int, Tuple[float, float, int]] = {}
                for (ci, wi), score in zip(s2_map, s2_scores):
                    ent = float(getattr(score, "entailment", 0.0))
                    con = float(getattr(score, "contradiction", 0.0))
                    cur = s2_best.get(ci)
                    if cur is None or ent > cur[0]:
                        s2_best[ci] = (ent, con, wi)
                for ci, (ent, con, wi) in s2_best.items():
                    if ent >= entailment_floor:
                        best[ci] = (ent, con, s1_win_passage_idx[wi], True)
    elif unrescued:
        # LEGACY rescue: re-score against 3-sentence windows (stride 2) of each
        # passage; the winning window's entailment/contradiction + its
        # passage's chunk_id replace the stage-1 result when it clears the
        # floor. Capped at ``_MAX_STAGE2_WINDOWS`` premises (first-N per
        # passage) to bound runtime.
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
            s2_pairs = []
            s2_map = []  # (claim_index, window_index)
            for ci in unrescued:
                claim = scorable[ci]
                for wi, premise in enumerate(window_premises):
                    s2_pairs.append((premise, claim))
                    s2_map.append((ci, wi))
            s2_scores = resolved.score_batch(pairs=s2_pairs)
            # Argmax windowed entailment per rescued claim.
            s2_best = {}
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
        computational_numeric_check=comp_check,
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
    "ENV_GROUNDEDNESS_COMPUTATIONAL",
    "resolve_groundedness_computational",
    "ENV_GROUNDEDNESS_S1_TOPK",
    "resolve_groundedness_s1_topk",
    "ENV_GROUNDEDNESS_FRONTIER",
    "ENV_GROUNDEDNESS_FRONTIER_WIDTH",
    "resolve_groundedness_frontier",
    "resolve_groundedness_frontier_width",
]
