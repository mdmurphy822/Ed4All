"""Deterministic answer-completeness analysis for the grounded-answer path
(PIECE B, 2026-06).

Pure, side-effect-free, stdlib-only. NO LLM, NO model load, NO embedding call.
Given a learner's question, the composed answer text, and the retrieved
passages, this module detects sub-questions that are (i) NOT addressed by the
answer AND (ii) ARE grounded in the retrieved passages — the
"uncovered-but-grounded" parts that should trigger a re-ask. This module is the
PURE core; a separate integration piece wires :func:`analyze_completeness` into
``lib.retrieval.grounded_answer``.

Motivating case (2026-06): a learner asks "how do i find the perimeter of a
rectangle? the circumference of a circle?" and the model answers only the
rectangle half. The circumference half is grounded in a retrieved passage but
goes unaddressed — a re-ask would recover it. The number-blind, paraphrase-blind
shingle/token arms cannot tell us this on their own; the completeness analysis
splits the question and tests each half independently.

Design bias — RECALL of the re-ask trigger, but UNDER-split the question:

  * Question splitting (:func:`split_question_parts`) biases toward UNDER-
    splitting. A false single-part is safe (we simply never re-ask). A false
    multi-part causes a needless re-ask. So the splitter only splits where the
    boundary is unambiguous (a ``?`` terminator) or a coordination/enumeration
    yields clearly distinct interrogative-ish clauses.

  * The "is this part answered?" predicate (:func:`part_addressed_by_answer`) is
    deliberately RECALL-leaning: it uses a LOWER token-coverage bar than the
    attribution keep-decision (which is a precision-of-keep check). We would
    rather conclude a part IS addressed (and not re-ask) than spuriously decide
    it is uncovered and re-ask on a question the answer actually handled.

  * The grounding gate (:func:`part_grounded_in_passages`) is the safety valve:
    a part is only an uncovered-but-grounded re-ask candidate when SOME passage
    actually supports it. An uncovered part with no supporting passage is NOT a
    re-ask trigger (re-asking would just produce another refusal).

Reuse contract: every text helper is imported from
:mod:`lib.retrieval.citation_attribution` (which itself reuses
:mod:`lib.retrieval._text` and the pair-claim-support thresholds), so this
module never diverges from attribution on what "normalize", "shingle support",
or "token coverage" mean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from lib.retrieval.citation_attribution import (
    ATTRIBUTION_SHINGLE_SIZE,
    _content_tokens,
    _normalize,
    _shingle_support,
    _token_coverage,
    passage_token_coverage,
)

__all__ = [
    "PART_ADDRESSED_MIN_SHINGLE",
    "PART_ADDRESSED_MIN_TOKEN_COVERAGE",
    "PART_GROUNDED_MIN_TOKEN_COVERAGE",
    "PartCompleteness",
    "CompletenessResult",
    "split_question_parts",
    "part_addressed_by_answer",
    "part_grounded_in_passages",
    "analyze_completeness",
]


# --------------------------------------------------------------------------- #
# Default thresholds (recall-leaning — see module docstring)
# --------------------------------------------------------------------------- #

#: Shingle floor for deciding a question PART is addressed by the answer.
#: Mirrors the attribution keep-decision's ``min_overlap`` spirit but is the
#: phrase arm of a recall-leaning OR — a part need only match EITHER the shingle
#: floor OR the (lower) token-coverage floor. 0.50 matches
#: ``citation_attribution.ADD_MIN_SHINGLE``: a verbatim half of the question
#: reproduced in the answer clears it, while a light paraphrase falls to the
#: token-coverage arm below (which is the dominant signal for "is this answered").
PART_ADDRESSED_MIN_SHINGLE = 0.50

#: Token-coverage floor for deciding a question PART is addressed by the answer.
#: DELIBERATELY LOWER than attribution's ``TOKEN_COVERAGE_FLOOR`` (0.80): that
#: floor calibrates the precision of a keep-vs-drop citation decision, whereas
#: this calibrates "did the answer touch this sub-question at all". A recall-
#: leaning 0.60 means a majority of the part's content tokens appearing in the
#: answer counts as addressed — biasing AWAY from a needless re-ask (a false
#: single-part is safe; a false "uncovered" triggers an unwanted re-ask).
PART_ADDRESSED_MIN_TOKEN_COVERAGE = 0.60

#: Token-coverage floor for deciding a question PART is GROUNDED in some passage.
#: Reuses ``passage_token_coverage`` (the exact attribution machinery). Held at
#: the attribution secondary floor 0.80: grounding is the SAFETY VALVE that gates
#: a re-ask, so it should be PRECISE — a part is only a re-ask candidate when a
#: passage genuinely supports it. A spurious grounding decision would re-ask on a
#: question the corpus cannot actually answer (producing another refusal).
PART_GROUNDED_MIN_TOKEN_COVERAGE = 0.80


# --------------------------------------------------------------------------- #
# Question splitting
# --------------------------------------------------------------------------- #

#: A leading interrogative stem worth carrying onto a trailing bare fragment.
#: Conservative: only the common English question openers, optionally with an
#: auxiliary verb, up to (but not including) the first content noun phrase. Used
#: ONLY to prepend context onto a trailing fragment like "the circumference of a
#: circle" so its content tokens read as an interrogative clause; the trailing
#: fragment's content tokens are what the coverage predicates actually use.
_INTERROGATIVE_STEM_RE = re.compile(
    r"^\s*(how|what|where|when|why|which|who|can|could|do|does|did|"
    r"is|are|should|would|will|find|calculate|compute|determine|explain)\b"
    r"(?:\s+(?:do|does|did|can|could|i|you|we|to|the|a|an))*"
    # Optionally consume a trailing action verb (e.g. "how do i FIND ...") so
    # the stem is fully removed and its stopwords don't dilute coverage.
    r"(?:\s+(?:find|calculate|compute|determine|explain|get|solve|use))?",
    re.IGNORECASE,
)

#: Secondary coordination/enumeration split points, applied to a SINGLE-``?``
#: question only (we never re-split an already ``?``-terminated clause). A split
#: fires only when it yields clearly distinct interrogative-ish clauses (see
#: :func:`_looks_interrogative`); otherwise we leave the clause whole (under-
#: split bias). Semicolon is the strong signal; " and " is admitted only between
#: two clauses that each look interrogative.
_SEMICOLON_SPLIT_RE = re.compile(r"\s*;\s*")
_AND_SPLIT_RE = re.compile(r"\s+\band\b\s+", re.IGNORECASE)

#: A clause "looks interrogative" if it opens with an interrogative stem OR
#: contains a content noun the question is plausibly asking about. We keep this
#: cheap: an interrogative opener, or simply >= 2 content tokens (a bare noun
#: phrase like "the circumference of a circle" carries the askable content).
def _looks_interrogative(clause: str) -> bool:
    if _INTERROGATIVE_STEM_RE.match(clause):
        return True
    return len(_content_tokens(clause)) >= 2


def _clean_fragment(text: str) -> str:
    """Whitespace-normalize and strip a stray leading/trailing connective."""
    frag = _normalize(text or "")
    # Drop a leading coordinator the splitter may have left attached.
    frag = re.sub(r"^(?:and|or|also|then)\b\s*", "", frag, flags=re.IGNORECASE)
    return frag.strip(" ,;")


def _match_text(part: str) -> str:
    """The part text used for lexical MATCHING (stem-stripped).

    The leading interrogative stem ("how do i find", "what is") is not askable
    CONTENT — it is scaffolding the answer/passages need not reproduce. Counting
    its stopword tokens ("how", "do", "find") as part of the part's content set
    DILUTES token coverage and would push a fully-answered/fully-grounded part
    below the floors. So matching is done against the part with its leading
    interrogative stem removed; the part's human-readable text (carrying the
    stem) is what :func:`split_question_parts` returns and what the result
    records. When stripping would leave nothing, the original part is used.
    """
    stripped = _INTERROGATIVE_STEM_RE.sub("", part or "", count=1).strip(" ,;")
    return stripped if _content_tokens(stripped) else _normalize(part or "")


def _carry_stem(parts: List[str]) -> List[str]:
    """Prepend the first part's interrogative stem onto bare trailing parts.

    "how do i find the perimeter of a rectangle" + "the circumference of a
    circle" → the second part is carried to "how do i find the circumference of
    a circle" when (a) it has no interrogative opener of its own and (b) the
    first part does. Cheap, best-effort: if the stem can't be cleanly read we
    leave the fragment untouched (its content tokens still drive coverage).
    """
    if len(parts) < 2:
        return parts
    head = parts[0]
    stem_match = _INTERROGATIVE_STEM_RE.match(head)
    if not stem_match:
        return parts
    stem = stem_match.group(0).strip()
    if not stem:
        return parts
    out = [parts[0]]
    for frag in parts[1:]:
        if _INTERROGATIVE_STEM_RE.match(frag):
            out.append(frag)  # already interrogative — leave it
        else:
            out.append(_normalize(f"{stem} {frag}"))
    return out


def split_question_parts(query: str) -> List[str]:
    """Deterministically split a question into sub-questions (UNDER-split bias).

    Primary split is on ``?`` (the motivating "how do i find the perimeter of a
    rectangle? the circumference of a circle?" splits cleanly into two). A
    single-``?`` (or no-``?``) question is then conservatively considered for a
    coordination/enumeration split on ``;`` and " and " — but ONLY when the split
    yields clauses that each look interrogative (an opener or >= 2 content
    tokens); otherwise the clause is kept whole.

    A short leading interrogative stem from the first part ("how do i find") is
    carried onto a trailing bare fragment ("the circumference of a circle") when
    cheaply doable, so the fragment reads as a clause — but the fragment's
    content tokens are what the coverage predicates actually use, so this is
    cosmetic.

    Returns a 1-element list for a single-part question. Whitespace is
    normalized and empty fragments are dropped.

    Bias: a false single-part is SAFE (we simply never re-ask); a false multi-
    part causes a needless re-ask. So splitting only fires on unambiguous
    boundaries.
    """
    raw = (query or "").strip()
    if not raw:
        return []

    # Primary: split on '?' terminators. Keep each interrogative clause.
    primary = [seg for seg in raw.split("?")]
    # A trailing segment with no '?' is the tail after the last '?'; include it
    # only if it carries content (it's the "the circumference of a circle?" case
    # where the user's final '?' makes the last real segment, and any leftover is
    # whitespace).
    question_parts: List[str] = []
    for seg in primary:
        frag = _clean_fragment(seg)
        if frag:
            question_parts.append(frag)

    if len(question_parts) >= 2:
        # Multiple '?'-delimited clauses: carry the stem and return.
        return _carry_stem(question_parts)

    # Single clause (zero or one '?'): consider a conservative coordination split.
    base = question_parts[0] if question_parts else _clean_fragment(raw)
    if not base:
        return []

    # A coordination/enumeration split is considered ONLY for a genuinely
    # interrogative query — one that contains at least one '?'. A query with no
    # '?' is a search/statement-style query (e.g. passage text used verbatim as a
    # semantic query, "Vector Stores and Embeddings ..."); splitting THAT on
    # " and " / ";" manufactures spurious parts and triggers needless re-asks.
    # Under-split bias: no '?' anywhere → single part.
    if "?" in raw:
        # Try semicolon first (strong boundary), then " and ".
        for splitter in (_SEMICOLON_SPLIT_RE, _AND_SPLIT_RE):
            clauses = [_clean_fragment(c) for c in splitter.split(base)]
            clauses = [c for c in clauses if c]
            if len(clauses) >= 2 and all(_looks_interrogative(c) for c in clauses):
                return _carry_stem(clauses)

    return [base]


# --------------------------------------------------------------------------- #
# Per-part predicates
# --------------------------------------------------------------------------- #


def part_addressed_by_answer(
    part: str,
    answer_text: str,
    *,
    min_shingle: float = PART_ADDRESSED_MIN_SHINGLE,
    min_token_coverage: float = PART_ADDRESSED_MIN_TOKEN_COVERAGE,
) -> "tuple[bool, Dict[str, float]]":
    """True iff the ANSWER addresses this question ``part`` (recall-leaning OR).

    Mirrors the attribution keep-decision shape (shingle OR token-coverage), but
    with a LOWER token-coverage bar (see :data:`PART_ADDRESSED_MIN_TOKEN_COVERAGE`)
    because this asks "did the answer touch this sub-question", a recall-leaning
    check, not "does this citation precisely support a claim". Returns the
    deciding signals so a decision-capture rationale can replay the call.

    The part is normalized against the answer via the same ``_normalize`` /
    ``_shingle_support`` / ``_token_coverage`` helpers attribution uses, so the
    two never diverge on what those terms mean.
    """
    part_norm = _normalize(_match_text(part))
    answer_norm = _normalize(answer_text or "")
    shingle = _shingle_support(part_norm, answer_norm)
    coverage = _token_coverage(part_norm, answer_norm)
    addressed = (shingle >= min_shingle) or (coverage >= min_token_coverage)
    signals: Dict[str, float] = {
        "shingle": round(shingle, 6),
        "token_coverage": round(coverage, 6),
        "min_shingle": float(min_shingle),
        "min_token_coverage": float(min_token_coverage),
    }
    return addressed, signals


def _passage_chunk_id(passage: object) -> str:
    cid = getattr(passage, "chunk_id", None)
    if cid is None and isinstance(passage, dict):
        cid = passage.get("chunk_id")
    return str(cid) if cid else ""


def _passage_text(passage: object) -> str:
    text = getattr(passage, "text", None)
    if text is None and isinstance(passage, dict):
        text = passage.get("text")
    return str(text or "")


def part_grounded_in_passages(
    part: str,
    passages: Sequence[object],
    *,
    min_token_coverage: float = PART_GROUNDED_MIN_TOKEN_COVERAGE,
) -> "tuple[bool, Dict[str, object]]":
    """True iff SOME passage supports this question ``part`` by token coverage.

    Reuses :func:`citation_attribution.passage_token_coverage` (which normalizes
    both sides and applies the content-token machinery) over each passage's
    ``.text``; the best-covering passage's id + score are returned in the signal
    dict. ``passages`` are duck-typed objects exposing ``.text`` (and optionally
    ``.chunk_id``); dicts with the same keys are also accepted.

    This is the SAFETY VALVE that gates a re-ask: it is held at the PRECISE
    attribution floor (0.80) so we only ever re-ask on a sub-question the corpus
    can actually answer — a spurious grounding decision would re-ask into another
    refusal.
    """
    match_text = _match_text(part)
    best_score = 0.0
    best_id: Optional[str] = None
    for passage in passages or []:
        text = _passage_text(passage)
        if not text.strip():
            continue
        score = passage_token_coverage(match_text, text)
        if score > best_score:
            best_score = score
            best_id = _passage_chunk_id(passage) or None
    grounded = best_score >= min_token_coverage
    signals: Dict[str, object] = {
        "best_token_coverage": round(best_score, 6),
        "best_chunk_id": best_id,
        "min_token_coverage": float(min_token_coverage),
    }
    return grounded, signals


# --------------------------------------------------------------------------- #
# Result shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PartCompleteness:
    """Per-question-part completeness record.

    ``addressed`` — the answer covers this part (:func:`part_addressed_by_answer`).
    ``grounded``  — some passage supports this part (:func:`part_grounded_in_passages`).
    A part is an uncovered-but-grounded re-ask candidate iff
    ``not addressed and grounded``.
    """

    text: str
    addressed: bool
    grounded: bool
    addressed_signals: Dict[str, float] = field(default_factory=dict)
    grounded_signals: Dict[str, object] = field(default_factory=dict)

    @property
    def uncovered_but_grounded(self) -> bool:
        return (not self.addressed) and self.grounded


@dataclass(frozen=True)
class CompletenessResult:
    """Answer-completeness analysis over one (query, answer, passages) triple.

    ``parts`` is the per-part record list (in question order). ``is_multipart``
    is ``len(parts) > 1`` — a single-part query makes the no-op trivially
    detectable (``uncovered_but_grounded`` is then empty by construction unless
    the lone part is itself uncovered-but-grounded, which the integration may
    still choose to ignore). ``thresholds`` records the floors used for
    replayable decision capture.
    """

    query: str
    parts: List[PartCompleteness] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)

    @property
    def is_multipart(self) -> bool:
        return len(self.parts) > 1

    @property
    def uncovered_but_grounded(self) -> List[str]:
        """Texts of parts that are NOT addressed AND ARE grounded (re-ask set)."""
        return [p.text for p in self.parts if p.uncovered_but_grounded]


def analyze_completeness(
    query: str,
    answer_text: str,
    passages: Sequence[object],
    *,
    addressed_min_shingle: float = PART_ADDRESSED_MIN_SHINGLE,
    addressed_min_token_coverage: float = PART_ADDRESSED_MIN_TOKEN_COVERAGE,
    grounded_min_token_coverage: float = PART_GROUNDED_MIN_TOKEN_COVERAGE,
) -> CompletenessResult:
    """Split ``query`` and analyze each part for completeness (pure, no model).

    For each sub-question part, runs :func:`part_addressed_by_answer` against the
    answer and :func:`part_grounded_in_passages` against the retrieved passages,
    returning a :class:`CompletenessResult`. The convenience
    ``result.uncovered_but_grounded`` lists the parts the answer DID NOT address
    but that ARE grounded in a passage — the re-ask candidates.

    The thresholds used are recorded on the result so a downstream decision
    capture can replay the analysis. Side-effect-free: no network, no model
    load, no embedding call.
    """
    thresholds: Dict[str, float] = {
        "addressed_min_shingle": float(addressed_min_shingle),
        "addressed_min_token_coverage": float(addressed_min_token_coverage),
        "grounded_min_token_coverage": float(grounded_min_token_coverage),
        "shingle_size": float(ATTRIBUTION_SHINGLE_SIZE),
    }
    parts: List[PartCompleteness] = []
    for part_text in split_question_parts(query):
        addressed, a_sig = part_addressed_by_answer(
            part_text,
            answer_text,
            min_shingle=addressed_min_shingle,
            min_token_coverage=addressed_min_token_coverage,
        )
        grounded, g_sig = part_grounded_in_passages(
            part_text,
            passages,
            min_token_coverage=grounded_min_token_coverage,
        )
        parts.append(
            PartCompleteness(
                text=part_text,
                addressed=addressed,
                grounded=grounded,
                addressed_signals=a_sig,
                grounded_signals=g_sig,
            )
        )
    return CompletenessResult(query=query or "", parts=parts, thresholds=thresholds)
