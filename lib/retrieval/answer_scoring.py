"""Additive per-question answer-scoring extensions for the grounded-answer eval
(retrieval-answer-eval-set §4 / P4).

Pure, deterministic, stdlib + repo-internal deps only (no LLM, no network on the
deterministic arms; an OPTIONAL injectable NLI classifier supplies the third
key-point arm when groundedness is enabled). Three scorers, all additive to the
v1.1 eval report — none changes an answer verdict:

  1. :func:`score_key_point_coverage` — for v1.1 questions carrying
     ``expected_key_points``, score each key point against the answer text with
     the SAME deterministic machinery citation attribution uses (shingle-4
     containment OR content-token coverage, both reused from
     :mod:`lib.retrieval.citation_attribution`; the normalization is the
     anchor-equivalent projection). When an NLI classifier is supplied (the
     groundedness arm is enabled), a key point also counts covered if the
     answer entails it at the SAME entailment floor groundedness uses — so the
     completeness verdict and the groundedness verdict never disagree on what
     "the answer supports this claim" means.

  2. :func:`score_part_coverage` — for ``multi_part`` questions: each covered
     part is ``answered`` when the answer supports >= 1 of its key points or its
     part_text; each ``covered:false`` part is ``correctly_flagged`` iff the
     answer contains an explicit does-not-cover acknowledgment NEAR the part's
     noun phrase (the ws3.v2 disclaimer pattern). This second metric is
     deliberately diagnostic/additive — its absence-detection heuristic is
     brittle (risk R7); it is NEVER a pinned milestone.

  3. :func:`score_population_breakdown` — joins the answer's CITED chunk_ids
     against the chunk population (``source`` / ``course``) derived exactly the
     way :mod:`lib.retrieval.gold_coverage` derives it (reused
     ``_population_of_chunk``), plus per-question
     ``expected_citation_population`` satisfaction (an answered question whose
     expected population is not ``"any"`` must cite >= 1 chunk of that
     population).

The deterministic key-point + part arms reuse
:func:`citation_attribution._shingle_support` /
:func:`citation_attribution._token_coverage` (and the shared
``_normalize``) rather than re-implementing them, so a key point is scored
"present in the answer" by exactly the rule a claim is scored "supported by a
passage".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from lib.retrieval.citation_attribution import (
    ATTRIBUTION_SHINGLE_SIZE,
    DEFAULT_MIN_OVERLAP,
    TOKEN_COVERAGE_FLOOR,
    _normalize,
    _shingle_support,
    _token_coverage,
)
from lib.retrieval.gold_coverage import _population_of_chunk

__all__ = [
    "KeyPointVerdict",
    "KeyPointCoverage",
    "PartVerdict",
    "PartCoverage",
    "PopulationBreakdown",
    "score_key_point_coverage",
    "score_part_coverage",
    "score_population_breakdown",
    "DISCLAIMER_PHRASES",
]


# --------------------------------------------------------------------------- #
# 1) Key-point coverage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KeyPointVerdict:
    """Per-key-point completeness verdict against the answer text.

    ``covered`` is the OR of the deterministic shingle/coverage arms and (when
    an NLI classifier is supplied) the entailment arm. ``arm`` names the arm
    that fired (``shingle`` / ``token_coverage`` / ``nli`` / ``none``) — the
    audit signal for which evidence carried the verdict.
    """

    key_point: str
    covered: bool
    shingle: float = 0.0
    coverage: float = 0.0
    entailment: Optional[float] = None
    arm: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "key_point": self.key_point,
            "covered": self.covered,
            "shingle": round(self.shingle, 6),
            "coverage": round(self.coverage, 6),
            "arm": self.arm,
        }
        if self.entailment is not None:
            out["entailment"] = round(self.entailment, 6)
        return out


@dataclass(frozen=True)
class KeyPointCoverage:
    """Roll-up of per-key-point verdicts for one question."""

    total: int
    covered: int
    verdicts: List[KeyPointVerdict] = field(default_factory=list)
    nli_used: bool = False

    @property
    def coverage_rate(self) -> float:
        return (self.covered / self.total) if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "covered": self.covered,
            "coverage_rate": round(self.coverage_rate, 6),
            "nli_used": self.nli_used,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def _nli_entailment(
    nli: Any, premise: str, hypothesis: str
) -> Optional[float]:
    """Best-effort single-pair entailment via the injected NLI classifier.

    Returns ``None`` (degrade) when the classifier can't score the pair — the
    deterministic arms still decide the verdict, never a fabricated 0.0.
    """
    try:
        scores = nli.score_batch(pairs=[(premise, hypothesis)])
    except Exception:  # noqa: BLE001 — NLI failure degrades to deterministic arms
        return None
    if not scores:
        return None
    return float(getattr(scores[0], "entailment", 0.0) or 0.0)


def score_key_point_coverage(
    answer_text: Optional[str],
    key_points: Sequence[str],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    token_coverage_floor: float = TOKEN_COVERAGE_FLOOR,
    nli: Optional[Any] = None,
    entailment_floor: Optional[float] = None,
) -> Optional[KeyPointCoverage]:
    """Score each ``key_point`` against ``answer_text``.

    A key point is ``covered`` iff (deterministic arms, reused from
    ``citation_attribution``):

      * ``shingle4(key_point, answer) >= min_overlap``  OR
      * ``token_coverage(key_point, answer) >= token_coverage_floor``

    OR, when ``nli`` is supplied (groundedness enabled), the answer entails the
    key point at ``entailment >= entailment_floor`` (the groundedness floor —
    passed in by the caller so the two never diverge). The answer is the NLI
    PREMISE and the key point the HYPOTHESIS: "does the answer support this
    expected point?".

    Returns ``None`` when the question carries no key points (the v1.0 case — no
    completeness slice to report), else a :class:`KeyPointCoverage`.
    """
    points = [str(k) for k in (key_points or []) if str(k).strip()]
    if not points:
        return None

    answer_norm = _normalize(answer_text or "")
    nli_used = nli is not None and entailment_floor is not None

    verdicts: List[KeyPointVerdict] = []
    covered_count = 0
    for kp in points:
        kp_norm = _normalize(kp)
        shingle = _shingle_support(kp_norm, answer_norm)
        coverage = _token_coverage(kp_norm, answer_norm)

        arm = "none"
        covered = False
        if shingle >= min_overlap:
            covered, arm = True, "shingle"
        elif coverage >= token_coverage_floor:
            covered, arm = True, "token_coverage"

        entailment: Optional[float] = None
        if nli_used:
            # Premise = answer, hypothesis = key point. Only consult NLI when the
            # deterministic arms did not already cover it (cheaper + the verdict
            # can only flip false->true, never the reverse).
            ent = _nli_entailment(nli, answer_text or "", kp)
            if ent is not None:
                entailment = ent
                if not covered and ent >= float(entailment_floor):
                    covered, arm = True, "nli"

        if covered:
            covered_count += 1
        verdicts.append(
            KeyPointVerdict(
                key_point=kp,
                covered=covered,
                shingle=shingle,
                coverage=coverage,
                entailment=entailment,
                arm=arm,
            )
        )

    return KeyPointCoverage(
        total=len(points),
        covered=covered_count,
        verdicts=verdicts,
        nli_used=nli_used,
    )


# --------------------------------------------------------------------------- #
# 2) Part coverage (multi_part questions)
# --------------------------------------------------------------------------- #

#: The ws3.v2 does-not-cover disclaimer phrases. Deterministic detection: an
#: answer "correctly flags" an uncovered part when one of these phrases appears
#: NEAR the part's noun phrase. Brittle by design (risk R7) — phrased
#: differently ("the material is silent on ...") it misses; this is a
#: diagnostic, additive metric, never a pinned milestone.
DISCLAIMER_PHRASES = (
    "does not cover",
    "does not include",
    "does not address",
    "does not discuss",
    "is not covered",
    "are not covered",
    "not covered",
    "not directly provided",
    "not provided",
    "not addressed",
    "not discussed",
    "not included",
    "no information",
    "isn't covered",
    "doesn't cover",
    "outside the scope",
    "out of scope",
    "beyond the scope",
)

#: Window (in characters of normalized lowercased answer text) within which a
#: disclaimer phrase and a part noun-phrase token must co-occur to count as a
#: flag for that part. Wide enough for one sentence, narrow enough that a
#: disclaimer about part A does not credit a totally unrelated part B.
_DISCLAIMER_PROXIMITY_CHARS = 160

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "at",
        "by", "is", "are", "was", "were", "be", "with", "as", "that", "this",
        "it", "its", "what", "how", "why", "when", "where", "which", "who",
        "does", "do", "did", "can", "could", "should", "would", "explain",
        "describe", "list", "define", "give", "provide", "discuss", "into",
        "from", "about", "between", "their", "they", "them",
    }
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]+")


def _noun_phrase_tokens(part_text: str) -> List[str]:
    """Content tokens of a part's text (lowercased, stopword-stripped, len>=3).

    The crude noun-phrase proxy: a disclaimer must mention one of these tokens
    to be credited to this part. No POS tagger (stdlib only); good enough for
    the diagnostic metric and honest about its crudeness (risk R7).
    """
    toks = _TOKEN_RE.findall((part_text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _disclaimer_flags_part(answer_norm_lower: str, part_text: str) -> bool:
    """True when a disclaimer phrase co-occurs within the proximity window of a
    content token from ``part_text``.

    Both inputs are already lowercased + whitespace-normalized.
    """
    if not answer_norm_lower:
        return False
    np_tokens = _noun_phrase_tokens(part_text)
    if not np_tokens:
        # No anchorable noun phrase: fall back to "any disclaimer present"
        # (still diagnostic — recorded, not pinned).
        return any(p in answer_norm_lower for p in DISCLAIMER_PHRASES)

    # Positions of every disclaimer phrase occurrence.
    disclaimer_spans: List[int] = []
    for phrase in DISCLAIMER_PHRASES:
        start = answer_norm_lower.find(phrase)
        while start != -1:
            disclaimer_spans.append(start)
            start = answer_norm_lower.find(phrase, start + 1)
    if not disclaimer_spans:
        return False

    for tok in np_tokens:
        idx = answer_norm_lower.find(tok)
        while idx != -1:
            for dpos in disclaimer_spans:
                if abs(dpos - idx) <= _DISCLAIMER_PROXIMITY_CHARS:
                    return True
            idx = answer_norm_lower.find(tok, idx + 1)
    return False


@dataclass(frozen=True)
class PartVerdict:
    """Per-part outcome for a multi_part question."""

    part_id: str
    covered: bool  # the gold's covered flag (is the part answerable?)
    # For covered:true parts → answered (answer supports the part).
    answered: Optional[bool] = None
    # For covered:false parts → correctly_flagged (answer acknowledges absence).
    correctly_flagged: Optional[bool] = None
    arm: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "part_id": self.part_id,
            "covered": self.covered,
            "arm": self.arm,
        }
        if self.answered is not None:
            out["answered"] = self.answered
        if self.correctly_flagged is not None:
            out["correctly_flagged"] = self.correctly_flagged
        return out


@dataclass(frozen=True)
class PartCoverage:
    """Roll-up of per-part verdicts for one multi_part question."""

    n_parts: int
    n_covered_parts: int
    n_answered: int           # covered parts the answer supported
    n_uncovered_parts: int
    n_correctly_flagged: int  # covered:false parts the answer acknowledged
    verdicts: List[PartVerdict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_parts": self.n_parts,
            "n_covered_parts": self.n_covered_parts,
            "n_answered": self.n_answered,
            "n_uncovered_parts": self.n_uncovered_parts,
            "n_correctly_flagged": self.n_correctly_flagged,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "_diagnostic": (
                "part absence-flagging is heuristic (risk R7); diagnostic, "
                "not a pinned milestone"
            ),
        }


def score_part_coverage(
    answer_text: Optional[str],
    parts: Sequence[Dict[str, Any]],
    *,
    key_points_by_part: Optional[Dict[str, Sequence[str]]] = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    token_coverage_floor: float = TOKEN_COVERAGE_FLOOR,
) -> Optional[PartCoverage]:
    """Score the parts of a ``multi_part`` question against ``answer_text``.

    For a ``covered:true`` part → ``answered`` iff the answer supports the
    part's text (or any of its key points) by the deterministic shingle/coverage
    arms. For a ``covered:false`` part → ``correctly_flagged`` iff the answer
    contains an explicit does-not-cover acknowledgment near the part's noun
    phrase (:func:`_disclaimer_flags_part`).

    Returns ``None`` when ``parts`` is empty/absent.
    """
    part_list = [p for p in (parts or []) if isinstance(p, dict)]
    if not part_list:
        return None

    answer_norm = _normalize(answer_text or "")
    answer_norm_lower = answer_norm.lower()
    kp_map = key_points_by_part or {}

    verdicts: List[PartVerdict] = []
    n_covered = n_answered = n_uncovered = n_flagged = 0
    for part in part_list:
        pid = str(part.get("part_id", ""))
        covered = bool(part.get("covered", False))
        if covered:
            n_covered += 1
            # Support targets: the part text plus any per-part key points.
            targets = [str(part.get("part_text", ""))]
            targets.extend(str(k) for k in kp_map.get(pid, []) if str(k).strip())
            answered = False
            arm = "none"
            for tgt in targets:
                tgt_norm = _normalize(tgt)
                if not tgt_norm:
                    continue
                if _shingle_support(tgt_norm, answer_norm) >= min_overlap:
                    answered, arm = True, "shingle"
                    break
                if _token_coverage(tgt_norm, answer_norm) >= token_coverage_floor:
                    answered, arm = True, "token_coverage"
                    break
            if answered:
                n_answered += 1
            verdicts.append(
                PartVerdict(part_id=pid, covered=True, answered=answered, arm=arm)
            )
        else:
            n_uncovered += 1
            flagged = _disclaimer_flags_part(
                answer_norm_lower, str(part.get("part_text", ""))
            )
            if flagged:
                n_flagged += 1
            verdicts.append(
                PartVerdict(
                    part_id=pid,
                    covered=False,
                    correctly_flagged=flagged,
                    arm="disclaimer" if flagged else "none",
                )
            )

    return PartCoverage(
        n_parts=len(part_list),
        n_covered_parts=n_covered,
        n_answered=n_answered,
        n_uncovered_parts=n_uncovered,
        n_correctly_flagged=n_flagged,
        verdicts=verdicts,
    )


# --------------------------------------------------------------------------- #
# 3) Per-population citation breakdown
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PopulationBreakdown:
    """Per-question citation-population breakdown.

    ``cited_source`` / ``cited_course`` count this question's CITED chunks by
    the population they belong to (derived the same way the coverage report
    does). ``expected_population`` is the authored
    ``expected_citation_population`` (default ``"any"``);
    ``expected_satisfied`` is None when expected is ``"any"`` or the question
    was not answered, else whether the citations satisfy the expectation.
    """

    cited_source: int
    cited_course: int
    expected_population: str
    expected_satisfied: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cited_source": self.cited_source,
            "cited_course": self.cited_course,
            "expected_population": self.expected_population,
            "expected_satisfied": self.expected_satisfied,
        }


def _populations_of_cited(
    cited_ids: Sequence[str], chunks_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, int]:
    """Tally cited chunk ids by population (unknown ids skipped — they cannot be
    classified against a chunk that is not in the pinned set)."""
    counts = {"source": 0, "course": 0}
    for cid in cited_ids:
        chunk = chunks_by_id.get(str(cid))
        if chunk is None:
            continue
        pop = _population_of_chunk(chunk)
        counts[pop] = counts.get(pop, 0) + 1
    return counts


def score_population_breakdown(
    cited_ids: Sequence[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    expected_population: str = "any",
    answered: bool = True,
) -> PopulationBreakdown:
    """Join an answered question's cited ids against the chunk population.

    ``expected_satisfied`` is True iff the answer (when ``answered``) cites at
    least one chunk of the expected population. For ``expected_population ==
    "both"``, both populations must be cited. ``"any"`` → no constraint
    (``expected_satisfied is None``). An unanswered question also reports
    ``None`` (nothing to satisfy).
    """
    expected = expected_population if expected_population in (
        "source", "course", "both", "any"
    ) else "any"
    counts = _populations_of_cited(cited_ids, chunks_by_id)
    n_source = counts.get("source", 0)
    n_course = counts.get("course", 0)

    satisfied: Optional[bool]
    if not answered or expected == "any":
        satisfied = None
    elif expected == "source":
        satisfied = n_source >= 1
    elif expected == "course":
        satisfied = n_course >= 1
    else:  # both
        satisfied = n_source >= 1 and n_course >= 1

    return PopulationBreakdown(
        cited_source=n_source,
        cited_course=n_course,
        expected_population=expected,
        expected_satisfied=satisfied,
    )
