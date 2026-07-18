"""Trivote Bloom voters — zero-shot NLI + verb-ontology + asserted-label.

``ED4ALL_BLOOM_TRIVOTE`` (default OFF) re-founds the
``bloom_classifier_disagreement`` signal on three *interpretable*
voters, retiring the unlicensed/undocumented
``cip29/bert-blooms-taxonomy-classifier`` and the sentiment member
(``distilbert-base-uncased-finetuned-sst-2-english``) from the legacy
:class:`lib.classifiers.bloom_bert_ensemble.BloomBertEnsemble`. The
gate's meaning becomes *"does the generator's own Bloom claim survive
independent checks"*.

The three voters:

1. **Asserted label** — the generator's OWN declared ``bloom_level``,
   read straight from the artifact metadata by the validator (objective
   ``bloom_level`` / dynamic-block-plan slot level / assessment item
   ``bloom_level``). Absent => the voter abstains. (Supplied by the
   validator; not computed here.)
2. **Zero-shot DeBERTa** (:func:`zero_shot_bloom`) — entailment of six
   per-Bloom-level hypothesis templates scored on the ALREADY-LICENSED
   ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` via the process-
   singleton :class:`lib.classifiers.nli_classifier.NliClassifier`
   (NO second DeBERTa load — the same model already resident for
   ``block_prose_entailment`` / ``claim_support`` / ``objective_
   entailment``). NLI unavailable => the voter abstains.
3. **Verb ontology** (:func:`verb_vote`) — the deterministic verb-based
   level from the canonical :func:`lib.ontology.bloom.detect_bloom_level`.
   No canonical verb in the text => the voter abstains (NOT a fabricated
   ``remember`` default).

Zero new model weights are introduced (the machine is HF-offline;
nothing is downloaded). This module is inference-only; the licensing
posture is documented in ``docs/LICENSING.md``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from lib.ontology.bloom import BLOOM_LEVELS, detect_bloom_level

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flag resolution.
# ---------------------------------------------------------------------------

#: Trivote master flag. Default OFF => byte-identical legacy 3-member
#: ensemble path in the validator. Parse-with-fallback truthy set.
_TRIVOTE_ENV_VAR = "ED4ALL_BLOOM_TRIVOTE"
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def resolve_trivote_mode() -> bool:
    """Return True when ``ED4ALL_BLOOM_TRIVOTE`` is truthy.

    Read at ``validate()`` time (not import time) so a per-test /
    per-run monkeypatch of the env var flips the mode without a module
    reload. Garbage / unset => False (legacy path).
    """
    raw = os.environ.get(_TRIVOTE_ENV_VAR, "").strip().lower()
    return raw in _TRUTHY_VALUES


# ---------------------------------------------------------------------------
# Zero-shot NLI voter.
# ---------------------------------------------------------------------------

#: Per-Bloom-level hypothesis templates for the zero-shot NLI voter.
#: Each completes the implicit stem "Given the task text, ...". Edit
#: these to re-calibrate the zero-shot voter; the keys MUST stay the
#: canonical :data:`lib.ontology.bloom.BLOOM_LEVELS` set (asserted by
#: the test suite).
_BLOOM_HYPOTHESIS_TEMPLATES: Dict[str, str] = {
    "remember": (
        "This task requires the student to remember or recall facts, "
        "terms, definitions, or basic concepts."
    ),
    "understand": (
        "This task requires the student to understand and explain ideas, "
        "concepts, or the meaning of information."
    ),
    "apply": (
        "This task requires the student to apply knowledge, procedures, or "
        "methods to solve a problem in a new situation."
    ),
    "analyze": (
        "This task requires the student to analyze information by breaking "
        "it into parts and examining how the parts relate."
    ),
    "evaluate": (
        "This task requires the student to evaluate, judge, critique, or "
        "justify a position based on criteria."
    ),
    "create": (
        "This task requires the student to create, design, compose, or "
        "produce a new original work."
    ),
}


def _normalize_scores(levels: List[str], values: List[float]) -> Dict[str, float]:
    """Turn six raw entailment scores into a distribution keyed by level.

    Sum-normalisation (``value / Σ values``) — NOT a temperature-softmax —
    because the DeBERTa entailment scores already live in ``[0, 1]``, so a
    softmax over that narrow range would flatten a strong signal to near-
    uniform and make a relative-confidence floor meaningless. Sum-
    normalisation preserves the relative entailment magnitudes: a lone
    strongly-entailed level keeps a high share. Returns ``{}`` when the
    total is non-positive (no signal => the caller abstains).
    """
    total = sum(values)
    if total <= 0:
        return {}
    return {lvl: value / total for lvl, value in zip(levels, values)}


def zero_shot_bloom(
    text: str,
    *,
    nli: Optional[Any] = None,
) -> Optional[Tuple[str, float, Dict[str, float]]]:
    """Zero-shot Bloom-level classification via NLI entailment.

    Scores entailment of the six per-level hypothesis templates
    (:data:`_BLOOM_HYPOTHESIS_TEMPLATES`) against ``text`` as the NLI
    premise, normalises the six entailment scores into a distribution
    (:func:`_normalize_scores`), and takes the argmax (lexicographic
    tie-break, mirroring the ensemble aggregator).

    Returns ``(level, confidence, per_level_scores)`` where ``level`` is
    a canonical :data:`~lib.ontology.bloom.BLOOM_LEVELS` member,
    ``confidence`` is the winner's normalised entailment share, and
    ``per_level_scores`` maps every level to its share (sums to 1.0).

    Returns ``None`` (=> the voter abstains, gracefully) when:

    - ``text`` is empty / whitespace, OR
    - the NLI model is unavailable (missing extras, load failure), OR
    - the batched forward pass raises.

    ``nli`` is injectable for tests (any object exposing
    ``score_batch(*, pairs) -> [NliScore-like]``). Production passes
    ``None`` and the function pulls the process-singleton via
    :meth:`lib.classifiers.nli_classifier.NliClassifier.get_or_load`
    (no second DeBERTa load).
    """
    if not text or not text.strip():
        return None

    classifier = nli
    if classifier is None:
        try:
            from lib.classifiers.nli_classifier import NliClassifier

            classifier = NliClassifier.get_or_load()
        except Exception as exc:  # noqa: BLE001 — graceful-degrade => abstain
            logger.debug("zero_shot_bloom: NLI singleton load failed: %s", exc)
            classifier = None
    if classifier is None:
        return None

    premise = text.strip()
    levels = list(BLOOM_LEVELS)
    pairs = [(premise, _BLOOM_HYPOTHESIS_TEMPLATES[lvl]) for lvl in levels]
    try:
        scores = classifier.score_batch(pairs=pairs)
    except Exception as exc:  # noqa: BLE001 — per-call NLI failure => abstain
        logger.warning("zero_shot_bloom: score_batch raised: %s", exc)
        return None

    if not scores or len(scores) != len(levels):
        logger.debug(
            "zero_shot_bloom: expected %d scores, got %s; abstaining",
            len(levels),
            len(scores) if scores else 0,
        )
        return None

    entail = [max(0.0, float(getattr(s, "entailment", 0.0))) for s in scores]
    per_level = _normalize_scores(levels, entail)
    if not per_level:
        return None
    # Argmax with lexicographic tie-break (sort by (-weight, level)) so
    # the winner is deterministic across re-runs, matching the ensemble
    # aggregator's tie-break convention.
    winner_level, winner_score = sorted(
        per_level.items(), key=lambda kv: (-kv[1], kv[0])
    )[0]
    return (winner_level, round(float(winner_score), 4), per_level)


# ---------------------------------------------------------------------------
# Verb-ontology voter.
# ---------------------------------------------------------------------------


def verb_vote(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Deterministic verb-ontology Bloom voter.

    Thin wrapper over the canonical
    :func:`lib.ontology.bloom.detect_bloom_level`. Returns
    ``(level, verb)`` on a canonical-verb match, or ``(None, None)`` when
    the text carries no canonical Bloom verb (=> the voter abstains — a
    genuine non-vote, never a fabricated ``remember`` default).
    """
    if not text or not text.strip():
        return (None, None)
    return detect_bloom_level(text)


# ---------------------------------------------------------------------------
# Trivote aggregation.
# ---------------------------------------------------------------------------

#: Canonical voter names, in a stable order for reporting.
_VOTER_NAMES: Tuple[str, ...] = ("asserted", "zero_shot", "verb")


def _norm_level(level: Optional[str]) -> Optional[str]:
    """Lowercase + validate a level against the canonical set.

    A non-canonical / empty level normalises to ``None`` so the voter
    abstains rather than casting a junk vote.
    """
    if not level or not isinstance(level, str):
        return None
    lowered = level.strip().lower()
    return lowered if lowered in BLOOM_LEVELS else None


def trivote_disagreement(
    asserted: Optional[str],
    zero_shot: Optional[str],
    verb: Optional[str],
) -> Dict[str, Any]:
    """Compute the trivote disagreement metric over the three votes.

    Each vote is a canonical Bloom level or ``None`` (abstention). The
    disagreement score is the **pairwise mismatch fraction over the
    non-abstaining voters** — for ``k`` present voters there are
    ``C(k, 2)`` ordered-independent pairs; the score is
    ``mismatched_pairs / C(k, 2)`` (``0.0`` when fewer than two voters
    are present).

    Returns a dict with:

    - ``votes``: ``{voter_name: level | None}`` (normalised).
    - ``present`` / ``abstentions``: voter-name partitions.
    - ``n_present``: count of non-abstaining voters.
    - ``pairwise_total`` / ``pairwise_mismatch``: the pair counts.
    - ``disagreement_score``: mismatch fraction (rounded 4dp).
    - ``skipped``: ``True`` when ``n_present < 2`` (structured skip — the
      caller must NOT fabricate a pass/fail from a single voter).
    - ``majority_level``: plurality level among present voters
      (lexicographic tie-break) or ``None``.
    - ``asserted_present`` / ``asserted_supported`` /
      ``asserted_unsupported``: whether the generator's own claim is
      backed by ≥1 independent (non-asserted) voter.
    - ``independent_levels``: the non-asserted present votes.
    """
    normalised = {
        "asserted": _norm_level(asserted),
        "zero_shot": _norm_level(zero_shot),
        "verb": _norm_level(verb),
    }
    present: List[Tuple[str, str]] = [
        (name, normalised[name]) for name in _VOTER_NAMES if normalised[name]
    ]
    abstentions = [name for name in _VOTER_NAMES if not normalised[name]]
    n = len(present)

    total_pairs = n * (n - 1) // 2
    mismatch_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if present[i][1] != present[j][1]:
                mismatch_pairs += 1
    disagreement_score = (
        round(mismatch_pairs / total_pairs, 4) if total_pairs > 0 else 0.0
    )

    majority_level: Optional[str] = None
    if present:
        counts: Dict[str, int] = {}
        for _, lvl in present:
            counts[lvl] = counts.get(lvl, 0) + 1
        majority_level = sorted(
            counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[0][0]

    asserted_level = normalised["asserted"]
    independent_levels = [lvl for name, lvl in present if name != "asserted"]
    asserted_present = asserted_level is not None
    asserted_supported = asserted_present and any(
        lvl == asserted_level for lvl in independent_levels
    )
    asserted_unsupported = (
        asserted_present
        and len(independent_levels) >= 1
        and not asserted_supported
    )

    return {
        "votes": normalised,
        "present": present,
        "abstentions": abstentions,
        "n_present": n,
        "pairwise_total": total_pairs,
        "pairwise_mismatch": mismatch_pairs,
        "disagreement_score": disagreement_score,
        "skipped": n < 2,
        "majority_level": majority_level,
        "asserted_present": asserted_present,
        "asserted_supported": asserted_supported,
        "asserted_unsupported": asserted_unsupported,
        "independent_levels": independent_levels,
    }


__all__ = [
    "resolve_trivote_mode",
    "zero_shot_bloom",
    "verb_vote",
    "trivote_disagreement",
    "_BLOOM_HYPOTHESIS_TEMPLATES",
]
