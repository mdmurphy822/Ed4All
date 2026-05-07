"""Shared NLI entailment axis for objective-delivery validators.

Used by :mod:`lib.validators.block_objective_delivery` and
:mod:`lib.validators.pair.objective_delivery` — the algorithm is the
same single-(premise, hypothesis) NLI score with a contradiction-floor
gate. Per-block-type / per-pair-kind threshold tables are passed in by
the caller; this module knows nothing about block_type vs pair_kind.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.3.2.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["score_nli_axis"]


def score_nli_axis(
    *,
    nli: Any,
    text: str,
    statement: Optional[str],
    entailment_threshold: float,
    contradiction_floor: float,
) -> Tuple[Optional[bool], Optional[float], Optional[float]]:
    """Score one (text, statement) pair via the NLI classifier.

    Returns ``(passed, entailment_score, contradiction_score)``.

    * ``passed=None`` when the axis was skipped (NLI loader unavailable,
      missing statement, or NLI exception). Caller decides whether to
      stamp UNVERIFIABLE.
    * ``passed=False`` when ``entailment < entailment_threshold AND
      contradiction > contradiction_floor`` (the canonical W1.7.C /
      W4.C miss criterion).
    * ``passed=True`` otherwise.
    """
    if nli is None or not (statement and text):
        return None, None, None
    try:
        score = nli.score_pair(premise=text, hypothesis=statement)
    except Exception as exc:  # noqa: BLE001 -- caller's defensive boundary
        logger.warning("NliClassifier.score_pair raised: %s", exc)
        return None, None, None
    entailment = float(score.entailment)
    contradiction = float(score.contradiction)
    if entailment < entailment_threshold and contradiction > contradiction_floor:
        return False, entailment, contradiction
    return True, entailment, contradiction
