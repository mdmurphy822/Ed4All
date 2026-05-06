"""Wave 3 W3.F-F3 — Distractor-entropy evaluator.

Per-question Shannon entropy of distractor token-set sizes; aggregate
mean across the corpus. Low entropy is the signal for "distractors
collapsed" (every distractor has the same length / vocabulary, so the
adapter can't tell them apart from the correct answer at training
time).

Reuses ``_tokenise`` from ``lib/validators/distractor_plausibility.py``
as the single source of truth for the distractor token set, so the eval
signal stays bytewise-aligned with the synthesis-time gate.

Aggregate output is folded into ``eval_report.json`` under the
top-level ``distractor_entropy`` block by
``Trainforge.eval.slm_eval_harness.SLMEvalHarness._run_distractor_entropy``.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from lib.validators.distractor_plausibility import _tokenise

logger = logging.getLogger(__name__)


#: Per-question entropy floor below which a question is flagged as
#: low-entropy (distractors collapsed). Calibrated against the
#: rdf-shacl-551-2 corpus: well-spread distractors hit ~1.0+, near-
#: degenerate ones drop below 0.5.
_LOW_ENTROPY_THRESHOLD: float = 0.5


def _shannon_entropy(values: List[int]) -> float:
    """Shannon entropy (in nats) of a discrete distribution given by
    the token-set sizes of each distractor.

    Each distractor contributes one bucket whose probability is its
    size / total sum. Empty input → entropy 0; single-bucket input →
    entropy 0 (no spread).
    """
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for v in values:
        if v <= 0:
            continue
        p = v / total
        entropy -= p * math.log(p)
    return entropy


class DistractorEntropyEvaluator:
    """Score the mean Shannon entropy of distractor token-set sizes.

    Args:
        low_entropy_threshold: Per-question entropy floor below which
            the question is counted as ``low_entropy``. Defaults to
            :data:`_LOW_ENTROPY_THRESHOLD` (``0.5``).
    """

    def __init__(
        self,
        *,
        low_entropy_threshold: float = _LOW_ENTROPY_THRESHOLD,
    ) -> None:
        self._low_threshold = float(low_entropy_threshold)

    def evaluate(
        self,
        prompts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score every prompt and return aggregate + per-question signals.

        Args:
            prompts: List of question dicts. Each entry SHOULD carry a
                ``distractors`` key whose value is a list of distractor
                strings. Missing / empty distractor lists yield entropy
                ``0.0`` and are counted in ``low_entropy_count``.

        Returns:
            Dict with three keys:

            * ``mean_distractor_entropy`` (float)
            * ``low_entropy_count`` (int — questions below threshold)
            * ``total_questions`` (int)
        """
        total = len(prompts)
        entropies: List[float] = []
        low_count = 0
        per_question: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts):
            distractors = prompt.get("distractors") or []
            if not isinstance(distractors, list):
                distractors = []
            sizes = [len(_tokenise(str(d))) for d in distractors]
            ent = _shannon_entropy(sizes)
            entropies.append(ent)
            is_low = ent < self._low_threshold
            if is_low:
                low_count += 1
            per_question.append({
                "question_id": prompt.get("question_id") or f"q-{idx}",
                "entropy": round(float(ent), 4),
                "distractor_count": len(distractors),
                "low_entropy": bool(is_low),
            })

        mean_entropy = (
            sum(entropies) / len(entropies) if entropies else 0.0
        )
        return {
            "mean_distractor_entropy": round(float(mean_entropy), 4),
            "low_entropy_count": int(low_count),
            "total_questions": int(total),
            "low_entropy_threshold": self._low_threshold,
            "per_question": per_question,
        }


__all__ = ["DistractorEntropyEvaluator"]
