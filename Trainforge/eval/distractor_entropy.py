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
from Trainforge.eval._per_type_helpers import (
    attach_relevance,
    bucket_per_question_records,
)

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

        # W7.B: per-question-type segmentation. Bucket the parallel-
        # indexed per_question records by question_type, drop the
        # empty-string ("type unknown") bucket from the per-type emit,
        # and stamp `relevant: bool` per the canonical relevance table.
        # distractor_entropy is structurally MC-only — non-MC buckets
        # have entropy=0 (single distractor or none), and `attach_relevance`
        # marks them not relevant so the W7.D gate can skip them.
        question_types = [str(p.get("question_type") or "") for p in prompts]
        buckets = bucket_per_question_records(
            per_question, question_types=question_types
        )
        per_question_type: Dict[str, Dict[str, Any]] = {}
        for qt, bucket in buckets.items():
            if not qt:
                continue  # skip the empty-string bucket from per-type emit
            bucket_total = len(bucket)
            entropy_sum = sum(float(r.get("entropy") or 0.0) for r in bucket)
            mean_bucket_entropy = (
                entropy_sum / bucket_total if bucket_total else 0.0
            )
            per_question_type[qt] = {
                "distractor_entropy_mean": round(float(mean_bucket_entropy), 4),
                "total_questions": int(bucket_total),
            }
        attach_relevance(per_question_type, metric_name="distractor_entropy")

        return {
            "mean_distractor_entropy": round(float(mean_entropy), 4),
            "low_entropy_count": int(low_count),
            "total_questions": int(total),
            "low_entropy_threshold": self._low_threshold,
            "per_question": per_question,
            "per_question_type": per_question_type,
        }


__all__ = ["DistractorEntropyEvaluator"]
