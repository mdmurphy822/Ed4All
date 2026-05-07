"""Wave 3 W3.F-F5 — Placeholder-rate evaluator.

Per-question scan of every text field (stem, distractors,
correct_answer, feedback) against the canonical 13-regex catalog at
:data:`lib.validators.assessment.ASSESSMENT_PLACEHOLDER_PATTERNS`. A
question is "contaminated" when any field hits any pattern.

Defense-in-depth metric — the W3 P0b Fix 6 path catches placeholder
strings at synthesis time. This evaluator probes the assembled corpus
one more time so any residual contamination shows up in the eval
report instead of silently passing into a training run.

No model calls; pure regex over the canonical pattern catalog.

Aggregate output is folded into ``eval_report.json`` under the
top-level ``placeholder_rate`` block by
``Trainforge.eval.slm_eval_harness.SLMEvalHarness._run_placeholder_rate``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from lib.validators.assessment import ASSESSMENT_PLACEHOLDER_PATTERNS
from Trainforge.eval._per_type_helpers import (
    attach_relevance,
    bucket_per_question_records,
)

logger = logging.getLogger(__name__)


#: Canonical fields scanned per question. Stem + correct_answer +
#: feedback are scalar text; distractors is a list[str].
_SCALAR_FIELDS = ("stem", "correct_answer", "feedback")


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class PlaceholderRateEvaluator:
    """Scan every question for placeholder-pattern contamination.

    Reuses the canonical 13-regex catalog at
    :data:`ASSESSMENT_PLACEHOLDER_PATTERNS` so the evaluator and the
    synthesis-time gate stay bytewise-aligned.
    """

    def evaluate(
        self,
        prompts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Scan every prompt's fields for placeholder hits.

        Args:
            prompts: List of question dicts. Each entry MAY carry:

                * ``stem`` (str)
                * ``distractors`` (List[str])
                * ``correct_answer`` (str)
                * ``feedback`` (str)
                * ``question_id`` (str) — for per-question reporting.

        Returns:
            Dict with four keys:

            * ``placeholder_rate`` (float in [0, 1])
            * ``total_questions`` (int)
            * ``contaminated_count`` (int)
            * ``hit_pattern_histogram`` (Dict[str, int]) — pattern ->
              total hit count across the corpus.
        """
        total = len(prompts)
        contaminated = 0
        histogram: Dict[str, int] = {}
        per_question: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts):
            scan_texts: List[str] = []
            for field in _SCALAR_FIELDS:
                scan_texts.append(_coerce_str(prompt.get(field)))
            distractors = prompt.get("distractors") or []
            if isinstance(distractors, list):
                for d in distractors:
                    scan_texts.append(_coerce_str(d))

            hits: List[str] = []
            for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
                pat_str = pattern.pattern
                hit_count = 0
                for text in scan_texts:
                    if not text:
                        continue
                    matches = pattern.findall(text)
                    hit_count += len(matches)
                if hit_count > 0:
                    hits.append(pat_str)
                    histogram[pat_str] = histogram.get(pat_str, 0) + hit_count
            if hits:
                contaminated += 1
            per_question.append({
                "question_id": prompt.get("question_id") or f"q-{idx}",
                "contaminated": bool(hits),
                "hit_patterns": hits,
            })

        rate = (contaminated / total) if total > 0 else 0.0

        # W7.A: per-question-type segmentation. Bucket the parallel-
        # indexed per_question records by question_type, drop the
        # empty-string ("type unknown") bucket from the per-type emit,
        # and stamp `relevant: bool` per the canonical relevance table.
        question_types = [str(p.get("question_type") or "") for p in prompts]
        buckets = bucket_per_question_records(
            per_question, question_types=question_types
        )
        per_question_type: Dict[str, Dict[str, Any]] = {}
        for qt, bucket in buckets.items():
            if not qt:
                continue  # skip the empty-string bucket from per-type emit
            placeholder_count = sum(1 for r in bucket if r.get("contaminated"))
            bucket_total = len(bucket)
            per_question_type[qt] = {
                "placeholder_rate": round(placeholder_count / bucket_total, 4)
                if bucket_total
                else 0.0,
                "total_questions": int(bucket_total),
                "placeholder_count": int(placeholder_count),
            }
        attach_relevance(per_question_type, metric_name="placeholder_rate")

        return {
            "placeholder_rate": round(rate, 4),
            "total_questions": int(total),
            "contaminated_count": int(contaminated),
            "hit_pattern_histogram": histogram,
            "per_question": per_question,
            "per_question_type": per_question_type,
        }


__all__ = ["PlaceholderRateEvaluator"]
