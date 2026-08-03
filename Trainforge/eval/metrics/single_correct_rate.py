"""Wave 3 W3.F-F2 — Single-correct-rate evaluator.

Parses the rendered Courseforge HTML for each emitted question and
counts how many ``<li>`` choices carry ``data-cf-correct="true"``. A
question is "single-correct" when exactly one choice carries the marker.

Defense-in-depth metric — the W3 P0b Fix 4 path catches this at
synthesis time (the ``is_correct`` count must be 1 for an
``assessment_item`` block to render). This evaluator probes the
rendered output one more time so any synthesis bug that emits zero or
multiple correct markers surfaces in the eval report instead of
silently passing through.

Aggregate output is folded into ``eval_report.json`` under the
top-level ``single_correct_rate`` block by
``Trainforge.eval.runners.slm_eval_harness.SLMEvalHarness._run_single_correct_rate``.

Wave 7 W7.B: the input shape changed from ``List[str]`` to
``List[Tuple[str, str]]`` of ``(html, question_type)`` so the per-
question records can carry their question_type label and the post-loop
:func:`bucket_per_question_records` projection can emit a
``per_question_type`` block scoped to MC + TF (see
:data:`Trainforge.eval.metrics._per_type_helpers.RELEVANT_QUESTION_TYPES`).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from Trainforge.eval.metrics._per_type_helpers import (
    attach_relevance,
    bucket_per_question_records,
)

logger = logging.getLogger(__name__)


#: Regex that matches every opening ``<li>`` element bearing the
#: ``data-cf-correct="true"`` attribute (single OR double-quoted).
#: Defensive against attribute-order shuffling: ``data-cf-correct``
#: may appear anywhere on the tag.
_CORRECT_LI_RE = re.compile(
    r"<li\b[^>]*\bdata-cf-correct\s*=\s*[\"']?true[\"']?[^>]*>",
    re.IGNORECASE,
)


class SingleCorrectRateEvaluator:
    """Score the fraction of rendered questions with exactly one
    ``<li data-cf-correct="true">`` choice.
    """

    def evaluate(
        self,
        rendered_html_blocks: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Count correct-marker occurrences per HTML block and aggregate.

        Args:
            rendered_html_blocks: List of ``(html, question_type)``
                tuples, one per emitted ``assessment_item`` block. The
                ``question_type`` is a lower-cased canonical string
                (e.g. ``"multiple_choice"``); the empty string marks
                an unknown / untyped question.

        Returns:
            Dict with the following keys:

            * ``single_correct_rate`` (float in [0, 1])
            * ``total_questions`` (int)
            * ``multi_correct_count`` (int — questions with >1 marker)
            * ``no_correct_count`` (int — questions with 0 markers)
            * ``per_question`` (List[Dict])
            * ``per_question_type`` (Dict[str, Dict]) — Wave 7 W7.B
              segmentation. Each bucket carries
              ``{single_correct_rate, total_questions, correct_count,
              relevant}``.
        """
        total = len(rendered_html_blocks)
        single = 0
        multi = 0
        none = 0
        per_question: List[Dict[str, Any]] = []
        question_types: List[str] = []

        for idx, entry in enumerate(rendered_html_blocks):
            # Defensive: legacy callers passing bare strings still work
            # (degraded — no per-type segmentation possible). Tuple-shape
            # is the canonical W7.B contract.
            if isinstance(entry, tuple):
                html, qt = entry
            else:
                html, qt = entry, ""
            text = html if isinstance(html, str) else ""
            qt_str = str(qt or "").lower()
            matches = _CORRECT_LI_RE.findall(text)
            count = len(matches)
            if count == 1:
                single += 1
                outcome = "single"
            elif count == 0:
                none += 1
                outcome = "none"
            else:
                multi += 1
                outcome = "multi"
            per_question.append({
                "index": idx,
                "correct_count": count,
                "outcome": outcome,
                "question_type": qt_str,
            })
            question_types.append(qt_str)

        rate = (single / total) if total > 0 else 0.0

        # W7.B: per-question-type segmentation. Bucket the parallel-
        # indexed per_question records by question_type, drop the
        # empty-string ("type unknown") bucket from the per-type emit,
        # and stamp `relevant: bool` per the canonical relevance table
        # (single_correct_rate is structurally MC + TF only).
        buckets = bucket_per_question_records(
            per_question, question_types=question_types
        )
        per_question_type: Dict[str, Dict[str, Any]] = {}
        for qt_key, bucket in buckets.items():
            if not qt_key:
                continue  # skip the empty-string bucket from per-type emit
            correct_count = sum(1 for r in bucket if r.get("outcome") == "single")
            bucket_total = len(bucket)
            per_question_type[qt_key] = {
                "single_correct_rate": round(correct_count / bucket_total, 4)
                if bucket_total
                else 0.0,
                "total_questions": int(bucket_total),
                "correct_count": int(correct_count),
            }
        attach_relevance(per_question_type, metric_name="single_correct_rate")

        return {
            "single_correct_rate": round(rate, 4),
            "total_questions": int(total),
            "multi_correct_count": int(multi),
            "no_correct_count": int(none),
            "per_question": per_question,
            "per_question_type": per_question_type,
        }


__all__ = ["SingleCorrectRateEvaluator"]
