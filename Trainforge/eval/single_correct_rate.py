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
``Trainforge.eval.slm_eval_harness.SLMEvalHarness._run_single_correct_rate``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

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
        rendered_html_blocks: List[str],
    ) -> Dict[str, Any]:
        """Count correct-marker occurrences per HTML block and aggregate.

        Args:
            rendered_html_blocks: List of HTML strings, one per emitted
                ``assessment_item`` block.

        Returns:
            Dict with four keys:

            * ``single_correct_rate`` (float in [0, 1])
            * ``total_questions`` (int)
            * ``multi_correct_count`` (int — questions with >1 marker)
            * ``no_correct_count`` (int — questions with 0 markers)
        """
        total = len(rendered_html_blocks)
        single = 0
        multi = 0
        none = 0
        per_question: List[Dict[str, Any]] = []

        for idx, html in enumerate(rendered_html_blocks):
            text = html if isinstance(html, str) else ""
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
            })

        rate = (single / total) if total > 0 else 0.0
        return {
            "single_correct_rate": round(rate, 4),
            "total_questions": int(total),
            "multi_correct_count": int(multi),
            "no_correct_count": int(none),
            "per_question": per_question,
        }


__all__ = ["SingleCorrectRateEvaluator"]
