"""Wave 3 W3.F-F1 — Answerable-rate evaluator.

Per-question Jaccard token overlap of the correct-answer text against
the union of cited chunk texts. A question is "answerable" when the
overlap is at least :data:`_DEFAULT_THRESHOLD` (``0.30``) — the same
floor the source-grounding gate (``DEFAULT_MIN_OVERLAP_JACCARD`` in
``lib/validators/assessment_retrieval_grounding.py``) uses for
authored-content blocks.

The single source of truth for the token-set + Jaccard helpers is
``lib/validators/assessment_retrieval_grounding.py::_content_tokens``
and ``_jaccard``. Reusing them keeps eval-time and gate-time scoring
byte-identical so a question that passed the synthesis gate doesn't
silently fail the eval probe (and vice-versa).

Aggregate output is folded into ``eval_report.json`` under the top-level
``answerable_rate`` block by
``Trainforge.eval.runners.slm_eval_harness.SLMEvalHarness._run_answerable_rate``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from lib.validators.assessment_retrieval_grounding import (
    _content_tokens,
    _jaccard,
)
from Trainforge.eval.metrics._per_type_helpers import (
    attach_relevance,
    bucket_per_question_records,
)

logger = logging.getLogger(__name__)


#: Per-question Jaccard floor below which a question is "unanswerable
#: from cited sources". Mirrors the ``DEFAULT_MIN_OVERLAP_JACCARD``
#: constant on the synthesis-side validator so the eval signal can't
#: drift below the gate's authoring floor.
_DEFAULT_THRESHOLD: float = 0.30


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class AnswerableRateEvaluator:
    """Score the fraction of questions whose correct-answer is grounded
    in the cited chunks.

    Args:
        threshold: Jaccard overlap floor for "answerable" classification.
            Defaults to :data:`_DEFAULT_THRESHOLD` (``0.30``).
    """

    def __init__(self, *, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = float(threshold)

    def evaluate(
        self,
        prompts: List[Dict[str, Any]],
        chunk_corpus: Dict[str, str],
    ) -> Dict[str, Any]:
        """Score every prompt and return aggregate + per-question signals.

        Args:
            prompts: List of question dicts. Each entry MAY carry the
                following keys (all optional, missing fields degrade
                gracefully):

                * ``correct_answer`` (str) — the canonical answer text.
                * ``source_chunks`` (List[str]) — cited chunk_ids.
                * ``question_id`` (str) — for per-question reporting.
            chunk_corpus: Mapping of ``chunk_id -> chunk_text`` for
                resolving the cited chunks. Missing chunk_ids are
                logged at debug level and excluded from the union.

        Returns:
            Dict with three keys:

            * ``answerable_rate`` (float in [0, 1])
            * ``total_questions`` (int)
            * ``answerable_count`` (int)
        """
        total = len(prompts)
        answerable = 0
        per_question: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts):
            answer_text = _coerce_str(prompt.get("correct_answer"))
            chunk_ids = prompt.get("source_chunks") or []
            if not isinstance(chunk_ids, list):
                chunk_ids = []
            answer_tokens = _content_tokens(answer_text)
            chunk_tokens: set = set()
            missing: List[str] = []
            for cid in chunk_ids:
                cid_str = _coerce_str(cid)
                text = chunk_corpus.get(cid_str)
                if text is None:
                    missing.append(cid_str)
                    continue
                chunk_tokens |= _content_tokens(text)
            overlap = _jaccard(answer_tokens, chunk_tokens)
            is_answerable = overlap >= self._threshold
            if is_answerable:
                answerable += 1
            per_question.append({
                "question_id": prompt.get("question_id") or f"q-{idx}",
                "overlap": round(float(overlap), 4),
                "answerable": bool(is_answerable),
                "missing_chunks": missing,
            })

        rate = (answerable / total) if total > 0 else 0.0

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
            answered = sum(1 for r in bucket if r.get("answerable"))
            bucket_total = len(bucket)
            per_question_type[qt] = {
                "answerable_rate": round(answered / bucket_total, 4)
                if bucket_total
                else 0.0,
                "total_questions": int(bucket_total),
                "answerable_count": int(answered),
            }
        attach_relevance(per_question_type, metric_name="answerable_rate")

        return {
            "answerable_rate": round(rate, 4),
            "total_questions": int(total),
            "answerable_count": int(answerable),
            "threshold": self._threshold,
            "per_question": per_question,
            "per_question_type": per_question_type,
        }


__all__ = ["AnswerableRateEvaluator"]
