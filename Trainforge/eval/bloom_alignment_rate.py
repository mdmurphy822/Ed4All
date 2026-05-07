"""Wave 3 W3.F-F4 — Bloom alignment-rate evaluator.

Per-question Bloom-level classification via the
:class:`lib.classifiers.bloom_bert_ensemble.BloomBertEnsemble` 3-member
ensemble. A question is "aligned" when the ensemble winner matches its
declared ``bloom_level``. Mirrors W2.E's per-block binary check at
corpus scope: the same classifier that gates synthesis-time blocks
re-runs against the eval probe stems so a corpus-scale alignment
regression surfaces in the eval report.

Graceful-degrade contract: when the BERT extras are absent (``[bert]``
extras not installed) or the ensemble fails to load any member,
:meth:`evaluate` returns ``{bloom_alignment_rate: None,
deps_missing: True}`` with a warning log instead of crashing the
harness. Mirrors the surface used by
:mod:`lib.validators.bloom_classifier_disagreement`.

Aggregate output is folded into ``eval_report.json`` under the
top-level ``bloom_alignment_rate`` block by
``Trainforge.eval.slm_eval_harness.SLMEvalHarness._run_bloom_alignment_rate``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from Trainforge.eval._per_type_helpers import (
    attach_relevance,
    bucket_per_question_records,
)

logger = logging.getLogger(__name__)


class BloomAlignmentRateEvaluator:
    """Score the fraction of questions whose ensemble-classified Bloom
    level matches the declared ``bloom_level``.

    Args:
        ensemble: Optional pre-instantiated
            :class:`BloomBertEnsemble`. When ``None``, the evaluator
            lazy-instantiates one on the first :meth:`evaluate` call.
            Tests inject a stub here to skip the heavy model load.
    """

    def __init__(self, ensemble: Any = None) -> None:
        self._ensemble = ensemble

    def _get_ensemble(self) -> Any:
        if self._ensemble is not None:
            return self._ensemble
        try:
            from lib.classifiers.bloom_bert_ensemble import (
                BloomBertEnsemble,
            )
        except ImportError as exc:
            logger.warning(
                "BloomAlignmentRateEvaluator: "
                "BloomBertEnsemble import failed (%s); deps missing.",
                exc,
            )
            return None
        try:
            self._ensemble = BloomBertEnsemble()
        except Exception as exc:  # noqa: BLE001 — graceful degrade
            logger.warning(
                "BloomAlignmentRateEvaluator: "
                "BloomBertEnsemble instantiation failed (%s); deps missing.",
                exc,
            )
            return None
        return self._ensemble

    def evaluate(
        self,
        prompts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Classify every question's stem and tally alignment.

        Args:
            prompts: List of question dicts. Each entry MAY carry:

                * ``stem`` (str) — question text to classify.
                * ``bloom_level`` (str) — declared canonical level.
                * ``question_id`` (str) — for per-question reporting.

                Questions missing ``bloom_level`` are skipped (not
                counted in either ``aligned_count`` or
                ``mismatched_count``).

        Returns:
            Dict carrying the alignment rate. When BERT extras are
            absent, returns
            ``{"bloom_alignment_rate": None, "deps_missing": True,
            "total_questions": <n>}``.
        """
        ensemble = self._get_ensemble()
        if ensemble is None:
            # W7.B: deps-missing branch must emit per_question_type=None
            # so the W7.D gate consumer can short-circuit cleanly without
            # crashing on a missing key.
            return {
                "bloom_alignment_rate": None,
                "deps_missing": True,
                "total_questions": len(prompts),
                "aligned_count": 0,
                "mismatched_count": 0,
                "per_question_type": None,
            }

        # Probe ensemble members loadability before classifying — this
        # mirrors the bloom_classifier_disagreement validator's pattern
        # and keeps the deps-missing path uniform across surfaces.
        try:
            members = ensemble._load_members()
        except Exception as exc:  # noqa: BLE001 — graceful degrade
            logger.warning(
                "BloomAlignmentRateEvaluator: "
                "ensemble._load_members() raised %s; deps missing.",
                exc,
            )
            return {
                "bloom_alignment_rate": None,
                "deps_missing": True,
                "total_questions": len(prompts),
                "aligned_count": 0,
                "mismatched_count": 0,
                "per_question_type": None,
            }
        if not members:
            return {
                "bloom_alignment_rate": None,
                "deps_missing": True,
                "total_questions": len(prompts),
                "aligned_count": 0,
                "mismatched_count": 0,
                "per_question_type": None,
            }

        aligned = 0
        mismatched = 0
        skipped = 0
        per_question: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts):
            declared = (prompt.get("bloom_level") or "").strip().lower()
            stem = prompt.get("stem") or ""
            qid = prompt.get("question_id") or f"q-{idx}"
            if not declared:
                skipped += 1
                per_question.append({
                    "question_id": qid,
                    "outcome": "skipped",
                    "reason": "no_declared_bloom_level",
                })
                continue
            try:
                result = ensemble.classify(str(stem))
            except Exception as exc:  # noqa: BLE001 — graceful degrade
                logger.warning(
                    "BloomAlignmentRateEvaluator: ensemble.classify "
                    "raised on %s (%s); skipping.",
                    qid, exc,
                )
                skipped += 1
                per_question.append({
                    "question_id": qid,
                    "outcome": "skipped",
                    "reason": "classify_error",
                })
                continue
            winner = (result.get("winner_level") or "").strip().lower()
            is_aligned = winner == declared and winner != "unknown"
            if is_aligned:
                aligned += 1
                outcome = "aligned"
            else:
                mismatched += 1
                outcome = "mismatched"
            per_question.append({
                "question_id": qid,
                "declared_level": declared,
                "winner_level": winner,
                "winner_score": float(result.get("winner_score") or 0.0),
                "outcome": outcome,
            })

        denom = aligned + mismatched
        rate: Optional[float] = (
            (aligned / denom) if denom > 0 else 0.0
        )

        # W7.B: per-question-type segmentation. Bucket the per_question
        # records (filtered to only those that produced an aligned /
        # mismatched outcome — skipped records aren't counted in the
        # rate denom and shouldn't pollute per-bucket rates either).
        # Stamp `relevant: bool` per the canonical relevance table —
        # bloom_alignment_rate is relevant across all 5 question types
        # so every bucket emits relevant=True.
        question_types = [str(p.get("question_type") or "") for p in prompts]
        scored_records: List[Dict[str, Any]] = []
        scored_types: List[str] = []
        for record, qt in zip(per_question, question_types):
            outcome = record.get("outcome")
            if outcome in ("aligned", "mismatched"):
                scored_records.append(record)
                scored_types.append(qt)
        buckets = bucket_per_question_records(
            scored_records, question_types=scored_types
        )
        per_question_type: Dict[str, Dict[str, Any]] = {}
        for qt_key, bucket in buckets.items():
            if not qt_key:
                continue  # skip the empty-string bucket from per-type emit
            aligned_count = sum(1 for r in bucket if r.get("outcome") == "aligned")
            bucket_total = len(bucket)
            per_question_type[qt_key] = {
                "bloom_alignment_rate": round(aligned_count / bucket_total, 4)
                if bucket_total
                else 0.0,
                "total_questions": int(bucket_total),
                "aligned_count": int(aligned_count),
            }
        attach_relevance(per_question_type, metric_name="bloom_alignment_rate")

        return {
            "bloom_alignment_rate": round(float(rate), 4),
            "total_questions": int(len(prompts)),
            "aligned_count": int(aligned),
            "mismatched_count": int(mismatched),
            "skipped_count": int(skipped),
            "deps_missing": False,
            "per_question": per_question,
            "per_question_type": per_question_type,
        }


__all__ = ["BloomAlignmentRateEvaluator"]
