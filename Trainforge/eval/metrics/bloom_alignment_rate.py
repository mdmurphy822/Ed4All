"""Bloom alignment-rate evaluator with explicit classifier abstention.

The evaluator retains the historical
:class:`lib.classifiers.bloom_bert_ensemble.BloomBertEnsemble` API. No
reliable classifier is currently provisioned and default member dispatch is
unimplemented, so the compatibility scaffold loads no members and the metric
returns ``bloom_alignment_rate=None``. Injected implementations can still use
the existing alignment contract: a question is aligned when the classifier
winner matches its declared ``bloom_level``. The configured MultiBERT training
path is staged but remains unproven and unavailable to this evaluator.

Abstention contract: when no classifier members are available,
:meth:`evaluate` returns ``{bloom_alignment_rate: None,
deps_missing: True}`` rather than inventing a score. The compatibility field
name ``deps_missing`` also covers the presently unprovisioned classifier.
This mirrors the surface used by
:mod:`lib.validators.bloom.classifier_disagreement`.

Aggregate output is folded into ``eval_report.json`` under the
top-level ``bloom_alignment_rate`` block by
``Trainforge.eval.runners.slm_eval_harness.SLMEvalHarness._run_bloom_alignment_rate``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from Trainforge.eval.metrics._per_type_helpers import (
    attach_relevance,
    bucket_per_question_records,
)

logger = logging.getLogger(__name__)


class BloomAlignmentRateEvaluator:
    """Score the fraction of questions whose classifier-assigned Bloom
    level matches the declared ``bloom_level``.

    Args:
        ensemble: Optional pre-instantiated
            :class:`BloomBertEnsemble`. When ``None``, the evaluator
            lazy-instantiates one on the first :meth:`evaluate` call.
            Tests may inject a stub that provides actual votes.
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
        """Classify question stems when a classifier is available.

        Args:
            prompts: List of question dicts. Each entry MAY carry:

                * ``stem`` (str) — question text to classify.
                * ``bloom_level`` (str) — declared canonical level.
                * ``question_id`` (str) — for per-question reporting.

                Questions missing ``bloom_level`` are skipped (not
                counted in either ``aligned_count`` or
                ``mismatched_count``).

        Returns:
            Dict carrying the alignment rate. When the compatibility
            scaffold has no usable classifier, returns
            ``{"bloom_alignment_rate": None, "deps_missing": True,
            "total_questions": <n>}``.
        """
        ensemble = self._get_ensemble()
        if ensemble is None:
            # The unavailable branch keeps ``per_question_type=None`` so
            # gate consumers can distinguish abstention from scored output.
            return {
                "bloom_alignment_rate": None,
                "deps_missing": True,
                "total_questions": len(prompts),
                "aligned_count": 0,
                "mismatched_count": 0,
                "per_question_type": None,
            }

        # Probe availability before classifying. The default compatibility
        # scaffold returns no members because reliable dispatch is not
        # provisioned; injected implementations may provide members.
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

        # Bucket per-question records that produced an aligned /
        # mismatched outcome — skipped records aren't counted in the
        # rate denom and shouldn't pollute per-bucket rates either).
        # Stamp `relevant: bool` per the canonical relevance table;
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
