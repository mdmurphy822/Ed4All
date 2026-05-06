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
            return {
                "bloom_alignment_rate": None,
                "deps_missing": True,
                "total_questions": len(prompts),
                "aligned_count": 0,
                "mismatched_count": 0,
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
            }
        if not members:
            return {
                "bloom_alignment_rate": None,
                "deps_missing": True,
                "total_questions": len(prompts),
                "aligned_count": 0,
                "mismatched_count": 0,
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
        return {
            "bloom_alignment_rate": round(float(rate), 4),
            "total_questions": int(len(prompts)),
            "aligned_count": int(aligned),
            "mismatched_count": int(mismatched),
            "skipped_count": int(skipped),
            "deps_missing": False,
            "per_question": per_question,
        }


__all__ = ["BloomAlignmentRateEvaluator"]
