"""Wave 3 W3.F-F6 — Source-support-rate evaluator.

Per-question NLI-driven entailment of the question stem + correct
answer (the hypothesis) against the union of the cited chunks (the
premise). A question is "supported" when the entailment probability is
at least :data:`_DEFAULT_ENTAILMENT_THRESHOLD` (``0.5``) AND the
contradiction probability is below
:data:`_DEFAULT_CONTRADICTION_THRESHOLD` (``0.3``).

Reuses :class:`lib.classifiers.nli_classifier.NliClassifier` (the
singleton-loaded DeBERTa-v3-base-mnli-fever-anli wrapper extracted in
Wave 2 W2.F) so the eval and the validator surfaces stay aligned on
the same pinned model revision.

Sibling — not replacement — to
:class:`Trainforge.eval.source_match.SourceMatchEvaluator`. Source-match
checks citation presence (did the model emit a chunk_id?);
source-support checks semantic entailment (do the cited chunks
actually support the answer?). A model can pass source-match while
flunking source-support (citing chunks that don't entail the claim).

Graceful-degrade contract: when transformers / torch extras are absent
or :meth:`NliClassifier.get_or_load` returns ``None``, this evaluator
returns ``{source_support_rate: None, deps_missing: True}`` with a
warning log instead of crashing the harness.

Aggregate output is folded into ``eval_report.json`` under the
top-level ``source_support_rate`` block by
``Trainforge.eval.slm_eval_harness.SLMEvalHarness._run_source_support_rate``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


#: Entailment-probability floor for "supported" classification.
_DEFAULT_ENTAILMENT_THRESHOLD: float = 0.5
#: Contradiction-probability ceiling for "supported" classification.
#: A pair scoring high entailment AND high contradiction is unstable;
#: we require contradiction below this threshold for a clean
#: "supported" verdict.
_DEFAULT_CONTRADICTION_THRESHOLD: float = 0.3


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class SourceSupportRateEvaluator:
    """Score the fraction of questions whose cited chunks entail the
    stem + correct answer via the NLI classifier.

    Args:
        classifier: Optional pre-loaded
            :class:`lib.classifiers.nli_classifier.NliClassifier`. When
            ``None``, the evaluator calls :meth:`NliClassifier.get_or_load`
            on the first :meth:`evaluate` call. Tests inject a stub here
            to skip the heavy model load.
        entailment_threshold: Entailment-probability floor. Defaults to
            :data:`_DEFAULT_ENTAILMENT_THRESHOLD` (``0.5``).
        contradiction_threshold: Contradiction-probability ceiling.
            Defaults to :data:`_DEFAULT_CONTRADICTION_THRESHOLD`
            (``0.3``).
    """

    def __init__(
        self,
        classifier: Any = None,
        *,
        entailment_threshold: float = _DEFAULT_ENTAILMENT_THRESHOLD,
        contradiction_threshold: float = _DEFAULT_CONTRADICTION_THRESHOLD,
    ) -> None:
        self._classifier = classifier
        self._ent_threshold = float(entailment_threshold)
        self._con_threshold = float(contradiction_threshold)

    def _get_classifier(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            from lib.classifiers.nli_classifier import NliClassifier
        except ImportError as exc:
            logger.warning(
                "SourceSupportRateEvaluator: NliClassifier import "
                "failed (%s); deps missing.",
                exc,
            )
            return None
        try:
            self._classifier = NliClassifier.get_or_load()
        except Exception as exc:  # noqa: BLE001 — graceful degrade
            logger.warning(
                "SourceSupportRateEvaluator: NliClassifier.get_or_load "
                "raised %s; deps missing.",
                exc,
            )
            return None
        return self._classifier

    def evaluate(
        self,
        prompts: List[Dict[str, Any]],
        chunk_corpus: Dict[str, str],
    ) -> Dict[str, Any]:
        """Score every prompt and return aggregate + per-question signals.

        Args:
            prompts: List of question dicts. Each entry MAY carry:

                * ``stem`` (str)
                * ``correct_answer`` (str)
                * ``source_chunks`` (List[str]) — cited chunk_ids.
                * ``question_id`` (str)
            chunk_corpus: Mapping of ``chunk_id -> chunk_text``.

        Returns:
            Dict carrying the support rate. When NLI extras are absent,
            returns
            ``{"source_support_rate": None, "deps_missing": True,
            "total_questions": <n>}``.
        """
        classifier = self._get_classifier()
        if classifier is None:
            return {
                "source_support_rate": None,
                "deps_missing": True,
                "total_questions": len(prompts),
                "supported_count": 0,
                "contradicted_count": 0,
            }

        supported = 0
        contradicted = 0
        skipped = 0
        per_question: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts):
            qid = prompt.get("question_id") or f"q-{idx}"
            chunk_ids = prompt.get("source_chunks") or []
            if not isinstance(chunk_ids, list) or not chunk_ids:
                skipped += 1
                per_question.append({
                    "question_id": qid,
                    "outcome": "skipped",
                    "reason": "no_source_chunks",
                })
                continue
            stem = _coerce_str(prompt.get("stem"))
            answer = _coerce_str(prompt.get("correct_answer"))
            hypothesis = (stem + " " + answer).strip()
            if not hypothesis:
                skipped += 1
                per_question.append({
                    "question_id": qid,
                    "outcome": "skipped",
                    "reason": "empty_hypothesis",
                })
                continue
            premise_parts: List[str] = []
            for cid in chunk_ids:
                text = chunk_corpus.get(_coerce_str(cid))
                if text:
                    premise_parts.append(_coerce_str(text))
            if not premise_parts:
                skipped += 1
                per_question.append({
                    "question_id": qid,
                    "outcome": "skipped",
                    "reason": "no_chunk_text",
                })
                continue
            premise = "\n".join(premise_parts)
            try:
                score = classifier.score_pair(
                    premise=premise, hypothesis=hypothesis
                )
            except Exception as exc:  # noqa: BLE001 — graceful degrade
                logger.warning(
                    "SourceSupportRateEvaluator: classifier.score_pair "
                    "raised on %s (%s); skipping.",
                    qid, exc,
                )
                skipped += 1
                per_question.append({
                    "question_id": qid,
                    "outcome": "skipped",
                    "reason": "score_error",
                })
                continue
            ent = float(getattr(score, "entailment", 0.0))
            con = float(getattr(score, "contradiction", 0.0))
            is_supported = (
                ent >= self._ent_threshold and con < self._con_threshold
            )
            is_contradicted = con >= self._con_threshold
            if is_supported:
                supported += 1
                outcome = "supported"
            elif is_contradicted:
                contradicted += 1
                outcome = "contradicted"
            else:
                outcome = "unsupported"
            per_question.append({
                "question_id": qid,
                "entailment": round(ent, 4),
                "contradiction": round(con, 4),
                "outcome": outcome,
            })

        denom = len(prompts) - skipped
        rate: Optional[float] = (
            (supported / denom) if denom > 0 else 0.0
        )
        return {
            "source_support_rate": round(float(rate), 4),
            "total_questions": int(len(prompts)),
            "supported_count": int(supported),
            "contradicted_count": int(contradicted),
            "skipped_count": int(skipped),
            "deps_missing": False,
            "entailment_threshold": self._ent_threshold,
            "contradiction_threshold": self._con_threshold,
            "per_question": per_question,
        }


__all__ = ["SourceSupportRateEvaluator"]
