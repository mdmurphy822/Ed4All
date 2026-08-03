"""Wave 108 — Phase B negative-grounding evaluator.

Mirrors :class:`FaithfulnessEvaluator` but operates on the
``negative_probes`` array emitted by :class:`HoldoutBuilder`. Each
probe asserts a (source, relation, target) triple that does NOT exist
in the graph; the correct response is therefore 'no'. A model that
always says 'yes' (template-recognizer regression class) scores 0.0
here regardless of how high its faithfulness score is on positive
probes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NegativeGroundingEvaluator:
    """Score a model's ability to reject (source, relation, target) probes
    that don't appear in the graph."""

    def __init__(
        self,
        holdout_split: Path,
        model_callable: Callable[[str], str],
        max_questions: Optional[int] = None,
    ) -> None:
        from Trainforge.eval.holdout_builder import load_holdout_split
        self.split = load_holdout_split(holdout_split)
        self.model_callable = model_callable
        self.max_questions = max_questions

    def evaluate(self) -> Dict[str, Any]:
        """Run the eval. Returns per-question + summary results.

        ``negative_grounding_accuracy`` is None when no negative_probes
        are present (legacy corpus); the harness folds that into the
        report and the gating validator treats it as an unscored
        signal.
        """
        from Trainforge.eval.faithfulness import _classify_response, _format_probe

        negs = self.split.get("negative_probes", []) or []
        if self.max_questions is not None:
            negs = negs[: self.max_questions]

        per_question: List[Dict[str, Any]] = []
        deny_count = 0
        affirm_count = 0
        errors: List[str] = []

        for probe in negs:
            text = _format_probe(probe)
            try:
                response = self.model_callable(text)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                per_question.append({
                    "probe": probe,
                    "prompt": text,
                    "response": None,
                    "outcome": "error",
                })
                continue

            classification = _classify_response(str(response))
            if classification == "deny":
                deny_count += 1
                outcome = "correct"
            elif classification == "affirm":
                affirm_count += 1
                outcome = "false_yes"
            else:
                outcome = "ambiguous"
            per_question.append({
                "probe": probe,
                "prompt": text,
                "response": response,
                "outcome": outcome,
            })

        scored_total = sum(
            1 for r in per_question if r["outcome"] != "error"
        )
        if scored_total == 0:
            return {
                "negative_grounding_accuracy": None,
                "false_yes_rate": 0.0,
                "scored_total": 0,
                "per_question_results": per_question,
                "errors": errors,
            }
        return {
            "negative_grounding_accuracy": deny_count / scored_total,
            "false_yes_rate": affirm_count / scored_total,
            "scored_total": scored_total,
            "per_question_results": per_question,
            "errors": errors,
        }


__all__ = ["NegativeGroundingEvaluator"]
