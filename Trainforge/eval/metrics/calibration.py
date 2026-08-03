"""Wave 92 — Layer 3 calibration / Expected Calibration Error.

Asks the model to attach a confidence to each answer and compares the
distribution of stated confidences to the actual accuracy. A
well-calibrated model says "I'm 70% sure" exactly when it's right
70% of the time.

The eval reuses the held-out probes from
:class:`FaithfulnessEvaluator`'s prompt set and wraps each in a
confidence-eliciting envelope:

    "Answer with 'yes' or 'no', then on a new line state your confidence
    on a 0-100 scale. {probe}"

Responses are parsed for both an answer-token and a number; ECE is
binned across confidence levels.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


_CONFIDENCE_PATTERN = re.compile(
    r"(?:confidence|certainty|sure)[^\d\n]*?(\d{1,3})\s*(?:%|out of 100)?",
    re.IGNORECASE,
)
_BARE_PERCENT_PATTERN = re.compile(r"\b(\d{1,3})\s*(?:%|out of 100)\b", re.IGNORECASE)
_AFFIRMATIVE = re.compile(r"\b(yes|true|correct)\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"\b(no|false|incorrect|wrong)\b", re.IGNORECASE)


def _parse_response(response: str) -> Dict[str, Any]:
    """Extract (answer, confidence_0_to_1) from a free-form response."""
    affirm = bool(_AFFIRMATIVE.search(response))
    deny = bool(_NEGATIVE.search(response))
    answer: Optional[str]
    if affirm and not deny:
        answer = "yes"
    elif deny and not affirm:
        answer = "no"
    else:
        answer = None

    confidence: Optional[float] = None
    m = _CONFIDENCE_PATTERN.search(response)
    if m:
        try:
            confidence = max(0.0, min(100.0, float(m.group(1)))) / 100.0
        except ValueError:
            confidence = None
    if confidence is None:
        m2 = _BARE_PERCENT_PATTERN.search(response)
        if m2:
            try:
                confidence = max(0.0, min(100.0, float(m2.group(1)))) / 100.0
            except ValueError:
                confidence = None
    return {"answer": answer, "confidence": confidence}


def _wrap_probe(probe: str) -> str:
    return (
        "Answer 'yes' or 'no'. On a new line state your confidence as a "
        "percentage (0-100). Question: " + probe
    )


class CalibrationEvaluator:
    """Compute Expected Calibration Error (ECE) for a model.

    ECE is the weighted absolute gap between confidence and accuracy
    across confidence bins. Lower is better. A perfectly calibrated
    model has ECE = 0.
    """

    def __init__(
        self,
        holdout_split: Path,
        model_callable: Callable[[str], str],
        bins: int = 10,
        max_questions: Optional[int] = None,
    ) -> None:
        from Trainforge.eval.faithfulness import _format_probe
        from Trainforge.eval.holdout_builder import load_holdout_split
        self._format_probe = _format_probe
        split = load_holdout_split(holdout_split)
        self.edges = split.get("withheld_edges", [])
        if max_questions is not None:
            self.edges = self.edges[:max_questions]
        self.model_callable = model_callable
        self.bins = max(2, int(bins))

    def evaluate(self) -> Dict[str, Any]:
        outcomes: List[Dict[str, Any]] = []
        for edge in self.edges:
            probe = self._format_probe(edge)
            wrapped = _wrap_probe(probe)
            try:
                response = str(self.model_callable(wrapped))
            except Exception as exc:  # noqa: BLE001
                outcomes.append({
                    "edge": edge,
                    "answer": None,
                    "confidence": None,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            parsed = _parse_response(response)
            # Held-out edges are TRUE in the graph, so "yes" is correct.
            correct = (parsed["answer"] == "yes")
            outcomes.append({
                "edge": edge,
                "response": response,
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "correct": correct,
            })

        ece, bin_summary = self._compute_ece(outcomes)
        return {
            "ece": ece,
            "bins": bin_summary,
            "total": len(outcomes),
            "scored": sum(1 for o in outcomes if o.get("confidence") is not None),
            "outcomes": outcomes,
        }

    def _compute_ece(
        self, outcomes: List[Dict[str, Any]],
    ) -> tuple[float, List[Dict[str, Any]]]:
        scored = [
            (o["confidence"], o["correct"])
            for o in outcomes if o.get("confidence") is not None
        ]
        if not scored:
            return 0.0, []

        n = len(scored)
        # Build bins: [0, 1/B), [1/B, 2/B), ..., [(B-1)/B, 1]
        bin_width = 1.0 / self.bins
        bin_summary: List[Dict[str, Any]] = []
        ece = 0.0
        for b in range(self.bins):
            lo = b * bin_width
            hi = lo + bin_width
            in_bin = [
                (c, ok) for (c, ok) in scored
                if (c >= lo and (c < hi or (b == self.bins - 1 and c <= hi)))
            ]
            if not in_bin:
                bin_summary.append({
                    "lo": lo, "hi": hi, "count": 0,
                    "avg_confidence": None, "accuracy": None, "gap": 0.0,
                })
                continue
            avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
            acc = sum(1 for _, ok in in_bin if ok) / len(in_bin)
            gap = abs(avg_conf - acc)
            ece += (len(in_bin) / n) * gap
            bin_summary.append({
                "lo": lo, "hi": hi, "count": len(in_bin),
                "avg_confidence": avg_conf, "accuracy": acc, "gap": gap,
            })
        return ece, bin_summary


__all__ = ["CalibrationEvaluator"]
