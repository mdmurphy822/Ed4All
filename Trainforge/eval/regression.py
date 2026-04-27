"""Wave 92 — Layer 5 regression: today's run vs the previously-promoted version.

The harness pins to ``vocabulary_ttl_hash + pedagogy_graph_hash`` from
the prior version's ``model_card.json::provenance``. When those hashes
match the current course state, the eval is comparable; when they
diverge the regression module surfaces a clear flag rather than
silently comparing apples to oranges.

Wave 92 is forward-compatible: the prior-version pointer file
(``models/_pointers.json``) is a Wave 93 artifact. This module raises
a clean ``FileNotFoundError`` with a useful message when the pointer
file doesn't exist yet, so production runs in the v0.3.0 timeframe
fail loudly instead of trying to compare against a nonexistent prior.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_POINTER_FILENAME = "_pointers.json"


class RegressionEvaluator:
    """Compare current eval scores to a previously-promoted version.

    Args:
        course_path: Path to ``LibV2/courses/<slug>/``. The eval looks
            up ``models/_pointers.json`` to find the prior version.
        current_eval_report: Dict matching the
            ``model_card.eval_scores`` shape for this run.
    """

    def __init__(
        self,
        course_path: Path,
        current_eval_report: Dict[str, Any],
    ) -> None:
        self.course_path = Path(course_path)
        self.current = dict(current_eval_report)

    def evaluate(self) -> Dict[str, Any]:
        prior_card = self._resolve_prior_card()
        prior_eval = prior_card.get("eval_scores") or {}
        prior_provenance = prior_card.get("provenance") or {}

        # Compare hashes that affect what's eval-able. If
        # vocabulary_ttl or pedagogy_graph changed, the comparison
        # may be noisy; we surface that to the caller rather than
        # blocking.
        comparable = True
        drift_warnings: List[str] = []
        for key in ("vocabulary_ttl_hash", "pedagogy_graph_hash"):
            if prior_provenance.get(key) and self.current.get("_provenance", {}).get(key):
                if prior_provenance[key] != self.current["_provenance"][key]:
                    comparable = False
                    drift_warnings.append(
                        f"Provenance drift: {key} changed since prior version. "
                        f"Comparison may not reflect a clean delta."
                    )

        deltas: Dict[str, Any] = {}
        for metric in ("faithfulness", "coverage", "baseline_delta"):
            if metric in prior_eval and metric in self.current:
                try:
                    deltas[metric] = {
                        "prior": prior_eval[metric],
                        "current": self.current[metric],
                        "delta": float(self.current[metric]) - float(prior_eval[metric]),
                    }
                except (TypeError, ValueError):
                    deltas[metric] = {
                        "prior": prior_eval[metric],
                        "current": self.current[metric],
                        "delta": None,
                    }

        return {
            "comparable": comparable,
            "drift_warnings": drift_warnings,
            "prior_model_id": prior_card.get("model_id"),
            "current_model_id": self.current.get("_model_id"),
            "deltas": deltas,
        }

    def _resolve_prior_card(self) -> Dict[str, Any]:
        """Load the previously-promoted model_card.json.

        Wave 93 will land the pointer file format. Wave 92 emits a
        descriptive ``FileNotFoundError`` so the call site sees what
        artifact is missing.
        """
        pointer_path = self.course_path / "models" / _POINTER_FILENAME
        if not pointer_path.exists():
            raise FileNotFoundError(
                f"No prior-version pointer found at {pointer_path}. "
                f"This file is created by Wave 93's ``libv2 models promote`` "
                f"CLI. Once a model has been promoted, regression eval will "
                f"compare against it. For the first model on a slug there "
                f"is no prior to compare to."
            )
        try:
            pointers = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(
                f"Failed to read prior-version pointers from {pointer_path}: {exc}"
            ) from exc

        prior_id = pointers.get("current") or pointers.get("promoted")
        if not prior_id:
            raise FileNotFoundError(
                f"Pointer file at {pointer_path} has no 'current' or "
                f"'promoted' key naming the prior model."
            )
        prior_card_path = self.course_path / "models" / prior_id / "model_card.json"
        if not prior_card_path.exists():
            raise FileNotFoundError(
                f"Pointer references {prior_id!r} but its model_card.json "
                f"is missing at {prior_card_path}."
            )
        return json.loads(prior_card_path.read_text(encoding="utf-8"))


__all__ = ["RegressionEvaluator"]
