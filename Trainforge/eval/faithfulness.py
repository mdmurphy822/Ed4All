"""Wave 92 — Layer 1 faithfulness eval.

Builds (Q, A) probes from withheld pedagogy-graph edges and asks the
model whether each probe holds. ``model_callable`` is a
``Callable[[str], str]`` so the harness doesn't bind to any particular
model API; tests use synthetic mocks and the production wiring uses
the trained adapter via ``transformers.pipeline``.

Each held-out edge becomes one probe. The probe text is templated
from the relation type:

    prerequisite_of(A, B)         -> "Is concept A a prerequisite for concept B?"
    teaches(C, X)                 -> "Does chunk C teach concept X?"
    interferes_with(M, X)         -> "Does misconception M interfere with concept X?"
    concept_supports_outcome(...) -> "Does concept X support outcome O?"
    derived_from_objective(...)   -> "Is concept X derived from objective O?"
    exemplifies(C, X)             -> "Does chunk C exemplify concept X?"
    assesses(A, X)                -> "Does assessment A assess concept X?"

Other relation types fall back to a generic
"Is the following statement true: <s> -[rel]-> <t>?" template.

Scoring: a response is judged "correct" when it contains an explicit
yes/affirmative token (case-insensitive, simple regex). Held-out edges
are TRUE statements about the graph, so the correct answer is always
yes; the eval doubles as a recall measurement.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


_AFFIRMATIVE = re.compile(
    r"\b(yes|true|correct|affirmative|holds?|valid|right)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(no|false|incorrect|wrong|invalid|does not|doesn'?t|never)\b",
    re.IGNORECASE,
)


_RELATION_TEMPLATES: Dict[str, str] = {
    "prerequisite_of": "Is the concept '{source}' a prerequisite for the concept '{target}'?",
    "teaches": "Does the chunk '{source}' teach the concept '{target}'?",
    "interferes_with": "Does the misconception '{source}' interfere with the concept '{target}'?",
    "concept_supports_outcome": "Does the concept '{source}' support the learning outcome '{target}'?",
    "derived_from_objective": "Is the concept '{source}' derived from the objective '{target}'?",
    "exemplifies": "Does the chunk '{source}' exemplify the concept '{target}'?",
    "assesses": "Does the assessment '{source}' assess the concept '{target}'?",
    "supports_outcome": "Does the component objective '{source}' support the terminal outcome '{target}'?",
    "follows": "Does '{source}' follow '{target}' in the curriculum order?",
    "belongs_to_module": "Does the chunk '{source}' belong to the module '{target}'?",
    "at_bloom_level": "Is the chunk '{source}' at Bloom level '{target}'?",
    # Wave 108 / Phase B: chunk_at_difficulty was dropped — every chunk
    # has a difficulty level so the probe was trivially-true and only
    # padded faithfulness scores. Held-out edges of that type fall
    # through to the generic template now.
    "assessment_validates_outcome": "Does the assessment '{source}' validate the outcome '{target}'?",
}


def _format_probe(edge: Dict[str, Any]) -> str:
    rel = edge.get("relation_type", "related_to")
    template = _RELATION_TEMPLATES.get(
        rel,
        "Is the following statement true: '{source}' -[{rel}]-> '{target}'?",
    )
    return template.format(
        source=edge.get("source", "?"),
        target=edge.get("target", "?"),
        rel=rel,
    )


def _classify_response(response: str) -> str:
    """Coarse classifier: 'affirm' / 'deny' / 'ambiguous'.

    The simple heuristic — affirmative beats negative when both
    appear — is conservative; ambiguous responses are treated as
    incorrect at the harness level.
    """
    affirm = bool(_AFFIRMATIVE.search(response))
    deny = bool(_NEGATIVE.search(response))
    if affirm and not deny:
        return "affirm"
    if deny and not affirm:
        return "deny"
    return "ambiguous"


class FaithfulnessEvaluator:
    """Score a model on held-out (true) graph facts."""

    def __init__(
        self,
        holdout_split: Path,
        model_callable: Callable[[str], str],
        max_questions: Optional[int] = None,
    ) -> None:
        """
        Args:
            holdout_split: Path to ``eval/holdout_split.json`` (built by
                :class:`HoldoutBuilder`).
            model_callable: A ``Callable[[str], str]`` that takes a
                prompt and returns the model's textual response.
            max_questions: Optional cap to keep eval cost bounded;
                first-N withheld edges are sampled.
        """
        from Trainforge.eval.holdout_builder import load_holdout_split
        self.split = load_holdout_split(holdout_split)
        self.model_callable = model_callable
        self.max_questions = max_questions

    def evaluate(self) -> Dict[str, Any]:
        """Run the eval. Returns per-question + summary results."""
        edges = self.split.get("withheld_edges", [])
        if self.max_questions is not None:
            edges = edges[: self.max_questions]

        per_question: List[Dict[str, Any]] = []
        correct = 0
        ambiguous = 0
        errors: List[str] = []

        for edge in edges:
            probe = _format_probe(edge)
            try:
                response = self.model_callable(probe)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                per_question.append({
                    "edge": edge,
                    "probe": probe,
                    "response": None,
                    "outcome": "error",
                })
                continue

            classification = _classify_response(str(response))
            outcome = "correct" if classification == "affirm" else (
                "ambiguous" if classification == "ambiguous" else "incorrect"
            )
            if outcome == "correct":
                correct += 1
            elif outcome == "ambiguous":
                ambiguous += 1
            per_question.append({
                "edge": edge,
                "probe": probe,
                "response": response,
                "outcome": outcome,
            })

        total = len(per_question)
        scored_total = total - sum(1 for r in per_question if r["outcome"] == "error")
        accuracy = correct / scored_total if scored_total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "total_questions": total,
            "correct": correct,
            "ambiguous": ambiguous,
            "scored_total": scored_total,
            "per_question_results": per_question,
            "errors": errors,
        }


__all__ = ["FaithfulnessEvaluator"]
