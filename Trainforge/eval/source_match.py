"""Wave 102 - Source-Match metric.

For every held-out probe (built from the Bloom-stratified pedagogy
holdout split), the ground-truth source chunk is known: it's the
``source`` endpoint of any edge whose ``source`` is a chunk-id
(``teaches``, ``exemplifies``, ``assesses``, ...). The model under
test is expected to cite that chunk_id in ``[brackets]`` somewhere in
its response when given a RAG prelude that included it.

Source-Match is the fraction of probes where the model's emitted
citation set contains the ground-truth chunk_id. It's the precision
companion to Faithfulness:

* Faithfulness asks "did the model affirm the held-out fact?"
* Source-Match asks "did the model cite the chunk that grounds it?"

A model that aces Faithfulness but flunks Source-Match is regurgitating
training-time memory rather than grounding to the retrieved corpus -
exactly what a RAG-on row of the ablation table is supposed to expose.

Wired into :meth:`SLMEvalHarness.run_all` alongside the other
layers; emitted under ``per_tier['source_match']`` and surfaced as a
top-level ``source_match`` key for the ablation renderer.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


# Default citation pattern: ``[chunk_<hex_or_underscore>]`` form.
# Permissive enough to catch both legacy ``chunk_abc123`` IDs and the
# Wave 75-style ``chunk_<16 hex>`` content-hashed IDs.
_DEFAULT_CITATION_RE = r"\[(chunk_[0-9a-zA-Z_]+)\]"


_CHUNK_PREFIXES = ("chunk_",)


def _is_chunk_id(value: Any) -> bool:
    """True when ``value`` looks like a ``chunk_*`` identifier."""
    if not isinstance(value, str):
        return False
    return any(value.startswith(p) for p in _CHUNK_PREFIXES)


class SourceMatchEvaluator:
    """Score the fraction of probes where the model cites the right chunk.

    Args:
        holdout_split: Path to the Wave 92 ``eval/holdout_split.json``
            (built by :class:`HoldoutBuilder`).
        model_callable: A ``Callable[[str], str]`` that consumes a probe
            (typically the same probe shape the
            :class:`FaithfulnessEvaluator` uses) and returns the model's
            response.
        citation_pattern: Regex for extracting cited chunk_ids from the
            response. Default catches the canonical
            ``[chunk_<token>]`` form.
        max_questions: Optional cap on how many probes to score; first-N
            chunk-anchored withheld edges are sampled.
    """

    def __init__(
        self,
        holdout_split: Path,
        model_callable: Callable[[str], str],
        *,
        citation_pattern: str = _DEFAULT_CITATION_RE,
        max_questions: Optional[int] = None,
    ) -> None:
        from Trainforge.eval.holdout_builder import load_holdout_split

        self.split = load_holdout_split(holdout_split)
        self.model_callable = model_callable
        self.citation_re = re.compile(citation_pattern)
        self.max_questions = max_questions

    def evaluate(self) -> Dict[str, Any]:
        """Run the eval; returns per-question + aggregate score."""
        from Trainforge.eval.faithfulness import _format_probe

        edges = self.split.get("withheld_edges", []) or []
        # Source-match is only meaningful for chunk-anchored edges. For
        # an edge like prerequisite_of(concept_a, concept_b) there's no
        # single chunk that grounds the fact, so we skip.
        chunk_anchored = [e for e in edges if _is_chunk_id(e.get("source"))]
        if self.max_questions is not None:
            chunk_anchored = chunk_anchored[: self.max_questions]

        per_question: List[Dict[str, Any]] = []
        matches = 0
        errors: List[str] = []

        for edge in chunk_anchored:
            probe = _format_probe(edge)
            ground_truth = edge.get("source")
            try:
                response = self.model_callable(probe)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                per_question.append({
                    "edge": edge,
                    "probe": probe,
                    "response": None,
                    "ground_truth_chunk_id": ground_truth,
                    "cited_chunk_ids": [],
                    "score": 0.0,
                    "outcome": "error",
                })
                continue

            cited = self.citation_re.findall(str(response))
            cited_set = list(dict.fromkeys(cited))  # de-dupe, preserve order
            score = 1.0 if ground_truth in cited_set else 0.0
            if score == 1.0:
                matches += 1
            per_question.append({
                "edge": edge,
                "probe": probe,
                "response": response,
                "ground_truth_chunk_id": ground_truth,
                "cited_chunk_ids": cited_set,
                "score": score,
                "outcome": "match" if score == 1.0 else "miss",
            })

        scored_total = sum(1 for r in per_question if r["outcome"] != "error")
        rate = matches / scored_total if scored_total > 0 else 0.0

        return {
            "source_match_rate": rate,
            "total_questions": len(per_question),
            "scored_total": scored_total,
            "matches": matches,
            "per_question": per_question,
            "errors": errors,
        }


__all__ = ["SourceMatchEvaluator"]
