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

from Trainforge.eval.chunk_ids import (
    chunk_ids_match,
    is_chunk_id,
    normalize_chunk_id,
)


logger = logging.getLogger(__name__)


# Wave 105: the default citation regex now accepts the four
# citation forms observed in trained-model output. RDF/SHACL corpus
# uses ``rdf_shacl_551_chunk_NNNNN`` IDs; we accept multiple
# citation formats and normalize back to the canonical ``chunk_NNNNN``
# form before comparison.
#
# Accepted forms (all extracted by the same alternation):
#   1. ``[chunk_00270]``                  — bracketed (the canonical form)
#   2. ``'chunk_00270'``                  — single-quoted bare suffix
#   3. ``'rdf_shacl_551_chunk_00270'``    — single-quoted full ID
#   4. ``chunk_00270``                    — bare token (no delimiter)
#
# Group 1 carries the matched chunk reference (with optional corpus
# prefix). The matcher strips the optional corpus prefix in
# :meth:`_normalize_citation` so a model that emits the long form is
# still credited when the ground-truth chunk_id is the short form.
_DEFAULT_CITATION_RE = (
    r"(?:"
    r"\[((?:[a-z0-9_]+_)?chunk_[0-9a-zA-Z_]+)\]"          # form 1: brackets
    r"|'((?:[a-z0-9_]+_)?chunk_[0-9a-zA-Z_]+)'"            # forms 2+3: single-quoted
    r"|(?<![A-Za-z0-9_'\[])((?:[a-z0-9_]+_)?chunk_[0-9a-zA-Z_]+)(?![A-Za-z0-9_'\]])"  # form 4: bare
    r")"
)


def _is_chunk_id(value: Any) -> bool:
    """True when ``value`` looks like a course chunk identifier."""
    return is_chunk_id(value)


def _normalize_citation(raw: str) -> str:
    """Strip an optional corpus prefix (e.g. ``rdf_shacl_551_``) so
    citations align with the canonical ``chunk_NNNNN`` ID space.

    A model can emit either ``chunk_00270`` or
    ``rdf_shacl_551_chunk_00270``; both should count as a match
    against ground_truth ``chunk_00270``. We find the rightmost
    occurrence of ``chunk_`` and keep everything from there.
    """
    normalized = normalize_chunk_id(raw)
    return normalized if normalized is not None else raw


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
            # Wave 108 / Phase B: multi-chunk ground truth. Prefer the
            # explicit set; fall back to edge.source for legacy splits.
            gt_ids_raw = edge.get("ground_truth_chunk_ids")
            if isinstance(gt_ids_raw, list) and gt_ids_raw:
                ground_truth_ids = [str(g) for g in gt_ids_raw]
            else:
                ground_truth_ids = [str(edge.get("source"))]
            ground_truth_normalized = [
                _normalize_citation(g) for g in ground_truth_ids
            ]
            # Back-compat: keep the legacy singular fields for trace
            # consumers that read them. The first element is the canonical
            # ground truth (the held-out edge's source).
            ground_truth = ground_truth_ids[0]
            try:
                response = self.model_callable(probe)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                per_question.append({
                    "edge": edge,
                    "probe": probe,
                    "response": None,
                    "ground_truth_chunk_id": ground_truth,
                    "ground_truth_chunk_id_normalized": ground_truth_normalized[0],
                    "ground_truth_chunk_ids": ground_truth_ids,
                    "ground_truth_chunk_ids_normalized": ground_truth_normalized,
                    "cited_chunk_ids": [],
                    "score": 0.0,
                    "outcome": "error",
                })
                continue

            # Wave 105: regex carries multiple alternation groups; pick
            # whichever group fired for each match, then normalize to
            # strip optional corpus prefix.
            raw_matches = self.citation_re.findall(str(response))
            cited_raw: List[str] = []
            for m in raw_matches:
                if isinstance(m, tuple):
                    # First non-empty alternation group is the citation.
                    citation = next((g for g in m if g), None)
                    if citation:
                        cited_raw.append(citation)
                else:
                    cited_raw.append(m)
            cited_set = list(dict.fromkeys(
                _normalize_citation(c) for c in cited_raw
            ))  # de-dupe, preserve order
            # Wave 108: credit the model when ANY ground-truth chunk is cited.
            score = 1.0 if any(
                chunk_ids_match(gt, cited)
                for gt in ground_truth_ids
                for cited in cited_set
            ) else 0.0
            if score == 1.0:
                matches += 1
            per_question.append({
                "edge": edge,
                "probe": probe,
                "response": response,
                "ground_truth_chunk_id": ground_truth,
                "ground_truth_chunk_id_normalized": ground_truth_normalized[0],
                "ground_truth_chunk_ids": ground_truth_ids,
                "ground_truth_chunk_ids_normalized": ground_truth_normalized,
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
