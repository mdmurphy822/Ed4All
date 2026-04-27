"""Wave 92 — Tier 3 cross-concept disambiguation.

For each ``interferes_with`` edge in the pedagogy graph
(misconception -> concept), pose a "distinguish X from Y" prompt and
verify the model's response surfaces the corpus-stored distinction.

Schema correction: misconceptions are first-class graph nodes
(class=Misconception, 34 of them in rdf-shacl-551-2). The
``interferes_with`` edge type — 365 edges in rdf-shacl-551-2 — is the
canonical anchor for misconception-to-concept linkage. There is **no
``misconception-of`` edge type** in the pedagogy graph; that slug
exists on the ``concept_graph_semantic`` artifact (a different
graph), so this module pulls only from ``pedagogy_graph.json``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_graph(course_path: Path) -> Dict[str, Any]:
    candidates = (
        "graph/pedagogy_graph.json",
        "pedagogy/pedagogy_graph.json",
        "pedagogy/pedagogy_model.json",
    )
    for rel in candidates:
        p = course_path / rel
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No pedagogy graph found in {course_path} (tried {candidates})"
    )


class DisambiguationEvaluator:
    """Distinguish-X-from-Y eval over interferes_with edges.

    For each misconception node M that interferes with concept C, we
    prompt the model to distinguish M from C and check the response
    contains a distinction signal. The "distinction signal" is one of:

    * Correction text from the corresponding chunk's ``misconceptions[]``
      array (when present).
    * The misconception's own ``label`` or ``statement`` text fragments
      that should NOT appear approvingly in a correct response.

    Score: each prompt is pass/fail; aggregate is the pass rate.
    """

    name = "disambiguation"

    def __init__(
        self,
        course_path: Path,
        model_callable: Callable[[str], str],
        max_pairs: Optional[int] = 50,
    ) -> None:
        self.course_path = Path(course_path)
        graph = _load_graph(self.course_path)
        self.misconceptions: Dict[str, Dict[str, Any]] = {
            n["id"]: n
            for n in graph.get("nodes", [])
            if n.get("class") == "Misconception"
        }
        self.concepts: Dict[str, Dict[str, Any]] = {
            n["id"]: n
            for n in graph.get("nodes", [])
            if n.get("class") == "Concept"
        }
        self.pairs: List[Dict[str, Any]] = []
        for e in graph.get("edges", []):
            if e.get("relation_type") != "interferes_with":
                continue
            mc = self.misconceptions.get(e.get("source", ""))
            concept = self.concepts.get(e.get("target", ""))
            if mc and concept:
                self.pairs.append({"misconception": mc, "concept": concept})
        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]

        # Pre-build correction lookup from chunks
        self._corrections = self._index_corrections()
        self.model_callable = model_callable

    def _index_corrections(self) -> Dict[str, List[str]]:
        """Map misconception statement fragments -> their corrections."""
        chunks_path = self.course_path / "corpus" / "chunks.jsonl"
        out: Dict[str, List[str]] = {}
        if not chunks_path.exists():
            return out
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            for mc in chunk.get("misconceptions") or []:
                if not isinstance(mc, dict):
                    continue
                statement = (mc.get("misconception") or mc.get("statement") or "").strip()
                correction = (mc.get("correction") or "").strip()
                if statement and correction:
                    out.setdefault(statement[:80].lower(), []).append(correction)
        return out

    def evaluate(self) -> Dict[str, Any]:
        per_pair: List[Dict[str, Any]] = []
        passed = 0
        for pair in self.pairs:
            mc = pair["misconception"]
            concept = pair["concept"]
            mc_label = mc.get("label", "") or mc.get("statement", "")
            concept_label = concept.get("label", "") or concept.get("id", "")
            statement = mc.get("statement", "") or mc_label
            prompt = (
                f"Some learners think: \"{statement}\" "
                f"How does this differ from the correct understanding of "
                f"'{concept_label}'?"
            )
            try:
                response = str(self.model_callable(prompt))
            except Exception as exc:  # noqa: BLE001
                per_pair.append({
                    "misconception_id": mc.get("id"),
                    "concept_id": concept.get("id"),
                    "outcome": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            corrections = self._corrections.get(statement[:80].lower(), [])
            ok, reason = self._scores_distinction(response, corrections, concept_label)
            if ok:
                passed += 1
            per_pair.append({
                "misconception_id": mc.get("id"),
                "concept_id": concept.get("id"),
                "prompt": prompt,
                "response": response,
                "correction_anchors": corrections,
                "outcome": "pass" if ok else "fail",
                "reason": reason,
            })

        total = len(self.pairs)
        return {
            "invariant": self.name,
            "pass_rate": passed / total if total > 0 else 0.0,
            "passed": passed,
            "total": total,
            "per_pair": per_pair,
        }

    @staticmethod
    def _scores_distinction(
        response: str,
        correction_anchors: List[str],
        concept_label: str,
    ) -> tuple[bool, str]:
        """Heuristic scorer.

        Pass when ANY of:
          1. Response includes any non-trivial token from the corpus
             correction text (when corrections are available), AND
             contains a distinguishing-language signal
             ("rather", "actually", "in fact", "the correct", etc.).
          2. No corrections were available to anchor on but the response
             explicitly invokes the concept_label and uses
             distinguishing language. (Soft fallback.)
        """
        signal = re.search(
            r"\b(rather|actually|in fact|the correct|distinguishes?|"
            r"differs?|whereas|whilst|while|on the other hand|"
            r"misconception|incorrect|wrong)\b",
            response,
            re.IGNORECASE,
        )
        if not signal:
            return False, "no distinguishing signal in response"

        if correction_anchors:
            for anchor in correction_anchors:
                anchor_tokens = [
                    t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", anchor)
                    if len(t) > 4
                ]
                if any(tok.lower() in response.lower() for tok in anchor_tokens):
                    return True, "correction anchor token matched"
            return False, "distinguishing signal but no correction anchor matched"

        if concept_label and concept_label.lower() in response.lower():
            return True, "concept_label invoked alongside distinguishing signal"
        return False, "distinguishing signal but concept_label not invoked"


__all__ = ["DisambiguationEvaluator"]
