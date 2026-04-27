"""Wave 92 — Layer 2 behavioral-invariant checks.

Three invariant classes, each presenting prompts and scoring whether
the model's response satisfies a structural rule. The rules are
sourced from the LibV2 pedagogy graph (4,160 prerequisite_of edges,
365 interferes_with edges) and the corpus's Bloom-level distribution
(remember=20, understand=86, apply=120, analyze=33, evaluate=20,
create=16) so they're corpus-aware without being hardcoded.

Each class implements ``.evaluate(model_callable) -> dict`` returning
per-prompt outcomes plus an aggregate ``pass_rate``. The harness
combines pass-rates into the model_card eval_scores.coverage proxy.
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Bloom-level signal patterns. Each level expects certain shape of
# response — definitions for "remember", procedures for "apply",
# tradeoff language for "evaluate". The patterns are intentionally
# coarse so the eval doesn't over-fit a particular phrasing.

_BLOOM_RESPONSE_PATTERNS: Dict[str, re.Pattern] = {
    "remember": re.compile(
        r"\b(is|means|refers to|defined as|definition|denotes|"
        r"signifies|stands for)\b",
        re.IGNORECASE,
    ),
    "understand": re.compile(
        r"\b(because|so that|the reason|explains?|why|since|"
        r"in order to|due to)\b",
        re.IGNORECASE,
    ),
    "apply": re.compile(
        r"\b(step|first|then|next|finally|use|apply|run|execute|"
        r"invoke|construct|write|implement)\b",
        re.IGNORECASE,
    ),
    "analyze": re.compile(
        r"\b(compare|contrast|differs?|differs? from|on the other hand|"
        r"whereas|whilst|in contrast|distinguish|break down)\b",
        re.IGNORECASE,
    ),
    "evaluate": re.compile(
        r"\b(tradeoff|trade-off|advantage|disadvantage|prefer|"
        r"better|worse|recommend|justif|critique|assess)\b",
        re.IGNORECASE,
    ),
    "create": re.compile(
        r"\b(design|propose|generate|build|formulate|construct|"
        r"new|novel|original)\b",
        re.IGNORECASE,
    ),
}


def _load_pedagogy_graph(course_path: Path) -> Dict[str, Any]:
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


def _load_misconception_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [n for n in graph.get("nodes", []) if n.get("class") == "Misconception"]


# ---------------------------------------------------------------------- #
# Prerequisite ordering invariant                                         #
# ---------------------------------------------------------------------- #


class PrerequisiteOrderInvariant:
    """The model must surface (or refuse) when a prerequisite is missing.

    For each ``prerequisite_of`` edge ``A → B``, we craft a prompt that
    asks the model about B in isolation and check that the response
    either (a) names A as a dependency, or (b) explicitly refuses /
    asks for prerequisite context. A response that confidently
    addresses B without acknowledging A is a violation.
    """

    name = "prerequisite_order"

    def __init__(
        self,
        course_path: Path,
        max_prompts: int = 30,
        seed: int = 42,
    ) -> None:
        graph = _load_pedagogy_graph(course_path)
        self.edges = [
            e for e in graph.get("edges", [])
            if e.get("relation_type") == "prerequisite_of"
        ]
        rng = random.Random(seed)
        rng.shuffle(self.edges)
        self.edges = self.edges[:max_prompts]

    def evaluate(self, model_callable: Callable[[str], str]) -> Dict[str, Any]:
        per_prompt: List[Dict[str, Any]] = []
        passed = 0
        for edge in self.edges:
            prereq = edge.get("source", "")
            target = edge.get("target", "")
            prompt = (
                f"Without assuming any background knowledge, explain '{target}'. "
                f"If there are concepts a learner must know first, list them."
            )
            try:
                response = str(model_callable(prompt))
            except Exception as exc:  # noqa: BLE001
                per_prompt.append({
                    "edge": edge,
                    "prompt": prompt,
                    "outcome": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            mentions_prereq = self._mentions(response, prereq)
            refuses = self._refuses(response)
            ok = mentions_prereq or refuses
            if ok:
                passed += 1
            per_prompt.append({
                "edge": edge,
                "prompt": prompt,
                "response": response,
                "mentions_prerequisite": mentions_prereq,
                "refuses": refuses,
                "outcome": "pass" if ok else "fail",
            })

        total = len(self.edges)
        return {
            "invariant": self.name,
            "pass_rate": passed / total if total > 0 else 0.0,
            "passed": passed,
            "total": total,
            "per_prompt": per_prompt,
        }

    @staticmethod
    def _mentions(response: str, target: str) -> bool:
        if not target:
            return False
        # Tokenize both at word boundaries so we don't
        # match "rdf" inside "rdfs:subPropertyOf" by accident.
        token = re.escape(str(target).lower())
        return bool(re.search(rf"\b{token}\b", response.lower()))

    @staticmethod
    def _refuses(response: str) -> bool:
        return bool(re.search(
            r"\b(prerequisite|background|prior knowledge|first need|need to know|"
            r"cannot answer|insufficient context)\b",
            response,
            re.IGNORECASE,
        ))


# ---------------------------------------------------------------------- #
# Bloom-level shape invariant                                             #
# ---------------------------------------------------------------------- #


class BloomLevelInvariant:
    """Response shape must match the target Bloom level.

    Buckets prompts by Bloom level and verifies each response carries
    the level's signature pattern (definition for remember, procedure
    for apply, tradeoff for evaluate, etc.).
    """

    name = "bloom_level"

    def __init__(
        self,
        course_path: Path,
        max_per_level: int = 5,
        seed: int = 42,
    ) -> None:
        # Build Bloom -> chunk_id mapping from the at_bloom_level edges.
        graph = _load_pedagogy_graph(course_path)
        bloom_buckets: Dict[str, List[str]] = {}
        for e in graph.get("edges", []):
            if e.get("relation_type") != "at_bloom_level":
                continue
            target = str(e.get("target", ""))
            level = target.split(":", 1)[1] if target.startswith("bloom:") else target
            bloom_buckets.setdefault(level, []).append(e.get("source", ""))

        rng = random.Random(seed)
        self.prompts: List[Dict[str, Any]] = []
        for level, chunks in bloom_buckets.items():
            rng.shuffle(chunks)
            for chunk_id in chunks[:max_per_level]:
                self.prompts.append({
                    "bloom_level": level,
                    "chunk_id": chunk_id,
                    "prompt": self._prompt_for_level(level, chunk_id),
                })

    @staticmethod
    def _prompt_for_level(level: str, chunk_id: str) -> str:
        if level == "remember":
            return f"Define the central concept introduced in chunk '{chunk_id}'."
        if level == "understand":
            return f"Explain why the concept in chunk '{chunk_id}' is the way it is."
        if level == "apply":
            return f"Walk through how to use the concept in chunk '{chunk_id}' step by step."
        if level == "analyze":
            return (
                f"Compare the concept in chunk '{chunk_id}' to a related but "
                f"distinct concept; what differs?"
            )
        if level == "evaluate":
            return (
                f"What are the tradeoffs of the approach in chunk "
                f"'{chunk_id}'? When would you prefer it?"
            )
        if level == "create":
            return (
                f"Propose a new application of the concept in chunk '{chunk_id}'."
            )
        return f"Discuss the concept in chunk '{chunk_id}'."

    def evaluate(self, model_callable: Callable[[str], str]) -> Dict[str, Any]:
        per_prompt: List[Dict[str, Any]] = []
        passed = 0
        for p in self.prompts:
            level = p["bloom_level"]
            pattern = _BLOOM_RESPONSE_PATTERNS.get(level)
            try:
                response = str(model_callable(p["prompt"]))
            except Exception as exc:  # noqa: BLE001
                per_prompt.append({**p, "outcome": "error", "error": str(exc)})
                continue
            ok = bool(pattern.search(response)) if pattern else True
            if ok:
                passed += 1
            per_prompt.append({
                **p,
                "response": response,
                "outcome": "pass" if ok else "fail",
            })

        total = len(self.prompts)
        return {
            "invariant": self.name,
            "pass_rate": passed / total if total > 0 else 0.0,
            "passed": passed,
            "total": total,
            "per_prompt": per_prompt,
        }


# ---------------------------------------------------------------------- #
# Misconception rejection invariant                                       #
# ---------------------------------------------------------------------- #


class MisconceptionRejectionInvariant:
    """Present each first-class misconception node as an assertion.

    The model must reject it. Anchored on the 365 ``interferes_with``
    edges from misconception nodes to concept nodes — those tell us
    which concept the misconception targets, so we can reject "with
    explanation" correctly.
    """

    name = "misconception_rejection"

    def __init__(self, course_path: Path) -> None:
        graph = _load_pedagogy_graph(course_path)
        self.misconceptions = _load_misconception_nodes(graph)
        # Build mc_id -> set of target concepts from interferes_with
        self._targets: Dict[str, Set[str]] = {}
        for e in graph.get("edges", []):
            if e.get("relation_type") != "interferes_with":
                continue
            src = e.get("source")
            tgt = e.get("target")
            if src and tgt:
                self._targets.setdefault(src, set()).add(tgt)

    def evaluate(self, model_callable: Callable[[str], str]) -> Dict[str, Any]:
        per_prompt: List[Dict[str, Any]] = []
        passed = 0
        for node in self.misconceptions:
            statement = node.get("statement") or node.get("label", "")
            mc_id = node.get("id", "")
            prompt = (
                f"True or false, with explanation: \"{statement}\""
            )
            try:
                response = str(model_callable(prompt))
            except Exception as exc:  # noqa: BLE001
                per_prompt.append({
                    "misconception_id": mc_id,
                    "prompt": prompt,
                    "outcome": "error",
                    "error": str(exc),
                })
                continue

            rejected = self._rejects(response)
            if rejected:
                passed += 1
            per_prompt.append({
                "misconception_id": mc_id,
                "prompt": prompt,
                "response": response,
                "rejected": rejected,
                "outcome": "pass" if rejected else "fail",
                "target_concepts": sorted(self._targets.get(mc_id, set())),
            })

        total = len(self.misconceptions)
        return {
            "invariant": self.name,
            "pass_rate": passed / total if total > 0 else 0.0,
            "passed": passed,
            "total": total,
            "per_prompt": per_prompt,
        }

    @staticmethod
    def _rejects(response: str) -> bool:
        return bool(re.search(
            r"\b(false|incorrect|wrong|not (?:true|correct)|misconception|"
            r"actually|in fact|rather|the correct)\b",
            response,
            re.IGNORECASE,
        ))


__all__ = [
    "PrerequisiteOrderInvariant",
    "BloomLevelInvariant",
    "MisconceptionRejectionInvariant",
]
