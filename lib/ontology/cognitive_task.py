"""Cognitive task-type loader.

Loads ``schemas/taxonomies/cognitive_task_type.json`` (the authoritative
15-value enum of observable cognitive task verbs) and exposes a detector
that mirrors :func:`lib.ontology.bloom.detect_bloom_level`.

Where ``bloom_level`` captures the cognitive process tier
(remember/understand/apply/analyze/evaluate/create) and ``question_type``
captures the assessment instrument (multiple_choice/short_answer/...),
``cognitive_task_type`` captures WHAT the learner is asked to do:
classify a thing, compute a value, debug a failure, critique a position.
Two apply-tier multiple-choice questions can differ entirely on this
axis without it that orthogonal signal is lost.

Authored 2026-05 in response to GPT feedback (item 5,
``plans/gptfeedback12may.txt``) requesting observable task-type tagging
as a separate axis next to Bloom level.

    COGNITIVE_TASK_TYPES         -- immutable tuple of the 15 task verbs.
    get_cognitive_task_types()   -- frozenset (membership-test view).
    detect_cognitive_task_type   -- (text) -> Optional[str]. Canonical
                                    detector. Returns the first
                                    whole-word match (longest-verb-first
                                    iteration) or None.

The schema is read once at import time. The detector lower-cases input
and uses whole-word matching (``\\b{verb}\\b``) so substrings like
"classifier" don't match "classify".
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

__all__ = [
    "COGNITIVE_TASK_TYPES",
    "get_cognitive_task_types",
    "detect_cognitive_task_type",
]


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_TASK_TYPE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "cognitive_task_type.json"
)


@lru_cache(maxsize=1)
def _load_enum() -> Tuple[str, ...]:
    """Read ``cognitive_task_type.json`` and return the enum tuple."""
    if not _TASK_TYPE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Cognitive task-type schema not found at {_TASK_TYPE_SCHEMA_PATH}."
        )
    with open(_TASK_TYPE_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    defs = schema.get("$defs") or {}
    node = defs.get("CognitiveTaskType") or {}
    enum = node.get("enum")
    if not isinstance(enum, list) or not enum:
        raise ValueError(
            f"Malformed cognitive_task_type.json: missing "
            f"$defs.CognitiveTaskType.enum at {_TASK_TYPE_SCHEMA_PATH}"
        )
    return tuple(enum)


#: The 15-value canonical task-verb tuple. Authoritative copy lives in
#: ``schemas/taxonomies/cognitive_task_type.json``; this constant materializes
#: the enum at import time for cheap membership / iteration access.
COGNITIVE_TASK_TYPES: Tuple[str, ...] = _load_enum()


# ---------------------------------------------------------------------------
# Public getters
# ---------------------------------------------------------------------------

def get_cognitive_task_types() -> FrozenSet[str]:
    """Return the canonical 15-value task-type set (immutable frozenset)."""
    return frozenset(COGNITIVE_TASK_TYPES)


# ---------------------------------------------------------------------------
# Canonical detector
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _detection_order() -> Tuple[str, ...]:
    """Build the iteration order for detect_cognitive_task_type.

    Sort key:
      1. Verb length descending (longer task verbs win — e.g. ``construct``
         (9 chars) beats ``compute`` (7 chars) when both somehow appear).
      2. Alphabetical for stable ordering on ties.

    Unlike ``lib.ontology.bloom``'s detection order, there is no
    "level priority" tiebreak — the task-type axis has no inherent
    higher/lower ranking among its 15 verbs.
    """
    verbs = list(COGNITIVE_TASK_TYPES)
    verbs.sort(key=lambda v: (-len(v), v))
    return tuple(verbs)


def detect_cognitive_task_type(text: str) -> Optional[str]:
    """Detect the observable cognitive task verb from free text.

    Lowercases the input, strips, and searches for each canonical task
    verb as a whole word (``\\b{verb}\\b``). Returns the first match in
    longest-verb-first iteration order, or ``None`` if no canonical verb
    is present.

    The detector mirrors :func:`lib.ontology.bloom.detect_bloom_level`'s
    "first whole-word match wins" contract; substring matches (e.g.
    ``"classifier"`` for ``"classify"``) deliberately don't fire so a
    field labeled "classifier_output" isn't mistagged as a `classify`
    task.

    Examples:
        >>> detect_cognitive_task_type("Classify the species by phylum")
        'classify'
        >>> detect_cognitive_task_type("Compute the molarity of the solution")
        'compute'
        >>> detect_cognitive_task_type("Debug the recursion")
        'debug'
        >>> detect_cognitive_task_type("No task verb here")
        >>> detect_cognitive_task_type("classifier output")
        # Returns None — 'classifier' is not a whole-word match for
        # 'classify'.
    """
    if not text:
        return None
    lowered = text.lower().strip()
    for verb in _detection_order():
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            return verb
    return None
