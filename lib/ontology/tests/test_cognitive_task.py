"""Tests for ``lib.ontology.cognitive_task`` — GPT Feedback (May 12) item 5.

Pins the contract of the 15-value observable task-verb axis that sits
orthogonal to ``bloom_level``. Detection mirrors
``lib.ontology.bloom.detect_bloom_level`` (whole-word match,
longest-verb-first iteration) so the two axes have parallel call shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from lib.ontology.cognitive_task import (
    COGNITIVE_TASK_TYPES,
    detect_cognitive_task_type,
    get_cognitive_task_types,
)


# ---------------------------------------------------------------------------
# Enum / loader invariants
# ---------------------------------------------------------------------------


def test_enum_is_15_values():
    """Schema enum and module constant agree on the canonical 15-value list."""
    assert len(COGNITIVE_TASK_TYPES) == 15


def test_enum_contains_canonical_task_verbs():
    """Every verb GPT feedback specifically called out is present."""
    required = {
        "classify",
        "compute",
        "critique",
        "debug",
        "explain",
        "compare",
        "construct",
    }
    assert required.issubset(set(COGNITIVE_TASK_TYPES))


def test_get_cognitive_task_types_returns_frozenset():
    out = get_cognitive_task_types()
    assert isinstance(out, frozenset)
    assert out == frozenset(COGNITIVE_TASK_TYPES)


# ---------------------------------------------------------------------------
# Canonical detection cases
# ---------------------------------------------------------------------------


CANONICAL_DETECTION_CASES = [
    ("Define photosynthesis", "define"),
    ("Identify the leading cause of failure", "identify"),
    ("Explain why the protocol requires retransmission", "explain"),
    ("Summarize the key findings of the paper", "summarize"),
    ("Classify the species by phylum", "classify"),
    ("Compare protozoa and helminths", "compare"),
    ("Apply Newton's second law to the situation", "apply"),
    ("Compute the molarity of the solution", "compute"),
    ("Analyze the call-graph for cycles", "analyze"),
    ("Debug the recursion that overflows the stack", "debug"),
    ("Predict the outcome of the experiment", "predict"),
    ("Infer the underlying cause from the symptoms", "infer"),
    ("Critique the argument's evidentiary basis", "critique"),
    ("Evaluate the trade-offs of the design", "evaluate"),
    ("Construct a state diagram for the parser", "construct"),
]


@pytest.mark.parametrize("text,expected", CANONICAL_DETECTION_CASES)
def test_canonical_detection_per_task_verb(text, expected):
    assert detect_cognitive_task_type(text) == expected


# ---------------------------------------------------------------------------
# Negative / edge cases
# ---------------------------------------------------------------------------


def test_no_task_verb_returns_none():
    assert detect_cognitive_task_type("There is no task verb here") is None


def test_empty_input_returns_none():
    assert detect_cognitive_task_type("") is None
    assert detect_cognitive_task_type(None) is None  # type: ignore[arg-type]


def test_substring_does_not_match():
    """A field labeled 'classifier' must NOT be tagged `classify`.

    Whole-word matching is the contract — same as Bloom's detector.
    """
    assert detect_cognitive_task_type("classifier output") is None


def test_lowercases_input():
    assert detect_cognitive_task_type("CLASSIFY THE SPECIES") == "classify"


def test_longest_verb_first_tiebreak():
    """When two verbs co-occur, the longer wins (deterministic order)."""
    # `construct` (9 chars) should beat `compute` (7 chars) when both appear.
    text = "Construct a procedure that compute distances between points"
    assert detect_cognitive_task_type(text) == "construct"
