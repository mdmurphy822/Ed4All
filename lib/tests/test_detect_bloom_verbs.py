"""Wave 58 — ``lib.ontology.bloom.detect_bloom_verbs`` canonical multi-match.

Companion to the singular ``detect_bloom_level``. Returns every canonical
verb that appears as a whole word in the input, ordered the same way the
singular detector iterates (longest-verb-first, higher-Bloom-level ties
winning). The invariant tested here:

    detect_bloom_verbs(text)[0] == detect_bloom_level(text)

when either returns a match. This is load-bearing for the Wave 58 emit
layer in Courseforge: ``bloomLevels[0]`` must equal the singular
``bloomLevel`` field.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest  # noqa: E402

from lib.ontology.bloom import (  # noqa: E402
    BLOOM_LEVELS,
    detect_bloom_level,
    detect_bloom_verbs,
)


# ---------------------------------------------------------------------- #
# 1. Basic behavior — single verb, no verb, empty
# ---------------------------------------------------------------------- #


def test_single_verb_statement_returns_one_match():
    matches = detect_bloom_verbs("Apply the framework to the data.")
    assert matches == [("apply", "apply")]


def test_no_canonical_verb_returns_empty_list():
    assert detect_bloom_verbs("No taxonomy-style verbs appear in this sentence.") == []


def test_empty_text_returns_empty_list():
    assert detect_bloom_verbs("") == []
    assert detect_bloom_verbs(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# 2. Multi-verb statement — every canonical verb emitted
# ---------------------------------------------------------------------- #


def test_multi_verb_returns_all_detected_in_canonical_order():
    """'Analyze and evaluate X' must emit both verbs, longest-first."""
    matches = detect_bloom_verbs("Analyze and evaluate the market trends")
    # ``evaluate`` is 8 chars, ``analyze`` is 7 — longest wins first.
    assert matches == [("evaluate", "evaluate"), ("analyze", "analyze")], (
        f"Expected longest-verb-first ordering; got {matches!r}"
    )


def test_multi_verb_from_different_levels():
    """'Solve and explain' → one 'apply' + one 'understand' verb."""
    matches = detect_bloom_verbs("Solve the equation and explain your reasoning.")
    # Both ``explain`` (understand, 7) and ``solve`` (apply, 5) are canonical
    # verbs. Longest-first => explain before solve.
    assert matches == [("understand", "explain"), ("apply", "solve")]


# ---------------------------------------------------------------------- #
# 3. Invariant: singular detector's match equals plural[0]
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Apply the framework to the data.",
        "Analyze and evaluate the market trends",
        "Design a scalable system.",
        "List the parts of a cell.",
        "Students will explain and solve.",
        "The student should apply.",
    ],
)
def test_plural_first_element_matches_singular(text):
    """detect_bloom_verbs(t)[0] == detect_bloom_level(t) when either has a match."""
    singular = detect_bloom_level(text)
    plural = detect_bloom_verbs(text)
    if singular == (None, None):
        assert plural == [], (
            f"Singular returned no match; plural should also be empty; got {plural!r}"
        )
    else:
        assert plural, (
            f"Singular matched {singular!r} but plural is empty for {text!r}"
        )
        assert plural[0] == singular, (
            f"Plural[0] ({plural[0]!r}) must equal singular ({singular!r}) for {text!r}"
        )


# ---------------------------------------------------------------------- #
# 4. Deduplication — same verb twice in text counts once
# ---------------------------------------------------------------------- #


def test_repeated_verb_deduplicated():
    """A verb repeated in the statement must appear once in the output."""
    matches = detect_bloom_verbs("Apply it, then apply it again.")
    assert matches == [("apply", "apply")], (
        f"Repeated verb should dedupe; got {matches!r}"
    )


# ---------------------------------------------------------------------- #
# 5. Levels in output are all canonical enum members
# ---------------------------------------------------------------------- #


def test_every_returned_level_is_canonical():
    matches = detect_bloom_verbs(
        "Recall the parts, explain the process, apply the rules, "
        "analyze the outcome, evaluate the merit, and design the next step."
    )
    for lvl, _verb in matches:
        assert lvl in BLOOM_LEVELS, (
            f"Returned level {lvl!r} not in canonical BLOOM_LEVELS {BLOOM_LEVELS!r}"
        )


# ---------------------------------------------------------------------- #
# 6. End-of-text / punctuation boundary cases (shares the Wave 55 matcher)
# ---------------------------------------------------------------------- #


def test_verb_at_end_of_text_is_detected():
    """Post-Wave-55 canonical detector catches end-of-text verbs."""
    matches = detect_bloom_verbs("The student should apply.")
    assert matches == [("apply", "apply")]


def test_verb_followed_by_comma_is_detected():
    matches = detect_bloom_verbs("First, analyze the data thoroughly.")
    assert matches == [("analyze", "analyze")]
