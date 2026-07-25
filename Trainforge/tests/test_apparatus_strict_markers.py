"""Regression net for the widened GENERIC apparatus marker set.

Covers ``ED4ALL_ASSESSMENT_APPARATUS_STRICT`` in
``Trainforge/generators/content_extractor.py``. The legacy marker set requires
a colon (``Solution:``), which an OCR'd scan lane routinely drops, so figure
captions, all-caps HOW-TO banners and glyph alt-text leaked into assessment
distractors AND correct answers.

Two contracts are pinned here:

1. **Default OFF is byte-identical** — with the flag unset every widened
   marker must be inert, so existing corpora keep their exact harvest.
2. **Flag ON catches the measured leak class with no false positives** on
   legitimate math prose (the false-positive corpus below is the adversarial
   set that drove excluding "showing" from the glyph participles).
"""

import pytest

from Trainforge.generators.content_extractor import (
    _is_apparatus_text,
    resolve_apparatus_strict,
)

_FLAG = "ED4ALL_ASSESSMENT_APPARATUS_STRICT"

# Real strings harvested from a scan-derived 313-item quiz set.
APPARATUS = [
    "A gray checkmark inside a circle, indicating correct or approved.",
    "Solution A gray checkmark inside a circle, indicating correct or complete.",
    "HOW TO ROUND WHOLE NUMBERS Round 23,658 to the nearest hundred.",
    "Figure 1.14 shows the names of the place values to the left of the point.",
    "Right-pointing arrow inside a square, indicating a forward or next action.",
    "Table 3.19 A right-pointing arrow inside a square, indicating navigation.",
]

# Legitimate subject prose that must NEVER be rejected. Several are adversarial
# near-misses: they mention a shape noun, cite a figure mid-sentence, or open
# with the word "Check"/"Solution" in an ordinary sentence.
LEGIT = [
    "A circle is the set of all points equidistant from a center point.",
    "The graph in Figure 1.14 illustrates the relationship between x and y.",
    "Check your solution by substituting the value back into the equation.",
    "Solution sets may be empty when the system is inconsistent.",
    "A box plot summarizes data, showing the median and quartiles.",
    "Draw a triangle, showing all three interior angles.",
    "The arrow notation f: A to B means f maps A into B.",
    "A square root of a number n is a value that, multiplied by itself, gives n.",
    "The distributive property states that a(b+c) = ab + ac.",
]


@pytest.mark.parametrize("text", APPARATUS)
def test_flag_off_is_byte_identical(monkeypatch, text):
    """Default OFF: every widened marker stays inert (legacy harvest)."""
    monkeypatch.delenv(_FLAG, raising=False)
    assert resolve_apparatus_strict() is False
    assert _is_apparatus_text(text) is False


@pytest.mark.parametrize("text", APPARATUS)
def test_flag_on_catches_apparatus(monkeypatch, text):
    monkeypatch.setenv(_FLAG, "1")
    assert _is_apparatus_text(text) is True


@pytest.mark.parametrize("text", LEGIT)
def test_flag_on_keeps_legitimate_prose(monkeypatch, text):
    monkeypatch.setenv(_FLAG, "1")
    assert _is_apparatus_text(text) is False


@pytest.mark.parametrize("text", LEGIT)
def test_flag_off_keeps_legitimate_prose(monkeypatch, text):
    monkeypatch.delenv(_FLAG, raising=False)
    assert _is_apparatus_text(text) is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("", False), ("0", False), ("false", False), ("garbage", False),
    ],
)
def test_flag_parse_with_fallback(monkeypatch, raw, expected):
    """Garbage / falsey resolves OFF rather than raising."""
    monkeypatch.setenv(_FLAG, raw)
    assert resolve_apparatus_strict() is expected


def test_legacy_colon_marker_still_fires_with_flag_off(monkeypatch):
    """The pre-existing marker set is untouched by the widening."""
    monkeypatch.delenv(_FLAG, raising=False)
    assert _is_apparatus_text("Solution: x = 4 Check: 2(4) = 8") is True
    assert _is_apparatus_text("Try It 3.14") is True


# ---- HTML-WRAPPED shape (the production emit) ----------------------------
#
# The widened markers are `^`-anchored, and the assessment emitter wraps
# choice / answer text in `<p>…</p>`. So the guard matched raw chunk strings
# in an offline A/B while passing the wrapped form the real emit produces —
# apparatus shipped anyway (159/626 distractors on a real build). The guard
# now also matches against the tag-stripped text.

WRAPPED_APPARATUS = [
    "<p>A gray checkmark inside a circle, indicating correct or complete.</p>",
    "<p>Figure 1.14 shows the names of the place values.</p>",
    "<p>HOW TO ROUND WHOLE NUMBERS Round 23,658.</p>",
    "<li>Right-pointing arrow inside a square, indicating a next action.</li>",
    "<p>Table 3.19 A right-pointing arrow, indicating navigation.</p>",
]

# A bare worked-example step label is pure scaffolding, never an answer.
BARE_STEP_LABELS = ["Step 1", "<p>Step 2.</p>", "Step 3:", "  step 10 "]

WRAPPED_LEGIT = [
    "<p>A circle is the set of all points equidistant from a center.</p>",
    "<p>Step 1 is to isolate the variable on one side.</p>",
    "<p>The graph in Figure 1.14 illustrates the relationship.</p>",
    "<p>A box plot summarizes data, showing the median.</p>",
    "<p>multiple of n if it is the product of a counting number and n</p>",
    "<p>The distributive property states that a(b+c) = ab + ac.</p>",
]


@pytest.mark.parametrize("text", WRAPPED_APPARATUS + BARE_STEP_LABELS)
def test_wrapped_apparatus_is_caught(monkeypatch, text):
    monkeypatch.setenv(_FLAG, "1")
    assert _is_apparatus_text(text) is True


@pytest.mark.parametrize("text", WRAPPED_LEGIT)
def test_wrapped_legitimate_prose_survives(monkeypatch, text):
    monkeypatch.setenv(_FLAG, "1")
    assert _is_apparatus_text(text) is False


@pytest.mark.parametrize("text", WRAPPED_APPARATUS + BARE_STEP_LABELS)
def test_wrapped_forms_inert_when_flag_off(monkeypatch, text):
    monkeypatch.delenv(_FLAG, raising=False)
    assert _is_apparatus_text(text) is False


def test_step_label_only_matches_a_STANDALONE_label(monkeypatch):
    """'Step N' inside a real sentence is content, not scaffolding."""
    monkeypatch.setenv(_FLAG, "1")
    assert _is_apparatus_text("Step 1") is True
    assert _is_apparatus_text("In Step 1 we isolate x, then Step 2 divides.") is False
