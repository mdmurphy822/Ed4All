"""Wave 25 Fix 5: dotted-numeric subsection → ``<h4>`` (and deeper levels).

Audit: on a textbook with dotted-numeric section numbering, dozens
to hundreds of short ``<p>`` bodies match
``^[0-9A-Z]+\\.[0-9]+.*`` (e.g. "4.8.1.1 Epistemological basis",
"1.7.1. Fully online learning", "8.4.1.4 Maintenance costs"). Pre-
Wave-25 the ``<h4>`` count was zero because the heuristic
classifier's regex didn't cover the dotted-numeric shape, and the
Wave-18 font-size promoter only fires when ``text_spans`` is
populated.

These tests exercise the heuristic path with ``text_spans`` absent.
"""

from __future__ import annotations

import pytest

from DART.converter.block_roles import BlockRole, RawBlock
from DART.converter.heuristic_classifier import HeuristicClassifier


def _classify(text: str):
    block = RawBlock(
        text=text, block_id="b1", page=1, extractor="pdftotext"
    )
    clf = HeuristicClassifier()
    return clf.classify_sync([block])[0]


@pytest.mark.unit
@pytest.mark.dart
class TestDottedNumericHeadings:
    def test_four_level_nested_heading_h5(self):
        # "4.8.1.1 Epistemological basis" → SUBSECTION_HEADING h5.
        result = _classify("4.8.1.1 Epistemological basis")
        assert result.role == BlockRole.SUBSECTION_HEADING
        assert result.attributes.get("level") == 5

    def test_three_level_nested_heading_h4(self):
        result = _classify("1.7.1. Fully online learning")
        assert result.role == BlockRole.SUBSECTION_HEADING
        assert result.attributes.get("level") == 4

    def test_three_level_no_trailing_dot(self):
        result = _classify("6.5.3 Why does this matter?")
        assert result.role == BlockRole.SUBSECTION_HEADING
        assert result.attributes.get("level") == 4

    def test_two_level_section_heading_h3(self):
        result = _classify("2.3 Implementation notes")
        # SUBSECTION_HEADING at level 3 — matches pre-Wave-25
        # PAPER_SECTION_NUMBERED behaviour for 1-dot forms. The
        # template layer renders <h3> based on the level attr.
        assert result.role == BlockRole.SUBSECTION_HEADING
        assert result.attributes.get("level") == 3

    def test_four_level_variant(self):
        result = _classify("8.4.1.4 Maintenance costs")
        assert result.role == BlockRole.SUBSECTION_HEADING
        assert result.attributes.get("level") == 5

    def test_single_digit_no_dot_not_matched_by_this_rule(self):
        # "1. Why this book?" — single-digit, no dot → should NOT
        # be classified as a dotted-numeric heading (Wave 21 list
        # rule picks it up instead, or chapter rule, etc.). The
        # invariant: result.attributes.get("dotted_number") is not
        # set.
        result = _classify("1. Why this book?")
        assert result.attributes.get("dotted_number") is None
