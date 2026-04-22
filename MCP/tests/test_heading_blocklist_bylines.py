"""Wave 27 HIGH-4 — heading blocklist for author bylines + math notation.

Previously, the Wave 24 byline detector in ``_is_low_signal_heading``
caught hyphenated-token names ("Ada-Lee Researcher") but leaked:

- 2-token full names without a lead-in ("Jane Smith", "John Smith")
- Single-initial names with parenthetical nicknames ("J.Q. (Buddy) Doe")
- "Edited by" / "Designed by" / "Illustrated by" lead-ins
- Math / logic notation residue ("C v ∀R.D", "∀x (P(x) → Q(x))")
- Formulaic-phrase lead-ins that pdftotext hoisted into headings
  ("The functional syntax equivalent is as follows:")

Wave 27 closes all five gaps while preserving legitimate chapter titles
that superficially look name-like ("European Union Policy",
"Creative Commons", "Digital Pedagogy" — anchored by at least one token
in the common-title-word set).
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import _content_gen_helpers as _cgh  # noqa: E402


class TestWave27LowSignalHeadings:
    """Headings that MUST be filtered out as low-signal chrome / residue."""

    def test_cover_design_byline_with_name(self):
        assert _cgh._is_low_signal_heading("Cover design by Author Name") is True

    def test_multi_author_transliteration(self):
        assert _cgh._is_low_signal_heading(
            "Ada-Lee Researcher Ben Otherwriter"
        ) is True

    def test_math_notation_short(self):
        assert _cgh._is_low_signal_heading("C v \u2200R.D") is True

    def test_functional_syntax_equivalent_phrase(self):
        assert _cgh._is_low_signal_heading(
            "The functional syntax equivalent is as follows:"
        ) is True

    def test_two_token_full_name(self):
        assert _cgh._is_low_signal_heading("Jane Smith") is True

    def test_edited_by_byline(self):
        assert _cgh._is_low_signal_heading(
            "Edited by Jane Smith and Robert Jones"
        ) is True

    def test_initialed_name_with_parenthetical(self):
        assert _cgh._is_low_signal_heading("J.Q. (Buddy) Doe") is True

    def test_pure_math_formula(self):
        assert _cgh._is_low_signal_heading("\u2200x (P(x) \u2192 Q(x))") is True

    def test_cover_design_alone(self):
        assert _cgh._is_low_signal_heading("Cover design") is True

    def test_designed_by_byline(self):
        assert _cgh._is_low_signal_heading("Designed by John Smith") is True

    def test_illustrated_by_byline(self):
        assert _cgh._is_low_signal_heading("Illustrated by Sarah Lee") is True

    def test_all_rights_reserved(self):
        assert _cgh._is_low_signal_heading("All rights reserved") is True

    def test_logical_syntax_equivalent(self):
        assert _cgh._is_low_signal_heading(
            "The logical syntax equivalent is:"
        ) is True

    def test_plain_two_name_byline_no_leadin(self):
        assert _cgh._is_low_signal_heading("John Smith") is True

    def test_three_name_byline(self):
        # Three-token byline with no common-title-word anchor.
        assert _cgh._is_low_signal_heading("Chen Wang Liu") is True

    def test_existential_quantifier_short(self):
        assert _cgh._is_low_signal_heading("\u2203x P(x)") is True


class TestWave27LegitimateHeadings:
    """Positive controls — real textbook chapter titles that MUST pass
    through the filter (``_is_low_signal_heading`` returns False).

    Regression guard against the Wave 27 byline / math filters over-
    triggering and demoting real chapter titles to low-signal chrome.
    """

    def test_introduction_to_digital_pedagogy(self):
        assert _cgh._is_low_signal_heading(
            "Introduction to Digital Pedagogy"
        ) is False

    def test_chapter_with_subtitle(self):
        assert _cgh._is_low_signal_heading("Chapter 1: Fundamentals") is False

    def test_european_union_policy(self):
        # Two+ tokens but contains a common-title-word ("european", "policy")
        # so the byline filter must not trip.
        assert _cgh._is_low_signal_heading("European Union Policy") is False

    def test_title_with_common_phrase(self):
        # A 5-token title anchored by common title words ("in", "a",
        # "Age") should not be demoted to low-signal chrome.
        assert _cgh._is_low_signal_heading(
            "Learning in a Connected Age"
        ) is False

    def test_science_of_learning(self):
        assert _cgh._is_low_signal_heading("The Science of Learning") is False

    def test_research_methods_in_education(self):
        assert _cgh._is_low_signal_heading(
            "Research Methods in Education"
        ) is False

    def test_digital_pedagogy_two_tokens(self):
        # Two tokens — "digital" is a common title adjective so the byline
        # detector's 2-3-token path must not fire.
        assert _cgh._is_low_signal_heading("Digital Pedagogy") is False

    def test_introduction_to_ontology(self):
        assert _cgh._is_low_signal_heading(
            "Introduction to Ontology"
        ) is False

    def test_fundamentals_of_accessibility(self):
        assert _cgh._is_low_signal_heading(
            "Fundamentals of Accessibility"
        ) is False

    def test_chapter_colon_title(self):
        assert _cgh._is_low_signal_heading(
            "Chapter 1: The Basics"
        ) is False
