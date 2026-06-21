"""Course-quality remediation P0: DART heading-classifier source fixes.

Two fixes guarding the OpenStax re-conversion (see
``plans/finegrain/course-quality-remediation-2026-06-21.md`` Lane 1):

1. **Font-size promoter content guard** — the heuristic classifier's
   font-size heading promoter (``_maybe_promote_by_font_size``) must
   REFUSE to promote answer-key / numeric runs even when they render at
   a heading-sized (>= 1.9x median) font. OpenStax exercise-answer
   blocks ("78 41. 900 42. 800", "51,493 ⓐ 1, ⓑ 4 …", "14 < 21 90.")
   were becoming spurious <h2>/<h3> and poisoning the document
   structure.

2. **Fused section-opener split** — real section titles ("1.1
   Introduction to Whole Numbers") are glued to their entire
   Learning-Objectives block into one long body-sized <p>, so they were
   never promoted to headings. The classifier now emits a
   SECTION_HEADING for the leading "N.M Title" prefix of such a fused
   opener.

All tests construct synthetic ``RawBlock`` / ``ExtractedTextSpan`` pairs
so no real PDF / PyMuPDF is required.
"""

from __future__ import annotations

import pytest

from DART.converter.block_roles import BlockRole, RawBlock
from DART.converter.extractor import ExtractedTextSpan
from DART.converter.heuristic_classifier import (
    HeuristicClassifier,
    _is_numeric_answer_key_heading,
)


def _span(
    *,
    text: str,
    page: int = 1,
    font_size: float = 11.0,
    is_bold: bool = False,
) -> ExtractedTextSpan:
    return ExtractedTextSpan(
        page=page,
        bbox=(0.0, 0.0, 500.0, 20.0),
        text=text,
        font_size=font_size,
        font_name="Times-Roman",
        is_bold=is_bold,
        is_italic=False,
    )


def _block(text: str, *, page: int = 1) -> RawBlock:
    return RawBlock(text=text, block_id="b1", page=page, extractor="pdftotext")


# ---------------------------------------------------------------------------
# Fix 1: font-size promoter content guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestNumericAnswerKeyHelper:
    def test_numeric_run_flagged(self):
        assert _is_numeric_answer_key_heading("78 41. 900 42. 800")

    def test_comparison_run_flagged(self):
        assert _is_numeric_answer_key_heading("14 < 21 90. 17 < 35")

    def test_circled_letter_answer_flagged(self):
        assert _is_numeric_answer_key_heading("51,493 ⓐ 1, ⓑ 4")

    def test_answer_key_solution_numbers_flagged(self):
        assert _is_numeric_answer_key_heading("550 47. 22,335 48. 39,075")

    def test_real_section_title_not_flagged(self):
        assert not _is_numeric_answer_key_heading(
            "1.1 Introduction to Whole Numbers"
        )

    def test_real_heading_keyword_not_flagged(self):
        assert not _is_numeric_answer_key_heading("Prime Factorization")

    def test_short_numbered_title_not_flagged(self):
        # One numeric token, several word tokens → density well below 0.5.
        assert not _is_numeric_answer_key_heading("3 Properties of Real Numbers")


@pytest.mark.unit
@pytest.mark.dart
class TestFontSizeContentGuard:
    def test_circled_letter_run_stays_paragraph_at_large_font(self):
        """A >1.9x-ratio circled-letter answer-key run must NOT be promoted.

        ``51,493 ⓐ 1, ⓑ 4`` is the canonical OpenStax answer-key noise the
        plan cites as becoming a spurious <h2>. It reaches the font-size
        promoter as a PARAGRAPH (no regex heading match), so without the
        content guard a >=1.9x font ratio would promote it.
        """
        noise = "51,493 ⓐ 1, ⓑ 4"
        spans = [_span(text=noise, font_size=24.0, is_bold=True)]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        out = classifier.classify_sync([_block(noise)])
        assert out[0].role == BlockRole.PARAGRAPH
        # Guard fired BEFORE the promotion attrs were stamped.
        assert "font_size" not in out[0].attributes

    def test_comparison_run_stays_paragraph_at_large_font(self):
        noise = "< 21 90 17 < 35"
        spans = [_span(text=noise, font_size=22.0)]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        out = classifier.classify_sync([_block(noise)])
        assert out[0].role == BlockRole.PARAGRAPH

    def test_numeric_answer_run_not_promoted_to_footnote_either(self):
        """A footnote-shaped numeric answer run must also stay PARAGRAPH.

        ``78 41. 900 42. 800`` opens with a digit + space, tripping the
        FOOTNOTE marker regex. The content guard rejects it from the
        FOOTNOTE arm too, so it degrades to PARAGRAPH (never a heading,
        never a phantom footnote).
        """
        noise = "78 41. 900 42. 800"
        out = HeuristicClassifier().classify_sync([_block(noise)])
        assert out[0].role == BlockRole.PARAGRAPH

    def test_genuine_title_still_promoted(self):
        """Regression guard must NOT block real title-cased headings.

        A long title-case heading that is NOT regex-classified (trailing
        word count keeps it out of the short-subheading arm) still gets
        promoted by the font-size pass.
        """
        title = "the properties of multiplication and division"
        spans = [_span(text=title, font_size=24.0)] + [
            _span(text=f"body {i}", font_size=11.0) for i in range(8)
        ]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        out = classifier.classify_sync([_block(title)])
        assert out[0].role == BlockRole.SECTION_HEADING


# ---------------------------------------------------------------------------
# Fix 2: fused section-opener split
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestFusedSectionOpenerSplit:
    def _classify(self, text: str):
        clf = HeuristicClassifier()
        return clf.classify_sync([_block(text)])[0]

    def test_fused_opener_yields_section_heading(self):
        fused = (
            "1.1 Introduction to Whole Numbers Learning Objectives By the "
            "end of this section, you will be able to: Identify counting "
            "numbers and whole numbers, model whole numbers, and "
            "identify the place value of a digit."
        )
        result = self._classify(fused)
        assert result.role == BlockRole.SECTION_HEADING
        assert result.attributes.get("split_from_fused_opener") is True
        assert result.attributes["heading_text"] == (
            "1.1 Introduction to Whole Numbers"
        )
        assert result.attributes["section_number"] == "1.1"
        assert result.attributes["level"] == 3

    def test_single_level_number_is_h2(self):
        fused = (
            "2 Introduction to the Language of Algebra Learning Objectives "
            "By the end of this section you will be able to use variables "
            "and algebraic symbols."
        )
        result = self._classify(fused)
        assert result.role == BlockRole.SECTION_HEADING
        assert result.attributes["level"] == 2
        assert result.attributes["heading_text"] == (
            "2 Introduction to the Language of Algebra"
        )

    def test_chapter_prefixed_opener(self):
        fused = (
            "Chapter 1 Foundations Learning Objectives In this chapter you "
            "will review the language of whole numbers and integers."
        )
        result = self._classify(fused)
        assert result.role == BlockRole.SECTION_HEADING
        assert "Foundations" in result.attributes["heading_text"]

    def test_learning_outcomes_marker_also_matches(self):
        fused = (
            "1.3 Add and Subtract Integers Learning Outcomes After "
            "completing this section, students model addition of integers."
        )
        result = self._classify(fused)
        assert result.role == BlockRole.SECTION_HEADING
        assert result.attributes["heading_text"] == (
            "1.3 Add and Subtract Integers"
        )

    def test_plain_paragraph_not_misclassified(self):
        # No leading number + no learning-objectives marker → stays prose.
        result = self._classify(
            "Whole numbers are the counting numbers and zero. We use them "
            "every day to count objects and measure quantities."
        )
        assert result.role == BlockRole.PARAGRAPH

    def test_standalone_numbered_heading_unaffected(self):
        # A short standalone "N.M Title" with no fused objectives still
        # routes through the existing dotted/paper-section regex path.
        result = self._classify("2.1 Related Work")
        assert result.role in {
            BlockRole.SECTION_HEADING,
            BlockRole.SUBSECTION_HEADING,
        }
