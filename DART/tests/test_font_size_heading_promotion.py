"""Wave 18 tests for font-size-based heading promotion.

The heuristic classifier promotes regex-classified ``PARAGRAPH`` blocks
to heading roles when a matching ``ExtractedTextSpan`` reports a font
size significantly larger than the document's median body font. Bold
tier kicks in between ``_BOLD_HEADING_RATIO`` and the main threshold.

All tests construct synthetic ``RawBlock`` / ``ExtractedTextSpan``
pairs so no real PDF / PyMuPDF is required.
"""

from __future__ import annotations

import pytest

from DART.converter.block_roles import BlockRole, RawBlock
from DART.converter.extractor import ExtractedTextSpan
from DART.converter.heuristic_classifier import (
    HeuristicClassifier,
    _match_block_to_spans,
)


def _span(
    *,
    text: str,
    page: int = 1,
    font_size: float = 11.0,
    is_bold: bool = False,
    is_italic: bool = False,
) -> ExtractedTextSpan:
    return ExtractedTextSpan(
        page=page,
        bbox=(0.0, 0.0, 500.0, 20.0),
        text=text,
        font_size=font_size,
        font_name="Times-Roman",
        is_bold=is_bold,
        is_italic=is_italic,
    )


def _block(text: str, *, page: int = 1) -> RawBlock:
    return RawBlock(text=text, block_id="abc123", page=page)


@pytest.mark.unit
@pytest.mark.dart
class TestMatchBlockToSpans:
    def test_exact_text_match_picks_span(self):
        spans = [
            _span(text="A body sentence.", font_size=11.0),
            _span(text="DISPLAY HEADING", font_size=20.0),
        ]
        match = _match_block_to_spans(_block("DISPLAY HEADING"), spans)
        assert match is not None
        assert match.font_size == 20.0

    def test_page_mismatch_rejects_span(self):
        spans = [_span(text="HEADING", page=2, font_size=18.0)]
        assert _match_block_to_spans(_block("HEADING", page=1), spans) is None

    def test_empty_spans_return_none(self):
        assert _match_block_to_spans(_block("anything"), []) is None

    def test_partial_overlap_below_threshold_rejected(self):
        spans = [_span(text="x", font_size=20.0)]  # tiny overlap
        assert (
            _match_block_to_spans(
                _block("This is a much longer body paragraph."), spans
            )
            is None
        )


@pytest.mark.unit
@pytest.mark.dart
class TestFontSizePromotion:
    def test_large_font_promotes_paragraph_to_section(self):
        spans = [_span(text="LARGE HEADING", font_size=24.0)] + [
            _span(text=f"body sentence {i}", font_size=11.0)
            for i in range(10)
        ]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        blocks = [_block("LARGE HEADING")]
        out = classifier.classify_sync(blocks)
        assert out[0].role == BlockRole.SECTION_HEADING
        assert "font_size" in out[0].attributes
        assert out[0].attributes["heading_text"] == "LARGE HEADING"

    def test_moderate_font_promotes_to_subsection(self):
        spans = [_span(text="Subsection Heading", font_size=17.0)]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        out = classifier.classify_sync([_block("Subsection Heading")])
        # Ratio ~1.55 -> SUBSECTION_HEADING.
        assert out[0].role == BlockRole.SUBSECTION_HEADING

    def test_bold_only_tiebreaker_promotes(self):
        spans = [_span(text="Bold Lead", font_size=13.0, is_bold=True)]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        out = classifier.classify_sync([_block("Bold Lead")])
        # Ratio ~1.18, bold -> SUBSECTION_HEADING via bold tier.
        assert out[0].role == BlockRole.SUBSECTION_HEADING

    def test_no_spans_baseline_preserved(self):
        classifier = HeuristicClassifier()
        out = classifier.classify_sync([_block("some body text")])
        assert out[0].role == BlockRole.PARAGRAPH
        # No font_size attribute when promotion didn't run.
        assert "font_size" not in out[0].attributes

    def test_explicit_regex_heading_not_demoted(self):
        """Chapter-classified block keeps its role even with small font."""
        # Tiny font for the "chapter" span would have demoted if allowed.
        spans = [_span(text="Chapter 3: Thermodynamics", font_size=9.0)]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        out = classifier.classify_sync([_block("Chapter 3: Thermodynamics")])
        # Regex hits CHAPTER_OPENER; font check is a no-op on non-PARAGRAPH.
        assert out[0].role == BlockRole.CHAPTER_OPENER

    def test_small_font_paragraph_stays_paragraph(self):
        spans = [_span(text="not a heading here", font_size=10.0)]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=11.0
        )
        out = classifier.classify_sync([_block("not a heading here")])
        assert out[0].role == BlockRole.PARAGRAPH

    def test_median_computed_lazily_from_spans(self):
        """Without explicit median, the classifier computes one from spans."""
        spans = [
            _span(text="A", font_size=10.0),
            _span(text="B", font_size=12.0),
            _span(text="HUGE", font_size=20.0),
        ]
        classifier = HeuristicClassifier(text_spans=spans)
        assert classifier._median_body_font_size is not None
        assert classifier._median_body_font_size > 0

    def test_heading_promotion_fires_on_page_aware_match(self):
        """Page number carries through when multiple pages have same text."""
        # Use text that won't be regex-matched as a subheading (contains
        # trailing punctuation so _is_valid_subheading rejects it) but
        # still has a large-font span. The promoter only acts on blocks
        # the regex left as PARAGRAPH.
        text = "this is ordinary body prose here."
        spans = [
            _span(text=text, page=1, font_size=10.0),
            _span(text=text, page=2, font_size=22.0),
        ]
        classifier = HeuristicClassifier(
            text_spans=spans, median_body_font_size=10.0
        )
        block_p2 = _block(text, page=2)
        out = classifier.classify_sync([block_p2])
        # Ratio 2.2 -> SECTION_HEADING.
        assert out[0].role == BlockRole.SECTION_HEADING
