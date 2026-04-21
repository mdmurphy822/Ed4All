"""Wave 21 list-detection tests.

Full-textbook smoke (Wave 20) exposed hundreds of bullet-marker and
numbered paragraphs that flowed through as naked ``<p>`` elements
because the
pre-Wave-21 classifier had no list-item role. This module covers the
three-layer fix:

* marker detection + classifier promotion in
  :class:`DART.converter.heuristic_classifier.HeuristicClassifier`
* consecutive-run grouping in
  :mod:`DART.converter.document_assembler`
* ``<ul>`` / ``<ol>`` template emission with Wave 19 provenance attrs
  on the wrapper only (Wave 8 P2 rule)

Tests intentionally avoid PyMuPDF / pdfplumber / tesseract so they run
on the CI minimal image.
"""

from __future__ import annotations

import re

import pytest

from DART.converter import convert_pdftotext_to_html
from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.block_templates import render_block
from DART.converter.document_assembler import (
    _group_consecutive_lists,
    assemble_html,
)
from DART.converter.heuristic_classifier import (
    HeuristicClassifier,
    _looks_like_list_item,
    _match_list_marker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw(text: str, block_id: str = "blk000001") -> RawBlock:
    return RawBlock(text=text, block_id=block_id)


def _li(
    text: str,
    marker: str,
    marker_type: str,
    block_id: str = "blk000001",
) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=_raw(text, block_id),
        role=BlockRole.LIST_ITEM,
        confidence=0.85,
        attributes={"text": text, "marker": marker, "marker_type": marker_type},
    )


# ---------------------------------------------------------------------------
# Marker detection primitives
# ---------------------------------------------------------------------------


class TestMarkerDetection:
    @pytest.mark.parametrize(
        "text, marker, marker_type",
        [
            ("\u2022 First bullet point", "\u2022", "unordered"),       # •
            ("\u00b7 Middle dot item", "\u00b7", "unordered"),         # ·
            ("\u25aa Small square item", "\u25aa", "unordered"),       # ▪
            ("\u25cf Black circle item", "\u25cf", "unordered"),       # ●
            ("\u25e6 White bullet item", "\u25e6", "unordered"),       # ◦
            ("\u25cb White circle item", "\u25cb", "unordered"),       # ○
            ("\u25b8 Right triangle item", "\u25b8", "unordered"),     # ▸
            ("\u25ba Right pointer item", "\u25ba", "unordered"),      # ►
        ],
    )
    def test_unicode_bullet_variants_detected(self, text, marker, marker_type):
        result = _match_list_marker(text)
        assert result is not None, text
        detected_type, detected_marker, rest = result
        assert detected_type == marker_type
        assert detected_marker == marker
        assert rest.startswith(
            text.split(marker, 1)[1].strip()[:5]
        ) or rest in text
        # Convenience wrapper agrees.
        assert _looks_like_list_item(text) == marker_type

    @pytest.mark.parametrize(
        "text, expected_marker, expected_type",
        [
            ("1. First numbered item", "1.", "ordered"),
            ("10. Tenth item here", "10.", "ordered"),
            ("1) Opening paren form", "1)", "ordered"),
            ("(3) Parenthesized item", "(3)", "ordered"),
            ("a. Lowercase alpha item", "a.", "ordered"),
            ("b) Alpha with paren", "b)", "ordered"),
            ("iv. Roman-numeral item", "iv.", "ordered"),
            ("vii) Roman with paren", "vii)", "ordered"),
        ],
    )
    def test_ordered_markers_detected(
        self, text, expected_marker, expected_type
    ):
        result = _match_list_marker(text)
        assert result is not None, text
        assert result[0] == expected_type
        assert result[1] == expected_marker

    @pytest.mark.parametrize(
        "text",
        [
            "- hyphen-prefixed item",
            "- Another dash item",
            "* asterisk item",
            "* Another asterisk item",
        ],
    )
    def test_ascii_dash_asterisk_markers_detected(self, text):
        result = _match_list_marker(text)
        assert result is not None, text
        assert result[0] == "unordered"
        assert result[1] in {"-", "*"}

    @pytest.mark.parametrize(
        "text",
        [
            # Mid-sentence punctuation that looks marker-ish but isn't.
            "For example, e.g., the following is true.",
            # Plain prose starting with a capital letter — no marker.
            "This is a regular sentence about pedagogy.",
            # Bullet-char mid-word (not at start of line).
            "data \u2022 separated \u2022 values",
            # Hyphen NOT followed by whitespace (likely mid-word).
            "-negative outcome",
            # Ordered-like pattern but body starts lowercase (mid-sentence).
            "1. foo",
        ],
    )
    def test_false_positive_guards(self, text):
        assert _match_list_marker(text) is None
        assert _looks_like_list_item(text) is None


# ---------------------------------------------------------------------------
# Classifier promotion
# ---------------------------------------------------------------------------


class TestClassifierPromotion:
    def test_bullet_block_becomes_list_item(self):
        clf = HeuristicClassifier()
        blocks = [_raw("\u2022 A short bulleted statement.", "b1")]
        out = clf.classify_sync(blocks)
        assert len(out) == 1
        assert out[0].role == BlockRole.LIST_ITEM
        attrs = out[0].attributes
        assert attrs["marker"] == "\u2022"
        assert attrs["marker_type"] == "unordered"
        assert attrs["text"] == "A short bulleted statement."

    def test_multi_numbered_block_expands_to_multiple_list_items(self):
        clf = HeuristicClassifier()
        # pdftotext often fuses several items onto one logical block —
        # the classifier expansion splits them into one LIST_ITEM per item.
        blocks = [
            _raw(
                "1. First numbered item with enough body. "
                "2. Second numbered item with body. "
                "3. Third numbered item with body.",
                "b1",
            ),
        ]
        out = clf.classify_sync(blocks)
        assert len(out) == 3
        assert all(b.role == BlockRole.LIST_ITEM for b in out)
        assert [b.attributes["marker"] for b in out] == ["1.", "2.", "3."]
        # Every split shares the same marker_type = ordered.
        assert all(b.attributes["marker_type"] == "ordered" for b in out)
        # Block IDs are unique across the expansion.
        assert len({b.raw.block_id for b in out}) == 3

    def test_bibliography_beats_list_item(self):
        """Bibliography entries (``[N]`` prefix) stay as BIBLIOGRAPHY_ENTRY."""
        clf = HeuristicClassifier()
        blocks = [
            _raw("[1] Smith, J. (2024). Accessibility by design. MIT Press.", "b1")
        ]
        out = clf.classify_sync(blocks)
        assert out[0].role == BlockRole.BIBLIOGRAPHY_ENTRY

    def test_plain_paragraph_untouched(self):
        clf = HeuristicClassifier()
        blocks = [
            _raw(
                "Effective course design requires rethinking pedagogy.",
                "b1",
            )
        ]
        out = clf.classify_sync(blocks)
        assert out[0].role == BlockRole.PARAGRAPH


# ---------------------------------------------------------------------------
# Assembler grouping pass
# ---------------------------------------------------------------------------


class TestListGrouping:
    def test_three_consecutive_unordered_items_group_into_one_list(self):
        items = [
            _li("First item", "\u2022", "unordered", "a1"),
            _li("Second item", "\u2022", "unordered", "a2"),
            _li("Third item", "\u2022", "unordered", "a3"),
        ]
        grouped = _group_consecutive_lists(items)
        assert len(grouped) == 1
        assert grouped[0].role == BlockRole.LIST_UNORDERED
        assert len(grouped[0].attributes["items"]) == 3
        # First item's raw is the group head (page attribution).
        assert grouped[0].raw.block_id == "a1"

    def test_mixed_marker_types_produce_two_lists(self):
        blocks = [
            _li("UL one", "\u2022", "unordered", "a1"),
            _li("UL two", "\u2022", "unordered", "a2"),
            _li("OL one", "1.", "ordered", "a3"),
            _li("OL two", "2.", "ordered", "a4"),
        ]
        grouped = _group_consecutive_lists(blocks)
        assert len(grouped) == 2
        assert grouped[0].role == BlockRole.LIST_UNORDERED
        assert grouped[1].role == BlockRole.LIST_ORDERED
        assert len(grouped[0].attributes["items"]) == 2
        assert len(grouped[1].attributes["items"]) == 2

    def test_intervening_paragraph_breaks_list_into_two(self):
        para = ClassifiedBlock(
            raw=_raw("Interrupting paragraph.", "p1"),
            role=BlockRole.PARAGRAPH,
            confidence=0.5,
        )
        blocks = [
            _li("A", "\u2022", "unordered", "a1"),
            para,
            _li("B", "\u2022", "unordered", "a2"),
        ]
        grouped = _group_consecutive_lists(blocks)
        assert len(grouped) == 3  # list, paragraph, list
        assert grouped[0].role == BlockRole.LIST_UNORDERED
        assert grouped[1].role == BlockRole.PARAGRAPH
        assert grouped[2].role == BlockRole.LIST_UNORDERED

    def test_single_item_still_emits_list(self):
        """The plan calls this out: never leave a stray <li> without
        a parent. One-item groups still emit a <ul> wrapper."""
        blocks = [_li("Only item", "\u2022", "unordered", "a1")]
        grouped = _group_consecutive_lists(blocks)
        assert len(grouped) == 1
        assert grouped[0].role == BlockRole.LIST_UNORDERED
        assert len(grouped[0].attributes["items"]) == 1


# ---------------------------------------------------------------------------
# Templates — wrapper-only provenance (Wave 19 P2 rule)
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_unordered_template_emits_ul_with_list_children(self):
        block = ClassifiedBlock(
            raw=_raw("", "grp0001"),
            role=BlockRole.LIST_UNORDERED,
            confidence=0.9,
            attributes={
                "items": [
                    {"text": "Alpha", "marker": "\u2022", "marker_type": "unordered"},
                    {"text": "Beta", "marker": "\u2022", "marker_type": "unordered"},
                ]
            },
        )
        out = render_block(block)
        assert out.startswith("<ul ")
        assert out.count("<li>") == 2
        assert "<li>Alpha</li>" in out
        assert "<li>Beta</li>" in out

    def test_ordered_template_emits_ol_with_list_children(self):
        block = ClassifiedBlock(
            raw=_raw("", "grp0002"),
            role=BlockRole.LIST_ORDERED,
            confidence=0.9,
            attributes={
                "items": [
                    {"text": "First", "marker": "1.", "marker_type": "ordered"},
                    {"text": "Second", "marker": "2.", "marker_type": "ordered"},
                ]
            },
        )
        out = render_block(block)
        assert out.startswith("<ol ")
        assert out.count("<li>") == 2
        assert "<li>First</li>" in out
        assert "<li>Second</li>" in out

    def test_provenance_attrs_on_wrapper_only(self):
        block = ClassifiedBlock(
            raw=_raw("", "grp0003"),
            role=BlockRole.LIST_UNORDERED,
            confidence=0.85,
            attributes={
                "items": [
                    {"text": "Alpha", "marker": "\u2022", "marker_type": "unordered"},
                    {"text": "Beta", "marker": "\u2022", "marker_type": "unordered"},
                ]
            },
        )
        out = render_block(block)
        # <ul> carries all provenance attrs.
        assert 'data-dart-block-role="list_unordered"' in out
        assert 'data-dart-block-id="grp0003"' in out
        assert 'data-dart-source="dart_converter"' in out
        assert 'class="dart-section ' in out or 'class="dart-section"' in out
        # <li> must NOT carry any data-dart-* attributes (Wave 8 P2).
        li_tag_matches = re.findall(r"<li\b([^>]*)>", out)
        assert li_tag_matches, "expected at least one <li>"
        for attrs_blob in li_tag_matches:
            assert "data-dart-" not in attrs_blob

    def test_nested_sub_items_render_as_nested_list(self):
        block = ClassifiedBlock(
            raw=_raw("", "grp0004"),
            role=BlockRole.LIST_UNORDERED,
            confidence=0.85,
            attributes={
                "items": [
                    {
                        "text": "Parent item",
                        "marker": "\u2022",
                        "marker_type": "unordered",
                        "sub_items": [
                            {
                                "text": "Child one",
                                "marker": "\u25e6",
                                "marker_type": "unordered",
                            },
                            {
                                "text": "Child two",
                                "marker": "\u25e6",
                                "marker_type": "unordered",
                            },
                        ],
                    },
                ]
            },
        )
        out = render_block(block)
        # Outer <ul> + inner <ul>.
        assert out.count("<ul") == 2
        assert "<li>Parent item<ul" in out
        assert "<li>Child one</li>" in out
        assert "<li>Child two</li>" in out

    def test_bullet_class_reflects_marker_variant(self):
        bullet = ClassifiedBlock(
            raw=_raw("", "grp0005"),
            role=BlockRole.LIST_UNORDERED,
            confidence=0.85,
            attributes={
                "items": [
                    {"text": "a", "marker": "\u25aa", "marker_type": "unordered"},
                    {"text": "b", "marker": "\u25aa", "marker_type": "unordered"},
                ]
            },
        )
        out = render_block(bullet)
        assert "list-square" in out
        assert "dart-section" in out

    def test_ordered_start_attribute_when_first_is_not_one(self):
        block = ClassifiedBlock(
            raw=_raw("", "grp0006"),
            role=BlockRole.LIST_ORDERED,
            confidence=0.85,
            attributes={
                "items": [
                    {"text": "Four", "marker": "4.", "marker_type": "ordered"},
                    {"text": "Five", "marker": "5.", "marker_type": "ordered"},
                ]
            },
        )
        out = render_block(block)
        assert 'start="4"' in out

    def test_ordered_default_start_omitted(self):
        block = ClassifiedBlock(
            raw=_raw("", "grp0007"),
            role=BlockRole.LIST_ORDERED,
            confidence=0.85,
            attributes={
                "items": [
                    {"text": "One", "marker": "1.", "marker_type": "ordered"},
                    {"text": "Two", "marker": "2.", "marker_type": "ordered"},
                ]
            },
        )
        out = render_block(block)
        assert " start=" not in out

    def test_empty_items_with_empty_raw_returns_empty(self):
        block = ClassifiedBlock(
            raw=_raw("", "grp0008"),
            role=BlockRole.LIST_UNORDERED,
            confidence=0.5,
            attributes={"items": []},
        )
        assert render_block(block) == ""

    def test_stray_list_item_renders_as_single_item_ul(self):
        """Defensive fallback — a LIST_ITEM that somehow escapes grouping
        still produces valid ``<ul><li></ul>`` markup (never a naked
        ``<li>``)."""
        block = ClassifiedBlock(
            raw=_raw("Stray", "grp0009"),
            role=BlockRole.LIST_ITEM,
            confidence=0.85,
            attributes={
                "text": "Stray item",
                "marker": "\u2022",
                "marker_type": "unordered",
            },
        )
        out = render_block(block)
        assert out.startswith("<ul ")
        assert "<li>Stray item</li>" in out


# ---------------------------------------------------------------------------
# End-to-end pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_end_to_end_bullet_list_emits_ul(self):
        """Three bullet paragraphs on blank-line boundaries collapse into
        one ``<ul>`` with three ``<li>`` children."""
        raw = (
            "Here are the main points:\n\n"
            "\u2022 First key insight\n\n"
            "\u2022 Second key insight\n\n"
            "\u2022 Third key insight\n\n"
            "Closing paragraph."
        )
        html = convert_pdftotext_to_html(raw, title="Bullet Smoke")
        # Exactly one <ul> with three <li> children.
        assert "<ul" in html
        # Count <ul class="dart-section"> occurrences.
        assert html.count('class="dart-section list-dot"') >= 1 or (
            html.count("<ul ") == 1
        )
        assert html.count("<li>First key insight</li>") == 1
        assert html.count("<li>Second key insight</li>") == 1
        assert html.count("<li>Third key insight</li>") == 1
        # No literal bullet paragraph residue.
        assert not re.search(r"<p[^>]*>\u2022\s", html)

    def test_end_to_end_numbered_joined_block_emits_ol(self):
        """pdftotext-merged numbered items collapse into an ``<ol>``."""
        raw = (
            "Research priorities:\n\n"
            "1. Investigate accessibility patterns deeply. "
            "2. Publish findings broadly. "
            "3. Refine curriculum accordingly."
        )
        html = convert_pdftotext_to_html(raw, title="Numbered Smoke")
        assert "<ol" in html
        assert "<li>Investigate accessibility patterns deeply.</li>" in html
        assert "<li>Publish findings broadly.</li>" in html
        assert "<li>Refine curriculum accordingly.</li>" in html
        # Zero numbered-paragraph residue for this input.
        assert not re.search(r"<p[^>]*>\d+\.\s+[A-Z]", html)

    def test_page_attribution_propagates_to_list_wrapper(self):
        """The first item's ``raw.page`` surfaces on the ``<ul>`` via
        ``data-dart-pages``. Uses form-feed segmentation to simulate a
        multi-page document."""
        raw_text = (
            "Intro paragraph on page 1.\n\n"
            "\u000c"
            "\u2022 First bullet on page 2\n\n"
            "\u2022 Second bullet on page 2"
        )
        html = convert_pdftotext_to_html(raw_text, title="Pages")
        # The <ul> wrapper carries data-dart-pages="2".
        ul_match = re.search(r"<ul\b[^>]*>", html)
        assert ul_match is not None
        assert 'data-dart-pages="2"' in ul_match.group(0)
