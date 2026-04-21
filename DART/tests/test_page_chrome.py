"""Wave 20 tests for ``DART.converter.page_chrome``.

Coverage: the frequency-based running header / footer / page-number
detector plus its stripping companion. All tests exercise the public
:func:`DART.converter.page_chrome.detect_page_chrome` + :func:`strip_page_chrome`
entry points against synthetic form-feed-delimited text — no PyMuPDF or
pdftotext dependency required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from DART.converter.page_chrome import (
    PageChrome,
    _normalise,
    _strip_trailing_digits,
    detect_page_chrome,
    strip_page_chrome,
)


_FORM_FEED = "\x0c"


def _make_pages(*page_texts: str) -> str:
    """Join a sequence of per-page strings with form-feed delimiters."""
    return _FORM_FEED.join(page_texts)


# ---------------------------------------------------------------------------
# Normalisation helpers (unit-level sanity checks so detector logic stays
# anchored to well-tested primitives)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestNormalisationHelpers:
    def test_normalise_strips_whitespace_and_lowercases(self):
        assert _normalise("  Running  Header  ") == "running header"

    def test_normalise_handles_nbsp_unicode(self):
        # NBSP (\xa0) should collapse to a regular space under NFKC.
        assert _normalise("Head\xa0Line") == "head line"

    def test_strip_trailing_digits_extracts_page_number(self):
        prefix, page = _strip_trailing_digits("teaching in a digital age 164")
        assert prefix == "teaching in a digital age"
        assert page == 164

    def test_strip_trailing_digits_handles_bare_number(self):
        prefix, page = _strip_trailing_digits("164")
        assert prefix == ""
        assert page == 164

    def test_strip_trailing_digits_returns_none_when_no_digits(self):
        prefix, page = _strip_trailing_digits("no digits here")
        assert prefix == "no digits here"
        assert page is None


# ---------------------------------------------------------------------------
# detect_page_chrome — positive detections
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestDetectChromePositive:
    def test_simple_identical_header_detected(self):
        pages: List[str] = []
        for i in range(1, 7):
            pages.append(
                f"Book Title\n\nBody content for page {i} goes here with many words.\n"
            )
        chrome = detect_page_chrome(_make_pages(*pages))
        assert "book title" in chrome.headers
        # Headers should now be absent from the stripped stream.
        stripped_joined = _FORM_FEED.join(chrome.stripped_pages)
        assert "Book Title" not in stripped_joined

    def test_header_with_page_number_variable_tail(self):
        pages = []
        for i in range(2, 10):
            pages.append(
                f"Teaching in a Digital Age {i}\n\n"
                "Paragraph of real textbook prose on this page.\n"
                "Additional content that should survive stripping.\n"
            )
        chrome = detect_page_chrome(_make_pages(*pages))
        # Fixed prefix detected, variable tail split off.
        assert any("teaching in a digital age" in h for h in chrome.headers)
        # Per-page page_number_lines populated (at least half the pages).
        assert len(chrome.page_number_lines) >= 4
        # The running header should no longer appear in stripped content.
        stripped_joined = _FORM_FEED.join(chrome.stripped_pages)
        assert "Teaching in a Digital Age 2" not in stripped_joined
        assert "Teaching in a Digital Age 9" not in stripped_joined

    def test_footer_only_chrome_detected(self):
        # Each page has enough unique body lines on top that only the
        # footer line is a repetition candidate at the bottom.
        pages = []
        for i in range(1, 7):
            pages.append(
                f"Unique opening line {i} aaa\n"
                f"Unique content paragraph {i} bbb\n"
                f"Unique body detail {i} ccc\n"
                f"Unique continuation {i} ddd\n"
                f"Unique fourth body sentence {i} eee\n\n"
                "© Copyright Campus Press"
            )
        chrome = detect_page_chrome(_make_pages(*pages))
        assert "© copyright campus press" in chrome.footers
        stripped_joined = _FORM_FEED.join(chrome.stripped_pages)
        assert "© Copyright Campus Press" not in stripped_joined

    def test_bare_page_number_only_chrome_detected(self):
        pages = []
        for i in range(1, 8):
            pages.append(
                f"Real prose content for page {i} with meaningful words.\n"
                "More content here.\n\n"
                f"{i}"
            )
        chrome = detect_page_chrome(_make_pages(*pages))
        # Page-number-only chrome surfaces via page_number_lines (bare-
        # number sentinel key). The headers/footers sets don't include
        # the sentinel itself (it's an internal marker) — but the
        # per-page number map proves detection worked.
        assert len(chrome.page_number_lines) >= 4


@pytest.mark.unit
@pytest.mark.dart
class TestDetectChromeThresholds:
    def test_chrome_on_35_percent_of_10_pages_detected(self):
        # 4 of 10 pages (40%) carry the chrome — above the 30% default.
        pages = []
        for i in range(10):
            if i < 4:
                pages.append(
                    f"Repeating Header\n\nPage {i} body prose content here."
                )
            else:
                pages.append(
                    f"Page {i} body prose content here without the header."
                )
        chrome = detect_page_chrome(_make_pages(*pages))
        assert "repeating header" in chrome.headers

    def test_chrome_on_20_percent_of_10_pages_not_detected(self):
        # 2 of 10 pages (20%) carry the would-be chrome — below the 30%
        # threshold, so it should stay as content.
        pages = []
        for i in range(10):
            if i < 2:
                pages.append(
                    f"Would Be Header\n\nPage {i} body prose content here."
                )
            else:
                pages.append(
                    f"Page {i} body prose content here without the header."
                )
        chrome = detect_page_chrome(_make_pages(*pages))
        assert "would be header" not in chrome.headers


@pytest.mark.unit
@pytest.mark.dart
class TestDetectChromeShortDocuments:
    def test_short_document_no_detection(self):
        # 3 pages is below the min_pages_to_analyze=4 default.
        pages = ["Same Header\n\nBody", "Same Header\n\nBody", "Same Header\n\nBody"]
        chrome = detect_page_chrome(_make_pages(*pages))
        assert chrome.headers == set()
        assert chrome.footers == set()
        # stripped_pages is the input verbatim.
        assert chrome.stripped_pages == pages

    def test_no_form_feeds_returns_empty(self):
        # Single blob without form-feeds = no page structure.
        raw = "Header\n\nBody content here\n\nMore body"
        chrome = detect_page_chrome(raw)
        assert chrome.headers == set()
        assert chrome.footers == set()


@pytest.mark.unit
@pytest.mark.dart
class TestDetectChromeFalsePositiveGuards:
    def test_long_content_line_repeating_not_flagged(self):
        # A sentence long enough to exceed _MAX_CHROME_LINE_LEN (80 chars).
        long_line = (
            "This is a long sentence appearing at the top of multiple "
            "pages but it is clearly real prose content, not a running "
            "header of any kind whatsoever."
        )
        pages = []
        for i in range(6):
            pages.append(f"{long_line}\n\nPage {i} body.")
        chrome = detect_page_chrome(_make_pages(*pages))
        assert not any(long_line.lower() in h for h in chrome.headers)

    def test_chapter_heading_repeating_not_flagged(self):
        # "Chapter 1" appearing on multiple pages must not be treated as
        # chrome — it's structural content.
        pages = []
        for i in range(6):
            pages.append(f"Chapter 1\n\nPage {i} body content here.")
        chrome = detect_page_chrome(_make_pages(*pages))
        assert "chapter 1" not in chrome.headers

    def test_short_prefix_with_variable_digit_not_flagged(self):
        # "A 1", "A 2", "A 3", ... — variable-tail chrome with a
        # sub-3-char fixed prefix. Ambiguous, so guarded out.
        pages = []
        for i in range(1, 7):
            pages.append(f"A {i}\n\nPage {i} body prose content.")
        chrome = detect_page_chrome(_make_pages(*pages))
        assert "a" not in chrome.headers


# ---------------------------------------------------------------------------
# strip_page_chrome
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestStripChrome:
    def test_strip_removes_detected_headers(self):
        pages = [
            f"Book Title\n\nPage {i} content prose body here."
            for i in range(1, 7)
        ]
        raw = _make_pages(*pages)
        chrome = detect_page_chrome(raw)
        stripped = strip_page_chrome(raw, chrome)
        assert "Book Title" not in stripped
        assert "Page 1 content prose body here" in stripped

    def test_strip_preserves_form_feeds(self):
        pages = [
            f"Title X\n\nPage {i} content prose body"
            for i in range(1, 7)
        ]
        raw = _make_pages(*pages)
        chrome = detect_page_chrome(raw)
        stripped = strip_page_chrome(raw, chrome)
        # Same number of pages survives — form feeds preserved so
        # downstream per-page attribution still works.
        assert stripped.count(_FORM_FEED) == raw.count(_FORM_FEED)

    def test_strip_is_idempotent(self):
        pages = [
            f"Bates Chapter\n\nPage {i} content prose body goes here."
            for i in range(1, 8)
        ]
        raw = _make_pages(*pages)
        chrome = detect_page_chrome(raw)
        once = strip_page_chrome(raw, chrome)
        twice = strip_page_chrome(once, chrome)
        assert once == twice

    def test_strip_on_empty_chrome_returns_input_unchanged(self):
        raw = "Just some content\n\nWith a second block."
        empty = PageChrome()
        assert strip_page_chrome(raw, empty) == raw


# ---------------------------------------------------------------------------
# Bbox-based confirmation (text_spans integration)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestBboxConfirmation:
    def test_spans_upgrade_chrome_detection(self):
        # Frequency signal alone is already strong enough; supplying
        # text_spans at the top of each page confirms (and never
        # weakens) the detection.
        pages = [
            f"Running Header\n\nPage {i} body text prose goes here."
            for i in range(1, 7)
        ]
        spans = []
        for i in range(1, 7):
            # Top-of-page header (y0 near 0, y1 small = top 10%).
            spans.append(
                SimpleNamespace(
                    page=i,
                    bbox=(10.0, 5.0, 200.0, 15.0),
                    text="Running Header",
                )
            )
            # Body (middle of page).
            spans.append(
                SimpleNamespace(
                    page=i,
                    bbox=(10.0, 400.0, 200.0, 450.0),
                    text=f"Page {i} body text prose goes here.",
                )
            )
            # Sentinel far-bottom span so the per-page height reads 800.
            spans.append(
                SimpleNamespace(
                    page=i,
                    bbox=(10.0, 790.0, 50.0, 800.0),
                    text="",
                )
            )
        chrome = detect_page_chrome(
            _make_pages(*pages), text_spans=spans
        )
        assert "running header" in chrome.headers


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestUnicodeNormalisation:
    def test_nbsp_header_still_detected(self):
        # Mix regular space + NBSP across pages so the raw bytes differ
        # but the normalised form matches.
        pages = []
        for i in range(1, 7):
            # Alternate NBSP and regular space.
            sep = "\xa0" if i % 2 == 0 else " "
            pages.append(
                f"Campus{sep}Guide\n\nPage {i} body prose content here."
            )
        chrome = detect_page_chrome(_make_pages(*pages))
        assert "campus guide" in chrome.headers
