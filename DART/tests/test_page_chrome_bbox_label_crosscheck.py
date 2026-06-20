"""OQ-3 tests: bbox printed-label CROSS-CHECK.

The page-chrome detector confirms ``(physical, printed)`` pairs only from
digit-tailed chrome lines in the pdftotext stream — sparse on corpora whose
page number sits alone in a margin. The bbox cross-check
(:func:`confirm_printed_labels_by_bbox` /
:func:`enrich_page_chrome_with_bbox`) scans PyMuPDF span bboxes in the
top/bottom margin bands for SHORT digit tokens and merges any
consistency-confirmed pairs into ``PageChrome.page_number_lines`` so MORE
pages resolve to ``printed`` / ``interpolated`` via
:func:`derive_page_label_map`.

These tests are fully synthetic — no PyMuPDF / pdftotext required. Spans are
``SimpleNamespace(page=..., bbox=(x0, y0, x1, y1), text=...)`` matching the
real :class:`ExtractedTextSpan` surface the cross-check reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from DART.converter.page_chrome import (
    PAGE_KIND_INTERPOLATED,
    PAGE_KIND_PHYSICAL,
    PAGE_KIND_PRINTED,
    PageChrome,
    confirm_printed_labels_by_bbox,
    derive_page_label_map,
    enrich_page_chrome_with_bbox,
)


_PAGE_HEIGHT = 800.0


def _footer_number_span(page: int, printed: int) -> SimpleNamespace:
    """A bare printed-number span in the BOTTOM 10% margin band."""
    return SimpleNamespace(
        page=page,
        bbox=(290.0, 760.0, 310.0, 775.0),  # y1=775 >= 0.90*800=720
        text=str(printed),
    )


def _header_number_span(page: int, printed: int) -> SimpleNamespace:
    """A bare printed-number span in the TOP 10% margin band."""
    return SimpleNamespace(
        page=page,
        bbox=(290.0, 5.0, 310.0, 20.0),  # y0=5 <= 0.10*800=80
        text=str(printed),
    )


def _body_span(page: int, text: str) -> SimpleNamespace:
    """A body-text span squarely in the MIDDLE of the page (no band)."""
    return SimpleNamespace(
        page=page,
        bbox=(60.0, 400.0, 540.0, 430.0),
        text=text,
    )


def _height_sentinel(page: int) -> SimpleNamespace:
    """A far-bottom empty-ish span so per-page height reads ~800."""
    return SimpleNamespace(
        page=page,
        bbox=(10.0, 790.0, 20.0, _PAGE_HEIGHT),
        text=".",
    )


@pytest.mark.unit
@pytest.mark.dart
class TestBboxLabelCrossCheck:
    def test_monotonic_footer_run_confirms_pairs(self):
        # Printed footer numbers 11..16 on physical pages 1..6 -> a clean
        # monotonic constant-offset (+10) run. Every page is confirmed.
        spans: List[SimpleNamespace] = []
        for phys in range(1, 7):
            printed = phys + 10
            spans.append(_footer_number_span(phys, printed))
            spans.append(_body_span(phys, f"Body prose for page {printed}."))
            spans.append(_height_sentinel(phys))

        additions = confirm_printed_labels_by_bbox(spans)
        assert additions == {p: str(p + 10) for p in range(1, 7)}

    def test_enrich_increases_printed_and_interpolated_labels(self):
        # Chrome alone confirms only ONE pair (physical 3 -> printed 13).
        # That single pair gives no monotonic run for interpolation, so
        # derive_page_label_map leaves the other pages physical.
        chrome_only = PageChrome(page_number_lines={3: "13"})
        total = 6
        before = derive_page_label_map(
            dict(chrome_only.page_number_lines), total_pages=total
        )
        printed_before = sum(
            1 for lbl in before.values() if lbl.kind == PAGE_KIND_PRINTED
        )
        interp_before = sum(
            1 for lbl in before.values() if lbl.kind == PAGE_KIND_INTERPOLATED
        )
        assert printed_before == 1
        assert interp_before == 0

        # Now add a monotonic bbox footer run (printed = physical + 10)
        # across pages 1..6. The bbox cross-check confirms the run; the
        # chrome pair at page 3 (printed 13) AGREES with it (3 + 10 = 13).
        chrome = PageChrome(page_number_lines={3: "13"})
        spans: List[SimpleNamespace] = []
        for phys in range(1, 7):
            spans.append(_footer_number_span(phys, phys + 10))
            spans.append(_height_sentinel(phys))
        added = enrich_page_chrome_with_bbox(chrome, spans)
        # Pages 1,2,4,5,6 added (page 3 already chrome-confirmed).
        assert added == 5

        after = derive_page_label_map(
            dict(chrome.page_number_lines), total_pages=total
        )
        printed_after = sum(
            1 for lbl in after.values() if lbl.kind == PAGE_KIND_PRINTED
        )
        # MORE printed labels than chrome-alone, and the offset run is now
        # consistent -> printed numbers track the +10 offset.
        assert printed_after > printed_before
        assert printed_after == 6
        for phys in range(1, 7):
            assert after[phys].printed == phys + 10

    def test_chrome_pair_never_overridden(self):
        # Chrome confirmed physical 3 -> printed 99 (an odd offset). A bbox
        # footer run says printed = physical + 10. The chrome pair must NOT
        # be overridden — its key is omitted from the additions.
        chrome = PageChrome(page_number_lines={3: "99"})
        spans: List[SimpleNamespace] = []
        for phys in range(1, 7):
            spans.append(_footer_number_span(phys, phys + 10))
            spans.append(_height_sentinel(phys))
        enrich_page_chrome_with_bbox(chrome, spans)
        # Page 3 keeps its chrome line; the bbox value (13) never replaces 99.
        assert chrome.page_number_lines[3] == "99"


@pytest.mark.unit
@pytest.mark.dart
class TestBboxAntiFabrication:
    def test_body_text_number_not_taken_as_label(self):
        # A number that lives in BODY text (middle of the page), never in a
        # margin band, must never become a confirmed label — even if it forms
        # a monotonic run by coincidence.
        spans: List[SimpleNamespace] = []
        for phys in range(1, 7):
            # The "number" is embedded in a body-position span.
            spans.append(_body_span(phys, str(phys + 10)))
            spans.append(_height_sentinel(phys))
        additions = confirm_printed_labels_by_bbox(spans)
        assert additions == {}

    def test_isolated_non_monotonic_margin_number_dropped(self):
        # Two pages, each with a single margin number, but they do NOT form a
        # monotonic constant-offset run (47 then 12) and corroborate no chrome
        # pair. Neither is confirmed.
        spans = [
            _footer_number_span(1, 47),
            _height_sentinel(1),
            _footer_number_span(2, 12),
            _height_sentinel(2),
        ]
        additions = confirm_printed_labels_by_bbox(spans)
        assert additions == {}

    def test_single_isolated_margin_number_without_chrome_dropped(self):
        # One lone margin number on one page, no run, no chrome to
        # corroborate -> dropped (a single arbitrary number is never a label).
        spans = [_footer_number_span(4, 88), _height_sentinel(4)]
        additions = confirm_printed_labels_by_bbox(spans)
        assert additions == {}

    def test_lone_margin_number_corroborating_chrome_is_kept(self):
        # A SINGLE margin candidate that AGREES with an existing chrome pair
        # (same physical page, same printed number) is high-confidence even
        # without a run — corroboration path. But since it duplicates the
        # chrome key, the addition is omitted (chrome wins). Confirm it does
        # NOT crash and returns no NEW key for the duplicate.
        chrome_lines = {4: "88"}
        spans = [_footer_number_span(4, 88), _height_sentinel(4)]
        additions = confirm_printed_labels_by_bbox(
            spans, existing_page_number_lines=chrome_lines
        )
        # Key 4 already chrome-confirmed -> omitted from additions.
        assert 4 not in additions


@pytest.mark.unit
@pytest.mark.dart
class TestBboxNoRegression:
    def test_no_spans_returns_empty(self):
        assert confirm_printed_labels_by_bbox([]) == {}
        assert confirm_printed_labels_by_bbox(None) == {}

    def test_enrich_no_spans_is_noop(self):
        chrome = PageChrome(page_number_lines={3: "13"})
        before = dict(chrome.page_number_lines)
        assert enrich_page_chrome_with_bbox(chrome, []) == 0
        assert enrich_page_chrome_with_bbox(chrome, None) == 0
        assert chrome.page_number_lines == before

    def test_missing_bbox_or_height_skips_gracefully(self):
        # Spans with no usable bbox / zero page height must not crash and
        # contribute no pairs.
        spans = [
            SimpleNamespace(page=1, bbox=(), text="47"),
            SimpleNamespace(page=2, bbox=(1.0, 2.0), text="48"),
            SimpleNamespace(page="x", bbox=(0, 0, 10, 10), text="49"),
        ]
        assert confirm_printed_labels_by_bbox(spans) == {}

    def test_top_margin_run_also_confirms(self):
        # Page numbers can live in the header. A monotonic TOP-band run is
        # confirmed the same as a bottom-band run.
        spans: List[SimpleNamespace] = []
        for phys in range(1, 6):
            spans.append(_header_number_span(phys, phys + 4))
            spans.append(_height_sentinel(phys))
        additions = confirm_printed_labels_by_bbox(spans)
        assert additions == {p: str(p + 4) for p in range(1, 6)}
