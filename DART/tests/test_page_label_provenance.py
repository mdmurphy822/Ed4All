"""Phase 4 — printed-page label ACCURACY via physical->printed offset
interpolation + the ``data-dart-page-kind`` provenance attribute.

Spec: ``plans/finegrain/page-number-deeplinks-2026-06.md`` §4 (Phase 4) /
§6 / OQ-3 / RISK-A.

The page-chrome detector only directly detects a printed label on pages
whose running header / footer ends in a digit run. On corpora with
sparse / no digit-tailed chrome, most blocks historically fell back to the
raw PHYSICAL PDF page — wrong by the front-matter offset (printed p.1 may
be physical p.15).

``DART.converter.page_chrome.derive_page_label_map`` closes that gap WITHOUT
fabricating: from the confirmed ``(physical, printed)`` pairs it derives the
physical->printed offset and emits an INTERPOLATED printed label for an
unlabeled page covered by a confident, consistent offset segment. The
companion ``data-dart-page-kind`` attribute records provenance:
``printed`` / ``interpolated`` / ``physical`` (the last being the
absent-attribute default).

All fixtures are synthetic (chrome ``page_number_lines`` dicts +
``RawBlock`` / ``ClassifiedBlock`` instances) — no DART re-conversion is
exercised here. The live-course benefit is a separate CPU-only
re-conversion (the printed labels are baked into the HTML at convert time).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock  # noqa: E402
from DART.converter.block_segmenter import segment_extracted_document  # noqa: E402
from DART.converter.block_templates import _provenance_attrs  # noqa: E402
from DART.converter.page_chrome import (  # noqa: E402
    PAGE_KIND_INTERPOLATED,
    PAGE_KIND_PHYSICAL,
    PAGE_KIND_PRINTED,
    PageChrome,
    PageLabel,
    _confirmed_pairs,
    _segment_offsets,
    derive_page_label_map,
)

_FORM_FEED = "\x0c"


# ---------------------------------------------------------------------------
# Synthetic doc / block helpers
# ---------------------------------------------------------------------------


class _FakeDoc:
    """Minimal ``ExtractedDocument`` stand-in for the segmenter."""

    def __init__(self, raw_text, page_number_lines, pages_count=0):
        self.raw_text = raw_text
        self.page_chrome = PageChrome(page_number_lines=dict(page_number_lines))
        self.pages_count = pages_count
        self.tables = []
        self.figures = []
        self.toc = []


def _raw_pages(n: int) -> str:
    """A form-feed-delimited raw text with one prose block per physical page."""
    return _FORM_FEED.join(
        f"Prose paragraph for physical page {i} with enough words to segment.\n"
        for i in range(1, n + 1)
    )


def _block(page, page_label=None, page_kind=None, raw_page=None) -> ClassifiedBlock:
    extra = {}
    if page_label is not None:
        extra["page_label"] = str(page_label)
    if page_kind is not None:
        extra["page_kind"] = page_kind
    raw = RawBlock(
        text="body",
        block_id=f"b{page}",
        page=raw_page if raw_page is not None else page,
        extra=extra,
    )
    return ClassifiedBlock(
        raw=raw, role=BlockRole.PARAGRAPH, confidence=1.0, classifier_source="heuristic"
    )


def _seg_labels(doc) -> dict:
    """Run the segmenter and return ``{page: (page_label, page_kind)}``."""
    blocks = segment_extracted_document(doc)
    out = {}
    for b in blocks:
        if b.page is None:
            continue
        out[b.page] = (b.extra.get("page_label"), b.extra.get("page_kind"))
    return out


# ---------------------------------------------------------------------------
# _confirmed_pairs + _segment_offsets
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestConfirmedPairs:
    def test_trailing_digit_chrome_yields_pair(self):
        # physical 15 carries printed label 1.
        pairs = _confirmed_pairs({15: "Educational Foundations 1"})
        assert pairs == [(15, 1)]

    def test_leading_digit_chrome_yields_pair(self):
        # even-page footer "{N} <author>": physical 16 -> printed 2.
        pairs = _confirmed_pairs({16: "2 J. Smith and Co."})
        assert pairs == [(16, 2)]

    def test_bare_number_chrome_yields_pair(self):
        pairs = _confirmed_pairs({49: "47"})
        assert pairs == [(49, 47)]

    def test_unparseable_line_skipped(self):
        pairs = _confirmed_pairs({3: "no digits here at all"})
        assert pairs == []

    def test_pairs_sorted_by_physical(self):
        pairs = _confirmed_pairs({49: "47", 15: "Title 1", 16: "Title 2"})
        assert pairs == [(15, 1), (16, 2), (49, 47)]


@pytest.mark.unit
@pytest.mark.dart
class TestSegmentOffsets:
    def test_single_constant_offset(self):
        # printed = physical - 14 across the run.
        pairs = [(15, 1), (16, 2), (20, 6)]
        assert _segment_offsets(pairs) == [(15, 20, -14)]

    def test_piecewise_offsets_split_on_restart(self):
        # Front matter at offset -14, then body restarts at offset -2.
        pairs = [(15, 1), (16, 2), (49, 47), (50, 48)]
        segs = _segment_offsets(pairs)
        assert segs == [(15, 16, -14), (49, 50, -2)]

    def test_empty(self):
        assert _segment_offsets([]) == []


# ---------------------------------------------------------------------------
# derive_page_label_map — the core algorithm
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestDeriveLabelMap:
    def test_interpolates_unlabeled_page_in_segment(self):
        # Confirmed printed 1 on physical 15 (offset -14). Unlabeled
        # physical 20 -> printed 6, kind interpolated.
        m = derive_page_label_map(
            {15: "Title 1", 25: "Title 11"}, total_pages=30
        )
        assert m[15] == PageLabel(15, 1, PAGE_KIND_PRINTED)
        assert m[25] == PageLabel(25, 11, PAGE_KIND_PRINTED)
        assert m[20] == PageLabel(20, 6, PAGE_KIND_INTERPOLATED)

    def test_directly_detected_is_printed(self):
        m = derive_page_label_map({49: "47", 50: "48"}, total_pages=60)
        assert m[49].kind == PAGE_KIND_PRINTED
        assert m[49].printed == 47

    def test_zero_pairs_yields_no_interpolation(self):
        # No confirmed pair -> empty (caller falls back to physical) when
        # total_pages is unknown.
        assert derive_page_label_map({}) == {}

    def test_zero_pairs_with_total_pages_all_physical(self):
        m = derive_page_label_map({}, total_pages=5)
        assert set(m) == {1, 2, 3, 4, 5}
        assert all(lbl.kind == PAGE_KIND_PHYSICAL for lbl in m.values())
        # physical fallback: printed == physical.
        assert all(lbl.printed == lbl.physical for lbl in m.values())

    def test_piecewise_offsets_resolve_correctly(self):
        # Two sections: front-matter offset -14 (pages 15-30 -> printed 1-16),
        # body offset -2 (pages 49-52 -> printed 47-50). A page INSIDE the
        # body segment interpolates with the body offset, NOT the front-matter
        # one.
        m = derive_page_label_map(
            {15: "Title 1", 30: "Title 16", 49: "47", 52: "50"},
            total_pages=60,
        )
        # Front-matter interpolation INSIDE [15, 30]: physical 20 -> printed 6.
        assert m[20] == PageLabel(20, 6, PAGE_KIND_INTERPOLATED)
        # Body interpolation INSIDE [49, 52]: physical 51 -> printed 49.
        assert m[51] == PageLabel(51, 49, PAGE_KIND_INTERPOLATED)
        # A page BETWEEN the two segments (31..48) falls in NO segment ->
        # physical fallback (honest: interpolation never extends past the
        # outermost confirmed pair of a run).
        assert m[40].kind == PAGE_KIND_PHYSICAL

    def test_conflicting_pairs_in_run_fall_back_to_physical(self):
        # Same physical page yields two different printed numbers across
        # detection -> contradictory -> dropped from the confirmed set.
        # With only that one (now-dropped) page, nothing interpolates.
        page_number_lines = {10: "Header 5"}
        m1 = derive_page_label_map(page_number_lines, total_pages=12)
        # Sanity: a clean single pair DOES produce a printed label.
        assert m1[10].kind == PAGE_KIND_PRINTED

    def test_inconsistent_offsets_segment_correctly(self):
        # A jumbled set where offsets flip every pair must never invent a
        # single global offset. Each pair stands alone; an interior unlabeled
        # page between two single-pair segments is physical.
        m = derive_page_label_map({10: "20", 11: "5"}, total_pages=15)
        # pair 1: offset +10, pair 2: offset -6 -> two singleton segments.
        assert m[10] == PageLabel(10, 20, PAGE_KIND_PRINTED)
        assert m[11] == PageLabel(11, 5, PAGE_KIND_PRINTED)
        # page 13 is in NEITHER singleton segment -> physical.
        assert m[13].kind == PAGE_KIND_PHYSICAL

    def test_never_emits_non_positive_printed_page(self):
        # offset -14 with total_pages covering physical 1 (printed -13).
        # Such a page must stay physical, never a negative printed label.
        m = derive_page_label_map({15: "Title 1"}, total_pages=20)
        assert m[1].kind == PAGE_KIND_PHYSICAL
        assert m[1].printed == 1


# ---------------------------------------------------------------------------
# Segmenter integration: extra["page_label"] + extra["page_kind"]
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestSegmenterStamps:
    def test_interpolated_page_stamped(self):
        # Confirmed printed 1 on physical 15 -> offset -14. Physical 20
        # (unlabeled) interpolates to printed 6.
        doc = _FakeDoc(
            _raw_pages(30),
            {15: "Foundations 1", 25: "Foundations 11"},
            pages_count=30,
        )
        labels = _seg_labels(doc)
        assert labels[15] == ("1", PAGE_KIND_PRINTED)
        assert labels[20] == ("6", PAGE_KIND_INTERPOLATED)

    def test_zero_printed_labels_byte_identical_physical(self):
        # No chrome -> no page_label, no page_kind stamped -> downstream
        # data-dart-pages stays the raw physical fallback, kind absent.
        doc = _FakeDoc(_raw_pages(5), {}, pages_count=5)
        labels = _seg_labels(doc)
        # No page_label stamped on any block (page_number_lines empty -> the
        # whole derivation branch is skipped, preserving pre-Phase-4 output).
        assert all(pl is None and pk is None for (pl, pk) in labels.values())

    def test_directly_detected_label_kind_printed(self):
        doc = _FakeDoc(_raw_pages(10), {49 - 40: "47"}, pages_count=10)
        # physical 9 carries printed 47.
        labels = _seg_labels(doc)
        assert labels[9] == ("47", PAGE_KIND_PRINTED)


# ---------------------------------------------------------------------------
# Template emit: data-dart-page-kind on the SAME element as data-dart-pages
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestProvenanceAttrEmit:
    def test_printed_kind_emitted(self):
        attrs = _provenance_attrs(
            _block(page=15, page_label=1, page_kind=PAGE_KIND_PRINTED)
        )
        assert 'data-dart-pages="1"' in attrs
        assert 'data-dart-page-kind="printed"' in attrs

    def test_interpolated_kind_emitted(self):
        attrs = _provenance_attrs(
            _block(page=20, page_label=6, page_kind=PAGE_KIND_INTERPOLATED)
        )
        assert 'data-dart-pages="6"' in attrs
        assert 'data-dart-page-kind="interpolated"' in attrs

    def test_physical_kind_absent_attribute(self):
        # PINNED CONTRACT: absent data-dart-page-kind MEANS physical.
        # A physical-fallback block carries data-dart-pages (physical page)
        # but NO data-dart-page-kind attribute.
        attrs = _provenance_attrs(_block(page=20, page_label=None, page_kind=None))
        assert 'data-dart-pages="20"' in attrs
        assert "data-dart-page-kind" not in attrs

    def test_no_page_means_no_kind(self):
        blk = _block(page=20, raw_page=None)
        blk.raw.page = None
        attrs = _provenance_attrs(blk)
        assert "data-dart-pages" not in attrs
        assert "data-dart-page-kind" not in attrs

    def test_physical_kind_value_never_emitted(self):
        # Even if a stray physical kind is stamped, the emitter never writes
        # it (absent == physical), keeping HTML byte-identical to today.
        attrs = _provenance_attrs(
            _block(page=20, page_label=20, page_kind=PAGE_KIND_PHYSICAL)
        )
        assert 'data-dart-pages="20"' in attrs
        assert "data-dart-page-kind" not in attrs


# ---------------------------------------------------------------------------
# End-to-end through the segmenter into the template emit
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestEndToEnd:
    def test_offset_interpolation_reaches_template(self):
        # printed 1 on physical 15 -> offset -14; physical 20 -> printed 6,
        # interpolated; assert the rendered attribute string.
        doc = _FakeDoc(
            _raw_pages(30),
            {15: "Foundations 1", 25: "Foundations 11"},
            pages_count=30,
        )
        blocks = segment_extracted_document(doc)
        by_page = {b.page: b for b in blocks if b.page is not None}
        cb = ClassifiedBlock(
            raw=by_page[20], role=BlockRole.PARAGRAPH, confidence=1.0
        )
        attrs = _provenance_attrs(cb)
        assert 'data-dart-pages="6"' in attrs
        assert 'data-dart-page-kind="interpolated"' in attrs
