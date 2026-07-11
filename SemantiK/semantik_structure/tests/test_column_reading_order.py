"""Fix A — column-aware reading order (SEMANTIK_COLUMN_ORDER, default OFF).

The pipeline historically ordered text blocks by a raw raster ``(y0, x0)``
sort with zero column awareness, so a 2-column page got line-interleaved
across the gutter. Fix A lifts the gutter-clustering math into
``semantik_structure.reading_order`` and re-keys the FB-ordering sites to
``(page, column_index, y0, x0)`` when the flag is on.

Invariants pinned here (all CPU-only, no model, no PDF):
  (a) flag OFF -> byte-identical to the legacy raster sort,
  (b) a synthetic 2-column page reads left-column-then-right-column (not
      interleaved) when ON,
  (c) a single-column page is identical whether the flag is on or off,
  (d) the FeatureBlock multiset is conserved (pure reorder).
"""
from __future__ import annotations

import os

import pytest

from semantik_structure.features import featurize
from semantik_structure.reading_order import (
    column_ids_for_bboxes,
    column_ids_for_x0s,
    resolve_column_order_mode,
)
from semantik_structure.types import RawBlock

_FLAG = "SEMANTIK_COLUMN_ORDER"


@pytest.fixture(autouse=True)
def _clear_flag():
    prev = os.environ.get(_FLAG)
    os.environ.pop(_FLAG, None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_FLAG, None)
        else:
            os.environ[_FLAG] = prev


# ---------------------------------------------------------------------------
# Resolver semantics (parse-with-fallback, default OFF).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_resolver_truthy_on(val):
    os.environ[_FLAG] = val
    assert resolve_column_order_mode() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage", "2x"])
def test_resolver_falsey_or_garbage_off(val):
    os.environ[_FLAG] = val
    assert resolve_column_order_mode() is False


def test_resolver_unset_off():
    assert resolve_column_order_mode() is False


# ---------------------------------------------------------------------------
# Core gutter clustering.
# ---------------------------------------------------------------------------


def test_single_column_one_bin():
    # All blocks left-aligned around x=72 (LETTER page) -> one column.
    x0s = [72.0, 74.0, 71.5, 73.0, 72.0]
    assert column_ids_for_x0s(x0s, 612.0) == [0, 0, 0, 0, 0]


def test_two_column_split():
    # Left col ~72, right col ~330 (gap 258 > 0.06*612=36.7) -> two columns.
    x0s = [72.0, 330.0, 73.0, 332.0, 71.0]
    assert column_ids_for_x0s(x0s, 612.0) == [0, 1, 0, 1, 0]


def test_empty():
    assert column_ids_for_x0s([], 612.0) == []


def test_float_exclusion_does_not_spawn_spurious_column():
    # A figure sits in the gutter between the two text columns; without
    # exclusion its internal fragment at x=200 could seed a phantom middle
    # column. The float bbox (150,0,300,100) covers only the gutter — the
    # x=200 fragment centroid lands in it, the real columns do not.
    bboxes = [
        (72.0, 10.0, 120.0, 20.0),    # left col text
        (200.0, 10.0, 260.0, 20.0),   # figure-internal fragment (inside float)
        (330.0, 10.0, 400.0, 20.0),   # right col text
        (73.0, 40.0, 120.0, 50.0),    # left col text
        (331.0, 40.0, 400.0, 50.0),   # right col text
    ]
    floats = [(150.0, 0.0, 300.0, 100.0)]
    cols = column_ids_for_bboxes(bboxes, 612.0, float_bboxes=floats)
    # Left col -> 0, right col -> 1; the float-internal block is assigned but
    # did not create a third column.
    assert max(cols) == 1
    assert cols[0] == 0 and cols[3] == 0
    assert cols[2] == 1 and cols[4] == 1


# ---------------------------------------------------------------------------
# End-to-end featurize ordering.
# ---------------------------------------------------------------------------


def _raw(text, page, x0, y0):
    return RawBlock(
        text=text,
        page=page,
        bbox=(x0, y0, x0 + 50.0, y0 + 12.0),
        page_width=612.0,
        page_height=792.0,
        font_size=11.0,
    )


def _two_column_page():
    # Interleaved-by-y input: L1, R1, L2, R2, L3, R3. Raster (y,x) sort keeps
    # this interleaving; column-major must regroup L1,L2,L3 then R1,R2,R3.
    return [
        _raw("L1", 1, 72.0, 100.0),
        _raw("R1", 1, 330.0, 102.0),
        _raw("L2", 1, 72.0, 130.0),
        _raw("R2", 1, 330.0, 132.0),
        _raw("L3", 1, 72.0, 160.0),
        _raw("R3", 1, 330.0, 162.0),
    ]


def _texts(fbs):
    return [fb.raw.text for fb in fbs]


def test_flag_off_byte_identical_raster():
    raw = _two_column_page()
    os.environ[_FLAG] = "0"
    off = _texts(featurize(list(raw)))
    os.environ.pop(_FLAG, None)
    default = _texts(featurize(list(raw)))
    assert off == default
    # Raster (y,x) sort keeps the interleaving.
    assert off == ["L1", "R1", "L2", "R2", "L3", "R3"]


def test_two_column_grouped_when_on():
    raw = _two_column_page()
    os.environ[_FLAG] = "1"
    on = _texts(featurize(list(raw)))
    # Left column top-to-bottom, then right column top-to-bottom.
    assert on == ["L1", "L2", "L3", "R1", "R2", "R3"]


def test_multiset_conserved():
    raw = _two_column_page()
    os.environ[_FLAG] = "0"
    off = sorted(_texts(featurize(list(raw))))
    os.environ[_FLAG] = "1"
    on = sorted(_texts(featurize(list(raw))))
    assert off == on  # pure reorder, no add/drop


def test_single_column_identical_on_off():
    raw = [
        _raw("A", 1, 72.0, 100.0),
        _raw("B", 1, 73.0, 120.0),
        _raw("C", 1, 71.0, 140.0),
    ]
    os.environ[_FLAG] = "0"
    off = _texts(featurize(list(raw)))
    os.environ[_FLAG] = "1"
    on = _texts(featurize(list(raw)))
    assert off == on == ["A", "B", "C"]


def test_multipage_column_major_within_page():
    # Page 2 also 2-column; ensure ordering is per-page column-major and pages
    # stay in page order.
    raw = _two_column_page() + [
        _raw("P2L", 2, 72.0, 100.0),
        _raw("P2R", 2, 330.0, 100.0),
    ]
    os.environ[_FLAG] = "1"
    on = _texts(featurize(list(raw)))
    assert on == ["L1", "L2", "L3", "R1", "R2", "R3", "P2L", "P2R"]
