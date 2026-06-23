"""Unit tests — GAP B: table-caption neighbour resolution.

``_find_caption_neighbor`` resolves the "Table N" / "Figure N" label FB that
becomes the table's WCAG 1.3.1 ``<caption>`` accessible name. A label is often
separated from the table body by a blank / spacer / page-furniture FB, so the
old immediate-neighbour-only scan (``first-1`` / ``last+1``) missed it and the
table shipped with no accessible name. The window-scan now finds an explicit
"Table N" label within a small window — without fabricating caption text.

Pure (FeatureBlock + TableCandidate + CouncilState), no GPU, no PDF IO.
"""

from __future__ import annotations

from dart_semantic.council.types import CouncilState
from dart_semantic.region_detection import TableCandidate
from dart_semantic.structure_graph import _find_caption_neighbor
from dart_semantic.types import FeatureBlock, RawBlock


def _fb(text: str, *, page: int = 1, y0: float = 0.0) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, y0, 50.0, y0 + 12.0),
        page_width=612.0,
        page_height=792.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _tc(member_indices):
    return TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 100.0),
        pages=[1],
        member_block_indices=list(member_indices),
        cell_grid=[["a", "b"], ["1", "2"]],
    )


def test_immediate_label_above_resolves():
    fbs = [_fb("Table 1: Cost data", y0=100), _fb("a b"), _fb("1 2")]
    tc = _tc([1, 2])
    idx = _find_caption_neighbor(tc, fbs, CouncilState(outputs={}))
    assert idx == 0


def test_windowed_label_above_with_spacer_resolves():
    # GAP B core case: a blank/spacer FB sits between the label and the body,
    # so the OLD immediate-neighbour scan (first-1 only) missed the label.
    fbs = [
        _fb("Table 2: Conversions", y0=100),  # 0 — the label
        _fb("", y0=98),                        # 1 — spacer FB
        _fb("a b"),                            # 2 — table body start
        _fb("1 2"),                            # 3
    ]
    tc = _tc([2, 3])
    idx = _find_caption_neighbor(tc, fbs, CouncilState(outputs={}))
    assert idx == 0


def test_windowed_label_below_resolves():
    fbs = [
        _fb("a b"),                       # 0 — body
        _fb("1 2"),                       # 1
        _fb("", y0=98),                   # 2 — spacer
        _fb("Table 3: Results", y0=90),   # 3 — label below
    ]
    tc = _tc([0, 1])
    idx = _find_caption_neighbor(tc, fbs, CouncilState(outputs={}))
    assert idx == 3


def test_no_label_returns_none_no_fabrication():
    # No "Table N" label anywhere nearby → None (the table may still warn;
    # we never invent a descriptive caption).
    fbs = [_fb("Some prose"), _fb("a b"), _fb("1 2"), _fb("more prose")]
    tc = _tc([1, 2])
    assert _find_caption_neighbor(tc, fbs, CouncilState(outputs={})) is None


def test_label_outside_window_not_linked():
    # A label > window FBs away is not linked (avoids grabbing an unrelated
    # distant "Table N" header).
    fbs = (
        [_fb("Table 9: far away", y0=100)]
        + [_fb("filler %d" % i) for i in range(5)]
        + [_fb("a b"), _fb("1 2")]
    )
    tc = _tc([6, 7])
    assert _find_caption_neighbor(tc, fbs, CouncilState(outputs={})) is None
