"""Task #26: ``Trainforge.chunker.union_source_pages`` unit tests.

``union_source_pages`` folds the per-block ``pages`` lists a chunker source-ref
list carries (harvested off ``data-semantik-pages`` / ``data-dart-pages`` block
attrs) into ONE sorted, deduped page-number union — the value stamped on the
additive top-level ``source_pages`` chunk field for retrieval-citation display.

Synthetic-only: no real course text (fixture rule) — the HTML strings below are
throwaway synthetic prose carrying the SemantiK provenance attrs the harvest
reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.chunker import harvest_dart_source_refs, union_source_pages


def test_union_across_two_refs_dedups_and_sorts():
    """(a) Two refs whose pages overlap → the sorted deduped union."""
    refs = [
        {"block_id": "s1", "pages": [62, 63]},
        {"block_id": "s2", "pages": [63]},
    ]
    assert union_source_pages(refs) == [62, 63]


def test_union_across_disjoint_refs_out_of_order():
    """Union sorts across refs even when the input order is descending."""
    refs = [
        {"block_id": "s1", "pages": [9]},
        {"block_id": "s2", "pages": [3, 4]},
        {"block_id": "s3", "pages": [1]},
    ]
    assert union_source_pages(refs) == [1, 3, 4, 9]


def test_refs_with_empty_or_missing_pages_yield_empty():
    """(b) Refs whose pages are empty / missing / None → ``[]``."""
    assert union_source_pages([{"block_id": "s1", "pages": []}]) == []
    assert union_source_pages([{"block_id": "s1"}]) == []
    assert union_source_pages([{"block_id": "s1", "pages": None}]) == []


def test_none_and_empty_input_yield_empty():
    """(c) ``None`` / ``[]`` input → ``[]`` (caller omits the field)."""
    assert union_source_pages(None) == []
    assert union_source_pages([]) == []


def test_garbage_entries_tolerated():
    """(d) Non-dict entries + garbage page tokens are dropped defensively."""
    refs = [
        "not-a-dict",
        None,
        42,
        {"block_id": "s1", "pages": [62, "63", "x", -1, 0, None, True]},
        {"block_id": "s2", "pages": ["64"]},
    ]
    # "63" coerces to 63; "x"/-1/0/None/True dropped; "64" coerces to 64.
    assert union_source_pages(refs) == [62, 63, 64]


def test_non_iterable_pages_value_tolerated():
    """A scalar (non-iterable) ``pages`` value never raises."""
    assert union_source_pages([{"block_id": "s1", "pages": 62}]) == []


def test_integration_via_harvest_over_synthetic_html():
    """(e) End-to-end: harvest two SemantiK-attr blocks then union their pages.

    Exercises the ``data-semantik-*`` attribute spelling (the CURRENT emit) so a
    future purge Stage-4 that drops the ``data-dart-*`` spelling doesn't silently
    orphan this test.
    """
    html = (
        "<main>"
        '<section data-semantik-block-id="s1" data-semantik-pages="62-63">'
        "<p>Synthetic paragraph one.</p></section>"
        '<section data-semantik-block-id="s2" data-semantik-pages="63">'
        "<p>Synthetic paragraph two.</p></section>"
        "</main>"
    )
    refs = harvest_dart_source_refs(html)
    assert [r["block_id"] for r in refs] == ["s1", "s2"]
    assert union_source_pages(refs) == [62, 63]
