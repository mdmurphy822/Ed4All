"""Build #23 Tier-3 — chunk-layer ``unit_subclass`` threading + ID-stability.

Covers: the pure ``aggregate_unit_subclass`` (subclass rides the resolved unit);
``merge_small_sections`` carrying ``merged_unit_subclass``; ``chunk_content``
threading it to the create_chunk callback; omit-when-absent on legacy sections;
and the CRITICAL invariant that adding subclass attributes changes NEITHER chunk
ids NOR chunk text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Trainforge.chunker.chunker import (
    ChunkerContext,
    aggregate_unit_subclass,
    chunk_content,
    merge_small_sections,
    section_subclass_signal,
)


@dataclass
class _Section:
    heading: str
    content: str
    word_count: int
    level: int = 3
    data_dart_unit: Optional[str] = None
    data_dart_subclass: Optional[str] = None
    data_dart_opener: Optional[str] = None
    data_dart_flows: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure aggregation — subclass rides the resolved unit
# ---------------------------------------------------------------------------
def test_aggregate_subclass_single_unit():
    assert aggregate_unit_subclass(["worked_example"], ["drill"]) == "drill"


def test_aggregate_subclass_straddle_tie_is_none():
    # Two distinct units → composite_unit is None → no subclass (honest).
    assert aggregate_unit_subclass(["worked_example", "exercise_set"], ["a", "b"]) is None


def test_aggregate_subclass_resolved_unit_no_subclass():
    assert aggregate_unit_subclass(["worked_example", "worked_example"], [None, None]) is None


def test_aggregate_subclass_plurality_picks_winning_units_subclass():
    units = ["we", "we", "es"]
    subs = ["symbolic-manipulation", "symbolic-manipulation", None]
    assert aggregate_unit_subclass(units, subs) == "symbolic-manipulation"


def test_section_subclass_signal_legacy_none():
    @dataclass
    class _Legacy:
        heading: str = "h"

    assert section_subclass_signal(_Legacy()) is None


# ---------------------------------------------------------------------------
# merge_small_sections carries merged_unit_subclass
# ---------------------------------------------------------------------------
def test_merge_carries_subclass():
    sections = [
        _Section("Example 9.1", "problem prose here now", 4,
                 data_dart_unit="worked_example",
                 data_dart_subclass="symbolic-manipulation"),
        _Section("Solution", "subtract then divide now", 4,
                 data_dart_unit="worked_example",
                 data_dart_subclass="symbolic-manipulation"),
    ]
    merged = merge_small_sections(sections, max_chunk_size=800)
    assert len(merged) == 1
    assert merged[0].merged_composite_unit == "worked_example"
    assert merged[0].merged_unit_subclass == "symbolic-manipulation"


def test_merge_legacy_sections_subclass_none():
    sections = [_Section("A", "a b c", 3), _Section("B", "d e f", 3)]
    merged = merge_small_sections(sections, max_chunk_size=800)
    assert merged[0].merged_unit_subclass is None


# ---------------------------------------------------------------------------
# chunk_content threads to create_chunk (mirrors the real stamping callback)
# ---------------------------------------------------------------------------
def _item(sections):
    return {
        "module_id": "m1", "item_id": "i1", "item_path": "week_01/i1.html",
        "title": "Page", "resource_type": "page", "raw_html": "<html></html>",
        "sections": sections, "misconceptions": [],
    }


def _stamping_ctx(captured):
    def _create_chunk(*, chunk_id, text, html, item, section_heading, chunk_type,
                      composite_unit=None, unit_roles=None, unit_subclass=None,
                      **_ignored):
        chunk = {"id": chunk_id, "text": text}
        if composite_unit:
            chunk["composite_unit"] = composite_unit
        if unit_subclass:
            chunk["unit_subclass"] = unit_subclass
        captured.append(chunk)
        return chunk

    return ChunkerContext(create_chunk=_create_chunk)


def test_chunk_content_stamps_unit_subclass():
    captured: List[Dict[str, Any]] = []
    sections = [
        _Section("Example 9.1", "simplify the radical expression fully now here", 7,
                 data_dart_unit="worked_example",
                 data_dart_subclass="symbolic-manipulation"),
    ]
    chunk_content([_item(sections)], "TEST_101", ctx=_stamping_ctx(captured))
    assert captured
    assert captured[0]["unit_subclass"] == "symbolic-manipulation"


def test_chunk_content_legacy_omits_subclass():
    captured: List[Dict[str, Any]] = []
    sections = [_Section("Intro", "plain prose about the topic here now again", 8)]
    chunk_content([_item(sections)], "TEST_101", ctx=_stamping_ctx(captured))
    assert captured
    assert "unit_subclass" not in captured[0]


# ---------------------------------------------------------------------------
# CRITICAL: subclass attributes do NOT change chunk ids or text
# ---------------------------------------------------------------------------
def test_id_and_text_stability_with_vs_without_subclass():
    def _sections(with_sub):
        return [
            _Section("Example 9.1", "simplify the radical expression fully now here", 7,
                     data_dart_unit="worked_example",
                     data_dart_subclass="symbolic-manipulation" if with_sub else None),
            _Section("Solution", "square both sides and isolate the variable now here", 9,
                     data_dart_unit="worked_example",
                     data_dart_subclass="symbolic-manipulation" if with_sub else None),
        ]

    cap_with: List[Dict[str, Any]] = []
    cap_without: List[Dict[str, Any]] = []
    chunk_content([_item(_sections(True))], "TEST_101", ctx=_stamping_ctx(cap_with))
    chunk_content([_item(_sections(False))], "TEST_101", ctx=_stamping_ctx(cap_without))

    assert [c["id"] for c in cap_with] == [c["id"] for c in cap_without]
    assert [c["text"] for c in cap_with] == [c["text"] for c in cap_without]
    # ONLY the subclass field differs.
    assert all("unit_subclass" in c for c in cap_with)
    assert all("unit_subclass" not in c for c in cap_without)
