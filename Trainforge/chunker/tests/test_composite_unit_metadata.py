"""Wave #22 quick-wins — chunk pedagogical-role metadata.

Covers the additive ``composite_unit`` (type string or null) +
``unit_roles`` (flow/opener roles) that the chunker aggregates from a
chunk's constituent sections and forwards to the ``create_chunk`` callback.

Synthetic ``_Section`` objects + a stub callback (no corpus files).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Trainforge.chunker.chunker import (
    ChunkerContext,
    aggregate_composite_unit,
    chunk_content,
    merge_small_sections,
    section_unit_signals,
)


@dataclass
class _Section:
    heading: str
    content: str
    word_count: int
    level: int = 3
    data_dart_unit: Optional[str] = None
    data_dart_opener: Optional[str] = None
    data_dart_flows: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_aggregate_unit_single_value():
    assert aggregate_composite_unit(["worked_example"]) == "worked_example"


def test_aggregate_unit_all_null():
    assert aggregate_composite_unit([None, None]) is None
    assert aggregate_composite_unit([]) is None


def test_aggregate_unit_plurality_winner():
    units = ["worked_example", "worked_example", "exercise_set", None]
    assert aggregate_composite_unit(units) == "worked_example"


def test_aggregate_unit_tie_is_null():
    # No clear majority → null (honest, no fabrication).
    assert aggregate_composite_unit(["worked_example", "exercise_set"]) is None


def test_section_signals_union_opener_and_flows():
    sec = _Section(
        "Example 9.1", "x", 1,
        data_dart_unit="worked_example",
        data_dart_opener="worked_example",
        data_dart_flows=["statement"],
    )
    unit, roles = section_unit_signals(sec)
    assert unit == "worked_example"
    assert set(roles) == {"worked_example", "statement"}


def test_section_signals_legacy_section_is_empty():
    # A section-like without the unit/opener attributes (legacy) → (None, []).
    @dataclass
    class _Legacy:
        heading: str = "h"
        content: str = "c"
        word_count: int = 1

    unit, roles = section_unit_signals(_Legacy())
    assert unit is None
    assert roles == []


# --------------------------------------------------------------------------
# merge_small_sections carries the metadata
# --------------------------------------------------------------------------


def test_merge_carries_unit_and_roles():
    sections = [
        _Section(
            "Example 9.1", "problem prose here now", 4,
            data_dart_unit="worked_example",
            data_dart_opener="worked_example",
            data_dart_flows=["statement"],
        ),
    ]
    merged = merge_small_sections(sections, max_chunk_size=800)
    assert len(merged) == 1
    assert merged[0].merged_composite_unit == "worked_example"
    assert merged[0].merged_unit_roles == ["statement", "worked_example"]


def test_merge_of_two_units_uses_plurality_and_unions_roles():
    sections = [
        _Section("A", "a b c", 3, data_dart_unit="worked_example",
                 data_dart_flows=["statement"]),
        _Section("B", "d e f", 3, data_dart_unit="worked_example",
                 data_dart_opener="solution", data_dart_flows=["solution-steps"]),
        _Section("C", "g h i", 3, data_dart_unit="exercise_set",
                 data_dart_opener="try_it"),
    ]
    merged = merge_small_sections(sections, max_chunk_size=800)
    assert len(merged) == 1  # all merge into one buffer
    assert merged[0].merged_composite_unit == "worked_example"  # 2 vs 1
    assert merged[0].merged_unit_roles == [
        "solution", "solution-steps", "statement", "try_it"
    ]


def test_merge_legacy_sections_have_no_metadata():
    sections = [
        _Section("A", "a b c", 3),
        _Section("B", "d e f", 3),
    ]
    merged = merge_small_sections(sections, max_chunk_size=800)
    assert merged[0].merged_composite_unit is None
    assert merged[0].merged_unit_roles == []


# --------------------------------------------------------------------------
# chunk_content threads the metadata to create_chunk
# --------------------------------------------------------------------------


def _item(sections):
    return {
        "module_id": "m1",
        "item_id": "i1",
        "item_path": "week_01/i1.html",
        "title": "Page",
        "resource_type": "page",
        "raw_html": "<html></html>",
        "sections": sections,
        "misconceptions": [],
    }


def _capturing_ctx(captured):
    def _create_chunk(*, chunk_id, text, html, item, section_heading, chunk_type,
                      composite_unit=None, unit_roles=None, **_ignored):
        captured.append(
            {"id": chunk_id, "unit": composite_unit, "roles": unit_roles}
        )
        return {"id": chunk_id, "text": text}

    return ChunkerContext(create_chunk=_create_chunk)


def test_chunk_content_stamps_pedagogical_metadata():
    captured: List[Dict[str, Any]] = []
    sections = [
        _Section(
            "Example 9.1", "simplify the radical expression fully now", 6,
            data_dart_unit="worked_example",
            data_dart_opener="worked_example",
            data_dart_flows=["statement"],
        ),
    ]
    result = chunk_content([_item(sections)], "TEST_101", ctx=_capturing_ctx(captured))
    assert list(result.chunks)
    assert captured
    assert captured[0]["unit"] == "worked_example"
    assert set(captured[0]["roles"]) == {"worked_example", "statement"}


def test_chunk_content_legacy_sections_pass_null_metadata():
    captured: List[Dict[str, Any]] = []
    sections = [_Section("Intro", "plain prose about the topic here now", 6)]
    chunk_content([_item(sections)], "TEST_101", ctx=_capturing_ctx(captured))
    assert captured
    # No unit/opener metadata → chunk_text_block only forwards the kwargs when
    # truthy, so the callback receives its defaults (None / None).
    assert captured[0]["unit"] is None
    assert captured[0]["roles"] is None
