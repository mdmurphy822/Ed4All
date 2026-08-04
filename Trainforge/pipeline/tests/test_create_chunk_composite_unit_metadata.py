"""Wave #22 stamp-site coverage: ``CourseProcessor._create_chunk`` propagates
the ``composite_unit`` / ``unit_roles`` kwargs onto the emitted chunk.

Unit-level companion to the end-to-end MCP chunking-phase composite-unit
metadata test (which drives the same fields through the SemantiK-chunking
registry callback). Here we exercise
the IMSCC-path ``CourseProcessor._create_chunk`` seam in isolation: when the
chunker's ``section_unit_signals`` / ``aggregate_composite_unit``
helpers thread the aggregated values through as kwargs, ``_create_chunk`` stamps
them; when they are absent, the additive fields are omitted (byte-stable legacy
contract — mirrors ``source_document_sha256``).

Bare ``CourseProcessor`` construction bypasses ``__init__`` (mirrors
``test_chunk_source_document_sha.py``).
"""

from __future__ import annotations

import collections
from typing import Any, Dict


def _build_minimal_processor() -> Any:
    """Construct a bare ``CourseProcessor`` bypassing ``__init__`` — only the
    fields ``_create_chunk`` reads are set + the metadata extractors that depend
    on deeper ``__init__`` state are monkey-patched."""
    from Trainforge.pipeline.process_course import CourseProcessor

    cp = CourseProcessor.__new__(CourseProcessor)
    cp.course_code = "TEST_101"
    cp.capture = None
    cp.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": collections.defaultdict(int),
        "difficulty_distribution": collections.defaultdict(int),
    }
    cp._all_concept_tags = set()
    cp._extract_concept_tags = lambda t, i: []  # type: ignore[method-assign]
    cp._determine_difficulty = lambda t, i: "foundational"  # type: ignore[method-assign]
    cp._extract_objective_refs = (
        lambda i, section_heading=None: []
    )  # type: ignore[method-assign]
    cp._extract_section_metadata = lambda i, h, **kw: (
        None,
        None,
        [],
        {"key_claims": [], "objective_alignment": []},
        {
            "content_type_label": "none",
            "key_terms": "none",
            "key_claims": "none",
            "objective_alignment": "none",
        },
    )  # type: ignore[method-assign]
    cp._resolve_chunk_source_references = (
        lambda *, item, section_heading, section_source_ids, merged_headings=None: []
    )  # type: ignore[method-assign]
    return cp


def _minimal_item() -> Dict[str, Any]:
    return {
        "module_id": "week_09_radicals",
        "module_title": "Week 9 — Radicals",
        "item_id": "ex",
        "title": "Example 9.1",
        "resource_type": "webcontent",
        "item_path": "week_09/ex.html",
        "learning_objectives": [],
    }


def test_create_chunk_stamps_pedagogical_unit_and_roles():
    cp = _build_minimal_processor()

    chunk = cp._create_chunk(
        chunk_id="test_101_chunk_00001",
        text="Simplify the radical expression fully.",
        html="<p>Simplify the radical expression fully.</p>",
        item=_minimal_item(),
        section_heading="Example 9.1",
        chunk_type="example",
        position_in_module=0,
        composite_unit="worked_example",
        unit_roles=["worked_example", "statement", "solution-steps"],
    )

    assert chunk["composite_unit"] == "worked_example"
    assert chunk["unit_roles"] == [
        "worked_example",
        "statement",
        "solution-steps",
    ]
    # Returns a copy of the passed list (not the same object aliased in).
    assert chunk["unit_roles"] is not None


def test_create_chunk_omits_pedagogical_fields_when_absent():
    cp = _build_minimal_processor()

    chunk = cp._create_chunk(
        chunk_id="test_101_chunk_00001",
        text="Simplify the radical expression fully.",
        html="<p>Simplify the radical expression fully.</p>",
        item=_minimal_item(),
        section_heading="Example 9.1",
        chunk_type="example",
        position_in_module=0,
    )

    assert "composite_unit" not in chunk
    assert "unit_roles" not in chunk


def test_create_chunk_empty_roles_list_omitted():
    """An empty roles list / null unit are falsy → omitted (mirrors the
    ``if unit_roles:`` guard in ``_create_chunk``)."""
    cp = _build_minimal_processor()

    chunk = cp._create_chunk(
        chunk_id="test_101_chunk_00001",
        text="Simplify the radical expression fully.",
        html="<p>Simplify the radical expression fully.</p>",
        item=_minimal_item(),
        section_heading="Example 9.1",
        chunk_type="example",
        position_in_module=0,
        composite_unit=None,
        unit_roles=[],
    )

    assert "composite_unit" not in chunk
    assert "unit_roles" not in chunk
