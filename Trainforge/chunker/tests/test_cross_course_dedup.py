"""W1b.4 — cross-course boilerplate dedup helper unit tests."""
from __future__ import annotations

from Trainforge.chunker.cross_course_dedup import (
    CrossCourseDedupIndex,
    chunk_content_hash,
    drop_boilerplate_chunks,
    normalize_for_dedup,
    resolve_cross_course_dedup_enabled,
)

_FOOTER = "Copyright 2026 Example Org. All rights reserved. Edit this page on GitHub."


def test_resolver_default_off():
    assert resolve_cross_course_dedup_enabled({}) is False
    assert resolve_cross_course_dedup_enabled({"ED4ALL_CROSS_COURSE_DEDUP": "1"}) is True
    assert resolve_cross_course_dedup_enabled({"ED4ALL_CROSS_COURSE_DEDUP": "x"}) is False


def test_normalize_collapses_case_punct_whitespace():
    assert normalize_for_dedup("Hello,   WORLD!") == "hello world"
    assert chunk_content_hash("Hello, WORLD!") == chunk_content_hash("hello world")
    assert chunk_content_hash("   ") == ""


def test_index_flags_hashes_across_courses():
    idx = CrossCourseDedupIndex()
    idx.add_course("course-a", [{"id": "a1", "text": _FOOTER}, {"id": "a2", "text": "Unique lesson content about photosynthesis and light reactions here."}])
    idx.add_course("course-b", [{"id": "b1", "text": _FOOTER}, {"id": "b2", "text": "Different lesson about the French revolution and its many causes today."}])
    boilerplate = idx.boilerplate_hashes()
    assert chunk_content_hash(_FOOTER) in boilerplate
    # Distinct real content never collides.
    assert len(boilerplate) == 1
    assert idx.is_boilerplate(_FOOTER) is True


def test_min_tokens_floor_protects_short_chunks():
    idx = CrossCourseDedupIndex(min_tokens=8)
    idx.add_course("a", [{"id": "1", "text": "short shared line"}])
    idx.add_course("b", [{"id": "2", "text": "short shared line"}])
    # Below the token floor → not tracked → not flagged.
    assert idx.boilerplate_hashes() == set()


def test_single_course_repeat_not_flagged():
    idx = CrossCourseDedupIndex(min_courses=2)
    idx.add_course("a", [{"id": "1", "text": _FOOTER}, {"id": "2", "text": _FOOTER}])
    # Appears twice but in ONE course → below min_courses.
    assert idx.boilerplate_hashes() == set()


def test_drop_boilerplate_partitions_and_preserves_objects():
    idx = CrossCourseDedupIndex()
    idx.add_course("a", [{"id": "a1", "text": _FOOTER}])
    idx.add_course("b", [{"id": "b1", "text": _FOOTER}])
    chunks = [
        {"id": "x", "text": _FOOTER},
        {"id": "y", "text": "Real content that only appears in this one course about tides."},
    ]
    kept, dropped = drop_boilerplate_chunks(chunks, idx.boilerplate_hashes())
    assert [c["id"] for c in kept] == ["y"]
    assert [c["id"] for c in dropped] == ["x"]
    # Anti-fabrication: kept objects are the same identities from input.
    assert kept[0] is chunks[1]


def test_drop_empty_boilerplate_is_noop():
    chunks = [{"id": "x", "text": _FOOTER}]
    kept, dropped = drop_boilerplate_chunks(chunks, set())
    assert kept == chunks and dropped == []
