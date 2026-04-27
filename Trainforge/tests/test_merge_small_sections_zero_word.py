"""Wave 83 Phase A regression test for zero-word section merge handling.

Pre-Wave-83, ``_merge_small_sections`` used ``buffer_wc == 0`` as the
"buffer is empty" signal. When the first section in the input had
``word_count == 0`` (the common Courseforge layout where a page-title
``<h1>`` precedes the first ``<h2>`` with no intervening text), the
algorithm would:

1. Enter the entry branch on iteration 0; set buffer_wc = 0.
2. Re-enter the entry branch on iteration 1 because buffer_wc was
   still 0 — silently REPLACING the buffer instead of merging.
3. Result: the h1 section was discarded; downstream chunk anchors
   shifted by one heading.

This was the load-bearing cause of the rdf-shacl-551 audit's 203/295
unbalanced-section chunks — every odd chunk lost its enclosing
``<section>`` open or close tag because the merge anchor was wrong.

Wave 83 replaces the numeric reset condition with an explicit
``buffer_started: bool`` so the state machine can't get stuck.
"""

from __future__ import annotations

from typing import List

from Trainforge.parsers.html_content_parser import ContentSection
from Trainforge.process_course import CourseProcessor


def _bare_processor() -> CourseProcessor:
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.MIN_CHUNK_SIZE = 100
    proc.MAX_CHUNK_SIZE = 800
    return proc


def _section(
    heading: str,
    word_count: int,
    *,
    level: int = 2,
    content: str = "",
    template_type: str = None,
) -> ContentSection:
    """Build a ContentSection for testing. ``content`` defaults to
    ``word_count`` placeholder words so the merger sees real text."""
    if not content:
        content = " ".join(f"w{i}" for i in range(word_count))
    return ContentSection(
        heading=heading,
        level=level,
        content=content,
        word_count=word_count,
        template_type=template_type,
    )


# ---------------------------------------------------------------------------
# rdf-shacl-551 layout reproducer
# ---------------------------------------------------------------------------


class TestZeroWordH1Merge:
    """Pin the exact rdf-shacl-551 layout: h1 page-title with 0 words,
    followed by content h2 sections."""

    def test_zero_word_h1_does_not_get_dropped(self):
        proc = _bare_processor()
        sections = [
            _section("RDF Triples and the Graph Model", 0, level=1),
            _section("This Page Supports", 22, content="objective list with 22 words " * 3),
            _section("The Core Idea", 269, template_type="explanation"),
            _section("Anatomy of a Triple", 238, template_type="explanation"),
        ]
        merged = proc._merge_small_sections(sections)

        # Pre-Wave-83 bug: 2 chunks emitted with h1 dropped, anchored to
        # "This Page Supports" / "The Core Idea".
        # Post-Wave-83: h1 starts the buffer, all sections merge until
        # MAX_CHUNK_SIZE is hit. With 0+22+269+238=529 ≤ 800, all four
        # sections merge into ONE chunk anchored to the h1 heading.
        assert len(merged) == 1
        heading, text, chunk_type, source_ids, _ = merged[0]
        assert heading == "RDF Triples and the Graph Model", (
            f"Expected merge buffer to anchor to h1 page title; got {heading!r}. "
            "Pre-Wave-83 the h1 was silently dropped by the buffer_wc==0 "
            "reset bug."
        )

    def test_h1_text_present_in_merged_buffer(self):
        # Even when h1's content is empty string, the merged text should
        # still contain content from all subsequent sections.
        proc = _bare_processor()
        sections = [
            _section("Page Title", 0, level=1, content=""),
            _section("First H2", 50),
            _section("Second H2", 50),
        ]
        merged = proc._merge_small_sections(sections)
        assert len(merged) == 1
        heading, text, _, _, _ = merged[0]
        assert heading == "Page Title"
        # Both h2 sections' text bodies should appear in the merged text.
        assert "w0" in text and "w49" in text  # words from the placeholder content


class TestNonZeroFirstSection:
    """Pin: when the first section has non-zero words, the legacy merge
    behavior is preserved (no regression)."""

    def test_first_section_non_zero_acts_as_anchor(self):
        proc = _bare_processor()
        sections = [
            _section("First Real Section", 100),
            _section("Second", 100),
            _section("Third", 100),
        ]
        merged = proc._merge_small_sections(sections)
        # All three fit within MAX_CHUNK_SIZE=800 → one merged chunk.
        assert len(merged) == 1
        heading, _, _, _, _ = merged[0]
        assert heading == "First Real Section"


class TestFlushBranch:
    """Pin: when the buffer flushes (next section would exceed
    MAX_CHUNK_SIZE), the new buffer initializes correctly without resetting
    buffer_started."""

    def test_flush_re_initializes_buffer_correctly(self):
        proc = _bare_processor()
        # Three sections of 400 words each: first two merge (800 = MAX),
        # third overflows and starts a new buffer.
        sections = [
            _section("A", 400),
            _section("B", 400),
            _section("C", 400),
        ]
        merged = proc._merge_small_sections(sections)
        assert len(merged) == 2
        assert merged[0][0] == "A"  # first chunk anchored to A
        assert merged[1][0] == "C"  # second chunk anchored to C (post-flush)

    def test_flush_after_zero_word_h1_still_works(self):
        # Combination case: zero-word h1 + sections that overflow.
        proc = _bare_processor()
        sections = [
            _section("Title", 0, level=1),
            _section("Big A", 500),
            _section("Big B", 500),  # 0+500+500=1000 > 800 → flush after Big A
        ]
        merged = proc._merge_small_sections(sections)
        # Title + Big A merge (0+500=500 ≤ 800), Big B overflows → flush.
        assert len(merged) == 2
        assert merged[0][0] == "Title"
        assert merged[1][0] == "Big B"


class TestMultipleZeroWordSections:
    """Pin: even multiple zero-word sections in a row don't trigger the
    state-machine confusion."""

    def test_consecutive_zero_word_sections(self):
        proc = _bare_processor()
        sections = [
            _section("Empty A", 0, content=""),
            _section("Empty B", 0, content=""),
            _section("Real Content", 100),
        ]
        merged = proc._merge_small_sections(sections)
        # All three merge (0+0+100=100 ≤ 800) → one chunk anchored to the
        # FIRST section's heading.
        assert len(merged) == 1
        heading, _, _, _, _ = merged[0]
        assert heading == "Empty A"


class TestEmptyInput:
    def test_empty_sections_list_returns_empty(self):
        proc = _bare_processor()
        assert proc._merge_small_sections([]) == []

    def test_single_zero_word_section_buffer_text_strip_excludes(self):
        # The final `if buffer_text.strip()` flush guard drops a buffer
        # whose text is empty/whitespace. A single zero-word section with
        # empty content therefore produces no output.
        proc = _bare_processor()
        sections = [_section("Empty", 0, content="")]
        merged = proc._merge_small_sections(sections)
        assert merged == []
