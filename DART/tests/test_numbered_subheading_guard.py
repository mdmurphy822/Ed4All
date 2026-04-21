"""Wave 25 Fix 7: numbered subheading followed by prose → SUBSECTION_HEADING.

Audit: "1. Why this book?", "2. The audience for the book", "3. Why an
'open' textbook?" on Bates — each on its own pdftotext block. The
Wave-21 LIST_ITEM classifier grabs them; the grouper can't merge them
across intervening prose; the stray-LIST_ITEM fallback emits a
one-item ``<ol>`` per heading. 14 single-``<ol>`` + 28 single-``<ul>``
wrappers total — a screen-reader reading-order catastrophe.

Fix: peek at ``block.neighbors["next"]``. When it carries ≥ 60 words
AND isn't itself a list item, the block is a numbered subsection
heading, NOT a list item. Promote to SUBSECTION_HEADING instead.
"""

from __future__ import annotations

import pytest

from DART.converter.block_roles import BlockRole, RawBlock
from DART.converter.heuristic_classifier import HeuristicClassifier


def _block_with_next(text: str, next_text: str, *, block_id: str = "b1") -> RawBlock:
    return RawBlock(
        text=text,
        block_id=block_id,
        page=1,
        extractor="pdftotext",
        neighbors={"prev": "", "next": next_text},
    )


@pytest.mark.unit
@pytest.mark.dart
class TestNumberedSubheadingGuard:
    def test_numbered_heading_with_long_prose_neighbor_is_subsection(self):
        long_prose = " ".join(["word"] * 80)
        block = _block_with_next("1. Why this book?", long_prose)
        clf = HeuristicClassifier()
        result = clf.classify_sync([block])[0]
        assert result.role == BlockRole.SUBSECTION_HEADING
        assert result.attributes.get("level") == 3

    def test_real_numbered_list_items_preserved(self):
        # Three consecutive short list items — each's neighbor is
        # another short item, so none gets demoted to subheading.
        # (Note: "N. Foo" text also matches the chapter regex, so
        # pre-Wave-25 these classified as CHAPTER_OPENER. We
        # explicitly verify the Wave-25 Fix-7 guard does NOT
        # demote them to SUBSECTION when the next block is short.
        # The invariant is simply: no SUBSECTION_HEADING emission.)
        b1 = _block_with_next("1. First step", "2. Second step", block_id="b1")
        b2 = _block_with_next("2. Second step", "3. Third step", block_id="b2")
        b3 = _block_with_next("3. Third step", "", block_id="b3")
        clf = HeuristicClassifier()
        out = clf.classify_sync([b1, b2, b3])
        # None should be SUBSECTION_HEADING — guard fires only when
        # the next block carries >= 60 words of prose.
        roles = [c.role for c in out]
        assert BlockRole.SUBSECTION_HEADING not in roles, (
            f"Wave-25 Fix-7 incorrectly demoted short list items: {roles}"
        )

    def test_numbered_with_no_next_neighbor_legacy_behavior(self):
        # "1." alone with no prose neighbor — guard is inapplicable
        # (the "long prose next" signal is absent), so the Wave 25
        # Fix 7 demotion does NOT fire. Pre-Wave-25 behavior is
        # preserved: the chapter regex captures this first and
        # classifies it as CHAPTER_OPENER.
        block = _block_with_next("1. Solitary item text here", "")
        clf = HeuristicClassifier()
        result = clf.classify_sync([block])[0]
        # Wave 25 guard inapplicable → legacy outcome preserved.
        assert result.role != BlockRole.SUBSECTION_HEADING

    def test_unordered_marker_not_affected(self):
        # Bullet-led block with long prose next — unordered markers
        # are NOT numbered subheadings; the guard is specific to
        # ordered markers.
        long_prose = " ".join(["word"] * 80)
        block = _block_with_next("\u2022 A bullet point", long_prose)
        clf = HeuristicClassifier()
        result = clf.classify_sync([block])[0]
        assert result.role == BlockRole.LIST_ITEM

    def test_mixed_three_items_then_heading(self):
        # 3 short blocks followed by a numbered heading whose next
        # neighbor is long prose. The heading-like block must be
        # SUBSECTION, the short ones must not be.
        long_prose = " ".join(["word"] * 80)
        i1 = _block_with_next("1. Step one", "2. Step two", block_id="i1")
        i2 = _block_with_next("2. Step two", "3. Step three", block_id="i2")
        i3 = _block_with_next("3. Step three", "4. Heading text", block_id="i3")
        heading = _block_with_next(
            "4. Heading text", long_prose, block_id="h1"
        )
        clf = HeuristicClassifier()
        out = clf.classify_sync([i1, i2, i3, heading])
        # First three: short next neighbor → guard doesn't fire →
        # NOT demoted.
        assert out[0].role != BlockRole.SUBSECTION_HEADING
        assert out[1].role != BlockRole.SUBSECTION_HEADING
        assert out[2].role != BlockRole.SUBSECTION_HEADING
        # Last: long prose neighbour → Wave 25 Fix 7 guard fires.
        assert out[3].role == BlockRole.SUBSECTION_HEADING

    def test_guard_preserves_heading_text_attr(self):
        long_prose = " ".join(["word"] * 80)
        block = _block_with_next(
            "2. The audience for the book", long_prose
        )
        clf = HeuristicClassifier()
        result = clf.classify_sync([block])[0]
        assert result.role == BlockRole.SUBSECTION_HEADING
        assert result.attributes.get("heading_text") == "The audience for the book"
        assert result.attributes.get("numbered_marker") == "2."
