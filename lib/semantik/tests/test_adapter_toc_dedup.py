"""TOC-quality dedup at the adapter seam (end-user-HTML audit, ch09 shots).

Rules exercised in ``adapter._build_toc_html``:
  (a)/(c) a child section whose text duplicates its parent (ancestor) chapter
          is never emitted;
  (b) repeated identical titles at the SAME level (sibling sections, or
      top-level chapters) collapse to the first occurrence;
  and the surviving anchor points at the FIRST occurrence.

Synthetic block IR only — no corpus files.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _build_toc_html,
    _mint_unique_sids,
)


def _section(text: str, idx: int) -> _AdapterBlock:
    """A genuine <h3> section-heading block (enters the TOC)."""
    return _AdapterBlock(
        html="",
        region_kind="heading",
        raw_block_index=idx,
        heading_level=3,
        raw_text=text,
        heading_text=text,
    )


def _toc(chapters):
    return _build_toc_html(chapters, _mint_unique_sids(chapters))


def test_child_identical_to_parent_is_dropped():
    # Rule (a)/(c): a "Chapter Outline" section under a "Chapter Outline"
    # chapter must not be emitted.
    ch = _AdapterChapter(
        title="Chapter Outline",
        blocks=[_section("Chapter Outline", 0), _section("9.1 Foo", 1)],
    )
    html = _toc([ch])
    # The chapter crumb appears once; the identical child does not.
    assert html.count(">Chapter Outline<") == 1
    assert ">9.1 Foo<" in html


def test_child_identical_to_parent_dropped_even_when_not_first():
    # Rule (c) is ancestor-scoped, not position-scoped.
    ch = _AdapterChapter(
        title="Chapter Outline",
        blocks=[_section("Intro", 0), _section("chapter  outline", 1)],
    )
    html = _toc([ch])
    assert ">Intro<" in html
    # Case/whitespace-insensitive: "chapter  outline" == parent → dropped.
    assert html.count("chapter  outline") == 0
    assert html.lower().count(">chapter outline<") == 1  # the parent only


def test_repeated_sibling_titles_dedupe_keep_first():
    ch = _AdapterChapter(
        title="Roots",
        blocks=[
            _section("9.1 Square Roots", 0),  # kept
            _section("9.2 Higher Roots", 1),
            _section("9.1 Square Roots", 2),  # dropped (rule b)
        ],
    )
    html = _toc([ch])
    assert html.count(">9.1 Square Roots<") == 1
    assert html.count(">9.2 Higher Roots<") == 1


def test_surviving_anchor_points_at_first_occurrence():
    first = _section("9.1 Square Roots", 0)
    dup = _section("9.1 Square Roots", 2)
    ch = _AdapterChapter(title="Roots", blocks=[first, dup])
    sid_map = _mint_unique_sids([ch])
    html = _build_toc_html([ch], sid_map)
    first_sid = sid_map[id(first)]
    dup_sid = sid_map[id(dup)]
    assert first_sid != dup_sid  # they mint distinct ids
    assert f'href="#{first_sid}"' in html
    assert f'href="#{dup_sid}"' not in html  # the dup li is gone entirely


def test_top_level_duplicate_chapters_dedupe():
    # Rule (b) also applies to same-level chapter crumbs.
    chapters = [
        _AdapterChapter(title="Review", blocks=[_section("A", 0)]),
        _AdapterChapter(title="Review", blocks=[_section("B", 1)]),
    ]
    html = _toc(chapters)
    assert html.count(">Review<") == 1


def test_distinct_titles_all_survive():
    # No false positives: distinct sections are all kept.
    ch = _AdapterChapter(
        title="Roots",
        blocks=[_section("Alpha", 0), _section("Beta", 1), _section("Gamma", 2)],
    )
    html = _toc([ch])
    for t in ("Alpha", "Beta", "Gamma"):
        assert html.count(f">{t}<") == 1


def test_cross_parent_section_dedupe_keeps_first(  # ITEM 3 (round-2 audit)
):
    # A section listed under BOTH its own chapter AND a later "Chapter 9 Review"
    # chapter (DIFFERENT parents) is deduped globally — first occurrence wins.
    chapters = [
        _AdapterChapter(
            title="Roots and Radicals",
            blocks=[
                _section("9.1 Simplify and Use Square Roots", 0),
                _section("9.4 Multiply Square Roots", 1),
            ],
        ),
        _AdapterChapter(
            title="Chapter 9 Review",
            blocks=[
                # Re-lists 9.1 under a different parent → dropped (cross-parent).
                _section("9.1 Simplify and Use Square Roots", 2),
                _section("Key Terms", 3),  # distinct → kept
            ],
        ),
    ]
    html = _toc(chapters)
    assert html.count(">9.1 Simplify and Use Square Roots<") == 1
    assert html.count(">Chapter 9 Review<") == 1
    assert ">Key Terms<" in html


def test_cross_parent_anchor_points_at_first_parent():
    # The surviving anchor for a cross-parent duplicate resolves to its FIRST
    # (real-chapter) occurrence, not the later review-chapter one.
    first = _section("9.1 Square Roots", 0)
    dup = _section("9.1 Square Roots", 5)
    chapters = [
        _AdapterChapter(title="Roots", blocks=[first]),
        _AdapterChapter(title="Chapter 9 Review", blocks=[dup]),
    ]
    sid_map = _mint_unique_sids(chapters)
    html = _build_toc_html(chapters, sid_map)
    assert f'href="#{sid_map[id(first)]}"' in html
    assert f'href="#{sid_map[id(dup)]}"' not in html
