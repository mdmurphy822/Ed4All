"""Fix 1 (section-boundary hard break) + Fix 3 (text-level fragment fold).

Covers the scan-corpus forensic-audit fixes:

* Fix 1 — ``ED4ALL_CHUNK_SECTION_HARD_BREAK`` forces a chunk break when the
  merger crosses a textbook SECTION boundary, so a chunk never fuses one
  section's closing apparatus with the next section's opener.
* Fix 3 — ``fold_fragment_texts`` folds sub-``fragment_floor``-word pieces
  produced by the sentence/code split (which the section-level fold never
  saw) so no split-produced runt is emitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Trainforge.chunker.chunker import (
    ChunkerContext,
    chunk_content,
    fold_fragment_texts,
    is_section_boundary_heading,
    merge_small_sections,
    resolve_chunk_section_hard_break,
)


@dataclass
class _Section:
    heading: str
    content: str
    word_count: int
    level: int = 3
    source_references: Optional[list] = None
    template_type: Optional[str] = None
    data_dart_opener: Optional[str] = None
    data_dart_unit: Optional[str] = None


# --------------------------------------------------------------------------
# Fix 1 — section-boundary hard break
# --------------------------------------------------------------------------


def test_resolve_section_hard_break_default_off():
    assert resolve_chunk_section_hard_break({}) is False
    assert resolve_chunk_section_hard_break(
        {"ED4ALL_CHUNK_SECTION_HARD_BREAK": "true"}
    ) is True
    assert resolve_chunk_section_hard_break(
        {"ED4ALL_CHUNK_SECTION_HARD_BREAK": "junk"}
    ) is False


def test_is_section_boundary_heading_signals():
    assert is_section_boundary_heading("1.2 Use the Language of Algebra")
    assert is_section_boundary_heading("Learning Objectives")
    assert is_section_boundary_heading("BE PREPARED :: 4.5")
    assert is_section_boundary_heading("4.5")
    # Non-openers: a single-number exercise label, an apparatus heading, prose.
    assert not is_section_boundary_heading("27.")
    assert not is_section_boundary_heading("Self Check")
    assert not is_section_boundary_heading("Square Root Notation")
    assert not is_section_boundary_heading("")


def test_hard_break_off_is_legacy_merge():
    # Two small adjacent sections that fit under max → legacy merge fuses them.
    sections = [
        _Section("1.1 Intro", "a b c", 3, level=2),
        _Section("1.2 Next Section", "d e f", 3, level=2),
    ]
    default = merge_small_sections(sections, max_chunk_size=100)
    assert len(default) == 1


def test_hard_break_splits_at_level2_section():
    sections = [
        _Section("1.1 Intro", "a b c", 3, level=2),
        _Section("1.2 Next Section", "d e f", 3, level=2),
    ]
    hb = merge_small_sections(sections, max_chunk_size=100, section_hard_break=True)
    assert len(hb) == 2
    assert hb[1].section_boundary is True


def test_hard_break_splits_at_opener_phrase_even_at_h3():
    # A "Learning Objectives" opener rendered at h3 still forces a break via
    # the textual arm (level 3 alone is not a boundary).
    sections = [
        _Section("Body prose", "a b c", 3, level=3),
        _Section("Learning Objectives", "x y z", 3, level=3),
    ]
    default = merge_small_sections(sections, max_chunk_size=100)
    assert len(default) == 1
    hb = merge_small_sections(sections, max_chunk_size=100, section_hard_break=True)
    assert len(hb) == 2


def test_hard_break_splits_at_data_dart_opener_attr():
    # A7 — a section carrying a data-dart-opener role (a promoted pedagogical
    # opener whose visible heading text is NOT a legacy opener phrase, e.g. an
    # h4 "Try It 9.1") is a soft sub-boundary under the hard-break flag, so a
    # worked example never fuses with the next TRY-IT block.
    sections = [
        _Section("Example 9.1", "problem prose here", 3, level=4),
        _Section("Try It 9.1", "x y z", 3, level=4, data_dart_opener="try_it"),
    ]
    # Off → the two fuse (legacy merge; neither heading is a legacy opener text
    # and h4 level is not a boundary).
    default = merge_small_sections(sections, max_chunk_size=100)
    assert len(default) == 1
    # On → the data-dart-opener attribute forces the break.
    hb = merge_small_sections(sections, max_chunk_size=100, section_hard_break=True)
    assert len(hb) == 2
    assert hb[1].section_boundary is True


def test_hard_break_splits_at_data_dart_unit_edge():
    # Wave #22 Tier-2 — a section at a composite-unit EDGE (data-dart-unit
    # harvested from the <section class="dart-unit"> wrapper) is a preferred
    # boundary under the hard-break flag, so a chunk never straddles two units.
    # The lead heading here is not a legacy opener phrase and sits at h3, so only
    # the unit signal can force the break.
    sections = [
        _Section("Prior content", "a b c", 3, level=3),
        _Section(
            "Exercises 9.3", "d e f", 3, level=3, data_dart_unit="exercise_set"
        ),
    ]
    default = merge_small_sections(sections, max_chunk_size=100)
    assert len(default) == 1  # off → fuse
    hb = merge_small_sections(
        sections, max_chunk_size=100, section_hard_break=True
    )
    assert len(hb) == 2  # on → the unit edge forces the break
    assert hb[1].section_boundary is True


def test_data_dart_unit_absent_is_noop_under_hard_break():
    sections = [
        _Section("Sub A", "a b c", 3, level=3),
        _Section("Sub B", "d e f", 3, level=3),
    ]
    hb = merge_small_sections(
        sections, max_chunk_size=100, section_hard_break=True
    )
    assert len(hb) == 1


def test_data_dart_opener_absent_is_noop_under_hard_break():
    # A plain h4 subsection with no opener attr and no opener text still merges.
    sections = [
        _Section("Sub A", "a b c", 4, level=4),
        _Section("Sub B", "d e f", 4, level=4),
    ]
    hb = merge_small_sections(sections, max_chunk_size=100, section_hard_break=True)
    assert len(hb) == 1


def test_hard_break_still_merges_ordinary_subsections():
    # Non-opener h3 subsections are NOT boundaries → they still merge (the fix
    # targets cross-SECTION fusion, not every subsection).
    sections = [
        _Section("Sub A", "a b c", 3, level=3),
        _Section("Sub B", "d e f", 3, level=3),
    ]
    hb = merge_small_sections(sections, max_chunk_size=100, section_hard_break=True)
    assert len(hb) == 1


def test_boundary_result_not_folded_backward_by_floor():
    # A sub-floor section opener must NOT be folded back into the prior section
    # (that would re-fuse across the exact boundary the hard break enforced).
    sections = [
        _Section(
            "1.1 Body", "one two three four five six seven eight nine ten", 10,
            level=2,
        ),
        _Section("1.2 Opener", "tiny", 1, level=2),
    ]
    res = merge_small_sections(
        sections, max_chunk_size=100, fragment_floor=5, section_hard_break=True
    )
    assert len(res) == 2
    assert res[1].combined_text == "tiny"


# --------------------------------------------------------------------------
# Fix 3 — text-level fragment fold (sentence/code split runts)
# --------------------------------------------------------------------------


def test_fold_fragment_texts_folds_trailing_runt():
    pieces = ["real content here with many words indeed enough", "tiny tail"]
    folded = fold_fragment_texts(pieces, fragment_floor=5)
    assert len(folded) == 1
    assert "tiny tail" in folded[0]


def test_fold_fragment_texts_carries_leading_runt_forward():
    pieces = ["tiny", "the rest of the content is plenty long enough here now"]
    folded = fold_fragment_texts(pieces, fragment_floor=5)
    assert len(folded) == 1
    assert folded[0].startswith("tiny")


def test_fold_fragment_texts_floor_zero_is_identity():
    pieces = ["a b", "c d"]
    assert fold_fragment_texts(pieces, 0) is pieces


def test_fold_fragment_texts_all_below_floor_collapses_to_one():
    pieces = ["a b", "c d"]
    folded = fold_fragment_texts(pieces, fragment_floor=100)
    assert len(folded) == 1


def _make_item() -> Dict[str, Any]:
    return {
        "module_id": "m1",
        "item_id": "i1",
        "item_path": "i1",
        "title": "T",
        "resource_type": "page",
        "raw_html": (
            "<p>one two three four five six seven eight nine ten eleven "
            "twelve. Tiny frag.</p>"
        ),
        "sections": [],
        "misconceptions": None,
    }


def _make_ctx() -> ChunkerContext:
    def _create_chunk(**kw: Any) -> Dict[str, Any]:
        text = kw["text"]
        return {
            "id": kw["chunk_id"],
            "text": text,
            "word_count": len(text.split()),
            "follows_chunk": kw.get("follows_chunk_id"),
        }

    return ChunkerContext(create_chunk=_create_chunk)


def test_chunk_content_folds_sentence_split_runt(monkeypatch):
    # Force a sentence split (max/target below the sentence length) so the
    # tiny trailing "Tiny frag." becomes its own piece; the floor must fold it.
    monkeypatch.setenv("ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR", "5")
    res = chunk_content(
        [_make_item()], "TEST_101", ctx=_make_ctx(),
        max_chunk_size=10, target_chunk_size=8,
    )
    assert len(res.chunks) == 1
    assert all(c["word_count"] >= 5 for c in res.chunks)


def test_chunk_content_runt_ships_when_floor_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR", raising=False)
    res = chunk_content(
        [_make_item()], "TEST_101", ctx=_make_ctx(),
        max_chunk_size=10, target_chunk_size=8,
    )
    # Legacy: the tiny trailing sentence ships as its own sub-floor chunk.
    assert len(res.chunks) == 2
