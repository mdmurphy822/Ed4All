"""Fix 1b — ``ED4ALL_CHUNK_SUBSECTION_BREAK`` size-guarded sub-section break.

The size-guarded successor to ``ED4ALL_CHUNK_SECTION_HARD_BREAK``. Two measured
facts frame this:

* (a) The CO-depth gap — GLM-converted chunks pack to the 800-word ceiling
  because ``merge_small_sections`` only breaks at ``N.M``-numbered / lexicon
  section openers; correctly-leveled but NON-numbered sub-section headings
  (h3/h4, e.g. "Use Place Value with Whole Numbers") do NOT trigger a break.
* (b) The failed prior attempt — ``ED4ALL_CHUNK_SECTION_HARD_BREAK=1`` flushed
  at EVERY matching heading with no size guard, producing a runt cascade
  (498 chunks / p50 42 words / 208 runts on the real corpus).

This flag breaks at a genuine content sub-section heading, but only once the
accumulating buffer has reached ``subsection_min_words`` (default 250) — a break
at a real pedagogical boundary once enough content accumulated, never a runt
cascade. Apparatus / pedagogy-label headings (exercise/review reprints, LO-intro
boilerplate, promoted openers, bare pedagogy labels) are excluded.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Trainforge.chunker.chunker import (
    ChunkerContext,
    SUBSECTION_MIN_WORDS_DEFAULT,
    _is_subsection_break_candidate,
    chunk_content,
    merge_small_sections,
    resolve_chunk_subsection_break,
    resolve_chunk_subsection_min_words,
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


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def _content_subsections(count: int, words_each: int = 55) -> List[_Section]:
    """(b)'s shape: many small heading-led content sub-sections.

    Each heading is a non-numbered content sub-section title (h3) that
    ``is_section_boundary_heading`` misses, so the legacy merge fuses them all
    to the 800-word ceiling.
    """
    return [
        _Section(
            heading=f"Topic {i}: Concept Heading Number {i}",
            content=_words(words_each),
            word_count=words_each,
            level=3,
        )
        for i in range(count)
    ]


# --------------------------------------------------------------------------
# Resolvers — parse-with-fallback, default off / default 250
# --------------------------------------------------------------------------


def test_resolve_subsection_break_default_off():
    assert resolve_chunk_subsection_break({}) is False
    assert resolve_chunk_subsection_break(
        {"ED4ALL_CHUNK_SUBSECTION_BREAK": "true"}
    ) is True
    assert resolve_chunk_subsection_break(
        {"ED4ALL_CHUNK_SUBSECTION_BREAK": "1"}
    ) is True
    assert resolve_chunk_subsection_break(
        {"ED4ALL_CHUNK_SUBSECTION_BREAK": "junk"}
    ) is False


def test_resolve_subsection_min_words_parse_with_fallback():
    assert resolve_chunk_subsection_min_words({}) == SUBSECTION_MIN_WORDS_DEFAULT
    assert SUBSECTION_MIN_WORDS_DEFAULT == 250
    assert resolve_chunk_subsection_min_words(
        {"ED4ALL_CHUNK_SUBSECTION_MIN_WORDS": "120"}
    ) == 120
    # Garbage / non-positive → default floor.
    assert resolve_chunk_subsection_min_words(
        {"ED4ALL_CHUNK_SUBSECTION_MIN_WORDS": "x"}
    ) == SUBSECTION_MIN_WORDS_DEFAULT
    assert resolve_chunk_subsection_min_words(
        {"ED4ALL_CHUNK_SUBSECTION_MIN_WORDS": "0"}
    ) == SUBSECTION_MIN_WORDS_DEFAULT
    assert resolve_chunk_subsection_min_words(
        {"ED4ALL_CHUNK_SUBSECTION_MIN_WORDS": "-5"}
    ) == SUBSECTION_MIN_WORDS_DEFAULT


# --------------------------------------------------------------------------
# Break-candidate classification — reuses apparatus/lexicon classification
# --------------------------------------------------------------------------


def test_break_candidate_content_headings():
    # The gap-closers: non-numbered content sub-section headings + numbered ones.
    assert _is_subsection_break_candidate(
        _Section("Use Place Value with Whole Numbers", "x", 1)
    )
    assert _is_subsection_break_candidate(
        _Section("1.2 Use the Language of Algebra", "x", 1)
    )
    # A long heading that merely CONTAINS an opener token is real content.
    assert _is_subsection_break_candidate(
        _Section("Solutions of Linear Equations", "x", 1)
    )


def test_break_candidate_excludes_apparatus_and_labels():
    # N.M EXERCISES / review reprints.
    assert not _is_subsection_break_candidate(_Section("1.2 Exercises", "x", 1))
    assert not _is_subsection_break_candidate(_Section("Section Exercises", "x", 1))
    assert not _is_subsection_break_candidate(_Section("Review Exercises", "x", 1))
    # Bare pedagogy-opener labels.
    assert not _is_subsection_break_candidate(_Section("Try It", "x", 1))
    assert not _is_subsection_break_candidate(_Section("Solution", "x", 1))
    assert not _is_subsection_break_candidate(_Section("Learning Objectives", "x", 1))
    # Promoted pedagogical opener via attribute (visible text not a legacy label).
    assert not _is_subsection_break_candidate(
        _Section("Try It 9.1", "x", 1, data_dart_opener="try_it")
    )
    # A continuation body (no heading) is not a heading-led boundary.
    assert not _is_subsection_break_candidate(_Section("", "x", 1))


# --------------------------------------------------------------------------
# Flag OFF — byte-identical to legacy merge
# --------------------------------------------------------------------------


def test_flag_off_byte_identical():
    secs = _content_subsections(30)
    legacy = merge_small_sections(secs, max_chunk_size=800)
    off_explicit = merge_small_sections(
        secs, max_chunk_size=800, subsection_break=False
    )
    off_with_floor = merge_small_sections(
        secs, max_chunk_size=800, subsection_break=False, subsection_min_words=250
    )
    assert off_explicit == legacy
    assert off_with_floor == legacy
    # (a): legacy packs to the ceiling — at least one chunk near the 800 cap.
    assert max(len(r.combined_text.split()) for r in legacy) > 700


def test_chunk_content_flag_off_byte_identical(monkeypatch):
    monkeypatch.delenv("ED4ALL_CHUNK_SUBSECTION_BREAK", raising=False)
    monkeypatch.delenv("ED4ALL_CHUNK_SUBSECTION_MIN_WORDS", raising=False)
    item = _make_item()
    baseline = chunk_content([item], "TEST_101", ctx=_make_ctx())
    # Env-off run is byte-identical text-wise to the baseline.
    monkeypatch.setenv("ED4ALL_CHUNK_SUBSECTION_BREAK", "0")
    again = chunk_content([_make_item()], "TEST_101", ctx=_make_ctx())
    assert [c["text"] for c in again.chunks] == [c["text"] for c in baseline.chunks]


# --------------------------------------------------------------------------
# Flag ON — no runt cascade, breaks land on sub-section boundaries, p50 in band
# --------------------------------------------------------------------------


def test_flag_on_no_runt_cascade():
    # (b) reproduction: 30 tiny heading-led sections. A naive break-at-every-
    # heading would emit 30 runts of ~55 words. The size guard collapses them
    # into >=250-word chunks instead.
    secs = _content_subsections(30, words_each=55)
    on = merge_small_sections(
        secs, max_chunk_size=800, subsection_break=True, subsection_min_words=250
    )
    wcs = [len(r.combined_text.split()) for r in on]
    # Every chunk but a legitimately tiny final tail clears the floor.
    assert all(w >= 250 for w in wcs[:-1])
    assert wcs[-1] >= 1
    # NOT a runt cascade: far fewer chunks than the 30-section count.
    assert len(on) < len(secs)
    # But finer than the legacy near-cap packing (fact (a) closed).
    legacy = merge_small_sections(secs, max_chunk_size=800)
    assert len(on) > len(legacy)


def test_flag_on_breaks_land_on_subsection_boundaries():
    secs = _content_subsections(30, words_each=55)
    on = merge_small_sections(
        secs, max_chunk_size=800, subsection_break=True, subsection_min_words=250
    )
    # Every emitted chunk opens on a genuine content sub-section heading (a break
    # candidate), never mid-section and never on an apparatus/label heading.
    for r in on:
        assert _is_subsection_break_candidate(_Section(r.heading, "x", 1))
    # Multiple real breaks happened.
    assert len(on) > 1


def test_flag_on_p50_between_floor_and_cap():
    secs = _content_subsections(40, words_each=55)
    floor = 250
    on = merge_small_sections(
        secs, max_chunk_size=800, subsection_break=True, subsection_min_words=floor
    )
    wcs = [len(r.combined_text.split()) for r in on]
    p50 = statistics.median(wcs)
    assert floor <= p50 <= 800
    # Never above the hard cap.
    assert max(wcs) <= 800


def test_apparatus_heading_does_not_force_a_break():
    # An exercise reprint sitting where the buffer is already past the floor must
    # NOT force a break (it is excluded from the candidate set) — it merges into
    # the current chunk; the NEXT content heading is the break point.
    secs = [
        _Section("Topic A", _words(140), 140, level=3),
        _Section("Topic B", _words(140), 140, level=3),  # buffer now 280 >= 250
        _Section("1.2 Exercises", _words(40), 40, level=3),  # apparatus, no break
        _Section("Topic C", _words(140), 140, level=3),  # content → break here
    ]
    on = merge_small_sections(
        secs, max_chunk_size=800, subsection_break=True, subsection_min_words=250
    )
    # Two chunks: [A + B + exercises] then [C]. The exercise reprint stayed fused
    # with the preceding content instead of opening its own chunk.
    assert len(on) == 2
    assert on[0].merged_headings == ["Topic A", "Topic B", "1.2 Exercises"]
    assert on[1].heading == "Topic C"


def test_below_floor_does_not_break():
    # Buffer never reaches the floor → no sub-section break fires; legacy merge.
    secs = _content_subsections(3, words_each=55)  # 165 words total < 250
    on = merge_small_sections(
        secs, max_chunk_size=800, subsection_break=True, subsection_min_words=250
    )
    legacy = merge_small_sections(secs, max_chunk_size=800)
    assert on == legacy
    assert len(on) == 1


def test_chunk_content_flag_on_no_runts(monkeypatch):
    monkeypatch.setenv("ED4ALL_CHUNK_SUBSECTION_BREAK", "1")
    monkeypatch.setenv("ED4ALL_CHUNK_SUBSECTION_MIN_WORDS", "250")
    item = _make_sectioned_item(30, words_each=55)
    res = chunk_content([item], "TEST_101", ctx=_make_ctx())
    wcs = [c["word_count"] for c in res.chunks]
    assert len(res.chunks) > 1
    assert all(w >= 250 for w in wcs[:-1])


# --------------------------------------------------------------------------
# chunk_content fixtures
# --------------------------------------------------------------------------


def _make_item() -> Dict[str, Any]:
    return {
        "module_id": "m1",
        "item_id": "i1",
        "item_path": "i1",
        "title": "T",
        "resource_type": "page",
        "raw_html": "<p>" + _words(40) + "</p>",
        "sections": [],
        "misconceptions": None,
    }


def _make_sectioned_item(count: int, words_each: int = 55) -> Dict[str, Any]:
    return {
        "module_id": "m1",
        "item_id": "i1",
        "item_path": "i1",
        "title": "T",
        "resource_type": "page",
        "raw_html": "<p>ignored</p>",
        "sections": _content_subsections(count, words_each),
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
