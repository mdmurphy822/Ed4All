"""Vendor-depth per-SECTION stage-2 windows (ED4ALL_OBJECTIVE_WINDOW_PER_SECTION).

Pins the section-window builder: the one-window-per-section shape on a
10-chapter / 71-section fixture tree, the ordered-walk boundary matching
(provenance / ordinal / text-head arms), the starved-section coalescing floor,
document-order + disjoint-cover preservation, the budget invariant inside an
oversized section, the flag resolver parse-with-fallback, the flag-off
byte-identical ``to_dict`` shape, and the stage-2 fingerprint mode/budget keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.chunk_window import (  # noqa: E402
    SECTION_MIN_CHUNKS,
    group_chunks_into_section_windows,
    group_chunks_into_windows,
    partition_chunks_by_section,
    resolve_window_per_section,
)

_SYS_PROMPT = "You are a synthesis author."
_DRAFT = "  (none)"

#: Generous serving window so a normal fixture section packs into ONE window.
_BIG_KW = dict(
    num_ctx=32768,
    system_prompt=_SYS_PROMPT,
    draft_block=_DRAFT,
    max_tokens=1024,
)


def _chunk(cid: str, slug: str, body: str, section_heading: str = "") -> dict:
    src: dict = {
        "source_references": [
            {"sourceId": f"semantik:{slug}#{cid}", "role": "primary"}
        ]
    }
    if section_heading:
        src["section_heading"] = section_heading
    return {"id": cid, "text": body, "source": src}


def _section(heading: str) -> dict:
    return {"headingText": heading, "id": heading.lower().replace(" ", "-")}


def _chapter(cid: str, source_file: str, section_headings) -> dict:
    return {
        "id": cid,
        "source_file": source_file,
        "chapter_text": "x",
        "sections": [_section(h) for h in section_headings],
    }


def _sectioned_corpus(cid: str, slug: str, headings, chunks_per_section=4):
    """One chapter + document-ordered chunks, ``chunks_per_section`` each.

    The first chunk of every section carries the section heading as its
    ``section_heading`` provenance (the boundary anchor); the rest carry a
    subsection-ish heading.
    """
    chapter = _chapter(cid, f"{slug}.html", headings)
    chunks = []
    for si, heading in enumerate(headings):
        for ci in range(chunks_per_section):
            head = heading if ci == 0 else f"Subtopic {si}.{ci}"
            chunks.append(
                _chunk(
                    f"{cid}_s{si}_c{ci}", slug,
                    f"Instructional prose for {heading} part {ci}. " * 5,
                    section_heading=head,
                )
            )
    return chapter, chunks


# ===========================================================================
# Resolver
# ===========================================================================


def test_resolver_default_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_WINDOW_PER_SECTION", raising=False)
    assert resolve_window_per_section() is False


def test_resolver_truthy_tokens(monkeypatch):
    for tok in ("1", "true", "YES", "On"):
        monkeypatch.setenv("ED4ALL_OBJECTIVE_WINDOW_PER_SECTION", tok)
        assert resolve_window_per_section() is True


def test_resolver_garbage_and_falsey_off(monkeypatch):
    for tok in ("0", "false", "no", "off", "banana", ""):
        monkeypatch.setenv("ED4ALL_OBJECTIVE_WINDOW_PER_SECTION", tok)
        assert resolve_window_per_section() is False


def test_resolver_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_WINDOW_PER_SECTION", "1")
    assert resolve_window_per_section(False) is False
    monkeypatch.delenv("ED4ALL_OBJECTIVE_WINDOW_PER_SECTION", raising=False)
    assert resolve_window_per_section(True) is True


# ===========================================================================
# 71-window shape on the fixture tree (the alg-glm-02 topology)
# ===========================================================================


def test_71_window_shape_on_10_chapter_71_section_tree():
    """10 chapters / 71 sections (the real corpus topology) → 71 windows."""
    section_counts = [10, 7, 6, 7, 6, 7, 6, 9, 8, 5]  # == 71
    assert sum(section_counts) == 71
    total_windows = 0
    for chi, n_secs in enumerate(section_counts):
        headings = [f"{chi + 1}.{s + 1} Topic {chi}-{s}" for s in range(n_secs)]
        chapter, chunks = _sectioned_corpus(
            f"ch{chi + 1}", f"book-ch{chi + 1:02d}", headings,
            chunks_per_section=4,
        )
        windows = group_chunks_into_section_windows(chapter, chunks, **_BIG_KW)
        assert len(windows) == n_secs, (
            f"chapter ch{chi + 1}: expected {n_secs} windows, "
            f"got {len(windows)}"
        )
        # Every window carries its section heading + a unique sequential index.
        assert [w.window_index for w in windows] == list(range(n_secs))
        assert [w.section_label for w in windows] == headings
        total_windows += len(windows)
    assert total_windows == 71


def test_disjoint_cover_in_document_order():
    """Concatenated window chunk_ids == the input chunk order exactly."""
    headings = ["1.1 Alpha", "1.2 Beta", "1.3 Gamma"]
    chapter, chunks = _sectioned_corpus("ch1", "one", headings)
    windows = group_chunks_into_section_windows(chapter, chunks, **_BIG_KW)
    flat = [cid for w in windows for cid in w.chunk_ids]
    assert flat == [c["id"] for c in chunks]


# ===========================================================================
# Boundary-matching arms
# ===========================================================================


def test_ordinal_match_survives_title_drift():
    """A chunk whose provenance shares the ``N.M`` ordinal but drifted title
    text still anchors its section boundary."""
    chapter = _chapter("ch1", "one.html", ["1.1 Whole Numbers", "1.2 Language of Algebra"])
    chunks = (
        [_chunk(f"a{i}", "one", "prose " * 20, "1.1 Whole Numbers")
         for i in range(3)]
        # Drifted title, same ordinal.
        + [_chunk(f"b{i}", "one", "prose " * 20, "1.2 The Language" if i == 0
                  else "Grouping Symbols")
           for i in range(3)]
    )
    groups = partition_chunks_by_section(chapter, chunks)
    assert len(groups) == 2
    assert [c["id"] for c in groups[1][1]] == ["b0", "b1", "b2"]


def test_text_head_match_when_provenance_is_subsection():
    """The section heading opening the chunk TEXT anchors the boundary even
    when ``section_heading`` provenance carries a subsection heading."""
    chapter = _chapter("ch1", "one.html", ["1.1 Alpha Topic", "1.2 Beta Topic"])
    chunks = (
        [_chunk(f"a{i}", "one", "prose " * 20, "1.1 Alpha Topic")
         for i in range(3)]
        + [_chunk("b0", "one", "1.2 Beta Topic\nOpening prose of the section.",
                  "Some Subheading")]
        + [_chunk(f"b{i}", "one", "prose " * 20, "Another Subheading")
         for i in range(1, 3)]
    )
    groups = partition_chunks_by_section(chapter, chunks)
    assert len(groups) == 2
    assert [c["id"] for c in groups[1][1]] == ["b0", "b1", "b2"]


def test_missed_boundary_chunks_stay_with_previous_section():
    """A section whose heading never matches attracts no chunks — its chunks
    stay with the preceding section (coalescing outcome, disjoint cover kept)."""
    chapter = _chapter(
        "ch1", "one.html", ["1.1 Alpha", "1.2 Never Matched", "1.3 Gamma"],
    )
    chunks = (
        [_chunk(f"a{i}", "one", "prose " * 20, "1.1 Alpha") for i in range(3)]
        # These belong to 1.2 but carry only subsection provenance + no
        # heading echo in the text head → boundary missed.
        + [_chunk(f"m{i}", "one", "prose " * 20, "Some Subtopic")
           for i in range(3)]
        + [_chunk(f"g{i}", "one", "prose " * 20, "1.3 Gamma") for i in range(3)]
    )
    groups = partition_chunks_by_section(chapter, chunks)
    labels = [label for label, _ in groups]
    assert "1.2 Never Matched" not in labels
    flat = [c["id"] for _, grp in groups for c in grp]
    assert flat == [c["id"] for c in chunks]
    # The missed section's chunks landed with 1.1 Alpha.
    assert [c["id"] for c in groups[0][1]] == ["a0", "a1", "a2", "m0", "m1", "m2"]


# ===========================================================================
# Coalescing floor (never a starved window)
# ===========================================================================


def test_starved_section_coalesces_into_previous():
    headings = ["1.1 Alpha", "1.2 Tiny", "1.3 Gamma"]
    chapter = _chapter("ch1", "one.html", headings)
    chunks = (
        [_chunk(f"a{i}", "one", "prose " * 20, "1.1 Alpha") for i in range(5)]
        + [_chunk(f"t{i}", "one", "prose " * 20, "1.2 Tiny")
           for i in range(SECTION_MIN_CHUNKS - 1)]  # below the floor
        + [_chunk(f"g{i}", "one", "prose " * 20, "1.3 Gamma") for i in range(4)]
    )
    windows = group_chunks_into_section_windows(chapter, chunks, **_BIG_KW)
    assert len(windows) == 2
    assert [w.section_label for w in windows] == ["1.1 Alpha", "1.3 Gamma"]
    for w in windows:
        assert len(w.chunk_ids) >= SECTION_MIN_CHUNKS
    # Tiny's chunks were absorbed by Alpha, order preserved.
    assert windows[0].chunk_ids == ["a0", "a1", "a2", "a3", "a4", "t0", "t1"]


def test_starved_first_section_merges_forward():
    headings = ["1.1 Tiny First", "1.2 Beta"]
    chapter = _chapter("ch1", "one.html", headings)
    chunks = (
        [_chunk("t0", "one", "prose " * 20, "1.1 Tiny First")]
        + [_chunk(f"b{i}", "one", "prose " * 20, "1.2 Beta") for i in range(4)]
    )
    windows = group_chunks_into_section_windows(chapter, chunks, **_BIG_KW)
    assert len(windows) == 1
    assert windows[0].section_label == "1.2 Beta"
    assert windows[0].chunk_ids == ["t0", "b0", "b1", "b2", "b3"]


def test_fewer_than_two_sections_degrades_to_legacy_group():
    """A chapter without a usable section tree degrades to legacy packing."""
    chapter = _chapter("ch1", "one.html", ["1.1 Only"])
    chunks = [_chunk(f"c{i}", "one", "prose " * 20) for i in range(4)]
    per_section = group_chunks_into_section_windows(chapter, chunks, **_BIG_KW)
    legacy = group_chunks_into_windows(chapter, chunks, **_BIG_KW)
    assert [w.to_dict() for w in per_section] == [w.to_dict() for w in legacy]


# ===========================================================================
# Budget invariant inside a section
# ===========================================================================


def test_oversized_section_splits_under_budget():
    """A section too big for one window splits — budget invariant holds and
    every split window keeps the SAME section label."""
    headings = ["1.1 Big", "1.2 Small"]
    chapter = _chapter("ch1", "one.html", headings)
    chunks = (
        [_chunk(f"b{i}", "one", "word " * 200,
                "1.1 Big" if i == 0 else "Sub") for i in range(12)]
        + [_chunk(f"s{i}", "one", "word " * 30, "1.2 Small") for i in range(3)]
    )
    small_kw = dict(_BIG_KW, num_ctx=1400, max_tokens=128)
    windows = group_chunks_into_section_windows(chapter, chunks, **small_kw)
    big_windows = [w for w in windows if w.section_label == "1.1 Big"]
    assert len(big_windows) >= 2
    for w in windows:
        assert w.estimated_prompt_tokens <= small_kw["num_ctx"]
    # Sequential unique indices across the whole chapter.
    assert [w.window_index for w in windows] == list(range(len(windows)))


# ===========================================================================
# Flag-off byte-identical + to_dict shape
# ===========================================================================


def test_legacy_to_dict_has_no_section_label_key():
    chapter = _chapter("ch1", "one.html", ["1.1 A", "1.2 B"])
    chunks = [_chunk(f"c{i}", "one", "prose " * 20) for i in range(4)]
    for w in group_chunks_into_windows(chapter, chunks, **_BIG_KW):
        assert "section_label" not in w.to_dict()


def test_per_section_to_dict_carries_section_label():
    chapter, chunks = _sectioned_corpus("ch1", "one", ["1.1 A", "1.2 B"])
    windows = group_chunks_into_section_windows(chapter, chunks, **_BIG_KW)
    assert [w.to_dict()["section_label"] for w in windows] == ["1.1 A", "1.2 B"]


# ===========================================================================
# Stage-2 fingerprint: mode + budget keys
# ===========================================================================


def test_fingerprint_folds_mode_and_budget():
    from MCP.tools.pipeline_tools import _stage2_window_fingerprint

    chapter, chunks = _sectioned_corpus("ch1", "one", ["1.1 A", "1.2 B"])
    w = group_chunks_into_section_windows(chapter, chunks, **_BIG_KW)[0]

    def _fp(**extra):
        return _stage2_window_fingerprint(
            chunk_ids=w.chunk_ids,
            chunks=w.chunks,
            draft_block=_DRAFT,
            model="m-7b",
            num_ctx=2048,
            system_prompt=_SYS_PROMPT,
            **extra,
        )

    legacy = _fp()
    # Default-absent kwargs are byte-identical to the legacy fingerprint.
    assert _fp(window_mode="", max_candidates=None) == legacy
    # Mode / budget each flip the fingerprint (sidecar invalidation).
    assert _fp(window_mode="per_section:1.1 A") != legacy
    assert _fp(max_candidates=12) != legacy
    assert _fp(window_mode="per_section:1.1 A", max_candidates=12) != _fp(
        window_mode="per_section:1.1 A"
    )
    # Distinct section labels fingerprint distinctly.
    assert _fp(window_mode="per_section:1.1 A") != _fp(
        window_mode="per_section:1.2 B"
    )
