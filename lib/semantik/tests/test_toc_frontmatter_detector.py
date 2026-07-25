"""Tests for the conservative phantom-TOC + front-matter detector.

Root-cause fix for the phantom-TOC defect (a PDF's full-book Table of Contents,
printed in the front matter, classified as a run of real chapter headings — a
phantom "Chapter 5: Systems of Linear Equations" / "Chapter 6: Polynomials"
appearing in a ch1-3 extract).

NO models / GPU. Exercises ``drop_toc_and_frontmatter`` on SYNTHETIC
``region_provenance`` mirroring the observed scanned-textbook case, then chains the post-detector
IR through ``build_chapters_ir`` → ``normalize_cascade_to_ed4all`` →
``SemantiKMarkersValidator``.

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest lib/semantik/tests/test_toc_frontmatter_detector.py -q
"""

from __future__ import annotations

import re

import pytest

from lib.semantik.cascade_ir import build_chapters_ir
from lib.semantik.toc_frontmatter_detector import drop_toc_and_frontmatter


# ---------------------------------------------------------------------------
# Synthetic region_provenance builders.
# ---------------------------------------------------------------------------


def _heading(idx: int, text: str, *, level: int = 1, raw: int = 0) -> dict:
    return {
        "region_index": idx,
        "region_kind": "heading",
        "role": "heading",
        "confidence": 0.9,
        "wcag_status": "passed",
        "first_raw_block_index": raw,
        "pages": [max(1, idx)],
        "heading_text": text,
        "level": level,
        "figure_alt": None,
        "raw_text": text,
    }


def _para(idx: int, text: str, *, raw: int = 0) -> dict:
    return {
        "region_index": idx,
        "region_kind": "paragraph",
        "role": "body",
        "confidence": 0.7,
        "wcag_status": "passed",
        "first_raw_block_index": raw,
        "pages": [max(1, idx)],
        "heading_text": None,
        "level": None,
        "figure_alt": None,
        "raw_text": text,
    }


def _scan_toc_case() -> list[dict]:
    """Front matter = [authors, Preface, "Table of Contents", TOC run],
    then real content: Chapter 1 + 1.1 + "Prime Factorization" + blocks.

    Mirrors the observed scanned-textbook defect: the book's full TOC is printed in the front
    matter and (without the detector) would sprout phantom Chapter 3/4/5/7/10
    headings inside a ch1-3 extract.
    """
    r = 0

    def nxt() -> int:
        nonlocal r
        r += 1
        return r

    return [
        # --- front-matter boilerplate ---
        _heading(0, "SENIOR CONTRIBUTING AUTHORS", raw=nxt()),
        _heading(1, "Preface", raw=nxt()),
        _heading(2, "Table of Contents", raw=nxt()),
        # --- the phantom TOC run (title + INCREASING page numbers) ---
        _heading(3, "Chapter 3 Math Models 301", raw=nxt()),
        _heading(4, "Chapter 4 Graphs 413", raw=nxt()),
        _heading(5, "Chapter 5 Systems 577", raw=nxt()),
        _heading(6, "Chapter 7 Factoring 807", raw=nxt()),
        _heading(7, "Chapter 10 Quadratic 1155", raw=nxt()),
        # --- real content begins here ---
        _heading(8, "Chapter 1: Foundations", level=1, raw=nxt()),
        _para(9, "Whole numbers are the counting numbers and zero.", raw=nxt()),
        _heading(10, "1.1 Introduction to Whole Numbers", level=2, raw=nxt()),
        _para(11, "Place value tells us the value of a digit.", raw=nxt()),
        _heading(12, "Prime Factorization", level=2, raw=nxt()),
        _para(13, "Every composite number factors into primes.", raw=nxt()),
        _heading(14, "Least Common Multiple", level=2, raw=nxt()),
        _para(15, "The LCM is the smallest shared multiple.", raw=nxt()),
        # A second real chapter so chapter count is non-trivial.
        _heading(16, "Chapter 2: Solving Linear Equations", level=1, raw=nxt()),
        _para(17, "Isolate the variable to solve for x.", raw=nxt()),
    ]


def _text_set(provenance: list[dict]) -> set[str]:
    return {
        str(p.get("heading_text") or p.get("raw_text") or "")
        for p in provenance
    }


# ---------------------------------------------------------------------------
# (1) scanned-textbook case — phantom TOC run + front-matter boilerplate dropped;
#     real chapters/sections survive.
# ---------------------------------------------------------------------------


def test_scan_toc_run_and_frontmatter_dropped():
    prov = _scan_toc_case()
    filtered, dropped = drop_toc_and_frontmatter(prov)
    texts = _text_set(filtered)

    # The TOC run (title + increasing pagenums) is gone.
    for phantom in (
        "Chapter 3 Math Models 301",
        "Chapter 4 Graphs 413",
        "Chapter 5 Systems 577",
        "Chapter 7 Factoring 807",
        "Chapter 10 Quadratic 1155",
    ):
        assert phantom not in texts, f"phantom-TOC entry survived: {phantom}"

    # Front-matter boilerplate is gone.
    for boiler in ("SENIOR CONTRIBUTING AUTHORS", "Preface", "Table of Contents"):
        assert boiler not in texts, f"front-matter survived: {boiler}"

    # Real chapters/sections survive untouched.
    for real in (
        "Chapter 1: Foundations",
        "1.1 Introduction to Whole Numbers",
        "Prime Factorization",
        "Least Common Multiple",
        "Chapter 2: Solving Linear Equations",
    ):
        assert real in texts, f"real content was wrongly dropped: {real}"

    # Observability: dropped count = 3 boilerplate + 5 TOC entries.
    assert dropped == 8, f"expected 8 drops, got {dropped}"


def test_scan_chapter_count_reflects_only_real_chapters():
    """The post-detector IR has only the 2 real chapters, no phantoms."""
    class _Res:
        region_provenance = _scan_toc_case()
        heading_tree: list = []

    chapters = build_chapters_ir(_Res())
    titles = [c.title for c in chapters]
    assert titles == [
        "Chapter 1: Foundations",
        "Chapter 2: Solving Linear Equations",
    ], f"phantom chapters present: {titles}"


# ---------------------------------------------------------------------------
# (2) Negative tests — never drop real content.
# ---------------------------------------------------------------------------


def test_single_real_chapter5_not_in_run_survives():
    """A real 'Chapter 5' with content (not in a TOC run) is NOT dropped."""
    prov = [
        _heading(0, "Chapter 5: Systems of Linear Equations", level=1, raw=1),
        _para(1, "A system is two or more equations sharing variables.", raw=2),
        _heading(2, "5.1 Solving by Graphing", level=2, raw=3),
        _para(3, "Graph each equation and find the intersection.", raw=4),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 0
    assert "Chapter 5: Systems of Linear Equations" in _text_set(filtered)
    assert len(filtered) == len(prov)


def test_section_titles_without_pagenumbers_not_dropped():
    """A run of plain section titles WITHOUT trailing page numbers — even in
    the front-matter zone before any chapter — is NOT a TOC run."""
    prov = [
        _heading(0, "Prime Factorization", level=2, raw=1),
        _heading(1, "Least Common Multiple", level=2, raw=2),
        _heading(2, "Greatest Common Factor", level=2, raw=3),
        _heading(3, "Order of Operations", level=2, raw=4),
        _heading(4, "Chapter 1: Foundations", level=1, raw=5),
        _para(5, "Content begins.", raw=6),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    # No trailing page numbers → no TOC run; none are front-matter boilerplate.
    assert dropped == 0, f"unexpected drops: {dropped}"
    assert len(filtered) == len(prov)


def test_short_run_below_threshold_not_dropped():
    """Fewer than _MIN_TOC_RUN (4) title+pagenum entries do NOT drop —
    too few to be unambiguously a printed TOC."""
    prov = [
        _heading(0, "Table of Contents", raw=1),
        _heading(1, "Chapter 3 Math Models 301", raw=2),
        _heading(2, "Chapter 4 Graphs 413", raw=3),
        _heading(3, "Chapter 1: Foundations", level=1, raw=4),
        _para(4, "Content.", raw=5),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    texts = _text_set(filtered)
    # The 2-entry run is below threshold → survives.
    assert "Chapter 3 Math Models 301" in texts
    assert "Chapter 4 Graphs 413" in texts
    # But the bare "Table of Contents" header IS dropped (front-matter).
    assert "Table of Contents" not in texts
    assert dropped == 1


def test_non_increasing_pagenums_not_dropped():
    """A contiguous run of title+pagenum lines whose page numbers do NOT
    trend upward is NOT treated as a TOC run."""
    prov = [
        _heading(0, "Some Heading 500", raw=1),
        _heading(1, "Another Heading 400", raw=2),
        _heading(2, "Third Heading 300", raw=3),
        _heading(3, "Fourth Heading 200", raw=4),
        _heading(4, "Chapter 1: Foundations", level=1, raw=5),
        _para(5, "Content.", raw=6),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 0, "decreasing pagenums must not be dropped as TOC"
    assert len(filtered) == len(prov)


def test_real_numbered_headings_never_form_a_run():
    """Real chapter/section headings (no trailing page numbers) — the exact
    anti-false-positive case from the spec — are NEVER dropped."""
    prov = [
        _heading(0, "Chapter 1: Foundations", level=1, raw=1),
        _para(1, "intro", raw=2),
        _heading(2, "1.1 Introduction to Whole Numbers", level=2, raw=3),
        _para(3, "place value", raw=4),
        _heading(4, "Chapter 2: Linear Equations", level=1, raw=5),
        _para(5, "solve", raw=6),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 0
    assert _text_set(filtered) == _text_set(prov)


# ---------------------------------------------------------------------------
# (2c) Part A — chapter-INDEX cluster (page-number-LESS) dropped; a real
#      chapter opener followed by content is NOT.
# ---------------------------------------------------------------------------


def test_chapter_index_cluster_without_pagenums_dropped(monkeypatch):
    """A back-to-back run (>=3) of 'Chapter N: Title' headings with NO content
    between consecutive entries — the observed scanned-textbook chapter-index cluster, which
    carries NO trailing page numbers — is dropped EVEN THOUGH it sits after the
    first real chapter anchor (so the front-matter TOC-run path never reaches
    it).

    Scoped to the index-cluster pass (zone pass OFF): the default page builder
    keys page off the region index, so the real opener at index 0 is
    artificially page-ADJACENT to the cluster — a layout the page-density zone
    pass (which assumes a real opener is page-SPARSE / page-DISTANT) does not
    model. The zone pass's own realistic-page coverage lives in the (2d)
    suite; here we isolate the index-cluster mechanism."""
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_ZONE", "off")
    prov = [
        # A real chapter opener anchors the front-matter zone end at index 0.
        _heading(0, "Chapter 1: Foundations", level=1, raw=1),
        _para(1, "Whole numbers are the counting numbers and zero.", raw=2),
        # The rendered chapter INDEX cluster — back-to-back, NO content between,
        # NO trailing page numbers (the page-number-less TOC variant).
        _heading(2, "Chapter 1: Foundations", level=1, raw=3),
        _heading(3, "Chapter 2: Solving Linear Equations", level=1, raw=4),
        _heading(4, "Chapter 3: Math Models", level=1, raw=5),
        _heading(5, "Chapter 5: Systems of Linear Equations", level=1, raw=6),
        _heading(6, "Chapter 7: Factoring", level=1, raw=7),
        _heading(7, "Chapter 10: Quadratic Equations", level=1, raw=8),
        # Real content resumes.
        _heading(8, "1.1 Introduction to Whole Numbers", level=2, raw=9),
        _para(9, "Place value tells us the value of a digit.", raw=10),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    texts = _text_set(filtered)
    # The whole index cluster (indices 2..7) is gone.
    for phantom in (
        "Chapter 2: Solving Linear Equations",
        "Chapter 3: Math Models",
        "Chapter 5: Systems of Linear Equations",
        "Chapter 7: Factoring",
        "Chapter 10: Quadratic Equations",
    ):
        assert phantom not in texts, f"index-cluster entry survived: {phantom}"
    # The real opener + content + first section survive.
    assert "Chapter 1: Foundations" in texts  # the FIRST (real) one stays
    assert "1.1 Introduction to Whole Numbers" in texts
    assert "Place value tells us the value of a digit." in texts
    # 6 cluster entries dropped (indices 2..7).
    assert dropped == 6, f"expected 6 index-cluster drops, got {dropped}"


def test_bare_ordinal_index_cluster_dropped(monkeypatch):
    """A bare-ordinal 'N Title' index cluster (no 'Chapter' word, no page
    numbers) is also dropped.

    Scoped to the index-cluster pass (zone pass OFF) for the same
    page-adjacency reason as the test above — the page-density zone pass has
    its own realistic-page coverage in the (2d) suite."""
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_ZONE", "off")
    prov = [
        _heading(0, "3 Math Models", level=1, raw=1),
        _heading(1, "5 Systems of Linear Equations", level=1, raw=2),
        _heading(2, "7 Factoring", level=1, raw=3),
        _heading(3, "10 Quadratic Equations", level=1, raw=4),
        _heading(4, "Chapter 1: Foundations", level=1, raw=5),
        _para(5, "Content begins.", raw=6),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    texts = _text_set(filtered)
    for phantom in ("3 Math Models", "5 Systems of Linear Equations", "7 Factoring"):
        assert phantom not in texts, f"bare-ordinal index survived: {phantom}"
    assert "Chapter 1: Foundations" in texts
    assert dropped == 4


def test_real_chapter_openers_with_content_not_a_cluster():
    """Real chapter openers each followed by SUBSTANTIAL content are NOT an
    index cluster (the content between consecutive openers breaks the run) —
    the exact anti-false-positive the spec demands."""
    prov = [
        _heading(0, "Chapter 1: Foundations", level=1, raw=1),
        _para(1, "Whole numbers intro.", raw=2),
        _heading(2, "1.1 Whole Numbers", level=2, raw=3),
        _para(3, "Place value.", raw=4),
        _heading(4, "Chapter 2: Linear Equations", level=1, raw=5),
        _para(5, "Solve for x.", raw=6),
        _heading(6, "2.1 Subtraction Property", level=2, raw=7),
        _para(7, "Add to both sides.", raw=8),
        _heading(8, "Chapter 3: Math Models", level=1, raw=9),
        _para(9, "Problem solving.", raw=10),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    # Nothing dropped — every chapter opener is separated by real content.
    assert dropped == 0, f"real openers wrongly dropped as a cluster: {dropped}"
    assert _text_set(filtered) == _text_set(prov)


def test_two_chapter_openers_below_index_run_threshold_kept():
    """Only TWO back-to-back chapter headings (< _MIN_CHAPTER_INDEX_RUN of 3)
    are NOT treated as an index cluster — too few to be unambiguous."""
    prov = [
        _heading(0, "Chapter 1: Foundations", level=1, raw=1),
        _heading(1, "Chapter 2: Linear Equations", level=1, raw=2),
        _para(2, "Content.", raw=3),
    ]
    filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 0, "a 2-entry run must not be dropped as an index cluster"
    assert len(filtered) == len(prov)


# ---------------------------------------------------------------------------
# (3) Flag OFF → byte-identical pass-through (no drops).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("off_value", ["0", "false", "no", "off", "OFF"])
def test_flag_off_byte_identical(monkeypatch, off_value):
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_TOC", off_value)
    prov = _scan_toc_case()
    filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 0
    assert filtered == prov  # byte-identical, no phantom drops


def test_flag_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_DROP_FRONTMATTER_TOC", raising=False)
    prov = _scan_toc_case()
    _filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 8, "detector must be DEFAULT ON when flag unset"


@pytest.mark.parametrize("on_value", ["1", "true", "yes", "on", "garbage"])
def test_flag_on_and_garbage_enables(monkeypatch, on_value):
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_TOC", on_value)
    prov = _scan_toc_case()
    _filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 8, "truthy / garbage values must keep detector ON"


# ---------------------------------------------------------------------------
# (2d) PAGE-DENSITY front-matter-zone pass — the robust positional fix. The
#      discriminator is page DENSITY, not "no content between entries", so it
#      catches the preface chapter-SUMMARY (each "Chapter N" followed by a real
#      summary paragraph) that the index-cluster pass (2c) misses.
# ---------------------------------------------------------------------------


def _hp(idx: int, text: str, page: int, *, level: int = 1, raw: int = 0) -> dict:
    """A heading on an EXPLICIT page (the page-density tests need page control,
    unlike the default builders that key page off the region index)."""
    h = _heading(idx, text, level=level, raw=raw)
    h["pages"] = [page]
    return h


def _pp(idx: int, text: str, page: int, *, raw: int = 0) -> dict:
    """A paragraph on an EXPLICIT page."""
    p = _para(idx, text, raw=raw)
    p["pages"] = [page]
    return p


def _preface_summary_case() -> list[dict]:
    """Mirrors the REAL scanned-textbook defect the index-cluster pass MISSES: a preface
    chapter-by-chapter SUMMARY — each "Chapter N: Title" FOLLOWED BY a real
    summary paragraph (content BETWEEN consecutive entries) — densely packed on
    pages 9-10, then the page-SPARSE real body starting page 13."""
    return [
        # Dense preface chapter-summary cluster on pages 9-10 (K=4 within P<=3).
        _hp(0, "Chapter 1: Foundations", 9, raw=1),
        _pp(1, "This chapter reviews whole numbers and integers.", 9, raw=2),
        _hp(2, "Chapter 2: Solving Linear Equations", 9, raw=3),
        _pp(3, "Here we develop the algebra of solving equations.", 9, raw=4),
        _hp(4, "Chapter 3: Math Models", 10, raw=5),
        _pp(5, "We apply algebra to real-world models and percents.", 10, raw=6),
        _hp(6, "Chapter 5: Systems of Linear Equations", 10, raw=7),
        _pp(7, "Systems are solved by graphing, substitution, elimination.", 10, raw=8),
        # The page-SPARSE, content-rich REAL body — page 13+.
        _hp(8, "Chapter Outline", 13, level=3, raw=9),
        _hp(9, "1.1 Introduction to Whole Numbers", 13, level=2, raw=10),
        _pp(10, "Place value tells us the value of a digit.", 13, raw=11),
        _pp(11, "Whole numbers are the counting numbers and zero.", 14, raw=12),
        _pp(12, "We can round whole numbers to a given place.", 16, raw=13),
    ]


def test_page_density_cluster_with_content_between_dropped():
    """The keystone case: a dense chapter cluster WITH summary paragraphs
    between consecutive entries — which the back-to-back index-cluster pass
    misses — IS dropped by the page-density zone pass; the page-sparse real
    body section survives."""
    prov = _preface_summary_case()
    filtered, dropped = drop_toc_and_frontmatter(prov)
    texts = _text_set(filtered)
    # Every phantom preface "Chapter N" heading is gone.
    for phantom in (
        "Chapter 1: Foundations",
        "Chapter 2: Solving Linear Equations",
        "Chapter 3: Math Models",
        "Chapter 5: Systems of Linear Equations",
    ):
        assert phantom not in texts, f"phantom preface-summary survived: {phantom}"
    # The page-sparse REAL body section is kept.
    assert "1.1 Introduction to Whole Numbers" in texts
    assert "Place value tells us the value of a digit." in texts
    assert dropped >= 4, f"expected >=4 cluster drops, got {dropped}"


def test_page_density_index_cluster_pass_alone_misses_it():
    """Documents the gap the zone pass closes: the index-cluster pass (which
    keys on NO-content-between) returns EMPTY for the content-between preface
    summary, but the zone pass (page-density) catches all four phantoms."""
    from lib.semantik.toc_frontmatter_detector import (
        _find_chapter_index_clusters,
        _find_frontmatter_zone_clusters,
    )

    prov = _preface_summary_case()
    idx_hits = _find_chapter_index_clusters(prov)
    zone_hits = _find_frontmatter_zone_clusters(prov)
    assert idx_hits == set(), (
        "index-cluster pass should MISS the content-between summary "
        f"(got {idx_hits})"
    )
    dropped_ris = {prov[i]["region_index"] for i in zone_hits}
    assert {0, 2, 4, 6} <= dropped_ris, (
        f"zone pass must catch the 4 phantom chapter headings, got {dropped_ris}"
    )


def test_page_density_sparse_real_body_not_dropped():
    """ANTI-FALSE-POSITIVE: page-SPARSE real body chapters (each spanning many
    pages of content before the next opener) NEVER satisfy the density window,
    so they are not dropped — even though they sit on early pages."""
    from lib.semantik.toc_frontmatter_detector import (
        _find_frontmatter_zone_clusters,
    )

    prov = [
        _hp(0, "Chapter 1: Foundations", 3, raw=1),
        _pp(1, "intro", 4, raw=2),
        _pp(2, "more", 6, raw=3),
        _pp(3, "more", 10, raw=4),
        _hp(4, "Chapter 2: Linear Equations", 14, raw=5),
        _pp(5, "solve", 15, raw=6),
        _hp(6, "Chapter 3: Math Models", 30, raw=7),
        _pp(7, "models", 31, raw=8),
    ]
    assert _find_frontmatter_zone_clusters(prov) == set()
    filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 0, f"page-sparse real body wrongly dropped: {dropped}"
    assert _text_set(filtered) == _text_set(prov)


def test_page_density_dense_cluster_deep_in_body_not_dropped():
    """ANTI-FALSE-POSITIVE (front-matter-zone guard): a dense chapter cluster
    DEEP in the body (first page beyond _ZONE_MAX_FRONTMATTER_PAGE) is NOT a
    front-matter cluster and is not dropped by the zone pass."""
    from lib.semantik.toc_frontmatter_detector import (
        _find_frontmatter_zone_clusters,
    )

    prov = [
        _hp(0, "Chapter 1: A", 50, raw=1),
        _hp(1, "Chapter 2: B", 50, raw=2),
        _hp(2, "Chapter 3: C", 51, raw=3),
        _hp(3, "Chapter 4: D", 52, raw=4),
        _pp(4, "content", 52, raw=5),
    ]
    assert _find_frontmatter_zone_clusters(prov) == set()


def test_page_density_below_k_threshold_not_dropped():
    """Fewer than _ZONE_MIN_CHAPTER_CLUSTER (4) dense headings → not a cluster."""
    from lib.semantik.toc_frontmatter_detector import (
        _find_frontmatter_zone_clusters,
    )

    prov = [
        _hp(0, "Chapter 1: A", 9, raw=1),
        _pp(1, "x", 9, raw=2),
        _hp(2, "Chapter 2: B", 9, raw=3),
        _pp(3, "x", 9, raw=4),
        _hp(4, "Chapter 3: C", 10, raw=5),
        _pp(5, "x", 10, raw=6),
        # Body-start far away so the 3-run can't extend.
        _hp(6, "Chapter 4: D", 40, raw=7),
        _pp(7, "x", 40, raw=8),
    ]
    assert _find_frontmatter_zone_clusters(prov) == set()


def test_page_density_diagnostics_stamped():
    """The page-density drop count is surfaced on the optional diagnostics
    out-param (the audit stamp mirroring resegment / TOC diagnostics)."""
    prov = _preface_summary_case()
    diag: dict = {}
    _filtered, _dropped = drop_toc_and_frontmatter(prov, diagnostics=diag)
    assert diag["frontmatter_zone_dropped"] >= 4
    assert diag["total_dropped"] >= diag["frontmatter_zone_dropped"]


@pytest.mark.parametrize("off_value", ["0", "false", "no", "off", "OFF"])
def test_zone_flag_off_no_density_drops(monkeypatch, off_value):
    """SEMANTIK_DROP_FRONTMATTER_ZONE off → the page-density pass does NOT
    fire (its drops are absent); the other passes are unaffected. On the
    content-between summary case (which ONLY the zone pass catches) this means
    the cluster survives, proving the gate governs exactly this pass."""
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_ZONE", off_value)
    prov = _preface_summary_case()
    diag: dict = {}
    filtered, _dropped = drop_toc_and_frontmatter(prov, diagnostics=diag)
    assert diag["frontmatter_zone_dropped"] == 0
    # With the zone pass off, the content-between preface summary survives
    # (the index-cluster pass misses it) — proving the gate's effect.
    assert "Chapter 3: Math Models" in _text_set(filtered)


def test_zone_flag_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_DROP_FRONTMATTER_ZONE", raising=False)
    prov = _preface_summary_case()
    diag: dict = {}
    drop_toc_and_frontmatter(prov, diagnostics=diag)
    assert diag["frontmatter_zone_dropped"] >= 4, "zone pass must default ON"


# ---------------------------------------------------------------------------
# (4) Chain — post-detector IR → adapter → SemantiKMarkersValidator.
# ---------------------------------------------------------------------------


def test_chain_ir_to_adapter_to_semantik_markers():
    from lib.semantik.adapter import normalize_cascade_to_ed4all
    from lib.validators.semantik_markers import SemantiKMarkersValidator

    class _Res:
        region_provenance = _scan_toc_case()
        heading_tree: list = []

    chapters = build_chapters_ir(_Res())

    # >= 2 real chapters, NO phantom-TOC chapters.
    titles = [c.title for c in chapters]
    assert len(chapters) >= 2, f"expected >=2 real chapters, got {titles}"
    for phantom in ("Math Models", "Graphs", "Systems", "Factoring", "Quadratic"):
        assert not any(phantom in t for t in titles), (
            f"phantom-TOC chapter leaked into IR: {titles}"
        )

    class _Adapt:
        exit_action = "ship_with_confidence"
        wcag_status = "passed"
        theta_score = 0.9
        flags: list = []
        lane_used = "fast-lane"
        lang = "en"

    res = _Adapt()
    res.chapters = chapters
    out = normalize_cascade_to_ed4all(res, pdf_stem="algebra_ch1")

    vres = SemantiKMarkersValidator().validate({"html_content": out["html"]})
    critical = [i for i in vres.issues if i.severity == "critical"]
    assert vres.passed, f"semantik_markers failed: {[i.code for i in critical]}"
    assert not critical

    # The phantom chapter titles never reached the HTML.
    for phantom in ("Math Models", "Systems 577", "Quadratic 1155"):
        assert phantom not in out["html"], f"phantom in HTML: {phantom}"
