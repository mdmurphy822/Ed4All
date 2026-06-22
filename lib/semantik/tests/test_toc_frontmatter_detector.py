"""Tests for the conservative phantom-TOC + front-matter detector.

Root-cause fix for the phantom-TOC defect (a PDF's full-book Table of Contents,
printed in the front matter, classified as a run of real chapter headings — a
phantom "Chapter 5: Systems of Linear Equations" / "Chapter 6: Polynomials"
appearing in a ch1-3 extract).

NO models / GPU. Exercises ``drop_toc_and_frontmatter`` on SYNTHETIC
``region_provenance`` mirroring the real EA2e case, then chains the post-detector
IR through ``build_chapters_ir`` → ``normalize_cascade_to_ed4all`` →
``DartMarkersValidator``.

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


def _ea2e_case() -> list[dict]:
    """Front matter = [authors, Preface, "Table of Contents", TOC run],
    then real content: Chapter 1 + 1.1 + "Prime Factorization" + blocks.

    Mirrors the real EA2e defect: the book's full TOC is printed in the front
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
# (1) EA2e case — phantom TOC run + front-matter boilerplate dropped;
#     real chapters/sections survive.
# ---------------------------------------------------------------------------


def test_ea2e_toc_run_and_frontmatter_dropped():
    prov = _ea2e_case()
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


def test_ea2e_chapter_count_reflects_only_real_chapters():
    """The post-detector IR has only the 2 real chapters, no phantoms."""
    class _Res:
        region_provenance = _ea2e_case()
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
# (3) Flag OFF → byte-identical pass-through (no drops).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("off_value", ["0", "false", "no", "off", "OFF"])
def test_flag_off_byte_identical(monkeypatch, off_value):
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_TOC", off_value)
    prov = _ea2e_case()
    filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 0
    assert filtered == prov  # byte-identical, no phantom drops


def test_flag_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("SEMANTIK_DROP_FRONTMATTER_TOC", raising=False)
    prov = _ea2e_case()
    _filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 8, "detector must be DEFAULT ON when flag unset"


@pytest.mark.parametrize("on_value", ["1", "true", "yes", "on", "garbage"])
def test_flag_on_and_garbage_enables(monkeypatch, on_value):
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_TOC", on_value)
    prov = _ea2e_case()
    _filtered, dropped = drop_toc_and_frontmatter(prov)
    assert dropped == 8, "truthy / garbage values must keep detector ON"


# ---------------------------------------------------------------------------
# (4) Chain — post-detector IR → adapter → DartMarkersValidator.
# ---------------------------------------------------------------------------


def test_chain_ir_to_adapter_to_dart_markers():
    from lib.semantik.adapter import normalize_cascade_to_ed4all
    from lib.validators.dart_markers import DartMarkersValidator

    class _Res:
        region_provenance = _ea2e_case()
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
    out = normalize_cascade_to_ed4all(res, pdf_stem="ea2e_ch1")

    vres = DartMarkersValidator().validate({"html_content": out["html"]})
    critical = [i for i in vres.issues if i.severity == "critical"]
    assert vres.passed, f"dart_markers failed: {[i.code for i in critical]}"
    assert not critical

    # The phantom chapter titles never reached the HTML.
    for phantom in ("Math Models", "Systems 577", "Quadratic 1155"):
        assert phantom not in out["html"], f"phantom in HTML: {phantom}"
