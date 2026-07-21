"""P3a tests for the SemantiK cascade → adapter chapters-IR converter.

Built + run with NO models / GPU. Exercises ``build_chapters_ir`` against a
SYNTHETIC ``region_provenance`` (the distilled per-region provenance the
cascade surfaces — see ``SemantiK/semantik_structure/cascade.py::
_build_region_provenance``), then chains the produced IR through the REAL
P2a adapter (``normalize_cascade_to_ed4all``) and the REAL Ed4All contract
validators (``SemantiKMarkersValidator``, ``source_refs``,
``SemanticStructureExtractor``).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest lib/semantik/tests/test_cascade_ir.py -q

The synthetic provenance is built FROM SCRATCH (no vendored SemantiK
fixture). It carries ≥2 chapters (two content-bearing level-1 headings), a
figure region (with alt + a 1.0 confidence that bands to omitted), page
numbers, and an answer-key NON-CONTENT heading that must NOT open a chapter
and must be filtered so a chapter never trips the >40-section collapse.
"""

from __future__ import annotations

import re

import pytest

from lib.semantik.adapter import normalize_cascade_to_ed4all
from lib.semantik.cascade_ir import build_chapters_ir


# ---------------------------------------------------------------------------
# Synthetic cascade-result fixture (built from scratch — region_provenance).
# ---------------------------------------------------------------------------


def _region_provenance() -> list[dict]:
    """A hand-built ``region_provenance`` list in DOCUMENT order.

    Mirrors the shape ``_build_region_provenance`` emits: each region dict
    carries ``region_index / region_kind / role / confidence / wcag_status /
    first_raw_block_index / pages / heading_text / level / figure_alt /
    raw_text``. Two content-bearing level-1 headings → two chapters; an
    answer-key non-content heading; a figure.
    """
    return [
        # --- Chapter 1 boundary (content-bearing level-1 heading) ---------
        {
            "region_index": 0,
            "region_kind": "heading",
            "role": "heading",
            "confidence": 0.95,
            "wcag_status": "passed",
            "first_raw_block_index": 0,
            "pages": [1],
            "heading_text": "Chapter 1: Foundations",
            "level": 1,
            "figure_alt": None,
            "raw_text": "Chapter 1: Foundations",
        },
        {
            "region_index": 1,
            "region_kind": "heading",
            "role": "heading",
            "confidence": 0.83,
            "wcag_status": "passed",
            "first_raw_block_index": 1,
            "pages": [1],
            "heading_text": "Introduction to Algebra",
            "level": 2,  # section heading — stays inside chapter 1
            "figure_alt": None,
            "raw_text": "Introduction to Algebra",
        },
        {
            "region_index": 2,
            "region_kind": "paragraph",
            "role": "body",
            "confidence": 0.61,
            "wcag_status": "passed",
            "first_raw_block_index": 2,
            "pages": [2, 3],  # contiguous range → "2-3"
            "heading_text": None,
            "level": None,
            "figure_alt": None,
            "raw_text": "The order of operations is PEMDAS.",
        },
        # A genuine bullet list (≥2 markers) → <ul><li> deterministic render.
        {
            "region_index": 3,
            "region_kind": "list",
            "role": "list",
            "confidence": 0.7,
            "wcag_status": "passed",
            "first_raw_block_index": 3,
            "pages": [2],
            "heading_text": None,
            "level": None,
            "figure_alt": None,
            "raw_text": "• Add integers. • Subtract integers. • Multiply integers.",
        },
        {
            "region_index": 4,
            "region_kind": "figure",
            "role": "figure",
            "confidence": 1.0,  # bands to omitted (§3.6)
            "wcag_status": "passed",
            "first_raw_block_index": 4,
            "pages": [3],
            "heading_text": "Number Line",
            "level": None,
            "figure_alt": "A number line from -5 to 5.",
            "raw_text": "Figure 1.1 A number line from -5 to 5.",
        },
        # An answer-key NON-CONTENT heading — must NOT open a chapter and
        # must be filtered out of the rendered sections (§3.4).
        {
            "region_index": 5,
            "region_kind": "heading",
            "role": "heading",
            "confidence": 0.4,
            "wcag_status": "passed",
            "first_raw_block_index": 6,
            "pages": [4],
            "heading_text": "78 41. 900 42. 800 43. 700",
            "level": 1,  # level-1 BUT non-content → still no chapter
            "figure_alt": None,
            "raw_text": "78 41. 900 42. 800 43. 700",
        },
        # --- Chapter 2 boundary (content-bearing level-1 heading) ---------
        {
            "region_index": 6,
            "region_kind": "heading",
            "role": "heading",
            "confidence": 0.9,
            "wcag_status": "passed",
            "first_raw_block_index": 10,
            "pages": [5],
            "heading_text": "Chapter 2: Linear Equations",
            "level": 1,
            "figure_alt": None,
            "raw_text": "Chapter 2: Linear Equations",
        },
        {
            "region_index": 7,
            "region_kind": "paragraph",
            "role": "body",
            "confidence": 0.79,
            "wcag_status": "passed",
            "first_raw_block_index": 11,
            "pages": [5],
            "heading_text": None,
            "level": None,
            "figure_alt": None,
            "raw_text": "Solve for x in 2x + 3 = 7.",
        },
        {
            "region_index": 8,
            "region_kind": "paragraph",
            "role": "body",
            "confidence": 0.6,
            "wcag_status": "passed",
            "first_raw_block_index": 12,
            "pages": [6],
            "heading_text": None,
            "level": None,
            "figure_alt": None,
            "raw_text": "A headingless prose block continues here.",
        },
    ]


class _SyntheticPipelineResult:
    """Duck-typed stand-in for the P3a-surfaced ``PipelineV2Result``."""

    def __init__(self):
        self.pdf = "sample_text_ch1.pdf"
        self.wcag_status = "passed"
        self.exit_action = "ship_with_confidence"
        self.theta_score = 0.91
        self.flags: list[str] = []
        self.lane_used = "fast-lane"
        self.lang = "en"
        self.region_provenance = _region_provenance()
        self.heading_tree = [
            (1, "Chapter 1: Foundations"),
            (2, "Introduction to Algebra"),
            (1, "Chapter 2: Linear Equations"),
        ]


# ---------------------------------------------------------------------------
# (a) build_chapters_ir grouping — ≥2 chapters, non-content heading absorbed.
# ---------------------------------------------------------------------------


def test_a_build_chapters_ir_groups_two_chapters():
    chapters = build_chapters_ir(_SyntheticPipelineResult())
    assert len(chapters) == 2, f"expected 2 chapters, got {len(chapters)}"
    assert chapters[0].title == "Chapter 1: Foundations"
    assert chapters[1].title == "Chapter 2: Linear Equations"

    # The answer-key non-content heading did NOT open a third chapter; it is
    # absorbed as a block under chapter 1 (the adapter later drops it).
    titles = [c.title for c in chapters]
    assert "78 41. 900 42. 800 43. 700" not in titles

    # The chapter-title headings are NOT re-emitted as in-chapter blocks.
    ch1_block_headings = [b.heading_text for b in chapters[0].blocks]
    assert "Chapter 1: Foundations" not in ch1_block_headings


def test_a_first_raw_block_index_carried_verbatim():
    """§3.3a — the determinism anchor is carried verbatim onto the block."""
    chapters = build_chapters_ir(_SyntheticPipelineResult())
    all_blocks = [b for c in chapters for b in c.blocks]
    indices = {b.raw_block_index for b in all_blocks}
    # The figure region anchored to raw block 4; the chapter-2 prose to 11/12.
    assert 4 in indices  # figure first_raw_block_index
    assert 11 in indices and 12 in indices


# ---------------------------------------------------------------------------
# (b) Chain P3a → adapter → SemantiKMarkersValidator (end-to-end on synthetic).
# ---------------------------------------------------------------------------


@pytest.fixture
def chained_out():
    chapters = build_chapters_ir(_SyntheticPipelineResult())
    # Hand the produced IR to the adapter via a duck-typed result carrying
    # the chapters IR + the doc-level signals the adapter reads.
    class _Res:
        chapters = None
        exit_action = "ship_with_confidence"
        wcag_status = "passed"
        theta_score = 0.91
        flags: list[str] = []
        lane_used = "fast-lane"
        lang = "en"

    res = _Res()
    res.chapters = chapters
    return normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")


def test_b_semantik_markers_validator_passes(chained_out):
    from lib.validators.semantik_markers import SemantiKMarkersValidator

    out = chained_out
    res = SemantiKMarkersValidator().validate({"html_content": out["html"]})
    critical = [i for i in res.issues if i.severity == "critical"]
    assert res.passed, f"semantik_markers failed: {[i.code for i in critical]}"
    assert not critical
    assert 'data-semantik-source=""' not in out["html"]
    assert 'data-semantik-source="synthesized"' in out["html"]
    # Page provenance survived the conversion.
    assert 'data-semantik-page-kind="physical"' in out["html"]
    assert 'data-semantik-pages="2-3"' in out["html"]  # contiguous-range collapse


def test_b_source_ids_resolve(chained_out):
    from lib.validators.source_refs import SOURCE_ID_RE

    out = chained_out
    slug = out["slug"]
    sidecar_ids = {
        s["section_id"] for s in out["synthesized_sidecar"]["sections"]
    }
    html_ids = set(re.findall(r'data-semantik-block-id="([^"]+)"', out["html"]))
    assert html_ids
    for bid in html_ids:
        source_id = f"semantik:{slug}#{bid}"
        assert SOURCE_ID_RE.match(source_id), f"bad sourceId: {source_id}"
        assert bid in sidecar_ids, f"{bid} unresolved against sidecar"


def test_b_structure_extractor_two_chapters_no_collapse(chained_out):
    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        SemanticStructureExtractor,
        _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
    )

    out = chained_out
    structure = SemanticStructureExtractor().extract(out["html"])
    chapters = structure["chapters"]
    assert len(chapters) >= 2, f"expected >=2 chapters, got {len(chapters)}"
    for ch in chapters:
        secs = ch.get("sections", [])
        assert len(secs) <= _STRUCTURE_COLLAPSE_SECTION_THRESHOLD


def test_b_figure_alt_carried(chained_out):
    out = chained_out
    alts = [
        s["data"].get("figure_alt")
        for s in out["synthesized_sidecar"]["sections"]
        if s["data"].get("figure_alt")
    ]
    assert "A number line from -5 to 5." in alts


# ---------------------------------------------------------------------------
# (c) Determinism — same provenance → identical block-id set across runs.
# ---------------------------------------------------------------------------


def test_c_determinism_identical_block_ids():
    def _run():
        chapters = build_chapters_ir(_SyntheticPipelineResult())

        class _Res:
            exit_action = "ship_with_confidence"
            wcag_status = "passed"
            theta_score = 0.91
            flags: list[str] = []
            lane_used = "fast-lane"
            lang = "en"

        res = _Res()
        res.chapters = chapters
        out = normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")
        return set(re.findall(r'data-semantik-block-id="([^"]+)"', out["html"]))

    ids_a = _run()
    ids_b = _run()
    assert ids_a == ids_b, "non-deterministic data-semantik-block-id set"
    assert ids_a, "no block ids emitted"

    # And the first_raw_block_index anchors are identical too.
    ch_a = build_chapters_ir(_SyntheticPipelineResult())
    ch_b = build_chapters_ir(_SyntheticPipelineResult())
    anchors_a = [b.raw_block_index for c in ch_a for b in c.blocks]
    anchors_b = [b.raw_block_index for c in ch_b for b in c.blocks]
    assert anchors_a == anchors_b


# ---------------------------------------------------------------------------
# (d) Coercion arms — .cascade dict and bare list both resolve.
# ---------------------------------------------------------------------------


def test_d_coercion_from_cascade_dict_and_bare_list():
    prov = _region_provenance()

    class _ResDict:
        region_provenance = None
        cascade = {"region_provenance": prov, "heading_tree": []}

    from_obj = build_chapters_ir(_SyntheticPipelineResult())
    from_cascade = build_chapters_ir(_ResDict())
    from_bare = build_chapters_ir(prov)

    # All three resolve the same chapter count + titles.
    assert len(from_obj) == len(from_cascade) == len(from_bare) == 2
    assert [c.title for c in from_cascade] == [c.title for c in from_bare]


def test_d_empty_provenance_yields_empty_ir():
    assert build_chapters_ir([]) == []
    assert build_chapters_ir({"region_provenance": []}) == []


# ---------------------------------------------------------------------------
# (e) Part B — section-number chapter derivation (no L1 openers in content).
# ---------------------------------------------------------------------------


def _h(idx: int, text: str, *, level: int = 2, raw: int = 0) -> dict:
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


def _p(idx: int, text: str, *, raw: int = 0) -> dict:
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


def prov_result(prov: list[dict]):
    """Chain a ``region_provenance`` list → IR → a duck-typed adapter result.

    Mirrors the ``chained_out`` fixture but for an inline-built provenance list
    (used by the overflow-continuation tests): ``build_chapters_ir`` groups the
    provenance into chapters, then a duck-typed result carrying that IR + the
    doc-level signals is returned for ``normalize_cascade_to_ed4all``.
    """
    chapters = build_chapters_ir(prov)

    class _Res:
        exit_action = "ship_with_confidence"
        wcag_status = "passed"
        theta_score = 0.91
        flags: list[str] = []
        lane_used = "fast-lane"
        lang = "en"

    res = _Res()
    res.chapters = chapters
    return res


def test_e_derive_chapters_by_section_number_multi_chapter():
    """Content with N.M sections across multiple leading numbers but NO real
    L1 chapter openers groups into one chapter per leading number."""
    prov = [
        _h(0, "1.1 Introduction to Whole Numbers", raw=1),
        _p(1, "Place value tells us the value of a digit.", raw=2),
        _h(2, "1.2 Use the Language of Algebra", raw=3),
        _p(3, "A variable is a letter standing for a number.", raw=4),
        _h(4, "2.1 Solve Equations", raw=5),
        _p(5, "Isolate the variable.", raw=6),
        _h(6, "2.2 Multiplication Property", raw=7),
        _p(7, "Multiply both sides.", raw=8),
        _h(8, "3.1 Problem-Solving Strategy", raw=9),
        _p(9, "Read the problem twice.", raw=10),
    ]
    chapters = build_chapters_ir(prov)
    # Three leading numbers → three chapters.
    assert len(chapters) == 3, [c.title for c in chapters]
    assert chapters[0].title == "Chapter 1"
    assert chapters[1].title == "Chapter 2"
    assert chapters[2].title == "Chapter 3"
    # Sections + their following prose attach to the right chapter.
    ch1_headings = [b.heading_text for b in chapters[0].blocks if b.heading_text]
    assert "1.1 Introduction to Whole Numbers" in ch1_headings
    assert "1.2 Use the Language of Algebra" in ch1_headings


def test_e_section_number_path_reuses_real_chapter_title():
    """When a real 'Chapter N: Title' heading text appears (even with no content
    behind it — e.g. the surviving opener of a dropped index), the derived
    chapter reuses that title for chapter N."""
    prov = [
        # A bare opener heading with no content (would be filtered as a boundary
        # producing an empty chapter on the legacy path); here it only supplies
        # the title for the section-derived chapter 1.
        _h(0, "Chapter 1: Foundations", level=1, raw=1),
        _h(1, "1.1 Whole Numbers", raw=2),
        _p(2, "Counting numbers and zero.", raw=3),
        _h(3, "2.1 Solve Equations", raw=4),
        _p(4, "Isolate x.", raw=5),
    ]
    chapters = build_chapters_ir(prov)
    titles = [c.title for c in chapters]
    # Chapter 1 reuses the real opener text; chapter 2 falls back to "Chapter 2".
    assert "Chapter 1: Foundations" in titles
    assert "Chapter 2" in titles


def _meta(idx: int, text: str, *, raw: int = 0) -> dict:
    """A ``metadata_drop`` furniture region (e.g. a dropped running header)."""
    return {
        "region_index": idx,
        "region_kind": "metadata_drop",
        "role": "metadata_drop",
        "confidence": None,
        "wcag_status": "failed",
        "first_raw_block_index": raw,
        "pages": [max(1, idx)],
        "heading_text": None,
        "level": 1,
        "figure_alt": None,
        "raw_text": text,
    }


def test_e_section_number_path_recovers_title_from_running_header():
    """D13 — when the real 'Chapter N: Title' opener never survived but the
    page-glued running header ("Chapter 1 Foundations 7/9/11") was dropped to
    metadata_drop furniture by the D6 backstop, the derived chapter recovers the
    fuller title (folio stripped) instead of falling back to a bare "Chapter N".
    """
    prov = [
        # The recurring page-glued running header the D6 backstop dropped. Only
        # its NAME ("Foundations") survives attached to the ordinal here — there
        # is NO real "Chapter 1: Foundations" content heading.
        _meta(0, "Chapter 1 Foundations 7"),
        _h(1, "1.1 Introduction to Whole Numbers", raw=1),
        _p(2, "Counting numbers and zero.", raw=2),
        _meta(3, "Chapter 1 Foundations 9"),
        _h(4, "1.2 Use the Language of Algebra", raw=3),
        _p(5, "A variable is a letter standing for a number.", raw=4),
        _meta(6, "Chapter 1 Foundations 11"),
    ]
    chapters = build_chapters_ir(prov)
    titles = [c.title for c in chapters]
    # Recovered "Chapter 1 Foundations", NOT bare "Chapter 1".
    assert titles == ["Chapter 1 Foundations"], titles
    # And it drives the document <h1> / <title> selection.
    from lib.semantik.adapter import _select_document_title

    assert (
        _select_document_title(chapters, "stem") == "Chapter 1 Foundations"
    )


def test_e_running_header_recovery_ignores_fused_content_paragraph():
    """A CONTENT paragraph that merely OPENS with the running-header text (a
    fused OCR block, region_kind='paragraph') must NOT be mined for the title —
    only genuine metadata_drop furniture is. Absent any furniture header, the
    bare "Chapter N" fallback is kept (no garbage title from a long sentence)."""
    prov = [
        # A fused content paragraph — NOT furniture. Ends in a digit ("7") that
        # a naive strip would treat as a folio, yielding a garbage title.
        _p(0, "Chapter 1 Foundations 55 Solution a Take 5 positive from 7", raw=1),
        _h(1, "1.1 Introduction to Whole Numbers", raw=2),
        _p(2, "Counting numbers.", raw=3),
    ]
    chapters = build_chapters_ir(prov)
    assert [c.title for c in chapters] == ["Chapter 1"], [
        c.title for c in chapters
    ]


def test_e_real_opener_wins_over_running_header_recovery():
    """A real 'Chapter N: Title' content heading still wins over the
    running-header-recovered title (the furniture recovery is a fallback tier,
    never an override)."""
    prov = [
        _h(0, "Chapter 1: Foundations of Algebra", level=1, raw=1),
        _meta(1, "Chapter 1 Foundations 7"),  # furniture — must NOT win
        _h(2, "1.1 Whole Numbers", raw=2),
        _p(3, "Counting numbers.", raw=3),
    ]
    chapters = build_chapters_ir(prov)
    assert "Chapter 1: Foundations of Algebra" in [c.title for c in chapters]


def test_e_l1_openers_present_keeps_legacy_path():
    """When real L1 chapter openers ARE present in content (each with content),
    the legacy boundary path is kept — the synthetic two-chapter fixture must
    not regress onto the section-number path."""
    chapters = build_chapters_ir(_SyntheticPipelineResult())
    # Same as test_a — legacy path, two L1-opener chapters.
    assert [c.title for c in chapters] == [
        "Chapter 1: Foundations",
        "Chapter 2: Linear Equations",
    ]


def test_e_leading_frontmatter_only_chapter_dropped():
    """A leading run of content-free (heading / metadata_drop only) regions
    before the first N.M section does NOT become a phantom chapter."""
    meta = {
        "region_index": 0,
        "region_kind": "metadata_drop",
        "role": "metadata_drop",
        "confidence": None,
        "wcag_status": "failed",
        "first_raw_block_index": 1,
        "pages": [1],
        "heading_text": None,
        "level": None,
        "figure_alt": None,
        "raw_text": "Running header noise.",
    }
    prov = [
        meta,
        _h(1, "Preface", level=2, raw=2),  # content-free heading
        _h(2, "1.1 Whole Numbers", raw=3),
        _p(3, "Counting numbers.", raw=4),
        _h(4, "2.1 Solve Equations", raw=5),
        _p(5, "Isolate x.", raw=6),
    ]
    chapters = build_chapters_ir(prov)
    titles = [c.title for c in chapters]
    # No leading "Document" / "Preface" content-free chapter.
    assert all("Document" not in t for t in titles), titles
    assert titles == ["Chapter 1", "Chapter 2"], titles


# ---------------------------------------------------------------------------
# (f) Deterministic per-kind inner-HTML rendering (lists / figures / prose).
# ---------------------------------------------------------------------------


def test_f_list_region_renders_ul():
    """A council ``list``-kind region with >=2 bullet markers renders <ul>."""
    from lib.semantik.cascade_ir import _block_html_for_kind

    html = _block_html_for_kind(
        "list", "• Add integers. • Subtract integers. • Multiply integers.", None
    )
    assert html.startswith("<ul>") and html.endswith("</ul>")
    assert html.count("<li>") == 3
    assert "<li>Add integers.</li>" in html


def test_f_noisy_list_region_falls_back_to_paragraph():
    """A ``list``-typed region with <2 bullet markers (council mis-type) is
    rendered as a <p>, NOT a one-item pseudo-list."""
    from lib.semantik.cascade_ir import _block_html_for_kind

    # No bullet glyphs at all (a mis-typed heading / single sentence).
    html = _block_html_for_kind("list", "Match primes vertically when possible.", None)
    assert html == "<p>Match primes vertically when possible.</p>"
    # A single leading bullet only → still one item → fall back to <p>.
    html2 = _block_html_for_kind("list", "• Only one bullet here.", None)
    assert html2.startswith("<p>")


def test_f_figure_region_renders_text_only_figure():
    """A ``figure``-kind region renders <figure><figcaption> with the alt
    caption and NEVER a broken <img src="">."""
    from lib.semantik.cascade_ir import _block_html_for_kind

    html = _block_html_for_kind(
        "figure", "Figure 1.1 A number line.", "A number line from -5 to 5."
    )
    assert html.startswith("<figure>") and html.endswith("</figure>")
    assert "<figcaption>A number line from -5 to 5.</figcaption>" in html
    assert "<img" not in html


def test_f_paragraph_and_table_render_as_paragraph():
    """Prose renders <p>; a ``table``-kind region (no upstream typed table on
    the real corpus) degrades to <p> (documented upstream-council limit)."""
    from lib.semantik.cascade_ir import _block_html_for_kind

    assert _block_html_for_kind("paragraph", "Plain prose.", None) == "<p>Plain prose.</p>"
    assert _block_html_for_kind("table", "Col A Col B 1 2", None) == "<p>Col A Col B 1 2</p>"


def test_f_list_items_are_escaped():
    """List item text is HTML-escaped (no raw < > & injection)."""
    from lib.semantik.cascade_ir import _block_html_for_kind

    html = _block_html_for_kind("list", "• a < b • c & d", None)
    assert "&lt;" in html and "&amp;" in html
    assert "<li>a &lt; b</li>" in html


# ---------------------------------------------------------------------------
# Defect 2 — the "(cont.)" heading cascade must not accumulate.
# ---------------------------------------------------------------------------


def test_overflow_continuation_title_does_not_accumulate_cont():
    """A chapter that overflows the per-chapter block budget several times spills
    into continuation chapters whose title reads 'Title (cont.)' EVERY time —
    never the accumulating 'Title (cont.) (cont.) (cont.)' cascade (defect 2:
    31 EA2e headings carried >=1 '(cont.)', one carried four).

    Uses the LEGACY boundary path (a single real 'Chapter N' L1 opener + a long
    run of headingless prose, no N.M sections → no section-number derivation),
    driving the _MAX_BLOCKS_PER_CHAPTER overflow guard repeatedly.
    """
    from lib.semantik.cascade_ir import _MAX_BLOCKS_PER_CHAPTER

    prov = [_h(0, "Chapter 9 Higher Roots", level=1, raw=0)]
    # 3x the per-chapter budget of headingless prose → >=2 overflow spills.
    for i in range(_MAX_BLOCKS_PER_CHAPTER * 3):
        prov.append(_p(i + 1, f"Prose body sentence number {i}.", raw=i + 1))

    chapters = build_chapters_ir(prov)
    titles = [c.title for c in chapters]
    # The overflow fired (more than one chapter from a single opener).
    assert len(titles) >= 3, titles
    # No title ever carries a doubled "(cont.) (cont.)" — the cascade signature.
    assert all("(cont.) (cont.)" not in t for t in titles), titles
    # Every continuation title has AT MOST one "(cont.)" suffix.
    for t in titles:
        assert t.count("(cont.)") <= 1, t
    # The base title is preserved on the continuations.
    assert all(t.startswith("Chapter 9 Higher Roots") for t in titles), titles


# ---------------------------------------------------------------------------
# Overflow continuations must be FLAGGED so the adapter renders them WITHOUT a
# visible heading (the 8x "Chapter Outline (cont.)" phantom-section defect).
# ---------------------------------------------------------------------------


def test_overflow_spill_flags_continuation_chapters():
    """The FIRST chapter from an opener is a real chapter (continuation=False);
    every overflow SPILL is flagged continuation=True so the adapter can demote
    its heading out of the HTML heading stream (downstream structure extraction
    reads each heading element as a new section)."""
    from lib.semantik.cascade_ir import _MAX_BLOCKS_PER_CHAPTER

    prov = [_h(0, "Chapter 9 Higher Roots", level=1, raw=0)]
    for i in range(_MAX_BLOCKS_PER_CHAPTER * 3):
        prov.append(_p(i + 1, f"Prose body sentence number {i}.", raw=i + 1))

    chapters = build_chapters_ir(prov)
    assert len(chapters) >= 3, [c.title for c in chapters]
    # First chapter is NOT a continuation; all spills ARE continuations.
    assert chapters[0].continuation is False
    assert all(c.continuation for c in chapters[1:]), [
        (c.title, c.continuation) for c in chapters
    ]
    # Every continuation carries the "(cont.)" title (the visual-only cue).
    assert all(c.title.endswith("(cont.)") for c in chapters[1:])


def test_overflow_continuation_renders_without_heading_element():
    """End-to-end: an overflowing chapter emits exactly ONE <h2> for the real
    section title; the spills render as aria-hidden presentation <div>s (NOT
    heading elements), so the HTML heading stream carries the title once."""
    from lib.semantik.cascade_ir import _MAX_BLOCKS_PER_CHAPTER

    prov = [_h(0, "Chapter 9 Higher Roots", level=1, raw=0)]
    for i in range(_MAX_BLOCKS_PER_CHAPTER * 3):
        prov.append(_p(i + 1, f"Prose body sentence number {i}.", raw=i + 1))

    out = normalize_cascade_to_ed4all(prov_result(prov), pdf_stem="ea2e_ch9")
    html = out["html"]
    # The base title appears as a heading EXACTLY once; no "(cont.)" heading.
    assert len(re.findall(r"<h2>Chapter 9 Higher Roots</h2>", html)) == 1
    assert "(cont.)</h2>" not in html
    assert re.search(r"<h2>[^<]*\(cont\.\)", html) is None
    # The spills survive as aria-hidden presentation divs (content preserved).
    assert 'class="semantik-continuation"' in html
    assert 'aria-hidden="true"' in html
    assert "Chapter 9 Higher Roots (cont.)" in html  # inside the div, not an h2
    # No content was dropped: every prose sentence is still present.
    assert "Prose body sentence number 0." in html
    assert f"Prose body sentence number {_MAX_BLOCKS_PER_CHAPTER * 3 - 1}." in html


def test_overflow_continuation_blocks_still_in_sidecar():
    """Sidecar consistency: continuation-chapter blocks are real content and
    MUST still appear as sidecar sections (block-id parity for the source_refs
    gate). The continuation FLAG only suppresses the visible chapter HEADING, it
    never drops the blocks. build_synthesized_sidecar emits per-BLOCK sections
    (never chapter titles), so there is no continuation title to dedupe there."""
    from lib.semantik.cascade_ir import _MAX_BLOCKS_PER_CHAPTER

    n = _MAX_BLOCKS_PER_CHAPTER * 3
    prov = [_h(0, "Chapter 9 Higher Roots", level=1, raw=0)]
    for i in range(n):
        prov.append(_p(i + 1, f"Prose body sentence number {i}.", raw=i + 1))

    out = normalize_cascade_to_ed4all(prov_result(prov), pdf_stem="ea2e_ch9")
    sidecar = out["synthesized_sidecar"]
    # All n prose blocks are present in the sidecar (none lost to the spill).
    assert len(sidecar["sections"]) == n
    # The chapter title never leaks in as a section (only per-block titles).
    titles = {s["section_title"] for s in sidecar["sections"]}
    assert not any("(cont.)" in t for t in titles), titles


# ---------------------------------------------------------------------------
# D2: caption-less figure placeholder is screen-reader-only (no chunk-text leak)
# ---------------------------------------------------------------------------


def test_captionless_figure_placeholder_is_sr_only_by_default(monkeypatch):
    """A caption-less figure ships its "Figure." placeholder inside an sr-only
    <span> so Trainforge's HTMLTextExtractor discards it from chunk text while
    the accessible name stays in the a11y tree."""
    monkeypatch.delenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", raising=False)
    from lib.semantik.cascade_ir import _render_figure_html, _TYPE_LEVEL_ALT

    html = _render_figure_html("", None, image_src=None, caption_text=None)
    assert '<figcaption><span class="sr-only">Figure.</span></figcaption>' in html
    assert _TYPE_LEVEL_ALT == "Figure."


def test_captionless_figure_placeholder_flag_off_is_visible(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", "0")
    from lib.semantik.cascade_ir import _render_figure_html

    html = _render_figure_html("", None, image_src=None, caption_text=None)
    assert html == "<figure><figcaption>Figure.</figcaption></figure>"  # byte-identical legacy


def test_real_caption_stays_visible_figcaption(monkeypatch):
    """A resolved caption is NEVER sr-only'd — it renders as a visible
    <figcaption> regardless of the flag."""
    from lib.semantik.cascade_ir import _render_figure_html

    for val in ("1", "0"):
        monkeypatch.setenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", val)
        html = _render_figure_html("", None, image_src=None,
                                   caption_text="A real widget diagram")
        assert "<figcaption>A real widget diagram</figcaption>" in html
        assert "sr-only" not in html
