"""P2a tests for the SemantiK v2 → Ed4All output-contract adapter.

Built + run with NO models / GPU. Exercises the adapter against a SYNTHETIC
cascade-result fixture and the REAL Ed4All contract validators
(``SemantiKMarkersValidator``, ``source_refs`` helpers, ``SemanticStructureExtractor``).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest lib/semantik/tests/test_adapter.py -q

The synthetic fixture (``_make_cascade_result``) is built FROM SCRATCH — no
vendored Semantic test fixture exists under ``SemantiK/`` (P0 vendored code
only). It mirrors the ``types.py`` region/feature-block shapes: each block
anchors to a RAW extracted-block document-order index (§3.3a), carries page
numbers, and the doc has ≥2 chapters + a figure + each of the three
``exit_action`` states.
"""

from __future__ import annotations

import re

import pytest

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _mint_sid,
    normalize_cascade_to_ed4all,
)


# ---------------------------------------------------------------------------
# Synthetic cascade-result fixture (built from scratch — §test item 4).
# ---------------------------------------------------------------------------


class _SyntheticCascadeResult:
    """Duck-typed stand-in for ``PipelineV2Result`` + the normalized IR.

    The real ``PipelineV2Result`` does NOT carry regions/feature_blocks in
    its serialized dict; the P3 seam attaches the normalized ``chapters`` IR
    derived from ``regions[i].feature_block_indices`` → ``feature_blocks[i].raw``.
    This synthetic result attaches the IR directly so P2a can validate the
    adapter without loading the cascade.
    """

    def __init__(self, *, exit_action: str):
        self.pdf = "sample_text_ch1.pdf"
        self.wcag_status = "passed"
        self.exit_action = exit_action
        self.theta_score = 0.91
        self.flags = [] if exit_action == "ship_with_confidence" else ["theta_low"]
        self.lane_used = "fast-lane"
        self.lang = "en"
        # ≥2 chapters, page numbers, a figure (§test item: fixture shape).
        # MODELS THE REAL CASCADE: heading regions carry region_kind="heading"
        # + heading_text; content (paragraph/list/figure) regions carry
        # heading_text=None (see cascade._build_region_provenance: heading_text
        # is set ONLY for heading/figure kinds). A content block therefore gets
        # a visually-hidden aria-labelledby label, NOT a fabricated visible h3.
        self.chapters = [
            _AdapterChapter(
                title="Chapter 1: Foundations",
                blocks=[
                    # A GENUINE heading region → visible <h3>.
                    _AdapterBlock(
                        html="",
                        region_kind="heading",
                        raw_block_index=0,
                        raw_text="Introduction to Algebra",
                        heading_text="Introduction to Algebra",
                        pages=[1],
                        confidence=0.83,
                        block_role="heading",
                        wcag_status="passed",
                    ),
                    # A content paragraph (no real heading) → sr-only label.
                    _AdapterBlock(
                        html="<p>Algebra builds on arithmetic.</p>",
                        region_kind="paragraph",
                        raw_block_index=1,
                        raw_text="Algebra builds on arithmetic.",
                        heading_text=None,
                        pages=[1],
                        confidence=0.83,
                        block_role="body",
                        wcag_status="passed",
                    ),
                    # A genuine bullet list → <ul><li> (deterministic render).
                    # The IR builder (cascade_ir._block_from_provenance) renders
                    # the <ul> from raw_text; the adapter emits block.html
                    # verbatim, so the fixture carries the rendered <ul>.
                    _AdapterBlock(
                        html=(
                            "<ul><li>Add integers.</li><li>Subtract integers."
                            "</li><li>Multiply integers.</li></ul>"
                        ),
                        region_kind="list",
                        raw_block_index=2,
                        raw_text="• Add integers. • Subtract integers. • Multiply integers.",
                        heading_text=None,
                        pages=[2],
                        confidence=0.7,
                        block_role="list",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html="<p>The order of operations is PEMDAS.</p>",
                        region_kind="paragraph",
                        raw_block_index=3,
                        raw_text="The order of operations is PEMDAS.",
                        heading_text=None,
                        pages=[2, 3],
                        confidence=0.61,
                        block_role="body",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html=(
                            "<figure><figcaption>A number line from -5 to 5."
                            "</figcaption></figure>"
                        ),
                        region_kind="figure",
                        raw_block_index=5,
                        raw_text="Figure 1.1 A number line from -5 to 5.",
                        heading_text="Number Line",
                        pages=[3],
                        confidence=1.0,  # bands to omitted
                        block_role="figure",
                        wcag_status="passed",
                        figure_alt="A number line from -5 to 5.",
                    ),
                    # A non-content (answer-key) heading that MUST be filtered
                    # so the chapter stays under the >40-section collapse.
                    _AdapterBlock(
                        html="",
                        region_kind="heading",
                        raw_block_index=6,
                        raw_text="78 41. 900 42. 800 43. 700",
                        heading_text="78 41. 900 42. 800 43. 700",
                        pages=[4],
                        confidence=0.4,
                    ),
                ],
            ),
            _AdapterChapter(
                title="Chapter 2: Linear Equations",
                blocks=[
                    _AdapterBlock(
                        html="",
                        region_kind="heading",
                        raw_block_index=10,
                        raw_text="Solving Linear Equations",
                        heading_text="Solving Linear Equations",
                        pages=[5],
                        confidence=0.79,
                        block_role="heading",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html="<p>Solve for x in 2x + 3 = 7.</p>",
                        region_kind="paragraph",
                        raw_block_index=11,
                        raw_text="Solve for x in 2x + 3 = 7.",
                        heading_text=None,
                        pages=[5],
                        confidence=0.79,
                        block_role="body",
                        wcag_status="passed",
                    ),
                    _AdapterBlock(
                        html="<p>A headingless prose block.</p>",
                        region_kind="paragraph",
                        raw_block_index=12,
                        raw_text="A headingless prose block continues here.",
                        heading_text=None,  # positional-fallback sid
                        pages=[6],
                        confidence=0.6,
                    ),
                ],
            ),
        ]


def _make_cascade_result(exit_action: str = "ship_with_confidence"):
    return _SyntheticCascadeResult(exit_action=exit_action)


@pytest.fixture
def result_html_and_sidecar():
    res = _make_cascade_result()
    out = normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")
    return out


# ---------------------------------------------------------------------------
# (a) §3.0 single-sid INVARIANT.
# ---------------------------------------------------------------------------


def test_a_single_sid_invariant(result_html_and_sidecar):
    """The sid anchors ``#{sid}`` == data-semantik-block-id == sidecar section_id,
    all from ONE mint fn. B4 (2026-07-04): a genuine HEADING block names its
    section via ``aria-labelledby`` → a VISIBLE ``<hN id={sid}>`` (descriptive
    region landmark). A CONTENT block carries the sid as a bare ``id`` on the
    ``<section>`` (so ``#{sid}`` deep-links) with NO aria-labelledby and NO
    sr-only label — so it is not a generic "Paragraph block" region."""
    out = result_html_and_sidecar
    html = out["html"]

    sections = re.findall(r"<section\b[^>]*>.*?</section>", html, re.DOTALL)
    assert sections, "no <section> wrappers emitted"

    sidecar_ids = {s["section_id"] for s in out["synthesized_sidecar"]["sections"]}
    assert sidecar_ids, "sidecar carried no sections"

    # B4: no section carries an sr-only "… block" label anymore.
    assert 'class="sr-only"' not in html

    seen_block_ids = set()
    for sec in sections:
        block_id = re.search(r'data-semantik-block-id="([^"]+)"', sec).group(1)
        aria_m = re.search(r'aria-labelledby="([^"]+)"', sec)
        if aria_m:
            # Heading block: aria-labelledby target is a VISIBLE <hN id={sid}>.
            aria = aria_m.group(1)
            label_match = re.search(r"<(h[2-6])\b[^>]*\bid=\"([^\"]+)\"[^>]*>", sec)
            assert label_match, f"no heading target in section: {sec[:120]}"
            assert aria == block_id == label_match.group(2), (
                f"sid divergence: aria={aria} block_id={block_id} "
                f"label_id={label_match.group(2)}"
            )
        else:
            # Content block: the <section> carries id={sid} directly.
            sec_open = sec[: sec.index(">") + 1]
            id_m = re.search(r'\bid="([^"]+)"', sec_open)
            assert id_m and id_m.group(1) == block_id, (
                f"content section missing id={block_id}: {sec_open}"
            )
        assert block_id in sidecar_ids, f"{block_id} not in sidecar id universe"
        seen_block_ids.add(block_id)

    # Every sidecar id appears in the HTML (full parity both directions).
    assert seen_block_ids == sidecar_ids

    # Heading-text-derived sids equal heading_slug(text) exactly (heading block).
    from gui.services.source_page import heading_slug

    res = _make_cascade_result()
    heading_block = res.chapters[0].blocks[0]  # the genuine "heading" region
    assert _mint_sid(heading_block) == heading_slug(heading_block.heading_text)
    # The #fragment for that sid is just #{sid}.
    assert f'href="#main-content"' in html  # skip-link fragment kept


def test_a_content_hash_mode_parity(monkeypatch):
    """§3.3 content-hash mode: HTML block_ids == sidecar section_ids, both
    16-hex hashes of the deterministic raw text."""
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "1")
    out = normalize_cascade_to_ed4all(
        _make_cascade_result(), pdf_stem="sample_text_ch1"
    )
    html = out["html"]
    html_ids = set(re.findall(r'data-semantik-block-id="([^"]+)"', html))
    sidecar_ids = {
        s["section_id"] for s in out["synthesized_sidecar"]["sections"]
    }
    assert html_ids == sidecar_ids
    assert all(re.fullmatch(r"[0-9a-f]{16}", i) for i in html_ids)


# ---------------------------------------------------------------------------
# (b) SemantiKMarkersValidator passes on adapter HTML.
# ---------------------------------------------------------------------------


def test_b_semantik_markers_validator_passes(result_html_and_sidecar):
    from lib.validators.semantik_markers import SemantiKMarkersValidator

    out = result_html_and_sidecar
    res = SemantiKMarkersValidator().validate({"html_content": out["html"]})
    critical = [i for i in res.issues if i.severity == "critical"]
    assert res.passed, f"semantik_markers failed: {[i.code for i in critical]}"
    assert not critical
    # No empty data-semantik-source anywhere.
    assert 'data-semantik-source=""' not in out["html"]
    assert "data-semantik-source=\"synthesized\"" in out["html"]


# ---------------------------------------------------------------------------
# (c) sourceId regex + source_refs parity.
# ---------------------------------------------------------------------------


def test_c_source_ids_match_regex_and_resolve(result_html_and_sidecar):
    from lib.validators.source_refs import SOURCE_ID_RE

    out = result_html_and_sidecar
    slug = out["slug"]
    sidecar_ids = {
        s["section_id"] for s in out["synthesized_sidecar"]["sections"]
    }
    # Every emitted block_id forms a valid sourceId AND resolves against the
    # sidecar id universe.
    html_ids = set(re.findall(r'data-semantik-block-id="([^"]+)"', out["html"]))
    assert html_ids
    for bid in html_ids:
        source_id = f"semantik:{slug}#{bid}"
        assert SOURCE_ID_RE.match(source_id), f"bad sourceId: {source_id}"
        assert bid in sidecar_ids, f"{bid} unresolved against sidecar"

    # Slug is the gentle transform (underscores preserved).
    assert slug == "sample_text_ch1"


# ---------------------------------------------------------------------------
# (d) SemanticStructureExtractor finds >=2 chapters, no >40-section collapse.
# ---------------------------------------------------------------------------


def test_d_structure_extractor_two_chapters_no_collapse(result_html_and_sidecar):
    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        SemanticStructureExtractor,
        _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
    )

    out = result_html_and_sidecar
    extractor = SemanticStructureExtractor()
    structure = extractor.extract(out["html"])
    chapters = structure["chapters"]
    assert len(chapters) >= 2, f"expected >=2 chapters, got {len(chapters)}"
    for ch in chapters:
        secs = ch.get("sections", [])
        assert len(secs) <= _STRUCTURE_COLLAPSE_SECTION_THRESHOLD


# ---------------------------------------------------------------------------
# (e) exit_action → success / certification_status mapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exit_action,expected_success,expected_cert",
    [
        ("ship_with_confidence", True, "certified"),
        ("ship_with_flag", True, "flagged"),
        ("non_certified_stamp", True, "non_certified"),
        ("hard_error", False, "error"),
        (None, False, "error"),
    ],
)
def test_e_exit_action_mapping(exit_action, expected_success, expected_cert):
    res = _make_cascade_result(exit_action=exit_action or "ship_with_confidence")
    res.exit_action = exit_action  # overwrite (None case)
    out = normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")
    assert out["success"] is expected_success
    assert out["certification_status"] == expected_cert
    assert out["method"] == "semantik_v2"


def test_required_output_keys_present(result_html_and_sidecar):
    """The §3.7 / config/workflows.yaml:898 required keys are surfaced."""
    out = result_html_and_sidecar
    for key in ("html", "html_length", "success", "method", "word_count"):
        assert key in out
    assert out["html_length"] == len(out["html"])
    assert out["word_count"] > 0


def test_no_fabricated_heading_no_body_duplication(result_html_and_sidecar):
    """The §1.1 defect: content blocks got a visible <h3> fabricated from
    their first body line, duplicating the prose. After the fix, a content
    block's aria-labelledby target is a visually-hidden <p class="sr-only">
    and the body text appears EXACTLY once (in the body, never echoed)."""
    out = result_html_and_sidecar
    html = out["html"]
    sections = re.findall(r"<section\b[^>]*>.*?</section>", html, re.DOTALL)

    for sec in sections:
        h3 = re.search(r"<h3[^>]*>(.*?)</h3>", sec, re.DOTALL)
        p = re.search(r"<p\b(?![^>]*class=\"sr-only\")[^>]*>(.*?)</p>", sec, re.DOTALL)
        if h3 and p:
            ht = h3.group(1).strip()
            pt = p.group(1).strip()
            # A visible <h3> must NOT duplicate the body paragraph's opening.
            assert not (ht and pt and pt.startswith(ht[:20])), (
                f"heading duplicates body: h3={ht[:40]!r} p={pt[:40]!r}"
            )

    # B4: content sections carry NO sr-only "Paragraph block" label anymore;
    # they anchor via a bare id on the <section> instead.
    assert 'class="sr-only"' not in html
    assert "Paragraph block" not in html
    # A genuine heading region still renders a VISIBLE heading.
    assert "<h3 id=" in html
    assert "Introduction to Algebra" in html


def test_genuine_bullet_list_renders_ul(result_html_and_sidecar):
    """A council ``list``-kind region with ≥2 bullet markers renders as a
    semantic <ul><li> (not a flattened <p>); items split from the raw text."""
    out = result_html_and_sidecar
    html = out["html"]
    assert "<ul>" in html and "</ul>" in html
    assert "<li>Add integers.</li>" in html
    assert "<li>Subtract integers.</li>" in html
    assert "<li>Multiply integers.</li>" in html


def test_document_h1_present_and_nesting(result_html_and_sidecar):
    """WCAG 2.4.6 — exactly one document <h1> at the top of <main>, with the
    title text; chapters stay <h2>, sections <h3> (h1 > h2 > h3 nesting)."""
    out = result_html_and_sidecar
    html = out["html"]
    h1s = re.findall(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
    assert len(h1s) == 1, f"expected exactly one <h1>, got {len(h1s)}"
    # The <h1> is the document title (same source as <title>).
    title = re.search(r"<title>(.*?)</title>", html).group(1)
    assert h1s[0].strip() == title.strip()
    # It sits at the top of <main>, before the first <article>/<h2>.
    main_idx = html.index('<main ')
    h1_idx = html.index("<h1")
    h2_idx = html.index("<h2")
    assert main_idx < h1_idx < h2_idx
    # Chapters are still <h2>, sections still <h3> (additive, not renumbered).
    assert "<h2>" in html and "<h3 id=" in html


def test_figure_and_page_provenance(result_html_and_sidecar):
    """Pages stamped as physical; figure_alt carried into the sidecar."""
    out = result_html_and_sidecar
    html = out["html"]
    assert 'data-semantik-page-kind="physical"' in html
    assert 'data-semantik-pages="2-3"' in html  # contiguous range collapse
    # 1.0-confidence figure omits data-semantik-confidence (§3.6).
    fig_sec = re.search(
        r'<section[^>]*data-semantik-block-role="figure"[^>]*>', html
    )
    assert fig_sec and "data-semantik-confidence" not in fig_sec.group(0)
    # figure_alt rides along into the sidecar.
    alts = [
        s["data"].get("figure_alt")
        for s in out["synthesized_sidecar"]["sections"]
        if s["data"].get("figure_alt")
    ]
    assert "A number line from -5 to 5." in alts


# ---------------------------------------------------------------------------
# Defect fixes — furniture-emission suppression + honest <h1> selection.
# ---------------------------------------------------------------------------


def test_metadata_drop_furniture_not_emitted_to_body():
    """A metadata_drop-role block (running header / footer / copyright line) is
    page furniture — its body text must NOT leak into the emitted HTML nor the
    synthesized sidecar (defect: SemantiK furniture-emission — metadata_drop
    bodies leaked into <p>s on an OCR scan corpus; a footer and running
    header repeated many times per chapter). Mirrors the assembler pass_9a
    metadata_drop drop; filtered from BOTH surfaces so block-id parity
    (source_refs gate) holds."""
    footer = "This sample book is available for free at http://example.org/x"
    running = "Chapter 9 Roots and Radicals 1035"
    result = type("R", (), {})()
    result.exit_action = "ship_with_confidence"
    result.wcag_status = "passed"
    result.lang = "en"
    result.chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _AdapterBlock(
                    html="<p>The square root of 225 is 15.</p>",
                    region_kind="paragraph",
                    raw_block_index=0,
                    raw_text="The square root of 225 is 15.",
                    pages=[2],
                    block_role="paragraph",
                ),
                _AdapterBlock(
                    html=f"<p>{footer}</p>",
                    region_kind="metadata_drop",
                    raw_block_index=1,
                    raw_text=footer,
                    pages=[2],
                    block_role="metadata_drop",
                ),
                _AdapterBlock(
                    html=f"<p>{running}</p>",
                    region_kind="metadata_drop",
                    raw_block_index=2,
                    raw_text=running,
                    pages=[3],
                    block_role="metadata_drop",
                ),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(result, pdf_stem="algebra_ch9")
    html = out["html"]
    # Real content survives; furniture bodies are gone.
    assert "The square root of 225 is 15." in html
    assert footer not in html
    assert "example.org" not in html
    assert running not in html
    assert 'data-semantik-block-role="metadata_drop"' not in html
    # Sidecar parity: no metadata_drop section survives either.
    sidecar_texts = [s["data"]["text"] for s in out["synthesized_sidecar"]["sections"]]
    assert footer not in sidecar_texts
    assert running not in sidecar_texts


def test_h1_prefers_chapter_n_over_frontmatter_outline():
    """The document <h1>/<title> prefers a real 'Chapter N <Title>' opener over
    a leading front-matter 'Chapter Outline' artifact (defect 3d — the h1 was
    minted from 'Chapter Outline' instead of the chapter title)."""
    result = type("R", (), {})()
    result.exit_action = "ship_with_confidence"
    result.wcag_status = "passed"
    result.lang = "en"
    result.chapters = [
        _AdapterChapter(
            title="Chapter Outline",
            blocks=[
                _AdapterBlock(
                    html="<p>Outline front matter.</p>",
                    region_kind="paragraph",
                    raw_block_index=0,
                    raw_text="Outline front matter.",
                    pages=[1],
                )
            ],
        ),
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _AdapterBlock(
                    html="<p>Radicals content.</p>",
                    region_kind="paragraph",
                    raw_block_index=1,
                    raw_text="Radicals content.",
                    pages=[2],
                )
            ],
        ),
    ]
    out = normalize_cascade_to_ed4all(result, pdf_stem="algebra_ch9")
    assert "<h1>Chapter 9 Roots and Radicals</h1>" in out["html"]
    assert "<title>Chapter 9 Roots and Radicals</title>" in out["html"]
    assert "<h1>Chapter Outline</h1>" not in out["html"]


def _one_block_chapter(title: str, idx: int) -> _AdapterChapter:
    return _AdapterChapter(
        title=title,
        blocks=[
            _AdapterBlock(
                html=f"<p>Body for {title}.</p>",
                region_kind="paragraph",
                raw_block_index=idx,
                raw_text=f"Body for {title}.",
                pages=[idx + 1],
            )
        ],
    )


def test_h1_excludes_chapter_n_review_banner():
    """The end-of-chapter 'CHAPTER 9 REVIEW' banner matches the 'Chapter N …'
    prefix but is apparatus, NOT the chapter title — it must NOT become the
    document <h1> (defect: h1 minted 'CHAPTER 9 REVIEW'). A real
    'Chapter 9 Roots and Radicals' opener wins even when the review banner is
    the FIRST 'Chapter N' candidate in document order."""
    result = type("R", (), {})()
    result.exit_action = "ship_with_confidence"
    result.wcag_status = "passed"
    result.lang = "en"
    result.chapters = [
        _one_block_chapter("Chapter Outline", 0),
        _one_block_chapter("CHAPTER 9 REVIEW", 1),
        _one_block_chapter("Chapter 9 Roots and Radicals", 2),
    ]
    out = normalize_cascade_to_ed4all(result, pdf_stem="algebra_ch9")
    assert "<h1>Chapter 9 Roots and Radicals</h1>" in out["html"]
    assert "CHAPTER 9 REVIEW" not in re.findall(r"<h1>(.*?)</h1>", out["html"])[0]


@pytest.mark.parametrize(
    "apparatus",
    [
        "Chapter 9 Review",
        "Chapter 9 Outline",
        "Chapter 9 Exercises",
        "Chapter 9 Exercise",
        "Chapter 9 Key Terms",
        "Chapter 9 Key Term",
        "Chapter 9 Practice Test",
    ],
)
def test_h1_excludes_apparatus_suffixes(apparatus: str):
    """review / outline / exercise(s) / key term(s) / practice test are all
    end-of-chapter apparatus qualifiers that disqualify a 'Chapter N …' heading
    from being the document title (case-insensitive)."""
    result = type("R", (), {})()
    result.exit_action = "ship_with_confidence"
    result.wcag_status = "passed"
    result.lang = "en"
    result.chapters = [
        _one_block_chapter(apparatus, 0),
        _one_block_chapter("Chapter 9 Roots and Radicals", 1),
    ]
    out = normalize_cascade_to_ed4all(result, pdf_stem="algebra_ch9")
    assert "<h1>Chapter 9 Roots and Radicals</h1>" in out["html"]


def test_h1_falls_back_when_no_qualifying_chapter_n():
    """When NO 'Chapter N <RealTitle>' opener qualifies (all candidates are
    apparatus / front-matter), fall back to the first chapter title — never
    fabricate, never pick the excluded review banner."""
    result = type("R", (), {})()
    result.exit_action = "ship_with_confidence"
    result.wcag_status = "passed"
    result.lang = "en"
    result.chapters = [
        _one_block_chapter("Chapter Outline", 0),
        _one_block_chapter("CHAPTER 9 REVIEW", 1),
    ]
    out = normalize_cascade_to_ed4all(result, pdf_stem="algebra_ch9")
    h1 = re.findall(r"<h1>(.*?)</h1>", out["html"])[0]
    assert h1 == "Chapter Outline"


def test_continuation_chapter_renders_no_heading_element():
    """A continuation chapter (§3.4 overflow spill) renders its title as an
    aria-hidden presentation <div>, NOT a heading element — so a repeated
    overflow does not mint phantom pseudo-section headings the structure
    extractor reads as new sections. A NON-continuation chapter still gets a
    visible <h2>."""
    result = type("R", (), {})()
    result.exit_action = "ship_with_confidence"
    result.wcag_status = "passed"
    result.lang = "en"
    real = _one_block_chapter("Higher Roots", 0)
    cont = _AdapterChapter(
        title="Higher Roots (cont.)",
        blocks=[
            _AdapterBlock(
                html="<p>More roots.</p>",
                region_kind="paragraph",
                raw_block_index=1,
                raw_text="More roots.",
                pages=[2],
            )
        ],
        continuation=True,
    )
    result.chapters = [real, cont]
    out = normalize_cascade_to_ed4all(result, pdf_stem="algebra_ch9")
    html = out["html"]
    # The real chapter's title is a visible <h2>; the continuation is NOT.
    assert "<h2>Higher Roots</h2>" in html
    assert "<h2>Higher Roots (cont.)</h2>" not in html
    assert re.search(r"<h2>[^<]*\(cont\.\)", html) is None
    # The continuation title survives as an aria-hidden presentation div.
    assert (
        '<div class="semantik-continuation" role="presentation" '
        'aria-hidden="true">Higher Roots (cont.)</div>' in html
    )
    # No content lost.
    assert "More roots." in html
