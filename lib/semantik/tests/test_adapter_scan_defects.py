"""Regression: adapter-seam scan-conversion defects (2026-07-03 VLM audit).

Defect 3 — obviously-fused headings refused (demoted to prose / continuation).
Defect 4 — end-of-chapter apparatus paragraphs promoted back to headings.
Defect 5 — colliding ``data-dart-block-id`` slugs uniquified at minting.

All synthetic IR built inline — no course-data path, no model, no cascade run.
"""

from __future__ import annotations

import re

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    build_synthesized_sidecar,
    normalize_cascade_to_ed4all,
)
from lib.semantik.heading_classifier import (
    APPARATUS_HEADING_NAMES,
    is_apparatus_heading,
    is_fused_heading,
    split_leading_apparatus_heading,
)


class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "x"
        self.lang = "en"


def _heading_block(text, idx):
    return _AdapterBlock(
        html="",
        region_kind="heading",
        raw_block_index=idx,
        raw_text=text,
        heading_text=text,
        pages=[idx + 1],
    )


def _para_block(text, idx, html=None):
    return _AdapterBlock(
        html=html if html is not None else f"<p>{text}</p>",
        region_kind="paragraph",
        raw_block_index=idx,
        raw_text=text,
        heading_text=None,
        pages=[idx + 1],
    )


# ---------------------------------------------------------------------------
# heading_classifier predicates.
# ---------------------------------------------------------------------------


def test_is_apparatus_heading_named_set():
    assert set(APPARATUS_HEADING_NAMES)  # non-empty named constant
    assert is_apparatus_heading("PRACTICE TEST")
    assert is_apparatus_heading("Review Exercises")
    assert is_apparatus_heading(") Key Terms")  # OCR gutter glyph tolerated
    assert not is_apparatus_heading("Practice Test on square roots today please")


def test_is_fused_heading():
    assert is_fused_heading(r"Chapter 9 $\sqrt{a}$ Denise wants $\sqrt{b}$ tiles")
    assert is_fused_heading(
        "This heading swallowed an entire sentence of real body prose text here."
    )
    assert not is_fused_heading("9.3 Add and Subtract Square Roots")
    assert not is_fused_heading("Chapter 9 Roots and Radicals")


# ---------------------------------------------------------------------------
# Defect 4 — apparatus promotion at the adapter seam.
# ---------------------------------------------------------------------------


def test_apparatus_paragraph_promoted_to_h3():
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("9.1 Simplify Square Roots", 0),
                _para_block("PRACTICE TEST", 1),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    # The apparatus paragraph now renders as a visible heading, not a <p> body.
    assert "<h3" in html and "PRACTICE TEST" in html
    assert "<p>PRACTICE TEST</p>" not in html


# ---------------------------------------------------------------------------
# Defect 3 — fused headings refused.
# ---------------------------------------------------------------------------


def test_fused_section_heading_demoted_to_paragraph():
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block(
                    r"Simplify $\sqrt{20}$ then $\sqrt{5}$ and add the results carefully",
                    0,
                ),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    # No <h3> minted from the fused blob; it renders as prose instead.
    assert "<h3" not in html
    assert "<p" in html


def test_fused_chapter_title_becomes_continuation_not_h2():
    fused_title = (
        r"Chapter 9 Roots and Radicals $\sqrt[4]{9c^8}$ Denise wants $\sqrt{81}$ tiles"
    )
    chapters = [
        _AdapterChapter(
            title=fused_title,
            blocks=[_para_block("Body content of the chapter.", 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    # The fused blob must not become a visible <h2> chapter heading.
    assert f"<h2>{fused_title}" not in html
    assert 'class="dart-continuation"' in html


# ---------------------------------------------------------------------------
# Defect 5 — duplicate block-id slug uniquification + HTML/sidecar parity.
# ---------------------------------------------------------------------------


def test_duplicate_heading_slugs_uniquified():
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("Radical Expressions", 0),
                _heading_block("Radical Expressions", 10),
                _heading_block("Radical Expressions", 20),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    ids = re.findall(r'data-dart-block-id="([^"]+)"', out["html"])
    assert len(ids) == len(set(ids)), f"duplicate block ids: {ids}"
    # Stable -2 / -3 suffix scheme (first keeps the bare base).
    assert ids == [
        "radical-expressions",
        "radical-expressions-2",
        "radical-expressions-3",
    ]


def test_html_sidecar_block_id_parity_under_collision():
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("Radical Expressions", 0),
                _heading_block("Radical Expressions", 5),
            ],
        )
    ]
    result = _Result(chapters)
    out = normalize_cascade_to_ed4all(result, pdf_stem="ea_ch9")
    html_ids = re.findall(r'data-dart-block-id="([^"]+)"', out["html"])
    sidecar = out["synthesized_sidecar"]
    sidecar_ids = [s["section_id"] for s in sidecar["sections"]]
    assert html_ids == sidecar_ids  # §3.3 parity holds under uniquification
    assert html_ids == ["radical-expressions", "radical-expressions-2"]


# ---------------------------------------------------------------------------
# Coordinator follow-up (Defect B) — heading-TYPED apparatus must not be EATEN.
# ---------------------------------------------------------------------------


def test_heading_typed_apparatus_not_eaten_by_noncontent_filter():
    """(ii) A heading-typed 'PRACTICE TEST' block previously vanished entirely
    (the `_is_noncontent_heading` emit filter classifies EOC apparatus names as
    answer-key noise and skipped the block at all three emit sites). It must
    survive as a visible heading."""
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("9.1 Simplify Square Roots", 0),
                _heading_block("PRACTICE TEST", 1),
                _para_block("In the following exercises, simplify.", 2),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    assert "PRACTICE TEST" in html
    assert '<h3 id="practice-test">PRACTICE TEST</h3>' in html
    # Sidecar parity: the block also exists in the valid-ID universe.
    sidecar_ids = [s["section_id"] for s in out["synthesized_sidecar"]["sections"]]
    assert "practice-test" in sidecar_ids


def test_normalized_latex_wrapped_apparatus_promoted():
    """(iii) After the fusion normalizes '\\section*{REVIEW EXERCISES}' to
    plain text, the downstream block (heading- OR paragraph-typed) must render
    as a heading, not vanish."""
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("REVIEW EXERCISES", 0),  # heading-typed arm
                _para_block("Review Exercises", 10),  # paragraph-typed arm
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    assert "REVIEW EXERCISES" in html
    ids = re.findall(r'<h3 id="([^"]+)"', html)
    assert "review-exercises" in ids
    assert "review-exercises-2" in ids  # both arms survive, ids uniquified


# ---------------------------------------------------------------------------
# Defect f — a leading apparatus heading fused into a paragraph is SPLIT.
# ---------------------------------------------------------------------------


_GLOSSARY_PROSE = (
    "index the number n written above the radical sign is called the index "
    "of the radical for every root we simplify in this section here."
)


def test_split_leading_apparatus_heading_predicate():
    # Fires: apparatus name (Title Case) + non-trivial prose.
    got = split_leading_apparatus_heading("Key Terms " + _GLOSSARY_PROSE)
    assert got is not None
    name, remainder = got
    assert name == "Key Terms"
    assert remainder.startswith("index the number")
    # ALL-CAPS form also fires.
    assert split_leading_apparatus_heading("KEY TERMS " + _GLOSSARY_PROSE) is not None
    # Does NOT fire: lowercase mid-sentence usage.
    assert split_leading_apparatus_heading("key terms are listed in the glossary here") is None
    # Does NOT fire: standalone name with no trailing prose (promoted whole).
    assert split_leading_apparatus_heading("Key Terms") is None
    # Does NOT fire: only a trivial remainder (< 3 words).
    assert split_leading_apparatus_heading("Key Terms and more") is None


def test_leading_apparatus_paragraph_split_into_heading_plus_remainder():
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("9.1 Simplify Square Roots", 0),
                _para_block("Key Terms " + _GLOSSARY_PROSE, 1),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    # An apparatus heading is minted; the glossary prose survives as a paragraph.
    assert '<h3 id="key-terms">Key Terms</h3>' in html
    assert "index the number n written above" in html
    # The fused "Key Terms <prose>" no longer renders as a single paragraph body.
    assert "<p>Key Terms index" not in html
    # Sidecar parity: both halves land in the valid-ID universe.
    sidecar_ids = [s["section_id"] for s in out["synthesized_sidecar"]["sections"]]
    assert "key-terms" in sidecar_ids


def test_leading_apparatus_split_preserves_provenance_on_both_halves():
    # The glossary paragraph is on physical page 2 (raw_block_index=1).
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("9.1 Simplify Square Roots", 0),  # page 1
                _para_block("Key Terms " + _GLOSSARY_PROSE, 1),  # page 2
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    # Both split halves inherit the source block's page provenance (page 2).
    assert html.count('data-dart-pages="2"') == 2
    # The unrelated heading keeps its own page-1 provenance.
    assert html.count('data-dart-pages="1"') == 1


def test_leading_apparatus_lowercase_midsentence_not_split():
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _para_block(
                    "key terms are listed in the glossary and defined for every reader here today.",
                    0,
                ),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    assert 'id="key-terms"' not in html
    assert "key terms are listed in the glossary" in html


def test_standalone_apparatus_paragraph_still_promotes():
    """The split must not regress the standalone-apparatus promotion path."""
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("9.1 Simplify Square Roots", 0),
                _para_block("Key Terms", 1),  # standalone, no trailing prose
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    assert html.count('<h3 id="key-terms">Key Terms</h3>') == 1
    assert "<p>Key Terms</p>" not in html


# ---------------------------------------------------------------------------
# Defect f, HEADING arm (ch09 live re-render) — the cascade typed the WHOLE
# fused "Key Terms <glossary>" blob as a heading region. The split must WIN
# over the is_fused_heading demotion for apparatus-LED fused headings.
# ---------------------------------------------------------------------------


def test_heading_typed_apparatus_fused_blob_split():
    """A region_kind='heading' block whose heading_text is 'Key Terms <glossary
    prose>' must produce an apparatus <h3> + a paragraph remainder — not the
    fused-heading demotion (which buried the apparatus name in prose)."""
    fused = (
        r"Key Terms index $\sqrt[n]{a}$ $n$ is called the index of the "
        "radical. like radicals are radical expressions with the same index "
        "and the same radicand."
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("9.1 Simplify Square Roots", 0),
                _heading_block(fused, 1),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    # The apparatus heading is recovered as a visible section heading …
    assert '<h3 id="key-terms">Key Terms</h3>' in html
    # … and the glossary prose survives as a paragraph, not a heading.
    assert "is called the index of the" in html
    assert "like radicals are radical expressions" in html
    assert "Key Terms index" not in html  # the fused blob no longer exists
    # Sidecar parity: both halves in the valid-ID universe.
    sidecar_ids = [s["section_id"] for s in out["synthesized_sidecar"]["sections"]]
    assert "key-terms" in sidecar_ids
    assert len(sidecar_ids) == 3  # 9.1 heading + apparatus heading + remainder


def test_standalone_apparatus_heading_still_plain_heading():
    """A legitimate short 'Key Terms' heading (standalone, no fused prose) is
    NOT split — it renders as a plain heading exactly once."""
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[_heading_block("Key Terms", 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    assert html.count('<h3 id="key-terms">Key Terms</h3>') == 1
    sidecar_ids = [s["section_id"] for s in out["synthesized_sidecar"]["sections"]]
    assert sidecar_ids == ["key-terms"]


def test_fused_non_apparatus_heading_still_demoted():
    """A fused heading NOT led by an apparatus name still takes the
    is_fused_heading demotion path (split must not swallow it)."""
    fused = (
        r"Simplify $\sqrt{20}$ then $\sqrt{5}$ and add the two results "
        "carefully before checking."
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[_heading_block(fused, 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    assert "<h3" not in html  # demoted, no heading minted
    assert "Simplify" in html  # text survives as prose


# ---------------------------------------------------------------------------
# RULE A (rerender_v5) — interior ALL-CAPS apparatus token split mid-block.
# ---------------------------------------------------------------------------


def test_interior_key_concepts_split_three_blocks_with_provenance():
    fused = (
        "whole numbers are the numbers 0, 1, 2, 3, and so on without end. "
        "KEY CONCEPTS 1.1 Introduction to Whole Numbers"
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 1 Foundations",
            blocks=[_para_block(fused, 4)],  # page 5
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    # apparatus heading minted from the interior token
    assert '<h3 id="key-concepts">KEY CONCEPTS</h3>' in html
    # preceding text stays as prose, remainder follows as prose
    assert "whole numbers are the numbers 0, 1, 2, 3" in html
    assert "1.1 Introduction to Whole Numbers" in html
    # the fused one-block form no longer exists
    assert "without end. KEY CONCEPTS 1.1" not in html
    # provenance: all THREE emitted sections carry the source block's page (5)
    assert html.count('data-dart-pages="5"') == 3
    # sidecar parity: three sections, same page range
    secs = out["synthesized_sidecar"]["sections"]
    assert len(secs) == 3
    assert all(s["page_range"] == [5, 5] for s in secs)
    assert [s["section_id"] for s in secs][1] == "key-concepts"


def test_interior_practice_test_split():
    fused = (
        r"\( \frac{m}{7} + \frac{10}{7} \) PRACTICE TEST Write as a whole "
        "number using digits: two hundred five thousand."
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 1 Foundations",
            blocks=[_para_block(fused, 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    assert '<h3 id="practice-test">PRACTICE TEST</h3>' in html
    assert "Write as a whole number using digits" in html
    assert "PRACTICE TEST Write as a whole" not in html


def test_interior_split_fires_on_flat_text_list_block():
    """The real ch01/ch04 residuals are council-typed 'list' glossary blobs
    whose html fell back to a flat <p>: the interior split must own them."""
    fused = (
        "whole numbers The whole numbers are the numbers 0, 1, 2, 3, and on. "
        "KEY CONCEPTS 1.1 Introduction to Whole Numbers"
    )
    block = _AdapterBlock(
        html=f"<p>{fused}</p>",  # flat-text fallback body
        region_kind="list",
        raw_block_index=0,
        raw_text=fused,
        heading_text=None,
        pages=[179],
    )
    chapters = [_AdapterChapter(title="Chapter 1 Foundations", blocks=[block])]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    assert '<h3 id="key-concepts">KEY CONCEPTS</h3>' in html
    assert "1.1 Introduction to Whole Numbers" in html
    assert "on. KEY CONCEPTS 1.1" not in html


def test_interior_split_never_flattens_real_list_markup():
    """A list block with REAL <ul> markup is never split/flattened even when
    its raw_text carries an interior apparatus token."""
    fused = (
        "first item text here always. KEY CONCEPTS second item text follows "
        "the banner token."
    )
    block = _AdapterBlock(
        html="<ul><li>first item text here always. KEY CONCEPTS</li>"
        "<li>second item text follows the banner token.</li></ul>",
        region_kind="list",
        raw_block_index=0,
        raw_text=fused,
        heading_text=None,
        pages=[1],
    )
    chapters = [_AdapterChapter(title="Chapter 1 Foundations", blocks=[block])]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    assert "<ul><li>" in html  # markup preserved
    assert '<h3 id="key-concepts">' not in html  # no split


def test_interior_lowercase_prose_mention_not_split():
    prose = (
        "In this chapter the key concepts 1.1 through 1.5 build the practice "
        "test skills every student needs to review carefully."
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 1 Foundations",
            blocks=[_para_block(prose, 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    assert "<h3" not in html  # no heading minted from lowercase mentions
    assert "the key concepts 1.1 through 1.5" in html
    assert len(out["synthesized_sidecar"]["sections"]) == 1


# ---------------------------------------------------------------------------
# RULE B (rerender_v5) — standalone "Introduction" paragraph typed as heading.
# ---------------------------------------------------------------------------


def test_standalone_introduction_paragraph_promoted():
    chapters = [
        _AdapterChapter(
            title="Chapter 4 Graphs",
            blocks=[
                _para_block("Introduction", 0),
                _para_block("Graphs describe how two quantities vary together.", 1),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch4")
    html = out["html"]
    assert '<h3 id="introduction">Introduction</h3>' in html
    assert "<p>Introduction</p>" not in html


def test_leading_introduction_heading_not_touched():
    """'Introduction' is standalone-arm ONLY: a real 'Introduction to Whole
    Numbers' heading is untouched and 'Introduction to Whole Numbers …' leading
    prose is never split."""
    chapters = [
        _AdapterChapter(
            title="Chapter 1 Foundations",
            blocks=[
                _heading_block("Introduction to Whole Numbers", 0),
                _para_block(
                    "Introduction to Whole Numbers begins with counting objects "
                    "one at a time carefully.",
                    1,
                ),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    # the real heading survives verbatim; the prose paragraph is NOT split
    assert (
        '<h3 id="introduction-to-whole-numbers">Introduction to Whole Numbers</h3>'
        in html
    )
    assert "begins with counting objects" in html
    # exactly one heading — no spurious 'Introduction' heading minted from prose
    assert html.count("<h3") == 1


# ---------------------------------------------------------------------------
# RULE C (rerender_v5) — phantom running-header h2 demotion + folio strip.
# ---------------------------------------------------------------------------


def test_folio_chapter_running_header_h2_demoted_to_banner():
    """'8 Chapter 1 Foundations' (1-digit folio — below the old 2-4 digit
    floor) is a running header: demoted to the aria-hidden continuation
    banner, never a visible h2."""
    chapters = [
        _AdapterChapter(
            title="Chapter 1 Foundations",
            blocks=[_para_block("Real chapter-opening prose lives here.", 0)],
        ),
        _AdapterChapter(
            title="8 Chapter 1 Foundations",
            blocks=[_para_block("Page-eight prose continues the chapter.", 1)],
        ),
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    assert "<h2>8 Chapter 1 Foundations</h2>" not in html
    assert 'class="dart-continuation"' in html
    assert "Page-eight prose continues" in html  # blocks under it still render
    # the REAL chapter opener h2 survives
    assert "<h2>Chapter 1 Foundations</h2>" in html


def test_bare_chapter_running_header_demoted_with_title_knowledge():
    """'Chapter 4 Graphs' x2 (below the x3 repeat rule) demotes when the
    render-time title knowledge says the chapter title IS 'Graphs'."""
    chapters = [
        _AdapterChapter(
            title="Chapter 4 Graphs",
            blocks=[_para_block("First page prose of the chapter here.", 0)],
        ),
        _AdapterChapter(
            title="Chapter 4 Graphs",
            blocks=[_para_block("Later page prose of the chapter here.", 1)],
        ),
    ]
    out = normalize_cascade_to_ed4all(
        _Result(chapters), pdf_stem="ea_ch4", title_override="Graphs"
    )
    html = out["html"]
    assert "<h2>Chapter 4 Graphs</h2>" not in html
    assert html.count('class="dart-continuation"') == 2
    # content under both survives; the h1 carries the honest title
    assert "First page prose" in html and "Later page prose" in html
    assert "<h1>Graphs</h1>" in html


def test_folio_prefixed_section_title_folio_stripped():
    """'130 The Real Numbers' keeps the REAL heading, sheds the folio."""
    chapters = [
        _AdapterChapter(
            title="130 The Real Numbers",
            blocks=[_para_block("Real numbers include rationals and irrationals.", 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch1")
    html = out["html"]
    assert "<h2>The Real Numbers</h2>" in html
    assert "130 The Real Numbers" not in html


def test_chapter_prefix_stripped_from_fused_solution_heading():
    """h3 'Chapter 6 Polynomials Solution' (running header fused onto a real
    label): the prefix is stripped and the bare 'Solution' takes the existing
    decorated-solution demotion."""
    chapters = [
        _AdapterChapter(
            title="Chapter 6 Polynomials",
            blocks=[
                _heading_block("Chapter 6 Polynomials Solution", 0),
                _para_block("The worked answer follows the label.", 1),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(
        _Result(chapters), pdf_stem="ea_ch6", title_override="Polynomials"
    )
    html = out["html"]
    assert "Chapter 6 Polynomials Solution" not in html
    # 'Solution' handling as-is: demoted to the pedagogy-solution paragraph
    assert '<p class="pedagogy-solution">Solution</p>' in html


def test_legit_numbered_section_heading_untouched():
    """'1.2 Use the Language of Algebra' (dot after the digits) is a genuine
    numbered section heading — no folio strip, no demotion."""
    chapters = [
        _AdapterChapter(
            title="Chapter 1 Foundations",
            blocks=[_heading_block("1.2 Use the Language of Algebra", 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(
        _Result(chapters), pdf_stem="ea_ch1", title_override="Foundations"
    )
    html = out["html"]
    assert ">1.2 Use the Language of Algebra</h3>" in html


# ---------------------------------------------------------------------------
# ch02 audit — literal nbsp-entity artifact runs scrubbed at render time.
# ---------------------------------------------------------------------------


def test_nbsp_entity_runs_scrubbed_from_html_and_sidecar():
    """A block carrying literal '&nbsp;' entity runs (the VLM table-spacing
    transcription defect — 520 visible "nbsp" occurrences in ch02) renders
    with single spaces: no literal nbsp survives in HTML or sidecar text."""
    fused = (
        "Divide. &nbsp; &nbsp; &nbsp; 45 by 9 &amp;nbsp; &amp;nbsp; then "
        "simplify the quotient completely."
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 2 The Language of Algebra",
            blocks=[_para_block(fused, 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch2")
    html = out["html"]
    assert "nbsp" not in html
    assert "Divide. 45 by 9 then simplify the quotient completely." in html
    # Sidecar parity: the sidecar text is the SAME scrubbed string.
    texts = [s["data"]["text"] for s in out["synthesized_sidecar"]["sections"]]
    assert all("nbsp" not in t for t in texts)
    assert any(
        "Divide. 45 by 9 then simplify the quotient completely." in t
        for t in texts
    )


def test_nbsp_word_without_entity_context_left_alone():
    """The bare word 'nbsp' in prose (no '&'/';' entity shape) is content —
    e.g. a sentence discussing the entity by name — and must not be touched."""
    prose = (
        "In HTML the nbsp entity inserts a non-breaking space between words "
        "so the browser never wraps there."
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 2 The Language of Algebra",
            blocks=[_para_block(prose, 0)],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch2")
    assert "the nbsp entity inserts a non-breaking space" in out["html"]
    texts = [s["data"]["text"] for s in out["synthesized_sidecar"]["sections"]]
    assert any(prose in t for t in texts)


def test_nbsp_scrub_helper_shapes():
    from lib.semantik.adapter import _scrub_entity_artifacts

    # runs collapse to ONE space (multi-escape included)
    assert (
        _scrub_entity_artifacts("a &nbsp; &amp;nbsp; &amp;amp;nbsp; b") == "a b"
    )
    # the bare 'nbsp;' fragment (semicolon required) folds too
    assert _scrub_entity_artifacts("a ;nbsp; nbsp; b") == "a ; b"
    # a letter-adjacent 'nbsp;' is NOT an entity artifact (word-boundary guard)
    assert _scrub_entity_artifacts("varnbsp; stays") == "varnbsp; stays"
    # no-artifact text passes through byte-identical (incl. bare word 'nbsp')
    for s in ("plain prose here", "the nbsp entity", "", None):
        assert _scrub_entity_artifacts(s) == s


def test_running_header_block_still_dropped():
    """(iv) The apparatus exemption must not weaken the running-header drop."""
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _heading_block("Chapter 9 Roots and Radicals 1039", 0),
                _para_block("Real body prose survives on this page.", 1),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ea_ch9")
    html = out["html"]
    assert "Chapter 9 Roots and Radicals 1039" not in html
    assert "Real body prose survives" in html
