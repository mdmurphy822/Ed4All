"""Tests for the Wave 12 ontology-aware DART converter package.

Covers:

    * ``BlockRole`` enum exhaustiveness
    * ``segment_pdftotext_output`` splitting + stable IDs
    * ``HeuristicClassifier`` role decisions
    * ``TEMPLATE_REGISTRY`` exhaustiveness + escape / provenance behaviour
    * ``assemble_html`` document shell shape
    * End-to-end pipeline smoke test
"""

from __future__ import annotations

import re

import pytest

from DART.converter import (
    BlockRole,
    ClassifiedBlock,
    HeuristicClassifier,
    RawBlock,
    TEMPLATE_REGISTRY,
    assemble_html,
    convert_pdftotext_to_html,
    render_block,
    segment_pdftotext_output,
)


# ---------------------------------------------------------------------------
# BlockRole enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestBlockRoleEnum:
    def test_block_roles_enum_count(self):
        # The plan specifies 30 roles across six groups; ensure no
        # accidental removal below that floor when subsequent waves add
        # roles.
        assert len(list(BlockRole)) >= 30

    def test_block_role_values_are_unique(self):
        values = [role.value for role in BlockRole]
        assert len(values) == len(set(values))

    def test_block_role_values_are_snake_case(self):
        snake = re.compile(r"^[a-z][a-z0-9_]*$")
        for role in BlockRole:
            assert snake.match(role.value), role.value


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestSegmenter:
    def test_segmenter_splits_on_blank_lines(self):
        raw = "Paragraph one goes here.\n\nParagraph two goes here.\n\nParagraph three."
        blocks = segment_pdftotext_output(raw)
        assert len(blocks) == 3
        assert blocks[0].text.startswith("Paragraph one")
        assert blocks[2].text.startswith("Paragraph three")

    def test_segmenter_produces_stable_block_ids(self):
        raw = "First block.\n\nSecond block text here."
        first = segment_pdftotext_output(raw)
        second = segment_pdftotext_output(raw)
        assert [b.block_id for b in first] == [b.block_id for b in second]

    def test_segmenter_populates_neighbours(self):
        raw = "Alpha block.\n\nBeta block.\n\nGamma block."
        blocks = segment_pdftotext_output(raw)
        assert blocks[0].neighbors["prev"] == ""
        assert blocks[0].neighbors["next"].startswith("Beta")
        assert blocks[1].neighbors["prev"].startswith("Alpha")
        assert blocks[1].neighbors["next"].startswith("Gamma")
        assert blocks[2].neighbors["next"] == ""

    def test_segmenter_assigns_pages_on_form_feed(self):
        raw = "Page one body.\n\fPage two body."
        blocks = segment_pdftotext_output(raw)
        assert blocks[0].page == 1
        assert blocks[1].page == 2

    def test_segmenter_rejoins_soft_hyphens(self):
        raw = "This is a demon-\nstration of soft hyphen rejoining."
        blocks = segment_pdftotext_output(raw)
        assert len(blocks) == 1
        assert "demonstration" in blocks[0].text

    def test_segmenter_returns_empty_on_empty_input(self):
        assert segment_pdftotext_output("") == []
        assert segment_pdftotext_output("   \n\n  \n") == []


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------


def _classify_text(text: str) -> ClassifiedBlock:
    blocks = segment_pdftotext_output(text)
    # Wave 14: ``HeuristicClassifier.classify`` is async for symmetry with
    # ``LLMClassifier``. Use the sync convenience helper here so the
    # existing unit tests don't need to grow an event loop.
    classified = HeuristicClassifier().classify_sync(blocks)
    assert len(classified) == 1
    return classified[0]


@pytest.mark.unit
@pytest.mark.dart
class TestHeuristicClassifier:
    def test_heuristic_classifier_chapter_opener(self):
        result = _classify_text("Chapter 3: Foo Bar")
        assert result.role == BlockRole.CHAPTER_OPENER
        assert result.confidence >= 0.8
        assert result.classifier_source == "heuristic"
        assert result.attributes["heading_text"] == "Foo Bar"

    def test_heuristic_classifier_numbered_chapter(self):
        result = _classify_text("11. Behaviorism and Learning")
        assert result.role == BlockRole.CHAPTER_OPENER

    def test_heuristic_classifier_abstract(self):
        result = _classify_text("Abstract")
        assert result.role == BlockRole.ABSTRACT

    def test_heuristic_classifier_paragraph_default(self):
        prose = (
            "This is a perfectly ordinary sentence that carries no "
            "structural signal and should land squarely in a paragraph."
        )
        result = _classify_text(prose)
        assert result.role == BlockRole.PARAGRAPH

    def test_heuristic_classifier_rejects_low_signal_heading(self):
        # "VANCOUVER BC" would otherwise pass the sub-heading regex.
        result = _classify_text("VANCOUVER BC")
        assert result.role != BlockRole.SUBSECTION_HEADING
        assert result.role != BlockRole.SECTION_HEADING

    def test_heuristic_classifier_accepts_valid_subheading(self):
        result = _classify_text("Learning From Reflection")
        assert result.role == BlockRole.SUBSECTION_HEADING

    def test_heuristic_classifier_toc_entry(self):
        result = _classify_text("Introduction . . . . . . 12")
        assert result.role == BlockRole.TOC_NAV

    def test_heuristic_classifier_copyright_line(self):
        result = _classify_text("(c) 2024 OpenStax. Licensed under CC BY 4.0.")
        assert result.role == BlockRole.COPYRIGHT_LICENSE

    def test_heuristic_classifier_arxiv_meta(self):
        result = _classify_text("arXiv:2403.01234v2 [cs.LG] 3 Mar 2024")
        assert result.role == BlockRole.AUTHOR_AFFILIATION

    def test_heuristic_classifier_email_affiliation(self):
        result = _classify_text("Jane Doe, jane@example.edu")
        assert result.role == BlockRole.AUTHOR_AFFILIATION

    def test_heuristic_classifier_paper_section(self):
        result = _classify_text("1 Introduction")
        assert result.role == BlockRole.SECTION_HEADING

    def test_heuristic_classifier_paper_subsection(self):
        result = _classify_text("2.1 Related Work")
        assert result.role == BlockRole.SUBSECTION_HEADING


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestTemplateRegistry:
    def test_template_registry_has_entry_per_role(self):
        for role in BlockRole:
            assert role in TEMPLATE_REGISTRY, f"missing template for {role}"

    def test_every_wrapper_template_emits_data_dart_block_role(self):
        """Wave 19: only wrapper templates (section/article/aside/figure/
        table/pre/math/blockquote/nav/dl) carry ``data-dart-*`` attributes.
        Leaf templates (paragraph/span/h1/h3/cite/a/li/figcaption/plain p)
        strip them per the Wave 8 P2 attribute-scope rule.
        """
        from DART.converter.block_templates import _WAVE19_LEAF_ROLES

        for role in BlockRole:
            dummy = ClassifiedBlock(
                raw=RawBlock(text="sample text", block_id="abc123"),
                role=role,
                confidence=0.5,
            )
            html_out = render_block(dummy)
            if role in _WAVE19_LEAF_ROLES:
                # Leaf roles must NOT carry the block-role provenance.
                assert f'data-dart-block-role="{role.value}"' not in html_out, (
                    f"leaf {role.value} should not emit data-dart-block-role"
                )
            else:
                # Wrapper roles keep the Wave 13 provenance contract.
                assert f'data-dart-block-role="{role.value}"' in html_out

    def test_template_escapes_html(self):
        evil = "<script>alert(1)</script>"
        dummy = ClassifiedBlock(
            raw=RawBlock(text=evil, block_id="evil01"),
            role=BlockRole.PARAGRAPH,
            confidence=0.5,
        )
        out = render_block(dummy)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestAssembler:
    def test_assembler_produces_valid_html5(self):
        blocks = [
            ClassifiedBlock(
                raw=RawBlock(text="Hello world.", block_id="b1"),
                role=BlockRole.PARAGRAPH,
                confidence=0.5,
            )
        ]
        html_out = assemble_html(blocks, "Test Doc", {})
        assert html_out.startswith("<!DOCTYPE html>")
        assert '<html lang="en">' in html_out
        assert "<main id=\"main-content\"" in html_out
        assert "<header role=\"banner\">" in html_out
        # Exactly one <h1> in the document shell. Wave 19 adds an id to
        # the shell heading so ``<main aria-labelledby>`` can target it.
        assert html_out.count("<h1 ") == 1
        assert html_out.count("</h1>") == 1

    def test_assembler_sweeps_metadata_into_aside(self):
        blocks = [
            ClassifiedBlock(
                raw=RawBlock(text="Body prose.", block_id="b1"),
                role=BlockRole.PARAGRAPH,
                confidence=0.5,
            ),
            ClassifiedBlock(
                raw=RawBlock(text="(c) 2024 Example Press.", block_id="b2"),
                role=BlockRole.COPYRIGHT_LICENSE,
                confidence=0.8,
            ),
        ]
        html_out = assemble_html(blocks, "Test", {})
        assert '<aside role="complementary"' in html_out
        assert "Example Press" in html_out

    def test_assembler_emits_caller_metadata(self):
        html_out = assemble_html([], "Test", {"authors": "Jane & John"})
        assert '<aside role="complementary"' in html_out
        # html.escape should encode the ampersand.
        assert "Jane &amp; John" in html_out


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestPipelineEndToEnd:
    def test_pipeline_end_to_end_parity(self):
        raw = (
            "Chapter 1: Foo\n\n"
            "This is the introductory paragraph with enough words to "
            "carry real content."
        )
        html_out = convert_pdftotext_to_html(raw, "Test Doc")
        assert html_out.startswith("<!DOCTYPE html>")
        # Chapter opener renders as a DPUB-ARIA doc-chapter <article>.
        # Wave 19: <article> carries ``class="dart-section"`` so the role
        # attribute is no longer adjacent to the opening tag.
        assert "<article " in html_out
        assert 'role="doc-chapter"' in html_out
        # Body paragraph is present and escaped.
        assert "introductory paragraph" in html_out
        # Provenance attributes present on wrappers only; leaf <p> strips.
        assert 'data-dart-block-role="chapter_opener"' in html_out
        # Wave 19: paragraphs are leaf nodes without data-dart-*.
        assert 'data-dart-block-role="paragraph"' not in html_out


# ---------------------------------------------------------------------------
# Wave 13: per-role snapshot tests
# ---------------------------------------------------------------------------


def _mk(role: BlockRole, text: str = "Sample text", **attrs) -> ClassifiedBlock:
    """Build a ClassifiedBlock with deterministic block_id + optional attrs."""
    return ClassifiedBlock(
        raw=RawBlock(text=text, block_id="blk000001"),
        role=role,
        confidence=0.5,
        attributes=attrs,
    )


def _assert_provenance(rendered: str, role: BlockRole) -> None:
    """Wave 19: wrapper templates preserve the provenance contract; leaf
    templates strip all ``data-dart-*`` attributes per the P2 rule.
    """
    from DART.converter.block_templates import _WAVE19_LEAF_ROLES

    if role in _WAVE19_LEAF_ROLES:
        # Leaf role — verify no data-dart-* attributes leaked.
        assert f'data-dart-block-role="{role.value}"' not in rendered, (
            f"leaf {role.value} should not emit data-dart-block-role"
        )
        assert 'data-dart-block-id="blk000001"' not in rendered, (
            f"leaf {role.value} should not emit data-dart-block-id"
        )
        return
    assert f'data-dart-block-role="{role.value}"' in rendered
    assert 'data-dart-block-id="blk000001"' in rendered
    # Wave 19 addition: wrappers now also carry data-dart-source.
    assert "data-dart-source=" in rendered, (
        f"wrapper {role.value} should emit data-dart-source"
    )


@pytest.mark.unit
@pytest.mark.dart
class TestStructuralTemplates:
    def test_chapter_opener_emits_dpub_role_and_microdata(self):
        out = render_block(
            _mk(BlockRole.CHAPTER_OPENER, "Chapter 1", heading_text="Foundations")
        )
        assert "<article " in out
        assert 'role="doc-chapter"' in out
        assert 'itemtype="https://schema.org/Chapter"' in out
        assert 'itemprop="name"' in out
        assert "Foundations" in out
        # Wave 19: outer <article> wrapper carries the ``dart-section`` class.
        assert 'class="dart-section"' in out
        _assert_provenance(out, BlockRole.CHAPTER_OPENER)

    def test_section_heading_wraps_region_with_aria(self):
        out = render_block(_mk(BlockRole.SECTION_HEADING, "Methods"))
        assert '<section' in out
        assert 'role="region"' in out
        assert 'aria-labelledby=' in out
        assert "<h2" in out
        # Wave 19: section wrapper carries the ``dart-section`` class.
        assert 'class="dart-section"' in out
        _assert_provenance(out, BlockRole.SECTION_HEADING)

    def test_subsection_heading_emits_h3(self):
        """Wave 19: SUBSECTION is now a leaf ``<h3 id="...">``. No
        ``<section>`` wrapper, no ``aria-labelledby``, no data-dart-*."""
        out = render_block(_mk(BlockRole.SUBSECTION_HEADING, "Baseline"))
        assert "<h3" in out
        assert "Baseline" in out
        # Leaf role — no data-dart-* attributes.
        assert "data-dart-block-role" not in out
        assert "data-dart-block-id" not in out
        _assert_provenance(out, BlockRole.SUBSECTION_HEADING)

    def test_paragraph_escapes_and_wraps(self):
        out = render_block(_mk(BlockRole.PARAGRAPH, "Hello <world>."))
        # Wave 19: paragraph is a leaf — plain ``<p>`` with no attrs.
        assert out.startswith("<p>")
        assert "Hello &lt;world&gt;." in out
        _assert_provenance(out, BlockRole.PARAGRAPH)

    def test_toc_nav_renders_ordered_list(self):
        out = render_block(
            _mk(BlockRole.TOC_NAV, "TOC", items=["Intro", "Methods", "Results"])
        )
        # Wave 18 promotes TOC_NAV to the DPUB-ARIA ``doc-toc`` role.
        assert '<nav role="doc-toc"' in out
        assert "<ol>" in out
        assert "<li>Intro</li>" in out
        assert "Contents" in out
        _assert_provenance(out, BlockRole.TOC_NAV)

    def test_page_break_has_doc_pagebreak_role(self):
        out = render_block(_mk(BlockRole.PAGE_BREAK, "", page="42"))
        assert 'role="doc-pagebreak"' in out
        assert 'aria-label="page 42"' in out
        _assert_provenance(out, BlockRole.PAGE_BREAK)


@pytest.mark.unit
@pytest.mark.dart
class TestEducationalTemplates:
    def test_learning_objectives_uses_learning_resource_microdata(self):
        out = render_block(
            _mk(
                BlockRole.LEARNING_OBJECTIVES,
                "",
                items=["Apply Bloom's taxonomy", "Design a rubric"],
            )
        )
        assert 'itemtype="https://schema.org/LearningResource"' in out
        assert 'itemprop="learningResourceType"' in out
        assert "<ul>" in out
        assert "<li>Apply Bloom&#x27;s taxonomy</li>" in out
        _assert_provenance(out, BlockRole.LEARNING_OBJECTIVES)

    def test_key_takeaways_uses_doc_tip(self):
        out = render_block(
            _mk(BlockRole.KEY_TAKEAWAYS, "", items=["Takeaway one"])
        )
        assert 'role="doc-tip"' in out
        assert "Key Takeaways" in out
        _assert_provenance(out, BlockRole.KEY_TAKEAWAYS)

    def test_activity_uses_doc_example(self):
        out = render_block(_mk(BlockRole.ACTIVITY, "Try this.", title="Reflect"))
        assert 'role="doc-example"' in out
        assert "Reflect" in out
        _assert_provenance(out, BlockRole.ACTIVITY)

    def test_self_check_has_aria_label(self):
        out = render_block(
            _mk(BlockRole.SELF_CHECK, "", items=["Q1?", "Q2?"])
        )
        assert 'role="doc-example"' in out
        assert 'aria-label="Self-check"' in out
        assert "<ol>" in out
        _assert_provenance(out, BlockRole.SELF_CHECK)

    def test_example_includes_role_and_title(self):
        out = render_block(_mk(BlockRole.EXAMPLE, "Body.", title="Worked Proof"))
        assert 'role="doc-example"' in out
        assert "Example: Worked Proof" in out
        _assert_provenance(out, BlockRole.EXAMPLE)

    def test_exercise_uses_doc_example_role(self):
        out = render_block(_mk(BlockRole.EXERCISE, "Solve.", title="Problem 1"))
        assert 'role="doc-example"' in out
        assert "Exercise: Problem 1" in out
        _assert_provenance(out, BlockRole.EXERCISE)

    def test_glossary_entry_uses_doc_glossary(self):
        out = render_block(
            _mk(
                BlockRole.GLOSSARY_ENTRY,
                "ignored",
                term="Ontology",
                definition="A formal specification of shared concepts.",
            )
        )
        assert '<dl role="doc-glossary"' in out
        assert '<dt itemprop="name">Ontology</dt>' in out
        assert 'itemprop="description"' in out
        _assert_provenance(out, BlockRole.GLOSSARY_ENTRY)


@pytest.mark.unit
@pytest.mark.dart
class TestReferenceTemplates:
    def test_abstract_uses_doc_abstract_role(self):
        out = render_block(_mk(BlockRole.ABSTRACT, "Short summary of the paper."))
        assert 'role="doc-abstract"' in out
        assert 'itemprop="abstract"' in out
        assert "Abstract" in out
        _assert_provenance(out, BlockRole.ABSTRACT)

    def test_bibliography_entry_uses_doc_endnote(self):
        out = render_block(
            _mk(
                BlockRole.BIBLIOGRAPHY_ENTRY,
                "Smith, J. (2024). Accessibility by design.",
                number="1",
            )
        )
        assert 'role="doc-endnote"' in out
        assert 'itemtype="https://schema.org/CreativeWork"' in out
        assert 'id="ref-1"' in out
        assert "<cite" in out
        _assert_provenance(out, BlockRole.BIBLIOGRAPHY_ENTRY)

    def test_footnote_uses_doc_footnote_with_backref(self):
        out = render_block(_mk(BlockRole.FOOTNOTE, "Foo details.", number="3"))
        assert 'role="doc-footnote"' in out
        assert 'id="fn-3"' in out
        assert "<sup>3</sup>" in out
        assert "\u21a9" in out
        _assert_provenance(out, BlockRole.FOOTNOTE)

    def test_citation_is_inline_cite(self):
        out = render_block(_mk(BlockRole.CITATION, "Doe 2023"))
        assert out.startswith("<cite")
        _assert_provenance(out, BlockRole.CITATION)

    def test_cross_reference_uses_doc_cross_reference(self):
        out = render_block(
            _mk(BlockRole.CROSS_REFERENCE, "See Chapter 2", target_id="ch2")
        )
        assert 'role="doc-cross-reference"' in out
        assert 'href="#ch2"' in out
        _assert_provenance(out, BlockRole.CROSS_REFERENCE)


@pytest.mark.unit
@pytest.mark.dart
class TestContentRichTemplates:
    def test_figure_with_src_emits_img(self):
        out = render_block(
            _mk(
                BlockRole.FIGURE,
                "A caption.",
                src="/img/figure-1.png",
                alt="a diagram",
                caption="Figure 1: Overview.",
                number="1",
            )
        )
        assert '<figure' in out
        assert 'itemtype="https://schema.org/ImageObject"' in out
        assert '<img src="/img/figure-1.png"' in out
        assert 'alt="a diagram"' in out
        assert 'itemprop="caption"' in out
        assert 'id="fig-1"' in out
        _assert_provenance(out, BlockRole.FIGURE)

    def test_figure_without_src_still_renders_caption(self):
        out = render_block(_mk(BlockRole.FIGURE, "Just a caption line."))
        assert "<figure" in out
        assert "<img" not in out
        assert "Just a caption line." in out
        _assert_provenance(out, BlockRole.FIGURE)

    def test_figure_caption_standalone(self):
        out = render_block(_mk(BlockRole.FIGURE_CAPTION, "Fig. 2 — title."))
        assert out.startswith("<figcaption")
        assert 'itemprop="caption"' in out
        _assert_provenance(out, BlockRole.FIGURE_CAPTION)

    def test_table_renders_caption_thead_tbody(self):
        out = render_block(
            _mk(
                BlockRole.TABLE,
                "",
                title="Results",
                headers=["Col A", "Col B"],
                rows=[["1", "2"], ["3", "4"]],
            )
        )
        assert '<table role="grid"' in out
        assert "<caption" in out
        assert "<thead>" in out
        assert "<tbody>" in out
        assert "<th>Col A</th>" in out
        assert "<td>3</td>" in out
        _assert_provenance(out, BlockRole.TABLE)

    def test_table_with_empty_data_still_renders(self):
        out = render_block(_mk(BlockRole.TABLE, "Empty table"))
        assert '<table role="grid"' in out
        assert "<thead></thead>" in out
        assert "<tbody></tbody>" in out
        _assert_provenance(out, BlockRole.TABLE)

    def test_code_block_uses_pre_region(self):
        out = render_block(
            _mk(
                BlockRole.CODE_BLOCK,
                "print('hi')",
                caption="Algorithm 1",
                number="1",
            )
        )
        assert '<pre role="region"' in out
        assert "<code>" in out
        assert "Algorithm 1" in out
        assert "print(&#x27;hi&#x27;)" in out
        _assert_provenance(out, BlockRole.CODE_BLOCK)

    def test_formula_math_uses_mathml_wrapper(self):
        out = render_block(
            _mk(BlockRole.FORMULA_MATH, "E = mc^2", fallback="E = mc^2")
        )
        assert "<math" in out
        assert 'xmlns="http://www.w3.org/1998/Math/MathML"' in out
        assert '<annotation encoding="text/plain">E = mc^2</annotation>' in out
        _assert_provenance(out, BlockRole.FORMULA_MATH)

    def test_blockquote_with_cite_and_footer(self):
        out = render_block(
            _mk(
                BlockRole.BLOCKQUOTE,
                "To be or not to be.",
                cite_url="https://example.org/hamlet",
                attribution="William Shakespeare",
            )
        )
        assert "<blockquote" in out
        assert 'cite="https://example.org/hamlet"' in out
        assert "<footer>" in out
        assert "William Shakespeare" in out
        _assert_provenance(out, BlockRole.BLOCKQUOTE)

    def test_epigraph_uses_doc_epigraph_role(self):
        out = render_block(
            _mk(BlockRole.EPIGRAPH, "Quote text.", attribution="Jane Doe")
        )
        assert 'role="doc-epigraph"' in out
        assert "<blockquote>" in out
        assert "Jane Doe" in out
        _assert_provenance(out, BlockRole.EPIGRAPH)

    def test_pullquote_uses_doc_pullquote_role(self):
        out = render_block(_mk(BlockRole.PULLQUOTE, "A striking phrase."))
        assert 'role="doc-pullquote"' in out
        # Wave 19: class list now includes the shared ``dart-section``.
        assert 'class="dart-section pullquote"' in out
        assert "<blockquote>" in out
        _assert_provenance(out, BlockRole.PULLQUOTE)


@pytest.mark.unit
@pytest.mark.dart
class TestCalloutTemplates:
    def test_callout_info_uses_note_role_and_sr_only(self):
        out = render_block(
            _mk(BlockRole.CALLOUT_INFO, "Important info.", title="Note")
        )
        assert 'role="note"' in out
        # Wave 19: ``dart-section`` is prepended to callout classes.
        assert 'class="dart-section callout callout-info"' in out
        assert 'class="sr-only">Information:' in out
        assert '<span aria-hidden="true">' in out
        _assert_provenance(out, BlockRole.CALLOUT_INFO)

    def test_callout_warning_uses_doc_notice_and_warning_class(self):
        out = render_block(
            _mk(BlockRole.CALLOUT_WARNING, "Heads up.", title="Caution")
        )
        assert 'role="doc-notice"' in out
        assert 'class="dart-section callout callout-warning"' in out
        assert 'class="sr-only">Warning:' in out
        _assert_provenance(out, BlockRole.CALLOUT_WARNING)

    def test_callout_tip_uses_doc_tip(self):
        out = render_block(_mk(BlockRole.CALLOUT_TIP, "Pro tip.", title="Hint"))
        assert 'role="doc-tip"' in out
        assert 'class="dart-section callout callout-tip"' in out
        assert 'class="sr-only">Tip:' in out
        _assert_provenance(out, BlockRole.CALLOUT_TIP)

    def test_callout_danger_uses_doc_notice_and_danger_class(self):
        out = render_block(
            _mk(BlockRole.CALLOUT_DANGER, "Severe warning.", title="Danger")
        )
        assert 'role="doc-notice"' in out
        assert 'class="dart-section callout callout-danger"' in out
        assert 'class="sr-only">Danger:' in out
        _assert_provenance(out, BlockRole.CALLOUT_DANGER)


@pytest.mark.unit
@pytest.mark.dart
class TestMetadataTemplates:
    def test_title_uses_h1_with_itemprop_name(self):
        out = render_block(_mk(BlockRole.TITLE, "Accessible Learning"))
        assert out.startswith("<h1")
        assert 'itemprop="name"' in out
        assert "Accessible Learning" in out
        _assert_provenance(out, BlockRole.TITLE)

    def test_author_affiliation_has_person_microdata(self):
        out = render_block(
            _mk(
                BlockRole.AUTHOR_AFFILIATION,
                "Jane Doe, ACME U.",
                name="Jane Doe",
                affiliation="ACME U.",
            )
        )
        assert 'itemtype="https://schema.org/Person"' in out
        assert 'itemprop="name"' in out
        assert 'itemprop="affiliation"' in out
        _assert_provenance(out, BlockRole.AUTHOR_AFFILIATION)

    def test_copyright_license_has_itemprop_license(self):
        out = render_block(
            _mk(BlockRole.COPYRIGHT_LICENSE, "(c) 2026 OpenStax. CC BY 4.0.")
        )
        assert 'itemprop="license"' in out
        _assert_provenance(out, BlockRole.COPYRIGHT_LICENSE)

    def test_keywords_has_itemprop_keywords(self):
        out = render_block(
            _mk(BlockRole.KEYWORDS, "accessibility, WCAG, HTML")
        )
        assert 'itemprop="keywords"' in out
        _assert_provenance(out, BlockRole.KEYWORDS)

    def test_bibliographic_metadata_renders_paragraph(self):
        out = render_block(
            _mk(BlockRole.BIBLIOGRAPHIC_METADATA, "ISBN 978-0-00-000000-0")
        )
        assert 'class="biblio-metadata"' in out
        _assert_provenance(out, BlockRole.BIBLIOGRAPHIC_METADATA)


# ---------------------------------------------------------------------------
# Wave 13: cross-template invariants + integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestTemplateInvariants:
    def test_no_template_emits_unescaped_script_tags(self):
        """Malicious text must never surface as raw <script> in any template."""
        payload = "<script>alert(1)</script>"
        # Figures / tables / glossary / blockquote all accept free-form
        # text via raw.text or via attributes; verify all 35 roles escape
        # a payload that would be damaging if unescaped.
        for role in BlockRole:
            block = ClassifiedBlock(
                raw=RawBlock(text=payload, block_id="evil01"),
                role=role,
                confidence=0.5,
                attributes={
                    "heading_text": payload,
                    "title": payload,
                    "caption": payload,
                    "term": payload,
                    "definition": payload,
                    "attribution": payload,
                    "name": payload,
                    "affiliation": payload,
                    "fallback": payload,
                    "alt": payload,
                    "items": [payload],
                    "headers": [payload],
                    "rows": [[payload]],
                },
            )
            out = render_block(block)
            assert "<script>" not in out, role
            assert "alert(1)" in out.replace("&lt;", "<").replace("&gt;", ">") or \
                "&lt;script&gt;" in out, role

    def test_every_wrapper_template_carries_block_id(self):
        """Wave 19: wrapper templates keep ``data-dart-block-id``; leaf
        templates strip it per the P2 attribute-scope rule."""
        from DART.converter.block_templates import _WAVE19_LEAF_ROLES

        for role in BlockRole:
            out = render_block(_mk(role, "sample"))
            if role in _WAVE19_LEAF_ROLES:
                assert 'data-dart-block-id="blk000001"' not in out, (
                    f"leaf {role.value} should not emit data-dart-block-id"
                )
            else:
                assert 'data-dart-block-id="blk000001"' in out, role


@pytest.mark.unit
@pytest.mark.dart
class TestAssembledDocumentValidity:
    def test_assembled_document_has_single_h1_and_unique_ids(self):
        """Minimal HTML5-validity check: one <h1>, no duplicate IDs."""
        blocks = [
            ClassifiedBlock(
                raw=RawBlock(text="Chapter 1: Foundations", block_id="a1"),
                role=BlockRole.CHAPTER_OPENER,
                confidence=0.9,
                attributes={"heading_text": "Foundations"},
            ),
            ClassifiedBlock(
                raw=RawBlock(text="Methods", block_id="a2"),
                role=BlockRole.SECTION_HEADING,
                confidence=0.9,
            ),
            # Two sections with identical heading text — IDs must not collide.
            ClassifiedBlock(
                raw=RawBlock(text="Methods", block_id="a3"),
                role=BlockRole.SECTION_HEADING,
                confidence=0.9,
            ),
            ClassifiedBlock(
                raw=RawBlock(text="[1] Doe (2024).", block_id="a4"),
                role=BlockRole.BIBLIOGRAPHY_ENTRY,
                confidence=0.8,
                attributes={"number": "1"},
            ),
            ClassifiedBlock(
                raw=RawBlock(text="[2] Smith (2025).", block_id="a5"),
                role=BlockRole.BIBLIOGRAPHY_ENTRY,
                confidence=0.8,
                attributes={"number": "2"},
            ),
        ]
        html_out = assemble_html(blocks, "Smoke Test", {})

        # Exactly one <h1> + one </h1>. Wave 19 adds an id attribute
        # so the shell heading matches ``<h1 `` (space) — never ``<h1>``.
        assert html_out.count("<h1 ") == 1
        assert html_out.count("</h1>") == 1

        # Bibliography wrapped in a single <ol role="doc-bibliography">.
        assert html_out.count('<ol role="doc-bibliography">') == 1
        assert html_out.count('role="doc-endnote"') == 2

        # No duplicate id="..." values.
        import re

        ids = re.findall(r'\bid="([^"]+)"', html_out)
        assert len(ids) == len(set(ids)), f"duplicate id(s): {ids}"

        # Rough tag-balance check: every element opened (<foo) with a
        # matching </foo> for a curated set of critical tags.
        for tag in ("html", "head", "body", "main", "header", "footer",
                    "article", "section", "ol", "table", "figure"):
            opens = len(re.findall(rf"<{tag}(?:\s|>)", html_out))
            closes = html_out.count(f"</{tag}>")
            assert opens == closes, f"unbalanced <{tag}>: {opens} opens vs {closes} closes"

    def test_assembled_document_injects_wcag_css(self):
        html_out = assemble_html([], "Test", {})
        assert ".sr-only" in html_out
        assert "prefers-reduced-motion" in html_out
        assert "prefers-color-scheme: dark" in html_out
        assert "outline: 3px solid" in html_out
        assert "scroll-margin-top: 80px" in html_out
        assert "min-height: 24px" in html_out
