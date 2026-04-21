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
    classified = HeuristicClassifier().classify(blocks)
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

    def test_every_template_emits_data_dart_block_role(self):
        for role in BlockRole:
            dummy = ClassifiedBlock(
                raw=RawBlock(text="sample text", block_id="abc123"),
                role=role,
                confidence=0.5,
            )
            html_out = render_block(dummy)
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
        # Exactly one <h1> in the document shell.
        assert html_out.count("<h1>") == 1
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
        # Chapter opener renders as a <section>.
        assert "<section" in html_out
        # Body paragraph is present and escaped.
        assert "introductory paragraph" in html_out
        # Provenance attributes present.
        assert 'data-dart-block-role="chapter_opener"' in html_out
        assert 'data-dart-block-role="paragraph"' in html_out
