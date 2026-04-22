"""Wave 15 tests: Dublin Core + schema.org JSON-LD + accessibility summary.

Covers :func:`DART.converter.document_assembler.assemble_html` extensions:

* Dublin Core ``<meta>`` tags: all-fields path, partial-metadata path,
  no-empty-content guarantee, multiple authors.
* Document-level schema.org JSON-LD: ``@type`` switching on
  ``document_type``, ``hasPart`` list derived from ``CHAPTER_OPENER``
  blocks, JSON validity.
* Accessibility summary JSON-LD: always emitted, structure validates,
  two JSON-LD blocks present in ``<head>``.
* ``<head>`` ordering: charset -> viewport -> title -> Dublin Core ->
  schema.org JSON-LD -> accessibility JSON-LD -> style bundle.
"""

from __future__ import annotations

import json
import re

import pytest

from DART.converter import (
    BlockRole,
    ClassifiedBlock,
    RawBlock,
    assemble_html,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chapter(number: int, heading: str) -> ClassifiedBlock:
    """Build a CHAPTER_OPENER with an explicit number attribute."""
    return ClassifiedBlock(
        raw=RawBlock(text=f"Chapter {number}: {heading}", block_id=f"chap{number:03d}"),
        role=BlockRole.CHAPTER_OPENER,
        confidence=0.9,
        attributes={"heading_text": heading, "number": str(number)},
    )


def _para(text: str, bid: str = "p001") -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(text=text, block_id=bid),
        role=BlockRole.PARAGRAPH,
        confidence=0.5,
    )


def _extract_jsonld(html_out: str) -> list[dict]:
    """Return every parsed JSON-LD payload in the document, in order."""
    payloads = []
    for match in re.finditer(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        html_out,
        flags=re.DOTALL,
    ):
        payloads.append(json.loads(match.group(1)))
    return payloads


# ---------------------------------------------------------------------------
# Dublin Core meta tags
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestDublinCore:
    def test_dublin_core_all_fields_emitted(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Sample Book",
            {
                "authors": "Jane Doe",
                "date": "2026-04-20",
                "language": "en",
                "rights": "CC BY 4.0",
                "subject": "accessibility, wcag, html",
            },
        )
        assert '<meta name="DC.title" content="Sample Book">' in html_out
        assert '<meta name="DC.creator" content="Jane Doe">' in html_out
        assert '<meta name="DC.date" content="2026-04-20">' in html_out
        assert '<meta name="DC.language" content="en">' in html_out
        assert '<meta name="DC.rights" content="CC BY 4.0">' in html_out
        assert '<meta name="DC.subject" content="accessibility, wcag, html">' in html_out

    def test_dublin_core_partial_metadata_omits_missing_fields(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Partial Doc",
            {"authors": "Jane Doe"},
        )
        # Title + creator always emitted (title is required; creator present).
        assert '<meta name="DC.title"' in html_out
        assert '<meta name="DC.creator" content="Jane Doe">' in html_out
        # Language defaults to "en" even when absent.
        assert '<meta name="DC.language" content="en">' in html_out
        # Date / rights / subject absent from metadata -> no tag at all.
        assert '<meta name="DC.date"' not in html_out
        assert '<meta name="DC.rights"' not in html_out
        assert '<meta name="DC.subject"' not in html_out

    def test_dublin_core_never_emits_empty_content_attribute(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Empty Fields",
            {
                "authors": "",
                "date": "",
                "rights": None,
                "subject": "   ",
            },
        )
        # Any DC.* meta must have a non-empty content attribute.
        for match in re.finditer(r'<meta name="DC\.[^"]+"\s+content="([^"]*)"', html_out):
            assert match.group(1).strip(), match.group(0)

    def test_dublin_core_handles_multiple_authors_as_list(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Collab Doc",
            {"authors": ["Jane Doe", "John Smith"]},
        )
        assert (
            '<meta name="DC.creator" content="Jane Doe, John Smith">' in html_out
        )

    def test_dublin_core_keywords_from_list_field(self):
        html_out = assemble_html(
            [_para("Body.")],
            "KW Doc",
            {"keywords": ["a11y", "wcag"]},
        )
        assert '<meta name="DC.subject" content="a11y, wcag">' in html_out

    def test_dublin_core_escapes_special_chars_in_content(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Special",
            {"authors": 'Jane "Quote" Doe & John'},
        )
        # Quotes in attribute values must be HTML-escaped.
        assert '"Quote"' not in html_out.split("DC.creator")[1][:200]
        assert "&quot;Quote&quot;" in html_out or "&#x27;Quote&#x27;" in html_out
        assert "&amp;" in html_out

    def test_dublin_core_defaults_language_to_en(self):
        html_out = assemble_html([_para("Body.")], "No Lang", {})
        assert '<meta name="DC.language" content="en">' in html_out


# ---------------------------------------------------------------------------
# Document-level schema.org JSON-LD
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestSchemaOrgDocumentJsonLd:
    def test_default_type_is_creative_work(self):
        html_out = assemble_html([_para("Body.")], "Doc", {})
        payloads = _extract_jsonld(html_out)
        # First JSON-LD is the document-level block.
        doc = payloads[0]
        assert doc["@type"] == "CreativeWork"
        assert doc["@context"] == "https://schema.org"

    def test_document_type_arxiv_becomes_scholarly_article(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Paper",
            {"document_type": "arxiv"},
        )
        doc = _extract_jsonld(html_out)[0]
        assert doc["@type"] == "ScholarlyArticle"

    def test_document_type_textbook_becomes_book(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Primer",
            {"document_type": "textbook"},
        )
        doc = _extract_jsonld(html_out)[0]
        assert doc["@type"] == "Book"

    def test_has_part_lists_every_chapter(self):
        blocks = [
            _chapter(1, "Foundations"),
            _para("Intro prose."),
            _chapter(2, "Methods"),
            _para("Methods prose."),
            _chapter(3, "Results"),
        ]
        html_out = assemble_html(blocks, "Three Chapters", {"document_type": "textbook"})
        doc = _extract_jsonld(html_out)[0]
        parts = doc["hasPart"]
        assert len(parts) == 3
        assert {p["@type"] for p in parts} == {"Chapter"}
        urls = [p["url"] for p in parts]
        assert urls == ["#chap-1", "#chap-2", "#chap-3"]
        names = [p["name"] for p in parts]
        assert names == ["Foundations", "Methods", "Results"]

    def test_has_part_omitted_when_no_chapters(self):
        html_out = assemble_html([_para("Only paragraph.")], "No Chapters", {})
        doc = _extract_jsonld(html_out)[0]
        assert "hasPart" not in doc

    def test_has_part_url_matches_chapter_article_id(self):
        """Every hasPart URL must point at an actual chapter anchor in
        the emitted body so the JSON-LD navigation graph is real."""
        blocks = [_chapter(2, "Methods")]
        html_out = assemble_html(blocks, "Doc", {"document_type": "textbook"})
        doc = _extract_jsonld(html_out)[0]
        parts = doc.get("hasPart", [])
        assert parts, "expected hasPart entries"
        for part in parts:
            anchor = part["url"].lstrip("#")
            assert f'id="{anchor}"' in html_out, anchor

    def test_has_part_derives_chapter_number_from_raw_text(self):
        """When the classifier strips the ``Chapter N:`` prefix, the raw
        block text is scraped so hasPart / article-id stay aligned."""
        block = ClassifiedBlock(
            raw=RawBlock(text="Chapter 7: Later Things", block_id="c7"),
            role=BlockRole.CHAPTER_OPENER,
            confidence=0.9,
            attributes={"heading_text": "Later Things"},
        )
        html_out = assemble_html([block, _para("Body.")], "Doc", {})
        doc = _extract_jsonld(html_out)[0]
        parts = doc.get("hasPart", [])
        assert parts and parts[0]["url"] == "#chap-7"
        assert 'id="chap-7"' in html_out

    def test_authors_emit_as_person_list(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Paper",
            {"authors": "Jane Doe, John Smith"},
        )
        doc = _extract_jsonld(html_out)[0]
        assert doc["author"] == [
            {"@type": "Person", "name": "Jane Doe"},
            {"@type": "Person", "name": "John Smith"},
        ]

    def test_schema_payload_is_valid_json(self):
        """Document-level JSON-LD must parse even with special characters."""
        html_out = assemble_html(
            [_para("Body.")],
            'Doc with "quotes" & <tags>',
            {
                "authors": 'Jane "Quote" Doe & Co.',
                "date": "2026-04-20",
            },
        )
        # Parsing the first JSON-LD payload must succeed.
        doc = _extract_jsonld(html_out)[0]
        assert "name" in doc
        # author must also parse.
        assert doc["author"][0]["@type"] == "Person"


# ---------------------------------------------------------------------------
# Accessibility summary JSON-LD
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestAccessibilityJsonLd:
    def test_accessibility_jsonld_always_emitted(self):
        html_out = assemble_html([_para("Body.")], "Minimal", {})
        payloads = _extract_jsonld(html_out)
        assert len(payloads) == 2
        acc = payloads[1]
        assert acc["@context"] == "https://schema.org"
        assert acc["@type"] == "CreativeWork"

    def test_accessibility_summary_carries_expected_features(self):
        html_out = assemble_html([_para("Body.")], "Doc", {})
        acc = _extract_jsonld(html_out)[1]
        assert set(acc["accessMode"]) == {"textual", "visual"}
        features = set(acc["accessibilityFeature"])
        required = {
            "structuralNavigation",
            "alternativeText",
            "tableOfContents",
            "readingOrder",
            "displayTransformability",
            "highContrastDisplay",
        }
        assert required.issubset(features)
        assert acc["accessibilityHazard"] == ["none"]
        assert "WCAG 2.2 AA" in acc["accessibilitySummary"]

    def test_two_jsonld_blocks_are_present_in_head(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Doc",
            {"document_type": "arxiv"},
        )
        head_match = re.search(r"<head\b[^>]*>(.*?)</head>", html_out, re.DOTALL)
        assert head_match is not None
        head = head_match.group(1)
        jsonld_tags = re.findall(
            r'<script type="application/ld\+json">', head
        )
        assert len(jsonld_tags) == 2


# ---------------------------------------------------------------------------
# <head> ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestHeadOrdering:
    def test_head_ordering_is_canonical(self):
        html_out = assemble_html(
            [_para("Body.")],
            "Ordering",
            {
                "authors": "Jane Doe",
                "date": "2026-04-20",
                "language": "en",
                "rights": "CC BY 4.0",
                "subject": "wcag",
                "document_type": "textbook",
            },
        )
        head_match = re.search(r"<head\b[^>]*>(.*?)</head>", html_out, re.DOTALL)
        assert head_match is not None
        head = head_match.group(1)

        # Each element must appear and be in the documented order.
        positions = {
            "charset": head.find('<meta charset="UTF-8">'),
            "viewport": head.find('name="viewport"'),
            "title": head.find("<title>"),
            "dc.title": head.find('name="DC.title"'),
            "jsonld_doc": head.find('"@type": "Book"'),
            "jsonld_acc": head.find('"accessibilityFeature"'),
            "style": head.find("<style>"),
        }
        for key, pos in positions.items():
            assert pos != -1, f"missing section: {key}"
        # Must be strictly ascending.
        order = [
            positions["charset"],
            positions["viewport"],
            positions["title"],
            positions["dc.title"],
            positions["jsonld_doc"],
            positions["jsonld_acc"],
            positions["style"],
        ]
        assert order == sorted(order), f"order violated: {positions}"

    def test_cross_references_do_not_corrupt_head(self):
        """Dublin Core + JSON-LD in <head> must survive the cross-ref pass."""
        blocks = [
            _chapter(1, "See Chapter 1 Introduction"),
            _para("This body mentions Chapter 1 and Chapter 2."),
        ]
        html_out = assemble_html(
            blocks,
            "Chapter Title Mentions Chapter 1",  # Title itself has "Chapter 1"
            {"authors": "Jane Doe"},
        )
        # Title meta must be untouched (no anchors injected).
        title_match = re.search(
            r'<meta name="DC\.title"\s+content="([^"]+)"', html_out
        )
        assert title_match is not None
        assert "<a" not in title_match.group(1)
