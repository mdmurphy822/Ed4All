"""Wave 20 integration tests — ``extract_document`` page-chrome path.

Validates the end-to-end wiring between
:func:`DART.converter.extractor.extract_document` and
:func:`DART.converter.page_chrome.detect_page_chrome`. Every test
monkeypatches pdftotext via ``subprocess.run`` so the tests stay
PyMuPDF / pdfplumber / tesseract free and run on a bare CI.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from DART.converter import extractor as extractor_module
from DART.converter.block_roles import BlockRole
from DART.converter.block_segmenter import segment_extracted_document
from DART.converter.extractor import (
    ExtractedDocument,
    PageChrome,
    extract_document,
)
from DART.converter.page_chrome import detect_page_chrome


_FORM_FEED = "\x0c"


def _fake_pdftotext(stdout: bytes):
    def _run(cmd, *args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    return _run


@pytest.fixture
def no_optional_deps(monkeypatch):
    """Disable every optional extractor for integration tests."""
    monkeypatch.setattr(
        extractor_module, "_extract_tables_pdfplumber", lambda pdf_path: []
    )
    monkeypatch.setattr(extractor_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        extractor_module,
        "_extract_figures",
        lambda pdf_path, *, llm=None, **_: [],
    )
    monkeypatch.setattr(extractor_module, "_open_pymupdf", lambda pdf_path: None)
    yield


# ---------------------------------------------------------------------------
# Wave 20 integration: extract_document populates page_chrome
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestExtractDocumentPageChrome:
    def test_multi_page_with_repeating_header_detected(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "textbook.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        pages = [
            f"Teaching in a Digital Age {i}\n\n"
            f"Unique prose one {i} aaa\n"
            f"Unique prose two {i} bbb\n"
            f"Unique prose three {i} ccc\n"
            for i in range(2, 10)
        ]
        raw = _FORM_FEED.join(pages).encode("utf-8")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext(raw))

        doc = extract_document(str(pdf))
        assert isinstance(doc, ExtractedDocument)
        assert isinstance(doc.page_chrome, PageChrome)
        # Chrome was detected.
        assert any(
            "teaching in a digital age" in h for h in doc.page_chrome.headers
        )
        # Raw text no longer carries the running-header variants.
        assert "Teaching in a Digital Age 2" not in doc.raw_text
        assert "Teaching in a Digital Age 9" not in doc.raw_text
        # Page-number mapping populated (at least half the pages numbered).
        assert len(doc.page_chrome.page_number_lines) >= 4

    def test_backward_compat_empty_chrome_on_short_doc(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "short.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        # 2 pages — below the min_pages_to_analyze=4 threshold.
        pages = [
            "Header Line\n\nBody of page 1.",
            "Header Line\n\nBody of page 2.",
        ]
        raw = _FORM_FEED.join(pages).encode("utf-8")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext(raw))

        doc = extract_document(str(pdf))
        # No chrome stripping on short docs.
        assert doc.page_chrome.headers == set()
        assert doc.page_chrome.footers == set()
        assert "Header Line" in doc.raw_text

    def test_single_page_no_chrome_detection(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "single.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        # No form-feed → single-page doc.
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext(b"Title\n\nBody of document.")
        )
        doc = extract_document(str(pdf))
        assert doc.page_chrome.headers == set()
        # Raw text survives unchanged.
        assert "Title" in doc.raw_text

    def test_all_unique_content_no_chrome_detected(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "content.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        # Five pages, no repeating lines.
        pages = [
            f"Opening paragraph for page {i} with completely unique words "
            f"{i*13} and ideas.\n\n"
            f"Closing paragraph for page {i} wrapping up uniquely "
            f"{i*19}.\n"
            for i in range(1, 6)
        ]
        raw = _FORM_FEED.join(pages).encode("utf-8")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext(raw))
        doc = extract_document(str(pdf))
        # No detected chrome on all-unique content.
        assert doc.page_chrome.headers == set()
        assert doc.page_chrome.footers == set()


# ---------------------------------------------------------------------------
# Wave 20 integration: segment_extracted_document picks up per-page pages
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestSegmenterPageAttribution:
    def test_raw_block_page_populated_from_form_feeds(self):
        # Build an ExtractedDocument directly — no pdftotext needed.
        raw = (
            "Page 1 content one.\n\nPage 1 content two."
            f"{_FORM_FEED}"
            "Page 2 content one.\n\nPage 2 content two."
            f"{_FORM_FEED}"
            "Page 3 content one.\n\nPage 3 content two."
        )
        doc = ExtractedDocument(raw_text=raw, source_pdf="test.pdf")
        blocks = segment_extracted_document(doc)
        # Every block has a page set (form-feeds make it knowable).
        for block in blocks:
            assert block.page is not None
        assert blocks[0].page == 1
        assert blocks[-1].page == 3

    def test_page_label_stamped_from_chrome_page_number_lines(self):
        # Build a PageChrome with a page-number map, feed it into
        # segment_extracted_document via an ExtractedDocument.
        raw = (
            "Body of page 1\n\nMore body page 1"
            f"{_FORM_FEED}"
            "Body of page 2\n\nMore body page 2"
            f"{_FORM_FEED}"
            "Body of page 3\n\nMore body page 3"
            f"{_FORM_FEED}"
            "Body of page 4\n\nMore body page 4"
        )
        chrome = PageChrome(
            headers={"book title"},
            page_number_lines={
                1: "Book Title 42",
                2: "Book Title 43",
                3: "Book Title 44",
                4: "Book Title 45",
            },
        )
        doc = ExtractedDocument(
            raw_text=raw,
            source_pdf="x.pdf",
            page_chrome=chrome,
        )
        blocks = segment_extracted_document(doc)
        # Every block in the prose gets stamped with its page's numeric
        # label (from the chrome mapping).
        for block in blocks:
            if block.page == 1:
                assert block.extra.get("page_label") == "42"
            elif block.page == 2:
                assert block.extra.get("page_label") == "43"

    def test_no_page_label_when_chrome_has_no_mapping(self):
        raw = (
            "Body line 1\n\nBody line 2"
            f"{_FORM_FEED}"
            "Body line 3\n\nBody line 4"
        )
        doc = ExtractedDocument(
            raw_text=raw,
            source_pdf="x.pdf",
            page_chrome=PageChrome(),
        )
        blocks = segment_extracted_document(doc)
        for block in blocks:
            assert "page_label" not in block.extra


# ---------------------------------------------------------------------------
# Wave 20 integration: block_templates emits data-dart-pages
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestTemplateEmissionPluralPages:
    def test_data_dart_pages_emitted_from_raw_page(self):
        from DART.converter.block_roles import ClassifiedBlock, RawBlock
        from DART.converter.block_templates import _provenance_attrs

        raw = RawBlock(text="Chapter 1 opener", block_id="abc123", page=5)
        clf = ClassifiedBlock(
            raw=raw,
            role=BlockRole.CHAPTER_OPENER,
            confidence=0.9,
            attributes={},
            classifier_source="heuristic",
        )
        attrs = _provenance_attrs(clf)
        assert 'data-dart-pages="5"' in attrs
        # Singular form must no longer be emitted.
        assert 'data-dart-page="' not in attrs

    def test_data_dart_pages_prefers_page_label(self):
        from DART.converter.block_roles import ClassifiedBlock, RawBlock
        from DART.converter.block_templates import _provenance_attrs

        raw = RawBlock(
            text="Body",
            block_id="abc123",
            page=5,
            extra={"page_label": "42"},
        )
        clf = ClassifiedBlock(
            raw=raw,
            role=BlockRole.PARAGRAPH,
            confidence=0.8,
            attributes={},
            classifier_source="heuristic",
        )
        # Note: _provenance_attrs is only called by WRAPPER templates.
        # Paragraph is a leaf — no data-dart-* attributes at all — so we
        # test the helper directly.
        attrs = _provenance_attrs(clf)
        # Label wins over raw page.
        assert 'data-dart-pages="42"' in attrs
        assert 'data-dart-pages="5"' not in attrs

    def test_data_dart_pages_omitted_when_no_page(self):
        from DART.converter.block_roles import ClassifiedBlock, RawBlock
        from DART.converter.block_templates import _provenance_attrs

        raw = RawBlock(text="Body", block_id="abc123", page=None)
        clf = ClassifiedBlock(
            raw=raw,
            role=BlockRole.PARAGRAPH,
            confidence=0.8,
            attributes={},
            classifier_source="heuristic",
        )
        attrs = _provenance_attrs(clf)
        assert "data-dart-pages" not in attrs
