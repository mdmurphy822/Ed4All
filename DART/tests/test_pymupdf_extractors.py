"""Wave 18 tests for PyMuPDF-backed peer extractors.

Each helper in ``DART.converter.extractor`` that touches PyMuPDF is
exercised with a monkeypatched ``fitz.Document`` fake so neither
PyMuPDF nor a real PDF is required at CI time. Same approach as the
Wave 17 figure-persistence suite.

Coverage:

* ``_extract_toc_pymupdf`` — happy path + missing API + empty TOC.
* ``_extract_metadata_pymupdf`` — normalisation + ISO-8601 date coerce.
* ``_extract_text_spans_pymupdf`` — spans collected, flags decoded.
* ``_extract_links_pymupdf`` — URI + internal goto translation.
* ``_find_tables_pymupdf`` — table harvesting + missing API degrade.
* ``extract_document`` integration — Wave 18 fields populated; table
  reconciliation prefers pdfplumber over PyMuPDF; PyMuPDF unavailable
  leaves fields empty; ``source`` attribute set on each extracted
  table.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import List, Optional

import pytest

from DART.converter import extractor as extractor_module
from DART.converter.extractor import (
    ExtractedDocument,
    ExtractedLink,
    ExtractedTable,
    ExtractedTOCEntry,
    ExtractedTextSpan,
    _extract_links_pymupdf,
    _extract_metadata_pymupdf,
    _extract_text_spans_pymupdf,
    _extract_toc_pymupdf,
    _find_tables_pymupdf,
    _normalise_pdf_date,
    extract_document,
    median_body_font_size,
)


# ---------------------------------------------------------------------------
# Fake fitz.Document + helpers
# ---------------------------------------------------------------------------


class _FakePage:
    """A test-only PyMuPDF-ish page with configurable helpers."""

    def __init__(
        self,
        *,
        page_dict=None,
        links=None,
        find_tables=None,
        expose_find_tables: bool = True,
    ):
        self._page_dict = page_dict or {"blocks": []}
        self._links = links or []
        self._find_tables = find_tables
        self._expose_find_tables = expose_find_tables

    def get_text(self, mode):
        assert mode == "dict"
        return self._page_dict

    def get_links(self):
        return self._links

    # ``find_tables`` is attached dynamically below so we can simulate
    # older PyMuPDF versions that lack the method entirely.
    def __getattr__(self, name):
        if name == "find_tables":
            if not self._expose_find_tables:
                raise AttributeError("find_tables")
            return self._find_tables
        raise AttributeError(name)


class _FakeFitzDoc:
    """A test-only PyMuPDF-ish Document that behaves like fitz.Document."""

    def __init__(self, pages: List[_FakePage], *, toc=None, metadata=None):
        self.pages = pages
        self._toc = toc if toc is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.closed = False

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        return self.pages[idx]

    def get_toc(self, simple=True):
        assert simple is True
        return self._toc

    def close(self):
        self.closed = True


def _fake_pdftotext_success(stdout: bytes):
    def _run(cmd, *args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    return _run


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestExtractTocPymupdf:
    def test_returns_entries_for_typical_outline(self):
        doc = _FakeFitzDoc(
            pages=[],
            toc=[
                [1, "Chapter 1: Introduction", 3],
                [2, "1.1 Motivation", 4],
                [2, "1.2 Related Work", 6],
                [1, "Chapter 2: Methods", 10],
            ],
        )
        entries = _extract_toc_pymupdf(doc)
        assert len(entries) == 4
        assert entries[0].level == 1
        assert entries[0].title == "Chapter 1: Introduction"
        assert entries[0].page == 3
        assert entries[1].level == 2
        assert entries[2].title == "1.2 Related Work"

    def test_degrades_when_get_toc_raises(self):
        class _Raising:
            def get_toc(self, simple=True):
                raise RuntimeError("broken")

        entries = _extract_toc_pymupdf(_Raising())
        assert entries == []

    def test_empty_toc_returns_empty_list(self):
        doc = _FakeFitzDoc(pages=[], toc=[])
        assert _extract_toc_pymupdf(doc) == []

    def test_skips_rows_with_invalid_shape(self):
        doc = _FakeFitzDoc(
            pages=[],
            toc=[
                [1, "Valid", 5],
                ["bad", "data", None],
                [1, "", 3],  # empty title skipped
                [1, "Another Valid", 7],
            ],
        )
        entries = _extract_toc_pymupdf(doc)
        titles = [e.title for e in entries]
        assert titles == ["Valid", "Another Valid"]


@pytest.mark.unit
@pytest.mark.dart
class TestExtractMetadataPymupdf:
    def test_normalises_typical_metadata(self):
        doc = _FakeFitzDoc(
            pages=[],
            metadata={
                "title": "  Educational Foundations  ",
                "author": "J. Smith",
                "subject": "Online Learning",
                "keywords": "learning, digital, open",
                "creator": "TeX",
                "producer": "pdfTeX-1.40",
                "creationDate": "D:20230315091500-07'00'",
                "modDate": "D:20240101000000Z",
                "format": "PDF 1.7",
            },
        )
        meta = _extract_metadata_pymupdf(doc)
        assert meta["title"] == "Educational Foundations"
        assert meta["author"] == "J. Smith"
        assert meta["subject"] == "Online Learning"
        assert meta["creationDate"] == "2023-03-15"
        assert meta["modDate"] == "2024-01-01"
        assert meta["creator"] == "TeX"
        # ``format``/``encryption`` are intentionally dropped.
        assert "format" not in meta

    def test_empty_strings_are_dropped(self):
        doc = _FakeFitzDoc(
            pages=[],
            metadata={"title": "", "author": "  ", "subject": "Only Subject"},
        )
        meta = _extract_metadata_pymupdf(doc)
        assert meta == {"subject": "Only Subject"}

    def test_degrades_on_bad_metadata_attr(self):
        class _NoMeta:
            @property
            def metadata(self):
                raise RuntimeError("kaboom")

        assert _extract_metadata_pymupdf(_NoMeta()) == {}


@pytest.mark.unit
@pytest.mark.dart
class TestNormalisePdfDate:
    def test_standard_pdf_spec_date(self):
        assert _normalise_pdf_date("D:20230315091500-07'00'") == "2023-03-15"

    def test_iso_date_passes_through(self):
        assert _normalise_pdf_date("2024-02-10") == "2024-02-10"

    def test_none_returns_none(self):
        assert _normalise_pdf_date(None) is None

    def test_empty_string_returns_none(self):
        assert _normalise_pdf_date("   ") is None


@pytest.mark.unit
@pytest.mark.dart
class TestExtractTextSpansPymupdf:
    def test_spans_collected_with_font_metadata(self):
        page_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "CHAPTER 1",
                                    "size": 24.0,
                                    "font": "Times-Bold",
                                    "flags": 16,  # bold bit set
                                    "bbox": (72, 72, 500, 100),
                                }
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "Body paragraph here.",
                                    "size": 11.0,
                                    "font": "Times-Roman",
                                    "flags": 0,
                                    "bbox": (72, 120, 500, 135),
                                }
                            ]
                        },
                    ],
                }
            ]
        }
        doc = _FakeFitzDoc(pages=[_FakePage(page_dict=page_dict)])
        spans = _extract_text_spans_pymupdf(doc)
        assert len(spans) == 2
        assert spans[0].text == "CHAPTER 1"
        assert spans[0].font_size == 24.0
        assert spans[0].is_bold is True
        assert spans[0].page == 1
        assert spans[1].is_bold is False

    def test_image_blocks_are_skipped(self):
        page_dict = {"blocks": [{"type": 1, "lines": []}]}
        doc = _FakeFitzDoc(pages=[_FakePage(page_dict=page_dict)])
        assert _extract_text_spans_pymupdf(doc) == []

    def test_empty_text_spans_are_skipped(self):
        page_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {"spans": [{"text": "", "size": 10.0, "flags": 0}]}
                    ],
                }
            ]
        }
        doc = _FakeFitzDoc(pages=[_FakePage(page_dict=page_dict)])
        assert _extract_text_spans_pymupdf(doc) == []


@pytest.mark.unit
@pytest.mark.dart
class TestExtractLinksPymupdf:
    def test_uri_and_goto_links_translated(self):
        page = _FakePage(
            links=[
                {
                    "kind": 2,
                    "from": (10, 20, 30, 40),
                    "uri": "https://example.org",
                },
                {
                    "kind": 1,
                    "from": (50, 60, 70, 80),
                    "page": 3,  # 0-indexed target -> 4
                },
            ]
        )
        doc = _FakeFitzDoc(pages=[page])
        links = _extract_links_pymupdf(doc)
        assert len(links) == 2
        assert links[0].uri == "https://example.org"
        assert links[0].dest_page is None
        assert links[1].uri is None
        assert links[1].dest_page == 4

    def test_degrades_when_get_links_raises(self):
        class _RaisingPage:
            def get_links(self):
                raise RuntimeError("boom")

        doc = _FakeFitzDoc(pages=[_RaisingPage()])
        assert _extract_links_pymupdf(doc) == []


@pytest.mark.unit
@pytest.mark.dart
class TestFindTablesPymupdf:
    def test_returns_tables_with_pymupdf_source(self):
        class _FakeTable:
            bbox = (0.0, 0.0, 400.0, 200.0)

            def extract(self):
                return [["A", "B"], ["1", "2"], ["3", "4"]]

        def _find():
            return SimpleNamespace(tables=[_FakeTable()])

        page = _FakePage(find_tables=_find)
        doc = _FakeFitzDoc(pages=[page])
        tables = _find_tables_pymupdf(doc)
        assert len(tables) == 1
        assert tables[0].source == "pymupdf"
        assert tables[0].header_rows == [["A", "B"]]
        assert tables[0].body_rows == [["1", "2"], ["3", "4"]]
        assert tables[0].page == 1

    def test_missing_find_tables_degrades(self):
        page = _FakePage(expose_find_tables=False)
        doc = _FakeFitzDoc(pages=[page])
        assert _find_tables_pymupdf(doc) == []

    def test_find_tables_raises_degrades(self):
        def _raise():
            raise AttributeError("not supported on this PyMuPDF")

        page = _FakePage(find_tables=_raise)
        doc = _FakeFitzDoc(pages=[page])
        assert _find_tables_pymupdf(doc) == []

    def test_iterable_style_table_finder(self):
        """Some PyMuPDF versions return an iterable directly, not .tables."""

        class _FakeTable:
            bbox = (0.0, 0.0, 100.0, 100.0)

            def extract(self):
                return [["hdr"], ["row"]]

        def _find():
            return [_FakeTable()]

        page = _FakePage(find_tables=_find)
        doc = _FakeFitzDoc(pages=[page])
        tables = _find_tables_pymupdf(doc)
        assert len(tables) == 1
        assert tables[0].source == "pymupdf"


@pytest.mark.unit
@pytest.mark.dart
class TestMedianBodyFontSize:
    def test_returns_median_of_sizes(self):
        spans = [
            ExtractedTextSpan(
                page=1, bbox=(0, 0, 1, 1), text="a", font_size=10.0,
                font_name="", is_bold=False, is_italic=False,
            ),
            ExtractedTextSpan(
                page=1, bbox=(0, 0, 1, 1), text="b", font_size=11.0,
                font_name="", is_bold=False, is_italic=False,
            ),
            ExtractedTextSpan(
                page=1, bbox=(0, 0, 1, 1), text="c", font_size=12.0,
                font_name="", is_bold=False, is_italic=False,
            ),
        ]
        assert median_body_font_size(spans) == 11.0

    def test_empty_returns_none(self):
        assert median_body_font_size([]) is None


# ---------------------------------------------------------------------------
# extract_document integration (Wave 18 fields + reconciliation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestExtractDocumentWave18:
    def _install_fake_fitz(self, monkeypatch, doc):
        """Patch ``extractor_module._open_pymupdf`` to return the fake doc."""
        monkeypatch.setattr(
            extractor_module,
            "_open_pymupdf",
            lambda path: doc,
        )

    def test_wave18_fields_populate_when_pymupdf_available(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext_success(b"Body prose.")
        )
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda p: []
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda n: None)
        monkeypatch.setattr(
            extractor_module,
            "_extract_figures",
            lambda *a, **kw: [],
        )

        page_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {"spans": [{"text": "Hi", "size": 11.0, "flags": 0}]}
                    ],
                }
            ]
        }
        fake_page = _FakePage(
            page_dict=page_dict,
            links=[{"kind": 2, "from": (1, 2, 3, 4), "uri": "https://e.org"}],
        )
        fake_doc = _FakeFitzDoc(
            pages=[fake_page],
            toc=[[1, "Chapter One", 1]],
            metadata={"title": "My Book", "creationDate": "D:20231201000000Z"},
        )
        self._install_fake_fitz(monkeypatch, fake_doc)

        out = extract_document(str(pdf))
        assert len(out.toc) == 1
        assert out.toc[0].title == "Chapter One"
        assert out.pdf_metadata.get("title") == "My Book"
        assert out.pdf_metadata.get("creationDate") == "2023-12-01"
        assert len(out.text_spans) == 1
        assert out.text_spans[0].text == "Hi"
        assert len(out.links) == 1
        assert out.links[0].uri == "https://e.org"

    def test_wave18_fields_empty_when_pymupdf_unavailable(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext_success(b"Prose only.")
        )
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda p: []
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda n: None)
        monkeypatch.setattr(
            extractor_module, "_extract_figures", lambda *a, **kw: []
        )
        # Simulate PyMuPDF unavailable.
        monkeypatch.setattr(extractor_module, "_open_pymupdf", lambda p: None)

        out = extract_document(str(pdf))
        assert out.toc == []
        assert out.pdf_metadata == {}
        assert out.text_spans == []
        assert out.links == []

    def test_pdfplumber_tables_win_over_pymupdf(self, monkeypatch, tmp_path):
        """Reconciliation: pdfplumber non-empty blocks PyMuPDF fallback."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext_success(b"text")
        )
        pdfp_table = ExtractedTable(
            page=1,
            bbox=(0, 0, 1, 1),
            header_rows=[["H"]],
            body_rows=[["b"]],
            source="pdfplumber",
        )
        monkeypatch.setattr(
            extractor_module,
            "_extract_tables_pdfplumber",
            lambda p: [pdfp_table],
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda n: None)
        monkeypatch.setattr(
            extractor_module, "_extract_figures", lambda *a, **kw: []
        )

        # PyMuPDF would return tables, but they should be ignored.
        class _FakeTable:
            bbox = (0, 0, 1, 1)

            def extract(self):
                return [["X"], ["y"]]

        pymupdf_page = _FakePage(
            find_tables=lambda: SimpleNamespace(tables=[_FakeTable()])
        )
        fake_doc = _FakeFitzDoc(pages=[pymupdf_page])
        self._install_fake_fitz(monkeypatch, fake_doc)

        out = extract_document(str(pdf))
        assert len(out.tables) == 1
        assert out.tables[0].source == "pdfplumber"
        assert out.tables[0].header_rows == [["H"]]

    def test_pymupdf_tables_used_when_pdfplumber_empty(
        self, monkeypatch, tmp_path
    ):
        """Reconciliation: pdfplumber=[] AND PyMuPDF present -> PyMuPDF wins."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext_success(b"text")
        )
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda p: []
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda n: None)
        monkeypatch.setattr(
            extractor_module, "_extract_figures", lambda *a, **kw: []
        )

        class _FakeTable:
            bbox = (0, 0, 1, 1)

            def extract(self):
                return [["Name", "Age"], ["Alice", "30"]]

        pymupdf_page = _FakePage(
            find_tables=lambda: SimpleNamespace(tables=[_FakeTable()])
        )
        fake_doc = _FakeFitzDoc(pages=[pymupdf_page])
        self._install_fake_fitz(monkeypatch, fake_doc)

        out = extract_document(str(pdf))
        assert len(out.tables) == 1
        assert out.tables[0].source == "pymupdf"
        assert out.tables[0].header_rows == [["Name", "Age"]]
        assert out.tables[0].body_rows == [["Alice", "30"]]

    def test_extracted_table_source_default_pdfplumber(self):
        """Data-class default preserves backward compat."""
        tbl = ExtractedTable(page=1, bbox=(0, 0, 1, 1))
        assert tbl.source == "pdfplumber"

    def test_extracted_document_wave18_defaults(self):
        doc = ExtractedDocument(raw_text="x", source_pdf="/t/x.pdf")
        assert doc.toc == []
        assert doc.pdf_metadata == {}
        assert doc.text_spans == []
        assert doc.links == []
