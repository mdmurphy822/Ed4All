"""Wave 16 tests for ``DART.converter.extractor``.

Coverage: pdftotext hard dependency + graceful degradation across
optional extractors (pdfplumber, tesseract, figure/alt pipeline).

Every test monkeypatches the optional-dependency call-surface so
neither pdfplumber nor tesseract nor pytesseract nor PyMuPDF need to
be installed at CI time. pdftotext itself is stubbed via a fake
``subprocess.run``.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import List, Optional

import pytest

from DART.converter import extractor as extractor_module
from DART.converter.extractor import (
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    extract_document,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fake_pdftotext_success(stdout: bytes):
    """Build a fake ``subprocess.run`` emitting ``stdout`` successfully."""

    def _run(cmd, *args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    return _run


def _fake_pdftotext_failure(returncode: int = 2, stderr: bytes = b"boom"):
    def _run(cmd, *args, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=b"", stderr=stderr)

    return _run


@pytest.fixture
def no_optional_deps(monkeypatch):
    """Ensure every optional dependency reads as unavailable.

    Simulates a bare CI environment: no pdfplumber, no tesseract
    binary, no PyMuPDF, no pytesseract. The extractor should still
    return a valid ``ExtractedDocument`` populated only from pdftotext.
    """
    monkeypatch.setattr(
        extractor_module, "_extract_tables_pdfplumber", lambda pdf_path: []
    )
    monkeypatch.setattr(extractor_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        extractor_module, "_extract_figures", lambda pdf_path, *, llm=None: []
    )
    yield


# ---------------------------------------------------------------------------
# pdftotext hard dependency
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestPdftotextHardDependency:
    def test_pdftotext_only_success(self, monkeypatch, tmp_path, no_optional_deps):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(
            subprocess,
            "run",
            _fake_pdftotext_success(b"Hello world.\n\nSecond paragraph."),
        )

        doc = extract_document(str(pdf))

        assert isinstance(doc, ExtractedDocument)
        assert doc.source_pdf == str(pdf)
        assert "Hello world." in doc.raw_text
        assert doc.tables == []
        assert doc.figures == []
        assert doc.ocr_text is None
        assert doc.pages_count >= 1

    def test_pdftotext_failure_raises_runtime_error(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_failure())

        with pytest.raises(RuntimeError, match="pdftotext failed"):
            extract_document(str(pdf))

    def test_pdftotext_missing_binary_raises(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        def _raise_missing(*args, **kwargs):
            raise FileNotFoundError("pdftotext")

        monkeypatch.setattr(subprocess, "run", _raise_missing)

        with pytest.raises(RuntimeError, match="pdftotext binary"):
            extract_document(str(pdf))

    def test_pages_count_form_feed_markers(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess,
            "run",
            _fake_pdftotext_success(b"page one\x0cpage two\x0cpage three"),
        )
        doc = extract_document(str(pdf))
        assert doc.pages_count == 3


# ---------------------------------------------------------------------------
# pdfplumber (tables)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestPdfplumberTables:
    def test_tables_populated_when_pdfplumber_works(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext_success(b"Some prose.")
        )
        # pdfplumber missing -> no import? patch the helper directly.
        fake_table = ExtractedTable(
            page=1,
            bbox=(0.0, 0.0, 100.0, 100.0),
            header_rows=[["Name", "Age"]],
            body_rows=[["Alice", "30"], ["Bob", "42"]],
            caption="Table 1: Demographics",
        )
        monkeypatch.setattr(
            extractor_module,
            "_extract_tables_pdfplumber",
            lambda pdf_path: [fake_table],
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            extractor_module, "_extract_figures", lambda pdf_path, *, llm=None: []
        )

        doc = extract_document(str(pdf))

        assert len(doc.tables) == 1
        assert doc.tables[0].header_rows == [["Name", "Age"]]
        assert doc.tables[0].body_rows == [["Alice", "30"], ["Bob", "42"]]
        assert doc.tables[0].caption == "Table 1: Demographics"

    def test_pdfplumber_broken_degrades_to_empty(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext_success(b"Prose.")
        )

        def _boom(pdf_path):
            raise RuntimeError("pdfplumber exploded")

        monkeypatch.setattr(extractor_module, "_extract_tables_pdfplumber", _boom)
        monkeypatch.setattr(extractor_module.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            extractor_module, "_extract_figures", lambda pdf_path, *, llm=None: []
        )

        # The extractor should catch the helper's exception and degrade to [].
        doc = extract_document(str(pdf))
        assert doc.tables == []


# ---------------------------------------------------------------------------
# Tesseract OCR
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestOcrPath:
    def test_ocr_skipped_when_tesseract_absent(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"Text."))
        # ``no_optional_deps`` already patches ``shutil.which`` to return None.

        doc = extract_document(str(pdf))
        assert doc.ocr_text is None

    def test_ocr_text_populated_when_all_deps_present(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"Text."))
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda pdf_path: []
        )
        monkeypatch.setattr(
            extractor_module, "_extract_figures", lambda pdf_path, *, llm=None: []
        )
        # Short-circuit the OCR helper to avoid requiring tesseract / PyMuPDF.
        monkeypatch.setattr(
            extractor_module, "_extract_ocr_text", lambda pdf_path: "ocr page"
        )

        doc = extract_document(str(pdf))
        assert doc.ocr_text == "ocr page"


# ---------------------------------------------------------------------------
# Figures + alt-text routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestFigureExtraction:
    def test_figures_empty_without_llm(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"T."))

        doc = extract_document(str(pdf))
        assert doc.figures == []

    def test_figure_alt_text_populated_when_llm_provided(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"T."))
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda pdf_path: []
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda name: None)

        # Short-circuit figure extraction with fake output that verifies
        # the ``llm`` kwarg propagates.
        seen_llm = []

        def _fake_figures(pdf_path, *, llm=None):
            seen_llm.append(llm)
            return [
                ExtractedFigure(
                    page=2,
                    bbox=(0.0, 0.0, 10.0, 10.0),
                    image_path="",
                    alt_text="A histogram of sales by quarter",
                    caption="Figure 1: Quarterly sales",
                )
            ]

        monkeypatch.setattr(extractor_module, "_extract_figures", _fake_figures)

        from MCP.orchestrator.llm_backend import MockBackend

        backend = MockBackend(responses=[])
        doc = extract_document(str(pdf), llm=backend)

        assert seen_llm == [backend]
        assert len(doc.figures) == 1
        assert doc.figures[0].alt_text == "A histogram of sales by quarter"

    def test_figure_extraction_broken_degrades_to_empty(
        self, monkeypatch, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"T."))
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda pdf_path: []
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda name: None)

        def _boom(pdf_path, *, llm=None):
            raise RuntimeError("PyMuPDF exploded")

        monkeypatch.setattr(extractor_module, "_extract_figures", _boom)

        doc = extract_document(str(pdf))
        assert doc.figures == []


# ---------------------------------------------------------------------------
# Dataclass defaults / shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestDataclassDefaults:
    def test_extracted_document_defaults(self):
        doc = ExtractedDocument(raw_text="t", source_pdf="/tmp/x.pdf")
        assert doc.tables == []
        assert doc.figures == []
        assert doc.ocr_text is None
        assert doc.pages_count == 0

    def test_extracted_table_defaults(self):
        tbl = ExtractedTable(page=1, bbox=(0.0, 0.0, 1.0, 1.0))
        assert tbl.header_rows == []
        assert tbl.body_rows == []
        assert tbl.caption is None

    def test_extracted_figure_defaults(self):
        fig = ExtractedFigure(page=1, bbox=(0.0, 0.0, 1.0, 1.0))
        assert fig.alt_text is None
        assert fig.caption is None
        assert fig.image_path == ""
