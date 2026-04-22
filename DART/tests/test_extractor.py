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
        extractor_module, "_extract_figures", lambda pdf_path, *, llm=None, **_: []
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
            extractor_module, "_extract_figures", lambda pdf_path, *, llm=None, **_: []
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
            extractor_module, "_extract_figures", lambda pdf_path, *, llm=None, **_: []
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
            extractor_module, "_extract_figures", lambda pdf_path, *, llm=None, **_: []
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
        # the ``llm`` kwarg propagates. Wave 17 adds ``figures_dir`` /
        # ``page_text_index`` kwargs to the helper; accept them via
        # ``**kwargs`` so this test remains signature-agnostic.
        seen_llm = []

        def _fake_figures(pdf_path, *, llm=None, **kwargs):
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

        def _boom(pdf_path, *, llm=None, **_):
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
        # Wave 18: new optional fields default empty.
        assert doc.toc == []
        assert doc.pdf_metadata == {}
        assert doc.text_spans == []
        assert doc.links == []

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


# ---------------------------------------------------------------------------
# Wave 17: figures_dir persistence + caption detector
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestFiguresDirPersistence:
    """Wave 17: ``figures_dir`` kwarg persists figure bytes to disk."""

    def test_figures_dir_none_keeps_image_path_empty(
        self, monkeypatch, tmp_path, no_optional_deps
    ):
        """Backward compat: ``figures_dir`` absent leaves image_path empty."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"Text."))

        # Stub _extract_figures to verify it was called without figures_dir.
        seen_kwargs = {}

        def _fake(pdf_path, *, llm=None, **kwargs):
            seen_kwargs.update(kwargs)
            return []

        monkeypatch.setattr(extractor_module, "_extract_figures", _fake)

        extract_document(str(pdf))
        # figures_dir is present in kwargs (default None)
        assert seen_kwargs.get("figures_dir") is None

    def test_figures_dir_persists_bytes_and_sets_relative_path(
        self, monkeypatch, tmp_path
    ):
        """End-to-end: figure bytes land on disk under ``figures_dir``."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"Prose."))
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda p: []
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda n: None)

        # Fake a PyMuPDF-style ExtractedImage namespace with bytes.
        img_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes"
        fake_img = SimpleNamespace(
            page=3,
            bbox=(0.0, 0.0, 10.0, 10.0),
            data=img_bytes,
            format="png",
            nearby_caption="",
        )

        # Patch PDFImageExtractor at import point inside _extract_figures.
        def _fake_extractor_factory(pdf_path):
            class _Ex:
                def extract_all(inner_self):
                    return [fake_img]

            return _Ex()

        # Instead of patching the class, patch _extract_figures' backbone
        # by intercepting the import. Simpler: provide a fake module via
        # monkeypatch.setattr on the ``image_extractor`` module attribute.
        import sys as _sys

        fake_module = SimpleNamespace(PDFImageExtractor=_fake_extractor_factory)
        monkeypatch.setitem(
            _sys.modules, "DART.pdf_converter.image_extractor", fake_module
        )

        figures_dir = tmp_path / "out_figures"
        doc = extract_document(str(pdf), figures_dir=figures_dir)

        assert len(doc.figures) == 1
        fig = doc.figures[0]
        # image_path is the relative filename (no directory prefix).
        assert fig.image_path != ""
        assert "/" not in fig.image_path
        # File exists on disk.
        assert (figures_dir / fig.image_path).exists()
        # Contents match input.
        assert (figures_dir / fig.image_path).read_bytes() == img_bytes
        # Filename follows {page:04d}-{hash8}.{ext} pattern.
        assert fig.image_path.startswith("0003-")
        assert fig.image_path.endswith(".png")

    def test_figure_persistence_is_idempotent(self, monkeypatch, tmp_path):
        """Same bytes -> same filename, no double-write on re-extract."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext_success(b"Prose."))
        monkeypatch.setattr(
            extractor_module, "_extract_tables_pdfplumber", lambda p: []
        )
        monkeypatch.setattr(extractor_module.shutil, "which", lambda n: None)

        img_bytes = b"identical-bytes-every-time"
        fake_img = SimpleNamespace(
            page=1,
            bbox=(0, 0, 1, 1),
            data=img_bytes,
            format="png",
            nearby_caption="",
        )

        def _fake_factory(pdf_path):
            class _Ex:
                def extract_all(inner_self):
                    return [fake_img]

            return _Ex()

        import sys as _sys

        monkeypatch.setitem(
            _sys.modules,
            "DART.pdf_converter.image_extractor",
            SimpleNamespace(PDFImageExtractor=_fake_factory),
        )

        figures_dir = tmp_path / "figs"
        doc1 = extract_document(str(pdf), figures_dir=figures_dir)
        first_path = doc1.figures[0].image_path

        doc2 = extract_document(str(pdf), figures_dir=figures_dir)
        second_path = doc2.figures[0].image_path

        assert first_path == second_path
        # Only one file on disk.
        assert len(list(figures_dir.iterdir())) == 1


@pytest.mark.unit
@pytest.mark.dart
class TestFigureCaptionDetector:
    """Wave 17: best-effort caption scrape from pdftotext output."""

    def test_caption_matches_figure_colon(self):
        page_text = (
            "Introductory prose.\n"
            "Figure 1.2: A histogram of quarterly sales figures.\n"
            "More prose.\n"
        )
        caption = extractor_module._find_caption_for_figure(
            page_text, page_number=1
        )
        assert caption is not None
        assert caption.startswith("Figure 1.2:")

    def test_caption_matches_fig_abbreviation(self):
        page_text = "Fig. 3: Sales over time.\n"
        caption = extractor_module._find_caption_for_figure(
            page_text, page_number=1
        )
        assert caption is not None
        assert "Fig. 3:" in caption

    def test_caption_matches_image_pattern(self):
        page_text = "Image 5: Reference photograph.\n"
        caption = extractor_module._find_caption_for_figure(
            page_text, page_number=1
        )
        assert caption is not None
        assert caption.startswith("Image 5:")

    def test_caption_matches_dash_separator(self):
        page_text = "Figure 4 - Outcome diagram illustrating flow.\n"
        caption = extractor_module._find_caption_for_figure(
            page_text, page_number=1
        )
        assert caption is not None
        assert "Figure 4" in caption

    def test_no_false_positives_on_prose(self):
        """Plain prose that mentions figures must not be captured."""
        page_text = (
            "The following data shows trends over time.\n"
            "We describe Figure interpretation in section 3.\n"  # no number
            "The figure illustrates our hypothesis.\n"  # lowercase prose
        )
        caption = extractor_module._find_caption_for_figure(
            page_text, page_number=1
        )
        assert caption is None

    def test_multiple_figures_claim_successive_captions(self):
        """``already_taken`` prevents multiple figures binding to one caption."""
        page_text = (
            "Figure 1.1: First caption.\n"
            "Figure 1.2: Second caption.\n"
        )
        claims: set = set()
        first = extractor_module._find_caption_for_figure(
            page_text, page_number=1, already_taken=claims
        )
        second = extractor_module._find_caption_for_figure(
            page_text, page_number=1, already_taken=claims
        )
        assert first is not None and first.startswith("Figure 1.1:")
        assert second is not None and second.startswith("Figure 1.2:")
        assert first != second
