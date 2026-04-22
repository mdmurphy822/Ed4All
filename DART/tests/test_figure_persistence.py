"""Wave 17 figure-persistence tests.

Targets the ``figures_dir``-backed persistence layer that writes the
bytes returned by the PyMuPDF extractor into an on-disk directory so
the emitted HTML has a real ``<img src>`` to point at.

All tests use monkeypatching; none require PyMuPDF at CI time.
"""

from __future__ import annotations

import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from DART.converter import extractor as extractor_module
from DART.converter.extractor import (
    _figure_filename,
    _find_caption_for_figure,
    _image_ext_from_format,
    _persist_image_bytes,
    extract_document,
)


def _fake_pdftotext(stdout: bytes):
    def _run(cmd, *args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    return _run


def _stub_pymupdf_with(monkeypatch, images):
    """Install a fake ``DART.pdf_converter.image_extractor`` module."""
    import sys as _sys

    def _factory(pdf_path):
        class _Ex:
            def extract_all(inner_self):
                return images

        return _Ex()

    monkeypatch.setitem(
        _sys.modules,
        "DART.pdf_converter.image_extractor",
        SimpleNamespace(PDFImageExtractor=_factory),
    )


def _stub_no_tables_no_ocr(monkeypatch):
    monkeypatch.setattr(
        extractor_module, "_extract_tables_pdfplumber", lambda p: []
    )
    monkeypatch.setattr(extractor_module.shutil, "which", lambda n: None)


@pytest.mark.unit
@pytest.mark.dart
class TestFilenameComputation:
    """Stable, deterministic filename rules."""

    def test_filename_format_is_page_dash_hash8_dot_ext(self):
        data = b"abc"
        name = _figure_filename(7, data, "png")
        digest8 = hashlib.sha256(data).hexdigest()[:8]
        assert name == f"0007-{digest8}.png"

    def test_filename_pads_page_to_four_digits(self):
        name = _figure_filename(1, b"x", "png")
        assert name.startswith("0001-")
        wide = _figure_filename(1234, b"x", "png")
        assert wide.startswith("1234-")

    def test_same_bytes_yield_same_filename(self):
        a = _figure_filename(5, b"same-bytes", "png")
        b = _figure_filename(5, b"same-bytes", "png")
        assert a == b

    def test_different_bytes_yield_different_hash(self):
        a = _figure_filename(5, b"one", "png")
        b = _figure_filename(5, b"two", "png")
        assert a != b


@pytest.mark.unit
@pytest.mark.dart
class TestImageExtRouting:
    """Known formats map cleanly; unknown formats fall back to png."""

    def test_png_ext(self):
        assert _image_ext_from_format("png") == "png"
        assert _image_ext_from_format("PNG") == "png"

    def test_jpeg_ext_preserved(self):
        assert _image_ext_from_format("jpeg") == "jpeg"

    def test_jpg_normalised_to_jpg(self):
        assert _image_ext_from_format("jpg") == "jpg"

    def test_unknown_format_falls_back_to_png(self):
        assert _image_ext_from_format("exotic") == "png"
        assert _image_ext_from_format(None) == "png"
        assert _image_ext_from_format("") == "png"


@pytest.mark.unit
@pytest.mark.dart
class TestPersistImageBytes:
    """Direct unit test of the persistence helper."""

    def test_writes_bytes_to_target_directory(self, tmp_path):
        out = tmp_path / "figs"
        filename = _persist_image_bytes(
            b"hello", out, page=2, fmt="png"
        )
        assert filename.endswith(".png")
        assert (out / filename).read_bytes() == b"hello"

    def test_idempotent_same_bytes_same_filename(self, tmp_path):
        out = tmp_path / "figs"
        a = _persist_image_bytes(b"xx", out, page=1, fmt="png")
        b = _persist_image_bytes(b"xx", out, page=1, fmt="png")
        assert a == b
        # Only one on-disk file.
        assert len(list(out.iterdir())) == 1

    def test_respects_jpeg_extension(self, tmp_path):
        out = tmp_path / "figs"
        filename = _persist_image_bytes(
            b"jpeg-bytes", out, page=1, fmt="jpeg"
        )
        assert filename.endswith(".jpeg")


@pytest.mark.unit
@pytest.mark.dart
class TestExtractDocumentFiguresDir:
    """``extract_document(..., figures_dir=...)`` integrates persistence."""

    def test_figures_dir_none_backward_compat(
        self, monkeypatch, tmp_path
    ):
        """Default (figures_dir=None) leaves image_path empty."""
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext(b"Prose."))
        _stub_no_tables_no_ocr(monkeypatch)
        _stub_pymupdf_with(
            monkeypatch,
            [
                SimpleNamespace(
                    page=1,
                    bbox=(0, 0, 1, 1),
                    data=b"img-bytes",
                    format="png",
                    nearby_caption="",
                )
            ],
        )
        doc = extract_document(str(pdf))
        assert doc.figures[0].image_path == ""

    def test_figures_dir_set_persists_and_sets_path(
        self, monkeypatch, tmp_path
    ):
        """figures_dir=<path> writes bytes + sets relative image_path."""
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext(b"Prose."))
        _stub_no_tables_no_ocr(monkeypatch)
        _stub_pymupdf_with(
            monkeypatch,
            [
                SimpleNamespace(
                    page=2,
                    bbox=(0, 0, 1, 1),
                    data=b"bytes-here",
                    format="png",
                    nearby_caption="",
                )
            ],
        )
        figures_dir = tmp_path / "my_figs"
        doc = extract_document(str(pdf), figures_dir=figures_dir)
        fig = doc.figures[0]
        assert fig.image_path != ""
        # Filename only (no directory).
        assert "/" not in fig.image_path
        assert (figures_dir / fig.image_path).read_bytes() == b"bytes-here"

    def test_figures_with_no_data_stay_empty(self, monkeypatch, tmp_path):
        """Vector regions with empty data must not crash; path stays ""."""
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext(b"Prose."))
        _stub_no_tables_no_ocr(monkeypatch)
        _stub_pymupdf_with(
            monkeypatch,
            [
                SimpleNamespace(
                    page=1,
                    bbox=(0, 0, 1, 1),
                    data=b"",  # empty
                    format="png",
                    nearby_caption="",
                )
            ],
        )
        figures_dir = tmp_path / "figs"
        doc = extract_document(str(pdf), figures_dir=figures_dir)
        assert doc.figures[0].image_path == ""

    def test_caption_bound_from_pdftotext_page(
        self, monkeypatch, tmp_path
    ):
        """Per-page caption scrape populates figure.caption."""
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        # Two pages separated by form-feed.
        raw = (
            b"Some prose on page one.\n"
            b"Figure 1.1: First page caption.\n"
            b"\x0c"
            b"Second page prose.\n"
            b"Figure 2.1: Second page caption.\n"
        )
        monkeypatch.setattr(subprocess, "run", _fake_pdftotext(raw))
        _stub_no_tables_no_ocr(monkeypatch)
        _stub_pymupdf_with(
            monkeypatch,
            [
                SimpleNamespace(
                    page=1,
                    bbox=(0, 0, 1, 1),
                    data=b"p1",
                    format="png",
                    nearby_caption="",
                ),
                SimpleNamespace(
                    page=2,
                    bbox=(0, 0, 1, 1),
                    data=b"p2",
                    format="png",
                    nearby_caption="",
                ),
            ],
        )
        figures_dir = tmp_path / "figs"
        doc = extract_document(str(pdf), figures_dir=figures_dir)
        captions = [fig.caption for fig in doc.figures]
        assert any(c and c.startswith("Figure 1.1:") for c in captions)
        assert any(c and c.startswith("Figure 2.1:") for c in captions)

    def test_caption_falls_back_to_nearby_when_no_match(
        self, monkeypatch, tmp_path
    ):
        """When pdftotext has no Figure N: line, use extractor nearby_caption."""
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext(b"Plain prose only.\n")
        )
        _stub_no_tables_no_ocr(monkeypatch)
        _stub_pymupdf_with(
            monkeypatch,
            [
                SimpleNamespace(
                    page=1,
                    bbox=(0, 0, 1, 1),
                    data=b"bytes",
                    format="png",
                    nearby_caption="PyMuPDF nearby caption.",
                )
            ],
        )
        figures_dir = tmp_path / "figs"
        doc = extract_document(str(pdf), figures_dir=figures_dir)
        assert doc.figures[0].caption == "PyMuPDF nearby caption."

    def test_caption_none_when_neither_source_present(
        self, monkeypatch, tmp_path
    ):
        """Graceful degradation: caption left None when we have nothing."""
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(
            subprocess, "run", _fake_pdftotext(b"Some ordinary prose.\n")
        )
        _stub_no_tables_no_ocr(monkeypatch)
        _stub_pymupdf_with(
            monkeypatch,
            [
                SimpleNamespace(
                    page=1,
                    bbox=(0, 0, 1, 1),
                    data=b"bytes",
                    format="png",
                    nearby_caption="",  # no upstream caption
                )
            ],
        )
        figures_dir = tmp_path / "figs"
        doc = extract_document(str(pdf), figures_dir=figures_dir)
        assert doc.figures[0].caption is None


@pytest.mark.unit
@pytest.mark.dart
class TestCaptionDetectorEdgeCases:
    """Supplementary edge cases for :func:`_find_caption_for_figure`."""

    def test_empty_text_returns_none(self):
        assert _find_caption_for_figure("", page_number=1) is None

    def test_whitespace_only_returns_none(self):
        assert _find_caption_for_figure("   \n\n  ", page_number=1) is None

    def test_already_taken_skips_matching_line(self):
        page_text = "Figure 1: Only caption here.\n"
        claims: set = {0}  # pre-claim the only line
        assert _find_caption_for_figure(
            page_text, page_number=1, already_taken=claims
        ) is None
