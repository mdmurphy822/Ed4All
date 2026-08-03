"""Unit tests for scripts/harness/rasterize_pdf_scan.py.

The source PDF is a tiny synthetic blank-page PDF built in-test via pikepdf — no
course data is read from disk (repo rule: tracked tests must not reference course
data). Requires the [semantik] extra (pypdfium2 + pikepdf + PIL).
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import rasterize_pdf_scan as rp  # noqa: E402

pikepdf = pytest.importorskip("pikepdf")
pdfium = pytest.importorskip("pypdfium2")
PIL_Image = pytest.importorskip("PIL.Image")


def _make_source_pdf(path: Path, n_pages: int = 3) -> Path:
    pdf = pikepdf.Pdf.new()
    for _ in range(n_pages):
        pdf.add_blank_page(page_size=(612, 792))
    pdf.save(str(path))
    pdf.close()
    return path


# ---- parse_page_range ----------------------------------------------------


def test_parse_page_range_start_end():
    assert rp.parse_page_range("2:4", 10) == (2, 4)


def test_parse_page_range_bare_n():
    assert rp.parse_page_range("3", 10) == (3, 3)


@pytest.mark.parametrize("spec", ["4:2", "0:3", "1:11", "garbage", "", "2:", ":3"])
def test_parse_page_range_invalid(spec):
    with pytest.raises(ValueError):
        rp.parse_page_range(spec, 10)


# ---- encoding ------------------------------------------------------------


def test_encode_jpeg_dctdecode_gray():
    img = PIL_Image.new("L", (8, 6), color=128)
    data, meta = rp.encode_image(img, "jpeg", 92)
    assert meta["Filter"] == "DCTDecode"
    assert meta["ColorSpace"] == "DeviceGray"
    assert meta["Width"] == 8 and meta["Height"] == 6
    assert isinstance(data, bytes) and len(data) > 0


def test_encode_png_flatedecode_gray_length():
    img = PIL_Image.new("L", (8, 6), color=200)
    data, meta = rp.encode_image(img, "png", 92)
    assert meta["Filter"] == "FlateDecode"
    assert meta["ColorSpace"] == "DeviceGray"
    assert len(zlib.decompress(data)) == 8 * 6


def test_encode_png_flatedecode_rgb_length():
    img = PIL_Image.new("RGB", (8, 6), color=(10, 20, 30))
    data, meta = rp.encode_image(img, "png", 92)
    assert meta["Filter"] == "FlateDecode"
    assert meta["ColorSpace"] == "DeviceRGB"
    assert len(zlib.decompress(data)) == 8 * 6 * 3


# ---- end-to-end rasterize ------------------------------------------------


def test_rasterize_page_count_and_image_only(tmp_path):
    src = _make_source_pdf(tmp_path / "src.pdf", n_pages=3)
    out = tmp_path / "out" / "slice.pdf"
    entry = rp.rasterize(src, (1, 2), 144, "jpeg", 92, "gray", out)
    assert entry["page_count"] == 2

    doc = pdfium.PdfDocument(str(out))
    try:
        assert len(doc) == 2
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            assert tp.count_chars() == 0  # image-only: no text objects
            tp.close()
            page.close()
    finally:
        doc.close()


def test_rasterize_mediabox_points_for_dpi_144(tmp_path):
    src = _make_source_pdf(tmp_path / "src.pdf", n_pages=1)
    out = tmp_path / "out.pdf"
    rp.rasterize(src, (1, 1), 144, "png", 92, "gray", out)
    doc = pikepdf.Pdf.open(str(out))
    try:
        mediabox = [float(x) for x in doc.pages[0].MediaBox]
        # 612x792pt @144dpi -> 1224x1584px -> back to 612x792pt.
        assert mediabox[2] == pytest.approx(612.0, abs=1.0)
        assert mediabox[3] == pytest.approx(792.0, abs=1.0)
    finally:
        doc.close()


def test_rasterize_xobject_filter_and_colorspace(tmp_path):
    src = _make_source_pdf(tmp_path / "src.pdf", n_pages=1)
    out_jpeg = tmp_path / "j.pdf"
    rp.rasterize(src, (1, 1), 96, "jpeg", 90, "gray", out_jpeg)
    doc = pikepdf.Pdf.open(str(out_jpeg))
    try:
        xobjects = doc.pages[0].Resources.XObject
        im = xobjects.Im0
        assert str(im.Filter) == "/DCTDecode"
        assert str(im.ColorSpace) == "/DeviceGray"
    finally:
        doc.close()

    out_png_rgb = tmp_path / "p.pdf"
    rp.rasterize(src, (1, 1), 96, "png", 90, "rgb", out_png_rgb)
    doc = pikepdf.Pdf.open(str(out_png_rgb))
    try:
        im = doc.pages[0].Resources.XObject.Im0
        assert str(im.Filter) == "/FlateDecode"
        assert str(im.ColorSpace) == "/DeviceRGB"
    finally:
        doc.close()


# ---- manifest ------------------------------------------------------------


def test_manifest_created_and_replaced(tmp_path):
    src = _make_source_pdf(tmp_path / "src.pdf", n_pages=3)
    manifest = tmp_path / "manifest.json"

    e1 = rp.rasterize(src, (1, 1), 96, "jpeg", 92, "gray", tmp_path / "a.pdf")
    rp.update_manifest(manifest, e1)
    doc = json.loads(manifest.read_text())
    assert doc["schema_version"] == rp.MANIFEST_SCHEMA_VERSION
    assert set(doc["entries"]) == {"a.pdf"}
    assert doc["entries"]["a.pdf"]["sha256"] == rp._sha256(tmp_path / "a.pdf")

    e2 = rp.rasterize(src, (2, 2), 96, "jpeg", 92, "gray", tmp_path / "b.pdf")
    rp.update_manifest(manifest, e2)
    doc = json.loads(manifest.read_text())
    assert set(doc["entries"]) == {"a.pdf", "b.pdf"}

    # re-raster the same out name -> entry replaced (last-writer-wins).
    e1b = rp.rasterize(src, (3, 3), 96, "jpeg", 92, "gray", tmp_path / "a.pdf")
    rp.update_manifest(manifest, e1b)
    doc = json.loads(manifest.read_text())
    assert doc["entries"]["a.pdf"]["pages"] == [3, 3]
    assert set(doc["entries"]) == {"a.pdf", "b.pdf"}


def test_main_end_to_end_writes_pdf_and_manifest(tmp_path):
    src = _make_source_pdf(tmp_path / "src.pdf", n_pages=3)
    out = tmp_path / "scan_in-300" / "ch01.pdf"
    rc = rp.main([
        "--pdf", str(src), "--pages", "1:2", "--dpi", "150",
        "--format", "jpeg", "--out", str(out),
    ])
    assert rc == 0
    assert out.is_file()
    manifest = out.parent / "manifest.json"
    assert manifest.is_file()
    doc = json.loads(manifest.read_text())
    assert "ch01.pdf" in doc["entries"]
