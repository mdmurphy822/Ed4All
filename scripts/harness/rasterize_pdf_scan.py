#!/usr/bin/env python3
"""Rasterize a page range of a PDF into a fresh IMAGE-ONLY PDF (+ a manifest).

Phase-1 OCR-input-quality tool for the scan-conversion improvement plan
(``plans/scan-conversion-improvements-2026-07.md``). A scan corpus rasterized at
200 DPI kills small glyphs (super/subscripts); Tesseract's sweet spot is ~300
DPI. This script re-rasters a chosen page range at an operator-chosen DPI into a
NEW image-only PDF so a fresh output path sidesteps the extract disk-cache
(whose key ignores ``SEMANTIK_OCR_RENDER_SCALE`` / the Tesseract config flags —
fresh files ⇒ a clean extraction). Point the SemantiK cascade at the 300-DPI
output with ``SEMANTIK_OCR_RENDER_SCALE=4.2`` (render ≈ 300 DPI so the source
resolution is actually sampled), and gate the change with
``scripts/harness/ocr_recall_ab.py`` (+3 pts or stop) before any full re-convert.

Arg-driven + generic: NO course paths are hardcoded. ``--out`` is REQUIRED so
the script never writes next to a live conversion output by default (mirrors
``scripts/ops/semantik_rerender.py``).

Image encoding (why img2pdf is NOT a dependency):
  * ``--format jpeg`` (default): PIL JPEG at ``--quality`` (default 92 — avoid
    q85 artifacts at high DPI), embedded via ``/Filter /DCTDecode``.
  * ``--format png``: lossless raw samples ``zlib.compress``ed with
    ``/Filter /FlateDecode`` — equivalent pixel quality to a PNG container
    without needing one, which is why ``img2pdf`` is unnecessary. NB: no PNG
    predictors, so files run larger than an optimal PNG; a 300-DPI gray chapter
    is several MB/page — ``jpeg`` q92 is the practical default for full-book
    re-rasters.

Deps: ``pypdfium2`` (render) + ``PIL`` (encode) + ``pikepdf`` (PDF assembly) —
all in the SemantiK ``[semantik]`` extra — plus stdlib zlib/hashlib/json.

Memory note: ``pikepdf`` holds all compressed page streams in RAM until
``save()``, so run PER-CHAPTER (the designed usage), not one whole-book
invocation.

Example
-------
    .venv/bin/python scripts/harness/rasterize_pdf_scan.py \
        --pdf book.pdf --pages 12:41 --dpi 300 --format jpeg --quality 92 \
        --out scan_in-300/ch02.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "rasterize_pdf_scan/1.0"
MANIFEST_SCHEMA_VERSION = "rasterize_pdf_scan/1.0"


def parse_page_range(spec: str, page_count: int) -> tuple[int, int]:
    """Parse ``"START:END"`` (1-based inclusive) or a bare ``"N"``.

    Validates ``1 <= start <= end <= page_count``; raises ``ValueError`` with
    the offending spec otherwise (argparse surfaces it to the operator).
    """
    raw = (spec or "").strip()
    try:
        if ":" in raw:
            a, b = raw.split(":", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid --pages spec (want START:END or N): {spec!r}"
        ) from None
    if not (1 <= start <= end <= page_count):
        raise ValueError(
            f"--pages {spec!r} out of range for a {page_count}-page PDF "
            f"(need 1 <= start <= end <= {page_count})"
        )
    return start, end


def render_page(pdf, page_index: int, dpi: int, color: str):
    """Render one 0-based page to a PIL image at ``dpi`` (scale = dpi/72.0).

    ``color`` ``gray`` → ``.convert("L")``; ``rgb`` → ``.convert("RGB")``.
    Native pypdfium2 bitmap + page handles are closed per page (the
    ``extract_shared`` leak-fix idiom) so the loop streams.
    """
    scale = dpi / 72.0
    page = pdf[page_index]
    bm = page.render(scale=scale)
    img = bm.to_pil()
    try:
        img = img.convert("L" if color == "gray" else "RGB")
    finally:
        bm.close()
        page.close()
    return img


def encode_image(img, fmt: str, quality: int) -> tuple[bytes, dict]:
    """Encode a PIL image to embeddable PDF-stream bytes + an XObject dict.

    Returns ``(stream_bytes, meta)`` where ``meta`` carries the PDF image
    XObject dictionary fields (Width / Height / ColorSpace / BitsPerComponent /
    Subtype / Filter). ``jpeg`` → DCTDecode (PIL JPEG); ``png`` → FlateDecode
    (raw samples via ``zlib.compress``).
    """
    w, h = img.width, img.height
    color_space = "DeviceGray" if img.mode == "L" else "DeviceRGB"
    if fmt == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        filt = "DCTDecode"
    elif fmt == "png":
        data = zlib.compress(img.tobytes())
        filt = "FlateDecode"
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unsupported --format: {fmt}")
    meta = {
        "Width": w,
        "Height": h,
        "ColorSpace": color_space,
        "BitsPerComponent": 8,
        "Subtype": "Image",
        "Filter": filt,
    }
    return data, meta


def add_image_page(out_pdf, img, dpi: int, fmt: str, quality: int) -> None:
    """Append one image-only page wrapping ``img`` to ``out_pdf`` (a pikepdf.Pdf)."""
    import pikepdf

    data, meta = encode_image(img, fmt, quality)
    w_pt = meta["Width"] * 72.0 / dpi
    h_pt = meta["Height"] * 72.0 / dpi

    xobject = pikepdf.Stream(out_pdf, data)
    xobject.Type = pikepdf.Name.XObject
    xobject.Subtype = pikepdf.Name.Image
    xobject.Width = meta["Width"]
    xobject.Height = meta["Height"]
    xobject.ColorSpace = pikepdf.Name(f"/{meta['ColorSpace']}")
    xobject.BitsPerComponent = 8
    xobject.Filter = pikepdf.Name(f"/{meta['Filter']}")
    xobject = out_pdf.make_indirect(xobject)

    content = f"q {w_pt:.4f} 0 0 {h_pt:.4f} 0 0 cm /Im0 Do Q".encode("ascii")
    content_stream = out_pdf.make_indirect(pikepdf.Stream(out_pdf, content))

    page_obj = out_pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=[0, 0, round(w_pt, 4), round(h_pt, 4)],
            Resources=pikepdf.Dictionary(
                XObject=pikepdf.Dictionary(Im0=xobject),
            ),
            Contents=content_stream,
        )
    )
    out_pdf.pages.append(pikepdf.Page(page_obj))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(manifest_path: Path, entry: dict) -> None:
    """Read-modify-write the sibling manifest, keyed by the output basename.

    Last-writer-wins on the same out name. Re-reads immediately before writing
    to narrow (not close) the concurrent-invocation window — this is an
    operator script, single-invocation-at-a-time.
    """
    doc: dict = {"schema_version": MANIFEST_SCHEMA_VERSION, "entries": {}}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
                doc = loaded
                doc["schema_version"] = MANIFEST_SCHEMA_VERSION
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable manifest → start fresh, do not crash
    doc["entries"][entry["out_basename"]] = entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def rasterize(
    pdf: Path,
    pages: tuple[int, int],
    dpi: int,
    fmt: str,
    quality: int,
    color: str,
    out: Path,
) -> dict:
    """Rasterize ``pages`` of ``pdf`` into the image-only PDF ``out``.

    Returns the manifest entry dict (also persisted by ``main``).
    """
    import pikepdf
    import pypdfium2 as pdfium

    start, end = pages
    out.parent.mkdir(parents=True, exist_ok=True)

    src = pdfium.PdfDocument(str(pdf))
    out_pdf = pikepdf.Pdf.new()
    try:
        for page_num in range(start, end + 1):
            img = render_page(src, page_num - 1, dpi, color)
            add_image_page(out_pdf, img, dpi, fmt, quality)
        out_pdf.save(str(out))
    finally:
        out_pdf.close()
        src.close()

    entry = {
        "out_basename": out.name,
        "source_pdf": str(pdf.resolve()),
        "pages": [start, end],
        "page_count": end - start + 1,
        "dpi": dpi,
        "format": fmt,
        "quality": quality if fmt == "jpeg" else None,
        "color": color,
        "sha256": _sha256(out),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": TOOL_VERSION,
    }
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rasterize a PDF page range into a fresh image-only PDF."
    )
    ap.add_argument("--pdf", required=True, type=Path, help="source PDF")
    ap.add_argument(
        "--pages", required=True,
        help="page range START:END (1-based inclusive) or a bare N",
    )
    ap.add_argument("--dpi", type=int, default=300, help="raster DPI (default 300)")
    ap.add_argument(
        "--format", choices=("png", "jpeg"), default="jpeg",
        help="image encoding (default jpeg)",
    )
    ap.add_argument(
        "--quality", type=int, default=92,
        help="JPEG quality (default 92; ignored for png)",
    )
    ap.add_argument(
        "--color", choices=("gray", "rgb"), default="gray",
        help="output color (default gray)",
    )
    ap.add_argument("--out", required=True, type=Path, help="output image-only PDF")
    ap.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest JSON path (default <out-dir>/manifest.json)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.pdf.is_file():
        ap.error(f"--pdf not found: {args.pdf}")

    import pypdfium2 as pdfium

    src = pdfium.PdfDocument(str(args.pdf))
    try:
        page_count = len(src)
    finally:
        src.close()

    try:
        pages = parse_page_range(args.pages, page_count)
    except ValueError as exc:
        ap.error(str(exc))

    entry = rasterize(
        args.pdf, pages, args.dpi, args.format, args.quality, args.color, args.out
    )
    manifest_path = args.manifest or (args.out.parent / "manifest.json")
    update_manifest(manifest_path, entry)

    print(
        f"rasterized {args.pdf.name} pages {pages[0]}:{pages[1]} "
        f"({entry['page_count']}p) @ {args.dpi}dpi {args.format} {args.color} "
        f"-> {args.out}"
    )
    print(f"  sha256={entry['sha256']}")
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
