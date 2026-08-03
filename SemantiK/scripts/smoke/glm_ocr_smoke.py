"""Smoke test: route detected math/table regions through GLM-OCR.

Picks up the existing extract_shared output for a PDF, identifies regions
where pypdfium2/pdfplumber are known weak (CID-heavy text / latex math
fonts / pdfplumber-detected tables), renders those region crops, runs
GLM-OCR on each, and writes a side-by-side markdown report for manual
inspection.

Output: eval/side_by_side/glm_ocr_<pdfname>_<timestamp>/regions.md plus
one PNG per region crop so the markdown is self-contained in a browser.

No production pipeline wiring — this is a diagnostic harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from semantik_structure.extract_shared import (
    extract_shared_cached,
    _pypdfium2_render_page_to_image,
)
from semantik_structure.region_detection import (
    Region,
    detect_math_regions,
    detect_table_regions,
)


def _crop_for_region(page_img, page_w_pdf: float, page_h_pdf: float,
                     bbox: tuple[float, float, float, float],
                     pad: float = 4.0, scale: float = 2.0):
    """Map a PDF-space bbox (top-left origin) onto the rendered image.

    page_img was produced with _pypdfium2_render_page_to_image(scale=scale)
    so image size = page_pdf_size * scale.
    """
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(page_w_pdf, x1 + pad)
    y1 = min(page_h_pdf, y1 + pad)
    px0 = int(x0 * scale)
    py0 = int(y0 * scale)
    px1 = int(x1 * scale)
    py1 = int(y1 * scale)
    return page_img.crop((px0, py0, px1, py1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("eval/side_by_side"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Process only the first N pages (smoke-test speed).")
    ap.add_argument("--max-regions", type=int, default=30,
                    help="Cap on total GLM-OCR calls to keep the smoke "
                         "quick (it's not cheap).")
    ap.add_argument("--render-scale", type=float, default=2.0)
    ap.add_argument("--model-path", default="zai-org/GLM-OCR")
    ap.add_argument("--quantize", action="store_true",
                    help="Load GLM-OCR in 4-bit (saves ~1.3GB VRAM).")
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"not found: {args.pdf}")

    run_name = args.run_name or (
        f"glm_ocr_{args.pdf.stem}_{datetime.now():%Y%m%d-%H%M}"
    )
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    print(f"[out] {out_dir}")

    # ---------- pipeline-free extraction pass ----------
    print(f"[extract] {args.pdf.name}")
    shared = extract_shared_cached(args.pdf)
    pages = shared.get("pages", [])
    if args.max_pages:
        pages = pages[:args.max_pages]

    # Pass 1: detect all regions across the doc, no cropping yet.
    all_tables: list[Region] = []
    all_math: list[Region] = []
    page_metrics: dict[int, tuple[float, float]] = {}
    for pg in pages:
        page_num = pg["page_num"]
        math_rs = detect_math_regions(pg)
        tab_rs = detect_table_regions(pg)
        if math_rs or tab_rs:
            page_metrics[page_num] = (
                float(pg.get("width", 612)),
                float(pg.get("height", 792)),
            )
        all_tables.extend(tab_rs)
        all_math.extend(math_rs)
    # Tables first across the entire doc so they never lose the region
    # cap to math-dense early pages.
    detected: list[Region] = all_tables + all_math
    if len(detected) > args.max_regions:
        detected = detected[:args.max_regions]

    # Pass 2: render + crop only the pages we actually need.
    pages_to_render = sorted({r.page_num for r in detected})
    page_imgs: dict[int, object] = {}
    for page_num in pages_to_render:
        page_imgs[page_num] = _pypdfium2_render_page_to_image(
            args.pdf, page_num, scale=args.render_scale,
        )
    all_regions: list[tuple[Region, object]] = []
    for r in detected:
        page_w, page_h = page_metrics[r.page_num]
        crop = _crop_for_region(page_imgs[r.page_num], page_w, page_h,
                                r.bbox, scale=args.render_scale)
        all_regions.append((r, crop))

    print(f"[regions] {len(all_regions)} regions to OCR "
          f"(math={sum(1 for r,_ in all_regions if r.kind=='math')}, "
          f"table={sum(1 for r,_ in all_regions if r.kind=='table')})")
    if not all_regions:
        (out_dir / "regions.md").write_text(
            "# No math or table regions detected.\n"
        )
        return

    # ---------- GLM-OCR ----------
    import time
    from semantik_structure.glmocr.region_enrichment.model import (
        load_glm_ocr, ocr_image, unload_glm_ocr,
        PROMPT_TEXT, PROMPT_FORMULA, PROMPT_TABLE,
    )
    print(f"[load] {args.model_path} (quantize={args.quantize})")
    t_load_start = time.time()
    glm = load_glm_ocr(args.model_path, quantize_4bit=args.quantize)
    t_load_end = time.time()

    rows: list[dict] = []
    per_page_time: dict[int, float] = {}
    per_page_regions: dict[int, int] = {}
    try:
        for i, (r, crop) in enumerate(all_regions):
            crop_path = crops_dir / f"region_{i:03d}_{r.kind}_p{r.page_num}.png"
            crop.save(crop_path)
            if r.kind == "math":
                prompt = PROMPT_FORMULA
            elif r.kind == "table":
                prompt = PROMPT_TABLE
            else:
                prompt = PROMPT_TEXT
            t0 = time.time()
            ocr_text = ocr_image(glm, crop, prompt=prompt)
            dt = time.time() - t0
            per_page_time[r.page_num] = per_page_time.get(r.page_num, 0.0) + dt
            per_page_regions[r.page_num] = per_page_regions.get(r.page_num, 0) + 1
            rows.append({
                "idx": i,
                "page": r.page_num,
                "kind": r.kind,
                "bbox": list(r.bbox),
                "src": r.src_text,
                "ocr": ocr_text,
                "crop": crop_path.relative_to(out_dir).as_posix(),
                "ocr_sec": dt,
            })
            print(f"  [{i+1}/{len(all_regions)}] page {r.page_num} {r.kind} "
                  f"{dt:.2f}s src={r.src_text[:40]!r} -> ocr_len={len(ocr_text)}")
    finally:
        unload_glm_ocr(glm)
    t_end = time.time()

    # ---------- timing summary ----------
    load_sec = t_load_end - t_load_start
    ocr_total_sec = sum(r["ocr_sec"] for r in rows)
    per_region = ocr_total_sec / max(1, len(rows))
    pages_seen = sorted(per_page_time)
    n_pages_total = len(pages)
    n_pages_with_regions = len(pages_seen)
    per_page_avg = ocr_total_sec / max(1, n_pages_with_regions)
    amortized_per_page_all = ocr_total_sec / max(1, n_pages_total)

    # ---------- side-by-side markdown ----------
    lines: list[str] = [
        f"# GLM-OCR smoke — {args.pdf.name}",
        "",
        f"- pdf: `{args.pdf}`",
        f"- regions: {len(rows)} "
        f"(math={sum(1 for r in rows if r['kind']=='math')}, "
        f"table={sum(1 for r in rows if r['kind']=='table')})",
        "",
        "## timing",
        "",
        f"- model load (incl. weight download cache hit): **{load_sec:.1f}s**",
        f"- ocr total: **{ocr_total_sec:.1f}s** across {len(rows)} regions",
        f"- per region (avg): **{per_region:.2f}s**",
        f"- per page *with regions* (avg): **{per_page_avg:.2f}s** "
        f"({n_pages_with_regions} pages had detected regions out of {n_pages_total})",
        f"- amortized per page *over whole doc*: **{amortized_per_page_all:.2f}s**",
        "",
        "### per-page breakdown",
        "",
        "| page | regions | ocr sec |",
        "|---:|---:|---:|",
    ]
    for p in pages_seen:
        lines.append(f"| {p} | {per_page_regions[p]} | {per_page_time[p]:.2f} |")
    lines.append("")
    for r in rows:
        lines.append(f"## #{r['idx']} — page {r['page']} — **{r['kind']}**")
        lines.append("")
        lines.append(f"![crop]({r['crop']})")
        lines.append("")
        lines.append(f"- bbox (pdf-space): {r['bbox']}")
        lines.append("")
        lines.append("**source text (pypdfium2/pdfplumber):**")
        lines.append("")
        lines.append("```")
        lines.append((r["src"] or "").rstrip())
        lines.append("```")
        lines.append("")
        lines.append("**GLM-OCR:**")
        lines.append("")
        lines.append("```")
        lines.append(r["ocr"].rstrip())
        lines.append("```")
        lines.append("")
    (out_dir / "regions.md").write_text("\n".join(lines))
    print(f"[done] {out_dir}/regions.md")


if __name__ == "__main__":
    main()
