"""Populate data/glm_ocr_cache/ for every pair's PDF before the build.

Self-contained: does not import from `data/builders/build_qwen_data.py` (which
isn't a package). Mirrors the same two-phase logic: scan for misses,
only load the model if any uncached region exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _pair_pdf_path(pair_path: Path) -> Path | None:
    from semantik_structure.prerender_cache import cache_path_for
    try:
        pair = json.loads(pair_path.read_text())
    except Exception:
        return None
    output_html = pair.get("output_html")
    if not output_html:
        return None
    local_pdf = pair.get("local_pdf")
    if local_pdf and Path(local_pdf).exists():
        return Path(local_pdf)
    cache_pdf = cache_path_for(output_html)
    if cache_pdf.exists():
        return cache_pdf
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dirs", type=Path, nargs="+",
                    default=[Path("data/pairs/wikipedia"),
                             Path("data/pairs/openstax"),
                             Path("data/pairs/federal_register"),
                             Path("data/pairs/gutenberg"),
                             Path("data/pairs/forms"),
                             Path("data/pairs/arxiv")])
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--model-path", default="zai-org/GLM-OCR")
    args = ap.parse_args()

    from semantik_structure.extract_shared import extract_shared_cached
    from semantik_structure.glm_ocr_cache import (
        CacheKey, get as cache_get, sha256_file,
    )
    from semantik_structure.region_detection import (
        detect_low_conf_tesseract_regions,
        detect_math_regions,
        detect_table_regions,
    )
    from semantik_structure.glm_ocr import PROMPT_FORMULA, PROMPT_TABLE, PROMPT_TEXT

    def _prompt_for(kind: str) -> str:
        return {"math": PROMPT_FORMULA,
                "table": PROMPT_TABLE}.get(kind, PROMPT_TEXT)

    pair_paths: list[Path] = []
    for d in args.pair_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("*.json"))
        if args.limit_per_source:
            files = files[:args.limit_per_source]
        pair_paths.extend(files)
    print(f"[plan] {len(pair_paths)} pair files", file=sys.stderr)

    # Phase 1: scan for cache misses
    t0 = time.time()
    pending_pairs: list[Path] = []
    total_regions = 0
    hit_regions = 0
    for p in pair_paths:
        pdf_path = _pair_pdf_path(p)
        if pdf_path is None:
            continue
        try:
            shared = extract_shared_cached(pdf_path)
        except Exception:
            continue
        pdf_sha = sha256_file(pdf_path)
        pair_has_miss = False
        for pg in shared.get("pages") or []:
            regions = (
                detect_table_regions(pg)
                + detect_math_regions(pg)
                + detect_low_conf_tesseract_regions(pg)
            )
            for r in regions:
                total_regions += 1
                key = CacheKey(pdf_sha=pdf_sha, page_num=int(pg["page_num"]),
                               bbox=r.bbox, prompt=_prompt_for(r.kind))
                if cache_get(key) is None:
                    pair_has_miss = True
                else:
                    hit_regions += 1
        if pair_has_miss:
            pending_pairs.append(p)
    scan_s = time.time() - t0
    print(f"[scan] {scan_s:.1f}s  regions_total={total_regions} "
          f"cached={hit_regions}  pairs_needing_ocr={len(pending_pairs)}",
          file=sys.stderr)
    if not pending_pairs:
        print("[done] all regions cached — skipping model load", file=sys.stderr)
        return

    # Phase 2: OCR the misses
    from semantik_structure.glm_ocr import load_glm_ocr, unload_glm_ocr
    from semantik_structure.glm_ocr_enrich import enrich_shared
    print(f"[load] {args.model_path}", file=sys.stderr)
    glm = load_glm_ocr(args.model_path)
    try:
        t1 = time.time()
        for i, p in enumerate(pending_pairs):
            pdf_path = _pair_pdf_path(p)
            if pdf_path is None:
                continue
            try:
                shared = extract_shared_cached(pdf_path)
                enrich_shared(shared, pdf_path, glm)
            except Exception as exc:
                print(f"[skip] {p.name}: {exc}", file=sys.stderr)
            if (i + 1) % 25 == 0:
                dt = (time.time() - t1) / 60
                print(f"[ocr] {i+1}/{len(pending_pairs)} "
                      f"elapsed={dt:.1f}min", file=sys.stderr)
    finally:
        unload_glm_ocr(glm)
    print(f"[done] total {(time.time()-t0)/60:.1f}min", file=sys.stderr)


if __name__ == "__main__":
    main()
