"""Refresh figure-catalog coverage stats from LOCALLY-fetched images.

Local-only recount (task R7). NO network, NO model imports, NO re-parse of
source HTML/XML. It only:
  1. reads ``data/figure_catalog/catalog.jsonl`` (the figure rows),
  2. reads ``data/figure_images/_fetch_manifest.jsonl`` (the fetch log),
  3. recomputes ``image_local`` / ``needs_image_fetch`` per catalog row, and
  4. rewrites ``coverage_report.json`` (every other stat is preserved verbatim
     so the report stays identical to ``build_figure_catalog.build_report``
     except for the two image-availability fields plus a recount provenance
     block).

Why a separate script: ``data/builders/build_figure_catalog.py`` always re-parses the
sources and hard-codes ``image_local: False`` (image bytes were not present
when it ran). It has no no-fetch recount mode, so the on-disk fetch results
have to be folded back in here.

Join key (catalog row -> fetched image): ``(source, doc_folder, stem)`` where
  * ``source``      = the catalog ``source`` ("arxiv"/"pmc"/"openstax"),
  * ``doc_folder``  = the manifest ``local_path`` doc segment; for openstax the
    catalog ``doc_id`` contains '/' which the fetcher flattened to '_',
  * ``stem``        = basename with a *real image extension* stripped
    (png/jpg/jpeg/gif/svg/webp/tif/tiff only). PMC catalog refs are
    extension-less (e.g. ``pbio.0030060.g001``) and must NOT have their
    trailing ``.g001`` segment chopped, so we strip image extensions only.

A catalog row counts as ``image_local`` when its key appears among the manifest
entries whose ``status == "ok"``. (All "ok" ``local_path`` values were verified
to exist on disk, so the manifest and the filesystem agree; manifest is used as
the authoritative join because arXiv assets live in nested subdirs that a flat
doc-folder filesystem walk would mis-key.)

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/analysis/refresh_figure_coverage.py
    # --dry-run to print the new numbers without writing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

_IMG_EXT_RE = re.compile(r"\.(png|jpg|jpeg|gif|svg|webp|tif|tiff)$", re.I)


def _stem(name: str) -> str:
    """basename with a real image extension stripped (PMC refs keep .gNNN)."""
    return _IMG_EXT_RE.sub("", os.path.basename(name))


def _catalog_doc_folder(source: str, doc_id: str) -> str:
    # The fetcher flattens openstax doc_ids ('book/section') to 'book_section'.
    return doc_id.replace("/", "_") if source == "openstax" else doc_id


def load_ok_keys(manifest_path: Path) -> tuple[set, Counter, int, int]:
    """Return (ok_keys, ok_per_source, ok_line_count, missing_on_disk)."""
    ok_keys: set[tuple[str, str, str]] = set()
    ok_per_source: Counter = Counter()
    ok_lines = 0
    missing_on_disk = 0
    repo_root = manifest_path.parent.parent.parent  # .../Semantic
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            ok_lines += 1
            lp = r["local_path"]
            parts = lp.split("/")
            # data/figure_images/<source>/<doc_folder>/.../<file>
            source = parts[2]
            doc_folder = parts[3]
            ok_per_source[source] += 1
            ok_keys.add((source, doc_folder, _stem(lp)))
            if not (repo_root / lp).exists():
                missing_on_disk += 1
    return ok_keys, ok_per_source, ok_lines, missing_on_disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-dir", type=Path,
                    default=Path("data/figure_catalog"))
    ap.add_argument("--images-dir", type=Path,
                    default=Path("data/figure_images"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    catalog_path = args.catalog_dir / "catalog.jsonl"
    report_path = args.catalog_dir / "coverage_report.json"
    manifest_path = args.images_dir / "_fetch_manifest.jsonl"

    ok_keys, ok_per_source, ok_lines, missing = load_ok_keys(manifest_path)

    total = Counter()
    local = Counter()
    with catalog_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = r["source"]
            total[src] += 1
            key = (src, _catalog_doc_folder(src, r["doc_id"]),
                   _stem(r["image_ref"]))
            if key in ok_keys:
                local[src] += 1

    total_rows = sum(total.values())
    local_rows = sum(local.values())
    needs_rows = total_rows - local_rows

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["image_local"] = local_rows
    report["needs_image_fetch"] = needs_rows
    report["image_local_by_source"] = dict(local)
    report["needs_image_fetch_by_source"] = {
        s: total[s] - local.get(s, 0) for s in total
    }
    report["coverage_recount"] = {
        "method": "join catalog rows to status==ok fetch-manifest entries "
                  "by (source, doc_folder, image-stem)",
        "manifest_ok_lines": ok_lines,
        "manifest_ok_by_source": dict(ok_per_source),
        "manifest_ok_paths_missing_on_disk": missing,
        "source": "data/figure_images/_fetch_manifest.jsonl",
    }

    print(f"total figures      : {total_rows}")
    print(f"image_local        : {local_rows}  by_source={dict(local)}")
    print(f"needs_image_fetch  : {needs_rows}")
    print(f"manifest ok lines  : {ok_lines}  by_source={dict(ok_per_source)}")
    print(f"ok paths missing on disk: {missing}")

    if args.dry_run:
        print("[dry-run] not writing")
        return
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {report_path}")


if __name__ == "__main__":
    main()
