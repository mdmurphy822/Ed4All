#!/usr/bin/env python3
"""Fetch OpenStax figure image assets for the figure/alt-text dataset.

Plans/09 §3a, task #22 follow-up. Mirror of
``scripts/fetch_ar5iv_figure_assets.py`` for the OpenStax CDN.

Reads ``data/figure_catalog/catalog.jsonl`` (source=="openstax") and fetches
each image POLITELY (serial, rate-limited, descriptive UA, retries) from
``https://openstax.org`` into ``data/figure_images/openstax/<doc>/<stem>.webp``.

Why OpenStax is the high-value source: it's the ONLY source in the catalog
where ``alt_raw`` is real human-authored alt text (1,278/1,288 of records).
Captions are the fallback ground truth for arXiv/PMC; OpenStax gives us *real*
accessibility writing to train/eval against.

Shared discipline (polite HTTP, license veto, manifest, catalog load, common
CLI args) lives in :mod:`scripts._figure_fetch_lib`. License: CC-BY (catalog
audit confirmed, 0 exclusions). CPU only — no torch/GPU.

Usage::
    python -m scripts.fetch_textbook_figure_assets --max 20      # smoke
    python -m scripts.fetch_textbook_figure_assets               # full
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from scripts._figure_fetch_lib import (
    MANIFEST,
    OUT,
    add_common_args,
    append_manifest,
    fetch_to_file,
    license_ok,
    load_catalog,
    safe_name,
)

OPENSTAX_BASE = "https://openstax.org"


def _safe_path(doc_id: str, ref: str, ext: str) -> Path:
    """``data/figure_images/openstax/<safe_doc>/<safe_stem>.<ext>``."""
    stem = ref.rsplit("/", 1)[-1] or "image"
    stem = safe_name(stem)
    if not stem.endswith(f".{ext}"):
        stem = f"{stem}.{ext}"
    return OUT / "openstax" / safe_name(doc_id) / stem


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, timeout=30, retries=2)
    args = ap.parse_args()

    rows = load_catalog(args.catalog)
    todo = [r for r in rows
            if r.get("source") == "openstax"
            and (r.get("image_ref") or "")
            and license_ok(r.get("license"))]
    skipped_lic = sum(1 for r in rows if r.get("source") == "openstax"
                      and (r.get("image_ref") or "") and not license_ok(r.get("license")))
    if args.max:
        todo = todo[:args.max]

    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "empty": 0, "error": 0, "http": 0}
    t0 = time.time()
    with MANIFEST.open("a") as mf:
        for i, rec in enumerate(todo, 1):
            ref = rec["image_ref"]
            ext = (rec.get("image_format_hint") or "webp").lower()
            doc_id = str(rec.get("doc_id", "unknown"))
            dest = _safe_path(doc_id, ref, ext)
            url = ref if ref.startswith("http") else OPENSTAX_BASE + (
                "" if ref.startswith("/") else "/") + ref
            status, nbytes, ctype = fetch_to_file(
                url, dest, timeout=args.timeout, retries=args.retries,
                delay=args.delay,
            )
            key = ("http" if status.startswith("http_")
                   else status if status in counts else "error")
            counts[key] = counts.get(key, 0) + 1
            append_manifest(
                mf, doc_id=doc_id, url=url, local_path=str(dest),
                status=status, n_bytes=nbytes, content_type=ctype,
            )
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] ok={counts['ok']} skip={counts['skip']} "
                      f"http4xx={counts['http']} err={counts['error']} "
                      f"({el:.0f}s, {i/max(el,1):.2f}/s)", flush=True)
            if status != "skip":
                time.sleep(args.delay)

    print(f"\n[done] requested={len(todo)} {counts} "
          f"| license-skipped={skipped_lic} | manifest={MANIFEST}")


if __name__ == "__main__":
    main()
