#!/usr/bin/env python3
"""Fetch PMC OpenAccess figure image assets for the figure/alt-text dataset.

Plans/09 §3a, task #22. Reads the figure catalog
(``data/figure_catalog/catalog.jsonl``), fetches PMC OpenAccess figure images
POLITELY, resolves the (extension-less) JATS graphic href to a real image, and
saves them to ``data/figure_images/pmc/<PMCID>/<file.ext>``. Appends to the
same resumable manifest the ar5iv fetcher writes
(``data/figure_images/_fetch_manifest.jsonl``).

OA channel
----------
The documented NCBI OA web service (``oa.fcgi``) still advertises the legacy
``ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/...`` package href, but that
anonymous-FTP tree has been DEPRECATED and those tarballs now 404/550 (the
``pair_from_pmc.py`` bulk extractor already had to move to the
``.../deprecated/oa_bulk/...`` path for the same reason). So the OA package is
not a usable image source today.

The robust, currently-working channel is the live OA article render on
``pmc.ncbi.nlm.nih.gov``:

  1. ``oa.fcgi?id=PMC<id>&tool=...&email=...`` (ONE light request per doc) —
     confirms the article is in the OA subset and returns the AUTHORITATIVE
     ``license=`` attribute, which we re-check against the commercial-OK policy.
  2. ``https://pmc.ncbi.nlm.nih.gov/articles/PMC<id>/`` (ONE request per doc) —
     the rendered article whose ``<img>`` tags point at the CDN
     (``https://cdn.ncbi.nlm.nih.gov/pmc/blobs/.../<stem>.<ext>``). The CDN URL
     already carries the *resolved* extension, so extension resolution is just
     a dict lookup of the catalog's extension-less ``image_ref`` stem against
     the stems found in the page. No blind jpg/gif/png probing required.

Both per-doc requests AMORTIZE across every figure in the article (~4
figures/doc in this catalog), so cost is ``2 + n_figs`` requests per doc.

Discipline (mirrors the ar5iv fetcher + the workers=1 Wikipedia rule)
--------------------------------------------------------------------
  * POLITE: strictly SERIAL, ``--delay`` between every network request
    (default 0.6s -> <2 req/s), descriptive User-Agent, NCBI etiquette params
    (``tool``/``email``) on the oa.fcgi endpoint, per-request timeout, bounded
    retries with EXPONENTIAL backoff that respects ``Retry-After`` on 429/503.
  * LICENSE: only fetches catalog records tagged commercial-OK, AND re-checks
    the live oa.fcgi license attribute. Anything smelling of NC/ND/SA is
    skipped.
  * RESUMABLE: skips assets already on disk; safe to re-run / interrupt.
  * CPU only — no torch, no GPU, no model loads.

Usage:
    python -m scripts.fetch_pmc_figure_assets --max 20      # smoke a batch
    python -m scripts.fetch_pmc_figure_assets               # full fetch
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.parse
from collections import OrderedDict
from pathlib import Path

from scripts._figure_fetch_lib import (
    MANIFEST,
    OUT,
    add_common_args,
    append_manifest,
    license_ok,
    load_catalog,
    polite_get,
    record_ref,
    safe_name,
)

OA_FCGI = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
ARTICLE_BASE = "https://pmc.ncbi.nlm.nih.gov/articles"
NCBI_TOOL = "DART-accessibility-research"
NCBI_EMAIL = "mdmurphy822@gmail.com"

# CDN image URLs embedded in the rendered article.
_CDN_IMG = re.compile(
    r"https://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/[^\s\"'<>]+?"
    r"\.(?:jpg|jpeg|png|gif|tif|tiff)",
    re.I,
)


# ---------------------------------------------------------------------------
# catalog helpers
# ---------------------------------------------------------------------------
def _stem(ref: str) -> str:
    """Catalog hrefs are extension-less, but be defensive and strip any tail."""
    tail = ref.rsplit("/", 1)[-1]
    # Drop a trailing image extension if one slipped in.
    return re.sub(r"\.(jpg|jpeg|png|gif|tif|tiff)$", "", tail, flags=re.I)


# ---------------------------------------------------------------------------
# OA service + article render
# ---------------------------------------------------------------------------
def oa_license(pmcid: str, *, timeout: int, retries: int, delay: float) -> tuple[bool, str]:
    """Query oa.fcgi. Return (in_oa_subset, license_str).

    in_oa_subset is False if the service reports an error (e.g. not OA).
    """
    qs = urllib.parse.urlencode({"id": pmcid, "tool": NCBI_TOOL, "email": NCBI_EMAIL})
    url = f"{OA_FCGI}?{qs}"
    try:
        body, _ctype, _status = polite_get(url, timeout=timeout, retries=retries, delay=delay)
    except Exception:  # noqa: BLE001 — treat service failure as "unknown"
        return (False, "")
    xml = body.decode("utf-8", "replace")
    if "<error" in xml:
        return (False, "")
    m = re.search(r'license="([^"]*)"', xml)
    return (True, m.group(1) if m else "")


def article_image_map(pmcid: str, *, timeout: int, retries: int,
                       delay: float) -> "OrderedDict[str, str]":
    """Fetch the rendered article and map figure stem -> CDN image URL.

    The CDN URL carries the resolved extension. First occurrence wins (the
    full-size figure blob appears before any thumbnail variants).
    """
    url = f"{ARTICLE_BASE}/{pmcid}/"
    body, _ctype, _status = polite_get(url, timeout=timeout, retries=retries, delay=delay)
    html = body.decode("utf-8", "replace")
    out: "OrderedDict[str, str]" = OrderedDict()
    for cdn in _CDN_IMG.findall(html):
        fn = cdn.rsplit("/", 1)[-1]
        stem = fn.rsplit(".", 1)[0]
        out.setdefault(stem, cdn)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, timeout=45, retries=3)
    ap.add_argument("--source", default="pmc", help="catalog source to fetch")
    ap.add_argument("--no-oa-check", action="store_true",
                    help="trust catalog license; skip the per-doc oa.fcgi re-check")
    args = ap.parse_args()

    rows = load_catalog(args.catalog)

    # Group catalog records by PMCID so the per-doc HTTP cost amortizes.
    by_doc: "OrderedDict[str, list[dict]]" = OrderedDict()
    skipped_lic = 0
    for r in rows:
        if r.get("source") != args.source:
            continue
        if not record_ref(r):
            continue
        if not license_ok(r.get("license")):
            skipped_lic += 1
            continue
        by_doc.setdefault(str(r.get("doc_id")), []).append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "unresolved": 0, "empty": 0,
              "error": 0, "http": 0, "lic_skip_doc": 0}
    fetched_images = 0
    t0 = time.time()

    with MANIFEST.open("a") as mf:
        for di, (pmcid, recs) in enumerate(by_doc.items(), 1):
            if args.max and fetched_images >= args.max:
                break

            # Decide which figures of this doc still need fetching (resume).
            pending = []
            for rec in recs:
                ref = record_ref(rec)
                stem = _stem(ref)
                # Extension unknown until resolved; check any existing file
                # for this stem in the doc dir.
                doc_dir = OUT / args.source / safe_name(pmcid)
                existing = (list(doc_dir.glob(safe_name(stem) + ".*"))
                            if doc_dir.exists() else [])
                existing = [p for p in existing if p.stat().st_size > 0]
                if existing:
                    counts["skip"] += 1
                    continue
                pending.append((rec, ref, stem))
            if not pending:
                continue

            # 1) Authoritative OA + license re-check (one light request).
            if not args.no_oa_check:
                in_oa, lic = oa_license(pmcid, timeout=args.timeout,
                                        retries=args.retries, delay=args.delay)
                if in_oa and not license_ok(lic):
                    counts["lic_skip_doc"] += 1
                    for rec, ref, _stem_ in pending:
                        append_manifest(
                            mf, doc_id=pmcid, url=f"{OA_FCGI}?id={pmcid}",
                            local_path="", status="license_skip",
                            n_bytes=0, content_type="",
                        )
                    continue

            # 2) Resolve stems -> CDN URLs from the rendered article.
            try:
                img_map = article_image_map(pmcid, timeout=args.timeout,
                                            retries=args.retries, delay=args.delay)
            except urllib.error.HTTPError as e:
                img_map = OrderedDict()
                article_err = f"http_{e.code}"
            except Exception as e:  # noqa: BLE001
                img_map = OrderedDict()
                article_err = f"error:{type(e).__name__}"
            else:
                article_err = ""

            # 3) Fetch each pending figure.
            for rec, ref, stem in pending:
                if args.max and fetched_images >= args.max:
                    break
                cdn = img_map.get(stem) or img_map.get(safe_name(stem))
                if not cdn:
                    # extension-resolution / page-parse miss
                    status = article_err or "unresolved"
                    counts["unresolved" if not article_err else
                           ("http" if article_err.startswith("http_") else "error")] += 1
                    append_manifest(
                        mf, doc_id=pmcid, url=f"{ARTICLE_BASE}/{pmcid}/",
                        local_path="", status=status, n_bytes=0, content_type="",
                    )
                    continue

                ext = cdn.rsplit(".", 1)[-1].lower()
                dest = (OUT / args.source / safe_name(pmcid)
                        / f"{safe_name(stem)}.{ext}")
                if dest.exists() and dest.stat().st_size > 0:
                    counts["skip"] += 1
                    continue
                try:
                    body, ctype, _status = polite_get(cdn, timeout=args.timeout,
                                                      retries=args.retries, delay=args.delay)
                    if not body:
                        counts["empty"] += 1
                        status, nbytes = "empty", 0
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(body)
                        counts["ok"] += 1
                        fetched_images += 1
                        status, nbytes = "ok", len(body)
                except urllib.error.HTTPError as e:
                    counts["http"] += 1
                    status, nbytes, ctype = f"http_{e.code}", 0, ""
                except Exception as e:  # noqa: BLE001
                    counts["error"] += 1
                    status, nbytes, ctype = f"error:{type(e).__name__}", 0, ""

                append_manifest(
                    mf, doc_id=pmcid, url=cdn,
                    local_path=str(dest) if status == "ok" else "",
                    status=status, n_bytes=nbytes, content_type=ctype,
                )

            if di % 25 == 0 or args.max:
                el = time.time() - t0
                print(f"[doc {di}/{len(by_doc)}] images_ok={counts['ok']} "
                      f"skip={counts['skip']} unresolved={counts['unresolved']} "
                      f"http4xx={counts['http']} err={counts['error']} "
                      f"lic_skip_doc={counts['lic_skip_doc']} "
                      f"({el:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"\n[done] docs={len(by_doc)} images_fetched={fetched_images} {counts} "
          f"| license-skipped-records={skipped_lic} "
          f"| elapsed={el:.0f}s | manifest={MANIFEST}")


if __name__ == "__main__":
    main()
