#!/usr/bin/env python3
"""Fetch ar5iv figure image assets for the figure/alt-text dataset.

Plans/09 §3a, task #22. Reads the figure catalog
(``data/figure_catalog/catalog.jsonl``), fetches arXiv figure images from
ar5iv POLITELY (serial, rate-limited, real UA, retries), and saves them to
``data/figure_images/<source>/<doc_id>/<file>``. Writes a resumable fetch
manifest.

Shared discipline (polite HTTP, license veto, manifest, catalog load, common
CLI args) lives in :mod:`scripts._figure_fetch_lib`; this file is the
ar5iv-specific path/URL mapping plus the fetch loop.

Discipline:
  * POLITE: serial, ``--delay`` between requests (default 0.6s), a descriptive
    User-Agent, per-request timeout + bounded retries (exponential backoff via
    the shared core). ar5iv is a free academic service — do not hammer it.
  * LICENSE: only fetches records the catalog tagged commercial-OK. Anything
    smelling of NC / ND / SA / arxiv-default is skipped (defensive — the cached
    ar5iv set is the CC-cleared "no-problem" subset, but we double-check).
  * RESUMABLE: skips assets already on disk; safe to re-run / interrupt.
  * CPU only — no torch, no GPU.

Usage:
    python -m scripts.fetch_ar5iv_figure_assets --max 20      # smoke a batch
    python -m scripts.fetch_ar5iv_figure_assets               # full fetch
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

from scripts._figure_fetch_lib import (
    MANIFEST,
    OUT,
    add_common_args,
    append_manifest,
    fetch_to_file,
    license_ok,
    load_catalog,
    polite_get,
    record_ref,
    safe_name,
)

AR5IV_BASE = "https://ar5iv.labs.arxiv.org"


def _safe_path(source: str, doc_id: str, ref: str) -> Path:
    # Keep the ar5iv asset sub-path so figures in one doc don't collide.
    tail = ref.split("/assets/", 1)[-1] if "/assets/" in ref else ref.rsplit("/", 1)[-1]
    tail = re.sub(r"[^A-Za-z0-9._/-]", "_", tail).lstrip("/")
    return OUT / source / safe_name(doc_id) / tail


def _url_for(ref: str) -> str:
    if ref.startswith("http"):
        return ref
    if ref.startswith("/static/"):
        # Site-chrome assets (logos/icons in captionless standalone-img
        # records) live on arxiv.org; the ar5iv host 404s them.
        return "https://arxiv.org" + ref
    return AR5IV_BASE + ("" if ref.startswith("/") else "/") + ref


LICENSE_MAP = OUT / "_arxiv_license_map.json"
OAI_URL = "https://export.arxiv.org/oai2"


def _oai_license(
    arxiv_id: str, *, timeout: int = 30, retries: int = 2, delay: float = 1.0
) -> str | None:
    """Fetch one paper's license URL via arXiv OAI-PMH.

    Transport note: the Atom API path
    (``dart_semantic.arxiv_license.fetch_licenses``) does NOT actually return
    the ``arxiv:license`` element (verified 2026-06-10: every entry comes back
    license-less, which ``is_commercial_ok`` rightly treats as not-OK — a
    blanket false veto). The OAI-PMH ``arXiv`` metadata format carries the
    authoritative ``<license>`` URL, so that is the transport here;
    CLASSIFICATION still reuses ``dart_semantic.arxiv_license``.
    """
    qs = urllib.parse.urlencode(
        {
            "verb": "GetRecord",
            "metadataPrefix": "arXiv",
            "identifier": f"oai:arXiv.org:{arxiv_id}",
        }
    )
    body, _ctype, _status = polite_get(
        f"{OAI_URL}?{qs}", timeout=timeout, retries=retries, delay=delay
    )
    m = re.search(r"<license>([^<]+)</license>", body.decode("utf-8", "replace"))
    return m.group(1).strip() if m else None


def verify_arxiv_licenses(
    doc_ids: list[str], map_path: Path = LICENSE_MAP, delay: float = 1.0
) -> dict[str, dict]:
    """Per-doc commercial-OK verification (feedback_license_policy: CC-BY /
    CC0 / PD / ODC-By only; arXiv default/assumed-license papers REJECTED).

    License URL via OAI-PMH (:func:`_oai_license`, one polite request per
    unknown doc), classification via
    ``dart_semantic.arxiv_license.is_commercial_ok``. Results are cached to
    ``map_path`` so re-runs cost zero API calls for known docs. Returns
    ``{doc_id: {"commercial_ok": bool, "license_url": ..., "reason": ...}}``.
    """
    from dart_semantic.arxiv_license import is_commercial_ok

    known: dict[str, dict] = {}
    if map_path.exists():
        known = json.loads(map_path.read_text(encoding="utf-8"))
    missing = sorted({str(d) for d in doc_ids if str(d) not in known})
    for i, doc_id in enumerate(missing, 1):
        try:
            url = _oai_license(doc_id, delay=delay)
        except Exception as e:  # noqa: BLE001 — unknown stays not-OK
            known[doc_id] = {
                "commercial_ok": False,
                "license_url": None,
                "reason": f"oai error: {type(e).__name__}",
            }
            continue
        ok, reason = is_commercial_ok(url)
        if ok and url and "by-sa" in url.lower():
            # Stricter than is_commercial_ok: share-alike is off the project
            # allowlist (feedback_license_policy / classify_license_url /
            # _figure_fetch_lib.BAD_LICENSE all veto SA) — fail closed here too.
            ok, reason = False, "rejected: CC-BY-SA (share-alike off allowlist)"
        known[doc_id] = {"commercial_ok": ok, "license_url": url, "reason": reason}
        if i % 25 == 0 or i == len(missing):
            print(f"[license] verified {i}/{len(missing)} docs", flush=True)
    if missing:
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(known, indent=1, sort_keys=True), encoding="utf-8")
    return known


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, timeout=30, retries=2)
    ap.add_argument("--source", default="arxiv", help="catalog source to fetch")
    ap.add_argument(
        "--verify-license",
        action="store_true",
        help="re-check each doc's license via arXiv OAI-PMH "
        "(classification: dart_semantic.arxiv_license, share-alike also "
        "vetoed) and skip docs that are not commercial-OK; results cached to "
        f"{LICENSE_MAP}",
    )
    args = ap.parse_args()

    rows = load_catalog(args.catalog)
    todo = [
        r
        for r in rows
        if r.get("source") == args.source and record_ref(r) and license_ok(r.get("license"))
    ]
    skipped_lic = sum(
        1
        for r in rows
        if r.get("source") == args.source and record_ref(r) and not license_ok(r.get("license"))
    )

    if args.verify_license and todo:
        lic_map = verify_arxiv_licenses([str(r.get("doc_id")) for r in todo])
        vetoed = [r for r in todo if not lic_map.get(str(r.get("doc_id")), {}).get("commercial_ok")]
        todo = [r for r in todo if lic_map.get(str(r.get("doc_id")), {}).get("commercial_ok")]
        skipped_lic += len(vetoed)
        print(
            f"[license] arXiv API verification: {len(todo)} records OK, "
            f"{len(vetoed)} vetoed (map: {LICENSE_MAP})"
        )

    if args.max:
        todo = todo[: args.max]

    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "empty": 0, "error": 0, "http": 0}
    t0 = time.time()
    with MANIFEST.open("a") as mf:
        for i, rec in enumerate(todo, 1):
            ref = record_ref(rec)
            dest = _safe_path(rec["source"], str(rec.get("doc_id", "unknown")), ref)
            url = _url_for(ref)
            status, nbytes, ctype = fetch_to_file(
                url,
                dest,
                timeout=args.timeout,
                retries=args.retries,
                delay=args.delay,
            )
            key = "http" if status.startswith("http_") else status if status in counts else "error"
            counts[key] = counts.get(key, 0) + 1
            append_manifest(
                mf,
                doc_id=rec.get("doc_id"),
                url=url,
                local_path=str(dest),
                status=status,
                n_bytes=nbytes,
                content_type=ctype,
            )
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(
                    f"[{i}/{len(todo)}] ok={counts['ok']} skip={counts['skip']} "
                    f"http4xx={counts['http']} err={counts['error']} "
                    f"({el:.0f}s, {i / max(el, 1):.1f}/s)",
                    flush=True,
                )
            # fetch_to_file already slept `delay` after every attempt; only
            # add the inter-record pause when we actually hit the network.
            if status != "skip":
                time.sleep(args.delay)

    print(
        f"\n[done] requested={len(todo)} {counts} "
        f"| license-skipped={skipped_lic} | manifest={MANIFEST}"
    )


if __name__ == "__main__":
    main()
