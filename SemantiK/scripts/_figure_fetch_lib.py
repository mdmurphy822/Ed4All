#!/usr/bin/env python3
"""Shared helpers for the figure-asset fetchers (Plans/09 §3a, task #22).

The three fetchers (``fetch_ar5iv_figure_assets``, ``fetch_textbook_figure_assets``,
``fetch_pmc_figure_assets``) all (a) read the same figure catalog, (b) write the
same resumable manifest, (c) enforce the same commercial-OK license veto, and
(d) speak POLITE HTTP (serial, rate-limited, descriptive UA, bounded retries).
This module is the single source of truth for those pieces; the fetchers are
thin source-specific loops over it.

The HTTP core is ported verbatim from the PMC fetcher's ``_get`` (the strongest
of the three: exponential backoff that honors ``Retry-After`` on 429/503, raises
4xx immediately, sleeps the polite delay AFTER every attempt).

CPU only — no torch, no GPU, no model loads. Imported as
``from scripts._figure_fetch_lib import ...`` (every fetcher runs via
``python -m scripts.fetch_...``).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
UA = ("DART-accessibility-research/0.1 "
      "(figure alt-text dataset; mailto:mdmurphy822@gmail.com)")
CATALOG = Path("data/figure_catalog/catalog.jsonl")
OUT = Path("data/figure_images")
MANIFEST = OUT / "_fetch_manifest.jsonl"

# License substrings that DISQUALIFY a record from generative use. Safe
# superset across all sources: openstax/pmc records never contain
# "arxiv-default" so carrying it here cannot cause a false veto.
BAD_LICENSE = re.compile(
    r"\b(nc|nd|sa|noncommercial|non-commercial|no-?deriv|share-?alike|"
    r"arxiv-?default|all-?rights)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def license_ok(license_value: object) -> bool:
    """True if the license string does NOT smell of NC/ND/SA/arxiv-default.

    None-safe (PMC semantics): a missing/empty license is permissive here —
    the per-doc oa.fcgi re-check, not this string match, is PMC's gate.
    """
    return bool(license_value) is False or not BAD_LICENSE.search(str(license_value))


def record_ref(rec: dict) -> str:
    """The image reference for a catalog record (image_ref | img_src | src)."""
    return rec.get("image_ref") or rec.get("img_src") or rec.get("src") or ""


def safe_name(name: str) -> str:
    """Filesystem-safe component: any char outside [A-Za-z0-9._-] -> '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def load_catalog(catalog: Path) -> list[dict]:
    """Parse the JSONL catalog into a list of dicts. SystemExit if missing."""
    if not catalog.exists():
        raise SystemExit(
            f"catalog not found: {catalog} (run the catalog build first)"
        )
    return [json.loads(line) for line in catalog.open() if line.strip()]


def add_common_args(ap: argparse.ArgumentParser, *, timeout: int = 30,
                    retries: int = 2) -> None:
    """Register the args every fetcher shares: --catalog --max --delay
    --timeout --retries. ``timeout``/``retries`` defaults differ by source."""
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--max", type=int, default=0, help="0 = all; else cap (smoke)")
    ap.add_argument("--delay", type=float, default=0.6,
                    help="seconds slept after every network request")
    ap.add_argument("--timeout", type=int, default=timeout)
    ap.add_argument("--retries", type=int, default=retries)


# ---------------------------------------------------------------------------
# polite HTTP
# ---------------------------------------------------------------------------
def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(int(value))
    except (TypeError, ValueError):
        return None  # HTTP-date form: fall back to exponential


def polite_get(url: str, *, timeout: int, retries: int, delay: float,
               ua: str = UA) -> tuple[bytes, str, int]:
    """Polite GET with exponential backoff honoring Retry-After.

    Returns (body, content_type, http_status). Raises the last error after
    exhausting retries; 4xx (except 429) are raised immediately (no retry).
    Sleeps ``delay`` AFTER every network attempt so we never exceed the rate.

    Ported verbatim from fetch_pmc_figure_assets._get (the strongest core).
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                status = resp.status
                time.sleep(delay)
                return body, ctype, status
        except urllib.error.HTTPError as e:
            time.sleep(delay)
            if e.code in (429, 503):
                # rate-limited / unavailable — back off, prefer Retry-After
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = _parse_retry_after(ra)
                if wait is None:
                    wait = min(60.0, (2 ** attempt) * 2.0)
                last_exc = e
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise
            if 400 <= e.code < 500:
                raise  # genuine client error — don't retry
            # 5xx (non-503): transient, exponential backoff
            last_exc = e
            if attempt < retries:
                time.sleep(min(60.0, (2 ** attempt) * 2.0))
                continue
            raise
        except Exception as e:  # noqa: BLE001 — transient network: backoff + retry
            time.sleep(delay)
            last_exc = e
            if attempt < retries:
                time.sleep(min(60.0, (2 ** attempt) * 2.0))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def fetch_to_file(url: str, dest: Path, *, timeout: int, retries: int,
                  delay: float, ua: str = UA) -> tuple[str, int, str]:
    """Fetch ``url`` to ``dest``, returning (status, n_bytes, content_type).

    status in {ok, skip, empty, http_<code>, error:<Type>}. Skips if the file
    already exists non-empty (resume). Built atop :func:`polite_get`.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", dest.stat().st_size, "")
    try:
        body, ctype, _status = polite_get(
            url, timeout=timeout, retries=retries, delay=delay, ua=ua,
        )
    except urllib.error.HTTPError as e:
        return (f"http_{e.code}", 0, "")
    except Exception as e:  # noqa: BLE001 — record the transient class
        return (f"error:{type(e).__name__}", 0, "")
    if not body:
        return ("empty", 0, ctype)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return ("ok", len(body), ctype)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def append_manifest(mf: IO[str], *, doc_id: object, url: str, local_path: str,
                    status: str, n_bytes: int, content_type: str) -> None:
    """Append one manifest line and flush.

    The JSON key for the byte count stays ``"bytes"`` (the on-disk schema the
    existing manifest and its consumers — build_figure_alt_dataset,
    refresh_figure_coverage — already use)."""
    mf.write(json.dumps({
        "doc_id": doc_id,
        "url": url,
        "local_path": local_path,
        "status": status,
        "bytes": n_bytes,
        "content_type": content_type,
    }) + "\n")
    mf.flush()
