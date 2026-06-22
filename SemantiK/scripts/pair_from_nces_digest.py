"""Fetch dense statistical tables from the NCES Digest of Education Statistics.

Source: nces.ed.gov — National Center for Education Statistics, a US federal
agency. Its publications are **US-government works in the public domain**
(no copyright), on the CC-BY/CC0/ODC-By/PD allowlist. This adds the
**dense_data** table_type (many-row numeric statistical tables) the table set
is thin on, and dilutes the PMC-dominated source mix (Plans/06 §0.1, §6).

The Digest menu page links ~556 table pages (``dt<yy>_<num>.<sub>.asp``). Each
page renders one data table (class ``tableMain``/``tabletop`` with ``<thead>``
+ ``<th scope="col">``) wrapped in several layout tables (nav, download links,
notes). We keep only the data table and hoist its "Table N. …" title into a
``<caption>``, emitting a pair whose ``output_html`` the multi-source table
builder ingests via its HTML5 path (source ``nces_digest``).

Polite by construction: single-threaded, rate-limited (``--delay``), capped
(``--limit``), descriptive User-Agent. Writes one pair JSON per table to
``data/pairs/nces_digest/``.

Usage::

    python scripts/pair_from_nces_digest.py --limit 150            # pilot
    python scripts/pair_from_nces_digest.py --limit 600 --delay 0.5  # bulk
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from lxml import html as lxml_html

MENU_URL = "https://nces.ed.gov/programs/digest/2022menu_tables.asp"
TABLE_BASE = "https://nces.ed.gov/programs/digest/d22/tables/"
LINK_RE = re.compile(r"dt22_(\d+)\.(\d+)\.asp", re.IGNORECASE)
TITLE_RE = re.compile(r"Table\s+[\d.]+\.\s+.{5,}", re.DOTALL)
UA = "Mozilla/5.0 (DART accessibility research dataset; +mailto:mdmurphy822@gmail.com)"
OUT_DIR = Path("data/pairs/nces_digest")


def _get(url: str, *, timeout: int = 30) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def list_table_urls() -> list[str]:
    html = _get(MENU_URL)
    if not html:
        sys.exit(f"[fatal] could not fetch menu page {MENU_URL}")
    ids = sorted({f"dt22_{a}.{b}.asp" for a, b in LINK_RE.findall(html)})
    return [TABLE_BASE + i for i in ids]


def _pick_data_table(doc) -> object | None:
    """The data table is the <table> with a <thead> (or the most <th>) and
    enough rows to be real."""
    best, best_th = None, 0
    for t in doc.xpath(".//table"):
        n_rows = len(t.xpath(".//tr"))
        n_th = len(t.xpath(".//th"))
        has_thead = bool(t.xpath(".//thead"))
        if n_rows < 3:
            continue
        if (has_thead or n_th > 0) and n_th >= best_th:
            best, best_th = t, n_th
    return best


def _page_title(doc) -> str | None:
    text = " ".join("".join(doc.itertext()).split())
    m = TITLE_RE.search(text)
    if not m:
        return None
    cap = m.group(0)
    # Trim at the end of the descriptive clause (first sentence-ish boundary
    # after the leading "Table N." — bounded so notes don't leak in).
    return cap[:300].strip()


def build_pair(url: str) -> dict | None:
    html = _get(url)
    if not html:
        return None
    try:
        doc = lxml_html.fromstring(html)
    except Exception:
        return None
    table = _pick_data_table(doc)
    if table is None:
        return None
    caption = _page_title(doc)
    # Inject the caption as the table's <caption> first child so the multi
    # builder's HTML5 path (which reads ./caption) picks it up.
    if caption and not table.xpath("./caption"):
        cap_el = lxml_html.Element("caption")
        cap_el.text = caption
        table.insert(0, cap_el)
    table_html = lxml_html.tostring(table, encoding="unicode")
    m = LINK_RE.search(url)
    table_id = f"dt22_{m.group(1)}.{m.group(2)}" if m else url.rsplit("/", 1)[-1]
    return {
        "source": "nces_digest",
        "table_id": table_id,
        "variant_id": table_id,
        "title": caption,
        "url": url,
        "output_html": table_html,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=150, help="max table pages to fetch")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip table ids already written")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    urls = list_table_urls()
    print(f"[menu] {len(urls)} table pages listed; fetching up to {args.limit}")

    n_ok = n_skip = n_fail = 0
    for url in urls[: args.limit]:
        m = LINK_RE.search(url)
        tid = f"dt22_{m.group(1)}.{m.group(2)}" if m else None
        out_path = args.out_dir / f"{tid}.json"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        pair = build_pair(url)
        if pair is None:
            n_fail += 1
        else:
            out_path.write_text(json.dumps(pair, ensure_ascii=False), encoding="utf-8")
            n_ok += 1
        if (n_ok + n_fail) % 25 == 0:
            print(f"[progress] ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)
        time.sleep(args.delay)

    print(f"[done] wrote {n_ok} pairs to {args.out_dir} (fail={n_fail}, skip={n_skip})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
