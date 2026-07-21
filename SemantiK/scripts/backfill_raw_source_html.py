"""Backfill `raw_source_html` on existing pair JSONs.

Phase 3c BERT-Semantic needs the rich semantic markup that
`emit_html.emit()` strips (ltx_authors, ltx_abstract, ltx_bibliography,
class="abstract", <address>, <cite>, <footer>, …). All future fetcher
runs save `raw_source_html` alongside `output_html`. This script
augments the 5,679 existing pair files in-place.

Strategy per source:
    arxiv      — copy from data/ar5iv_html_cache/<arxiv_id>.html
                 (zero network, fully local)
    wikipedia  — refetch from REST API (workers=1 to avoid 429)
    wikiquote  — refetch from REST API (workers=1)
    gutenberg  — refetch from gutenberg.org (small batch, 65 books)
    openstax   — refetch from openstax.org
    federal_register — refetch the XML doc (saved as
                 raw_source_xml not raw_source_html)
    forms      — N/A (output_html IS the only HTML for synth forms)
    synthetic_blockquote_code — copy output_html → raw_source_html

Usage:
    # dry run by source — count what's missing
    python scripts/backfill_raw_source_html.py --dry-run

    # backfill arxiv from local cache (fast, no network)
    python scripts/backfill_raw_source_html.py --source arxiv

    # backfill wikipedia (slow — ~70 min for 2K pairs at workers=1)
    python scripts/backfill_raw_source_html.py --source wikipedia
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests


PAIRS_ROOT = Path("data/pairs")
ARXIV_CACHE = Path("data/ar5iv_html_cache")
USER_AGENT = "semantik-structure/0.0.1"


# ---------------------------------------------------------------------
# Source-specific backfill functions
# ---------------------------------------------------------------------


def backfill_arxiv(pair: dict) -> tuple[bool, str]:
    """Look up the cached ar5iv HTML by arxiv_id."""
    arxiv_id = pair.get("arxiv_id")
    if not arxiv_id:
        return False, "no arxiv_id"
    cache_path = ARXIV_CACHE / f"{arxiv_id}.html"
    if not cache_path.exists():
        return False, f"no cache for {arxiv_id}"
    pair["raw_source_html"] = cache_path.read_text()
    return True, "ok"


_WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"
_WQ_REST = "https://en.wikiquote.org/api/rest_v1/page/html/{title}"
_GUTENBERG = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}-images.html"


def backfill_wikipedia(pair: dict) -> tuple[bool, str]:
    title = pair.get("article_title")
    if not title:
        return False, "no article_title"
    url = _WIKI_REST.format(title=quote(title.replace(" ", "_")))
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        return False, f"fetch: {exc}"
    pair["raw_source_html"] = r.text
    return True, "ok"


def backfill_wikiquote(pair: dict) -> tuple[bool, str]:
    title = pair.get("article_title")
    if not title:
        return False, "no article_title"
    url = _WQ_REST.format(title=quote(title.replace(" ", "_")))
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        return False, f"fetch: {exc}"
    pair["raw_source_html"] = r.text
    return True, "ok"


def backfill_gutenberg(pair: dict) -> tuple[bool, str]:
    book_id = pair.get("book_id")
    if not book_id:
        return False, "no book_id"
    # Try image-bearing HTML edition first (matches pair_from_gutenberg
    # default URL), then plain.
    for url_template in (
        "https://www.gutenberg.org/cache/epub/{}/pg{}-images.html",
        "https://www.gutenberg.org/cache/epub/{}/pg{}.html",
    ):
        url = url_template.format(book_id, book_id)
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.ok:
                pair["raw_source_html"] = r.text
                return True, "ok"
        except Exception:
            pass
    return False, "fetch: all gutenberg URLs failed"


def backfill_openstax(pair: dict) -> tuple[bool, str]:
    book = pair.get("book")
    page = pair.get("page")
    if not book or not page:
        return False, "no book/page"
    url = f"https://openstax.org/books/{book}/pages/{page}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        return False, f"fetch: {exc}"
    pair["raw_source_html"] = r.text
    return True, "ok"


def backfill_federal_register(pair: dict) -> tuple[bool, str]:
    """Refetch the original XML — saved under raw_source_xml since
    Federal Register publishes XML, not HTML."""
    # Federal Register pairs don't store the xml_url; reconstruct from
    # the document_number via the public API.
    number = pair.get("document_number")
    if not number:
        return False, "no document_number"
    meta_url = f"https://www.federalregister.gov/api/v1/documents/{number}.json"
    try:
        r = requests.get(meta_url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        meta = r.json()
        xml_url = meta.get("full_text_xml_url")
        if not xml_url:
            return False, "no xml_url in metadata"
        r2 = requests.get(xml_url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r2.raise_for_status()
    except Exception as exc:
        return False, f"fetch: {exc}"
    pair["raw_source_xml"] = r2.text
    return True, "ok"


def backfill_synthetic(pair: dict) -> tuple[bool, str]:
    """Synth: output_html IS the raw."""
    pair["raw_source_html"] = pair.get("output_html", "")
    return True, "ok"


def backfill_forms(pair: dict) -> tuple[bool, str]:
    """Forms: no upstream HTML exists; emit_html.emit() output IS the
    only HTML representation. Mark this explicitly so future readers
    don't treat absence as a backfill failure."""
    pair["raw_source_html"] = pair.get("output_html", "")
    pair["raw_source_html_note"] = "synthesized: pdf-form has no upstream HTML"
    return True, "ok"


BACKFILLERS = {
    "arxiv": backfill_arxiv,
    "wikipedia": backfill_wikipedia,
    "wikiquote": backfill_wikiquote,
    "gutenberg": backfill_gutenberg,
    "openstax": backfill_openstax,
    "federal_register": backfill_federal_register,
    "synthetic_blockquote_code": backfill_synthetic,
    "pdf_form": backfill_forms,
    # Directory name for forms is "forms"; mapping below handles both.
    "forms": backfill_forms,
}


# Map directory name → source slug (where they differ)
DIR_TO_SOURCE = {
    "synthetic_blockquote_code": "synthetic_blockquote_code",
    "forms": "pdf_form",
}


def list_pairs(source_dir: Path) -> list[Path]:
    return sorted(source_dir.glob("*.json"))


def needs_backfill(pair: dict, source: str) -> bool:
    if source == "federal_register":
        return "raw_source_xml" not in pair
    return "raw_source_html" not in pair


# For wikipedia/wikiquote, sibling section-pairs share one raw_source_html
# (they're sections of the same article). Group by article_title and
# fetch ONCE per article — turns ~12× wasteful re-fetches into one.
def _article_dedup_key(pair: dict, source: str) -> str | None:
    if source in ("wikipedia", "wikiquote"):
        return pair.get("article_title")
    if source == "gutenberg":
        # All chapters of one book share the raw HTML.
        bid = pair.get("book_id")
        return f"book_{bid}" if bid else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="Process only this source directory "
                    "(default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count what would be backfilled, don't write")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N successful backfills (debug)")
    ap.add_argument("--rate-limit-sleep", type=float, default=1.0,
                    help="Seconds to sleep between fetches for network sources")
    args = ap.parse_args()

    if args.source:
        source_dirs = [PAIRS_ROOT / args.source]
    else:
        source_dirs = sorted(d for d in PAIRS_ROOT.iterdir() if d.is_dir())

    grand_totals = Counter()

    for sdir in source_dirs:
        if not sdir.exists():
            print(f"[skip] {sdir} does not exist", file=sys.stderr)
            continue
        dir_name = sdir.name
        source = DIR_TO_SOURCE.get(dir_name, dir_name)
        backfiller = BACKFILLERS.get(source)
        if backfiller is None:
            print(f"[skip] no backfiller for source={source!r} "
                  f"(dir={dir_name})", file=sys.stderr)
            continue

        pairs = list_pairs(sdir)
        print(f"\n[{dir_name}] {len(pairs)} pair files; source={source}",
              file=sys.stderr)
        local_totals = Counter({"total": len(pairs)})
        is_network_source = source in {
            "wikipedia", "wikiquote", "gutenberg", "openstax",
            "federal_register",
        }
        # Per-article cache so we fetch ONCE per article and apply to
        # all its section-pairs. Saves ~12× requests on wikipedia.
        article_cache: dict[str, str] = {}

        for i, p in enumerate(pairs, start=1):
            try:
                pair = json.loads(p.read_text())
            except Exception as exc:
                print(f"  [{i}/{len(pairs)}] LOAD-FAIL {p.name}: {exc}",
                      file=sys.stderr)
                local_totals["load_error"] += 1
                continue

            if not needs_backfill(pair, source):
                local_totals["already"] += 1
                continue

            # Dedup: reuse cached raw HTML if a sibling-pair already fetched it.
            dkey = _article_dedup_key(pair, source)
            cache_hit = False
            if dkey is not None and dkey in article_cache:
                pair["raw_source_html"] = article_cache[dkey]
                cache_hit = True
                local_totals["cache_hit"] += 1
            else:
                ok, msg = backfiller(pair)
                if not ok:
                    if local_totals["fail"] < 5:
                        print(f"  [{i}/{len(pairs)}] FAIL {p.name}: {msg}",
                              file=sys.stderr)
                    local_totals["fail"] += 1
                    continue
                if dkey is not None:
                    article_cache[dkey] = pair.get("raw_source_html", "")

            if args.dry_run:
                local_totals["would_backfill"] += 1
            else:
                p.write_text(json.dumps(pair, ensure_ascii=False))
                local_totals["backfilled"] += 1

            if (local_totals["backfilled"] + local_totals["would_backfill"]
                    + local_totals["already"] + local_totals["fail"]) % 100 == 0:
                print(f"  [{i}/{len(pairs)}] backfilled="
                      f"{local_totals['backfilled']} cache_hits="
                      f"{local_totals['cache_hit']} fail="
                      f"{local_totals['fail']}",
                      file=sys.stderr)

            # Only sleep on actual fetches (not cache hits).
            if is_network_source and not args.dry_run and not cache_hit:
                time.sleep(args.rate_limit_sleep)

            if (args.limit
                and local_totals["backfilled"] >= args.limit):
                print(f"  [limit] stopped at backfilled={args.limit}",
                      file=sys.stderr)
                break

        for k, v in local_totals.items():
            grand_totals[k] += v
        print(f"[{dir_name}] {dict(local_totals)}", file=sys.stderr)

    print(f"\n[grand totals] {dict(grand_totals)}")


if __name__ == "__main__":
    main()
