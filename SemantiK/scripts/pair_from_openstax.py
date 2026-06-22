"""Generate (features → HTML) training pairs from OpenStax books.

OpenStax publishes 127 textbooks under CC-BY 4.0 (commercial-OK) or
CC-BY-NC-SA (non-commercial — rejected per feedback_license_policy).
This ingester:

  1. Pulls the book catalog via OpenStax's CMS API and filters to
     commercial-permissive licenses.
  2. For each allowed book: fetch any page once, extract the book tree
     from the embedded __PRELOADED_STATE__ Redux blob to enumerate all
     pages.
  3. For each page: fetch the rendered HTML, strip OpenStax chrome,
     normalize into a clean WCAG-ish HTML fragment, render the fragment
     to PDF, OCR-extract features, save the pair.

Output shape matches pair_from_wikipedia.py / pair_from_arxiv.py so
data/build_classifier_data.py consumes it without changes.

Usage:
    python scripts/pair_from_textbook.py --book algebra-and-trigonometry --out-dir data/pairs/openstax
    python scripts/pair_from_textbook.py --all --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool

OPENSTAX = "https://openstax.org"
CMS = f"{OPENSTAX}/apps/cms/api/v2"
USER_AGENT = "dart-semantic/0.0.1"

# Commercial-OK license prefixes (matches arxiv_license.py).
COMMERCIAL_OK_PREFIXES = (
    "https://creativecommons.org/licenses/by/",
    "http://creativecommons.org/licenses/by/",
    "https://creativecommons.org/publicdomain/",
    "http://creativecommons.org/publicdomain/",
)


def commercial_ok(license_url: str | None) -> bool:
    if not license_url:
        return False
    return any(license_url.startswith(p) for p in COMMERCIAL_OK_PREFIXES)


# ---------- catalog ----------

def list_commercial_books() -> list[dict]:
    """Return [{slug, title, license_url, cnx_id}] for every CC-BY book."""
    out: list[dict] = []
    offset = 0
    while True:
        r = requests.get(
            f"{CMS}/pages/",
            params={"type": "books.Book", "limit": 100, "offset": offset,
                    "fields": "slug,title,license_url,cnx_id"},
            headers={"User-Agent": USER_AGENT}, timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            break
        for b in items:
            slug = b.get("meta", {}).get("slug") or b.get("slug")
            if not slug:
                continue
            if commercial_ok(b.get("license_url")):
                out.append({
                    "slug": slug,
                    "title": b.get("title", slug),
                    "license_url": b.get("license_url"),
                    "cnx_id": b.get("cnx_id"),
                })
        offset += len(items)
        if offset >= data.get("meta", {}).get("total_count", offset):
            break
    return out


# ---------- book tree extraction ----------

PRELOADED_PREFIX = "window.__PRELOADED_STATE__ = "


def fetch_book_tree(book_slug: str) -> list[tuple[str, str]]:
    """Return [(page_slug, page_title), ...] for every page leaf in the book.

    Pulls the TOC from the __PRELOADED_STATE__ embedded in any page's HTML.
    We request a stable entry point ("1-introduction") that exists on
    every OpenStax book; if that fails we walk the catalog to find one.
    """
    probe_urls = [
        f"{OPENSTAX}/books/{book_slug}/pages/preface",
        f"{OPENSTAX}/books/{book_slug}/pages/1-introduction",
    ]
    html = None
    for url in probe_urls:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        if r.ok and PRELOADED_PREFIX in r.text:
            html = r.text
            break
    if html is None:
        raise RuntimeError(f"could not find preloaded state for {book_slug}")

    state = _extract_preloaded_state(html)
    tree = state.get("content", {}).get("book", {}).get("tree")
    if not tree:
        raise RuntimeError(f"no book tree in preloaded state for {book_slug}")

    slugs: list[tuple[str, str]] = []

    def walk(node):
        if "contents" in node and node["contents"]:
            for c in node["contents"]:
                walk(c)
        else:
            slug = node.get("slug")
            title_html = node.get("title", "")
            # Titles are HTML fragments with <span> wrappers; strip tags.
            title = BeautifulSoup(title_html, "lxml").get_text(" ", strip=True)
            if slug:
                slugs.append((slug, title))

    walk(tree)
    return slugs


def _extract_preloaded_state(html: str) -> dict:
    """Parse the balanced-brace JS object after `window.__PRELOADED_STATE__ =`."""
    idx = html.find(PRELOADED_PREFIX)
    if idx < 0:
        raise RuntimeError("PRELOADED_STATE not found in page")
    start = idx + len(PRELOADED_PREFIX)
    depth = 0
    in_str = False
    esc = False
    end = start
    for i in range(start, len(html)):
        c = html[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    return json.loads(html[start:end])


# ---------- page content ----------

# CSS selectors for the meaningful part of an OpenStax page. OpenStax
# wraps content in a section with data-type="chapter" or data-type="page";
# inside that everything is semantic (h1..h4, p, ul/ol, table, figure).
CONTENT_SELECTORS = [
    "section[data-type='page']",
    "section[data-type='chapter']",
    "[data-type='page']",
    "[data-type='chapter']",
    "main article",
    "main",
]

# Stripped as OpenStax chrome: navigation links, metadata, attribution boilerplate.
STRIP_SELECTORS = [
    "nav", "[role='navigation']", "header[role='banner']",
    "footer", "[role='contentinfo']", ".os-figure-link-container",
    ".os-eoc-links", ".footnote-refs", ".print-reference",
    "[data-type='footnote-refs']", ".os-hide-on-small",
]


def extract_page_content(book_slug: str, page_slug: str) -> tuple[str, str, str]:
    """Return (title, accessible_html_body, raw_source_html) for one page.

    The accessible HTML is wrapped in a <main> and includes only the
    semantic content. raw_source_html is the unmodified upstream
    response — used by Phase 3c BERT-Semantic for richer doc-role
    classes that the chrome-stripping step removes.
    """
    url = f"{OPENSTAX}/books/{book_slug}/pages/{page_slug}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    raw_source_html = r.text
    soup = BeautifulSoup(raw_source_html, "lxml")

    for sel in STRIP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    container = None
    for sel in CONTENT_SELECTORS:
        container = soup.select_one(sel)
        if container:
            break
    if container is None:
        raise RuntimeError(f"no content container on {url}")

    # Title: prefer h1 inside the container; fall back to page_slug.
    h1 = container.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else page_slug

    # Build a minimal clean HTML document.
    body_html = _clean_container(container)
    html_doc = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>{_escape(title)}</title>'
        '<style>body{font-family:Georgia,serif;max-width:7in;margin:1in auto;'
        'line-height:1.5;color:#111}h1,h2,h3{margin-top:1.2em}'
        'table{border-collapse:collapse;margin:1em 0}'
        'th,td{border:1px solid #444;padding:0.3em 0.6em}'
        'th{background:#eee}figure{margin:1em 0;text-align:center}'
        'a{padding:2px 1px}pre{white-space:pre-wrap;word-break:break-word}'
        '</style></head><body><main>'
        f'{body_html}'
        '</main></body></html>'
    )
    return title, html_doc, raw_source_html


def _clean_container(container) -> str:
    """Normalize OpenStax markup to WCAG-clean fragments.

    Strips data-* attributes, class names, IDs; keeps structural tags
    (h1-h6, p, ul/ol/li, dl/dt/dd, table/tr/th/td, figure/figcaption,
    blockquote, pre, a, em, strong, code, math). Everything else is
    unwrapped (text preserved, tag removed).
    """
    # Keep this tag set; anything else becomes its children unwrapped.
    keep = {"h1", "h2", "h3", "h4", "h5", "h6", "p",
            "ul", "ol", "li", "dl", "dt", "dd",
            "table", "thead", "tbody", "tr", "th", "td", "caption",
            "figure", "figcaption", "img",
            "blockquote", "pre", "code",
            "a", "em", "strong", "i", "b",
            "math"}
    keep_attrs = {"href", "src", "alt", "colspan", "rowspan", "scope",
                  "cite", "lang", "alttext", "display"}

    # Unwrap tags not in keep (but preserve text).
    for tag in list(container.find_all(True)):
        if tag.name not in keep:
            tag.unwrap()

    # Strip unwanted attributes.
    for tag in container.find_all(True):
        for attr in list(tag.attrs):
            if attr not in keep_attrs:
                del tag.attrs[attr]

    # Make sure every img has an alt attribute (empty string for decorative).
    for img in container.find_all("img"):
        if "alt" not in img.attrs or img["alt"] is None:
            img["alt"] = ""

    # Ensure every <th> has a scope; default to col.
    for th in container.find_all("th"):
        if "scope" not in th.attrs:
            th["scope"] = "col"

    return str(container)


def _escape(s: str) -> str:
    import html as h
    return h.escape(s, quote=False)


# ---------- per-page processing ----------

def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s.lower()).strip("_")
    return s[:maxlen] or "untitled"


def process_page(validator: HtmlValidator,
                 work: tuple[str, str, str, str]) -> dict:
    """Worker: fetch + render + feature-extract for one (book, page)."""
    book_slug, page_slug, page_title, out_dir_str = work
    out_dir = Path(out_dir_str)
    stats = {"book": book_slug, "page": page_slug, "title": page_title,
             "ok": 0, "fetch_error": 0, "empty": 0,
             "axe_drop": 0}

    try:
        title, html_doc, raw_source_html = extract_page_content(book_slug, page_slug)
    except Exception as exc:
        stats["fetch_error"] = 1
        stats["msg"] = str(exc)
        return stats

    # Gate: does our axe-core setup accept this HTML?
    result = validator.check(html_doc)
    if not result.ok:
        stats["axe_drop"] = 1
        stats["msg"] = "; ".join(result.reasons[:2])
        return stats

    # Render + feature-extract.
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "page.pdf"
        validator.render_pdf(html_doc, pdf_path)
        input_ocr = pdf_to_ocr_text(pdf_path)

    if not input_ocr.strip():
        stats["empty"] = 1
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{slugify(book_slug)}__{slugify(page_slug, 40)}.json"
    pair = {
        "source": "openstax",
        "book": book_slug,
        "page": page_slug,
        "title": title,
        "variant_id": f"{book_slug}/{page_slug}",
        "input_ocr": input_ocr,
        "output_html": html_doc,
        "raw_source_html": raw_source_html,
    }
    (out_dir / fname).write_text(json.dumps(pair, ensure_ascii=False))
    stats["ok"] = 1
    return stats


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--book", help="Single book slug (e.g. algebra-and-trigonometry)")
    g.add_argument("--all", action="store_true",
                   help="Process every CC-BY book in the catalog")
    ap.add_argument("--out-dir", type=Path, default=Path("data/pairs/openstax"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-pages", type=int, default=None,
                    help="Cap on pages per book (useful for smoke tests)")
    ap.add_argument("--limit-books", type=int, default=None,
                    help="Cap on total books processed when --all")
    args = ap.parse_args()

    if args.book:
        books = [{"slug": args.book, "title": args.book, "license_url": "manual", "cnx_id": None}]
    else:
        print("[catalog] pulling commercial-OK book list...", file=sys.stderr)
        books = list_commercial_books()
        print(f"[catalog] {len(books)} commercial-OK books", file=sys.stderr)
        if args.limit_books:
            books = books[:args.limit_books]

    # Build the full work list across books.
    work_items: list[tuple[str, str, str, str]] = []
    for book in books:
        slug = book["slug"]
        try:
            pages = fetch_book_tree(slug)
        except Exception as exc:
            print(f"[{slug}] toc error: {exc}", file=sys.stderr)
            continue
        if args.limit_pages:
            pages = pages[:args.limit_pages]
        print(f"[{slug}] {len(pages)} pages", file=sys.stderr)
        for page_slug, page_title in pages:
            work_items.append((slug, page_slug, page_title, str(args.out_dir)))

    # Skip already-processed pairs.
    existing = {p.stem for p in args.out_dir.glob("*.json")} if args.out_dir.exists() else set()
    work_items = [w for w in work_items
                  if f"{slugify(w[0])}__{slugify(w[1], 40)}" not in existing]
    print(f"[plan] {len(work_items)} pages to process "
          f"({len(existing)} already done)", file=sys.stderr)

    totals = {"ok": 0, "fetch_error": 0, "empty": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_page, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        if done % 25 == 0 or stats.get("fetch_error") or stats.get("empty"):
            status = next((k for k in ("ok", "fetch_error", "empty", "axe_drop")
                           if stats.get(k)), "?")
            print(f"[{done}/{len(work_items)}] {stats['book']}/{stats['page']:40} "
                  f"{status}", file=sys.stderr)

    elapsed = time.time() - start
    rate = totals["ok"] / max(1, done) * 100
    print(f"\n[summary] processed={done}  ok={totals['ok']} ({rate:.1f}%)  "
          f"fetch_err={totals['fetch_error']}  empty={totals['empty']}  "
          f"axe_drop={totals['axe_drop']}  in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
