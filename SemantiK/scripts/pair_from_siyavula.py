"""Generate (features → HTML) training pairs from Siyavula CC-BY ePUBs.

!!! PARKED 2026-06-15 — DO NOT RUN INTO TRAINING AS-IS. Siyavula's CC-BY
ePUBs render every equation as an alt-less ``<img class="math-inline">``
with NO LaTeX/MathML fallback, so math sections produce axe-PASSING but
semantically INACCESSIBLE ground truth (math → silent decorative image).
Shipping that would train DART to hide math from screen readers — the
opposite of the product's purpose. The license/download/extraction
machinery below is correct and verified; what's missing before use is a
math-image-density guard (prose-only) or a math-recovery step. Kept
in-tree as documented, not wired into data/build_semantic_data.py.

Siyavula publishes open school textbooks (Grades 4-12 maths & science).
Per title it offers two license flavours, and only one is usable for us:

  - branded PDF + ``*_CC-BY-ND.epub``  → CC-BY-**ND** (NoDerivatives) → REJECTED
  - ``*_CC-BY.epub``                   → CC-BY 4.0                    → USABLE

DART's output is a derivative work, so the NoDerivatives files are off
limits per feedback_license_policy. This ingester fetches ONLY the
``*_CC-BY.epub`` files, verifies the embedded Creative Commons license is
neither NC nor ND (fail-closed — see ``epub_license_ok``), then for each
content section in the ePUB spine: cleans to WCAG-ish HTML, runs the
axe-core gate, renders to PDF, and OCR-extracts input features.

Two Siyavula-specific gotchas, both load-bearing:

  1. The CC-BY ePUBs are served via Git LFS. A non-browser User-Agent
     gets a 134-byte LFS *pointer* instead of the file, so we send a
     browser UA and verify the download is a complete, valid zip.
  2. The OPF asserts no ``dc:rights``; the license is carried by the
     ``*_CC-BY.epub`` filename plus an embedded
     ``creativecommons.org/licenses/by/4.0`` URL in the content. We gate
     on the embedded URL, not the filename.

Output shape matches pair_from_textbook.py / pair_from_wikipedia.py so
data/build_semantic_data.py consumes it without changes.

Usage:
    python scripts/pair_from_siyavula.py --book Gr10_Mathematics_Learner_Eng
    python scripts/pair_from_siyavula.py --all --workers 4 --limit-books 3
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool

SIYAVULA = "https://www.siyavula.com"
READ_PAGE = f"{SIYAVULA}/read"

# Siyavula gates the LFS-backed ePUBs on User-Agent: a tool UA gets the
# 134-byte LFS pointer, a browser UA gets the real binary.
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)

# Any Creative Commons license URL whose token set includes one of these is
# non-commercial or no-derivatives — both rejected for a commercial,
# transform-the-document product.
_FORBIDDEN_CC_TOKENS = ("-nc", "-nd")
_CC_URL_RE = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/[0-9.]+", re.I)


# ---------- catalog ----------


def list_ccby_epubs() -> list[dict]:
    """Return [{book_id, subject, url}] for every ``*_CC-BY.epub`` on /read."""
    r = requests.get(READ_PAGE, headers={"User-Agent": BROWSER_UA}, timeout=30)
    r.raise_for_status()
    out: list[dict] = []
    seen: set[str] = set()
    for href in re.findall(r'href="([^"]*_CC-BY\.epub)"', r.text):
        url = href if href.startswith("http") else urljoin_site(href)
        if url in seen:
            continue
        seen.add(url)
        fname = href.rsplit("/", 1)[-1]
        subject = (
            href.split("/downloads/books/", 1)[-1].split("/", 1)[0]
            if "/downloads/books/" in href
            else "unknown"
        )
        out.append(
            {
                "book_id": fname[: -len("_CC-BY.epub")],
                "subject": subject,
                "url": url,
            }
        )
    return out


def urljoin_site(href: str) -> str:
    return (
        href
        if href.startswith("http")
        else f"{SIYAVULA}{href if href.startswith('/') else '/' + href}"
    )


# ---------- download (LFS-aware) ----------


def download_epub(url: str, dest: Path) -> None:
    """Download an ePUB to ``dest``, verifying it is complete and a real zip.

    Raises if the response is the Git LFS pointer (wrong UA) or truncated.
    """
    with requests.get(url, headers={"User-Agent": BROWSER_UA}, stream=True, timeout=120) as r:
        r.raise_for_status()
        expected = int(r.headers.get("Content-Length", 0))
        ctype = r.headers.get("Content-Type", "")
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    got = dest.stat().st_size
    if expected and got != expected:
        raise RuntimeError(f"truncated download: got {got} of {expected} bytes")
    if got < 1024 or not zipfile.is_zipfile(dest):
        head = dest.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError("got Git LFS pointer, not the ePUB (UA gating)")
        raise RuntimeError(f"not a valid ePUB (size={got}, type={ctype})")


# ---------- license gate (fail-closed) ----------


def epub_license_ok(zf: zipfile.ZipFile) -> tuple[bool, list[str]]:
    """True iff the ePUB embeds a commercial-OK CC license and NO NC/ND one.

    Fail-closed: returns False if no Creative Commons URL is found at all
    (we will not ship a pair whose license we cannot positively confirm),
    or if any embedded CC URL carries an ``-nc`` or ``-nd`` token.
    """
    found: set[str] = set()
    for name in zf.namelist():
        if not name.lower().endswith((".html", ".xhtml", ".xml", ".opf")):
            continue
        try:
            text = zf.read(name).decode("utf-8", "replace")
        except Exception:
            continue
        for tokens in _CC_URL_RE.findall(text):
            found.add(tokens.lower())
    if not found:
        return False, []
    bad = [t for t in found if any(tok in t for tok in _FORBIDDEN_CC_TOKENS)]
    return (not bad), sorted(found)


# ---------- spine / content extraction ----------

# Spine docs that are navigation or front/back matter, not body content.
_SKIP_DOC_RE = re.compile(r"(nav|cover|title-?page|copyright|front-?matter|colophon)", re.I)


def _opf_path(zf: zipfile.ZipFile) -> str:
    container = zf.read("META-INF/container.xml").decode("utf-8", "replace")
    m = re.search(r'full-path="([^"]+)"', container)
    if not m:
        raise RuntimeError("no rootfile in container.xml")
    return m.group(1)


def epub_sections(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return [(content_zip_path, title)] for each body section in spine order."""
    opf_path = _opf_path(zf)
    opf_dir = posixpath.dirname(opf_path)
    opf = BeautifulSoup(zf.read(opf_path).decode("utf-8", "replace"), "xml")

    manifest = {it.get("id"): it.get("href") for it in opf.find_all("item") if it.get("id")}
    sections: list[tuple[str, str]] = []
    for ref in opf.find_all("itemref"):
        href = manifest.get(ref.get("idref"))
        if not href:
            continue
        if _SKIP_DOC_RE.search(href):
            continue
        zip_path = posixpath.normpath(posixpath.join(opf_dir, href))
        if zip_path not in zf.namelist():
            continue
        sections.append((zip_path, posixpath.basename(href)))
    return sections


def extract_section_html(zf: zipfile.ZipFile, content_path: str) -> tuple[str, str, str]:
    """Return (title, clean_html_doc, raw_source_html) for one ePUB section."""
    raw_source_html = zf.read(content_path).decode("utf-8", "replace")
    soup = BeautifulSoup(raw_source_html, "lxml")

    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    container = soup.body or soup
    heading = container.find(re.compile(r"^h[1-6]$"))
    title = heading.get_text(" ", strip=True) if heading else ""

    body_html = _clean_container(container)
    html_doc = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8">'
        f"<title>{_escape(title or 'Section')}</title>"
        "<style>body{font-family:Georgia,serif;max-width:7in;margin:1in auto;"
        "line-height:1.5;color:#111}h1,h2,h3{margin-top:1.2em}"
        "table{border-collapse:collapse;margin:1em 0}"
        "th,td{border:1px solid #444;padding:0.3em 0.6em}"
        "th{background:#eee}figure{margin:1em 0;text-align:center}"
        "a{padding:2px 1px}pre{white-space:pre-wrap;word-break:break-word}"
        "</style></head><body><main>"
        f"{body_html}"
        "</main></body></html>"
    )
    return title, html_doc, raw_source_html


def _clean_container(container) -> str:
    """Normalize CNXMLPlus markup to WCAG-clean fragments.

    Keeps structural tags; unwraps everything else (preserving text);
    strips presentational attributes; guarantees img@alt and th@scope.
    Mirrors pair_from_textbook._clean_container so the two sources emit
    structurally identical ground truth.
    """
    keep = {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "caption",
        "figure",
        "figcaption",
        "img",
        "blockquote",
        "pre",
        "code",
        "a",
        "em",
        "strong",
        "i",
        "b",
        "math",
    }
    keep_attrs = {
        "href",
        "src",
        "alt",
        "colspan",
        "rowspan",
        "scope",
        "cite",
        "lang",
        "alttext",
        "display",
    }

    for tag in list(container.find_all(True)):
        if tag.name not in keep:
            tag.unwrap()
    for tag in container.find_all(True):
        for attr in list(tag.attrs):
            if attr not in keep_attrs:
                del tag.attrs[attr]
    for img in container.find_all("img"):
        if "alt" not in img.attrs or img["alt"] is None:
            img["alt"] = ""
    for th in container.find_all("th"):
        if "scope" not in th.attrs:
            th["scope"] = "col"
    return str(container)


def _escape(s: str) -> str:
    import html as h

    return h.escape(s, quote=False)


def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s.lower()).strip("_")
    return s[:maxlen] or "untitled"


# ---------- per-section processing ----------


def process_section(validator: HtmlValidator, work: tuple[str, str, str, str]) -> dict:
    """Worker: extract + gate + render + feature-extract for one section."""
    book_id, epub_path, content_path, out_dir_str = work
    out_dir = Path(out_dir_str)
    stats = {
        "book": book_id,
        "page": content_path,
        "ok": 0,
        "fetch_error": 0,
        "empty": 0,
        "axe_drop": 0,
    }

    try:
        with zipfile.ZipFile(epub_path) as zf:
            title, html_doc, raw_source_html = extract_section_html(zf, content_path)
    except Exception as exc:
        stats["fetch_error"] = 1
        stats["msg"] = str(exc)
        return stats

    result = validator.check(html_doc)
    if not result.ok:
        stats["axe_drop"] = 1
        stats["msg"] = "; ".join(result.reasons[:2])
        return stats

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "page.pdf"
        validator.render_pdf(html_doc, pdf_path)
        input_ocr = pdf_to_ocr_text(pdf_path)

    if not input_ocr.strip():
        stats["empty"] = 1
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)
    page_slug = slugify(posixpath.basename(content_path).rsplit(".", 1)[0], 40)
    fname = f"{slugify(book_id)}__{page_slug}.json"
    pair = {
        "source": "siyavula",
        "book": book_id,
        "page": content_path,
        "title": title,
        "variant_id": f"{book_id}/{page_slug}",
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
    g.add_argument(
        "--book", help="Substring filter on a CC-BY book_id (e.g. Gr10_Mathematics_Learner_Eng)"
    )
    g.add_argument("--all", action="store_true", help="Process every CC-BY ePUB in the catalog")
    ap.add_argument("--out-dir", type=Path, default=Path("data/pairs/siyavula"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--limit-pages", type=int, default=None, help="Cap on sections per book (smoke tests)"
    )
    ap.add_argument("--limit-books", type=int, default=None, help="Cap on total books processed")
    args = ap.parse_args()

    print("[catalog] pulling CC-BY ePUB list...", file=sys.stderr)
    books = list_ccby_epubs()
    print(f"[catalog] {len(books)} CC-BY ePUBs", file=sys.stderr)
    if args.book:
        books = [b for b in books if args.book.lower() in b["book_id"].lower()]
        print(f"[catalog] {len(books)} match --book {args.book!r}", file=sys.stderr)
    if args.limit_books:
        books = books[: args.limit_books]

    # Download each ePUB once into a run-scoped temp dir, license-gate it,
    # then enumerate its sections. The temp dir stays open for the whole
    # pool run because workers read the files by path.
    tmpdir = tempfile.TemporaryDirectory(prefix="siyavula_epubs_")
    work_items: list[tuple[str, str, str, str]] = []
    for book in books:
        epub_path = Path(tmpdir.name) / f"{book['book_id']}.epub"
        try:
            download_epub(book["url"], epub_path)
        except Exception as exc:
            print(f"[{book['book_id']}] download error: {exc}", file=sys.stderr)
            continue
        with zipfile.ZipFile(epub_path) as zf:
            ok, urls = epub_license_ok(zf)
            if not ok:
                print(
                    f"[{book['book_id']}] LICENSE REJECT (embedded: {urls or 'none'})",
                    file=sys.stderr,
                )
                continue
            sections = epub_sections(zf)
        if args.limit_pages:
            sections = sections[: args.limit_pages]
        print(f"[{book['book_id']}] license OK {urls}; {len(sections)} sections", file=sys.stderr)
        for content_path, _title in sections:
            work_items.append((book["book_id"], str(epub_path), content_path, str(args.out_dir)))

    # Skip already-processed pairs.
    existing = {p.stem for p in args.out_dir.glob("*.json")} if args.out_dir.exists() else set()
    work_items = [
        w
        for w in work_items
        if f"{slugify(w[0])}__{slugify(posixpath.basename(w[2]).rsplit('.', 1)[0], 40)}"
        not in existing
    ]
    print(
        f"[plan] {len(work_items)} sections to process ({len(existing)} already done)",
        file=sys.stderr,
    )

    totals = {"ok": 0, "fetch_error": 0, "empty": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_section, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        if done % 25 == 0 or stats.get("fetch_error"):
            status = next(
                (k for k in ("ok", "fetch_error", "empty", "axe_drop") if stats.get(k)), "?"
            )
            print(
                f"[{done}/{len(work_items)}] {stats['book']}/{stats['page']:45} {status}",
                file=sys.stderr,
            )

    elapsed = time.time() - start
    rate = totals["ok"] / max(1, done) * 100
    print(
        f"\n[summary] processed={done}  ok={totals['ok']} ({rate:.1f}%)  "
        f"fetch_err={totals['fetch_error']}  empty={totals['empty']}  "
        f"axe_drop={totals['axe_drop']}  in {elapsed:.1f}s",
        file=sys.stderr,
    )
    tmpdir.cleanup()


if __name__ == "__main__":
    main()
