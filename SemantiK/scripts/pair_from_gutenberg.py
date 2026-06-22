"""Generate training pairs from Project Gutenberg books.

Each Gutenberg book is chapter-split, producing one (input_ocr, output_html)
pair per chapter. Pre-1928 US works are public domain; post-1928 works on
Gutenberg are all under licenses that permit redistribution + derivatives.

Usage:
    python scripts/pair_from_gutenberg.py --ids 1342 84 2701 --workers 4
    python scripts/pair_from_gutenberg.py --ids-file scripts/gutenberg_ids.txt
    python scripts/pair_from_gutenberg.py --catalog 50  # random sample from catalog
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import requests

from dart_semantic import emit_html, ir
from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.parse_gutenberg import parse_gutenberg
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


USER_AGENT = "dart-semantic/0.0.1"

GUTENBERG_CSS = """
body { font-family: Georgia, serif; max-width: 7in; margin: 1in auto; color: #111; line-height: 1.55; }
h1, h2 { text-align: center; margin-top: 2em; }
p { text-indent: 1.5em; margin: 0.4em 0; }
blockquote { margin: 1em 2em; font-style: italic; color: #444; }
a { padding: 2px 1px; }
"""

HTML_URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}-images.html",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.html",
    "https://www.gutenberg.org/files/{id}/{id}-h/{id}-h.htm",
]


def wrap_html(raw_html: str) -> str:
    return raw_html.replace("<head>", f"<head><style>{GUTENBERG_CSS}</style>", 1)


def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s.lower()).strip("_")
    return s[:maxlen] or "book"


def fetch_book_html(book_id: int) -> tuple[str, str]:
    """Return (html, canonical_url). Tries the standard URL templates in order."""
    last_exc = None
    for tmpl in HTML_URL_TEMPLATES:
        url = tmpl.format(id=book_id)
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.ok and len(r.text) > 5000:
                return r.text, url
            last_exc = RuntimeError(f"status {r.status_code} / len {len(r.text)}")
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"no HTML URL responded for book {book_id}: {last_exc}")


def process_book(validator: HtmlValidator,
                 work: tuple[int, str]) -> dict:
    """Worker: fetch + chapter-split + emit pairs for one Gutenberg book."""
    book_id, out_dir_str = work
    out_dir = Path(out_dir_str)
    stats = {"book_id": book_id, "ok": 0, "chapters": 0,
             "fetch_error": 0, "parse_error": 0,
             "emitter_drop": 0, "axe_drop": 0}

    try:
        html, url = fetch_book_html(book_id)
    except Exception as exc:
        stats["fetch_error"] = 1
        stats["msg"] = str(exc)
        return stats

    try:
        chapters = parse_gutenberg(html, book_id=book_id, source_url=url)
    except Exception as exc:
        stats["parse_error"] = 1
        stats["msg"] = str(exc)
        return stats
    stats["chapters"] = len(chapters)

    for idx, chapter_doc in enumerate(chapters):
        try:
            raw = emit_html.emit(chapter_doc)
        except emit_html.EmitterError:
            stats["emitter_drop"] += 1
            continue
        html_doc = wrap_html(raw)

        result = validator.check(html_doc)
        if not result.ok:
            stats["axe_drop"] += 1
            continue

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "chapter.pdf"
            validator.render_pdf(html_doc, pdf_path)
            input_ocr = pdf_to_ocr_text(pdf_path)
        if not input_ocr.strip():
            stats["axe_drop"] += 1
            continue

        chapter_slug = slugify(chapter_doc.title.split("—", 1)[-1].strip())
        pair = {
            "source": "gutenberg",
            "book_id": book_id,
            "chapter_idx": idx,
            "title": chapter_doc.title,
            "url": chapter_doc.source_url,
            "variant_id": f"gutenberg_{book_id}_{idx:03d}",
            "input_ocr": input_ocr,
            "output_html": html_doc,
            # Whole-book raw Gutenberg HTML — same string for every
            # chapter from this book. Phase 3c reads it for richer
            # semantic markup that emit_html.emit() strips.
            "raw_source_html": html,
        }
        fname = f"gutenberg_{book_id}__{idx:03d}_{chapter_slug}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / fname).write_text(json.dumps(pair, ensure_ascii=False))
        stats["ok"] += 1

    return stats


# ---------- book id sources ----------

def load_ids_from_file(path: Path) -> list[int]:
    ids: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            ids.append(int(line))
        except ValueError:
            continue
    return ids


# A curated starter set — classics with clean HTML, varied prose styles,
# multiple genres. All pre-1928 (public domain in US). The user can
# extend with --ids-file for larger runs.
CURATED_IDS = [
    1342,   # Pride and Prejudice — Austen
    84,     # Frankenstein — Shelley
    2701,   # Moby Dick — Melville
    11,     # Alice in Wonderland — Carroll
    98,     # A Tale of Two Cities — Dickens
    1661,   # Sherlock Holmes — Doyle
    345,    # Dracula — Stoker
    74,     # The Adventures of Tom Sawyer — Twain
    158,    # Emma — Austen
    2542,   # A Doll's House — Ibsen
    766,    # David Copperfield — Dickens
    16328,  # Beowulf
    1184,   # The Count of Monte Cristo — Dumas
    174,    # The Picture of Dorian Gray — Wilde
    236,    # Jungle Book — Kipling
    43,     # Dr. Jekyll and Mr. Hyde — Stevenson
    219,    # Heart of Darkness — Conrad
    215,    # The Call of the Wild — London
    2591,   # Grimm's Fairy Tales
    145,    # Middlemarch — Eliot
]


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", type=int, nargs="+", help="Specific Gutenberg book IDs")
    g.add_argument("--ids-file", type=Path,
                   help="Path to file with one book id per line")
    g.add_argument("--curated", action="store_true",
                   help=f"Process the {len(CURATED_IDS)}-book curated classics list")
    ap.add_argument("--out-dir", type=Path, default=Path("data/pairs/gutenberg"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of books processed")
    args = ap.parse_args()

    if args.ids:
        ids = args.ids
    elif args.ids_file:
        ids = load_ids_from_file(args.ids_file)
    else:
        ids = CURATED_IDS
    if args.limit:
        ids = ids[:args.limit]

    # Skip already-processed books.
    existing = (
        {int(re.match(r"gutenberg_(\d+)__", p.stem).group(1))
         for p in args.out_dir.glob("gutenberg_*.json")
         if re.match(r"gutenberg_(\d+)__", p.stem)}
        if args.out_dir.exists() else set()
    )
    ids = [i for i in ids if i not in existing]
    print(f"[plan] {len(ids)} books ({len(existing)} already done)", file=sys.stderr)

    work_items = [(i, str(args.out_dir)) for i in ids]
    totals = {"ok": 0, "fetch_error": 0, "parse_error": 0,
              "emitter_drop": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_book, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        outcome = (f"ok ({stats.get('ok')} ch of {stats.get('chapters')})"
                   if stats.get("ok") else next(
                       (k for k in ("fetch_error", "parse_error",
                                    "emitter_drop", "axe_drop")
                        if stats.get(k)), "?"))
        print(f"[{done}/{len(ids)}] book {stats['book_id']:6}  "
              f"{outcome:25} {stats.get('msg', '')[:50]}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\n[summary] books={done}  chapter_pairs={totals['ok']}  "
          f"fetch_err={totals['fetch_error']}  parse_err={totals['parse_error']}  "
          f"emit_drop={totals['emitter_drop']}  axe_drop={totals['axe_drop']}  "
          f"in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
