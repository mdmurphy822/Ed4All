"""Generate (features → StructuredBlock[] JSON) training pairs from Wikipedia.

Flow:
    article title
        -> fetch REST HTML
        -> parse_wikipedia.parse_article -> list[ir.Document] (one per h2 section)
        -> for each doc:
             emit_html(doc)  ->  axe gate
                          |        |
                          |        v
                          |    drop on fail
                          v
             render PDF -> pdf_to_ocr_text -> input_ocr
             emit_blocks(doc) -> StructuredBlock JSON -> output_json
             write pair JSON

Usage:
    python scripts/pair_from_wikipedia.py "Accessibility" \\
        --out-dir data/pairs/wikipedia
    python scripts/pair_from_wikipedia.py --titles data/seeds/seed_titles.txt --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from dart_semantic import emit_html, ir  # legacy IR + emitter used for ground-truth HTML
from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.parse_wikipedia import fetch_article_html, parse_article
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


def _section_has_multi_header_table(doc: ir.Document) -> bool:
    """True if any ir.Table in the section has a multi-row header,
    a colspan>=2 header cell, or a rowspan>=2 header cell.

    Used by --require-multi-header-table to gate wiki harvesting toward
    table_specialist's weak spot (header_row precision/recall on
    multi-level header tables — see eval_table_specialist_per_source)."""
    for block in doc.blocks:
        if not isinstance(block, ir.Table):
            continue
        if len(block.header_rows) > 1:
            return True
        for row in block.header_rows:
            for cell in row:
                if cell.colspan > 1 or cell.rowspan > 1:
                    return True
    return False

WIKI_CSS = """
body { font-family: Georgia, serif; max-width: 7in; margin: 1in auto; color: #111; line-height: 1.45; }
h1 { margin: 0 0 0.6em; }
h2 { margin-top: 1.2em; }
h3 { margin-top: 1em; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #444; padding: 0.35em 0.7em; text-align: left; vertical-align: top; }
th { background: #eee; }
ul, ol { margin: 0.6em 0; padding-left: 1.6em; }
figure { margin: 1em 0; }
figcaption { font-style: italic; font-size: 0.9em; color: #444; }
pre, code { font-family: "DejaVu Sans Mono", monospace; }
/* Wrap long lines so <pre> never becomes a scrollable region (axe flags
   those without tabindex=0). Wrapping is the accessible default. */
pre { background: #f7f7f7; padding: 0.8em; white-space: pre-wrap; word-break: break-word; }
/* Give inline links a bit of padding so a dense paragraph doesn't trip
   SC 2.5.8 target-size. This doesn't change visual layout noticeably. */
a { padding: 2px 1px; }
"""


def wrap_html(raw_html: str) -> str:
    return raw_html.replace("<head>", f"<head><style>{WIKI_CSS}</style>", 1)


def slugify(s: str, maxlen: int = 50) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in s.lower()).strip("_")
    return cleaned[:maxlen] or "untitled"


def process_article(
    validator: HtmlValidator, work: tuple[str, str, bool, float],
) -> dict:
    """Worker function: process one article end-to-end.

    work = (title, out_dir_str, require_multi_header, throttle_secs).
    Returns a stats dict with counts; the caller aggregates across all
    workers.
    """
    title, out_dir_str, require_multi_header, throttle_secs = work
    out_dir = Path(out_dir_str)
    stats = {"attempted": 0, "ok": 0, "emitter_drop": 0, "axe_drop": 0,
             "fetch_error": 0, "no_multi_header": 0}

    if throttle_secs > 0:
        time.sleep(throttle_secs)

    try:
        raw = fetch_article_html(title)
    except Exception as exc:
        stats["fetch_error"] += 1
        stats["title"] = title
        stats["msg"] = f"fetch failed: {exc}"
        return stats

    docs = parse_article(raw, title,
                         article_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
    article_slug = slugify(title)

    for idx, doc in enumerate(docs):
        stats["attempted"] += 1
        section_slug = slugify(doc.title.split("—", 1)[-1].strip())

        if require_multi_header and not _section_has_multi_header_table(doc):
            stats["no_multi_header"] += 1
            continue

        try:
            raw_html = emit_html.emit(doc)
        except emit_html.EmitterError:
            stats["emitter_drop"] += 1
            continue
        html_doc = wrap_html(raw_html)

        result = validator.check(html_doc)
        if not result.ok:
            stats["axe_drop"] += 1
            continue

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "section.pdf"
            validator.render_pdf(html_doc, pdf_path)
            input_ocr = pdf_to_ocr_text(pdf_path)

        # build_classifier_data.py reads output_html for label extraction;
        # the old output_json field is no longer needed.
        pair = {
            "source": "wikipedia",
            "variant_id": f"{article_slug}__{idx:02d}",
            "article_title": title,
            "section_title": doc.title,
            "url": doc.source_url,
            "input_ocr": input_ocr,
            "output_html": html_doc,
            # raw Parsoid HTML for the whole article (same string for
            # every section pair from this article — Phase 3c reader
            # needs the rich semantic markup emit_html.emit() strips).
            "raw_source_html": raw,
        }
        path = out_dir / f"{article_slug}__{idx:02d}_{section_slug}.json"
        path.write_text(json.dumps(pair, ensure_ascii=False))
        stats["ok"] += 1

    stats["title"] = title
    stats["sections"] = len(docs)
    return stats


def read_titles(path: str) -> list[str]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("title", nargs="?", help="Single Wikipedia title")
    grp.add_argument("--titles", help="Path to file with one title per line, or '-' for stdin")
    ap.add_argument("--out-dir", type=Path, default=Path("data/pairs/wikipedia"))
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (each holds its own Chromium ~400MB)")
    ap.add_argument("--require-multi-header-table", action="store_true",
                    help="Skip sections that don't contain at least one "
                         "table with a multi-row header (len>1) or "
                         "colspan/rowspan>=2 header cell. Targets "
                         "table_specialist's header_row weakness.")
    ap.add_argument("--throttle-secs", type=float, default=0.0,
                    help="Sleep between articles (per worker) to stay "
                         "under MediaWiki REST rate limits. Use ~2s for "
                         "single-worker runs across long seed lists.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip titles whose article slug already has at "
                         "least one pair file in --out-dir.")
    args = ap.parse_args()

    titles = [args.title] if args.title else read_titles(args.titles)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        existing_slugs = {
            p.name.split("__", 1)[0]
            for p in args.out_dir.glob("*.json")
        }
        before = len(titles)
        titles = [t for t in titles if slugify(t) not in existing_slugs]
        print(f"[skip-existing] kept {len(titles)}/{before} titles "
              f"({before - len(titles)} already have pairs)",
              file=sys.stderr)

    totals = {"attempted": 0, "ok": 0, "emitter_drop": 0, "axe_drop": 0,
              "fetch_error": 0, "no_multi_header": 0}
    start = time.time()

    work_items = [
        (t, str(args.out_dir), args.require_multi_header_table,
         args.throttle_secs)
        for t in titles
    ]
    done = 0
    for stats in run_in_pool(process_article, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        title = stats.get("title", "?")
        if stats.get("fetch_error"):
            print(f"[{done}/{len(titles)}] FETCH-FAIL {title}: {stats.get('msg', '')}",
                  file=sys.stderr)
        else:
            print(f"[{done}/{len(titles)}] {title[:50]:50} "
                  f"sections={stats.get('sections', 0):>3} "
                  f"ok={stats['ok']:>3} "
                  f"emitter-drop={stats['emitter_drop']:>2} "
                  f"axe-drop={stats['axe_drop']:>2} "
                  f"no-multi-hdr={stats.get('no_multi_header', 0):>2}",
                  file=sys.stderr)

    elapsed = time.time() - start
    rate = totals["ok"] / totals["attempted"] * 100 if totals["attempted"] else 0
    print(f"\n[summary] articles={len(titles)}  attempted={totals['attempted']}  "
          f"ok={totals['ok']} ({rate:.1f}%)  emitter_drops={totals['emitter_drop']}  "
          f"axe_drops={totals['axe_drop']}  fetch_errors={totals['fetch_error']}  "
          f"in {elapsed:.1f}s  workers={args.workers}")


if __name__ == "__main__":
    main()
