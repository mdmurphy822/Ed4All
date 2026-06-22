"""Generate training pairs from Federal Register documents.

Federal Register publishes ~200 Rules, Proposed Rules, and Notices per
day, all with structured XML available via the public API. Content is
federal-government public domain — no license filter needed.

Pipeline:
  1. List documents via api.federalregister.gov/api/v1/documents.json,
     paginated.
  2. For each doc: fetch the XML, parse into our legacy IR, emit HTML.
  3. Run the HTML through axe; render to PDF; feature-extract; save pair.

Output matches the pair_from_* contract (input_ocr + output_html).

Usage:
    # Most-recent 50 docs (smoke test)
    python scripts/pair_from_federal_register.py --limit 50

    # Everything published since a date
    python scripts/pair_from_federal_register.py --since 2024-01-01 \\
        --out-dir data/pairs/federal_register --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import requests

from dart_semantic import emit_html, ir  # legacy IR + emitter for ground-truth HTML
from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.parse_federal_register import parse_fr_xml
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


API = "https://www.federalregister.gov/api/v1/documents.json"
USER_AGENT = "dart-semantic/0.0.1"

FR_CSS = """
body { font-family: Georgia, serif; max-width: 7in; margin: 1in auto; color: #111; line-height: 1.45; }
h1 { margin: 0 0 0.6em; }
h2, h3 { margin-top: 1em; }
p { text-align: justify; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #444; padding: 0.35em 0.7em; }
th { background: #eee; }
ul, ol { margin: 0.5em 0; padding-left: 1.5em; }
blockquote { margin: 1em 2em; color: #444; font-style: italic; }
a { padding: 2px 1px; }
pre { white-space: pre-wrap; word-break: break-word; }
"""


def wrap_html(raw_html: str) -> str:
    return raw_html.replace("<head>", f"<head><style>{FR_CSS}</style>", 1)


def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:maxlen] or "doc"


def list_documents(since: str | None,
                   until: str | None,
                   limit: int | None) -> list[dict]:
    """Page through the FR API and return document metadata."""
    out: list[dict] = []
    page = 1
    per_page = 100
    while True:
        params = {
            "per_page": per_page,
            "page": page,
            "order": "newest",
            "fields[]": [
                "document_number", "title", "type", "publication_date",
                "html_url", "full_text_xml_url", "pdf_url",
            ],
        }
        # Pre-2000 documents generally don't have XML in the archive. Default
        # to a safe floor unless the caller asked for something older.
        params["conditions[publication_date][gte]"] = since or "2015-01-01"
        if until:
            params["conditions[publication_date][lte]"] = until
        r = requests.get(API, params=params,
                         headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        out.extend(results)
        if limit and len(out) >= limit:
            return out[:limit]
        if page >= data.get("total_pages", 1):
            break
        page += 1
        # Polite pacing — FR API has no auth and generous rate limits.
        time.sleep(0.2)
    return out


def process_document(validator: HtmlValidator,
                     work: tuple[dict, str]) -> dict:
    """Worker: fetch XML + render + pair for one Federal Register document."""
    doc_meta, out_dir_str = work
    out_dir = Path(out_dir_str)
    number = doc_meta["document_number"]
    stats = {"number": number, "title": doc_meta.get("title", ""),
             "ok": 0, "fetch_error": 0, "parse_error": 0,
             "emitter_drop": 0, "axe_drop": 0}

    xml_url = doc_meta.get("full_text_xml_url")
    if not xml_url:
        stats["fetch_error"] = 1
        stats["msg"] = "no xml_url in metadata"
        return stats

    try:
        r = requests.get(xml_url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        xml_text = r.text
    except Exception as exc:
        stats["fetch_error"] = 1
        stats["msg"] = f"xml fetch: {exc}"
        return stats

    try:
        document = parse_fr_xml(
            xml_text,
            document_number=number,
            title=doc_meta.get("title", number),
            source_url=doc_meta.get("html_url"),
        )
    except (ir.IRError, Exception) as exc:
        stats["parse_error"] = 1
        stats["msg"] = f"parse: {exc}"
        return stats

    try:
        raw_html = emit_html.emit(document)
    except emit_html.EmitterError as exc:
        stats["emitter_drop"] = 1
        stats["msg"] = f"emit: {exc}"
        return stats
    html_doc = wrap_html(raw_html)

    result = validator.check(html_doc)
    if not result.ok:
        stats["axe_drop"] = 1
        stats["msg"] = "; ".join(result.reasons[:2])
        return stats

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / f"{number}.pdf"
        validator.render_pdf(html_doc, pdf_path)
        input_ocr = pdf_to_ocr_text(pdf_path)

    if not input_ocr.strip():
        stats["axe_drop"] = 1
        stats["msg"] = "empty ocr"
        return stats

    pair = {
        "source": "federal_register",
        "document_number": number,
        "title": doc_meta.get("title", ""),
        "type": doc_meta.get("type", ""),
        "publication_date": doc_meta.get("publication_date", ""),
        "url": doc_meta.get("html_url"),
        "variant_id": number,
        "input_ocr": input_ocr,
        "output_html": html_doc,
        # Original Federal Register XML (carries doc metadata, agency
        # tags, RIN/CFR references, footnote markup that emit_html
        # flattens). Phase 3c reads this for legal/boilerplate signal.
        "raw_source_xml": xml_text,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(number)}.json"
    path.write_text(json.dumps(pair, ensure_ascii=False))
    stats["ok"] = 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="Lower bound publication date (YYYY-MM-DD)")
    ap.add_argument("--until", help="Upper bound publication date (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=100,
                    help="Maximum number of documents to ingest")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/pairs/federal_register"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"[catalog] listing Federal Register documents...", file=sys.stderr)
    docs = list_documents(args.since, args.until, args.limit)
    print(f"[catalog] {len(docs)} documents queued", file=sys.stderr)

    # Skip already-processed.
    existing = ({p.stem for p in args.out_dir.glob("*.json")}
                if args.out_dir.exists() else set())
    work_items = [(d, str(args.out_dir)) for d in docs
                  if slugify(d["document_number"]) not in existing]
    print(f"[plan] {len(work_items)} to process "
          f"({len(existing)} already done)", file=sys.stderr)

    totals = {"ok": 0, "fetch_error": 0, "parse_error": 0,
              "emitter_drop": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_document, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        if done % 10 == 0 or any(stats.get(k) for k in
                                  ("fetch_error", "parse_error", "emitter_drop")):
            outcome = next((k for k in ("ok", "fetch_error", "parse_error",
                                        "emitter_drop", "axe_drop")
                            if stats.get(k)), "?")
            print(f"[{done}/{len(work_items)}] {stats['number']:18} "
                  f"{outcome:14} {stats.get('msg', '')[:50]}",
                  file=sys.stderr)

    elapsed = time.time() - start
    rate = totals["ok"] / max(1, done) * 100
    print(f"\n[summary] processed={done}  ok={totals['ok']} ({rate:.1f}%)  "
          f"fetch_err={totals['fetch_error']}  parse_err={totals['parse_error']}  "
          f"emit_drop={totals['emitter_drop']}  axe_drop={totals['axe_drop']}  "
          f"in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
