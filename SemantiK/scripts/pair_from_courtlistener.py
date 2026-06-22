"""Generate training pairs from CourtListener federal opinions.

License: U.S. judicial opinions are public domain (Banks v. Manchester,
128 U.S. 244 (1888)). The Free Law Project re-distributes opinion text
and source PDFs as public-domain works. This fetcher restricts the pull
to ``court_jurisdiction=F`` (federal courts) at fetch time. State
courts are deferred until per-state license review is done.

Anonymous-access constraints (verified 2026-05-05):

  * The CourtListener ``v4`` Search API is open without a token and
    returns paginated opinion metadata (``court_jurisdiction``,
    ``caseName``, ``judge``, ``opinions[].download_url``,
    ``opinions[].local_path``, ``opinions[].snippet`` capped at 500
    chars).
  * The ``v4`` ``opinions/<id>/`` and ``clusters/<id>/`` detail
    endpoints require a token (HTTP 401 anon).
  * The HTML opinion page on courtlistener.com is behind CloudFront WAF
    and returns 202 with empty body to programmatic UAs.
  * The ``storage.courtlistener.com`` static origin serves opinion PDFs
    (and Free Law's mirrored court source PDFs) publicly. We pull the
    PDF, extract its text via pdftotext (CPU, no GPU), synthesize a
    minimal accessible HTML doc from the case metadata + extracted
    text, and route it through the standard validate -> render -> OCR
    loop. The CourtListener extractor (extract_courtlistener_blocks)
    accepts both rich-HTML and plain-text-wrapped-in-<pre> modes; we
    use a structured HTML form (case-name + paragraphs) that the
    extractor walks for title / author / body roles.

Output schema matches data.build_semantic_data.extract_courtlistener_blocks:
  raw_source_html (the structured HTML the extractor walks)
  output_html  (the same HTML, CSS-wrapped, used to render the PDF)
  input_ocr    (pdf_to_ocr_text on the rendered PDF)
  variant_id   (cluster_id-opinion_id pair)
  source       "courtlistener"

Usage:
    # Smoke (5 federal opinions)
    python scripts/pair_from_courtlistener.py --limit 5 \\
        --out-dir data/pairs/courtlistener_smoke

    # Bulk
    python scripts/pair_from_courtlistener.py --limit 1000 \\
        --out-dir data/pairs/courtlistener --workers 2
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


USER_AGENT = "dart-semantic/0.0.1"
SEARCH_API = "https://www.courtlistener.com/api/rest/v4/search/"
STORAGE_BASE = "https://storage.courtlistener.com"

CL_CSS = """
body { font-family: 'Times New Roman', Times, serif; max-width: 6.5in;
       margin: 1in auto; color: #111; line-height: 1.5; }
h1 { font-size: 16pt; margin: 0 0 0.6em; text-align: center; }
.case_name { font-size: 14pt; font-weight: bold; text-align: center;
             margin: 0.6em 0 0.2em; }
.byline, .author { font-size: 11pt; font-style: italic;
                   text-align: center; margin: 0.4em 0; }
.citation { font-size: 10.5pt; color: #444; text-align: center;
            margin: 0.2em 0 0.6em; }
p { text-align: justify; margin: 0.4em 0; }
.footnote { font-size: 9.5pt; border-top: 1px solid #999;
            margin-top: 1em; padding-top: 0.3em; color: #333; }
.opinion_caption { text-align: center; font-weight: bold; }
"""


def slugify(s: str, maxlen: int = 70) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:maxlen] or "doc"


def wrap_html(body_html: str, title: str) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{(title or "Federal Opinion").strip()[:200]}</title>'
        f'<style>{CL_CSS}</style></head><body>{body_html}</body></html>'
    )


# --------------------------------------------------------------------------
# Search API — discover federal opinions
# --------------------------------------------------------------------------

def _http_get_json(url: str, params: dict | None = None,
                   max_retries: int = 4) -> dict:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params,
                             headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"http {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            print(f"[retry] {url}: {exc} (sleep {delay:.1f}s)",
                  file=sys.stderr)
            time.sleep(delay)
            delay *= 2.0
    return {}


def list_federal_opinions(target: int) -> list[dict]:
    """Return up to `target` opinion records (federal jurisdiction).

    One record carries enough info for the worker to fetch a PDF without
    further API calls (cluster_id, opinion_id, caseName, judge, court,
    dateFiled, download_url, local_path).
    """
    out: list[dict] = []
    next_url: str | None = None
    page = 0
    while len(out) < target:
        page += 1
        try:
            if next_url:
                # Use the absolute next URL verbatim — the v4 cursor is
                # already URL-encoded inside that string and re-passing
                # it through requests' params kwarg double-encodes it.
                data = _http_get_json(next_url, params=None)
            else:
                data = _http_get_json(SEARCH_API, params={
                    "type": "o",
                    "page_size": 100,
                    "court_jurisdiction": "F",
                    "order_by": "dateFiled desc",
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] search page {page}: {exc}", file=sys.stderr)
            break

        results = data.get("results", []) or []
        if not results:
            break
        for r in results:
            if r.get("court_jurisdiction") != "F":
                continue  # belt-and-suspenders
            for op in r.get("opinions", []) or []:
                local_path = op.get("local_path") or ""
                download_url = op.get("download_url") or ""
                if not (local_path or download_url):
                    continue
                out.append({
                    "cluster_id": r.get("cluster_id"),
                    "opinion_id": op.get("id"),
                    "case_name": r.get("caseName") or "",
                    "court": r.get("court") or "",
                    "court_id": r.get("court_id") or "",
                    "judge": r.get("judge") or "",
                    "date_filed": r.get("dateFiled") or "",
                    "docket_number": r.get("docketNumber") or "",
                    "citations": r.get("citation") or [],
                    "absolute_url": r.get("absolute_url") or "",
                    "local_path": local_path,
                    "download_url": download_url,
                    "snippet": op.get("snippet") or "",
                    "type": op.get("type") or "",
                    "per_curiam": op.get("per_curiam", False),
                })
                if len(out) >= target:
                    break
            if len(out) >= target:
                break

        next_url = data.get("next") or None
        if not next_url:
            break
        time.sleep(0.6)  # ~1.5 req/s — polite to anon search

    return out


# --------------------------------------------------------------------------
# PDF fetch + text extraction
# --------------------------------------------------------------------------

def fetch_opinion_pdf(meta: dict, cache_dir: Path) -> Path | None:
    """Try storage.courtlistener.com (Free Law mirror) first. If absent,
    fall back to the upstream court ``download_url``. Returns None on
    any non-PDF response."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    op_id = meta.get("opinion_id")
    if not op_id:
        return None
    local = cache_dir / f"opinion_{op_id}.pdf"
    if local.exists() and local.stat().st_size > 1024:
        return local

    candidates: list[str] = []
    local_path = (meta.get("local_path") or "").lstrip("/")
    if local_path.lower().endswith(".pdf"):
        candidates.append(f"{STORAGE_BASE}/{local_path}")
    download_url = meta.get("download_url") or ""
    if download_url.lower().endswith(".pdf"):
        candidates.append(download_url)

    for url in candidates:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT},
                             timeout=120, stream=True)
            if r.status_code != 200:
                continue
            ctype = r.headers.get("Content-Type", "").lower()
            if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                continue
            tmp = local.with_suffix(local.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
            if tmp.stat().st_size <= 1024:
                tmp.unlink(missing_ok=True)
                continue
            tmp.rename(local)
            time.sleep(0.4)  # polite spacing on storage origin
            return local
        except Exception:
            continue
    return None


def pdf_to_text(pdf_path: Path) -> str:
    """Extract opinion text via pdftotext (layout mode disabled — we
    want flowing prose so the synthesized HTML's <p> blocks line up
    with paragraph boundaries)."""
    try:
        cp = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", "-nopgbrk", str(pdf_path), "-"],
            capture_output=True, timeout=60, check=False,
        )
        if cp.returncode != 0:
            return ""
        return cp.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Text -> structured HTML synthesis
# --------------------------------------------------------------------------

_FOOTNOTE_LINE_RE = re.compile(r"^\s*(\d{1,3}|\*+|†+)\s+\S+")


def text_to_paragraphs(text: str) -> list[str]:
    """Split pdftotext output into paragraphs by blank-line gaps.

    Drops obvious page-number lines and very short fragments."""
    paras: list[str] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        joined = " ".join(s.strip() for s in buf if s.strip())
        joined = re.sub(r"\s+", " ", joined).strip()
        if len(joined) >= 20:
            paras.append(joined)
        buf.clear()

    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
            continue
        # Drop bare page numbers
        if re.fullmatch(r"\s*\d{1,4}\s*", s):
            continue
        buf.append(s)
    flush()
    return paras


def build_html(meta: dict, paragraphs: list[str]) -> tuple[str, str]:
    title = meta.get("case_name") or "Federal Opinion"
    parts: list[str] = []
    parts.append(f'<h1 class="opinion_caption">{title}</h1>')
    parts.append(f'<p class="case_name">{title}</p>')

    cite_bits: list[str] = []
    for c in meta.get("citations") or []:
        if isinstance(c, str) and c.strip():
            cite_bits.append(c.strip())
    if meta.get("docket_number"):
        cite_bits.append(f"No. {meta['docket_number']}")
    if meta.get("court"):
        cite_bits.append(meta["court"])
    if meta.get("date_filed"):
        cite_bits.append(meta["date_filed"])
    if cite_bits:
        parts.append(f'<p class="citation">{"; ".join(cite_bits)}</p>')

    judge = (meta.get("judge") or "").strip()
    if judge:
        parts.append(f'<p class="byline">{judge}</p>')

    # Heuristic footnote split: paragraphs at the tail that start with a
    # small integer + space go into a <div class="footnote"> block. This
    # gives the extractor a footer-class signal it can pick up.
    body_paras = paragraphs[:]
    footnote_paras: list[str] = []
    while body_paras and _FOOTNOTE_LINE_RE.match(body_paras[-1] or ""):
        footnote_paras.insert(0, body_paras.pop())
        if len(footnote_paras) > 10:  # safety cap
            break

    for p in body_paras:
        parts.append(f"<p>{p}</p>")
    for fn in footnote_paras:
        parts.append(f'<p class="footnote">{fn}</p>')

    body_html = "\n".join(parts)
    return wrap_html(body_html, title), title


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def process_opinion(validator: HtmlValidator, work: tuple) -> dict:
    meta, out_dir_str, cache_dir_str = work
    out_dir = Path(out_dir_str)
    cache_dir = Path(cache_dir_str)
    op_id = meta.get("opinion_id")
    stats = {
        "op_id": op_id, "court": meta.get("court_id", ""),
        "ok": 0, "fetch_error": 0, "parse_error": 0,
        "emit_drop": 0, "axe_drop": 0, "msg": "",
    }

    pdf_path = fetch_opinion_pdf(meta, cache_dir)
    if pdf_path is None:
        stats["fetch_error"] = 1
        stats["msg"] = "no pdf"
        return stats

    text = pdf_to_text(pdf_path)
    if not text or len(text.strip()) < 200:
        stats["parse_error"] = 1
        stats["msg"] = "pdftotext empty/short"
        return stats

    paragraphs = text_to_paragraphs(text)
    if len(paragraphs) < 2:
        stats["parse_error"] = 1
        stats["msg"] = "no paragraphs"
        return stats

    try:
        html_doc, title = build_html(meta, paragraphs)
    except Exception as exc:
        stats["emit_drop"] = 1
        stats["msg"] = f"emit: {exc}"
        return stats

    result = validator.check(html_doc)
    if not result.ok:
        stats["axe_drop"] = 1
        stats["msg"] = "; ".join(result.reasons[:2])
        return stats

    with tempfile.TemporaryDirectory() as tmp:
        rendered_pdf = Path(tmp) / f"opinion_{op_id}.pdf"
        validator.render_pdf(html_doc, rendered_pdf)
        input_ocr = pdf_to_ocr_text(rendered_pdf)

    if not input_ocr.strip():
        stats["axe_drop"] = 1
        stats["msg"] = "empty ocr"
        return stats

    variant = f"cl_{meta.get('cluster_id')}_{op_id}"
    pair = {
        "source": "courtlistener",
        "cluster_id": meta.get("cluster_id"),
        "opinion_id": op_id,
        "case_name": meta.get("case_name"),
        "court": meta.get("court"),
        "court_id": meta.get("court_id"),
        "court_jurisdiction": "F",
        "date_filed": meta.get("date_filed"),
        "docket_number": meta.get("docket_number"),
        "judge": meta.get("judge"),
        "citations": meta.get("citations"),
        "url": ("https://www.courtlistener.com" + meta.get("absolute_url", "")
                if meta.get("absolute_url") else None),
        "variant_id": variant,
        "input_ocr": input_ocr,
        "output_html": html_doc,
        # The extractor reads raw_source_html. We feed it the same
        # structured HTML doc (its plaintext-mode and html-mode paths
        # both run cleanly on this — see extract_courtlistener_blocks).
        "raw_source_html": html_doc,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(variant)}.json"
    path.write_text(json.dumps(pair, ensure_ascii=False))
    stats["ok"] = 1
    return stats


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000,
                    help="Number of federal opinions to keep.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/pairs/courtlistener"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("data/cache/courtlistener"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--oversample", type=float, default=1.4,
                    help="Multiplier for catalog size (some opinions "
                         "won't have a PDF on storage).")
    args = ap.parse_args()

    target = max(1, int(args.limit * args.oversample))
    print(f"[catalog] requesting {target} federal opinions "
          f"(target_kept={args.limit})", file=sys.stderr)
    metas = list_federal_opinions(target)
    print(f"[catalog] {len(metas)} opinion records returned",
          file=sys.stderr)

    existing = ({p.stem for p in args.out_dir.glob("*.json")}
                if args.out_dir.exists() else set())
    work_items: list[tuple] = []
    for m in metas:
        variant = f"cl_{m.get('cluster_id')}_{m.get('opinion_id')}"
        if slugify(variant) in existing:
            continue
        work_items.append((m, str(args.out_dir), str(args.cache_dir)))
        if len(work_items) >= args.limit:
            break
    print(f"[plan] {len(work_items)} opinions to process "
          f"({len(existing)} already done)", file=sys.stderr)

    totals = {"ok": 0, "fetch_error": 0, "parse_error": 0,
              "emit_drop": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_opinion, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        if done % 20 == 0 or any(stats.get(k) for k in
                                  ("fetch_error", "parse_error",
                                   "emit_drop", "axe_drop")):
            outcome = next((k for k in ("ok", "fetch_error", "parse_error",
                                        "emit_drop", "axe_drop")
                            if stats.get(k)), "?")
            print(f"[{done}/{len(work_items)}] op_id={stats['op_id']} "
                  f"court={stats['court']:<10} {outcome:14} "
                  f"{stats.get('msg', '')[:55]}", file=sys.stderr)

    elapsed = time.time() - start
    rate = totals["ok"] / max(1, done) * 100
    print(f"\n[summary] processed={done}  ok={totals['ok']} ({rate:.1f}%)  "
          f"fetch_err={totals['fetch_error']}  "
          f"parse_err={totals['parse_error']}  "
          f"emit_drop={totals['emit_drop']}  "
          f"axe_drop={totals['axe_drop']}  in {elapsed:.1f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
