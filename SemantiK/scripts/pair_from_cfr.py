"""Generate training pairs from govinfo.gov CFR XML volumes.

The Code of Federal Regulations is published by the U.S. Government
Publishing Office at:

    https://www.govinfo.gov/bulkdata/CFR/<year>/title-<n>/CFR-<year>-title<n>-vol<v>.xml

Each volume contains many <PART> elements. Each PART becomes one
training pair (one PDF). This source is the single best yield for
the rare doc-role classes Phase 3c needs:

  legal     — front-matter notices, "Authority:" + "Source:" lines
  metadata  — Part / Section numbers, citation references, EARs
  footer    — footnotes, edition notes, agency boilerplate

License: 17 U.S.C. 105 — works of the U.S. Government are not
subject to copyright. Public domain. No per-doc filter required.

Usage:
    # Smoke (one title, max 5 parts)
    python scripts/pair_from_cfr.py --year 2024 --titles 1 --limit 5 \
        --out-dir data/pairs/cfr_smoke

    # Bulk
    python scripts/pair_from_cfr.py --year 2024 --titles 1,2,5,12 --limit 100 \
        --out-dir data/pairs/cfr --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from dart_semantic import emit_html, ir
from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.parse_cfr import parse_cfr_part
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


USER_AGENT = "dart-semantic/0.0.1"
BULK_BASE = "https://www.govinfo.gov/bulkdata/CFR"

CFR_CSS = """
body { font-family: 'Times New Roman', Times, serif; max-width: 7in;
       margin: 1in auto; color: #111; line-height: 1.5; }
h1 { font-size: 18pt; margin: 0 0 0.6em; text-align: center; }
h2 { font-size: 13pt; margin: 1.2em 0 0.4em; }
h3, h4 { margin-top: 1em; }
p { text-align: justify; margin: 0.4em 0; }
blockquote { margin: 0.8em 1.5em; color: #444; font-size: 10.5pt; }
ul, ol { margin: 0.4em 0; padding-left: 1.5em; }
"""


def wrap_html(raw_html: str) -> str:
    return raw_html.replace("<head>", f"<head><style>{CFR_CSS}</style>", 1)


def slugify(s: str, maxlen: int = 70) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:maxlen] or "doc"


# --------------------------------------------------------------------------
# Volume listing & download
# --------------------------------------------------------------------------

def list_volume_urls(year: int, titles: list[int]) -> list[tuple[int, str]]:
    """Discover CFR-<year>-title<n>-vol*.xml urls for each requested title.

    Returns [(title_num, url), ...]. Tries vol1..vol9 and stops at the
    first 404 per title.
    """
    out: list[tuple[int, str]] = []
    for t in titles:
        for vol in range(1, 10):
            url = (f"{BULK_BASE}/{year}/title-{t}/"
                   f"CFR-{year}-title{t}-vol{vol}.xml")
            try:
                r = requests.head(url, headers={"User-Agent": USER_AGENT},
                                  timeout=15, allow_redirects=True)
                if r.status_code == 200:
                    out.append((t, url))
                else:
                    break
            except Exception:
                break
            time.sleep(0.1)
    return out


def fetch_volume(url: str, cache_dir: Path) -> Path:
    """Download a CFR volume XML to cache and return the local path."""
    name = url.rsplit("/", 1)[-1]
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / name
    if local.exists() and local.stat().st_size > 0:
        return local
    print(f"[fetch] {url}", file=sys.stderr)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120,
                     stream=True)
    r.raise_for_status()
    tmp = local.with_suffix(local.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(64 * 1024):
            if chunk:
                f.write(chunk)
    tmp.rename(local)
    return local


# --------------------------------------------------------------------------
# Volume splitting — yield PART elements with metadata
# --------------------------------------------------------------------------

def iter_parts_in_volume(volume_path: Path,
                         title_num: int,
                         year: int) -> list[dict]:
    """Walk a CFR volume XML and yield one work-item dict per PART.

    Each item carries enough info for a worker to (a) re-parse just
    that PART without re-loading the whole volume and (b) write a
    self-contained pair JSON.
    """
    # Extract <PART> elements as serialized XML strings so we can
    # ship them through the worker pool by value (pickle-safe) and
    # also stash them as `raw_source_xml` on the pair.
    items: list[dict] = []
    try:
        tree = ET.parse(volume_path)
    except ET.ParseError as exc:
        print(f"[warn] {volume_path.name}: {exc}", file=sys.stderr)
        return items
    root = tree.getroot()

    # Find vol number from filename: ...vol3.xml
    vol_match = re.search(r"vol(\d+)\.xml$", volume_path.name)
    volume = vol_match.group(1) if vol_match else "1"

    # Pull out the volume-wide front matter (BTITLE / FMTR) once so it
    # can be attached to the first PART of each volume. Without this
    # the per-PART pairs would never carry legal-class signal — the
    # OENOTICE / SUDOCS / GPO blocks live only at the volume root.
    front_matter_xml = ""
    fmtr = root.find("FMTR")
    if fmtr is not None:
        # Strip the verbose TOC to keep size manageable.
        for toc in list(fmtr.findall("TOC")) + list(fmtr.findall("CFRTOC")):
            fmtr.remove(toc)
        front_matter_xml = ET.tostring(fmtr, encoding="unicode")

    parts = list(root.iter("PART"))
    for part_el in parts:
        part_xml = ET.tostring(part_el, encoding="unicode")
        # Attach the volume's front-matter (OENOTICE / GPO / SUDOCS) to
        # EVERY PART. The first iteration of this fetcher only attached
        # it to the first PART of each volume, but the resulting
        # 6-volumes × ~5 legal-text-paragraphs that survive alignment
        # was insufficient (~19 legal rows total). Repeating the front
        # matter across all PARTs is the boilerplate the model NEEDS to
        # learn — every CFR volume in the wild carries this preamble,
        # so duplicating it across our pairs reflects the actual
        # distribution.
        items.append({
            "title_num": str(title_num),
            "volume": volume,
            "year": str(year),
            "part_xml": part_xml,
            "front_matter_xml": front_matter_xml,
        })
    return items


# --------------------------------------------------------------------------
# Per-PART worker
# --------------------------------------------------------------------------

def process_part(validator: HtmlValidator, work: tuple) -> dict:
    item, out_dir_str = work
    out_dir = Path(out_dir_str)

    title_num = item["title_num"]
    volume = item["volume"]
    year = item["year"]
    part_xml = item["part_xml"]

    stats = {"title": title_num, "vol": volume,
             "ok": 0, "parse_error": 0,
             "emitter_drop": 0, "axe_drop": 0,
             "msg": "", "stem": ""}

    try:
        part_el = ET.fromstring(part_xml)
    except ET.ParseError as exc:
        stats["parse_error"] = 1
        stats["msg"] = f"xml parse: {exc}"
        return stats

    source_url = (f"{BULK_BASE}/{year}/title-{title_num}/"
                  f"CFR-{year}-title{title_num}-vol{volume}.xml")

    # If the volume front matter is attached to this work item, parse it
    # and pass into the parser so its OENOTICE / GPO / SUDOCS notices
    # ride along inside the rendered PDF.
    fm_el: ET.Element | None = None
    fm_xml = item.get("front_matter_xml") or ""
    if fm_xml:
        try:
            fm_el = ET.fromstring(fm_xml)
        except ET.ParseError:
            fm_el = None

    try:
        document = parse_cfr_part(
            part_el,
            title_num=title_num,
            volume=volume,
            year=year,
            source_url=source_url,
            front_matter_el=fm_el,
        )
    except ir.IRError as exc:
        stats["parse_error"] = 1
        stats["msg"] = f"parse: {exc}"
        return stats
    except Exception as exc:
        stats["parse_error"] = 1
        stats["msg"] = f"parse-exc: {exc}"
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
        pdf_path = Path(tmp) / "doc.pdf"
        validator.render_pdf(html_doc, pdf_path)
        input_ocr = pdf_to_ocr_text(pdf_path)

    if not input_ocr.strip():
        stats["axe_drop"] = 1
        stats["msg"] = "empty ocr"
        return stats

    stem = slugify(f"title{title_num}_vol{volume}_{document.source_id}",
                   maxlen=80)
    pair = {
        "source": "cfr",
        "title_num": title_num,
        "volume": volume,
        "year": year,
        "part_title": document.title,
        "url": source_url,
        "variant_id": stem,
        "input_ocr": input_ocr,
        "output_html": html_doc,
        # The PART XML — extractor walks this for doc_role labels.
        # When the fetcher attaches front-matter (first PART of each
        # volume), wrap both inside a <CFRDOC> root so a single XML
        # parse from the extractor sees both subtrees.
        "raw_source_xml": (
            f"<CFRDOC>{item.get('front_matter_xml','')}{part_xml}</CFRDOC>"
            if item.get("front_matter_xml") else part_xml
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(pair, ensure_ascii=False))
    stats["ok"] = 1
    stats["stem"] = stem
    return stats


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def parse_titles(s: str) -> list[int]:
    out: list[int] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--titles", default="1,2,5,12,17,40",
                    help="Comma-separated CFR title numbers (or ranges 1-3)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on number of PARTs to process")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/pairs/cfr"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("data/cache/cfr_xml"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    titles = parse_titles(args.titles)
    print(f"[catalog] discovering volumes for titles {titles} year {args.year}",
          file=sys.stderr)
    vol_urls = list_volume_urls(args.year, titles)
    print(f"[catalog] {len(vol_urls)} volumes found", file=sys.stderr)

    items: list[dict] = []
    for title_num, url in vol_urls:
        try:
            local = fetch_volume(url, args.cache_dir)
        except Exception as exc:
            print(f"[warn] fetch fail {url}: {exc}", file=sys.stderr)
            continue
        items.extend(iter_parts_in_volume(local, title_num, args.year))
        if args.limit and len(items) >= args.limit:
            break
    if args.limit:
        items = items[:args.limit]
    print(f"[plan] {len(items)} PARTs queued", file=sys.stderr)

    existing = ({p.stem for p in args.out_dir.glob("*.json")}
                if args.out_dir.exists() else set())
    print(f"[plan] {len(existing)} already done in {args.out_dir}",
          file=sys.stderr)

    work_items = [(it, str(args.out_dir)) for it in items]

    totals = {"ok": 0, "parse_error": 0, "emitter_drop": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_part, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        if done % 10 == 0 or any(stats.get(k) for k in
                                  ("parse_error", "emitter_drop", "axe_drop")):
            outcome = next((k for k in ("ok", "parse_error", "emitter_drop",
                                        "axe_drop")
                            if stats.get(k)), "?")
            print(f"[{done}/{len(work_items)}] T{stats['title']:3} "
                  f"v{stats['vol']:1} {outcome:14} {stats.get('msg','')[:50]}",
                  file=sys.stderr)

    elapsed = time.time() - start
    rate = totals["ok"] / max(1, done) * 100
    print(f"\n[summary] processed={done}  ok={totals['ok']} ({rate:.1f}%)  "
          f"parse_err={totals['parse_error']}  "
          f"emit_drop={totals['emitter_drop']}  "
          f"axe_drop={totals['axe_drop']}  "
          f"in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
