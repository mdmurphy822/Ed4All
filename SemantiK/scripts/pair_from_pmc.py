"""Generate training pairs from PubMed Central OA Commercial subset.

NCBI's PMC OA "Commercial Use" bulk subset ships JATS-XML article
files licensed CC-BY / CC-BY-SA / CC0 only. The bulk listing lives at:

    https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_comm/xml/
        oa_comm_xml.PMC<bucket>xxxxxx.baseline.<yyyy-mm-dd>.filelist.csv
        oa_comm_xml.PMC<bucket>xxxxxx.baseline.<yyyy-mm-dd>.tar.gz

The CSV carries a per-article `License` column. Even within the OA
Commercial bucket some articles carry stricter terms — the per-article
filter below drops anything not in {``CC BY``, ``CC BY-SA``, ``CC0``,
``Public Domain``}. Equivalent CC URLs (creativecommons.org/licenses/by/,
/by-sa/, /publicdomain/zero/) inside the JATS ``<license>`` tag are also
checked as a second-pass safety belt.

Pipeline:
  1. Download the bucket filelist CSV.
  2. License-filter in the CSV (and dedupe against `existing` outputs).
  3. Pull the matching tarball, stream `tarfile` extraction filtered to
     the kept PMCIDs (no full extract — saves ~5x disk).
  4. Per article: re-verify license inside the JATS XML, render an
     HTML doc from JATS, axe-check, render to PDF, OCR, save pair.

License: PMC OA Commercial subset = CC-BY / CC-BY-SA / CC0 (commercial
reuse permitted). State validated at fetch time.

Usage:
    # Smoke (5 articles from the smallest bucket)
    python scripts/pair_from_pmc.py --bucket PMC000xxxxxx --limit 5 \\
        --out-dir data/pairs/pmc_smoke

    # Bulk (1000 articles across two buckets)
    python scripts/pair_from_pmc.py \\
        --buckets PMC000xxxxxx,PMC001xxxxxx --limit 1000 \\
        --out-dir data/pairs/pmc --workers 2
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


USER_AGENT = "dart-semantic/0.0.1"
BULK_BASE = ("https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/"
             "oa_bulk/oa_comm/xml")
BASELINE = "2026-01-23"  # most-recent baseline as of 2026-05-05

# License accept-list. The PMC CSV `License` column uses the codes on
# the left; the JATS-XML license-ref attribute or visible URL uses the
# CC URL forms on the right. Either signal is enough.
ACCEPT_CODES = {
    "CC BY", "CC BY-SA", "CC0", "PUBLIC DOMAIN", "CC-BY", "CC-BY-SA",
}
ACCEPT_URL_RE = re.compile(
    r"creativecommons\.org/(licenses/by/|licenses/by-sa/|publicdomain/zero/"
    r"|publicdomain/mark/)",
    re.I,
)

# Share-alike-excluded variants. CC-BY-SA is commercial-OK but copyleft,
# and is OFF the project allowlist (CC-BY / CC0 / ODC-By) — see
# feedback_license_policy. When PMC_EXCLUDE_SHARE_ALIKE=1 (set by the
# --exclude-share-alike flag) both license gates drop SA. The toggle is
# read LIVE from the environment so it propagates to fork AND spawn
# ProcessPoolExecutor workers (env is inherited either way).
_SA_CODES = {"CC BY-SA", "CC-BY-SA"}
ACCEPT_CODES_NO_SA = ACCEPT_CODES - _SA_CODES
ACCEPT_URL_RE_NO_SA = re.compile(
    r"creativecommons\.org/(licenses/by/|publicdomain/zero/"
    r"|publicdomain/mark/)",
    re.I,
)


def _exclude_sa() -> bool:
    return os.environ.get("PMC_EXCLUDE_SHARE_ALIKE") == "1"


def _accept_codes() -> set[str]:
    return ACCEPT_CODES_NO_SA if _exclude_sa() else ACCEPT_CODES


def _accept_url_re() -> "re.Pattern[str]":
    return ACCEPT_URL_RE_NO_SA if _exclude_sa() else ACCEPT_URL_RE

PMC_CSS = """
body { font-family: 'Times New Roman', Times, serif; max-width: 7in;
       margin: 1in auto; color: #111; line-height: 1.45; }
h1 { font-size: 16pt; margin: 0 0 0.6em; }
h2 { font-size: 13pt; margin: 1.2em 0 0.4em; }
h3 { font-size: 11.5pt; margin: 1em 0 0.3em; font-style: italic; }
p { text-align: justify; margin: 0.4em 0; }
.affil { font-size: 10pt; color: #444; }
.doi, .pubmeta { font-size: 9.5pt; color: #555; }
.refs { font-size: 9.5pt; }
.refs li { margin: 0.3em 0; }
blockquote { margin: 0.8em 1.5em; color: #444; font-size: 10.5pt; }
"""


def slugify(s: str, maxlen: int = 70) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:maxlen] or "doc"


def wrap_html(body_html: str, title: str) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{(title or "PMC Article").strip()[:200]}</title>'
        f'<style>{PMC_CSS}</style></head><body>{body_html}</body></html>'
    )


# --------------------------------------------------------------------------
# Filelist CSV — license filter at fetch time
# --------------------------------------------------------------------------

def _csv_url(bucket: str) -> str:
    return f"{BULK_BASE}/oa_comm_xml.{bucket}.baseline.{BASELINE}.filelist.csv"


def _tar_url(bucket: str) -> str:
    return f"{BULK_BASE}/oa_comm_xml.{bucket}.baseline.{BASELINE}.tar.gz"


def fetch_filelist(bucket: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _csv_url(bucket)
    local = cache_dir / f"{bucket}.filelist.csv"
    if local.exists() and local.stat().st_size > 0:
        return local
    print(f"[fetch] {url}", file=sys.stderr)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    local.write_bytes(r.content)
    return local


def select_pmcids(filelist_csv: Path, limit: int) -> tuple[list[str], int]:
    """Return (kept PMCIDs, n_dropped_by_license)."""
    kept: list[str] = []
    dropped = 0
    with filelist_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            license_code = (row.get("License") or "").strip().upper()
            retracted = (row.get("Retracted") or "").strip().lower()
            if retracted == "yes":
                dropped += 1
                continue
            if license_code not in _accept_codes():
                dropped += 1
                continue
            pmcid = (row.get("AccessionID") or "").strip()
            if pmcid:
                kept.append(pmcid)
            if len(kept) >= limit:
                break
    return kept, dropped


# --------------------------------------------------------------------------
# Tarball streaming — pull only XMLs for kept PMCIDs
# --------------------------------------------------------------------------

def fetch_tarball(bucket: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _tar_url(bucket)
    local = cache_dir / f"{bucket}.tar.gz"
    if local.exists() and local.stat().st_size > 0:
        return local
    print(f"[fetch] {url}", file=sys.stderr)
    tmp = local.with_suffix(local.suffix + ".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT},
                      timeout=600, stream=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(256 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.rename(local)
    return local


def extract_xml_for_pmcids(tar_path: Path,
                           pmcids: set[str],
                           out_dir: Path) -> dict[str, Path]:
    """Stream the tarball; write `<PMCID>.xml` for each requested id.

    Returns {pmcid: xml_path}. Skips PMCIDs already present in `out_dir`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    # Avoid re-extracting if cached.
    for pmcid in list(pmcids):
        cached = out_dir / f"{pmcid}.xml"
        if cached.exists() and cached.stat().st_size > 0:
            found[pmcid] = cached
    remaining = pmcids - set(found.keys())
    if not remaining:
        return found

    print(f"[extract] streaming {tar_path.name} for {len(remaining)} XMLs",
          file=sys.stderr)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            base = name.rsplit("/", 1)[-1]
            if not base.endswith(".xml"):
                continue
            pmcid = base[:-4]  # strip .xml
            if pmcid not in remaining:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            data = fobj.read()
            target = out_dir / f"{pmcid}.xml"
            target.write_bytes(data)
            found[pmcid] = target
            remaining.discard(pmcid)
            if not remaining:
                break
    return found


# --------------------------------------------------------------------------
# Per-article verification & rendering
# --------------------------------------------------------------------------

def jats_license_ok(xml_text: str) -> bool:
    """Second-pass safety belt — verify the JATS payload itself carries
    a permissive license. Anything that fails this is dropped."""
    try:
        soup = BeautifulSoup(xml_text, "lxml-xml")
    except Exception:
        return False
    url_re = _accept_url_re()
    license_els = soup.find_all("license")
    # SA veto: when share-alike is excluded, drop the article if ANY license
    # signal mentions by-sa / share-alike. This catches CSV mislabels and
    # stops the "CC BY" text heuristic below from falsely matching a
    # "CC BY-SA" string (since "CC BY" is a substring of it).
    if _exclude_sa():
        blob = " ".join(
            (el.get("xlink:href") or "") + " " + (el.get("href") or "")
            + " " + (el.get("license-type") or "") + " " + el.get_text(" ")
            for el in license_els
        ).lower()
        if any(m in blob for m in ("by-sa", "by sa", "sharealike", "share-alike")):
            return False
    if not license_els:
        # Some PMC articles place the rights in a copyright-statement only.
        # If neither tag carries a CC marker, drop.
        copyright_text = " ".join(
            el.get_text(" ") for el in soup.find_all("copyright-statement"))
        return bool(url_re.search(copyright_text))
    ok_types = {"cc-by", "cc0", "ccby"}
    if not _exclude_sa():
        ok_types |= {"cc-by-sa", "ccbysa", "ccby-sa"}
    for el in license_els:
        href = el.get("xlink:href") or el.get("href") or ""
        license_type = (el.get("license-type") or "").lower()
        text = el.get_text(" ")
        if license_type in {"open-access", "openaccess"} and \
                not url_re.search(href + " " + text):
            # open-access alone is not sufficient — we need a CC marker.
            continue
        if url_re.search(href + " " + text):
            return True
        if license_type in ok_types:
            return True
        # Heuristic: visible "CC BY" / "CC0" / "Public Domain" text inside
        # the license block is also a strong positive signal. (Any SA string
        # was already vetoed above when share-alike is excluded.)
        compact = text.strip().upper()
        if any(code in compact for code in
               ("CC BY", "CC0", "CC-BY", "PUBLIC DOMAIN")):
            return True
    return False


def jats_to_html(xml_text: str) -> tuple[str, str]:
    """Render a JATS article to a minimal accessible HTML doc.

    Returns (html_doc, title). The HTML preserves <p>, headings, refs,
    and the JATS `<license>` block — which is exactly the structure the
    PMC extractor walks at training time.
    """
    soup = BeautifulSoup(xml_text, "lxml-xml")

    title_el = soup.find("article-title")
    title = (title_el.get_text(" ").strip() if title_el else "").strip()

    parts: list[str] = []
    if title:
        parts.append(f"<h1>{title}</h1>")

    # Authors (one row in extractor; visible in HTML as a paragraph).
    for cg in soup.find_all("contrib-group"):
        names: list[str] = []
        for surname in cg.find_all("surname"):
            given = surname.find_previous_sibling("given-names")
            given_t = given.get_text(" ").strip() if given else ""
            sur_t = surname.get_text(" ").strip()
            if sur_t:
                names.append((given_t + " " + sur_t).strip())
        if names:
            parts.append(f'<p class="affil">{", ".join(names)}</p>')

    # Pub metadata (journal title, year, volume, issue) — extractor keys
    # off the JATS XML (raw_source_xml), but we render visible HTML too
    # so axe-core gets a non-empty body.
    bits: list[str] = []
    j = soup.find("journal-title")
    if j:
        bits.append(j.get_text(" ").strip())
    yr = soup.find("year")
    if yr:
        bits.append(yr.get_text(" ").strip())
    vol = soup.find("volume")
    if vol:
        bits.append(f"vol. {vol.get_text(' ').strip()}")
    iss = soup.find("issue")
    if iss:
        bits.append(f"no. {iss.get_text(' ').strip()}")
    if bits:
        parts.append(f'<p class="pubmeta">{"; ".join(bits)}</p>')

    # License block — keep visible so the extractor's legal-class signal
    # rides along inside the rendered PDF.
    for lic in soup.find_all("license"):
        lic_text = lic.get_text(" ").strip()
        if lic_text:
            parts.append(f"<p class=\"pubmeta\">{lic_text}</p>")
    for cs in soup.find_all("copyright-statement"):
        cs_text = cs.get_text(" ").strip()
        if cs_text:
            parts.append(f"<p class=\"pubmeta\">{cs_text}</p>")

    # Abstract
    abs_el = soup.find("abstract")
    if abs_el:
        for p in abs_el.find_all("p"):
            t = p.get_text(" ").strip()
            if t:
                parts.append(f"<p>{t}</p>")

    # Body sections (sec)
    body = soup.find("body")
    if body:
        for sec in body.find_all("sec", recursive=True):
            t_el = sec.find("title", recursive=False)
            if t_el:
                heading = t_el.get_text(" ").strip()
                if heading:
                    parts.append(f"<h2>{heading}</h2>")
            for p in sec.find_all("p", recursive=False):
                t = p.get_text(" ").strip()
                if t:
                    parts.append(f"<p>{t}</p>")
        # Top-level paragraphs not inside a section
        for p in body.find_all("p", recursive=False):
            t = p.get_text(" ").strip()
            if t:
                parts.append(f"<p>{t}</p>")

    # References
    refs = soup.find_all("ref")
    if refs:
        parts.append("<h2>References</h2>")
        parts.append("<ol class=\"refs\">")
        for ref in refs:
            t = ref.get_text(" ").strip()
            if t:
                parts.append(f"<li>{t}</li>")
        parts.append("</ol>")

    # Footnotes
    for fn in soup.find_all(["fn", "notes"]):
        t = fn.get_text(" ").strip()
        if t:
            parts.append(f"<p class=\"pubmeta\">{t}</p>")

    body_html = "\n".join(parts) if parts else "<p>(empty article)</p>"
    return wrap_html(body_html, title), title


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def process_article(validator: HtmlValidator, work: tuple) -> dict:
    xml_path_str, out_dir_str = work
    xml_path = Path(xml_path_str)
    out_dir = Path(out_dir_str)
    pmcid = xml_path.stem
    stats = {"pmcid": pmcid, "ok": 0, "license_drop": 0,
             "parse_error": 0, "emit_drop": 0, "axe_drop": 0,
             "msg": ""}

    try:
        xml_text = xml_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        stats["parse_error"] = 1
        stats["msg"] = f"read: {exc}"
        return stats

    if not jats_license_ok(xml_text):
        stats["license_drop"] = 1
        stats["msg"] = "jats license filter"
        return stats

    try:
        html_doc, title = jats_to_html(xml_text)
    except Exception as exc:
        stats["emit_drop"] = 1
        stats["msg"] = f"emit: {exc}"
        return stats

    if len(html_doc) < 400:
        stats["emit_drop"] = 1
        stats["msg"] = "html too short"
        return stats

    result = validator.check(html_doc)
    if not result.ok:
        stats["axe_drop"] = 1
        stats["msg"] = "; ".join(result.reasons[:2])
        return stats

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / f"{pmcid}.pdf"
        validator.render_pdf(html_doc, pdf_path)
        input_ocr = pdf_to_ocr_text(pdf_path)

    if not input_ocr.strip():
        stats["axe_drop"] = 1
        stats["msg"] = "empty ocr"
        return stats

    pair = {
        "source": "pmc",
        "pmcid": pmcid,
        "title": title,
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        "variant_id": pmcid,
        "input_ocr": input_ocr,
        "output_html": html_doc,
        # JATS XML — extractor walks this for doc-role labels.
        "raw_source_xml": xml_text,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(pmcid)}.json"
    path.write_text(json.dumps(pair, ensure_ascii=False))
    stats["ok"] = 1
    return stats


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def parse_buckets(s: str) -> list[str]:
    return [b.strip() for b in s.split(",") if b.strip()]


def main() -> None:
    global BASELINE  # noqa: PLW0603
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets",
                    default="PMC000xxxxxx,PMC001xxxxxx,PMC002xxxxxx",
                    help="Comma-separated bucket prefixes "
                         "(PMC000xxxxxx..PMC011xxxxxx)")
    ap.add_argument("--bucket", default=None,
                    help="Shorthand for a single bucket. Overrides --buckets.")
    ap.add_argument("--limit", type=int, default=1000,
                    help="Total articles to KEEP after license filter.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/pairs/pmc"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("data/cache/pmc"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--baseline", default=BASELINE,
                    help="Override the baseline date stamp.")
    ap.add_argument("--exclude-share-alike", action="store_true",
                    help="Drop CC-BY-SA articles, keeping only CC-BY / CC0 / "
                         "Public-Domain (the project allowlist; see "
                         "feedback_license_policy). Sets PMC_EXCLUDE_SHARE_ALIKE=1 "
                         "so the filter also reaches worker processes.")
    args = ap.parse_args()
    BASELINE = args.baseline  # type: ignore[assignment]
    if args.exclude_share_alike:
        # Set in the env (not just a local) so fork/spawn pool workers,
        # which re-read it live in _exclude_sa(), apply the same gate.
        os.environ["PMC_EXCLUDE_SHARE_ALIKE"] = "1"
        print("[license] share-alike EXCLUDED: keeping CC-BY / CC0 / PD only",
              file=sys.stderr)

    if args.bucket:
        buckets = [args.bucket]
    else:
        buckets = parse_buckets(args.buckets)

    print(f"[catalog] buckets={buckets} target_kept={args.limit}",
          file=sys.stderr)

    existing = ({p.stem for p in args.out_dir.glob("*.json")}
                if args.out_dir.exists() else set())
    print(f"[plan] {len(existing)} pairs already on disk", file=sys.stderr)

    total_kept: list[Path] = []
    license_dropped_total = 0
    remaining_budget = max(0, args.limit - len(existing))

    for bucket in buckets:
        if remaining_budget <= 0:
            break
        try:
            csv_path = fetch_filelist(bucket, args.cache_dir)
        except Exception as exc:
            print(f"[warn] csv fetch fail {bucket}: {exc}", file=sys.stderr)
            continue

        # Slightly over-sample so the JATS-level license filter still
        # leaves us with enough articles after the second pass.
        oversample = int(remaining_budget * 1.1) + 5
        kept_ids, dropped = select_pmcids(csv_path, oversample)
        license_dropped_total += dropped
        # Filter out already-done.
        kept_ids = [p for p in kept_ids if p not in existing]
        if not kept_ids:
            continue

        try:
            tar_path = fetch_tarball(bucket, args.cache_dir)
        except Exception as exc:
            print(f"[warn] tar fetch fail {bucket}: {exc}", file=sys.stderr)
            continue

        xml_dir = args.cache_dir / f"{bucket}_xml"
        xml_paths = extract_xml_for_pmcids(
            tar_path, set(kept_ids), xml_dir)
        # Order by the kept_ids list.
        for pmcid in kept_ids:
            p = xml_paths.get(pmcid)
            if p:
                total_kept.append(p)
                if len(total_kept) >= remaining_budget:
                    break
        if len(total_kept) >= remaining_budget:
            break
        time.sleep(0.5)  # polite gap between bucket fetches

    if not total_kept:
        print("[summary] nothing to process", file=sys.stderr)
        print(f"[summary] license_dropped(csv)={license_dropped_total}",
              file=sys.stderr)
        return

    print(f"[plan] {len(total_kept)} JATS XMLs queued "
          f"(license_dropped(csv)={license_dropped_total})",
          file=sys.stderr)

    work_items = [(str(p), str(args.out_dir)) for p in total_kept]
    totals = {"ok": 0, "license_drop": 0, "parse_error": 0,
              "emit_drop": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_article, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        if done % 25 == 0 or any(stats.get(k) for k in
                                  ("license_drop", "parse_error",
                                   "emit_drop", "axe_drop")):
            outcome = next((k for k in ("ok", "license_drop", "parse_error",
                                        "emit_drop", "axe_drop")
                            if stats.get(k)), "?")
            print(f"[{done}/{len(work_items)}] {stats['pmcid']:12} "
                  f"{outcome:14} {stats.get('msg', '')[:55]}",
                  file=sys.stderr)

    elapsed = time.time() - start
    rate = totals["ok"] / max(1, done) * 100
    license_dropped_total += totals["license_drop"]
    print(f"\n[summary] processed={done}  ok={totals['ok']} ({rate:.1f}%)  "
          f"license_drop_total={license_dropped_total} "
          f"(csv {license_dropped_total - totals['license_drop']} + "
          f"jats {totals['license_drop']})  "
          f"parse_err={totals['parse_error']}  "
          f"emit_drop={totals['emit_drop']}  "
          f"axe_drop={totals['axe_drop']}  in {elapsed:.1f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
