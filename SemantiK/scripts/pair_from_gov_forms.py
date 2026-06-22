"""Generate training pairs from US federal fillable-PDF forms.

Source: IRS + GSA public-domain fillable PDFs. Each PDF is processed by:
  1. pymupdf widgets() → (field_name, type, bbox, tooltip)
  2. pymupdf text layer → positional text blocks
  3. label matcher: for each widget, pick nearest text-block as its label
  4. emit_html → WCAG <form> + <fieldset> + <label>/<input>/<select>
  5. feature-extract from the ORIGINAL PDF (real form typography)
  6. pair {input_ocr, output_html}

This fills the forms gap that arXiv and Wikipedia don't cover. Real
federal forms give us authentic layouts (multi-column, tiny print,
dense field labels); our synthetic-forms generator gives us clean
bulk volume. Hybrid recommended.

Usage:
    python scripts/pair_from_gov_forms.py --urls data/seeds/seed_gov_forms.txt
    python scripts/pair_from_gov_forms.py --irs-catalog 30

Public domain — no license filter needed for federal forms.
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

from dart_semantic import emit_html, ir
from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.parse_pdf_form import parse_pdf_form
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


USER_AGENT = "dart-semantic/0.0.1"

FORM_CSS = """
body { font-family: Arial, sans-serif; max-width: 7in; margin: 1in auto; color: #111; }
h1 { margin: 0 0 0.5em; }
form { margin-top: 1em; }
fieldset { border: 1px solid #888; padding: 0.8em 1em; margin: 0.8em 0; }
legend { font-weight: bold; padding: 0 0.4em; }
.field { margin: 0.6em 0; }
label { display: block; margin-bottom: 0.2em; font-weight: 600; }
input, select, textarea {
    min-width: 24px; min-height: 24px;
    padding: 4px 6px; font-size: 14px;
    border: 1px solid #666;
}
input[type=checkbox], input[type=radio] {
    margin-right: 0.4em;
}
"""


def wrap_html(raw_html: str) -> str:
    return raw_html.replace("<head>", f"<head><style>{FORM_CSS}</style>", 1)


def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:maxlen] or "form"


# ---------- starter URL list ----------

# Curated IRS forms (public domain). The URL pattern
# https://www.irs.gov/pub/irs-pdf/<short>.pdf is stable — IRS retires forms
# only infrequently. These are all fillable.
CURATED_IRS_FORMS = [
    ("w4",    "W-4 Employee's Withholding Certificate"),
    ("w9",    "W-9 Request for Taxpayer Identification Number"),
    ("form-a", "Form 1040 U.S. Individual Income Tax Return"),
    ("form-asa","Schedule A (Form 1040) Itemized Deductions"),
    ("form-asb","Schedule B (Form 1040) Interest and Ordinary Dividends"),
    ("form-asc","Schedule C (Form 1040) Profit or Loss From Business"),
    ("form-asd","Schedule D (Form 1040) Capital Gains and Losses"),
    ("form-ase","Schedule E (Form 1040) Supplemental Income and Loss"),
    ("f1099misc","Form 1099-MISC Miscellaneous Income"),
    ("f1099nec","Form 1099-NEC Nonemployee Compensation"),
    ("f2441", "Form 2441 Child and Dependent Care Expenses"),
    ("f4562", "Form 4562 Depreciation and Amortization"),
    ("f8606", "Form 8606 Nondeductible IRAs"),
    ("f8863", "Form 8863 Education Credits"),
    ("f8962", "Form 8962 Premium Tax Credit"),
    ("fss4",  "Form SS-4 Application for Employer Identification Number"),
    ("fss5",  "Form SS-5 Application for Social Security Card"),
]

IRS_URL_TEMPLATE = "https://www.irs.gov/pub/irs-pdf/{short}.pdf"


def process_form(validator: HtmlValidator,
                 work: tuple[str, str, str]) -> dict:
    """Worker: download PDF, parse form, emit pair."""
    url, title, out_dir_str = work
    out_dir = Path(out_dir_str)
    stats = {"url": url, "title": title, "ok": 0,
             "fetch_error": 0, "parse_error": 0,
             "emitter_drop": 0, "axe_drop": 0}

    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        stats["fetch_error"] = 1
        stats["msg"] = str(exc)
        return stats

    # Save the downloaded PDF alongside the pair JSON so v2 data builders
    # can re-extract widget metadata from the real PDF instead of having
    # to re-render the output_html (which is our synthetic reconstruction
    # and has no fillable widgets to extract from).
    variant_id = slugify(url.rsplit("/", 1)[-1].replace(".pdf", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{variant_id}.pdf"
    pdf_path.write_bytes(r.content)

    try:
        document = parse_pdf_form(pdf_path, title=title, source_url=url)
    except ir.IRError as exc:
        stats["parse_error"] = 1
        stats["msg"] = str(exc)
        pdf_path.unlink(missing_ok=True)
        return stats

    try:
        raw_html = emit_html.emit(document)
    except emit_html.EmitterError as exc:
        stats["emitter_drop"] = 1
        stats["msg"] = f"emit: {exc}"
        pdf_path.unlink(missing_ok=True)
        return stats
    html_doc = wrap_html(raw_html)

    result = validator.check(html_doc)
    if not result.ok:
        stats["axe_drop"] = 1
        stats["msg"] = "; ".join(result.reasons[:2])
        pdf_path.unlink(missing_ok=True)
        return stats

    # Use the ORIGINAL form PDF for feature extraction — we want the
    # real scanned-form visual style (cramped multi-column field grids,
    # tiny labels, tabular rows), not a rendered-from-HTML clean form.
    input_ocr = pdf_to_ocr_text(pdf_path)

    if not input_ocr.strip():
        stats["axe_drop"] = 1
        stats["msg"] = "empty ocr"
        pdf_path.unlink(missing_ok=True)
        return stats

    pair = {
        "source": "pdf_form",
        "url": url,
        "title": title,
        "variant_id": variant_id,
        "input_ocr": input_ocr,
        "output_html": html_doc,
        "local_pdf": str(pdf_path),
    }
    (out_dir / f"{variant_id}.json").write_text(
        json.dumps(pair, ensure_ascii=False))
    stats["ok"] = 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--curated-irs", action="store_true",
                   help=f"Process the {len(CURATED_IRS_FORMS)}-form curated IRS starter set")
    g.add_argument("--urls-file", type=Path,
                   help="Path to file with 'url | title' per line")
    ap.add_argument("--out-dir", type=Path, default=Path("data/pairs/forms"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    work: list[tuple[str, str, str]] = []
    if args.curated_irs:
        for short, title in CURATED_IRS_FORMS:
            work.append((IRS_URL_TEMPLATE.format(short=short), title,
                         str(args.out_dir)))
    else:
        for line in args.urls_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                url, title = [s.strip() for s in line.split("|", 1)]
            else:
                url, title = line, line.rsplit("/", 1)[-1]
            work.append((url, title, str(args.out_dir)))

    existing = ({p.stem for p in args.out_dir.glob("*.json")}
                if args.out_dir.exists() else set())
    work = [w for w in work if slugify(w[0].rsplit("/", 1)[-1].replace(".pdf", "")) not in existing]
    print(f"[plan] {len(work)} forms to process "
          f"({len(existing)} already done)", file=sys.stderr)

    totals = {"ok": 0, "fetch_error": 0, "parse_error": 0,
              "emitter_drop": 0, "axe_drop": 0}
    start = time.time()
    done = 0
    for stats in run_in_pool(process_form, work, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        outcome = next((k for k in ("ok", "fetch_error", "parse_error",
                                    "emitter_drop", "axe_drop")
                        if stats.get(k)), "?")
        title_s = stats.get("title", "")[:50]
        print(f"[{done}/{len(work)}] {title_s:50} {outcome:14} "
              f"{stats.get('msg', '')[:50]}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\n[summary] forms={done}  ok={totals['ok']}  "
          f"fetch_err={totals['fetch_error']}  parse_err={totals['parse_error']}  "
          f"emit_drop={totals['emitter_drop']}  axe_drop={totals['axe_drop']}  "
          f"in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
