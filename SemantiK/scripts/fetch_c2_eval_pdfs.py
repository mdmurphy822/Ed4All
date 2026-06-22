"""Fetch eval-ONLY non-arXiv PDFs for the C2 eval-corpus strata (Plan 12 V3.2/C2).

Downloads a modest, license-clean set of real-world documents into
``data/eval_corpus/c2_pdfs/<source>/`` with a ``_fetch_manifest.jsonl`` per
source (url, fetched_at, sha256, bytes, license provenance, failures). These
PDFs back NO training pairs — they exist purely so
``scripts/build_eval_corpus_manifest.py`` can emit ``c2_*`` strata.

Sources (ALL US federal government works — public domain, 17 U.S.C. §105;
court opinions per Banks v. Manchester, 128 U.S. 244 (1888)):

- ``gov_forms``     5 IRS fillable forms (the pair_from_gov_forms.py targets,
                    but this time the PDFs are KEPT).
- ``courtlistener`` 5 FRESH federal opinions via the anon v4 Search API,
                    excluding every opinion id already cached under
                    ``data/cache/courtlistener`` (those back trained pairs).
- ``govinfo``       CFR accessibility parts (36 CFR 1194 Section 508,
                    36 CFR 1191 ADA guidelines) + 2 recent Federal Register
                    rules, all <20MB.
- ``scanned_ocr``   GPO page scans from Statutes at Large (ADA 1990, Civil
                    Rights Act 1964, Voting Rights Act 1965) — image-heavy
                    inputs for the Tesseract path. The embedded GPO OCR text
                    layer (if any) is measured and recorded honestly.

Polite HTTP: serial, ~1 req/s (``--delay``), descriptive UA, bounded retries
(scripts._figure_fetch_lib.polite_get). Failures are RECORDED in the manifest
and the run continues — partial coverage honestly recorded beats fabricated
completeness.

CPU only — no torch, no GPU.

Usage::

    .venv/bin/python -m scripts.fetch_c2_eval_pdfs            # all sources
    .venv/bin/python -m scripts.fetch_c2_eval_pdfs --sources gov_forms,govinfo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._figure_fetch_lib import polite_get  # noqa: E402

UA = "DART-accessibility-research/0.1 (C2 eval corpus; mailto:mdmurphy822@gmail.com)"
DEFAULT_OUT_ROOT = Path("data/eval_corpus/c2_pdfs")
DEFAULT_CL_CACHE = Path("data/cache/courtlistener")
LICENSE = "us-gov-pd"
MAX_PDF_BYTES = 20 * 1024 * 1024  # mirror the manifest builder's runtime cap

_LICENSE_NOTES = {
    "gov_forms": "US federal government work (IRS form) — public domain, 17 U.S.C. §105.",
    "courtlistener": (
        "US federal judicial opinion — public domain "
        "(Banks v. Manchester, 128 U.S. 244 (1888)); Free Law Project mirror."
    ),
    "govinfo": "US federal government work (GPO/govinfo.gov) — public domain, 17 U.S.C. §105.",
    "scanned_ocr": (
        "US federal government work (GPO Statutes at Large page scan) — "
        "public domain, 17 U.S.C. §105."
    ),
}

# ---------------------------------------------------------------------------
# Curated URL sets (HEAD-verified 2026-06-10; all 200 + application/pdf, <20MB)
# ---------------------------------------------------------------------------
# IRS shorts follow pair_from_gov_forms.CURATED_IRS_FORMS, corrected to the
# filenames IRS actually serves today (w4/w9 are published as fw4/form-b).
GOV_FORMS = [
    ("https://www.irs.gov/pub/irs-pdf/form-a.pdf", "Form 1040 U.S. Individual Income Tax Return"),
    ("https://www.irs.gov/pub/irs-pdf/form-b.pdf", "W-9 Request for Taxpayer Identification Number"),
    ("https://www.irs.gov/pub/irs-pdf/fw4.pdf", "W-4 Employee's Withholding Certificate"),
    (
        "https://www.irs.gov/pub/irs-pdf/fss4.pdf",
        "Form SS-4 Application for Employer Identification Number",
    ),
    ("https://www.irs.gov/pub/irs-pdf/f8863.pdf", "Form 8863 Education Credits"),
]

GOVINFO = [
    (
        "https://www.govinfo.gov/content/pkg/sample-cfr-vol/pdf/"
        "sample-cfr-part.pdf",
        "36 CFR Part 1194 — Electronic and Information Technology Accessibility"
        " Standards (Section 508)",
    ),
    (
        "https://www.govinfo.gov/content/pkg/CFR-2024-title36-vol3/pdf/"
        "CFR-2024-title36-vol3-part1191.pdf",
        "36 CFR Part 1191 — ADA Accessibility Guidelines for Buildings and Facilities",
    ),
    (
        "https://www.govinfo.gov/content/pkg/FR-2026-06-10/pdf/2026-11609.pdf",
        "Federal Register rule 2026-11609 — Endangered and Threatened Wildlife and Plants (9 pp)",
    ),
    (
        "https://www.govinfo.gov/content/pkg/FR-2026-06-10/pdf/0000-00000.pdf",
        "Federal Register rule 0000-00000 — Venezuela Sanctions Regulations"
        " Web General Licenses (4 pp)",
    ),
]

SCANNED_OCR = [
    (
        "https://www.govinfo.gov/content/pkg/STATUTE-104/pdf/STATUTE-104-Pg327.pdf",
        "Americans with Disabilities Act of 1990, 104 Stat. 327 (GPO page scan)",
    ),
    (
        "https://www.govinfo.gov/content/pkg/STATUTE-78/pdf/STATUTE-78-Pg241.pdf",
        "Civil Rights Act of 1964, 78 Stat. 241 (GPO page scan)",
    ),
    (
        "https://www.govinfo.gov/content/pkg/STATUTE-79/pdf/STATUTE-79-Pg437.pdf",
        "Voting Rights Act of 1965, 79 Stat. 437 (GPO page scan)",
    ),
]

CL_SEARCH_API = "https://www.courtlistener.com/api/rest/v4/search/"
CL_STORAGE_BASE = "https://storage.courtlistener.com"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pdf_stats(path: Path) -> dict:
    """{n_pages, text_chars} via pdfplumber; honest measurement of the text layer.

    Never fails the fetch — a parse error is itself worth recording.
    """
    try:
        import pdfplumber  # the pipeline's PDF lib (pymupdf is not in this venv)

        with pdfplumber.open(path) as doc:
            n_pages = len(doc.pages)
            text_chars = sum(len(p.extract_text() or "") for p in doc.pages)
        return {"n_pages": n_pages, "text_chars": text_chars}
    except Exception as exc:  # noqa: BLE001 — record, don't fail
        return {"pdf_stats_error": f"{type(exc).__name__}: {exc}"}


def _record(mf: IO[str], rec: dict) -> None:
    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    mf.flush()


def _fetch_one(
    mf: IO[str],
    *,
    source: str,
    stratum: str,
    url: str,
    title: str,
    dest: Path,
    delay: float,
    timeout: int,
    retries: int,
    extra: dict | None = None,
    log=print,
) -> bool:
    """Fetch ``url`` -> ``dest``, append one manifest line. True on a kept PDF."""
    base = {
        "source": source,
        "stratum": stratum,
        "url": url,
        "title": title,
        "local_path": str(dest),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "license": LICENSE,
        "license_note": _LICENSE_NOTES[source],
    }
    if extra:
        base.update(extra)

    if dest.exists() and dest.stat().st_size > 0:
        body = dest.read_bytes()
        base.update(
            status="skip_exists",
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            **_pdf_stats(dest),
        )
        _record(mf, base)
        log(f"[c2:{source}] skip (exists) {dest.name}")
        return True

    try:
        body, ctype, _ = polite_get(url, timeout=timeout, retries=retries, delay=delay, ua=UA)
    except urllib.error.HTTPError as e:
        base.update(status=f"http_{e.code}", bytes=0, note="fetch failed; recorded, not retried")
        _record(mf, base)
        log(f"[c2:{source}] FAIL http_{e.code} {url}")
        return False
    except Exception as e:  # noqa: BLE001 — record the transient class
        base.update(status=f"error:{type(e).__name__}", bytes=0, note=str(e)[:200])
        _record(mf, base)
        log(f"[c2:{source}] FAIL {type(e).__name__} {url}")
        return False

    # PDF header may sit after a preamble (spec allows it within the first
    # 1024 bytes — court-stamped opinions really do this, e.g. a 139-byte
    # "Cas:...:Doc:FinalOpinion:" banner before %PDF).
    if body[:1024].find(b"%PDF") < 0:
        base.update(status="not_pdf", bytes=len(body), content_type=ctype)
        _record(mf, base)
        log(f"[c2:{source}] FAIL not_pdf ({ctype}) {url}")
        return False
    if len(body) > MAX_PDF_BYTES:
        base.update(
            status="too_large",
            bytes=len(body),
            note=f"exceeds {MAX_PDF_BYTES} byte eval cap; not kept",
        )
        _record(mf, base)
        log(f"[c2:{source}] FAIL too_large ({len(body) / 1e6:.1f}MB) {url}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    base.update(
        status="ok",
        bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        content_type=ctype,
        **_pdf_stats(dest),
    )
    _record(mf, base)
    log(f"[c2:{source}] ok {dest.name} ({len(body) / 1e6:.2f}MB)")
    return True


def _slug(url: str) -> str:
    name = url.rsplit("/", 1)[-1]
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


# ---------------------------------------------------------------------------
# static-list sources
# ---------------------------------------------------------------------------
def fetch_static_source(
    out_root: Path,
    source: str,
    items: list[tuple[str, str]],
    *,
    stratum: str,
    delay: float,
    timeout: int,
    retries: int,
    log=print,
) -> tuple[int, int]:
    sdir = out_root / source
    sdir.mkdir(parents=True, exist_ok=True)
    ok = 0
    with (sdir / "_fetch_manifest.jsonl").open("a", encoding="utf-8") as mf:
        for url, title in items:
            ok += _fetch_one(
                mf,
                source=source,
                stratum=stratum,
                url=url,
                title=title,
                dest=sdir / _slug(url),
                delay=delay,
                timeout=timeout,
                retries=retries,
                log=log,
            )
    return ok, len(items)


# ---------------------------------------------------------------------------
# courtlistener — fresh federal opinions, excluding trained-pair cache ids
# ---------------------------------------------------------------------------
def _cached_opinion_ids(cl_cache: Path) -> set[str]:
    """Opinion ids of every cached PDF (opinion_<id>.pdf) — ALL back trained pairs."""
    out: set[str] = set()
    if cl_cache.is_dir():
        for p in cl_cache.glob("opinion_*.pdf"):
            out.add(p.stem.removeprefix("opinion_"))
    return out


def fetch_courtlistener(
    out_root: Path,
    *,
    want: int,
    cl_cache: Path,
    delay: float,
    timeout: int,
    retries: int,
    max_pages: int = 5,
    log=print,
) -> tuple[int, int]:
    source = "courtlistener"
    sdir = out_root / source
    sdir.mkdir(parents=True, exist_ok=True)
    exclude = _cached_opinion_ids(cl_cache)
    log(f"[c2:courtlistener] excluding {len(exclude)} cached (trained-pair) opinion ids")

    ok = attempted = 0
    next_url: str | None = (
        f"{CL_SEARCH_API}?type=o&page_size=100&court_jurisdiction=F&order_by=dateFiled%20desc"
    )
    with (sdir / "_fetch_manifest.jsonl").open("a", encoding="utf-8") as mf:
        for _page in range(max_pages):
            if ok >= want or not next_url:
                break
            try:
                body, _, _ = polite_get(
                    next_url, timeout=timeout, retries=retries, delay=delay, ua=UA
                )
                data = json.loads(body)
            except Exception as e:  # noqa: BLE001 — record, continue with what we have
                _record(
                    mf,
                    {
                        "source": source,
                        "url": next_url,
                        "status": f"search_error:{type(e).__name__}",
                        "note": str(e)[:200],
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                log(f"[c2:courtlistener] search FAIL {type(e).__name__}: {e}")
                break

            for r in data.get("results", []) or []:
                if ok >= want:
                    break
                if r.get("court_jurisdiction") != "F":
                    continue
                for op in r.get("opinions", []) or []:
                    op_id = op.get("id")
                    local_path = (op.get("local_path") or "").lstrip("/")
                    if not op_id or str(op_id) in exclude:
                        continue
                    if not local_path.lower().endswith(".pdf"):
                        continue
                    attempted += 1
                    got = _fetch_one(
                        mf,
                        source=source,
                        stratum="c2_courtlistener",
                        url=f"{CL_STORAGE_BASE}/{local_path}",
                        title=(r.get("caseName") or "")[:200],
                        dest=sdir / f"opinion_{op_id}.pdf",
                        delay=delay,
                        timeout=timeout,
                        retries=retries,
                        extra={
                            "opinion_id": op_id,
                            "cluster_id": r.get("cluster_id"),
                            "docket_number": r.get("docketNumber") or "",
                            "court": r.get("court") or "",
                            "date_filed": r.get("dateFiled") or "",
                            "held_out_check": (
                                f"opinion id absent from {cl_cache} "
                                f"({len(exclude)} trained-pair ids)"
                            ),
                        },
                        log=log,
                    )
                    ok += got
                    break  # at most one opinion per cluster
            next_url = data.get("next") or None
    return ok, attempted


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
ALL_SOURCES = ("gov_forms", "courtlistener", "govinfo", "scanned_ocr")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--sources", type=str, default=",".join(ALL_SOURCES))
    ap.add_argument("--cl-cache", type=Path, default=DEFAULT_CL_CACHE)
    ap.add_argument("--n-courtlistener", type=int, default=5)
    ap.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds slept after every request (be polite: 1 req/s)",
    )
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args(argv)

    sources = tuple(s for s in args.sources.split(",") if s)
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)} (known: {ALL_SOURCES})")

    kw = dict(delay=args.delay, timeout=args.timeout, retries=args.retries)
    totals: dict[str, tuple[int, int]] = {}
    if "gov_forms" in sources:
        totals["gov_forms"] = fetch_static_source(
            args.out_root, "gov_forms", GOV_FORMS, stratum="c2_gov_forms", **kw
        )
    if "courtlistener" in sources:
        totals["courtlistener"] = fetch_courtlistener(
            args.out_root, want=args.n_courtlistener, cl_cache=args.cl_cache, **kw
        )
    if "govinfo" in sources:
        totals["govinfo"] = fetch_static_source(
            args.out_root, "govinfo", GOVINFO, stratum="c2_govinfo", **kw
        )
    if "scanned_ocr" in sources:
        totals["scanned_ocr"] = fetch_static_source(
            args.out_root, "scanned_ocr", SCANNED_OCR, stratum="scanned_ocr", **kw
        )

    print("\n[c2 summary]")
    for src, (ok, tried) in totals.items():
        print(f"  {src:14s} {ok}/{tried} kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
