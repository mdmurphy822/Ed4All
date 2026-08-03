"""Build a figure ground-truth catalog from LOCAL sources only.

Staging manifest for an image/alt-text dataset (Plans/09 section 3a, task #22).
This is CPU-only HTML/XML parsing prep: NO GPU, NO network, NO model imports.

The GROUND TRUTH per figure is the *caption*:
    * arXiv (ar5iv LaTeXML HTML): the ``<figcaption>`` text.
    * PMC (JATS XML):             the ``<fig>``'s ``<caption>`` text.
    * OpenStax (rendered HTML):   the ``<figcaption>`` text (CC-BY).
The ``alt`` attribute is captured but is NOT the target — on ar5iv it's the
placeholder "Refer to caption"; on OpenStax/PMC it can be a genuine long
description (also captured, still not the caption target).

Image bytes are NOT present locally. Every record carries ``image_ref`` (the
verbatim src/href to obtain later) and ``image_local: false``.

Disambiguation (ar5iv renders MATH and the site logo as <img> too):
    * Only emit a record when the <figure> contains a <figcaption>.
    * The img src must be a raster/vector asset (.png/.jpg/.jpeg/.svg/.gif),
      NOT a data: URI and NOT the ar5iv logo (/assets/ar5iv.png).
    * Skip figures whose only images are inline-math data-URIs.

License discipline (feedback_license_policy — commercial-OK only):
    * arXiv ar5iv_html_cache = the CC-cleared ar5iv "no-problem" subset.
    * PMC: read the per-doc JATS <license>; flag anything that is NOT clearly
      CC-BY / CC0 / Public-Domain so it can be excluded.
    * OpenStax = CC-BY.
    * Wikipedia/Commons (CC-BY-SA) is intentionally NOT a source here.

Parsing reuses the repo's existing approach: BeautifulSoup with the ``lxml``
tree-builder for HTML (as in ``semantik_structure/parse_ar5iv.py``) and the
``lxml-xml`` tree-builder for JATS (as in ``scripts/pair_from_pmc.py``).

Usage (always pin the GPU off — a QLoRA train owns it):
    CUDA_VISIBLE_DEVICES="" python data/builders/build_figure_catalog.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Allowed image asset extensions (raster + svg). .eps/.pdf are vector but
# off this allowlist per the disambiguation spec; figures whose only asset is
# one of those (or extension-less) are skipped.
ALLOWED_EXTS = {"png", "jpg", "jpeg", "svg", "gif"}

# ar5iv site logo — never a real figure image.
AR5IV_LOGO_RE = re.compile(r"/assets/ar5iv", re.I)

_WS_RE = re.compile(r"\s+")
_EXT_RE = re.compile(r"\.([a-z0-9]+)(?:[?#]|$)", re.I)
# "Figure 12 " / "Fig. 3:" leading label that OpenStax/some sources prepend.
_FIG_LABEL_RE = re.compile(r"^\s*(?:figure|fig\.?)\s*\d+\s*[:.—-]?\s*", re.I)

# License URL -> clean tag. Order matters (check specific before generic).
_LICENSE_URL_RE = re.compile(
    r"creativecommons\.org/(licenses/by-sa/|licenses/by/"
    r"|publicdomain/zero/|publicdomain/mark/)",
    re.I,
)

# ---------------------------------------------------------------------------
# Captionless / decorative classification (Plans/09 §2 item 2: decorative ->
# alt="" with NO generation). Heuristic per the decorative spec: tiny images,
# ornaments/logos/spacer graphics are decorative; everything else captionless
# is "uncaptioned_content" (content-bearing but without caption ground truth).
# ---------------------------------------------------------------------------

# Path smells of chrome/ornament graphics (logo, icon, spacer, rules...).
# Matched against the FULL src path: site chrome often lives in telltale
# directories ("/static/browse/", "/icons/social/") with neutral basenames.
DECORATIVE_NAME_RE = re.compile(
    r"(?:^|[/_.-])(logos?|icons?|spacer|ornament|bullet|divider|hrule|rule|"
    r"border|banner|button|arrow|seal|orcid|email|signature|deco\w*|"
    r"header|footer|watermark|sprite|badge|social|static)(?=[/_.\-\d]|$)",
    re.I,
)
# min(width, height) at or below this many px -> tiny ornament/spacer.
TINY_MAX_PX = 32
# max/min dimension ratio at or above this -> rule / spacer graphic.
EXTREME_ASPECT = 15.0

_PX_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:px)?\s*$", re.I)


def _px(value: object) -> float | None:
    """Parse an img width/height attribute to px. None for %, em, missing."""
    if value is None:
        return None
    m = _PX_RE.match(str(value))
    return float(m.group(1)) if m else None


def classify_decorative(
    src: str,
    width: object = None,
    height: object = None,
    alt: str | None = None,
) -> tuple[str, list[str]]:
    """Classify a captionless image per the Plans/09 decorative spec.

    Returns ``(decorative_class, signals)`` where ``decorative_class`` is
    ``"decorative"`` (tiny / ornament / logo / spacer -> target alt="") or
    ``"uncaptioned_content"`` (genuinely content-bearing but uncaptioned ->
    needs a generated alt, NOT alt=""). ``signals`` lists which heuristics
    fired; empty for uncaptioned_content.
    """
    signals: list[str] = []
    # Author-declared decorative: alt attribute PRESENT but empty. A missing
    # alt (None) is no signal — most sources simply omit it.
    if alt is not None and not alt.strip():
        signals.append("explicit_empty_alt")
    if DECORATIVE_NAME_RE.search(src or ""):
        signals.append("name_smell")
    w, h = _px(width), _px(height)
    if w is not None and h is not None and w > 0 and h > 0:
        if min(w, h) <= TINY_MAX_PX:
            signals.append("tiny")
        if max(w, h) / min(w, h) >= EXTREME_ASPECT:
            signals.append("extreme_aspect")
    elif (w or h) is not None:
        d = w if w is not None else h
        if d is not None and 0 < d <= TINY_MAX_PX:
            signals.append("tiny")
    return ("decorative" if signals else "uncaptioned_content", signals)


def captionless_fields(
    src: str, width: object = None, height: object = None, alt: str | None = None
) -> dict:
    """The marker fields every captionless catalog record carries."""
    cls, signals = classify_decorative(src, width, height, alt)
    return {
        "figcaption": None,  # caption: null — no caption ground truth
        "captionless": True,
        "decorative_candidate": True,
        "decorative_class": cls,
        "decorative_signals": signals,
    }


def _norm_ws(text: str | None) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def _image_format_hint(src: str) -> str | None:
    """Best-effort format guess from the src/href.

    For PMC, JATS graphic hrefs are extension-less (e.g. 'pbio.0030060.g001');
    PMC OA figure assets are predominantly JPEG, but without an explicit
    extension we return None rather than guess wrong.
    """
    if not src:
        return None
    # webp delivery hint embedded in OpenStax CDN paths: 'f=webp'.
    if re.search(r"[?&/]f=webp\b", src, re.I):
        return "webp"
    m = _EXT_RE.search(src)
    if not m:
        return None
    ext = m.group(1).lower()
    if ext == "jpeg":
        return "jpg"
    # PMC JATS graphic hrefs are extension-less but end in a figure-id segment
    # like '.g001' / '.e004' that the extension regex mis-reads. Only return a
    # hint for genuinely-recognized image formats.
    if ext not in ALLOWED_EXTS and ext != "tif" and ext != "tiff":
        return None
    return ext


def _ext_of(src: str) -> str | None:
    m = _EXT_RE.search(src or "")
    return m.group(1).lower() if m else None


# --------------------------------------------------------------------------
# License tagging (commercial-OK discipline)
# --------------------------------------------------------------------------


def classify_license_url(text_or_url: str) -> tuple[str, bool]:
    """Return (clean_tag, commercial_ok) from a license URL/text blob.

    commercial_ok is False for CC-BY-SA (share-alike — off the project
    allowlist for a generative captioner) and for anything with no
    recognizable CC-BY / CC0 / Public-Domain marker.
    """
    blob = text_or_url or ""
    low = blob.lower()
    if "by-sa" in low or "sharealike" in low or "share-alike" in low:
        return ("CC-BY-SA", False)
    m = _LICENSE_URL_RE.search(blob)
    if m:
        seg = m.group(1).lower()
        if seg.startswith("licenses/by-sa"):
            return ("CC-BY-SA", False)
        if seg.startswith("licenses/by"):
            return ("CC-BY", True)
        if "publicdomain/zero" in seg:
            return ("CC0", True)
        if "publicdomain/mark" in seg:
            return ("Public-Domain", True)
    # Text heuristics for blocks without a clean URL.
    if "cc0" in low or "public domain" in low or "publicdomain" in low:
        return ("CC0/PD", True)
    if "cc by" in low or "cc-by" in low or "attribution" in low:
        return ("CC-BY", True)
    return ("UNKNOWN", False)


def pmc_license(soup: BeautifulSoup) -> tuple[str, bool]:
    """Resolve a PMC article's effective license tag and commercial_ok flag.

    Reads JATS <license> (xlink:href / license-type / visible text), falling
    back to <copyright-statement>. Any share-alike signal forces CC-BY-SA
    (commercial_ok=False) so it gets flagged.
    """
    license_els = soup.find_all("license")
    signals: list[str] = []
    for el in license_els:
        href = el.get("xlink:href") or el.get("href") or ""
        ltype = el.get("license-type") or ""
        signals.append(f"{href} {ltype} {el.get_text(' ')}")
    if not license_els:
        signals = [el.get_text(" ") for el in soup.find_all("copyright-statement")]
    if not signals:
        return ("UNKNOWN", False)
    blob = " ".join(signals)
    # If ANY signal carries share-alike, veto.
    if re.search(r"by-sa|sharealike|share-alike", blob, re.I):
        return ("CC-BY-SA", False)
    return classify_license_url(blob)


# --------------------------------------------------------------------------
# arXiv / ar5iv (HTML) extraction
# --------------------------------------------------------------------------


def _figure_context(fig: Tag, label_text: str) -> str:
    """Cheap surrounding context: the nearest preceding/following <p> that
    references the figure's label (e.g. mentions 'Figure 3'). Empty if none
    is cheaply found.
    """
    if not label_text:
        return ""
    # Build a loose matcher for the figure number from the caption label.
    m = re.search(r"(\d+)", label_text)
    if not m:
        return ""
    num = m.group(1)
    pat = re.compile(rf"\bfig(?:ure)?\.?\s*{re.escape(num)}\b", re.I)
    # Look at a handful of nearby paragraphs without walking the whole doc.
    for finder in (fig.find_all_previous, fig.find_all_next):
        seen = 0
        for p in finder("p"):
            seen += 1
            if seen > 12:
                break
            txt = _norm_ws(p.get_text(" "))
            if txt and pat.search(txt):
                return txt[:600]
    return ""


def _usable_img_src(img: Tag) -> str | None:
    """The img's src when it is a usable raster/vector asset, else None.

    Drops data: URIs (inline math), the ar5iv logo, and off-allowlist
    extensions — the same disambiguation the captioned path always applied.
    """
    src = (img.get("src") or "").strip()
    if not src or src.startswith("data:"):
        return None
    if AR5IV_LOGO_RE.search(src):
        return None
    if _ext_of(src) not in ALLOWED_EXTS:
        return None
    return src


def _arxiv_base_record(doc_id: str, fig_index: int | None, src: str) -> dict:
    return {
        "source": "arxiv",
        "doc_id": doc_id,
        "figure_index": fig_index,
        "image_ref": src,
        "image_format_hint": _image_format_hint(src),
        "surrounding_context": "",
        "license": "CC (ar5iv no-problem subset)",
        "image_local": False,
    }


def extract_arxiv(html_path: Path, include_captionless: bool = False) -> list[dict]:
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    if "<figcaption" not in raw and not (include_captionless and "<img" in raw):
        return []
    soup = BeautifulSoup(raw, "lxml")
    doc_id = html_path.stem
    records: list[dict] = []
    fig_index = 0
    # NET-NEW captionless records get figure_index: null + their own ordinal
    # (captionless_index) so the captioned figure_index sequence is IDENTICAL
    # with and without the flag (consumers join captioned rows on it).
    capless_index = 0
    for fig in soup.find_all("figure"):
        figcap = fig.find("figcaption")
        caption_text = _norm_ws(figcap.get_text(" ")) if figcap is not None else ""

        # Collect candidate images: raster/vector assets only, drop data: URIs
        # and the ar5iv logo. A figure of only math data-URIs yields none.
        chosen_img: Tag | None = None
        for img in fig.find_all("img"):
            if _usable_img_src(img) is not None:
                chosen_img = img
                break
        if chosen_img is None:
            continue  # math-only / no usable raster asset

        src = (chosen_img.get("src") or "").strip()
        alt = chosen_img.get("alt")
        alt_norm = alt if (alt is not None and alt.strip()) else None

        if figcap is None or not caption_text:
            # Captionless figure: dropped by default (no caption ground
            # truth), cataloged as a decorative candidate behind the flag.
            if not include_captionless:
                continue
            capless_index += 1
            records.append(
                {
                    **_arxiv_base_record(doc_id, None, src),
                    "captionless_index": capless_index,
                    "alt_raw": alt_norm,
                    **captionless_fields(
                        src, chosen_img.get("width"), chosen_img.get("height"), alt
                    ),
                }
            )
            continue

        fig_index += 1
        records.append(
            {
                **_arxiv_base_record(doc_id, fig_index, src),
                "figcaption": caption_text,
                "alt_raw": alt_norm,
                "surrounding_context": _figure_context(fig, caption_text),
            }
        )

    if include_captionless:
        # Standalone <img> outside any <figure>: the classic home of
        # ornaments / logos / spacer graphics (decorative candidates).
        for img in soup.find_all("img"):
            if img.find_parent("figure") is not None:
                continue
            src = _usable_img_src(img)
            if src is None:
                continue
            alt = img.get("alt")
            capless_index += 1
            records.append(
                {
                    **_arxiv_base_record(doc_id, None, src),
                    "captionless_index": capless_index,
                    "alt_raw": alt if (alt is not None and alt.strip()) else None,
                    "standalone_img": True,
                    **captionless_fields(src, img.get("width"), img.get("height"), alt),
                }
            )
    return records


# --------------------------------------------------------------------------
# PMC (JATS XML) extraction
# --------------------------------------------------------------------------


def extract_pmc(
    json_path: Path, include_captionless: bool = False
) -> tuple[list[dict], dict | None]:
    """Return (records, flag) where flag is set if the doc is license-excluded."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    xml = data.get("raw_source_xml") or ""
    pmcid = data.get("pmcid") or data.get("variant_id") or json_path.stem
    if not xml:
        return [], None
    soup = BeautifulSoup(xml, "lxml-xml")

    lic_tag, commercial_ok = pmc_license(soup)
    if not commercial_ok:
        # Flag and exclude — do not emit records for non-commercial-OK docs.
        # The license veto applies to captionless candidates identically.
        return [], {"doc_id": pmcid, "source": "pmc", "license": lic_tag}

    records: list[dict] = []
    fig_index = 0
    for fig in soup.find_all("fig"):
        cap = fig.find("caption")
        # JATS caption may hold a <title> + <p>(s). Join in document order.
        caption_text = _norm_ws(cap.get_text(" ")) if cap is not None else ""
        graphic = fig.find("graphic")
        if graphic is None:
            continue
        href = (graphic.get("xlink:href") or graphic.get("href") or "").strip()
        if not href:
            continue
        # JATS hrefs are extension-less; no logo/data-URI concern here.
        fig_index += 1
        rec = {
            "source": "pmc",
            "doc_id": pmcid,
            "figure_index": fig_index,
            "image_ref": href,
            "image_format_hint": _image_format_hint(href),
            "figcaption": caption_text,
            "alt_raw": None,  # JATS <fig> carries no alt; caption is the GT
            "surrounding_context": "",
            "license": lic_tag,
            "image_local": False,
        }
        if include_captionless and not caption_text:
            # Captionless <fig>: historically emitted with figcaption "";
            # in captionless mode mark it as a decorative candidate
            # (figcaption: null). JATS graphics carry no width/height/alt.
            rec.update(captionless_fields(href))
        records.append(rec)
    return records, None


# --------------------------------------------------------------------------
# OpenStax (rendered HTML) extraction — optional, CC-BY
# --------------------------------------------------------------------------


def extract_openstax(json_path: Path, include_captionless: bool = False) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    html = data.get("output_html") or ""
    if "<figure" not in html and not (include_captionless and "<img" in html):
        return []
    doc_id = data.get("variant_id") or json_path.stem
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []
    fig_index = 0
    capless_index = 0  # net-new captionless records get their own ordinal

    def _base(src: str) -> dict:
        return {
            "source": "openstax",
            "doc_id": doc_id,
            "figure_index": fig_index,
            "image_ref": src,
            "image_format_hint": _image_format_hint(src),
            "surrounding_context": "",
            "license": "CC-BY",
            "image_local": False,
        }

    for fig in soup.find_all("figure"):
        figcap = fig.find("figcaption")
        caption_text = _norm_ws(figcap.get_text(" ")) if figcap is not None else ""
        # OpenStax prepends a "Figure N" label inside <figcaption>; strip it.
        caption_text = _FIG_LABEL_RE.sub("", caption_text).strip()
        img = fig.find("img")
        if img is None:
            continue
        src = (img.get("src") or "").strip()
        if not src or src.startswith("data:"):
            continue
        alt = img.get("alt")
        alt_norm = alt if (alt is not None and alt.strip()) else None

        if figcap is None:
            # No <figcaption> at all: dropped by default, cataloged as a
            # decorative candidate behind the flag. NET-NEW record: does NOT
            # consume the captioned figure_index (kept identical to default
            # mode); carries its own captionless_index instead.
            if not include_captionless:
                continue
            capless_index += 1
            records.append(
                {
                    **_base(src),
                    "figure_index": None,
                    "captionless_index": capless_index,
                    "alt_raw": alt_norm,
                    **captionless_fields(src, img.get("width"), img.get("height"), alt),
                }
            )
            continue

        fig_index += 1
        rec = {
            **_base(src),
            "figcaption": caption_text,
            "alt_raw": alt_norm,
        }
        if include_captionless and not caption_text:
            # Label-only caption ("Figure N" strips to empty): historically
            # emitted with figcaption ""; in captionless mode mark it as a
            # decorative candidate (figcaption: null).
            rec.update(captionless_fields(src, img.get("width"), img.get("height"), alt))
        records.append(rec)

    if include_captionless:
        # Standalone <img> outside any <figure> (icons, inline graphics).
        for img in soup.find_all("img"):
            if img.find_parent("figure") is not None:
                continue
            src = (img.get("src") or "").strip()
            if not src or src.startswith("data:"):
                continue
            alt = img.get("alt")
            capless_index += 1
            records.append(
                {
                    **_base(src),
                    "figure_index": None,
                    "captionless_index": capless_index,
                    "alt_raw": alt if (alt is not None and alt.strip()) else None,
                    "standalone_img": True,
                    **captionless_fields(src, img.get("width"), img.get("height"), alt),
                }
            )
    return records


# --------------------------------------------------------------------------
# Coverage report
# --------------------------------------------------------------------------


def _percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def build_report(records: list[dict], flagged: list[dict]) -> dict:
    from collections import Counter

    per_source = Counter(r["source"] for r in records)
    per_license = Counter(r["license"] for r in records)
    # figcaption is None for captionless (decorative-candidate) records.
    nonempty = [r for r in records if (r["figcaption"] or "").strip()]
    captionless = [r for r in records if r.get("captionless")]

    char_lens = sorted(len(r["figcaption"]) for r in nonempty)
    tok_lens = sorted(len(r["figcaption"].split()) for r in nonempty)

    def buckets(vals: list[int], edges: list[int]) -> dict:
        out = {}
        labels = []
        prev = 0
        for e in edges:
            labels.append(f"{prev}-{e}")
            prev = e + 1
        labels.append(f"{edges[-1] + 1}+")
        for lab in labels:
            out[lab] = 0
        for v in vals:
            placed = False
            prev = 0
            for i, e in enumerate(edges):
                if v <= e:
                    out[labels[i]] += 1
                    placed = True
                    break
                prev = e + 1
            if not placed:
                out[labels[-1]] += 1
        return out

    needs_fetch = sum(1 for r in records if not r["image_local"])
    local = sum(1 for r in records if r["image_local"])

    return {
        "total_figures": len(records),
        "per_source": dict(per_source),
        "per_license": dict(per_license),
        "figcaption_nonempty": len(nonempty),
        "figcaption_empty": len(records) - len(nonempty),
        "figcaption_char_length": {
            "p50": _percentile(char_lens, 50),
            "p90": _percentile(char_lens, 90),
            "p99": _percentile(char_lens, 99),
            "min": char_lens[0] if char_lens else 0,
            "max": char_lens[-1] if char_lens else 0,
            "buckets": buckets(char_lens, [50, 100, 200, 400, 800, 1600]),
        },
        "figcaption_token_length": {
            "p50": _percentile(tok_lens, 50),
            "p90": _percentile(tok_lens, 90),
            "p99": _percentile(tok_lens, 99),
            "buckets": buckets(tok_lens, [10, 25, 50, 100, 200]),
        },
        "needs_image_fetch": needs_fetch,
        "image_local": local,
        "license_excluded_docs": flagged,
        "license_excluded_count": len(flagged),
        "captionless": {
            "total": len(captionless),
            "per_source": dict(Counter(r["source"] for r in captionless)),
            "per_class": dict(Counter(r["decorative_class"] for r in captionless)),
            "per_signal": dict(Counter(s for r in captionless for s in r["decorative_signals"])),
            "standalone_img": sum(1 for r in captionless if r.get("standalone_img")),
        },
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/figure_catalog"))
    ap.add_argument("--no-openstax", action="store_true", help="Skip OpenStax source.")
    ap.add_argument(
        "--include-captionless",
        action="store_true",
        help="ALSO catalog captionless figures / standalone images as "
        'decorative candidates (figcaption: null, "decorative_candidate": '
        "true, heuristic decorative_class). Default OFF: catalog unchanged.",
    )
    args = ap.parse_args()

    root: Path = args.root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    flagged: list[dict] = []

    # arXiv ----------------------------------------------------------------
    arxiv_dir = root / "ar5iv_html_cache"
    arxiv_files = sorted(arxiv_dir.glob("*.html")) if arxiv_dir.exists() else []
    print(f"[arxiv] scanning {len(arxiv_files)} html files", file=sys.stderr)
    for fp in arxiv_files:
        records.extend(extract_arxiv(fp, include_captionless=args.include_captionless))

    # PMC ------------------------------------------------------------------
    pmc_dir = root / "pairs" / "pmc"
    pmc_files = sorted(pmc_dir.glob("*.json")) if pmc_dir.exists() else []
    print(f"[pmc] scanning {len(pmc_files)} json files", file=sys.stderr)
    for fp in pmc_files:
        recs, flag = extract_pmc(fp, include_captionless=args.include_captionless)
        records.extend(recs)
        if flag:
            flagged.append(flag)

    # OpenStax (optional) --------------------------------------------------
    if not args.no_openstax:
        os_dir = root / "pairs" / "openstax"
        os_files = sorted(os_dir.glob("*.json")) if os_dir.exists() else []
        print(f"[openstax] scanning {len(os_files)} json files", file=sys.stderr)
        for fp in os_files:
            records.extend(extract_openstax(fp, include_captionless=args.include_captionless))

    # Write catalog --------------------------------------------------------
    catalog_path = out_dir / "catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = build_report(records, flagged)
    report_path = out_dir / "coverage_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {len(records)} figure records -> {catalog_path}", file=sys.stderr)
    print(f"[done] coverage report -> {report_path}", file=sys.stderr)
    print(
        json.dumps(report["per_source"]) + " | excluded=" + str(report["license_excluded_count"]),
        file=sys.stderr,
    )
    if args.include_captionless:
        print("[captionless] " + json.dumps(report["captionless"]), file=sys.stderr)


if __name__ == "__main__":
    main()
