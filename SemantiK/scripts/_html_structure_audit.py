"""Semantic-structure fidelity audit for DART output HTML.

axe-core answers "is the markup VALID per WCAG rules"; it does NOT answer
"is this document correctly STRUCTURED" — a doc whose headings were all
flattened to <p> can pass axe clean. This module measures the structure
itself, so the eval reports BOTH (axe verdict + structure fidelity).

Two tiers:
  Tier 1 — intrinsic (any HTML, no ground truth): the inferred heading
    OUTLINE (eyeball vs the source), a semantic-element CENSUS, and
    structural SMELLS that axe is blind to, each tagged with the WCAG SC
    it endangers.
  Tier 2 — extrinsic (paired docs): ``diff_structure`` compares a
    predicted census against a ground-truth census (e.g. ar5iv HTML).

Note: DART's assembler normalizes heading levels (heading_tree.py:
normalize_heading_levels promotes the first heading to h1 and closes
skips), so a level-skip smell should be rare on real DART output — we
still check, because a regression there is exactly what we'd want to see.
"""

from __future__ import annotations

import sys
from collections import Counter

from bs4 import BeautifulSoup

_GENERIC_LINK_TEXT = {"click here", "here", "read more", "more", "link", "this", "details"}
_LANDMARK_TAGS = ("main", "nav", "header", "footer", "aside")


def _text(el) -> str:
    return " ".join(el.get_text(" ", strip=True).split())


def audit_structure(html: str, *, n_pages: int | None = None) -> dict:
    """Return {outline, census, smells, coverage} for a DART HTML document."""
    soup = BeautifulSoup(html or "", "lxml")

    # ---- heading outline (level, text) in document order ----
    outline: list[dict] = []
    heading_levels: list[int] = []
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        lvl = int(h.name[1])
        heading_levels.append(lvl)
        outline.append({"level": lvl, "text": _text(h)[:120]})

    # ---- element census ----
    tables = soup.find_all("table")
    tables_with_header = sum(
        1 for t in tables if t.find("th") is not None or t.find("thead") is not None
    )
    tables_with_scope = sum(1 for t in tables if t.find(attrs={"scope": True}) is not None)

    imgs = soup.find_all("img")
    img_missing_alt = sum(1 for i in imgs if not i.has_attr("alt"))
    img_empty_alt = sum(1 for i in imgs if i.has_attr("alt") and not i["alt"].strip())
    img_with_alt = sum(1 for i in imgs if i.has_attr("alt") and i["alt"].strip())
    # math rendered as a bare image (no MathML sibling) — the Siyavula trap
    math_as_img = sum(
        1
        for i in imgs
        if "math" in " ".join(i.get("class", [])).lower()
        or "equation" in (i.get("src", "") or "").lower()
    )

    links = soup.find_all("a")
    links_generic = sum(1 for a in links if _text(a).lower() in _GENERIC_LINK_TEXT)
    links_empty = sum(1 for a in links if not _text(a) and not a.find("img"))

    landmarks = {tag: len(soup.find_all(tag)) for tag in _LANDMARK_TAGS}
    landmarks["role_main"] = len(soup.find_all(attrs={"role": "main"}))
    html_tag = soup.find("html")
    lang_present = bool(html_tag and html_tag.get("lang"))

    h_counts = Counter(heading_levels)
    census = {
        "headings_total": len(outline),
        "headings_by_level": {f"h{i}": h_counts.get(i, 0) for i in range(1, 7) if h_counts.get(i)},
        "paragraphs": len(soup.find_all("p")),
        "ul": len(soup.find_all("ul")),
        "ol": len(soup.find_all("ol")),
        "li": len(soup.find_all("li")),
        "tables": len(tables),
        "tables_with_header": tables_with_header,
        "tables_with_scope": tables_with_scope,
        "figures": len(soup.find_all("figure")),
        "figcaptions": len(soup.find_all("figcaption")),
        "images": len(imgs),
        "img_with_alt": img_with_alt,
        "img_empty_alt_decorative": img_empty_alt,
        "img_missing_alt": img_missing_alt,
        "mathml": len(soup.find_all("math")),
        "math_as_image": math_as_img,
        "links": len(links),
        "landmarks": landmarks,
        "lang_attr": lang_present,
    }

    # ---- coverage ratios ----
    coverage = {
        "table_header_coverage": round(tables_with_header / len(tables), 3) if tables else None,
        "img_alt_coverage": round(img_with_alt / (img_with_alt + img_missing_alt), 3)
        if (img_with_alt + img_missing_alt)
        else None,
        "has_main_landmark": bool(landmarks["main"] or landmarks["role_main"]),
        "has_lang": lang_present,
    }

    # ---- structural smells (axe-blind / structure-correctness) ----
    smells: list[dict] = []

    def smell(code, sc, detail, count=1):
        smells.append({"code": code, "sc": sc, "detail": detail, "count": count})

    # heading-level skip (assembler should prevent; flag regressions)
    skips = [
        (heading_levels[i - 1], heading_levels[i])
        for i in range(1, len(heading_levels))
        if heading_levels[i] - heading_levels[i - 1] > 1
    ]
    if skips:
        smell("heading_level_skip", "1.3.1", f"{len(skips)} jump(s) e.g. {skips[0]}", len(skips))
    if outline and heading_levels[0] != 1:
        smell("no_h1_start", "1.3.1", f"first heading is h{heading_levels[0]}, not h1")
    if not outline:
        detail = "document has zero headings"
        if n_pages and n_pages >= 2:
            detail += f" across {n_pages} pages (likely flattened structure)"
        smell("flattened_document", "1.3.1", detail)
    # degenerate headings: text that is a single char / token / fragment is
    # almost always an OCR fragment misclassified as a heading (the classic
    # v1 failure: <h2>p</h2>, <h3>e</h3>). The council should not produce these.
    degenerate = [h for h in outline if len(h["text"].strip()) < 3 or h["text"].strip().isdigit()]
    if degenerate:
        ex = ", ".join(repr(h["text"]) for h in degenerate[:4])
        smell(
            "degenerate_headings",
            "1.3.1",
            f"{len(degenerate)} heading(s) are 1-2 char / numeric fragments (e.g. {ex})",
            len(degenerate),
        )

    # over-promoted body text: a real heading is a short label, not a sentence.
    # Text that STARTS lowercase or runs long (>9 words) is almost always a
    # paragraph/column fragment mis-classified as a heading (is_heading FP).
    def _frag(t: str) -> bool:
        t = t.strip()
        if not t or len(t) < 3 or t.isdigit():
            return False  # already covered by degenerate_headings
        return t[0].islower() or len(t.split()) > 9

    fragments = [h for h in outline if _frag(h["text"])]
    if fragments:
        ex = "; ".join(repr(h["text"][:45]) for h in fragments[:3])
        smell(
            "sentence_fragment_headings",
            "1.3.1",
            f"{len(fragments)}/{len(outline)} heading(s) look like body-text fragments, "
            f"not section labels (start lowercase or >9 words), e.g. {ex}",
            len(fragments),
        )
    if tables and tables_with_header < len(tables):
        smell(
            "tables_without_headers",
            "1.3.1",
            f"{len(tables) - tables_with_header}/{len(tables)} tables lack <th>/<thead>",
            len(tables) - tables_with_header,
        )
    if img_missing_alt:
        smell(
            "images_missing_alt",
            "1.1.1",
            f"{img_missing_alt} <img> with no alt attribute",
            img_missing_alt,
        )
    if math_as_img:
        smell(
            "math_as_image",
            "1.1.1",
            f"{math_as_img} equation(s) rendered as <img> (no MathML)",
            math_as_img,
        )
    if not coverage["has_main_landmark"]:
        smell("no_main_landmark", "1.3.1/2.4.1", "no <main> or role=main landmark")
    if not lang_present:
        smell("missing_lang", "3.1.1", "<html> has no lang attribute")
    if links_generic:
        smell(
            "generic_link_text",
            "2.4.4",
            f"{links_generic} link(s) with generic text",
            links_generic,
        )
    if links_empty:
        smell(
            "empty_link_text",
            "2.4.4",
            f"{links_empty} link(s) with no discernible text",
            links_empty,
        )

    return {"outline": outline, "census": census, "coverage": coverage, "smells": smells}


def diff_structure(pred: dict, truth: dict) -> dict:
    """Tier 2: compare predicted vs ground-truth census (both from audit_structure)."""
    pc, tc = pred["census"], truth["census"]
    keys = (
        "headings_total",
        "paragraphs",
        "ul",
        "ol",
        "li",
        "tables",
        "figures",
        "images",
        "mathml",
        "links",
    )
    deltas = {
        k: {"pred": pc.get(k), "truth": tc.get(k), "delta": (pc.get(k, 0) - tc.get(k, 0))}
        for k in keys
    }
    return {
        "deltas": deltas,
        "heading_count_abs_err": abs(pc.get("headings_total", 0) - tc.get("headings_total", 0)),
    }


def _fmt(audit: dict) -> str:
    out = ["STRUCTURE AUDIT", "  outline:"]
    for h in audit["outline"][:40]:
        out.append(f"    {'  ' * (h['level'] - 1)}h{h['level']}: {h['text'][:80]}")
    if len(audit["outline"]) > 40:
        out.append(f"    ... (+{len(audit['outline']) - 40} more headings)")
    out.append(f"  census: {audit['census']}")
    out.append(f"  coverage: {audit['coverage']}")
    out.append(f"  smells ({len(audit['smells'])}):")
    for s in audit["smells"]:
        out.append(f"    [{s['sc']}] {s['code']}: {s['detail']}")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Structure-fidelity audit of an HTML file.")
    ap.add_argument("html", help="path to an HTML file")
    ap.add_argument("--pages", type=int, default=None)
    args = ap.parse_args()
    a = audit_structure(open(args.html, encoding="utf-8").read(), n_pages=args.pages)
    print(_fmt(a))
    sys.exit(0)
