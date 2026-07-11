#!/usr/bin/env python3
"""Re-render a SemantiK ``{stem}_accessible.html`` through the CURRENT adapter
WITHOUT re-running the (model-heavy) cascade.

Motivation
----------
The SemantiK adapter (``lib/semantik/adapter.py``) + IR builder
(``lib/semantik/cascade_ir.py``) are pure-Python and deterministic. When a bug
in that seam is fixed (e.g. the overflow-continuation heading defect, or the
document-title selection defect), every already-converted chapter can be
regenerated from persisted artifacts — no GPU, no models, no re-extraction.

Two input modes
---------------
1. ``--ir <cascade_ir.json>`` (LOSSLESS, forward path): the
   ``{stem}_accessible.cascade_ir.json`` sidecar the conversion seam now
   persists (region_provenance + heading_tree + doc-level signals — exactly the
   bridge shape ``from_bridge_json`` consumes). Chapters are rebuilt via
   ``build_chapters_ir`` and rendered via ``normalize_cascade_to_ed4all``, so
   the output is byte-identical to a fresh conversion under the current code.

2. ``--from-html <accessible.html>`` (RECONSTRUCTION, back-compat): reconstruct
   the adapter chapter/block IR from a previously-emitted ``_accessible.html``
   (+ optional ``_synthesized.json`` for raw_text fidelity) and re-render it.
   This unlocks chapters converted BEFORE the cascade-IR sidecar existed. The
   reconstruction is faithful for HTML re-render (block html is carried
   verbatim; sids re-mint identically) but LOSSY for a few sidecar-only fields
   (region_kind is approximated from ``data-dart-block-role``; figure alt / cell
   grids are not recovered) — use ``--ir`` whenever the sidecar is present.

Arg-driven + generic: no course paths are hardcoded. ``--output`` is REQUIRED
so the script never writes next to (or over) a live conversion output by
default.

Example
-------
    python scripts/semantik_rerender.py \
        --from-html DART/output/foo-ch09_accessible.html \
        --synthesized DART/output/foo-ch09_accessible_synthesized.json \
        --output /tmp/rerender/foo-ch09_accessible.html --diff

    python scripts/semantik_rerender.py \
        --ir DART/output/foo-ch09_accessible.cascade_ir.json \
        --output /tmp/rerender/foo-ch09_accessible.html --diff
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root on sys.path so ``lib.semantik`` imports resolve when the script is
# invoked directly (``python scripts/semantik_rerender.py``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.semantik.adapter import (  # noqa: E402
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)
from lib.semantik.cascade_ir import build_chapters_ir  # noqa: E402


# ---------------------------------------------------------------------------
# Small duck-typed cascade-result stand-in (the adapter reads these attrs).
# ---------------------------------------------------------------------------
class _RerenderResult:
    """Duck-typed ``PipelineV2Result`` carrying just what the adapter reads."""

    def __init__(self, *, chapters: List[_AdapterChapter], **signals: Any):
        self.chapters = chapters
        self.exit_action = signals.get("exit_action") or "ship_with_confidence"
        self.wcag_status = signals.get("wcag_status") or "passed"
        self.theta_score = signals.get("theta_score")
        self.flags = list(signals.get("flags") or [])
        self.lane_used = signals.get("lane_used")
        self.lang = signals.get("lang") or "en"


def _parse_pages(value: Optional[str]) -> List[int]:
    """Parse a ``data-dart-pages`` value ("3" / "3-5" / "3,5,7") to ints."""
    if not value:
        return []
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        elif part.isdigit():
            out.append(int(part))
    return out


_CH_TOKEN_RE = re.compile(r"(ch\d{1,3})", re.IGNORECASE)


def _resolve_title_override(
    pdf_stem: str,
    explicit_title: Optional[str],
    title_map_path: Optional[str],
) -> Optional[str]:
    """Resolve the document-title override for a re-render (fix d).

    ``--title`` wins outright. Otherwise ``--title-map`` (a ``{stem_or_chNN:
    title}`` JSON) is consulted by (1) the exact ``pdf_stem`` and (2) the
    ``chNN`` token extracted from the stem (case-insensitive). Returns ``None``
    when nothing matches (the adapter keeps its title heuristic).
    """
    if explicit_title and explicit_title.strip():
        return explicit_title
    if not title_map_path:
        return None
    mapping = json.loads(Path(title_map_path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        return None
    if pdf_stem in mapping:
        return mapping[pdf_stem]
    # Case-insensitive stem match, then the chNN token.
    lower = {str(k).lower(): v for k, v in mapping.items()}
    if pdf_stem.lower() in lower:
        return lower[pdf_stem.lower()]
    m = _CH_TOKEN_RE.search(pdf_stem)
    if m and m.group(1).lower() in lower:
        return lower[m.group(1).lower()]
    return None


# ---------------------------------------------------------------------------
# Mode 1 — lossless re-render from the persisted cascade-IR sidecar.
# ---------------------------------------------------------------------------
def _chapters_from_ir(ir_doc: Dict[str, Any]) -> _RerenderResult:
    """Build chapters via ``build_chapters_ir`` on the persisted bridge IR."""
    chapters = build_chapters_ir(ir_doc)
    return _RerenderResult(
        chapters=chapters,
        exit_action=ir_doc.get("exit_action"),
        wcag_status=ir_doc.get("wcag_status"),
        theta_score=ir_doc.get("theta_score"),
        flags=ir_doc.get("flags"),
        lane_used=ir_doc.get("lane_used"),
        lang=ir_doc.get("lang"),
    )


# ---------------------------------------------------------------------------
# Mode 2 — reconstruct the adapter IR from a previously-emitted HTML file.
# ---------------------------------------------------------------------------
_SID_POSITIONAL_RE = re.compile(r"^s(\d+)$")
_CONT_SUFFIX_RE = re.compile(r"(?:\s*\(cont\.\))+\s*$")


def _chapters_from_html(
    html: str, synth: Optional[Dict[str, Any]] = None
) -> _RerenderResult:
    """Reconstruct the adapter chapter/block IR from emitted accessible HTML.

    Each ``<article role="doc-chapter">`` → one chapter (continuation detected
    from a ``dart-continuation`` presentation div OR a trailing "(cont.)" in the
    ``<h2>`` — so a re-render of the pre-fix buggy HTML reproduces the fixed
    output). Each ``<section class="dart-section">`` → one block; the sr-only
    label is stripped, the remaining inner HTML is carried verbatim.
    """
    from bs4 import BeautifulSoup, Tag

    raw_text_by_sid: Dict[str, str] = {}
    # SEMANTIK_OCR_CONFUSABLE_REPAIR — the sidecar carries the ORIGINAL (pre-
    # repair) text under ``data.repair.original_text`` when a block was repaired;
    # preferring it keeps the content-hash sid basis (raw_text) verbatim across a
    # re-render (the repaired text stays the rendered body + sidecar ``text``).
    orig_text_by_sid: Dict[str, str] = {}
    if synth:
        for sec in synth.get("sections", []):
            sid = sec.get("section_id")
            if sid:
                data = sec.get("data", {}) or {}
                raw_text_by_sid[str(sid)] = data.get("text", "") or ""
                repair = data.get("repair")
                if isinstance(repair, dict) and repair.get("original_text"):
                    orig_text_by_sid[str(sid)] = str(repair["original_text"])

    soup = BeautifulSoup(html, "html.parser")
    chapters: List[_AdapterChapter] = []
    for article in soup.find_all("article", attrs={"role": "doc-chapter"}):
        h2 = article.find("h2", recursive=False)
        cont_div = article.find(
            "div", attrs={"class": "dart-continuation"}, recursive=False
        )
        if h2 is not None:
            title = h2.get_text(strip=True)
        elif cont_div is not None:
            title = cont_div.get_text(strip=True)
        else:
            title = ""
        continuation = cont_div is not None or bool(
            _CONT_SUFFIX_RE.search(title)
        )
        blocks: List[_AdapterBlock] = []
        # ``recursive=True`` so sections wrapped in a ``dart-callout-group``
        # boxing div (opener + its following content) are still discovered in
        # document order; sections are never nested in sections, so no double
        # count. (Direct-child-only would miss every grouped section.)
        for section in article.find_all(
            "section", attrs={"class": "dart-section"}, recursive=True
        ):
            # DART->semantik purge Stage 3 (dual-READ): the parsed HTML may be
            # freshly-emitted ``data-semantik-*`` OR a legacy ``data-dart-*``
            # corpus, so every attribute read falls back to the dart spelling.
            sid = (
                section.get("data-semantik-block-id")
                or section.get("data-dart-block-id")
                or ""
            )
            block_role = section.get("data-semantik-block-role") or section.get(
                "data-dart-block-role"
            )
            pages = _parse_pages(
                section.get("data-semantik-pages")
                or section.get("data-dart-pages")
            )
            conf = section.get("data-semantik-confidence") or section.get(
                "data-dart-confidence"
            )
            try:
                confidence = float(conf) if conf is not None else None
            except ValueError:
                confidence = None
            wcag_status = section.get("data-semantik-wcag") or section.get(
                "data-dart-wcag"
            )
            # SEMANTIK_OCR_CONFUSABLE_REPAIR — preserve the repair annotation on
            # re-render so a --from-html round-trip does not silently drop the
            # data-semantik-repair attribute (the body html is already the
            # repaired bytes; we only need to re-stamp the marker + count).
            repair_marker = section.get("data-semantik-repair") or section.get(
                "data-dart-repair"
            )
            repair_count_raw = section.get(
                "data-semantik-repair-count"
            ) or section.get("data-dart-repair-count")
            try:
                repair_count = int(repair_count_raw) if repair_count_raw else 0
            except (TypeError, ValueError):
                repair_count = 0

            h3 = section.find("h3", recursive=False)
            if h3 is not None:
                # Genuine heading block → visible <h3>; no inner body html.
                heading_text = h3.get_text(strip=True)
                region_kind = "heading"
                inner_html = ""
                raw_text = raw_text_by_sid.get(sid, heading_text)
            else:
                heading_text = None
                region_kind = block_role or "paragraph"
                # Everything except the sr-only aria label is the block body.
                parts: List[str] = []
                for child in section.children:
                    if isinstance(child, Tag):
                        classes = child.get("class") or []
                        if child.name == "p" and "sr-only" in classes:
                            continue
                        parts.append(str(child))
                    else:
                        text = str(child).strip()
                        if text:
                            parts.append(text)
                inner_html = "".join(parts).strip()
                raw_text = raw_text_by_sid.get(sid, "")

            # A repaired block's sid basis is the ORIGINAL text (kept verbatim);
            # fall back to the sidecar text when the original was not carried.
            has_repair = bool(repair_marker) and repair_count > 0
            if has_repair and sid in orig_text_by_sid:
                raw_text = orig_text_by_sid[sid]
            # ``repaired_text`` only needs to be non-None to re-stamp the marker
            # (the repaired body already lives in ``inner_html``).
            repaired_text = (
                raw_text_by_sid.get(sid) if has_repair else None
            )

            m = _SID_POSITIONAL_RE.match(sid)
            raw_block_index = int(m.group(1)) if m else 0
            blocks.append(
                _AdapterBlock(
                    html=inner_html,
                    region_kind=region_kind,
                    raw_block_index=raw_block_index,
                    raw_text=raw_text,
                    heading_text=heading_text,
                    pages=pages,
                    confidence=confidence,
                    block_role=block_role,
                    wcag_status=wcag_status,
                    repaired_text=repaired_text,
                    ocr_repair_count=repair_count if has_repair else 0,
                )
            )
        chapters.append(
            _AdapterChapter(
                title=title, blocks=blocks, continuation=continuation
            )
        )
    return _RerenderResult(chapters=chapters)


# ---------------------------------------------------------------------------
# Diff reporting.
# ---------------------------------------------------------------------------
_H1_RE = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)
_H2_RE = re.compile(r"<h2>(.*?)</h2>", re.DOTALL)


def _heading_summary(html: str) -> Dict[str, Any]:
    return {
        "bytes": len(html.encode("utf-8")),
        "h1": _H1_RE.findall(html),
        "h2_count": len(_H2_RE.findall(html)),
        "h2_unique": sorted(set(_H2_RE.findall(html))),
    }


def _print_diff(old_html: str, new_html: str) -> None:
    old = _heading_summary(old_html)
    new = _heading_summary(new_html)
    print("--- heading-stream diff (old -> re-rendered) ---")
    print(f"bytes:    {old['bytes']} -> {new['bytes']}")
    print(f"h1:       {old['h1']} -> {new['h1']}")
    print(f"h2 count: {old['h2_count']} -> {new['h2_count']}")
    print(
        f"h2 unique count: {len(old['h2_unique'])} -> {len(new['h2_unique'])}"
    )
    removed = [t for t in old["h2_unique"] if t not in set(new["h2_unique"])]
    print(f"h2 titles removed from stream ({len(removed)}):")
    for t in removed[:60]:
        print(f"    - {t}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ir",
        help="Path to a persisted {stem}_accessible.cascade_ir.json (lossless).",
    )
    src.add_argument(
        "--from-html",
        dest="from_html",
        help="Path to a previously-emitted {stem}_accessible.html (reconstruct).",
    )
    ap.add_argument(
        "--synthesized",
        help="Optional {stem}_synthesized.json (raw_text fidelity for --from-html).",
    )
    ap.add_argument(
        "--input-dir",
        help="Optional dir + --stem to resolve artifacts by convention.",
    )
    ap.add_argument("--stem", help="Artifact stem (with --input-dir).")
    ap.add_argument(
        "--output", required=True, help="Destination HTML path (required)."
    )
    ap.add_argument(
        "--pdf-stem",
        help="pdf_stem for slug/sid minting (default: derived from output/input).",
    )
    ap.add_argument(
        "--title",
        help=(
            "Explicit document <h1>/<title> override (bypasses the OCR "
            "running-header title heuristic). Wins over --title-map."
        ),
    )
    ap.add_argument(
        "--title-map",
        dest="title_map",
        help=(
            'JSON file mapping {stem_or_chNN: title} for batch runs — the '
            "pdf_stem (or its chNN token) selects the title override."
        ),
    )
    ap.add_argument("--diff", action="store_true", help="Print a heading-diff.")
    args = ap.parse_args(argv)

    # Resolve inputs (optionally by --input-dir + --stem convention).
    ir_path = Path(args.ir) if args.ir else None
    html_path = Path(args.from_html) if args.from_html else None
    synth_path = Path(args.synthesized) if args.synthesized else None
    if args.input_dir and args.stem:
        d = Path(args.input_dir)
        if ir_path is None and html_path is None:
            cand_ir = d / f"{args.stem}_accessible.cascade_ir.json"
            cand_html = d / f"{args.stem}_accessible.html"
            if cand_ir.is_file():
                ir_path = cand_ir
            elif cand_html.is_file():
                html_path = cand_html
        if synth_path is None:
            cand = d / f"{args.stem}_accessible_synthesized.json"
            if cand.is_file():
                synth_path = cand

    old_html = ""
    if ir_path is not None:
        ir_doc = json.loads(ir_path.read_text(encoding="utf-8"))
        result = _chapters_from_ir(ir_doc)
        pdf_stem = args.pdf_stem or ir_doc.get("pdf") or ir_path.stem
        # For the diff, compare against the sibling emitted HTML if present.
        sib = ir_path.parent / (
            ir_path.name.replace(".cascade_ir.json", ".html")
        )
        if sib.is_file():
            old_html = sib.read_text(encoding="utf-8")
    else:
        old_html = html_path.read_text(encoding="utf-8")
        synth = None
        if synth_path and synth_path.is_file():
            synth = json.loads(synth_path.read_text(encoding="utf-8"))
        result = _chapters_from_html(old_html, synth)
        pdf_stem = args.pdf_stem or html_path.stem.replace("_accessible", "")

    title_override = _resolve_title_override(
        pdf_stem, args.title, args.title_map
    )
    adapter_out = normalize_cascade_to_ed4all(
        result, pdf_stem=pdf_stem, title_override=title_override
    )
    new_html = adapter_out["html"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(new_html, encoding="utf-8")
    print(f"wrote {out_path} ({len(new_html.encode('utf-8'))} bytes)")

    if args.diff and old_html:
        _print_diff(old_html, new_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
