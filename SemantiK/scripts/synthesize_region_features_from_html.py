"""Synthesize Structure ``table_region`` training rows from HTML tables.

Hardens BERT-Structure's ``table_region`` binary head (see
``dart_semantic/council/structure.py`` —
``TABLE_REGION_LABELS = ("not_table_region", "table_region")``,
``head_table_region``) by turning the project's new HTML table corpus
(PMC JATS / OpenStax HTML5 / NCES / CFR+FedReg GPOTABLE) into labeled
rows in the **exact** Structure dataset format that
``data/build_structure_data.py`` emits and ``train_structure.py``
consumes.

Why this exists
---------------
``data/structure_dataset`` was built from
``data/pairs/{wikipedia, openstax, federal_register, gutenberg, forms,
arxiv, synthetic_blockquote_code}`` — it is **PMC/CFR-table absent**, and
Plans/06 §5.6 explicitly tracks the ``Structure`` ``table_region``
extension as a deferred item. Without table-rich rows from the new
sources, the ``table_region`` head sees almost no positive examples from
the document types DART's buyers actually submit, and the layout
side-channel never learns the geometric signature of a real rendered
table.

Pipeline (CPU-only — Playwright/Chromium render + pdfplumber extract)
--------------------------------------------------------------------
For each HTML ``<table>`` in the corpus:

  1. Wrap it in a minimal accessible HTML doc with synthetic CONTEXT
     blocks above and below (a heading + two paragraphs). These context
     blocks become genuine ``not_table_region`` negatives that share a
     page with the positive table blocks — the same train/inference
     distribution the live pipeline produces.
  2. Render to PDF with the **same** Chromium ``page.pdf()`` path the
     live pipeline uses (``HtmlValidator.render_pdf``).
  3. Run the **same** pdfplumber/feature extractor the live pipeline
     uses (``extract_shared``) — merged text blocks + pdfplumber table
     bboxes.
  4. For every merged block, compute ``in_table`` via the **same**
     ``data.build_structure_data._block_in_any_table`` and the 20-dim
     layout vector via the **same**
     ``data.build_structure_data.compute_span_layout_features``. No
     divergent reimplementation — the runtime
     (``dart_semantic/council/structure.py``) mirrors these and MUST see
     the same distribution at train and inference time.
  5. Label: ``table_region=1`` if the block sits inside a
     pdfplumber-detected table region OR its text aligns (Jaccard) to a
     known table cell; the synthetic context blocks are
     ``table_region=0``. ``structural_role`` follows the builder's
     convention (table cell text → ``paragraph``; context heading →
     ``heading``; context paragraphs → ``paragraph``).
  6. Emit rows byte-compatible with ``data/build_structure_data.py``:
     ``{text, layout, labels:{structural_role, is_heading, table_region,
     is_image_block, list_nesting}, html_tag, source, pair}``.

No silent fallbacks (repo convention)
-------------------------------------
If the emitted row schema (keys / value dtypes / layout dim) does not
match the canonical Structure row, the run raises
:class:`StructureSchemaMismatch` rather than coercing — same spirit as
the Stage-13 stub raising ``StageThirteenStubRequired``.

This is the **pilot** entry point. Bulk render of all ~9K tables is a
later step (do not run it here). Default ``--limit-per-source`` keeps the
pilot small.

Usage (pilot)::

    python scripts/synthesize_region_features_from_html.py \
        --limit-per-source 10 \
        --out-dir data/structure_region_pilot

Network usage: NONE — local pair files only; Chromium renders local HTML.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter
from html import escape
from pathlib import Path

from data.build_structure_data import (
    LAYOUT_FEATURE_DIM,
    LAYOUT_FEATURE_NAMES,
    ROLE_NAMES,
    ROLE_TO_ID,
    _block_in_any_table,
    compute_span_layout_features,
)
from data.build_table_specialist_data import parse_html_tables
from dart_semantic.classify import Role
from dart_semantic.extract_shared import extract_shared
from dart_semantic.text_utils import jaccard_overlap
from dart_semantic.validate import HtmlValidator


# ---------------------------------------------------------------------------
# Typed exceptions (no silent fallbacks)
# ---------------------------------------------------------------------------


class StructureSchemaMismatch(Exception):
    """Raised when an emitted row is not byte-compatible with the
    canonical ``data/build_structure_data.py`` Structure row. Refuse to
    write a divergent dataset rather than silently coercing it."""


class TableExtractionError(Exception):
    """Raised when a source declares a table-bearing format but no usable
    ``<table>`` markup can be recovered — surfaces the failure instead of
    emitting an all-negative (label-leaking) row set."""


# ---------------------------------------------------------------------------
# Canonical row contract — must match data/build_structure_data.py exactly
# ---------------------------------------------------------------------------

# Top-level keys + the value type each must hold. Mirrors the dict built
# in ``data.build_structure_data.process_pair``.
_REQUIRED_TOP_KEYS: dict[str, type] = {
    "text": str,
    "layout": list,
    "labels": dict,
    "html_tag": str,
    "source": str,
    "pair": str,
}
_REQUIRED_LABEL_KEYS = (
    "structural_role",
    "is_heading",
    "table_region",
    "is_image_block",
    "list_nesting",
)


def _validate_row(row: dict) -> None:
    """Raise :class:`StructureSchemaMismatch` unless ``row`` matches the
    canonical Structure row schema exactly. Called on every emitted row —
    a single bad row aborts the build (no silent coercion)."""
    extra = set(row) - set(_REQUIRED_TOP_KEYS)
    if extra:
        raise StructureSchemaMismatch(f"unexpected top-level keys: {sorted(extra)}")
    for key, typ in _REQUIRED_TOP_KEYS.items():
        if key not in row:
            raise StructureSchemaMismatch(f"missing top-level key: {key!r}")
        if not isinstance(row[key], typ):
            raise StructureSchemaMismatch(
                f"key {key!r} expected {typ.__name__}, got "
                f"{type(row[key]).__name__}"
            )
    layout = row["layout"]
    if len(layout) != LAYOUT_FEATURE_DIM:
        raise StructureSchemaMismatch(
            f"layout dim {len(layout)} != expected {LAYOUT_FEATURE_DIM} "
            f"({LAYOUT_FEATURE_NAMES})"
        )
    for v in layout:
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise StructureSchemaMismatch(f"non-finite layout value: {v!r}")
    labels = row["labels"]
    extra_lbl = set(labels) - set(_REQUIRED_LABEL_KEYS)
    if extra_lbl:
        raise StructureSchemaMismatch(f"unexpected label keys: {sorted(extra_lbl)}")
    for key in _REQUIRED_LABEL_KEYS:
        if key not in labels:
            raise StructureSchemaMismatch(f"missing label key: {key!r}")
        if not isinstance(labels[key], int) or isinstance(labels[key], bool):
            raise StructureSchemaMismatch(
                f"label {key!r} must be int, got {type(labels[key]).__name__}"
            )
    role = labels["structural_role"]
    if not (0 <= role < len(ROLE_NAMES)):
        raise StructureSchemaMismatch(f"structural_role {role} out of range")


# ---------------------------------------------------------------------------
# Table corpus — raw <table> HTML strings per source
# ---------------------------------------------------------------------------


def _raw_html_tables(html: str) -> list[str]:
    """Slice the raw ``<table>...</table>`` substrings out of an HTML
    stream (case-insensitive, nesting-naive — table-in-table is rare in
    this corpus and the cell-role builder's parser also collapses it).
    We keep the ORIGINAL markup (not a re-serialized grid) so the
    rendered PDF reflects the accessible HTML the corpus actually emits.
    """
    out: list[str] = []
    low = html.lower()
    cursor = 0
    while True:
        start = low.find("<table", cursor)
        if start < 0:
            break
        end = low.find("</table>", start)
        if end < 0:
            break
        end += len("</table>")
        out.append(html[start:end])
        cursor = end
    return out


def tables_from_pair(pair: dict) -> list[str]:
    """Return raw HTML5 ``<table>`` strings for one pair, normalizing
    JATS / GPOTABLE sources to the same accessible HTML5 the cell-role
    builder uses. Mirrors ``build_table_specialist_data.process_pair``'s
    source handling so the table set is the same one the rest of Plans/06
    operates on."""
    html = pair.get("output_html") or pair.get("raw_source_html") or ""
    raw_xml = pair.get("raw_source_xml") or ""
    if raw_xml:
        from data.jats_tables import jats_to_html5_tables

        try:
            jats = "\n".join(t for t, _cap in jats_to_html5_tables(raw_xml))
        except Exception as exc:  # noqa: BLE001
            raise TableExtractionError(f"jats normalize: {exc}") from exc
        if jats:
            html = f"{html}\n{jats}" if html else jats
    if raw_xml and "gpotable" in raw_xml.lower():
        from data.gpotable_tables import gpotable_to_html5_tables

        try:
            gpo = "\n".join(t for t, _cap in gpotable_to_html5_tables(raw_xml))
        except Exception as exc:  # noqa: BLE001
            raise TableExtractionError(f"gpotable normalize: {exc}") from exc
        if gpo:
            html = f"{html}\n{gpo}" if html else gpo

    raw = _raw_html_tables(html)
    # Keep only data tables: 2x2+ and not sparse — same gate the cell-role
    # builder applies, so layout/pseudo tables don't pollute the positives.
    kept: list[str] = []
    for table_html, parsed in zip(raw, parse_html_tables(html)):
        if parsed.n_rows < 2 or parsed.n_cols < 2:
            continue
        non_empty = total = 0
        for i in range(parsed.n_rows):
            row = parsed.grid[i] if i < len(parsed.grid) else []
            for j in range(min(parsed.n_cols, len(row))):
                total += 1
                if (row[j].get("text") or "").strip():
                    non_empty += 1
        if total == 0 or non_empty / total < 0.5:
            continue
        kept.append(table_html)
    return kept


def _table_cell_texts(table_html: str) -> set[str]:
    """Normalized set of cell text fragments for one table — used to
    align rendered PDF blocks back to table membership via Jaccard."""
    cells: set[str] = set()
    for parsed in parse_html_tables(table_html):
        for row in parsed.grid:
            for cell in row:
                t = " ".join((cell.get("text") or "").split()).strip()
                if t:
                    cells.add(t)
    return cells


# ---------------------------------------------------------------------------
# Accessible wrapper doc
# ---------------------------------------------------------------------------

# Synthetic context provides genuine ``not_table_region`` negatives that
# share the page geometry with the table positives. Distinct, recognizable
# text so we never mistake a context block for a cell during alignment.
_CONTEXT_HEADING = "Synthetic Region Context Heading"
_CONTEXT_PARA_BEFORE = (
    "This introductory paragraph precedes the data table and exists only "
    "to provide a not table region negative example for the Structure "
    "table_region head."
)
_CONTEXT_PARA_AFTER = (
    "This closing paragraph follows the data table and likewise provides a "
    "not table region negative example sharing the same rendered page "
    "geometry as the positive table blocks above."
)

_CONTEXT_BLOCKS = (
    ("h2", _CONTEXT_HEADING, "heading", 1),
    ("p", _CONTEXT_PARA_BEFORE, "paragraph", 0),
    ("p", _CONTEXT_PARA_AFTER, "paragraph", 0),
)


def wrap_table_doc(table_html: str, *, title: str) -> str:
    """Minimal accessible HTML5 doc: lang, title, heading + paragraph
    before the table, paragraph after. The table markup is kept verbatim.
    Rendered by Chromium with print_background so cell borders survive
    into the PDF for pdfplumber's table detector."""
    safe_title = escape(title or "Table")
    css = (
        "body{font-family:Georgia,serif;font-size:12pt;margin:48px;}"
        "table{border-collapse:collapse;margin:24px 0;}"
        "th,td{border:1px solid #000;padding:4px 8px;}"
        "th{background:#eee;}"
        "h2{font-size:18pt;}"
    )
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{safe_title}</title><style>{css}</style></head><body>"
        f"<h2>{escape(_CONTEXT_HEADING)}</h2>"
        f"<p>{escape(_CONTEXT_PARA_BEFORE)}</p>"
        f"{table_html}"
        f"<p>{escape(_CONTEXT_PARA_AFTER)}</p>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Render + extract + label one table
# ---------------------------------------------------------------------------


def _context_match(text: str) -> tuple[str, int] | None:
    """If a rendered block is one of our synthetic context blocks, return
    (role_name, is_heading). Else None. High Jaccard threshold so a table
    cell never accidentally claims a context label."""
    best: tuple[str, int] | None = None
    best_score = 0.55
    for _tag, ctext, role, is_h in _CONTEXT_BLOCKS:
        s = jaccard_overlap(text, ctext)
        if s > best_score:
            best_score = s
            best = (role, is_h)
    return best


def synthesize_rows_for_table(
    validator: HtmlValidator,
    table_html: str,
    *,
    source: str,
    pair_id: str,
    title: str,
    max_pos_rows: int | None = None,
) -> tuple[list[dict], dict]:
    """Render one wrapped table to PDF, extract features, emit labeled
    rows. Returns (rows, per-table stats).

    ``max_pos_rows`` caps how many ``table_region=1`` rows a single table
    contributes. Some real tables are enormous (a Federal Register
    county-by-county fee schedule renders ~9.5K cells); without a cap one
    table would swamp the head's positive distribution. Negatives (the
    synthetic context blocks) are always kept. ``None`` = no cap."""
    stats = {
        "pdf_ok": False,
        "blocks": 0,
        "rows": 0,
        "pos": 0,
        "neg": 0,
        "pos_capped": 0,
        "in_table_pdf": 0,
        "error": None,
    }
    cell_texts = _table_cell_texts(table_html)
    doc = wrap_table_doc(table_html, title=title)
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "rendered.pdf"
        try:
            validator.render_pdf(doc, pdf_path)
        except Exception as exc:  # noqa: BLE001
            stats["error"] = f"render: {exc}"
            return rows, stats
        stats["pdf_ok"] = True
        try:
            shared = extract_shared(pdf_path)
        except Exception as exc:  # noqa: BLE001
            stats["error"] = f"extract: {exc}"
            return rows, stats

    for page in shared.get("pages", []):
        merged = page.get("merged", {}).get("text_blocks", []) or []
        if not merged:
            continue
        page_w = float(page.get("width", 612.0))
        page_h = float(page.get("height", 792.0))
        sizes = [b["font_size"] for b in merged if b.get("font_size") is not None]
        page_median_fs = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
        heights = [
            b["bbox"][3] - b["bbox"][1]
            for b in merged
            if (b.get("bbox") and b["bbox"][3] > b["bbox"][1])
        ]
        page_median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
        merged_sorted = sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))

        for block in merged_sorted:
            stats["blocks"] += 1
            text = (block.get("text") or "").strip()
            if not text:
                continue

            # in_table via the SAME pdfplumber-bbox test the live builder
            # uses — this is the authoritative inference-time signal.
            in_table = _block_in_any_table(block, page)
            if in_table:
                stats["in_table_pdf"] += 1

            ctx = _context_match(text)
            if ctx is not None:
                # Synthetic context negative.
                role_name, is_heading = ctx
                table_region = 0
                html_tag = "h2" if is_heading else "p"
            else:
                # Treat as table content if pdfplumber says in_table OR the
                # block text aligns to a known cell (borderless-table
                # backstop — mirrors the HTML <table>-ancestor backstop in
                # build_structure_data.py).
                aligns_cell = any(
                    jaccard_overlap(text, c) >= 0.30 for c in cell_texts
                )
                table_region = 1 if (in_table or aligns_cell) else 0
                # Table cell text shape is labeled paragraph (the builder's
                # th/td/caption -> PARAGRAPH convention). Non-aligning,
                # non-context stray blocks (rare) also default to paragraph.
                role_name = "paragraph"
                is_heading = 0
                html_tag = "td"

            layout_vec = compute_span_layout_features(
                block,
                page_w=page_w,
                page_h=page_h,
                page_median_fs=page_median_fs,
                page_median_h=page_median_h,
                in_table=in_table,
            )
            role_enum = (
                Role.HEADING if role_name == "heading" else Role.PARAGRAPH
            )
            role_id = ROLE_TO_ID[role_enum]
            row = {
                "text": text,
                "layout": layout_vec,
                "labels": {
                    "structural_role": int(role_id),
                    "is_heading": int(is_heading),
                    "table_region": int(table_region),
                    "is_image_block": 0,
                    "list_nesting": 0,
                },
                "html_tag": html_tag,
                "source": source,
                "pair": pair_id,
            }
            if (
                table_region
                and max_pos_rows is not None
                and stats["pos"] >= max_pos_rows
            ):
                stats["pos_capped"] += 1
                continue
            _validate_row(row)
            rows.append(row)
            if table_region:
                stats["pos"] += 1
            else:
                stats["neg"] += 1

    stats["rows"] = len(rows)
    return rows, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


DEFAULT_PAIR_DIRS = [
    ("pmc", Path("data/pairs/pmc")),
    ("openstax", Path("data/pairs/openstax")),
    ("nces_digest", Path("data/pairs/nces_digest")),
    ("cfr", Path("data/pairs/cfr")),
    ("federal_register", Path("data/pairs/federal_register")),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("data/structure_region_pilot")
    )
    ap.add_argument(
        "--limit-per-source",
        type=int,
        default=10,
        help="Pilot cap on PAIR FILES scanned per source (not tables).",
    )
    ap.add_argument(
        "--max-tables-per-source",
        type=int,
        default=12,
        help="Pilot cap on tables actually rendered per source.",
    )
    ap.add_argument(
        "--max-pos-rows-per-table",
        type=int,
        default=200,
        help="Cap on table_region=1 rows from a single table so one "
        "giant table (e.g. a Federal Register county fee schedule, "
        "~9.5K cells) can't swamp the head's positive distribution. "
        "0 = no cap.",
    )
    ap.add_argument(
        "--reference-row",
        type=Path,
        default=Path("data/structure_dataset/train.jsonl"),
        help="A real Structure row to diff schema against (optional).",
    )
    args = ap.parse_args()

    # Optional: confirm our row contract matches a real dataset row's keys.
    if args.reference_row.exists():
        first = args.reference_row.read_text(encoding="utf-8").splitlines()[0]
        ref = json.loads(first)
        ref_top = set(ref) & set(_REQUIRED_TOP_KEYS) | (
            set(ref) - set(_REQUIRED_TOP_KEYS)
        )
        if set(ref) != set(_REQUIRED_TOP_KEYS):
            raise StructureSchemaMismatch(
                f"reference row top-level keys {sorted(ref)} != contract "
                f"{sorted(_REQUIRED_TOP_KEYS)}"
            )
        if set(ref["labels"]) != set(_REQUIRED_LABEL_KEYS):
            raise StructureSchemaMismatch(
                f"reference row label keys {sorted(ref['labels'])} != contract "
                f"{sorted(_REQUIRED_LABEL_KEYS)}"
            )
        if len(ref["layout"]) != LAYOUT_FEATURE_DIM:
            raise StructureSchemaMismatch(
                f"reference layout dim {len(ref['layout'])} != "
                f"{LAYOUT_FEATURE_DIM}"
            )
        print(
            f"[schema] reference row {args.reference_row.name} matches "
            f"contract ({len(_REQUIRED_TOP_KEYS)} keys, "
            f"layout dim {LAYOUT_FEATURE_DIM}) OK",
            file=sys.stderr,
        )
        _ = ref_top  # silence linter

    all_rows: list[dict] = []
    totals = {
        "pairs_scanned": 0,
        "tables_in": 0,
        "pdf_ok": 0,
        "blocks": 0,
        "rows": 0,
        "pos": 0,
        "neg": 0,
        "pos_capped": 0,
        "render_fail": 0,
    }
    per_source: dict[str, Counter] = {}

    with HtmlValidator() as validator:
        for source, pdir in DEFAULT_PAIR_DIRS:
            if not pdir.exists():
                print(f"[skip] {source}: {pdir} missing", file=sys.stderr)
                continue
            sc = per_source.setdefault(source, Counter())
            n_tables = 0
            for pair_path in sorted(pdir.glob("*.json"))[: args.limit_per_source]:
                if n_tables >= args.max_tables_per_source:
                    break
                totals["pairs_scanned"] += 1
                try:
                    pair = json.loads(pair_path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    print(f"[err] {pair_path.name}: read {exc}", file=sys.stderr)
                    continue
                try:
                    tables = tables_from_pair(pair)
                except TableExtractionError as exc:
                    print(
                        f"[err] {pair_path.name}: {exc}", file=sys.stderr
                    )
                    continue
                for ti, table_html in enumerate(tables):
                    if n_tables >= args.max_tables_per_source:
                        break
                    totals["tables_in"] += 1
                    sc["tables_in"] += 1
                    rows, stats = synthesize_rows_for_table(
                        validator,
                        table_html,
                        source=source,
                        pair_id=f"{pair_path.stem}#t{ti}",
                        title=pair.get("title") or pair_path.stem,
                        max_pos_rows=(
                            args.max_pos_rows_per_table
                            if args.max_pos_rows_per_table > 0
                            else None
                        ),
                    )
                    if stats["error"]:
                        totals["render_fail"] += 1
                        print(
                            f"[fail] {source}/{pair_path.stem}#t{ti}: "
                            f"{stats['error']}",
                            file=sys.stderr,
                        )
                        continue
                    n_tables += 1
                    totals["pdf_ok"] += 1
                    totals["blocks"] += stats["blocks"]
                    totals["rows"] += stats["rows"]
                    totals["pos"] += stats["pos"]
                    totals["neg"] += stats["neg"]
                    totals["pos_capped"] += stats["pos_capped"]
                    sc["pdf_ok"] += 1
                    sc["rows"] += stats["rows"]
                    sc["pos"] += stats["pos"]
                    sc["neg"] += stats["neg"]
                    sc["pos_capped"] += stats["pos_capped"]
                    all_rows.extend(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "region_rows.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated": "2026-05-27",
        "totals": totals,
        "per_source": {
            s: dict(c) for s, c in per_source.items()
        },
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_feature_names": LAYOUT_FEATURE_NAMES,
        "row_schema": {
            "top_keys": list(_REQUIRED_TOP_KEYS),
            "label_keys": list(_REQUIRED_LABEL_KEYS),
        },
        "out_path": str(out_path),
    }
    (args.out_dir / "pilot_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\n=== PILOT SUMMARY ===", file=sys.stderr)
    print(json.dumps(totals, indent=2), file=sys.stderr)
    for s, c in per_source.items():
        print(f"  {s:18} {dict(c)}", file=sys.stderr)
    print(f"[write] {out_path}  ({len(all_rows)} rows)", file=sys.stderr)
    print(
        f"[write] {args.out_dir / 'pilot_report.json'}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
