"""Build the multi-source Qwen-table LoRA adapter dataset (Plans/06).

Supersedes the ar5iv-only ``data/build_table_qwen_data.py``: it adds OpenStax
(HTML5) and PMC (JATS) sources so the table adapter trains on diverse source /
format / table-type, not just LaTeXML ``ltx_tabular`` idioms. All sources
normalize to the SAME accessible HTML5 ``<table>`` target the ar5iv builder
emits, so the adapter learns one target contract.

Per-source parsing:

  * **arxiv** (``raw_source_html``, ar5iv) — reuse the ar5iv extraction
    verbatim from ``build_table_qwen_data`` (equation-table filter, figure
    caption hoist, ``ltx_th_*`` role classes).
  * **openstax** (``output_html``, HTML5) — already near-target ``<table>`` with
    ``<th scope>``; a data-vs-layout filter drops presentational/layout tables.
  * **pmc** (``raw_source_xml``, JATS) — normalized to HTML5 by
    ``data/jats_tables.py`` (thead ``<td>``→``<th scope=col>``, caption hoist,
    graphic-only skip).

All sources then share the ar5iv builder's grid → role → target / request
pipeline (``_expand_grid``, ``_cell_role_from_classes``, ``_detect_header_rows``,
``_build_target_html``, ``_build_request_payload_via_live``) — those already key
on ``scope=`` and ``<th>`` so they work unchanged on the normalized HTML5.

Each row is tagged with ``source`` and a heuristic ``table_type`` (plan §1b) so
the build can report a balanced spread. Splits use
``data._splits.stable_split_for_id`` on a per-source stable doc id, so the same
document lands in the same split here as in every other ``qwen_*`` builder.

CPU-ONLY. The optional ``--max-token-len`` filter lazy-loads the trainer's HF
tokenizer (no GPU) and drops rows whose wrapped prompt+target exceeds the
adapter's ``max_len`` (mirrors the prose/math builders).

Usage::

    python -m data.build_table_qwen_multi --max-token-len 1024
    python -m data.build_table_qwen_multi --sources pmc --max-pairs 50 \\
        --out-dir /tmp/qwen_table_multi_smoke      # smoke one source
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from lxml import html as lxml_html

from dart_semantic.qwen_specialists.chat_format import wrap_for_qwen

from data._splits import stable_split_for_id

# Reuse the battle-tested ar5iv extraction core. These helpers are
# source-agnostic (they read scope= / <th> / spans off any lxml table).
from data.build_table_qwen_data import (
    _build_request_payload_via_live,
    _build_target_html,
    _cell_role_from_classes,
    _detect_header_rows,
    _expand_grid,
    _figure_caption_text,
    _is_equation_table,
    _is_pseudo_kv_table,
    _iter_tables,
    _ws_token_count,
)
from data.jats_tables import jats_to_html5_tables
from data.gpotable_tables import gpotable_to_html5_tables

DEFAULT_PAIRS_ROOT = Path("data/pairs")
DEFAULT_OUT_DIR = Path("data/qwen_table_dataset_multi")
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ALL_SOURCES = ("arxiv", "openstax", "pmc", "cfr", "federal_register", "nces_digest")
# Sources whose tables live as HTML5 <table> in `output_html` (parsed
# identically: lxml + ./caption hoist + data-vs-layout filter).
_HTML5_SOURCES = ("openstax", "nces_digest")


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
@dataclass
class SkipCounters:
    not_data_table: int = 0
    pseudo_table: int = 0
    too_small: int = 0
    all_empty_cells: int = 0
    layout_table: int = 0       # HTML source: no <th>/<thead>, presentational
    serialize_error: int = 0
    over_max_tokens: int = 0
    duplicate_table: int = 0


@dataclass
class BuildStats:
    n_pairs_seen: int = 0
    n_docs: int = 0
    n_tables_inspected: int = 0
    n_rows_emitted: int = 0
    rows_per_split: Counter = field(default_factory=Counter)
    rows_per_source: Counter = field(default_factory=Counter)
    rows_per_type: Counter = field(default_factory=Counter)
    source_type: Counter = field(default_factory=Counter)
    n_with_caption: int = 0
    n_with_scope: int = 0
    target_token_hist: Counter = field(default_factory=Counter)
    skip: SkipCounters = field(default_factory=SkipCounters)


_TOKEN_HIST_EDGES = [0, 50, 100, 200, 400, 800, 1600, 3200, 25600]


def _bucket(n: int, edges: list[int]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= n < edges[i + 1]:
            return f"[{edges[i]},{edges[i+1]})"
    return f"[{edges[-1]},inf)"


# --------------------------------------------------------------------------- #
# table_type heuristic (Plans/06 §1b)
# --------------------------------------------------------------------------- #
def classify_table_type(
    grid: list[list[dict[str, Any]]],
    header_row_indices: list[int],
    source: str,
) -> str:
    """Coarse type tag for stratified reporting / balanced sampling.

    regulatory > matrix > dense_data > scientific > simple, first match wins.
    """
    if source in ("cfr", "federal_register"):
        return "regulatory"

    n_rows = len(grid)
    n_cols = max((len(r) for r in grid), default=0)

    has_row_header = any(
        c.get("role") == "header_row"
        for row in grid for c in row if c.get("is_origin", True)
    )
    has_col_header = any(
        c.get("role") == "header_col"
        for row in grid for c in row if c.get("is_origin", True)
    )
    # matrix / cross-tab: labels on BOTH axes.
    if has_row_header and has_col_header:
        return "matrix"

    # dense data / "dataset" table: many rows, mostly numeric body.
    body_rows = [i for i in range(n_rows) if i not in set(header_row_indices)]
    numeric_cells = total = 0
    for i in body_rows:
        for c in grid[i]:
            if not c.get("is_origin", True):
                continue
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            total += 1
            if any(ch.isdigit() for ch in txt):
                numeric_cells += 1
    digit_frac = numeric_cells / total if total else 0.0
    if n_rows >= 12 and digit_frac >= 0.6:
        return "dense_data"

    # scientific results: multi-row header or spanning cells.
    has_span = any(
        c.get("is_origin", True) and (c.get("rowspan", 1) > 1 or c.get("colspan", 1) > 1)
        for row in grid for c in row
    )
    if len(header_row_indices) >= 2 or has_span:
        return "scientific"

    return "simple"


# --------------------------------------------------------------------------- #
# Shared row builder (caption passed in; works for any normalized HTML5 table)
# --------------------------------------------------------------------------- #
def build_row(
    *,
    table_elem,
    caption_text: str | None,
    source: str,
    doc_id: str,
    source_pair: str,
    table_idx: int,
    require_header: bool,
    skip: SkipCounters,
) -> dict[str, Any] | None:
    """Return a JSONL-serializable adapter row, or None to skip.

    Mirrors ``build_table_qwen_data.build_row_for_table`` but takes the caption
    as a parameter (each source locates it differently) and tags source /
    table_type. ``require_header`` enables the data-vs-layout filter for HTML
    sources (a table with no header cell is almost always layout).
    """
    if _is_equation_table(table_elem):       # arxiv ltx_eqn blocks → math adapter
        skip.not_data_table += 1
        return None

    grid = _expand_grid(table_elem)
    if not grid or not grid[0]:
        skip.too_small += 1
        return None
    n_rows = len(grid)
    n_cols = max((len(r) for r in grid), default=0)
    if n_rows < 2 or n_cols < 2:
        skip.too_small += 1
        return None

    if _is_pseudo_kv_table(grid):
        skip.pseudo_table += 1
        return None

    n_non_empty = sum(
        1 for row in grid for c in row
        if c.get("is_origin", True) and (c.get("text") or "").strip()
    )
    if n_non_empty == 0:
        skip.all_empty_cells += 1
        return None

    # Role assignment (two-pass, identical to the ar5iv builder).
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell.get("is_origin", True) and cell.get("elem") is not None:
                cell["role"] = _cell_role_from_classes(
                    cell["elem"], is_th=(cell["tag"] == "th"),
                    rowspan=cell["rowspan"], colspan=cell["colspan"],
                    row_idx=i, col_idx=j,
                )
            elif not cell.get("is_origin", True):
                cell.setdefault("role", "span")
            else:
                cell["role"] = "data"
    for row in grid:
        last_role = "data"
        for cell in row:
            if cell.get("is_origin", True):
                last_role = cell.get("role", "data")
            else:
                cell["role"] = last_role if last_role != "data" else "span"

    header_row_indices = _detect_header_rows(grid, table_elem)

    # Data-vs-layout filter for HTML sources: no header cell anywhere AND no
    # thead → presentational/layout table, drop it.
    if require_header:
        any_header = any(
            c.get("role") in ("header_col", "header_row")
            for row in grid for c in row if c.get("is_origin", True)
        )
        if not any_header and not header_row_indices:
            skip.layout_table += 1
            return None

    try:
        target_html = _build_target_html(
            grid=grid, caption_text=caption_text,
            header_row_indices=header_row_indices,
        )
    except Exception:
        skip.serialize_error += 1
        return None

    request_payload = _build_request_payload_via_live(
        grid=grid, n_rows=n_rows, n_cols=n_cols, bordered=True,
        header_row_indices=header_row_indices, caption_text=caption_text,
    )
    table_type = classify_table_type(grid, header_row_indices, source)

    has_rowspan = any(
        c.get("is_origin", True) and c.get("rowspan", 1) > 1
        for row in grid for c in row
    )
    has_colspan = any(
        c.get("is_origin", True) and c.get("colspan", 1) > 1
        for row in grid for c in row
    )
    n_th = sum(
        1 for row in grid for c in row
        if c.get("is_origin", True) and c.get("tag") == "th"
    )
    return {
        "source": source,
        "source_pair": source_pair,
        "doc_id": doc_id,
        "arxiv_id": doc_id,          # kept for back-compat with existing readers
        "table_idx": table_idx,
        "table_type": table_type,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "request_payload": request_payload,
        "target_html": target_html,
        "n_target_tokens": _ws_token_count(target_html),
        "extraction_metadata": {
            "n_th_cells": n_th,
            "has_caption": bool(caption_text),
            "has_rowspan": has_rowspan,
            "has_colspan": has_colspan,
        },
    }


# --------------------------------------------------------------------------- #
# Per-source table iterators → yield (table_elem, caption, doc_id)
# --------------------------------------------------------------------------- #
def _doc_id_for(source: str, pair: dict, pair_path: Path) -> str | None:
    if source == "arxiv":
        return pair.get("arxiv_id")
    if source == "pmc":
        return pair.get("pmcid")
    if source == "openstax":
        # book+page is a stable per-table-bearing-doc id.
        return pair.get("variant_id") or f"{pair.get('book','')}__{pair.get('page','')}"
    if source == "nces_digest":
        return pair.get("table_id") or pair.get("variant_id") or pair_path.stem
    if source == "federal_register":
        return pair.get("document_number") or pair_path.stem
    if source == "cfr":
        return pair.get("variant_id") or pair_path.stem
    return pair.get("arxiv_id") or pair_path.stem


def iter_source_tables(source: str, pair: dict):
    """Yield ``(table_elem, caption_text, require_header)`` for one pair."""
    if source == "arxiv":
        html = pair.get("raw_source_html") or ""
        for elem in _iter_tables(html):
            # ar5iv caption is hoisted from the wrapping <figure>.
            yield elem, _figure_caption_text(elem), False
        return
    if source in _HTML5_SOURCES:
        html = pair.get("output_html") or pair.get("raw_source_html") or ""
        if not html:
            return
        try:
            root = lxml_html.fromstring(html)
        except Exception:
            return
        # output_html may be a full page (tables nested) OR a single bare
        # <table> (then fromstring returns the table as root, which .//table
        # would miss) — cover both.
        tables = ([root] if root.tag == "table" else []) + root.xpath(".//table")
        for elem in tables:
            cap = None
            caps = elem.xpath("./caption")
            if caps:
                cap = " ".join("".join(caps[0].itertext()).split()) or None
            yield elem, cap, True
        return
    if source == "pmc":
        for html5, cap in jats_to_html5_tables(pair.get("raw_source_xml") or ""):
            try:
                elem = lxml_html.fromstring(html5)
            except Exception:
                continue
            # jats already emitted <caption>; don't double it via param.
            yield elem, cap, True
        return
    if source in ("cfr", "federal_register"):
        for html5, cap in gpotable_to_html5_tables(pair.get("raw_source_xml") or ""):
            try:
                elem = lxml_html.fromstring(html5)
            except Exception:
                continue
            yield elem, cap, True
        return
    raise ValueError(f"unknown source: {source}")


def iter_pairs(source: str, pairs_root: Path) -> Iterator[Path]:
    yield from sorted((pairs_root / source).glob("*.json"))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-root", type=Path, default=DEFAULT_PAIRS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--sources", nargs="+", default=list(ALL_SOURCES),
                    choices=ALL_SOURCES)
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="per-source cap on pair files (0 = all); smoke tests.")
    ap.add_argument("--max-token-len", type=int, default=0,
                    help="drop rows whose wrapped prompt+target exceeds this "
                         "(0 = no tokenizer filter). Mirrors prose/math.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--progress-every", type=int, default=400)
    args = ap.parse_args()

    tok = None
    if args.max_token_len > 0:
        print(f"[tok] loading {args.base_model} for length filter")
        tok = _load_tokenizer(args.base_model)

    stats = BuildStats()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        s: (args.out_dir / f"{s}.jsonl").open("w", encoding="utf-8")
        for s in ("train", "val", "test")
    }
    seen_docs: dict[str, str] = {}            # doc_id -> split
    seen_table_sigs: dict[str, set] = defaultdict(set)

    try:
        for source in args.sources:
            n_src_pairs = 0
            for pair_path in iter_pairs(source, args.pairs_root):
                if args.max_pairs and n_src_pairs >= args.max_pairs:
                    break
                n_src_pairs += 1
                stats.n_pairs_seen += 1
                try:
                    pair = json.loads(pair_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                doc_id = _doc_id_for(source, pair, pair_path)
                if not doc_id:
                    continue
                # Per-doc dedup: pair files are sectioned (multiple per doc).
                if doc_id in seen_docs:
                    split = seen_docs[doc_id]
                else:
                    split = stable_split_for_id(doc_id, 0.80, 0.10)
                    seen_docs[doc_id] = split
                    stats.n_docs += 1

                for table_idx, (elem, cap, require_header) in enumerate(
                    iter_source_tables(source, pair)
                ):
                    stats.n_tables_inspected += 1
                    row = build_row(
                        table_elem=elem, caption_text=cap, source=source,
                        doc_id=doc_id, source_pair=pair_path.stem,
                        table_idx=table_idx, require_header=require_header,
                        skip=stats.skip,
                    )
                    if row is None:
                        continue
                    # Within-doc content dedup.
                    sig = (row["n_rows"], row["n_cols"],
                           hash(row["request_payload"]["payload"]["prompt"]))
                    if sig in seen_table_sigs[doc_id]:
                        stats.skip.duplicate_table += 1
                        continue
                    seen_table_sigs[doc_id].add(sig)

                    # Tokenizer-aware length filter.
                    n_wrap = None
                    if tok is not None:
                        prompt = row["request_payload"]["payload"]["prompt"]
                        wrapped = wrap_for_qwen(tok, prompt, target=row["target_html"])
                        n_wrap = len(tok(wrapped, add_special_tokens=False)["input_ids"])
                        if n_wrap > args.max_token_len:
                            stats.skip.over_max_tokens += 1
                            continue
                    row["n_wrap_tokens"] = n_wrap
                    row["split"] = split

                    handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats.n_rows_emitted += 1
                    stats.rows_per_split[split] += 1
                    stats.rows_per_source[source] += 1
                    stats.rows_per_type[row["table_type"]] += 1
                    stats.source_type[f"{source}/{row['table_type']}"] += 1
                    if row["extraction_metadata"]["has_caption"]:
                        stats.n_with_caption += 1
                    if row["extraction_metadata"]["n_th_cells"] > 0:
                        stats.n_with_scope += 1
                    stats.target_token_hist[
                        _bucket(row["n_target_tokens"], _TOKEN_HIST_EDGES)
                    ] += 1

                if stats.n_pairs_seen % args.progress_every == 0:
                    print(f"[progress] {source} pairs={n_src_pairs} "
                          f"docs={stats.n_docs} rows={stats.n_rows_emitted}",
                          flush=True)
            print(f"[source-done] {source}: pairs={n_src_pairs} "
                  f"rows_so_far={stats.n_rows_emitted}", flush=True)
    finally:
        for h in handles.values():
            h.close()

    n = max(1, stats.n_rows_emitted)
    coverage = {
        "schema_version": 1,
        "sources": args.sources,
        "max_token_len": args.max_token_len,
        "n_pairs_seen": stats.n_pairs_seen,
        "n_docs": stats.n_docs,
        "n_tables_inspected": stats.n_tables_inspected,
        "n_rows_emitted": stats.n_rows_emitted,
        "rows_per_split": dict(stats.rows_per_split),
        "rows_per_source": dict(stats.rows_per_source),
        "rows_per_type": dict(stats.rows_per_type),
        "source_x_type": dict(sorted(stats.source_type.items())),
        "pct_with_caption": round(100 * stats.n_with_caption / n, 2),
        "pct_with_th_scope": round(100 * stats.n_with_scope / n, 2),
        "target_token_hist": dict(sorted(stats.target_token_hist.items())),
        "skip": {
            "not_data_table": stats.skip.not_data_table,
            "pseudo_table": stats.skip.pseudo_table,
            "too_small": stats.skip.too_small,
            "all_empty_cells": stats.skip.all_empty_cells,
            "layout_table": stats.skip.layout_table,
            "serialize_error": stats.skip.serialize_error,
            "over_max_tokens": stats.skip.over_max_tokens,
            "duplicate_table": stats.skip.duplicate_table,
        },
    }
    (args.out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))

    print()
    print(f"[done] rows={stats.n_rows_emitted} "
          f"(train={stats.rows_per_split.get('train',0)} "
          f"val={stats.rows_per_split.get('val',0)} "
          f"test={stats.rows_per_split.get('test',0)})")
    print(f"[done] by source: {dict(stats.rows_per_source)}")
    print(f"[done] by type:   {dict(stats.rows_per_type)}")
    print(f"[save] coverage -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
