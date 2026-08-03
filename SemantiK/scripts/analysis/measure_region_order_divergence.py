"""Stage-5 region-order divergence calibration driver (ITEM5 prong 2).

Batch-runs the cascade over a directory of PDFs (or individual PDFs) with
``SEMANTIK_REGION_ORDER`` in SHADOW (``fb``, the default), harvests the
``geom_order`` divergence audit (normalized inversion distance between the
FB-index order and the geometry-derived order, per page + doc-aggregated), and
prints a per-doc / per-page table + a divergence histogram. A ``--json`` dump
feeds the Phase-C calibration record.

Structure / reading-order is VALID under the mock runtime (production HTML
renders from ``raw_text``; the geometric order derives from Region geometry, not
Stage-6 output), so this runs ``--runtime mock`` by default — no GPU, no council
models required. Dev-only; NO flag row.

Usage:
    .venv/bin/python SemantiK/scripts/analysis/measure_region_order_divergence.py \
        <dir-or-pdf> [<dir-or-pdf> ...] \
        [--mode fb|geom|off] [--runtime mock|real] \
        [--python <interp>] [--json out.json] [--limit N]

The corpora are supplied as arguments (no course slugs / hardcoded paths in this
file). For the calibration flip evidence run three genre classes: (a) 2-column
arXiv papers, (b) multi-column regulatory PDFs, (c) a single-column textbook —
see the ITEM5 spec §6.3.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]  # SemantiK/
_RUN_CASCADE = _REPO_ROOT / "scripts" / "run_cascade_json.py"


def _discover_pdfs(paths: list[str], limit: int | None) -> list[Path]:
    pdfs: list[Path] = []
    for p in paths:
        pp = Path(p).expanduser()
        if pp.is_dir():
            pdfs.extend(sorted(pp.rglob("*.pdf")))
        elif pp.is_file() and pp.suffix.lower() == ".pdf":
            pdfs.append(pp)
        else:
            print(f"[warn] skipping non-PDF / missing path: {pp}", file=sys.stderr)
    # De-dup preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for pdf in pdfs:
        if pdf not in seen:
            seen.add(pdf)
            out.append(pdf)
    if limit is not None:
        out = out[:limit]
    return out


def _run_one(pdf: Path, *, mode: str, runtime: str, python: str) -> dict[str, Any] | None:
    """Run the cascade bridge on ONE pdf and return its ``geom_order`` audit."""
    env = dict(os.environ)
    env["SEMANTIK_REGION_ORDER"] = mode
    env.setdefault("SEMANTIK_ALLOW_THETA_STUB", "1")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_json = Path(tf.name)
    try:
        proc = subprocess.run(
            [python, str(_RUN_CASCADE), "--pdf", str(pdf.resolve()),
             "--runtime", runtime, "--out-json", str(out_json)],
            env=env, cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            print(f"[warn] {pdf.name}: cascade rc={proc.returncode}", file=sys.stderr)
        data = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — dev tool, keep going
        print(f"[warn] {pdf.name}: {exc}", file=sys.stderr)
        return None
    finally:
        out_json.unlink(missing_ok=True)
    if data.get("error"):
        print(f"[warn] {pdf.name}: {data['error']}", file=sys.stderr)
        return None
    return data.get("geom_order")


def _histogram(values: list[float], *, buckets: int = 10) -> list[tuple[str, int]]:
    counts = [0] * buckets
    for v in values:
        b = min(buckets - 1, int(v * buckets))
        counts[b] += 1
    return [
        (f"[{i / buckets:.1f},{(i + 1) / buckets:.1f})", counts[i])
        for i in range(buckets)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="PDF file(s) or directory(ies)")
    ap.add_argument("--mode", default="fb", choices=["fb", "geom", "off"],
                    help="SEMANTIK_REGION_ORDER value (default fb = shadow)")
    ap.add_argument("--runtime", default="mock", help="cascade runtime (default mock)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the bridge")
    ap.add_argument("--json", dest="json_out", default=None, help="write the full record here")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of PDFs")
    args = ap.parse_args(argv)

    pdfs = _discover_pdfs(args.paths, args.limit)
    if not pdfs:
        print("no PDFs found", file=sys.stderr)
        return 2

    docs: list[dict[str, Any]] = []
    all_page_div: list[float] = []
    for pdf in pdfs:
        audit = _run_one(pdf, mode=args.mode, runtime=args.runtime, python=args.python)
        if not audit:
            continue
        pages = audit.get("pages", []) or []
        doc_div = float(audit.get("doc_divergence", 0.0) or 0.0)
        n_pages_gt0 = sum(1 for p in pages if float(p.get("divergence", 0.0) or 0.0) > 0.0)
        n_fallback = sum(1 for p in pages if p.get("fallback"))
        all_page_div.extend(float(p.get("divergence", 0.0) or 0.0) for p in pages)
        docs.append(
            {
                "pdf": pdf.name,
                "doc_divergence": doc_div,
                "n_pages": len(pages),
                "n_pages_divergent": n_pages_gt0,
                "n_fallback_pages": n_fallback,
                "bboxless_regions": audit.get("bboxless_regions", 0),
                "warnings": audit.get("warnings", []),
                "pages": pages,
            }
        )

    # ---- report ----
    print(f"\nSEMANTIK_REGION_ORDER={args.mode}  runtime={args.runtime}  "
          f"docs={len(docs)}/{len(pdfs)}\n")
    print(f"{'doc':<40} {'doc_div':>8} {'pages':>6} {'div>0':>6} {'fallbk':>7} {'bboxless':>9}")
    print("-" * 80)
    for d in docs:
        print(f"{d['pdf'][:40]:<40} {d['doc_divergence']:>8.4f} "
              f"{d['n_pages']:>6} {d['n_pages_divergent']:>6} "
              f"{d['n_fallback_pages']:>7} {d['bboxless_regions']:>9}")

    if docs:
        corpus_doc_div = sum(d["doc_divergence"] for d in docs) / len(docs)
        max_doc_div = max(d["doc_divergence"] for d in docs)
        print("-" * 80)
        print(f"corpus mean doc_divergence: {corpus_doc_div:.4f}   "
              f"max: {max_doc_div:.4f}   pages measured: {len(all_page_div)}")
        print("\nper-page divergence histogram:")
        for label, count in _histogram(all_page_div):
            bar = "#" * min(50, count)
            print(f"  {label:<12} {count:>5}  {bar}")

    record = {
        "mode": args.mode,
        "runtime": args.runtime,
        "n_docs": len(docs),
        "n_pdfs": len(pdfs),
        "corpus_mean_doc_divergence": (
            sum(d["doc_divergence"] for d in docs) / len(docs) if docs else 0.0
        ),
        "docs": docs,
    }
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
