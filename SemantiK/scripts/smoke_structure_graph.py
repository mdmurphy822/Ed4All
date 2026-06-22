"""End-to-end smoke for Stage 5 (structure_graph).

Walks: PDF → ``run_council`` → ``arbitrate`` → ``build_structure_graph``.
Prints a kind distribution summary, the first N regions, and (optionally)
dumps the full Region list as JSON.

Usage:
    .venv/bin/python scripts/smoke_structure_graph.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --max-regions-print 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _summarise(regions_out: list, *, max_print: int) -> dict:
    kind_counts: Counter = Counter(r.kind for r in regions_out)
    fb_counts = [len(r.feature_block_indices) for r in regions_out]
    total_fbs = sum(fb_counts)
    avg_fb = total_fbs / max(1, len(regions_out))
    max_fb = max(fb_counts) if fb_counts else 0

    print()
    print(f"total regions          : {len(regions_out)}")
    print(f"kind distribution      : {dict(kind_counts)}")
    print(f"total FBs claimed      : {total_fbs}")
    print(f"avg FB-per-region      : {avg_fb:.2f}")
    print(f"max FB-per-region      : {max_fb}")
    print()

    n = min(len(regions_out), max_print)
    print(f"first {n} regions:")
    print(f"  {'idx':>4} {'kind':<14} {'#fb':<4} preview")
    for i, r in enumerate(regions_out[:n]):
        preview = _region_preview(r)
        print(
            f"  {i:>4} {r.kind:<14} {len(r.feature_block_indices):<4} {preview}"
        )

    return {
        "kind_counts": dict(kind_counts),
        "n_regions": len(regions_out),
        "total_fbs": total_fbs,
        "avg_fb_per_region": avg_fb,
        "max_fb_per_region": max_fb,
    }


def _region_preview(r) -> str:
    payload = r.payload or {}
    if r.kind in ("paragraph", "heading", "blockquote", "code_block",
                  "metadata_drop"):
        text = (payload.get("text") or "")[:80].replace("\n", " ")
        return f"'{text}'"
    if r.kind == "list":
        items = payload.get("items") or []
        n = len(items)
        ordered = payload.get("ordered")
        first = (items[0].get("text") if items else "")[:40]
        return f"<list n_items={n} ordered={ordered} first='{first}'>"
    if r.kind == "table":
        return (
            f"<table {payload.get('n_rows')}x{payload.get('n_cols')} "
            f"bordered={payload.get('bordered')} "
            f"caption_fb={payload.get('caption_fb_index')}>"
        )
    if r.kind == "math":
        src = (payload.get("src_text") or "")[:50].replace("\n", " ")
        return f"<math '{src}'>"
    if r.kind == "figure":
        return f"<figure caption_fb={payload.get('caption_fb_index')}>"
    if r.kind == "form":
        return f"<form n_widgets={len(payload.get('widgets') or [])}>"
    return f"<{r.kind}>"


def _regions_to_jsonable(regions_out: list) -> list[dict]:
    out: list[dict] = []
    for r in regions_out:
        # Strip non-serialisable payload pieces (cell_grid stays as-is —
        # it's already nested lists of strings).
        payload = dict(r.payload or {})
        out.append({
            "kind": r.kind,
            "feature_block_indices": list(r.feature_block_indices),
            "payload": payload,
            "aria_hints": list(r.aria_hints),
            "provenance": dict(r.provenance or {}),
            "source_region_id": r.source_region_id,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-regions-print", type=int, default=30)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument(
        "--is-heading-threshold",
        type=float,
        default=None,
        help="Override Stage-5 is_heading confidence floor (default: "
             "build_structure_graph's default of 0.7; raised from 0.5 "
             "on 2026-05-06 after a 5-PDF over-firing diagnostic).",
    )
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    print("=" * 70)
    print(f"Stage 5 smoke: {args.pdf}")
    print("=" * 70)

    from dart_semantic.council.cross_reranker import arbitrate
    from dart_semantic.council.orchestrator import run_council
    from dart_semantic.structure_graph import build_structure_graph

    t0 = time.time()
    state, regions, feature_blocks = run_council(
        args.pdf, max_pages=args.max_pages,
    )
    t1 = time.time()
    print(f"[orchestrator]    wall: {t1 - t0:.2f}s")
    decisions = arbitrate(state, regions)
    t2 = time.time()
    print(f"[arbitrate]       wall: {t2 - t1:.2f}s")

    print(f"[feature_blocks]  count: {len(feature_blocks)}")

    t3 = time.time()
    build_kwargs: dict = {}
    if args.is_heading_threshold is not None:
        build_kwargs["is_heading_threshold"] = args.is_heading_threshold
    regions_out = build_structure_graph(
        state, feature_blocks, regions, decisions, **build_kwargs,
    )
    t4 = time.time()
    print(f"[structure_graph] wall: {t4 - t3:.2f}s")

    summary = _summarise(regions_out, max_print=args.max_regions_print)

    # Coverage invariant — also asserted inside build_structure_graph,
    # but reprint here for the smoke harness's reporting.
    covered = {
        idx for r in regions_out for idx in r.feature_block_indices
    }
    assert len(covered) == len(feature_blocks), (
        f"coverage gap: {len(covered)} of {len(feature_blocks)} FBs claimed"
    )
    print(f"[coverage]        OK ({len(covered)} / {len(feature_blocks)})")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pdf": str(args.pdf),
            "wall_seconds": {
                "orchestrator": t1 - t0,
                "arbitrate": t2 - t1,
                "structure_graph": t4 - t3,
            },
            "summary": summary,
            "regions": _regions_to_jsonable(regions_out),
        }
        args.save_json.write_text(json.dumps(payload, indent=2))
        print(f"[save] wrote {args.save_json}")


if __name__ == "__main__":
    main()
