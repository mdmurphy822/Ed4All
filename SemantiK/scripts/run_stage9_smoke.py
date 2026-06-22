"""Stage 9 smoke harness — proves the assembler wires up end-to-end.

Walks one PDF through every stage Stage 9 depends on:

    Stage 1+2 (extract + featurize)
    Stage 3 (council)
    Stage 4 (cross-BERT reranker)
    Stage 5 (structure_graph)
    Stage 6 (Qwen specialists, mock runtime, K candidates per region)
    Stage 7 (per-region HARD gate)
    Stage 8 (per-region SOFT reranker)
    Stage 9 (this stage — assembler 9a + 9b + 9c)

Usage::

    .venv/bin/python scripts/run_stage9_smoke.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --k 2 \\
        --max-regions-per-kind 3 \\
        --save-html /tmp/stage9_smoke.html
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart_semantic.assembler import AssemblerConfig, assemble_document  # noqa: E402
from dart_semantic.council.cross_reranker import arbitrate  # noqa: E402
from dart_semantic.council.orchestrator import run_council  # noqa: E402
from dart_semantic.gates import gate_per_region, rerank_per_region  # noqa: E402
from dart_semantic.qwen_specialists.runner import run_qwen_specialists  # noqa: E402
from dart_semantic.structure_graph import Region, build_structure_graph  # noqa: E402


def _cap_regions_per_kind(regions: list[Region], *, cap: int) -> list[Region]:
    if cap <= 0:
        return list(regions)
    counts: Counter[str] = Counter()
    out: list[Region] = []
    for r in regions:
        if counts[r.kind] >= cap:
            continue
        out.append(r)
        counts[r.kind] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-regions-per-kind", type=int, default=3)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--save-html", type=Path, default=None)
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    print(f"[smoke9] running council on {pdf_path}", flush=True)
    state, council_regions, feature_blocks = run_council(pdf_path)

    print(
        f"[smoke9] arbitrate over {len(council_regions)} council regions",
        flush=True,
    )
    decisions = arbitrate(state, council_regions)

    print("[smoke9] build_structure_graph", flush=True)
    structure_regions = build_structure_graph(
        state, feature_blocks, council_regions, decisions,
    )
    print(
        f"[smoke9] structure_graph emitted {len(structure_regions)} regions",
        flush=True,
    )

    capped = _cap_regions_per_kind(
        structure_regions, cap=args.max_regions_per_kind,
    )
    capped_by_kind: Counter[str] = Counter(r.kind for r in capped)
    print(
        f"[smoke9] capped region count by kind: {dict(capped_by_kind)}",
        flush=True,
    )

    print(
        f"[smoke9] running Stage 6 (mock runtime, k={args.k}) on "
        f"{len(capped)} regions",
        flush=True,
    )
    candidates = run_qwen_specialists(
        capped, feature_blocks,
        k=args.k, runtime_mode="mock",
    )

    print("[smoke9] running Stage 7 (per-region hard gate)", flush=True)
    survivors, _gate_results = gate_per_region(
        candidates, capped, feature_blocks,
    )
    n_all_k_fail = sum(1 for v in survivors.values() if not v)

    print("[smoke9] running Stage 8 (per-region soft reranker)", flush=True)
    top_per_region = rerank_per_region(survivors, capped, feature_blocks)

    print(
        "[smoke9] running Stage 9 (assembler — 9a + 9b + 9c, mock runtime)",
        flush=True,
    )
    assembled = assemble_document(
        top_per_region, capped, feature_blocks,
        council_state=state,
        runtime_mode="mock",
        config=AssemblerConfig(),
    )

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("Stage 9 smoke results")
    print("=" * 60)
    print(f"total regions       : {len(capped)}")
    print(f"all-K-fail regions  : {n_all_k_fail}")
    print(f"html length (chars) : {len(assembled.html)}")
    print(f"gaps_found          : {[g.kind.value for g in assembled.gaps_found]}")
    print(f"gaps_resolved       : {[g.kind.value for g in assembled.gaps_resolved]}")
    print(f"gaps_fallback       : {[g.kind.value for g in assembled.gaps_fallback]}")
    print(f"heading_tree levels : {[lvl for lvl, _ in assembled.heading_tree]}")
    print(f"landmarks           : {assembled.landmarks}")
    print(f"sub_task_log keys   : {list(assembled.sub_task_log.keys())}")
    for k, v in assembled.sub_task_log.items():
        print(f"  - {k}: {v}")
    print(f"elapsed             : {elapsed:.2f}s")

    if args.save_html:
        args.save_html.parent.mkdir(parents=True, exist_ok=True)
        args.save_html.write_text(assembled.html, encoding="utf-8")
        print(f"[smoke9] wrote html to {args.save_html}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        out_payload: dict = {
            "pdf": str(pdf_path),
            "k": args.k,
            "max_regions_per_kind": args.max_regions_per_kind,
            "n_regions_capped": len(capped),
            "all_k_fail_count": n_all_k_fail,
            "elapsed_sec": elapsed,
            "html_length": len(assembled.html),
            "gaps_found": [g.kind.value for g in assembled.gaps_found],
            "gaps_resolved": [g.kind.value for g in assembled.gaps_resolved],
            "gaps_fallback": [g.kind.value for g in assembled.gaps_fallback],
            "heading_tree": [list(p) for p in assembled.heading_tree],
            "landmarks": assembled.landmarks,
            "sub_task_log": assembled.sub_task_log,
        }
        args.save_json.write_text(json.dumps(out_payload, indent=2))
        print(f"[smoke9] dumped json to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
