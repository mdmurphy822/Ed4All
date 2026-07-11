"""Stage 11 smoke harness — drives the assembler output through the
document-level SOFT reranker.

Walks one PDF through every stage Stage 11 depends on:

    Stage 1+2 (extract + featurize)
    Stage 3   (council)
    Stage 4   (cross-BERT reranker)
    Stage 5   (structure_graph)
    Stage 6   (Qwen specialists, mock runtime, K candidates per region)
    Stage 7   (per-region HARD gate)
    Stage 8   (per-region SOFT reranker)
    Stage 9   (assembler)
    Stage 10  (document-level HARD gate)
    Stage 11  (this stage — document-level SOFT reranker)

Usage::

    .venv/bin/python scripts/run_stage11_smoke.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --max-regions-per-kind 3 \\
        --save-json /tmp/stage11_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantik_structure.assembler import AssemblerConfig, assemble_document  # noqa: E402
from semantik_structure.council.cross_reranker import arbitrate  # noqa: E402
from semantik_structure.council.orchestrator import run_council  # noqa: E402
from semantik_structure.gates import (  # noqa: E402
    gate_document,
    gate_per_region,
    rerank_per_region,
)
from semantik_structure.qwen_specialists.runner import run_qwen_specialists  # noqa: E402
from semantik_structure.soft_reranker import score_document  # noqa: E402
from semantik_structure.structure_graph import Region, build_structure_graph  # noqa: E402
from semantik_structure.validate import HtmlValidator  # noqa: E402


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
    ap.add_argument("--max-regions-per-kind", type=int, default=3)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    print(f"[smoke11] running council on {pdf_path}", flush=True)
    state, council_regions, feature_blocks = run_council(pdf_path)
    decisions = arbitrate(state, council_regions)
    structure_regions = build_structure_graph(
        state, feature_blocks, council_regions, decisions,
    )
    capped = _cap_regions_per_kind(
        structure_regions, cap=args.max_regions_per_kind,
    )
    print(
        f"[smoke11] capped region count by kind: "
        f"{dict(Counter(r.kind for r in capped))}",
        flush=True,
    )

    print(
        f"[smoke11] running Stage 6 (mock runtime, k={args.k}) on "
        f"{len(capped)} regions",
        flush=True,
    )
    candidates = run_qwen_specialists(
        capped, feature_blocks,
        k=args.k, runtime_mode="mock",
    )

    with HtmlValidator() as validator:
        print("[smoke11] running Stage 7 (per-region hard gate)", flush=True)
        survivors, _ = gate_per_region(
            candidates, capped, feature_blocks, validator=validator,
        )

        print("[smoke11] running Stage 8 (per-region soft reranker)", flush=True)
        top_per_region = rerank_per_region(survivors, capped, feature_blocks)

        print("[smoke11] running Stage 9 (assembler, mock runtime)", flush=True)
        assembled = assemble_document(
            top_per_region, capped, feature_blocks,
            council_state=state,
            runtime_mode="mock",
            config=AssemblerConfig(),
        )

        print(
            "[smoke11] running Stage 10 (document-level hard gate)", flush=True,
        )
        gate_result = gate_document(assembled, validator=validator)

    print(
        "[smoke11] running Stage 11 (document-level soft reranker)", flush=True,
    )
    scored = score_document(assembled)

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("Stage 11 smoke results")
    print("=" * 60)
    print(f"html length (chars) : {len(assembled.html)}  (tie-break value)")
    print(f"stage10 gate_passed : {gate_result.passed}")
    print(f"composite score     : {scored.score:.4f}")
    print(f"lane                : {scored.lane}")
    print("breakdown:")
    for axis, val in scored.breakdown.items():
        print(f"  {axis:24s} : {val:.4f}")
    print(f"elapsed             : {elapsed:.2f}s")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pdf": str(pdf_path),
            "html_length": len(assembled.html),
            "stage10_gate_passed": gate_result.passed,
            "composite": scored.score,
            "lane": scored.lane,
            "breakdown": scored.breakdown,
            "elapsed_sec": elapsed,
        }
        args.save_json.write_text(json.dumps(payload, indent=2))
        print(f"[smoke11] dumped json to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
