"""Stage 7 smoke harness — proves the per-region hard gate wires up.

Walks one PDF end-to-end:

    Stage 1+2 (extract + featurize) →
    Stage 3 (council)               →
    Stage 4 (cross-BERT reranker)   →
    Stage 5 (structure_graph)       →
    Stage 6 (Qwen specialists, mock runtime, K candidates per region) →
    Stage 7 (per-region HARD gate; this stage)

The mock runtime emits deterministic stub strings unrelated to the
source text. So almost every candidate **should** fail
text-preservation in this run — that's the expected v1 behavior with
the mock runtime, and the harness flags it. Real Qwen generation will
preserve text.

Usage::

    .venv/bin/python scripts/run_stage7_smoke.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --k 2 \\
        --max-regions-per-kind 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.orchestrator import run_council
from semantik_structure.gates import gate_per_region
from semantik_structure.gates.types import GateCheck
from semantik_structure.qwen_specialists.runner import run_qwen_specialists
from semantik_structure.structure_graph import Region, build_structure_graph


def _cap_regions_per_kind(
    regions: list[Region],
    *,
    cap: int,
) -> list[Region]:
    """Same helper as Stage 6 smoke — keep ≤ cap regions per kind."""
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


def _summarize_failures(results: dict[int, list]) -> dict[str, int]:
    """Bucket failures by check kind across all (region, candidate)
    pairs. Each candidate's failures are counted once per check that
    failed (a candidate that fails both text-preserve and axe
    contributes to both buckets)."""
    buckets: dict[str, int] = defaultdict(int)
    for _region_idx, gr_list in results.items():
        for gr in gr_list:
            if gr.passed:
                continue
            for c in gr.checks:
                if not c.passed:
                    buckets[c.check.value] += 1
    return dict(buckets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-regions-per-kind", type=int, default=3)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument(
        "--text-preservation-threshold", type=float, default=0.85,
        help="τ floor for text-preserve check (default 0.85).",
    )
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    print(f"[smoke7] running council on {pdf_path}", flush=True)
    state, council_regions, feature_blocks = run_council(pdf_path)

    print(
        f"[smoke7] arbitrate over {len(council_regions)} council regions",
        flush=True,
    )
    decisions = arbitrate(state, council_regions)

    print("[smoke7] build_structure_graph", flush=True)
    structure_regions = build_structure_graph(
        state, feature_blocks, council_regions, decisions,
    )
    print(
        f"[smoke7] structure_graph emitted {len(structure_regions)} regions",
        flush=True,
    )

    capped = _cap_regions_per_kind(
        structure_regions, cap=args.max_regions_per_kind,
    )
    capped_by_kind: Counter[str] = Counter(r.kind for r in capped)
    print(
        f"[smoke7] capped region count by kind: {dict(capped_by_kind)}",
        flush=True,
    )

    print(
        f"[smoke7] running Stage 6 (mock runtime, k={args.k}) on "
        f"{len(capped)} regions",
        flush=True,
    )
    candidates = run_qwen_specialists(
        capped, feature_blocks,
        k=args.k, runtime_mode="mock",
    )
    total_in = sum(len(v) for v in candidates.values())
    print(f"[smoke7] Stage 6 produced {total_in} candidates", flush=True)

    print("[smoke7] running Stage 7 (per-region hard gate)", flush=True)
    survivors, results = gate_per_region(
        candidates, capped, feature_blocks,
        text_preservation_threshold=args.text_preservation_threshold,
    )

    total_out = sum(len(v) for v in survivors.values())
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("Stage 7 smoke results")
    print("=" * 60)
    print(f"candidates in : {total_in}")
    print(f"candidates out: {total_out}")
    if total_in:
        rate = 100.0 * total_out / total_in
        print(f"survival rate : {rate:.1f}%")

    # All-K-fail regions: which kinds, which indices.
    all_fail_indices = [
        idx for idx, kept in survivors.items() if not kept
    ]
    all_fail_kinds = Counter(
        capped[idx].kind for idx in all_fail_indices
        if 0 <= idx < len(capped)
    )
    print(f"all-K-fail regions: {len(all_fail_indices)} / {len(survivors)}")
    if all_fail_kinds:
        print(f"  by kind: {dict(all_fail_kinds)}")
    if all_fail_indices:
        print(f"  indices: {all_fail_indices[:20]}"
              f"{' ...' if len(all_fail_indices) > 20 else ''}")

    # Per-check failure breakdown.
    failures = _summarize_failures(results)
    print("per-check failure counts:")
    for check_id in (
        GateCheck.HTML5_VALIDATOR,
        GateCheck.TEXT_PRESERVE,
        GateCheck.MATHML_VALIDITY,
        GateCheck.AXE_WCAG22AA,
    ):
        print(f"  {check_id.value}: {failures.get(check_id.value, 0)}")

    print()
    print(f"elapsed={elapsed:.2f}s")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "pdf": str(pdf_path),
            "k": args.k,
            "max_regions_per_kind": args.max_regions_per_kind,
            "tau": args.text_preservation_threshold,
            "n_regions_capped": len(capped),
            "candidates_in": total_in,
            "candidates_out": total_out,
            "all_k_fail_count": len(all_fail_indices),
            "all_k_fail_indices": all_fail_indices,
            "all_k_fail_by_kind": dict(all_fail_kinds),
            "failures_by_check": failures,
            "elapsed_sec": elapsed,
            "survivors": {
                str(idx): [
                    {"adapter": c.adapter.value, "request_id": c.request_id}
                    for c in cands
                ]
                for idx, cands in survivors.items()
            },
            "results": {
                str(idx): [
                    {
                        "passed": gr.passed,
                        "checks": [
                            {
                                "check": c.check.value,
                                "passed": c.passed,
                                "message": c.message,
                            }
                            for c in gr.checks
                        ],
                    }
                    for gr in gr_list
                ]
                for idx, gr_list in results.items()
            },
        }
        args.save_json.write_text(json.dumps(out_payload, indent=2))
        print(f"[smoke7] dumped json to {args.save_json}")

    # The smoke harness does NOT non-zero on all-K-fail (which is
    # expected when the mock runtime is fed). It only fails on a true
    # crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
