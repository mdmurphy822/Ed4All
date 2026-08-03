"""Stage 8 smoke harness — proves the per-region soft reranker wires up.

Walks one PDF end-to-end through every stage Stage 8 depends on:

    Stage 1+2 (extract + featurize) →
    Stage 3 (council)               →
    Stage 4 (cross-BERT reranker)   →
    Stage 5 (structure_graph)       →
    Stage 6 (Qwen specialists, mock runtime, K candidates per region) →
    Stage 7 (per-region HARD gate)  →
    Stage 8 (per-region SOFT reranker; this stage)

With the upgraded MockRuntime (kind-aware HTML5 fragments instead of
``<mock-output .../>`` stubs), candidates pass html5 well-formedness
and most pass text-preservation. Stage 8 then scores survivors and
emits a top-1 RankedCandidate per region.

Usage::

    .venv/bin/python scripts/smoke/run_stage8_smoke.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --k 2 \\
        --max-regions-per-kind 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.orchestrator import run_council
from semantik_structure.gates import gate_per_region, rerank_per_region
from semantik_structure.qwen_specialists.runner import run_qwen_specialists
from semantik_structure.structure_graph import Region, build_structure_graph


def _cap_regions_per_kind(
    regions: list[Region],
    *,
    cap: int,
) -> list[Region]:
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
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-regions-per-kind", type=int, default=3)
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    print(f"[smoke8] running council on {pdf_path}", flush=True)
    state, council_regions, feature_blocks = run_council(pdf_path)

    print(
        f"[smoke8] arbitrate over {len(council_regions)} council regions",
        flush=True,
    )
    decisions = arbitrate(state, council_regions)

    print("[smoke8] build_structure_graph", flush=True)
    structure_regions = build_structure_graph(
        state, feature_blocks, council_regions, decisions,
    )
    print(
        f"[smoke8] structure_graph emitted {len(structure_regions)} regions",
        flush=True,
    )

    capped = _cap_regions_per_kind(
        structure_regions, cap=args.max_regions_per_kind,
    )
    capped_by_kind: Counter[str] = Counter(r.kind for r in capped)
    print(
        f"[smoke8] capped region count by kind: {dict(capped_by_kind)}",
        flush=True,
    )

    print(
        f"[smoke8] running Stage 6 (mock runtime, k={args.k}) on "
        f"{len(capped)} regions",
        flush=True,
    )
    candidates = run_qwen_specialists(
        capped, feature_blocks,
        k=args.k, runtime_mode="mock",
    )
    total_in = sum(len(v) for v in candidates.values())
    print(f"[smoke8] Stage 6 produced {total_in} candidates", flush=True)

    print("[smoke8] running Stage 7 (per-region hard gate)", flush=True)
    survivors, _gate_results = gate_per_region(
        candidates, capped, feature_blocks,
    )
    total_survivors = sum(len(v) for v in survivors.values())
    n_all_k_fail = sum(1 for v in survivors.values() if not v)
    print(
        f"[smoke8] Stage 7 left {total_survivors} survivors; "
        f"{n_all_k_fail} all-K-fail regions",
        flush=True,
    )

    print("[smoke8] running Stage 8 (per-region soft reranker)", flush=True)
    top_per_region = rerank_per_region(survivors, capped, feature_blocks)

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("Stage 8 smoke results")
    print("=" * 60)
    print(f"total regions       : {len(capped)}")
    print(f"candidates in       : {total_in}")
    print(f"Stage 7 survivors   : {total_survivors}")
    print(f"all-K-fail regions  : {n_all_k_fail}")

    n_top1 = sum(1 for rc in top_per_region.values() if rc is not None)
    print(f"Stage 8 top-1 count : {n_top1}")
    if n_top1:
        scores = [rc.score for rc in top_per_region.values() if rc is not None]
        print(
            f"  score min/avg/max : "
            f"{min(scores):.3f} / {sum(scores)/len(scores):.3f} / "
            f"{max(scores):.3f}",
        )

    print()
    print("per-region top-1 (kind, survivors, score, breakdown):")
    for idx in sorted(top_per_region.keys()):
        rc = top_per_region[idx]
        n_surv = len(survivors.get(idx, []))
        kind = capped[idx].kind if 0 <= idx < len(capped) else "?"
        if rc is None:
            print(f"  r{idx:>3}  {kind:<16}  surv={n_surv}  TOP1=None (all-K-fail)")
            continue
        b = rc.breakdown
        if b.get("singleton"):
            print(
                f"  r{idx:>3}  {kind:<16}  surv={n_surv}  "
                f"score={rc.score:.3f}  [singleton]"
            )
        else:
            print(
                f"  r{idx:>3}  {kind:<16}  surv={n_surv}  "
                f"score={rc.score:.3f}  "
                f"[diff={b.get('diff_from_src', 0):.2f} "
                f"head={b.get('heading_fit', 0):.2f} "
                f"tbl={b.get('table_semantics', 0):.2f} "
                f"aria={b.get('aria_restraint', 0):.2f} "
                f"nbr={b.get('neighbor_consistency', 0):.2f}]"
            )

    print()
    print(f"elapsed={elapsed:.2f}s")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "pdf": str(pdf_path),
            "k": args.k,
            "max_regions_per_kind": args.max_regions_per_kind,
            "n_regions_capped": len(capped),
            "candidates_in": total_in,
            "stage7_survivors": total_survivors,
            "all_k_fail_count": n_all_k_fail,
            "stage8_top1_count": n_top1,
            "elapsed_sec": elapsed,
            "top_per_region": {
                str(idx): (
                    None if rc is None else {
                        "kind": capped[idx].kind,
                        "score": rc.score,
                        "breakdown": rc.breakdown,
                        "candidate_text": rc.candidate.text,
                        "request_id": rc.candidate.request_id,
                    }
                )
                for idx, rc in top_per_region.items()
            },
        }
        args.save_json.write_text(json.dumps(out_payload, indent=2))
        print(f"[smoke8] dumped json to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
