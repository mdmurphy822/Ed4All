"""Stage 6 smoke harness — proves the wiring, not the model quality.

Walks one PDF end-to-end:

    Stage 1+2 (extract + featurize)  →
    Stage 3 (council)                →
    Stage 4 (cross-BERT reranker)    →
    Stage 5 (structure_graph)        →
    Stage 6 (Qwen specialists, mock runtime)

The mock runtime emits deterministic synthetic strings, so the
candidates are not real HTML — the point is to confirm:

  * the 3-pass shape (PROSE → TABLE → MATH) is hit,
  * every routed region gets exactly K candidates,
  * the dict-of-lists key space matches the input region indices.

Usage::

    .venv/bin/python scripts/run_stage6_smoke.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --k 2 \\
        --max-regions-per-kind 3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.orchestrator import run_council
from semantik_structure.qwen_specialists.routing import adapter_for
from semantik_structure.qwen_specialists.runner import run_qwen_specialists
from semantik_structure.structure_graph import Region, build_structure_graph


def _cap_regions_per_kind(
    regions: list[Region],
    *,
    cap: int,
) -> list[Region]:
    """Keep at most ``cap`` regions per kind, preserving original order
    for those we keep. Stable subset → key indices in the Stage-6 output
    match the indices in the returned list, not the original list.
    """
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


def _build_md_dump(
    regions: list[Region],
    candidates_by_idx: dict[int, list],
) -> str:
    lines: list[str] = ["# Stage 6 smoke dump", ""]
    for idx, region in enumerate(regions):
        lines.append(f"## region {idx} — kind={region.kind}")
        lines.append(f"- fb_indices: {region.feature_block_indices}")
        try:
            lines.append(f"- adapter: {adapter_for(region.kind).value}")
        except ValueError:
            lines.append(f"- adapter: <unrouted>")
        lines.append(f"- payload keys: {sorted(region.payload.keys())}")
        cands = candidates_by_idx.get(idx, [])
        lines.append(f"- candidates: {len(cands)}")
        for c_idx, c in enumerate(cands):
            lines.append(f"  - [{c_idx}] {c.text[:200]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-regions-per-kind", type=int, default=5)
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    print(f"[smoke] running council on {pdf_path}", flush=True)
    state, council_regions, feature_blocks = run_council(pdf_path)

    print(
        f"[smoke] arbitrate over {len(council_regions)} council regions",
        flush=True,
    )
    decisions = arbitrate(state, council_regions)

    print(f"[smoke] build_structure_graph", flush=True)
    structure_regions = build_structure_graph(
        state, feature_blocks, council_regions, decisions,
    )
    print(
        f"[smoke] structure_graph emitted {len(structure_regions)} regions",
        flush=True,
    )

    capped = _cap_regions_per_kind(
        structure_regions, cap=args.max_regions_per_kind,
    )
    capped_by_kind: Counter[str] = Counter(r.kind for r in capped)
    print(f"[smoke] capped region count by kind: {dict(capped_by_kind)}",
          flush=True)

    print(
        f"[smoke] running Stage 6 (mock runtime, k={args.k}) on "
        f"{len(capped)} regions",
        flush=True,
    )
    candidates = run_qwen_specialists(
        capped, feature_blocks,
        k=args.k, runtime_mode="mock",
    )

    # Per-pass distribution: bucket candidate keys by adapter.
    by_adapter: dict[str, int] = defaultdict(int)
    for idx in candidates:
        try:
            adapter = adapter_for(capped[idx].kind).value
        except ValueError:
            adapter = "unrouted"
        by_adapter[adapter] += len(candidates[idx])

    print("[smoke] candidate counts by adapter:")
    for adapter, n in sorted(by_adapter.items()):
        print(f"  {adapter}: {n}")

    total = sum(len(v) for v in candidates.values())
    print(f"[smoke] total candidates: {total} "
          f"(regions={len(candidates)}, k={args.k})")

    # Sanity assertions — these mirror the validation contract.
    assert set(candidates.keys()).issubset(set(range(len(capped)))), (
        "Stage 6 returned keys outside range(len(regions))"
    )
    assert all(len(v) == args.k for v in candidates.values()), (
        f"Stage 6 returned candidate lists != k={args.k}"
    )

    if args.save_md:
        args.save_md.parent.mkdir(parents=True, exist_ok=True)
        args.save_md.write_text(_build_md_dump(capped, candidates))
        print(f"[smoke] dumped markdown to {args.save_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
