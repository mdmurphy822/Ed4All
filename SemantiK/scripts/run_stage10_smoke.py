"""Stage 10 smoke harness — drives the assembler output through the
document-level HARD gate.

Walks one PDF through every stage Stage 10 depends on:

    Stage 1+2 (extract + featurize)
    Stage 3   (council)
    Stage 4   (cross-BERT reranker)
    Stage 5   (structure_graph)
    Stage 6   (Qwen specialists, mock runtime, K candidates per region)
    Stage 7   (per-region HARD gate)
    Stage 8   (per-region SOFT reranker)
    Stage 9   (assembler)
    Stage 10  (this stage — document-level HARD gate)

Usage::

    .venv/bin/python scripts/run_stage10_smoke.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --max-regions-per-kind 3 \\
        --save-json /tmp/stage10_smoke.json
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

from dart_semantic.assembler import AssemblerConfig, assemble_document  # noqa: E402
from dart_semantic.council.cross_reranker import arbitrate  # noqa: E402
from dart_semantic.council.orchestrator import run_council  # noqa: E402
from dart_semantic.gates import (  # noqa: E402
    gate_document,
    gate_per_region,
    rerank_per_region,
)
from dart_semantic.qwen_specialists.runner import run_qwen_specialists  # noqa: E402
from dart_semantic.structure_graph import Region, build_structure_graph  # noqa: E402
from dart_semantic.validate import HtmlValidator  # noqa: E402


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

    print(f"[smoke10] running council on {pdf_path}", flush=True)
    state, council_regions, feature_blocks = run_council(pdf_path)
    decisions = arbitrate(state, council_regions)
    structure_regions = build_structure_graph(
        state, feature_blocks, council_regions, decisions,
    )
    capped = _cap_regions_per_kind(
        structure_regions, cap=args.max_regions_per_kind,
    )
    print(
        f"[smoke10] capped region count by kind: "
        f"{dict(Counter(r.kind for r in capped))}",
        flush=True,
    )

    print(
        f"[smoke10] running Stage 6 (mock runtime, k={args.k}) on "
        f"{len(capped)} regions",
        flush=True,
    )
    candidates = run_qwen_specialists(
        capped, feature_blocks,
        k=args.k, runtime_mode="mock",
    )

    print("[smoke10] running Stage 7 (per-region hard gate)", flush=True)
    # Open one HtmlValidator and reuse it through Stages 7 and 10.
    with HtmlValidator() as validator:
        survivors, _ = gate_per_region(
            candidates, capped, feature_blocks, validator=validator,
        )

        print("[smoke10] running Stage 8 (per-region soft reranker)", flush=True)
        top_per_region = rerank_per_region(survivors, capped, feature_blocks)

        print("[smoke10] running Stage 9 (assembler, mock runtime)", flush=True)
        assembled = assemble_document(
            top_per_region, capped, feature_blocks,
            council_state=state,
            runtime_mode="mock",
            config=AssemblerConfig(),
        )

        print(
            "[smoke10] running Stage 10 (document-level hard gate)", flush=True,
        )
        result = gate_document(assembled, validator=validator)

    elapsed = time.perf_counter() - t0

    failures = [c for c in result.checks if not c.passed]
    metrics = [
        {
            "check": c.check.value,
            "passed": c.passed,
            "message": c.message,
        }
        for c in result.checks
    ]

    print()
    print("=" * 60)
    print("Stage 10 smoke results")
    print("=" * 60)
    print(f"html length (chars) : {len(assembled.html)}")
    print(f"gate_passed         : {result.passed}")
    print(f"checks_run          : {len(result.checks)}")
    if failures:
        print("failures:")
        for f in failures:
            print(f"  - {f.check.value}: {f.message}")
    else:
        print("failures            : none")
    print("metrics:")
    for m in metrics:
        status = "PASS" if m["passed"] else "FAIL"
        print(f"  [{status}] {m['check']}: {m['message']}")
    print(f"elapsed             : {elapsed:.2f}s")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pdf": str(pdf_path),
            "html_length": len(assembled.html),
            "gate_passed": result.passed,
            "failures": [
                {"check": f.check.value, "message": f.message}
                for f in failures
            ],
            "metrics": metrics,
            "elapsed_sec": elapsed,
        }
        args.save_json.write_text(json.dumps(payload, indent=2))
        print(f"[smoke10] dumped json to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
