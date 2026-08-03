"""Stage 12 + Stage 13 smoke harness.

Drives the assembler output through:

    Stage 1+2 (extract + featurize)
    Stage 3   (council)
    Stage 4   (cross-BERT reranker)
    Stage 5   (structure_graph)
    Stage 6   (Qwen specialists, mock runtime)
    Stage 7   (per-region HARD gate)
    Stage 8   (per-region SOFT reranker)
    Stage 9   (assembler)
    Stage 10  (document-level HARD gate)
    Stage 11  (document-level SOFT reranker)
    Stage 12  (this stage — ThetaEvaluator)
    Stage 13  (this stage — exit decider)

Usage::

    .venv/bin/python scripts/smoke/run_stage12_smoke.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --max-regions-per-kind 3 \\
        --save-json /tmp/stage12_smoke.json

The cascade body has been extracted into :func:`run_full_cascade`, which
now lives in ``semantik_structure/cascade.py``, so the corpus-level eval
driver (`scripts/eval/eval_full_cascade.py`) can call it once per PDF while
the smoke harness keeps its original CLI surface and print contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantik_structure.cascade import run_full_cascade  # noqa: E402
from semantik_structure.theta import StageThirteenStubRequired  # noqa: E402
from semantik_structure.validate import HtmlValidator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--max-regions-per-kind", type=int, default=3)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument(
        "--enable-glm-ocr-stage", action="store_true",
        help=(
            "Enable optional Stage 5b: GLM-OCR enrichment of "
            "council-confirmed table regions. Cache-only (CLI doesn't "
            "load the GLM model) — only cached bboxes get text. Pre-warm "
            "the cache via scripts/datasets/glm_ocr_prepass.py."
        ),
    )
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        # Smoke harness keeps its [smoke12] prefix verbatim.
        # The cascade lib emits [cascade] prefixes; rewrite for the
        # smoke harness so output is identical to the pre-refactor.
        print(msg.replace("[cascade]", "[smoke12]"), flush=True)

    # Theta strict-by-default banner (mirror eval driver — see
    # semantik_structure/theta/_module_state.py).
    if os.environ.get("SEMANTIK_ALLOW_THETA_STUB", "").strip() == "1":
        print(
            "[smoke12] WARNING: SEMANTIK_ALLOW_THETA_STUB=1 — semantic_preservation "
            "may fall back to 0.7 stub_v1; theta_score is NOT a real "
            "measurement when this flag is set AND the trained model is "
            "missing/broken.",
            flush=True,
        )

    if args.enable_glm_ocr_stage:
        print(
            "[smoke12] Stage 5b: GLM-OCR table enrichment ENABLED "
            "(cache-only — only cached bboxes get text; misses are "
            "silently skipped, no GPU calls).",
            flush=True,
        )

    print(f"[smoke12] running council on {pdf_path}", flush=True)
    # Real driver: open one validator and run the cascade through it.
    with HtmlValidator() as validator:
        try:
            result = run_full_cascade(
                pdf_path,
                validator=validator,
                max_regions_per_kind=args.max_regions_per_kind,
                k=args.k,
                runtime_mode="mock",
                enable_glm_ocr_stage=args.enable_glm_ocr_stage,
                # Suppress duplicate council banner (already printed above).
                log=lambda m: _log(m) if "running council" not in m else None,
            )
        except StageThirteenStubRequired as exc:
            # Stage 13 routed to an unimplemented v1 lane. Print the
            # diagnostic loud and exit non-zero — do NOT degrade silently
            # to a flagged confidence metric.
            print()
            print("=" * 60, file=sys.stderr)
            print("STAGE 13 STUB FAILURE — refusing to silently degrade.", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print(f"lane required : {exc.lane_required}", file=sys.stderr)
            print(f"reason        : {exc.reason}", file=sys.stderr)
            print(f"partial report:", file=sys.stderr)
            print(f"  wcag_status : {exc.report.wcag_status}", file=sys.stderr)
            print(f"  lane        : {exc.report.lane}", file=sys.stderr)
            print(f"  theta_score : {exc.report.theta_score}", file=sys.stderr)
            print(f"  flags       : {exc.report.flags}", file=sys.stderr)
            return 3

    elapsed = time.perf_counter() - t0

    theta = result["theta"]
    print()
    print("=" * 60)
    print("Stage 12 + 13 smoke results")
    print("=" * 60)
    print(f"html length (chars) : {result['html_length']}")
    print(f"stage10 wcag        : {result['wcag_status_under_mock']}")
    print(f"theta_score         : {theta['theta_score']}")
    print(f"lane_used           : {result['lane_used']}")
    print(f"offline_retry_fired : {result['offline_retry_fired']}")
    print(f"offline_retry_won   : {result['offline_retry_won']}")
    action_val = theta["action"]
    if hasattr(action_val, "value"):
        action_val = action_val.value
    print(f"action              : {action_val}")
    flags_val = [
        f.value if hasattr(f, "value") else f
        for f in (theta["flags"] or [])
    ]
    print(f"flags               : {flags_val}")
    print()
    print("dimension breakdown:")
    for dim_name, dim in theta["dimensions"].items():
        score = dim["score"] if isinstance(dim, dict) else dim.score
        breakdown = dim["breakdown"] if isinstance(dim, dict) else dim.breakdown
        print(f"  {dim_name:32s} : {score:.4f}  {breakdown}")
    print(f"elapsed             : {elapsed:.2f}s")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(theta)
        payload["pdf"] = str(pdf_path)
        payload["html_length"] = result["html_length"]
        payload["elapsed_sec"] = elapsed
        args.save_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[smoke12] dumped json to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
