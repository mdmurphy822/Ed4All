"""Aggregate per-PDF result.json files written by make_pdf_vs_html.py
into a single eval/results/v7_family_<ts>.json summary.

Companion to scripts/eval_v7_family.sh — the orchestrator passes the
list of run names that have just completed (one per PDF). For each
run, we read eval/side_by_side/<run>/result.json and compute miss /
agree / override rates, then aggregate role + source counts across
runs and tally verdicts.

Missing or malformed result.json files surface as a "failed" run in
the totals; we do not fail the aggregator on partial results so the
orchestrator can still produce a summary even when one PDF crashed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_run(run_name: str, side_by_side_root: Path) -> dict:
    rj = side_by_side_root / run_name / "result.json"
    if not rj.exists():
        return {"run_name": run_name, "status": "missing_result_json"}
    try:
        data = json.loads(rj.read_text())
    except json.JSONDecodeError as exc:
        return {"run_name": run_name, "status": f"malformed_result_json: {exc}"}

    src = data.get("source_counts") or {}
    n = int(data.get("n_classified_blocks") or 0)
    miss = int(src.get("model:qwen_miss", 0))
    agree = int(src.get("qwen3:agree", 0))
    override = int(src.get("qwen3:override", 0))

    def _rate(v: int) -> float | None:
        return round(v / n, 4) if n else None

    return {
        "run_name": run_name,
        "status": "ok",
        "pdf_path": data.get("pdf_path"),
        "adapter": data.get("adapter"),
        "n_raw_blocks": data.get("n_raw_blocks"),
        "n_classified_blocks": n,
        "role_counts": data.get("role_counts") or {},
        "source_counts": dict(src),
        "qwen_miss_rate": _rate(miss),
        "agree_rate": _rate(agree),
        "override_rate": _rate(override),
        "axe_ok": data.get("axe_ok"),
        "escalation_verdict": data.get("escalation_verdict"),
        "escalation_reasons": data.get("escalation_reasons") or [],
        "stage_errors": data.get("stage_errors") or [],
        "html_chars": data.get("html_chars"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_names", nargs="+",
                    help="Run names whose result.json files to aggregate.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to write the aggregated summary JSON.")
    ap.add_argument("--side-by-side-root", type=Path,
                    default=Path("eval/side_by_side"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--classifier", default=None)
    args = ap.parse_args()

    runs = [_load_run(name, args.side_by_side_root) for name in args.run_names]

    ok_runs = [r for r in runs if r["status"] == "ok"]
    failed_runs = [{"run_name": r["run_name"], "status": r["status"]}
                   for r in runs if r["status"] != "ok"]

    agg_role: Counter[str] = Counter()
    agg_src: Counter[str] = Counter()
    n_total = 0
    miss_total = 0
    verdicts: Counter[str] = Counter()
    for r in ok_runs:
        agg_role.update(r["role_counts"])
        agg_src.update(r["source_counts"])
        n_total += r["n_classified_blocks"]
        miss_total += int(r["source_counts"].get("model:qwen_miss", 0))
        v = r.get("escalation_verdict") or "unknown"
        verdicts[v] += 1

    summary = {
        "adapter": args.adapter,
        "classifier": args.classifier,
        "runs": runs,
        "totals": {
            "n_runs": len(runs),
            "n_ok": len(ok_runs),
            "n_failed": len(failed_runs),
            "failed_runs": failed_runs,
            "agg_role_counts": dict(agg_role),
            "agg_source_counts": dict(agg_src),
            "agg_n_classified_blocks": n_total,
            "agg_qwen_miss_rate": (round(miss_total / n_total, 4)
                                   if n_total else None),
            "verdicts": dict(verdicts),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    # Also print a terse human-readable summary to stdout so the
    # orchestrator can pipe it into the summary log.
    print(f"[summary] {len(ok_runs)}/{len(runs)} runs ok "
          f"(failed: {[r['run_name'] for r in failed_runs] or 'none'})")
    for r in ok_runs:
        print(f"  {r['run_name']:40s} "
              f"blocks={r['n_classified_blocks']:5d} "
              f"miss={r['qwen_miss_rate']:.2%} "
              f"override={r['override_rate']:.2%} "
              f"verdict={r['escalation_verdict']}")
    if n_total:
        print(f"  agg miss rate: {miss_total / n_total:.2%} "
              f"({miss_total}/{n_total})")
    print(f"[summary] -> {args.out}")


if __name__ == "__main__":
    main()
