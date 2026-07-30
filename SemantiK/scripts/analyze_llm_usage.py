"""Passive P1 measurement over the P3 ``llm_usage.jsonl`` metering ledger.

Reads the rows the ``semantik_structure.llm_usage_meter`` tap appends at
SemantiK's four Super-120B call sites and reports, per site (filtered by default
to the reasoning-bearing ``super_resegment`` + ``reasoning_qc`` sites — the ones
whose thinking/effort config is DESIGN INTENT):

* **thinking-token share** — ``reasoning_tokens / completion_tokens`` — when the
  seat breaks reasoning tokens out in ``usage.completion_tokens_details``;
* otherwise the **completion-token distribution** (min / p50 / p95 / max / mean)
  PLUS, when present, the ``reasoning_chars`` proxy distribution — a char-length
  stand-in for thinking effort on a seat (e.g. the live Nemotron-3 Super on vLLM
  0.21.0) that folds reasoning tokens into ``completion_tokens`` and returns the
  reasoning TEXT on ``message.reasoning`` instead of a token count.

REPORT-ONLY: reads persisted rows, touches no thinking flag, changes no
behaviour. Point it at a run ledger (``runtime/state/runs/<id>/llm_usage.jsonl``) or a
standalone sidecar (``<SEMANTIK_DATA_DIR>/llm_usage/llm_usage.jsonl``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

# The reasoning-bearing sites (P1 focus). --all-sites lifts the filter.
_REASONING_SITES = ("super_resegment", "reasoning_qc")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summ(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 2),
        "p50": round(_percentile(values, 50), 2),
        "p95": round(_percentile(values, 95), 2),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2),
    }


def _read_rows(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def analyze(path: Path, sites: tuple[str, ...] | None) -> dict:
    by_site: dict[str, list[dict]] = {}
    for row in _read_rows(path):
        site = str(row.get("site", "?"))
        if sites is not None and site not in sites:
            continue
        by_site.setdefault(site, []).append(row)

    report: dict = {"ledger": str(path), "sites": {}}
    for site, rows in sorted(by_site.items()):
        completion = [float(r.get("completion_tokens", 0) or 0) for r in rows]
        reasoning_tok = [
            float(r["reasoning_tokens"])
            for r in rows
            if isinstance(r.get("reasoning_tokens"), (int, float))
        ]
        reasoning_chars = [
            float(r["reasoning_chars"])
            for r in rows
            if isinstance(r.get("reasoning_chars"), (int, float))
        ]
        entry: dict = {
            "calls": len(rows),
            "completion_tokens": _summ(completion),
            "duration_ms": _summ([float(r.get("duration_ms", 0) or 0) for r in rows]),
        }
        if reasoning_tok:
            shares = [
                rt / ct
                for rt, ct in zip(reasoning_tok, completion)
                if ct > 0
            ]
            entry["thinking_token_share"] = _summ(shares)
            entry["reasoning_tokens"] = _summ(reasoning_tok)
            entry["reasoning_token_source"] = "usage.completion_tokens_details"
        elif reasoning_chars:
            entry["reasoning_chars"] = _summ(reasoning_chars)
            entry["reasoning_token_source"] = (
                "message.reasoning (char proxy — seat exposes no reasoning-token count)"
            )
        else:
            entry["reasoning_token_source"] = "none (no reasoning channel on these rows)"
        report["sites"][site] = entry
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ledger", type=Path, help="Path to an llm_usage.jsonl ledger.")
    ap.add_argument(
        "--all-sites",
        action="store_true",
        help="Report every site, not just the reasoning-bearing ones.",
    )
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = ap.parse_args()

    if not args.ledger.is_file():
        raise SystemExit(f"ledger not found: {args.ledger}")

    sites = None if args.all_sites else _REASONING_SITES
    report = analyze(args.ledger, sites)

    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"llm_usage ledger: {report['ledger']}")
    if not report["sites"]:
        print("  (no matching rows — try --all-sites)")
        return
    for site, e in report["sites"].items():
        print(f"\n[{site}]  calls={e['calls']}  source={e['reasoning_token_source']}")
        print(f"  completion_tokens: {e['completion_tokens']}")
        print(f"  duration_ms:       {e['duration_ms']}")
        if "thinking_token_share" in e:
            print(f"  thinking_token_share: {e['thinking_token_share']}")
            print(f"  reasoning_tokens:     {e['reasoning_tokens']}")
        elif "reasoning_chars" in e:
            print(f"  reasoning_chars (proxy): {e['reasoning_chars']}")


if __name__ == "__main__":
    main()
