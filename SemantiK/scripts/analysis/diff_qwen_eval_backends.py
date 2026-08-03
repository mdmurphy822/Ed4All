"""Quantization-drift diff between the two Qwen math-adapter eval backends.

Compares two eval runs produced by ``scripts/eval/eval_qwen_math_adapter.py`` —
the ``safetensors`` backend (trained HF weights, 4-bit) vs the ``gguf``
backend (production-quantized) — on the IDENTICAL 300-row sample, and
emits a markdown drift report.

The aggregate report JSON
(``data/eval_reports/qwen_math_adapter_<backend>.json``) carries a
pre-aggregated ``overall`` / ``per_class`` block, but the row-level
disagreement set requires per-row scores, which the JSON does not store.
So this script joins the two ``.samples.jsonl`` files on the row identity
key ``(arxiv_id, math_idx)`` and RESCORES every row with the eval's own
``score_one`` — the single source of truth — exactly as the eval itself
does (the report's ``"rescored": true`` flag). Aggregates are recomputed
from those rescored rows so the markdown is internally consistent rather
than mixing a stored block with freshly scored disagreements.

No GPU, no model, no torch: ``score_one`` and friends are pure-Python /
stdlib, and importing the eval module does not pull torch (the heavy
imports live inside the backend functions).

Usage::

    python -m scripts.analysis.diff_qwen_eval_backends \\
        --safetensors-report data/eval_reports/qwen_math_adapter_safetensors.json \\
        --gguf-report        data/eval_reports/qwen_math_adapter_gguf.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Boolean-rate metrics (fraction True), recomputed from per-row scores.
_RATE_METRICS = (
    "exact",
    "norm",
    "id_stripped_exact",
    "alttext_present",
    "alttext_match",
    "wellformed",
    "truncated",
)
# Mean-of-float metrics.
_MEAN_METRICS = (
    "id_stripped_sim",
    "char_sim",
)
# Order metrics appear in the markdown tables.
_METRIC_ORDER = (
    "exact",
    "norm",
    "id_stripped_exact",
    "id_stripped_sim",
    "alttext_present",
    "alttext_match",
    "wellformed",
    "truncated",
    "char_sim",
)
# Stable per-class row order (rare types first, then inline), with any
# unexpected class appended.
_CLASS_ORDER = ("inline", "display", "numbered", "multiline", "matrix")

# Cap for the disagreement listing (full count still reported).
_DISAGREE_CAP = 25


def _load_eval_module():
    """Load eval_qwen_math_adapter for its score_one — no torch pulled in."""
    path = REPO_ROOT / "scripts" / "eval" / "eval_qwen_math_adapter.py"
    spec = importlib.util.spec_from_file_location("_eqma", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"cannot load eval module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_samples(path: Path) -> dict[tuple[str, int], dict]:
    """Read a .samples.jsonl into {(arxiv_id, math_idx): row}."""
    if not path.exists():
        raise FileNotFoundError(f"samples file not found: {path}")
    out: dict[tuple[str, int], dict] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("arxiv_id"), r.get("math_idx"))
            if key in out:
                raise ValueError(
                    f"duplicate identity key {key} in {path} (line {lineno})"
                )
            out[key] = r
    return out


def _rescore(samples: dict[tuple[str, int], dict], score_one) -> dict:
    """Map identity key -> {math_type, scores} by rescoring gen vs gold."""
    out = {}
    for key, r in samples.items():
        out[key] = {
            "math_type": r.get("math_type", "inline"),
            "scores": score_one(r["gen"], r["gold"]),
        }
    return out


def _aggregate(rows: list[dict]) -> dict:
    """Recompute rate + mean metrics from a list of {scores: ...} rows."""
    n = len(rows)
    agg: dict[str, float] = {"n": n}
    if n == 0:
        for k in _RATE_METRICS + _MEAN_METRICS:
            agg[k] = 0.0
        return agg
    for k in _RATE_METRICS:
        agg[k] = sum(1 for r in rows if r["scores"][k]) / n
    for k in _MEAN_METRICS:
        agg[k] = sum(r["scores"][k] for r in rows) / n
    return agg


def _fmt(metric: str, value: float) -> str:
    return f"{value:.4f}"


def _delta_table(safe: dict, gguf: dict) -> list[str]:
    lines = [
        "| metric | safetensors | gguf | delta (st - gguf) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for m in _METRIC_ORDER:
        s, g = safe[m], gguf[m]
        lines.append(f"| {m} | {_fmt(m, s)} | {_fmt(m, g)} | {s - g:+.4f} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--safetensors-report",
        type=Path,
        default=REPO_ROOT / "data/eval_reports/qwen_math_adapter_safetensors.json",
    )
    ap.add_argument(
        "--gguf-report",
        type=Path,
        default=REPO_ROOT / "data/eval_reports/qwen_math_adapter_gguf.json",
    )
    ap.add_argument(
        "--safetensors-samples",
        type=Path,
        default=None,
        help="defaults to the safetensors report with .json -> .samples.jsonl",
    )
    ap.add_argument(
        "--gguf-samples",
        type=Path,
        default=None,
        help="defaults to the gguf report with .json -> .samples.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "data/eval_reports/qwen_math_drift_safetensors_vs_gguf.md",
    )
    args = ap.parse_args()

    def _samples_default(report: Path) -> Path:
        return report.with_suffix(".samples.jsonl")

    safe_samples_path = args.safetensors_samples or _samples_default(
        args.safetensors_report
    )
    gguf_samples_path = args.gguf_samples or _samples_default(args.gguf_report)

    eqma = _load_eval_module()
    score_one = eqma.score_one

    # ---- aggregate report metadata (for the header) -----------------------
    safe_report = json.loads(args.safetensors_report.read_text(encoding="utf-8"))
    gguf_report = json.loads(args.gguf_report.read_text(encoding="utf-8"))

    # ---- join the samples on (arxiv_id, math_idx) -------------------------
    safe_samples = _load_samples(safe_samples_path)
    gguf_samples = _load_samples(gguf_samples_path)

    safe_keys = set(safe_samples)
    gguf_keys = set(gguf_samples)
    shared = safe_keys & gguf_keys
    only_safe = safe_keys - gguf_keys
    only_gguf = gguf_keys - safe_keys

    mismatch = bool(only_safe or only_gguf)

    safe_scored = _rescore(safe_samples, score_one)
    gguf_scored = _rescore(gguf_samples, score_one)

    # ---- aggregates over the SHARED rows ----------------------------------
    safe_rows = [safe_scored[k] for k in shared]
    gguf_rows = [gguf_scored[k] for k in shared]
    safe_overall = _aggregate(safe_rows)
    gguf_overall = _aggregate(gguf_rows)

    # ---- per-class aggregates ---------------------------------------------
    safe_by_class: dict[str, list[dict]] = defaultdict(list)
    gguf_by_class: dict[str, list[dict]] = defaultdict(list)
    for k in shared:
        safe_by_class[safe_scored[k]["math_type"]].append(safe_scored[k])
        gguf_by_class[gguf_scored[k]["math_type"]].append(gguf_scored[k])
    classes = sorted(
        set(safe_by_class) | set(gguf_by_class),
        key=lambda c: (_CLASS_ORDER.index(c) if c in _CLASS_ORDER else 99, c),
    )

    # ---- row-level id_stripped_exact disagreement set ---------------------
    disagreements = []
    for k in sorted(shared, key=lambda x: (x[0] or "", x[1])):
        s, g = safe_scored[k], gguf_scored[k]
        if s["scores"]["id_stripped_exact"] != g["scores"]["id_stripped_exact"]:
            disagreements.append(
                {
                    "arxiv_id": k[0],
                    "math_idx": k[1],
                    "math_type": s["math_type"],
                    "safe_idse": s["scores"]["id_stripped_exact"],
                    "gguf_idse": g["scores"]["id_stripped_exact"],
                    "safe_trunc": s["scores"]["truncated"],
                    "gguf_trunc": g["scores"]["truncated"],
                }
            )

    # ---- build markdown ---------------------------------------------------
    md: list[str] = []
    md.append("# Qwen math adapter — quantization drift: safetensors vs gguf")
    md.append("")
    md.append(f"- safetensors report: `{args.safetensors_report}`")
    md.append(f"- gguf report: `{args.gguf_report}`")
    md.append(f"- safetensors samples: `{safe_samples_path}`")
    md.append(f"- gguf samples: `{gguf_samples_path}`")
    md.append(
        f"- base_model: `{safe_report.get('base_model')}` "
        f"(gguf: `{gguf_report.get('base_model')}`)"
    )
    md.append(
        f"- sample: safetensors n={len(safe_keys)}, gguf n={len(gguf_keys)}, "
        f"shared n={len(shared)}"
    )
    md.append(
        "- metrics RESCORED from samples via eval_qwen_math_adapter.score_one "
        "(deltas = safetensors - gguf)"
    )
    md.append("")

    if mismatch:
        md.append("## ⚠ SAMPLE MISMATCH")
        md.append("")
        md.append(
            "The two backends did NOT score the identical sample. Aggregate "
            "and per-class tables below cover ONLY the shared rows; the "
            "disagreement set is restricted to shared rows. Resolve the "
            "mismatch before trusting the drift numbers."
        )
        md.append("")
        md.append(f"- rows only in safetensors ({len(only_safe)}):")
        for k in sorted(only_safe, key=lambda x: (x[0] or "", x[1]))[:50]:
            md.append(f"  - `{k[0]}` idx={k[1]}")
        if len(only_safe) > 50:
            md.append(f"  - ... and {len(only_safe) - 50} more")
        md.append(f"- rows only in gguf ({len(only_gguf)}):")
        for k in sorted(only_gguf, key=lambda x: (x[0] or "", x[1]))[:50]:
            md.append(f"  - `{k[0]}` idx={k[1]}")
        if len(only_gguf) > 50:
            md.append(f"  - ... and {len(only_gguf) - 50} more")
        md.append("")
    else:
        md.append(
            f"Both backends cover the identical {len(shared)} rows. "
            "No sample mismatch."
        )
        md.append("")

    # 1. aggregate delta
    md.append("## 1. Aggregate delta")
    md.append("")
    md.extend(_delta_table(safe_overall, gguf_overall))
    md.append("")

    # 2. per-class delta
    md.append("## 2. Per-class delta (by math_type)")
    md.append("")
    for cls in classes:
        sa = _aggregate(safe_by_class.get(cls, []))
        ga = _aggregate(gguf_by_class.get(cls, []))
        md.append(f"### {cls} (n safetensors={sa['n']}, gguf={ga['n']})")
        md.append("")
        md.extend(_delta_table(sa, ga))
        md.append("")

    # 3. disagreement set
    md.append("## 3. id_stripped_exact disagreement set")
    md.append("")
    md.append(
        f"Rows where `id_stripped_exact` flips between backends: "
        f"**{len(disagreements)}** of {len(shared)} shared rows."
    )
    md.append("")
    if disagreements:
        md.append(
            "| arxiv_id | math_idx | class | st id_exact | gguf id_exact "
            "| st trunc | gguf trunc |"
        )
        md.append("| --- | ---: | --- | :---: | :---: | :---: | :---: |")
        for d in disagreements[:_DISAGREE_CAP]:
            md.append(
                f"| {d['arxiv_id']} | {d['math_idx']} | {d['math_type']} "
                f"| {d['safe_idse']} | {d['gguf_idse']} "
                f"| {d['safe_trunc']} | {d['gguf_trunc']} |"
            )
        if len(disagreements) > _DISAGREE_CAP:
            md.append("")
            md.append(
                f"_... {len(disagreements) - _DISAGREE_CAP} more not shown "
                f"(cap {_DISAGREE_CAP})._"
            )
    else:
        md.append("_No disagreements: id_stripped_exact agrees on every shared row._")
    md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[drift] wrote {args.out}")
    print(
        f"[drift] shared={len(shared)} mismatch={mismatch} "
        f"disagreements={len(disagreements)}"
    )
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
