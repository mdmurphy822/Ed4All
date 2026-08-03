"""Cascade-level eval driver — Stage 1..13 over a held-out arXiv slice.

Single process, single GPU. NO PDF-level parallelism: the smoke
harness drives one PDF at a time through the shared validator and
shared (mock) Qwen runtime. Spawning workers would (a) fight the
single 8GB GPU and (b) thrash the Chromium context — both already
known foot-guns for this stack.

Usage::

    .venv/bin/python scripts/eval/eval_full_cascade.py \\
        --max-arxiv 5 \\
        --out data/eval_reports/full_cascade_dev.json

Resume::

    .venv/bin/python scripts/eval/eval_full_cascade.py \\
        --resume data/eval_reports/full_cascade_dev.json \\
        --max-arxiv 5

The resume path also doubles as the output path — partial JSON is
flushed via tmpfile + os.replace after every PDF and on SIGINT, so the
file on disk is always parseable.

The pipeline runs with ``runtime_mode="mock"`` (the LoRA adapters do
not yet exist on disk). The output JSON marks this clearly:

- top-level ``"mock_runtime": true``
- top-level ``"warning"`` describing what mock means
- mock-derived aggregate keys are suffixed ``_under_mock`` so they
  never get silently graphed against future real-runtime numbers.

Reference (raw_source_html) comparisons are reported as **deltas only**
— never as pass/fail booleans, never aggregated into a pass-rate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantik_structure.theta import StageThirteenStubRequired  # noqa: E402
from semantik_structure.validate import HtmlValidator  # noqa: E402

from semantik_structure.cascade import run_full_cascade  # noqa: E402


# Pair-id stem regex: "0704_1551__00_quantum_zeno_effect..." -> the
# stem is the file's basename minus .json. We compare stems verbatim
# against dataset `pair` fields.
_DATASET_FILES = {
    "structure_train": "data/structure_dataset/train.jsonl",
    "structure_val": "data/structure_dataset/val.jsonl",
    "semantic_train": "data/semantic_dataset/train.jsonl",
    "semantic_val": "data/semantic_dataset/val.jsonl",
}

WARNING_TEXT = (
    "All Stage 6 / Stage 9 outputs were produced by the MOCK Qwen runtime "
    "(no LoRA adapters on disk). axe pass-rates and violation distributions "
    "are NOT representative of the production cascade and MUST NOT be "
    "compared against future real-runtime runs. Aggregate keys are suffixed "
    "'_under_mock' to make this fail loudly if grafted onto a real-run "
    "dashboard."
)


def _read_pair_field_set(jsonl_path: Path) -> set[str]:
    """Collect the `pair` field of every record in a dataset jsonl.

    Empty / missing returns an empty set (so missing semantic_dataset
    is silent, per the plan)."""
    out: set[str] = set()
    if not jsonl_path.exists():
        return out
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = d.get("pair")
            if isinstance(pid, str):
                out.add(pid)
    return out


def _build_trained_pair_id_set(repo_root: Path) -> tuple[set[str], dict[str, int]]:
    """Union of pair-IDs across all train/val splits used to fit Semantic / Structure."""
    sizes: dict[str, int] = {}
    union: set[str] = set()
    for label, rel in _DATASET_FILES.items():
        ids = _read_pair_field_set(repo_root / rel)
        sizes[label] = len(ids)
        union |= ids
    return union, sizes


def _list_arxiv_pair_stems(arxiv_dir: Path) -> list[str]:
    """Return sorted list of pair-stems (basename minus .json) under data/pairs/arxiv."""
    out: list[str] = []
    for p in sorted(arxiv_dir.glob("*.json")):
        out.append(p.stem)
    return out


def _resolve_pair_pdf(arxiv_dir: Path, stem: str) -> Path | None:
    """Read the pair JSON and return the local_pdf path if it exists."""
    pair_path = arxiv_dir / f"{stem}.json"
    try:
        with pair_path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return None
    rel = d.get("local_pdf")
    if not isinstance(rel, str) or not rel:
        return None
    pdf = Path(rel)
    if not pdf.is_absolute():
        # Try repo-root interpretation.
        repo_root = arxiv_dir.resolve().parents[1]
        pdf = (repo_root / rel).resolve()
    return pdf if pdf.exists() else None


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write payload to path via tmpfile + os.replace (atomic on POSIX)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def _aggregate(per_pdf: list[dict]) -> dict[str, Any]:
    """Compute aggregate metrics from per-PDF results.

    Keys derived from the MOCK runtime are suffixed `_under_mock` so
    they never get silently graphed against future real-runtime
    numbers (gate 3).

    Stage 13 stub failures (architecture rows that v1 doesn't
    implement) are bucketed separately into ``n_stage13_stub`` so they
    don't get conflated with genuine cascade crashes (``n_failed``)
    or with valid runs (``n_ok``).
    """
    n_total = len(per_pdf)
    n_stage13_stub = sum(1 for r in per_pdf if r.get("stage13_stub_required"))
    n_failed = sum(1 for r in per_pdf if r.get("error") and not r.get("stage13_stub_required"))
    n_ok = n_total - n_failed - n_stage13_stub

    # Stage 10 axe pass-rate (mock).
    n_stage10_pass = sum(
        1 for r in per_pdf if not r.get("error") and r.get("stage10_pass_under_mock")
    )
    pass_rate = (n_stage10_pass / n_ok) if n_ok else None

    # Stage 7 violation distribution (mock).
    s7_counter: Counter[str] = Counter()
    for r in per_pdf:
        if r.get("error"):
            continue
        for k, v in (r.get("stage7_violations_under_mock") or {}).items():
            s7_counter[k] += int(v)

    # Stage 7 skipped (no-measurement) distribution (mock). A passed
    # check with skipped=True did NOT actually run against real source
    # signal — surfaced separately so silent gate-coverage holes
    # (gap-fill regions with empty source text, lxml-missing math
    # regions, doc with zero headings, etc.) don't masquerade as
    # healthy passes in the violation table.
    s7_skipped: Counter[str] = Counter()
    for r in per_pdf:
        if r.get("error"):
            continue
        for k, v in (r.get("stage7_skipped_under_mock") or {}).items():
            s7_skipped[k] += int(v)

    # Stage 10 axe rule distribution (mock).
    s10_counter: Counter[str] = Counter()
    for r in per_pdf:
        if r.get("error"):
            continue
        ax = r.get("stage10_axe_under_mock") or {}
        for v in ax.get("violations", []) or []:
            rid = v.get("id") if isinstance(v, dict) else None
            if rid:
                s10_counter[rid] += 1

    # Theta score distribution (mock-driven, since assembled HTML is mock).
    theta_scores = [
        r["theta"]["theta_score"]
        for r in per_pdf
        if not r.get("error") and r.get("theta") and r["theta"].get("theta_score") is not None
    ]
    if theta_scores:
        theta_avg = sum(theta_scores) / len(theta_scores)
        theta_min = min(theta_scores)
        theta_max = max(theta_scores)
    else:
        theta_avg = theta_min = theta_max = None

    # Per-stage timing averages.
    stage_totals: dict[str, list[float]] = {}
    for r in per_pdf:
        if r.get("error"):
            continue
        for k, v in (r.get("stages") or {}).items():
            stage_totals.setdefault(k, []).append(float(v))
    stage_avg = {k: (sum(v) / len(v)) if v else 0.0 for k, v in stage_totals.items()}

    # Region-kind histogram across all PDFs.
    rk_counter: Counter[str] = Counter()
    for r in per_pdf:
        if r.get("error"):
            continue
        for k, v in (r.get("region_kinds") or {}).items():
            rk_counter[k] += int(v)

    # Stage 13 stub-required breakdown by lane (when any).
    s13_lanes: Counter[str] = Counter()
    for r in per_pdf:
        lane = r.get("stage13_stub_required")
        if lane:
            s13_lanes[lane] += 1

    # Stage 13 offline-retry orchestration metrics. ``n_offline_retries``
    # counts PDFs where the orchestrator dispatched the offline lane;
    # ``n_offline_lane_won`` counts those where reconciliation kept the
    # offline report (architecture.md §6.3 — beat fast theta by
    # DELTA_THETA_IMPROVE OR offline cleared WCAG that fast couldn't).
    n_offline_retries = sum(
        1 for r in per_pdf if not r.get("error") and r.get("offline_retry_fired")
    )
    n_offline_lane_won = sum(
        1 for r in per_pdf if not r.get("error") and r.get("offline_retry_won")
    )

    # Council per-head signal-coverage rollup. For each head, sum the
    # `missing` count across PDFs (an FB the head SHOULD have scored
    # but didn't). A nonzero total here means at least one run took the
    # orchestrator's partial-failure path — the cross-reranker /
    # assembler will have used conservative defaults, which can mask
    # quality regressions. Surfaced loudly per the no-silent-fallbacks
    # invariant.
    coverage_missing: Counter[str] = Counter()
    coverage_expected: Counter[str] = Counter()
    coverage_observed: Counter[str] = Counter()
    coverage_errors: dict[str, int] = {}
    for r in per_pdf:
        if r.get("error"):
            continue
        cov = r.get("council_signal_coverage") or {}
        for head, entry in cov.items():
            if not isinstance(entry, dict):
                continue
            coverage_missing[head] += int(entry.get("missing", 0) or 0)
            coverage_expected[head] += int(entry.get("expected", 0) or 0)
            coverage_observed[head] += int(entry.get("observed", 0) or 0)
            if entry.get("error"):
                coverage_errors[head] = coverage_errors.get(head, 0) + 1

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_stage13_stub": n_stage13_stub,
        "stage13_stub_by_lane": dict(s13_lanes),
        "n_offline_retries": n_offline_retries,
        "n_offline_lane_won": n_offline_lane_won,
        "stage10_pass_rate_under_mock": pass_rate,
        "stage7_violations_under_mock": dict(s7_counter),
        "stage7_skipped_under_mock": dict(s7_skipped),
        "stage10_violations_under_mock": dict(s10_counter),
        "theta_score_avg_under_mock": theta_avg,
        "theta_score_min_under_mock": theta_min,
        "theta_score_max_under_mock": theta_max,
        "per_stage_avg_seconds": stage_avg,
        "region_kinds_total": dict(rk_counter),
        # merge_or_split "missing" == page-boundary count by design (builder
        # pairs within-page only; expected is global N-1) — see the expected
        # computation in semantik_structure/council/orchestrator.py.
        "council_signal_coverage_missing_total": dict(coverage_missing),
        "council_signal_coverage_expected_total": dict(coverage_expected),
        "council_signal_coverage_observed_total": dict(coverage_observed),
        "council_signal_coverage_error_pdf_count": dict(coverage_errors),
    }


def _summary_text(payload: dict) -> str:
    agg = payload.get("aggregate", {})
    # The aggregate/per-PDF JSON keys retain the legacy ``_under_mock``
    # SUFFIX regardless of runtime (loud-fail guard — see header docstring),
    # but the data is real whenever runtime_mode=="real". Make the
    # human-readable labels honest about which runtime produced them.
    # Real runs always stamp runtime_mode (see payload assembly below);
    # default to the conservative "mock" when a caller omits it.
    mode = payload.get("runtime_mode") or "mock"
    is_real = mode == "real"
    # tag substituted into the per-metric labels below
    mtag = "real-runtime" if is_real else "under mock"
    lines = []
    lines.append("# full-cascade eval summary")
    lines.append(f"timestamp:        {payload.get('timestamp')}")
    if is_real:
        lines.append(
            f"runtime_mode:     {mode}  (REAL — production GGUF adapters; "
            f"metric keys keep legacy '_under_mock' suffix but data is real)"
        )
    else:
        lines.append(f"runtime_mode:     {mode}  (NOT production)")
    if payload.get("theta_stub_allowed"):
        lines.append(
            "theta_stub_allowed: TRUE  (semantic_preservation may include "
            "0.7 stub_v1 placeholder; theta_score is NOT a real measurement)"
        )
    else:
        lines.append("theta_stub_allowed: false  (strict; raises if model unavailable)")
    lines.append(f"n_total:          {agg.get('n_total')}")
    lines.append(f"n_ok:             {agg.get('n_ok')}")
    lines.append(f"n_failed:         {agg.get('n_failed')}")
    n_stub = agg.get("n_stage13_stub", 0)
    if n_stub:
        lines.append(
            f"n_stage13_stub:   {n_stub}  (loud-fail; lanes={agg.get('stage13_stub_by_lane')})"
        )
    n_retries = agg.get("n_offline_retries", 0)
    n_won = agg.get("n_offline_lane_won", 0)
    lines.append(f"n_offline_retries: {n_retries}  (n_won={n_won})")
    pr = agg.get("stage10_pass_rate_under_mock")
    lines.append(f"stage10_pass_rate ({mtag}): {pr if pr is None else f'{pr:.4f}'}")
    lines.append(
        f"theta_score (avg/min/max, {mtag}): "
        f"{agg.get('theta_score_avg_under_mock')} / "
        f"{agg.get('theta_score_min_under_mock')} / "
        f"{agg.get('theta_score_max_under_mock')}"
    )
    lines.append("")
    lines.append("per_stage_avg_seconds:")
    for k, v in (agg.get("per_stage_avg_seconds") or {}).items():
        lines.append(f"  {k:12s}: {v:.3f}")
    lines.append("")
    lines.append(f"top stage7 violation rules ({mtag}) (count):")
    s7 = agg.get("stage7_violations_under_mock") or {}
    for rid, n in sorted(s7.items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"  {rid:40s} {n}")
    lines.append("")
    lines.append(f"stage7 skipped (no measurement) ({mtag}) (count):")
    s7s = agg.get("stage7_skipped_under_mock") or {}
    if s7s:
        for rid, n in sorted(s7s.items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"  {rid:40s} {n}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"top stage10 violation rules ({mtag}) (count):")
    s10 = agg.get("stage10_violations_under_mock") or {}
    for rid, n in sorted(s10.items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"  {rid:40s} {n}")
    lines.append("")
    lines.append("council per-head signal coverage (expected / observed / missing):")
    cov_exp = agg.get("council_signal_coverage_expected_total") or {}
    cov_obs = agg.get("council_signal_coverage_observed_total") or {}
    cov_mis = agg.get("council_signal_coverage_missing_total") or {}
    cov_err = agg.get("council_signal_coverage_error_pdf_count") or {}
    head_names = sorted(set(cov_exp) | set(cov_obs) | set(cov_mis) | set(cov_err))
    if head_names:
        for head in head_names:
            exp = int(cov_exp.get(head, 0) or 0)
            obs = int(cov_obs.get(head, 0) or 0)
            mis = int(cov_mis.get(head, 0) or 0)
            err = int(cov_err.get(head, 0) or 0)
            err_suffix = f"  errors_in={err}_pdfs" if err else ""
            lines.append(f"  {head:24s} {exp:>8d} / {obs:>8d} / {mis:>8d}{err_suffix}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"warning: {payload.get('warning')}")
    return "\n".join(lines) + "\n"


def _gather_eval_pair_ids(
    repo_root: Path,
    *,
    arxiv_pair_ids: list[str],
    side_by_side: bool,
    side_by_side_paths: list[Path],
) -> list[str]:
    """All eval pair-id-likes for the held-out assertion.

    For arxiv we use the pair-stem. For side-by-side we use the
    directory name (these don't have pair stems in the dataset, but
    we keep them in the assertion in case a future dataset tags
    them). The assertion will only fire on actual dataset pair-IDs.
    """
    out = list(arxiv_pair_ids)
    if side_by_side:
        for p in side_by_side_paths:
            out.append(p.parent.name)
    return out


def _select_arxiv_pairs(
    repo_root: Path,
    *,
    max_arxiv: int,
    exclude_stems: set[str],
    log,
) -> list[tuple[str, Path]]:
    """Walk data/pairs/arxiv, drop any stem in exclude_stems, cap at max_arxiv.

    Skips entries whose local_pdf doesn't exist on disk."""
    arxiv_dir = repo_root / "data/pairs/arxiv"
    all_stems = _list_arxiv_pair_stems(arxiv_dir)
    log(f"[eval] {len(all_stems)} arxiv pair stems on disk")
    log(f"[eval] {len(exclude_stems)} trained pair-ids to exclude")

    selected: list[tuple[str, Path]] = []
    skipped_no_pdf = 0
    for stem in all_stems:
        if len(selected) >= max_arxiv:
            break
        if stem in exclude_stems:
            continue
        pdf = _resolve_pair_pdf(arxiv_dir, stem)
        if pdf is None:
            skipped_no_pdf += 1
            continue
        selected.append((stem, pdf))

    log(f"[eval] selected {len(selected)} arxiv pairs (skipped {skipped_no_pdf} without local pdf)")
    return selected


def _run_cascade_isolated(
    pdf: Path,
    *,
    repo_root: Path,
    runtime: str,
    k: int,
    max_regions_per_kind: int,
    enable_glm_ocr_stage: bool,
    log,
) -> dict[str, Any]:
    """Run one PDF's cascade in a FRESH subprocess (per-PDF process isolation).

    On the 8GB card a single long-running process accumulates VRAM across PDFs
    (cached council BERTs + theta + figure captioner + a per-PDF Qwen GGUF; the
    llama.cpp CUDA allocator also pools), so the 2nd+ PDF OOMs on GGUF load.
    Isolating each PDF in its own process makes the OS reclaim ALL VRAM on exit
    — the canonical pattern for batch GPU inference under a hard VRAM budget.
    Reuses scripts/pdf_to_html.py in report-only mode (--report, no --out;
    same run_full_cascade result dict).
    """
    script = repo_root / "scripts" / "pdf_to_html.py"
    with tempfile.NamedTemporaryFile(
        suffix=".json",
        prefix="r10_isolated_",
        delete=False,
    ) as tf:
        out_path = Path(tf.name)
    cmd = [
        sys.executable,
        str(script),
        "--pdf",
        str(pdf),
        "--report",
        str(out_path),
        "--runtime",
        runtime,
        "--k",
        str(k),
        "--max-regions-per-kind",
        str(max_regions_per_kind),
    ]
    if enable_glm_ocr_stage:
        cmd.append("--enable-glm-ocr-stage")
    # Inherit env but ensure the real theta model is used (never the stub) so
    # the isolated run matches the in-process contract.
    env = dict(os.environ)
    env.pop("SEMANTIK_ALLOW_THETA_STUB", None)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    try:
        result = json.loads(out_path.read_text())
    except Exception as exc:  # subprocess died before writing valid JSON
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise RuntimeError(
            f"isolated cascade subprocess produced no parseable report "
            f"(rc={proc.returncode}): {exc}\n--- tail ---\n{tail}"
        )
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass
    if "error" in result:
        # Surface the in-subprocess failure as the loop's normal failure path.
        raise RuntimeError(f"isolated cascade failed: {result.get('error')}")
    return result


def _maybe_clear_cookies(validator: HtmlValidator) -> None:
    """Best-effort cookie clear between PDFs. Silently no-ops if absent."""
    ctx = getattr(validator, "_context", None)
    if ctx is None:
        return
    fn = getattr(ctx, "clear_cookies", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-arxiv", type=int, default=40, help="Max number of arxiv held-out pairs (default 40)."
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Eval-corpus manifest JSON (scripts/datasets/build_eval_corpus_manifest.py "
        "output, e.g. data/eval_corpus/manifest_v1.json). When given, evaluate "
        "EXACTLY the manifest's PDFs — the built-in arXiv selection, "
        "--max-arxiv and --side-by-side discovery are bypassed. The held-out "
        "assertion still runs over the manifest's pair_ids.",
    )
    ap.add_argument(
        "--side-by-side",
        dest="side_by_side",
        action="store_true",
        default=False,
        help="Include PDFs supplied by --side-by-side-dir (off by default).",
    )
    ap.add_argument("--no-side-by-side", dest="side_by_side", action="store_false")
    ap.add_argument(
        "--side-by-side-dir",
        action="append",
        type=Path,
        default=[],
        help="Explicit benchmark directory containing input.pdf; repeatable.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Default data/eval_reports/full_cascade_<ts>.json",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing report (use as --out too).",
    )
    ap.add_argument(
        "--warmup-pdf",
        type=Path,
        default=None,
        help="Run this PDF through the cascade once to warm caches (not recorded).",
    )
    ap.add_argument(
        "--invalidate-extract-cache",
        action="store_true",
        help="Wipe data/extract_cache/* before running.",
    )
    ap.add_argument(
        "--invalidate-prerender-cache",
        action="store_true",
        help="Wipe data/prerender_cache/* before running.",
    )
    ap.add_argument(
        "--summary", action="store_true", help="Also write a .summary.txt next to the JSON."
    )
    ap.add_argument(
        "--max-regions-per-kind", type=int, default=0, help="Cap regions per kind (0 = no cap)."
    )
    ap.add_argument("--k", type=int, default=2, help="K candidates per region from Stage 6.")
    ap.add_argument(
        "--runtime",
        choices=("mock", "real"),
        default="mock",
        help="Qwen Stage-6 runtime. 'mock' (default): deterministic "
        "stub output. 'real': llama-cpp over the trained GGUFs "
        "(requires every adapter's GGUF wired in config.yaml). "
        "Real runs write to a distinct report and are flagged "
        "runtime_mode=real so they never mix with mock numbers.",
    )
    ap.add_argument(
        "--isolate-per-pdf",
        action="store_true",
        help="Run each PDF's cascade in a fresh subprocess so the OS reclaims "
        "all VRAM between PDFs. Use for multi-PDF --runtime real when a "
        "single process accumulates VRAM (cached BERTs + "
        "theta + captioner + per-PDF Qwen GGUF) and OOMs on the 2nd PDF.",
    )
    ap.add_argument(
        "--enable-glm-ocr-stage",
        action="store_true",
        help=(
            "Enable optional Stage 5b: GLM-OCR enrichment of "
            "council-confirmed table regions. Cache-only (the eval "
            "driver doesn't load the GLM model) — pre-warm via "
            "scripts/datasets/glm_ocr_prepass.py if you want hits."
        ),
    )
    args = ap.parse_args(argv)
    if args.manifest is not None and (args.side_by_side or args.side_by_side_dir):
        ap.error("--manifest cannot be combined with side-by-side inputs")
    if args.side_by_side_dir:
        args.side_by_side = True
    if args.side_by_side and not args.side_by_side_dir:
        ap.error("--side-by-side requires at least one --side-by-side-dir")

    def log(msg: str) -> None:
        print(msg, flush=True)

    # ---------- pinned-manifest mode ----------
    # A manifest pins the exact PDF set (stratified, seed-built, trained-pair-
    # excluded at build time). Missing PDFs are a hard error — never silently
    # evaluate a smaller corpus than the manifest promises.
    manifest_payload: dict[str, Any] | None = None
    if args.manifest is not None:
        manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        m_entries = manifest_payload.get("entries") or []
        if not m_entries:
            raise RuntimeError(f"--manifest {args.manifest} has no entries")
        m_missing = [e["pdf_path"] for e in m_entries if not Path(e["pdf_path"]).exists()]
        if m_missing:
            raise RuntimeError(
                f"--manifest lists {len(m_missing)} PDFs missing on disk (first 5): {m_missing[:5]}"
            )
        log(
            f"[eval] manifest mode: {len(m_entries)} PDFs from {args.manifest} "
            f"(schema={manifest_payload.get('schema_version')}; built-in "
            "arXiv/side-by-side selection bypassed)"
        )

    # ---------- Stage 5b banner ----------
    if args.enable_glm_ocr_stage:
        log(
            "[eval] Stage 5b: GLM-OCR table enrichment ENABLED "
            "(cache-only — only cached bboxes get text; pre-warm via "
            "scripts/datasets/glm_ocr_prepass.py)."
        )

    # ---------- theta stub-fallback banner ----------
    # The theta semantic-preservation cross-encoder is strict-by-default
    # (see semantik_structure/theta/_module_state.py). When the model is
    # missing/broken, evaluate() raises unless SEMANTIK_ALLOW_THETA_STUB=1
    # is set. Surface the active state loudly so a human reading the
    # log knows whether theta_score includes a 0.7 placeholder.
    _stub_allowed = os.environ.get("SEMANTIK_ALLOW_THETA_STUB", "").strip() == "1"
    if _stub_allowed:
        log(
            "[eval] WARNING: SEMANTIK_ALLOW_THETA_STUB=1 — semantic_preservation "
            "may fall back to the 0.7 stub_v1 placeholder. Composite "
            "theta_score is NOT a real measurement when this flag is set "
            "AND the trained model is missing/broken."
        )
    else:
        log(
            "[eval] strict-by-default: theta semantic_preservation must "
            "load successfully OR every PDF will fail. Set "
            "SEMANTIK_ALLOW_THETA_STUB=1 to permit stub fallback."
        )

    # ---------- output path / resume ----------
    if args.resume is not None:
        out_path = args.resume
    elif args.out is not None:
        out_path = args.out
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # Real runs get a distinct prefix so their numbers never sit in the
        # same filename namespace as the mock baselines.
        tag = "real" if args.runtime == "real" else "mock"
        out_path = repo_root / f"data/eval_reports/full_cascade_{tag}_{ts}.json"

    resumed_payload: dict[str, Any] | None = None
    if args.resume is not None and args.resume.exists():
        try:
            resumed_payload = json.loads(args.resume.read_text())
            log(
                f"[eval] resuming from {args.resume} "
                f"(already has {len(resumed_payload.get('per_pdf', []))} entries)"
            )
        except Exception as exc:
            log(f"[eval] WARN: --resume target unreadable ({exc!r}); starting fresh")
            resumed_payload = None

    # ---------- caches ----------
    if args.invalidate_extract_cache:
        cache_dir = repo_root / "data/extract_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            log(f"[eval] wiped {cache_dir}")
    if args.invalidate_prerender_cache:
        cache_dir = repo_root / "data/prerender_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            log(f"[eval] wiped {cache_dir}")

    # ---------- held-out construction ----------
    trained, sizes = _build_trained_pair_id_set(repo_root)
    log(f"[eval] trained pair-ids per dataset: {sizes}")
    log(f"[eval] trained pair-ids union size: {len(trained)}")

    arxiv_pairs: list[tuple[str, Path]] = []
    side_by_side_paths: list[Path] = []
    if manifest_payload is None:
        arxiv_pairs = _select_arxiv_pairs(
            repo_root,
            max_arxiv=args.max_arxiv,
            exclude_stems=trained,
            log=log,
        )

        if args.side_by_side:
            for directory in args.side_by_side_dir:
                p = directory.expanduser() / "input.pdf"
                if not p.is_file():
                    raise FileNotFoundError(f"benchmark input does not exist: {p}")
                side_by_side_paths.append(p)
            log(f"[eval] side-by-side PDFs: {len(side_by_side_paths)} (weakly-held-out)")

    # ---------- HARD held-out assertion (gate 2) ----------
    if manifest_payload is not None:
        # Manifest entries carry their pair_id (or none, for never-paired
        # PDFs — use the PDF stem, which can't collide with dataset ids).
        eval_pair_ids = [
            e.get("pair_id") or Path(e["pdf_path"]).stem for e in manifest_payload["entries"]
        ]
    else:
        arxiv_stems = [s for s, _ in arxiv_pairs]
        eval_pair_ids = _gather_eval_pair_ids(
            repo_root,
            arxiv_pair_ids=arxiv_stems,
            side_by_side=args.side_by_side,
            side_by_side_paths=side_by_side_paths,
        )
    overlap = sorted(set(eval_pair_ids) & trained)
    assert not overlap, (
        "Held-out assertion FAILED: the following eval pair-ids appear in "
        "structure_dataset/{train,val} or semantic_dataset/{train,val}: "
        f"{overlap[:10]}{'...' if len(overlap) > 10 else ''}"
    )
    log("[eval] held-out assertion: OK (no overlap with trained pair-ids)")

    # ---------- already-done set (resume) ----------
    # Resume now tracks by canonical PDF path (post-dedupe). Old reports
    # used per-pair-stem entries; we still read those by falling back to
    # the entry's `pdf` field which is present in both schemas.
    done_pdfs: set[str] = set()
    per_pdf: list[dict] = []
    if resumed_payload is not None:
        per_pdf = list(resumed_payload.get("per_pdf", []))
        for r in per_pdf:
            if isinstance(r, dict) and r.get("pdf"):
                done_pdfs.add(str(r["pdf"]))
        log(f"[eval] {len(done_pdfs)} PDFs already in resumed report; will skip")

    # ---------- assemble work list (DEDUPED by PDF) ----------
    # Multiple arxiv pair-stems can resolve to the same local PDF (one
    # per section of the same paper). Running the cascade per pair-stem
    # is N× wasted compute and skews aggregate metrics by weighting
    # multi-section papers more heavily. Dedupe to one cascade run per
    # unique PDF, recording every pair-id that maps to it for traceability.
    pdf_to_pair_ids: dict[str, list[str]] = {}
    pdf_to_sources: dict[str, list[str]] = {}
    pdf_path_obj: dict[str, Path] = {}

    def _add(pair_id: str, pdf_path: Path, source: str) -> None:
        key = str(pdf_path.resolve())
        pdf_to_pair_ids.setdefault(key, []).append(pair_id)
        srcs = pdf_to_sources.setdefault(key, [])
        if source not in srcs:
            srcs.append(source)
        pdf_path_obj[key] = pdf_path

    if manifest_payload is not None:
        for e in manifest_payload["entries"]:
            label = e.get("pair_id") or Path(e["pdf_path"]).stem
            _add(label, Path(e["pdf_path"]), e.get("source") or "manifest")
    for stem, pdf in arxiv_pairs:
        _add(stem, pdf, "arxiv")
    for p in side_by_side_paths:
        _add(p.parent.name, p, "side_by_side")

    work: list[tuple[Path, list[str], list[str]]] = []
    for key in pdf_path_obj:
        work.append(
            (pdf_path_obj[key], sorted(pdf_to_pair_ids[key]), pdf_to_sources[key]),
        )
    # Stable order: arxiv-first by PDF path, side-by-side last (preserves
    # the prior log ordering for human readability).
    work.sort(key=lambda t: ("side_by_side" in t[2], str(t[0])))

    n_pair_ids = sum(len(pids) for _, pids, _ in work)
    if n_pair_ids != len(work):
        log(
            f"[eval] dedupe: {n_pair_ids} pair-ids -> {len(work)} unique PDFs "
            f"({n_pair_ids - len(work)} duplicate runs eliminated)"
        )
    log(f"[eval] total work units: {len(work)} unique PDFs")

    # ---------- payload skeleton ----------
    payload: dict[str, Any] = {
        "schema_version": "full_cascade_eval/1.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mock_runtime": args.runtime == "mock",
        "runtime_mode": args.runtime,
        "warning": (
            WARNING_TEXT
            if args.runtime == "mock"
            else "REAL runtime over the trained GGUF adapters (runtime_mode=real, "
            "distinct filename, mock_runtime=false). NOTE: per-PDF and aggregate "
            "keys retain the legacy '_under_mock' SUFFIX, but the DATA here is "
            "real-runtime — do not confuse with mock baselines. theta_score is "
            "the 0.7 stub placeholder whenever theta_stub_allowed=true (theta v1 "
            "is mode-collapsed; diagnostic-only)."
        ),
        "theta_stub_allowed": _stub_allowed,
        "config": {
            "max_arxiv": args.max_arxiv,
            "side_by_side": args.side_by_side,
            "max_regions_per_kind": args.max_regions_per_kind,
            "k": args.k,
            "invalidate_extract_cache": args.invalidate_extract_cache,
            "invalidate_prerender_cache": args.invalidate_prerender_cache,
            "manifest": str(args.manifest) if args.manifest is not None else None,
        },
        "trained_pair_ids_size": len(trained),
        "trained_dataset_sizes": sizes,
        "n_arxiv_selected": len(arxiv_pairs),
        "n_side_by_side": len(side_by_side_paths),
        "manifest_schema_version": (
            manifest_payload.get("schema_version") if manifest_payload is not None else None
        ),
        "n_manifest_entries": (
            len(manifest_payload["entries"]) if manifest_payload is not None else None
        ),
        "side_by_side_note": (
            "side-by-side PDFs were used during development; treat as "
            "weakly-held-out, not formally held-out."
        ),
        "out_path": str(out_path),
        "per_pdf": per_pdf,
        "aggregate": _aggregate(per_pdf),
    }

    # Carry over resumed metadata where it doesn't conflict.
    if resumed_payload is not None:
        payload["resumed_from"] = str(args.resume)

    # ---------- SIGINT handler — flush partial JSON cleanly ----------
    flushed_on_signal = {"done": False}

    def _flush_partial(reason: str) -> None:
        payload["aggregate"] = _aggregate(payload["per_pdf"])
        payload["interrupted"] = reason
        _atomic_write_json(out_path, payload)
        log(f"[eval] partial JSON flushed -> {out_path}  (reason: {reason})")

    def _on_sigint(signum, frame):  # noqa: ARG001
        if flushed_on_signal["done"]:
            return
        flushed_on_signal["done"] = True
        _flush_partial("SIGINT")
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    # ---------- one validator + one runtime, shared across PDFs ----------
    t_run0 = time.perf_counter()
    with HtmlValidator() as validator:
        # Optional warmup — runs but is NOT recorded.
        if args.warmup_pdf is not None and args.warmup_pdf.exists():
            log(f"[eval] warmup on {args.warmup_pdf}")
            try:
                run_full_cascade(
                    args.warmup_pdf,
                    validator=validator,
                    max_regions_per_kind=args.max_regions_per_kind,
                    k=args.k,
                    runtime_mode=args.runtime,
                    log=lambda m: None,
                )
            except Exception as exc:
                log(f"[eval] warmup raised {exc!r} (ignored, continuing)")

        for i, (pdf, pair_ids, sources) in enumerate(work, 1):
            pdf_str = str(pdf)
            if pdf_str in done_pdfs:
                log(f"[eval] [{i}/{len(work)}] skip (already done): {pair_ids[0]}")
                continue

            label = pair_ids[0]
            if len(pair_ids) > 1:
                label = f"{pair_ids[0]} (+{len(pair_ids) - 1} more)"
            log(f"[eval] [{i}/{len(work)}] {','.join(sources)}: {label}  ({pdf})")
            entry: dict[str, Any] = {
                "pair_ids": pair_ids,
                "sources": sources,
                "pdf": pdf_str,
            }
            t0 = time.perf_counter()
            try:
                if args.isolate_per_pdf:
                    result = _run_cascade_isolated(
                        pdf,
                        repo_root=repo_root,
                        runtime=args.runtime,
                        k=args.k,
                        max_regions_per_kind=args.max_regions_per_kind,
                        enable_glm_ocr_stage=args.enable_glm_ocr_stage,
                        log=log,
                    )
                else:
                    result = run_full_cascade(
                        pdf,
                        validator=validator,
                        max_regions_per_kind=args.max_regions_per_kind,
                        k=args.k,
                        runtime_mode=args.runtime,
                        enable_glm_ocr_stage=args.enable_glm_ocr_stage,
                        log=lambda m: None,  # quiet per-PDF
                    )
                entry.update(result)
            except KeyboardInterrupt:
                # Let the SIGINT handler flush.
                raise
            except StageThirteenStubRequired as exc:
                # Architecture row routes to an unimplemented v1 lane.
                # Tag distinctly so the aggregate doesn't conflate this
                # with a genuine cascade crash and so a future loud-fail
                # consumer can grep ``stage13_stub_required`` and act.
                entry["stage13_stub_required"] = exc.lane_required
                entry["error"] = f"StageThirteenStubRequired: {exc!s}"
                entry["partial_theta"] = {
                    "wcag_status": exc.report.wcag_status,
                    "lane": exc.report.lane,
                    "theta_score": exc.report.theta_score,
                    "flags": list(exc.report.flags or []),
                }
                entry["elapsed_total"] = time.perf_counter() - t0
                log(
                    f"[eval]   STAGE13_STUB({exc.lane_required}): "
                    f"theta={exc.report.theta_score} wcag={exc.report.wcag_status}"
                )
            except Exception as exc:
                tb = traceback.format_exc(limit=3)
                entry["error"] = f"{type(exc).__name__}: {exc!r}"
                entry["traceback"] = tb
                entry["elapsed_total"] = time.perf_counter() - t0
                log(f"[eval]   FAILED: {entry['error']}")

            payload["per_pdf"].append(entry)
            payload["aggregate"] = _aggregate(payload["per_pdf"])
            _atomic_write_json(out_path, payload)
            _maybe_clear_cookies(validator)

    payload["completed"] = True
    payload["elapsed_total_sec"] = time.perf_counter() - t_run0
    payload["aggregate"] = _aggregate(payload["per_pdf"])
    _atomic_write_json(out_path, payload)

    if args.summary:
        summary_path = out_path.with_suffix(out_path.suffix + ".summary.txt")
        summary_path.write_text(_summary_text(payload))
        log(f"[eval] summary -> {summary_path}")

    log(f"[eval] DONE. report -> {out_path}")
    log(f"[eval] aggregate: {json.dumps(payload['aggregate'], indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
