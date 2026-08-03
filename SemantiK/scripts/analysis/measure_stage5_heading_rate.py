"""Stage 5 heading-rate diagnostic.

Measures, per PDF in a small diverse sample, how many FeatureBlocks
Structure's ``is_heading`` head fires "heading" on at each of a series
of confidence thresholds. The Stage-5 default threshold (currently
0.5) determines how many FBs become heading-kind regions; we want to
know whether the default lets through implausible heading counts on
ML / math / form / prose PDFs.

For one PDF, we additionally compare against a reference HTML's
<h1>..<h6> count to ground the rate in a "true" heading number.

Saves the full measurement to ``data/eval_reports/stage5_heading_rate.json``
and prints a thresholds × PDFs table.

Usage:
    .venv/bin/python scripts/analysis/measure_stage5_heading_rate.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# External arXiv paper repo (machine-specific) for the one out-of-tree probe
# below; supplied via SEMANTIK_ARXIV_REPO, falling back to ``~/arxiv-repo/papers``.
# No hardcoded absolute path in tracked code.
_ARXIV_REPO = Path(
    os.environ.get("SEMANTIK_ARXIV_REPO", str(Path.home() / "arxiv-repo" / "papers"))
).expanduser()


# Diverse 5-PDF sample. Four are eval/side_by_side fixtures (covering
# tables-heavy ML, math-heavy physics, prose whitepaper, ML / quantum
# theory), one is an arxiv local PDF outside the eval/ set so we don't
# overfit to fixtures we're already optimizing for.
_PDFS: list[tuple[str, str, Path, Path | None]] = [
    (
        "tables_heavy_ml",
        "ML paper, tables-heavy (KOSS state spaces, 28pp)",
        _REPO_ROOT / "eval/side_by_side/tables_heavy_ml_v7/input.pdf",
        # Use v7 output.html as the only available HTML reference.
        # NOT a perfect ground truth (v7 itself over-fires headings),
        # but it's the closest reference we have for this fixture.
        _REPO_ROOT / "eval/side_by_side/tables_heavy_ml_v7/output.html",
    ),
    (
        "math_heavy_short",
        "Math-heavy physics paper (firewall black holes)",
        _REPO_ROOT / "eval/side_by_side/math_heavy_short_v7/input.pdf",
        None,
    ),
    (
        "shades_of_accessibility",
        "Long-prose accessibility whitepaper (legal/policy)",
        _REPO_ROOT / "eval/side_by_side/shades_of_accessibility_v7/input.pdf",
        None,
    ),
    (
        "2209.03909_quantum",
        "ML+physics arxiv paper (structured negativity)",
        _REPO_ROOT / "eval/side_by_side/2209.03909v1_v7adapter/input.pdf",
        None,
    ),
    (
        "0705.1680_bayesian_nn",
        "cs.CE / Bayesian Neural Networks for option pricing",
        _ARXIV_REPO / "other/cs_CE/0705.1680v1_Option Pricing Using Bayesian Neural Networks.pdf",
        None,
    ),
]

_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)


def _count_html_headings(html_path: Path) -> int:
    if not html_path.exists():
        return -1
    html = html_path.read_text(errors="ignore")
    return len(re.findall(r"<h[1-6]\b", html, re.IGNORECASE))


def _measure_one(pdf_path: Path) -> dict:
    """Run the council on one PDF, then count is_heading@threshold for
    each threshold without rebuilding the structure graph.

    We bypass build_structure_graph entirely — heading-pass logic is
    "is_heading top-1 == 'heading' and conf >= threshold" applied
    per-FB, exactly the predicate we're measuring. Avoiding the full
    Stage-5 build lets us reuse a single council run across all
    thresholds.
    """
    from semantik_structure.council.orchestrator import run_council

    t0 = time.time()
    state, regions, feature_blocks = run_council(pdf_path)
    elapsed = time.time() - t0

    # Pull all is_heading TypedSignals indexed by FB index.
    #
    # Stage 5's heading-pass reads the *calibrated* twin (post-hoc
    # temperature applied at inference time; see
    # ``semantik_structure/council/structure.py``). Mirror that here so the
    # heading-rate measurement reflects what Stage 5 actually emits.
    # Fall back to the raw ``is_heading`` signal for pre-calibration
    # checkpoints where the calibrated twin doesn't exist.
    structure_out = (state.outputs or {}).get("structure")
    is_heading_by_fb: dict[int, tuple[str, float]] = {}
    raw_by_fb: dict[int, tuple[str, float]] = {}
    if structure_out is not None:
        for sig in structure_out.signals:
            labels = list(sig.top_k_labels or [])
            confs = list(sig.top_k_confidences or [])
            if not labels:
                continue
            entry = (labels[0], float(confs[0]))
            if sig.head_name == "is_heading_calibrated":
                is_heading_by_fb[int(sig.region_id)] = entry
            elif sig.head_name == "is_heading":
                raw_by_fb[int(sig.region_id)] = entry
    # Backfill from raw if calibrated isn't present (legacy checkpoint).
    for fb_idx, entry in raw_by_fb.items():
        is_heading_by_fb.setdefault(fb_idx, entry)

    n_fb = len(feature_blocks)
    counts: dict[float, int] = {}
    for thr in _THRESHOLDS:
        c = 0
        for fb_idx in range(n_fb):
            label, conf = is_heading_by_fb.get(fb_idx, (None, 0.0))
            if label == "heading" and conf >= thr:
                c += 1
        counts[thr] = c

    # Confidence-distribution stats — useful for deciding where to set
    # the floor.
    heading_confs = [
        conf for (label, conf) in is_heading_by_fb.values()
        if label == "heading"
    ]
    heading_confs.sort()

    return {
        "n_feature_blocks": n_fb,
        "n_regions_table_or_math": len(regions),
        "wall_seconds": elapsed,
        "counts_by_threshold": {f"{t:.1f}": counts[t] for t in _THRESHOLDS},
        "rate_by_threshold": {
            f"{t:.1f}": (counts[t] / max(1, n_fb)) for t in _THRESHOLDS
        },
        "n_signals_with_heading_top1": len(heading_confs),
        "heading_conf_min": heading_confs[0] if heading_confs else None,
        "heading_conf_max": heading_confs[-1] if heading_confs else None,
        "heading_conf_p50": (
            heading_confs[len(heading_confs) // 2]
            if heading_confs else None
        ),
    }


def main() -> None:
    out: dict = {
        "thresholds": list(_THRESHOLDS),
        "pdfs": {},
    }

    for label, desc, pdf_path, ref_html in _PDFS:
        if not pdf_path.exists():
            print(f"[skip] {label}: PDF not found at {pdf_path}",
                  file=sys.stderr)
            continue
        print(f"\n=== {label} :: {pdf_path} ===")
        m = _measure_one(pdf_path)
        if ref_html is not None:
            m["reference_html"] = str(ref_html)
            m["reference_h1_h6_count"] = _count_html_headings(ref_html)
        m["description"] = desc
        m["pdf_path"] = str(pdf_path)
        out["pdfs"][label] = m
        print(f"  n_feature_blocks         : {m['n_feature_blocks']}")
        print(f"  council wall             : {m['wall_seconds']:.2f}s")
        print(f"  counts by threshold      : {m['counts_by_threshold']}")
        print(
            f"  rate by threshold (%)    : "
            f"{ {t: round(100 * m['rate_by_threshold'][t], 2) for t in m['rate_by_threshold']} }"
        )
        if "reference_h1_h6_count" in m:
            print(
                f"  reference HTML headings  : {m['reference_h1_h6_count']} "
                f"(from {ref_html})"
            )

    # Pretty table.
    print()
    print("=" * 78)
    print("Heading-rate (count) by PDF × threshold")
    print("=" * 78)
    header = f"{'pdf':<26} {'n_fb':>5}  " + "  ".join(
        f"thr={t:.1f}" for t in _THRESHOLDS
    )
    print(header)
    print("-" * len(header))
    for label, _desc, _pdf, _ref in _PDFS:
        if label not in out["pdfs"]:
            continue
        m = out["pdfs"][label]
        cells = "  ".join(
            f"{m['counts_by_threshold'][f'{t:.1f}']:>6}" for t in _THRESHOLDS
        )
        print(f"{label:<26} {m['n_feature_blocks']:>5}  {cells}")

    print()
    print("Heading-rate (%) by PDF × threshold")
    print("-" * 78)
    print(header)
    print("-" * len(header))
    for label, _desc, _pdf, _ref in _PDFS:
        if label not in out["pdfs"]:
            continue
        m = out["pdfs"][label]
        cells = "  ".join(
            f"{100 * m['rate_by_threshold'][f'{t:.1f}']:>6.2f}"
            for t in _THRESHOLDS
        )
        print(f"{label:<26} {m['n_feature_blocks']:>5}  {cells}")

    out_path = _REPO_ROOT / "data/eval_reports/stage5_heading_rate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
