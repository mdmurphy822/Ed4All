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
    .venv/bin/python scripts/analysis/measure_stage5_heading_rate.py \
        --input-manifest <INPUT_MANIFEST>

The manifest is a JSON array of objects with ``label`` and ``pdf`` keys and
optional ``description`` and ``reference_html`` keys. Inputs are deliberately
operator-supplied; this tracked diagnostic contains no corpus paths.
"""
from __future__ import annotations

import json
import argparse
import re
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)


def _load_inputs(path: Path) -> list[tuple[str, str, Path, Path | None]]:
    """Load and validate explicit diagnostic inputs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("--input-manifest must contain a non-empty JSON array")
    rows = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not item.get("label") or not item.get("pdf"):
            raise ValueError(f"manifest row {index} requires label and pdf")
        pdf = Path(item["pdf"]).expanduser()
        if not pdf.is_file():
            raise FileNotFoundError(f"manifest PDF does not exist: {pdf}")
        ref = Path(item["reference_html"]).expanduser() if item.get("reference_html") else None
        if ref is not None and not ref.is_file():
            raise FileNotFoundError(f"manifest reference HTML does not exist: {ref}")
        rows.append((str(item["label"]), str(item.get("description", "")), pdf, ref))
    return rows


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "data/eval_reports/stage5_heading_rate.json")
    args = parser.parse_args(argv)
    pdfs = _load_inputs(args.input_manifest)
    out: dict = {
        "thresholds": list(_THRESHOLDS),
        "pdfs": {},
    }

    for label, desc, pdf_path, ref_html in pdfs:
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
    for label, _desc, _pdf, _ref in pdfs:
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
    for label, _desc, _pdf, _ref in pdfs:
        if label not in out["pdfs"]:
            continue
        m = out["pdfs"][label]
        cells = "  ".join(
            f"{100 * m['rate_by_threshold'][f'{t:.1f}']:>6.2f}"
            for t in _THRESHOLDS
        )
        print(f"{label:<26} {m['n_feature_blocks']:>5}  {cells}")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
