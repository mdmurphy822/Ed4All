"""Shadow-run BERT-Structure (v2, 4-head) vs DistilBERT classifier_v5
(v1) on held-out arXiv PDFs.

For each PDF:
    1. extract_shared → shared JSON (cached)
    2. featurize_from_shared → list[FeatureBlock]
    3. v1: classify.classify_blocks(features, distilbert_classifier_v5)
       → role per block
    4. v2: dart_semantic.council.structure.run_inputs(adapter, features)
       → 4 typed signals per block (structural_role, is_heading,
       table_region, list_nesting)

Then per-PDF reports:
    * agreement: how often v1.role == v2.structural_role
    * v2-exclusive signals (binary heads v1 didn't have):
        is_heading=1 count, table_region=1 count, list_nesting>0 count
    * disagreement table: top-N v1↔v2 confusion buckets with text
      samples for spot-checking

Usage:
    python scripts/shadow_run_structure.py \\
        --pdfs path/to/pdf1.pdf path/to/pdf2.pdf ... \\
        --out-dir eval/results/structure_v2_shadow
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dart_semantic import classify
from dart_semantic.extract_shared import extract_shared_cached
from dart_semantic.features import featurize_from_shared
from dart_semantic.council.base import (
    load_lora_adapter,
    load_shared_backbone,
)
from dart_semantic.council import structure as bert_structure


V1_ADAPTER_PATH = Path("models/classifier_v5/final")
V2_ADAPTER_PATH = Path("models/council/structure/final")


def run_v1(features) -> list[str]:
    """Run DistilBERT classifier_v5 over the feature blocks."""
    model = classify.load_classifier(V1_ADAPTER_PATH)
    classified = classify.classify_blocks(features, model=model, batch_size=64)
    return [c.role for c in classified]


def run_v2(features) -> dict[str, list[str]]:
    """Run BERT-Structure (v2, 4-head). Returns per-head argmax labels
    keyed by head name."""
    spec = bert_structure.ADAPTER_SPEC
    backbone = load_shared_backbone()
    adapter = load_lora_adapter(spec, backbone)
    out = bert_structure.run_inputs(adapter, features)
    # 4 signals per region: structural_role, is_heading, table_region,
    # list_nesting — collect by region_id, head.
    by_region: dict[int, dict[str, str]] = defaultdict(dict)
    for sig in out.signals:
        if sig.top_k_labels:
            by_region[sig.region_id][sig.head_name] = sig.top_k_labels[0]
    n = len(features)
    role = [by_region[i].get("structural_role", "?") for i in range(n)]
    is_heading = [by_region[i].get("is_heading", "?") for i in range(n)]
    table_region = [by_region[i].get("table_region", "?") for i in range(n)]
    list_nesting = [by_region[i].get("list_nesting", "?") for i in range(n)]
    return {
        "structural_role": role,
        "is_heading": is_heading,
        "table_region": table_region,
        "list_nesting": list_nesting,
    }


def _normalize_v2_role_for_compare(v2_role: str) -> str:
    """v2's structural_role tags use the 6-class subset. v1 emits the
    full 21-class Role enum. For agreement-counting we map v1's
    table-cell roles to "in_table" stand-in (v2 doesn't try to
    classify them) — those rows don't count for agreement, they're
    measured by the table_region binary head instead. v1 'paragraph' /
    'heading' / 'list_item' / 'blockquote' / 'code_block' /
    'form_label' map directly. Anything else from v1 we mark as
    "other_v1" so we can see how often v1 emits dead-class roles."""
    return v2_role  # identity — comparison done in compare_one_pdf


V1_OVERLAP = {
    "paragraph", "heading", "list_item",
    "blockquote", "code_block", "form_label",
}
V1_TABLE_FAMILY = {
    "table", "table_row", "table_header_cell",
    "table_data_cell", "table_caption",
}


def compare_one_pdf(pdf_path: Path, out_dir: Path) -> dict:
    """Run both models on one PDF and produce a per-doc comparison
    report dict."""
    print(f"[{pdf_path.name}] extract...", file=sys.stderr, flush=True)
    t0 = time.time()
    shared = extract_shared_cached(pdf_path)
    features = featurize_from_shared(shared)
    print(f"  {len(features)} feature blocks  ({time.time()-t0:.1f}s)",
          file=sys.stderr)

    print(f"[{pdf_path.name}] v1 (DistilBERT classifier_v5)...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    v1_roles = run_v1(features)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

    print(f"[{pdf_path.name}] v2 (BERT-Structure 4-head)...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    v2_out = run_v2(features)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

    n = len(features)

    # ---- Agreement on overlap classes ----
    overlap_total = 0
    overlap_agree = 0
    confusions: Counter = Counter()
    overlap_samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for i in range(n):
        v1r = v1_roles[i]
        v2r = v2_out["structural_role"][i]
        # v1's table-family: skip from role agreement — v2 measures it
        # via table_region binary
        if v1r in V1_TABLE_FAMILY:
            continue
        # only score where v1's emission is in the v2 vocab
        if v1r not in V1_OVERLAP:
            continue
        overlap_total += 1
        if v1r == v2r:
            overlap_agree += 1
        else:
            confusions[(v1r, v2r)] += 1
            if len(overlap_samples[(v1r, v2r)]) < 3:
                txt = (getattr(features[i].raw, "text", "") or "")[:160]
                overlap_samples[(v1r, v2r)].append({
                    "block_idx": i,
                    "text": txt,
                })

    # ---- v2-exclusive binary head counts (signals v1 doesn't have) ----
    is_heading_pos = sum(1 for v in v2_out["is_heading"] if v == "heading")
    table_region_pos = sum(
        1 for v in v2_out["table_region"] if v == "table_region")
    list_nest_dist = Counter(v2_out["list_nesting"])

    # ---- v1 role distribution ----
    v1_role_dist = Counter(v1_roles)
    v2_role_dist = Counter(v2_out["structural_role"])

    # ---- v1 table family vs v2 table_region cross-reference ----
    v1_table_count = sum(1 for r in v1_roles if r in V1_TABLE_FAMILY)
    # Where v1 says "table_*", does v2 agree it's table_region?
    v1_table_v2_agree = 0
    for i in range(n):
        if v1_roles[i] in V1_TABLE_FAMILY:
            if v2_out["table_region"][i] == "table_region":
                v1_table_v2_agree += 1

    return {
        "pdf": str(pdf_path),
        "n_blocks": n,
        "overlap_total": overlap_total,
        "overlap_agree": overlap_agree,
        "overlap_rate": (overlap_agree / overlap_total) if overlap_total else 0.0,
        "v1_role_dist": dict(v1_role_dist),
        "v2_role_dist": dict(v2_role_dist),
        "is_heading_pos": is_heading_pos,
        "table_region_pos": table_region_pos,
        "list_nesting_dist": dict(list_nest_dist),
        "v1_table_count": v1_table_count,
        "v1_table_v2_table_region_agree": v1_table_v2_agree,
        "top_confusions": [
            {"v1": v1, "v2": v2, "n": n,
             "samples": overlap_samples[(v1, v2)]}
            for (v1, v2), n in confusions.most_common(10)
        ],
    }


def render_report(reports: list[dict]) -> str:
    """Pretty-print a summary table across all PDFs."""
    lines = []
    lines.append("# BERT-Structure v2 (4-head) vs DistilBERT classifier_v5 — Shadow Run")
    lines.append("")
    lines.append(f"PDFs: {len(reports)}")
    lines.append("")

    lines.append("## Per-PDF agreement on overlap classes (paragraph, heading, list_item, blockquote, code_block, form_label)")
    lines.append("")
    lines.append("| PDF | blocks | overlap_n | agree | rate |")
    lines.append("|---|---|---|---|---|")
    total_blocks = total_overlap = total_agree = 0
    for r in reports:
        name = Path(r["pdf"]).stem
        if len(name) > 50:
            name = name[:47] + "..."
        lines.append(
            f"| {name} | {r['n_blocks']} | {r['overlap_total']} | "
            f"{r['overlap_agree']} | {r['overlap_rate']:.3f} |"
        )
        total_blocks += r["n_blocks"]
        total_overlap += r["overlap_total"]
        total_agree += r["overlap_agree"]
    rate = total_agree / total_overlap if total_overlap else 0.0
    lines.append(
        f"| **all** | **{total_blocks}** | **{total_overlap}** | "
        f"**{total_agree}** | **{rate:.3f}** |"
    )
    lines.append("")

    lines.append("## v2-exclusive signals (heads v1 doesn't have)")
    lines.append("")
    lines.append("| PDF | is_heading=1 | table_region=1 | depth>0 |")
    lines.append("|---|---|---|---|")
    for r in reports:
        name = Path(r["pdf"]).stem
        if len(name) > 50:
            name = name[:47] + "..."
        depth_pos = sum(v for k, v in r["list_nesting_dist"].items()
                        if k != "depth_0")
        lines.append(
            f"| {name} | {r['is_heading_pos']} | "
            f"{r['table_region_pos']} | {depth_pos} |"
        )
    lines.append("")

    lines.append("## Cross-reference: v1 table family vs v2 table_region head")
    lines.append("")
    lines.append("| PDF | v1.table* | v2.table_region | overlap |")
    lines.append("|---|---|---|---|")
    for r in reports:
        name = Path(r["pdf"]).stem
        if len(name) > 50:
            name = name[:47] + "..."
        lines.append(
            f"| {name} | {r['v1_table_count']} | "
            f"{r['table_region_pos']} | "
            f"{r['v1_table_v2_table_region_agree']} |"
        )
    lines.append("")

    lines.append("## Top disagreements (v1 ≠ v2 on overlap classes)")
    lines.append("")
    for r in reports:
        name = Path(r["pdf"]).stem
        if len(name) > 50:
            name = name[:47] + "..."
        if not r["top_confusions"]:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for conf in r["top_confusions"][:5]:
            lines.append(
                f"- **{conf['v1']} → {conf['v2']}**: {conf['n']}× "
            )
            for s in conf["samples"][:2]:
                lines.append(f"  - block {s['block_idx']}: "
                             f"`{s['text']}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="+", required=True,
                    help="Held-out PDFs to shadow-run")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("eval/results/structure_v2_shadow"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for pdf in args.pdfs:
        report = compare_one_pdf(Path(pdf), args.out_dir)
        reports.append(report)
        # Per-PDF JSON for drilldown
        out_json = args.out_dir / f"{Path(pdf).stem[:50]}.json"
        out_json.write_text(json.dumps(report, indent=2))
        print(f"  wrote {out_json}", file=sys.stderr)

    md = render_report(reports)
    summary = args.out_dir / "summary.md"
    summary.write_text(md)
    print(f"\n[done] summary -> {summary}", file=sys.stderr)
    print(md)


if __name__ == "__main__":
    main()
