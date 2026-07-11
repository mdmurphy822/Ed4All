"""Smoke-validate the v8 alignment fix on a few representative pairs.

Compares old-version (min_overlap=0.3, no hyphenation handling, no
length sanity) vs new-version (the current build_qwen_data.align logic)
on the same pair files. Reports for each:
  - n_blocks            : feature blocks
  - n_targets_old       : how many got a target under the old rules
  - n_targets_new       : how many under the new rules
  - n_heading_old/new   : how many of those were `heading`
  - suspect_old/new     : heading targets matching the same heuristics
                         used in the v7 audit (lowercase-start,
                         very-long, sentence-with-terminal,
                         page-running-header, toc/page-num tail)

Goal: heading-suspect rate drops from ~50% to <10%; total target
count shouldn't collapse (some loss is expected and good — those
were noise — but a >40% drop in coverage would mean we over-tightened).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantik_structure.extract_shared import extract_shared_cached
from semantik_structure.features import featurize_from_shared
from semantik_structure.text_utils import jaccard_overlap
from data.build_qwen_data import (
    align_to_ground_truth as align_new,
    extract_html_blocks,
)


# Old version of align_to_ground_truth (pre-v8). Kept here as a baseline
# for the smoke comparison so we can quantify the lift the fix gives.
def align_old(features, html_blocks, *, min_overlap: float = 0.3):
    labels: dict[int, str] = {}
    cursor = 0
    window = 8
    for i, fb in enumerate(features):
        text = fb.raw.text
        best_idx, best_score = -1, 0.0
        for j in range(max(0, cursor - 2),
                       min(len(html_blocks), cursor + window)):
            s = jaccard_overlap(text, html_blocks[j][0])
            if s > best_score:
                best_score = s
                best_idx = j
        if best_idx >= 0 and best_score >= min_overlap:
            labels[i] = html_blocks[best_idx][1]
            cursor = max(cursor, best_idx + 1)
    return labels


def heading_suspect(text: str) -> str | None:
    t = text.strip()
    if not t:
        return None
    if (re.search(r"\.\s*\d+\s*$", t)
            or (re.search(r"\d+\s*$", t) and len(t) > 30)):
        return "toc_or_pagenum_tail"
    if re.match(r"^[A-Z][a-zA-Z ]{3,30}\s\d{1,3}$", t):
        return "page_running_header"
    if t[0].islower():
        return "lowercase_start"
    if len(t) >= 100:
        return "very_long"
    if t.rstrip().endswith((".", "!", "?")) and len(t) > 60:
        return "sentence_with_terminal"
    return None


def audit_pair(pair_path: Path) -> dict:
    pair = json.loads(pair_path.read_text())
    output_html = pair.get("output_html") or ""
    if not output_html:
        return {"pair": pair_path.name, "error": "no_output_html"}

    local_pdf = pair.get("local_pdf")
    if local_pdf and Path(local_pdf).exists():
        pdf = Path(local_pdf)
    else:
        from semantik_structure.prerender_cache import cache_path_for
        pdf = cache_path_for(output_html)
        if not pdf.exists():
            return {"pair": pair_path.name, "error": "no_pdf"}

    shared = extract_shared_cached(pdf)
    features = featurize_from_shared(shared)
    html_blocks = extract_html_blocks(output_html)
    if not html_blocks:
        return {"pair": pair_path.name, "error": "no_html_blocks"}

    old_labels = align_old(features, html_blocks)
    new_labels = align_new(features, html_blocks)

    def report(labels, tag):
        n_total = len(labels)
        n_heading = sum(1 for r in labels.values() if r == "heading")
        suspect = Counter()
        sample = {}
        for i, role in labels.items():
            if role != "heading":
                continue
            text = features[i].raw.text or ""
            cat = heading_suspect(text)
            if cat:
                suspect[cat] += 1
                sample.setdefault(cat, []).append(text[:80])
        return {
            f"n_targets_{tag}": n_total,
            f"n_heading_{tag}": n_heading,
            f"suspect_{tag}": dict(suspect),
            f"suspect_total_{tag}": sum(suspect.values()),
            f"suspect_pct_{tag}": (round(100 * sum(suspect.values()) / n_heading, 1)
                                   if n_heading else 0),
            f"sample_{tag}": {k: v[:3] for k, v in sample.items()},
        }

    out = {
        "pair": pair_path.name,
        "n_blocks": len(features),
        "n_html_blocks": len(html_blocks),
    }
    out.update(report(old_labels, "old"))
    out.update(report(new_labels, "new"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", type=Path)
    ap.add_argument("--show-samples", action="store_true",
                    help="Print up to 3 sample texts per suspect category.")
    args = ap.parse_args()

    totals_old = Counter()
    totals_new = Counter()
    for p in args.pairs:
        r = audit_pair(p)
        if "error" in r:
            print(f"[skip] {r['pair']}: {r['error']}")
            continue
        print(f"\n=== {r['pair']} ({r['n_blocks']} blocks, "
              f"{r['n_html_blocks']} html_blocks) ===")
        print(f"  OLD targets={r['n_targets_old']:5d}  "
              f"heading={r['n_heading_old']:4d}  "
              f"suspect={r['suspect_total_old']:4d} "
              f"({r['suspect_pct_old']:5.1f}%)  {r['suspect_old']}")
        print(f"  NEW targets={r['n_targets_new']:5d}  "
              f"heading={r['n_heading_new']:4d}  "
              f"suspect={r['suspect_total_new']:4d} "
              f"({r['suspect_pct_new']:5.1f}%)  {r['suspect_new']}")
        if args.show_samples:
            for tag in ("old", "new"):
                samples = r[f"sample_{tag}"]
                if samples:
                    print(f"  -- {tag} samples --")
                    for cat, texts in samples.items():
                        for t in texts:
                            print(f"     [{tag}/{cat}] {t!r}")

        totals_old["targets"] += r["n_targets_old"]
        totals_old["heading"] += r["n_heading_old"]
        totals_old["suspect"] += r["suspect_total_old"]
        totals_new["targets"] += r["n_targets_new"]
        totals_new["heading"] += r["n_heading_new"]
        totals_new["suspect"] += r["suspect_total_new"]

    print("\n=== aggregate ===")
    for tag, t in (("old", totals_old), ("new", totals_new)):
        sp = (100 * t["suspect"] / t["heading"]) if t["heading"] else 0
        print(f"  {tag.upper():4s}  targets={t['targets']:6d}  "
              f"heading={t['heading']:5d}  "
              f"suspect={t['suspect']:5d}  ({sp:5.1f}%)")

    if totals_old["heading"] and totals_new["targets"]:
        coverage_drop = 100 * (1 - totals_new["targets"] / totals_old["targets"])
        suspect_drop = (100 *
                        (1 - (totals_new["suspect"] / max(1, totals_new["heading"])) /
                         (totals_old["suspect"] / max(1, totals_old["heading"]))))
        print(f"  coverage drop:        {coverage_drop:5.1f}%")
        print(f"  suspect-rate drop:    {suspect_drop:5.1f}%")


if __name__ == "__main__":
    main()
