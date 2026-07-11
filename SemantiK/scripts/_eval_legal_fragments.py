"""Real-runtime FRAGMENT A/B for the Structure head (Plan 14 §6 gate 3).

Council-only (no Qwen): runs ``run_council`` on each held-out opinion and
counts, per doc, the structure-graph output:
  - FRAGMENT = a ``paragraph`` Region whose text starts lowercase (a
    continuation that should have merged — the loop's quality metric).
  - headings = count of ``heading`` Regions (must NOT drop vs baseline —
    Plan 14 forbids trading heading recall for fewer fragments).

Point the council at a candidate head via the env override added to
``council/structure.py``:

    # shipped head (baseline)
    PYTHONPATH=. .venv/bin/python -m scripts._eval_legal_fragments
    # new legal head
    DART_STRUCTURE_ADAPTER_DIR=models/council/structure_legal/final \
      PYTHONPATH=. .venv/bin/python -m scripts._eval_legal_fragments

Deterministic council inference → run once per head and diff.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.orchestrator import run_council
from semantik_structure.structure_graph import build_structure_graph

# The 9 held-out eval opinions (mirrors the constructor --holdout; never trained).
HOLDOUT = [
    "opinion_11297311",
    "opinion_11298455",
    "opinion_11320152",
    "opinion_11313568",
    "opinion_11315258",
    "opinion_11314119",
    "opinion_11297479",
    "opinion_11311457",
    "opinion_11314738",
]


def _region_text(region, feature_blocks) -> str:
    txt = region.payload.get("text") if isinstance(region.payload, dict) else None
    if not txt:
        parts = []
        for i in region.feature_block_indices:
            if 0 <= i < len(feature_blocks):
                fb = feature_blocks[i]
                parts.append(getattr(fb, "text", "") or "")
        txt = " ".join(parts)
    return " ".join((txt or "").split())


def _starts_lower(text: str) -> bool:
    t = text.lstrip(" \"'“”‘’([")
    return bool(t) and t[0].islower()


def main() -> None:
    pdf_dir = Path("data/cache/courtlistener")
    stems = sys.argv[1:] or HOLDOUT
    head = os.environ.get("DART_STRUCTURE_ADAPTER_DIR", "models/council/structure/final")
    print(f"=== FRAGMENT A/B  head={head} ===")
    print(f"{'opinion':>20} {'regions':>8} {'headings':>9} {'FRAGMENTS':>10}")
    totals = {"regions": 0, "headings": 0, "frags": 0}
    detail: list[tuple[str, list[str]]] = []
    heading_dump: list[tuple[str, list[str]]] = []
    for stem in stems:
        pdf = pdf_dir / f"{stem}.pdf"
        if not pdf.exists():
            print(f"{stem:>20}  (missing pdf)")
            continue
        state, council_regions, feature_blocks = run_council(pdf)
        decisions = arbitrate(state, council_regions)
        regions = build_structure_graph(state, feature_blocks, council_regions, decisions)
        heading_texts = [
            _region_text(r, feature_blocks)[:60] for r in regions if r.kind == "heading"
        ]
        headings = len(heading_texts)
        frags = []
        for r in regions:
            if r.kind != "paragraph":
                continue
            t = _region_text(r, feature_blocks)
            if _starts_lower(t):
                frags.append(t[:55])
        print(f"{stem:>20} {len(regions):>8} {headings:>9} {len(frags):>10}")
        totals["regions"] += len(regions)
        totals["headings"] += headings
        totals["frags"] += len(frags)
        if frags:
            detail.append((stem, frags))
        heading_dump.append((stem, heading_texts))
    print(f"{'TOTAL':>20} {totals['regions']:>8} {totals['headings']:>9} {totals['frags']:>10}")
    print("\n--- HEADING texts (check for real-vs-FP suppression) ---")
    for stem, hs in heading_dump:
        print(f"  {stem} ({len(hs)}):")
        for h in hs:
            print(f"      H: {h!r}")
    print("\n--- fragment texts ---")
    for stem, frags in detail:
        for f in frags:
            print(f"  {stem}: {f!r}")


if __name__ == "__main__":
    main()
