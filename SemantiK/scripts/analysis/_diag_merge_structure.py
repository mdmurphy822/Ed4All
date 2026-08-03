"""Diagnostic: per-line-block council signals to see why paragraph merging fails.

Council-only (no Qwen) — runs run_council and dumps, per FeatureBlock on a
chosen page: structural_role top-1+conf, is_heading top-1+conf, and the
merge_or_split same_logical_block signal for the pair (k, k+1). Shows exactly
where paragraph runs break (is_heading FP / low paragraph conf / not_same).

    PYTHONPATH=. .venv/bin/python -m scripts._diag_merge_structure <pdf> [page]
"""

from __future__ import annotations

import sys
from pathlib import Path

from semantik_structure.council.orchestrator import run_council
from semantik_structure.structure_graph import _get_signal, _top1


def _fb_text(fb) -> str:
    return getattr(fb, "text", None) or (fb.get("text") if isinstance(fb, dict) else "") or ""


def _fb_page(fb) -> int | None:
    for attr in ("page_num", "page", "page_index"):
        v = getattr(fb, attr, None)
        if v is None and isinstance(fb, dict):
            v = fb.get(attr)
        if v is not None:
            return v
    return None


def main() -> None:
    pdf = Path(sys.argv[1])
    want_page = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    state, regions, feature_blocks = run_council(pdf)
    print(f"n feature_blocks: {len(feature_blocks)} | regions: {len(regions)}")

    pages = [_fb_page(fb) for fb in feature_blocks]
    idxs = [i for i, p in enumerate(feature_blocks) if pages[i] == want_page]
    if not idxs:
        idxs = list(range(min(30, len(feature_blocks))))
        print(f"(page {want_page} not matched; showing first {len(idxs)} blocks)")
    print(f"\npage {want_page}: {len(idxs)} blocks\n")
    print(f"{'idx':>4} {'role(conf)':>22} {'isHd(conf)':>14} {'merge→next(conf)':>22}  text")
    for i in idxs:
        rs = _get_signal(state, "structure", "structural_role", i)
        rl, rc = _top1(rs)
        hs = _get_signal(state, "structure", "is_heading", i)
        hl, hc = _top1(hs)
        ms = _get_signal(state, "merge_or_split", "same_logical_block", i)
        ml, mc = _top1(ms) if ms is not None else ("MISSING", 0.0)
        txt = " ".join(_fb_text(feature_blocks[i]).split())[:70]
        flag = ""
        if hl == "heading" or (hl and "head" in str(hl).lower()):
            flag += " <isHEAD>"
        if rl != "paragraph":
            flag += f" <role={rl}>"
        print(
            f"{i:>4} {str(rl) + '(' + format(rc, '.2f') + ')':>22} "
            f"{str(hl) + '(' + format(hc, '.2f') + ')':>14} "
            f"{str(ml) + '(' + format(mc, '.2f') + ')':>22}  {txt}{flag}"
        )


if __name__ == "__main__":
    main()
