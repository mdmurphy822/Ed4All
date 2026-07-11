"""End-to-end smoke for Stage 4 v1 (cross-BERT reranker).

Walks: PDF → ``run_council`` (5 BERTs swapped sequentially on one
shared backbone) → ``arbitrate`` (rule-based §3.2 arbitration) → routing
decisions printed to stdout.

Inline validation (Stage-4 spec — five arbiter unit checks + topo-order
test) runs first; any failure short-circuits before we touch the GPU.

Usage:
    .venv/bin/python scripts/smoke_cross_reranker.py \\
        --pdf eval/side_by_side/tables_heavy_ml_v7/input.pdf \\
        --max-regions-print 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Inline arbiter unit tests — pure-function, no GPU
# ---------------------------------------------------------------------------


def _make_signal(head_name, region_id, top_k_labels, top_k_confidences):
    from semantik_structure.council.types import TypedSignal
    return TypedSignal(
        head_name=head_name,
        region_id=region_id,
        top_k_labels=list(top_k_labels),
        top_k_confidences=list(top_k_confidences),
    )


def _validate_arbiter() -> None:
    from semantik_structure.council.cross_reranker import arbitrate
    from semantik_structure.council.types import (
        BertOutput,
        CouncilState,
        MathCandidate,
        TableCandidate,
    )

    # --- Test 1: confirmed table candidate → route="table" ---
    tc_yes = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 50.0),
        pages=[1],
        evidence_features={"n_rows": 3, "n_cols": 3},
        member_block_indices=[0, 1, 2],
        cell_grid=[["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
        header_row_indices=[0],
        bordered=True,
    )
    structure_signals = [
        _make_signal("table_region", 0, ["table_region", "not_table_region"], [0.9, 0.1]),
        _make_signal("table_region", 1, ["table_region", "not_table_region"], [0.85, 0.15]),
        _make_signal("table_region", 2, ["table_region", "not_table_region"], [0.95, 0.05]),
        _make_signal("structural_role", 0, ["paragraph"], [0.7]),
        _make_signal("structural_role", 1, ["paragraph"], [0.7]),
        _make_signal("structural_role", 2, ["paragraph"], [0.7]),
    ]
    state = CouncilState(outputs={
        "structure": BertOutput(bert_name="structure", signals=structure_signals),
    })
    decisions = arbitrate(state, [tc_yes])
    assert decisions[0].route == "table", decisions[0]
    assert decisions[0].final_role == "table", decisions[0]
    print("  [t1 confirmed-table] PASS")

    # --- Test 2: rejected table candidate → prose + detector_demoted ---
    tc_no_signals = [
        _make_signal("table_region", 0, ["not_table_region", "table_region"], [0.95, 0.05]),
        _make_signal("table_region", 1, ["not_table_region", "table_region"], [0.9, 0.1]),
        _make_signal("structural_role", 0, ["paragraph"], [0.8]),
        _make_signal("structural_role", 1, ["paragraph"], [0.7]),
    ]
    tc_no = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 50.0, 25.0),
        pages=[1],
        evidence_features={"n_rows": 2, "n_cols": 2},
        member_block_indices=[0, 1],
        cell_grid=[["x", "y"], ["z", "w"]],
        header_row_indices=[0],
        bordered=False,
    )
    state2 = CouncilState(outputs={
        "structure": BertOutput(bert_name="structure", signals=tc_no_signals),
    })
    decisions = arbitrate(state2, [tc_no])
    assert decisions[0].route == "prose", decisions[0]
    assert "detector_demoted" in decisions[0].flags, decisions[0]
    print("  [t2 rejected-table] PASS")

    # --- Test 3: math candidate with math_font_frac=0.8 → route="math" ---
    math_yes = MathCandidate(
        kind="math",
        bbox=(0.0, 0.0, 100.0, 30.0),
        pages=[1],
        evidence_features={},
        member_block_indices=[],
        glyph_density_features={"math_font_frac": 0.8, "cid_token_count": 0.0},
    )
    decisions = arbitrate(CouncilState(outputs={}), [math_yes])
    assert decisions[0].route == "math", decisions[0]
    assert decisions[0].final_role == "math", decisions[0]
    print("  [t3 confirmed-math] PASS")

    # --- Test 4: image block at >0.5 confidence → prose + image_block_demoted ---
    # We need a region whose member_block_indices map to Structure
    # signals that say is_image_block. The cleanest way to exercise the
    # image_block_demoted path is to use a TableCandidate that is
    # itself rejected by the table detector (so route becomes prose),
    # while is_image_block fires on the same members.
    img_signals = [
        _make_signal("table_region", 0, ["not_table_region", "table_region"], [0.95, 0.05]),
        _make_signal("is_image_block", 0, ["image_block", "not_image_block"], [0.85, 0.15]),
        _make_signal("structural_role", 0, ["paragraph"], [0.6]),
    ]
    img_region = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 100.0),
        pages=[1],
        evidence_features={"n_rows": 1, "n_cols": 1},
        member_block_indices=[0],
        cell_grid=[["img"]],
        header_row_indices=[0],
        bordered=False,
    )
    state4 = CouncilState(outputs={
        "structure": BertOutput(bert_name="structure", signals=img_signals),
    })
    decisions = arbitrate(state4, [img_region])
    assert decisions[0].route == "prose", decisions[0]
    assert "image_block_demoted" in decisions[0].flags, decisions[0]
    print("  [t4 image-block-demoted] PASS")

    # --- Test 5: Structure heading@0.4 + Semantic body@0.7 ---
    # Demoted table → prose path; then semantic_override fires.
    sig5_struct = [
        _make_signal("table_region", 0, ["not_table_region", "table_region"], [0.99, 0.01]),
        _make_signal("structural_role", 0, ["heading"], [0.4]),
    ]
    sig5_sem = [
        _make_signal("doc_role", 0, ["body"], [0.7]),
    ]
    region5 = TableCandidate(
        kind="table",
        bbox=(0.0, 0.0, 100.0, 50.0),
        pages=[1],
        evidence_features={"n_rows": 1, "n_cols": 1},
        member_block_indices=[0],
        cell_grid=[["x"]],
        header_row_indices=[0],
        bordered=False,
    )
    state5 = CouncilState(outputs={
        "structure": BertOutput(bert_name="structure", signals=sig5_struct),
        "semantic": BertOutput(bert_name="semantic", signals=sig5_sem),
    })
    decisions = arbitrate(state5, [region5])
    assert decisions[0].final_role == "body", decisions[0]
    assert "semantic_override" in decisions[0].flags, decisions[0]
    print("  [t5 semantic-override] PASS")


def _validate_topo_order() -> None:
    from semantik_structure.council.routing import topological_order
    order = topological_order()
    assert len(order) == 5, order
    assert "merge_or_split" in order
    assert "structure" in order
    assert "semantic" in order
    assert "table_specialist" in order
    assert "math_specialist" in order
    # merge_or_split has no upstream → can be at any root position;
    # semantic depends on structure → structure must come first.
    assert order.index("structure") < order.index("semantic"), order
    assert order.index("structure") < order.index("table_specialist"), order
    print(f"  [topo] PASS  order={order}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _summarise(decisions, regions, *, max_print: int) -> dict:
    route_counts: Counter = Counter(d.route for d in decisions)
    flag_counts: Counter = Counter()
    for d in decisions:
        for flag in d.flags:
            flag_counts[flag] += 1

    print()
    print(f"total regions      : {len(decisions)}")
    print(f"route distribution : {dict(route_counts)}")
    print(f"flag frequency     : {dict(flag_counts)}")
    print()
    n = min(len(decisions), max_print)
    print(f"first {n} decisions:")
    print(
        f"  {'rid':>4} {'route':<6} {'final_role':<18} flags / preview"
    )
    for d in decisions[:n]:
        region = regions[d.region_id]
        preview = _region_text_preview(region)
        flags_s = ",".join(d.flags) if d.flags else "-"
        role_s = d.final_role if d.final_role is not None else "-"
        print(
            f"  {d.region_id:>4} {d.route:<6} {role_s:<18} "
            f"[{flags_s}]  {preview!r}"
        )

    return {
        "route_counts": dict(route_counts),
        "flag_counts": dict(flag_counts),
        "n_regions": len(decisions),
    }


def _region_text_preview(region) -> str:
    """One-line preview of a RegionCandidate for printing."""
    feats = getattr(region, "evidence_features", {}) or {}
    if region.kind == "table":
        n_rows = feats.get("n_rows", "?")
        n_cols = feats.get("n_cols", "?")
        return f"<table {n_rows}x{n_cols}>"
    if region.kind == "math":
        src = (feats.get("src_text") or "")[:60].replace("\n", " ")
        return f"<math '{src}'>"
    return f"<{region.kind}>"


def _state_to_jsonable(state) -> dict:
    out_d: dict = {}
    for name, bert_out in state.outputs.items():
        out_d[name] = {
            "bert_name": bert_out.bert_name,
            "backbone_version": bert_out.backbone_version,
            "adapter_version": bert_out.adapter_version,
            "n_signals": len(bert_out.signals),
            "head_names": sorted({s.head_name for s in bert_out.signals}),
        }
    return {
        "document_id": state.document_id,
        "council_version": state.council_version,
        "outputs": out_d,
    }


def _decisions_to_jsonable(decisions) -> list[dict]:
    return [
        {
            "region_id": d.region_id,
            "route": d.route,
            "final_role": d.final_role,
            "flags": list(d.flags),
        }
        for d in decisions
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-regions-print", type=int, default=50)
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("inline validation")
    print("=" * 70)
    _validate_topo_order()
    _validate_arbiter()
    print()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    print("=" * 70)
    print(f"end-to-end smoke: {args.pdf}")
    print("=" * 70)

    from semantik_structure.council.cross_reranker import arbitrate
    from semantik_structure.council.orchestrator import run_council

    t0 = time.time()
    state, regions, _feature_blocks = run_council(
        args.pdf, max_pages=args.max_pages,
    )
    t1 = time.time()
    print(f"[orchestrator] wall: {t1 - t0:.2f}s")
    decisions = arbitrate(state, regions)
    t2 = time.time()
    print(f"[arbitrate]   wall: {t2 - t1:.2f}s")

    summary = _summarise(decisions, regions, max_print=args.max_regions_print)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pdf": str(args.pdf),
            "wall_seconds": {"orchestrator": t1 - t0, "arbitrate": t2 - t1},
            "summary": summary,
            "state": _state_to_jsonable(state),
            "decisions": _decisions_to_jsonable(decisions),
        }
        args.save_json.write_text(json.dumps(payload, indent=2))
        print(f"[save] wrote {args.save_json}")


if __name__ == "__main__":
    main()
