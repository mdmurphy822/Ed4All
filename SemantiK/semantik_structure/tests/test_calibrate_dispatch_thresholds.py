"""ITEM6 Phase 3 — dispatch-threshold calibration harness (CPU-only, no model).

Loads ``SemantiK/scripts/calibration/calibrate_dispatch_thresholds.py`` by path and pins the
sweep math at two hand-computed grid points + the pre-ITEM6 (no role_top_k)
non-zero exit contract + tau-monotonicity of dispatch_frac.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "calibration"
    / "calibrate_dispatch_thresholds.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("calibrate_dispatch_thresholds", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


# Synthetic doc: 3 regions.
#   R0 top1=0.9 top2=0.8 (margin 0.1) — CORRECTED.
#   R1 top1=0.4 top2=0.3 (margin 0.1) — not corrected.
#   R2 top1=0.95 top2=0.05 (margin 0.9) — not corrected.
_PROV = {
    "region_provenance": [
        {"region_index": 0, "role_top_k": [["heading", 0.9], ["paragraph", 0.8]]},
        {"region_index": 1, "role_top_k": [["code_block", 0.4], ["paragraph", 0.3]]},
        {"region_index": 2, "role_top_k": [["paragraph", 0.95], ["list_item", 0.05]]},
    ]
}
_REVIEW = {
    "structure_review": [
        {"block_id": 0, "verdict": "corrected", "kind_before": "heading", "kind_after": "paragraph"},
        {"block_id": 1, "verdict": "ok", "kind_before": "code_block", "kind_after": "code_block"},
        {"block_id": 2, "verdict": "ok", "kind_before": "paragraph", "kind_after": "paragraph"},
    ]
}


def _rows_by_grid(rows):
    return {(tau, delta): (frac, rec, prec) for tau, delta, frac, rec, prec in rows}


def test_sweep_two_grid_points(tmp_path):
    mod = _load_module()
    prov = _write(tmp_path, "prov.json", _PROV)
    rev = _write(tmp_path, "review.json", _REVIEW)
    rows = mod._collect([prov], [rev])
    grid = _rows_by_grid(mod._sweep(rows))

    # tau=0.50, delta=0.15: R0 dispatches (margin<0.15, corrected), R1 dispatches
    # (top1<0.5, not corrected), R2 does not. dispatched=2, hit=1, pos=1, total=3.
    frac, rec, prec = grid[(0.50, 0.15)]
    assert round(frac, 4) == round(2 / 3, 4)
    assert rec == 1.0
    assert prec == 0.5

    # tau=0.30, delta=0.05: nothing clears either arm -> empty dispatch set.
    frac, rec, prec = grid[(0.30, 0.05)]
    assert frac == 0.0
    assert rec == 0.0
    assert prec == 0.0


def test_dispatch_frac_monotone_in_tau(tmp_path):
    mod = _load_module()
    prov = _write(tmp_path, "prov.json", _PROV)
    rev = _write(tmp_path, "review.json", _REVIEW)
    rows = mod._collect([prov], [rev])
    grid = _rows_by_grid(mod._sweep(rows))
    # Holding delta fixed, dispatch_frac is non-decreasing in tau (top1<tau grows).
    for delta in (0.05, 0.15, 0.40):
        prev = -1.0
        for tau in mod._TAUS:
            frac = grid[(tau, delta)][0]
            assert frac >= prev - 1e-9, f"non-monotone at delta={delta}"
            prev = frac


def test_missing_role_top_k_exits_nonzero(tmp_path):
    mod = _load_module()
    prov = _write(
        tmp_path, "prov.json", {"region_provenance": [{"region_index": 0}]}
    )
    rev = _write(tmp_path, "review.json", {"structure_review": []})
    rc = mod.main(["--provenance", prov, "--review-audit", rev])
    assert rc == 1


def test_paired_arg_count_mismatch_exits_two(tmp_path):
    mod = _load_module()
    prov = _write(tmp_path, "prov.json", _PROV)
    rev = _write(tmp_path, "review.json", _REVIEW)
    # Two provenance paths, one review-audit -> unpaired -> exit 2 (no crash).
    rc = mod.main(["--provenance", prov, "--provenance", prov, "--review-audit", rev])
    assert rc == 2
