"""Synthetic-region smoke for the ITEM1 regroup calibration harness
(``SemantiK/scripts/regroup_calibration_ab.py``).

CPU-only, no PDF, no cascade run beyond the ``block_resegment`` /
``pedagogical_units`` detector SoT. Feeds a hand-built ``region_provenance``
list (one clean 3-region unit + an anchor-free control) through the detector
replay path and asserts the metric fields, plus a direct assertion that the
dual over-merge oracle fires on a seeded unit-start member. Determinism-sensitive
(the report ordering must be stable under any PYTHONHASHSEED).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HARNESS_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "regroup_calibration_ab.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "regroup_calibration_ab", _HARNESS_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HARNESS = _load_harness()


def _prov(first_raw: int, raw_text: str, kind: str = "paragraph") -> dict:
    return {
        "first_raw_block_index": first_raw,
        "raw_text": raw_text,
        "region_kind": kind,
    }


def test_clean_three_region_unit_merges_no_over_merge():
    # EXAMPLE label (anchor) + Solution body + trailing body sentence → one run.
    prov = [
        _prov(0, "EXAMPLE 1.1 Simplify the expression."),
        _prov(1, "Solution First we combine like terms."),
        _prov(2, "Therefore the simplified result is four."),
    ]
    result = HARNESS._analyse_detector("unit.json", prov)

    assert result["anchor_count"] == 1
    assert result["regroup_ops"] == 1
    assert result["regions_folded_hist"] == {"3": 1}
    assert result["n_regions_off"] == 3
    assert result["n_regions_on"] == 1  # 3 → 1 after the fold
    assert result["id_set_delta"] == 2  # two body sourceIds fold away
    assert result["over_merge_flags"] == 0
    assert result["under_merge_off"] == 1  # the label had an absorbable follower
    assert result["under_merge_on"] == 0  # ...and it merged
    assert result["under_merge_reduction"] == 1.0


def test_anchor_free_doc_is_a_no_op():
    prov = [
        _prov(0, "The mitochondrion is the powerhouse of the cell."),
        _prov(1, "It generates most of the cell's supply of ATP."),
        _prov(2, "This process is called cellular respiration."),
    ]
    result = HARNESS._analyse_detector("plain.json", prov)

    assert result["anchor_count"] == 0
    assert result["regroup_ops"] == 0
    assert result["id_set_delta"] == 0
    assert result["n_regions_off"] == result["n_regions_on"] == 3
    assert result["over_merge_flags"] == 0
    assert result["under_merge_off"] == 0


def test_over_merge_oracle_fires_on_seeded_unit_start_member():
    # A non-anchor member whose leading text OPENS a new unit is a wrong fuse.
    from types import SimpleNamespace

    def region(text: str):
        return SimpleNamespace(payload={"text": text})

    # Primary oracle (_pedagogical_class_for) OR the lexicon opener oracle fires.
    assert HARNESS._over_merge_member(region("EXAMPLE 2.3 Solve for x.")) is not None
    assert HARNESS._over_merge_member(region("TRY IT 2.4 Now you try.")) is not None
    # Body / answer openers and plain prose are NOT wrong fuses.
    assert HARNESS._over_merge_member(region("Solution First we factor.")) is None
    assert HARNESS._over_merge_member(region("The sky is blue today.")) is None


def test_rollup_gate_flags():
    per_doc = [
        HARNESS._analyse_detector(
            "unit.json",
            [
                _prov(0, "EXAMPLE 1.1 Simplify the expression."),
                _prov(1, "Solution First we combine like terms."),
            ],
        )
    ]
    roll = HARNESS._rollup(per_doc, "detector")
    assert roll["over_merge_gate_pass"] is True
    assert roll["label_only_reduction_gate_pass"] is True
    assert roll["all_gates_pass"] is True
