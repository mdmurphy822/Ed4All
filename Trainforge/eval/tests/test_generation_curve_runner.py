"""Tests for the W5 C0..C5 generation-quality curve runner (§5 case 10).

Runs the curve on a fake ``eval_fn`` whose candidate quality improves per mode
(C0 worst → C5 best) and asserts the per-arm ``block_prose_grounding_rate`` is
monotone-non-decreasing and the delta table attributes the lift to each
technique. Fakes only — no real model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Trainforge.eval.generation_curve_runner import (  # noqa: E402
    CURVE_SCHEMA_VERSION,
    DEFAULT_MODES,
    run_generation_curve,
)


# Monotone grounding per mode (C0 worst → C5 best).
_MODE_GROUNDING = {"C0": 0.30, "C1": 0.71, "C2": 0.78, "C3": 0.85, "C4": 0.87, "C5": 0.93}
_MODE_OBJ = {"C0": 0.20, "C1": 0.50, "C2": 0.62, "C3": 0.70, "C4": 0.72, "C5": 0.84}
_MODE_CONTRA = {"C0": 7, "C1": 2, "C2": 1, "C3": 0, "C4": 0, "C5": 0}


def _fake_eval(repo_root, course_slug, *, nli=None, embedder=None,
               technique_config="C5", write=False, **kw):
    g = _MODE_GROUNDING[technique_config]
    return {
        "headline": {
            "block_prose_grounding_rate": g,
            "objective_grounding_rate": _MODE_OBJ[technique_config],
            "contradicted_block_count": _MODE_CONTRA[technique_config],
        },
        "success_condition": {"met": g >= 0.90, "pinned_at": "MEASURE-FIRST"},
    }


def test_curve_monotone_and_deltas():
    curve = run_generation_curve(
        _REPO_ROOT, "test-course", eval_fn=_fake_eval, write=False
    )
    assert curve["schema_version"] == CURVE_SCHEMA_VERSION == "1.0"
    modes = [arm["mode"] for arm in curve["arms"]]
    assert modes == DEFAULT_MODES
    rates = [arm["headline"]["block_prose_grounding_rate"] for arm in curve["arms"]]
    # Monotone non-decreasing.
    assert all(b >= a for a, b in zip(rates, rates[1:])), rates

    # Delta table attributes the lift per technique.
    techniques = [d["technique"] for d in curve["deltas"]]
    assert techniques == [
        "chunk_scoped", "free_text_envelope", "self_verify", "refine", "best_of_n_nli"
    ]
    # C0→C1 is the biggest grounding jump (chunk_scoped).
    c0c1 = next(d for d in curve["deltas"] if d["from"] == "C0")
    assert c0c1["block_prose_grounding_rate"] == pytest.approx(0.41)
    assert c0c1["contradicted_block_count"] == -5  # 7 → 2

    # Ceiling reports the C5 number + reserves a Spark row note.
    assert curve["ceiling"]["mode"] == "C5"
    assert curve["ceiling"]["block_prose_grounding_rate"] == pytest.approx(0.93)
    assert "C5-70b" in curve["ceiling"]["note"]


def test_curve_subset_modes():
    curve = run_generation_curve(
        _REPO_ROOT, "test-course", modes=["C1", "C5"], eval_fn=_fake_eval, write=False
    )
    assert [a["mode"] for a in curve["arms"]] == ["C1", "C5"]
    assert len(curve["deltas"]) == 1


def test_curve_reselect_basis_recorded():
    curve = run_generation_curve(
        _REPO_ROOT, "test-course", eval_fn=_fake_eval,
        reselect_from_c3=True, write=False,
    )
    by_mode = {a["mode"]: a["arm_basis"] for a in curve["arms"]}
    assert by_mode["C0"] == "resynthesized"
    assert by_mode["C5"] == "reselected"


def test_curve_writes_file(tmp_path):
    out = tmp_path / "curve.json"
    curve = run_generation_curve(
        _REPO_ROOT, "test-course", eval_fn=_fake_eval,
        write=True, output_path=out,
    )
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "1.0"
    # Schema-validate the curve.
    schema_path = _REPO_ROOT / "schemas" / "eval" / "generation_quality_curve.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(on_disk, schema)
