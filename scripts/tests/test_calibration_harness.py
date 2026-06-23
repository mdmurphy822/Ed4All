"""Unit tests for scripts/calibration_harness.py.

All fixtures are SYNTHESIZED at fixture-load time (a hand-built validation-report dict
and a hand-built block_quality_rollup dict) — NO real course slug / path is referenced,
honoring the tier-2 discipline (tracked test code never pins a corpus slug). The tests
exercise the aggregation + flip heuristic + corpus-identity collapsing logic in
isolation from disk discovery.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "calibration_harness.py"
_spec = importlib.util.spec_from_file_location("calibration_harness", _SCRIPT)
ch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ch
_spec.loader.exec_module(ch)


# --------------------------------------------------------------------------------------
# Synthesized fixtures (NOT a real course)
# --------------------------------------------------------------------------------------
def _make_corpus(corpus_id: str, *, fired_gates: dict[str, int], total_blocks: int):
    """Build a CorpusResult by feeding a synthesized validation-report dict through the
    real reader, so the test exercises the production parse path.

    ``fired_gates`` maps gate_id -> number of blocks on which it FAILS (passed=False).
    """
    per_block = []
    for i in range(total_blocks):
        gate_results = []
        for gid, n_fail in fired_gates.items():
            gate_results.append(
                {
                    "gate_id": gid,
                    "passed": i >= n_fail,  # first n_fail blocks fail this gate
                    "issue_count": 1 if i < n_fail else 0,
                    "action": "regenerate" if i < n_fail else None,
                }
            )
        per_block.append({"block_id": f"blk_{i:03d}", "gate_results": gate_results})
    report_dict = {"per_block": per_block}

    res = ch.CorpusResult(
        corpus_id=corpus_id,
        origin="courseforge_export",
        path=f"/synth/{corpus_id}",
        corpus_key=ch._corpus_key(corpus_id),
    )
    # Drive the REAL production reader against a synthesized on-disk report.
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "report.json"
        rp.write_text(json.dumps(report_dict), encoding="utf-8")
        ch.read_validation_report(res, rp)
    return res


# --------------------------------------------------------------------------------------
# corpus_key derivation
# --------------------------------------------------------------------------------------
def test_corpus_key_strips_timestamp_prefix_and_variant():
    assert ch._corpus_key("PROJ-sample-course-a-20260621094139") == "sample-course-a"
    assert ch._corpus_key("PROJ-OPENSTAX_ALG_9-20260513113631") == "sample-course-a"
    assert ch._corpus_key("sample-course-b") == "sample-course-a"
    assert ch._corpus_key("biology-201") == "biology-201"


def test_corpus_key_dash_underscore_equivalence():
    # The dash vs underscore variants of one corpus must collapse to one key (the bug
    # that originally inflated distinct count to 2).
    assert ch._corpus_key("sample-course-a") == ch._corpus_key("OPENSTAX_ALG_9")


# --------------------------------------------------------------------------------------
# aggregation + flip heuristic
# --------------------------------------------------------------------------------------
def test_single_corpus_is_never_flip_ready():
    # One distinct corpus, even with a perfectly in-band fire rate, can never flip.
    c = _make_corpus(
        "synthcorpus-A-20260101000000",
        fired_gates={"co_terminal_alignment": 0},  # 0 fires -> rate 0.0 (in band)
        total_blocks=20,
    )
    report = ch.aggregate([c])
    assert report["distinct_corpus_count"] == 1
    fam = "WS3 CO<->TO semantic alignment"
    g = report["gate_families"][fam]
    assert g["flip_ready"] is False
    assert "needs >=2 corpora" in g["flip_blocked_reason"]


def test_two_distinct_corpora_in_band_flip_ready():
    # Two DISTINCT corpora, both with fire_rate within the documented band -> flip_ready.
    a = _make_corpus(
        "alpha-corpus-20260101000000",
        fired_gates={"co_terminal_alignment": 0},
        total_blocks=20,
    )
    b = _make_corpus(
        "beta-corpus-20260202000000",
        fired_gates={"co_terminal_alignment": 0},
        total_blocks=20,
    )
    report = ch.aggregate([a, b])
    assert report["distinct_corpus_count"] == 2
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    assert g["fire_rate"] == 0.0
    assert g["expected_band"]["in_band"] is True
    assert g["flip_ready"] is True
    assert g["flip_blocked_reason"] is None


def test_two_distinct_corpora_out_of_band_blocked():
    # Two distinct corpora but a fire_rate ABOVE the band -> not flip_ready, FP-audit msg.
    a = _make_corpus(
        "alpha-corpus-20260101000000",
        fired_gates={"co_terminal_alignment": 18},  # 18/20 = 0.9, way over 0.10 band
        total_blocks=20,
    )
    b = _make_corpus(
        "beta-corpus-20260202000000",
        fired_gates={"co_terminal_alignment": 16},  # 16/20 = 0.8
        total_blocks=20,
    )
    report = ch.aggregate([a, b])
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    assert g["corpora_count"] == 2
    assert g["fire_rate"] > 0.10
    assert g["flip_ready"] is False
    assert "manual FP audit" in g["flip_blocked_reason"]


def test_same_corpus_multiple_runs_count_as_one():
    # Ten timestamped runs of ONE underlying corpus must NOT satisfy the >=2 requirement.
    runs = [
        _make_corpus(
            f"PROJ-mybook-9-full7b-2026010{i}000000",
            fired_gates={"co_terminal_alignment": 0},
            total_blocks=10,
        )
        for i in range(1, 6)
    ]
    report = ch.aggregate(runs)
    assert report["distinct_corpus_count"] == 1
    assert report["contributing_run_count"] == 5
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    assert g["flip_ready"] is False
    assert "needs >=2 corpora" in g["flip_blocked_reason"]


def test_latest_run_is_representative():
    # When runs share a key, the lexicographically-latest (newest timestamp) wins, and
    # the per-corpus row carries the underlying key.
    old = _make_corpus(
        "PROJ-mybook-9-20260101000000",
        fired_gates={"co_terminal_alignment": 9},  # old run: many fires
        total_blocks=10,
    )
    new = _make_corpus(
        "PROJ-mybook-9-20260301000000",
        fired_gates={"co_terminal_alignment": 0},  # new run: clean
        total_blocks=10,
    )
    report = ch.aggregate([old, new])
    assert report["distinct_corpus_count"] == 1
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    # Representative is the NEW (clean) run -> fire_rate 0, not the old run's 0.9.
    assert g["fire_rate"] == 0.0
    assert g["corpora"][0]["corpus"] == "PROJ-mybook-9-20260301000000"


def test_rollup_reader_block_pass_drives_rubric_family():
    # Synthesized block_quality_rollup_report.json -> IB6.1 rubric family fires on
    # block_pass=False rows.
    import json
    import tempfile

    rollup = {
        "schema_version": "1.0",
        "per_block": [
            {"block_id": "b0", "module": "w1", "block_pass": True, "mean": 2.5,
             "weakest_dim": None, "accessibility_gate_fail": False},
            {"block_id": "b1", "module": "w1", "block_pass": False, "mean": 1.2,
             "weakest_dim": "feedback", "accessibility_gate_fail": False},
        ],
    }
    res = ch.CorpusResult(
        corpus_id="synthrollup-A", origin="libv2", path="/synth",
        corpus_key="synthrollup-a",
    )
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "block_quality_rollup_report.json"
        rp.write_text(json.dumps(rollup), encoding="utf-8")
        ch.read_block_quality_rollup(res, rp)
    obs = res.observations["IB6.1 eight-dimension block-quality rubric"]
    assert obs.evaluated == 2
    assert obs.fired == 1  # only b1 failed
    assert abs(obs.fire_rate - 0.5) < 1e-9


def test_report_has_all_gate_families():
    # Every configured framework gate family appears in the report even with no data.
    report = ch.aggregate([])
    assert set(report["gate_families"]) == {f.family for f in ch.GATE_FAMILIES}
    for g in report["gate_families"].values():
        assert g["flip_ready"] is False  # no data -> never ready
