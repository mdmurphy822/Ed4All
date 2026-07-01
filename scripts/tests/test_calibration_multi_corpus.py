"""W8.3 unit tests — multi-corpus discovery + across-corpora aggregate.

Hermetic: the >=2-corpus aggregation path is exercised against a SYNTHETIC fixture
built at test time by ``scripts/tests/fixtures/build_calibration_corpora.py`` (invented,
generic strings — NO real course slug/path/prose). No real LibV2/Courseforge corpus is
read; the ``--runs-dir`` / ``extra_roots`` overrides keep discovery inside ``tmp_path``.
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

_BUILDER = (
    Path(__file__).resolve().parent / "fixtures" / "build_calibration_corpora.py"
)
_bspec = importlib.util.spec_from_file_location("build_calibration_corpora", _BUILDER)
builder = importlib.util.module_from_spec(_bspec)
sys.modules[_bspec.name] = builder
_bspec.loader.exec_module(builder)


_UDL_FAMILY = "IB4 UDL multiple-means coverage"


# --------------------------------------------------------------------------------------
# Fixture builder — two distinct synthetic corpora
# --------------------------------------------------------------------------------------
def test_builder_creates_two_distinct_corpora(tmp_path):
    dest = builder.build(tmp_path / "fix")
    corpora = ch.discover_corpora(course_filter=None, runs_dir=dest)
    keys = {c.corpus_key for c in corpora}
    assert builder.ALPHA_CORPUS_KEY in keys
    assert builder.BETA_CORPUS_KEY in keys
    report = ch.aggregate(corpora)
    assert report["distinct_corpus_count"] == 2
    assert report["sufficient_corpora"] is True


def test_builder_is_deterministic(tmp_path):
    a = builder.build(tmp_path / "a")
    b = builder.build(tmp_path / "b")
    ra = (a / "synth-alpha-export" / "04_rewrite" / "02_validation_report" / "report.json").read_text()
    rb = (b / "synth-alpha-export" / "04_rewrite" / "02_validation_report" / "report.json").read_text()
    assert ra == rb  # regenerable, byte-stable


# --------------------------------------------------------------------------------------
# Across-corpora aggregate (W8.3)
# --------------------------------------------------------------------------------------
def test_across_corpora_aggregate_max_and_min(tmp_path):
    dest = builder.build(tmp_path / "fix")
    corpora = ch.discover_corpora(course_filter=None, runs_dir=dest)
    report = ch.aggregate(corpora)
    g = report["gate_families"][_UDL_FAMILY]
    ac = g["across_corpora"]
    assert ac["n_corpora"] == 2
    # alpha fires 1/10 = 0.1, beta 0/10 = 0.0 -> max 0.1, min 0.0, mean 0.05.
    assert abs(ac["max_fire_rate"] - 0.1) < 1e-9
    assert ac["min_fire_rate"] == 0.0
    assert abs(ac["mean_fire_rate"] - 0.05) < 1e-9
    # worst corpus is the alpha export (the one that fired).
    assert ac["worst_corpus"] is not None
    assert "alpha" in ac["worst_corpus"]


def test_across_corpora_present_for_every_family_even_empty():
    report = ch.aggregate([])
    for g in report["gate_families"].values():
        ac = g["across_corpora"]
        assert ac["n_corpora"] == 0
        assert ac["max_fire_rate"] == 0.0
        assert ac["worst_corpus"] is None


def test_pooled_in_band_but_worst_corpus_spike_blocks_flip():
    # A gate clean POOLED (well inside band) but spiking on ONE corpus must NOT be
    # flip_ready — the worst-corpus rate is the FP signal. udl_coverage band is [0, 0.10].
    # corpus A: 1/1000 blocks fire (0.001). corpus B: 5/10 fire (0.5, out of band).
    # Pooled = 6/1010 ~= 0.0059 (in band), but worst = 0.5 (out) -> blocked.
    a = _make_udl_corpus("alpha-corpus-20260101000000", fired=1, total=1000)
    b = _make_udl_corpus("beta-corpus-20260202000000", fired=5, total=10)
    report = ch.aggregate([a, b])
    g = report["gate_families"][_UDL_FAMILY]
    assert g["corpora_count"] == 2
    assert g["expected_band"]["in_band"] is True  # pooled is in band
    assert g["across_corpora"]["max_in_band"] is False
    assert g["flip_ready"] is False
    assert "worst-corpus" in g["flip_blocked_reason"]


def test_two_clean_corpora_flip_ready_max_in_band(tmp_path):
    dest = builder.build(tmp_path / "fix")
    corpora = ch.discover_corpora(course_filter=None, runs_dir=dest)
    report = ch.aggregate(corpora)
    g = report["gate_families"][_UDL_FAMILY]
    # Both corpora within the [0, 0.10] band (0.1 and 0.0) -> flip_ready.
    assert g["across_corpora"]["max_in_band"] is True
    assert g["flip_ready"] is True
    assert g["flip_blocked_reason"] is None


# --------------------------------------------------------------------------------------
# extra_roots discovery + env knob
# --------------------------------------------------------------------------------------
def test_extra_roots_discovered_hermetically(tmp_path):
    fixture = builder.build(tmp_path / "fix")
    empty = tmp_path / "empty"
    empty.mkdir()
    # runs_dir=empty skips the default LibV2/Courseforge scan (stays hermetic); the
    # extra_roots union brings in the fixture's two exports.
    corpora = ch.discover_corpora(
        course_filter=None, runs_dir=empty, extra_roots=[fixture]
    )
    keys = {c.corpus_key for c in corpora}
    assert keys == {builder.ALPHA_CORPUS_KEY, builder.BETA_CORPUS_KEY}


def test_resolve_extra_corpora_roots_env(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv(
        ch.ENV_EXTRA_CORPORA, f"{real}{__import__('os').pathsep}{missing}"
    )
    roots = ch.resolve_extra_corpora_roots()
    assert roots == [real.resolve()]  # garbage/missing path dropped (parse-with-fallback)


def test_resolve_extra_corpora_roots_unset(monkeypatch):
    monkeypatch.delenv(ch.ENV_EXTRA_CORPORA, raising=False)
    assert ch.resolve_extra_corpora_roots() == []


def test_resolve_extra_corpora_roots_explicit_wins(tmp_path, monkeypatch):
    env_dir = tmp_path / "env_dir"
    env_dir.mkdir()
    cli_dir = tmp_path / "cli_dir"
    cli_dir.mkdir()
    monkeypatch.setenv(ch.ENV_EXTRA_CORPORA, str(env_dir))
    # Explicit (CLI) list wins over the env var.
    roots = ch.resolve_extra_corpora_roots([str(cli_dir)])
    assert roots == [cli_dir.resolve()]


# --------------------------------------------------------------------------------------
# --max-corpora cap
# --------------------------------------------------------------------------------------
def test_max_corpora_cap_keeps_highest_coverage():
    corpora = [
        _make_udl_corpus(f"corpus-{c}-2026010{i}000000", fired=0, total=10)
        for i, c in enumerate("abc", start=1)
    ]
    uncapped = ch.aggregate(corpora)
    assert uncapped["distinct_corpus_count"] == 3
    capped = ch.aggregate(corpora, max_corpora=2)
    assert capped["distinct_corpus_count"] == 2
    # Non-positive cap is ignored (parse-with-fallback).
    assert ch.aggregate(corpora, max_corpora=0)["distinct_corpus_count"] == 3


# --------------------------------------------------------------------------------------
# helper — build a CorpusResult via the REAL reader from a synthesized report
# --------------------------------------------------------------------------------------
def _make_udl_corpus(corpus_id: str, *, fired: int, total: int):
    import json
    import tempfile

    per_block = []
    for i in range(total):
        per_block.append(
            {
                "block_id": f"blk_{i:04d}",
                "gate_results": [
                    {
                        "gate_id": "udl_coverage",
                        "passed": i >= fired,
                        "issue_count": 1 if i < fired else 0,
                        "action": "regenerate" if i < fired else None,
                    }
                ],
            }
        )
    res = ch.CorpusResult(
        corpus_id=corpus_id,
        origin="courseforge_export",
        path=f"/synth/{corpus_id}",
        corpus_key=ch._corpus_key(corpus_id),
    )
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "report.json"
        rp.write_text(json.dumps({"per_block": per_block}), encoding="utf-8")
        ch.read_validation_report(res, rp)
    return res
