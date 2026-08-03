"""Checkpoint selection driven by downstream probe metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.training.probes.checkpoint_selection import (  # noqa: E402
    CheckpointProbeResult,
    combine_probe_metrics,
    enumerate_epoch_checkpoints,
    run_checkpoint_selection,
    select_best_checkpoint,
    unwired_probe_runner,
)


def _mk_ckpts(root: Path, steps):
    for s in steps:
        (root / f"checkpoint-{s}").mkdir(parents=True)
    (root / "not-a-checkpoint").mkdir()


def test_enumerate_sorted_by_step(tmp_path):
    _mk_ckpts(tmp_path, [30, 10, 20])
    got = enumerate_epoch_checkpoints(tmp_path)
    assert [s for s, _ in got] == [10, 20, 30]


def test_enumerate_empty_dir(tmp_path):
    assert enumerate_epoch_checkpoints(tmp_path / "nope") == []


def test_select_best_higher_is_better():
    results = [
        CheckpointProbeResult(Path("a"), 10, {"gold_keypoint_coverage": 0.5}),
        CheckpointProbeResult(Path("b"), 20, {"gold_keypoint_coverage": 0.8}),
        CheckpointProbeResult(Path("c"), 30, {"gold_keypoint_coverage": 0.7}),
    ]
    best = select_best_checkpoint(results, metric="gold_keypoint_coverage")
    assert best.step == 20


def test_select_tie_break_prefers_earlier_step():
    results = [
        CheckpointProbeResult(Path("a"), 30, {"gold_keypoint_coverage": 0.8}),
        CheckpointProbeResult(Path("b"), 10, {"gold_keypoint_coverage": 0.8}),
    ]
    best = select_best_checkpoint(results, metric="gold_keypoint_coverage")
    assert best.step == 10  # overfit-preferring tie-break


def test_select_pair_loss_lower_is_better():
    results = [
        CheckpointProbeResult(Path("a"), 10, {"pair_loss": 1.5}),
        CheckpointProbeResult(Path("b"), 20, {"pair_loss": 0.9}),
    ]
    best = select_best_checkpoint(results, metric="pair_loss")
    assert best.step == 20


def test_combine_composite_both_present():
    v = combine_probe_metrics(
        {"gold_keypoint_coverage": 0.8, "sympy_correctness": 0.6}
    )
    assert v == pytest.approx(0.7)


def test_combine_composite_one_present_renormalizes():
    v = combine_probe_metrics({"gold_keypoint_coverage": 0.8})
    assert v == pytest.approx(0.8)


def test_combine_composite_none_present():
    assert combine_probe_metrics({"other": 1.0}) is None


def test_run_selection_writes_report(tmp_path):
    _mk_ckpts(tmp_path, [10, 20])

    def probe(ckpt_dir: Path):
        step = int(ckpt_dir.name.split("-")[1])
        return {"gold_keypoint_coverage": 0.5 if step == 10 else 0.9}

    selected = run_checkpoint_selection(
        tmp_path, probe, metric="gold_keypoint_coverage",
    )
    assert selected.step == 20
    report = json.loads((tmp_path / "checkpoint_selection.json").read_text())
    assert report["selected_step"] == 20
    assert len(report["checkpoints"]) == 2


def test_run_selection_skips_failing_probe(tmp_path):
    _mk_ckpts(tmp_path, [10, 20])

    def probe(ckpt_dir: Path):
        if ckpt_dir.name.endswith("10"):
            raise ValueError("corrupt checkpoint")
        return {"gold_keypoint_coverage": 0.7}

    selected = run_checkpoint_selection(tmp_path, probe)
    assert selected.step == 20


def test_unwired_probe_runner_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        unwired_probe_runner(tmp_path)
