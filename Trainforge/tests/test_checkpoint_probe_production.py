from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import Trainforge.training.checkpoint_probe as checkpoint_probe_facade
from Trainforge.training.peft_trainer import PEFTTrainer
from Trainforge.training.probes.checkpoint import (
    CourseCheckpointProbe,
    preflight_checkpoint_probe,
)
from Trainforge.training.probes.checkpoint_selection import (
    run_checkpoint_selection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_checkpoint_probe_facade_preserves_symbol_identity():
    assert (
        checkpoint_probe_facade.CourseCheckpointProbe
        is CourseCheckpointProbe
    )
    assert (
        checkpoint_probe_facade.preflight_checkpoint_probe
        is preflight_checkpoint_probe
    )


def test_legacy_checkpoint_probe_module_help_is_cpu_only():
    result = subprocess.run(
        [sys.executable, "-m", "Trainforge.training.checkpoint_probe", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "CPU-only preflight" in result.stdout
    assert "--course-dir" in result.stdout
    assert "--metric" in result.stdout


def _gold_course(root: Path) -> Path:
    course = root / "course"
    path = course / "retrieval_eval" / "gold_set.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "questions": [
            {
                "question_id": "g1",
                "question_text": "Name both primary colors.",
                "expected_key_points": ["red", "blue"],
            },
            {
                "question_id": "g2",
                "question_text": "What shape has three sides?",
                "expected_key_points": ["triangle has three sides"],
            },
        ],
    }))
    return course


def _checkpoint(root: Path, stage: str, step: int) -> Path:
    path = root / ".trainer_checkpoints" / stage / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(f"{stage}-{step}".encode())
    (path / "adapter_config.json").write_text("{}")
    return path


def test_probe_preflight_and_resume_avoid_duplicate_generations(tmp_path):
    course = _gold_course(tmp_path)
    assert preflight_checkpoint_probe(
        course, metric="gold_keypoint_coverage"
    )["gold_key_points"] == 3
    calls = []
    captures = []

    class Capture:
        def log_decision(self, **kwargs):
            captures.append(kwargs)

    def factory(_checkpoint):
        def model(prompt):
            calls.append(prompt)
            return (
                "red and blue"
                if "colors" in prompt
                else "A triangle has three sides."
            )
        return model

    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={},
    )
    checkpoint = _checkpoint(tmp_path, "sft", 1)
    probe = CourseCheckpointProbe(
        course_dir=course,
        stage="sft",
        state_root=tmp_path / "state",
        spec=trainer.spec,
        training_config={"use_4bit": False},
        metric="gold_keypoint_coverage",
        capture=Capture(),
        model_callable_factory=factory,
    )
    assert probe(checkpoint)["gold_keypoint_coverage"] == 1.0
    assert probe(checkpoint)["gold_keypoint_coverage"] == 1.0
    assert len(calls) == 2
    assert len(captures) == 2
    assert all(len(row["rationale"]) >= 20 for row in captures)


def test_improved_dpo_selects_and_promotes_from_isolated_namespace(tmp_path):
    course = _gold_course(tmp_path)
    output = tmp_path / "adapter"
    sft1 = _checkpoint(output, "sft", 10)
    sft2 = _checkpoint(output, "sft", 20)
    dpo1 = _checkpoint(output, "dpo", 30)
    dpo2 = _checkpoint(output, "dpo", 40)
    scores = {
        str(sft1): 0.4, str(sft2): 0.9,
        str(dpo1): 0.95, str(dpo2): 0.7,
    }

    def factory(stage, _root):
        def probe(path):
            assert f"/{stage}/" in str(path)
            return {"gold_keypoint_coverage": scores[str(path)]}
        return probe

    decisions = []

    class Capture:
        def log_decision(self, **kwargs):
            decisions.append(kwargs)

    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={
            "checkpoint_selection_metric": "gold_keypoint_coverage",
            "early_stopping_patience": 2,
        },
        course_dir=course,
        decision_capture=Capture(),
        checkpoint_probe_factory=factory,
    )
    sft_selected = trainer._select_and_promote_checkpoint(
        stage="sft",
        checkpoint_dir=sft1.parent,
        output_dir=output,
    )
    assert sft_selected.read_bytes() == b"sft-20"
    dpo_selected = trainer._select_and_promote_checkpoint(
        stage="dpo",
        checkpoint_dir=dpo1.parent,
        output_dir=output,
    )
    assert dpo_selected.read_bytes() == b"dpo-30"
    assert json.loads(
        (sft1.parent / "checkpoint_selection.json").read_text()
    )["selected_step"] == 20
    assert json.loads(
        (dpo1.parent / "checkpoint_selection.json").read_text()
    )["selected_step"] == 30
    assert len(decisions) == 2
    for event in decisions:
        alternatives = event.get("alternatives_considered") or []
        assert alternatives
        assert all(isinstance(item, dict) for item in alternatives)
        assert all(item.get("option") for item in alternatives)
        assert all(item.get("reason_rejected") for item in alternatives)
        assert any(
            str(signal) in " ".join(
                item["reason_rejected"] for item in alternatives
            )
            for signal in (10, 20, 30, "gold_keypoint_coverage")
        )


@pytest.mark.parametrize(
    ("dpo_score", "accepted"),
    [
        (0.89, False),
        (0.90, True),
        (0.91, True),
    ],
    ids=["regression-rejected", "equal-accepted", "improvement-accepted"],
)
def test_dpo_promotion_requires_selected_sft_baseline_or_better(
    tmp_path, dpo_score, accepted,
):
    output = tmp_path / "adapter"
    sft = _checkpoint(output, "sft", 10)
    dpo = _checkpoint(output, "dpo", 20)
    scores = {str(sft): 0.90, str(dpo): dpo_score}

    def factory(_stage, _root):
        return lambda path: {
            "gold_keypoint_coverage": scores[str(path)]
        }

    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={
            "checkpoint_selection_metric": "gold_keypoint_coverage",
            "early_stopping_patience": 1,
        },
        checkpoint_probe_factory=factory,
    )
    selected_sft = trainer._select_and_promote_checkpoint(
        stage="sft",
        checkpoint_dir=sft.parent,
        output_dir=output,
    )
    assert selected_sft.read_bytes() == b"sft-10"

    if not accepted:
        with pytest.raises(RuntimeError, match="regressed against"):
            trainer._select_and_promote_checkpoint(
                stage="dpo",
                checkpoint_dir=dpo.parent,
                output_dir=output,
            )
        # A rejected DPO candidate must not overwrite the safe SFT adapter.
        assert (output / "adapter_model.safetensors").read_bytes() == b"sft-10"
        return

    selected_dpo = trainer._select_and_promote_checkpoint(
        stage="dpo",
        checkpoint_dir=dpo.parent,
        output_dir=output,
    )
    assert selected_dpo.read_bytes() == b"dpo-20"


def test_patience_stops_selection_window_before_late_recovery(tmp_path):
    for step in (1, 2, 3, 4):
        _checkpoint(tmp_path, "sft", step)
    scores = {1: 0.8, 2: 0.7, 3: 0.6, 4: 0.99}

    selected = run_checkpoint_selection(
        tmp_path / ".trainer_checkpoints" / "sft",
        lambda path: {
            "gold_keypoint_coverage": scores[int(path.name.split("-")[1])]
        },
        metric="gold_keypoint_coverage",
        early_stopping_patience=2,
        strict=True,
    )
    assert selected.step == 1
    report = json.loads(
        (tmp_path / ".trainer_checkpoints" / "sft"
         / "checkpoint_selection.json").read_text()
    )
    assert report["early_stopping_triggered"] is True
    assert report["stopped_after_step"] == 3


@pytest.mark.parametrize(
    "result, match",
    [
        ({}, "missing or non-finite"),
        ({"gold_keypoint_coverage": float("nan")}, "missing or non-finite"),
    ],
)
def test_strict_selection_fails_loud_on_invalid_metric(tmp_path, result, match):
    _checkpoint(tmp_path, "sft", 1)
    with pytest.raises(RuntimeError, match=match):
        run_checkpoint_selection(
            tmp_path / ".trainer_checkpoints" / "sft",
            lambda _path: result,
            metric="gold_keypoint_coverage",
            strict=True,
        )


def test_selection_enabled_without_course_or_factory_fails_loud(tmp_path):
    trainer = PEFTTrainer(
        base_model="qwen2.5-1.5b",
        training_config={
            "checkpoint_selection_metric": "gold_keypoint_coverage",
        },
    )
    with pytest.raises(RuntimeError, match="no course_dir"):
        trainer._select_and_promote_checkpoint(
            stage="sft",
            checkpoint_dir=tmp_path,
            output_dir=tmp_path / "out",
        )
