"""Focused contract tests for run-scoped live training telemetry."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from Trainforge.training.telemetry import (
    LATEST_FILENAME,
    STREAM_FILENAME,
    TrainingTelemetryWriter,
    _TrainingTelemetryMixin,
    tokenized_dataset_metrics,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def memory_allocated() -> int:
        return 100

    @staticmethod
    def memory_reserved() -> int:
        return 200

    @staticmethod
    def max_memory_allocated() -> int:
        return 300

    @staticmethod
    def max_memory_reserved() -> int:
        return 400

    @staticmethod
    def mem_get_info() -> tuple[int, int]:
        return 600, 1000


class _Callback(_TrainingTelemetryMixin):
    pass


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_public_schema_tracks_writer_version_and_events() -> None:
    schema_path = (
        Path(__file__).parents[2]
        / "schemas/events/training_telemetry.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == 1
    assert set(schema["properties"]["event"]["enum"]) == {
        "stage_start", "progress", "checkpoint", "stage_end", "selection",
    }


def test_writer_is_versioned_atomic_and_resume_deduplicated(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-contract")
    clock = _Clock()
    writer = TrainingTelemetryWriter(
        tmp_path, stage="sft", clock=clock,
    )
    clock.value += 4
    first = writer.emit(
        "progress",
        status="running",
        global_step=2,
        max_steps=10,
        epoch=0.2,
        metrics={"loss": 1.25},
    )
    assert first is not None
    assert first["schema_version"] == 1
    assert first["run_id"] == "run-contract"
    assert first["metrics"]["eta_seconds"] == 16.0

    resumed_writer = TrainingTelemetryWriter(
        tmp_path, stage="sft", clock=clock,
    )
    assert resumed_writer.emit(
        "progress",
        status="running",
        global_step=2,
        max_steps=10,
        metrics={"loss": 99.0},
    ) is None
    records = _records(tmp_path / STREAM_FILENAME)
    assert records == [first]
    latest = json.loads(
        (tmp_path / LATEST_FILENAME).read_text(encoding="utf-8")
    )
    assert latest == first
    assert not list(tmp_path.glob("*.tmp"))


def test_callback_emits_live_progress_gpu_headroom_and_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-live")
    callback = _Callback()
    callback._init_telemetry(
        run_dir=tmp_path,
        stage="dpo",
        pair_count=24,
        configured_max_seq_length=4096,
        torch_module=SimpleNamespace(cuda=_Cuda()),
        resumed=True,
    )
    state = SimpleNamespace(global_step=4, max_steps=20, epoch=0.5)
    control = SimpleNamespace(should_training_stop=False)

    callback.on_train_begin(None, state, control)
    callback.on_log(
        None,
        state,
        control,
        logs={
            "loss": 0.75,
            "learning_rate": 2e-5,
            "grad_norm": 0.9,
            "tokens_per_second": 1200.0,
            "bad": float("nan"),
        },
    )
    callback.on_save(None, state, control)
    state.global_step = 20
    state.epoch = 2.0
    callback.on_train_end(None, state, control)

    records = _records(tmp_path / STREAM_FILENAME)
    assert [record["event"] for record in records] == [
        "stage_start", "progress", "checkpoint", "stage_end",
    ]
    assert records[0]["metrics"]["pair_count"] == 24
    assert records[0]["metrics"]["resumed_from_checkpoint"] == 1
    progress = records[1]
    assert progress["metrics"]["progress_fraction"] == 0.2
    assert progress["metrics"]["cuda_headroom_fraction"] == 0.6
    assert progress["metrics"]["tokens_per_second"] == 1200.0
    assert "bad" not in progress["metrics"]
    assert records[-1]["status"] == "completed"


def test_callback_marks_interrupted_training_paused(tmp_path: Path) -> None:
    callback = _Callback()
    callback._init_telemetry(
        run_dir=tmp_path,
        stage="sft",
        pair_count=8,
        configured_max_seq_length=2048,
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        ),
        resumed=False,
    )
    state = SimpleNamespace(global_step=3, max_steps=12, epoch=0.25)
    control = SimpleNamespace(should_training_stop=True)
    callback.on_train_end(None, state, control)
    assert _records(tmp_path / STREAM_FILENAME)[0]["status"] == "paused"


def test_tokenized_dataset_metrics_are_exact() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    metrics = tokenized_dataset_metrics(
        tokenizer=Tokenizer(),
        sequences=[
            "one two three",
            "one two three four five",
            "one two three four five six seven eight nine",
        ],
        completion_sequences=["three", "four five", "eight nine"],
        max_seq_length=6,
    )
    assert metrics["sequence_length_p50"] == 5
    assert metrics["sequence_length_p95"] == 9
    assert metrics["sequence_length_max"] == 9
    assert metrics["truncated_examples"] == 1
    assert metrics["non_padding_tokens"] == 14
    assert metrics["padding_fraction"] == 4 / 18
    assert metrics["completion_mask_fraction"] == 5 / 14
