"""Regression coverage for the training-pair synthesis pause boundary.

``GracefulStopRequested`` subclasses ``RuntimeError``.  If the pipeline tool
turns it into an error envelope, the workflow runner can stamp a partially
synthesized phase complete and discard its resumable checkpoint.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from MCP.tools.pipeline_tools import _build_tool_registry
from lib.generation.stop_control import GracefulStopRequested


def test_required_training_route_is_scoped_to_synthesis() -> None:
    config_path = Path(__file__).resolve().parents[2] / "config" / "workflows.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phases = config["workflows"]["textbook_to_course"]["phases"]
    by_name = {phase["name"]: phase for phase in phases}

    expected = {
        "param": "required_training",
        "source": "workflow_params",
        "key": "with_training",
    }
    assert expected in by_name["training_synthesis"]["inputs_from"]
    for phase_name, phase in by_name.items():
        if phase_name == "training_synthesis":
            continue
        assert expected not in (phase.get("inputs_from") or [])


def test_pipeline_synthesizer_propagates_graceful_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir = tmp_path / "trainforge"
    chunks_dir = corpus_dir / "imscc_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    import Trainforge.synthesize_training as synthesis

    def _stop(**_kwargs):
        raise GracefulStopRequested("training_synthesis.pair_loop", 17)

    monkeypatch.setattr(synthesis, "run_synthesis", _stop)

    tool = _build_tool_registry()["synthesize_training"]
    with pytest.raises(GracefulStopRequested) as exc_info:
        asyncio.run(
            tool(
                corpus_dir=str(corpus_dir),
                course_code="FXTEST_101",
                provider="mock",
            )
        )

    assert exc_info.value.site_id == "training_synthesis.pair_loop"


def test_pipeline_synthesizer_propagates_generator_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir = tmp_path / "trainforge"
    chunks_dir = corpus_dir / "imscc_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    import Trainforge.synthesize_training as synthesis

    def _fail(**_kwargs):
        raise ConnectionError("teacher endpoint unavailable")

    monkeypatch.setattr(synthesis, "run_synthesis", _fail)

    tool = _build_tool_registry()["synthesize_training"]
    with pytest.raises(RuntimeError, match="teacher endpoint unavailable"):
        asyncio.run(
            tool(
                corpus_dir=str(corpus_dir),
                course_code="FXTEST_101",
                provider="local",
                required_training=True,
            )
        )


def test_pipeline_synthesizer_rejects_empty_pair_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir = tmp_path / "trainforge"
    chunks_dir = corpus_dir / "imscc_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    specs = corpus_dir / "training_specs"
    specs.mkdir()
    (specs / "instruction_pairs.jsonl").write_text("", encoding="utf-8")
    (specs / "preference_pairs.jsonl").write_text("", encoding="utf-8")

    import Trainforge.synthesize_training as synthesis

    class _Stats:
        instruction_pairs_emitted = 0
        preference_pairs_emitted = 0
        chunks_eligible = 1
        chunks_total = 1

        @staticmethod
        def as_dict():
            return {}

    monkeypatch.setattr(synthesis, "run_synthesis", lambda **_kwargs: _Stats())

    tool = _build_tool_registry()["synthesize_training"]
    with pytest.raises(RuntimeError, match="no usable training corpus"):
        asyncio.run(
            tool(
                corpus_dir=str(corpus_dir),
                course_code="FXTEST_101",
                provider="local",
                required_training=True,
            )
        )


def test_optional_pipeline_synthesizer_preserves_legacy_error_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir = tmp_path / "trainforge"
    chunks_dir = corpus_dir / "imscc_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    import Trainforge.synthesize_training as synthesis

    def _fail(**_kwargs):
        raise ConnectionError("teacher endpoint unavailable")

    monkeypatch.setattr(synthesis, "run_synthesis", _fail)

    tool = _build_tool_registry()["synthesize_training"]
    result = asyncio.run(
        tool(
            corpus_dir=str(corpus_dir),
            course_code="FXTEST_101",
            provider="local",
        )
    )

    assert "synthesize_training failed: teacher endpoint unavailable" in result
