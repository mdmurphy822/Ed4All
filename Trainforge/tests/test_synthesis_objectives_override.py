"""Regression tests for the opt-in synthesis objectives-path override."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from Trainforge.synthesize_training import run_synthesis


class _PreflightObserved(RuntimeError):
    """Stop immediately after recording the holdout preflight arguments."""


def _course(tmp_path: Path) -> Path:
    course = tmp_path / "workflow-course"
    chunks = course / "imscc_chunks" / "chunks.jsonl"
    chunks.parent.mkdir(parents=True)
    chunks.write_text(
        json.dumps({"id": "chunk-1", "text": "Grounded source text."}) + "\n",
        encoding="utf-8",
    )
    (course / "objectives.json").write_text(
        json.dumps({"objectives": [{"id": "workflow-objective"}]}) + "\n",
        encoding="utf-8",
    )
    return course


def _observe_holdout_objectives(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    observed: dict[str, Any] = {}

    def fake_loader(**kwargs: Any) -> None:
        observed.update(kwargs)
        raise _PreflightObserved

    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_HOLDOUT_EXCLUSION", "true")
    monkeypatch.setattr(
        "Trainforge.synthesis_holdout.load_synthesis_holdout_registry",
        fake_loader,
    )
    return observed


def test_environment_override_reaches_holdout_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    course = _course(tmp_path)
    pinned = tmp_path / "archive" / "objectives.json"
    pinned.parent.mkdir()
    pinned.write_text('{"objectives": [{"id": "pinned"}]}\n', encoding="utf-8")
    observed = _observe_holdout_objectives(monkeypatch)
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH", str(pinned))

    with pytest.raises(_PreflightObserved):
        run_synthesis(course, "COURSE", provider="mock")

    assert observed["objectives_path"] == pinned
    assert not (course / "training_specs").exists()


def test_explicit_argument_precedes_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    course = _course(tmp_path)
    environment_path = tmp_path / "environment-objectives.json"
    environment_path.write_text('{"objectives": []}\n', encoding="utf-8")
    explicit_path = tmp_path / "explicit-objectives.json"
    explicit_path.write_text('{"objectives": []}\n', encoding="utf-8")
    observed = _observe_holdout_objectives(monkeypatch)
    monkeypatch.setenv(
        "TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH", str(environment_path)
    )

    with pytest.raises(_PreflightObserved):
        run_synthesis(
            course,
            "COURSE",
            provider="mock",
            objectives_path=explicit_path,
        )

    assert observed["objectives_path"] == explicit_path


def test_unset_override_retains_corpus_objectives_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    course = _course(tmp_path)
    observed = _observe_holdout_objectives(monkeypatch)
    monkeypatch.delenv("TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH", raising=False)

    with pytest.raises(_PreflightObserved):
        run_synthesis(course, "COURSE", provider="mock")

    assert observed["objectives_path"] == course / "objectives.json"


def test_missing_override_fails_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    course = _course(tmp_path)
    missing = tmp_path / "missing-objectives.json"
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_OBJECTIVES_PATH", str(missing))

    with pytest.raises(FileNotFoundError, match="Configured synthesis objectives"):
        run_synthesis(course, "COURSE", provider="mock")

    assert not (course / "training_specs").exists()
