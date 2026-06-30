"""Regression net for DifficultyProvenanceValidator (warning-day-1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.assessment.irt_difficulty import DEFAULT_MIN_RESPONSES  # noqa: E402
from lib.validators.difficulty_provenance import DifficultyProvenanceValidator  # noqa: E402

_FLAG = "TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD"


def _codes(result):
    return {i.code for i in result.issues}


def test_flag_off_no_op(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    result = DifficultyProvenanceValidator().validate({"chunks": [{"id": "c1"}]})
    assert result.passed is True
    assert "DIFFICULTY_PROVENANCE_SCAFFOLD_DISABLED" in _codes(result)


def test_flag_on_all_tagged_passes(monkeypatch):
    monkeypatch.setenv(_FLAG, "true")
    chunks = [
        {"id": "c1", "difficulty_provenance": "heuristic"},
        {"id": "c2", "difficulty_provenance": "calibrated",
         "difficulty_irt": {"difficulty_b": 0.5, "discrimination_a": 1.0,
                            "n_responses": DEFAULT_MIN_RESPONSES + 1}},
    ]
    result = DifficultyProvenanceValidator().validate({"chunks": chunks})
    assert result.passed is True
    assert result.warning_count == 0


def test_flag_on_irt_without_responses_warns(monkeypatch):
    monkeypatch.setenv(_FLAG, "true")
    # An IRT block with n_responses < min OR heuristic provenance is a violation.
    chunks = [
        {"id": "c1", "difficulty_provenance": "calibrated",
         "difficulty_irt": {"difficulty_b": 0.5, "discrimination_a": 1.0,
                            "n_responses": DEFAULT_MIN_RESPONSES - 1}},
        {"id": "c2", "difficulty_provenance": "heuristic",
         "difficulty_irt": {"difficulty_b": 0.5, "discrimination_a": 1.0,
                            "n_responses": DEFAULT_MIN_RESPONSES + 100}},
    ]
    result = DifficultyProvenanceValidator().validate({"chunks": chunks})
    assert "DIFFICULTY_IRT_WITHOUT_RESPONSES" in _codes(result)
    assert result.metadata["irt_without_responses"] == 2
    # Warning-day-1: warnings never block.
    assert result.passed is True


def test_flag_on_missing_provenance_warns(monkeypatch):
    monkeypatch.setenv(_FLAG, "true")
    chunks = [{"id": "c1"}]  # no difficulty_provenance
    result = DifficultyProvenanceValidator().validate({"chunks": chunks})
    assert "DIFFICULTY_PROVENANCE_MISSING" in _codes(result)
    assert result.metadata["missing_provenance"] == 1


def test_reads_chunks_from_course_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(_FLAG, "true")
    import json
    chunkdir = tmp_path / "imscc_chunks"
    chunkdir.mkdir(parents=True)
    (chunkdir / "chunks.jsonl").write_text(
        json.dumps({"id": "c1", "difficulty_provenance": "heuristic"}) + "\n",
        encoding="utf-8",
    )
    result = DifficultyProvenanceValidator().validate({"course_dir": str(tmp_path)})
    assert result.metadata["chunks_checked"] == 1
    assert result.passed is True
