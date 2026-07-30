"""Trainforge/scripts/model_dpo_yield.py — Bloom-ladder addendum AD-02.

Covers the CLI wrapper: course-dir chunk/objectives resolution (LibV2
layout via ``resolve_imscc_chunks_path`` + the same objectives search chain
``training_synthesis`` uses), JSON vs human-table emit, and the three exit
codes (0 at/above floor, 1 below floor / famine, 2 missing required input).
Arithmetic itself is covered by ``lib/validators/tests/test_dpo_yield_projection.py``
against the same shared core (``project_dpo_yield``) — this suite is
CLI-surface only.

No course slugs anywhere — every fixture is built under ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Trainforge.scripts import model_dpo_yield


_OBJ_STATEMENT = (
    "Explain how absolute value represents distance magnitude for a "
    "number line position"
)
_CLAIM = (
    "absolute value equals the original number for every number line position"
)
_CORRECTION = (
    "absolute value equals the non negative magnitude for every number "
    "line position"
)
_FILLER = " ".join([
    "Distance magnitude concepts recur across many worked examples in "
    "this unit for number line position study."
] * 6)


def _admitted_chunk(chunk_id: str = "chunk-001") -> dict:
    return {
        "id": chunk_id,
        "text": _FILLER + " " + _CLAIM + ". " + _CORRECTION + ".",
        "bloom_level": "understand",
        "learning_outcome_refs": ["to-01"],
        "misconceptions": [{
            "misconception": _CLAIM,
            "correction": _CORRECTION,
            "mechanism_evidence": _CORRECTION,
            "bloom_level": "understand",
        }],
    }


def _make_course_dir(tmp_path: Path, *, chunks: list) -> Path:
    course_dir = tmp_path / "course"
    (course_dir / "imscc_chunks").mkdir(parents=True)
    chunks_path = course_dir / "imscc_chunks" / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")
    (course_dir / "objectives.json").write_text(json.dumps([
        {"id": "TO-01", "statement": _OBJ_STATEMENT, "bloom_level": "understand"},
    ]))
    return course_dir


def test_exit_zero_when_at_or_above_floor(tmp_path: Path, capsys):
    course_dir = _make_course_dir(tmp_path, chunks=[_admitted_chunk()])
    code = model_dpo_yield.main([
        "--course-dir", str(course_dir), "--min-dpo-pairs", "1",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "verdict" in out
    assert "OK" in out


def test_exit_one_when_below_floor(tmp_path: Path, capsys):
    course_dir = _make_course_dir(tmp_path, chunks=[_admitted_chunk()])
    code = model_dpo_yield.main([
        "--course-dir", str(course_dir), "--min-dpo-pairs", "5",
    ])
    assert code == 1
    out = capsys.readouterr().out
    assert "FAMINE" in out


def test_json_output_shape(tmp_path: Path, capsys):
    course_dir = _make_course_dir(tmp_path, chunks=[_admitted_chunk()])
    code = model_dpo_yield.main([
        "--course-dir", str(course_dir), "--min-dpo-pairs", "1", "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cards_found"] == 1
    assert payload["arm_a_admitted"] == 1
    assert payload["projected_admissible"] == 1
    assert payload["min_dpo_pairs"] == 1
    assert payload["deficit"] is False
    assert payload["chunks_path"].endswith("imscc_chunks/chunks.jsonl")
    assert payload["objectives_path"].endswith("objectives.json")


def test_exit_two_when_chunks_missing(tmp_path: Path, capsys):
    course_dir = tmp_path / "empty_course"
    course_dir.mkdir()
    code = model_dpo_yield.main(["--course-dir", str(course_dir)])
    assert code == 2
    err = capsys.readouterr().err
    assert "no chunks.jsonl resolvable" in err


def test_exit_two_when_objectives_missing(tmp_path: Path, capsys):
    course_dir = tmp_path / "course"
    (course_dir / "imscc_chunks").mkdir(parents=True)
    chunks_path = course_dir / "imscc_chunks" / "chunks.jsonl"
    chunks_path.write_text(json.dumps(_admitted_chunk()) + "\n")
    code = model_dpo_yield.main(["--course-dir", str(course_dir)])
    assert code == 2
    err = capsys.readouterr().err
    assert "no canonical objectives artifact resolvable" in err


def test_explicit_objectives_path_override(tmp_path: Path, capsys):
    course_dir = tmp_path / "course"
    (course_dir / "imscc_chunks").mkdir(parents=True)
    chunks_path = course_dir / "imscc_chunks" / "chunks.jsonl"
    chunks_path.write_text(json.dumps(_admitted_chunk()) + "\n")
    objectives_elsewhere = tmp_path / "external_objectives.json"
    objectives_elsewhere.write_text(json.dumps([
        {"id": "TO-01", "statement": _OBJ_STATEMENT, "bloom_level": "understand"},
    ]))
    code = model_dpo_yield.main([
        "--course-dir", str(course_dir),
        "--objectives-path", str(objectives_elsewhere),
        "--min-dpo-pairs", "1",
    ])
    assert code == 0


def test_default_min_dpo_pairs_is_fifty(tmp_path: Path, capsys):
    course_dir = _make_course_dir(tmp_path, chunks=[_admitted_chunk()])
    code = model_dpo_yield.main(["--course-dir", str(course_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["min_dpo_pairs"] == 50
    assert code == 1  # one admitted pair, default floor 50 -> famine


@pytest.mark.parametrize("cards", [1, 2, 3])
def test_multi_chunk_admission_count_matches_projection(
    tmp_path: Path, capsys, cards: int,
):
    chunks = [_admitted_chunk(f"chunk-{i}") for i in range(cards)]
    course_dir = _make_course_dir(tmp_path, chunks=chunks)
    code = model_dpo_yield.main([
        "--course-dir", str(course_dir), "--min-dpo-pairs", str(cards), "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["projected_admissible"] == cards
    assert code == 0
