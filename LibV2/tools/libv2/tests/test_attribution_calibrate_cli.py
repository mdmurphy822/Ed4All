"""CliRunner tests for ``libv2 attribution-calibrate`` (retrieval-answer-eval P4).

CI-safe: monkeypatches the lib's live retriever so no real index / LLM / slug is
touched; builds a hermetic course gold set in tmp_path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402

_SLUG = "syn-attrib-101"


class _FakePassage:
    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text


def _stage(repo_root: Path) -> Path:
    course_dir = repo_root / "courses" / _SLUG
    eval_dir = course_dir / "retrieval_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (course_dir / "manifest.json").write_text(
        json.dumps({"slug": _SLUG, "classification": {}}), encoding="utf-8"
    )
    gold = {
        "schema_version": "1.1",
        "course_slug": _SLUG,
        "questions": [
            {
                "question_id": "gq-syn-attrib-101-0001",
                "question_text": "what is a widget",
                "expected_key_points": [
                    "A widget is a small reusable interface component",
                ],
                "relevant_passages": [{"chunk_id": "rel1"}],
            }
        ],
    }
    (eval_dir / "gold_set.json").write_text(json.dumps(gold), encoding="utf-8")
    return course_dir


def test_attribution_calibrate_writes_artifact(tmp_path, monkeypatch):
    repo_root = tmp_path / "libv2"
    course_dir = _stage(repo_root)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(repo_root))

    def _fake_retrieve(libv2_root, course_slug, query, *, engine, limit):
        return [
            _FakePassage(
                "rel1", "A widget is a small reusable interface component here."
            ),
            _FakePassage("noise1", "Unrelated geology sentence about rocks."),
        ]

    monkeypatch.setattr(
        "lib.retrieval.attribution_calibrate._default_retrieve", _fake_retrieve
    )

    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "attribution-calibrate", "--course", _SLUG, "--engine", "lexical",
    ])
    assert result.exit_code == 0, result.output
    written = list((course_dir / "retrieval_eval").glob(
        "attribution_calibration_*.json"
    ))
    assert len(written) == 1
    doc = json.loads(written[0].read_text(encoding="utf-8"))
    assert doc["course_slug"] == _SLUG
    assert doc["schema_version"] == "1.0"
    assert "recommended" in doc
    assert "add_min_shingle_knob_warranted" in doc


def test_attribution_calibrate_no_write(tmp_path, monkeypatch):
    repo_root = tmp_path / "libv2"
    _stage(repo_root)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(repo_root))
    monkeypatch.setattr(
        "lib.retrieval.attribution_calibrate._default_retrieve",
        lambda *a, **k: [_FakePassage("rel1", "A widget is a small reusable component.")],
    )
    runner = CliRunner()
    result = runner.invoke(libv2_main, [
        "--repo", str(repo_root),
        "attribution-calibrate", "--course", _SLUG, "--no-write",
    ])
    assert result.exit_code == 0, result.output
    # --no-write → JSON printed, no artifact on disk.
    assert not list(
        (repo_root / "courses" / _SLUG / "retrieval_eval").glob(
            "attribution_calibration_*.json"
        )
    )
    assert "recommended" in result.output
