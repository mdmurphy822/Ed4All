"""Wave 91 Action B: CLI default-on / opt-out tests for
``synthesize_training.py``.

The Wave 91 contract:
    - ``--curriculum-from-graph`` defaults to ON in the CLI.
    - ``--no-graph`` opts out for legacy corpora.
    - When the default is honored and no pedagogy_graph.json is
      present, ``run_synthesis`` raises ``FileNotFoundError`` so the
      regression where synthesis silently produced graph-less ordering
      is impossible.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import (  # noqa: E402
    build_parser,
    main,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


def _make_working_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    return dst


def test_cli_curriculum_from_graph_defaults_to_true():
    parser = build_parser()
    args = parser.parse_args([
        "--corpus", "/tmp/dummy",
        "--course-code", "TST",
    ])
    assert args.curriculum_from_graph is True
    assert args.no_graph is False


def test_cli_no_graph_flag_can_be_set():
    parser = build_parser()
    args = parser.parse_args([
        "--corpus", "/tmp/dummy",
        "--course-code", "TST",
        "--no-graph",
    ])
    assert args.no_graph is True


def test_default_run_without_graph_raises_filenotfound(tmp_path):
    """The fixture has no pedagogy_graph.json, so the default-on
    curriculum-from-graph path must raise."""
    working = _make_working_copy(tmp_path)
    parser = build_parser()
    args = parser.parse_args([
        "--corpus", str(working),
        "--course-code", "MINI_TRAINING_101",
    ])
    # Sanity: default is True.
    assert args.curriculum_from_graph is True
    with pytest.raises(FileNotFoundError) as excinfo:
        main(args)
    msg = str(excinfo.value)
    assert "pedagogy_graph.json" in msg
    # Error message must point users at the opt-out path.
    assert "--no-graph" in msg


def test_no_graph_opt_out_runs_to_completion(tmp_path):
    """With --no-graph, synthesis runs without a pedagogy graph (legacy)."""
    working = _make_working_copy(tmp_path)
    parser = build_parser()
    args = parser.parse_args([
        "--corpus", str(working),
        "--course-code", "MINI_TRAINING_101",
        "--no-graph",
    ])
    stats = main(args)
    assert stats.instruction_pairs_emitted >= 1
    assert stats.preference_pairs_emitted >= 1
    assert stats.curriculum_from_graph is False
