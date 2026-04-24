"""Wave 74 Session 3 — ``--skip-dart`` phase-skip mechanic.

Locks the workflow runner's behaviour when a workflow's params carry
``skip_dart=True``:

* ``_synthesize_dart_skip_output`` walks the provided DART output dir
  and emits a dict with the keys downstream phases' ``inputs_from``
  pulls (``output_paths``, ``html_paths``, ``html_path``, plus
  ``_completed``/``_skipped``/``_gates_passed`` markers).
* The synthesised ordering honours the workflow's ``pdf_paths`` when
  each PDF has a matching ``{stem}_accessible.html``; otherwise falls
  back to directory-sorted order.
* When the dir is missing or empty, the method returns ``None`` and
  the caller surfaces the failure upstream instead of silently passing.
* ``_should_skip_phase`` no longer strips the synthesised dict for
  ``dart_conversion`` (pre-implementation, it would have overwritten
  ``phase_outputs["dart_conversion"]`` with a bare
  ``{_skipped: True, _completed: True}`` placeholder, dropping the
  html_paths downstream phases need).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from MCP.core.workflow_runner import WorkflowRunner


@pytest.fixture
def runner_stub() -> WorkflowRunner:
    """Minimal WorkflowRunner — we only exercise pure helpers below.

    The ``executor`` and ``config`` fields are unused by
    ``_synthesize_dart_skip_output`` / ``_should_skip_phase``, so a
    sentinel tuple keeps the constructor happy without hauling in a
    real OrchestratorConfig.
    """
    return WorkflowRunner(executor=object(), config=object())


def test_synthesize_dart_skip_output_emits_expected_keys(tmp_path, runner_stub):
    # Two DART HTMLs in a scratch dir.
    a = tmp_path / "alpha_accessible.html"
    b = tmp_path / "beta_accessible.html"
    a.write_text("<html>a</html>")
    b.write_text("<html>beta</html>")

    params = {
        "skip_dart": True,
        "dart_output_dir": str(tmp_path),
        "pdf_paths": [
            str(tmp_path / "alpha.pdf"),
            str(tmp_path / "beta.pdf"),
        ],
    }

    out = runner_stub._synthesize_dart_skip_output(params)
    assert out is not None

    # Downstream phases (staging, libv2_archival) pull these exact keys
    # via inputs_from in config/workflows.yaml.
    assert out["_completed"] is True
    assert out["_skipped"] is True
    assert out["_gates_passed"] is True
    assert out["success"] is True
    assert out["html_length"] > 0

    # Ordering must follow the corpus PDF order so staging's
    # {stem}_accessible.html lookup matches 1:1.
    expected_paths = [str(a), str(b)]
    assert out["output_paths"] == ",".join(expected_paths)
    assert out["html_paths"] == ",".join(expected_paths)
    assert out["output_path"] == expected_paths[0]
    assert out["html_path"] == expected_paths[0]


def test_synthesize_falls_back_to_sorted_dir_when_no_pdf_paths(
    tmp_path, runner_stub
):
    (tmp_path / "zulu_accessible.html").write_text("z")
    (tmp_path / "alpha_accessible.html").write_text("a")

    out = runner_stub._synthesize_dart_skip_output(
        {"skip_dart": True, "dart_output_dir": str(tmp_path)}
    )
    assert out is not None
    # Sorted directory order: alpha before zulu.
    paths = out["output_paths"].split(",")
    assert Path(paths[0]).name == "alpha_accessible.html"
    assert Path(paths[1]).name == "zulu_accessible.html"


def test_synthesize_returns_none_when_dir_missing(tmp_path, runner_stub):
    out = runner_stub._synthesize_dart_skip_output(
        {
            "skip_dart": True,
            "dart_output_dir": str(tmp_path / "no_such_subdir"),
        }
    )
    assert out is None


def test_synthesize_returns_none_when_dir_empty(tmp_path, runner_stub):
    out = runner_stub._synthesize_dart_skip_output(
        {"skip_dart": True, "dart_output_dir": str(tmp_path)}
    )
    assert out is None


def test_should_skip_phase_preserves_synthesised_output(runner_stub):
    """Regression guard: ``_should_skip_phase`` must NOT return True
    for dart_conversion when skip_dart is set, because the phase loop
    would then overwrite phase_outputs["dart_conversion"] with a bare
    {_skipped, _completed} placeholder — dropping html_paths.

    The correct mechanic is: pre-populate phase_outputs in
    run_workflow BEFORE the loop, then rely on the already-completed
    guard (``if phase_outputs[phase_name].get("_completed")``).
    """
    dart_phase = SimpleNamespace(name="dart_conversion", optional=False)
    assert (
        runner_stub._should_skip_phase(dart_phase, {"skip_dart": True}) is False
    )


def test_should_skip_phase_still_handles_trainforge(runner_stub):
    """Regression guard: the optional trainforge_assessment branch is
    untouched — --no-assessments must still elide it.
    """
    tf_phase = SimpleNamespace(name="trainforge_assessment", optional=True)
    assert (
        runner_stub._should_skip_phase(
            tf_phase, {"generate_assessments": False}
        )
        is True
    )
    assert (
        runner_stub._should_skip_phase(
            tf_phase, {"generate_assessments": True}
        )
        is False
    )
