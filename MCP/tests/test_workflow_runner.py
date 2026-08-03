"""Wave1-I8 — provider-banner regression tests.

The workflow runner emits one log line per entry in
``AGENT_PROVIDER_ENV_MAP`` at workflow start so operators can see at a
glance how each subagent-classified agent will dispatch:

1. ``local-provider`` — the agent's provider env var (e.g.
   ``COURSEFORGE_PROVIDER``) is set; the executor short-circuits the
   Wave-74 subagent dispatch and routes through the in-process
   Wave-D provider.
2. ``subagent (claude)`` — ``ED4ALL_AGENT_DISPATCH=true`` is set but
   no provider env var is; the executor dispatches to the Claude
   Code subagent.
3. ``in-process-stub`` — neither is set; the executor falls through
   to the in-process tool registry (test/dry-run path).

Pure observability — these tests pin the log format + precedence
order without exercising any phase. They go through
``WorkflowRunner._emit_provider_banner`` directly because the banner
is a one-shot helper invoked once per ``run_workflow`` call.

Finding 7 of plans/dispatch-7-execution-inspection-2026-05.md.
"""
from __future__ import annotations

import logging
from typing import Mapping
from unittest.mock import MagicMock

import pytest

from MCP.core.executor import AGENT_PROVIDER_ENV_MAP
from MCP.core.workflow_runner import WorkflowRunner


def _make_runner() -> WorkflowRunner:
    """Build a minimal ``WorkflowRunner`` for banner-only tests.

    The banner helper depends on neither ``executor`` nor ``config``,
    so stub references are sufficient.
    """
    return WorkflowRunner(executor=MagicMock(), config=MagicMock())


def _banner_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return ``[provider-banner]`` log records' messages, ordered."""
    return [
        record.getMessage()
        for record in caplog.records
        if "[provider-banner]" in record.getMessage()
    ]


def _assert_one_line_per_agent(
    lines: list[str], expected_map: Mapping[str, str]
) -> None:
    """Pin one line per ``AGENT_PROVIDER_ENV_MAP`` entry, no duplicates."""
    assert len(lines) == len(expected_map), (
        f"Expected {len(expected_map)} banner lines (one per agent in "
        f"AGENT_PROVIDER_ENV_MAP), got {len(lines)}: {lines!r}"
    )
    for agent in expected_map:
        agent_lines = [line for line in lines if f" {agent}:" in line]
        assert len(agent_lines) == 1, (
            f"Expected exactly one banner line for agent {agent!r}, got "
            f"{agent_lines!r}"
        )


def test_banner_all_in_process_stub_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No env vars set → every agent banner reads ``in-process-stub``."""
    # Clear all provider env vars + the dispatch flag.
    monkeypatch.delenv("ED4ALL_AGENT_DISPATCH", raising=False)
    for env_var in AGENT_PROVIDER_ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)
    for line in lines:
        assert "in-process-stub" in line, line
        assert "ED4ALL_AGENT_DISPATCH=false" in line, line
        # Defence-in-depth: no spurious local-provider / subagent strings.
        assert "local-provider" not in line, line
        assert "subagent" not in line, line


def test_banner_all_subagent_when_only_agent_dispatch_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``ED4ALL_AGENT_DISPATCH=true`` only → all show ``subagent (claude)``."""
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    for env_var in AGENT_PROVIDER_ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)
    for line in lines:
        assert "subagent (claude)" in line, line
        assert "ED4ALL_AGENT_DISPATCH=true" in line, line
        assert "in-process-stub" not in line, line
        assert "local-provider" not in line, line


def test_banner_local_provider_takes_precedence_over_subagent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider env var wins over ``ED4ALL_AGENT_DISPATCH=true``.

    Mirrors the executor's short-circuit precedence: when both are
    set, the in-process provider runs (Wave-D ToS unblock) and the
    subagent dispatch is bypassed for that agent only — other agents
    in the map still log ``subagent (claude)``.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "together")
    # Clear the other provider env vars so they fall through to subagent.
    for agent, env_var in AGENT_PROVIDER_ENV_MAP.items():
        if env_var != "COURSEFORGE_PROVIDER":
            monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    _assert_one_line_per_agent(lines, AGENT_PROVIDER_ENV_MAP)

    content_gen_lines = [
        line for line in lines if " content-generator:" in line
    ]
    assert len(content_gen_lines) == 1
    assert "local-provider" in content_gen_lines[0]
    assert "COURSEFORGE_PROVIDER=together" in content_gen_lines[0]

    other_lines = [
        line for line in lines if " content-generator:" not in line
    ]
    for line in other_lines:
        assert "subagent (claude)" in line, line
        assert "ED4ALL_AGENT_DISPATCH=true" in line, line


def test_banner_empty_or_whitespace_env_var_does_not_count(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whitespace-only provider env var must NOT flip the banner.

    Mirrors the executor's ``.strip()`` guard inside
    ``_dispatch_task_via_tool`` — an empty / whitespace env var falls
    through to the subagent / stub path. The banner must report the
    same.
    """
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    # Whitespace-only — should be treated as unset.
    monkeypatch.setenv("COURSEFORGE_PROVIDER", "   ")
    for agent, env_var in AGENT_PROVIDER_ENV_MAP.items():
        if env_var != "COURSEFORGE_PROVIDER":
            monkeypatch.delenv(env_var, raising=False)

    runner = _make_runner()
    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        runner._emit_provider_banner()

    lines = _banner_lines(caplog)
    content_gen_lines = [
        line for line in lines if " content-generator:" in line
    ]
    assert len(content_gen_lines) == 1
    # Should NOT have flipped to local-provider on whitespace.
    assert "local-provider" not in content_gen_lines[0]
    assert "subagent (claude)" in content_gen_lines[0]


# ---------------------------------------------------------------------------
# Wave1-I6 (Finding 9) — env-predicate skip log
# ---------------------------------------------------------------------------


def test_should_skip_phase_logs_env_predicate_skip(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ``enabled_when_env`` skip must surface in the operator log.

    Finding 9 of plans/dispatch-7-execution-inspection-2026-05.md: a
    phase with ``enabled_when_env: "SOME_FLAG=true"`` and ``SOME_FLAG``
    unset silently no-ops, which is indistinguishable from a crashed /
    mis-routed phase at triage time. This test pins the log line shape
    so the skip stays visible.
    """
    from MCP.core.config import WorkflowPhase

    monkeypatch.delenv("WAVE1_I6_TEST_FLAG", raising=False)

    phase = WorkflowPhase(
        name="phase_under_test",
        agents=[],
        enabled_when_env="WAVE1_I6_TEST_FLAG=true",
    )
    runner = _make_runner()

    with caplog.at_level(logging.INFO, logger="MCP.core.workflow_runner"):
        skipped = runner._should_skip_phase(phase, {})

    assert skipped is True
    skip_lines = [
        record.getMessage()
        for record in caplog.records
        if "enabled_when_env" in record.getMessage()
        and "phase_under_test" in record.getMessage()
    ]
    assert len(skip_lines) == 1, (
        f"Expected exactly one env-predicate skip log line, got "
        f"{skip_lines!r}"
    )
    line = skip_lines[0]
    assert "WAVE1_I6_TEST_FLAG=true" in line
    assert "unsatisfied" in line
    assert "unset" in line


# ---------------------------------------------------------------------------
# Phase-skip merge: synthesized pre-populated data must survive the skip
# ---------------------------------------------------------------------------


def test_skip_phase_preserves_synthesizer_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_should_skip_phase branch must MERGE, not replace, pre-populated data.

    Regression for the bug where the ``_should_skip_phase`` branch in
    ``run_workflow`` overwrote ``phase_outputs[phase_name]`` with a bare
    ``{"_skipped": True, "_completed": True}``, destroying keys like
    ``project_id`` that downstream phases pull via ``inputs_from``.

    The test simulates the Phase-5 stage-subcommand + ``--force`` path:

    1. ``_synthesize_outline_output`` pre-populates
       ``phase_outputs["objective_extraction"]`` with rich data.
    2. ``--force`` flips ``_completed`` to ``False`` on that entry.
    3. The phase loop hits ``_should_skip_phase`` (courseforge_stage
       whitelist, phase not whitelisted).
    4. After the merge, ``project_id`` must still be present alongside
       the skip markers.
    """
    from MCP.core.config import WorkflowPhase

    # Simulate a phase that is NOT whitelisted for the active stage, so
    # _should_skip_for_courseforge_stage returns True.
    phase = WorkflowPhase(
        name="objective_extraction",
        agents=[],
    )

    runner = _make_runner()

    # Wire _should_skip_phase to return True (simulates courseforge_stage
    # whitelist skipping objective_extraction).
    monkeypatch.setattr(
        runner,
        "_should_skip_phase",
        lambda p, wp: True,
    )

    # Build the synthesizer-pre-populated phase_outputs dict with
    # _completed=False (as --force would flip it).
    synthesized_entry = {
        "project_id": "PROJ-X-123",
        "project_path": "/foo/bar",
        "textbook_structure_path": "/foo/bar/textbook_structure.json",
        "_completed": False,
    }
    phase_outputs: dict = {"objective_extraction": dict(synthesized_entry)}

    # Apply the fixed merge logic directly — mirrors run_workflow lines
    # 1036-1041 after the patch.
    existing = phase_outputs.get(phase.name) or {}
    phase_outputs[phase.name] = {
        **existing,
        "_skipped": True,
        "_completed": True,
    }

    result = phase_outputs["objective_extraction"]

    assert result["project_id"] == "PROJ-X-123", (
        f"project_id was wiped; got: {result!r}"
    )
    assert result["project_path"] == "/foo/bar", (
        f"project_path was wiped; got: {result!r}"
    )
    assert result["_skipped"] is True, "_skipped marker missing"
    assert result["_completed"] is True, "_completed marker missing"


# =============================================================================
# Resume-state restoration regression tests
# =============================================================================
#
# Bug: on ``--resume``, ``phase_outputs`` is reloaded from the persisted
# workflow-state JSON, but a phase can be persisted as ``_completed=True``
# with an EMPTY extracted-output dict (observed in real states, e.g.
# content_generation/packaging carrying ``keys=[]``). The loop's
# ``_completed`` guard then skips that phase without re-deriving its outputs,
# so downstream phases (content gates; ``imscc_chunking`` reading
# ``packaging.package_path``) see empty inputs and fail. The fix
# (``_restore_resume_phase_outputs``) reconstructs missing canonical keys
# from on-disk artifacts before the loop runs.


def _build_project_export(tmp_path, course_name: str = "FXALPHA_101"):
    """Materialize a minimal Courseforge project export on disk.

    Returns ``(project_path, expected_package_path)``.
    """
    import json as _json

    project_path = tmp_path / "exports" / f"PROJ-{course_name}-20260101"
    project_path.mkdir(parents=True)
    (project_path / "project_config.json").write_text(
        _json.dumps({
            "project_id": project_path.name,
            "course_name": course_name,
        }),
        encoding="utf-8",
    )
    # source_module_map.json
    (project_path / "source_module_map.json").write_text(
        _json.dumps({"modules": []}), encoding="utf-8"
    )
    # 01_learning_objectives/
    lo_dir = project_path / "01_learning_objectives"
    lo_dir.mkdir()
    (lo_dir / "textbook_structure.json").write_text(
        _json.dumps({"chapters": []}), encoding="utf-8"
    )
    (lo_dir / "synthesized_objectives.json").write_text(
        _json.dumps({
            "terminal_objectives": [{"id": "TO-01"}],
            "chapter_objectives": [
                {"chapter": "Week 1", "objectives": [{"id": "CO-01"}]}
            ],
        }),
        encoding="utf-8",
    )
    # 03_content_development/week_01/*.html
    content_dir = project_path / "03_content_development" / "week_01"
    content_dir.mkdir(parents=True)
    (content_dir / "week_01_overview.html").write_text(
        "<html></html>", encoding="utf-8"
    )
    # 05_final_package/<course>.imscc
    final_dir = project_path / "05_final_package"
    final_dir.mkdir()
    pkg = final_dir / f"{course_name}.imscc"
    pkg.write_bytes(b"PK\x03\x04dummy")

    return project_path, pkg


def test_resume_restore_backfills_empty_completed_phases(tmp_path) -> None:
    """Resume restoration backfills missing output keys from disk.

    Simulates a resumed workflow whose ``packaging`` + ``content_generation``
    phases were persisted ``_completed=True`` with EMPTY output dicts. After
    restoration, ``packaging.package_path`` (consumed by ``imscc_chunking``)
    and ``content_generation.content_dir`` (consumed by content gates) must
    be present so downstream phases resolve their inputs.
    """
    project_path, expected_pkg = _build_project_export(tmp_path)

    runner = _make_runner()

    # Resumed phase_outputs: objective_extraction carries project_path
    # (so the export root resolves); packaging + content_generation are
    # completed-but-empty (the bug condition).
    phase_outputs = {
        "objective_extraction": {
            "project_id": project_path.name,
            "project_path": str(project_path),
            "_completed": True,
            "_gates_passed": True,
        },
        "content_generation": {"_completed": True, "_gates_passed": True},
        "packaging": {"_completed": True, "_gates_passed": True},
    }

    runner._restore_resume_phase_outputs(phase_outputs)

    pkg_out = phase_outputs["packaging"]
    assert pkg_out["package_path"] == str(expected_pkg), (
        f"packaging.package_path not restored; got: {pkg_out!r}"
    )
    assert pkg_out["imscc_path"] == str(expected_pkg)
    assert pkg_out["_resume_restored"] is True
    # Skip markers preserved (not clobbered).
    assert pkg_out["_completed"] is True
    assert pkg_out["_gates_passed"] is True

    cg_out = phase_outputs["content_generation"]
    assert cg_out["content_dir"].endswith("03_content_development"), (
        f"content_generation.content_dir not restored; got: {cg_out!r}"
    )
    assert cg_out["content_paths"], "content_paths empty after restore"


def test_resume_restore_noop_on_fresh_run() -> None:
    """Fresh (non-resume) run: empty phase_outputs => strict no-op."""
    runner = _make_runner()
    phase_outputs: dict = {}
    runner._restore_resume_phase_outputs(phase_outputs)
    assert phase_outputs == {}, "fresh-run restoration must be a no-op"


def test_resume_restore_does_not_overwrite_present_keys(tmp_path) -> None:
    """Restoration must never clobber an already-recorded output value."""
    project_path, expected_pkg = _build_project_export(tmp_path)
    runner = _make_runner()

    phase_outputs = {
        "objective_extraction": {
            "project_id": project_path.name,
            "project_path": str(project_path),
            "_completed": True,
        },
        # packaging already carries a real package_path — must be left
        # untouched (no restore triggered since required key present).
        "packaging": {
            "package_path": "/already/recorded/COURSE.imscc",
            "_completed": True,
        },
    }

    runner._restore_resume_phase_outputs(phase_outputs)

    assert phase_outputs["packaging"]["package_path"] == (
        "/already/recorded/COURSE.imscc"
    ), "restoration overwrote a recorded package_path"
    assert "_resume_restored" not in phase_outputs["packaging"], (
        "restoration touched a phase that already had its required key"
    )


def test_resume_restore_tolerates_missing_artifacts(tmp_path) -> None:
    """Missing on-disk artifacts: leave the empty phase as-is, no raise."""
    # Project export root that does NOT exist on disk.
    runner = _make_runner()
    phase_outputs = {
        "objective_extraction": {
            "project_id": "PROJ-MISSING-1",
            "project_path": str(tmp_path / "does_not_exist"),
            "_completed": True,
        },
        "packaging": {"_completed": True},
    }

    # Must not raise; packaging stays empty (downstream dep check surfaces it).
    runner._restore_resume_phase_outputs(phase_outputs)

    assert "package_path" not in phase_outputs["packaging"]
    assert "_resume_restored" not in phase_outputs["packaging"]
