"""Wave 29 CLI exit-code propagation tests (Defect 3).

Pre-Wave-29 ``python3 -m cli.main run ...`` returned exit code 0 even
when phase summary showed ``gates=fail`` on screen. Auto-test harnesses
couldn't detect real pipeline failures. Wave 29 pipes the gate-failure
signal through:

* ``OrchestratorResult.gates_passed`` aggregates across all phases.
* ``cli.commands.run._any_gate_failed`` scans ``phase_results``.
* Non-zero exit code (``2``) propagates when any phase reported
  ``gates_passed=False`` OR the workflow status isn't ``ok``.
* ``--dry-run`` stays exit 0 (no execution, no gates).
* ``--resume`` honours the resumed workflow's final gate status.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.commands.run import _any_gate_failed


# --------------------------------------------------------------------- #
# Fake OrchestratorResult helper
# --------------------------------------------------------------------- #


class _FakeResult:
    """Lightweight stand-in for ``OrchestratorResult`` used in mocks."""

    def __init__(self, status: str, phase_results: dict, error: str = None):
        self.status = status
        self.phase_results = phase_results
        self.error = error
        self.dispatched_phases = []
        self.phase_outputs = {}
        self.workflow_id = "WF-FAKE-123"

    def to_dict(self):
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "phase_results": self.phase_results,
            "error": self.error,
            "gates_passed": not _any_gate_failed(self),
        }


# --------------------------------------------------------------------- #
# _any_gate_failed unit tests
# --------------------------------------------------------------------- #


def test_any_gate_failed_true_when_any_phase_failed():
    r = _FakeResult(
        "ok",
        {
            "phase_a": {"gates_passed": True, "completed": 1, "task_count": 1},
            "phase_b": {"gates_passed": False, "completed": 1, "task_count": 1},
        },
    )
    assert _any_gate_failed(r) is True


def test_any_gate_failed_false_when_all_pass():
    r = _FakeResult(
        "ok",
        {
            "phase_a": {"gates_passed": True},
            "phase_b": {"gates_passed": True},
        },
    )
    assert _any_gate_failed(r) is False


def test_any_gate_failed_false_when_no_phase_results():
    r = _FakeResult("ok", {})
    assert _any_gate_failed(r) is False


def test_any_gate_failed_false_when_key_absent():
    """Phases with no gates configured don't set gates_passed — treat
    absence as pass."""
    r = _FakeResult(
        "ok",
        {"phase_a": {"completed": 1, "task_count": 1}},
    )
    assert _any_gate_failed(r) is False


# --------------------------------------------------------------------- #
# CLI invocation exit-code tests
# --------------------------------------------------------------------- #


def _fake_created():
    return {"workflow_id": "WF-FAKE-123", "status": "CREATED"}


def test_cli_exit_zero_when_all_gates_pass():
    runner = CliRunner()
    fake = _FakeResult(
        "ok",
        {
            "dart_conversion": {"gates_passed": True, "completed": 1, "task_count": 1},
            "content_generation": {"gates_passed": True, "completed": 5, "task_count": 5},
        },
    )
    with (
        patch("cli.commands.run._create_textbook_workflow", new=AsyncMock(return_value=_fake_created())),
        patch("cli.commands.run._build_orchestrator") as build_mock,
    ):
        orch = build_mock.return_value
        orch.run = AsyncMock(return_value=fake)
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
            ],
        )
    assert result.exit_code == 0, result.output


def test_cli_exit_two_when_any_gate_failed():
    """When a phase reports ``gates_passed=False``, exit code must
    propagate as 2 — even if workflow status is still ``ok``."""
    runner = CliRunner()
    fake = _FakeResult(
        "ok",
        {
            "dart_conversion": {"gates_passed": True, "completed": 1, "task_count": 1},
            "trainforge_assessment": {"gates_passed": False, "completed": 1, "task_count": 1},
        },
    )
    with (
        patch("cli.commands.run._create_textbook_workflow", new=AsyncMock(return_value=_fake_created())),
        patch("cli.commands.run._build_orchestrator") as build_mock,
    ):
        orch = build_mock.return_value
        orch.run = AsyncMock(return_value=fake)
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
            ],
        )
    assert result.exit_code == 2, result.output


def test_cli_exit_two_when_workflow_failed():
    """Status != ok also yields exit 2."""
    runner = CliRunner()
    fake = _FakeResult(
        "failed",
        {"dart_conversion": {"gates_passed": True}},
        error="Some error",
    )
    with (
        patch("cli.commands.run._create_textbook_workflow", new=AsyncMock(return_value=_fake_created())),
        patch("cli.commands.run._build_orchestrator") as build_mock,
    ):
        orch = build_mock.return_value
        orch.run = AsyncMock(return_value=fake)
        result = runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "TEST_101",
            ],
        )
    assert result.exit_code == 2, result.output


def test_cli_dry_run_always_exit_zero():
    """``--dry-run`` doesn't execute; it always exits 0."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "textbook-to-course",
            "--corpus",
            "inputs/pdfs/fake.pdf",
            "--course-name",
            "TEST_101",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_resume_exit_two_when_gates_failed():
    """``--resume`` honours the resumed workflow's gate status."""
    runner = CliRunner()
    fake = _FakeResult(
        "ok",
        {"trainforge_assessment": {"gates_passed": False}},
    )
    with patch("cli.commands.run._build_orchestrator") as build_mock:
        orch = build_mock.return_value
        orch.run = AsyncMock(return_value=fake)
        result = runner.invoke(
            cli,
            ["run", "textbook-to-course", "--resume", "WF-FAKE-123"],
        )
    assert result.exit_code == 2, result.output


def test_orchestrator_result_gates_passed_aggregator():
    """``OrchestratorResult.gates_passed`` aggregates across phases."""
    from MCP.orchestrator.pipeline_orchestrator import OrchestratorResult

    ok = OrchestratorResult(
        workflow_id="W1",
        status="ok",
        phase_results={
            "p1": {"gates_passed": True},
            "p2": {"gates_passed": True},
        },
    )
    assert ok.gates_passed is True
    assert ok.to_dict()["gates_passed"] is True

    bad = OrchestratorResult(
        workflow_id="W2",
        status="ok",
        phase_results={
            "p1": {"gates_passed": True},
            "p2": {"gates_passed": False},
        },
    )
    assert bad.gates_passed is False
    assert bad.to_dict()["gates_passed"] is False
