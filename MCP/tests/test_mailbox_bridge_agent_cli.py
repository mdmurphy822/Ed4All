"""Wave 74 Session 2 — ``ed4all mailbox-bridge peek-agent`` /
``complete-agent`` CLI coverage.

These tests pin the operator-side CLI for ``kind="agent_task"`` items.
The companion ``kind="llm_call"`` commands (``peek`` / ``complete``)
already route through this file as well because Session 2 added a
kind-filter to the existing ``peek`` to stop it silently claiming
agent tasks.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.commands.mailbox_bridge import mailbox_bridge_group  # noqa: E402
from MCP.orchestrator.local_dispatcher import LocalDispatcher  # noqa: E402
from MCP.orchestrator.task_mailbox import TaskMailbox  # noqa: E402


# ----------------------------------------------------------- test helpers


def _seed_agent_task(mb: TaskMailbox, task_id: str, **overrides: Any) -> None:
    spec: Dict[str, Any] = {
        "kind": "agent_task",
        "agent_type": "content-generator",
        "tool_name": "generate_course_content",
        "task_params": {"project_id": "P", "course_name": "C"},
        "phase_context": {"phase_name": "content_generation"},
        "agent_spec_path": "Courseforge/agents/content-generator.md",
    }
    spec.update(overrides)
    mb.put_pending(task_id, spec)


def _seed_llm_call(mb: TaskMailbox, task_id: str, **overrides: Any) -> None:
    spec: Dict[str, Any] = {
        "kind": "llm_call",
        "prompt": "say hi",
        "max_tokens": 64,
    }
    spec.update(overrides)
    mb.put_pending(task_id, spec)


# ----------------------------------------------------------- peek-agent


class TestPeekAgent:
    def test_empty_queue_returns_empty_array(self, tmp_path: Path):
        # Ensure the mailbox directory exists.
        TaskMailbox(run_id="RUN_EMPTY", base_dir=tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            mailbox_bridge_group,
            [
                "peek-agent",
                "--run-id", "RUN_EMPTY",
                "--base-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload == []

    def test_skips_llm_call_returns_only_agent_task(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_MIX", base_dir=tmp_path)
        _seed_llm_call(mb, "llm_01")
        _seed_agent_task(mb, "content-generator-abc123", agent_type="content-generator")
        _seed_llm_call(mb, "llm_02")
        _seed_agent_task(
            mb, "assessment-generator-def456",
            agent_type="assessment-generator",
            tool_name="generate_assessments",
        )

        runner = CliRunner()
        result = runner.invoke(
            mailbox_bridge_group,
            [
                "peek-agent",
                "--run-id", "RUN_MIX",
                "--max", "10",
                "--base-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        returned_ids = sorted(entry["task_id"] for entry in payload)
        assert returned_ids == [
            "assessment-generator-def456",
            "content-generator-abc123",
        ]
        # The llm_call tasks are still pending — NOT claimed.
        pending = mb.list_pending()
        assert sorted(pending) == ["llm_01", "llm_02"]
        # Flat shape: fields at the top level.
        first = {e["task_id"]: e for e in payload}["content-generator-abc123"]
        assert first["agent_type"] == "content-generator"
        assert first["tool_name"] == "generate_course_content"
        assert first["task_params"] == {"project_id": "P", "course_name": "C"}
        assert first["agent_spec_path"] == "Courseforge/agents/content-generator.md"
        assert first["run_id"] == "RUN_MIX"
        assert "created_at" in first

    def test_honors_max(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_MAX", base_dir=tmp_path)
        for i in range(5):
            _seed_agent_task(mb, f"content-generator-{i:03d}")

        runner = CliRunner()
        result = runner.invoke(
            mailbox_bridge_group,
            [
                "peek-agent",
                "--run-id", "RUN_MAX",
                "--max", "2",
                "--base-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload) == 2

        # Three tasks are still pending, two are in_progress.
        assert len(mb.list_pending()) == 3
        assert len(mb.list_in_progress()) == 2

    def test_peek_llm_call_ignores_agent_tasks(self, tmp_path: Path):
        """Regression guard: the existing `peek` subcommand must not
        claim kind=agent_task items — otherwise the peek-agent
        helper never sees them."""
        mb = TaskMailbox(run_id="RUN_REGRESSION", base_dir=tmp_path)
        _seed_agent_task(mb, "content-generator-aaa")
        _seed_llm_call(mb, "llm_xx")

        runner = CliRunner()
        result = runner.invoke(
            mailbox_bridge_group,
            [
                "peek",
                "--run-id", "RUN_REGRESSION",
                "--max", "10",
                "--base-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        returned_ids = [entry["task_id"] for entry in payload]
        assert returned_ids == ["llm_xx"]
        # agent task is still pending.
        assert mb.list_pending() == ["content-generator-aaa"]


# ------------------------------------------------------- complete-agent


class TestCompleteAgent:
    def test_success_path_writes_envelope(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_OK", base_dir=tmp_path)
        _seed_agent_task(mb, "content-generator-xyz")
        mb.claim("content-generator-xyz")

        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps({
                "success": True,
                "artifacts": ["week_01.html", "week_02.html"],
                "outputs": {"weeks_generated": 2},
            }),
            encoding="utf-8",
        )

        runner = CliRunner()
        invocation = runner.invoke(
            mailbox_bridge_group,
            [
                "complete-agent",
                "--run-id", "RUN_OK",
                "--task-id", "content-generator-xyz",
                "--result-file", str(result_file),
                "--base-dir", str(tmp_path),
            ],
        )
        assert invocation.exit_code == 0, invocation.output
        ack = json.loads(invocation.output)
        assert ack == {"ok": True, "task_id": "content-generator-xyz"}

        envelope = mb.read_completion("content-generator-xyz")
        assert envelope["success"] is True
        assert envelope["result"]["success"] is True
        assert envelope["result"]["artifacts"] == ["week_01.html", "week_02.html"]
        assert envelope["result"]["outputs"] == {"weeks_generated": 2}
        # Operator-stamped dispatch_mode.
        assert envelope["result"]["dispatch_mode"] == "operator"
        # TaskMailbox stamps task_id + run_id + completed_at.
        assert envelope["task_id"] == "content-generator-xyz"
        assert envelope["run_id"] == "RUN_OK"
        assert "completed_at" in envelope

    def test_error_path_writes_failure_envelope(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_FAIL", base_dir=tmp_path)
        _seed_agent_task(mb, "content-generator-fail")
        mb.claim("content-generator-fail")

        runner = CliRunner()
        invocation = runner.invoke(
            mailbox_bridge_group,
            [
                "complete-agent",
                "--run-id", "RUN_FAIL",
                "--task-id", "content-generator-fail",
                "--error", "operator timeout fetching source pages",
                "--error-code", "OPERATOR_TIMEOUT",
                "--base-dir", str(tmp_path),
            ],
        )
        assert invocation.exit_code == 0, invocation.output

        envelope = mb.read_completion("content-generator-fail")
        assert envelope["success"] is False
        assert envelope["error"] == "operator timeout fetching source pages"
        assert envelope["error_code"] == "OPERATOR_TIMEOUT"

    def test_error_without_code_defaults_to_operator_error(self, tmp_path: Path):
        mb = TaskMailbox(run_id="RUN_DEFAULT_CODE", base_dir=tmp_path)
        _seed_agent_task(mb, "content-generator-fail2")
        mb.claim("content-generator-fail2")

        runner = CliRunner()
        invocation = runner.invoke(
            mailbox_bridge_group,
            [
                "complete-agent",
                "--run-id", "RUN_DEFAULT_CODE",
                "--task-id", "content-generator-fail2",
                "--error", "something broke",
                "--base-dir", str(tmp_path),
            ],
        )
        assert invocation.exit_code == 0, invocation.output
        envelope = mb.read_completion("content-generator-fail2")
        assert envelope["success"] is False
        assert envelope["error_code"] == "OPERATOR_ERROR"

    def test_requires_result_file_or_error(self, tmp_path: Path):
        runner = CliRunner()
        invocation = runner.invoke(
            mailbox_bridge_group,
            [
                "complete-agent",
                "--run-id", "R",
                "--task-id", "T",
                "--base-dir", str(tmp_path),
            ],
        )
        assert invocation.exit_code != 0
        assert "One of --result-file or --error is required" in invocation.output

    def test_rejects_both_result_file_and_error(self, tmp_path: Path):
        result_file = tmp_path / "r.json"
        result_file.write_text(
            json.dumps({"success": True, "artifacts": [], "outputs": {}}),
            encoding="utf-8",
        )
        runner = CliRunner()
        invocation = runner.invoke(
            mailbox_bridge_group,
            [
                "complete-agent",
                "--run-id", "R",
                "--task-id", "T",
                "--result-file", str(result_file),
                "--error", "nope",
                "--base-dir", str(tmp_path),
            ],
        )
        assert invocation.exit_code != 0
        assert "mutually exclusive" in invocation.output

    def test_rejects_non_object_result_file(self, tmp_path: Path):
        result_file = tmp_path / "bad.json"
        result_file.write_text("[1, 2, 3]", encoding="utf-8")
        runner = CliRunner()
        invocation = runner.invoke(
            mailbox_bridge_group,
            [
                "complete-agent",
                "--run-id", "R",
                "--task-id", "T",
                "--result-file", str(result_file),
                "--base-dir", str(tmp_path),
            ],
        )
        assert invocation.exit_code != 0
        assert "JSON object" in invocation.output

    def test_rejects_result_file_missing_required_keys(self, tmp_path: Path):
        result_file = tmp_path / "partial.json"
        # Missing artifacts + outputs.
        result_file.write_text(
            json.dumps({"success": True}),
            encoding="utf-8",
        )
        runner = CliRunner()
        invocation = runner.invoke(
            mailbox_bridge_group,
            [
                "complete-agent",
                "--run-id", "R",
                "--task-id", "T",
                "--result-file", str(result_file),
                "--base-dir", str(tmp_path),
            ],
        )
        assert invocation.exit_code != 0
        assert "missing required agent-task keys" in invocation.output


# ------------------------------------------------------- round-trip loop


@pytest.mark.asyncio
async def test_round_trip_dispatcher_to_operator_cli_and_back(tmp_path: Path, monkeypatch):
    """End-to-end: dispatcher.dispatch_task writes a pending agent task
    → CLI ``peek-agent`` claims it → CLI ``complete-agent`` writes the
    completion envelope → dispatcher.dispatch_task unblocks and
    returns the operator-written tool-shape result.

    This exercises the full per-task bridge loop that Session 2's
    CLI subcommands exist to drive.
    """
    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    monkeypatch.setenv("ED4ALL_AGENT_TIMEOUT_SECONDS", "5")

    run_id = "ROUND_TRIP_RUN"
    disp = LocalDispatcher(
        mailbox_base_dir=tmp_path,
        mailbox_poll_interval=0.02,
    )

    # The operator runs on a background thread so the dispatcher's
    # wait_for_completion has something to unblock on. Real ops would
    # be a Claude Code outer session driving subprocess Bash calls,
    # but the loop is the same: peek-agent to claim, dispatch subagent,
    # complete-agent to reply.
    result_file = tmp_path / "operator_result.json"
    result_file.write_text(
        json.dumps({
            "success": True,
            "artifacts": ["generated_page.html"],
            "outputs": {"pages_generated": 1},
        }),
        encoding="utf-8",
    )

    runner = CliRunner()
    claimed_task_id: List[str] = []

    def operator_loop():
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            peek = runner.invoke(
                mailbox_bridge_group,
                [
                    "peek-agent",
                    "--run-id", run_id,
                    "--base-dir", str(tmp_path),
                    "--max", "1",
                ],
            )
            if peek.exit_code != 0:
                return
            payload = json.loads(peek.output)
            if payload:
                task_id = payload[0]["task_id"]
                claimed_task_id.append(task_id)
                complete = runner.invoke(
                    mailbox_bridge_group,
                    [
                        "complete-agent",
                        "--run-id", run_id,
                        "--task-id", task_id,
                        "--result-file", str(result_file),
                        "--base-dir", str(tmp_path),
                    ],
                )
                assert complete.exit_code == 0, complete.output
                return
            time.sleep(0.02)

    op = threading.Thread(target=operator_loop, daemon=True)
    op.start()

    result = await disp.dispatch_task(
        task_name="generate_course_content",
        agent_type="content-generator",
        task_params={"project_id": "P"},
        run_id=run_id,
    )
    op.join(timeout=2.0)

    assert claimed_task_id, "operator loop never saw a pending task"
    assert result["success"] is True
    assert result["artifacts"] == ["generated_page.html"]
    assert result["outputs"] == {"pages_generated": 1}
    # CLI stamps the operator dispatch_mode; dispatcher passes it
    # through unchanged.
    assert result["dispatch_mode"] == "operator"
    assert result["mailbox_task_id"] == claimed_task_id[0]
