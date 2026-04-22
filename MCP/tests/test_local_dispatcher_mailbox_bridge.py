"""Wave 34 tests: LocalDispatcher mailbox bridge.

Covers the new dispatch path introduced in Wave 34:

  LocalDispatcher.dispatch_phase
    |
    +-- agent_tool injected        -> call callable directly (bypass)
    +-- LOCAL_DISPATCHER_ALLOW_STUB -> stubbed PhaseOutput
    +-- otherwise                  -> TaskMailbox put_pending +
                                       wait_for_completion, consume
                                       envelope, return PhaseOutput
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.orchestrator.local_dispatcher import LocalDispatcher  # noqa: E402
from MCP.orchestrator.task_mailbox import TaskMailbox  # noqa: E402
from MCP.orchestrator.worker_contracts import PhaseInput, PhaseOutput  # noqa: E402


def _phase_input(
    phase_name: str = "content_generation",
    run_id: str = "RUN_W34_001",
) -> PhaseInput:
    return PhaseInput(
        run_id=run_id,
        workflow_type="textbook_to_course",
        phase_name=phase_name,
        phase_config={"agents": ["content-generator"], "max_concurrent": 4},
        params={"course_name": "SYNTH_101", "duration_weeks": 2},
        mode="local",
    )


class MockWatcher:
    """Test double for the outer Claude Code session watcher.

    Polls a TaskMailbox for pending tasks and writes synthetic completion
    envelopes back. Runs in a background thread so the dispatcher can
    block on ``wait_for_completion`` realistically.
    """

    def __init__(
        self,
        mailbox: TaskMailbox,
        *,
        envelope_factory=None,
        delay_seconds: float = 0.0,
        poll_interval: float = 0.02,
    ):
        self.mailbox = mailbox
        self.envelope_factory = envelope_factory or self._default_envelope
        self.delay_seconds = delay_seconds
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.claimed_tasks: List[Dict[str, Any]] = []

    @staticmethod
    def _default_envelope(task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        phase_name = (
            spec.get("phase_input", {}).get("phase_name") or "unknown_phase"
        )
        run_id = spec.get("phase_input", {}).get("run_id") or ""
        return {
            "success": True,
            "result": {
                "run_id": run_id,
                "phase_name": phase_name,
                "status": "ok",
                "outputs": {
                    "dispatched_via": "mock_watcher",
                    "task_id": task_id,
                },
            },
        }

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            pending = self.mailbox.list_pending()
            for task_id in pending:
                try:
                    spec = self.mailbox.claim(task_id)
                except Exception:  # noqa: BLE001
                    continue
                self.claimed_tasks.append(spec)
                if self.delay_seconds:
                    time.sleep(self.delay_seconds)
                envelope = self.envelope_factory(task_id, spec)
                self.mailbox.complete(task_id, envelope)
            time.sleep(self.poll_interval)


class TestAgentToolBypassesMailbox:
    @pytest.mark.asyncio
    async def test_callable_injection_skips_mailbox(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        calls: list = []

        async def fake_agent(request):
            calls.append(request)
            return json.dumps({
                "run_id": "RUN_W34_001",
                "phase_name": "content_generation",
                "outputs": {"via": "direct_callable"},
                "status": "ok",
            })

        dispatcher = LocalDispatcher(
            agent_tool=fake_agent,
            project_root=tmp_path,
            mailbox_base_dir=tmp_path / "runs",
            mailbox_timeout_seconds=0.1,  # intentionally tiny — must not hit it
        )
        result = await dispatcher.dispatch_phase(_phase_input())

        assert result.status == "ok"
        assert result.outputs == {"via": "direct_callable"}
        assert len(calls) == 1
        # Mailbox should be untouched.
        mailbox_root = tmp_path / "runs" / "RUN_W34_001" / "mailbox"
        # Either it doesn't exist, or it exists but has no task files.
        if mailbox_root.exists():
            assert list((mailbox_root / "pending").iterdir()) == []


class TestStubFlagBypassesMailbox:
    @pytest.mark.asyncio
    async def test_stub_flag_still_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")
        dispatcher = LocalDispatcher(
            project_root=tmp_path,
            mailbox_base_dir=tmp_path / "runs",
            mailbox_timeout_seconds=0.1,
        )
        result = await dispatcher.dispatch_phase(_phase_input())
        assert result.status == "ok"
        assert result.outputs.get("dispatch_mode") == "stub"


class TestMailboxBridge:
    @pytest.mark.asyncio
    async def test_mock_watcher_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        runs_root = tmp_path / "runs"
        mb = TaskMailbox(run_id="RUN_W34_BRIDGE", base_dir=runs_root)
        watcher = MockWatcher(mb)
        watcher.start()
        try:
            dispatcher = LocalDispatcher(
                project_root=tmp_path,
                mailbox_base_dir=runs_root,
                mailbox_timeout_seconds=5.0,
                mailbox_poll_interval=0.02,
            )
            result = await dispatcher.dispatch_phase(
                _phase_input(run_id="RUN_W34_BRIDGE"),
            )
        finally:
            watcher.stop()

        assert result.status == "ok"
        assert result.outputs.get("dispatched_via") == "mock_watcher"
        assert len(watcher.claimed_tasks) == 1
        # The claimed task spec should carry the phase_input and the prompt.
        spec = watcher.claimed_tasks[0]
        assert spec["phase_input"]["phase_name"] == "content_generation"
        assert "prompt" in spec
        assert spec["subagent_type"] == "content-generator"
        # Metrics tag the mailbox task id for traceability.
        assert "mailbox_task_id" in result.metrics

    @pytest.mark.asyncio
    async def test_mailbox_timeout_fails_with_error_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        dispatcher = LocalDispatcher(
            project_root=tmp_path,
            mailbox_base_dir=tmp_path / "runs",
            mailbox_timeout_seconds=0.1,
            mailbox_poll_interval=0.02,
        )
        result = await dispatcher.dispatch_phase(
            _phase_input(run_id="RUN_W34_TIMEOUT"),
        )
        assert result.status == "fail"
        assert "MAILBOX_TIMEOUT" in (result.error or "")
        assert result.metrics.get("error_code") == "MAILBOX_TIMEOUT"

    @pytest.mark.asyncio
    async def test_watcher_failure_envelope_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        runs_root = tmp_path / "runs"
        mb = TaskMailbox(run_id="RUN_W34_FAIL", base_dir=runs_root)

        def fail_envelope(task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "success": False,
                "error": "subagent hit a validation error on LO refs",
                "error_code": "SUBAGENT_VALIDATION",
            }

        watcher = MockWatcher(mb, envelope_factory=fail_envelope)
        watcher.start()
        try:
            dispatcher = LocalDispatcher(
                project_root=tmp_path,
                mailbox_base_dir=runs_root,
                mailbox_timeout_seconds=5.0,
                mailbox_poll_interval=0.02,
            )
            result = await dispatcher.dispatch_phase(
                _phase_input(run_id="RUN_W34_FAIL"),
            )
        finally:
            watcher.stop()

        assert result.status == "fail"
        assert "validation error" in (result.error or "")
        assert result.metrics.get("error_code") == "SUBAGENT_VALIDATION"

    @pytest.mark.asyncio
    async def test_12_concurrent_tasks_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Stress: 12 dispatched phases with a single MockWatcher.

        Validates that concurrent put_pending calls don't collide and
        each dispatcher call returns the completion meant for its task.
        """
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        runs_root = tmp_path / "runs"
        mb = TaskMailbox(run_id="RUN_W34_CC", base_dir=runs_root)

        def identity_envelope(task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "success": True,
                "result": {
                    "run_id": "RUN_W34_CC",
                    "phase_name": spec["phase_input"]["phase_name"],
                    "outputs": {"echo_phase": spec["phase_input"]["phase_name"]},
                    "status": "ok",
                },
            }

        watcher = MockWatcher(mb, envelope_factory=identity_envelope)
        watcher.start()
        try:
            dispatcher = LocalDispatcher(
                project_root=tmp_path,
                mailbox_base_dir=runs_root,
                mailbox_timeout_seconds=10.0,
                mailbox_poll_interval=0.02,
            )
            coros = [
                dispatcher.dispatch_phase(
                    _phase_input(
                        phase_name=f"phase_{i:02d}", run_id="RUN_W34_CC",
                    )
                )
                for i in range(12)
            ]
            results = await asyncio.gather(*coros)
        finally:
            watcher.stop()

        assert all(r.status == "ok" for r in results)
        phase_names = sorted(r.phase_name for r in results)
        assert phase_names == [f"phase_{i:02d}" for i in range(12)]
        for r in results:
            assert r.outputs.get("echo_phase") == r.phase_name

    @pytest.mark.asyncio
    async def test_cleanup_after_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """After a successful round-trip, the mailbox should not retain
        pending / in_progress / completed files for the completed task."""
        monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
        runs_root = tmp_path / "runs"
        mb = TaskMailbox(run_id="RUN_W34_CLEAN", base_dir=runs_root)
        watcher = MockWatcher(mb)
        watcher.start()
        try:
            dispatcher = LocalDispatcher(
                project_root=tmp_path,
                mailbox_base_dir=runs_root,
                mailbox_timeout_seconds=5.0,
                mailbox_poll_interval=0.02,
            )
            await dispatcher.dispatch_phase(
                _phase_input(run_id="RUN_W34_CLEAN"),
            )
        finally:
            watcher.stop()

        assert mb.list_pending() == []
        assert mb.list_in_progress() == []
        # Completion was cleaned up too (dispatcher owns the envelope now).
        assert mb.list_completed() == []
