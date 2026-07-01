"""Tests for the APIDispatcher (Wave 7)."""
from __future__ import annotations

import asyncio

import pytest

from MCP.orchestrator.api_dispatcher import (
    APIDispatcher,
    APIDispatcherStubNotAllowed,
    _ALLOW_STUB_ENV,
)
from MCP.orchestrator.llm_backend import MockBackend
from MCP.orchestrator.worker_contracts import PhaseInput, PhaseOutput


def _phase_input(phase: str = "content_generation") -> PhaseInput:
    return PhaseInput(
        run_id="RUN_API_001",
        workflow_type="textbook_to_course",
        phase_name=phase,
        phase_config={"agents": ["content-generator"], "max_concurrent": 3},
        params={"week_range": "1-3"},
        mode="api",
        llm_factory=lambda: MockBackend(responses=["generated"]),
    )


class TestAPIDispatcherStub:
    @pytest.mark.asyncio
    async def test_stub_when_no_worker(self):
        dispatcher = APIDispatcher()
        result = await dispatcher.dispatch_phase(_phase_input())
        assert isinstance(result, PhaseOutput)
        assert result.status == "ok"
        assert result.outputs.get("dispatch_mode") == "stub"

    @pytest.mark.asyncio
    async def test_dispatched_tracked(self):
        dispatcher = APIDispatcher()
        await dispatcher.dispatch_phase(_phase_input(phase="p1"))
        await dispatcher.dispatch_phase(_phase_input(phase="p2"))
        dispatched = await dispatcher.after_run(workflow_id="W1", result={})
        assert dispatched == ["p1", "p2"]


class TestAPIDispatcherTaskStubGate:
    """W2.5: dispatch_task's fake-success stub must be gated behind
    LOCAL_DISPATCHER_ALLOW_STUB so a real run never silently succeeds
    with empty outputs."""

    @pytest.mark.asyncio
    async def test_dispatch_task_raises_when_stub_flag_off(self, monkeypatch):
        monkeypatch.delenv(_ALLOW_STUB_ENV, raising=False)
        dispatcher = APIDispatcher()
        with pytest.raises(APIDispatcherStubNotAllowed):
            await dispatcher.dispatch_task(
                task_name="generate_content",
                agent_type="content-generator",
                task_params={"week": 1},
                run_id="RUN_API_002",
            )

    @pytest.mark.asyncio
    async def test_dispatch_task_raises_on_garbage_flag(self, monkeypatch):
        # Parse-with-fallback: a non-truthy token stays off → still raises.
        monkeypatch.setenv(_ALLOW_STUB_ENV, "banana")
        dispatcher = APIDispatcher()
        with pytest.raises(APIDispatcherStubNotAllowed):
            await dispatcher.dispatch_task(
                task_name="generate_content",
                agent_type="content-generator",
                task_params={},
                run_id="RUN_API_003",
            )

    @pytest.mark.asyncio
    async def test_dispatch_task_stub_when_flag_on(self, monkeypatch):
        monkeypatch.setenv(_ALLOW_STUB_ENV, "1")
        dispatcher = APIDispatcher()
        result = await dispatcher.dispatch_task(
            task_name="generate_content",
            agent_type="content-generator",
            task_params={"week": 1},
            run_id="RUN_API_004",
        )
        assert result["success"] is True
        assert result["dispatch_mode"] == "api_stub"
        assert result["agent_type"] == "content-generator"
        assert result["tool_name"] == "generate_content"
        assert result["outputs"] == {}


class TestAPIDispatcherWorker:
    @pytest.mark.asyncio
    async def test_worker_invoked(self):
        async def worker(pi: PhaseInput) -> PhaseOutput:
            backend = pi.llm_factory()
            text = await backend.complete("sys", "go")
            return PhaseOutput(
                run_id=pi.run_id,
                phase_name=pi.phase_name,
                outputs={"generated": text},
                status="ok",
            )

        dispatcher = APIDispatcher()
        result = await dispatcher.dispatch_phase(_phase_input(), worker=worker)
        assert result.status == "ok"
        assert result.outputs["generated"] == "generated"

    @pytest.mark.asyncio
    async def test_worker_exception_becomes_fail(self):
        async def bad_worker(pi: PhaseInput) -> PhaseOutput:
            raise RuntimeError("kaboom")

        dispatcher = APIDispatcher()
        result = await dispatcher.dispatch_phase(_phase_input(), worker=bad_worker)
        assert result.status == "fail"
        assert "kaboom" in (result.error or "")


class TestAPIDispatcherBatch:
    @pytest.mark.asyncio
    async def test_batch_respects_concurrency(self):
        concurrent_peak = {"max": 0, "active": 0}
        lock = asyncio.Lock()

        async def worker(pi: PhaseInput) -> PhaseOutput:
            async with lock:
                concurrent_peak["active"] += 1
                concurrent_peak["max"] = max(
                    concurrent_peak["max"], concurrent_peak["active"]
                )
            await asyncio.sleep(0.01)
            async with lock:
                concurrent_peak["active"] -= 1
            return PhaseOutput(
                run_id=pi.run_id, phase_name=pi.phase_name, status="ok"
            )

        dispatcher = APIDispatcher()
        inputs = [_phase_input(phase=f"p{i}") for i in range(10)]
        results = await dispatcher.dispatch_batch(
            inputs, worker, max_concurrent=3
        )
        assert len(results) == 10
        assert all(r.status == "ok" for r in results)
        # Peak concurrent should not exceed max_concurrent
        assert concurrent_peak["max"] <= 3

    @pytest.mark.asyncio
    async def test_batch_all_complete(self):
        async def worker(pi: PhaseInput) -> PhaseOutput:
            return PhaseOutput(
                run_id=pi.run_id,
                phase_name=pi.phase_name,
                outputs={"ran": pi.phase_name},
                status="ok",
            )

        dispatcher = APIDispatcher()
        inputs = [_phase_input(phase=f"phase_{i}") for i in range(5)]
        results = await dispatcher.dispatch_batch(inputs, worker)
        names = {r.outputs["ran"] for r in results}
        assert names == {f"phase_{i}" for i in range(5)}
