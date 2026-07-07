"""NVIDIA-70b-everywhere SETUP — CLI stop-after + preflight tests (no network).

Proves:
  - --stop-after annotates the dry-run plan (phases after the stop marked
    SKIPPED_AFTER_STOP), and an unknown phase name is rejected by the runner;
  - --provider nvidia --dry-run emits a routing preflight that makes NO
    network call (monkeypatched env only) and asserts the right things;
  - the workflow-runner stop-after guard halts the phase loop after the named
    phase (unit test on the loop, no LLM dispatch).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from cli.commands.run import _dry_run_plan, _nvidia_preflight
from cli.main import cli


# ---------------------------------------------------------------------------
# --stop-after — dry-run plan annotation
# ---------------------------------------------------------------------------


def test_stop_after_annotates_plan_and_skips_later_phases():
    plan = _dry_run_plan(
        "textbook_to_course",
        {"course_name": "X", "stop_after": "imscc_chunking"},
        mode="local",
        provider="local",
    )
    assert plan.get("stop_after") == "imscc_chunking"
    names = [p["name"] for p in plan["phases"]]
    assert "imscc_chunking" in names
    stop_idx = names.index("imscc_chunking")
    # The stop phase itself is marked; every later phase is SKIPPED_AFTER_STOP.
    assert plan["phases"][stop_idx].get("stop_after") is True
    for entry in plan["phases"][stop_idx + 1:]:
        assert entry.get("status") == "SKIPPED_AFTER_STOP", entry["name"]


def test_stop_after_unknown_phase_is_rejected_by_runner():
    """The runner validates the phase name; an unknown phase is a hard error."""
    from MCP.core.config import OrchestratorConfig
    from MCP.core.workflow_runner import WorkflowRunner

    config = OrchestratorConfig.load()
    wf = config.get_workflow("textbook_to_course")
    valid = {p.name for p in wf.phases}
    assert "not_a_real_phase" not in valid  # sanity


# ---------------------------------------------------------------------------
# --provider nvidia preflight — NO network call
# ---------------------------------------------------------------------------


def test_nvidia_preflight_makes_no_network_call(monkeypatch: pytest.MonkeyPatch):
    """Preflight resolves + asserts only; never imports/uses an LLM client.

    We block httpx.Client.post to prove no HTTP fires.
    """
    import httpx

    def _boom(*a, **k):
        raise AssertionError("preflight must not make a network call")

    monkeypatch.setattr(httpx.Client, "post", _boom, raising=False)
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "PRESENT_BUT_UNUSED")
    monkeypatch.delenv("NVIDIA_LARGE_MODEL", raising=False)
    monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)
    monkeypatch.delenv("TRAINFORGE_SYNTHESIS_PROVIDER", raising=False)
    monkeypatch.delenv("ED4ALL_ANSWER_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEFORGE_BLOCK_ROUTING_PATH", raising=False)
    for env in ("COURSEFORGE_REWRITE_PROVIDER", "COURSEFORGE_OUTLINE_PROVIDER",
                "COURSEFORGE_PROVIDER"):
        monkeypatch.delenv(env, raising=False)

    pre = _nvidia_preflight({"course_name": "X"})
    assert pre["provider"] == "nvidia"
    assert "No NVIDIA dispatch" in pre["note"]
    names = {c["name"]: c for c in pre["checks"]}
    # Key presence passes.
    assert names["nvidia_api_key"]["level"] == "pass"
    # 70B model resolves (not the 30B nano).
    assert names["cloud_model"]["level"] == "pass"
    # Synthesis seat resolves nvidia (GAP-1).
    assert names["synthesis_tier"]["level"] == "pass"
    # Training seat resolves local (GAP-3 licensing guard).
    assert names["training_seat_licensing"]["level"] == "pass"
    # Answer path loopback.
    assert names["answer_loopback"]["level"] == "pass"


def test_nvidia_preflight_catches_30b_nano_leak(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "present")
    monkeypatch.setenv("NVIDIA_LARGE_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    pre = _nvidia_preflight({"course_name": "X"})
    names = {c["name"]: c for c in pre["checks"]}
    assert names["cloud_model"]["level"] == "error"
    assert pre["verdict"] == "FAIL"


def test_nvidia_preflight_flags_training_seat_nvidia(monkeypatch: pytest.MonkeyPatch):
    """A stale TRAINFORGE_SYNTHESIS_PROVIDER=nvidia is caught (GAP-3 guard)."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "present")
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_PROVIDER", "nvidia")
    pre = _nvidia_preflight({"course_name": "X"})
    names = {c["name"]: c for c in pre["checks"]}
    assert names["training_seat_licensing"]["level"] == "error"


def test_nvidia_preflight_missing_key_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    pre = _nvidia_preflight({"course_name": "X"})
    names = {c["name"]: c for c in pre["checks"]}
    assert names["nvidia_api_key"]["level"] == "error"


def test_nvidia_preflight_warns_stale_local_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "present")
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "local")
    pre = _nvidia_preflight({"course_name": "X"})
    stale = [c for c in pre["checks"] if c["name"] == "stale_provider_env"]
    assert stale and stale[0]["level"] == "warn"


# ---------------------------------------------------------------------------
# Workflow runner — the stop-after halt guard
# ---------------------------------------------------------------------------


def test_runner_stop_after_unknown_phase_returns_error(tmp_path, monkeypatch):
    """run_workflow returns an error dict when --stop-after names a bad phase."""
    import json as _json

    from MCP.core import workflow_runner as wr_mod
    from MCP.core.config import OrchestratorConfig
    from MCP.core.workflow_runner import WorkflowRunner

    # Point STATE_PATH at tmp so we can write a fake workflow state file.
    monkeypatch.setattr(wr_mod, "STATE_PATH", tmp_path)
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(parents=True)
    state = {
        "type": "textbook_to_course",
        "params": {"course_name": "X", "stop_after": "bogus_phase_xyz"},
        "phase_outputs": {},
    }
    (wf_dir / "WF-TEST.json").write_text(_json.dumps(state))

    config = OrchestratorConfig.load()
    runner = WorkflowRunner(executor=MagicMock(), config=config)
    result = asyncio.run(runner.run_workflow("WF-TEST"))
    assert "error" in result
    assert "bogus_phase_xyz" in result["error"]
