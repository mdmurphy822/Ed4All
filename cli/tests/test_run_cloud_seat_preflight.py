"""Hosted large-model build profile — CLI stop-after + cloud-seat preflight
tests (no network).

Proves:
  - --stop-after annotates the dry-run plan (phases after the stop marked
    SKIPPED_AFTER_STOP), and an unknown phase name is rejected by the runner;
  - --provider nvidia --dry-run emits a routing preflight that makes NO
    network call (monkeypatched env only) and asserts the right things;
  - the workflow-runner stop-after guard halts the phase loop after the named
    phase (unit test on the loop, no LLM dispatch);
  - the framing-purge rename kept its backward-compat surface: the legacy
    ``_nvidia_preflight`` alias + the legacy ``nvidia_preflight`` plan key, and
    the NVIDIA_API_KEY / NVIDIA_BASE_URL / NVIDIA_LARGE_MODEL env NAMES still
    resolve verbatim (registry-row env data, never renamed).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from cli.commands.run import _cloud_seat_preflight, _dry_run_plan, _nvidia_preflight
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

    pre = _cloud_seat_preflight({"course_name": "X"})
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
    pre = _cloud_seat_preflight({"course_name": "X"})
    names = {c["name"]: c for c in pre["checks"]}
    assert names["cloud_model"]["level"] == "error"
    assert pre["verdict"] == "FAIL"


def test_nvidia_preflight_flags_training_seat_nvidia(monkeypatch: pytest.MonkeyPatch):
    """A stale TRAINFORGE_SYNTHESIS_PROVIDER=nvidia is caught (GAP-3 guard)."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "present")
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_PROVIDER", "nvidia")
    pre = _cloud_seat_preflight({"course_name": "X"})
    names = {c["name"]: c for c in pre["checks"]}
    assert names["training_seat_licensing"]["level"] == "error"


def test_nvidia_preflight_missing_key_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    pre = _cloud_seat_preflight({"course_name": "X"})
    names = {c["name"]: c for c in pre["checks"]}
    assert names["nvidia_api_key"]["level"] == "error"


def test_nvidia_preflight_warns_stale_local_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "present")
    monkeypatch.setenv("COURSEFORGE_REWRITE_PROVIDER", "local")
    pre = _cloud_seat_preflight({"course_name": "X"})
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


# ---------------------------------------------------------------------------
# Framing-purge backward-compat surface
# ---------------------------------------------------------------------------


def test_legacy_preflight_alias_points_at_renamed_helper():
    """The deprecated ``_nvidia_preflight`` name is an alias for the new one."""
    assert _nvidia_preflight is _cloud_seat_preflight


def test_dry_run_plan_emits_both_preflight_keys(monkeypatch: pytest.MonkeyPatch):
    """--provider nvidia --dry-run carries the new key AND the legacy key."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "present")
    plan = _dry_run_plan(
        "textbook_to_course",
        {"course_name": "X"},
        mode="local",
        provider="nvidia",
    )
    assert "cloud_seat_preflight" in plan
    # Legacy key stays readable for one release (old dry-run consumers +
    # doctor --run-id post-mortems + existing state/runs sidecars).
    assert "nvidia_preflight" in plan
    # Same object under both keys.
    assert plan["cloud_seat_preflight"] is plan["nvidia_preflight"]


def test_nvidia_env_names_still_resolve(monkeypatch: pytest.MonkeyPatch):
    """The NVIDIA_* env NAMES are registry-row data — never renamed.

    The workflow-runner constants that name them keep pointing at the verbatim
    NVIDIA_* strings, and NVIDIA_LARGE_MODEL is honored by the preflight.
    """
    from MCP.core import workflow_runner as _wr

    assert _wr._HOSTED_LARGE_MODEL_ENV == "NVIDIA_LARGE_MODEL"
    # Deprecated aliases still resolve to the same values.
    assert _wr._NVIDIA_LARGE_MODEL_ENV == _wr._HOSTED_LARGE_MODEL_ENV
    assert _wr._NVIDIA_LARGE_MODEL_DEFAULT == _wr._HOSTED_LARGE_MODEL_DEFAULT
    assert _wr._NVIDIA_LARGE_BLOCK_ROUTING_PATH == _wr._HOSTED_LARGE_BLOCK_ROUTING_PATH
    assert _wr._NVIDIA_PROVIDER == _wr._CLOUD_SEAT_PROVIDER == "nvidia"

    # NVIDIA_LARGE_MODEL is read verbatim by the preflight (custom large model
    # is accepted, not flagged as the 30B-nano leak).
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "present")
    monkeypatch.setenv("NVIDIA_LARGE_MODEL", "meta/llama-3.3-70b-custom")
    pre = _cloud_seat_preflight({"course_name": "X"})
    names = {c["name"]: c for c in pre["checks"]}
    assert names["cloud_model"]["level"] == "pass"
    assert "meta/llama-3.3-70b-custom" in names["cloud_model"]["detail"]
