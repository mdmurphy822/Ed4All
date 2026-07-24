"""Sandbox tests for the operator-assistant tool whitelist.

The load-bearing contracts: unknown tool names are refused (never
executed), ``stop_run`` rejects anything that is not a canonical run id
BEFORE a subprocess argv is built, and subprocess calls use fixed argv
lists (no shell).
"""

from __future__ import annotations

import json

import pytest

from lib.assistant import tools as assistant_tools
from lib.assistant.tools import RUN_ID_RE, dispatch_tool


# --------------------------------------------------------------------------- #
# Whitelist dispatch
# --------------------------------------------------------------------------- #


def test_unknown_tool_is_refused_not_executed():
    result = dispatch_tool("read_file", {"path": "/etc/passwd"})
    assert result.startswith("Refused: tool 'read_file'")
    assert "campaign_status" in result  # names the real whitelist


def test_shell_shaped_tool_name_refused():
    result = dispatch_tool("bash", {"command": "rm -rf /"})
    assert result.startswith("Refused: tool 'bash'")


def test_extra_arguments_are_dropped_not_forwarded():
    # get_help only accepts `topic`; a smuggled kwarg must not raise or leak.
    result = dispatch_tool("get_help", {"topic": "run", "path": "/etc/passwd"})
    assert "ed4all run" in result


def test_registry_and_schemas_agree():
    schema_names = {
        entry["function"]["name"] for entry in assistant_tools.TOOL_SCHEMAS
    }
    assert schema_names == set(assistant_tools.TOOL_REGISTRY)


# --------------------------------------------------------------------------- #
# stop_run — run-id validation is the injection guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_id",
    [
        "WF-20260101-abcdef01; rm -rf /",
        "$(reboot)",
        "WF-20260101-ABCDEF01",  # uppercase hex not canonical
        "WF-2026-abcdef01",
        "wf-20260101-abcdef01",
        "--all",
        "",
        "WF-20260101-abcdef01\nWF-20260101-deadbeef",
    ],
)
def test_stop_run_rejects_injection_shaped_ids(monkeypatch, bad_id):
    def _explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("subprocess must not run for an invalid run id")

    monkeypatch.setattr(assistant_tools.subprocess, "run", _explode)
    result = dispatch_tool("stop_run", {"run_id": bad_id})
    assert result.startswith("Refused:")
    assert "Nothing was stopped" in result


def test_stop_run_valid_id_uses_fixed_argv(monkeypatch):
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "Stop sentinel written\n"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["argv"] = argv
        assert kwargs.get("shell") is not True
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("stop_run", {"run_id": "WF-20260420-abc12345"})
    assert "WF-20260420-abc12345" in result
    # Fixed argv list: interpreter -m cli.main stop <validated id>.
    assert isinstance(calls["argv"], list)
    assert calls["argv"][-2:] == ["stop", "WF-20260420-abc12345"]
    assert calls["argv"][1:3] == ["-m", "cli.main"]


def test_stop_run_requires_run_id_argument():
    result = dispatch_tool("stop_run", {})
    assert result.startswith("Refused:")


def test_run_id_regex_shape():
    assert RUN_ID_RE.match("WF-20260420-abc12345")
    assert not RUN_ID_RE.match("WF-20260420-abc1234")  # 7 hex chars
    assert not RUN_ID_RE.match("WF-20260420-abc123456")  # 9 hex chars


# --------------------------------------------------------------------------- #
# campaign_status / run_status / start_next_book
# --------------------------------------------------------------------------- #


def test_campaign_status_missing_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(assistant_tools, "CAMPAIGN_DIR", tmp_path / "nope")
    assert "no campaign configured" in dispatch_tool("campaign_status", {})


def test_campaign_status_counts_by_status(monkeypatch, tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "campaign": "test-campaign",
                "books": [
                    {"slug": "book-a", "status": "built"},
                    {"slug": "book-b", "status": "running"},
                    {"slug": "book-c", "status": "pending"},
                    {"slug": "book-d", "status": "pending"},
                ],
            }
        )
    )
    monkeypatch.setattr(assistant_tools, "CAMPAIGN_DIR", campaign)
    result = dispatch_tool("campaign_status", {})
    assert "4 book(s)" in result
    assert "pending=2" in result
    assert "running=1" in result
    assert "Currently running: book-b" in result


def test_run_status_reads_state_runs(monkeypatch, tmp_path):
    state = tmp_path / "state"
    run_dir = state / "runs" / "WF-20260420-abc12345"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "paused",
                "workflow_type": "textbook_to_course",
                "created_at": "2026-07-21T10:00:00",
            }
        )
    )
    monkeypatch.setattr(assistant_tools, "STATE_PATH", state)
    result = dispatch_tool("run_status", {})
    assert "WF-20260420-abc12345" in result
    assert "status=paused" in result


def test_start_next_book_refuses_without_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(assistant_tools, "CAMPAIGN_DIR", tmp_path / "absent")
    result = dispatch_tool("start_next_book", {})
    assert result.startswith("Refused:")
    assert "does not exist" in result


def test_start_next_book_refuses_when_run_active(monkeypatch, tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "run_next.py").write_text("print('driver')\n")
    monkeypatch.setattr(assistant_tools, "CAMPAIGN_DIR", campaign)
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: True)

    def _explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("driver must not spawn while a run is active")

    monkeypatch.setattr(assistant_tools.subprocess, "Popen", _explode)
    result = dispatch_tool("start_next_book", {})
    assert result.startswith("Refused:")
    assert "already active" in result


def test_start_next_book_spawns_detached(monkeypatch, tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "run_next.py").write_text("print('driver')\n")
    monkeypatch.setattr(assistant_tools, "CAMPAIGN_DIR", campaign)
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)

    spawned = {}

    class _Proc:
        pid = 4242

    def _fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "Popen", _fake_popen)
    result = dispatch_tool("start_next_book", {})
    assert "pid 4242" in result
    assert spawned["argv"][-1].endswith("run_next.py")
    assert spawned["kwargs"]["start_new_session"] is True
    # Log landed in the campaign logs dir.
    assert list((campaign / "logs").glob("assistant_start_*.log"))


# --------------------------------------------------------------------------- #
# get_help — curated static data only
# --------------------------------------------------------------------------- #


def test_get_help_known_topics():
    assert "stop sentinel" in dispatch_tool("get_help", {"topic": "stop"})
    assert "--resume" in dispatch_tool("get_help", {"topic": "resume"})
    assert "ED4ALL_CAMPAIGN_DIR" in dispatch_tool("get_help", {"topic": "campaign"})
    assert "ED4ALL_SEAT_BASE_URLS" in dispatch_tool("get_help", {"topic": "seats"})


def test_get_help_unknown_topic_lists_topics():
    result = dispatch_tool("get_help", {"topic": "kubernetes"})
    assert "Available help topics" in result
    assert "run" in result
