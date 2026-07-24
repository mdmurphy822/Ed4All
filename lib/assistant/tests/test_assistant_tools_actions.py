"""Refusal-first guard tests for the MUTATING assistant tools: start_book,
resume_run, pause_all, clear_stop_all, start_seat/stop_seat co-residency,
support_bundle. Every guard must refuse BEFORE any subprocess spawns."""

from __future__ import annotations

import json

import pytest

from lib.assistant import tools as assistant_tools
from lib.assistant.tools import dispatch_tool


@pytest.fixture()
def fake_campaign(monkeypatch, tmp_path):
    state = tmp_path / "state"
    (state / "runs").mkdir(parents=True)
    campaign = tmp_path / "campaign"
    (campaign / "logs").mkdir(parents=True)
    (campaign / "run_next.py").write_text("print('driver')\n")
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "campaign": "t",
                "books": [
                    {"slug": "book-pending", "status": "pending"},
                    {"slug": "book-running", "status": "running"},
                ],
            }
        )
    )
    monkeypatch.setattr(assistant_tools, "STATE_PATH", state)
    monkeypatch.setattr(assistant_tools, "CAMPAIGN_DIR", campaign)
    monkeypatch.setattr(
        assistant_tools, "LIBV2_COURSES", tmp_path / "libv2" / "courses"
    )
    return {"state": state, "campaign": campaign}


def _no_spawn(monkeypatch):
    def _explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("no subprocess may spawn on a refusal path")

    monkeypatch.setattr(assistant_tools.subprocess, "Popen", _explode)
    monkeypatch.setattr(assistant_tools.subprocess, "run", _explode)


# --------------------------------------------------------------------------- #
# start_book
# --------------------------------------------------------------------------- #


def test_start_book_rejects_unknown_and_nonpending(fake_campaign, monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    _no_spawn(monkeypatch)
    assert dispatch_tool("start_book", {"slug": "ghost"}).startswith("Refused:")
    result = dispatch_tool("start_book", {"slug": "book-running"})
    assert result.startswith("Refused:")
    assert "not 'pending'" in result
    assert dispatch_tool("start_book", {"slug": "../evil"}).startswith("Refused:")


def test_start_book_refuses_when_run_active(fake_campaign, monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: True)
    _no_spawn(monkeypatch)
    result = dispatch_tool("start_book", {"slug": "book-pending"})
    assert result.startswith("Refused:")
    assert "already active" in result


def test_start_book_happy_path_fixed_argv(fake_campaign, monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    spawned = {}

    class _Proc:
        pid = 777

    def _fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "Popen", _fake_popen)
    result = dispatch_tool("start_book", {"slug": "book-pending"})
    assert "pid 777" in result
    assert spawned["argv"][-2:] == ["--slug", "book-pending"]
    assert spawned["kwargs"]["start_new_session"] is True


# --------------------------------------------------------------------------- #
# resume_run
# --------------------------------------------------------------------------- #


def test_resume_run_rejects_bad_id(fake_campaign, monkeypatch):
    _no_spawn(monkeypatch)
    for bad in ("WF-1-2", "--force", "WF-20260101-abcdef01;reboot", ""):
        assert dispatch_tool("resume_run", {"run_id": bad}).startswith("Refused:")


def test_resume_run_refuses_under_stop_all(fake_campaign, monkeypatch):
    (fake_campaign["state"] / "runs" / "STOP_ALL").write_text("")
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    _no_spawn(monkeypatch)
    result = dispatch_tool("resume_run", {"run_id": "WF-20260101-abcdef01"})
    assert result.startswith("Refused:")
    assert "STOP_ALL" in result


def test_resume_run_refuses_when_run_active(fake_campaign, monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: True)
    _no_spawn(monkeypatch)
    result = dispatch_tool("resume_run", {"run_id": "WF-20260101-abcdef01"})
    assert result.startswith("Refused:")


def test_resume_run_happy_path_plain_resume_argv(fake_campaign, monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    spawned = {}

    class _Proc:
        pid = 888

    def _fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "Popen", _fake_popen)
    result = dispatch_tool("resume_run", {"run_id": "WF-20260101-abcdef01"})
    assert "spawned detached" in result
    assert spawned["argv"][-2:] == ["--resume", "WF-20260101-abcdef01"]
    assert "--force" not in spawned["argv"]  # PLAIN resume, never --force


# --------------------------------------------------------------------------- #
# pause_all / clear_stop_all
# --------------------------------------------------------------------------- #


def test_pause_all_fixed_argv(monkeypatch):
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "STOP_ALL written\n"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["argv"] = argv
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("pause_all", {})
    assert calls["argv"][-2:] == ["stop", "--all"]
    assert "STOP_ALL" in result


@pytest.mark.parametrize("confirm", [False, "true", "yes", 1, None])
def test_clear_stop_all_requires_exact_true(monkeypatch, confirm):
    _no_spawn(monkeypatch)
    result = dispatch_tool("clear_stop_all", {"confirm": confirm})
    assert result.startswith("Refused:")
    assert "Nothing was cleared" in result


def test_clear_stop_all_confirmed(monkeypatch):
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "cleared\n"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["argv"] = argv
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("clear_stop_all", {"confirm": True})
    assert calls["argv"][-2:] == ["stop", "--clear-all"]
    assert "STOP_ALL cleared" in result


# --------------------------------------------------------------------------- #
# start_seat / stop_seat — registry + co-residency law
# --------------------------------------------------------------------------- #

_REGISTRY = {
    "spark-super": "http://localhost:8001",
    "spark-glm": "http://localhost:8002",
    "spark-qwen": "http://localhost:8003",
    "spark-nano": "http://localhost:8004",
}


def _wire_seats(monkeypatch, live):
    monkeypatch.setattr(assistant_tools, "parse_seat_registry", lambda: dict(_REGISTRY))
    live_urls = {_REGISTRY[s] for s in live}
    monkeypatch.setattr(
        assistant_tools, "_probe_ready", lambda url: url in live_urls
    )


def test_start_seat_rejects_unregistered(monkeypatch):
    _wire_seats(monkeypatch, live=set())
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    result = dispatch_tool("start_seat", {"seat": "evil-seat"})
    assert result.startswith("Refused:")
    assert "spark-nano" in result  # names the registry


def test_start_seat_refused_during_run(monkeypatch):
    _wire_seats(monkeypatch, live=set())
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: True)
    result = dispatch_tool("start_seat", {"seat": "spark-nano"})
    assert result.startswith("Refused:")
    assert "seat schedule" in result


def test_start_seat_super_never_coresident(monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    # super up, nano requested → refuse
    _wire_seats(monkeypatch, live={"spark-super"})
    result = dispatch_tool("start_seat", {"seat": "spark-nano"})
    assert result.startswith("Refused:")
    assert "spark-super" in result
    # nano up, super requested → refuse
    _wire_seats(monkeypatch, live={"spark-nano"})
    result = dispatch_tool("start_seat", {"seat": "spark-super"})
    assert result.startswith("Refused:")


def test_start_seat_budget_ceiling(monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    # glm(0.25) + qwen(0.40) live; nano(0.30) would sum to 0.95 <= budget → OK;
    # but qwen+glm+nano+... use a case that overflows: glm+qwen live, start
    # nano = 0.95 exactly (allowed), so overflow case: qwen+nano live, start
    # glm+? — use fractions: qwen(0.40)+nano(0.30) live, start glm(0.25)=0.95 ok.
    # Overflow: patch fraction table for a clear >0.95 combo.
    monkeypatch.setattr(
        assistant_tools,
        "SEAT_GPU_FRACTION",
        {"spark-glm": 0.5, "spark-qwen": 0.5, "spark-nano": 0.3, "spark-super": 0.7292},
    )
    _wire_seats(monkeypatch, live={"spark-glm"})
    result = dispatch_tool("start_seat", {"seat": "spark-qwen"})
    assert result.startswith("Refused:")
    assert "0.95" in result


def test_start_seat_already_serving_is_noop(monkeypatch):
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    _wire_seats(monkeypatch, live={"spark-nano"})
    result = dispatch_tool("start_seat", {"seat": "spark-nano"})
    assert "already serving" in result


# --------------------------------------------------------------------------- #
# Campaign-mode start_seat guard: the assistant may start ONLY its own chat
# seat (ED4ALL_ASSISTANT_SEAT, default spark-nano) — never spark-super or any
# other pipeline seat (owned by the pipeline seat schedule).
# --------------------------------------------------------------------------- #


def test_campaign_mode_start_seat_refuses_super(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSISTANT_SEAT", raising=False)
    _wire_seats(monkeypatch, live=set())
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    result = dispatch_tool(
        "start_seat", {"seat": "spark-super"}, campaign_mode=True
    )
    assert result.startswith("Refused:")
    assert "own chat seat" in result
    assert "spark-nano" in result  # names the allowed seat / the rule
    assert "spark-super" in result


def test_campaign_mode_start_seat_refuses_any_foreign_seat(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSISTANT_SEAT", raising=False)
    _wire_seats(monkeypatch, live=set())
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    # A non-super pipeline seat is refused too — only the OWN seat is allowed.
    result = dispatch_tool(
        "start_seat", {"seat": "spark-qwen"}, campaign_mode=True
    )
    assert result.startswith("Refused:")
    assert "own chat seat" in result


def test_campaign_mode_start_seat_allows_own_seat(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSISTANT_SEAT", raising=False)
    # nano is live → own-seat guard passes, normal logic returns the noop.
    _wire_seats(monkeypatch, live={"spark-nano"})
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    result = dispatch_tool(
        "start_seat", {"seat": "spark-nano"}, campaign_mode=True
    )
    assert "own chat seat" not in result  # guard did NOT fire
    assert "already serving" in result


def test_campaign_mode_start_seat_honors_assistant_seat_env(monkeypatch):
    # Point the assistant's own seat at spark-glm; then spark-glm is allowed
    # and the default spark-nano becomes a foreign seat.
    monkeypatch.setenv("ED4ALL_ASSISTANT_SEAT", "spark-glm")
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    _wire_seats(monkeypatch, live={"spark-glm"})
    allowed = dispatch_tool(
        "start_seat", {"seat": "spark-glm"}, campaign_mode=True
    )
    assert "own chat seat" not in allowed
    refused = dispatch_tool(
        "start_seat", {"seat": "spark-nano"}, campaign_mode=True
    )
    assert refused.startswith("Refused:")
    assert "spark-glm" in refused  # names the (env-overridden) own seat


def test_non_campaign_start_seat_unchanged(monkeypatch):
    # Non-campaign path: the own-seat guard NEVER fires — spark-super is subject
    # only to the co-residency law (here: no live seats → it would start).
    monkeypatch.delenv("ED4ALL_ASSISTANT_SEAT", raising=False)
    _wire_seats(monkeypatch, live=set())
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)

    calls = {}

    def _fake_coherent(seat):
        calls["seat"] = seat
        return type("R", (), {"ok": True, "reason": "ok", "recreated": False})()

    import lib.vllm_container_lifecycle as _lifecycle

    monkeypatch.setattr(_lifecycle, "start_seat_coherent", _fake_coherent)
    result = dispatch_tool("start_seat", {"seat": "spark-super"})
    assert "own chat seat" not in result
    assert calls.get("seat") == "spark-super"  # foreign seat proceeds normally


# --------------------------------------------------------------------------- #
# Read-only (campaign-tick) dispatch guard: mutating base tools are refused.
# --------------------------------------------------------------------------- #

_MUTATING_BASE_TOOLS = (
    "start_next_book", "start_book", "resume_run", "stop_run", "pause_all",
    "clear_stop_all", "start_seat", "stop_seat", "support_bundle",
)


@pytest.mark.parametrize("tool", _MUTATING_BASE_TOOLS)
def test_readonly_dispatch_refuses_mutating_base_tools(tool):
    result = dispatch_tool(tool, {}, readonly=True)
    assert result.startswith("Refused:")
    assert "read-only campaign-tick surface" in result


def test_readonly_dispatch_allows_readonly_base_tool():
    # A read-only tool passes the guard: it never emits the mutating-tool
    # refusal (run_status returns its own recent-runs summary text).
    result = dispatch_tool("run_status", {}, readonly=True)
    assert "read-only campaign-tick surface" not in result


def test_readonly_tool_names_exclude_every_mutating_tool():
    for tool in _MUTATING_BASE_TOOLS:
        assert tool not in assistant_tools.READONLY_TOOL_NAMES
    # …and READONLY_TOOL_SCHEMAS carries exactly the read-only names.
    schema_names = {s["function"]["name"] for s in assistant_tools.READONLY_TOOL_SCHEMAS}
    assert schema_names == set(assistant_tools.READONLY_TOOL_NAMES)


def test_stop_seat_refused_during_run(monkeypatch):
    _wire_seats(monkeypatch, live={"spark-nano"})
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: True)
    result = dispatch_tool("stop_seat", {"seat": "spark-nano"})
    assert result.startswith("Refused:")


def test_stop_seat_rejects_unregistered(monkeypatch):
    _wire_seats(monkeypatch, live=set())
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    assert dispatch_tool("stop_seat", {"seat": "ghost"}).startswith("Refused:")


def test_stop_seat_already_down_is_noop(monkeypatch):
    _wire_seats(monkeypatch, live=set())
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    assert "already down" in dispatch_tool("stop_seat", {"seat": "spark-nano"})


# --------------------------------------------------------------------------- #
# support_bundle
# --------------------------------------------------------------------------- #


def test_support_bundle_fixed_argv_and_path(monkeypatch, tmp_path):
    state = tmp_path / "state"
    monkeypatch.setattr(assistant_tools, "STATE_PATH", state)
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "bundle ok\n"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["argv"] = argv
        # simulate the CLI writing the bundle
        out = argv[argv.index("--output") + 1]
        with open(out, "wb") as handle:
            handle.write(b"tarball")
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("support_bundle", {})
    assert "support-bundle" in calls["argv"]
    out_arg = calls["argv"][calls["argv"].index("--output") + 1]
    assert out_arg.startswith(str(state / "support_bundles"))
    assert "7 bytes" in result
