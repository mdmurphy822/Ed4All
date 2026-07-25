"""Campaign MUTATING tools — launch / resume / stop: preflights, fixed-argv
spawns, and DecisionCapture wiring.

Every refusal path asserts NO subprocess spawns (an ``_explode`` fake). Every
happy path asserts the EXACT fixed argv, ``start_new_session=True``, and that
``shell`` is never truthy. Hermetic: path constants monkeypatched into
``tmp_path``; ``/proc`` is a fake tree; the launcher is never really run.
"""

from __future__ import annotations

import json
import re

import pytest

from lib.assistant import campaign_tools as ct
from lib.assistant.campaign_tools import dispatch_campaign_tool
from lib.assistant.campaign_tools import ed4all_run_pids as _REAL_ED4ALL_RUN_PIDS


class _FakeFlagError(ValueError):
    pass


_ALLOWED = {"COURSEFORGE_TWO_PASS", "ED4ALL_PLANNING_GATE_RETRIES"}
_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./+:=,-]{0,199}$")


class _FakeFlags:
    CampaignFlagError = _FakeFlagError

    @staticmethod
    def validate_overlay(overlay):
        out = {}
        for key, value in overlay.items():
            if key not in _ALLOWED:
                raise _FakeFlagError(f"unknown or disallowed key: {key}")
            if not _VALUE_RE.match(str(value)):
                raise _FakeFlagError(f"invalid value for {key}")
            out[key] = str(value)
        return out


@pytest.fixture()
def env(monkeypatch, tmp_path):
    campaign = tmp_path / "campaign"
    pending = campaign / "pending-runs"
    review = campaign / "review-queue"
    inputs = tmp_path / "inputs"
    state = tmp_path / "state"
    for path in (campaign, pending, review, inputs, state):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ct, "CAMPAIGN_DIR", campaign)
    monkeypatch.setattr(ct, "PENDING_RUNS_DIR", pending)
    monkeypatch.setattr(ct, "REVIEW_QUEUE_DIR", review)
    monkeypatch.setattr(ct, "LAUNCHED_RUNS_PATH", campaign / "launched-runs.jsonl")
    monkeypatch.setattr(ct, "LAUNCHER_SH", campaign / "launch_book.sh")
    monkeypatch.setattr(ct, "INPUTS_ROOT", inputs)
    monkeypatch.setattr(ct, "STATE_PATH", state)
    monkeypatch.setattr(ct, "_campaign_flags", lambda: _FakeFlags)
    # Default: preflight clear (no live run) unless a test overrides it.
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [])
    return {
        "campaign": campaign,
        "pending": pending,
        "review": review,
        "inputs": inputs,
        "state": state,
    }


def _no_spawn(monkeypatch):
    def _explode(*args, **kwargs):  # pragma: no cover — must not be reached
        raise AssertionError("no subprocess may spawn on a refusal path")

    monkeypatch.setattr(ct.subprocess, "Popen", _explode)
    monkeypatch.setattr(ct.subprocess, "run", _explode)


class _Rec:
    def __init__(self):
        self.decisions = []

    def log_decision(self, **kwargs):
        self.decisions.append(kwargs)


def _write_overlay(env, name="sample-book-a", env_map=None):
    corpus = env["inputs"] / "book.pdf"
    corpus.write_text("x")
    overlay = env["pending"] / f"{name}.json"
    overlay.write_text(
        json.dumps(
            {
                "version": 1,
                "created": "2026-07-22T00:00:00Z",
                "corpus": str(corpus.resolve()),
                "env": env_map or {"COURSEFORGE_TWO_PASS": "true"},
                "note": None,
                "prepared_by": "assistant",
            }
        )
    )
    return overlay


def _write_launched(env, *wf_ids):
    """Seed the campaign launch-manifest so the given wf_ids are campaign-owned
    (the scope MUTATING resume/stop tools require)."""
    path = env["campaign"] / "launched-runs.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        for wf_id in wf_ids:
            handle.write(json.dumps({
                "ts": "2026-07-22T00:00:00Z", "name": None, "corpus": "x",
                "overlay_path": None, "env": {}, "pid": 1, "log_path": "x.log",
                "wf_id": wf_id, "book_slug": None,
            }) + "\n")
    return path


def _make_fake_proc(base, pidmap):
    root = base / "proc"
    root.mkdir(exist_ok=True)
    for pid, tokens in pidmap.items():
        pid_dir = root / str(pid)
        pid_dir.mkdir(exist_ok=True)
        (pid_dir / "cmdline").write_bytes(
            b"\x00".join(t.encode() for t in tokens) + b"\x00"
        )
    return root


# --------------------------------------------------------------------------- #
# campaign_launch_run — refusals (no spawn)
# --------------------------------------------------------------------------- #


def test_launch_refuses_bad_name(env, monkeypatch):
    _no_spawn(monkeypatch)
    assert dispatch_campaign_tool("campaign_launch_run", {"name": "../evil"}).startswith(
        "Refused:"
    )


def test_launch_refuses_missing_overlay(env, monkeypatch):
    _no_spawn(monkeypatch)
    result = dispatch_campaign_tool("campaign_launch_run", {"name": "ghost"})
    assert result.startswith("Refused:")
    assert "no prepared overlay" in result


def test_launch_refuses_tampered_overlay_corpus(env, monkeypatch):
    _no_spawn(monkeypatch)
    overlay = env["pending"] / "tampered.json"
    overlay.write_text(json.dumps({"version": 1, "corpus": "/etc/passwd", "env": {}}))
    result = dispatch_campaign_tool("campaign_launch_run", {"name": "tampered"})
    assert result.startswith("Refused:")
    assert "corpus" in result


def test_launch_refuses_on_stop_all_both_locations(env, monkeypatch):
    _write_overlay(env)
    for rel in ("STOP_ALL", "runs/STOP_ALL"):
        _no_spawn(monkeypatch)
        sentinel = env["state"] / rel
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("")
        result = dispatch_campaign_tool("campaign_launch_run", {"name": "sample-book-a"})
        assert result.startswith("Refused:")
        assert "STOP_ALL" in result
        sentinel.unlink()


def test_launch_refuses_on_existing_run_pair_in_fake_proc(env, monkeypatch, tmp_path):
    _write_overlay(env)
    _no_spawn(monkeypatch)
    # A fake /proc with an existing adjacent ed4all,run pair -> single-owner refusal.
    proc = _make_fake_proc(tmp_path, {321: ["ed4all", "run", "textbook-to-course"]})
    # Exercise the GENUINE ed4all_run_pids against the fake /proc tree.
    monkeypatch.setattr(
        ct, "ed4all_run_pids", lambda proc_root="/proc": _REAL_ED4ALL_RUN_PIDS(proc)
    )
    result = dispatch_campaign_tool("campaign_launch_run", {"name": "sample-book-a"})
    assert result.startswith("Refused:")
    assert "single-owner" in result


# --------------------------------------------------------------------------- #
# campaign_launch_run — happy path
# --------------------------------------------------------------------------- #


def test_launch_happy_path_fixed_argv_and_capture(env, monkeypatch):
    overlay = _write_overlay(env)
    rec = _Rec()
    monkeypatch.setattr(ct, "_get_capture", lambda course_code="campaign": rec)
    spawned = {}

    class _Proc:
        pid = 4242

    def _fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(ct.subprocess, "Popen", _fake_popen)

    result = dispatch_campaign_tool("campaign_launch_run", {"name": "sample-book-a"})
    assert "pid 4242" in result
    assert spawned["argv"] == ["bash", str(ct.LAUNCHER_SH), "--overlay", str(overlay)]
    assert spawned["kwargs"]["start_new_session"] is True
    assert spawned["kwargs"].get("shell") is not True
    assert "shell" not in spawned["kwargs"]  # shell never passed
    # DecisionCapture fired exactly once with the launch decision type.
    assert len(rec.decisions) == 1
    assert rec.decisions[0]["decision_type"] == "campaign_run_launch"
    assert len(rec.decisions[0]["rationale"]) >= 20
    assert "sample-book-a" in rec.decisions[0]["rationale"]


# --------------------------------------------------------------------------- #
# campaign_resume_run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_id",
    [
        "WF-20260101-abcdef01; rm -rf /",
        "$(reboot)",
        "WF-20260101-ABCDEF01",
        "wf-20260101-abcdef01",
        "--force",
        "",
        "WF-20260101-abcdef01\nWF-20260101-deadbeef",
    ],
)
def test_resume_rejects_injection_ids_before_subprocess(env, monkeypatch, bad_id):
    _no_spawn(monkeypatch)
    result = dispatch_campaign_tool("campaign_resume_run", {"run_id": bad_id})
    assert result.startswith("Refused:")
    assert "Nothing was resumed" in result


def test_resume_refuses_on_stop_all(env, monkeypatch):
    _no_spawn(monkeypatch)
    # Campaign-owned run: the manifest check passes, so we exercise the
    # STOP_ALL preflight (not the scoping refusal).
    _write_launched(env, "WF-20260722-abcdef01")
    (env["state"] / "STOP_ALL").write_text("")
    result = dispatch_campaign_tool(
        "campaign_resume_run", {"run_id": "WF-20260722-abcdef01"}
    )
    assert result.startswith("Refused:")
    assert "STOP_ALL" in result


def test_resume_refuses_non_manifest_run(env, monkeypatch):
    # A shape-valid run id that the campaign never launched (empty manifest) is
    # refused BEFORE any preflight/spawn — the anti-foreign-mutation scope.
    _no_spawn(monkeypatch)
    _write_launched(env, "WF-20260722-00000001")  # a DIFFERENT campaign run
    result = dispatch_campaign_tool(
        "campaign_resume_run", {"run_id": "WF-20260722-abcdef01"}
    )
    assert result.startswith("Refused:")
    assert "launch-manifest" in result
    assert "Nothing was resumed" in result


def test_resume_happy_path_plain_no_force(env, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(ct, "_get_capture", lambda course_code="campaign": rec)
    spawned = {}

    class _Proc:
        pid = 55

    def _fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(ct.subprocess, "Popen", _fake_popen)
    rid = "WF-20260722-abcdef01"
    _write_launched(env, rid)  # campaign-owned → passes the manifest scope
    # Pass a stray force kwarg — the whitelist must drop it; argv must be plain.
    result = dispatch_campaign_tool("campaign_resume_run", {"run_id": rid, "force": True})
    assert f"Resume of {rid}" in result
    # WORKFLOW_NAME positional is required by `ed4all run` even on --resume
    # (resume path ignores it, resumes by run_id); the point of this test is
    # that the stray force kwarg is DROPPED — no --force ever reaches the argv.
    assert spawned["argv"] == ["ed4all", "run", "textbook_to_course", "--resume", rid]
    assert "--force" not in spawned["argv"]
    assert spawned["kwargs"]["start_new_session"] is True
    assert "shell" not in spawned["kwargs"]
    assert len(rec.decisions) == 1
    assert rec.decisions[0]["decision_type"] == "campaign_run_control"


# --------------------------------------------------------------------------- #
# campaign_stop_run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_id",
    ["WF-1; rm -rf /", "$(reboot)", "not-a-wf", "", "WF-20260101-ABCDEF01"],
)
def test_stop_rejects_bad_ids_before_subprocess(env, monkeypatch, bad_id):
    _no_spawn(monkeypatch)
    result = dispatch_campaign_tool("campaign_stop_run", {"run_id": bad_id})
    assert result.startswith("Refused:")
    assert "Nothing was stopped" in result


def test_stop_refuses_non_manifest_run(env, monkeypatch):
    # A shape-valid run id the campaign never launched is refused BEFORE the
    # subprocess — foreign runs are not the campaign's to stop.
    _no_spawn(monkeypatch)
    _write_launched(env, "WF-20260722-00000001")  # a DIFFERENT campaign run
    result = dispatch_campaign_tool(
        "campaign_stop_run", {"run_id": "WF-20260722-abcdef01"}
    )
    assert result.startswith("Refused:")
    assert "launch-manifest" in result
    assert "Nothing was stopped" in result


def test_stop_happy_path_fixed_argv_and_capture(env, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(ct, "_get_capture", lambda course_code="campaign": rec)
    called = {}

    class _Proc:
        returncode = 0
        stdout = "Stop sentinel dropped\n"
        stderr = ""

    def _fake_run(argv, **kwargs):
        called["argv"] = argv
        called["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(ct.subprocess, "run", _fake_run)
    rid = "WF-20260722-abcdef01"
    _write_launched(env, rid)  # campaign-owned → passes the manifest scope
    result = dispatch_campaign_tool("campaign_stop_run", {"run_id": rid})
    assert rid in result
    assert called["argv"] == ["ed4all", "stop", rid]
    assert called["kwargs"].get("shell") is not True
    assert "shell" not in called["kwargs"]
    assert len(rec.decisions) == 1
    assert rec.decisions[0]["decision_type"] == "campaign_run_control"


def test_stop_capture_fires_with_real_decision_capture(env, monkeypatch):
    from lib.decision_capture import DecisionCapture

    captured = {}
    real_get = ct._get_capture

    def _tracking(course_code="campaign"):
        cap = real_get(course_code)
        captured["cap"] = cap
        return cap

    class _Proc:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    monkeypatch.setattr(ct.subprocess, "run", lambda argv, **k: _Proc())
    _write_launched(env, "WF-20260722-abcdef01")  # campaign-owned
    ct._get_capture = _tracking
    try:
        dispatch_campaign_tool("campaign_stop_run", {"run_id": "WF-20260722-abcdef01"})
        cap = captured["cap"]
        assert isinstance(cap, DecisionCapture)
        assert len(cap.decisions) == 1
        assert cap.decisions[0]["decision_type"] == "campaign_run_control"
    finally:
        ct._get_capture = real_get
        cap = captured.get("cap")
        if cap is not None:
            cap.close()
