"""Campaign tool set — queue / prepare / status / report + registry + dispatch
+ the public preflight / review-queue helpers.

Hermetic: every module path constant is monkeypatched into ``tmp_path``; the
campaign_flags dependency is injected via the ``_campaign_flags`` seam; ``/proc``
is a fake tree; no subprocess of a real seat/run ever happens. Tests NEVER touch
the real campaign, state/, or inputs/ trees.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from lib.assistant import campaign_tools as ct
from lib.assistant.campaign_tools import dispatch_campaign_tool


# --------------------------------------------------------------------------- #
# Fake campaign_flags (mirrors the S3 contract enough for integration)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


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
    return {
        "campaign": campaign,
        "pending": pending,
        "review": review,
        "inputs": inputs,
        "state": state,
    }


def _no_spawn(monkeypatch):
    def _explode(*args, **kwargs):  # pragma: no cover — must not be reached
        raise AssertionError("no subprocess may spawn on this path")

    monkeypatch.setattr(ct.subprocess, "Popen", _explode)
    monkeypatch.setattr(ct.subprocess, "run", _explode)


def _make_fake_proc(base, pidmap):
    root = base / "proc"
    root.mkdir(exist_ok=True)
    for pid, tokens in pidmap.items():
        pid_dir = root / str(pid)
        pid_dir.mkdir(exist_ok=True)
        (pid_dir / "cmdline").write_bytes(
            b"\x00".join(t.encode() for t in tokens) + b"\x00"
        )
    (root / "not-a-pid").mkdir(exist_ok=True)
    return root


class _Rec:
    def __init__(self):
        self.decisions = []

    def log_decision(self, **kwargs):
        self.decisions.append(kwargs)


# --------------------------------------------------------------------------- #
# Registry <-> schemas <-> whitelist agreement (mirror tools.py's test)
# --------------------------------------------------------------------------- #


def test_registry_and_schemas_agree():
    schema_names = {entry["function"]["name"] for entry in ct.CAMPAIGN_TOOL_SCHEMAS}
    assert schema_names == set(ct.CAMPAIGN_TOOL_REGISTRY)


def test_whitelist_and_required_cover_registry():
    assert set(ct._CAMPAIGN_TOOL_ARG_WHITELIST) == set(ct.CAMPAIGN_TOOL_REGISTRY)
    # Required args are a subset of the whitelist for every tool.
    for name, required in ct._CAMPAIGN_TOOL_REQUIRED_ARGS.items():
        assert set(required) <= set(ct._CAMPAIGN_TOOL_ARG_WHITELIST[name])
    # Schema `required` matches the enforced required set.
    for entry in ct.CAMPAIGN_TOOL_SCHEMAS:
        name = entry["function"]["name"]
        schema_required = set(entry["function"]["parameters"]["required"])
        assert schema_required == set(ct._CAMPAIGN_TOOL_REQUIRED_ARGS.get(name, ()))


def test_resume_schema_never_exposes_force():
    entry = next(
        e for e in ct.CAMPAIGN_TOOL_SCHEMAS
        if e["function"]["name"] == "campaign_resume_run"
    )
    props = entry["function"]["parameters"]["properties"]
    assert set(props) == {"run_id"}
    assert "force" not in props
    assert "force" not in ct._CAMPAIGN_TOOL_ARG_WHITELIST["campaign_resume_run"]


# --------------------------------------------------------------------------- #
# dispatch: unknown / extra args / missing required
# --------------------------------------------------------------------------- #


def test_dispatch_unknown_tool_refused_not_executed(env, monkeypatch):
    _no_spawn(monkeypatch)
    result = dispatch_campaign_tool("no_such_tool", {"x": 1})
    assert result.startswith("Refused:")
    assert "not in the campaign tool whitelist" in result


def test_dispatch_missing_required_refused(env):
    assert dispatch_campaign_tool("campaign_prepare_run", {}).startswith(
        "Refused: campaign_prepare_run requires a corpus argument"
    )
    assert dispatch_campaign_tool("campaign_report", {"kind": "campaign_note"}).startswith(
        "Refused: campaign_report requires a summary argument"
    )


def test_dispatch_extra_args_dropped(env):
    # A stray kwarg would TypeError the tool fn if it were NOT dropped; the call
    # succeeding proves the whitelist filter dropped it.
    result = dispatch_campaign_tool(
        "campaign_report",
        {"kind": "campaign_note", "summary": "hello there", "bogus": 1, "force": True},
    )
    assert result.startswith("Wrote review report")


# --------------------------------------------------------------------------- #
# dispatch: readonly (campaign-tick) surface refuses mutating campaign tools
# --------------------------------------------------------------------------- #

_MUTATING_CAMPAIGN_TOOLS = (
    "campaign_prepare_run",
    "campaign_launch_run",
    "campaign_resume_run",
    "campaign_stop_run",
    "campaign_prepare_training",
    "campaign_launch_training",
)


@pytest.mark.parametrize("tool", _MUTATING_CAMPAIGN_TOOLS)
def test_readonly_dispatch_refuses_mutating_campaign_tools(env, monkeypatch, tool):
    _no_spawn(monkeypatch)
    result = dispatch_campaign_tool(tool, {}, readonly=True)
    assert result.startswith("Refused:")
    assert "read-only tick surface" in result


def test_readonly_dispatch_allows_readonly_campaign_tools(env, monkeypatch):
    _no_spawn(monkeypatch)
    # campaign_queue / campaign_run_status / campaign_report all pass the guard.
    assert not dispatch_campaign_tool("campaign_queue", {}, readonly=True).startswith(
        "Refused:"
    )
    assert not dispatch_campaign_tool(
        "campaign_run_status", {}, readonly=True
    ).startswith("Refused:")
    assert dispatch_campaign_tool(
        "campaign_report",
        {"kind": "campaign_note", "summary": "tick observation"},
        readonly=True,
    ).startswith("Wrote review report")


def test_readonly_campaign_names_are_the_observe_report_subset():
    assert set(ct.CAMPAIGN_READONLY_TOOL_NAMES) == {
        "campaign_queue",
        "campaign_run_status",
        "campaign_report",
        "campaign_training_status",
    }
    schema_names = {s["function"]["name"] for s in ct.CAMPAIGN_READONLY_TOOL_SCHEMAS}
    assert schema_names == set(ct.CAMPAIGN_READONLY_TOOL_NAMES)
    for tool in _MUTATING_CAMPAIGN_TOOLS:
        assert tool not in ct.CAMPAIGN_READONLY_TOOL_NAMES


# --------------------------------------------------------------------------- #
# validate_corpus_path
# --------------------------------------------------------------------------- #


def test_validate_corpus_path_accepts_under_inputs(env):
    book = env["inputs"] / "book.pdf"
    book.write_text("x")
    resolved = ct.validate_corpus_path(str(book))
    assert resolved == book.resolve()


def test_validate_corpus_path_rejects_outside(env, tmp_path):
    outside = tmp_path / "outside.pdf"
    outside.write_text("x")
    with pytest.raises(ValueError):
        ct.validate_corpus_path(str(outside))


def test_validate_corpus_path_rejects_symlink_escape(env, tmp_path):
    target = tmp_path / "secret.pdf"
    target.write_text("x")
    link = env["inputs"] / "link.pdf"
    os.symlink(target, link)
    # realpath resolves the symlink OUT of inputs/ -> refused.
    with pytest.raises(ValueError):
        ct.validate_corpus_path(str(link))


def test_validate_corpus_path_rejects_missing(env):
    with pytest.raises(ValueError):
        ct.validate_corpus_path(str(env["inputs"] / "ghost.pdf"))


# --------------------------------------------------------------------------- #
# campaign_prepare_run
# --------------------------------------------------------------------------- #


def test_prepare_writes_exact_s5_shape(env):
    corpus = env["inputs"] / "sample-book-a"
    corpus.mkdir()
    result = dispatch_campaign_tool(
        "campaign_prepare_run",
        {"corpus": str(corpus), "env_overlay": {"COURSEFORGE_TWO_PASS": "true"}, "note": "pilot book"},
    )
    assert result.startswith("Prepared run overlay")
    overlay = env["pending"] / "sample-book-a.json"
    assert overlay.is_file()
    doc = json.loads(overlay.read_text())
    assert doc["version"] == 1
    assert doc["created"].endswith("Z") and "T" in doc["created"]
    assert doc["corpus"] == str(corpus.resolve())
    assert doc["env"] == {"COURSEFORGE_TWO_PASS": "true"}
    assert doc["note"] == "pilot book"
    assert doc["prepared_by"] == "assistant"


def test_prepare_defaults_empty_env_and_null_note(env):
    corpus = env["inputs"] / "manual.pdf"
    corpus.write_text("x")
    dispatch_campaign_tool("campaign_prepare_run", {"corpus": str(corpus)})
    doc = json.loads((env["pending"] / "manual.json").read_text())
    assert doc["env"] == {}
    assert doc["note"] is None


def test_prepare_refuses_unknown_flag_naming_it(env):
    corpus = env["inputs"] / "b.pdf"
    corpus.write_text("x")
    result = dispatch_campaign_tool(
        "campaign_prepare_run",
        {"corpus": str(corpus), "env_overlay": {"ED4ALL_HOME": "/evil"}},
    )
    assert result.startswith("Refused:")
    assert "ED4ALL_HOME" in result  # the offending key is named
    assert not list(env["pending"].glob("*.json"))  # nothing written


def test_prepare_refuses_bad_value_naming_it(env):
    corpus = env["inputs"] / "b.pdf"
    corpus.write_text("x")
    result = dispatch_campaign_tool(
        "campaign_prepare_run",
        {"corpus": str(corpus), "env_overlay": {"COURSEFORGE_TWO_PASS": "a b; rm -rf /"}},
    )
    assert result.startswith("Refused:")
    assert "COURSEFORGE_TWO_PASS" in result
    assert not list(env["pending"].glob("*.json"))


def test_prepare_refuses_corpus_outside_inputs_no_file(env, tmp_path):
    outside = tmp_path / "evil.pdf"
    outside.write_text("x")
    result = dispatch_campaign_tool("campaign_prepare_run", {"corpus": str(outside)})
    assert result.startswith("Refused:")
    assert not list(env["pending"].glob("*.json"))


# --------------------------------------------------------------------------- #
# campaign_queue
# --------------------------------------------------------------------------- #


def test_campaign_queue_summarizes(env):
    manifest = {
        "campaign": "demo",
        "books": [
            {"slug": "b-late", "status": "pending", "wave": 2, "pages": 100},
            {"slug": "b-early", "status": "pending", "wave": 1, "pages": 50},
            {"slug": "b-done", "status": "completed"},
        ],
    }
    (env["campaign"] / "manifest.json").write_text(json.dumps(manifest))
    # a prepared overlay + an open blocking report
    (env["pending"] / "prep-a.json").write_text("{}")
    ct.write_review_report("run_failure", summary="boom", book_slug="b-done")

    out = dispatch_campaign_tool("campaign_queue", {})
    assert "3 book(s)" in out
    # pending sorted by (wave, pages): b-early before b-late
    assert out.index("b-early") < out.index("b-late")
    assert "prep-a" in out
    assert "Open review reports: 1" in out
    assert "b-done" in out  # blocking slug surfaced


# --------------------------------------------------------------------------- #
# campaign_run_status
# --------------------------------------------------------------------------- #


def test_run_status_rejects_bad_run_id(env, monkeypatch):
    _no_spawn(monkeypatch)
    assert dispatch_campaign_tool(
        "campaign_run_status", {"run_id": "$(reboot)"}
    ).startswith("Refused:")


def test_run_status_all_lists_active_runs(env):
    wf_dir = env["state"] / "workflows"
    wf_dir.mkdir()
    (wf_dir / "WF-20260722-abcdef01.json").write_text(
        json.dumps(
            {"status": "RUNNING", "params": {"course_name": "SAMPLE_BOOK_A"}, "current_phase": "packaging"}
        )
    )
    (wf_dir / "WF-20260722-abcdef02.json").write_text(
        json.dumps({"status": "COMPLETED", "params": {"course_name": "OLD"}})
    )
    out = dispatch_campaign_tool("campaign_run_status", {})
    assert "WF-20260722-abcdef01" in out
    assert "packaging" in out
    assert "WF-20260722-abcdef02" not in out  # completed is not "active"


def test_run_status_one_reads_record(env):
    wf_dir = env["state"] / "workflows"
    wf_dir.mkdir()
    rid = "WF-20260722-abcdef01"
    (wf_dir / f"{rid}.json").write_text(
        json.dumps({"status": "FAILED", "params": {"course_name": "SAMPLE_BOOK_A"}, "failed_phase": "course_planning"})
    )
    out = dispatch_campaign_tool("campaign_run_status", {"run_id": rid})
    assert rid in out
    assert "course_planning" in out


# --------------------------------------------------------------------------- #
# ed4all_run_pids / preflight_launch
# --------------------------------------------------------------------------- #


def test_ed4all_run_pids_matches_adjacent_pair(env, tmp_path, monkeypatch):
    me = os.getpid()
    proc = _make_fake_proc(
        tmp_path,
        {
            321: ["ed4all", "run", "textbook-to-course"],
            400: ["python3", "foo.py"],
            500: ["ed4all", "status"],  # not adjacent run
            610: ["/usr/local/bin/ed4all", "run"],  # path-form basename
            me: ["ed4all", "run", "self"],  # must be excluded
        },
    )
    pids = ct.ed4all_run_pids(proc)
    assert set(pids) == {321, 610}
    assert me not in pids


def test_ed4all_run_pids_never_uses_pgrep(env, monkeypatch, tmp_path):
    def _explode(*a, **k):  # pragma: no cover
        raise AssertionError("ed4all_run_pids must never shell out to pgrep")

    monkeypatch.setattr(ct.subprocess, "run", _explode)
    monkeypatch.setattr(ct.subprocess, "Popen", _explode)
    proc = _make_fake_proc(tmp_path, {})
    assert ct.ed4all_run_pids(proc) == []


def test_preflight_launch_clear_then_stop_all(env, tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [])
    assert ct.preflight_launch() is None
    (env["state"] / "STOP_ALL").write_text("")
    assert "STOP_ALL" in (ct.preflight_launch() or "")


def test_preflight_launch_second_sentinel_location(env, monkeypatch):
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [])
    (env["state"] / "runs").mkdir()
    (env["state"] / "runs" / "STOP_ALL").write_text("")
    assert "STOP_ALL" in (ct.preflight_launch() or "")


def test_preflight_launch_single_owner(env, monkeypatch):
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [999])
    refusal = ct.preflight_launch()
    assert refusal is not None and "single-owner" in refusal


# --------------------------------------------------------------------------- #
# write_review_report / campaign_report / blocking_reports
# --------------------------------------------------------------------------- #


def test_campaign_report_writes_s6_and_index(env):
    result = dispatch_campaign_tool(
        "campaign_report",
        {
            "kind": "gate_anomaly",
            "summary": "kg_quality coverage regressed",
            "run_id": "WF-20260722-abcdef01",
            "phase": "concept_extraction",
            "error_class": "GateFailure",
            "log_excerpt": "some log tail",
            "book_slug": "sample-book-a",
        },
    )
    assert result.startswith("Wrote review report")
    reports = list(env["review"].glob("*-gate_anomaly.json"))
    assert len(reports) == 1
    doc = json.loads(reports[0].read_text())
    assert set(doc) == {
        "version", "ts", "kind", "book_slug", "run_id", "phase",
        "error_class", "log_excerpt", "summary", "verdict", "status",
    }
    assert doc["kind"] == "gate_anomaly"
    assert doc["verdict"] is None  # tools always write verdict=null
    assert doc["status"] == "open"
    assert doc["book_slug"] == "sample-book-a"
    index = (env["review"] / "INDEX.md").read_text()
    assert "# Campaign review queue" in index
    assert "[gate_anomaly] sample-book-a" in index
    assert reports[0].name in index


def test_campaign_report_rejects_bad_kind_and_run_id(env):
    assert dispatch_campaign_tool(
        "campaign_report", {"kind": "nonsense", "summary": "x y z pad"}
    ).startswith("Refused:")
    assert dispatch_campaign_tool(
        "campaign_report",
        {"kind": "campaign_note", "summary": "x y z pad", "run_id": "not-a-wf"},
    ).startswith("Refused:")


def test_write_review_report_collision_suffix(env):
    p1 = ct.write_review_report("campaign_note", summary="a")
    p2 = ct.write_review_report("campaign_note", summary="b")
    # Same-second stamps → the second gets a -2 suffix (never overwrites).
    assert p1 != p2
    assert p2.name.endswith("-2.json") or p1.stem != p2.stem


def test_blocking_reports_classification(env):
    slug = "sample-book-a"
    ct.write_review_report("run_failure", summary="fail", book_slug=slug)  # blocking
    ct.write_review_report("gate_anomaly", summary="gate", book_slug=slug)  # blocking
    ct.write_review_report("campaign_note", summary="note", book_slug=slug)  # not blocking
    blocking = ct.blocking_reports(slug)
    kinds = {r["kind"] for r in blocking}
    assert kinds == {"run_failure", "gate_anomaly"}


def test_blocking_reports_respects_clear_and_resolved(env):
    slug = "b"
    # A run_failure that Claude marked CLEAR is not blocking.
    p = ct.write_review_report("run_failure", summary="fail", book_slug=slug)
    doc = json.loads(p.read_text())
    doc["verdict"] = "CLEAR"
    p.write_text(json.dumps(doc))
    # A resolved gate_anomaly is not blocking.
    p2 = ct.write_review_report("gate_anomaly", summary="g", book_slug=slug)
    doc2 = json.loads(p2.read_text())
    doc2["status"] = "resolved"
    p2.write_text(json.dumps(doc2))
    # An explicit BLOCK verdict on a campaign_note IS blocking.
    p3 = ct.write_review_report("campaign_note", summary="n", book_slug=slug)
    doc3 = json.loads(p3.read_text())
    doc3["verdict"] = "BLOCK"
    p3.write_text(json.dumps(doc3))

    blocking = ct.blocking_reports(slug)
    assert len(blocking) == 1
    assert blocking[0]["verdict"] == "BLOCK"


def test_blocking_reports_unparseable_is_blocking(env):
    (env["review"] / "20260722T000000Z-run_failure.json").write_text("{not json")
    blocking = ct.blocking_reports("any-book")
    assert len(blocking) == 1
    assert blocking[0]["verdict"] == "BLOCK"
    assert blocking[0]["kind"] == "unparseable"


# --------------------------------------------------------------------------- #
# Self-diagnosis (spec req 3)
# --------------------------------------------------------------------------- #


def test_dispatch_self_diagnosis_writes_assistant_error(env, monkeypatch):
    def _boom():
        raise RuntimeError("kaboom")

    monkeypatch.setitem(ct.CAMPAIGN_TOOL_REGISTRY, "campaign_queue", _boom)
    result = dispatch_campaign_tool("campaign_queue", {})
    assert result.startswith("Tool campaign_queue failed:")
    assert "RuntimeError" in result
    reports = list(env["review"].glob("*-assistant_error.json"))
    assert len(reports) == 1
    doc = json.loads(reports[0].read_text())
    assert doc["kind"] == "assistant_error"
    assert doc["error_class"] == "RuntimeError"


def test_dispatch_self_diagnosis_report_write_cannot_mask_error(env, monkeypatch):
    def _boom():
        raise RuntimeError("kaboom")

    def _bad_report(*a, **k):
        raise OSError("disk full")

    monkeypatch.setitem(ct.CAMPAIGN_TOOL_REGISTRY, "campaign_queue", _boom)
    monkeypatch.setattr(ct, "write_review_report", _bad_report)
    # Even when the report write ALSO fails, the error return survives.
    result = dispatch_campaign_tool("campaign_queue", {})
    assert result.startswith("Tool campaign_queue failed:")


# --------------------------------------------------------------------------- #
# Capture-fires regression (recording stub + real DecisionCapture)
# --------------------------------------------------------------------------- #


def test_report_capture_fires_with_recording_stub(env, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(ct, "_get_capture", lambda course_code="campaign": rec)
    dispatch_campaign_tool(
        "campaign_report",
        {"kind": "campaign_note", "summary": "monitoring tick summary padding"},
    )
    assert len(rec.decisions) == 1
    decision = rec.decisions[0]
    assert decision["decision_type"] == "campaign_review_report"
    assert decision["operation"] == "campaign_tools"
    assert len(decision["rationale"]) >= 20


def test_report_capture_fires_with_real_decision_capture(env):
    from lib.decision_capture import DecisionCapture

    captured = {}
    real_get = ct._get_capture

    def _tracking(course_code="campaign"):
        cap = real_get(course_code)
        captured["cap"] = cap
        return cap

    # Use the REAL DecisionCapture (repo conftest isolates capture dirs).
    ct._get_capture = _tracking  # not monkeypatch: restored in finally
    try:
        result = dispatch_campaign_tool(
            "campaign_report",
            {"kind": "campaign_note", "summary": "real capture path exercise pad"},
        )
        assert result.startswith("Wrote review report")
        cap = captured["cap"]
        assert isinstance(cap, DecisionCapture)
        assert len(cap.decisions) == 1
        assert cap.decisions[0]["decision_type"] == "campaign_review_report"
        assert len(cap.decisions[0]["rationale"]) >= 20
    finally:
        ct._get_capture = real_get
        cap = captured.get("cap")
        if cap is not None:
            cap.close()


def test_capture_failure_never_eats_result(env, monkeypatch):
    def _bad_capture(course_code="campaign"):
        raise RuntimeError("capture backend down")

    monkeypatch.setattr(ct, "_get_capture", _bad_capture)
    result = dispatch_campaign_tool(
        "campaign_report",
        {"kind": "campaign_note", "summary": "capture-down but result survives pad"},
    )
    assert result.startswith("Wrote review report")
