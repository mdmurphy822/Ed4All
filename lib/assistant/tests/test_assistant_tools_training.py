"""Stage-B (LoRA training) campaign tools — prepare / launch / status.

Covers: every prepare failure mode (loud, mutation-free), launch refusals
(STOP_ALL, live build, live training, seat still serving after teardown),
teardown stopping ONLY registered containers, fixed-argv exactness (no
--force, no extra flags, slug charset validation), the ``kind: "training"``
launched-runs row + backward-compat read of kind-less rows, readonly-mode
exposure of ONLY the status tool, and the engine-prompt training rules.

Hermetic: path constants monkeypatched into ``tmp_path``; docker / seat
probes / proc scans / env checks are fakes; no subprocess ever spawns on a
refusal path.
"""

from __future__ import annotations

import json

import pytest

from lib.assistant import campaign_tools as ct
from lib.assistant.campaign_tools import dispatch_campaign_tool

SLUG = "sample-book-a"
WF_ID = "WF-20260722-abc12345"


class _Rec:
    def __init__(self):
        self.decisions = []

    def log_decision(self, **kwargs):
        self.decisions.append(kwargs)


@pytest.fixture()
def env(monkeypatch, tmp_path):
    campaign = tmp_path / "campaign"
    pending = campaign / "pending-runs"
    review = campaign / "review-queue"
    inputs = tmp_path / "inputs"
    state = tmp_path / "state"
    libv2 = tmp_path / "libv2"
    for path in (campaign, pending, review, inputs, state, libv2):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ct, "CAMPAIGN_DIR", campaign)
    monkeypatch.setattr(ct, "PENDING_RUNS_DIR", pending)
    monkeypatch.setattr(ct, "REVIEW_QUEUE_DIR", review)
    monkeypatch.setattr(ct, "LAUNCHED_RUNS_PATH", campaign / "launched-runs.jsonl")
    monkeypatch.setattr(ct, "INPUTS_ROOT", inputs)
    monkeypatch.setattr(ct, "STATE_PATH", state)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2))
    monkeypatch.delenv("ED4ALL_CAMPAIGN_BASE_MODEL", raising=False)

    # Green-path defaults; individual tests flip these to exercise refusals.
    monkeypatch.setattr(ct, "training_env_problems", lambda: [])
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [])
    monkeypatch.setattr(ct, "trainforge_train_pids", lambda proc_root="/proc": [])
    monkeypatch.setattr(ct, "_seat_registry", lambda: {})
    monkeypatch.setattr(ct, "_seat_probe", lambda url: False)
    monkeypatch.setattr(
        ct, "_poll_training_wf_id", lambda launch_time, timeout=None: WF_ID
    )

    # A trainable manifest book + its pairs + the reviewer approval marker.
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "books": [
                    {"slug": SLUG, "status": "built"},
                    {"slug": "sample-pending", "status": "pending"},
                ]
            }
        )
    )
    pairs = libv2 / "courses" / SLUG / "training_specs" / "instruction_pairs.jsonl"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_text('{"instruction": "x", "response": "y"}\n')
    approvals = review / "approvals"
    approvals.mkdir(parents=True, exist_ok=True)
    (approvals / f"{SLUG}.training-approved").write_text("approved\n")

    return {
        "campaign": campaign,
        "review": review,
        "state": state,
        "libv2": libv2,
        "pairs": pairs,
        "approvals": approvals,
    }


def _no_spawn(monkeypatch):
    def _explode(*args, **kwargs):  # pragma: no cover — must not be reached
        raise AssertionError("no subprocess may spawn on a refusal path")

    monkeypatch.setattr(ct.subprocess, "Popen", _explode)
    monkeypatch.setattr(ct.subprocess, "run", _explode)


def _failing_checks(result):
    return {name for name, c in result["checks"].items() if not c["ok"]}


# --------------------------------------------------------------------------- #
# prepare_training_run — every failure mode, loudly, mutating nothing
# --------------------------------------------------------------------------- #


def test_prepare_happy_path_all_checks_pass(env):
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is True
    assert result["error"] is None
    assert _failing_checks(result) == set()
    assert set(result["checks"]) == {
        "slug", "book_status", "pairs", "base_model", "approval",
        "env_ready", "stop_all", "no_live_build", "no_live_training",
    }
    assert str(env["pairs"]) == result["checks"]["pairs"]["detail"]


def test_prepare_mutates_nothing(env):
    before = sorted(p for p in env["campaign"].rglob("*"))
    ct.prepare_training_run(SLUG)
    assert sorted(p for p in env["campaign"].rglob("*")) == before
    assert not (env["campaign"] / "launched-runs.jsonl").exists()


@pytest.mark.parametrize(
    "bad_slug",
    ["../evil", "slug; rm -rf /", "$(reboot)", "UPPER", "", "a b", "x\ny"],
)
def test_prepare_rejects_invalid_slug_charset(env, bad_slug):
    result = ct.prepare_training_run(bad_slug)
    assert result["ok"] is False
    assert "slug" in _failing_checks(result)
    # Short-circuit: an invalid slug never reaches a path/argv-consuming check.
    assert set(result["checks"]) == {"slug"}


def test_prepare_fails_slug_not_in_manifest(env):
    result = ct.prepare_training_run("sample-ghost")
    assert result["ok"] is False
    assert {"slug", "book_status"} <= _failing_checks(result)
    assert "not in the campaign manifest" in result["error"]


def test_prepare_fails_untrainable_status(env):
    result = ct.prepare_training_run("sample-pending")
    assert result["ok"] is False
    assert "book_status" in _failing_checks(result)
    assert "pending" in result["checks"]["book_status"]["detail"]


def test_prepare_fails_missing_pairs(env):
    env["pairs"].unlink()
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is False
    assert "pairs" in _failing_checks(result)
    assert "--skip-training" in result["checks"]["pairs"]["detail"]


def test_prepare_fails_empty_pairs_file(env):
    env["pairs"].write_text("")
    result = ct.prepare_training_run(SLUG)
    assert "pairs" in _failing_checks(result)


def test_prepare_accepts_training_dir_pairs_fallback(env):
    env["pairs"].unlink()
    alt = env["libv2"] / "courses" / SLUG / "training" / "instruction_pairs.jsonl"
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_text('{"instruction": "x"}\n')
    result = ct.prepare_training_run(SLUG)
    assert "pairs" not in _failing_checks(result)
    assert result["checks"]["pairs"]["detail"] == str(alt)


def test_prepare_unknown_base_model_is_loud_never_fallback(env, monkeypatch):
    monkeypatch.setenv("ED4ALL_CAMPAIGN_BASE_MODEL", "not-a-model")
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is False
    assert "base_model" in _failing_checks(result)
    detail = result["checks"]["base_model"]["detail"]
    assert "not-a-model" in detail
    assert "never a fallback" in detail
    # The default model name must NOT be silently substituted anywhere.
    assert ct.DEFAULT_CAMPAIGN_BASE_MODEL not in detail.split("Supported")[0]


def test_prepare_default_base_model_resolves(env):
    result = ct.prepare_training_run(SLUG)
    assert result["checks"]["base_model"]["ok"] is True
    assert ct.DEFAULT_CAMPAIGN_BASE_MODEL in result["checks"]["base_model"]["detail"]


def test_prepare_fails_missing_approval_marker(env):
    (env["approvals"] / f"{SLUG}.training-approved").unlink()
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is False
    assert "approval" in _failing_checks(result)
    assert "the assistant never does" in result["checks"]["approval"]["detail"]


def test_prepare_fails_env_not_ready(env, monkeypatch):
    monkeypatch.setattr(
        ct, "training_env_problems", lambda: ["mamba_ssm missing — kernels required"]
    )
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is False
    assert "env_ready" in _failing_checks(result)
    assert "mamba_ssm" in result["error"]


@pytest.mark.parametrize("rel", ["STOP_ALL", "runs/STOP_ALL"])
def test_prepare_fails_on_stop_all_both_locations(env, rel):
    sentinel = env["state"] / rel
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("")
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is False
    assert "stop_all" in _failing_checks(result)
    assert "STOP_ALL" in result["error"]


def test_prepare_fails_on_live_build(env, monkeypatch):
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [321])
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is False
    assert "no_live_build" in _failing_checks(result)
    assert "321" in result["checks"]["no_live_build"]["detail"]


def test_prepare_fails_on_live_training_not_double_counted_as_build(env, monkeypatch):
    # A live trainforge_train shows up in BOTH proc scans; it must be reported
    # as a training conflict, not (also) as a build conflict.
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [99])
    monkeypatch.setattr(ct, "trainforge_train_pids", lambda proc_root="/proc": [99])
    result = ct.prepare_training_run(SLUG)
    assert result["ok"] is False
    assert "no_live_training" in _failing_checks(result)
    assert "no_live_build" not in _failing_checks(result)


def test_prepare_reports_all_failures_at_once(env, monkeypatch):
    env["pairs"].unlink()
    (env["approvals"] / f"{SLUG}.training-approved").unlink()
    monkeypatch.setattr(ct, "training_env_problems", lambda: ["transformers too old"])
    result = ct.prepare_training_run(SLUG)
    assert {"pairs", "approval", "env_ready"} <= _failing_checks(result)
    for token in ("pairs", "approval", "transformers too old"):
        assert token in result["error"]


# --------------------------------------------------------------------------- #
# trainforge_train_pids — adjacent-token /proc scan (never pgrep -f)
# --------------------------------------------------------------------------- #


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


def test_trainforge_train_pids_matches_adjacent_triple_only(tmp_path):
    proc = _make_fake_proc(
        tmp_path,
        {
            101: ["/usr/bin/ed4all", "run", "trainforge_train", "--course-code", "x"],
            102: ["ed4all", "run", "textbook-to-course", "--corpus", "y"],
            103: ["vim", "ed4all", "run", "notes-trainforge_train"],
            104: ["bash", "-c", "ed4all run trainforge_train"],  # one token, not a triple
        },
    )
    assert ct.trainforge_train_pids(proc) == [101]


# --------------------------------------------------------------------------- #
# launch_training_run — refusals (no spawn, no docker)
# --------------------------------------------------------------------------- #


def _no_docker(monkeypatch):
    def _explode(container):  # pragma: no cover — must not be reached
        raise AssertionError("docker must not run on a refusal path")

    monkeypatch.setattr(ct, "_docker_stop", _explode)


def test_launch_refuses_when_prepare_fails_stop_all(env, monkeypatch):
    _no_spawn(monkeypatch)
    _no_docker(monkeypatch)
    (env["state"] / "STOP_ALL").write_text("")
    result = ct.launch_training_run(SLUG)
    assert result["ok"] is False
    assert result["pid"] is None
    assert "STOP_ALL" in result["error"]


def test_launch_refuses_on_live_build(env, monkeypatch):
    _no_spawn(monkeypatch)
    _no_docker(monkeypatch)
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [777])
    result = ct.launch_training_run(SLUG)
    assert result["ok"] is False
    assert "build is active" in result["error"]


def test_launch_refuses_on_live_training(env, monkeypatch):
    _no_spawn(monkeypatch)
    _no_docker(monkeypatch)
    monkeypatch.setattr(ct, "ed4all_run_pids", lambda proc_root="/proc": [88])
    monkeypatch.setattr(ct, "trainforge_train_pids", lambda proc_root="/proc": [88])
    result = ct.launch_training_run(SLUG)
    assert result["ok"] is False
    assert "trainforge_train run is already active" in result["error"]


def test_launch_refuses_invalid_slug_no_spawn(env, monkeypatch):
    _no_spawn(monkeypatch)
    _no_docker(monkeypatch)
    result = ct.launch_training_run("evil;slug")
    assert result["ok"] is False
    assert "not a valid book slug" in result["error"]


def test_launch_fails_loudly_when_seat_still_serving_after_teardown(env, monkeypatch):
    _no_spawn(monkeypatch)
    stops = []
    monkeypatch.setattr(
        ct, "_seat_registry",
        lambda: {"http://localhost:8001": "vllm-super", "http://localhost:8004": "vllm-nano"},
    )
    monkeypatch.setattr(ct, "_docker_stop", lambda c: stops.append(c) or True)
    monkeypatch.setattr(
        ct, "_seat_probe", lambda url: url == "http://localhost:8001"
    )
    result = ct.launch_training_run(SLUG)
    assert result["ok"] is False
    assert "seat teardown FAILED" in result["error"]
    assert "http://localhost:8001" in result["error"]
    # Both registered containers were stopped (stop attempted before verify).
    assert sorted(stops) == ["vllm-nano", "vllm-super"]


def test_teardown_stops_only_registered_containers(env, monkeypatch):
    stops = []
    monkeypatch.setattr(
        ct, "_seat_registry",
        lambda: {"http://localhost:8001": "vllm-super", "http://localhost:8004": "vllm-nano"},
    )
    monkeypatch.setattr(ct, "_docker_stop", lambda c: stops.append(c) or True)
    assert ct.teardown_vllm_seats() is None
    assert sorted(stops) == ["vllm-nano", "vllm-super"]  # nothing discovered


def test_teardown_empty_registry_is_clean_noop(env, monkeypatch):
    _no_docker(monkeypatch)
    assert ct.teardown_vllm_seats() is None


# --------------------------------------------------------------------------- #
# launch_training_run — happy path: fixed argv + provenance row + capture
# --------------------------------------------------------------------------- #


@pytest.fixture()
def happy_launch(env, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(ct, "_get_capture", lambda course_code="campaign": rec)
    stops = []
    monkeypatch.setattr(
        ct, "_seat_registry", lambda: {"http://localhost:8001": "vllm-super"}
    )
    monkeypatch.setattr(ct, "_docker_stop", lambda c: stops.append(c) or True)
    spawned = {}

    class _Proc:
        pid = 4321

    def _fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(ct.subprocess, "Popen", _fake_popen)
    return {"rec": rec, "stops": stops, "spawned": spawned}


def test_launch_happy_path_fixed_argv_exact(env, happy_launch):
    result = ct.launch_training_run(SLUG)
    assert result["ok"] is True
    assert result["pid"] == 4321
    assert result["wf_id"] == WF_ID
    spawned = happy_launch["spawned"]
    assert spawned["argv"] == [
        "ed4all", "run", "trainforge_train",
        "--course-code", SLUG, "--base-model", "nemotron3-nano-30b",
    ]
    assert "--force" not in spawned["argv"]
    assert len(spawned["argv"]) == 7  # no extra flags, ever
    assert spawned["kwargs"]["start_new_session"] is True
    assert "shell" not in spawned["kwargs"]
    assert happy_launch["stops"] == ["vllm-super"]
    # Log path shape.
    assert "/logs/train-" in result["log_path"]
    assert SLUG in result["log_path"]


def test_launch_appends_training_kind_row(env, happy_launch):
    result = ct.launch_training_run(SLUG)
    rows = ct._read_launched_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "training"
    assert row["book_slug"] == SLUG
    assert row["pid"] == 4321
    assert row["wf_id"] == WF_ID
    assert row["log_path"] == result["log_path"]
    # Training runs join the campaign run-id scope (resume/stop-able).
    assert WF_ID in ct._campaign_run_ids()


def test_launch_capture_fires_once_with_dynamic_rationale(env, happy_launch):
    ct.launch_training_run(SLUG)
    rec = happy_launch["rec"]
    assert len(rec.decisions) == 1
    decision = rec.decisions[0]
    assert decision["decision_type"] == "campaign_training_launch"
    assert len(decision["rationale"]) >= 20
    for token in (SLUG, "4321", "nemotron3-nano-30b"):
        assert token in decision["rationale"]


def test_launch_result_shape_is_stable_pilot_contract(env, happy_launch):
    result = ct.launch_training_run(SLUG)
    assert set(result) == {"ok", "pid", "log_path", "wf_id", "error"}
    prep = ct.prepare_training_run(SLUG)
    assert set(prep) == {"ok", "error", "checks"}


# --------------------------------------------------------------------------- #
# launched-runs kind field — backward compat with kind-less Stage-A rows
# --------------------------------------------------------------------------- #


def test_kindless_rows_read_as_build_and_training_rows_filter(env):
    path = env["campaign"] / "launched-runs.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({  # legacy Stage-A row: NO kind field
            "ts": "2026-07-21T00:00:00Z", "name": "book-a", "corpus": "x.pdf",
            "overlay_path": None, "env": {}, "pid": 1, "log_path": "a.log",
            "wf_id": "WF-20260721-00000001", "book_slug": "book-a",
        }) + "\n")
        handle.write(json.dumps({
            "ts": "2026-07-22T00:00:00Z", "name": f"train-{SLUG}", "corpus": None,
            "overlay_path": None, "env": {}, "pid": 2, "log_path": "t.log",
            "wf_id": WF_ID, "book_slug": SLUG, "kind": "training",
        }) + "\n")
    rows = ct._read_launched_rows()
    assert [ct._row_kind(r) for r in rows] == ["build", "training"]
    training = ct.launched_training_rows()
    assert len(training) == 1 and training[0]["book_slug"] == SLUG
    # Existing reader stays compatible: BOTH wf_ids are campaign-scoped.
    assert ct._campaign_run_ids() == {"WF-20260721-00000001", WF_ID}


# --------------------------------------------------------------------------- #
# campaign_training_status (read-only) + dispatch/readonly wiring
# --------------------------------------------------------------------------- #


def test_training_status_reports_rows_wf_status_log_tail_and_env(env):
    log = env["campaign"] / "logs" / "t.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("line-1\nline-2\nfinal-line\n")
    wf_dir = env["state"] / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{WF_ID}.json").write_text(json.dumps({"status": "RUNNING"}))
    with open(env["campaign"] / "launched-runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "ts": "2026-07-22T00:00:00Z", "name": f"train-{SLUG}", "corpus": None,
            "overlay_path": None, "env": {}, "pid": 7, "log_path": str(log),
            "wf_id": WF_ID, "book_slug": SLUG, "kind": "training",
        }) + "\n")
    out = ct.campaign_training_status()
    assert SLUG in out
    assert WF_ID in out
    assert "status=RUNNING" in out
    assert "final-line" in out
    assert "Training env: READY" in out


def test_training_status_slug_filter_and_empty(env):
    out = ct.campaign_training_status("sample-other")
    assert "none launched" in out


def test_training_status_env_problems_surface(env, monkeypatch):
    monkeypatch.setattr(ct, "training_env_problems", lambda: ["mamba_ssm missing"])
    out = ct.campaign_training_status()
    assert "NOT READY" in out
    assert "mamba_ssm missing" in out


def test_training_status_refuses_invalid_slug(env):
    out = ct.campaign_training_status("../evil")
    assert out.startswith("Refused:")


def test_dispatch_prepare_training_refusal_string(env, monkeypatch):
    _no_spawn(monkeypatch)
    (env["approvals"] / f"{SLUG}.training-approved").unlink()
    out = dispatch_campaign_tool("campaign_prepare_training", {"slug": SLUG})
    assert out.startswith("Refused:")
    assert "approval" in out


def test_dispatch_prepare_training_ok_string(env):
    out = dispatch_campaign_tool("campaign_prepare_training", {"slug": SLUG})
    assert "Training prepare OK" in out
    assert "campaign_launch_training" in out


def test_dispatch_launch_training_happy(env, happy_launch):
    out = dispatch_campaign_tool("campaign_launch_training", {"slug": SLUG})
    assert "Training launched" in out
    assert "pid 4321" in out


def test_dispatch_launch_training_rejects_metachar_slug(env, monkeypatch):
    _no_spawn(monkeypatch)
    _no_docker(monkeypatch)
    out = dispatch_campaign_tool("campaign_launch_training", {"slug": "a;b|c"})
    assert out.startswith("Refused:")
    assert "Nothing was launched" in out


def test_dispatch_requires_slug(env):
    for name in ("campaign_prepare_training", "campaign_launch_training"):
        out = dispatch_campaign_tool(name, {})
        assert out.startswith("Refused:")
        assert "slug" in out


def test_readonly_mode_exposes_only_training_status(env, monkeypatch):
    _no_spawn(monkeypatch)
    _no_docker(monkeypatch)
    for name in ("campaign_prepare_training", "campaign_launch_training"):
        out = dispatch_campaign_tool(name, {"slug": SLUG}, readonly=True)
        assert out.startswith("Refused:")
        assert "mutating" in out
    out = dispatch_campaign_tool("campaign_training_status", {}, readonly=True)
    assert not out.startswith("Refused:")
    assert "Training" in out


def test_readonly_schema_slice_includes_training_status_only(env):
    names = {s["function"]["name"] for s in ct.CAMPAIGN_READONLY_TOOL_SCHEMAS}
    assert "campaign_training_status" in names
    assert "campaign_launch_training" not in names
    assert "campaign_prepare_training" not in names


def test_full_schema_set_includes_all_three_training_tools(env):
    names = {s["function"]["name"] for s in ct.CAMPAIGN_TOOL_SCHEMAS}
    assert {
        "campaign_prepare_training",
        "campaign_launch_training",
        "campaign_training_status",
    } <= names


def test_arg_whitelist_drops_extra_flags(env, happy_launch):
    # A smuggled base_model / force / extra_flag never reaches the tool.
    out = dispatch_campaign_tool(
        "campaign_launch_training",
        {"slug": SLUG, "base_model": "evil-model", "force": True, "flags": "--force"},
    )
    assert "Training launched" in out
    assert happy_launch["spawned"]["argv"][-1] == "nemotron3-nano-30b"


# --------------------------------------------------------------------------- #
# resolve_campaign_base_model + env-ready helper
# --------------------------------------------------------------------------- #


def test_resolve_campaign_base_model_default_and_override(monkeypatch):
    monkeypatch.delenv("ED4ALL_CAMPAIGN_BASE_MODEL", raising=False)
    assert ct.resolve_campaign_base_model() == "nemotron3-nano-30b"
    monkeypatch.setenv("ED4ALL_CAMPAIGN_BASE_MODEL", "qwen2.5-1.5b")
    assert ct.resolve_campaign_base_model() == "qwen2.5-1.5b"
    monkeypatch.setenv("ED4ALL_CAMPAIGN_BASE_MODEL", "   ")
    assert ct.resolve_campaign_base_model() == "nemotron3-nano-30b"


def test_training_env_problems_reports_missing_kernels(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name in ("mamba_ssm", "causal_conv1d"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    joined = "; ".join(ct.training_env_problems())
    assert "mamba_ssm" in joined
    assert "causal_conv1d" in joined


# --------------------------------------------------------------------------- #
# Engine prompts + tool exposure (training rules are stated, tick is read-only)
# --------------------------------------------------------------------------- #


def test_campaign_prompt_states_training_lifecycle_and_approval_rule():
    from lib.assistant.engine import CAMPAIGN_SYSTEM_PROMPT

    assert "campaign_launch_training" in CAMPAIGN_SYSTEM_PROMPT
    assert "campaign_prepare_training" in CAMPAIGN_SYSTEM_PROMPT
    assert "training-approved" in CAMPAIGN_SYSTEM_PROMPT
    assert "you NEVER write it" in CAMPAIGN_SYSTEM_PROMPT
    assert "NEVER launch training while a build runs" in CAMPAIGN_SYSTEM_PROMPT
    assert "NEVER launch a build while training runs" in CAMPAIGN_SYSTEM_PROMPT


def test_campaign_tick_prompt_training_is_observe_only():
    from lib.assistant.engine import CAMPAIGN_TICK_SYSTEM_PROMPT

    assert "campaign_training_status" in CAMPAIGN_TICK_SYSTEM_PROMPT
    assert "launch training" in CAMPAIGN_TICK_SYSTEM_PROMPT
    assert "campaign_launch_training" not in CAMPAIGN_TICK_SYSTEM_PROMPT


def test_engine_campaign_mode_serves_training_schemas():
    from lib.assistant.engine import AssistantEngine

    class _Client:
        model = "m"
        base_url = "http://localhost:1/v1"
        max_tokens = 10
        last_seat = None

    engine = AssistantEngine(client=_Client(), mode="campaign")
    names = {s["function"]["name"] for s in engine.tool_schemas}
    assert {
        "campaign_prepare_training",
        "campaign_launch_training",
        "campaign_training_status",
    } <= names

    tick = AssistantEngine(client=_Client(), mode="campaign-tick")
    tick_names = {s["function"]["name"] for s in tick.tool_schemas}
    assert "campaign_training_status" in tick_names
    assert "campaign_launch_training" not in tick_names
    assert "campaign_prepare_training" not in tick_names
