"""Sandbox tests for the READ tool families added by the harness expansion:
workflow/run/gate reports, bounded log tails, doctor, build cost,
aggregator reports, flag lookup, library listing, objectives, ask_course.

Load-bearing contracts: every model-supplied id/slug/enum value is
validated BEFORE a path or argv is built; log tails resolve only through
the two whitelisted patterns with a 1..200 line clamp; results are capped
at MAX_TOOL_RESULT_CHARS.
"""

from __future__ import annotations

import json

import pytest

from lib.assistant import tools as assistant_tools
from lib.assistant.tools import MAX_TOOL_RESULT_CHARS, dispatch_tool


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def fake_state(monkeypatch, tmp_path):
    """Isolated runtime/state/ + campaign + LibV2 roots wired into the tools module."""
    state = tmp_path / "state"
    (state / "workflows").mkdir(parents=True)
    campaign = tmp_path / "campaign"
    (campaign / "logs").mkdir(parents=True)
    libv2_courses = tmp_path / "libv2" / "courses"
    libv2_courses.mkdir(parents=True)
    monkeypatch.setattr(assistant_tools, "STATE_PATH", state)
    monkeypatch.setattr(assistant_tools, "CAMPAIGN_DIR", campaign)
    monkeypatch.setattr(assistant_tools, "LIBV2_COURSES", libv2_courses)
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "campaign": "t",
                "books": [
                    {"slug": "book-pending", "status": "pending"},
                    {"slug": "book-failed", "status": "failed"},
                ],
            }
        )
    )
    return {"state": state, "campaign": campaign, "libv2_courses": libv2_courses}


def _write_failed_workflow(state_dir, run_id="WF-20260721-abcd1234"):
    doc = {
        "id": run_id,
        "type": "textbook_to_course",
        "status": "FAILED",
        "params": {"course_name": "book-failed"},
        "created_at": "2026-07-21T12:00:00",
        "updated_at": "2026-07-21T12:20:00",
        "failed_phase": "course_planning",
        "failure_reason": "zero successful tasks and no outputs (dispatched=1, complete=0)",
        "tasks": [{"status": "PENDING"}, {"status": "COMPLETED"}],
        "phase_outputs": {
            "semantik_conversion": {
                "_completed": True,
                "_gates_passed": True,
                "_gate_results": [
                    {
                        "gate_id": "semantik_markers",
                        "passed": True,
                        "severity": "critical",
                        "issues": [],
                    }
                ],
            },
            "course_planning": {
                "_completed": False,
                "_gates_passed": False,
                "_gate_results": [
                    {
                        "gate_id": "objective_source_refs",
                        "passed": False,
                        "severity": "critical",
                        "issues": [
                            {
                                "severity": "critical",
                                "code": "ORPHANED_CITATIONS",
                                "message": "3 citations resolve to nothing",
                            },
                            {
                                "severity": "warning",
                                "code": "OBJECTIVE_NO_GROUNDING_SOURCE",
                                "message": "CO-04 has no grounding source",
                            },
                        ],
                    }
                ],
            },
        },
    }
    (state_dir / "workflows" / f"{run_id}.json").write_text(json.dumps(doc))
    return run_id


# --------------------------------------------------------------------------- #
# run_report / gate_report — id validation + summarization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_id",
    ["../../etc/passwd", "WF-20260101-ABCDEF01", "$(reboot)", "", "TTC_ok_but_not_for_state"],
)
def test_run_report_rejects_bad_ids(fake_state, bad_id):
    result = dispatch_tool("run_report", {"run_id": bad_id})
    assert result.startswith("Refused:")


def test_run_report_summarizes_failed_run(fake_state):
    run_id = _write_failed_workflow(fake_state["state"])
    result = dispatch_tool("run_report", {"run_id": run_id})
    assert f"Run {run_id}" in result
    assert "status=FAILED" in result
    assert "failed_phase=course_planning" in result
    assert "zero successful tasks" in result
    assert "course_planning: completed=False" in result
    assert len(result) <= MAX_TOOL_RESULT_CHARS + 100


def test_run_report_missing_state_is_reported(fake_state):
    result = dispatch_tool("run_report", {"run_id": "WF-20260101-deadbeef"})
    assert "no workflow state" in result


def test_gate_report_failing_gates_first_with_codes(fake_state):
    run_id = _write_failed_workflow(fake_state["state"])
    result = dispatch_tool("gate_report", {"run_id": run_id})
    assert "1 FAILED" in result
    assert "objective_source_refs" in result
    assert "ORPHANED_CITATIONS" in result


def test_gate_report_phase_filter_and_unknown_phase(fake_state):
    run_id = _write_failed_workflow(fake_state["state"])
    only = dispatch_tool("gate_report", {"run_id": run_id, "phase": "course_planning"})
    assert "semantik_markers" not in only
    unknown = dispatch_tool("gate_report", {"run_id": run_id, "phase": "nope"})
    assert "no phase 'nope'" in unknown


def test_gate_report_requires_run_id(fake_state):
    assert dispatch_tool("gate_report", {}).startswith("Refused:")


# --------------------------------------------------------------------------- #
# tail_log — whitelisted resolution + clamping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_target",
    ["../secrets", "a/b", "book pending", "x" * 80, "", ".hidden/../../etc"],
)
def test_tail_log_rejects_path_escape_targets(fake_state, bad_target):
    result = dispatch_tool("tail_log", {"target": bad_target})
    assert result.startswith("Refused:") or "not found" in result
    # nothing outside the whitelisted dirs was ever read — shape guard only


def test_tail_log_campaign_slug_and_line_clamp(fake_state):
    log = fake_state["campaign"] / "logs" / "book-failed.log"
    log.write_text("\n".join(f"line-{i}" for i in range(500)))
    result = dispatch_tool("tail_log", {"target": "book-failed", "lines": 9999})
    assert "last 200 line(s)" in result  # clamped to MAX_LOG_TAIL_LINES
    assert "line-499" in result
    low = dispatch_tool("tail_log", {"target": "book-failed", "lines": -5})
    assert "last 1 line(s)" in low or "last 50 line(s)" in low


def test_tail_log_run_id_resolves_under_state_runs(fake_state):
    run_dir = fake_state["state"] / "runs" / "WF-20260721-abcd1234"
    run_dir.mkdir(parents=True)
    (run_dir / "phase.log").write_text("alpha\nbeta\ngamma\n")
    result = dispatch_tool(
        "tail_log", {"target": "WF-20260721-abcd1234", "lines": 2}
    )
    assert "beta" in result and "gamma" in result and "alpha" not in result


def test_tail_log_ttc_run_id_accepted(fake_state):
    run_dir = fake_state["state"] / "runs" / "TTC_book_20260721_010101"  # synthetic run id, slug-guard: allow
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("ttc-tail\n")
    result = dispatch_tool("tail_log", {"target": "TTC_book_20260721_010101"})  # synthetic run id, slug-guard: allow
    assert "ttc-tail" in result


def test_tail_log_missing_log_reported(fake_state):
    result = dispatch_tool("tail_log", {"target": "book-pending"})
    assert "no campaign log" in result


# --------------------------------------------------------------------------- #
# doctor — group enum + fixed argv + redaction
# --------------------------------------------------------------------------- #


def test_doctor_rejects_unknown_group(monkeypatch):
    def _explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("subprocess must not run for an unknown group")

    monkeypatch.setattr(assistant_tools.subprocess, "run", _explode)
    result = dispatch_tool("doctor", {"group": "rm -rf /"})
    assert result.startswith("Refused:")
    assert "environment" in result  # names the enum


def test_doctor_json_structured_summary(monkeypatch):
    """--json path: verdict + counts, every WARN/FAIL with remediation kept,
    OK/INFO counted but not detailed, secrets filtered, fixed argv."""
    calls = {}

    payload = {
        "results": [
            {"name": "gpu_ok", "group": "gpu", "severity": "ok",
             "summary": "gpu nominal", "detail": "lots of detail", "remediation": "", "data": {}},
            {"name": "env_info", "group": "environment", "severity": "info",
             "summary": "info only", "detail": "", "remediation": "", "data": {}},
            {"name": "provider_key", "group": "provider", "severity": "warn",
             "summary": "no ANTHROPIC_API_KEY=sk-ant-" + "a" * 95,
             "detail": "", "remediation": "export the key", "data": {}},
            {"name": "gpu_fit_nli", "group": "gpu", "severity": "fail",
             "summary": "NLI will not fit", "detail": "", "remediation": "free VRAM", "data": {}},
        ],
        "exit_code": 2,
        "summary": "DANGER: 1 failure",
    }

    class _Proc:
        returncode = 2
        stdout = json.dumps(payload)
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["argv"] = argv
        assert kwargs.get("shell") is not True
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("doctor", {"group": "gpu"})
    # --json requested, fixed argv, enum group forwarded
    assert calls["argv"][-3:] == ["-g", "gpu", "--json"]
    assert "cli.main" in calls["argv"]
    # verdict + severity counts
    assert "DANGER" in result
    assert "fail=1" in result and "warn=1" in result and "ok=1" in result
    # every WARN/FAIL surfaced with remediation; FAIL before WARN
    assert "gpu_fit_nli" in result and "free VRAM" in result
    assert "provider_key" in result and "export the key" in result
    assert result.index("gpu_fit_nli") < result.index("provider_key")
    # OK/INFO bodies dropped
    assert "lots of detail" not in result
    # secrets filtered
    assert "sk-ant-" + "a" * 95 not in result


def test_doctor_text_fallback_when_json_unavailable(monkeypatch):
    """Older CLI (no --json): the first call's non-JSON output triggers the
    formatted-text fallback; argv, verdict, and redaction still hold."""
    seen = []

    class _Proc:
        returncode = 1
        stdout = "check ok\nANTHROPIC_API_KEY=sk-ant-" + "a" * 95 + "\n"
        stderr = ""

    def _fake_run(argv, **kwargs):
        seen.append(list(argv))
        assert kwargs.get("shell") is not True
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("doctor", {"group": "environment"})
    # first invocation asked for --json, then fell back to the plain argv
    assert seen[0][-3:] == ["-g", "environment", "--json"]
    assert seen[-1][-2:] == ["-g", "environment"]
    assert "cli.main" in seen[-1]
    assert "DEGRADED" in result
    assert "sk-ant-" + "a" * 95 not in result  # secrets filtered


# --------------------------------------------------------------------------- #
# build_cost / aggregator_report — slug + enum validation
# --------------------------------------------------------------------------- #


def test_build_cost_rejects_unknown_slug(fake_state):
    assert dispatch_tool("build_cost", {"course_slug": "not-a-course"}).startswith(
        "Refused:"
    )
    assert dispatch_tool("build_cost", {"course_slug": "../../etc"}).startswith(
        "Refused:"
    )


def test_build_cost_happy_path(fake_state):
    course = fake_state["libv2_courses"] / "demo-course"
    course.mkdir()
    (course / "build_cost_report.json").write_text(
        json.dumps(
            {
                "run_id": "TTC_demo",
                "totals": {
                    "phase_count": 2,
                    "measured_phase_count": 2,
                    "total_wall_clock_seconds": 7300.0,
                },
                "phases": [
                    {"phase": "semantik_conversion", "wall_clock_seconds": 7000.0},
                    {"phase": "staging", "wall_clock_seconds": 300.0},
                ],
            }
        )
    )
    result = dispatch_tool("build_cost", {"course_slug": "demo-course"})
    assert "2.0h" in result
    assert "semantik_conversion" in result
    # slowest first
    assert result.index("semantik_conversion") < result.index("staging")


def test_aggregator_report_rejects_non_enum_name(fake_state):
    course = fake_state["libv2_courses"] / "demo-course"
    course.mkdir()
    result = dispatch_tool(
        "aggregator_report",
        {"course_slug": "demo-course", "name": "../../manifest"},
    )
    assert result.startswith("Refused:")
    assert "promotion_chain" in result  # names the enum


def test_aggregator_report_summarizes_not_dumps(fake_state):
    course = fake_state["libv2_courses"] / "demo-course"
    course.mkdir()
    (course / "courseforge_promotion_chain_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "course_status": "certified_accessible",
                "chain": [{"stage": f"s{i}", "blob": "x" * 500} for i in range(40)],
            }
        )
    )
    result = dispatch_tool(
        "aggregator_report",
        {"course_slug": "demo-course", "name": "promotion_chain"},
    )
    assert "course_status: certified_accessible" in result
    assert "chain: list[40]" in result
    assert "x" * 500 not in result  # summarized, never a raw dump
    assert len(result) <= MAX_TOOL_RESULT_CHARS + 100


def test_aggregator_report_missing_file(fake_state):
    course = fake_state["libv2_courses"] / "demo-course"
    course.mkdir()
    result = dispatch_tool(
        "aggregator_report", {"course_slug": "demo-course", "name": "kg_quality"}
    )
    assert "no kg_quality report" in result


# --------------------------------------------------------------------------- #
# flag_lookup — bounded doc grep
# --------------------------------------------------------------------------- #


def test_flag_lookup_bounded_to_five(monkeypatch, tmp_path):
    doc = tmp_path / "flags.md"
    rows = "\n".join(
        f"| `ED4ALL_FAKE_{i}` | unset | Purpose {i} " + "z" * 600 + " |"
        for i in range(9)
    )
    doc.write_text("# flags\n" + rows + "\n")
    monkeypatch.setattr(assistant_tools, "BEHAVIOR_FLAGS_MD", doc)
    monkeypatch.setattr(assistant_tools, "ROOT_CLAUDE_MD", tmp_path / "none.md")
    result = dispatch_tool("flag_lookup", {"name_or_prefix": "ED4ALL_FAKE"})
    assert result.count("ED4ALL_FAKE_") == 5  # capped at 5 matches
    assert len(result) <= MAX_TOOL_RESULT_CHARS + 100


def test_flag_lookup_rejects_garbage_query():
    assert dispatch_tool("flag_lookup", {"name_or_prefix": "a; rm -rf /"}).startswith(
        "Refused:"
    )
    assert dispatch_tool("flag_lookup", {"name_or_prefix": "x"}).startswith("Refused:")


def test_flag_lookup_real_docs_find_known_flag():
    result = dispatch_tool(
        "flag_lookup", {"name_or_prefix": "ED4ALL_ASSISTANT_BASE_URL"}
    )
    assert "ED4ALL_ASSISTANT_BASE_URL" in result
    assert "8004" in result


# --------------------------------------------------------------------------- #
# list_workflows / library_courses / course_objectives
# --------------------------------------------------------------------------- #


def test_list_workflows_real_config_lists_phases():
    result = dispatch_tool("list_workflows", {})
    assert "textbook_to_course" in result
    assert "semantik_conversion" in result
    assert len(result) <= MAX_TOOL_RESULT_CHARS + 100


def test_library_courses_presence_booleans(fake_state):
    course = fake_state["libv2_courses"] / "demo-course"
    (course / "semantik_chunks").mkdir(parents=True)
    (course / "semantik_chunks" / "chunks.jsonl").write_text("{}\n")
    (course / "manifest.json").write_text("{}")
    bare = fake_state["libv2_courses"] / "bare-course"
    bare.mkdir()
    result = dispatch_tool("library_courses", {})
    assert "demo-course  manifest=y chunks=y index=n" in result
    assert "bare-course  manifest=n chunks=n index=n" in result


def test_course_objectives_libv2_archive_shape(fake_state):
    course = fake_state["libv2_courses"] / "demo-course"
    course.mkdir()
    (course / "objectives.json").write_text(
        json.dumps(
            {
                "terminal_outcomes": [
                    {"id": "TO-01", "bloom_level": "apply", "statement": "Apply the thing."}
                ],
                "component_objectives": [{"id": "CO-01"}, {"id": "CO-02"}],
            }
        )
    )
    result = dispatch_tool("course_objectives", {"course_slug": "demo-course"})
    assert "1 terminal objective(s), 2 course objective(s)" in result
    assert "TO-01 [apply] Apply the thing." in result


def test_course_objectives_unknown_slug(fake_state):
    assert dispatch_tool(
        "course_objectives", {"course_slug": "ghost"}
    ).startswith("Refused:")


# --------------------------------------------------------------------------- #
# ask_course — refusal guards
# --------------------------------------------------------------------------- #


def test_ask_course_refuses_without_vector_index(fake_state, monkeypatch):
    course = fake_state["libv2_courses"] / "demo-course"
    course.mkdir()
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    result = dispatch_tool(
        "ask_course", {"course_slug": "demo-course", "question": "what is x?"}
    )
    assert result.startswith("Refused:")
    assert "no vector index" in result


def test_ask_course_refuses_while_run_active(fake_state, monkeypatch):
    course = fake_state["libv2_courses"] / "demo-course"
    (course / "vector_index").mkdir(parents=True)
    (course / "vector_index" / "manifest.json").write_text("{}")
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: True)
    result = dispatch_tool(
        "ask_course", {"course_slug": "demo-course", "question": "what is x?"}
    )
    assert result.startswith("Refused:")
    assert "GPU contention" in result


def test_ask_course_rejects_unknown_slug(fake_state):
    assert dispatch_tool(
        "ask_course", {"course_slug": "ghost", "question": "hi"}
    ).startswith("Refused:")


def test_ask_course_requires_question(fake_state, monkeypatch):
    course = fake_state["libv2_courses"] / "demo-course"
    (course / "vector_index").mkdir(parents=True)
    (course / "vector_index" / "manifest.json").write_text("{}")
    monkeypatch.setattr(assistant_tools, "_ed4all_run_active", lambda: False)
    assert dispatch_tool(
        "ask_course", {"course_slug": "demo-course", "question": "  "}
    ).startswith("Refused:")


# --------------------------------------------------------------------------- #
# Registry hygiene
# --------------------------------------------------------------------------- #


def test_registry_schemas_and_whitelists_agree():
    schema_names = {
        entry["function"]["name"] for entry in assistant_tools.TOOL_SCHEMAS
    }
    assert schema_names == set(assistant_tools.TOOL_REGISTRY)
    assert set(assistant_tools._TOOL_ARG_WHITELIST) == set(
        assistant_tools.TOOL_REGISTRY
    )
    # every required arg is also whitelisted
    for name, required in assistant_tools._TOOL_REQUIRED_ARGS.items():
        assert set(required) <= set(assistant_tools._TOOL_ARG_WHITELIST[name])
