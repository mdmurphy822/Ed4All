"""Unit tests for :mod:`lib.diagnostics.core`.

Hermetic: no GPU, no network, no env dependence — every check is a plain
callable constructed in-test. Covers the registry (register / list / clear
/ ordered+filtered run / raising-check isolation), the exit-code contract,
the grouped report renderer + verdict line, and JSON serialization.
"""

from __future__ import annotations

import pytest

from lib.diagnostics.core import (
    CheckContext,
    CheckResult,
    Severity,
    clear_registry,
    format_report,
    register,
    registered_checks,
    resolve_exit_code,
    resolve_verdict,
    results_to_json,
    run_checks,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def _ok(name: str, group: str = "g") -> CheckResult:
    return CheckResult(name=name, group=group, severity=Severity.OK, summary=f"{name} ok")


def _warn(name: str, group: str = "g") -> CheckResult:
    return CheckResult(
        name=name,
        group=group,
        severity=Severity.WARN,
        summary=f"{name} warn",
        remediation=f"fix {name}",
    )


def _fail(name: str, group: str = "g") -> CheckResult:
    return CheckResult(
        name=name,
        group=group,
        severity=Severity.FAIL,
        summary=f"{name} fail",
        remediation=f"abort {name}",
    )


def _info(name: str, group: str = "g") -> CheckResult:
    return CheckResult(
        name=name,
        group=group,
        severity=Severity.INFO,
        summary=f"{name} info",
    )


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #


def test_register_and_list_in_order() -> None:
    def a(_ctx: CheckContext):
        return [_ok("a")]

    def b(_ctx: CheckContext):
        return [_ok("b")]

    register("ga", a)
    register("gb", b)

    pairs = registered_checks()
    assert [g for g, _ in pairs] == ["ga", "gb"]
    assert [fn for _, fn in pairs] == [a, b]


def test_clear_registry() -> None:
    register("g", lambda _ctx: [_ok("x")])
    assert registered_checks()
    clear_registry()
    assert registered_checks() == []


def test_registered_checks_returns_copy() -> None:
    register("g", lambda _ctx: [])
    snapshot = registered_checks()
    snapshot.clear()
    assert len(registered_checks()) == 1


def test_run_checks_runs_in_registration_order() -> None:
    order: list[str] = []

    def a(_ctx: CheckContext):
        order.append("a")
        return [_ok("a")]

    def b(_ctx: CheckContext):
        order.append("b")
        return [_ok("b")]

    register("g1", a)
    register("g2", b)
    results = run_checks(CheckContext())
    assert order == ["a", "b"]
    assert [r.name for r in results] == ["a", "b"]


def test_run_checks_filters_by_group() -> None:
    register("gpu", lambda _ctx: [_ok("gpu1", "gpu")])
    register("env", lambda _ctx: [_ok("env1", "env")])

    results = run_checks(CheckContext(), groups=["gpu"])
    assert [r.name for r in results] == ["gpu1"]


def test_run_checks_multi_result_check() -> None:
    register("g", lambda _ctx: [_ok("one"), _warn("two")])
    results = run_checks(CheckContext())
    assert [r.name for r in results] == ["one", "two"]


def test_run_checks_isolates_raising_check() -> None:
    def boom(_ctx: CheckContext):
        raise RuntimeError("kaboom")

    register("g_before", lambda _ctx: [_ok("before")])
    register("g_boom", boom)
    register("g_after", lambda _ctx: [_ok("after")])

    results = run_checks(CheckContext())
    names = [r.name for r in results]
    assert "before" in names
    assert "after" in names  # run continued past the raising check
    err = next(r for r in results if r.name == "g_boom_error")
    assert err.severity is Severity.WARN
    assert "check errored" in err.summary
    assert "kaboom" in err.summary


def test_run_checks_empty_registry() -> None:
    assert run_checks(CheckContext()) == []


# --------------------------------------------------------------------- #
# resolve_exit_code
# --------------------------------------------------------------------- #


def test_exit_code_fail_is_2() -> None:
    assert resolve_exit_code([_ok("a"), _warn("b"), _fail("c")]) == 2


def test_exit_code_warn_is_1() -> None:
    assert resolve_exit_code([_ok("a"), _warn("b")]) == 1


def test_exit_code_all_ok_is_0() -> None:
    assert resolve_exit_code([_ok("a"), _ok("b")]) == 0


def test_exit_code_empty_is_0() -> None:
    assert resolve_exit_code([]) == 0


def test_exit_code_info_only_is_0() -> None:
    # INFO is below WARN — informational, never escalates the exit code.
    assert resolve_exit_code([_info("a"), _info("b")]) == 0


def test_exit_code_info_plus_ok_is_0() -> None:
    assert resolve_exit_code([_info("a"), _ok("b")]) == 0


def test_exit_code_info_plus_warn_is_1() -> None:
    assert resolve_exit_code([_info("a"), _warn("b")]) == 1


def test_exit_code_info_plus_fail_is_2() -> None:
    assert resolve_exit_code([_info("a"), _fail("b")]) == 2


# --------------------------------------------------------------------- #
# resolve_verdict
# --------------------------------------------------------------------- #


def test_verdict_danger_on_fail() -> None:
    verdict = resolve_verdict([_ok("a"), _fail("boom")])
    assert verdict.startswith("DANGER:")
    assert "boom" in verdict


def test_verdict_degraded_on_warn() -> None:
    verdict = resolve_verdict([_ok("a"), _warn("slow")])
    assert verdict.startswith("DEGRADED:")
    assert "slow" in verdict


def test_verdict_ok() -> None:
    assert resolve_verdict([_ok("a")]) == "OK"
    assert resolve_verdict([]) == "OK"


def test_verdict_info_only_is_ok() -> None:
    # INFO is below WARN — an INFO-only set does NOT trigger DEGRADED.
    assert resolve_verdict([_info("a"), _info("b")]) == "OK"
    assert resolve_verdict([_info("a"), _ok("b")]) == "OK"


def test_verdict_info_plus_warn_degraded_lists_only_warn() -> None:
    verdict = resolve_verdict([_info("note"), _warn("slow")])
    assert verdict.startswith("DEGRADED:")
    assert "slow" in verdict
    # INFO names are never listed in the DEGRADED summary.
    assert "note" not in verdict


def test_verdict_info_plus_fail_danger_lists_only_fail() -> None:
    verdict = resolve_verdict([_info("note"), _warn("slow"), _fail("boom")])
    assert verdict.startswith("DANGER:")
    assert "boom" in verdict
    # Only FAIL names are listed under DANGER — not INFO (nor WARN).
    assert "note" not in verdict


# --------------------------------------------------------------------- #
# format_report
# --------------------------------------------------------------------- #


def test_format_report_groups_and_markers() -> None:
    results = [
        _ok("gpu1", "gpu"),
        _fail("gpu2", "gpu"),
        _warn("env1", "env"),
    ]
    report = format_report(results)
    assert "[gpu]" in report
    assert "[env]" in report
    # group sections appear in first-seen order
    assert report.index("[gpu]") < report.index("[env]")
    assert "✓" in report
    assert "✗" in report
    assert "⚠" in report


def test_format_report_shows_remediation_for_warn_and_fail() -> None:
    report = format_report([_warn("slow"), _fail("boom")])
    assert "fix slow" in report
    assert "abort boom" in report


def test_format_report_renders_info_marker_under_group() -> None:
    results = [_ok("gpu1", "gpu"), _info("note", "gpu"), _warn("env1", "env")]
    report = format_report(results)
    assert "[gpu]" in report
    # INFO renders with its distinct 'ℹ' marker, in its group.
    assert "ℹ" in report
    assert "note info" in report
    gpu_section = report[report.index("[gpu]"):report.index("[env]")]
    assert "ℹ" in gpu_section
    # An INFO-only-vs-OK set still verdicts OK (INFO never degrades).
    assert report.rstrip().endswith(resolve_verdict(results))


def test_format_report_hides_remediation_for_ok() -> None:
    ok = CheckResult(
        name="fine",
        group="g",
        severity=Severity.OK,
        summary="all good",
        remediation="should-not-appear",
    )
    report = format_report([ok])
    assert "should-not-appear" not in report


def test_format_report_ends_with_verdict() -> None:
    assert format_report([_fail("boom")]).rstrip().endswith(
        resolve_verdict([_fail("boom")])
    )
    assert format_report([_ok("a")]).rstrip().endswith("OK")


def test_format_report_empty() -> None:
    report = format_report([])
    assert "no checks registered" in report
    assert report.rstrip().endswith("OK")


# --------------------------------------------------------------------- #
# results_to_json
# --------------------------------------------------------------------- #


def test_results_to_json_round_trips_and_flattens_severity() -> None:
    result = CheckResult(
        name="gpu_fit_nli",
        group="gpu",
        severity=Severity.FAIL,
        summary="would oom",
        detail="long detail",
        remediation="free vram",
        data={"need_mib": 900, "free_mib": 200},
    )
    payload = results_to_json([result])
    assert payload == [
        {
            "name": "gpu_fit_nli",
            "group": "gpu",
            "severity": "fail",
            "summary": "would oom",
            "detail": "long detail",
            "remediation": "free vram",
            "data": {"need_mib": 900, "free_mib": 200},
        }
    ]


def test_results_to_json_is_json_serializable() -> None:
    import json

    payload = results_to_json([_ok("a"), _warn("b"), _fail("c")])
    text = json.dumps(payload)
    assert '"severity": "ok"' in text
    assert '"severity": "warn"' in text
    assert '"severity": "fail"' in text


def test_results_to_json_data_is_copied() -> None:
    src = {"k": "v"}
    result = CheckResult(
        name="n", group="g", severity=Severity.OK, summary="s", data=src
    )
    payload = results_to_json([result])
    payload[0]["data"]["k"] = "mutated"
    assert src["k"] == "v"
