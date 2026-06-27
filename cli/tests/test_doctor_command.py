"""Tests for ``ed4all doctor`` (multi-check preflight diagnostics).

GPU-free + deterministic: the command iterates the
:mod:`lib.diagnostics` registry, so the tests patch at the registry
seam (``doctor_mod.run_checks``) to return constructed
:class:`~lib.diagnostics.CheckResult` lists. They never touch a real
GPU, ollama, or torch.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from cli.commands import doctor as doctor_mod
from cli.commands.doctor import doctor_command
from lib.diagnostics import CheckResult, Severity, registered_checks


# ---------------------------------------------------------------------- #
# Builders
# ---------------------------------------------------------------------- #


def _result(name: str, group: str, severity: Severity, **overrides) -> CheckResult:
    base = dict(
        name=name,
        group=group,
        severity=severity,
        summary=f"{name} summary",
        detail=f"{name} detail",
        remediation="do the thing" if severity is not Severity.OK else "",
        data={"k": name},
    )
    base.update(overrides)
    return CheckResult(**base)


def _all_ok_results() -> list[CheckResult]:
    return [
        _result("gpu_fit_nli", "gpu", Severity.OK),
        _result("window_budget", "window", Severity.OK),
        _result("environment_provider", "environment", Severity.OK),
    ]


def _patch_run_checks(monkeypatch, results, *, capture=None):
    """Patch the registry seam to return ``results``.

    If ``capture`` is a dict, the ``groups=`` kwarg run_checks received is
    recorded under ``capture['groups']``.
    """

    def fake_run_checks(context, groups=None):
        if capture is not None:
            capture["groups"] = groups
            capture["base_url"] = context.base_url
        return results

    monkeypatch.setattr(doctor_mod, "run_checks", fake_run_checks)


# ---------------------------------------------------------------------- #
# Bootstrap
# ---------------------------------------------------------------------- #


def test_bootstrap_registers_exactly_three_groups_and_is_idempotent():
    doctor_mod._bootstrap_checks()
    groups_once = {g for g, _ in registered_checks()}
    assert groups_once == {"gpu", "window", "environment"}

    # Idempotent: a second bootstrap clears-then-registers, still three.
    doctor_mod._bootstrap_checks()
    groups_twice = {g for g, _ in registered_checks()}
    assert groups_twice == {"gpu", "window", "environment"}


# ---------------------------------------------------------------------- #
# Report rendering
# ---------------------------------------------------------------------- #


def test_doctor_prints_all_groups_and_ok_verdict(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _all_ok_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 0, result.output
    # All three groups' output present.
    assert "[gpu]" in result.output
    assert "[window]" in result.output
    assert "[environment]" in result.output
    # Each group's result summary appears.
    assert "gpu_fit_nli summary" in result.output
    assert "window_budget summary" in result.output
    assert "environment_provider summary" in result.output
    # Overall verdict.
    assert "OK" in result.output
    # Default run threads no group filter and no base-url.
    assert capture["groups"] is None
    assert capture["base_url"] is None


def test_doctor_threads_base_url(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _all_ok_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--base-url", "http://x:1234"])

    assert result.exit_code == 0, result.output
    assert capture["base_url"] == "http://x:1234"


# ---------------------------------------------------------------------- #
# Exit-code gating
# ---------------------------------------------------------------------- #


def test_doctor_exit_2_and_danger_on_fail(monkeypatch):
    results = [
        _result("gpu_fit_nli", "gpu", Severity.FAIL),
        _result("window_budget", "window", Severity.OK),
    ]
    _patch_run_checks(monkeypatch, results)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 2, result.output
    assert "DANGER" in result.output


def test_doctor_exit_1_and_degraded_on_warn(monkeypatch):
    results = [
        _result("gpu_fit_nli", "gpu", Severity.WARN),
        _result("window_budget", "window", Severity.OK),
    ]
    _patch_run_checks(monkeypatch, results)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 1, result.output
    assert "DEGRADED" in result.output


def test_doctor_exit_0_on_all_ok(monkeypatch):
    _patch_run_checks(monkeypatch, _all_ok_results())

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_doctor_never_raises_on_degraded_environment(monkeypatch):
    """A WARN-heavy (degraded/unavailable) result set must exit cleanly."""
    results = [
        _result("gpu_error", "gpu", Severity.WARN),
        _result("window_budget", "window", Severity.WARN),
        _result("environment_provider", "environment", Severity.WARN),
    ]
    _patch_run_checks(monkeypatch, results)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    assert result.exit_code == 1, result.output


# ---------------------------------------------------------------------- #
# JSON output
# ---------------------------------------------------------------------- #


def test_doctor_json_emits_results_exit_code_and_summary(monkeypatch):
    results = [
        _result("gpu_fit_nli", "gpu", Severity.OK),
        _result("window_budget", "window", Severity.OK),
    ]
    _patch_run_checks(monkeypatch, results)

    result = CliRunner().invoke(doctor_command, ["--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 2
    assert payload["exit_code"] == 0
    assert payload["summary"] == "OK"
    # Severities serialize as strings.
    for entry in payload["results"]:
        assert isinstance(entry["severity"], str)
    assert {e["severity"] for e in payload["results"]} == {"ok"}


def test_doctor_json_reflects_danger_exit_code(monkeypatch):
    results = [_result("gpu_fit_nli", "gpu", Severity.FAIL)]
    _patch_run_checks(monkeypatch, results)

    result = CliRunner().invoke(doctor_command, ["--json"])

    assert result.exit_code == 2, result.output
    payload = json.loads(result.output)
    assert payload["exit_code"] == 2
    assert "DANGER" in payload["summary"]
    assert payload["results"][0]["severity"] == "fail"


# ---------------------------------------------------------------------- #
# Group filter
# ---------------------------------------------------------------------- #


def test_doctor_group_filter_passes_groups_to_run_checks(monkeypatch):
    capture: dict = {}
    # Only a gpu result comes back (simulating the registry filter).
    _patch_run_checks(monkeypatch, [_result("gpu_fit_nli", "gpu", Severity.OK)], capture=capture)

    result = CliRunner().invoke(doctor_command, ["--group", "gpu"])

    assert result.exit_code == 0, result.output
    assert capture["groups"] == ("gpu",)
    assert "[gpu]" in result.output
    assert "[window]" not in result.output
    assert "[environment]" not in result.output


def test_doctor_multiple_group_filters(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _all_ok_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--group", "gpu", "--group", "window"])

    assert result.exit_code == 0, result.output
    assert capture["groups"] == ("gpu", "window")


# ---------------------------------------------------------------------- #
# Group validation (Finding #2 — unknown group must not exit 0/OK)
# ---------------------------------------------------------------------- #


def test_doctor_unknown_group_exits_2_and_runs_nothing(monkeypatch):
    capture: dict = {}
    # If the command (wrongly) reached run_checks, this would record "groups".
    _patch_run_checks(monkeypatch, _all_ok_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--group", "gpuu"])

    # A typo'd group fails LOUD (exit 2), never silently exits 0/OK.
    assert result.exit_code == 2, result.output
    # Short-circuited BEFORE run_checks (it was never called).
    assert "groups" not in capture
    # stderr names the bad group + lists the valid groups.
    assert "gpuu" in result.output
    assert "Valid groups:" in result.output
    assert "window" in result.output
    assert "environment" in result.output


def test_doctor_unknown_group_among_valid_still_exits_2(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _all_ok_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--group", "gpu", "--group", "nope"])

    assert result.exit_code == 2, result.output
    assert "groups" not in capture
    assert "nope" in result.output


def test_doctor_valid_group_gpu_runs(monkeypatch):
    capture: dict = {}
    _patch_run_checks(
        monkeypatch, [_result("gpu_fit_nli", "gpu", Severity.OK)], capture=capture
    )

    result = CliRunner().invoke(doctor_command, ["--group", "gpu"])

    assert result.exit_code == 0, result.output
    assert capture["groups"] == ("gpu",)
    assert "[gpu]" in result.output


# ---------------------------------------------------------------------- #
# JSON back-compat (Finding #4 — snapshot + verdicts keys preserved)
# ---------------------------------------------------------------------- #


def _gpu_snapshot_result() -> CheckResult:
    return _result(
        "gpu_snapshot",
        "gpu",
        Severity.OK,
        data={
            "free_mib": 1234,
            "total_mib": 8192,
            "cuda_available": True,
            "probe_source": "nvml",
            "resident_models": [],
            "error": None,
        },
    )


def _gpu_fit_result() -> CheckResult:
    return _result(
        "gpu_fit_nli",
        "gpu",
        Severity.OK,
        detail="fits fine",
        data={
            "consumer": "nli",
            "device": "cuda",
            "need_mib": 400,
            "free_mib": 1234,
            "outcome": "fits",
        },
    )


def test_doctor_json_includes_snapshot_and_verdicts_from_gpu(monkeypatch):
    results = [_gpu_snapshot_result(), _gpu_fit_result()]
    _patch_run_checks(monkeypatch, results)

    result = CliRunner().invoke(doctor_command, ["--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # New general key still present.
    assert isinstance(payload["results"], list)
    # Back-compat snapshot reconstructed from the gpu_snapshot result.data.
    assert payload["snapshot"]["free_mib"] == 1234
    assert payload["snapshot"]["total_mib"] == 8192
    assert payload["snapshot"]["cuda_available"] is True
    assert payload["snapshot"]["probe_source"] == "nvml"
    # Back-compat verdicts reconstructed from the gpu_fit_* result(s).
    assert isinstance(payload["verdicts"], list)
    assert len(payload["verdicts"]) == 1
    verdict = payload["verdicts"][0]
    assert verdict["consumer"] == "nli"
    assert verdict["device_requested"] == "cuda"
    assert verdict["need_mib"] == 400
    assert verdict["free_mib"] == 1234
    assert verdict["outcome"] == "fits"
    assert verdict["detail"] == "fits fine"


def test_doctor_json_snapshot_verdicts_empty_without_gpu(monkeypatch):
    # window-only run: gpu group did not run → snapshot/verdicts null-or-empty.
    _patch_run_checks(monkeypatch, [_result("window_budget", "window", Severity.OK)])

    result = CliRunner().invoke(doctor_command, ["--json", "--group", "window"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["snapshot"] is None
    assert payload["verdicts"] == []
    # No crash; the general results key is still present.
    assert isinstance(payload["results"], list)
