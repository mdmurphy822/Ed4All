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
            capture["run_config"] = context.run_config
        return results

    monkeypatch.setattr(doctor_mod, "run_checks", fake_run_checks)


# ---------------------------------------------------------------------- #
# Bootstrap
# ---------------------------------------------------------------------- #


def test_bootstrap_registers_exactly_four_groups_and_is_idempotent():
    doctor_mod._bootstrap_checks()
    groups_once = {g for g, _ in registered_checks()}
    assert groups_once == {"gpu", "window", "environment", "provider"}

    # Idempotent: a second bootstrap clears-then-registers, still four.
    doctor_mod._bootstrap_checks()
    groups_twice = {g for g, _ in registered_checks()}
    assert groups_twice == {"gpu", "window", "environment", "provider"}


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
    # Bare default runs exactly gpu/window/environment — provider EXCLUDED.
    assert capture["groups"] == ("gpu", "window", "environment")
    assert "provider" not in capture["groups"]
    assert "[provider]" not in result.output
    # Bare default threads no run_config and no base-url.
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


# ---------------------------------------------------------------------- #
# Provider / seat preflight: --run, --provider, --mode, --ping, --group provider
# ---------------------------------------------------------------------- #


import contextlib  # noqa: E402 — grouped with the provider-preflight tests


@contextlib.contextmanager
def _noop_run_env(*_a, **_k):
    """Stand-in for ``applied_run_env`` so tests never apply the real fanout."""
    yield {}


def _provider_results() -> list[CheckResult]:
    return [_result("seat_content", "provider", Severity.OK)]


def test_doctor_bare_excludes_provider_group(monkeypatch):
    """No new flags → exactly gpu/window/environment; provider absent + no run_config."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _all_ok_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, [])

    assert result.exit_code == 0, result.output
    assert capture["groups"] == ("gpu", "window", "environment")
    assert "provider" not in capture["groups"]
    assert capture["run_config"] is None


def test_doctor_run_adds_provider_group_and_threads_run_config(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)
    # Avoid any real fanout / nvidia preflight work.
    monkeypatch.setattr(doctor_mod, "applied_run_env", _noop_run_env)
    monkeypatch.setattr(
        doctor_mod, "_nvidia_preflight", lambda params: {"verdict": "PASS"}
    )

    result = CliRunner().invoke(
        doctor_command, ["--run", "textbook_to_course", "--mode", "api"]
    )

    assert result.exit_code == 0, result.output
    assert "provider" in capture["groups"]
    assert capture["groups"] == ("gpu", "window", "environment", "provider")
    rc = capture["run_config"]
    assert rc is not None
    assert rc["workflow"] == "textbook_to_course"
    assert rc["provider_hint"] == ""
    assert rc["mode"] == "api"
    assert rc["ping"] is False


def test_doctor_run_unknown_workflow_exits_2_and_runs_nothing(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _all_ok_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--run", "not_a_workflow"])

    assert result.exit_code == 2, result.output
    # Short-circuited BEFORE run_checks (never called).
    assert "groups" not in capture
    # stderr names the bogus workflow.
    assert "not_a_workflow" in result.output


def test_doctor_ping_without_run_is_bare_plus_ping(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--ping"])

    assert result.exit_code == 0, result.output
    assert "provider" in capture["groups"]
    rc = capture["run_config"]
    assert rc is not None
    assert rc["workflow"] is None  # bare + ping
    assert rc["ping"] is True
    # No --run → no nvidia_preflight key computed.
    assert "nvidia_preflight" not in rc


def test_doctor_group_provider_runs_only_provider(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--group", "provider"])

    assert result.exit_code == 0, result.output
    assert capture["groups"] == ("provider",)
    # Explicit -g provider but no --run/--ping → bare (run_config None).
    assert capture["run_config"] is None
    assert "[provider]" in result.output


def test_doctor_run_nvidia_preflight_lands_in_run_config(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)
    monkeypatch.setattr(doctor_mod, "applied_run_env", _noop_run_env)
    sentinel = {"verdict": "WARN", "checks": [{"level": "warn", "name": "x"}]}
    monkeypatch.setattr(doctor_mod, "_nvidia_preflight", lambda params: sentinel)

    result = CliRunner().invoke(
        doctor_command,
        ["--run", "textbook_to_course", "--provider", "nvidia"],
    )

    assert result.exit_code == 0, result.output
    rc = capture["run_config"]
    assert rc["provider_hint"] == "nvidia"
    assert rc["nvidia_preflight"] is sentinel


def test_doctor_run_nvidia_preflight_failure_degrades_to_none(monkeypatch):
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)
    monkeypatch.setattr(doctor_mod, "applied_run_env", _noop_run_env)

    def _boom(params):
        raise RuntimeError("preflight blew up")

    monkeypatch.setattr(doctor_mod, "_nvidia_preflight", _boom)

    result = CliRunner().invoke(
        doctor_command,
        ["--run", "textbook_to_course", "--provider", "nvidia"],
    )

    # The command still succeeds; the failed preflight degrades to None.
    assert result.exit_code == 0, result.output
    rc = capture["run_config"]
    assert rc["nvidia_preflight"] is None


def test_doctor_run_non_nvidia_provider_skips_nvidia_preflight(monkeypatch):
    """A non-nvidia run threads run_config but the nvidia_preflight stays None."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)

    def _should_not_run(params):  # pragma: no cover - asserted not called
        raise AssertionError("_nvidia_preflight must not run for a non-nvidia run")

    monkeypatch.setattr(doctor_mod, "_nvidia_preflight", _should_not_run)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    result = CliRunner().invoke(
        doctor_command, ["--run", "textbook_to_course", "--provider", "local"]
    )

    assert result.exit_code == 0, result.output
    rc = capture["run_config"]
    assert rc["nvidia_preflight"] is None


# ---------------------------------------------------------------------- #
# Finding #7 — stage-subcommand --run aliases to textbook_to_course before
# the fanout (run_config / applied_run_env must see the canonical workflow).
# ---------------------------------------------------------------------- #


def test_doctor_run_courseforge_stage_aliases_to_textbook_to_course(monkeypatch):
    """``--run courseforge_outline`` must feed the ALIASED ``textbook_to_course``
    workflow to run_config (mirroring ``ed4all run``'s pre-fanout aliasing)."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)
    monkeypatch.setattr(doctor_mod, "applied_run_env", _noop_run_env)
    monkeypatch.setattr(
        doctor_mod, "_nvidia_preflight", lambda params: {"verdict": "PASS"}
    )

    result = CliRunner().invoke(doctor_command, ["--run", "courseforge_outline"])

    assert result.exit_code == 0, result.output
    rc = capture["run_config"]
    assert rc is not None
    # The fanout must model textbook_to_course, NOT courseforge_outline.
    assert rc["workflow"] == "textbook_to_course"


def test_doctor_run_all_courseforge_stage_subcommands_alias(monkeypatch):
    """Every courseforge-* stage subcommand aliases to textbook_to_course."""
    for sub in (
        "courseforge",
        "courseforge_outline",
        "courseforge_validate",
        "courseforge_rewrite",
    ):
        capture: dict = {}
        _patch_run_checks(monkeypatch, _provider_results(), capture=capture)
        monkeypatch.setattr(doctor_mod, "applied_run_env", _noop_run_env)
        monkeypatch.setattr(
            doctor_mod, "_nvidia_preflight", lambda params: {"verdict": "PASS"}
        )

        result = CliRunner().invoke(doctor_command, ["--run", sub])

        assert result.exit_code == 0, (sub, result.output)
        assert capture["run_config"]["workflow"] == "textbook_to_course", sub


def test_doctor_run_non_aliased_workflow_passes_through(monkeypatch):
    """A non-aliased workflow (textbook_to_course) is fed through unchanged."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)
    monkeypatch.setattr(doctor_mod, "applied_run_env", _noop_run_env)
    monkeypatch.setattr(
        doctor_mod, "_nvidia_preflight", lambda params: {"verdict": "PASS"}
    )

    result = CliRunner().invoke(doctor_command, ["--run", "rag_training"])

    assert result.exit_code == 0, result.output
    assert capture["run_config"]["workflow"] == "rag_training"


def test_doctor_run_aliased_workflow_is_what_nvidia_preflight_sees(monkeypatch):
    """The aliased workflow is also what _compute_nvidia_preflight models."""
    capture: dict = {}
    seen: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)

    @contextlib.contextmanager
    def _spy_run_env(workflow, provider, *_a, **_k):
        seen["workflow"] = workflow
        yield {}

    monkeypatch.setattr(doctor_mod, "applied_run_env", _spy_run_env)
    monkeypatch.setattr(
        doctor_mod, "_nvidia_preflight", lambda params: {"verdict": "PASS"}
    )

    result = CliRunner().invoke(
        doctor_command, ["--run", "courseforge_rewrite", "--provider", "nvidia"]
    )

    assert result.exit_code == 0, result.output
    # _compute_nvidia_preflight opened applied_run_env with the aliased workflow.
    assert seen["workflow"] == "textbook_to_course"


# ---------------------------------------------------------------------- #
# Finding #9 — --ping/--run must keep the provider group even when an
# explicit --group list omits it (the only network probe lives there).
# ---------------------------------------------------------------------- #


def test_doctor_ping_with_explicit_group_still_includes_provider(monkeypatch):
    """``--ping --group gpu`` must NOT filter provider out (silent no-op probe)."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--ping", "--group", "gpu"])

    assert result.exit_code == 0, result.output
    # provider was unioned into the explicit selection.
    assert "provider" in capture["groups"]
    assert "gpu" in capture["groups"]
    rc = capture["run_config"]
    assert rc is not None
    assert rc["ping"] is True


def test_doctor_run_with_explicit_group_still_includes_provider(monkeypatch):
    """``--run ... --group window`` also force-includes the provider group."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)
    monkeypatch.setattr(doctor_mod, "applied_run_env", _noop_run_env)
    monkeypatch.setattr(
        doctor_mod, "_nvidia_preflight", lambda params: {"verdict": "PASS"}
    )

    result = CliRunner().invoke(
        doctor_command, ["--run", "textbook_to_course", "--group", "window"]
    )

    assert result.exit_code == 0, result.output
    assert "provider" in capture["groups"]
    assert "window" in capture["groups"]


def test_doctor_ping_with_explicit_provider_group_not_duplicated(monkeypatch):
    """An explicit ``--group provider`` with --ping is not double-added."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--ping", "--group", "provider"])

    assert result.exit_code == 0, result.output
    assert capture["groups"].count("provider") == 1


def test_doctor_ping_alone_still_includes_provider(monkeypatch):
    """``--ping`` with no --group keeps the full run-group set incl. provider."""
    capture: dict = {}
    _patch_run_checks(monkeypatch, _provider_results(), capture=capture)

    result = CliRunner().invoke(doctor_command, ["--ping"])

    assert result.exit_code == 0, result.output
    assert "provider" in capture["groups"]


def test_doctor_explicit_group_without_ping_unchanged(monkeypatch):
    """Without --ping/--run, an explicit --group is NOT augmented with provider."""
    capture: dict = {}
    _patch_run_checks(
        monkeypatch, [_result("gpu_fit_nli", "gpu", Severity.OK)], capture=capture
    )

    result = CliRunner().invoke(doctor_command, ["--group", "gpu"])

    assert result.exit_code == 0, result.output
    assert capture["groups"] == ("gpu",)
    assert "provider" not in capture["groups"]
