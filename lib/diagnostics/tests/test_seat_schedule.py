"""Unit tests for :mod:`lib.diagnostics.seat_schedule` — the ``seat`` group.

Hermetic: no real docker, no real network, no real ollama/vLLM seat. The docker
``ps -a`` query and the ``/v1/models`` probe are monkeypatched; the registries
are driven by env. Asserts registry parse validity (malformed tokens surfaced),
loopback + duplicate-port, container existence (docker absent → INFO), per-seat
liveness (down at rest → INFO), launch-spec path checks, the assistant-seat
sub-check, and the never-raising contract.
"""

from __future__ import annotations

import os

import pytest

from lib.diagnostics import seat_schedule as seat
from lib.diagnostics.core import (
    CheckContext,
    Severity,
    clear_registry,
    registered_checks,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    clear_registry()
    for var in (
        "ED4ALL_SEAT_BASE_URLS",
        "ED4ALL_VLLM_CONTAINERS",
        "ED4ALL_SEAT_LAUNCH_SPECS",
        "ED4ALL_ASSISTANT_BASE_URL",
        "ED4ALL_ASSISTANT_SEAT",
        "ED4ALL_ASSISTANT_SEAT_PRIORITY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    clear_registry()


def _by_name(results, name):
    return next(r for r in results if r.name == name)


# --------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------- #


def test_register_seat_checks_no_import_side_effect() -> None:
    assert registered_checks() == []
    seat.register_seat_checks()
    pairs = registered_checks()
    assert [g for g, _ in pairs] == ["seat"]
    assert pairs[0][1] is seat.seat_checks


# --------------------------------------------------------------------- #
# 1. Registry parse validity
# --------------------------------------------------------------------- #


def test_registry_parse_clean_ok(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "spark-super=http://localhost:8001")
    res = _by_name(
        seat._check_registry_parse(CheckContext()), "seat_registry_ed4all_seat_base_urls"
    )
    assert res.severity is Severity.OK
    assert res.data["valid"] == 1


def test_registry_parse_malformed_token_warns(monkeypatch) -> None:
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS", "spark-super=http://localhost:8001,garbage_no_eq"
    )
    res = _by_name(
        seat._check_registry_parse(CheckContext()), "seat_registry_ed4all_seat_base_urls"
    )
    assert res.severity is Severity.WARN
    assert res.data["malformed"] == ["garbage_no_eq"]
    assert res.data["valid"] == 1


def test_registry_parse_unconfigured_info(monkeypatch) -> None:
    res = _by_name(
        seat._check_registry_parse(CheckContext()), "seat_registry_ed4all_vllm_containers"
    )
    assert res.severity is Severity.INFO


def test_audit_registry_tokens_launch_spec_grammar() -> None:
    # ';' separator, split on FIRST '=' only (spec may contain '=').
    valid, bad = seat._audit_registry_tokens(
        "a=/opt/x.sh --env FOO=bar;b=", launch_spec=True
    )
    assert valid == 1  # 'a=...' valid; 'b=' has empty spec → bad
    assert bad == ["b="]


# --------------------------------------------------------------------- #
# 2. Loopback + duplicate-port
# --------------------------------------------------------------------- #


def test_loopback_all_ok(monkeypatch) -> None:
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS",
        "a=http://localhost:8001,b=http://127.0.0.1:8002",
    )
    results = seat._check_loopback_and_ports(CheckContext())
    assert _by_name(results, "seat_loopback").severity is Severity.OK
    assert _by_name(results, "seat_port_collision").severity is Severity.OK


def test_loopback_non_loopback_warns(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "a=http://10.0.0.5:8001")
    res = _by_name(seat._check_loopback_and_ports(CheckContext()), "seat_loopback")
    assert res.severity is Severity.WARN
    assert "a" in res.data["non_loopback"]


def test_duplicate_port_warns(monkeypatch) -> None:
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS",
        "a=http://localhost:8001,b=http://localhost:8001",
    )
    res = _by_name(seat._check_loopback_and_ports(CheckContext()), "seat_port_collision")
    assert res.severity is Severity.WARN
    assert "localhost:8001" in res.data["collisions"]


def test_is_loopback_variants() -> None:
    assert seat._is_loopback("http://localhost:8001")
    assert seat._is_loopback("http://127.0.0.1:8001")
    assert seat._is_loopback("http://[::1]:8001")
    assert not seat._is_loopback("http://example.com:8001")
    assert not seat._is_loopback("not a url")


# --------------------------------------------------------------------- #
# 3. Container existence (docker mocked)
# --------------------------------------------------------------------- #


def test_container_present_ok(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_VLLM_CONTAINERS", "http://localhost:8001=vllm-super")
    monkeypatch.setattr(seat, "_docker_container_names", lambda: {"vllm-super", "other"})
    res = _by_name(seat._check_container_existence(CheckContext()), "seat_containers")
    assert res.severity is Severity.OK


def test_container_missing_warns(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_VLLM_CONTAINERS", "http://localhost:8001=vllm-super")
    monkeypatch.setattr(seat, "_docker_container_names", lambda: {"other"})
    res = _by_name(seat._check_container_existence(CheckContext()), "seat_containers")
    assert res.severity is Severity.WARN
    assert "vllm-super" in res.data["missing"].values()


def test_container_docker_absent_info(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_VLLM_CONTAINERS", "http://localhost:8001=vllm-super")
    monkeypatch.setattr(seat, "_docker_container_names", lambda: None)
    res = _by_name(seat._check_container_existence(CheckContext()), "seat_containers")
    assert res.severity is Severity.INFO
    assert res.data["docker_available"] is False


def test_docker_container_names_parses_and_never_raises(monkeypatch) -> None:
    class _P:
        returncode = 0
        stdout = "vllm-super\nvllm-nano\n"
        stderr = ""

    monkeypatch.setattr(seat.subprocess, "run", lambda *a, **k: _P())
    assert seat._docker_container_names() == {"vllm-super", "vllm-nano"}

    def boom(*a, **k):
        raise FileNotFoundError("docker not installed")

    monkeypatch.setattr(seat.subprocess, "run", boom)
    assert seat._docker_container_names() is None


# --------------------------------------------------------------------- #
# 4. Per-seat liveness
# --------------------------------------------------------------------- #


def test_seat_liveness_info_live_and_down(monkeypatch) -> None:
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS",
        "up=http://localhost:8001,down=http://localhost:8002",
    )

    def fake_probe(url, **k):
        if "8001" in url:
            return True, ["m"], None
        return False, [], "refused"

    monkeypatch.setattr("lib.diagnostics.run_env.probe_v1_models", fake_probe)
    results = seat._check_seat_liveness(CheckContext())
    assert all(r.severity is Severity.INFO for r in results)  # down at rest is not an error
    assert _by_name(results, "seat_live_up").data["live"] is True
    assert _by_name(results, "seat_live_down").data["live"] is False


# --------------------------------------------------------------------- #
# 5. Launch-spec paths
# --------------------------------------------------------------------- #


def test_launch_spec_executable_ok(monkeypatch, tmp_path) -> None:
    script = tmp_path / "launch-super.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setenv("ED4ALL_SEAT_LAUNCH_SPECS", f"spark-super={script}")
    res = _by_name(seat._check_launch_specs(CheckContext()), "seat_launch_spark-super")
    assert res.severity is Severity.OK


def test_launch_spec_missing_warns(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "ED4ALL_SEAT_LAUNCH_SPECS", f"spark-super={tmp_path / 'nope.sh'}"
    )
    res = _by_name(seat._check_launch_specs(CheckContext()), "seat_launch_spark-super")
    assert res.severity is Severity.WARN
    assert "does not exist" in res.summary


def test_launch_spec_command_is_info(monkeypatch) -> None:
    monkeypatch.setenv(
        "ED4ALL_SEAT_LAUNCH_SPECS", "spark-super=docker run -d --name x img"
    )
    res = _by_name(seat._check_launch_specs(CheckContext()), "seat_launch_spark-super")
    assert res.severity is Severity.INFO
    assert res.data["kind"] == "command"


def test_launch_spec_none_configured_info(monkeypatch) -> None:
    res = _by_name(seat._check_launch_specs(CheckContext()), "seat_launch_specs")
    assert res.severity is Severity.INFO


# --------------------------------------------------------------------- #
# 6. Assistant seat (P1-3)
# --------------------------------------------------------------------- #


def test_assistant_base_url_default_loopback_ok(monkeypatch) -> None:
    res = _by_name(
        seat._check_assistant_seat(CheckContext()), "assistant_base_url_loopback"
    )
    assert res.severity is Severity.OK  # default http://localhost:8004/v1


def test_assistant_base_url_non_loopback_warns(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_ASSISTANT_BASE_URL", "http://gpu-box:8004/v1")
    res = _by_name(
        seat._check_assistant_seat(CheckContext()), "assistant_base_url_loopback"
    )
    assert res.severity is Severity.WARN


def test_assistant_priority_none_resolve_warns(monkeypatch) -> None:
    # No seat registry → none of the default priority names resolve.
    res = _by_name(
        seat._check_assistant_seat(CheckContext()), "assistant_seat_priority"
    )
    assert res.severity is Severity.WARN


def test_assistant_priority_partial_info(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "spark-nano=http://localhost:8004")
    # default priority spark-super,spark-nano → nano resolves, super does not.
    res = _by_name(
        seat._check_assistant_seat(CheckContext()), "assistant_seat_priority"
    )
    assert res.severity is Severity.INFO
    assert "spark-super" in res.data["resolved"]


def test_assistant_seat_startable_ok(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "spark-nano=http://localhost:8004")
    monkeypatch.setenv("ED4ALL_SEAT_LAUNCH_SPECS", "spark-nano=/opt/seats/launch-nano.sh")
    res = _by_name(
        seat._check_assistant_seat(CheckContext()), "assistant_seat_startable"
    )
    assert res.severity is Severity.OK


def test_assistant_seat_not_startable_warns(monkeypatch) -> None:
    # spark-nano in the registry but NO launch spec → cannot autostart/self-heal.
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "spark-nano=http://localhost:8004")
    res = _by_name(
        seat._check_assistant_seat(CheckContext()), "assistant_seat_startable"
    )
    assert res.severity is Severity.WARN
    assert "ED4ALL_SEAT_LAUNCH_SPECS" in res.summary


# --------------------------------------------------------------------- #
# Never-raises contract
# --------------------------------------------------------------------- #


def test_seat_checks_never_raises(monkeypatch) -> None:
    # Even if every underlying probe blows up, the entry point returns results.
    def boom(*a, **k):
        raise RuntimeError("everything on fire")

    monkeypatch.setattr(seat, "_docker_container_names", boom)
    monkeypatch.setattr("lib.diagnostics.run_env.probe_v1_models", boom)
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "a=http://localhost:8001")
    results = seat.seat_checks(CheckContext())
    assert isinstance(results, list) and results
    assert all(r.group == "seat" for r in results)


def test_run_subcheck_isolates_raise() -> None:
    def boom(ctx):
        raise ValueError("kaboom")

    boom.__name__ = "_check_boom"
    results = seat._run_subcheck(boom, CheckContext())
    assert len(results) == 1
    assert results[0].severity is Severity.WARN
    assert results[0].group == "seat"
