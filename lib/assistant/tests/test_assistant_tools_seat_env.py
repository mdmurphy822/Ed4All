"""Tests for the seat-swap environment setup tool family: seat_env_doctor
(env audit via the lifecycle lib's own parsers), list_gpu_containers
(mocked subprocess, bounded), and generate_seat_env (validation rejections
+ golden-ish snippet output; never writes a file)."""

from __future__ import annotations

import pytest

from lib.assistant import tools as assistant_tools
from lib.assistant.tools import dispatch_tool


def _clear_seat_env(monkeypatch):
    for var in (
        "ED4ALL_SEAT_SCHEDULE",
        "ED4ALL_SEAT_BASE_URLS",
        "ED4ALL_VLLM_CONTAINERS",
        "ED4ALL_SEAT_LAUNCH_SPECS",
        "ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS",
        "ED4ALL_SEAT_COHERENCE_ATTEMPTS",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# seat_env_doctor
# --------------------------------------------------------------------------- #


def test_seat_env_doctor_healthy_stack(monkeypatch, tmp_path):
    _clear_seat_env(monkeypatch)
    script = tmp_path / "launch-a.sh"
    script.write_text("#!/bin/sh\n")
    monkeypatch.setenv("ED4ALL_SEAT_SCHEDULE", "1")
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "seat-a=http://localhost:8001")
    monkeypatch.setenv("ED4ALL_VLLM_CONTAINERS", "http://localhost:8001=vllm-a")
    monkeypatch.setenv("ED4ALL_SEAT_LAUNCH_SPECS", f"seat-a={script}")
    result = dispatch_tool("seat_env_doctor", {})
    assert "[ok] ED4ALL_SEAT_SCHEDULE: ON" in result
    assert "seat seat-a: http://localhost:8001 -> container vllm-a" in result
    assert f"launch script {script} exists" in result
    assert "mismatch=" not in result.splitlines()[0]  # no mismatch findings
    assert "1200s (default)" in result
    assert "3 (default)" in result


def test_seat_env_doctor_missing_container_mapping(monkeypatch):
    _clear_seat_env(monkeypatch)
    monkeypatch.setenv("ED4ALL_SEAT_SCHEDULE", "1")
    monkeypatch.setenv("ED4ALL_SEAT_BASE_URLS", "seat-a=http://localhost:8001")
    result = dispatch_tool("seat_env_doctor", {})
    assert "[missing] ED4ALL_VLLM_CONTAINERS" in result
    assert "NO entry in ED4ALL_VLLM_CONTAINERS" in result


def test_seat_env_doctor_dead_launch_spec_and_non_loopback(monkeypatch, tmp_path):
    _clear_seat_env(monkeypatch)
    monkeypatch.setenv("ED4ALL_SEAT_SCHEDULE", "1")
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS",
        "seat-a=http://localhost:8001,seat-b=http://spark:8002",
    )
    monkeypatch.setenv(
        "ED4ALL_VLLM_CONTAINERS",
        "http://localhost:8001=vllm-a,http://spark:8002=vllm-b",
    )
    monkeypatch.setenv(
        "ED4ALL_SEAT_LAUNCH_SPECS",
        f"seat-a={tmp_path / 'gone.sh'};seat-b={tmp_path / 'gone.sh'}",
    )
    result = dispatch_tool("seat_env_doctor", {})
    assert "does not exist" in result  # dead launch-spec path
    assert "NOT loopback" in result  # seat-b's URL


def test_seat_env_doctor_duplicate_port_and_overrides(monkeypatch):
    _clear_seat_env(monkeypatch)
    monkeypatch.setenv("ED4ALL_SEAT_SCHEDULE", "1")
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS",
        "seat-a=http://localhost:8001,seat-b=http://127.0.0.1:8001",
    )
    monkeypatch.setenv("ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("ED4ALL_SEAT_COHERENCE_ATTEMPTS", "5")
    result = dispatch_tool("seat_env_doctor", {})
    assert "duplicates seat" in result
    assert "600s (override)" in result
    assert "5 (override)" in result


def test_seat_env_doctor_all_unset(monkeypatch):
    _clear_seat_env(monkeypatch)
    result = dispatch_tool("seat_env_doctor", {})
    assert "off/unset" in result
    assert "[missing] ED4ALL_SEAT_BASE_URLS" in result


# --------------------------------------------------------------------------- #
# list_gpu_containers
# --------------------------------------------------------------------------- #


def test_list_gpu_containers_bounded_rows(monkeypatch):
    calls = []

    class _Proc:
        returncode = 0
        stdout = "\n".join(
            f"vllm-{i}\tUp 2 hours\tnvcr.io/img:{i}" for i in range(40)
        )
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        assert kwargs.get("shell") is not True
        return _Proc()

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("list_gpu_containers", {})
    assert calls[0][:3] == ["docker", "ps", "-a"]
    assert "40 total, showing 30" in result
    assert "vllm-0 / Up 2 hours" in result
    assert "vllm-39" not in result  # bounded to 30


def test_list_gpu_containers_sg_fallback(monkeypatch):
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)

        class _Proc:
            stderr = ""

        proc = _Proc()
        if argv[0] == "docker":
            proc.returncode = 1
            proc.stdout = ""
        else:
            proc.returncode = 0
            proc.stdout = "vllm-x\tExited\timg"
        return proc

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("list_gpu_containers", {})
    assert calls[1][:3] == ["sg", "docker", "-c"]
    assert "vllm-x / Exited / img" in result


def test_list_gpu_containers_docker_absent(monkeypatch):
    def _fake_run(argv, **kwargs):
        raise OSError("no docker")

    monkeypatch.setattr(assistant_tools.subprocess, "run", _fake_run)
    result = dispatch_tool("list_gpu_containers", {})
    assert "docker unavailable" in result


# --------------------------------------------------------------------------- #
# generate_seat_env
# --------------------------------------------------------------------------- #


def _seat(name="seat-a", url="http://localhost:8001", container="vllm-a", **kw):
    entry = {"seat_name": name, "base_url": url, "container": container}
    entry.update(kw)
    return entry


@pytest.mark.parametrize(
    "bad_seats,fragment",
    [
        ([], "non-empty list"),
        ("not-a-list", "non-empty list"),
        ([_seat(name="Bad_Name")], "seat_name"),
        ([_seat(name="x" * 40)], "seat_name"),
        ([_seat(url="http://spark:8001")], "loopback"),
        ([_seat(url="ftp://localhost:8001")], "loopback"),
        ([_seat(container="bad name!")], "container"),
        ([_seat(), _seat(url="http://localhost:8002")], "duplicate seat_name"),
        ([_seat(), _seat(name="seat-b")], "duplicate base_url"),
        ([_seat(launch_script="relative/path.sh")], "absolute"),
        ([_seat(launch_script="/does/not/exist.sh")], "absolute"),
        (
            [_seat(name=f"seat-{i}", url=f"http://localhost:{8000 + i}",
                   container=f"c{i}") for i in range(9)],
            "at most 8 seats",
        ),
    ],
)
def test_generate_seat_env_rejections(bad_seats, fragment):
    result = dispatch_tool("generate_seat_env", {"seats": bad_seats})
    assert result.startswith("Refused:"), result
    assert fragment in result


def test_generate_seat_env_golden_snippet(tmp_path):
    script = tmp_path / "launch-a.sh"
    script.write_text("#!/bin/sh\n")
    seats = [
        _seat(launch_script=str(script)),
        _seat(name="seat-b", url="http://127.0.0.1:8002", container="vllm-b"),
    ]
    result = dispatch_tool(
        "generate_seat_env",
        {"seats": seats, "load_timeout": 900, "coherence_attempts": 4},
    )
    assert 'export ED4ALL_SEAT_SCHEDULE=1' in result
    assert (
        'export ED4ALL_SEAT_BASE_URLS="seat-a=http://localhost:8001,'
        'seat-b=http://127.0.0.1:8002"' in result
    )
    assert (
        'export ED4ALL_VLLM_CONTAINERS="http://localhost:8001=vllm-a,'
        'http://127.0.0.1:8002=vllm-b"' in result
    )
    assert f'export ED4ALL_SEAT_LAUNCH_SPECS="seat-a={script}"' in result
    assert "export ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS=900" in result
    assert "export ED4ALL_SEAT_COHERENCE_ATTEMPTS=4" in result
    assert "seat-schedule.env.example" in result  # points at the template
    # It generated TEXT only — nothing was written anywhere under tmp_path.
    assert list(tmp_path.iterdir()) == [script]


def test_generate_seat_env_defaults_and_no_specs():
    result = dispatch_tool("generate_seat_env", {"seats": [_seat()]})
    assert "export ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS=1200" in result
    assert "export ED4ALL_SEAT_COHERENCE_ATTEMPTS=3" in result
    assert 'export ED4ALL_SEAT_LAUNCH_SPECS="' not in result
    assert "cannot self-heal" in result  # explains the omission


def test_seat_setup_help_topic():
    result = dispatch_tool("get_help", {"topic": "seat-setup"})
    assert "list_gpu_containers" in result
    assert "generate_seat_env" in result
    assert "seat_env_doctor" in result
    assert "seat-schedule.env.example" in result
