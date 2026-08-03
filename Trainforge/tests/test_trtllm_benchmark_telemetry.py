from __future__ import annotations

import io
import json

from Trainforge.generators.postprocessing.trtllm_benchmark_telemetry import (
    TrtllmTelemetrySampler,
    parse_trtllm_log,
    telemetry_preflight,
)

ITERATION = (
    "2026-07-27T14:44:57Z iter = 9, num_scheduled_requests: 3, "
    "states = {'num_ctx_tokens': 7000, 'num_generation_tokens': 800}"
)


def test_parser_reports_batch_budget_not_dynamic_kv():
    result = parse_trtllm_log([
        ITERATION,
        "2026-07-27T14:44:58Z HTTP/1.1 503",
        "2026-07-27T14:44:59Z request aborted after disconnect",
    ])
    assert result["peak_scheduled_sequences"] == 3
    assert result["peak_scheduled_token_usage"] == 7800
    assert result["peak_scheduled_token_headroom"] == 392
    assert result["http_status_counts"] == {"503": 1}
    assert result["abort_disconnect_count"] == 2
    assert "not KV-cache occupancy" in result["labels"]["scheduled_tokens"]
    assert str(result["dynamic_kv_free"]).startswith("unavailable")


def test_preflight_binds_container_command_and_requires_iteration():
    def runner(command, **_kwargs):
        if command[1] == "inspect":
            return json.dumps([
                "trtllm-serve", "/snapshots/served", "--max_num_tokens", "8192",
                "--max_seq_len", "262144",
            ])
        return ITERATION

    result = telemetry_preflight(
        runner=runner, expected_model="served",
        model_snapshot={"data": [{"id": "served"}]},
    )
    assert result["status"] == "accepted"
    assert result["max_num_tokens"] == 8192
    assert result["served_context_tokens"] == 262144
    assert result["expected_model"] == "served"

    try:
        telemetry_preflight(runner=lambda command, **kwargs: (
            json.dumps([
                "trtllm-serve", "--max_num_tokens", "8192",
                "--max_seq_len", "262144",
            ])
            if command[1] == "inspect" else "idle"
        ))
    except RuntimeError as exc:
        assert "no parseable scheduled-token" in str(exc)
    else:
        raise AssertionError("missing scheduled-token source must fail closed")


def test_preflight_binds_localhost_port_model_and_charges_absolute_deadline():
    clock = [100.0]
    timeouts = []

    def runner(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        clock[0] += 2.0
        if command[1] == "inspect" and "Config.Cmd" in command[-1]:
            return json.dumps([
                "trtllm-serve", "/models/snapshots/served",
                "--max_num_tokens", "8192",
                "--max_seq_len", "262144",
            ])
        if command[1] == "inspect":
            return json.dumps({"8123/tcp": [
                {"HostIp": "0.0.0.0", "HostPort": "8123"},
            ]})
        return ITERATION

    result = telemetry_preflight(
        runner=runner, expected_model="served",
        model_snapshot={"data": [{"id": "served"}]},
        base_url="http://localhost:8123/v1",
        hard_deadline=108.0, monotonic=lambda: clock[0],
    )
    assert result["published_port"] == 8123
    assert result["canonical_base_host"] == "127.0.0.1"
    assert timeouts == [8.0, 6.0, 4.0]

    try:
        telemetry_preflight(
            runner=runner, expected_model="served",
            model_snapshot={"data": [{"id": "served"}]},
            base_url="http://127.0.0.1:9999/v1",
            hard_deadline=200.0, monotonic=lambda: clock[0],
        )
    except RuntimeError as exc:
        assert "does not map" in str(exc)
    else:
        raise AssertionError("wrong published port must fail closed")


def test_sampler_lifecycle_terminates_follower_and_persists_sources(tmp_path):
    class Process:
        def __init__(self):
            self.stdout = io.StringIO(ITERATION + "\n")
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    commands = []

    def popen(command, **_kwargs):
        commands.append(command)
        return process

    sampler = TrtllmTelemetrySampler(
        tmp_path, interval=0.01, popen=popen,
        runner=lambda command, **kwargs: "2026, 1, proc, 10",
    )
    sampler.start()
    __import__("time").sleep(0.03)
    summary = sampler.stop()
    assert process.terminated
    assert commands[0][:4] == ["docker", "logs", "--follow", "--since"]
    assert "--timestamps" in commands[0]
    assert summary["peak_scheduled_sequences"] == 3
    assert summary["sampler_state"] == {
        "started": True,
        "stop_requested": True,
        "process_present": True,
        "process_poll": 0,
        "process_stopped": True,
        "terminate_requested": True,
        "kill_requested": False,
        "threads_total": 2,
        "threads_alive": 0,
        "threads_stopped": True,
        "errors": [],
    }
    assert (tmp_path / "trtllm.log").is_file()
    assert (tmp_path / "system.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_sampler_start_is_transactional_on_partial_thread_failure(tmp_path):
    class Process:
        stdout = io.StringIO("")
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    created = [0]

    def thread_factory(**kwargs):
        created[0] += 1
        if created[0] == 2:
            class Broken:
                def start(self):
                    raise RuntimeError("thread start failed")
            return Broken()
        return __import__("threading").Thread(**kwargs)

    sampler = TrtllmTelemetrySampler(
        tmp_path, popen=lambda *args, **kwargs: process,
        thread_factory=thread_factory,
    )
    try:
        sampler.start()
    except RuntimeError as exc:
        assert "thread start failed" in str(exc)
    else:
        raise AssertionError("partial sampler start must fail")
    assert process.terminated
    assert process.poll() is not None
    assert all(not thread.is_alive() for thread in sampler._threads)


def test_sampler_stop_fails_if_process_cannot_be_proven_dead(tmp_path):
    class Stuck:
        stdout = io.StringIO("")

        @staticmethod
        def poll():
            return None

        @staticmethod
        def terminate():
            return None

        @staticmethod
        def wait(timeout=None):
            raise __import__("subprocess").TimeoutExpired("docker", timeout)

        @staticmethod
        def kill():
            return None

    sampler = TrtllmTelemetrySampler(
        tmp_path, popen=lambda *args, **kwargs: Stuck(),
        hard_deadline=100.0, monotonic=lambda: 99.0,
    )
    sampler.start()
    try:
        sampler.stop()
    except RuntimeError as exc:
        assert "failed to drain" in str(exc)
    else:
        raise AssertionError("undrained telemetry process must fail closed")


def test_sampler_reports_observed_kill_and_exit_state(tmp_path):
    class Process:
        stdout = io.StringIO("")
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            return None

        def wait(self, timeout=None):
            if self.returncode is None:
                raise __import__("subprocess").TimeoutExpired("docker", timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    sampler = TrtllmTelemetrySampler(
        tmp_path, popen=lambda *args, **kwargs: Process(),
    )
    sampler.start()
    state = sampler.stop()["sampler_state"]
    assert state["terminate_requested"] is True
    assert state["kill_requested"] is True
    assert state["process_poll"] == -9
    assert state["process_stopped"] is True
    assert state["threads_alive"] == 0
