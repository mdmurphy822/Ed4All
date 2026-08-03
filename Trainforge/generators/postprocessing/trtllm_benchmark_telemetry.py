"""Non-invasive TRT-LLM benchmark telemetry from container logs and NVML CLI."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import urlsplit

MAX_SCHEDULED_TOKENS = 8192
CONTAINER = "trtllm-super"
STATIC_KV_FACTS = {
    "source": "server_startup_log",
    "kv_blocks": 28964,
    "tokens_per_block": 32,
    "kv_token_capacity": 926848,
    "kv_cache_gib": 3.54,
    "max_sequence_blocks": 8193,
    "max_seq_len": 262144,
}

_ITERATION = re.compile(
    r"num_scheduled_requests\s*[:=]\s*(\d+).*?"
    r"num_ctx_tokens['\"]?\s*[:=]\s*(\d+).*?"
    r"num_generation_tokens['\"]?\s*[:=]\s*(\d+)"
)
_HTTP_STATUS = re.compile(r"(?:status(?:_code)?|HTTP/\d(?:\.\d)?)\D{0,8}(\d{3})")
_ABORT = re.compile(r"\b(abort(?:ed|ing)?|disconnect(?:ed|ion)?|cancelled)\b", re.I)


def parse_trtllm_log(lines: Iterable[str]) -> dict[str, Any]:
    """Summarize only values observable in timestamped TRT-LLM log lines."""
    peak_sequences = 0
    peak_tokens = 0
    iterations = 0
    statuses: dict[str, int] = {}
    abort_disconnects = 0
    for line in lines:
        match = _ITERATION.search(line)
        if match:
            scheduled, context, generation = map(int, match.groups())
            peak_sequences = max(peak_sequences, scheduled)
            peak_tokens = max(peak_tokens, context + generation)
            iterations += 1
        for status in _HTTP_STATUS.findall(line):
            statuses[status] = statuses.get(status, 0) + 1
        abort_disconnects += len(_ABORT.findall(line))
    return {
        "iteration_samples": iterations,
        "peak_scheduled_sequences": peak_sequences if iterations else None,
        "peak_scheduled_token_usage": peak_tokens if iterations else None,
        "peak_scheduled_token_headroom": (
            MAX_SCHEDULED_TOKENS - peak_tokens if iterations else None
        ),
        "scheduled_token_limit": MAX_SCHEDULED_TOKENS,
        "http_status_counts": dict(sorted(statuses.items())),
        "abort_disconnect_count": abort_disconnects,
        "labels": {
            "scheduled_tokens": (
                "num_ctx_tokens + num_generation_tokens; batch-token budget "
                "usage, not KV-cache occupancy"
            ),
            "gpu_memory": (
                "nvidia-smi process allocation and system utilization; not "
                "exact free unified VRAM"
            ),
            "throughput": (
                "client HTTP attempt ledger is authoritative for tok/s and latency"
            ),
        },
        "dynamic_kv_free": "unavailable_without_service_reseat_or_new_server_metrics",
        "dynamic_queue_delay": "unavailable_from_current_non_streaming_interfaces",
        "static_kv_startup_facts": dict(STATIC_KV_FACTS),
    }


def _run(command: list[str], *, timeout: float = 10.0) -> str:
    return subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    ).stdout


def telemetry_preflight(
    *, container: str = CONTAINER, runner: Callable[..., str] = _run,
    expected_model: Optional[str] = None,
    model_snapshot: Optional[Mapping[str, Any]] = None,
    base_url: Optional[str] = None,
    hard_deadline: Optional[float] = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Fail closed unless container config and scheduled-token logs are bound."""
    def run(command: list[str]) -> str:
        timeout = 10.0
        if hard_deadline is not None:
            timeout = min(timeout, hard_deadline - monotonic())
            if timeout <= 0:
                raise RuntimeError("telemetry preflight exceeded absolute cell deadline")
        return runner(command, timeout=timeout)

    inspect = run([
        "docker", "inspect", container, "--format", "{{json .Config.Cmd}}",
    ])
    command = json.loads(inspect.strip())
    model_argument = str(command[1]) if len(command) > 1 else ""
    command_snapshot = Path(model_argument).name
    if "--max_num_tokens" not in command:
        raise RuntimeError("TRT-LLM command omits --max_num_tokens")
    index = command.index("--max_num_tokens")
    if index + 1 >= len(command) or int(command[index + 1]) != MAX_SCHEDULED_TOKENS:
        raise RuntimeError("TRT-LLM scheduled-token limit is not bound to 8192")
    if "--max_seq_len" not in command:
        raise RuntimeError("TRT-LLM command omits --max_seq_len")
    seq_index = command.index("--max_seq_len")
    if seq_index + 1 >= len(command):
        raise RuntimeError("TRT-LLM command has no --max_seq_len value")
    served_context_tokens = int(command[seq_index + 1])
    if served_context_tokens <= 0:
        raise RuntimeError("TRT-LLM --max_seq_len must be positive")
    if expected_model is not None:
        served = {
            str(item.get("id")) for item in (model_snapshot or {}).get("data", [])
            if isinstance(item, Mapping)
        }
        if expected_model not in served:
            raise RuntimeError("telemetry preflight model is absent from /models snapshot")
        if command_snapshot != expected_model:
            raise RuntimeError(
                "container command model snapshot does not match configured/served model"
            )
    published_port = None
    if base_url is not None:
        ports = json.loads(run([
            "docker", "inspect", container, "--format",
            "{{json .NetworkSettings.Ports}}",
        ]).strip())
        split = urlsplit(base_url)
        requested_host = (split.hostname or "").lower()
        if requested_host == "localhost":
            requested_host = "127.0.0.1"
        requested_port = split.port or (443 if split.scheme == "https" else 80)
        candidates = []
        for bindings in ports.values():
            for binding in bindings or []:
                host = str(binding.get("HostIp") or "").lower()
                if host in {"", "0.0.0.0", "::"}:
                    host = "127.0.0.1"
                if host == "localhost":
                    host = "127.0.0.1"
                candidates.append((host, int(binding["HostPort"])))
        if (requested_host, requested_port) not in candidates:
            raise RuntimeError("base URL host/port does not map to named container")
        published_port = requested_port
    recent = run([
        "docker", "logs", "--since", "24h", "--timestamps", container,
    ])
    parsed = parse_trtllm_log(recent.splitlines())
    if not parsed["iteration_samples"]:
        raise RuntimeError(
            "TRT-LLM log source has no parseable scheduled-token iteration"
        )
    return {
        "status": "accepted",
        "container": container,
        "max_num_tokens": MAX_SCHEDULED_TOKENS,
        "served_context_tokens": served_context_tokens,
        "expected_model": expected_model,
        "container_model_argument_sha256": __import__("hashlib").sha256(
            model_argument.encode()
        ).hexdigest(),
        "container_model_snapshot": command_snapshot,
        "canonical_base_host": (
            "127.0.0.1" if base_url and urlsplit(base_url).hostname == "localhost"
            else (urlsplit(base_url).hostname if base_url else None)
        ),
        "published_port": published_port,
        "command_sha256": __import__("hashlib").sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "parser_probe": parsed,
    }


class TrtllmTelemetrySampler:
    """Capture isolated container logs plus host/GPU observations."""

    def __init__(
        self, root: Path, *, container: str = CONTAINER, interval: float = 2.0,
        popen: Callable[..., Any] = subprocess.Popen,
        runner: Callable[..., str] = _run,
        hard_deadline: Optional[float] = None,
        monotonic: Callable[[], float] = time.monotonic,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        self.root = Path(root)
        self.container = container
        self.interval = interval
        self._popen = popen
        self._runner = runner
        self._hard_deadline = hard_deadline
        self._monotonic = monotonic
        self._thread_factory = thread_factory
        self._stop = threading.Event()
        self._process: Any = None
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stop_requested = False
        self._terminate_requested = False
        self._kill_requested = False
        self._lifecycle_errors: list[str] = []
        self._since = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self._process = self._popen(
                ["docker", "logs", "--follow", "--since", self._since,
                 "--timestamps", self.container],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1,
            )
            candidates = [
                self._thread_factory(
                    target=self._run_worker,
                    args=(self._capture_logs, "capture_thread_failed"),
                    daemon=True,
                ),
                self._thread_factory(
                    target=self._run_worker,
                    args=(self._sample_system, "system_thread_failed"),
                    daemon=True,
                ),
            ]
            self._threads = []
            for thread in candidates:
                thread.start()
                self._threads.append(thread)
            self._started = True
        except BaseException as exc:
            try:
                self.stop()
            except BaseException as cleanup_exc:
                raise RuntimeError(
                    "telemetry start failed and partial resources did not drain"
                ) from cleanup_exc
            raise exc

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._stop_requested = self._stop.is_set()
        cleanup_error: Optional[BaseException] = None
        def remaining(cap: float) -> float:
            if self._hard_deadline is None:
                return cap
            return max(0.0, min(cap, self._hard_deadline - self._monotonic()))
        if self._process is not None and self._process.poll() is None:
            self._terminate_requested = True
            self._process.terminate()
            try:
                self._process.wait(timeout=remaining(2.0))
            except subprocess.TimeoutExpired:
                self._kill_requested = True
                self._process.kill()
                try:
                    self._process.wait(timeout=remaining(1.0))
                except subprocess.TimeoutExpired as exc:
                    self._lifecycle_errors.append("process_wait_timeout_after_kill")
                    cleanup_error = exc
        for thread in self._threads:
            thread.join(timeout=remaining(2.0))
        if (
            (self._process is not None and self._process.poll() is None)
            or any(thread.is_alive() for thread in self._threads)
        ):
            cleanup_error = cleanup_error or RuntimeError(
                "telemetry sampler failed to drain within absolute cell deadline"
            )
        if cleanup_error is not None:
            raise RuntimeError(
                "telemetry sampler failed to drain within absolute cell deadline"
            ) from cleanup_error
        process_poll = self._process.poll() if self._process is not None else None
        threads_alive = sum(thread.is_alive() for thread in self._threads)
        sampler_state = {
            "started": self._started,
            "stop_requested": self._stop_requested,
            "process_present": self._process is not None,
            "process_poll": process_poll,
            "process_stopped": self._process is not None and process_poll is not None,
            "terminate_requested": self._terminate_requested,
            "kill_requested": self._kill_requested,
            "threads_total": len(self._threads),
            "threads_alive": threads_alive,
            "threads_stopped": bool(self._threads) and threads_alive == 0,
            "errors": list(self._lifecycle_errors),
        }
        raw_path = self.root / "trtllm.log"
        summary = parse_trtllm_log(
            raw_path.read_text(encoding="utf-8").splitlines()
            if raw_path.exists() else []
        )
        summary["sampler_state"] = sampler_state
        _write_json(self.root / "summary.json", summary)
        return summary

    def _run_worker(self, target: Callable[[], None], error_code: str) -> None:
        try:
            target()
        except BaseException:
            self._lifecycle_errors.append(error_code)
            self._stop.set()

    def _capture_logs(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        path = self.root / "trtllm.log"
        with path.open("a", encoding="utf-8") as handle:
            for line in self._process.stdout:
                handle.write(line)
                handle.flush()
                if self._stop.is_set():
                    break
            os.fsync(handle.fileno())

    def _sample_system(self) -> None:
        path = self.root / "system.jsonl"
        while not self._stop.is_set():
            row: dict[str, Any] = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "mem_available_kib": _mem_available_kib(),
            }
            for name, command in {
                "compute_apps": [
                    "nvidia-smi", "--query-compute-apps=timestamp,pid,process_name,"
                    "used_memory", "--format=csv,noheader,nounits",
                ],
                "gpu": [
                    "nvidia-smi", "--query-gpu=timestamp,utilization.gpu,"
                    "utilization.memory,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
            }.items():
                try:
                    budget = (
                        self._hard_deadline - self._monotonic()
                        if self._hard_deadline is not None else 1.0
                    )
                    if budget <= 0:
                        raise TimeoutError("absolute telemetry deadline reached")
                    row[name] = self._runner(
                        command, timeout=min(1.0, budget),
                    ).splitlines()
                except Exception as exc:
                    row[name] = {"unavailable": type(exc).__name__}
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._stop.wait(self.interval)


def _mem_available_kib(path: Path = Path("/proc/meminfo")) -> Optional[int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
