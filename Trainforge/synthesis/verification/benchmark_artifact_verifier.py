"""Content-addressed verification reports for benchmark operational evidence."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from Trainforge.generators.trtllm_benchmark_telemetry import parse_trtllm_log

TELEMETRY_REPORT_SCHEMA = "ed4all.benchmark-telemetry-verification.v1"
RECONCILIATION_REPORT_SCHEMA = "ed4all.http-reconciliation-verification.v1"
_SHA256 = frozenset("0123456789abcdef")
INTENT_VERSION = "ed4all.provider-call-intent.v1"
BOUND_INTENT_VERSION = "ed4all.provider-call-intent.v2"


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _finish(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "report_sha256": _hash(_canonical(body))}


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def read_call_intents(path: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read and strictly validate the independent logical-call authority."""
    source = Path(path)
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if not source.is_file():
        return [], [{"code": "intent_manifest_missing"}]
    required = {
        "intent_version", "utc", "run_id", "cell_id", "unit", "kind",
        "stage", "logical_attempt", "request_sha256", "model",
        "max_tokens", "temperature", "response_schema_sha256",
        "response_dialect", "contract_sha256",
    }
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append({"code": "intent_json_invalid", "line": line_number})
            continue
        version = row.get("intent_version") if isinstance(row, dict) else None
        bound = version == BOUND_INTENT_VERSION
        expected_keys = (
            required | {"synthesis_contract_sha256"} if bound else required
        )
        if not isinstance(row, dict) or set(row) != expected_keys:
            errors.append({"code": "intent_schema_invalid", "line": line_number})
            continue
        if (
            row["intent_version"] not in {INTENT_VERSION, BOUND_INTENT_VERSION}
            or row["kind"] not in {"initial", "repair", "dialect"}
            or isinstance(row["logical_attempt"], bool)
            or not isinstance(row["logical_attempt"], int)
            or row["logical_attempt"] <= 0
            or not _valid_hash(row["request_sha256"])
            or not _valid_hash(row["response_schema_sha256"])
            or not _valid_hash(row["contract_sha256"])
            or (
                bound
                and not _valid_hash(row["synthesis_contract_sha256"])
            )
            or not all(
                isinstance(row[field], str) and row[field]
                for field in ("run_id", "cell_id", "unit", "stage", "model")
            )
        ):
            errors.append({"code": "intent_value_invalid", "line": line_number})
            continue
        contract = {
            "stage": row["stage"],
            "model": row["model"],
            "max_tokens": row["max_tokens"],
            "temperature": row["temperature"],
            "response_schema_sha256": row["response_schema_sha256"],
            "response_dialect": row["response_dialect"],
        }
        if bound:
            contract["synthesis_contract_sha256"] = row[
                "synthesis_contract_sha256"
            ]
        if row["contract_sha256"] != _hash(
            json.dumps(
                contract, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ):
            errors.append({"code": "intent_contract_identity_mismatch", "line": line_number})
        expected_kind = (
            "dialect"
            if row["stage"] == "staged_synthesis:dialect_preflight"
            else "initial" if row["logical_attempt"] == 1 else "repair"
        )
        if row["kind"] != expected_kind:
            errors.append({"code": "intent_kind_invalid", "line": line_number})
        rows.append(row)
    identity = lambda row: (
        row["run_id"], row["cell_id"], row["unit"], row["stage"],
        row["logical_attempt"], row["request_sha256"], row["model"],
        row["contract_sha256"],
    )
    counts = Counter(identity(row) for row in rows)
    if any(count != 1 for count in counts.values()):
        errors.append({"code": "intent_identity_duplicate"})
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        groups[(row["unit"], row["stage"])].append(row["logical_attempt"])
    if any(sorted(values) != list(range(1, len(values) + 1)) for values in groups.values()):
        errors.append({"code": "intent_attempt_sequence_invalid"})
    return rows, errors


def verify_telemetry_artifacts(
    telemetry_root: Path | str,
    *,
    preflight: Mapping[str, Any],
    expected_preflight_sha256: str,
    expected_container: str,
    expected_port: int,
    expected_model: str,
    expected_max_num_tokens: int,
    expected_interval_seconds: float,
    sampler_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify raw telemetry and bind it to the exact benchmark runtime."""
    root = Path(telemetry_root)
    errors: list[dict[str, Any]] = []
    preflight_bytes = _canonical(dict(preflight))
    if not _valid_hash(expected_preflight_sha256) or (
        _hash(preflight_bytes) != expected_preflight_sha256
    ):
        errors.append({"code": "preflight_hash_mismatch"})
    bindings = {
        "container": preflight.get("container"),
        "published_port": preflight.get("published_port"),
        "expected_model": preflight.get("expected_model"),
        "max_num_tokens": preflight.get("max_num_tokens"),
    }
    expected_bindings = {
        "container": expected_container,
        "published_port": expected_port,
        "expected_model": expected_model,
        "max_num_tokens": expected_max_num_tokens,
    }
    if bindings != expected_bindings:
        errors.append({"code": "preflight_identity_mismatch"})
    parser_probe = preflight.get("parser_probe")
    if (
        preflight.get("status") != "accepted"
        or not isinstance(parser_probe, Mapping)
        or not isinstance(parser_probe.get("iteration_samples"), int)
        or parser_probe.get("iteration_samples", 0) <= 0
    ):
        errors.append({"code": "preflight_parser_probe_invalid"})
    if (
        isinstance(expected_interval_seconds, bool)
        or not isinstance(expected_interval_seconds, (int, float))
        or expected_interval_seconds <= 0
    ):
        errors.append({"code": "sampler_interval_invalid"})

    artifacts: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for name in ("trtllm.log", "system.jsonl"):
        path = root / name
        if not path.is_file():
            errors.append({"code": "telemetry_artifact_missing", "artifact": name})
            payload = b""
        else:
            payload = path.read_bytes()
        payloads[name] = payload
        artifacts[name] = {"sha256": _hash(payload), "bytes": len(payload)}

    parsed = parse_trtllm_log(
        payloads["trtllm.log"].decode("utf-8", errors="replace").splitlines()
    )
    if not parsed["iteration_samples"]:
        errors.append({"code": "iteration_samples_empty"})
    system_samples = 0
    for line_number, line in enumerate(
        payloads["system.jsonl"].decode("utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append({"code": "system_sample_invalid", "line": line_number})
            continue
        if not isinstance(row, Mapping) or "utc" not in row:
            errors.append({"code": "system_sample_schema_invalid", "line": line_number})
        else:
            system_samples += 1
    if not system_samples:
        errors.append({"code": "system_samples_empty"})

    state = dict(sampler_state)
    process_poll = state.get("process_poll")
    lifecycle_clean = (
        set(state) == {
            "started", "stop_requested", "process_present", "process_poll",
            "process_stopped", "terminate_requested", "kill_requested",
            "threads_total", "threads_alive", "threads_stopped", "errors",
        }
        and state.get("started") is True
        and state.get("stop_requested") is True
        and state.get("process_present") is True
        and isinstance(process_poll, int)
        and not isinstance(process_poll, bool)
        and state.get("process_stopped") is (process_poll is not None)
        and isinstance(state.get("terminate_requested"), bool)
        and isinstance(state.get("kill_requested"), bool)
        and (
            state.get("kill_requested") is False
            or state.get("terminate_requested") is True
        )
        and isinstance(state.get("threads_total"), int)
        and not isinstance(state.get("threads_total"), bool)
        and state.get("threads_total", 0) > 0
        and isinstance(state.get("threads_alive"), int)
        and not isinstance(state.get("threads_alive"), bool)
        and state.get("threads_alive") == 0
        and state.get("threads_stopped") is True
        and state.get("errors") == []
    )
    if not lifecycle_clean:
        errors.append({"code": "sampler_stop_not_clean"})
    body = {
        "schema_version": TELEMETRY_REPORT_SCHEMA,
        "status": "verified" if not errors else "rejected",
        "preflight_sha256": expected_preflight_sha256,
        "bindings": expected_bindings,
        "interval_seconds": expected_interval_seconds,
        "parser": {
            "schema": "trtllm_iteration_status_abort.v1",
            "status": "verified" if parsed["iteration_samples"] else "rejected",
            "iteration_samples": parsed["iteration_samples"],
            "system_samples": system_samples,
            "summary_sha256": _hash(_canonical(parsed)),
        },
        "sampler": state,
        "sampler_state_sha256": _hash(_canonical(state)),
        "artifacts": artifacts,
        "errors": errors,
    }
    return _finish(body)


def verify_http_reconciliation(
    ledger_path: Path | str,
    *,
    intent_manifest_path: Path | str | None = None,
    expected_identities: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify attempt identity bijection, terminal state, and raw artifacts."""
    path = Path(ledger_path)
    errors: list[dict[str, Any]] = []
    payload = path.read_bytes() if path.is_file() else b""
    if not path.is_file():
        errors.append({"code": "ledger_missing"})
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        payload.decode("utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append({"code": "ledger_json_invalid", "line": line_number})
            continue
        if not isinstance(row, Mapping):
            errors.append({"code": "ledger_row_invalid", "line": line_number})
        else:
            rows.append(row)

    intent_rows: list[dict[str, Any]] = []
    if intent_manifest_path is not None:
        intent_rows, intent_errors = read_call_intents(intent_manifest_path)
        errors.extend(intent_errors)
        expected_identities = [
            {
                "unit": row["unit"], "stage": row["stage"],
                "attempt": row["logical_attempt"],
                "request_sha256": row["request_sha256"],
            }
            for row in intent_rows
        ]
    elif expected_identities is None:
        errors.append({"code": "intent_manifest_missing"})
        expected_identities = ()
    fields = ("unit", "stage", "attempt", "request_sha256")
    def identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in fields)

    expected = Counter(identity(item) for item in expected_identities)
    if any(None in item for item in expected):
        errors.append({"code": "expected_identity_invalid"})
    started: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    terminal: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        target = (
            started if row.get("event") == "http_attempt_started"
            else terminal if row.get("event") == "http_attempt_terminal"
            else None
        )
        if target is None:
            errors.append({"code": "ledger_event_invalid"})
        else:
            target[identity(row)].append(row)
    observed = Counter({item: len(values) for item, values in started.items()})
    if observed != expected:
        errors.append({"code": "started_identity_mismatch"})
    if Counter({item: len(values) for item, values in terminal.items()}) != expected:
        errors.append({"code": "terminal_identity_mismatch"})

    raw_artifacts: dict[str, dict[str, Any]] = {}
    for item in sorted(set(started) | set(terminal), key=str):
        if len(started[item]) != 1 or len(terminal[item]) != 1:
            errors.append({"code": "attempt_not_bijective", "identity": list(item)})
            continue
        terminal_row = terminal[item][0]
        terminal_state = terminal_row.get("state")
        if terminal_state is None:
            terminal_state = (
                "completed" if isinstance(terminal_row.get("http_status"), int)
                else "failed" if terminal_row.get("exception_class") else None
            )
        if terminal_state not in {"completed", "failed"}:
            errors.append({"code": "terminal_state_invalid", "identity": list(item)})
        for ref_field, row in (
            ("request_raw_ref", started[item][0]),
            ("response_raw_ref", terminal_row),
        ):
            ref = row.get(ref_field)
            artifact = Path(str(ref.get("path"))) if isinstance(ref, Mapping) else None
            if artifact is None or not artifact.is_file():
                errors.append({"code": f"{ref_field}_missing", "identity": list(item)})
                continue
            raw = artifact.read_bytes()
            actual = _hash(raw)
            raw_artifacts[actual] = {"sha256": actual, "bytes": len(raw)}
            if ref.get("sha256") != actual or ref.get("bytes") != len(raw):
                errors.append({"code": f"{ref_field}_mismatch", "identity": list(item)})

    body = {
        "schema_version": RECONCILIATION_REPORT_SCHEMA,
        "status": "verified" if not errors else "rejected",
        "ledger": {"sha256": _hash(payload), "bytes": len(payload)},
        "intent_manifest": {
            "sha256": _hash(Path(intent_manifest_path).read_bytes())
            if intent_manifest_path is not None
            and Path(intent_manifest_path).is_file() else _hash(b""),
            "count": len(intent_rows),
        },
        "intent_identity": {
            "run_ids": sorted({row["run_id"] for row in intent_rows}),
            "cell_ids": sorted({row["cell_id"] for row in intent_rows}),
            "models": sorted({row["model"] for row in intent_rows}),
            "contract_sha256s": sorted({
                row["contract_sha256"] for row in intent_rows
            }),
        },
        "expected_attempts": sum(expected.values()),
        "started_attempts": sum(len(value) for value in started.values()),
        "terminal_attempts": sum(len(value) for value in terminal.values()),
        "identity_set_sha256": _hash(_canonical([
            list(item) for item in sorted(expected.elements(), key=str)
        ])),
        "raw_artifacts": dict(sorted(raw_artifacts.items())),
        "errors": errors,
    }
    return _finish(body)


def verified_report(report: Mapping[str, Any], schema: str) -> bool:
    """Return true only for an untampered, accepted verifier report."""
    if report.get("schema_version") != schema or report.get("status") != "verified":
        return False
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    return _valid_hash(digest) and digest == _hash(_canonical(body))
