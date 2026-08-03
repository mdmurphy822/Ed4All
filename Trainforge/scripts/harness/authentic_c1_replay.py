#!/usr/bin/env python3
"""Deterministic zero-network replay of an authentic staged C1 attempt graph.

This harness consumes the durable HTTP audit artifacts produced by a prior
run.  It never reconstructs absent responses and never contacts a provider.
The original request and response bytes remain the authority; replay emits a
new, deterministic adjudication/journal/checkpoint/publication evidence tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from Trainforge.generators.staged_synthesis_micro import (
    MICRO_CONTRACT_VERSION,
    MICRO_DPO_PROJECTION,
    MICRO_RELEASE_CONTRACT_VERSION,
    MICRO_SFT_PROJECTION,
    _CHOSEN_SCHEMA,
    _CLAIM_SCHEMA,
    _TASK_SCHEMA,
    micro_contract_fingerprint,
    validate_typed_givens,
)


AUTHENTIC_REPLAY_SCHEMA = "ed4all.authentic-c1-replay.v1"
NORMALIZATION_SCHEMA = "ed4all.authentic-c1-normalization.v1"
REPLAY_SYNTHESIS_SEED = 0
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _stable(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(_stable(value).encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} must be an object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{json.dumps(value, indent=2, sort_keys=True)}\n".encode("utf-8")
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(f"{_stable(dict(row))}\n".encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_raw_ref(ref: Mapping[str, Any], *, repository_root: Path) -> Path:
    raw = Path(str(ref.get("path") or ""))
    path = raw if raw.is_absolute() else repository_root / raw
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"recorded raw artifact is missing: {raw}")
    payload = resolved.read_bytes()
    expected = str(ref.get("sha256") or "")
    if not _HASH.fullmatch(expected) or _digest_bytes(payload) != expected:
        raise ValueError(f"recorded raw artifact hash differs: {raw}")
    if len(payload) != int(ref.get("bytes", -1)):
        raise ValueError(f"recorded raw artifact byte count differs: {raw}")
    return resolved


def build_authentic_inventory(
    *,
    source_cell: Path,
    frozen_manifest: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Freeze the exact 13-attempt, three-row request/response authority."""
    attempts = _read_jsonl(source_cell / "http_attempts.jsonl")
    intents = _read_jsonl(source_cell / "call-intents.jsonl")
    capture_paths = sorted(
        (source_cell / "audit" / "runtime/training-captures").rglob("decisions_*.jsonl")
    )
    if len(capture_paths) != 1:
        raise ValueError("authentic C1 authority requires one decision stream")
    capture_lines = [
        line.encode("utf-8")
        for line in capture_paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    captures = [json.loads(line) for line in capture_lines]
    manifest = _read_jsonl(frozen_manifest)
    by_row = {str(row.get("chunk_id") or ""): row for row in manifest}
    if len(by_row) != len(manifest) or "" in by_row:
        raise ValueError("frozen manifest row identities are not unique")
    if len(attempts) % 2:
        raise ValueError("HTTP attempt ledger has an unmatched event")
    terminals = []
    for offset in range(0, len(attempts), 2):
        started, terminal = attempts[offset:offset + 2]
        if (
            started.get("event") != "http_attempt_started"
            or terminal.get("event") != "http_attempt_terminal"
        ):
            raise ValueError("HTTP attempt events are not exact start/terminal pairs")
        identity = ("unit", "stage", "attempt", "request_sha256", "model")
        if any(started.get(key) != terminal.get(key) for key in identity):
            raise ValueError("HTTP start/terminal identity differs")
        terminals.append(terminal)
    if len(terminals) != 13 or len(intents) != 13 or len(captures) != 13:
        raise ValueError("authentic C1 authority must contain exactly 13 attempts")

    inventory_rows = []
    last_by_unit: dict[str, str] = {}
    authentic_rows: list[str] = []
    request_hashes: set[str] = set()
    response_hashes: set[str] = set()
    for position, (intent, terminal, capture, capture_bytes) in enumerate(
        zip(intents, terminals, captures, capture_lines)
    ):
        for key in ("unit", "stage", "logical_attempt", "request_sha256", "model"):
            expected = (
                terminal.get("attempt") if key == "logical_attempt"
                else terminal.get(key)
            )
            if intent.get(key) != expected:
                raise ValueError(f"intent/terminal mismatch at attempt {position}: {key}")
        request_path = _resolve_raw_ref(
            terminal["request_raw_ref"], repository_root=repository_root,
        )
        response_path = _resolve_raw_ref(
            terminal["response_raw_ref"], repository_root=repository_root,
        )
        request_sha = _digest_bytes(request_path.read_bytes())
        response_sha = _digest_bytes(response_path.read_bytes())
        if request_sha in request_hashes or response_sha in response_hashes:
            raise ValueError("authentic request/response bytes are not bijective")
        request_hashes.add(request_sha)
        response_hashes.add(response_sha)
        unit = str(terminal["unit"])
        capture_context = json.loads(str(capture.get("context") or "{}"))
        if (
            capture.get("decision_type") != "synthesis_provider_call"
            or capture_context.get("intent_request_sha256") != request_sha
            or capture_context.get("intent_unit") != unit
            or capture_context.get("intent_stage") != terminal["stage"]
        ):
            raise ValueError(
                f"decision capture is not bijective at attempt {position}"
            )
        chunk_id = unit.split(":", 1)[0] if unit != "dialect-preflight" else None
        manifest_row = by_row.get(str(chunk_id)) if chunk_id else None
        if chunk_id and manifest_row is None:
            raise ValueError(f"authentic attempt row is absent from manifest: {chunk_id}")
        if chunk_id and chunk_id not in authentic_rows:
            authentic_rows.append(chunk_id)
        attempt_identity = _digest({
            "position": position,
            "unit": unit,
            "stage": terminal["stage"],
            "logical_attempt": terminal["attempt"],
            "request_sha256": request_sha,
            "response_sha256": response_sha,
        })
        inventory_rows.append({
            "position": position,
            "attempt_identity_sha256": attempt_identity,
            "parent_attempt_identity_sha256": last_by_unit.get(unit),
            "unit": unit,
            "row_id": chunk_id,
            "row_sha256": (
                manifest_row.get("chunk_sha256") if manifest_row else None
            ),
            "kind": manifest_row.get("kind") if manifest_row else "dialect",
            "stage": terminal["stage"],
            "logical_attempt": terminal["attempt"],
            "request_sha256": request_sha,
            "request_bytes": request_path.stat().st_size,
            "response_sha256": response_sha,
            "response_bytes": response_path.stat().st_size,
            "model": terminal["model"],
            "finish_reason": terminal.get("finish_reason"),
            "usage": terminal.get("usage") or {},
            "http_status": terminal.get("http_status"),
            "transport": {
                "endpoint_scheme": str(terminal.get("endpoint") or "").split(":", 1)[0],
                "timeout_seconds": terminal.get("actual_timeout_seconds"),
            },
            "authentic_validator_capture": {
                "capture_event_sha256": _digest_bytes(capture_bytes),
                "prompt_sha256": capture_context.get("prompt_sha256"),
                "response_content_sha256": capture_context.get(
                    "response_sha256"
                ),
                "validation_error": capture_context.get("validation_error"),
                "validation_evidence": capture_context.get(
                    "validation_evidence"
                ),
            },
        })
        last_by_unit[unit] = attempt_identity
    if len(authentic_rows) != 3:
        raise ValueError("authentic C1 authority must span exactly three rows")
    missing = [
        {
            "row_id": row["chunk_id"],
            "row_sha256": row["chunk_sha256"],
            "kind": row["kind"],
            "status": "missing_by_construction",
            "authentic_attempt_count": 0,
        }
        for row in manifest
        if row["chunk_id"] not in authentic_rows
    ]
    if len(missing) != 5:
        raise ValueError("exactly five frozen C1 rows must be missing by construction")
    return {
        "schema_version": AUTHENTIC_REPLAY_SCHEMA,
        "attempt_count": 13,
        "authentic_row_count": 3,
        "authentic_row_order": authentic_rows,
        "frozen_manifest_sha256": _digest_bytes(frozen_manifest.read_bytes()),
        "source_http_ledger_sha256": _digest_bytes(
            (source_cell / "http_attempts.jsonl").read_bytes()
        ),
        "source_intent_manifest_sha256": _digest_bytes(
            (source_cell / "call-intents.jsonl").read_bytes()
        ),
        "source_decision_stream_sha256": _digest_bytes(
            capture_paths[0].read_bytes()
        ),
        "attempts": inventory_rows,
        "missing_rows": missing,
    }


def _schema_for(stage: str) -> Mapping[str, Any] | None:
    if stage.endswith("dialect_preflight"):
        return {
            "type": "object", "required": ["probe"],
            "properties": {"probe": {"const": "ready"}},
            "additionalProperties": False,
        }
    if "micro_A_" in stage:
        return _TASK_SCHEMA
    if "micro_B_" in stage:
        return _CLAIM_SCHEMA
    if "micro_D_" in stage:
        return _CHOSEN_SCHEMA
    return None


def _adjudicate_response(
    response_bytes: bytes, *, attempt: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = json.loads(response_bytes)
    choices = envelope.get("choices") if isinstance(envelope, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        return {"outcome": "invalid_envelope", "critical": True}
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    reasoning = (
        message.get("reasoning_content")
        if isinstance(message, dict) else None
    )
    if finish == "length":
        authentic = attempt.get("authentic_validator_capture") or {}
        return {
            "outcome": "output_truncated",
            "critical": True,
            "parse": "barred_before_partial_parse",
            "publication_eligible": False,
            "authentic_validator": {
                "outcome": "rejected",
                "validation_error": authentic.get("validation_error"),
                "validation_evidence": authentic.get("validation_evidence"),
                "capture_event_sha256": authentic.get(
                    "capture_event_sha256"
                ),
            },
        }
    if reasoning:
        return {
            "outcome": "reasoning_content_present",
            "critical": True,
            "publication_eligible": False,
        }
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return {"outcome": "missing_content", "critical": True}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {
            "outcome": "invalid_json", "critical": True,
            "publication_eligible": False,
        }
    schema = _schema_for(str(attempt["stage"]))
    errors = (
        sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda error: list(error.absolute_path),
        )
        if schema is not None else []
    )
    semantic_error = None
    if not errors and "micro_A_" in str(attempt["stage"]):
        semantic_error = validate_typed_givens(value.get("generated_givens"))
    replay_result = {
        "outcome": (
            "schema_rejected" if errors
            else "semantic_rejected" if semantic_error
            else "parsed_valid"
        ),
        "critical": False,
        "parse": "complete_json",
        "schema_errors": [
            {
                "path": "/" + "/".join(map(str, error.absolute_path)),
                "validator": error.validator,
            }
            for error in errors
        ],
        "semantic_error": semantic_error,
        "content_sha256": _digest(value),
        "publication_eligible": False,
    }
    authentic = attempt.get("authentic_validator_capture") or {}
    authentic_error = authentic.get("validation_error")
    replay_result["authentic_validator"] = {
        "outcome": "rejected" if authentic_error else "accepted",
        "validation_error": authentic_error,
        "validation_evidence": authentic.get("validation_evidence"),
        "capture_event_sha256": authentic.get("capture_event_sha256"),
    }
    return replay_result


@contextmanager
def _deny_network() -> Iterable[list[dict[str, Any]]]:
    denied: list[dict[str, Any]] = []
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    def blocked_socket(*args: Any, **kwargs: Any) -> Any:
        denied.append({"api": "socket.socket", "args_count": len(args)})
        raise RuntimeError("outbound networking is mechanically denied")

    def blocked_connection(*args: Any, **kwargs: Any) -> Any:
        denied.append({"api": "socket.create_connection", "args_count": len(args)})
        raise RuntimeError("outbound networking is mechanically denied")

    socket.socket = blocked_socket  # type: ignore[assignment]
    socket.create_connection = blocked_connection  # type: ignore[assignment]
    try:
        yield denied
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]


def replay_once(
    *,
    inventory: Mapping[str, Any],
    source_cell: Path,
    repository_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("replay output must start from a clean empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_terminals = [
        row for row in _read_jsonl(source_cell / "http_attempts.jsonl")
        if row.get("event") == "http_attempt_terminal"
    ]
    if len(source_terminals) != len(inventory["attempts"]):
        raise ValueError("source terminal count differs from frozen inventory")
    journals = []
    telemetry = []
    captures = []
    row_attempts: dict[str, list[dict[str, Any]]] = {}
    network_attempt_ledger: list[dict[str, Any]] = []
    with _deny_network() as denied:
        # A deliberate proof probe establishes that the guard is active.  It
        # is not a provider attempt and is kept separate from the empty ledger.
        try:
            socket.create_connection(("replay.invalid", 443))
        except RuntimeError:
            pass
        for authority, terminal in zip(inventory["attempts"], source_terminals):
            response_path = _resolve_raw_ref(
                terminal["response_raw_ref"], repository_root=repository_root,
            )
            request_path = _resolve_raw_ref(
                terminal["request_raw_ref"], repository_root=repository_root,
            )
            if _digest_bytes(request_path.read_bytes()) != authority["request_sha256"]:
                raise ValueError("replay request authority differs from inventory")
            if _digest_bytes(response_path.read_bytes()) != authority["response_sha256"]:
                raise ValueError("replay response authority differs from inventory")
            adjudication = _adjudicate_response(
                response_path.read_bytes(), attempt=authority,
            )
            identity = authority["attempt_identity_sha256"]
            journals.extend([
                {
                    "attempt_identity_sha256": identity,
                    "state": "started",
                    "position": authority["position"],
                },
                {
                    "attempt_identity_sha256": identity,
                    "state": "terminal",
                    "position": authority["position"],
                    "adjudication": adjudication,
                },
            ])
            telemetry.append({
                "attempt_identity_sha256": identity,
                "position": authority["position"],
                "stage": authority["stage"],
                "finish_reason": authority["finish_reason"],
                "usage": authority["usage"],
                "response_bytes": authority["response_bytes"],
                "outcome": adjudication["outcome"],
            })
            captures.append({
                "decision_type": "authentic_c1_recorded_response",
                "attempt_identity_sha256": identity,
                "request_sha256": authority["request_sha256"],
                "response_sha256": authority["response_sha256"],
                "stage": authority["stage"],
                "model": authority["model"],
                "rationale": (
                    f"Replayed authentic attempt position {authority['position']} "
                    f"at {authority['stage']} with exact recorded byte hashes."
                ),
            })
            if authority["row_id"]:
                row_attempts.setdefault(authority["row_id"], []).append({
                    "authority": authority, "adjudication": adjudication,
                })
    checkpoints = []
    for row_id in inventory["authentic_row_order"]:
        attempts = row_attempts[row_id]
        last = attempts[-1]
        critical = any(item["adjudication"]["critical"] for item in attempts)
        checkpoints.append({
            "row_id": row_id,
            "row_sha256": last["authority"]["row_sha256"],
            "attempt_count": len(attempts),
            "terminal_disposition": (
                "critical_truncation_hold"
                if last["adjudication"]["outcome"] == "output_truncated"
                else "bounded_no_progress_hold"
            ),
            "critical": critical,
            "accepted": False,
            "projection": None,
        })
    publication = {
        "schema_version": AUTHENTIC_REPLAY_SCHEMA,
        "release_contract_version": MICRO_RELEASE_CONTRACT_VERSION,
        "synthesis_contract_version": MICRO_CONTRACT_VERSION,
        "synthesis_contract_sha256": micro_contract_fingerprint(),
        "synthesis_seed": REPLAY_SYNTHESIS_SEED,
        "seed_provenance": (
            "replay-assigned compatibility identity; the historical HTTP "
            "requests did not carry a synthesis seed"
        ),
        "projection_contracts": {
            "instruction": MICRO_SFT_PROJECTION,
            "preference": MICRO_DPO_PROJECTION,
        },
        "published_count": 0,
        "published": [],
        "terminal_holds": checkpoints,
    }
    _write_jsonl(output_dir / "journal.jsonl", journals)
    _write_jsonl(output_dir / "checkpoint.jsonl", checkpoints)
    _write_jsonl(output_dir / "decision-captures.jsonl", captures)
    _write_jsonl(output_dir / "telemetry.jsonl", telemetry)
    _write_json(output_dir / "classification.json", {
        "rows": checkpoints,
        "attempt_outcomes": [row["outcome"] for row in telemetry],
    })
    _write_json(output_dir / "publication.json", publication)
    _write_json(output_dir / "zero-network-proof.json", {
        "socket_guard_active": bool(denied),
        "denied_guard_probes": denied,
        "provider_call_count": 0,
        "connect_call_count": 0,
        "network_attempt_ledger": network_attempt_ledger,
    })
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.name == "verification.json" or not path.is_file():
            continue
        files[path.name] = _digest_bytes(path.read_bytes())
    verification = {
        "status": "accepted",
        "attempt_count": len(telemetry),
        "journal_started": sum(row["state"] == "started" for row in journals),
        "journal_terminal": sum(row["state"] == "terminal" for row in journals),
        "checkpoint_rows": len(checkpoints),
        "published_count": 0,
        "network_attempt_count": 0,
        "files": files,
        "semantic_sha256": _digest({
            "journals": journals,
            "checkpoints": checkpoints,
            "captures": captures,
            "telemetry": telemetry,
            "publication": publication,
        }),
    }
    _write_json(output_dir / "verification.json", verification)
    return verification


def run_replay_twice(
    *,
    source_cell: Path,
    frozen_manifest: Path,
    repository_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    inventory = build_authentic_inventory(
        source_cell=source_cell,
        frozen_manifest=frozen_manifest,
        repository_root=repository_root,
    )
    _write_json(evidence_root / "inventory.json", inventory)
    _write_json(evidence_root / "request-response-bijection.json", {
        "attempt_count": 13,
        "unique_request_count": len({
            row["request_sha256"] for row in inventory["attempts"]
        }),
        "unique_response_count": len({
            row["response_sha256"] for row in inventory["attempts"]
        }),
        "status": "accepted",
    })
    _write_json(evidence_root / "missing-five-record.json", {
        "rows": inventory["missing_rows"],
        "count": len(inventory["missing_rows"]),
        "construction_rule": "no authentic terminal HTTP response existed",
    })
    normalization = {
        "schema_version": NORMALIZATION_SCHEMA,
        "normalized_fields": [],
        "statement": (
            "Replay artifacts intentionally omit wall-clock timestamps and "
            "temporary-root names, so byte comparison needs no field removal."
        ),
        "implementation_sha256": _digest(
            {"stable_json": "sorted-keys,compact-utf8", "normalized_fields": []}
        ),
    }
    _write_json(evidence_root / "normalization-contract.json", normalization)
    first = replay_once(
        inventory=inventory, source_cell=source_cell,
        repository_root=repository_root, output_dir=evidence_root / "run-1",
    )
    second = replay_once(
        inventory=inventory, source_cell=source_cell,
        repository_root=repository_root, output_dir=evidence_root / "run-2",
    )
    comparison = {
        "status": "accepted" if first == second else "rejected",
        "run_1_semantic_sha256": first["semantic_sha256"],
        "run_2_semantic_sha256": second["semantic_sha256"],
        "semantic_equal": first["semantic_sha256"] == second["semantic_sha256"],
        "byte_hash_maps_equal": first["files"] == second["files"],
    }
    if comparison["status"] != "accepted":
        raise ValueError("two clean authentic replays differ")
    _write_json(evidence_root / "semantic-diff.json", comparison)
    _write_json(evidence_root / "zero-network-proof.json", {
        "run_1": json.loads(
            (evidence_root / "run-1" / "zero-network-proof.json").read_text()
        ),
        "run_2": json.loads(
            (evidence_root / "run-2" / "zero-network-proof.json").read_text()
        ),
        "status": "accepted",
    })
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cell", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    run_replay_twice(
        source_cell=args.source_cell,
        frozen_manifest=args.frozen_manifest,
        repository_root=args.repository_root,
        evidence_root=args.evidence_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
