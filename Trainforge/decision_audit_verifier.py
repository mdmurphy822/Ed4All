"""Deterministic verification of staged-synthesis decision-capture artifacts.

This module is deliberately read-only with respect to the artifacts it audits.
The caller must close the owning ``DecisionCapture`` before invoking
``verify_decision_audit`` and pass the exact expected provider-call identities.
That makes a missing, duplicate, or unrelated event a hard audit failure rather
than allowing a count-only reconciliation to hide it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Trainforge.benchmark_artifact_verifier import read_call_intents

_SHA256_LEN = 64
_PROVIDER_DECISION_TYPE = "synthesis_provider_call"
_IDENTITY_FIELDS = ("chunk_id", "stage", "attempt")
_NLI_FIELDS = (
    "decision_type",
    "stage",
    "attempt",
    "premise_sha256",
    "hypothesis_sha256",
    "entailment",
    "contradiction",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LEN
        and all(char in "0123456789abcdef" for char in value)
    )


def _identity(value: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        return (
            str(value["chunk_id"]),
            str(value["stage"]),
            int(value["attempt"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "provider identity requires chunk_id, stage, and integer attempt"
        ) from exc


def _identity_json(identity: tuple[str, str, int]) -> dict[str, Any]:
    return dict(zip(_IDENTITY_FIELDS, identity))


def _read_jsonl(path: Path, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append({"code": "decision_json_invalid", "line": line_number})
            continue
        if not isinstance(row, dict):
            errors.append({"code": "decision_row_not_object", "line": line_number})
            continue
        rows.append(row)
    return rows


def _verify_raw_ref(
    ref: Any,
    *,
    path_field: str,
    hash_field: str,
    bytes_field: str,
    event_index: int,
    kind: str,
    errors: list[dict[str, Any]],
) -> tuple[str | None, int | None]:
    if not isinstance(ref, Mapping):
        errors.append({"code": f"{kind}_ref_missing", "event_index": event_index})
        return None, None
    raw_path = ref.get(path_field)
    path = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
    if path is None or not path.is_file():
        errors.append({"code": f"{kind}_artifact_missing", "event_index": event_index})
        return None, None
    payload = path.read_bytes()
    expected_hash = ref.get(hash_field)
    expected_bytes = ref.get(bytes_field)
    if not _is_sha256(expected_hash) or _sha256(payload) != expected_hash:
        errors.append({"code": f"{kind}_hash_mismatch", "event_index": event_index})
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or len(payload) != expected_bytes
    ):
        errors.append({"code": f"{kind}_byte_count_mismatch", "event_index": event_index})
    if ref.get("hash_algorithm", "sha256") != "sha256":
        errors.append({"code": f"{kind}_hash_algorithm_invalid", "event_index": event_index})
    return _sha256(payload), len(payload)


def verify_decision_audit(
    decision_artifact_path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_provider_identities: Iterable[Mapping[str, Any]] | None = None,
    intent_manifest_path: Path | str | None = None,
    capture_closed: bool,
) -> dict[str, Any]:
    """Verify a closed staged-synthesis decision artifact and all its evidence.

    The returned report contains no absolute paths or timestamps. Its
    ``report_sha256`` covers the canonical JSON serialization of every other
    report field, making it suitable for inclusion in a later commit artifact.
    A rejected report is returned (not raised) so the complete deterministic
    error set can be persisted for diagnosis.
    """
    path = Path(decision_artifact_path)
    errors: list[dict[str, Any]] = []
    if not capture_closed:
        errors.append({"code": "capture_not_closed"})
    if not path.is_file():
        errors.append({"code": "decision_artifact_missing"})
        artifact_bytes = b""
        rows: list[dict[str, Any]] = []
    else:
        artifact_bytes = path.read_bytes()
        rows = _read_jsonl(path, errors)
    observed_artifact_sha256 = _sha256(artifact_bytes)
    if (
        not _is_sha256(expected_artifact_sha256)
        or observed_artifact_sha256 != expected_artifact_sha256
    ):
        errors.append({"code": "decision_artifact_hash_mismatch"})

    intent_rows: list[dict[str, Any]] = []
    if intent_manifest_path is not None:
        intent_rows, intent_errors = read_call_intents(intent_manifest_path)
        errors.extend(intent_errors)
        expected_provider_identities = [
            {
                "chunk_id": row["unit"], "stage": row["stage"],
                "attempt": row["logical_attempt"],
            }
            for row in intent_rows
        ]
    elif expected_provider_identities is None:
        expected_provider_identities = ()
        errors.append({"code": "intent_manifest_missing"})
    try:
        expected = Counter(_identity(item) for item in expected_provider_identities)
    except ValueError:
        expected = Counter()
        errors.append({"code": "expected_identity_invalid"})

    provider_rows = [
        row for row in rows if row.get("decision_type") == _PROVIDER_DECISION_TYPE
    ]
    observed: Counter[tuple[str, str, int]] = Counter()
    event_ids: set[str] = set()
    nli_count = 0

    for event_index, row in enumerate(provider_rows):
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            errors.append({"code": "event_id_missing", "event_index": event_index})
        elif event_id in event_ids:
            errors.append({"code": "event_id_duplicate", "event_index": event_index})
        else:
            event_ids.add(event_id)

        try:
            context = json.loads(row.get("context") or "")
        except (TypeError, json.JSONDecodeError):
            context = None
        if not isinstance(context, Mapping):
            errors.append({"code": "context_invalid", "event_index": event_index})
            continue
        try:
            identity = (
                (
                    str(context["intent_unit"]),
                    str(context["intent_stage"]),
                    int(context["intent_logical_attempt"]),
                )
                if intent_manifest_path is not None else _identity(context)
            )
        except (KeyError, TypeError, ValueError):
            errors.append({"code": "provider_identity_invalid", "event_index": event_index})
            continue
        observed[identity] += 1
        if intent_manifest_path is not None:
            matching = [
                item for item in intent_rows
                if (
                    item["unit"], item["stage"], item["logical_attempt"]
                ) == identity
            ]
            if len(matching) != 1 or any(
                context.get(context_key) != matching[0][row_key]
                for context_key, row_key in (
                    ("intent_run_id", "run_id"),
                    ("intent_cell_id", "cell_id"),
                    ("intent_kind", "kind"),
                    ("intent_request_sha256", "request_sha256"),
                    ("intent_model", "model"),
                    ("intent_contract_sha256", "contract_sha256"),
                )
            ):
                errors.append({
                    "code": "provider_intent_binding_mismatch",
                    "event_index": event_index,
                })

        inputs = row.get("inputs_ref")
        outputs = row.get("outputs")
        if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)) or len(inputs) != 1:
            errors.append({"code": "prompt_ref_cardinality", "event_index": event_index})
        else:
            prompt_digest, _ = _verify_raw_ref(
                inputs[0],
                path_field="path_or_id",
                hash_field="content_hash",
                bytes_field="size_bytes",
                event_index=event_index,
                kind="prompt",
                errors=errors,
            )
            if row.get("prompt_ref") != inputs[0].get("path_or_id"):
                errors.append({"code": "prompt_ref_identity_mismatch", "event_index": event_index})
            if prompt_digest is not None and context.get("prompt_sha256") != prompt_digest:
                errors.append({"code": "prompt_context_hash_mismatch", "event_index": event_index})
        if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)) or len(outputs) != 1:
            errors.append({"code": "response_ref_cardinality", "event_index": event_index})
        else:
            response_digest, _ = _verify_raw_ref(
                outputs[0],
                path_field="path",
                hash_field="content_hash",
                bytes_field="size_bytes",
                event_index=event_index,
                kind="response",
                errors=errors,
            )
            if response_digest is not None and context.get("response_sha256") != response_digest:
                errors.append({"code": "response_context_hash_mismatch", "event_index": event_index})

        evidence = context.get("validation_evidence")
        if not isinstance(evidence, Mapping):
            errors.append({"code": "validation_evidence_missing", "event_index": event_index})
            continue
        if (
            not _is_sha256(evidence.get("response_schema_sha256"))
            or not isinstance(evidence.get("required_keys"), list)
            or evidence.get("validator_stage") != context.get("stage")
            or evidence.get("validator_attempt") != context.get("attempt")
        ):
            errors.append({"code": "validation_evidence_invalid", "event_index": event_index})
        nli_rows = evidence.get("nli_scores")
        if not isinstance(nli_rows, list):
            errors.append({"code": "nli_evidence_missing", "event_index": event_index})
            continue
        for nli_index, nli in enumerate(nli_rows):
            nli_count += 1
            if (
                not isinstance(nli, Mapping)
                or any(field not in nli for field in _NLI_FIELDS)
                or not _is_sha256(nli.get("premise_sha256"))
                or not _is_sha256(nli.get("hypothesis_sha256"))
                or nli.get("stage") != context.get("stage")
                or nli.get("attempt") != context.get("attempt")
                or isinstance(nli.get("entailment"), bool)
                or not isinstance(nli.get("entailment"), (int, float))
                or not 0.0 <= float(nli.get("entailment")) <= 1.0
                or isinstance(nli.get("contradiction"), bool)
                or not isinstance(nli.get("contradiction"), (int, float))
                or not 0.0 <= float(nli.get("contradiction")) <= 1.0
            ):
                errors.append({
                    "code": "nli_evidence_invalid",
                    "event_index": event_index,
                    "nli_index": nli_index,
                })

    for identity in sorted(set(expected) | set(observed)):
        expected_count = expected[identity]
        observed_count = observed[identity]
        if expected_count != observed_count:
            errors.append({
                "code": "provider_identity_count_mismatch",
                "identity": _identity_json(identity),
                "expected": expected_count,
                "observed": observed_count,
            })

    core = {
        "schema_version": 1,
        "status": "accepted" if not errors else "rejected",
        "decision_artifact_sha256": observed_artifact_sha256,
        "decision_artifact_bytes": len(artifact_bytes),
        "intent_manifest_sha256": (
            _sha256(Path(intent_manifest_path).read_bytes())
            if intent_manifest_path is not None
            and Path(intent_manifest_path).is_file() else None
        ),
        "intent_count": len(intent_rows),
        "decision_rows": len(rows),
        "provider_events_expected": sum(expected.values()),
        "provider_events_observed": len(provider_rows),
        "provider_identities_verified": sum(
            min(expected[key], observed[key]) for key in set(expected) | set(observed)
        ),
        "nli_evidence_rows_verified": nli_count,
        "capture_closed": bool(capture_closed),
        "errors": errors,
    }
    return {**core, "report_sha256": _sha256(_canonical(core))}


def write_decision_audit_report(path: Path | str, report: Mapping[str, Any]) -> str:
    """Create an immutable canonical report and return its byte hash."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(dict(report))
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return _sha256(payload)
