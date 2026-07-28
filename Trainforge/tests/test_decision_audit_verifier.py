from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from Trainforge.decision_audit_verifier import (
    verify_decision_audit,
    write_decision_audit_report,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _event(tmp_path: Path) -> dict:
    prompt = b"Produce one grounded response."
    response = b'{"answer":"Atomic operations commit together."}'
    prompt_path = tmp_path / "prompt.txt"
    response_path = tmp_path / "response.txt"
    prompt_path.write_bytes(prompt)
    response_path.write_bytes(response)
    context = {
        "chunk_id": "chunk-generic",
        "stage": "sft_realization",
        "attempt": 1,
        "prompt_sha256": _sha(prompt),
        "response_sha256": _sha(response),
        "validation_evidence": {
            "response_schema_sha256": "a" * 64,
            "required_keys": ["answer"],
            "validator_stage": "sft_realization",
            "validator_attempt": 1,
            "nli_scores": [{
                "decision_type": "semantic_coverage",
                "stage": "sft_realization",
                "attempt": 1,
                "premise_sha256": "b" * 64,
                "hypothesis_sha256": "c" * 64,
                "entailment": 0.99,
                "contradiction": 0.001,
            }],
        },
    }
    return {
        "event_id": "EVT_0123456789abcdef",
        "decision_type": "synthesis_provider_call",
        "context": json.dumps(context, sort_keys=True, separators=(",", ":")),
        "prompt_ref": str(prompt_path),
        "inputs_ref": [{
            "path_or_id": str(prompt_path),
            "content_hash": _sha(prompt),
            "hash_algorithm": "sha256",
            "size_bytes": len(prompt),
        }],
        "outputs": [{
            "path": str(response_path),
            "content_hash": _sha(response),
            "hash_algorithm": "sha256",
            "size_bytes": len(response),
        }],
    }


def _write_artifact(path: Path, events: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in events).encode()
    path.write_bytes(payload)
    return _sha(payload)


def _verify(tmp_path: Path, events: list[dict], **overrides):
    artifact = tmp_path / "decisions.jsonl"
    digest = _write_artifact(artifact, events)
    kwargs = {
        "expected_artifact_sha256": digest,
        "expected_provider_identities": [{
            "chunk_id": "chunk-generic",
            "stage": "sft_realization",
            "attempt": 1,
        }],
        "capture_closed": True,
    }
    kwargs.update(overrides)
    return verify_decision_audit(artifact, **kwargs)


def test_accepts_complete_closed_audit_and_emits_deterministic_hash(tmp_path: Path):
    first = _verify(tmp_path, [_event(tmp_path)])
    second = verify_decision_audit(
        tmp_path / "decisions.jsonl",
        expected_artifact_sha256=first["decision_artifact_sha256"],
        expected_provider_identities=[{
            "chunk_id": "chunk-generic",
            "stage": "sft_realization",
            "attempt": 1,
        }],
        capture_closed=True,
    )
    assert first == second
    assert first["status"] == "accepted"
    assert first["provider_events_observed"] == 1
    assert first["nli_evidence_rows_verified"] == 1
    assert len(first["report_sha256"]) == 64


def test_rejects_decision_artifact_tampering(tmp_path: Path):
    event = _event(tmp_path)
    artifact = tmp_path / "decisions.jsonl"
    original_digest = _write_artifact(artifact, [event])
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    report = verify_decision_audit(
        artifact,
        expected_artifact_sha256=original_digest,
        expected_provider_identities=[{
            "chunk_id": "chunk-generic", "stage": "sft_realization", "attempt": 1,
        }],
        capture_closed=True,
    )
    assert "decision_artifact_hash_mismatch" in {
        error["code"] for error in report["errors"]
    }


@pytest.mark.parametrize(("ref_kind", "filename"), [
    ("prompt", "prompt.txt"),
    ("response", "response.txt"),
])
def test_rejects_raw_artifact_tampering(
    tmp_path: Path, ref_kind: str, filename: str,
):
    event = _event(tmp_path)
    (tmp_path / filename).write_bytes(b"tampered")
    report = _verify(tmp_path, [event])
    assert f"{ref_kind}_hash_mismatch" in {
        error["code"] for error in report["errors"]
    }
    assert f"{ref_kind}_byte_count_mismatch" in {
        error["code"] for error in report["errors"]
    }


def test_rejects_wrong_and_orphan_provider_identities(tmp_path: Path):
    event = _event(tmp_path)
    report = _verify(
        tmp_path,
        [event],
        expected_provider_identities=[{
            "chunk_id": "another-chunk", "stage": "sft_realization", "attempt": 1,
        }],
    )
    mismatches = [
        error for error in report["errors"]
        if error["code"] == "provider_identity_count_mismatch"
    ]
    assert len(mismatches) == 2
    assert {item["observed"] for item in mismatches} == {0, 1}


def test_rejects_duplicate_provider_identity_and_event_id(tmp_path: Path):
    event = _event(tmp_path)
    report = _verify(tmp_path, [event, deepcopy(event)])
    codes = {error["code"] for error in report["errors"]}
    assert "event_id_duplicate" in codes
    assert "provider_identity_count_mismatch" in codes


@pytest.mark.parametrize("mutation", ["missing_validation", "bad_nli_hash", "wrong_nli_identity"])
def test_rejects_validation_and_nli_coverage_tampering(
    tmp_path: Path, mutation: str,
):
    event = _event(tmp_path)
    context = json.loads(event["context"])
    if mutation == "missing_validation":
        del context["validation_evidence"]
    elif mutation == "bad_nli_hash":
        context["validation_evidence"]["nli_scores"][0]["premise_sha256"] = "bad"
    else:
        context["validation_evidence"]["nli_scores"][0]["attempt"] = 2
    event["context"] = json.dumps(context, sort_keys=True, separators=(",", ":"))
    report = _verify(tmp_path, [event])
    assert report["status"] == "rejected"
    assert {
        error["code"] for error in report["errors"]
    } & {"validation_evidence_missing", "nli_evidence_invalid"}


def test_rejects_open_capture_even_when_artifacts_are_complete(tmp_path: Path):
    report = _verify(tmp_path, [_event(tmp_path)], capture_closed=False)
    assert report["status"] == "rejected"
    assert {"capture_not_closed"} == {
        error["code"] for error in report["errors"]
    }


def test_report_writer_is_canonical_immutable_and_commit_bindable(tmp_path: Path):
    report = _verify(tmp_path, [_event(tmp_path)])
    path = tmp_path / "audit-report.json"
    byte_hash = write_decision_audit_report(path, report)
    assert byte_hash == _sha(path.read_bytes())
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        write_decision_audit_report(path, report)


def _intent_bound_event(tmp_path: Path) -> tuple[dict, Path]:
    event = _event(tmp_path)
    context = json.loads(event["context"])
    contract = {
        "stage": "staged_synthesis:sft_realization",
        "model": "generic-model", "max_tokens": 800, "temperature": 0.4,
        "response_schema_sha256": "a" * 64,
        "response_dialect": "openai_json_schema_strict",
    }
    contract_sha256 = _sha(json.dumps(
        contract, sort_keys=True, separators=(",", ":"),
    ).encode())
    binding = {
        "intent_run_id": "run", "intent_cell_id": "cell",
        "intent_unit": "unit", "intent_kind": "initial",
        "intent_stage": "staged_synthesis:sft_realization",
        "intent_logical_attempt": 1,
        "intent_request_sha256": "d" * 64,
        "intent_model": "generic-model",
        "intent_contract_sha256": contract_sha256,
    }
    context.update(binding)
    event["context"] = json.dumps(context, sort_keys=True, separators=(",", ":"))
    manifest = tmp_path / "call-intents.jsonl"
    manifest.write_text(json.dumps({
        "intent_version": "ed4all.provider-call-intent.v1",
        "utc": "2026-01-01T00:00:00Z",
        "run_id": "run", "cell_id": "cell", "unit": "unit",
        "kind": "initial", "stage": "staged_synthesis:sft_realization",
        "logical_attempt": 1, "request_sha256": "d" * 64,
        "model": "generic-model", "max_tokens": 800, "temperature": 0.4,
        "response_schema_sha256": "a" * 64,
        "response_dialect": "openai_json_schema_strict",
        "contract_sha256": contract_sha256,
    }) + "\n", encoding="utf-8")
    return event, manifest


def test_intent_manifest_detects_missing_extra_duplicate_and_tampered_capture(
    tmp_path: Path,
):
    event, manifest = _intent_bound_event(tmp_path)
    artifact = tmp_path / "decisions.jsonl"
    empty_hash = _write_artifact(artifact, [])
    missing = verify_decision_audit(
        artifact, expected_artifact_sha256=empty_hash,
        intent_manifest_path=manifest, capture_closed=True,
    )
    assert "provider_identity_count_mismatch" in {
        item["code"] for item in missing["errors"]
    }

    duplicated = _write_artifact(artifact, [event, deepcopy(event)])
    extra = verify_decision_audit(
        artifact, expected_artifact_sha256=duplicated,
        intent_manifest_path=manifest, capture_closed=True,
    )
    assert {"event_id_duplicate", "provider_identity_count_mismatch"} <= {
        item["code"] for item in extra["errors"]
    }

    event["event_id"] = "EVT_distinct"
    context = json.loads(event["context"])
    context["intent_request_sha256"] = "f" * 64
    event["context"] = json.dumps(context, sort_keys=True, separators=(",", ":"))
    tampered_hash = _write_artifact(artifact, [event])
    tampered = verify_decision_audit(
        artifact, expected_artifact_sha256=tampered_hash,
        intent_manifest_path=manifest, capture_closed=True,
    )
    assert "provider_intent_binding_mismatch" in {
        item["code"] for item in tampered["errors"]
    }
