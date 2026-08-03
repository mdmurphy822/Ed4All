from __future__ import annotations

import hashlib
import json

from Trainforge.synthesis.verification.benchmark_artifact_verifier import (
    RECONCILIATION_REPORT_SCHEMA,
    TELEMETRY_REPORT_SCHEMA,
    verified_report,
    verify_http_reconciliation,
    verify_telemetry_artifacts,
    read_call_intents,
)
from Trainforge.generators.http_attempt_ledger import DurableCallIntentManifest

ITERATION = (
    "iter=1 num_scheduled_requests=2 "
    "num_ctx_tokens=100 num_generation_tokens=20"
)


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _telemetry(tmp_path):
    root = tmp_path / "telemetry"
    root.mkdir()
    (root / "trtllm.log").write_text(ITERATION + "\n", encoding="utf-8")
    (root / "system.jsonl").write_text(
        json.dumps({"utc": "2026-01-01T00:00:00Z", "gpu": []}) + "\n",
        encoding="utf-8",
    )
    preflight = {
        "status": "accepted", "container": "runtime",
        "published_port": 8000, "expected_model": "model",
        "max_num_tokens": 8192, "parser_probe": {"iteration_samples": 1},
    }
    state = {
        "started": True, "stop_requested": True,
        "process_present": True, "process_poll": 0,
        "process_stopped": True, "terminate_requested": True,
        "kill_requested": False, "threads_total": 2,
        "threads_alive": 0, "threads_stopped": True, "errors": [],
    }
    report = verify_telemetry_artifacts(
        root, preflight=preflight,
        expected_preflight_sha256=hashlib.sha256(_canonical(preflight)).hexdigest(),
        expected_container="runtime", expected_port=8000,
        expected_model="model", expected_max_num_tokens=8192,
        expected_interval_seconds=2.0, sampler_state=state,
    )
    return root, preflight, state, report


def test_telemetry_report_binds_artifacts_identity_samples_and_stop(tmp_path):
    root, _preflight, _state, report = _telemetry(tmp_path)
    assert verified_report(report, TELEMETRY_REPORT_SCHEMA)
    assert report["parser"]["iteration_samples"] == 1
    assert report["parser"]["system_samples"] == 1
    assert report["artifacts"]["trtllm.log"]["sha256"] == hashlib.sha256(
        (root / "trtllm.log").read_bytes()
    ).hexdigest()
    assert report["sampler_state_sha256"] == hashlib.sha256(
        _canonical(report["sampler"])
    ).hexdigest()


def test_telemetry_rejects_artifact_identity_sample_and_stop_tampering(tmp_path):
    root, preflight, state, report = _telemetry(tmp_path)
    report["parser"]["iteration_samples"] = 99
    assert not verified_report(report, TELEMETRY_REPORT_SCHEMA)
    (root / "trtllm.log").write_text("idle\n", encoding="utf-8")
    rejected = verify_telemetry_artifacts(
        root, preflight=preflight,
        expected_preflight_sha256=hashlib.sha256(_canonical(preflight)).hexdigest(),
        expected_container="wrong", expected_port=8000, expected_model="model",
        expected_max_num_tokens=8192, expected_interval_seconds=2.0,
        sampler_state={**state, "process_stopped": False},
    )
    codes = {item["code"] for item in rejected["errors"]}
    assert {"preflight_identity_mismatch", "iteration_samples_empty",
            "sampler_stop_not_clean"} <= codes
    assert not verified_report(rejected, TELEMETRY_REPORT_SCHEMA)


def test_telemetry_rejects_unproven_or_inconsistent_lifecycle_state(tmp_path):
    root, preflight, state, _report = _telemetry(tmp_path)
    adversarial_states = [
        {**state, "process_poll": None},
        {**state, "threads_alive": 1},
        {**state, "threads_total": 0},
        {**state, "errors": ["capture_thread_failed"]},
        {key: value for key, value in state.items() if key != "process_poll"},
        {**state, "process_stopped": 1},
        {**state, "threads_alive": False},
        {**state, "kill_requested": True, "terminate_requested": False},
    ]
    for dirty_state in adversarial_states:
        report = verify_telemetry_artifacts(
            root, preflight=preflight,
            expected_preflight_sha256=hashlib.sha256(
                _canonical(preflight)
            ).hexdigest(),
            expected_container="runtime", expected_port=8000,
            expected_model="model", expected_max_num_tokens=8192,
            expected_interval_seconds=2.0, sampler_state=dirty_state,
        )
        assert not verified_report(report, TELEMETRY_REPORT_SCHEMA), dirty_state
        assert {"code": "sampler_stop_not_clean"} in report["errors"]


def test_bound_micro_intent_contract_rejects_fingerprint_tampering(tmp_path):
    path = tmp_path / "intents.jsonl"
    fingerprint = "a" * 64
    manifest = DurableCallIntentManifest(
        path, run_id="run", cell_id="cell",
        synthesis_contract_sha256=fingerprint,
    )
    with manifest.unit():
        manifest.admit(
            unit="chunk:instruction", stage="staged_synthesis:micro_B_claim_0_attempt_1",
            payload={"model": "served", "max_tokens": 64, "temperature": 0,
                     "response_format": {"json_schema": {"schema": {}}}},
            response_dialect="openai_json_schema_strict",
        )
    rows, errors = read_call_intents(path)
    assert not errors
    assert rows[0]["synthesis_contract_sha256"] == fingerprint
    assert rows[0]["intent_version"] == "ed4all.provider-call-intent.v2"

    tampered = dict(rows[0])
    tampered["synthesis_contract_sha256"] = "b" * 64
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    _rows, errors = read_call_intents(path)
    assert {"code": "intent_contract_identity_mismatch", "line": 1} in errors


def _audit(tmp_path):
    request = tmp_path / "request.raw"
    response = tmp_path / "response.raw"
    request.write_bytes(b"request")
    response.write_bytes(b"response")
    ident = {
        "unit": "u1", "stage": "plan", "attempt": 1,
        "request_sha256": hashlib.sha256(b"logical request").hexdigest(),
    }
    def ref(path):
        raw = path.read_bytes()
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw)}
    rows = [
        {**ident, "event": "http_attempt_started", "request_raw_ref": ref(request)},
        {**ident, "event": "http_attempt_terminal", "state": "completed",
         "response_raw_ref": ref(response)},
    ]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return ledger, ident, request, response


def test_reconciliation_report_is_commit_bindable_and_bijective(tmp_path):
    ledger, ident, _request, _response = _audit(tmp_path)
    report = verify_http_reconciliation(ledger, expected_identities=[ident])
    assert verified_report(report, RECONCILIATION_REPORT_SCHEMA)
    assert report["expected_attempts"] == report["started_attempts"] == (
        report["terminal_attempts"]
    ) == 1


def test_reconciliation_rejects_raw_identity_terminal_and_report_tampering(tmp_path):
    ledger, ident, _request, response = _audit(tmp_path)
    response.write_bytes(b"tampered")
    report = verify_http_reconciliation(
        ledger, expected_identities=[{**ident, "unit": "different"}],
    )
    codes = {item["code"] for item in report["errors"]}
    assert {"started_identity_mismatch", "terminal_identity_mismatch",
            "response_raw_ref_mismatch"} <= codes
    assert not verified_report(report, RECONCILIATION_REPORT_SCHEMA)
    report["status"] = "verified"
    assert not verified_report(report, RECONCILIATION_REPORT_SCHEMA)


def test_reconciliation_rejects_duplicate_and_unclean_terminal_state(tmp_path):
    ledger, ident, _request, _response = _audit(tmp_path)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[1]["state"] = "running"
    rows.append(rows[0])
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = verify_http_reconciliation(ledger, expected_identities=[ident])
    codes = {item["code"] for item in report["errors"]}
    assert "attempt_not_bijective" in codes
    assert "started_identity_mismatch" in codes


def _intent_manifest(tmp_path, ident, *, duplicate=False):
    path = tmp_path / "call-intents.jsonl"
    contract = {
        "stage": ident["stage"], "model": "generic-model",
        "max_tokens": 800, "temperature": 0.4,
        "response_schema_sha256": "b" * 64,
        "response_dialect": "openai_json_schema_strict",
    }
    row = {
        "intent_version": "ed4all.provider-call-intent.v1",
        "utc": "2026-01-01T00:00:00Z",
        "run_id": "run", "cell_id": "cell",
        "unit": ident["unit"], "kind": "initial", "stage": ident["stage"],
        "logical_attempt": ident["attempt"],
        "request_sha256": ident["request_sha256"],
        "model": "generic-model", "max_tokens": 800, "temperature": 0.4,
        "response_schema_sha256": "b" * 64,
        "response_dialect": "openai_json_schema_strict",
        "contract_sha256": hashlib.sha256(
            json.dumps(
                contract, sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    path.write_text(
        "".join(json.dumps(row) + "\n" for _ in range(2 if duplicate else 1)),
        encoding="utf-8",
    )
    return path


def test_manifest_authority_detects_wholly_missing_and_extra_http_pairs(tmp_path):
    ledger, ident, _request, _response = _audit(tmp_path)
    manifest = _intent_manifest(tmp_path, ident)
    assert verified_report(
        verify_http_reconciliation(ledger, intent_manifest_path=manifest),
        RECONCILIATION_REPORT_SCHEMA,
    )
    ledger.write_text("", encoding="utf-8")
    missing = verify_http_reconciliation(ledger, intent_manifest_path=manifest)
    assert {"started_identity_mismatch", "terminal_identity_mismatch"} <= {
        item["code"] for item in missing["errors"]
    }
    ledger2, ident2, _request2, _response2 = _audit(tmp_path)
    rows = ledger2.read_text(encoding="utf-8").splitlines()
    ledger2.write_text("\n".join(rows + rows) + "\n", encoding="utf-8")
    extra = verify_http_reconciliation(ledger2, intent_manifest_path=manifest)
    assert {"started_identity_mismatch", "terminal_identity_mismatch",
            "attempt_not_bijective"} <= {
        item["code"] for item in extra["errors"]
    }


def test_manifest_duplicate_and_tamper_cannot_redefine_expected_http_calls(tmp_path):
    ledger, ident, _request, _response = _audit(tmp_path)
    duplicate = _intent_manifest(tmp_path, ident, duplicate=True)
    report = verify_http_reconciliation(ledger, intent_manifest_path=duplicate)
    assert "intent_identity_duplicate" in {
        item["code"] for item in report["errors"]
    }
    manifest = _intent_manifest(tmp_path, ident)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["request_sha256"] = "d" * 64
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    tampered = verify_http_reconciliation(ledger, intent_manifest_path=manifest)
    assert {"started_identity_mismatch", "terminal_identity_mismatch"} <= {
        item["code"] for item in tampered["errors"]
    }
