import hashlib
import json

import pytest

from Trainforge.scripts.harness.staged_window_abcd_pilot import (
    CELL_PUBLICATION_SCHEMA,
    _digest,
    _row_invariant_errors,
    _stable,
    summarize,
    verify_cell_publication,
)


def _row(**changes):
    row = {
        "chunk_id": "generic-unit",
        "kind": "instruction",
        "variant": "D_production_contract",
        "repetition": 0,
        "accepted": True,
        "stage_validity": True,
        "truncated": False,
        "leakage_passed": True,
        "error_code": None,
        "error_details": None,
        "scoring_error": None,
        "scores": {"accepted": True},
        "pair_sha256": "a" * 64,
    }
    row.update(changes)
    return row


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"truncated": True}, "truncated_row"),
        ({"leakage_passed": False}, "leakage_not_passed"),
        ({"leakage_passed": None}, "leakage_passed_not_boolean"),
        ({"accepted": 1}, "accepted_not_boolean"),
        ({"stage_validity": None}, "stage_validity_not_boolean"),
        ({"pair_sha256": None}, "accepted_state_inconsistent"),
        ({"error_code": "transport_error"}, "accepted_state_inconsistent"),
        ({"scoring_error": "bad score"}, "accepted_state_inconsistent"),
        (
            {"error_details": {"terminal_content_rejection": True}},
            "accepted_state_inconsistent",
        ),
        (
            {
                "accepted": False,
                "stage_validity": True,
                "scores": {"accepted": False},
            },
            "failure_state_inconsistent",
        ),
    ],
)
def test_row_invariants_fail_closed(changes, reason):
    assert reason in _row_invariant_errors(_row(**changes))


def _committed_cell(tmp_path):
    model = "generic-model"
    run_id = "generic-run-c16"
    preflight = {
        "pilot_run_id": run_id,
        "cell_id": "cell-generic",
        "max_tokens": 800,
        "temperature": 0.4,
        "telemetry_identity": {"expected_model": model},
        "intent_manifest_preflight_sha256": hashlib.sha256(b"").hexdigest(),
        "intent_manifest_preflight_count": 0,
    }
    preflight_bytes = json.dumps(preflight).encode()
    (tmp_path / "preflight.json").write_bytes(preflight_bytes)
    preflight_hash = hashlib.sha256(preflight_bytes).hexdigest()
    rows = [_row()]
    result_bytes = "".join(f"{_stable(row)}\n" for row in rows).encode()
    (tmp_path / "results.jsonl").write_bytes(result_bytes)
    (tmp_path / "summary.json").write_text(
        json.dumps(summarize(rows), sort_keys=True), encoding="utf-8",
    )
    reconciliation = {
        "status": "accepted", "started": 1, "terminal": 1, "errors": [],
    }
    (tmp_path / "http-reconciliation.json").write_text(
        json.dumps(reconciliation), encoding="utf-8",
    )
    audit_core = {
        "status": "accepted", "capture_closed": True,
        "provider_events_expected": 1, "provider_events_observed": 1,
        "provider_identities_verified": 1, "errors": [],
    }
    audit_report = {
        **audit_core,
        "report_sha256": hashlib.sha256(
            (json.dumps(
                audit_core, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ) + "\n").encode()
        ).hexdigest(),
    }
    audit_bytes = (
        json.dumps(audit_report, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False) + "\n"
    ).encode()
    (tmp_path / "decision-audit-report.json").write_bytes(audit_bytes)
    audit_hash = hashlib.sha256(audit_bytes).hexdigest()
    verifier_hashes = {}
    for name, schema in (
        ("telemetry-verification.json",
         "ed4all.benchmark-telemetry-verification.v1"),
        ("http-reconciliation-verification.json",
         "ed4all.http-reconciliation-verification.v1"),
    ):
        core = {"schema_version": schema, "status": "verified", "errors": []}
        if name == "http-reconciliation-verification.json":
            core["intent_identity"] = {
                "run_ids": [], "cell_ids": [], "models": [],
                "contract_sha256s": [],
            }
        report = {
            **core,
            "report_sha256": hashlib.sha256(
                (json.dumps(
                    core, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ) + "\n").encode()
            ).hexdigest(),
        }
        report_bytes = (
            json.dumps(report, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n"
        ).encode()
        (tmp_path / name).write_bytes(report_bytes)
        verifier_hashes[name] = hashlib.sha256(report_bytes).hexdigest()
    (tmp_path / "call-intents.jsonl").write_bytes(b"")
    names = (
        "call-intents.jsonl",
        "decision-audit-report.json",
        "http-reconciliation-verification.json",
        "http-reconciliation.json", "results.jsonl", "summary.json",
        "telemetry-verification.json",
    )
    artifacts = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in names
    }
    commit_body = {
        "schema_version": CELL_PUBLICATION_SCHEMA,
        "status": "committed",
        "artifacts": artifacts,
        "model_id": model,
        "pilot_run_id": run_id,
        "cell_id": "cell-generic",
        "preflight_sha256": preflight_hash,
        "decision_audit_report_sha256": audit_hash,
        "telemetry_report_sha256": verifier_hashes[
            "telemetry-verification.json"
        ],
        "reconciliation_report_sha256": verifier_hashes[
            "http-reconciliation-verification.json"
        ],
        "intent_manifest_sha256": hashlib.sha256(b"").hexdigest(),
        "intent_count": 0,
    }
    commit = {**commit_body, "commit_sha256": _digest(commit_body)}
    (tmp_path / "success-commit.json").write_text(
        json.dumps(commit), encoding="utf-8",
    )
    return model, run_id, preflight_hash, rows


def test_publication_verifier_recomputes_complete_trust_chain(tmp_path):
    model, run_id, preflight_hash, rows = _committed_cell(tmp_path)
    verified = verify_cell_publication(
        tmp_path,
        expected_model=model,
        expected_run_id=run_id,
        expected_preflight_sha256=preflight_hash,
        classified_rows=rows,
    )
    assert verified["status"] == "verified"


def _staged_candidate(tmp_path):
    model, run_id, preflight_hash, rows = _committed_cell(tmp_path)
    authority = tmp_path / "success-commit.json"
    commit = json.loads(authority.read_text(encoding="utf-8"))
    authority.unlink()
    return model, run_id, preflight_hash, rows, commit


def _rewrite_candidate_rows(tmp_path, commit, rows):
    (tmp_path / "results.jsonl").write_bytes(
        "".join(f"{_stable(row)}\n" for row in rows).encode()
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(summarize(rows), sort_keys=True), encoding="utf-8",
    )
    for name in ("results.jsonl", "summary.json"):
        commit["artifacts"][name] = hashlib.sha256(
            (tmp_path / name).read_bytes()
        ).hexdigest()
    body = {key: value for key, value in commit.items() if key != "commit_sha256"}
    commit["commit_sha256"] = _digest(body)


@pytest.mark.parametrize(
    "row",
    [
        _row(leakage_passed=False),
        _row(
            accepted=False, stage_validity=False, pair_sha256=None,
            error_code="objective_delivery:unsupported",
            scores={"accepted": False},
        ),
    ],
)
def test_staged_candidate_rejects_leakage_and_unclassified_codes(tmp_path, row):
    model, run_id, preflight_hash, _rows, commit = _staged_candidate(tmp_path)
    _rewrite_candidate_rows(tmp_path, commit, [row])
    with pytest.raises(ValueError):
        verify_cell_publication(
            tmp_path, expected_model=model, expected_run_id=run_id,
            expected_preflight_sha256=preflight_hash,
            classified_rows=[row], candidate_commit=commit,
            artifact_dir=tmp_path,
        )


def test_publication_rejects_coordinated_rehashed_manifest_identity_mismatch(
    tmp_path,
):
    model, run_id, preflight_hash, rows = _committed_cell(tmp_path)
    contract = {
        "stage": "staged_synthesis:dialect_preflight",
        "model": "foreign-model", "max_tokens": 800, "temperature": 0.4,
        "response_schema_sha256": "a" * 64,
        "response_dialect": "openai_json_schema_strict",
    }
    intent = {
        "intent_version": "ed4all.provider-call-intent.v1",
        "utc": "2026-01-01T00:00:00Z", "run_id": "foreign-run",
        "cell_id": "foreign-cell", "unit": "dialect-preflight",
        "kind": "dialect", "stage": contract["stage"], "logical_attempt": 1,
        "request_sha256": "b" * 64, "model": contract["model"],
        "max_tokens": 800, "temperature": 0.4,
        "response_schema_sha256": "a" * 64,
        "response_dialect": "openai_json_schema_strict",
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    intent_bytes = (json.dumps(intent, sort_keys=True) + "\n").encode()
    (tmp_path / "call-intents.jsonl").write_bytes(intent_bytes)
    report_path = tmp_path / "http-reconciliation-verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    core["intent_identity"] = {
        "run_ids": ["foreign-run"], "cell_ids": ["foreign-cell"],
        "models": ["foreign-model"],
        "contract_sha256s": [intent["contract_sha256"]],
    }
    report = {**core, "report_sha256": hashlib.sha256(
        (json.dumps(core, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()}
    report_bytes = (
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    report_path.write_bytes(report_bytes)
    commit_path = tmp_path / "success-commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    for name, payload in (
        ("call-intents.jsonl", intent_bytes),
        (report_path.name, report_bytes),
    ):
        commit["artifacts"][name] = hashlib.sha256(payload).hexdigest()
    commit["intent_manifest_sha256"] = commit["artifacts"]["call-intents.jsonl"]
    commit["intent_count"] = 1
    commit["reconciliation_report_sha256"] = commit["artifacts"][report_path.name]
    body = {key: value for key, value in commit.items() if key != "commit_sha256"}
    commit["commit_sha256"] = _digest(body)
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic identity"):
        verify_cell_publication(
            tmp_path, expected_model=model, expected_run_id=run_id,
            expected_preflight_sha256=preflight_hash, classified_rows=rows,
        )


@pytest.mark.parametrize(
    "attack",
    [
        "result_bytes", "summary", "audit_report", "commit_digest",
        "classified_state", "dual",
    ],
)
def test_publication_verifier_rejects_adversarial_mutation(tmp_path, attack):
    model, run_id, preflight_hash, rows = _committed_cell(tmp_path)
    classified = rows
    if attack == "result_bytes":
        path = tmp_path / "results.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
    elif attack == "summary":
        (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    elif attack == "audit_report":
        path = tmp_path / "decision-audit-report.json"
        report = json.loads(path.read_text())
        report["status"] = "rejected"
        path.write_text(json.dumps(report), encoding="utf-8")
    elif attack == "commit_digest":
        path = tmp_path / "success-commit.json"
        commit = json.loads(path.read_text())
        commit["commit_sha256"] = "0" * 64
        path.write_text(json.dumps(commit), encoding="utf-8")
    elif attack == "classified_state":
        classified = [{**rows[0], "accepted": False}]
    else:
        (tmp_path / "finalization_in_progress.json").write_text(
            '{"status":"committed"}', encoding="utf-8",
        )
    with pytest.raises(ValueError):
        verify_cell_publication(
            tmp_path,
            expected_model=model,
            expected_run_id=run_id,
            expected_preflight_sha256=preflight_hash,
            classified_rows=classified,
        )


def test_publication_verifier_rejects_missing_authority(tmp_path):
    with pytest.raises(ValueError):
        verify_cell_publication(
            tmp_path,
            expected_model="generic-model",
            expected_run_id="generic-run",
            expected_preflight_sha256="0" * 64,
            classified_rows=[],
        )
