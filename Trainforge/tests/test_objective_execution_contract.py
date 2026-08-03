from __future__ import annotations

import copy

import pytest

from Trainforge.generators.staged.objective_contract import (
    CANDIDATE_COMPLETION_CAPS,
    DEFAULT_CANDIDATE_COMPLETION_CAP,
    ObjectiveExecutionContractError,
    build_private_sidecar,
    content_sha256,
    derive_objective_requirements,
    load_private_sidecar,
    reconcile_completion_execution,
    validate_candidate_completion,
    write_private_sidecar,
)


def _objective(*, verb: str = "compute"):
    return {
        "id": "objective-example",
        "statement": f"{verb.title()} a justified terminal response.",
        "bloom_level": "apply",
        "bloom_verb": verb,
        "action_object": "the requested response",
        "condition": "given two supported source claims",
        "degree": "to include the first claim and second claim with accuracy",
    }


def _fixture():
    contract = derive_objective_requirements(_objective())
    completion = "Use both supported claims. Therefore, the result is 12."
    proof = {"proof_type": "deterministic", "ledger": "fixture-proof"}
    release = {"prompt": "Compute the response.", "completion": completion}
    sidecar = build_private_sidecar(
        requirement_contract=contract,
        release_pair=release,
        claims=[{"claim_id": "claim-a", "source_chunk_ids": ["chunk-a"]}],
        source_bindings=[
            {"source_chunk_id": "chunk-a", "source_sha256": "a" * 64}
        ],
        steps=[
            {
                "step_id": "step-a",
                "requirement_ids": [
                    item["requirement_id"] for item in contract["requirements"]
                ],
                "claim_ids": ["claim-a"],
            }
        ],
        result={"text": "12", "completion_span": [52, 54]},
        private_evidence=[
            {
                "claim_id": "claim-a",
                "quote": "private source quote",
                "source_span": [0, 20],
            }
        ],
        proofs=[proof],
        fingerprints={
            "prompt": "b" * 64,
            "schema": "c" * 64,
            "validator": "d" * 64,
            "threshold": "e" * 64,
        },
        stage_calls=[
            {
                "call_id": "call-a",
                "finish_reason": "stop",
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "truncated": False,
            }
        ],
        decision_capture_id="decision-a",
    )
    records = []
    for requirement in contract["requirements"]:
        records.append(
            {
                "requirement_id": requirement["requirement_id"],
                "status": "delivered",
                "completion_spans": [[0, 3]],
                "completion_span_text": "Use",
                "proof_sha256": content_sha256(proof),
                "validator_fingerprint": "d" * 64,
            }
        )
    return contract, completion, release, sidecar, records


def test_requirement_derivation_is_stable_and_projects_all_abcd_dimensions():
    first = derive_objective_requirements(_objective())
    second = derive_objective_requirements(copy.deepcopy(_objective()))
    assert first == second
    assert [item["kind"] for item in first["requirements"]] == [
        "action",
        "condition",
        "content_obligation",
        "content_obligation",
        "content_obligation",
        "performance_criterion",
        "result",
    ]
    assert all(item["requirement_id"].startswith("req-") for item in first["requirements"])
    assert all(len(item["requirement_sha256"]) == 64 for item in first["requirements"])
    assert first["cognitive_task_type"] == "compute"
    assert first["result_required"] is True


def test_requirement_reordering_and_value_change_change_contract_hash():
    original = _objective()
    changed = _objective()
    changed["degree"] = "to include the second claim and first claim with accuracy"
    assert (
        derive_objective_requirements(original)["objective_contract_sha256"]
        != derive_objective_requirements(changed)["objective_contract_sha256"]
    )


def test_result_is_derived_from_action_semantics_not_subject_allowlist():
    assert derive_objective_requirements(_objective(verb="classify"))[
        "result_required"
    ]
    assert derive_objective_requirements(_objective(verb="explain"))[
        "result_required"
    ]
    declarative = derive_objective_requirements(_objective(verb="define"))
    assert declarative["cognitive_task_type"] == "define"
    assert declarative["result_required"] is False
    assert not any(
        item["kind"] == "result" for item in declarative["requirements"]
    )


def test_capacity_abstains_before_any_provider_is_needed():
    with pytest.raises(
        ObjectiveExecutionContractError,
        match="exceeds configured capacity",
    ):
        derive_objective_requirements(_objective(), max_requirements=2)


@pytest.mark.parametrize("cap", CANDIDATE_COMPLETION_CAPS)
def test_each_candidate_cap_accepts_an_untruncated_short_completion(cap):
    telemetry = validate_candidate_completion("complete", cap=cap)
    assert telemetry["selected_cap"] == cap
    assert telemetry["truncated"] is False


def test_micro_contract_completion_cap_defaults_to_600_and_never_truncates():
    # PACKAGE A: This test pins the MICRO contract completion cap, which is
    # deliberately NOT changing. The general SFT answer caps in
    # staged/provider.py are being raised from 600→1200 chars, but the
    # MICRO contract's objective-execution completion bounds remain exactly 600
    # (a separate, uncapped MICRO path for objective realization).
    assert DEFAULT_CANDIDATE_COMPLETION_CAP == 600
    assert validate_candidate_completion("x" * 599)["completion_characters"] == 599
    for kwargs, message in (
        ({"completion": "x" * 600}, "reached or exceeded"),
        ({"completion": "x", "finish_reason": "length"}, "length finish"),
        ({"completion": "x", "truncated": True}, "was truncated"),
        ({"completion": "x", "cap": 700}, "must be one of"),
    ):
        with pytest.raises(ObjectiveExecutionContractError, match=message):
            validate_candidate_completion(**kwargs)


def test_sidecar_is_hash_bound_and_keeps_private_evidence_off_release_pair():
    contract, _, release, sidecar, _ = _fixture()
    sidecar_core = dict(sidecar)
    digest = sidecar_core.pop("sidecar_sha256")
    assert digest == content_sha256(sidecar_core)
    assert "private_evidence" in sidecar
    assert "private_evidence" not in release
    assert sidecar["objective_contract_sha256"] == contract[
        "objective_contract_sha256"
    ]
    assert sidecar["release_content_sha256"] == content_sha256(release)


def test_release_payload_hash_excludes_only_detached_objective_audit_link():
    _, _, release, sidecar, _ = _fixture()
    linked = {
        **release,
        "pair_objective_execution": [{
            "sidecar_sha256": sidecar["sidecar_sha256"],
        }],
        "pair_objective_execution_pass_rate": 1.0,
    }
    assert sidecar["release_content_sha256"] == content_sha256(release)
    from Trainforge.generators.staged.objective_contract import (
        release_content_sha256,
    )
    assert release_content_sha256(linked) == release_content_sha256(release)
    assert release_content_sha256({
        **linked, "completion": f"{release['completion']} changed",
    }) != release_content_sha256(release)


def test_private_sidecar_persistence_is_content_addressed_and_verified(tmp_path):
    _, _, _, sidecar, _ = _fixture()
    path = write_private_sidecar(tmp_path, sidecar)
    assert path.name == f"{sidecar['sidecar_sha256']}.json"
    assert write_private_sidecar(tmp_path, sidecar) == path
    assert load_private_sidecar(
        path, expected_sha256=sidecar["sidecar_sha256"],
    ) == sidecar
    with pytest.raises(
        ObjectiveExecutionContractError, match="hash mismatch",
    ):
        load_private_sidecar(path, expected_sha256="0" * 64)


def test_sidecar_rejects_length_finish_truncation_and_invalid_token_counts():
    contract = derive_objective_requirements(_objective())
    base = {
        "requirement_contract": contract,
        "release_pair": {"completion": "answer"},
        "claims": [],
        "source_bindings": [],
        "steps": [],
        "result": {"text": "answer"},
        "private_evidence": [],
        "proofs": [],
        "fingerprints": {"validator": "a" * 64},
        "decision_capture_id": "decision-a",
    }
    for call, message in (
        (
            {
                "finish_reason": "length",
                "truncated": False,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
            "length finish",
        ),
        (
            {
                "finish_reason": "stop",
                "truncated": True,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
            "was truncated",
        ),
        (
            {
                "finish_reason": "stop",
                "truncated": False,
                "prompt_tokens": -1,
                "completion_tokens": 1,
            },
            "non-negative integer",
        ),
    ):
        with pytest.raises(ObjectiveExecutionContractError, match=message):
            build_private_sidecar(stage_calls=[call], **base)


def test_reconciliation_accepts_only_complete_completion_coverage():
    contract, completion, release, sidecar, records = _fixture()
    audit = reconcile_completion_execution(
        completion=completion,
        requirement_contract=contract,
        execution_records=records,
        sidecar=sidecar,
        release_pair=release,
        validator_fingerprint="d" * 64,
    )
    assert audit["completion_only"] is True
    assert audit["requirement_pass_rate"] == 1.0
    assert audit["result_delivered"] is True


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing_requirement", "do not reconcile"),
        ("prompt_offset", "out of bounds"),
        ("altered_text", "span text mismatch"),
        ("proof", "proof hash mismatch"),
        ("validator", "validator fingerprint mismatch"),
        ("sidecar", "sidecar hash mismatch"),
        ("release", "release content hash mismatch"),
        ("completion_binding", "exactly match release"),
        ("result", "sidecar required result is missing"),
    ],
)
def test_reconciliation_fails_closed_on_tampering(mutation, message):
    contract, completion, release, sidecar, records = _fixture()
    if mutation == "missing_requirement":
        records.pop()
    elif mutation == "prompt_offset":
        records[0]["completion_spans"] = [[-1, 2]]
    elif mutation == "altered_text":
        records[0]["completion_span_text"] = "Not"
    elif mutation == "proof":
        records[0]["proof_sha256"] = "0" * 64
    elif mutation == "validator":
        records[0]["validator_fingerprint"] = "0" * 64
    elif mutation == "sidecar":
        sidecar["steps"][0]["step_id"] = "tampered"
    elif mutation == "release":
        release["completion"] += " changed"
    elif mutation == "completion_binding":
        completion = "Use"
    elif mutation == "result":
        sidecar_core = dict(sidecar)
        sidecar_core.pop("sidecar_sha256")
        sidecar_core["result"] = None
        sidecar = {
            **sidecar_core,
            "sidecar_sha256": content_sha256(sidecar_core),
        }
    with pytest.raises(ObjectiveExecutionContractError, match=message):
        reconcile_completion_execution(
            completion=completion,
            requirement_contract=contract,
            execution_records=records,
            sidecar=sidecar,
            release_pair=release,
            validator_fingerprint="d" * 64,
        )
