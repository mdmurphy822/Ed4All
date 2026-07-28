from __future__ import annotations

import copy
import hashlib

from Trainforge.generators.objective_execution_contract import (
    build_private_sidecar,
    content_sha256,
    derive_objective_requirements,
    reconcile_completion_execution,
)
from lib.validators.pair.promotion import (
    _objective_execution_answer_support_proof,
    _replay_objective_execution_public_authority,
)


def _authority_fixture():
    contract = derive_objective_requirements({
        "id": "objective-generic",
        "statement": "Explain the supported relationship.",
        "bloom_level": "understand",
        "bloom_verb": "explain",
        "action_object": "the supported relationship",
    })
    requirements = contract["requirements"]
    texts = [
        (
            "Therefore the supported relationship holds."
            if item["kind"] == "result"
            else f"Execute requirement {index + 1} from the supported relationship."
        )
        for index, item in enumerate(requirements)
    ]
    completion = "\n\n".join(texts)
    steps = []
    records = []
    proofs = []
    cursor = 0
    for index, requirement in enumerate(requirements):
        text = texts[index]
        if index:
            cursor += 2
        span = [cursor, cursor + len(text)]
        cursor = span[1]
        proof = {
            "proof_type": "hybrid",
            "requirement_id": requirement["requirement_id"],
            "requirement_sha256": requirement["requirement_sha256"],
            "realization_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "claim_ids": ["claim-one"],
            "completion_span": span,
            "validator_fingerprint": "v" * 64,
            "scores": {
                "requirement_entailment": 1.0,
                "requirement_contradiction": 0.0,
                "claim_support_entailment": 1.0,
                "claim_support_contradiction": 0.0,
            },
        }
        proofs.append(proof)
        steps.append({
            "step_id": f"step-{index + 1:02d}",
            "requirement_ids": [requirement["requirement_id"]],
            "claim_ids": ["claim-one"],
            "realization": text,
            "completion_span": span,
        })
        records.append({
            "requirement_id": requirement["requirement_id"],
            "status": "delivered",
            "completion_spans": [span],
            "completion_span_text": text,
            "proof_sha256": content_sha256(proof),
            "validator_fingerprint": "v" * 64,
        })
    pair = {
        "prompt": "Explain the relationship.",
        "completion": completion,
        "source_chunk_id": "chunk-one",
    }
    final_assembly = {
        "contract_version": "ed4all.objective-execution-final-assembly.v1",
        "ordered_step_ids": [item["step_id"] for item in steps],
        "separator": "\n\n",
        "item_sha256": [
            hashlib.sha256(item["realization"].encode()).hexdigest()
            for item in steps
        ],
        "completion_spans": [item["completion_span"] for item in steps],
        "assembled_sha256": hashlib.sha256(completion.encode()).hexdigest(),
    }
    sidecar = build_private_sidecar(
        requirement_contract=contract,
        release_pair=pair,
        claims=[{
            "claim_id": "claim-one",
            "claim": "The relationship is supported.",
            "source_chunk_ids": ["chunk-one"],
            "source_block_id": "block-one",
        }],
        source_bindings=[{
            "source_chunk_id": "chunk-one",
            "source_sha256": "s" * 64,
        }],
        steps=steps,
        result=steps[-1],
        private_evidence=[{
            "claim_id": "claim-one",
            "quote": "The relationship is supported.",
            "quote_sha256": hashlib.sha256(
                b"The relationship is supported."
            ).hexdigest(),
            "source_block_id": "block-one",
            "source_span": [0, 30],
        }],
        proofs=proofs,
        fingerprints={"validator": "v" * 64},
        stage_calls=[{
            "call_id": "decision-one",
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "truncated": False,
        }],
        decision_capture_id="decision-one",
        final_assembly=final_assembly,
    )
    audit = reconcile_completion_execution(
        completion=completion,
        requirement_contract=contract,
        execution_records=records,
        sidecar=sidecar,
        release_pair=pair,
        validator_fingerprint="v" * 64,
    )
    pair["_objective_execution_candidate"] = {
        "pair_objective_execution": [audit],
        "pair_objective_execution_pass_rate": 1.0,
        "execution_records": records,
        "claim_support_rate": 1.0,
        "claim_contradicted_rate": 0.0,
    }
    return pair, sidecar


def test_private_objective_authority_and_public_replay_are_exact():
    pair, sidecar = _authority_fixture()
    authority = _objective_execution_answer_support_proof(
        pair,
        answer=pair["completion"],
        source_chunk_id="chunk-one",
        authorized_private_sidecar=sidecar,
    )
    assert authority is not None
    pair["_objective_execution_candidate"][
        "answer_support_authority"
    ] = authority
    assert _replay_objective_execution_public_authority(
        pair, answer=pair["completion"], source_chunk_id="chunk-one",
    ) == authority


def test_objective_authority_fails_closed_on_answer_sidecar_or_order_tamper():
    pair, sidecar = _authority_fixture()
    for mutation in ("answer", "sidecar", "order", "cross_pair"):
        candidate = copy.deepcopy(pair)
        private = copy.deepcopy(sidecar)
        if mutation == "answer":
            candidate["completion"] += " altered"
        elif mutation == "sidecar":
            private["steps"][0]["realization"] = "altered"
        elif mutation == "order":
            private["steps"].reverse()
            core = dict(private)
            core.pop("sidecar_sha256")
            private["sidecar_sha256"] = content_sha256(core)
        else:
            candidate["source_chunk_id"] = "chunk-two"
        assert _objective_execution_answer_support_proof(
            candidate,
            answer=candidate["completion"],
            source_chunk_id=candidate["source_chunk_id"],
            authorized_private_sidecar=private,
        ) is None
