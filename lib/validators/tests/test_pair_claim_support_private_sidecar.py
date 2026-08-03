from __future__ import annotations

import copy
import hashlib
import json

import pytest

from lib.classifiers.nli_classifier import NliScore
from lib.validators.pair.claim_support import PairClaimSupportValidator
from Trainforge.generators.staged.objective_contract import (
    release_content_sha256,
)


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _ExactQuoteNli:
    def __init__(self, quote):
        self.quote = quote
        self.premises = []

    def score_pair(self, *, premise, hypothesis):
        self.premises.append(premise)
        if premise == self.quote:
            return NliScore(entailment=0.99, neutral=0.01, contradiction=0.0)
        return NliScore(entailment=0.01, neutral=0.98, contradiction=0.01)

    def score_batch(self, *, pairs):
        return [
            self.score_pair(premise=premise, hypothesis=hypothesis)
            for premise, hypothesis in pairs
        ]


def _fixture():
    sentence = "Equal coefficients describe the same affine line."
    quote = "Both equations have identical coefficients and are the same line."
    source_block_id = "block-one"
    source_chunk_id = "chunk-one"
    provenance = {
        "claim_realizations": {"claim-one": sentence},
        "claim_evidence": [{
            "claim_id": "claim-one",
            "claim": sentence,
            "evidence_quote_sha256": _text_hash(quote),
            "source_block_id": source_block_id,
        }],
    }
    provenance["provenance_sha256"] = _hash(provenance)
    pair = {
        "chunk_id": source_chunk_id,
        "completion": sentence,
        "provenance": provenance,
        "_objective_execution_candidate": {"private": "detached"},
        "_objective_execution_private_sidecar": {"not": "released"},
    }
    sidecar_core = {
        "contract_version": "ed4all.objective-execution-sidecar.v1",
        "claims": [{
            "claim_id": "claim-one",
            "claim": sentence,
            "source_block_id": source_block_id,
            "source_chunk_ids": [source_chunk_id],
        }],
        "source_bindings": [{
            "source_chunk_id": source_chunk_id,
            "source_sha256": "a" * 64,
        }],
        "private_evidence": [{
            "claim_id": "claim-one",
            "quote": quote,
            "source_span": [0, len(quote)],
        }],
        "release_content_sha256": release_content_sha256(pair),
    }
    sidecar = {**sidecar_core, "sidecar_sha256": _hash(sidecar_core)}
    chunk = {
        "id": source_chunk_id,
        "text": f"Context. {quote} More context.",
    }
    return pair, sidecar, chunk, quote


def test_verified_private_quote_is_first_exact_nli_premise():
    pair, sidecar, chunk, quote = _fixture()
    nli = _ExactQuoteNli(quote)
    status, reason, fields = PairClaimSupportValidator(nli=nli).validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        authorized_private_sidecar=sidecar,
    )
    assert (status, reason) == ("validated", None)
    assert fields["claim_support_rate"] == 1.0
    assert nli.premises[0] == quote
    assert "evidence_quote" not in pair["provenance"]["claim_evidence"][0]
    assert (
        sidecar["release_content_sha256"]
        == release_content_sha256(pair)
    )


def test_detached_objective_handoff_fields_do_not_change_release_hash():
    pair, sidecar, _, _ = _fixture()
    expected = sidecar["release_content_sha256"]
    pair["_objective_execution_candidate"] = {"different": "runtime value"}
    pair["_objective_execution_private_sidecar"] = {"different": "sidecar"}
    pair["pair_objective_execution"] = [{"detached": True}]
    pair["pair_objective_execution_pass_rate"] = 1.0
    assert release_content_sha256(pair) == expected


def test_hashed_private_contract_without_sidecar_fails_closed():
    pair, _, chunk, quote = _fixture()
    nli = _ExactQuoteNli(quote)
    status, reason, fields = PairClaimSupportValidator(nli=nli).validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
    )
    assert (status, reason) == ("rejected", "unsupported_claim")
    assert fields["claim_support_rate"] == 0.0
    assert nli.premises == []


def test_missing_sidecar_fails_closed_even_when_nli_dependency_is_missing():
    pair, _, chunk, _ = _fixture()
    status, reason, fields = PairClaimSupportValidator().validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
    )
    assert (status, reason) == ("rejected", "unsupported_claim")
    assert fields["deps_missing"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "sidecar_hash",
        "release_hash",
        "claim",
        "source_block",
        "source_chunk",
        "quote_hash",
        "quote_substring",
        "provenance",
    ],
)
def test_private_sidecar_tampering_fails_closed(mutation):
    pair, sidecar, chunk, quote = _fixture()
    if mutation == "sidecar_hash":
        sidecar["sidecar_sha256"] = "0" * 64
    elif mutation == "release_hash":
        core = dict(sidecar)
        core.pop("sidecar_sha256")
        core["release_content_sha256"] = "0" * 64
        sidecar = {**core, "sidecar_sha256": _hash(core)}
    elif mutation == "claim":
        core = dict(sidecar)
        core.pop("sidecar_sha256")
        core["claims"][0]["claim"] = "Different claim."
        sidecar = {**core, "sidecar_sha256": _hash(core)}
    elif mutation == "source_block":
        core = dict(sidecar)
        core.pop("sidecar_sha256")
        core["claims"][0]["source_block_id"] = "different-block"
        sidecar = {**core, "sidecar_sha256": _hash(core)}
    elif mutation == "source_chunk":
        core = dict(sidecar)
        core.pop("sidecar_sha256")
        core["claims"][0]["source_chunk_ids"] = ["different-chunk"]
        sidecar = {**core, "sidecar_sha256": _hash(core)}
    elif mutation == "quote_hash":
        pair["provenance"]["claim_evidence"][0][
            "evidence_quote_sha256"
        ] = "0" * 64
        pair["provenance"]["provenance_sha256"] = _hash({
            key: value for key, value in pair["provenance"].items()
            if key != "provenance_sha256"
        })
        core = dict(sidecar)
        core.pop("sidecar_sha256")
        core["release_content_sha256"] = release_content_sha256(pair)
        sidecar = {**core, "sidecar_sha256": _hash(core)}
    elif mutation == "quote_substring":
        chunk["text"] = "The authorized quote is absent from this source."
    elif mutation == "provenance":
        pair["provenance"]["claim_realizations"]["claim-one"] = "Tampered."
    nli = _ExactQuoteNli(quote)
    status, reason, fields = PairClaimSupportValidator(nli=nli).validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        authorized_private_sidecar=sidecar,
    )
    assert (status, reason) == ("rejected", "unsupported_claim")
    assert fields["claim_support_rate"] == 0.0
    assert nli.premises == []


def test_legacy_inline_quote_contract_is_unchanged():
    pair, _, chunk, quote = _fixture()
    evidence = pair["provenance"]["claim_evidence"][0]
    evidence.pop("evidence_quote_sha256")
    evidence["evidence_quote"] = quote
    pair["provenance"]["provenance_sha256"] = _hash({
        key: value for key, value in pair["provenance"].items()
        if key != "provenance_sha256"
    })
    nli = _ExactQuoteNli(quote)
    status, reason, fields = PairClaimSupportValidator(nli=nli).validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
    )
    assert (status, reason) == ("validated", None)
    assert fields["claim_support_rate"] == 1.0
