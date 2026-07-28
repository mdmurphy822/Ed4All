from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from Trainforge.generators._openai_compatible_client import OpenAICompatibleClient
from Trainforge.generators._synthesis_common import SynthesisProviderError
from Trainforge.generators.staged_synthesis_provider import (
    StagedSynthesisProvider,
    _canonical_evidence_quote,
    _coverage_units,
    _claim_repair_diff,
    _claim_repair_guard,
    _deterministic_entailment_basis,
    _equation_signatures,
    _first_distinctive_verbatim_span,
    _first_verbatim_span,
    _is_standalone_discourse_marker,
    _operational_condition_error,
    _relation_preserving_support,
    staged_synthesis_v4_enabled,
)
from Trainforge.generators.synthesis_window_contract import (
    WINDOW_CONTRACT_VERSION,
    build_evidence_window,
)
from Trainforge.generators.instruction_factory import synthesize_instruction_pair
from Trainforge.generators.preference_factory import synthesize_preference_pair


class _Client:
    @staticmethod
    def _extract_json_lenient(raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


class _LengthClient(_Client):
    def post_with_usage(self, payload, *, task):
        self.meter_task = task
        return self._post_with_retry(payload)

    def _post_with_retry(self, payload):
        self.payload = payload
        return {
            "choices": [{
                "message": {"content": '{"objective_id":'},
                "finish_reason": "length",
            }],
            "usage": {},
        }, 0

    @staticmethod
    def _extract_text(body):
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_usage(body):
        return body["usage"]


@pytest.mark.parametrize("name,value", [
    ("max_stage_repairs", -1),
    ("max_stage_repairs", True),
    ("max_leakage_repairs", -1),
])
def test_stage_repair_budgets_fail_closed_before_provider_call(name, value):
    provider = object.__new__(StagedSynthesisProvider)
    kwargs = {
        "stage": "synthetic",
        "chunk_id": "generic-chunk",
        "system": "system",
        "user": "user",
        "required_keys": (),
        "validator": lambda value: None,
        name: value,
    }
    with pytest.raises(SynthesisProviderError) as caught:
        provider._call_stage(**kwargs)
    assert caught.value.code == "staged_repair_budget_invalid"


def test_conversion_condition_requires_nonvacuous_input_and_operation():
    condition = (
        "after converting each equation to slope-intercept form"
    )
    assert _operational_condition_error(
        "Convert 2x + 3y = 6 and 4x + 5y = 7 to slope-intercept form, "
        "then compare their coefficients.",
        condition,
    ) is None
    assert _operational_condition_error(
        "Convert y = 2x + 3 and y = 4x - 1 to slope-intercept form, "
        "then compare them.",
        condition,
    )
    assert _operational_condition_error(
        "Compare y = 2x + 3 and y = 4x - 1.",
        condition,
    )


def test_conversion_condition_accepts_operational_semantic_paraphrase():
    assert _operational_condition_error(
        "Rewrite 2x + 3y = 6 so y is isolated, and then inspect the slope "
        "and vertical intercept.",
        "after converting to slope-intercept form",
    ) is None


class _ReasoningClient(_LengthClient):
    def _post_with_retry(self, payload):
        self.payload = payload
        return {
            "choices": [{
                "message": {
                    "content": '{"label":"ready"}',
                    "reasoning_content": "hidden reasoning must be absent",
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        }, 0


class _Capture:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


class _Base:
    api_url = "http://test.invalid/v1/chat/completions"
    base_url = "http://test.invalid/v1"
    client = object()
    _oa_client = _Client()
    _model = "test-model"
    _provider_name = "local"
    _provenance_provider = "local"
    _max_tokens = 800
    class _PlanNli:
        @staticmethod
        def score_pair(*, premise, hypothesis):
            premise_l = premise.lower()
            hypothesis_l = hypothesis.lower()
            is_bad_answer = (
                "successful operations" in hypothesis_l
                and ("remain" in hypothesis_l or "keep" in hypothesis_l)
            )
            describes_error = any(
                marker in hypothesis_l
                for marker in (
                    "partial-success preservation", "keep earlier successful",
                    "mistakes execution order",
                )
            ) and "successful operations" in premise_l
            contrast = (
                "no partial" in premise_l and "remain" in hypothesis_l
            )
            if describes_error:
                return type("_Score", (), {
                    "entailment": 0.99, "contradiction": 0.001,
                })()
            if is_bad_answer or contrast:
                return type("_Score", (), {
                    "entailment": 0.01, "contradiction": 0.99,
                })()
            return type("_Score", (), {
                "entailment": 0.99, "contradiction": 0.001,
            })()

    _plan_nli_scorer = _PlanNli()

    def __init__(self, responses, capture=None):
        self.responses = iter(responses)
        self._capture = capture
        self.prompts = []
        self.schemas = []

    def _chat_completion_raw_structured(
        self, messages, *, schema, max_tokens=None,
    ):
        assert schema["type"] == "object"
        self.prompts.append(messages)
        self.schemas.append(schema)
        self.request_max_tokens = getattr(self, "request_max_tokens", [])
        self.request_max_tokens.append(
            self._max_tokens if max_tokens is None else max_tokens
        )
        return next(self.responses), {
            "prompt_tokens": 11,
            "completion_tokens": 7,
        }, 0


@pytest.fixture(autouse=True)
def _served_context_window(monkeypatch):
    monkeypatch.setenv(
        "TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768",
    )


def _chunk():
    return {
        "id": "chunk-test",
        "text": (
            "A transaction groups related operations into one logical unit. "
            "Atomicity means the operations either all succeed or all fail, "
            "which prevents a partially applied update."
        ),
        "learning_outcome_refs": ["co-01"],
        "bloom_level": "analyze",
        "content_type_label": "explanation",
        "concept_tags": ["transaction atomicity"],
        "misconceptions": [{
            "id": "mc-01",
            "statement": "Atomic transactions preserve successful partial work.",
        }],
        "synthesis_focus_objective": {
            "id": "co-01",
            "statement": "Analyze how transaction atomicity prevents partial updates.",
            "bloom_level": "analyze",
        },
    }


def _sft_plan():
    return json.dumps({
        "objective_id": "co-01",
        "bloom_level": "analyze",
        "supported_claims": [{
            "claim": "Atomicity prevents partial updates.",
            "evidence_quote": (
                "Atomicity means the operations either all succeed or all fail"
            ),
        }],
        "learner_task": "Examine the relationship between atomicity and partial updates.",
        "misconception_affordance": {
            "error_mechanism": "",
            "faulty_step": "",
            "causal_rationale": "",
        },
    })


def _dpo_plan():
    value = json.loads(_sft_plan())
    value["misconception_affordance"] = {
        "error_mechanism": "partial-success preservation",
        "faulty_step": "Keep earlier successful operations after a later failure.",
        "causal_rationale": (
            "The learner mistakes execution order for permission to retain "
            "partial transactional state."
        ),
    }
    return json.dumps(value)


def test_flag_is_opt_in_and_garbage_is_legacy():
    assert not staged_synthesis_v4_enabled({})
    assert not staged_synthesis_v4_enabled(
        {"TRAINFORGE_STAGED_SYNTHESIS_V4": "garbage"}
    )


def test_evidence_quote_is_canonicalized_across_punctuation_only():
    source = "Atomicity means: operations either all succeed, or all fail."
    assert _canonical_evidence_quote(
        "Atomicity means operations either all succeed or all fail", source,
    ) == "Atomicity means: operations either all succeed, or all fail"
    assert _canonical_evidence_quote(
        "Atomicity sometimes preserves partial work", source,
    ) is None
    assert staged_synthesis_v4_enabled(
        {"TRAINFORGE_STAGED_SYNTHESIS_V4": "true"}
    )


def test_leakage_boundary_is_exactly_fifty_characters():
    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwx-tail"
    assert _first_verbatim_span(source[:49], source) is None
    assert _first_verbatim_span(source[:50], source) == source[:50].lower()


def test_leakage_subtracts_only_frequency_proven_mandatory_contract_text():
    mandatory = (
        "Determine the product and sum for the supplied symbolic values "
        "under the stated condition."
    )
    contexts = [
        {"context_id": "context-alpha", "text": mandatory},
        {"context_id": "context-beta", "text": f"Reusable task: {mandatory}"},
    ]
    assert _first_distinctive_verbatim_span(
        mandatory, mandatory,
        mandatory_contract_text=[mandatory],
        nondistinctive_contexts=contexts,
    ) is None
    narrative = (
        "A distinctive narrative explains a particular sequence of events "
        "whose wording belongs only to this evidence source."
    )
    assert _first_distinctive_verbatim_span(
        f"{mandatory} {narrative}", f"{mandatory} {narrative}",
        mandatory_contract_text=[mandatory],
        nondistinctive_contexts=contexts,
    ) is not None


def test_leakage_does_not_exempt_single_context_or_objective_by_itself():
    copied = (
        "Compare the product and sum using the supplied values and explain "
        "the relationship between both results."
    )
    assert _first_distinctive_verbatim_span(
        copied, copied,
        mandatory_contract_text=[copied],
        nondistinctive_contexts=[
            {"context_id": "only-context", "text": copied},
        ],
    ) is not None


def test_leakage_frequency_normalizes_structural_math_placeholders():
    mandatory = (
        "For x = 12, determine the product and sum under the given condition."
    )
    contexts = [
        {
            "context_id": "variant-one",
            "text": "For y = 8, determine the product and sum under the given condition.",
        },
        {
            "context_id": "variant-two",
            "text": "For z = 19, determine the product and sum under the given condition.",
        },
    ]
    assert _first_distinctive_verbatim_span(
        mandatory, mandatory,
        mandatory_contract_text=[mandatory],
        nondistinctive_contexts=contexts,
    ) is None


def test_relation_signature_accepts_only_complete_relation_preservation():
    evidence = "Multiply alpha by beta when gamma is positive."
    assert _relation_preserving_support(
        evidence, "Multiplication alpha by beta when gamma is positive.",
    )
    assert not _relation_preserving_support(
        evidence, "Multiply beta by alpha when gamma is positive.",
    )
    assert not _relation_preserving_support(
        evidence, "Do not multiply alpha by beta when gamma is positive.",
    )
    assert not _relation_preserving_support(
        evidence,
        "Multiply alpha by beta when gamma is positive and add delta.",
    )


def test_deterministic_clause_entailment_preserves_governor_operator_and_polarity():
    authoritative = (
        "When two objects move in opposite directions, their relative speed "
        "is the sum of their individual speeds, not the difference."
    )
    supported = (
        "When two objects move in opposite directions, their relative speed "
        "is the sum of their individual speeds"
    )
    assert (
        _deterministic_entailment_basis(authoritative, supported)
        == "exact_authoritative_clause"
    )
    assert _deterministic_entailment_basis(
        authoritative,
        "When two objects move in opposite directions, their relative speed "
        "is the difference of their individual speeds",
    ) is None
    assert _deterministic_entailment_basis(
        authoritative,
        "When two objects move in opposite directions, their relative speed "
        "is not the sum of their individual speeds",
    ) is None
    assert _deterministic_entailment_basis(
        authoritative,
        "When two objects move in opposite directions, their relative price "
        "is the sum of their individual speeds",
    ) is None
    assert _deterministic_entailment_basis(
        authoritative,
        "Their relative speed is the sum of their individual speeds",
    ) is None


@pytest.mark.parametrize(
    "marker",
    [
        "Therefore", "THUS!", "However…", "Consequently;", "Moreover:",
        "As a result", "On the other hand.",
    ],
)
def test_coverage_units_never_adjudicate_standalone_discourse_marker(marker):
    units = _coverage_units(
        f"{marker}, the supported relation remains true."
    )
    assert units == ["the supported relation remains true."]
    assert all(not _is_standalone_discourse_marker(unit) for unit in units)


def test_discourse_marker_filter_preserves_substantive_and_multilingual_units():
    assert _coverage_units(
        "Therefore, an unsupported satellite controls the result."
    ) == ["an unsupported satellite controls the result."]
    assert _coverage_units(
        "The result, however surprising, remains constrained."
    ) == ["The result", "however surprising", "remains constrained."]
    assert _coverage_units(
        "The system therefore preserves the operator and its negation."
    ) == [
        "The system therefore preserves the operator",
        "its negation.",
    ]
    # The inventory is intentionally not expanded through guessed
    # translations; an unfamiliar marker remains visible to adjudication.
    assert _coverage_units("Por lo tanto.") == ["Por lo tanto."]
    assert not _is_standalone_discourse_marker("Por lo tanto.")


def test_sft_uses_plan_then_realization_and_captures_each_call():
    capture = _Capture()
    provider = StagedSynthesisProvider(_Base([
        _sft_plan(),
        json.dumps({
            "prompt": (
                "Analyze the relationship between transaction atomicity and "
                "the prevention of partially applied updates."
            ),
            "completion": (
                "Atomicity treats related operations as a unit: they all "
                "succeed or all fail, so no partial update remains."
            ),
            "covered_claim_indices": [0],
        }),
    ], capture))
    result = provider.paraphrase_instruction(
        {"provider": "mock", "prompt": "draft", "completion": "draft"},
        _chunk(),
    )
    assert result["provider"] == "local"
    assert result["observed_bloom"] == "analyze"
    assert result["objective_contract"]["id"] == "co-01"
    assert len(result["synthesis_plan_sha256"]) == 64
    assert [call["decision_type"] for call in capture.calls] == [
        "synthesis_provider_call",
        "synthesis_provider_call",
    ]
    assert all(call["prompt_ref"] for call in capture.calls)
    assert all(call["inputs_ref"] for call in capture.calls)
    assert all(call["outputs"] for call in capture.calls)
    assert all(call["task_id"].startswith("T-") for call in capture.calls)
    realization_prompt = base_prompt = provider._base.prompts[1][1]["content"]
    assert "Atomicity means the operations either all succeed or all fail" not in (
        base_prompt
    )
    assert "Atomicity prevents partial updates." in realization_prompt


def test_incomplete_realization_is_repaired_at_realization_stage_only():
    incomplete = json.dumps({
        "prompt": (
            "Analyze the relationship between transaction atomicity and "
            "the prevention of partially applied updates."
        ),
        "completion": (
            "Atomicity treats related operations as one unit and prevents "
            "partial state from remaining after a failure."
        ),
        "covered_claim_indices": [],
    })
    complete = json.loads(incomplete)
    complete["covered_claim_indices"] = [0]
    base = _Base([_sft_plan(), incomplete, json.dumps(complete)])
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert len(base.prompts) == 3
    assert "covered_claim_indices must exactly cover" in (
        base.prompts[2][-1]["content"]
    )


def test_completion_max_boundary_passes_and_plus_one_is_regenerated():
    # PACKAGE A: Update boundary from 600→1200. Now test that 1200 chars is
    # accepted but 1201 is regenerated (rejection triggers a repair request).
    prompt = "P" * 400
    at_limit = json.dumps({
        "prompt": prompt,
        "completion": "C" * 1200,
        "covered_claim_indices": [0],
    })
    over_limit = json.dumps({
        "prompt": prompt,
        "completion": "C" * 1201,
        "covered_claim_indices": [0],
    })
    accepted = StagedSynthesisProvider(
        _Base([_sft_plan(), at_limit])
    ).paraphrase_instruction({}, _chunk())
    assert len(accepted["completion"]) == 1200

    base = _Base([_sft_plan(), over_limit, at_limit])
    repaired = StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert len(repaired["completion"]) == 1200
    assert "completion length 1201 is outside 1..1200" in (
        base.prompts[2][-1]["content"]
    )
    assert "C" * 1201 in base.prompts[2][-2]["content"]


def test_concise_semantically_supported_math_answer_is_not_length_rejected():
    base = _Base([
        _sft_plan(),
        json.dumps({
            "prompt": "Analyze transaction atomicity.",
            "completion": "Atomicity prevents partial updates.",
            "covered_claim_indices": [0],
        }),
    ])
    result = StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert result["completion"] == "Atomicity prevents partial updates."


def test_actual_concise_math_answer_x_equals_two_passes_semantic_boundary():
    chunk = {
        "id": "chunk-math",
        "text": (
            "For the equation x plus 3 equals 5, subtracting 3 from both "
            "sides gives x = 2."
        ),
        "learning_outcome_refs": ["co-02"],
        "bloom_level": "apply",
        "synthesis_focus_objective": {
            "id": "co-02",
            "statement": "Apply inverse operations to solve a linear equation.",
            "bloom_level": "apply",
        },
    }
    plan = json.dumps({
        "objective_id": "co-02",
        "bloom_level": "apply",
        "supported_claims": [{
            "claim": "x = 2",
            "evidence_quote": "subtracting 3 from both sides gives x = 2",
        }],
        "learner_task": "Solve the linear equation using inverse operations.",
        "misconception_affordance": {
            "error_mechanism": "",
            "faulty_step": "",
            "causal_rationale": "",
        },
    })
    base = _Base([
        plan,
        json.dumps({
            "prompt": "Solve x + 3 = 5.",
            "completion": "x = 2",
            "covered_claim_indices": [0],
        }),
    ])
    result = StagedSynthesisProvider(base).paraphrase_instruction({}, chunk)
    assert result["completion"] == "x = 2"


def test_context_preflight_fails_before_transport(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "100")
    base = _Base([_sft_plan()])
    with pytest.raises(SynthesisProviderError) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert caught.value.code == "staged_context_window_exceeded"
    assert base.prompts == []


def test_verified_preflight_context_replaces_unset_ambient_env(monkeypatch):
    monkeypatch.delenv(
        "TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", raising=False,
    )
    base = _Base([
        _sft_plan(),
        json.dumps({
            "prompt": (
                "Analyze how transaction atomicity prevents partially "
                "applied updates."
            ),
            "completion": (
                "Atomicity makes related operations all succeed or all fail, "
                "so a partial update cannot remain."
            ),
            "covered_claim_indices": [0],
        }),
    ])
    provider = StagedSynthesisProvider(base)
    provider.bind_verified_served_context({
        "status": "accepted",
        "served_context_tokens": 262144,
        "parser_probe": {
            "static_kv_startup_facts": {"max_seq_len": 262144},
        },
    })
    result = provider.paraphrase_instruction({}, _chunk())
    assert result["completion"]
    assert len(base.prompts) == 2


def test_verified_preflight_context_mismatch_fails_closed():
    provider = StagedSynthesisProvider(_Base([]))
    with pytest.raises(SynthesisProviderError) as caught:
        provider.bind_verified_served_context({
            "status": "accepted",
            "served_context_tokens": 262144,
            "parser_probe": {
                "static_kv_startup_facts": {"max_seq_len": 131072},
            },
        })
    assert caught.value.code == "staged_context_window_unverified"


def test_polynomial_error_becomes_captured_terminal_content_rejection(
    monkeypatch,
):
    import sympy

    def polynomial_error(*_args, **_kwargs):
        raise sympy.PolynomialError("non-polynomial expression")

    monkeypatch.setattr(sympy, "Poly", polynomial_error)
    capture = _Capture()
    base = _Base([json.dumps({"claim": "x = 1"})] * 3, capture)
    provider = StagedSynthesisProvider(base)

    def validator(_value):
        assert _equation_signatures("x = 1") == set()
        return "symbolic equation has no valid polynomial signature"

    with pytest.raises(SynthesisProviderError) as caught:
        provider._call_stage(
            stage="plan_sft",
            chunk_id="unit-test",
            system="Return strict JSON.",
            user="Return one claim.",
            required_keys=("claim",),
            validator=validator,
            response_schema={
                "type": "object",
                "properties": {"claim": {"type": "string"}},
                "required": ["claim"],
                "additionalProperties": False,
            },
        )
    assert caught.value.code == "staged_plan_sft_invalid"
    # The canonical-response progress guard stops before a redundant third call.
    assert len(base.prompts) == 2
    assert len(capture.calls) == 2
    reasons = [
        json.loads(call["context"])["validation_evidence"]["validator_reason"]
        for call in capture.calls
    ]
    assert reasons[0] == "symbolic equation has no valid polynomial signature"
    assert reasons[1].startswith("repair made no canonical progress:")


def test_pointer_prefixed_failure_targets_only_that_field_and_stops_no_progress():
    capture = _Capture()
    repeated = json.dumps({
        "learner_task": "Describe the topic.",
        "generated_givens": [],
    })
    base = _Base([repeated, repeated], capture)
    provider = StagedSynthesisProvider(base)

    with pytest.raises(SynthesisProviderError) as caught:
        provider._call_stage(
            stage="micro_A_task_design",
            chunk_id="generic-unit",
            system="Thinking is disabled. Return strict JSON.",
            user="Return a compact learner task.",
            required_keys=("learner_task", "generated_givens"),
            validator=lambda _value: (
                "/learner_task omits the semantic analysis obligation"
            ),
            response_schema={
                "type": "object",
                "properties": {
                    "learner_task": {"type": "string", "maxLength": 400},
                    "generated_givens": {"type": "array", "maxItems": 6},
                },
                "required": ["learner_task", "generated_givens"],
                "additionalProperties": False,
            },
        )

    assert caught.value.code == "staged_micro_A_task_design_invalid"
    assert len(base.prompts) == 2
    repair = base.prompts[1][-1]["content"]
    assert "Edit all and only these JSON pointers: ['/learner_task']" in repair
    assert "Preserve every other field byte-for-byte" in repair
    reasons = [
        json.loads(call["context"])["validation_evidence"]["validator_reason"]
        for call in capture.calls
    ]
    assert reasons[1].startswith("repair made no canonical progress:")


def test_numeric_closed_world_repair_names_scope_and_corrects_second_attempt():
    capture = _Capture()
    invalid = json.dumps({
        "prompt": "Compare 4 values across 8 cases.",
        "chosen": "Use 1 extra step.",
    })
    corrected = json.dumps({
        "prompt": "Compare the supplied values across the stated cases.",
        "chosen": "Use the supplied procedure.",
    })
    base = _Base([invalid, corrected], capture)
    provider = StagedSynthesisProvider(base)

    def validator(value):
        if any(character.isdigit() for character in value["prompt"] + value["chosen"]):
            return (
                "illegal numeric literals=['4', '8', '1']; affected JSON "
                "pointers=['/prompt', '/chosen']; allowed numeric "
                "literals=['-2', '2', '3']"
            )
        return None

    result = provider._call_stage(
        stage="micro_D_sft_realization",
        chunk_id="generic-unit",
        system="Thinking is disabled. Return strict JSON.",
        user="Realize the immutable task without adding numeric literals.",
        required_keys=("prompt", "chosen"),
        validator=validator,
        response_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "maxLength": 400},
                "chosen": {"type": "string", "maxLength": 400},
            },
            "required": ["prompt", "chosen"],
            "additionalProperties": False,
        },
    )

    assert result.value == json.loads(corrected)
    assert len(base.prompts) == 2
    repair = base.prompts[1][-1]["content"]
    assert "illegal numeric literals=['4', '8', '1']" in repair
    assert "['/prompt', '/chosen']" in repair
    assert "allowed numeric literals=['-2', '2', '3']" in repair
    assert "Remove, rephrase, or replace ONLY" in repair
    assert "canonical JSON must differ" in repair
    assert "never reissue" in repair


def test_numeric_closed_world_identical_repair_fails_no_progress():
    capture = _Capture()
    invalid = json.dumps({
        "prompt": "Compare 4 values across 8 cases.",
        "chosen": "Use 1 extra step.",
    })
    base = _Base([invalid, invalid], capture)
    provider = StagedSynthesisProvider(base)
    error = (
        "illegal numeric literals=['4', '8', '1']; affected JSON "
        "pointers=['/prompt', '/chosen']; allowed numeric "
        "literals=['-2', '2', '3']"
    )

    with pytest.raises(SynthesisProviderError) as caught:
        provider._call_stage(
            stage="micro_D_sft_realization",
            chunk_id="generic-unit",
            system="Thinking is disabled. Return strict JSON.",
            user="Realize the immutable task without adding numeric literals.",
            required_keys=("prompt", "chosen"),
            validator=lambda _value: error,
            response_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "maxLength": 400},
                    "chosen": {"type": "string", "maxLength": 400},
                },
                "required": ["prompt", "chosen"],
                "additionalProperties": False,
            },
        )

    assert caught.value.code == "staged_micro_D_sft_realization_invalid"
    assert len(base.prompts) == 2
    reasons = [
        json.loads(call["context"])["validation_evidence"]["validator_reason"]
        for call in capture.calls
    ]
    assert reasons[1].startswith("repair made no canonical progress:")


def test_stage_a_numeric_repair_is_compact_and_task_only():
    invalid = json.dumps({
        "objective_id": "co-generic",
        "bloom_level": "analyze",
        "learner_task": "Compare 2 values and determine whether 7 applies.",
        "generated_givens": [],
    })
    corrected = json.dumps({
        "objective_id": "co-generic",
        "bloom_level": "analyze",
        "learner_task": "Compare the supplied values and determine the result.",
        "generated_givens": [],
    })
    base = _Base([invalid, corrected])
    error = (
        "illegal numeric literals=['7']; affected JSON pointers="
        "['/learner_task', '/generated_givens']; allowed numeric literals=['2']"
    )
    result = StagedSynthesisProvider(base)._call_stage(
        stage="micro_A_task_design",
        chunk_id="generic-unit",
        system="Return strict JSON.",
        user="Design the learner task.",
        required_keys=(
            "objective_id", "bloom_level", "learner_task", "generated_givens",
        ),
        validator=lambda value: (
            error if "7" in value["learner_task"] else None
        ),
        response_schema={
            "type": "object",
            "properties": {
                "objective_id": {"type": "string"},
                "bloom_level": {"type": "string"},
                "learner_task": {"type": "string"},
                "generated_givens": {"type": "array"},
            },
            "required": [
                "objective_id", "bloom_level", "learner_task",
                "generated_givens",
            ],
        },
    )
    assert result.value == json.loads(corrected)
    repair = base.prompts[1][-1]["content"]
    assert "NUMERIC FAILURE" in repair
    assert "EDITABLE=['/learner_task']" in repair
    assert "illegal numeric literals=['7']" in repair
    assert "allowed numeric literals=['2']" in repair
    assert "Generated-given metadata is orchestrator-owned" in repair
    assert "Preserve objective_id, bloom_level" in repair
    assert len(repair) < 900


def test_stage_a_repair_finish_reason_length_is_fatal():
    base = _Base([])
    base._oa_client = _LengthClient()
    base._temperature = 0.0
    base._chat_completion_raw_structured = None
    provider = StagedSynthesisProvider(base)
    with pytest.raises(SynthesisProviderError) as caught:
        provider._call_stage(
            stage="micro_A_task_design",
            chunk_id="generic-unit",
            system="Return strict JSON.",
            user="Design the learner task.",
            required_keys=("learner_task",),
            validator=lambda _value: None,
            response_schema={
                "type": "object",
                "properties": {"learner_task": {"type": "string"}},
                "required": ["learner_task"],
            },
        )
    assert caught.value.code == "staged_output_truncated"


def test_finish_reason_length_is_never_accepted():
    capture = _Capture()
    base = _Base([], capture)
    base._oa_client = _LengthClient()
    base._temperature = 0.0
    base._chat_completion_raw_structured = None
    with pytest.raises(SynthesisProviderError) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert caught.value.code == "staged_output_truncated"
    assert base._oa_client.payload["response_format"]["type"] == "json_schema"
    assert len(capture.calls) == 1
    assert "staged_output_truncated" in capture.calls[0]["rationale"]


def test_explicit_stage_max_tokens_reaches_actual_request_and_length_is_fatal():
    base = _Base([])
    base._oa_client = _LengthClient()
    base._temperature = 0.0
    base._chat_completion_raw_structured = None
    provider = StagedSynthesisProvider(base)
    with pytest.raises(SynthesisProviderError) as caught:
        provider._structured_completion(
            [{"role": "system", "content": "Thinking is disabled."}],
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "maxLength": 16},
                },
                "required": ["label"],
                "additionalProperties": False,
            },
            max_output_tokens=2048,
        )
    assert caught.value.code == "staged_output_truncated"
    assert base._oa_client.payload["max_tokens"] == 2048


def test_structured_transport_rejects_reasoning_when_thinking_is_off():
    capture = _Capture()
    base = _Base([], capture)
    base._oa_client = _ReasoningClient()
    base._temperature = 0.0
    base._chat_completion_raw_structured = None
    with pytest.raises(SynthesisProviderError) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert caught.value.code == "staged_thinking_not_disabled"
    assert len(capture.calls) == 1
    assert "staged_thinking_not_disabled" in capture.calls[0]["rationale"]


def test_staged_v4_calls_are_metered_once_and_decision_captured(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-STAGED-V4")
    monkeypatch.setenv("ED4ALL_ACTIVE_PHASE", "training_synthesis")
    responses = iter([
        _sft_plan(),
        json.dumps({
            "prompt": (
                "Analyze how transaction atomicity prevents partially "
                "applied updates in the described operation."
            ),
            "completion": (
                "Atomicity groups the operations so they all succeed or all "
                "fail, preventing a partially applied update."
            ),
            "covered_claim_indices": [0],
        }),
    ])
    requests_seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests_seen
        requests_seen += 1
        if requests_seen == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"content": next(responses)},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    capture = _Capture()
    base = _Base([], capture)
    base._chat_completion_raw_structured = None
    base._temperature = 0.0
    base._oa_client = OpenAICompatibleClient(
        base_url="http://test.invalid/v1",
        model="snapshot-sha",
        provider_label="local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=lambda _seconds: None,
    )
    provider = StagedSynthesisProvider(base)
    provider._model = "snapshot-sha"
    provider.paraphrase_instruction({}, _chunk())

    rows = [
        json.loads(line)
        for line in (
            tmp_path / "runs" / "RUN-STAGED-V4" / "llm_usage.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert [row["task"] for row in rows] == [
        "staged_synthesis:plan_sft",
        "staged_synthesis:sft_realization",
    ]
    assert all(row["phase"] == "training_synthesis" for row in rows)
    assert all(row["model"] == "snapshot-sha" for row in rows)
    assert all(row["total_tokens"] == 18 for row in rows)
    assert len(capture.calls) == 2
    assert requests_seen == 3  # one retry + two successful logical calls


@pytest.mark.parametrize(
    ("choice", "expected_code"),
    [
        (
            {
                "message": {"content": '{"objective_id":'},
                "finish_reason": "length",
            },
            "staged_output_truncated",
        ),
        (
            {
                "message": {
                    "content": '{"objective_id":"co-01"}',
                    "reasoning_content": "reasoning must not be emitted",
                },
                "finish_reason": "stop",
            },
            "staged_thinking_not_disabled",
        ),
    ],
)
def test_staged_preparse_failure_has_one_usage_and_one_capture(
    monkeypatch, tmp_path, choice, expected_code
):
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", f"RUN-{expected_code}")
    monkeypatch.setenv("ED4ALL_ACTIVE_PHASE", "training_synthesis")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                },
            },
        )

    capture = _Capture()
    base = _Base([], capture)
    base._chat_completion_raw_structured = None
    base._temperature = 0.0
    base._oa_client = OpenAICompatibleClient(
        base_url="http://test.invalid/v1",
        model="test-model",
        provider_label="local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=lambda _seconds: None,
    )
    with pytest.raises(SynthesisProviderError) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert caught.value.code == expected_code
    rows = (
        tmp_path / "runs" / f"RUN-{expected_code}" / "llm_usage.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert len(capture.calls) == 1
    assert expected_code in capture.calls[0]["rationale"]


def test_raw_audit_artifacts_retain_complete_bytes(tmp_path):
    capture = _Capture()
    capture.output_dir = tmp_path
    raw_realization = json.dumps({
        "prompt": (
            "Analyze the relationship between transaction atomicity and "
            "the prevention of partially applied updates."
        ),
        "completion": "C" * 600,
        "covered_claim_indices": [0],
    })
    provider = StagedSynthesisProvider(
        _Base([_sft_plan(), raw_realization], capture)
    )
    provider.paraphrase_instruction({}, _chunk())
    response_ref = capture.calls[-1]["outputs"][0]
    artifact = __import__("pathlib").Path(response_ref["path"])
    assert artifact.read_bytes() == raw_realization.encode("utf-8")
    assert artifact.stat().st_mode & 0o777 == 0o600
    assert response_ref["size_bytes"] == len(raw_realization.encode("utf-8"))


def test_leakage_repair_targets_only_failed_field():
    leaked = json.dumps({
        "prompt": (
            "Analyze the relationship between transaction atomicity and "
            "the prevention of partially applied updates."
        ),
        "completion": (
            "Atomicity means the operations either all succeed or all fail, "
            "which prevents a partially applied update."
        ),
        "covered_claim_indices": [0],
    })
    repaired = json.dumps({
        "prompt": (
            "Analyze the relationship between transaction atomicity and "
            "the prevention of partially applied updates."
        ),
        "completion": (
            "A transaction commits its related work as one unit or rolls that "
            "work back, preventing an incomplete state change."
        ),
        "covered_claim_indices": [0],
    })
    base = _Base([_sft_plan(), leaked, repaired])
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    repair = base.prompts[2][-1]["content"]
    assert "field 'completion'" in repair
    assert "Completely paraphrase only 'completion'" in repair
    assert "Preserve every other field" in repair


def test_dpo_rejected_requires_explicit_faulty_reasoning():
    chosen = json.dumps({
        "prompt": (
            "Analyze why transaction atomicity prevents a partial update "
            "when one related operation fails."
        ),
        "chosen": (
            "Atomicity makes the operations fail together, so no partial "
            "transaction update remains."
        ),
        "covered_claim_indices": [0],
    })
    weak = json.dumps({
        "rejected": (
            "Atomicity keeps successful operations even when another operation "
            "in the transaction fails."
        ),
        "distorted_claim_index": 0,
        "error_mechanism": "partial-success preservation",
        "faulty_step": "Keep earlier successful operations after a later failure.",
        "causal_rationale": (
            "The learner mistakes execution order for permission to retain "
            "partial transactional state."
        ),
    })
    explicit = json.dumps({
        "rejected": (
            "Because successful operations ran first, they remain applied even "
            "when a later operation in the transaction fails."
        ),
        "distorted_claim_index": 0,
        "error_mechanism": "partial-success preservation",
        "faulty_step": "Keep earlier successful operations after a later failure.",
        "causal_rationale": (
            "The learner mistakes execution order for permission to retain "
            "partial transactional state."
        ),
    })
    base = _Base([_dpo_plan(), chosen, weak, explicit])
    result = StagedSynthesisProvider(base).paraphrase_preference({}, _chunk())
    assert result["rejected"].startswith("Because")
    assert "faulty causal reasoning" in base.prompts[3][-1]["content"]


def test_dpo_realizes_chosen_and_rejected_in_separate_calls():
    base = _Base([
        _dpo_plan(),
        json.dumps({
            "prompt": (
                "Analyze why transaction atomicity prevents a partial update "
                "when one related operation fails."
            ),
            "chosen": (
                "Atomicity makes the related operations fail together, leaving "
                "none of their updates partially applied."
            ),
            "covered_claim_indices": [0],
        }),
        json.dumps({
            "rejected": (
                "Because successful operations already ran, they remain applied "
                "even when another operation in the transaction fails."
            ),
            "distorted_claim_index": 0,
            "error_mechanism": "partial-success preservation",
            "faulty_step": (
                "Keep earlier successful operations after a later failure."
            ),
            "causal_rationale": (
                "The learner mistakes execution order for permission to "
                "retain partial transactional state."
            ),
        }),
    ])
    result = StagedSynthesisProvider(base).paraphrase_preference(
        {"provider": "mock", "prompt": "draft", "chosen": "x", "rejected": "y"},
        _chunk(),
    )
    assert len(base.prompts) == 3
    assert result["chosen"] != result["rejected"]
    assert "FIXED_PROMPT" in base.prompts[2][-1]["content"]


def test_every_nli_call_has_hash_only_contemporaneous_audit_evidence():
    class _RecordingNli:
        def __init__(self):
            self.calls = []

        def score_pair(self, *, premise, hypothesis):
            self.calls.append((premise, hypothesis))
            return _Base._PlanNli.score_pair(
                premise=premise, hypothesis=hypothesis,
            )

    capture = _Capture()
    scorer = _RecordingNli()
    chunk = _chunk()
    chunk["synthesis_focus_objective"]["action_object"] = (
        "transaction atomicity and partial updates"
    )
    plan = json.loads(_dpo_plan())
    plan["learner_task"] = (
        "Examine how all-or-nothing transactions avoid incomplete state."
    )
    chosen = {
        "prompt": "Analyze how all-or-nothing transactions avoid incomplete state.",
        "chosen": (
            "Atomicity makes the related operations fail together, leaving "
            "none of their updates partially applied."
        ),
        "covered_claim_indices": [0],
    }
    rejected = {
        "rejected": (
            "Because successful operations already ran, they remain applied "
            "even when another operation in the transaction fails."
        ),
        "distorted_claim_index": 0,
        "error_mechanism": "partial-success preservation",
        "faulty_step": "Keep earlier successful operations after a later failure.",
        "causal_rationale": (
            "The learner mistakes execution order for permission to retain "
            "partial transactional state."
        ),
    }
    base = _Base([
        json.dumps(plan), json.dumps(chosen), json.dumps(rejected),
    ], capture)
    base._plan_nli_scorer = scorer

    StagedSynthesisProvider(base).paraphrase_preference({}, chunk)

    evidence = []
    for call in capture.calls:
        context = json.loads(call["context"])
        evidence.extend(
            context["validation_evidence"].get("nli_scores", [])
        )
    assert len(evidence) == len(scorer.calls)
    assert {
        row["decision_type"] for row in evidence
    } >= {
        "plan_quote_claim",
        "semantic_coverage",
        "objective_task_delivery",
        "dpo_error_mechanism",
        "dpo_faulty_step",
        "dpo_causal_rationale",
        "dpo_chosen_contrast",
        "dpo_evidence_support",
    }
    for row, (premise, hypothesis) in zip(evidence, scorer.calls):
        assert row["stage"] in {
            "plan_dpo", "dpo_chosen_realization",
            "dpo_rejected_realization",
        }
        assert row["attempt"] == 1
        assert row["premise_sha256"] == hashlib.sha256(
            premise.encode("utf-8")
        ).hexdigest()
        assert row["hypothesis_sha256"] == hashlib.sha256(
            hypothesis.encode("utf-8")
        ).hexdigest()
        is_negative = row["decision_type"] in {
                "dpo_chosen_contrast", "dpo_evidence_support",
        }
        assert row["entailment"] == pytest.approx(0.01 if is_negative else 0.99)
        assert row["contradiction"] == pytest.approx(0.99 if is_negative else 0.001)
        assert premise not in json.dumps(row)
        assert hypothesis not in json.dumps(row)


def test_exact_validation_failure_is_fed_back_to_same_stage():
    corrected = json.loads(_sft_plan())
    # Objective identity is the only invalid field; repair must preserve the
    # independently valid task rather than silently rewriting the whole plan.
    corrected["learner_task"] = "Analyze atomicity."
    base = _Base([
        json.dumps({
            "objective_id": "co-99",
            "bloom_level": "analyze",
            "supported_claims": [{
                "claim": "Atomicity prevents partial updates.",
                "evidence_quote": (
                    "Atomicity means the operations either all succeed or all fail"
                ),
            }],
            "learner_task": "Analyze atomicity.",
            "misconception_affordance": {
                "error_mechanism": "",
                "faulty_step": "",
                "causal_rationale": "",
            },
        }),
        json.dumps(corrected),
        json.dumps({
            "prompt": (
                "Analyze the relationship between transaction atomicity and "
                "the prevention of partially applied updates."
            ),
            "completion": (
                "Atomicity makes related operations succeed or fail together, "
                "preventing a partially applied update."
            ),
            "covered_claim_indices": [0],
        }),
    ])
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    correction = base.prompts[1][-1]["content"]
    assert "objective_id differs from canonical objective" in correction
    assert "JSON pointers: ['/objective_id']" in correction


def test_plan_rejects_quote_not_present_in_source_after_bounded_repairs():
    bad = json.dumps({
        "objective_id": "co-01",
        "bloom_level": "analyze",
        "supported_claims": [{
            "claim": "Atomicity prevents partial updates.",
            "evidence_quote": "This quote was fabricated and is not present.",
        }],
        "learner_task": "Analyze atomicity.",
        "misconception_affordance": {
            "error_mechanism": "",
            "faulty_step": "",
            "causal_rationale": "",
        },
    })
    base = _Base([bad, bad, bad])
    provider = StagedSynthesisProvider(base)
    with pytest.raises(SynthesisProviderError, match="exhausted") as caught:
        provider.paraphrase_instruction({}, _chunk())
    assert caught.value.details["terminal_content_rejection"] is True
    assert caught.value.details["stage"] == "plan_sft"
    assert caught.value.details["prompt_ref"].startswith("sha256:")
    assert caught.value.details["response_ref"].startswith("sha256:")
    for repair_prompt in base.prompts[1:]:
        repair = repair_prompt[-1]["content"]
        assert "only editable plan element is supported_claims[0]" in repair
        assert "FROZEN_VALID_CLAIMS_SHA256=" in repair
        assert "Do not edit, reorder, merge, or paraphrase" in repair


def test_claim_repair_guard_allows_only_exact_replace_or_delete():
    plan = {
        "objective_id": "objective-generic",
        "supported_claims": [
            {"claim": "invalid", "evidence_quote": "absent"},
            {"claim": "frozen", "evidence_quote": "present"},
        ],
        "generated_givens": [{
            "symbol": "n", "value": 2, "unit": "items",
            "role": "input", "synthetic": True,
            "provenance": "generated",
        }],
        "task": {"verb": "analyze", "constraints": ["preserve"]},
    }
    guard = _claim_repair_guard(plan, 0)

    replaced = json.loads(json.dumps(plan))
    replaced["supported_claims"][0] = {
        "claim": "replacement", "evidence_quote": "present",
    }
    assert _claim_repair_diff(replaced, guard) is None

    deleted = json.loads(json.dumps(plan))
    del deleted["supported_claims"][0]
    assert _claim_repair_diff(deleted, guard) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "reorder",
        "insert",
        "mutate_frozen",
        "mutate_nested_task",
        "mutate_generated_givens",
    ],
)
def test_claim_repair_guard_rejects_adversarial_mutations(mutation):
    plan = {
        "objective_id": "objective-generic",
        "supported_claims": [
            {"claim": "invalid", "evidence_quote": "absent"},
            {"claim": "frozen-a", "evidence_quote": "present-a"},
            {"claim": "frozen-b", "evidence_quote": "present-b"},
        ],
        "generated_givens": [{
            "symbol": "n", "value": 2, "unit": "items",
            "role": "input", "synthetic": True,
            "provenance": "generated",
        }],
        "task": {"verb": "analyze", "constraints": ["preserve"]},
    }
    guard = _claim_repair_guard(plan, 0)
    candidate = json.loads(json.dumps(plan))
    if mutation == "reorder":
        candidate["supported_claims"][1:] = reversed(
            candidate["supported_claims"][1:]
        )
    elif mutation == "insert":
        candidate["supported_claims"].append({
            "claim": "inserted", "evidence_quote": "present",
        })
    elif mutation == "mutate_frozen":
        candidate["supported_claims"][1]["claim"] = "changed"
    elif mutation == "mutate_nested_task":
        candidate["task"]["constraints"].append("changed")
    else:
        candidate["generated_givens"][0]["value"] = 3

    diff = _claim_repair_diff(candidate, guard)
    assert diff is not None
    assert diff["reason"]
    assert diff["baseline_sha256"] == guard["baseline_sha256"]
    assert diff["candidate_sha256"] != guard["baseline_sha256"]


def test_call_stage_rejects_mutation_then_accepts_scoped_claim_replacement():
    original = {
        "supported_claims": [
            {"claim": "invalid", "evidence_quote": "absent"},
            {"claim": "frozen", "evidence_quote": "present"},
        ],
        "generated_givens": [{
            "symbol": "n", "value": 2, "unit": "items",
            "role": "input", "synthetic": True,
            "provenance": "generated",
        }],
        "task": {"verb": "analyze", "constraints": ["preserve"]},
    }
    adversarial = json.loads(json.dumps(original))
    adversarial["supported_claims"][0] = {
        "claim": "replacement", "evidence_quote": "present",
    }
    adversarial["generated_givens"][0]["value"] = 99
    repaired = json.loads(json.dumps(original))
    repaired["supported_claims"][0] = {
        "claim": "replacement", "evidence_quote": "present",
    }
    capture = _Capture()
    provider = StagedSynthesisProvider(_Base([
        json.dumps(original), json.dumps(adversarial), json.dumps(repaired),
    ], capture=capture))

    result = provider._call_stage(
        stage="generic_plan",
        chunk_id="generic-chunk",
        system="Return JSON.",
        user="Plan.",
        required_keys=("supported_claims", "generated_givens", "task"),
        validator=lambda value: (
            "supported_claims[0] is invalid"
            if value["supported_claims"][0]["claim"] == "invalid" else None
        ),
        response_schema={"type": "object"},
    )

    assert result.value == repaired
    assert result.requests == 3
    evidence = json.loads(capture.calls[1]["context"])[
        "validation_evidence"
    ]
    diff = evidence["repair_immutability"]
    assert diff["reason"] == "claim repair mutated a noneditable field"
    repair_prompt = provider._base.prompts[2][-1]["content"]
    assert "BASELINE_SHA256=" in repair_prompt
    assert "REJECTED_CANDIDATE_SHA256=" in repair_prompt


@pytest.mark.parametrize(
    ("stage", "method", "responses"),
    [
        ("plan_sft", "instruction", ["bad", "bad", "bad"]),
        ("sft_realization", "instruction", [
            _sft_plan(), "bad", "bad", "bad",
        ]),
        ("plan_dpo", "preference", ["bad", "bad", "bad"]),
        ("dpo_chosen_realization", "preference", [
            _dpo_plan(), "bad", "bad", "bad",
        ]),
        ("dpo_rejected_realization", "preference", [
            _dpo_plan(),
            json.dumps({
                "prompt": (
                    "Analyze why transaction atomicity prevents a partial "
                    "update when one related operation fails."
                ),
                "chosen": (
                    "Atomicity makes related operations fail together, "
                    "leaving no partial update."
                ),
                "covered_claim_indices": [0],
            }),
            "bad", "bad", "bad",
        ]),
    ],
)
def test_every_staged_contract_exhaustion_carries_terminal_rejection_evidence(
    stage, method, responses,
):
    provider = StagedSynthesisProvider(_Base(responses))
    with pytest.raises(SynthesisProviderError) as caught:
        if method == "instruction":
            provider.paraphrase_instruction({}, _chunk())
        else:
            provider.paraphrase_preference({}, _chunk())
    assert caught.value.code == f"staged_{stage}_invalid"
    assert caught.value.details == {
        "terminal_content_rejection": True,
        "stage": stage,
        "validation_error": "response was not a JSON object",
        "prompt_ref": caught.value.details["prompt_ref"],
        "response_ref": caught.value.details["response_ref"],
    }
    assert caught.value.details["prompt_ref"].startswith("sha256:")
    assert caught.value.details["response_ref"].startswith("sha256:")


@pytest.mark.parametrize("kind", ["instruction", "preference"])
def test_exhausted_staged_content_contract_is_terminal_pair_rejection(kind):
    capture = _Capture()
    provider = StagedSynthesisProvider(
        _Base(["not-json", "still-not-json", "never-json"], capture)
    )
    if kind == "instruction":
        result = synthesize_instruction_pair(
            _chunk(), seed=7, provider="local", paraphrase_provider=provider,
            capture=capture,
        )
    else:
        result = synthesize_preference_pair(
            _chunk(), seed=7, provider="local", paraphrase_provider=provider,
            capture=capture,
        )
    assert result.pair is None
    assert result.quality["reason"] == f"staged_plan_{'sft' if kind == 'instruction' else 'dpo'}_invalid"
    evidence = result.quality["rejection_evidence"]
    assert evidence["terminal_content_rejection"] is True
    assert evidence["validation_error"] == "response was not a JSON object"
    assert evidence["prompt_ref"].startswith("sha256:")
    assert evidence["response_ref"].startswith("sha256:")
    assert any(
        call["decision_type"] in {
            "instruction_pair_synthesis", "preference_pair_generation",
        }
        and "terminal-rejection" in call["task_id"]
        for call in capture.calls
    )


def test_focus_mismatch_fails_before_provider_dispatch():
    chunk = _chunk()
    chunk["bloom_level"] = "remember"
    base = _Base([])
    with pytest.raises(SynthesisProviderError, match="canonical"):
        StagedSynthesisProvider(base).paraphrase_instruction({}, chunk)
    assert base.prompts == []


def test_window_selects_objective_blocks_and_preserves_misconception_polarity():
    chunk = _chunk()
    chunk["html"] = """
      <section data-cf-block-id="rule-1" data-cf-objective-id="co-01">
        Atomicity rolls back the entire unit after one operation fails.
      </section>
      <aside class="misconception-claim"
             data-cf-block-id="wrong-1" data-cf-objective-id="co-01">
        Earlier successful operations always remain applied.
      </aside>
      <aside class="misconception-correction"
             data-cf-block-id="fix-1" data-cf-objective-id="co-01">
        Atomicity also reverses earlier operations in the same transaction.
      </aside>
      <section data-cf-block-id="other-1" data-cf-objective-id="co-99">
        Unrelated material must not enter this synthesis window.
      </section>
    """
    window = build_evidence_window(
        chunk, chunk["synthesis_focus_objective"],
    )
    assert window["contract_version"] == WINDOW_CONTRACT_VERSION
    assert [block["block_id"] for block in window["blocks"]] == [
        "rule-1", "wrong-1", "fix-1",
    ]
    assert [block["polarity"] for block in window["blocks"]] == [
        "factual", "incorrect", "corrective",
    ]
    assert "Unrelated material" not in json.dumps(window)


def test_multi_claim_realization_schema_and_prompt_require_exact_indices():
    plan = json.loads(_sft_plan())
    plan["supported_claims"].append({
        "claim": "A transaction groups related operations.",
        "evidence_quote": (
            "A transaction groups related operations into one logical unit"
        ),
    })
    base = _Base([
        json.dumps(plan),
        json.dumps({
            "prompt": (
                "Analyze how transaction grouping and atomic failure behavior "
                "work together to prevent partial updates."
            ),
            "completion": (
                "The transaction forms one logical unit, and atomicity makes "
                "that unit succeed or fail together, preventing partial state."
            ),
            "covered_claim_indices": [0, 1],
        }),
    ])
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert base.schemas[1]["properties"]["covered_claim_indices"]["const"] == [0, 1]
    assert '"covered_claim_indices":[0,1]' in base.prompts[1][1]["content"]


def test_reported_indices_cannot_spoof_semantic_claim_coverage():
    class _SelectiveNli:
        @staticmethod
        def score_pair(*, premise, hypothesis):
            entailed = (
                "partial updates" in premise.lower()
                or "all succeed or all fail" in premise.lower()
                or "partial updates" not in hypothesis.lower()
            )
            return type("_Score", (), {
                "entailment": 0.99 if entailed else 0.01,
                "contradiction": 0.001,
            })()

    plan = json.loads(_sft_plan())
    plan["supported_claims"].append({
        "claim": "A transaction groups related operations.",
        "evidence_quote": (
            "A transaction groups related operations into one logical unit"
        ),
    })
    omitted = json.dumps({
        "prompt": "Analyze transaction grouping and atomicity.",
        "completion": "A transaction groups related operations.",
        "covered_claim_indices": [0, 1],
    })
    base = _Base([json.dumps(plan), omitted, omitted, omitted])
    base._plan_nli_scorer = _SelectiveNli()
    with pytest.raises(SynthesisProviderError) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert "missing atomic claim 0" in caught.value.details["validation_error"]
    assert "best_output_unit=0" in caught.value.details["validation_error"]


def test_grounded_reinforcement_is_allowed_with_exact_indices():
    duplicated = json.dumps({
        "prompt": "Analyze transaction atomicity.",
        "completion": (
            "Atomicity prevents partial updates. "
            "Partial updates are prevented by atomicity."
        ),
        "covered_claim_indices": [0],
    })
    base = _Base([_sft_plan(), duplicated])
    result = StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert result["completion"] == json.loads(duplicated)["completion"]


def test_later_overlapping_plan_claim_is_deleted_before_realization():
    plan = json.loads(_sft_plan())
    plan["supported_claims"].append({
        "claim": "Partial updates are prevented by transaction atomicity.",
        "evidence_quote": (
            "Atomicity means the operations either all succeed or all fail"
        ),
    })
    base = _Base([json.dumps(plan), _sft_plan(), json.dumps({
        "prompt": "Analyze transaction atomicity.",
        "completion": "Atomicity prevents partial updates.",
        "covered_claim_indices": [0],
    })])
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    repair = base.prompts[1][-1]["content"]
    assert "overlaps semantically" in repair
    assert "only editable plan element is supported_claims[1]" in repair
    assert len(base.prompts) == 3


def test_missing_claim_repair_names_best_unit_score_and_obligation():
    class _MissingSecondClaimNli:
        @staticmethod
        def score_pair(*, premise, hypothesis):
            hypothesis_l = hypothesis.lower()
            premise_l = premise.lower()
            matched = not (
                "groups related operations" in hypothesis_l
                and "groups related operations" not in premise_l
            )
            return type("_Score", (), {
                "entailment": 0.99 if matched else 0.20,
                "contradiction": 0.001,
            })()

    plan = json.loads(_sft_plan())
    plan["supported_claims"].append({
        "claim": "A transaction groups related operations.",
        "evidence_quote": (
            "A transaction groups related operations into one logical unit"
        ),
    })
    omitted = json.dumps({
        "prompt": "Analyze transaction grouping and atomicity.",
        "completion": "Atomicity prevents partial updates.",
        "covered_claim_indices": [0, 1],
    })
    corrected = json.dumps({
        "prompt": "Analyze transaction grouping and atomicity.",
        "completion": (
            "Atomicity prevents partial updates. "
            "A transaction groups related operations."
        ),
        "covered_claim_indices": [0, 1],
    })
    base = _Base([json.dumps(plan), omitted, corrected])
    base._plan_nli_scorer = _MissingSecondClaimNli()
    StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    repair = base.prompts[2][-1]["content"]
    assert "missing atomic claim 1" in repair
    assert "best_output_unit=0" in repair
    assert "entailment=0.200" in repair
    assert "missing_obligation=" in repair
    assert "explicitly entail the exact missing immutable claim" in repair
    assert "Preserve /prompt" in repair
    assert "never reissue identical canonical output" in repair


def test_scoped_missing_claim_repair_corrects_and_identical_fails_no_progress():
    invalid = json.dumps({
        "prompt": "Determine the number of solutions for the system.",
        "chosen": "infinitely many",
    })
    corrected = json.dumps({
        "prompt": "Determine the number of solutions for the system.",
        "chosen": (
            "Equal slopes and equal y-intercepts make the lines coincident, "
            "so the system has infinitely many solutions."
        ),
    })
    error = (
        "missing atomic claim 0: 'Equal slopes and equal y-intercepts make "
        "the lines coincident and give infinitely many solutions.'; "
        "missing_claim_id='claim-generic'; required_entailment=explicitly "
        "realize immutable claim 0 in /chosen; affected JSON pointers=['/chosen']"
    )
    schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "chosen": {"type": "string"},
        },
        "required": ["prompt", "chosen"],
    }
    base = _Base([invalid, corrected])
    result = StagedSynthesisProvider(base)._call_stage(
        stage="micro_D_dpo_chosen",
        chunk_id="generic-unit",
        system="Return strict JSON.",
        user="Realize every immutable claim.",
        required_keys=("prompt", "chosen"),
        validator=lambda value: (
            error if value["chosen"] == "infinitely many" else None
        ),
        response_schema=schema,
    )
    assert result.value == json.loads(corrected)
    repair = base.prompts[1][-1]["content"]
    assert "claim-generic" in repair
    assert "['/chosen']" in repair
    assert "exact missing immutable claim text/index/ID" in repair

    capture = _Capture()
    repeated = _Base([invalid, invalid], capture)
    with pytest.raises(SynthesisProviderError):
        StagedSynthesisProvider(repeated)._call_stage(
            stage="micro_D_dpo_chosen",
            chunk_id="generic-unit",
            system="Return strict JSON.",
            user="Realize every immutable claim.",
            required_keys=("prompt", "chosen"),
            validator=lambda _value: error,
            response_schema=schema,
        )
    reasons = [
        json.loads(call["context"])["validation_evidence"]["validator_reason"]
        for call in capture.calls
    ]
    assert reasons[1].startswith("repair made no canonical progress:")


def test_flat_mixed_misconception_source_fails_loud_before_dispatch():
    chunk = _chunk()
    chunk["text"] += (
        " A common misconception is that successful partial work remains."
    )
    base = _Base([])
    with pytest.raises(
        SynthesisProviderError, match="unresolved misconception polarity",
    ) as caught:
        StagedSynthesisProvider(base).paraphrase_instruction({}, chunk)
    assert caught.value.code == "staged_evidence_window_unavailable"
    assert caught.value.details["terminal_content_rejection"] is True
    assert base.prompts == []


def test_correct_rejected_answer_is_adversarially_rejected():
    chosen = json.dumps({
        "prompt": "Analyze why atomicity prevents partial updates.",
        "chosen": (
            "Atomicity makes the operations fail together, so no partial "
            "transaction update remains."
        ),
        "covered_claim_indices": [0],
    })
    correct_rejected = json.dumps({
        "rejected": (
            "Because atomicity treats all operations as one unit, they all "
            "fail together and therefore no partial update remains."
        ),
        "distorted_claim_index": 0,
        "error_mechanism": "partial-success preservation",
        "faulty_step": "Keep earlier successful operations after a later failure.",
        "causal_rationale": (
            "The learner mistakes execution order for permission to retain "
            "partial transactional state."
        ),
    })
    base = _Base([
        _dpo_plan(), chosen,
        correct_rejected, correct_rejected, correct_rejected,
    ])
    with pytest.raises(SynthesisProviderError) as caught:
        StagedSynthesisProvider(base).paraphrase_preference({}, _chunk())
    assert caught.value.details["validation_error"] in {
        "rejected does not express declared faulty_step",
        "rejected does not locally contrast the chosen answer",
        "rejected is supported by factual evidence",
    }


def test_strict_schemas_use_xgrammar_supported_json_schema_subset():
    provider = StagedSynthesisProvider(_Base([]))
    schemas = [
        provider._realization_schema(("prompt", "completion"), [0, 1]),
    ]
    supported = {
        "type", "additionalProperties", "required", "properties", "items",
        "enum", "const", "uniqueItems", "minimum", "minItems", "maxItems",
        "minLength", "maxLength",
    }

    def assert_subset(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key in supported or key in {
                    "prompt", "completion", "covered_claim_indices",
                }
                assert_subset(child)
        elif isinstance(value, list):
            for child in value:
                assert_subset(child)

    for schema in schemas:
        assert_subset(schema)
        __import__("jsonschema").Draft202012Validator.check_schema(schema)


def test_full_objective_contract_is_enforced_at_plan_and_realization():
    chunk = _chunk()
    chunk["text"] += (
        " Given a failed transaction, atomic rollback prevents partial updates "
        "without omitting rollback effects."
    )
    chunk["synthesis_focus_objective"].update({
        "bloom_verb": "differentiate",
        "action_object": "atomic and partial updates",
        "condition": "given a failed transaction",
        "degree": "without omitting rollback effects",
    })
    valid_plan = json.loads(_sft_plan())
    valid_plan["learner_task"] = (
        "Differentiate atomic and partial updates given a failed transaction "
        "without omitting rollback effects."
    )
    missing_contract = json.loads(_sft_plan())
    missing_contract["learner_task"] = "Analyze atomicity."
    valid_realization = json.dumps({
        "prompt": (
            "Differentiate atomic and partial updates given a failed transaction "
            "without omitting rollback effects."
        ),
        "completion": "Atomicity prevents partial updates.",
        "covered_claim_indices": [0],
    })
    base = _Base([
        json.dumps(missing_contract), json.dumps(valid_plan), valid_realization,
    ])
    StagedSynthesisProvider(base).paraphrase_instruction({}, chunk)
    assert "omits canonical bloom_verb" in base.prompts[1][-1]["content"]

    bad_realization = json.loads(valid_realization)
    bad_realization["prompt"] = "Analyze transaction atomicity."
    base = _Base([
        json.dumps(valid_plan),
        json.dumps(bad_realization),
        valid_realization,
    ])
    StagedSynthesisProvider(base).paraphrase_instruction({}, chunk)
    assert "omits canonical bloom_verb" in base.prompts[2][-1]["content"]


def test_package_a_raised_answer_caps_accept_deep_worked_answers():
    # PACKAGE A regression test: 600→1200 char caps + 3→5 supported_claims.
    # Verify that completion answers up to 1200 chars pass validation,
    # and that prompts remain capped at 400 chars (unchanged).

    # Test 1: Completion at new 1200-char cap is accepted
    prompt = "P" * 400
    completion_at_limit = json.dumps({
        "prompt": prompt,
        "completion": "C" * 1200,
        "covered_claim_indices": [0],
    })
    accepted = StagedSynthesisProvider(
        _Base([_sft_plan(), completion_at_limit])
    ).paraphrase_instruction({}, _chunk())
    assert len(accepted["completion"]) == 1200

    # Test 2: Completion over 1200-char limit is rejected and triggers repair
    over_limit_completion = json.dumps({
        "prompt": prompt,
        "completion": "C" * 1201,
        "covered_claim_indices": [0],
    })
    base = _Base([_sft_plan(), over_limit_completion, completion_at_limit])
    repaired = StagedSynthesisProvider(base).paraphrase_instruction({}, _chunk())
    assert len(repaired["completion"]) == 1200
    # Verify the rejection message shows the new 1..1200 bound
    assert "1..1200" in base.prompts[2][-1]["content"]

    # Test 3: Prompt is still bounded at 400 (unchanged by PACKAGE A)
    assert len(prompt) == 400
