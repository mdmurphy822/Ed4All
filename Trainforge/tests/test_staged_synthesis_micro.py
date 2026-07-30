from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import Trainforge.generators.staged_synthesis_micro as micro_module
from Trainforge.generators._synthesis_common import SynthesisProviderError
from lib.ontology.misconception_id import canonical_mc_id
from Trainforge.generators.staged_synthesis_micro import (
    MICRO_CONTRACT_VERSION,
    MICRO_COMPLETION_CAP_CANDIDATES,
    MICRO_DEFAULT_COMPLETION_CAP,
    MICRO_DPO_PROJECTION,
    MICRO_GATE_D_EXECUTION_POLICY,
    MICRO_GATE_D_STAGE_LIMITS,
    MICRO_LEGACY_EXECUTION_POLICY,
    MICRO_RELEASE_CONTRACT_VERSION,
    MICRO_SFT_PROJECTION,
    MICRO_STAGE_MAX_TOKENS,
    MICRO_STAGE_A_SEED_CONTRACT_VERSION,
    MICRO_STAGE_D_REALIZATION_VIEW_VERSION,
    MICRO_STAGE_D_REALIZATION_CONTRACT_VERSION,
    MICRO_STAGE_D_PROMPT_CONTRACT_VERSION,
    MICRO_STAGE_D_PROMPT_MAX_CHARS,
    MICRO_STAGE_E_AUTHORITY_CONTRACT_VERSION,
    MICRO_STAGE_TOKEN_BUDGET_VERSION,
    MicroResumeStore,
    MicroStagedSynthesisProvider,
    assemble_claims,
    assemble_claim_realizations,
    assemble_objective_execution,
    compose_bounded_learner_task,
    deterministic_stage_a_task,
    deterministic_scenario_givens,
    immutable_artifact_error,
    micro_contract_components,
    micro_contract_fingerprint,
    micro_preference_eligibility,
    staged_synthesis_micro_v1_enabled,
    stage_d_realization_view,
    deterministic_stage_d_prompt,
    unnecessary_generated_givens_error,
    validate_typed_givens,
    verify_expected_input_hashes,
    _leakage_error,
    _allowed_realization_numeric_inventory,
    _claim_checklist,
    _realization_numeric_error,
    _stage_d_realization_schema,
    _stage_e_authority,
    _stage_e_exact_faulty_step_proof,
    _stage_e_selection_schema,
    _resolve_stage_e_selection,
    _scoped_coverage_error,
    _task_numeric_error,
    _prompt_numeric_error,
)
from Trainforge.synthesize_training import build_parser
from Trainforge.generators.staged_synthesis_provider import (
    _coverage_units,
    _operational_condition_error,
    _relation_operator_mutation,
    affine_two_line_relation_proof,
)
from Trainforge.generators.objective_execution_contract import (
    derive_objective_requirements,
)


def test_authenticated_relation_operator_mutation_fails_closed() -> None:
    assert _relation_operator_mutation(
        "The false relation is 0 ≠ -5.", "The false relation is 0 = -5.",
    )
    assert _relation_operator_mutation(
        "The false relation is 0 ≠ -5.",
        "The elimination result is inconsistent.",
    ) is None


def test_affine_two_line_proof_is_exact_and_fingerprinted() -> None:
    source = (
        "<p><code>x + y = 2</code></p><p><code>2x + 2y = 4</code></p>"
        "<p>The equations describe the same line.</p>"
    )
    proof = affine_two_line_relation_proof(
        source,
        "The equations represent the same line and have infinitely many solutions.",
    )
    assert proof is not None
    assert proof["relation"] == "same_line"
    assert len(proof["proof_sha256"]) == 64
    assert affine_two_line_relation_proof(
        "Convert 3x - 2y = 4 to y = (3/2)x - 2.",
        "The equations represent the same line and have infinitely many solutions.",
    ) is not None
    assert affine_two_line_relation_proof(
        source, "The equations have no solution.",
    ) is None


def _bounded_card(action_object: str = "relationships") -> dict:
    return {
        "id": "co-generic",
        "bloom_level": "analyze",
        "bloom_verb": "analyze",
        "action_object": action_object,
        "conditions": ["given source evidence"],
        "content_obligations": [action_object],
        "content_obligation_anchors": [],
        "performance_criteria": ["with every relation justified"],
    }


def _given(value: str = "12") -> dict:
    return {
        "symbol": "n",
        "value": value,
        "unit": "items",
        "role": "count",
        "synthetic": True,
        "provenance": "generated",
    }


def _claim(*, text: str = "The product is commutative.", block: str = "b-1") -> dict:
    return {
        "claim": text,
        "evidence_quote": text,
        "source_block_id": block,
    }


def _gate_d_binding(seed: int = 17) -> dict:
    candidate = {
        "schema": "ed4all.training-synthesis-gate-d-subset.v1",
        "full_manifest_sha256": "1" * 64,
        "eligibility_sha256": "2" * 64,
        "ordered_identity_sha256": "3" * 64,
        "go_artifact_sha256": "4" * 64,
        "go_decision_sha256": "5" * 64,
        "strict_dialect_capability_sha256": "6" * 64,
        "synthesis_seed": seed,
        "run_id": "generic-gate-d-run",
        "output_dir": "generic-gate-d-output",
        "row_count": 1,
        "row": {"position": 0, "kind": "preference"},
    }
    candidate["subset_sha256"] = hashlib.sha256(
        json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return candidate


def _stage_d_artifact() -> dict:
    return assemble_claims(
        [
            {
                "claim": (
                    "Total separation equals the sum of the distances each "
                    "object travels."
                ),
                "evidence_quote": (
                    "When objects move in opposite directions, their total "
                    "separation distance equals the sum of the distances each "
                    "travels."
                ),
                "source_block_id": "generic-source-1",
                "source_role": "evidence",
                "source_polarity": "factual",
            },
            {
                "claim": (
                    "Relative speed equals the sum of the individual speeds."
                ),
                "evidence_quote": (
                    "When two objects move in opposite directions, their "
                    "relative speed is the sum of their individual speeds."
                ),
                "source_block_id": "generic-source-2",
                "source_role": "evidence",
                "source_polarity": "factual",
            },
        ],
        objective_id="co-generic",
        learner_task="Analyze opposite-direction motion.",
        generated_givens=[],
    )


class _StageDScore:
    def score_pair(self, *, premise, hypothesis):
        conflict = (
            "difference of the individual speeds" in hypothesis.casefold()
            or "does not equal the sum" in hypothesis.casefold()
        )
        return SimpleNamespace(
            entailment=0.01 if conflict else 0.99,
            contradiction=0.99 if conflict else 0.01,
        )


class _StageDBase:
    _capture = None
    _model = "test-model"
    _provider_name = "local"
    _provenance_provider = "local"
    _plan_nli_scorer = _StageDScore()


class _StageDProbe(MicroStagedSynthesisProvider):
    def __init__(self, response):
        super().__init__(_StageDBase(), synthesis_seed=0)
        self.response = response
        self.calls = 0
        self.validation_errors = []

    def _call_stage(self, **kwargs):
        self.calls += 1
        error = kwargs["validator"](self.response)
        self.validation_errors.append(error)
        if error:
            raise SynthesisProviderError(error, code="stage_d_probe_invalid")
        return SimpleNamespace(value=dict(self.response))


def test_micro_flag_is_explicit_default_off_and_garbage_off():
    assert not staged_synthesis_micro_v1_enabled({})
    assert not staged_synthesis_micro_v1_enabled(
        {"TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1": "garbage"}
    )
    assert staged_synthesis_micro_v1_enabled(
        {"TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1": "yes"}
    )


def test_cli_exposes_explicit_micro_contract_without_changing_legacy_default():
    parser = build_parser()
    legacy = parser.parse_args(["--corpus", "course", "--course-code", "GEN-1"])
    micro = parser.parse_args([
        "--corpus", "course", "--course-code", "GEN-1",
        "--synthesis-contract", "micro-v1",
    ])
    assert legacy.synthesis_contract is None
    assert micro.synthesis_contract == "micro-v1"


def test_contract_fingerprint_binds_every_schema_prompt_and_gate():
    components = micro_contract_components()
    assert components["version"] == MICRO_CONTRACT_VERSION
    assert components["release_contract_version"] == MICRO_RELEASE_CONTRACT_VERSION
    assert components["projection_contracts"] == {
        "instruction": MICRO_SFT_PROJECTION,
        "preference": MICRO_DPO_PROJECTION,
    }
    assert components["execution_policy_contract"] == {
        "default": MICRO_LEGACY_EXECUTION_POLICY,
        "trusted_harness_selectable": MICRO_GATE_D_EXECUTION_POLICY,
        "gate_d_limits": MICRO_GATE_D_STAGE_LIMITS,
    }
    assert set(components["stages"]) == set("ABCDEFQ")
    assert set(components["schemas_sha256"]) == {
        "task", "claim", "sft", "chosen", "misconception", "rejected"
    }
    assert set(components["systems_sha256"]) == {"A", "B", "D", "E", "F"}
    assert components["entailment_floor"] == 0.70
    assert components["contradiction_ceiling"] == 0.50
    assert components["stage_token_budget"] == {
        "version": MICRO_STAGE_TOKEN_BUDGET_VERSION,
        "max_tokens": MICRO_STAGE_MAX_TOKENS,
        "deterministic_stages": ["A", "C"],
    }
    assert len(components["router_sha256"]) == 64
    assert set(components["validator_sha256"]) == {
        "typed_givens", "given_necessity", "immutable_artifact",
        "numeric_closed_world", "claim", "prompt_numeric", "leakage",
    }
    assert set(components["implementation_sha256"]) == {
        "resume_store", "journal_dispatch", "stop_poll", "task_design",
        "stage_budget_dispatch", "execution_policy", "claim_slot", "assembly_router",
        "realization", "instruction_flow", "preference_flow",
    }
    assert len(micro_contract_fingerprint()) == 64


def test_expected_manifest_and_eligibility_hashes_fail_closed_before_calls(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    eligibility = tmp_path / "eligibility.json"
    manifest.write_bytes(b'{"unit":"generic"}\n')
    eligibility.write_bytes(b'{"eligible":true}\n')
    expected_manifest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected_eligibility = hashlib.sha256(eligibility.read_bytes()).hexdigest()
    assert verify_expected_input_hashes(
        manifest_path=manifest,
        eligibility_path=eligibility,
        expected_manifest_sha256=expected_manifest,
        expected_eligibility_sha256=expected_eligibility,
    ) == {
        "manifest_sha256": expected_manifest,
        "eligibility_sha256": expected_eligibility,
    }
    with pytest.raises(SynthesisProviderError) as caught:
        verify_expected_input_hashes(
            manifest_path=manifest,
            eligibility_path=eligibility,
            expected_manifest_sha256="0" * 64,
            expected_eligibility_sha256=expected_eligibility,
        )
    assert caught.value.code == "staged_micro_input_identity_mismatch"


@pytest.mark.parametrize("value", ["0", "-7", "3.1415", ".5"])
def test_typed_generated_givens_accept_decimal_numeric_fuzz(value):
    assert validate_typed_givens([_given(value)]) is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"synthetic": False},
        {"provenance": "source"},
        {"value": "NaN"},
        {"value": "1e9"},
        {"symbol": "bad symbol"},
        {"unit": ""},
    ],
)
def test_typed_generated_givens_reject_invalid_numeric_or_provenance(mutation):
    given = {**_given(), **mutation}
    assert validate_typed_givens([given])


@pytest.mark.parametrize(
    "value",
    [
        "9" * 33,
        "1000000000001",
        "1234567890123456",
        "0.123456789",
        "1+2",
    ],
)
def test_typed_generated_givens_reject_runaway_magnitude_precision_and_text(value):
    assert validate_typed_givens([_given(value)])


@pytest.mark.parametrize(
    ("role", "unit"),
    [
        ("count", "seconds"),
        ("time", "items"),
        ("distance", "minutes"),
        ("rate", "meters"),
    ],
)
def test_typed_generated_givens_reject_incoherent_role_and_unit(role, unit):
    assert validate_typed_givens([{**_given(), "role": role, "unit": unit}])


def test_generated_givens_are_empty_unless_each_scalar_is_task_necessary():
    assert unnecessary_generated_givens_error(
        "Compare the two relations.", [],
    ) is None
    assert unnecessary_generated_givens_error(
        "For n = 12 items, determine the total.", [_given()],
    ) is None
    assert unnecessary_generated_givens_error(
        "Determine the total.", [_given()],
    )


def test_opaque_given_label_is_metadata_when_scalar_is_used():
    given = {**_given("12"), "symbol": "c2", "unit": "none"}
    assert validate_typed_givens([given]) is None
    assert unnecessary_generated_givens_error(
        "Compare the equations 2x + 3y = 6 and 4x + 6y = 12.",
        [given],
    ) is None


def test_generated_scalar_must_be_used_and_named_symbol_must_match():
    assert unnecessary_generated_givens_error(
        "Compare the two equations.", [{**_given("12"), "symbol": "c2"}],
    )
    assert unnecessary_generated_givens_error(
        "Let n = 13, then compare that result with 12 items.", [_given("12")],
    )
    assert unnecessary_generated_givens_error(
        "For n = 12 items, determine the total.", [_given("12")],
    ) is None


def test_live_stage_a_response_replays_under_scalar_authority_contract():
    """Regression provenance: functional-v1.3 Gate-D retry1 Stage A."""
    matches = list(
        (Path(__file__).resolve().parents[2] / "plans/release-evidence").rglob(
            "response-5902498ad5f323ad2636c3358b769af873eb4a9950ca7cd9f2d098148821bb3e.txt"
        )
    )
    if not matches:
        pytest.skip("ignored live response evidence is unavailable")
    assert len(matches) == 1
    path = matches[0]
    response = json.loads(path.read_text(encoding="utf-8"))
    assert validate_typed_givens(response["generated_givens"]) is None
    assert unnecessary_generated_givens_error(
        response["learner_task"], response["generated_givens"],
    ) is None


def test_canary_002_stage_a_response_replays_as_vacuous_condition_failure():
    """Regression provenance: functional-v1.3 canary-002 Stage A."""
    matches = list(
        (Path(__file__).resolve().parents[2] / "plans/release-evidence").rglob(
            "response-2677726b97847700208b494e005bd1a5a9810fb6bd577bcb317fc5930aa78ea6.txt"
        )
    )
    if not matches:
        pytest.skip("ignored live response evidence is unavailable")
    assert len(matches) == 1
    response = json.loads(matches[0].read_text(encoding="utf-8"))
    assert _operational_condition_error(
        response["learner_task"],
        "after converting to slope-intercept form",
    )


def test_canary004_stage_d_numeric_failure_replays_and_corrects():
    """Regression provenance: functional-v1.3 canary004 Stage D."""
    root = Path(__file__).resolve().parents[2] / "plans/release-evidence"
    response_matches = list(root.rglob(
        "response-548c0b1cecfdc6bac72c16c920953e5cf23a7f4663f33204ca4c3f5335a8f0b7.txt"
    ))
    prompt_matches = list(root.rglob(
        "prompt-2d90afe1b38cccfcd5c8643d98825b04ec89b09f6078d252650a416f6ae70084.txt"
    ))
    if not response_matches or not prompt_matches:
        pytest.skip("ignored live Stage-D evidence is unavailable")
    assert len(response_matches) == len(prompt_matches) == 1
    initial = json.loads(response_matches[0].read_text(encoding="utf-8"))
    messages = json.loads(prompt_matches[0].read_text(encoding="utf-8"))
    artifact_text = messages[1]["content"].split(
        "IMMUTABLE_ARTIFACT=", 1,
    )[1].split("\nReturn only", 1)[0]
    artifact = json.loads(artifact_text)
    allowed = _allowed_realization_numeric_inventory(artifact)
    assert allowed == ["-2", "2", "3"]
    error = _realization_numeric_error(
        initial, artifact=artifact, answer_field="chosen",
    )
    assert "illegal numeric literals=['4', '8', '1']" in error
    assert "['/prompt', '/chosen']" in error
    corrected = {
        "prompt": (
            "Given two equations in different forms, convert both to the same "
            "linear representation and compare their slopes and intercepts."
        ),
        "chosen": initial["chosen"],
    }
    assert _realization_numeric_error(
        corrected, artifact=artifact, answer_field="chosen",
    ) is None


def test_canary005_stage_a_numeric_failure_has_joint_repair_scope():
    """Regression provenance: functional-v1.3 canary005 Stage A."""
    root = Path(__file__).resolve().parents[2] / "plans/release-evidence"
    response_matches = list(root.rglob(
        "response-acbf2beb7b169b3aba52f2c63ed04bcf962b73e13b8c6ebed1abe2bb0326b696.txt"
    ))
    if not response_matches:
        pytest.skip("ignored live Stage-A evidence is unavailable")
    assert len(response_matches) == 1
    initial = json.loads(response_matches[0].read_text(encoding="utf-8"))
    artifact = {
        "claims": [{
            "claim": "Use coefficients 2, 3, and 4.",
            "evidence_quote": "Use coefficients 2, 3, and 4.",
        }],
        "generated_givens": [],
    }
    error = _task_numeric_error(initial["learner_task"], artifact)
    assert "illegal numeric literals=['6', '6', '15']" in error
    assert "['/learner_task', '/generated_givens']" in error
    assert "allowed numeric literals=['2', '3', '4']" in error


def test_canary006_stage_d_missing_claim_replay_is_scoped_and_correctable():
    """Regression provenance: functional-v1.3 canary006 Stage D."""
    root = Path(__file__).resolve().parents[2] / "plans/release-evidence"
    prompt_matches = list(root.rglob(
        "prompt-12ff4464fa780f4b6228ad96a5ef7d9e019dcad099db699dc1c2636b416e9a5d.txt"
    ))
    response_matches = list(root.rglob(
        "response-473582244ea53c13f883bea841ad4b5418942b80dbd566a132f8b589b87c468a.txt"
    ))
    if not prompt_matches or not response_matches:
        pytest.skip("ignored live Stage-D evidence is unavailable")
    assert len(prompt_matches) == len(response_matches) == 1
    messages = json.loads(prompt_matches[0].read_text(encoding="utf-8"))
    artifact_text = messages[1]["content"].split(
        "IMMUTABLE_ARTIFACT=", 1,
    )[1].split("\nALLOWED_NUMERIC_LITERALS", 1)[0]
    artifact = json.loads(artifact_text)
    omitted = json.loads(response_matches[0].read_text(encoding="utf-8"))
    checklist = _claim_checklist(artifact)
    assert checklist[0]["claim_id"] == "2737cba6290c73ab"
    assert omitted["chosen"] == "infinitely many"

    class _CoverageNli:
        @staticmethod
        def score_pair(*, premise, hypothesis):
            complete = premise.strip().lower() != "infinitely many"
            return type("_Score", (), {
                "entailment": 0.99 if complete else 0.013,
                "contradiction": 0.006,
            })()

    base = type(
        "_Base", (), {
            "_capture": None,
            "_model": "test",
            "_provider_name": "local",
            "_plan_nli_scorer": _CoverageNli(),
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    plan = {
        "supported_claims": [{
            "claim": item["claim"],
            "evidence_quote": item["evidence_quote"],
        } for item in artifact["claims"]],
        "generated_givens": artifact["generated_givens"],
        "learner_task": artifact["learner_task"],
    }
    raw_error = provider._semantic_coverage_error(
        omitted["chosen"], plan, prompt=omitted["prompt"],
    )
    assert raw_error.startswith("missing atomic claim 0:")
    scoped = _scoped_coverage_error(
        raw_error, artifact=artifact, answer_field="chosen",
    )
    assert "missing_claim_id='2737cba6290c73ab'" in scoped
    assert "required_entailment=explicitly realize immutable claim 0" in scoped
    assert "affected JSON pointers=['/chosen']" in scoped
    corrected = (
        "The equations have equal slopes and equal y-intercepts, so the lines "
        "are coincident and the system has infinitely many solutions."
    )
    assert provider._semantic_coverage_error(
        corrected, plan, prompt=omitted["prompt"],
    ) is None


def test_canary009_stage_d_grounded_relation_short_circuits_low_nli():
    """Regression provenance: functional-v1.3 canary009 Stage D."""
    root = Path(__file__).resolve().parents[2] / "plans/release-evidence"
    prompt_matches = list(root.rglob(
        "prompt-1af48fd85b52f1e95e979177776fbe57185c763a4004ef95f3f01dc3a2338300.txt"
    ))
    response_matches = list(root.rglob(
        "response-98efb639749270493c97cc783ceb9991cd580d25e683c3ee89e4ebd01e63be62.txt"
    ))
    if not prompt_matches or not response_matches:
        pytest.skip("ignored live Stage-D evidence is unavailable")
    assert len(prompt_matches) == len(response_matches) == 1
    messages = json.loads(prompt_matches[0].read_text(encoding="utf-8"))
    artifact_text = messages[1]["content"].split(
        "IMMUTABLE_ARTIFACT=", 1,
    )[1].split("\nCLAIM_CHECKLIST=", 1)[0]
    artifact = json.loads(artifact_text)
    d1 = json.loads(response_matches[0].read_text(encoding="utf-8"))
    relation = (
        "When two objects move in opposite directions, their relative speed "
        "is the sum of their individual speeds"
    )
    assert relation in d1["chosen"]

    class _CoverageNli:
        calls = []

        @classmethod
        def score_pair(cls, *, premise, hypothesis):
            cls.calls.append((premise, hypothesis))
            is_relation_unit = hypothesis == relation
            return type("_Score", (), {
                "entailment": 0.333 if is_relation_unit else 0.99,
                "contradiction": 0.204 if is_relation_unit else 0.006,
            })()

    base = type(
        "_Base", (), {
            "_capture": None,
            "_model": "test",
            "_provider_name": "local",
            "_plan_nli_scorer": _CoverageNli(),
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    plan = {
        "supported_claims": [{
            "claim": item["claim"],
            "evidence_quote": item["evidence_quote"],
        } for item in artifact["claims"]],
        "generated_givens": artifact["generated_givens"],
        "learner_task": artifact["learner_task"],
    }
    assert provider._semantic_coverage_error(
        d1["chosen"], plan, prompt=d1["prompt"],
    ) is None
    assert not any(
        premise == artifact["claims"][1]["claim"] and hypothesis == relation
        for premise, hypothesis in _CoverageNli.calls
    )

    d2_matches = list(root.rglob(
        "response-bc9efe75739add4ccb78d66b4aa8828cdf29bc0ccb3c5c76417c3e971ff582ac.txt"
    ))
    assert len(d2_matches) == 1
    d2 = json.loads(d2_matches[0].read_text(encoding="utf-8"))
    error = provider._semantic_coverage_error(
        d2["chosen"], plan, prompt=d2["prompt"],
    )
    assert error.startswith("missing atomic claim 1:")

    genuine = _scoped_coverage_error(
        "substantive output unit 2 is ungrounded: 'A novel assertion'; "
        "missing_obligation=remove or ground only this substantive unit",
        artifact=artifact,
        answer_field="chosen",
    )
    assert "affected JSON pointers=['/chosen']" in genuine
    assert "repair_scope=edit only the named genuinely unsupported clause" in genuine
    assert "do not delete any deterministically proven immutable claim" in genuine
    assert artifact["claims"][0]["stable_id"] in genuine
    assert artifact["claims"][1]["stable_id"] in genuine


def test_canary007_stage_a_normalizes_exact_live_scenario_numbers():
    """Regression provenance: functional-v1.3 canary007 Stage A."""
    root = Path(__file__).resolve().parents[2] / "plans/release-evidence"
    a1_matches = list(root.rglob(
        "response-a863155408d3b270689fce77ef00a08a45bd8aff4a9ee7f8647023d3b561bd8a.txt"
    ))
    a2_matches = list(root.rglob(
        "response-be9a986815a397b4b558216d86db9559e4c1919be5aafbe5c99dfb5669f48680.txt"
    ))
    if not a1_matches or not a2_matches:
        pytest.skip("ignored live Stage-A evidence is unavailable")
    assert len(a1_matches) == len(a2_matches) == 1
    a1 = json.loads(a1_matches[0].read_text(encoding="utf-8"))
    a2 = json.loads(a2_matches[0].read_text(encoding="utf-8"))
    givens, evidence, error = deterministic_scenario_givens(
        a1["learner_task"], allowed_numeric_text="2 3 4",
    )
    assert error is None
    assert [(item["symbol"], item["value"], item["unit"]) for item in givens] == [
        ("generated_given_0", "6", "scalar"),
        ("generated_given_1", "12", "scalar"),
    ]
    assert len(givens[0]["occurrences"]) == 2
    assert all(
        occurrence["pointer"] == "/learner_task"
        for item in givens for occurrence in item["occurrences"]
    )
    assert unnecessary_generated_givens_error(a1["learner_task"], givens) is None
    assert evidence["contract_version"] == MICRO_STAGE_A_SEED_CONTRACT_VERSION
    assert "generated_givens" not in __import__(
        "Trainforge.generators.staged_synthesis_micro",
        fromlist=["_TASK_SCHEMA"],
    )._TASK_SCHEMA["properties"]
    assert "generated_givens" in a2
    assert any(symbol in a2["learner_task"] for symbol in ("b1", "b2"))
    assert all(item["symbol"] not in a1["learner_task"] for item in givens)


def test_deterministic_scenario_givens_excludes_structural_numbers():
    givens, evidence, error = deterministic_scenario_givens(
        "Compare x^2 with b1 in equation 3.", allowed_numeric_text="",
    )
    assert error is None
    assert givens == []
    assert evidence["normalized_values"] == []


def test_deterministic_scenario_givens_units_sign_decimal_and_duplicates():
    givens, _evidence, error = deterministic_scenario_givens(
        "Compare -2.5 meters with -2.5 meters over 7 seconds.",
        allowed_numeric_text="",
    )
    assert error is None
    assert [
        (item["value"], item["unit"], item["role"]) for item in givens
    ] == [
        ("-2.5", "meters", "scenario_given"),
        ("7", "seconds", "scenario_given"),
    ]
    assert len(givens[0]["occurrences"]) == 2
    assert validate_typed_givens(givens) is None
    percent, _evidence, error = deterministic_scenario_givens(
        "Compare a 12% rate.", allowed_numeric_text="",
    )
    assert error is None
    assert percent[0]["unit"] == "percent"


@pytest.mark.parametrize("task", [
    "Analyze the system; the solution is 3.",
    "Determine the count; there are 4 solutions.",
])
def test_deterministic_scenario_givens_never_legalizes_numeric_answers(task):
    givens, evidence, error = deterministic_scenario_givens(
        task, allowed_numeric_text="",
    )
    assert givens == []
    assert evidence == {}
    assert "answer-bearing numeric conclusion" in error


def test_deterministic_scenario_givens_rejects_ambiguous_required_unit():
    givens, evidence, error = deterministic_scenario_givens(
        "Compare a rate of 3 per cycle.", allowed_numeric_text="",
    )
    assert givens == []
    assert evidence == {}
    assert "ambiguous required unit context" in error


def test_micro_preference_eligibility_fails_before_any_model_call():
    chunk = {
        "id": "generic-no-misconception",
        "text": (
            "Two equations can be compared by converting them to the same "
            "representation and examining their corresponding properties."
        ),
        "learning_outcome_refs": ["co-1"],
        "bloom_level": "analyze",
        "synthesis_focus_objective": {
            "id": "co-1",
            "statement": "Analyze corresponding equation properties.",
            "bloom_level": "analyze",
        },
    }
    eligibility = micro_preference_eligibility(
        chunk, focus=chunk["synthesis_focus_objective"],
    )
    assert eligibility["eligible"] is False
    assert eligibility["reason"] == "preference_misconception_candidate_missing"
    assert eligibility["telemetry"]["candidate_count"] == 0

    class _NoCallBase:
        _capture = None
        _model = "test"
        _provider_name = "local"
        _plan_nli_scorer = None
        prompts = []

        def _chat_completion_raw_structured(self, *_args, **_kwargs):
            self.prompts.append("unexpected")
            raise AssertionError("preference eligibility must precede model calls")

    base = _NoCallBase()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    with pytest.raises(SynthesisProviderError) as caught:
        provider.paraphrase_preference({}, chunk)
    assert caught.value.code == "preference_misconception_candidate_missing"
    assert caught.value.details["model_calls"] == 0
    assert caught.value.details["eligibility"] == eligibility["telemetry"]
    assert base.prompts == []


def test_micro_preference_eligibility_and_stage_e_share_candidates():
    misconception = "atomicity preserves partial updates"
    correction = "atomicity prevents partial updates"
    canonical_id = canonical_mc_id(misconception, correction, "analyze")
    chunk = {
        "id": "generic-source-backed-misconception",
        "text": (
            "A mistaken view says atomicity preserves partial updates. "
            "The correction is that atomicity prevents partial updates by "
            "making all grouped operations succeed or fail together."
        ),
        "misconceptions": [{
            "id": canonical_id,
            "misconception": misconception,
            "correction": correction,
            "mechanism_evidence": (
                "all grouped operations succeed or fail together"
            ),
        }],
    }
    focus = {
        "id": "co-1",
        "statement": "Analyze how atomicity prevents partial updates.",
        "bloom_level": "analyze",
    }
    eligibility = micro_preference_eligibility(chunk, focus=focus)
    assert eligibility["eligible"] is True
    assert eligibility["reason"] is None
    assert eligibility["telemetry"]["candidate_ids"] == [canonical_id]
    assert eligibility["candidates"][0]["source_role"] == "misconception_claim"
    assert eligibility["candidates"][0]["source_polarity"] == (
        "incorrect_with_correction"
    )
    assert eligibility["candidates"][0]["mechanism_evidence"]


def test_micro_preference_eligibility_reason_codes_missing_correction():
    chunk = {
        "id": "generic-missing-correction",
        "text": "The source states every operation may remain partial.",
        "misconceptions": [{
            "misconception": "every operation may remain partial",
        }],
    }
    result = micro_preference_eligibility(
        chunk,
        focus={
            "id": "co-1",
            "statement": "Analyze operation grouping.",
            "bloom_level": "analyze",
        },
    )
    assert result["reason"] == "preference_misconception_correction_missing"
    assert result["telemetry"]["rejection_counts"]["correction_missing"] == 1


def test_every_micro_schema_string_and_array_is_finitely_bounded():
    from Trainforge.generators import staged_synthesis_micro as micro

    schemas = (
        micro._TASK_SCHEMA,
        micro._CLAIM_SCHEMA,
        micro._SFT_SCHEMA,
        micro._CHOSEN_SCHEMA,
        micro._MISCONCEPTION_SCHEMA,
        micro._REJECTED_SCHEMA,
    )

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "string" and "enum" not in node and "const" not in node:
                assert "maxLength" in node
            if node.get("type") == "array":
                assert "maxItems" in node
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    for schema in schemas:
        visit(schema)


def test_micro_system_directives_explicitly_disable_reasoning():
    from Trainforge.generators import staged_synthesis_micro as micro

    for directive in (
        micro._TASK_SYSTEM,
        micro._CLAIM_SYSTEM,
        micro._REALIZE_SYSTEM,
        micro._MISCONCEPTION_SYSTEM,
        micro._REJECTED_SYSTEM,
    ):
        assert "Thinking is disabled" in directive
        assert "do not emit reasoning" in directive


def test_stage_token_budget_is_positive_bounded_and_fingerprint_authoritative(
    monkeypatch,
):
    assert MICRO_STAGE_MAX_TOKENS == {
        "A": 2048,
        "B": 1536,
        "D": 1536,
        "E": 1280,
        "F": 1024,
    }
    assert all(1 <= value <= 4096 for value in MICRO_STAGE_MAX_TOKENS.values())
    before = micro_contract_fingerprint()
    monkeypatch.setitem(MICRO_STAGE_MAX_TOKENS, "A", 2049)
    assert micro_contract_fingerprint() != before


def test_unknown_or_overridden_micro_stage_budget_fails_closed():
    provider = object.__new__(MicroStagedSynthesisProvider)
    with pytest.raises(SynthesisProviderError) as unknown:
        provider._call_stage(stage="micro_G_unknown")
    assert unknown.value.code == "staged_micro_token_budget_unknown"
    with pytest.raises(SynthesisProviderError) as override:
        provider._call_stage(
            stage="micro_A_task_design", max_output_tokens=800,
        )
    assert override.value.code == "staged_micro_token_budget_override"


@pytest.mark.parametrize("seed", [None, True, -1, 2**64, "7"])
def test_micro_provider_requires_explicit_bounded_integer_seed(seed):
    with pytest.raises(SynthesisProviderError) as caught:
        MicroStagedSynthesisProvider(object(), synthesis_seed=seed)
    assert caught.value.code == "staged_micro_synthesis_seed_invalid"


@pytest.mark.parametrize("cap", [None, True, 599, 601, 2000, "800"])
def test_micro_provider_rejects_unqualified_completion_cap(cap):
    with pytest.raises(SynthesisProviderError) as caught:
        MicroStagedSynthesisProvider(
            object(), synthesis_seed=0, completion_cap=cap,
        )
    assert caught.value.code == "staged_micro_completion_cap_invalid"


def test_candidate_completion_cap_is_schema_bound_and_release_bound():
    artifact = assemble_claims(
        [{
            "claim": "A supported claim.",
            "evidence_quote": "A supported claim.",
            "source_block_id": "source-1",
            "source_role": "evidence",
            "source_polarity": "factual",
        }],
        objective_id="co-generic",
        learner_task="Explain the supported claim.",
        generated_givens=[],
    )
    for cap in MICRO_COMPLETION_CAP_CANDIDATES:
        schema = _stage_d_realization_schema(
            artifact, completion_cap=cap,
        )
        assert schema["properties"]["claim_realizations"]["items"][
            "properties"
        ]["realization"]["maxLength"] == cap

    class Base:
        _model = "generic-model"
        _provider_name = "local"

    default = MicroStagedSynthesisProvider(Base(), synthesis_seed=17)
    candidate = MicroStagedSynthesisProvider(
        Base(), synthesis_seed=17, completion_cap=800,
    )
    assert default._completion_cap == MICRO_DEFAULT_COMPLETION_CAP
    assert default._release_identity()["completion_cap"] == 600
    assert candidate._release_identity()["completion_cap"] == 800
    assert candidate._release_identity() != default._release_identity()


def test_stage_d_objective_schema_and_assembly_bind_steps_result_and_spans():
    contract = derive_objective_requirements({
        "id": "co-generic",
        "statement": "Analyze the supported relationship.",
        "bloom_level": "analyze",
        "bloom_verb": "analyze",
        "action_object": "the supported relationship",
    })
    artifact = assemble_claims(
        [{
            "claim": "The relationship is supported.",
            "evidence_quote": "The relationship is supported.",
            "source_block_id": "source-1",
            "source_role": "evidence",
            "source_polarity": "factual",
        }],
        objective_id="co-generic",
        learner_task="Analyze the supported relationship.",
        generated_givens=[],
        requirement_contract=contract,
    )
    claim_id = artifact["claims"][0]["stable_id"]
    worked = [
        item for item in contract["requirements"]
        if item["kind"] != "result"
    ]
    result = next(
        item for item in contract["requirements"]
        if item["kind"] == "result"
    )
    schema = _stage_d_realization_schema(artifact, completion_cap=800)
    assert schema["required"] == [
        "claim_realizations", "worked_steps", "result",
    ]
    assert schema["properties"]["worked_steps"]["minItems"] == len(worked)
    value = {
        "claim_realizations": [{
            "claim_id": claim_id,
            "realization": "The relationship is supported.",
        }],
        "worked_steps": [{
            "requirement_id": item["requirement_id"],
            "claim_ids": [claim_id],
            "realization": f"Execute {item['kind']} from the supported claim.",
        } for item in worked],
        "result": {
            "requirement_id": result["requirement_id"],
            "claim_ids": [claim_id],
            "realization": "Therefore, the supported relationship holds.",
        },
    }
    assembled = assemble_objective_execution(value, artifact=artifact)
    assert len(assembled["steps"]) == len(worked) + 1
    assert assembled["result"]["requirement_id"] == result["requirement_id"]
    for row in assembled["steps"]:
        start, end = row["completion_span"]
        assert assembled["text"][start:end] == row["realization"]


def test_release_identity_binds_seed_projections_provider_model_and_fingerprint():
    class Base:
        _model = "generic-model"
        _provider_name = "local"

    provider = MicroStagedSynthesisProvider(Base(), synthesis_seed=17)
    identity = provider._release_identity()
    assert identity == {
        "release_contract_version": MICRO_RELEASE_CONTRACT_VERSION,
        "synthesis_contract_version": MICRO_CONTRACT_VERSION,
        "synthesis_contract_sha256": micro_contract_fingerprint(),
        "synthesis_seed": 17,
        "projections": {
            "instruction": MICRO_SFT_PROJECTION,
            "preference": MICRO_DPO_PROJECTION,
        },
        "provider": "local",
        "model": "generic-model",
        "completion_cap": MICRO_DEFAULT_COMPLETION_CAP,
    }
    assert provider._decision_identity_context() == identity
    assert "execution_policy" not in identity
    assert (
        provider.execution_policy_identity()["version"]
        == MICRO_LEGACY_EXECUTION_POLICY
    )


def test_gate_d_policy_requires_verified_binding_and_drifts_release_identity():
    class Base:
        _model = "generic-model"
        _provider_name = "local"

    provider = MicroStagedSynthesisProvider(Base(), synthesis_seed=17)
    legacy = provider._release_identity()
    invalid = _gate_d_binding()
    invalid["subset_sha256"] = "0" * 64
    with pytest.raises(SynthesisProviderError) as caught:
        provider.bind_trusted_gate_d_single_pass(invalid)
    assert caught.value.code == "staged_micro_gate_d_binding_invalid"
    policy = provider.bind_trusted_gate_d_single_pass(_gate_d_binding())
    assert policy["version"] == MICRO_GATE_D_EXECUTION_POLICY
    assert policy["limits"] == MICRO_GATE_D_STAGE_LIMITS
    assert provider._release_identity() != legacy
    with pytest.raises(SynthesisProviderError) as rebound:
        provider.bind_trusted_gate_d_single_pass(_gate_d_binding())
    assert rebound.value.code == "staged_micro_execution_policy_rebind"


def test_gate_d_planned_stage_identities_are_exact_and_c_is_no_call():
    class Base:
        _model = "generic-model"
        _provider_name = "local"

    provider = MicroStagedSynthesisProvider(Base(), synthesis_seed=17)
    with pytest.raises(SynthesisProviderError):
        provider.planned_stage_identities({}, kind="preference")
    provider.bind_trusted_gate_d_single_pass(_gate_d_binding())
    chunk = {
        "id": "generic-chunk",
        "text": (
            "A factual source block supports the objective. A mistaken view "
            "says the relation is absent. The correction is that the source "
            "relation is present."
        ),
        "misconceptions": [{
            "misconception": "the relation is absent",
            "correction": "the source relation is present",
        }],
        "learning_outcome_refs": ["co-1"],
        "bloom_level": "analyze",
        "synthesis_focus_objective": {
            "id": "co-1",
            "statement": "Analyze the source-backed relation.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "action_object": "the source-backed relation",
        },
    }
    planned = provider.planned_stage_identities(chunk, kind="preference")
    assert [(row["family"], row["slot"], row["stage"], row["model_call"])
            for row in planned] == [
        ("A", None, "micro_A_task_design", False),
        ("B", 0, "micro_B_claim_0_attempt_1", True),
        ("C", None, "micro_C_assembly", False),
        ("D", None, "micro_D_dpo_chosen", True),
        ("E", None, "micro_E_misconception_selection", True),
        ("F", None, "micro_F_one_fault_rejected", True),
        ("Q", None, "micro_Q_finalization", False),
    ]
    assert len({row["unit"] for row in planned}) == len(planned)
    assert len({row["execution_policy_sha256"] for row in planned}) == 1
    assert planned[0]["model_call"] is False
    assert planned[0]["input_identity"]["contract_version"] == (
        MICRO_STAGE_A_SEED_CONTRACT_VERSION
    )
    assert planned[0]["input_identity"]["synthesis_seed"] == 17
    assert len(planned[0]["input_identity"]["objective_card_sha256"]) == 64


@pytest.mark.parametrize(
    ("verb", "level", "subject", "condition", "criterion"),
    [
        (
            "analyze", "analyze", "relationships in distributed systems",
            "given two recorded service events",
            "by distinguishing causal order from clock order",
        ),
        (
            "evaluate", "evaluate", "competing interpretations of a poem",
            "using evidence from the supplied passage",
            "by justifying the stronger interpretation",
        ),
        (
            "create", "create", "an accessible navigation design",
            "for keyboard-only interaction",
            "by satisfying the stated interaction constraints",
        ),
    ],
)
def test_stage_a_v3_deterministically_preserves_every_objective_dimension(
    verb, level, subject, condition, criterion,
):
    card = {
        "id": "co-generic",
        "statement": f"{verb.capitalize()} {subject}.",
        "bloom_level": level,
        "bloom_verb": verb,
        "action_object": subject,
        "condition": condition,
        "degree": criterion,
    }
    first = deterministic_stage_a_task(card, synthesis_seed=29)
    second = deterministic_stage_a_task(card, synthesis_seed=29)
    assert first == second
    assert first["generated_givens"] == []
    assert first["_stage_a_deterministic"]["model_calls"] == 0
    assert first["_stage_a_deterministic"]["decision_capture_events"] == 0
    assert first["_stage_a_deterministic"]["contract_sha256"]
    assert first["_normalization_evidence"]["contract_sha256"] == (
        first["_stage_a_deterministic"]["contract_sha256"]
    )
    assert first["_stage_a_deterministic"]["telemetry"] == {
        "deterministic_events": 1,
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    task = first["learner_task"].lower()
    for required in (verb, subject, condition, criterion):
        assert required.lower() in task
    for obligation in first["_objective_card"]["content_obligations"]:
        assert obligation.lower() in task
    assert "source-grounded evidence" in task


def test_canary010_exact_objective_card_is_valid_on_first_zero_call_assembly():
    root = Path(__file__).resolve().parents[2] / "plans/release-evidence"
    matches = list((root / "training-synthesis-functional-v1.3.0"
                    / "02-one-row-canary"
                    / "functional-production-dpo-00183-seed0-010").rglob(
        "prompt-357e74d9e98cc3aa3ce8bdb36bcf4ab6c1753b4ca60f4db63aafc1b6479fdbb3.txt"
    ))
    if not matches:
        pytest.skip("ignored live canary010 Stage-A evidence is unavailable")
    assert len(matches) == 1
    messages = json.loads(matches[0].read_text(encoding="utf-8"))
    objective_text = messages[1]["content"].split(
        "OBJECTIVE=", 1,
    )[1].split("\nReturn objective_id", 1)[0]
    objective = json.loads(objective_text)
    task = deterministic_stage_a_task(objective, synthesis_seed=0)
    assert task["objective_id"] == "co-104"
    assert task["bloom_level"] == "analyze"
    assert task["generated_givens"] == []
    rendered = task["learner_task"].lower()
    for key in (
        "bloom_verb", "action_object",
    ):
        assert objective[key].lower() in rendered
    for key in (
        "conditions", "content_obligations", "performance_criteria",
    ):
        assert all(item.lower() in rendered for item in objective[key])


def test_stage_a_v3_incomplete_card_fails_loud_before_any_model_call():
    with pytest.raises(SynthesisProviderError) as caught:
        deterministic_stage_a_task({
            "id": "co-incomplete",
            "statement": "Analyze the supplied material.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
        }, synthesis_seed=0)
    assert caught.value.code == "staged_micro_A_objective_card_incomplete"
    assert caught.value.details["model_calls"] == 0
    assert caught.value.details["missing_fields"] == ["action_object"]


def test_stage_a_v3_provider_path_never_dispatches_or_emits_fake_capture():
    class Base:
        _model = "generic-model"
        _provider_name = "local"

    provider = MicroStagedSynthesisProvider(Base(), synthesis_seed=11)
    provider._call_stage = lambda **_kwargs: pytest.fail(
        "deterministic Stage A must not dispatch"
    )
    focus = {
        "id": "co-1",
        "statement": "Analyze relationships in distributed systems.",
        "bloom_level": "analyze",
        "bloom_verb": "analyze",
        "action_object": "relationships in distributed systems",
        "condition": "given two recorded service events",
        "degree": "by distinguishing causal order from clock order",
    }
    chunk = {
        "id": "generic-chunk",
        "text": (
            "Relationships in distributed systems can be analyzed from two "
            "recorded service events by distinguishing causal order from "
            "clock order."
        ),
        "learning_outcome_refs": ["co-1"],
        "bloom_level": "analyze",
        "synthesis_focus_objective": focus,
    }
    task = provider._task_design(chunk)
    assert task["_stage_a_deterministic"]["model_calls"] == 0
    assert task["_normalization_evidence"]["provider_decision_capture_id"] == ""
    assert "_decision_capture_id" not in task


def test_stage_d_realization_view_projects_minimum_generic_domain_columns():
    artifact = assemble_claims(
        [{
            "claim": "Atomic execution prevents a partially committed update.",
            "evidence_quote": (
                "A transaction is atomic when all operations commit together "
                "or every operation is rolled back after a failure."
            ),
            "source_block_id": "block-transaction",
            "source_role": "explanation",
            "source_polarity": "factual",
        }],
        objective_id="co-generic",
        learner_task="Analyze how atomic execution prevents partial updates.",
        generated_givens=[],
    )
    view = stage_d_realization_view(artifact)
    serialized = json.dumps(view, sort_keys=True)
    assert view["contract_version"] == MICRO_STAGE_D_REALIZATION_VIEW_VERSION
    assert view["private_artifact_sha256"] == artifact["artifact_sha256"]
    assert view["claims"][0]["source_role"] == "explanation"
    assert view["claims"][0]["source_polarity"] == "factual"
    assert view["claims"][0]["private_evidence_quote_sha256"]
    assert "evidence_quote" not in view["claims"][0]
    assert artifact["claims"][0]["evidence_quote"] not in serialized
    assert artifact["claims"][0]["claim"] in serialized
    assert len(serialized) < len(json.dumps(artifact, sort_keys=True))


def test_canary011_stage_d_leakage_replay_uses_private_quotes_and_can_progress():
    root = (
        Path(__file__).resolve().parents[2]
        / "plans/release-evidence/training-synthesis-functional-v1.3.0"
        / "02-one-row-canary"
        / "functional-production-dpo-00183-seed0-011"
    )
    prompt_path = root / "micro-journals/raw_staged_calls" / (
        "prompt-b8aebacccd61f884a27a346c0517d6a1b04b0beab7b587211132527fc09f1b1d.txt"
    )
    first_path = root / "micro-journals/raw_staged_calls" / (
        "response-49365b1f66a541a42b1ec97914743fa0e793ec8e77b378ae49f81c8cb4ba10f9.txt"
    )
    repeated_path = root / "micro-journals/raw_staged_calls" / (
        "response-5a356b8ec4593b1b9437a16099281fa99d7374fa50d273377cca3f1236fd0f63.txt"
    )
    if not all(path.exists() for path in (prompt_path, first_path, repeated_path)):
        pytest.skip("ignored live canary011 Stage-D evidence is unavailable")
    messages = json.loads(prompt_path.read_text(encoding="utf-8"))
    artifact_text = messages[1]["content"].split(
        "IMMUTABLE_ARTIFACT=", 1,
    )[1].split("\nCLAIM_CHECKLIST=", 1)[0]
    artifact = json.loads(artifact_text)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    repeated = json.loads(repeated_path.read_text(encoding="utf-8"))
    assert first == repeated
    source = " ".join(
        item["evidence_quote"] for item in artifact["claims"]
    )
    leakage = _leakage_error(
        first["chosen"], source=source, artifact=artifact, pointer="/chosen",
    )
    assert "category=private_source_overlap" in leakage
    assert "affected JSON pointers=['/chosen']" in leakage

    view = stage_d_realization_view(artifact)
    serialized = json.dumps(view, sort_keys=True)
    assert all(
        item["evidence_quote"] not in serialized for item in artifact["claims"]
    )
    corrected = (
        "As the objects head away from their shared origin, combine how far "
        "each one travels to obtain the growing gap. The rate at which that "
        "gap opens is found by adding the two speeds rather than subtracting "
        "one from the other."
    )
    assert _leakage_error(
        corrected, source=source, artifact=artifact, pointer="/chosen",
    ) is None

    class _CoverageNli:
        @staticmethod
        def score_pair(*, premise, hypothesis):
            return type("_Score", (), {
                "entailment": 0.99,
                "contradiction": 0.001,
            })()

    base = type(
        "_Base", (), {
            "_capture": None,
            "_model": "test",
            "_provider_name": "local",
            "_plan_nli_scorer": _CoverageNli(),
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    plan = {
        "supported_claims": [{
            "claim": item["claim"],
            "evidence_quote": item["evidence_quote"],
        } for item in artifact["claims"]],
        "generated_givens": artifact["generated_givens"],
        "learner_task": artifact["learner_task"],
    }
    assert provider._semantic_coverage_error(
        corrected, plan, prompt=first["prompt"],
    ) is None


def test_canary012_stage_d_never_scores_standalone_therefore():
    root = (
        Path(__file__).resolve().parents[2]
        / "plans/release-evidence/training-synthesis-functional-v1.3.0"
        / "02-one-row-canary"
        / "functional-production-dpo-00183-seed0-012"
    )
    prompt_path = root / "micro-journals/raw_staged_calls" / (
        "prompt-080967944f6b1a83fc55c79f7da27bf99231db845c12f2f6d06cacbd69bd215e.txt"
    )
    response_path = root / "micro-journals/raw_staged_calls" / (
        "response-5f6a52bbd5a19809a45f194174e96faecef85d2c0d4313cdbbb2339aacdb3d74.txt"
    )
    if not prompt_path.exists() or not response_path.exists():
        pytest.skip("ignored live canary012 Stage-D evidence is unavailable")
    messages = json.loads(prompt_path.read_text(encoding="utf-8"))
    view_text = messages[1]["content"].split(
        "REALIZATION_VIEW=", 1,
    )[1].split("\nThe chosen must", 1)[0]
    view = json.loads(view_text)
    d1 = json.loads(response_path.read_text(encoding="utf-8"))
    assert "Therefore," in d1["chosen"]
    assert "Therefore" not in _coverage_units(d1["chosen"])

    class _CoverageNli:
        calls = []

        @classmethod
        def score_pair(cls, *, premise, hypothesis):
            cls.calls.append((premise, hypothesis))
            return type("_Score", (), {
                "entailment": 0.99,
                "contradiction": 0.001,
            })()

    base = type(
        "_Base", (), {
            "_capture": None,
            "_model": "test",
            "_provider_name": "local",
            "_plan_nli_scorer": _CoverageNli(),
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    plan = {
        "supported_claims": [{
            "claim": item["claim"],
            "evidence_quote": item["claim"],
        } for item in view["claims"]],
        "generated_givens": view["generated_givens"],
        "learner_task": view["learner_task"],
    }
    assert provider._semantic_coverage_error(
        d1["chosen"], plan, prompt=d1["prompt"],
    ) is None
    assert not any(
        hypothesis.strip().casefold() == "therefore"
        for _premise, hypothesis in _CoverageNli.calls
    )


def test_marker_does_not_hide_following_unsupported_clause():
    supported = "A verified checksum detects an altered payload."
    unsupported = "An unrelated satellite controls every checksum."

    class _CoverageNli:
        @staticmethod
        def score_pair(*, premise, hypothesis):
            bad = premise == unsupported
            return type("_Score", (), {
                "entailment": 0.05 if bad else 0.99,
                "contradiction": 0.01,
            })()

    base = type(
        "_Base", (), {
            "_capture": None,
            "_model": "test",
            "_provider_name": "local",
            "_plan_nli_scorer": _CoverageNli(),
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    error = provider._semantic_coverage_error(
        f"{supported} Therefore, {unsupported}",
        {
            "supported_claims": [{
                "claim": supported,
                "evidence_quote": supported,
            }],
            "generated_givens": [],
            "learner_task": "Explain the verified checksum behavior.",
        },
        prompt="Explain the verified checksum behavior.",
    )
    assert "is ungrounded" in error
    assert unsupported in error


@pytest.mark.parametrize(
    "response_sha256",
    [
        "68733c6b2d278ec5acfaa87aba6e121dbd3787dddff6e5140e392709f05c39c1",
        "dd5cfb0faa882dac17f93e7284a4e5d6a7ef3e23309bf4fa9043f7ab9c5f1ce6",
        "c6bd99852478d0131bb1f13521ba0cef5fb17a0b95233075d39c43967405fb0b",
    ],
)
def test_canary013_d1_d2_d3_whole_answer_responses_fail_v2_structure(
    response_sha256,
):
    """The three oscillating canary responses cannot enter the v2 validator."""
    root = Path(__file__).resolve().parents[2] / "plans/release-evidence"
    matches = list(root.rglob(f"response-{response_sha256}.txt"))
    if not matches:
        pytest.skip("ignored canary-013 response evidence is unavailable")
    assert len(matches) == 1
    response = json.loads(matches[0].read_text(encoding="utf-8"))
    assert set(response) == {"prompt", "chosen"}
    probe = _StageDProbe(response)
    with pytest.raises(SynthesisProviderError) as caught:
        probe._realize(
            chunk_id="generic-chunk",
            artifact=_stage_d_artifact(),
            preference=True,
        )
    assert caught.value.code == "stage_d_probe_invalid"
    assert probe.calls == 1
    assert probe.validation_errors == [
        "Stage-D model envelope must contain only claim_realizations; "
        "affected JSON pointers=['/']"
    ]


def test_canary013_v2_realizes_exact_ids_once_and_assembles_in_stage_c_order():
    artifact = _stage_d_artifact()
    ids = [item["stable_id"] for item in artifact["claims"]]
    response = {
        "claim_realizations": [
            {
                "claim_id": ids[0],
                "realization": (
                    "Total separation equals the sum of the distances each "
                    "object travels."
                ),
            },
            {
                "claim_id": ids[1],
                "realization": (
                    "Relative speed equals the sum of the individual speeds."
                ),
            },
        ],
    }
    probe = _StageDProbe(response)
    realized = probe._realize(
        chunk_id="generic-chunk", artifact=artifact, preference=True,
    )
    assert probe.calls == 1
    assert probe.validation_errors == [None]
    assert list(realized["_claim_realization_map"]) == ids
    assert realized["_assembled_realization_provenance"][
        "ordered_claim_ids"
    ] == ids
    first = realized["chosen"].index(response["claim_realizations"][0]["realization"])
    second = realized["chosen"].index(response["claim_realizations"][1]["realization"])
    assert first < second


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            "The lunar archive catalogs unrelated telescope maintenance.",
            "contains obvious unrelated assertion",
        ),
        (
            "Relative speed is the difference of the individual speeds.",
            "does not entail its bound claim",
        ),
        (
            "Relative speed does not equal the sum of the individual speeds.",
            "does not entail its bound claim",
        ),
        (
            "Relative speed equals 3 times an individual speed.",
            "contains illegal numeric literals=['3']",
        ),
        (
            (
                "When two objects move in opposite directions, their relative "
                "speed is the sum of their individual speeds."
            ),
            "provider output contains distinctive source span",
        ),
    ],
)
def test_canary013_v2_rejects_unsupported_operator_negation_and_leakage(
    mutation, expected,
):
    artifact = _stage_d_artifact()
    ids = [item["stable_id"] for item in artifact["claims"]]
    response = {
        "claim_realizations": [
            {
                "claim_id": ids[0],
                "realization": artifact["claims"][0]["claim"],
            },
            {"claim_id": ids[1], "realization": mutation},
        ],
    }
    probe = _StageDProbe(response)
    with pytest.raises(SynthesisProviderError) as caught:
        probe._realize(
            chunk_id="generic-chunk", artifact=artifact, preference=True,
        )
    assert caught.value.code == "stage_d_probe_invalid"
    assert expected in probe.validation_errors[0]
    assert "/claim_realizations/1/realization" in probe.validation_errors[0]


@pytest.mark.parametrize(
    "claim_ids",
    [
        lambda ids: [ids[1], ids[0]],
        lambda ids: [ids[0], ids[0]],
        lambda ids: [ids[0], "f" * 16],
        lambda ids: [ids[0]],
    ],
)
def test_canary013_v2_fails_closed_on_reordered_duplicate_unknown_or_missing_ids(
    claim_ids,
):
    artifact = _stage_d_artifact()
    ids = [item["stable_id"] for item in artifact["claims"]]
    response = {
        "claim_realizations": [
            {"claim_id": claim_id, "realization": "Bound realization."}
            for claim_id in claim_ids(ids)
        ],
    }
    probe = _StageDProbe(response)
    with pytest.raises(SynthesisProviderError):
        probe._realize(
            chunk_id="generic-chunk", artifact=artifact, preference=True,
        )
    assert "exactly one known row" in probe.validation_errors[0]


def test_canary013_repair_cannot_mutate_valid_item_or_prompt():
    artifact = _stage_d_artifact()
    ids = [item["stable_id"] for item in artifact["claims"]]
    first = {
        "claim_realizations": [
            {"claim_id": ids[0], "realization": artifact["claims"][0]["claim"]},
            {
                "claim_id": ids[1],
                "realization": (
                    "The lunar archive catalogs unrelated telescope maintenance."
                ),
            },
        ],
    }
    second = {
        "claim_realizations": [
            {"claim_id": ids[0], "realization": "Mutated valid realization."},
            {"claim_id": ids[1], "realization": artifact["claims"][1]["claim"]},
        ],
    }

    class _RepairProbe(_StageDProbe):
        def __init__(self):
            super().__init__(first)

        def _call_stage(self, **kwargs):
            self.calls += 1
            for candidate in (first, second):
                error = kwargs["validator"](candidate)
                self.validation_errors.append(error)
            raise SynthesisProviderError(
                self.validation_errors[-1], code="stage_d_probe_invalid",
            )

    probe = _RepairProbe()
    with pytest.raises(SynthesisProviderError):
        probe._realize(
            chunk_id="generic-chunk", artifact=artifact, preference=True,
        )
    assert "contains obvious unrelated assertion" in probe.validation_errors[0]
    assert "mutated or reordered frozen state" in probe.validation_errors[1]


@pytest.mark.parametrize(("width", "leaks"), [(49, False), (50, True)])
def test_canary014_stage_b_public_claim_verbatim_boundary(width, leaks):
    distinctive = "".join(chr(ord("a") + (index % 26)) for index in range(width))
    block = {
        "block_id": "generic-source",
        "text": f"Private evidence states {distinctive} and remains exact.",
    }
    value = {
        "claim": distinctive,
        "evidence_quote": f"Private evidence states {distinctive}",
        "source_block_id": "generic-source",
    }

    class _EntailingScore:
        @staticmethod
        def score_pair(**_kwargs):
            return SimpleNamespace(entailment=0.99, contradiction=0.0)

    base = _StageDBase()
    base._plan_nli_scorer = _EntailingScore()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    error = provider._validate_claim(value, block=block)
    assert bool(error) is leaks
    if leaks:
        assert "distinctive private-source span" in error
        assert "affected JSON pointers=['/claim']" in error


def test_canary014_stage_b_repair_freezes_exact_private_provenance():
    quote = (
        "When objects move in opposite directions, their total separation "
        "distance is the sum of the distances each travels."
    )
    alternate = "Opposite motion provides a second exact private sentence."
    block = {
        "block_id": "generic-source",
        "text": f"{quote} {alternate}",
    }
    guard = {}
    provider = MicroStagedSynthesisProvider(_StageDBase(), synthesis_seed=0)
    leaking = {
        "claim": quote,
        "evidence_quote": quote,
        "source_block_id": "generic-source",
    }
    error = provider._validate_claim(
        leaking, block=block, private_guard=guard,
    )
    assert "repair only /claim" in error
    assert guard == {
        "source_block_id": "generic-source",
        "evidence_quote": quote,
    }
    mutation = {
        "claim": "Opposite motion has a supported outcome.",
        "evidence_quote": alternate,
        "source_block_id": "generic-source",
    }
    error = provider._validate_claim(
        mutation, block=block, private_guard=guard,
    )
    assert "mutated frozen private provenance fields" in error
    assert "editable JSON pointers=['/claim']" in error


def test_canary014_leakage_safe_motion_claim_is_exact_stage_d_zero_nli():
    quote = (
        "When objects move in opposite directions, their total separation "
        "distance is the sum of the distances each travels."
    )
    safe_claim = (
        "The distance between two objects increases by the sum of how far each "
        "has traveled when they move away from each other."
    )
    block = {"block_id": "generic-source", "text": quote}

    class _ClaimScore:
        calls = 0

        @classmethod
        def score_pair(cls, **_kwargs):
            cls.calls += 1
            return SimpleNamespace(entailment=0.99, contradiction=0.0)

    base = _StageDBase()
    base._plan_nli_scorer = _ClaimScore()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    claim = {
        "claim": safe_claim,
        "evidence_quote": quote,
        "source_block_id": "generic-source",
    }
    assert provider._validate_claim(claim, block=block) is None
    assert _ClaimScore.calls == 1
    artifact = assemble_claims(
        [{**claim, "source_role": "evidence", "source_polarity": "factual"}],
        objective_id="co-generic",
        learner_task="Analyze opposite-direction motion.",
        generated_givens=[],
    )
    claim_id = artifact["claims"][0]["stable_id"]
    response = {
        "claim_realizations": [{
            "claim_id": claim_id,
            "realization": safe_claim,
        }],
    }

    class _NoNli:
        @staticmethod
        def score_pair(**_kwargs):
            raise AssertionError("exact immutable Stage-B claim must bypass NLI")

    class _Base(_StageDBase):
        _plan_nli_scorer = _NoNli()

    class _Probe(_StageDProbe):
        def __init__(self):
            MicroStagedSynthesisProvider.__init__(
                self, _Base(), synthesis_seed=0,
            )
            self.response = response
            self.calls = 0
            self.validation_errors = []

    realized = _Probe()._realize(
        chunk_id="generic-chunk", artifact=artifact, preference=True,
    )
    assert realized["chosen"] == safe_claim


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        (
            "Objects do not move apart and their separation is not a sum.",
            "not entailed",
        ),
        (
            "Their separation distance is the difference of distances traveled.",
            "not entailed",
        ),
        (
            "Their separation distance is 3 times either distance.",
            "illegal numeric literals=['3']",
        ),
        (
            "A satellite archive determines the objects' separation.",
            "not entailed",
        ),
    ],
)
def test_canary014_stage_b_rejects_polarity_operator_numeric_and_unsupported(
    claim, expected,
):
    quote = (
        "When objects move in opposite directions, their total separation "
        "distance is the sum of the distances each travels."
    )
    block = {"block_id": "generic-source", "text": quote}

    class _RejectingScore:
        @staticmethod
        def score_pair(**_kwargs):
            return SimpleNamespace(entailment=0.01, contradiction=0.99)

    base = _StageDBase()
    base._plan_nli_scorer = _RejectingScore()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    error = provider._validate_claim(
        {
            "claim": claim,
            "evidence_quote": quote,
            "source_block_id": "generic-source",
        },
        block=block,
    )
    if expected == "not entailed":
        assert (
            "not entailed" in error
            or "contradicts its exact quote" in error
        )
    else:
        assert expected in error


def test_canary014_public_claim_contract_drifts_fingerprint_and_resume_identity(
    monkeypatch,
):
    before = micro_contract_fingerprint()
    monkeypatch.setattr(
        micro_module,
        "MICRO_STAGE_B_PUBLIC_CLAIM_VERSION",
        "ed4all.micro-stage-b-public-claim.drift",
    )
    after = micro_contract_fingerprint()
    assert after != before
    assert micro_contract_components()["stage_b_public_claim_contract"][
        "version"
    ].endswith(".drift")


def test_canary023_exact_stage_b_replay_uses_audited_formula_relation_proof():
    quote = (
        "Key Idea: When objects move in opposite directions, their total "
        "separation distance equals the sum of the distances each travels, "
        "found using D = rt with consistent units."
    )
    claims = [
        (
            "When objects move in opposite directions, the total distance "
            "between them increases by the sum of the distances each object "
            "travels, calculated using distance equals rate times time with "
            "consistent units."
        ),
        (
            "When objects travel in opposite directions, the total distance "
            "between them increases by the sum of each object's individual "
            "distance, calculated using distance equals rate times time with "
            "consistent units."
        ),
    ]

    class _NoNli:
        @staticmethod
        def score_pair(**_kwargs):
            raise AssertionError("deterministic formula proof must precede NLI")

    base = _StageDBase()
    base._plan_nli_scorer = _NoNli()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    block = {"block_id": "generic-source", "text": quote}
    for claim in claims:
        assert provider._validate_claim({
            "claim": claim,
            "evidence_quote": quote,
            "source_block_id": "generic-source",
        }, block=block) is None
    assert len(provider._validation_audit.derivations) == 2
    assert {
        item["proof"] for item in provider._validation_audit.derivations
    } == {"formula_relation_proof.v1"}
    assert all(
        len(item["premise_sha256"]) == len(item["hypothesis_sha256"]) == 64
        for item in provider._validation_audit.derivations
    )


@pytest.mark.parametrize(
    "claim",
    [
        # Polarity.
        "The sum relation does not hold; distance equals rate times time.",
        # Operator.
        "The quantities have a sum, and distance equals rate plus time.",
        # Repeated-operand/role drift.
        "The quantities have a sum, and distance equals rate times rate.",
        # Unit-constraint drift.
        "The quantities have a sum, and distance equals rate times time "
        "with inconsistent units.",
        # Unsupported numeric material is rejected before semantic proof.
        "The quantities have a sum, and distance equals rate times time "
        "with consistent units and a factor of 9.",
    ],
)
def test_canary023_formula_relation_proof_rejects_semantic_drift(claim):
    quote = (
        "The total separation is the sum of traveled distances, using "
        "D = rt with consistent units."
    )

    class _RejectingNli:
        @staticmethod
        def score_pair(**_kwargs):
            return SimpleNamespace(entailment=0.0, contradiction=1.0)

    base = _StageDBase()
    base._plan_nli_scorer = _RejectingNli()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    error = provider._validate_claim({
        "claim": claim,
        "evidence_quote": quote,
        "source_block_id": "generic-source",
    }, block={"block_id": "generic-source", "text": quote})
    assert error is not None


def test_canary023_formula_proof_source_drifts_fingerprint_and_resume_contract(
    monkeypatch,
):
    before = micro_contract_fingerprint()
    monkeypatch.setattr(
        micro_module,
        "formula_relation_proof",
        lambda _premise, _hypothesis: None,
    )
    after = micro_contract_fingerprint()
    assert after != before
    assert (
        micro_contract_components()["stage_b_public_claim_contract"][
            "formula_relation_proof_sha256"
        ]
        != micro_module._sha(
            __import__("inspect").getsource(
                __import__(
                    "Trainforge.generators.staged_synthesis_provider",
                    fromlist=["formula_relation_proof"],
                ).formula_relation_proof
            )
        )
    )


def test_canary024_exact_542_character_prompt_composes_losslessly_under_400():
    card = {
        "bloom_verb": "analyze",
        "action_object": (
            "uniform motion problems involving objects moving in opposite "
            "directions"
        ),
        "conditions": [
            "given a scenario where two objects start from the same point "
            "and move in opposite directions"
        ],
        "content_obligations": [
            "uniform motion problems involving objects moving in opposite "
            "directions"
        ],
        "performance_criteria": [
            "by correctly identifying that total distance equals the sum of "
            "each object's distance"
        ],
    }
    prompt = compose_bounded_learner_task(card)
    assert len(prompt) == 347
    assert prompt.endswith(".")
    for value in (
        card["action_object"],
        *card["conditions"],
        *card["content_obligations"],
        *card["performance_criteria"],
    ):
        assert value in prompt
    assert prompt.count(card["action_object"]) == 1


def test_canary024_bounded_composer_preserves_true_multi_obligation_and_math():
    card = _bounded_card("relations D = rt in consistent units")
    card["content_obligations"] = [
        card["action_object"],
        "compare α + β with γ",
        "justify 12 km using consistent units",
    ]
    prompt = compose_bounded_learner_task(card)
    assert "compare α + β with γ" in prompt
    assert "justify 12 km using consistent units" in prompt
    assert micro_module._scenario_numeric_literals(prompt) == {"12"}
    coverage = micro_module._learner_task_coverage(card, prompt)
    assert {row["kind"] for row in coverage} >= {
        "bloom", "action", "condition", "obligation", "criterion",
    }
    assert any(row["formula_literals"] == ["D = rt"] for row in coverage)
    assert any(row["numeric_literals"] == ["12"] for row in coverage)


def test_canary024_bounded_composer_399_400_401_boundary():
    def prompt_with_length(target: int) -> tuple[str, dict]:
        for width in range(1, 500):
            card = _bounded_card("x" * width)
            try:
                prompt = compose_bounded_learner_task(card, max_chars=target)
            except SynthesisProviderError:
                continue
            if len(prompt) == target:
                return prompt, card
        raise AssertionError(f"could not construct {target}-character fixture")

    assert len(prompt_with_length(399)[0]) == 399
    assert len(prompt_with_length(400)[0]) == 400
    prompt_401, card_401 = prompt_with_length(401)
    assert len(prompt_401) == 401
    with pytest.raises(SynthesisProviderError) as caught:
        compose_bounded_learner_task(card_401, max_chars=400)
    assert caught.value.code == "staged_micro_A_prompt_budget_unsatisfied"


def test_canary024_impossible_prompt_fails_closed_without_model_or_capture():
    card = _bounded_card("irreducible " + "x" * 500)
    with pytest.raises(SynthesisProviderError) as caught:
        deterministic_stage_a_task(card, synthesis_seed=0)
    assert caught.value.code == "staged_micro_A_prompt_budget_unsatisfied"
    assert caught.value.details["terminal_content_rejection"] is True


def test_canary024_prompt_is_seed_stable_and_projection_identical():
    card = _bounded_card()
    first = deterministic_stage_a_task(card, synthesis_seed=0)
    second = deterministic_stage_a_task(card, synthesis_seed=999)
    assert first["learner_task"] == second["learner_task"]
    assert first["_stage_a_deterministic"]["prompt_coverage"] == (
        second["_stage_a_deterministic"]["prompt_coverage"]
    )
    assert deterministic_stage_d_prompt(first) == deterministic_stage_d_prompt(
        second
    )
    assert micro_module._TASK_SCHEMA["properties"]["learner_task"][
        "maxLength"
    ] == 400


def test_canary024_composer_source_drifts_fingerprint_and_resume_identity(
    monkeypatch,
):
    before = micro_contract_fingerprint()
    monkeypatch.setattr(
        micro_module,
        "compose_bounded_learner_task",
        lambda _card, *, max_chars=400: "Analyze a stable relation.",
    )
    assert micro_contract_fingerprint() != before


def test_canary015_exact_payload_stalls_after_prompt_and_length_remains_fatal():
    root = (
        Path(__file__).resolve().parents[2]
        / "plans/release-evidence/training-synthesis-functional-v1.3.0"
        / "02-one-row-canary"
        / "functional-production-dpo-00183-seed0-015"
    )
    path = root / "audit/http-raw" / (
        "response-f45d9be6f4a99f5e59a19166449590fcb95b05af31fbfd8be1f09fc0463260a1.bin"
    )
    if not path.exists():
        pytest.skip("ignored canary015 provider payload is unavailable")
    payload = json.loads(path.read_bytes())
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    assert choice["finish_reason"] == "length"
    assert payload["usage"]["completion_tokens"] == 1536
    assert len(content.rstrip()) == 412
    assert len(content) == 1881
    assert content.rstrip().endswith('"')
    assert '"claim_realizations"' not in content
    with pytest.raises(json.JSONDecodeError):
        json.loads(content)
    # The transport contract intentionally remains fail-closed on every length
    # finish. The correction removes the unbounded model-authored prompt rather
    # than accepting this capped response.
    assert set(_stage_d_realization_schema(_stage_d_artifact())["required"]) == {
        "claim_realizations"
    }


@pytest.mark.parametrize("claim_count", [1, 2, 3])
def test_canary015_per_artifact_schema_binds_cardinality_and_known_ids(
    claim_count,
):
    claims = [
        {
            "claim": f"Generic supported relation {index}.",
            "evidence_quote": f"Generic supported relation {index}.",
            "source_block_id": f"generic-source-{index}",
            "source_role": "evidence",
            "source_polarity": "factual",
        }
        for index in range(claim_count)
    ]
    artifact = assemble_claims(
        claims,
        objective_id="co-generic",
        learner_task="Analyze the supported relations.",
        generated_givens=[],
    )
    schema = _stage_d_realization_schema(artifact)
    rows = schema["properties"]["claim_realizations"]
    expected = [item["stable_id"] for item in artifact["claims"]]
    assert rows["minItems"] == rows["maxItems"] == claim_count
    assert rows["items"]["properties"]["claim_id"]["enum"] == expected
    assert schema["additionalProperties"] is False


def test_canary015_legacy_overlong_prompt_now_fails_authoritative_400_bound():
    root = (
        Path(__file__).resolve().parents[2]
        / "plans/release-evidence/training-synthesis-functional-v1.3.0"
        / "02-one-row-canary"
        / "functional-production-dpo-00183-seed0-015"
        / "micro-journals/micro_synthesis_state"
    )
    matches = list(root.glob("*.jsonl"))
    if not matches:
        pytest.skip("ignored canary015 journal is unavailable")
    terminal_c = next(
        row
        for line in matches[0].read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["stage"] == "C"
        and row["state"] == "terminal"
    )
    artifact = terminal_c["artifact"]
    assert len(artifact["learner_task"]) > MICRO_STAGE_D_PROMPT_MAX_CHARS
    with pytest.raises(SynthesisProviderError) as caught:
        deterministic_stage_d_prompt(artifact)
    assert caught.value.code == "staged_micro_D_prompt_invalid"


@pytest.mark.parametrize(
    "learner_task",
    ["", "Sentence without terminator", "x" * (MICRO_STAGE_D_PROMPT_MAX_CHARS + 1)],
)
def test_canary015_deterministic_prompt_fails_closed_not_truncates(learner_task):
    artifact = {**_stage_d_artifact(), "learner_task": learner_task}
    with pytest.raises(SynthesisProviderError) as caught:
        deterministic_stage_d_prompt(artifact)
    assert caught.value.code == "staged_micro_D_prompt_invalid"


def test_canary015_prompt_contract_drifts_fingerprint(monkeypatch):
    before = micro_contract_fingerprint()
    monkeypatch.setattr(
        micro_module,
        "MICRO_STAGE_D_PROMPT_CONTRACT_VERSION",
        "ed4all.micro-stage-d-prompt.drift",
    )
    assert micro_contract_fingerprint() != before
    component = micro_contract_components()["stage_d_prompt_contract"]
    assert component["version"].endswith(".drift")
    assert component["model_fields"] == []


def _stage_e_candidate(
    statement="A grouped operation may retain partial success.",
    correction="A grouped operation succeeds or fails as one unit.",
    bloom="analyze",
):
    return {
        "id": canonical_mc_id(statement, correction, bloom),
        "misconception": statement,
        "correction": correction,
        "mechanism_evidence": f"{statement} {correction}",
        "bloom_level": bloom,
        "source_block_id": "generic-source",
        "source_role": "misconception_claim",
        "source_polarity": "incorrect_with_correction",
    }


def test_canary016_exact_fault_selects_canonical_id_without_free_mechanism():
    root = (
        Path(__file__).resolve().parents[2]
        / "plans/release-evidence/training-synthesis-functional-v1.3.0"
        / "02-one-row-canary"
        / "functional-production-dpo-00183-seed0-016"
    )
    request_path = root / "audit/http-raw" / (
        "request-2239f14fe64511e43569c82b59c1d85417ba82b4a8d13f452783f5db33c57af9.bin"
    )
    response_path = root / "micro-journals/raw_staged_calls" / (
        "response-c30adebd2f37b5acd28313659dc432af4e856ea3908acd82cde3b31e4d053417.txt"
    )
    if not request_path.exists() or not response_path.exists():
        pytest.skip("ignored canary016 Gate-E evidence is unavailable")
    request = json.loads(request_path.read_bytes())
    user = request["messages"][-1]["content"]
    candidates = json.loads(
        user.split("CANDIDATES=", 1)[1].split("\nReturn one", 1)[0]
    )
    candidate = {**candidates[0], "bloom_level": "analyze"}
    candidate["id"] = canonical_mc_id(
        candidate["misconception"],
        candidate["correction"],
        candidate["bloom_level"],
    )
    old = json.loads(response_path.read_text(encoding="utf-8"))
    selection = {
        "misconception_id": candidate["id"],
        "rationale": old["rationale"],
    }
    assert set(selection) == {"misconception_id", "rationale"}
    assert _resolve_stage_e_selection(
        [candidate], selection["misconception_id"],
    ) == (0, candidate)
    schema = _stage_e_selection_schema([candidate])
    assert schema["properties"]["misconception_id"]["enum"] == [candidate["id"]]
    assert "faulty_step" not in schema["properties"]
    assert "error_mechanism" not in schema["properties"]


def test_canary016_id_matches_shared_ontology_and_is_reorder_stable():
    first = _stage_e_candidate()
    second = _stage_e_candidate(
        "A checksum guarantees an unchanged payload.",
        "A checksum detects a changed payload but cannot guarantee absence.",
    )
    assert first["id"] == canonical_mc_id(
        first["misconception"], first["correction"], first["bloom_level"],
    )
    assert _resolve_stage_e_selection([first, second], second["id"])[1] == second
    assert _resolve_stage_e_selection([second, first], second["id"])[1] == second


@pytest.mark.parametrize("mutation", ["malformed", "mc_" + "f" * 16])
def test_canary016_unknown_or_malformed_id_never_index_falls_back(mutation):
    candidate = _stage_e_candidate()
    with pytest.raises(SynthesisProviderError) as caught:
        _resolve_stage_e_selection([candidate], mutation)
    assert caught.value.code == "staged_micro_E_authority_unresolved"


def test_canary016_rejects_candidate_content_mismatch_and_duplicate_collision():
    candidate = _stage_e_candidate()
    with pytest.raises(SynthesisProviderError) as mismatch:
        _stage_e_authority([
            {**candidate, "correction": "Mutated correction."},
        ])
    assert mismatch.value.code == "staged_micro_E_authority_invalid"
    with pytest.raises(SynthesisProviderError) as duplicate:
        _stage_e_authority([candidate, dict(candidate)])
    assert duplicate.value.code == "staged_micro_E_authority_invalid"


def test_canary016_faulty_step_operator_change_remains_unsupported():
    candidate = _stage_e_candidate(
        "A relative rate is found by subtracting component rates.",
        "The relative rate is found by adding component rates.",
    )

    class _Score:
        entailment = 0.01
        contradiction = 0.99

    class _Nli:
        @staticmethod
        def score_pair(**_kwargs):
            return _Score()

    base = _StageDBase()
    base._plan_nli_scorer = _Nli()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    error = provider._stage_e_faulty_step_error(
        candidate,
        "The faulty step multiplies unrelated component rates.",
    )
    assert "not supported by indexed misconception evidence" in error


def test_canary016_stage_e_authority_contract_drifts_fingerprint(monkeypatch):
    before = micro_contract_fingerprint()
    monkeypatch.setattr(
        micro_module,
        "MICRO_STAGE_E_AUTHORITY_CONTRACT_VERSION",
        "ed4all.micro-stage-e-misconception-authority.drift",
    )
    assert micro_contract_fingerprint() != before
    component = micro_contract_components()["stage_e_authority_contract"]
    assert component["version"].endswith(".drift")
    assert component["model_fields"] == ["misconception_id", "rationale"]
    assert component["deterministic_fields"] == [
        "misconception_index", "error_mechanism", "faulty_step",
    ]


def test_canary025_exact_authority_derives_faulty_step_without_nli():
    statement = (
        "when two objects move in opposite directions, their relative speed is "
        "found by subtracting their speeds rather than adding them"
    )
    candidate = _stage_e_candidate(
        statement,
        "Opposite-direction relative speed is the sum of the component speeds.",
    )
    index, resolved, proof = _stage_e_exact_faulty_step_proof(
        [candidate], candidate["id"],
    )
    assert index == 0
    assert resolved["misconception"] == statement
    assert proof["proof"] == "exact_canonical_faulty_step"
    assert proof["normalization"] == "none-byte-exact-utf8"
    assert proof["faulty_step_sha256"] == hashlib.sha256(
        statement.encode("utf-8")
    ).hexdigest()


def test_canary025_exact_replay_resolves_terminal_gate_e_output():
    root = (
        Path(__file__).resolve().parents[2]
        / "plans/release-evidence/training-synthesis-functional-v1.3.0"
        / "02-one-row-canary"
        / "functional-production-dpo-00183-seed0-025"
    )
    request_path = root / "audit/http-raw" / (
        "request-e0d05e5a26f7c11f35f4b556206b9ad4b1479baaf7233c0aa062e91e547a6c94.bin"
    )
    response_path = root / "micro-journals/raw_staged_calls" / (
        "response-8e9a00c4ac52bd58df8b4bc6bcfea9b40461b999f35a4984fa5927657bbba867.txt"
    )
    if not request_path.exists() or not response_path.exists():
        pytest.skip("ignored canary025 Gate-E evidence is unavailable")
    request = json.loads(request_path.read_bytes())
    user = request["messages"][-1]["content"]
    candidates = json.loads(
        user.split("CANONICAL_CANDIDATES=", 1)[1].split(
            "\nReturn one", 1,
        )[0]
    )
    response = json.loads(response_path.read_text(encoding="utf-8"))
    index, candidate, proof = _stage_e_exact_faulty_step_proof(
        candidates, response["misconception_id"],
    )
    assert index == 0
    assert response["faulty_step"] == candidate["misconception"]
    assert proof["faulty_step_sha256"] == hashlib.sha256(
        response["faulty_step"].encode("utf-8")
    ).hexdigest()
    schema = _stage_e_selection_schema(candidates)
    assert schema["required"] == ["misconception_id", "rationale"]
    assert "faulty_step" not in schema["properties"]


def test_canary025_unicode_authority_is_byte_exact_not_semantically_normalized():
    composed = _stage_e_candidate(
        "Café subtraction is valid.", "Addition is required.",
    )
    decomposed_text = "Cafe\u0301 subtraction is valid."
    decomposed = _stage_e_candidate(
        decomposed_text, "Addition is required.",
    )
    assert composed["id"] != decomposed["id"]
    _index, resolved, proof = _stage_e_exact_faulty_step_proof(
        [decomposed], decomposed["id"],
    )
    assert resolved["misconception"] == decomposed_text
    assert proof["faulty_step_sha256"] != hashlib.sha256(
        composed["misconception"].encode("utf-8")
    ).hexdigest()


def test_canary025_exact_authority_fails_closed_on_id_content_and_collision():
    candidate = _stage_e_candidate()
    with pytest.raises(SynthesisProviderError):
        _stage_e_exact_faulty_step_proof([candidate], "mc_" + "f" * 16)
    with pytest.raises(SynthesisProviderError):
        _stage_e_exact_faulty_step_proof(
            [{**candidate, "misconception": "Polarity reversed."}],
            candidate["id"],
        )
    with pytest.raises(SynthesisProviderError):
        _stage_e_exact_faulty_step_proof(
            [candidate, dict(candidate)], candidate["id"],
        )


def test_canary025_polarity_and_operator_mutation_cannot_reuse_authority_id():
    candidate = _stage_e_candidate(
        "Relative speed is found by subtracting the component speeds.",
        "Relative speed is found by adding the component speeds.",
    )
    for mutated in (
        "Relative speed is not found by subtracting the component speeds.",
        "Relative speed is found by multiplying the component speeds.",
    ):
        with pytest.raises(SynthesisProviderError) as caught:
            _stage_e_exact_faulty_step_proof(
                [{**candidate, "misconception": mutated}], candidate["id"],
            )
        assert caught.value.code == "staged_micro_E_authority_invalid"


def test_gate_d_rejects_attempt_two_repeat_and_repair_budget_before_base_call():
    class Base:
        _model = "generic-model"
        _provider_name = "local"
        calls = 0

    provider = MicroStagedSynthesisProvider(Base(), synthesis_seed=17)
    provider.bind_trusted_gate_d_single_pass(_gate_d_binding())
    with pytest.raises(SynthesisProviderError) as attempt:
        provider._call_stage(stage="micro_B_claim_0_attempt_2")
    assert attempt.value.code == "staged_micro_gate_d_attempt_forbidden"
    with pytest.raises(SynthesisProviderError) as repair:
        provider._call_stage(
            stage="micro_A_task_design", max_stage_repairs=1,
        )
    assert repair.value.code == "staged_micro_gate_d_repair_budget_invalid"
    assert Base.calls == 0


def test_gate_d_provider_failure_is_terminal_and_legacy_claim_retry_is_unchanged(
    monkeypatch,
):
    class Base:
        _model = "generic-model"
        _provider_name = "local"

    class Result:
        value = _claim()

    gate = MicroStagedSynthesisProvider(Base(), synthesis_seed=17)
    gate.bind_trusted_gate_d_single_pass(_gate_d_binding())
    gate_calls = []

    def fail_once(self, **kwargs):
        gate_calls.append(kwargs["stage"])
        raise SynthesisProviderError("validator rejected", code="synthetic_invalid")

    from Trainforge.generators import staged_synthesis_provider as staged_base
    monkeypatch.setattr(staged_base.StagedSynthesisProvider, "_call_stage", fail_once)
    with pytest.raises(SynthesisProviderError) as terminal:
        gate._claim_slot(
            chunk_id="generic-chunk", slot=0,
            block={
                "block_id": "b-1", "polarity": "factual",
                "text": "The product is commutative.",
            },
        )
    assert terminal.value.code == "staged_micro_gate_d_stage_terminal"
    assert gate_calls == ["micro_B_claim_0_attempt_1"]
    with pytest.raises(SynthesisProviderError) as repeated:
        gate._claim_slot(
            chunk_id="generic-chunk", slot=0,
            block={
                "block_id": "b-1", "polarity": "factual",
                "text": "The product is commutative.",
            },
        )
    assert repeated.value.code == "staged_micro_gate_d_stage_terminal"
    assert gate_calls == ["micro_B_claim_0_attempt_1"]

    legacy = MicroStagedSynthesisProvider(Base(), synthesis_seed=17)
    legacy_calls = []

    def fail_then_pass(self, **kwargs):
        legacy_calls.append(kwargs["stage"])
        if len(legacy_calls) == 1:
            raise SynthesisProviderError("validator rejected", code="synthetic_invalid")
        return Result()

    monkeypatch.setattr(
        staged_base.StagedSynthesisProvider, "_call_stage", fail_then_pass,
    )
    assert legacy._claim_slot(
        chunk_id="generic-chunk", slot=0,
        block={
            "block_id": "b-1", "polarity": "factual",
            "text": "The product is commutative.",
        },
    ) == _claim()
    assert legacy_calls == [
        "micro_B_claim_0_attempt_1",
        "micro_B_claim_0_attempt_2",
    ]


def test_schema_maximum_payloads_fit_stage_token_caps_by_byte_upper_bound():
    from Trainforge.generators import staged_synthesis_micro as micro

    payloads = {
        "A": {
            "objective_id": "o" * 64,
            "bloom_level": "understand",
            "learner_task": "t" * 400,
        },
        "B": {
            "claim": "c" * 300,
            "evidence_quote": "q" * 500,
            "source_block_id": "b" * 128,
        },
        "D": {"prompt": "p" * 400, "completion": "c" * 600},
        "E": {
            "misconception_index": 0,
            "error_mechanism": "e" * 240,
            "faulty_step": "f" * 240,
            "rationale": "r" * 320,
        },
        "F": {"rejected": "r" * 600},
    }
    # UTF-8 bytes are a conservative token upper bound for these ASCII
    # schema maxima, so fitting here proves the cap does not disable features.
    for stage, payload in payloads.items():
        assert len(json.dumps(payload).encode("utf-8")) < MICRO_STAGE_MAX_TOKENS[stage]


def test_assembly_dedupes_with_stable_identity_order_and_hashes():
    artifact = assemble_claims(
        [
            _claim(block="z"),
            _claim(block="z"),
            _claim(text="A sum combines terms.", block="a"),
        ],
        objective_id="CO-1",
        learner_task="Compare the relations.",
        generated_givens=[],
    )
    assert [row["source_block_id"] for row in artifact["claims"]] == ["a", "z"]
    assert len({row["stable_id"] for row in artifact["claims"]}) == 2
    assert len(artifact["claims_sha256"]) == 64
    assert immutable_artifact_error(
        artifact, artifact["artifact_sha256"]
    ) is None
    tampered = json.loads(json.dumps(artifact))
    tampered["claims"][0]["claim"] = "Changed."
    assert immutable_artifact_error(tampered, artifact["artifact_sha256"])


def test_empty_obligations_route_by_objective_semantics_not_block_id(monkeypatch):
    import Trainforge.generators.staged_synthesis_micro as micro

    provider = object.__new__(MicroStagedSynthesisProvider)
    provider._task_design = lambda _chunk: {
        "objective_id": "co-1",
        "learner_task": "Analyze rollback behavior after a failed transaction.",
        "generated_givens": [],
        "_objective_card": {
            "id": "co-1",
            "statement": "Analyze transaction rollback semantics.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "action_object": "",
            "content_obligations": [],
        },
    }
    observed = []
    provider._claim_slot = lambda **kwargs: (
        observed.append(kwargs["block"]["block_id"])
        or _claim(
            text=kwargs["block"]["text"],
            block=kwargs["block"]["block_id"],
        )
    )
    provider._journaled = lambda _store, _cache, **kwargs: kwargs["call"]()
    monkeypatch.setattr(micro, "build_evidence_window", lambda *_args, **_kwargs: {
        "blocks": [
            {
                "block_id": "block-001",
                "text": "A garden contains several flowering plants.",
                "polarity": "factual",
            },
            {
                "block_id": "block-900",
                "text": (
                    "A failed transaction uses rollback behavior to restore "
                    "the prior state."
                ),
                "polarity": "factual",
            },
        ],
    })

    _task, artifact = provider._assemble(
        {"id": "generic-chunk"}, kind="instruction", store=None, cache={},
    )

    assert observed[0] == "block-900"
    assert len({row["stable_id"] for row in artifact["claims"]}) == 2


def test_prompt_numeric_validation_cannot_be_seeded_by_answer():
    artifact = assemble_claims(
        [_claim(text="Objects can be compared by their attributes.")],
        objective_id="CO-1",
        learner_task="Analyze the objects.",
        generated_givens=[],
    )
    assert _prompt_numeric_error("Analyze 999 objects.", artifact)
    # Supplying the same value in an eventual answer must not affect the
    # independent learner-task/prompt decision.
    assert _prompt_numeric_error("Analyze 999 objects.", artifact)
    with_given = assemble_claims(
        [_claim(text="Objects can be compared by their attributes.")],
        objective_id="CO-1",
        learner_task="Analyze 999 objects.",
        generated_givens=[_given("999")],
    )
    assert _prompt_numeric_error("Analyze 999 objects.", with_given) is None


def test_micro_leakage_rejects_exact_distinctive_fifty_character_copy():
    narrative = (
        "A remarkably distinctive source sentence describes copper gears "
        "turning beneath a violet observatory at midnight."
    )
    artifact = assemble_claims(
        [_claim(text=narrative)],
        objective_id="CO-1",
        learner_task="Explain the mechanism in your own words.",
        generated_givens=[],
    )
    assert _leakage_error(narrative, source=narrative, artifact=artifact)
    assert _leakage_error(
        "Explain the mechanism in your own words.",
        source=narrative,
        artifact=artifact,
    ) is None


def test_resume_journal_round_trip_hash_chain_and_duplicate_rejection(tmp_path):
    store = MicroResumeStore(tmp_path / "micro.jsonl", fingerprint="f" * 64)
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1, state="started"
    )
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1,
        state="terminal", artifact={"objective_id": "co-1"},
    )
    assert store.load() == {"instruction:A": {"objective_id": "co-1"}}
    # Even byte-identical duplicate terminals make unit completion ambiguous.
    store.append(
        unit="instruction:A", stage="A", slot=None, attempt=1,
        state="terminal", artifact={"objective_id": "co-1"},
    )
    with pytest.raises(SynthesisProviderError) as caught:
        store.load()
    assert caught.value.code == "staged_micro_duplicate_identity"


def test_resume_journal_rejects_started_without_terminal(tmp_path):
    store = MicroResumeStore(tmp_path / "micro.jsonl", fingerprint="f" * 64)
    store.append(
        unit="preference:B:slot-0", stage="B", slot=0,
        attempt=1, state="started",
    )
    with pytest.raises(SynthesisProviderError) as caught:
        store.load()
    assert caught.value.code == "staged_micro_resume_ambiguous"


def test_journaled_content_rejection_closes_and_replays_without_call(
    tmp_path,
):
    store = MicroResumeStore(tmp_path / "micro.jsonl", fingerprint="f" * 64)
    base = type(
        "Base", (), {
            "_capture": None, "_model": "test", "_provider_name": "local",
            "_plan_nli_scorer": None,
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=17)
    calls = 0

    def reject():
        nonlocal calls
        calls += 1
        raise SynthesisProviderError(
            "closed invalid realization",
            code="staged_micro_D_sft_invalid",
            details={
                "terminal_content_rejection": True,
                "validation_error": "requirement missing",
            },
        )

    with pytest.raises(SynthesisProviderError):
        provider._journaled(
            store, {}, kind="instruction", stage="D", slot=None,
            attempt=1, call=reject,
        )
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert [row["state"] for row in rows] == ["started", "terminal"]
    assert rows[-1]["outcome"] == "content_rejected"
    assert rows[-1]["terminal_evidence"]["reason_code"] == (
        "staged_micro_D_sft_invalid"
    )
    with pytest.raises(SynthesisProviderError) as replay:
        store.load()
    assert replay.value.code == "staged_micro_D_sft_invalid"
    assert calls == 1
    assert store.load(allow_failure_outcomes=True) == {}


def test_journaled_unknown_error_remains_open_not_content_rejected(tmp_path):
    store = MicroResumeStore(tmp_path / "micro.jsonl", fingerprint="f" * 64)
    base = type(
        "Base", (), {
            "_capture": None, "_model": "test", "_provider_name": "local",
            "_plan_nli_scorer": None,
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=17)
    with pytest.raises(RuntimeError):
        provider._journaled(
            store, {}, kind="instruction", stage="A", slot=None,
            attempt=1, call=lambda: (_ for _ in ()).throw(RuntimeError("crash")),
        )
    with pytest.raises(SynthesisProviderError) as caught:
        store.load()
    assert caught.value.code == "staged_micro_resume_ambiguous"


def test_journaled_success_is_sealed_before_post_call_stop(monkeypatch, tmp_path):
    from lib.generation import stop_control

    store = MicroResumeStore(tmp_path / "micro.jsonl", fingerprint="f" * 64)
    base = type(
        "Base", (), {
            "_capture": None, "_model": "test", "_provider_name": "local",
            "_plan_nli_scorer": None,
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=17)
    checks = iter((False, True))
    monkeypatch.setattr(stop_control, "stop_requested", lambda: next(checks))
    with pytest.raises(stop_control.GracefulStopRequested):
        provider._journaled(
            store, {}, kind="instruction", stage="A", slot=None,
            attempt=1, call=lambda: {"task": "sealed"},
        )
    assert store.load() == {"instruction:A": {"task": "sealed"}}


def test_terminal_rejection_evidence_tamper_fails_closed(tmp_path):
    path = tmp_path / "micro.jsonl"
    store = MicroResumeStore(path, fingerprint="f" * 64)
    store.append(
        unit="instruction:D", stage="D", slot=None, attempt=1,
        state="started",
    )
    evidence = {
        "contract": "ed4all.micro-terminal-outcome.v1",
        "outcome": "content_rejected",
        "reason_code": "staged_micro_D_sft_invalid",
        "exception_class": "SynthesisProviderError",
        "message_sha256": "a" * 64,
        "details_sha256": "b" * 64,
        "unit": "instruction:D",
        "stage": "D",
        "slot": None,
        "attempt": 1,
        "logical_intent": {
            "kind": "instruction", "unit": "instruction:D", "stage": "D",
            "slot": None, "attempt": 1,
        },
        "external_linkage": {
            "decision_capture_id": "",
            "http_attempt_ledger": "cell-bound-authority",
            "call_intent_ledger": "cell-bound-authority",
        },
    }
    store.append(
        unit="instruction:D", stage="D", slot=None, attempt=1,
        state="terminal", outcome="content_rejected",
        terminal_evidence=evidence,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["terminal_evidence"]["reason_code"] = "tampered"
    # Re-sealing only the outer row cannot bypass the inner evidence seal.
    rows[-1]["row_sha256"] = store._row_hash(rows[-1])
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(SynthesisProviderError) as caught:
        store.load(allow_failure_outcomes=True)
    assert caught.value.code == "staged_micro_resume_corrupt"


def test_resume_journal_rejects_contract_drift_and_hash_tamper(tmp_path):
    path = tmp_path / "micro.jsonl"
    store = MicroResumeStore(path, fingerprint="f" * 64)
    store.append(unit="instruction:C", stage="C", slot=None, attempt=1, state="started")
    drifted = MicroResumeStore(path, fingerprint="e" * 64)
    with pytest.raises(SynthesisProviderError) as caught:
        drifted.load()
    assert caught.value.code == "staged_micro_resume_drift"

    row = json.loads(path.read_text())
    row["attempt"] = 2
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SynthesisProviderError):
        store.load()


def test_resume_journal_rejects_conflicting_duplicate_terminal(tmp_path):
    store = MicroResumeStore(tmp_path / "micro.jsonl", fingerprint="f" * 64)
    store.append(unit="instruction:D", stage="D", slot=None, attempt=1, state="started")
    store.append(
        unit="instruction:D", stage="D", slot=None, attempt=1,
        state="terminal", artifact={"completion": "one"},
    )
    store.append(
        unit="instruction:D", stage="D", slot=None, attempt=1,
        state="terminal", artifact={"completion": "two"},
    )
    with pytest.raises(SynthesisProviderError) as caught:
        store.load()
    assert caught.value.code == "staged_micro_duplicate_identity"


def test_exact_quote_provenance_is_block_local_and_nli_only_after_fast_paths():
    class Score:
        entailment = 0.99
        contradiction = 0.0

    class Nli:
        calls = []

        def score_pair(self, *, premise, hypothesis):
            self.calls.append((premise, hypothesis))
            return Score()

    base = type(
        "Base",
        (),
        {
            "_capture": None,
            "_model": "test",
            "_provider_name": "local",
            "_plan_nli_scorer": Nli(),
        },
    )()
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=17)
    block = {
        "block_id": "block-a",
        "polarity": "factual",
        "text": "A product combines factors by multiplication.",
    }
    exact = {
        "claim": "A product combines factors by multiplication.",
        "evidence_quote": "A product combines factors by multiplication.",
        "source_block_id": "block-a",
    }
    assert provider._validate_claim(exact, block=block) is None
    assert base._plan_nli_scorer.calls == []
    wrong_block = {**exact, "source_block_id": "block-b"}
    assert provider._validate_claim(wrong_block, block=block)


def test_stop_after_claim_assembly_prevents_realization_dispatch(monkeypatch):
    from lib.generation import stop_control

    base = type(
        "Base",
        (),
        {
            "_capture": None,
            "_model": "test",
            "_provider_name": "local",
            "_plan_nli_scorer": None,
        },
    )()

    class Provider(MicroStagedSynthesisProvider):
        realize_calls = 0

        def _assemble(self, chunk, *, kind, store, cache):
            artifact = assemble_claims(
                [_claim()],
                objective_id="co-1",
                learner_task="Explain the relation.",
                generated_givens=[],
            )
            return {
                "bloom_level": "understand",
                "_objective_card": {"id": "co-1"},
            }, artifact

        def _realize(self, **kwargs):
            self.realize_calls += 1
            return {"prompt": "p", "completion": "c"}

    provider = Provider(base, synthesis_seed=17)
    checks = iter((False, True))
    monkeypatch.setattr(stop_control, "stop_requested", lambda: next(checks, True))
    with pytest.raises(stop_control.GracefulStopRequested):
        provider.paraphrase_instruction(
            {"prompt": "draft", "completion": "draft"},
            {"id": "generic-chunk"},
        )
    assert provider.realize_calls == 0


def test_complete_af_flows_capture_one_dynamic_event_per_admitted_call(
    monkeypatch,
):
    monkeypatch.setenv(
        "TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS", "32768",
    )
    class Capture:
        calls = []

        def __init__(self):
            self.calls = []
            self.decisions = []

        def log_decision(self, **kwargs):
            self.calls.append(kwargs)
            self.decisions.append({
                "event_id": f"EVT_mock_{len(self.decisions) + 1:04d}",
                **kwargs,
            })

    class Score:
        entailment = 0.99
        contradiction = 0.0

    class Nli:
        @staticmethod
        def score_pair(*, premise, hypothesis):
            return Score()

    class IntentManifest:
        def __init__(self):
            self.rows = []
            self.active = None

        def admit(self, request_sha):
            sequence = len(self.rows) + 1
            row = {
                "run_id": "synthetic-run",
                "cell_id": "synthetic-cell",
                "unit": f"unit-{sequence}",
                "kind": "structured_completion",
                "stage": f"strict-call-{sequence}",
                "logical_attempt": 1,
                "request_sha256": request_sha,
                "model": "synthetic-model",
                "contract_sha256": "c" * 64,
            }
            self.rows.append(row)
            self.active = row

        def current(self):
            return self.active

    class Ledger:
        def __init__(self, manifest):
            self._intent_manifest = manifest

    class Client:
        @staticmethod
        def _extract_json_lenient(raw):
            return json.loads(raw)

    class Base:
        api_url = "http://test.invalid/v1/chat/completions"
        base_url = "http://test.invalid/v1"
        client = object()
        _oa_client = Client()
        _model = "synthetic-model"
        _provider_name = "local"
        _provenance_provider = "local"
        _max_tokens = 800
        _plan_nli_scorer = Nli()

        def __init__(self, responses, capture, manifest):
            self.responses = iter(responses)
            self._capture = capture
            self.manifest = manifest
            self.prompts = []

        def _chat_completion_raw_structured(self, messages, *, schema, max_tokens):
            rendered = json.dumps(messages, sort_keys=True)
            self.manifest.admit(hashlib.sha256(rendered.encode()).hexdigest())
            self.prompts.append(messages)
            self.request_max_tokens = getattr(self, "request_max_tokens", [])
            self.request_max_tokens.append(max_tokens)
            return next(self.responses), {
                "prompt_tokens": 13, "completion_tokens": 8,
            }, 0

    chunk = {
        "id": "generic-chunk",
        "text": (
            "A transaction groups operations into one logical unit. Atomicity "
            "means operations all succeed or all fail, preventing partial "
            "updates. A mistaken view says atomic transactions preserve "
            "successful partial work."
        ),
        "learning_outcome_refs": ["co-1"],
        "bloom_level": "analyze",
        "misconceptions": [{
            "misconception": (
                "Atomic transactions preserve successful partial work."
            ),
            "correction": "Atomicity prevents partial updates.",
        }],
        "synthesis_focus_objective": {
            "id": "co-1",
            "statement": "Analyze how atomicity prevents partial updates.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "action_object": "how atomicity prevents partial updates",
        },
    }
    task = json.dumps({
        "objective_id": "co-1",
        "bloom_level": "analyze",
        "learner_task": "Analyze how atomicity prevents incomplete outcomes.",
    })
    claim = json.dumps({
        "claim": "Atomicity prevents incomplete transactional outcomes.",
        "evidence_quote": (
            "Atomicity means operations all succeed or all fail, preventing "
            "partial updates."
        ),
        "source_block_id": "generic-chunk",
    })
    claim_payload = json.loads(claim)
    claim_id = assemble_claims(
        [{
            **claim_payload,
            "source_role": "flat_source",
            "source_polarity": "factual",
        }],
        objective_id="co-1",
        learner_task="unused",
        generated_givens=[],
    )["claims"][0]["stable_id"]
    requirement_contract = derive_objective_requirements(
        chunk["synthesis_focus_objective"],
    )
    worked_requirements = [
        item for item in requirement_contract["requirements"]
        if item["kind"] != "result"
    ]
    result_requirement = next(
        item for item in requirement_contract["requirements"]
        if item["kind"] == "result"
    )
    objective_execution = {
        "worked_steps": [{
            "requirement_id": item["requirement_id"],
            "claim_ids": [claim_id],
            "realization": (
                "Analyze the all-or-none relationship: atomic execution keeps "
                "the operation group indivisible."
            ),
        } for item in worked_requirements],
        "result": {
            "requirement_id": result_requirement["requirement_id"],
            "claim_ids": [claim_id],
            "realization": (
                "Therefore an incomplete transactional result cannot remain."
            ),
        },
    }
    sft = json.dumps({
        "claim_realizations": [{
            "claim_id": claim_id,
            "realization": (
                "Atomic execution keeps the operation group indivisible, so an "
                "incomplete transactional result cannot remain."
            ),
        }], **objective_execution,
    })
    chosen = json.dumps({
        "claim_realizations": [{
            "claim_id": claim_id,
            "realization": (
                "Atomic execution keeps the operation group indivisible, so an "
                "incomplete transactional result cannot remain."
            ),
        }], **objective_execution,
    })
    selection = json.dumps({
        "misconception_id": canonical_mc_id(
            "Atomic transactions preserve successful partial work.",
            "Atomicity prevents partial updates.",
            "analyze",
        ),
        "rationale": (
            "The selected source-backed misconception reverses the all-or-none "
            "transaction guarantee."
        ),
    })
    rejected = json.dumps({
        "rejected": (
            "Earlier successful operations remain applied even when a later "
            "operation fails, because partial success is preserved."
        ),
    })

    capture = Capture()
    manifest = IntentManifest()
    base = Base([claim, sft, claim, chosen, selection, rejected],
                capture, manifest)
    provider = MicroStagedSynthesisProvider(base, synthesis_seed=0)
    provider._pilot_attempt_ledger = Ledger(manifest)
    instruction = provider.paraphrase_instruction(
        {
            "prompt": "draft", "completion": "draft",
            "decision_capture_id": "fabricated-draft-id",
        },
        chunk,
    )
    preference = provider.paraphrase_preference(
        {
            "prompt": "draft", "chosen": "draft", "rejected": "draft",
            "decision_capture_id": "fabricated-draft-id",
        },
        chunk,
    )
    for pair in (instruction, preference):
        evidence = pair["provenance"]["claim_evidence"][0]
        assert "evidence_quote" not in evidence
        assert len(evidence["evidence_quote_sha256"]) == 64
        sidecar = pair["_objective_execution_private_sidecar"]
        assert sidecar["private_evidence"][0]["quote"]
        assert pair["_objective_execution_candidate"][
            "pair_objective_execution_pass_rate"
        ] == 1.0

    assert len(manifest.rows) == len(capture.calls) == 6
    assert base.request_max_tokens == [
        1536, 1536, 1536, 1536, 1280, 1024,
    ]
    contexts = [json.loads(call["context"]) for call in capture.calls]
    observed_stages = [context["stage"] for context in contexts]
    assert not any("micro_A_" in stage for stage in observed_stages)
    assert any("micro_B_" in stage for stage in observed_stages)
    assert any("micro_D_" in stage for stage in observed_stages)
    assert any("micro_E_" in stage for stage in observed_stages)
    assert any("micro_F_" in stage for stage in observed_stages)
    assert sum("micro_E_" in stage for stage in observed_stages) == 1
    assert sum("micro_F_" in stage for stage in observed_stages) == 1
    e_context = next(
        context for context in contexts
        if "micro_E_" in context["stage"]
    )
    e_candidates = micro_preference_eligibility(
        chunk, focus=provider._focus(chunk),
    )["candidates"]
    expected_e_schema_sha = hashlib.sha256(
        json.dumps(
            _stage_e_selection_schema(e_candidates),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()
    ).hexdigest()
    assert e_context["validation_evidence"][
        "response_schema_sha256"
    ] == expected_e_schema_sha
    assert {
        row["proof"]
        for row in e_context["validation_evidence"]["derivation_ledger"]
    } == {"exact_canonical_faulty_step"}
    assert "Atomic transactions preserve successful partial work." in (
        base.prompts[-1][-1]["content"]
    )
    # A and C are deterministic and therefore have no transport intents; their
    # immutable contract evidence is carried into downstream prompts.
    assert all(call["decision_type"] == "synthesis_provider_call"
               for call in capture.calls)
    assert all(context["attempt"] >= 1 for context in contexts)
    assert all(context["intent_run_id"] == "synthetic-run"
               for context in contexts)
    assert all(context["intent_cell_id"] == "synthetic-cell"
               for context in contexts)
    assert all(context["intent_contract_sha256"] == "c" * 64
               for context in contexts)
    assert [
        context["max_output_tokens"] for context in contexts
    ] == base.request_max_tokens
    assert [
        context["intent_request_sha256"] for context in contexts
    ] == [row["request_sha256"] for row in manifest.rows]
    assert all(context["synthesis_seed"] == 0 for context in contexts)
    assert all(context["projections"] == {
        "instruction": MICRO_SFT_PROJECTION,
        "preference": MICRO_DPO_PROJECTION,
    } for context in contexts)
    assert instruction["projection_contract"] == MICRO_SFT_PROJECTION
    assert preference["projection_contract"] == MICRO_DPO_PROJECTION
    assert instruction["synthesis_seed"] == preference["synthesis_seed"] == 0
    assert instruction["seed"] == preference["seed"] == 0
    assert instruction["decision_capture_id"] == capture.decisions[1]["event_id"]
    assert preference["decision_capture_id"] == capture.decisions[-1]["event_id"]
    assert "fabricated" not in preference["decision_capture_id"]
    assert preference["decision_capture_id"] != instruction["decision_capture_id"]
    assert preference["chunk_id"] == preference["source_chunk_id"] == chunk["id"]
    assert preference["lo_refs"] == ["co-1"]
    assert preference["provider"] == "local"
    assert preference["provenance"]["source_chunk_id"] == chunk["id"]
    assert preference["provenance"]["source_refs"] == ["generic-chunk"]
    assert preference["provenance"]["provider"] == "local"
    assert preference["source_row_identity"]["chunk_id"] == chunk["id"]
    assert len(preference["source_row_identity"]["chunk_sha256"]) == 64
    preference_schema = json.loads(
        (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "schemas/knowledge/preference_pair.schema.json"
        ).read_text(encoding="utf-8")
    )
    pytest.importorskip("jsonschema").validate(
        instance=preference, schema=preference_schema,
    )
    instruction_schema = json.loads(
        (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "schemas/knowledge/instruction_pair.schema.json"
        ).read_text(encoding="utf-8")
    )
    pytest.importorskip("jsonschema").validate(
        instance=instruction, schema=instruction_schema,
    )
    assert json.loads(json.dumps(
        preference, sort_keys=True, ensure_ascii=False,
    )) == preference
    assert (
        instruction["release_identity_sha256"]
        == preference["release_identity_sha256"]
        == contexts[0]["semantic_identity_sha256"]
    )
    assert any("SLOT=0" in call["decision"] or "micro_B_claim_0" in context["stage"]
               for call, context in zip(capture.calls, contexts))
    assert all("stage=" in call["rationale"] and "attempt=" in call["rationale"]
               for call in capture.calls)
    quality = preference["preference_quality"]
    assert quality["chosen_score"] == 1.0
    assert quality["chosen_minus_rejected_margin"] > 0
    assert quality["one_fault_separation_margin"] > 0
    assert quality["confidence"] > 0
    assert quality["fault_taxonomy"]
    assert len(quality["rationale_sha256"]) == 64
