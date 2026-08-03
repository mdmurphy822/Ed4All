"""Versioned micro-call synthesis for source-grounded training pairs.

This module is deliberately opt-in.  The legacy and staged-v4 providers are
not changed unless ``TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1`` is explicitly
truthy.  Calls use the staged provider's audited strict-JSON transport,
DecisionCapture, intent/HTTP ledger, context-window, and terminal-error seams.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import inspect
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from lib.generation import stop_control
from lib.ontology.misconception_id import canonical_mc_id
from Trainforge.generators.providers._synthesis_common import SynthesisProviderError
from Trainforge.generators.staged.provider import (
    ENV_SERVED_CONTEXT_TOKENS,
    StagedSynthesisProvider,
    _BLOOM,
    _PLAN_CONTRADICTION_CEILING,
    _PLAN_ENTAILMENT_FLOOR,
    _NUMBER_RE,
    _SUPERSCRIPT_SEQUENCE_RE,
    _SUPERSCRIPT_TRANSLATION,
    _clean,
    _claim_repair_diff,
    _claim_repair_guard,
    _deterministic_entailment_basis,
    _deterministic_derivation_ledger,
    _equal_canonical_proposition,
    _explicit_relation_conflict,
    _first_distinctive_verbatim_span,
    formula_relation_proof,
    affine_two_line_relation_proof,
    _relation_operator_mutation,
    _relation_preserving_support,
    _number_key,
    _normalized_proposition,
    _scenario_numeric_literals,
    _sentences,
    _sha,
    _stable_json,
    _symbolic_math_supports,
)
from Trainforge.generators.staged.window_contract import (
    build_evidence_window,
    canonical_quote,
    objective_card,
    resolve_bloom_windows_enabled,
)
from Trainforge.generators.staged.objective_contract import (
    OBJECTIVE_REQUIREMENT_CONTRACT_VERSION,
    build_private_sidecar,
    content_sha256,
    derive_objective_requirements,
    load_private_sidecar,
    objective_requirement_contract_components,
    reconcile_completion_execution,
    write_private_sidecar,
)


MICRO_CONTRACT_VERSION = "ed4all.staged-synthesis-micro.v1"
MICRO_RELEASE_CONTRACT_VERSION = "training-synthesis-release.v1.2.0"
MICRO_SFT_PROJECTION = "ed4all-sft-chat.v2"
MICRO_DPO_PROJECTION = "ed4all-dpo-preference.v2"
MICRO_STAGE_TOKEN_BUDGET_VERSION = "micro-stage-max-tokens.v1"
MICRO_STAGE_A_SEED_CONTRACT_VERSION = "ed4all.micro-stage-a-seeds.v4"
MICRO_STAGE_B_PUBLIC_CLAIM_VERSION = "ed4all.micro-stage-b-public-claim.v2"
MICRO_STAGE_D_REALIZATION_VIEW_VERSION = (
    "ed4all.micro-stage-d-realization-view.v1"
)
MICRO_STAGE_D_REALIZATION_CONTRACT_VERSION = (
    "ed4all.micro-stage-d-claim-realizations.v3"
)
MICRO_STAGE_D_PROMPT_CONTRACT_VERSION = "ed4all.micro-stage-d-prompt.v1"
MICRO_STAGE_D_PROMPT_MAX_CHARS = 400
MICRO_COMPLETION_CAP_CANDIDATES = (600, 800, 1000, 1200, 1600)
MICRO_DEFAULT_COMPLETION_CAP = 600
MICRO_STAGE_E_AUTHORITY_CONTRACT_VERSION = (
    "ed4all.micro-stage-e-misconception-authority.v2"
)
MICRO_STAGE_MAX_TOKENS = {
    "A": 2048,
    "B": 1536,
    "D": 1536,
    "E": 1280,
    "F": 1024,
}
ENV_STAGED_SYNTHESIS_MICRO_V1 = "TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1"
_TRUE = frozenset({"1", "true", "yes", "on"})
_STAGES = ("A", "B", "C", "D", "E", "F", "Q")
_TASK_SYSTEM = (
    "Design one objective-bound learner task. Do not extract evidence or write "
    "an answer. Thinking is disabled: do not emit reasoning. Return only strict JSON."
)
_CLAIM_SYSTEM = (
    "Return one leakage-safe PUBLIC atomic evidence claim and one exact PRIVATE "
    "quote from the assigned source block. The public claim must faithfully "
    "paraphrase the private quote without copying a distinctive 50-character "
    "span from the quote or assigned block. Never return a list. Thinking is "
    "disabled: do not emit reasoning. Return only strict JSON."
)
_REALIZE_SYSTEM = (
    "Realize the immutable assembled task and claims. Add no claims or numeric "
    "facts. Thinking is disabled: do not emit reasoning. Return only strict JSON."
)
_MISCONCEPTION_SYSTEM = (
    "Select one canonical misconception ID from the supplied authority and "
    "write only its rationale. The orchestrator derives the exact faulty step; "
    "do not return it, invent an error-mechanism label, or write an answer. "
    "Thinking is disabled: do not emit reasoning. Return only strict JSON."
)
_REJECTED_SYSTEM = (
    "Write only one rejected answer exhibiting exactly the selected fault. "
    "Do not alter any supplied metadata or introduce numeric facts. Thinking "
    "is disabled: do not emit reasoning."
)


def staged_synthesis_micro_v1_enabled(
    environment: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve the explicit default-off switch; garbage remains legacy."""
    source = os.environ if environment is None else environment
    return (
        str(source.get(ENV_STAGED_SYNTHESIS_MICRO_V1, "")).strip().lower()
        in _TRUE
    )


# ---------------------------------------------------------------------------
# Generation-unit identity.
#
# ``_resume_store`` keys the per-unit resume FILE PATH on an identity whose
# only caller-supplied fields are ``variant`` and ``repetition`` — every other
# field it re-derives from the chunk and draft it was already handed.  That
# makes ``variant`` the sole discriminator between two generation units of the
# same chunk and kind, and a blank one silently collapses them onto one file
# (variant 1 then replays variant 0's terminal artifacts stage for stage).
#
# There are two ways to supply it, in precedence order:
#   1. ``draft["_micro_manifest_identity"]`` — the pilot harness stamps its
#      Latin-square cell coordinates there and that binding stays authoritative.
#   2. :func:`bind_micro_generation_unit` — the production path wraps each
#      generation unit in this context manager.  ``production_variant_token``
#      derives the token from ``(kind, variant_index)``, the canonical
#      production unit key already used by the checkpoint cache and the
#      generation journal, so the micro resume store stays 1:1 with the units
#      the outer journal tracks.
# Neither present is a LOUD failure — never a defaulted identity.
# ---------------------------------------------------------------------------
MICRO_PRODUCTION_REPETITION = 0

_MICRO_UNIT_BINDING: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = (
    contextvars.ContextVar("_MICRO_UNIT_BINDING", default=None)
)


def production_variant_token(kind: str, variant_index: int) -> str:
    """Return the injective-in-``variant_index`` production variant token."""
    return f"{_clean(kind)}-{int(variant_index)}"


@contextlib.contextmanager
def bind_micro_generation_unit(
    *, kind: str, variant_index: int,
    repetition: int = MICRO_PRODUCTION_REPETITION,
) -> "Iterator[Dict[str, Any]]":
    """Bind the current generation unit for micro's resume-store identity.

    Thread-safe: the binding lives in a :mod:`contextvars` variable, and worker
    threads start from a fresh context, so a binding established inside a
    worker is invisible to every other worker.  Non-micro providers never read
    it, so wrapping a call is a no-op for them.
    """
    binding = {
        "kind": _clean(kind),
        "variant": production_variant_token(kind, variant_index),
        "repetition": int(repetition),
    }
    token = _MICRO_UNIT_BINDING.set(binding)
    try:
        yield binding
    finally:
        _MICRO_UNIT_BINDING.reset(token)


def current_micro_generation_unit() -> Optional[Dict[str, Any]]:
    """Return the bound generation unit, or ``None`` when unbound."""
    binding = _MICRO_UNIT_BINDING.get()
    return dict(binding) if isinstance(binding, Mapping) else None


def build_micro_manifest_identity(
    chunk: Mapping[str, Any], *, kind: str, variant: Any,
    repetition: Any = MICRO_PRODUCTION_REPETITION,
    draft: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the canonical micro draft identity for one generation unit.

    Single source of truth: ``_resume_store`` builds BOTH the expected identity
    and (on the bound path) the supplied one through this function, so the two
    cannot drift apart into a spurious ``staged_micro_draft_identity_invalid``.
    """
    variant_token = _clean(variant)
    if not variant_token:
        # A blank variant passes an equality check against ``_clean(None)``
        # while destroying the only per-unit discriminator in the resume-store
        # path.  Refuse it rather than let two units share one state file.
        raise SynthesisProviderError(
            "micro synthesis draft identity requires a non-blank variant: it "
            "keys the per-unit resume-store path, so a blank one lets two "
            "generation units of the same chunk replay each other's stages",
            code="staged_micro_draft_identity_variant_blank",
        )
    try:
        repetition_index = int(repetition)
    except (TypeError, ValueError) as exc:
        raise SynthesisProviderError(
            "micro synthesis draft identity requires an integer repetition",
            code="staged_micro_draft_identity_invalid",
        ) from exc
    return {
        "chunk_id": _clean(chunk.get("id") or chunk.get("chunk_id")),
        "chunk_sha256": _sha(_stable_json(chunk)),
        "kind": kind,
        "variant": variant_token,
        "repetition": repetition_index,
        "draft": {
            "provider": _clean(draft.get("provider")),
            "source_chunk_id": _clean(draft.get("source_chunk_id")),
        },
    }


def _schema_sha(schema: Mapping[str, Any]) -> str:
    return _sha(_stable_json(schema))


def micro_contract_components() -> Dict[str, Any]:
    """Return every behavior-bearing component used by resume/publication."""
    schemas = {
        "task": _TASK_SCHEMA,
        "claim": _CLAIM_SCHEMA,
        "sft": _SFT_SCHEMA,
        "chosen": _CHOSEN_SCHEMA,
        "misconception": _MISCONCEPTION_SCHEMA,
        "rejected": _REJECTED_SCHEMA,
    }
    router_contract = {
        "ordering": ["source_block_id", "block_text_sha256"],
        "eligible_polarities": ["corrective", "factual"],
        "max_claim_slots": 3,
        "one_block_per_slot": True,
    }
    return {
        "version": MICRO_CONTRACT_VERSION,
        "release_contract_version": MICRO_RELEASE_CONTRACT_VERSION,
        "projection_contracts": {
            "instruction": MICRO_SFT_PROJECTION,
            "preference": MICRO_DPO_PROJECTION,
        },
        "stages": list(_STAGES),
        "systems_sha256": {
            "A": _sha(_TASK_SYSTEM),
            "B": _sha(_CLAIM_SYSTEM),
            "D": _sha(_REALIZE_SYSTEM),
            "E": _sha(_MISCONCEPTION_SYSTEM),
            "F": _sha(_REJECTED_SYSTEM),
        },
        "schemas_sha256": {key: _schema_sha(value) for key, value in schemas.items()},
        "entailment_floor": _PLAN_ENTAILMENT_FLOOR,
        "contradiction_ceiling": _PLAN_CONTRADICTION_CEILING,
        "claim_slots": 3,
        "attempts_per_claim_slot": 2,
        "stage_token_budget": {
            "version": MICRO_STAGE_TOKEN_BUDGET_VERSION,
            "max_tokens": dict(MICRO_STAGE_MAX_TOKENS),
            "deterministic_stages": ["A", "C"],
        },
        "stage_a_seed_contract": {
            "version": MICRO_STAGE_A_SEED_CONTRACT_VERSION,
            "model_fields": [],
            "source_fields": [
                "objective_id", "bloom_level", "bloom_verb", "action_object",
                "conditions", "content_obligations", "performance_criteria",
            ],
            "assembler_sha256": _sha(
                inspect.getsource(deterministic_stage_a_task)
            ),
            "bounded_composer_sha256": _sha(
                inspect.getsource(compose_bounded_learner_task)
            ),
            "prompt_max_chars": MICRO_STAGE_D_PROMPT_MAX_CHARS,
            "normalizer_sha256": _sha(
                inspect.getsource(deterministic_scenario_givens)
            ),
        },
        "stage_b_public_claim_contract": {
            "version": MICRO_STAGE_B_PUBLIC_CLAIM_VERSION,
            "validator_sha256": _sha(
                inspect.getsource(
                    MicroStagedSynthesisProvider._validate_claim
                )
            ),
            "formula_relation_proof_sha256": _sha(
                inspect.getsource(formula_relation_proof)
            ),
            "private_fields": ["evidence_quote", "source_block_id"],
            "editable_repair_fields": ["claim"],
            "verbatim_boundary_characters": 50,
        },
        "stage_d_realization_view_contract": {
            "version": MICRO_STAGE_D_REALIZATION_VIEW_VERSION,
            "projector_sha256": _sha(
                inspect.getsource(stage_d_realization_view)
            ),
            "private_fields": ["evidence_quote", "distinctive_source_text"],
        },
        "stage_d_realization_contract": {
            "version": MICRO_STAGE_D_REALIZATION_CONTRACT_VERSION,
            "schema_sha256": _schema_sha(_REALIZATION_SCHEMA),
            "schema_factory_sha256": _sha(
                inspect.getsource(_stage_d_realization_schema)
            ),
            "assembler_sha256": _sha(
                inspect.getsource(assemble_claim_realizations)
            ),
            "coverage_unit": "whole_bound_claim_realization",
            "model_calls": 1,
            "completion_cap_candidates": list(
                MICRO_COMPLETION_CAP_CANDIDATES
            ),
            "default_completion_cap": MICRO_DEFAULT_COMPLETION_CAP,
            "overflow_policy": "reject_never_truncate",
            "objective_requirement_contract_version": (
                OBJECTIVE_REQUIREMENT_CONTRACT_VERSION
            ),
            "objective_requirement_contract": (
                objective_requirement_contract_components()
            ),
            "objective_requirement_derivation_sha256": _sha(
                inspect.getsource(derive_objective_requirements)
            ),
        },
        "stage_d_prompt_contract": {
            "version": MICRO_STAGE_D_PROMPT_CONTRACT_VERSION,
            "max_chars": MICRO_STAGE_D_PROMPT_MAX_CHARS,
            "assembler_sha256": _sha(
                inspect.getsource(deterministic_stage_d_prompt)
            ),
            "authority": "stage_c_immutable_learner_task",
            "model_fields": [],
        },
        "preference_eligibility_contract": {
            "version": "ed4all.micro-preference-eligibility.v1",
            "predicate_sha256": _sha(
                inspect.getsource(micro_preference_eligibility)
            ),
            "consumer_stages": ["preference_preflight", "E"],
        },
        "stage_e_authority_contract": {
            "version": MICRO_STAGE_E_AUTHORITY_CONTRACT_VERSION,
            "id_authority": (
                "lib.ontology.misconception_id.canonical_mc_id"
            ),
            "schema_factory_sha256": _sha(
                inspect.getsource(_stage_e_selection_schema)
            ),
            "authority_validator_sha256": _sha(
                inspect.getsource(_stage_e_authority)
            ),
            "resolver_sha256": _sha(
                inspect.getsource(_resolve_stage_e_selection)
            ),
            "exact_faulty_step_proof_sha256": _sha(
                inspect.getsource(_stage_e_exact_faulty_step_proof)
            ),
            "model_fields": ["misconception_id", "rationale"],
            "deterministic_fields": [
                "misconception_index", "error_mechanism", "faulty_step",
            ],
        },
        "numeric_contract": "closed-world.v1",
        "router_sha256": _sha(_stable_json(router_contract)),
        "assembly_sha256": _sha(inspect.getsource(assemble_claims)),
        "validator_sha256": {
            "typed_givens": _sha(inspect.getsource(validate_typed_givens)),
            "given_necessity": _sha(
                inspect.getsource(unnecessary_generated_givens_error)
            ),
            "immutable_artifact": _sha(inspect.getsource(immutable_artifact_error)),
            "numeric_closed_world": _sha(inspect.getsource(numeric_closed_world_error)),
            "claim": _sha(inspect.getsource(MicroStagedSynthesisProvider._validate_claim)),
            "prompt_numeric": _sha(
                inspect.getsource(_prompt_numeric_error)
                + inspect.getsource(_task_numeric_error)
            ),
            "leakage": _sha(inspect.getsource(_leakage_error)),
        },
        "implementation_sha256": {
            "resume_store": _sha(inspect.getsource(MicroResumeStore)),
            "journal_dispatch": _sha(
                inspect.getsource(MicroStagedSynthesisProvider._journaled)
            ),
            "stop_poll": _sha(
                inspect.getsource(MicroStagedSynthesisProvider._check_stop)
            ),
            "stage_budget_dispatch": _sha(
                inspect.getsource(MicroStagedSynthesisProvider._call_stage)
            ),
            "task_design": _sha(
                inspect.getsource(MicroStagedSynthesisProvider._task_design)
            ),
            "claim_slot": _sha(
                inspect.getsource(MicroStagedSynthesisProvider._claim_slot)
            ),
            "assembly_router": _sha(
                inspect.getsource(MicroStagedSynthesisProvider._assemble)
            ),
            "realization": _sha(
                inspect.getsource(MicroStagedSynthesisProvider._realize)
                + inspect.getsource(_claim_checklist)
                + inspect.getsource(_scoped_coverage_error)
            ),
            "instruction_flow": _sha(
                inspect.getsource(
                    MicroStagedSynthesisProvider.paraphrase_instruction
                )
            ),
            "preference_flow": _sha(
                inspect.getsource(
                    MicroStagedSynthesisProvider.paraphrase_preference
                )
            ),
        },
    }


def micro_contract_fingerprint() -> str:
    return _sha(_stable_json(micro_contract_components()))


def verify_expected_input_hashes(
    *,
    manifest_path: Path,
    eligibility_path: Path,
    expected_manifest_sha256: str,
    expected_eligibility_sha256: str,
) -> Dict[str, str]:
    """Fail closed on cohort identity before any provider construction/call."""
    expected = {
        "manifest_sha256": str(expected_manifest_sha256).lower(),
        "eligibility_sha256": str(expected_eligibility_sha256).lower(),
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in expected.values()):
        raise SynthesisProviderError(
            "expected manifest and eligibility SHA-256 values are required",
            code="staged_micro_expected_hash_missing",
        )
    actual = {
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "eligibility_sha256": hashlib.sha256(
            Path(eligibility_path).read_bytes()
        ).hexdigest(),
    }
    if actual != expected:
        raise SynthesisProviderError(
            "manifest or eligibility identity differs from the explicit expectation",
            code="staged_micro_input_identity_mismatch",
            details={"expected": expected, "actual": actual},
        )
    return actual


_GIVEN = {
    "type": "object",
    "additionalProperties": False,
    "required": ["symbol", "value", "unit", "role", "synthetic", "provenance"],
    "properties": {
        "symbol": {
            "type": "string",
            "minLength": 1,
            "maxLength": 32,
            "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$",
        },
        "value": {
            "type": "string",
            "minLength": 1,
            "maxLength": 32,
            "pattern": "^-?(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+)$",
        },
        "unit": {"type": "string", "minLength": 1, "maxLength": 40},
        "role": {
            "type": "string",
            "enum": ["rate", "distance", "time", "count", "constant", "other"],
        },
        "synthetic": {"const": True},
        "provenance": {"const": "generated"},
    },
}
_TASK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objective_id", "bloom_level", "learner_task"],
    "properties": {
        "objective_id": {"type": "string", "minLength": 1, "maxLength": 64},
        "bloom_level": {"type": "string", "enum": sorted(_BLOOM)},
        "learner_task": {"type": "string", "minLength": 1, "maxLength": 400},
    },
}
_CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claim", "evidence_quote", "source_block_id"],
    "properties": {
        "claim": {"type": "string", "minLength": 1, "maxLength": 300},
        "evidence_quote": {"type": "string", "minLength": 1, "maxLength": 500},
        "source_block_id": {"type": "string", "minLength": 1, "maxLength": 128},
    },
}
_REALIZATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["claim_realizations"],
    "properties": {
        "claim_realizations": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["claim_id", "realization"],
                "properties": {
                    "claim_id": {
                        "type": "string", "minLength": 16, "maxLength": 16,
                        "pattern": "^[0-9a-f]{16}$",
                    },
                    "realization": {
                        "type": "string", "minLength": 1, "maxLength": 500,
                    },
                },
            },
        },
    },
}
_SFT_SCHEMA = _REALIZATION_SCHEMA
_CHOSEN_SCHEMA = _REALIZATION_SCHEMA
_MISCONCEPTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["misconception_id", "rationale"],
    "properties": {
        "misconception_id": {
            "type": "string", "minLength": 19, "maxLength": 19,
            "pattern": "^mc_[0-9a-f]{16}$",
        },
        "rationale": {"type": "string", "minLength": 20, "maxLength": 320},
    },
}
_REJECTED_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["rejected"],
    "properties": {"rejected": {"type": "string", "minLength": 1, "maxLength": 600}},
}


def _stage_d_realization_schema(
    artifact: Mapping[str, Any], *,
    completion_cap: int = MICRO_DEFAULT_COMPLETION_CAP,
) -> Dict[str, Any]:
    """Bind the finite model envelope to immutable Stage-C claim identity."""
    claim_ids = [
        _clean(item.get("stable_id")) for item in artifact.get("claims") or []
    ]
    if (
        not 1 <= len(claim_ids) <= 3
        or len(set(claim_ids)) != len(claim_ids)
        or any(not re.fullmatch(r"[0-9a-f]{16}", item) for item in claim_ids)
    ):
        raise SynthesisProviderError(
            "Stage-D schema requires one to three unique immutable claim IDs",
            code="staged_micro_D_schema_identity_invalid",
        )
    schema = json.loads(_stable_json(_REALIZATION_SCHEMA))
    rows = schema["properties"]["claim_realizations"]
    rows["minItems"] = len(claim_ids)
    rows["maxItems"] = len(claim_ids)
    rows["items"]["properties"]["claim_id"] = {
        "type": "string",
        "enum": claim_ids,
    }
    rows["items"]["properties"]["realization"]["maxLength"] = completion_cap
    contract = artifact.get("objective_requirement_contract")
    if isinstance(contract, Mapping):
        requirements = [
            dict(item) for item in contract.get("requirements") or []
            if isinstance(item, Mapping)
        ]
        worked = [
            item for item in requirements if item.get("kind") != "result"
        ]
        result = [
            item for item in requirements if item.get("kind") == "result"
        ]
        if not worked:
            raise SynthesisProviderError(
                "Stage-D objective contract has no executable requirements",
                code="staged_micro_D_objective_requirements_invalid",
            )
        schema["required"].extend(["worked_steps", "result"])
        schema["properties"]["worked_steps"] = {
            "type": "array",
            "minItems": len(worked),
            "maxItems": len(worked),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "requirement_id", "claim_ids", "realization",
                ],
                "properties": {
                    "requirement_id": {
                        "type": "string",
                        "enum": [item["requirement_id"] for item in worked],
                    },
                    "claim_ids": {
                        "type": "array", "minItems": 1,
                        "maxItems": len(claim_ids), "uniqueItems": True,
                        "items": {"type": "string", "enum": claim_ids},
                    },
                    "realization": {
                        "type": "string", "minLength": 1,
                        "maxLength": completion_cap,
                    },
                },
            },
        }
        schema["properties"]["result"] = (
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "requirement_id", "claim_ids", "realization",
                ],
                "properties": {
                    "requirement_id": {
                        "type": "string",
                        "enum": [item["requirement_id"] for item in result],
                    },
                    "claim_ids": {
                        "type": "array", "minItems": 1,
                        "maxItems": len(claim_ids), "uniqueItems": True,
                        "items": {"type": "string", "enum": claim_ids},
                    },
                    "realization": {
                        "type": "string", "minLength": 1,
                        "maxLength": completion_cap,
                    },
                },
            }
            if result else {"type": "null"}
        )
    return schema


def _stage_e_selection_schema(
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Bind Gate E to the exact canonical misconception authority."""
    ids = [item["id"] for item in _stage_e_authority(candidates)]
    schema = json.loads(_stable_json(_MISCONCEPTION_SCHEMA))
    schema["properties"]["misconception_id"] = {
        "type": "string", "minLength": 19, "maxLength": 19, "enum": ids,
    }
    return schema


def _stage_e_authority(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Validate canonical IDs and content before exposing Gate-E authority."""
    rows = [dict(item) for item in candidates]
    ids: list[str] = []
    for item in rows:
        statement = _clean(item.get("misconception"))
        correction = _clean(item.get("correction"))
        bloom = _clean(item.get("bloom_level"))
        candidate_id = _clean(item.get("id"))
        expected = canonical_mc_id(statement, correction, bloom)
        if (
            not statement
            or not correction
            or not _clean(item.get("mechanism_evidence"))
            or candidate_id != expected
        ):
            raise SynthesisProviderError(
                "Stage-E candidate ID/content authority mismatch",
                code="staged_micro_E_authority_invalid",
            )
        ids.append(candidate_id)
    if not ids or len(set(ids)) != len(ids):
        raise SynthesisProviderError(
            "Stage-E authority contains duplicate or colliding IDs",
            code="staged_micro_E_authority_invalid",
        )
    return rows


def _resolve_stage_e_selection(
    candidates: Sequence[Mapping[str, Any]], misconception_id: Any,
) -> tuple[int, Dict[str, Any]]:
    """Resolve exactly one enumerated canonical ID; never index-fallback."""
    rows = _stage_e_authority(candidates)
    selected = _clean(misconception_id)
    matches = [
        (index, item) for index, item in enumerate(rows)
        if item["id"] == selected
    ]
    if len(matches) != 1:
        raise SynthesisProviderError(
            "misconception_id does not resolve to exactly one canonical candidate",
            code="staged_micro_E_authority_unresolved",
        )
    return matches[0]


def _stage_e_exact_faulty_step_proof(
    candidates: Sequence[Mapping[str, Any]], misconception_id: Any,
) -> tuple[int, Dict[str, Any], Dict[str, Any]]:
    """Resolve and prove byte-exact source-owned misconception content.

    No Unicode or semantic normalization participates in content authority.
    The canonical Python string is copied as-is; UTF-8 is used only to hash its
    exact audit bytes.
    """
    index, candidate = _resolve_stage_e_selection(candidates, misconception_id)
    faulty_step = candidate.get("misconception")
    if not isinstance(faulty_step, str) or not faulty_step:
        raise SynthesisProviderError(
            "resolved Stage-E authority lacks exact misconception content",
            code="staged_micro_E_authority_invalid",
        )
    proof = {
        "proof": "exact_canonical_faulty_step",
        "contract_version": MICRO_STAGE_E_AUTHORITY_CONTRACT_VERSION,
        "misconception_id": candidate["id"],
        "candidate_index": index,
        "faulty_step_sha256": hashlib.sha256(
            faulty_step.encode("utf-8")
        ).hexdigest(),
        "normalization": "none-byte-exact-utf8",
        "source_block_id": _clean(candidate.get("source_block_id")),
    }
    return index, candidate, proof


def deterministic_stage_d_prompt(artifact: Mapping[str, Any]) -> str:
    """Return the complete bounded Stage-A task; never truncate a sentence."""
    prompt = _clean(artifact.get("learner_task"))
    if (
        not prompt
        or len(prompt) > MICRO_STAGE_D_PROMPT_MAX_CHARS
        or prompt[-1] not in ".?!"
    ):
        raise SynthesisProviderError(
            "immutable Stage-D prompt is empty, over bound, or sentence-incomplete",
            code="staged_micro_D_prompt_invalid",
            details={
                "terminal_content_rejection": True,
                "stage": "micro_D",
                "prompt_chars": len(prompt),
                "prompt_max_chars": MICRO_STAGE_D_PROMPT_MAX_CHARS,
            },
        )
    return prompt


def validate_typed_givens(value: Any) -> Optional[str]:
    if not isinstance(value, list) or len(value) > 6:
        return "generated_givens must be an array of at most six items"
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            return f"generated_givens[{index}] is not an object"
        symbol = _clean(item.get("symbol"))
        scalar = _clean(item.get("value"))
        unit = _clean(item.get("unit"))
        role = _clean(item.get("role"))
        if (
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", symbol)
            or symbol in seen
            or len(scalar) > 32
            or not re.fullmatch(r"-?(?:\d+(?:\.\d+)?|\.\d+)", scalar)
            or not unit
            or len(unit) > 40
            or role not in {
                "rate", "distance", "time", "count", "constant", "other",
                "scenario_given",
            }
            or item.get("synthetic") is not True
            or item.get("provenance") != "generated"
        ):
            return f"generated_givens[{index}] violates typed provenance"
        try:
            decimal = Decimal(scalar)
        except InvalidOperation:
            return f"generated_givens[{index}].value is not a finite scalar"
        digits = decimal.as_tuple().digits
        fractional_digits = max(0, -decimal.as_tuple().exponent)
        if (
            not decimal.is_finite()
            or abs(decimal) > Decimal("1000000000000")
            or len(digits) > 15
            or fractional_digits > 8
        ):
            return (
                f"generated_givens[{index}].value exceeds magnitude or "
                "precision bounds"
            )
        normalized_unit = re.sub(r"[^a-z0-9/]+", "", unit.casefold())
        incompatible = {
            "count": {
                "s", "sec", "secs", "second", "seconds", "min", "minute",
                "minutes", "m", "meter", "meters", "km", "kilometer",
                "kilometers", "m/s", "km/h", "mph",
            },
            "time": {
                "item", "items", "object", "objects", "person", "people",
                "meter", "meters", "km", "kilometer", "kilometers", "m/s",
                "km/h", "mph",
            },
            "distance": {
                "item", "items", "second", "seconds", "minute", "minutes",
                "hour", "hours", "m/s", "km/h", "mph",
            },
            "rate": {
                "item", "items", "object", "objects", "second", "seconds",
                "minute", "minutes", "hour", "hours", "meter", "meters",
                "km", "kilometer", "kilometers",
            },
            "constant": {"m/s", "km/h", "mph"},
        }
        if normalized_unit in incompatible.get(role, set()):
            return f"generated_givens[{index}] has incoherent role and unit"
        seen.add(symbol)
    return None


def deterministic_scenario_givens(
    learner_task: Any, *, allowed_numeric_text: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """Normalize novel Stage-A scenario scalars into stable seed columns."""
    task = _clean(learner_task)
    allowed = set(_scenario_numeric_literals(allowed_numeric_text))
    structural_spans: list[tuple[int, int]] = []
    structural_spans.extend(
        match.span(1)
        for match in re.finditer(
            r"\^\s*(-?(?:\d+(?:\.\d+)?|\.\d+))", task,
        )
    )
    structural_spans.extend(
        match.span(1)
        for match in re.finditer(r"(?<=[A-Za-z_])(\d+)", task)
    )
    structural_spans.extend(
        match.span(1)
        for match in re.finditer(
            r"\b(?:equation|step|case)\s+(\d+)\b", task,
            flags=re.IGNORECASE,
        )
    )
    answer_pattern = re.compile(
        r"\b(?:answer|result|solution)\s+(?:is|equals)\s*"
        r"-?(?:\d+(?:\.\d+)?|\.\d+)"
        r"|\bthere\s+(?:are|is)\s+-?(?:\d+(?:\.\d+)?|\.\d+)"
        r"|\b(?:therefore|thus|hence)\b[^.]{0,40}"
        r"-?(?:\d+(?:\.\d+)?|\.\d+)",
        flags=re.IGNORECASE,
    )
    if answer_pattern.search(task):
        return [], {}, (
            "/learner_task contains an answer-bearing numeric conclusion; "
            "scenario normalization cannot legalize results"
        )

    occurrences: Dict[str, list[Dict[str, Any]]] = {}
    for match in _NUMBER_RE.finditer(task):
        if any(
            start <= match.start() and match.end() <= end
            for start, end in structural_spans
        ):
            continue
        key = _number_key(match.group(0))
        if key in allowed:
            continue
        occurrences.setdefault(key, []).append({
            "pointer": "/learner_task",
            "start": match.start(),
            "end": match.end(),
            "occurrence": len(occurrences.get(key, [])),
        })
    ordered = sorted(
        occurrences,
        key=lambda key: occurrences[key][0]["start"],
    )
    givens: list[Dict[str, Any]] = []
    for index, key in enumerate(ordered):
        end = occurrences[key][0]["end"]
        suffix = task[end:end + 32]
        unit_match = re.match(
            r"\s*(?:%|[A-Za-z]+(?:/[A-Za-z]+)?)"
            r"(?=\s|[.,;:!?)]|$)",
            suffix,
        )
        candidate = _clean(unit_match.group(0)) if unit_match else ""
        if (
            re.match(r"\s*(?:/|\bper\b)", suffix, flags=re.IGNORECASE)
            and candidate.casefold() not in {"m/s", "km/h"}
        ):
            return [], {}, (
                "/learner_task has ambiguous required unit context for "
                f"numeric literal {key!r}"
            )
        if candidate.casefold() not in {
            "%", "item", "items", "object", "objects", "person", "people",
            "second", "seconds", "minute", "minutes", "hour", "hours",
            "meter", "meters", "kilometer", "kilometers", "km", "m",
            "mph", "m/s", "km/h", "dollar", "dollars", "percent",
        }:
            candidate = ""
        unit = "percent" if candidate == "%" else (candidate or "scalar")
        givens.append({
            "symbol": f"generated_given_{index}",
            "value": key,
            "unit": unit,
            "role": "scenario_given",
            "synthetic": True,
            "provenance": "generated",
            "source_pointer": "/learner_task",
            "occurrences": occurrences[key],
        })
    evidence = {
        "contract_version": MICRO_STAGE_A_SEED_CONTRACT_VERSION,
        "allowed_numeric_literals": sorted(
            allowed, key=lambda item: (Decimal(item), item),
        ),
        "normalized_values": [item["value"] for item in givens],
        "normalization_sha256": _sha(_stable_json(givens)),
    }
    return givens, evidence, None


def deterministic_stage_a_task(
    objective: Mapping[str, Any], *, synthesis_seed: int,
) -> Dict[str, Any]:
    """Assemble a learner task from source-owned objective-card fields.

    Stage A is an expression-like deterministic transformation, not a
    generative inference surface. The seed selects only generic pedagogical
    sentence ordering; it never changes or regenerates objective content.
    """
    canonical_dimensions = {
        "conditions", "content_obligations", "performance_criteria",
        "content_obligation_anchors",
    }
    card = (
        dict(objective)
        if canonical_dimensions <= set(objective)
        else objective_card(objective)
    )
    required_scalars = ("id", "bloom_level", "bloom_verb", "action_object")
    missing = [key for key in required_scalars if not _clean(card.get(key))]
    if missing:
        raise SynthesisProviderError(
            "deterministic Stage A objective card is incomplete: "
            + ", ".join(missing),
            code="staged_micro_A_objective_card_incomplete",
            details={
                "terminal_content_rejection": True,
                "stage": "micro_A",
                "model_calls": 0,
                "missing_fields": missing,
            },
        )
    bloom = _clean(card["bloom_level"]).lower()
    if bloom not in _BLOOM:
        raise SynthesisProviderError(
            "deterministic Stage A objective card has an invalid Bloom action",
            code="staged_micro_A_objective_card_incomplete",
            details={
                "terminal_content_rejection": True,
                "stage": "micro_A",
                "model_calls": 0,
                "missing_fields": ["bloom_level_or_verb"],
            },
        )
    dimensions = {
        "conditions": [
            _clean(item) for item in card.get("conditions") or [] if _clean(item)
        ],
        "content_obligations": [
            _clean(item)
            for item in card.get("content_obligations") or []
            if _clean(item)
        ],
        "performance_criteria": [
            _clean(item)
            for item in card.get("performance_criteria") or []
            if _clean(item)
        ],
    }
    action = _clean(card["bloom_verb"])
    action_object = _clean(card["action_object"])
    seed_material = _stable_json({
        "contract_version": MICRO_STAGE_A_SEED_CONTRACT_VERSION,
        "objective_card": card,
        "synthesis_seed": synthesis_seed,
    })
    variant = 0
    learner_task = compose_bounded_learner_task(
        card, max_chars=MICRO_STAGE_D_PROMPT_MAX_CHARS,
    )
    prompt_coverage = _learner_task_coverage(card, learner_task)
    contract_identity = {
        "version": MICRO_STAGE_A_SEED_CONTRACT_VERSION,
        "assembler_sha256": _sha(inspect.getsource(deterministic_stage_a_task)),
        "objective_card_sha256": _sha(_stable_json(card)),
        "synthesis_seed": synthesis_seed,
        "template_variant": variant,
    }
    contract_identity["contract_sha256"] = _sha(_stable_json(contract_identity))
    return {
        "objective_id": card["id"],
        "bloom_level": bloom,
        "learner_task": learner_task,
        "generated_givens": [],
        "_normalization_evidence": {
            "contract_version": MICRO_STAGE_A_SEED_CONTRACT_VERSION,
            "contract_sha256": contract_identity["contract_sha256"],
            "allowed_numeric_literals": sorted(
                _scenario_numeric_literals(_stable_json(card)),
                key=lambda item: (Decimal(item), item),
            ),
            "normalized_values": [],
            "normalization_sha256": _sha("[]"),
            "provider_decision_capture_id": "",
        },
        "_stage_a_deterministic": {
            **contract_identity,
            "schema": "ed4all.micro-stage-a-deterministic.v3",
            "model_calls": 0,
            "decision_capture_events": 0,
            "learner_task_sha256": _sha(learner_task),
            "prompt_coverage": prompt_coverage,
            "prompt_coverage_sha256": _sha(_stable_json(prompt_coverage)),
            "telemetry": {
                "deterministic_events": 1,
                "model_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
        "_objective_card": card,
    }


def compose_bounded_learner_task(
    card: Mapping[str, Any], *, max_chars: int = 400,
) -> str:
    """Compose a complete objective task within a hard projection boundary.

    Structured fields are authoritative. Exact duplicate obligations are
    represented once by the action object; no semantic field is sliced.
    """
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    action = _clean(card.get("bloom_verb"))
    action_object = _clean(card.get("action_object"))
    if not action or not action_object:
        raise SynthesisProviderError(
            "bounded learner task lacks its action/Bloom authority",
            code="staged_micro_A_prompt_budget_unsatisfied",
        )

    def unique(field: str) -> list[str]:
        values: list[str] = []
        seen: set[tuple[str, ...]] = set()
        for raw in card.get(field) or []:
            value = _clean(raw).rstrip(".?!")
            key = _normalized_proposition(value)
            if value and key not in seen:
                seen.add(key)
                values.append(value)
        return values

    conditions = unique("conditions")
    obligations = unique("content_obligations")
    criteria = unique("performance_criteria")
    action_key = _normalized_proposition(action_object)
    uncovered_obligations = [
        value for value in obligations
        if _normalized_proposition(value) != action_key
    ]
    represented = [action, action_object, *conditions, *obligations, *criteria]
    expected_numeric = _scenario_numeric_literals(_stable_json({
        "action": action,
        "action_object": action_object,
        "conditions": conditions,
        "content_obligations": obligations,
        "performance_criteria": criteria,
    }))
    groups = [
        f"{action.capitalize()} {action_object.rstrip('.?!')}.",
        *(["Conditions: " + "; ".join(conditions) + "."] if conditions else []),
        *(
            ["Obligations: " + "; ".join(uncovered_obligations) + "."]
            if uncovered_obligations else []
        ),
        *(["Criteria: " + "; ".join(criteria) + "."] if criteria else []),
        "Use source-grounded evidence; explain reasoning; do not copy an answer.",
    ]
    alternatives = [" ".join(groups)]
    compact_requirements = [
        *(conditions or []), *uncovered_obligations, *criteria,
    ]
    if compact_requirements:
        alternatives.append(
            f"{action.capitalize()} {action_object.rstrip('.?!')}. "
            f"Requirements: {'; '.join(compact_requirements)}. "
            "Use source-grounded evidence; explain reasoning; do not copy an answer."
        )

    def contains_tokens(container: tuple[str, ...], needle: tuple[str, ...]) -> bool:
        return bool(needle) and any(
            container[index:index + len(needle)] == needle
            for index in range(len(container) - len(needle) + 1)
        )

    valid: list[str] = []
    missing: list[str] = []
    for candidate in alternatives:
        normalized = _normalized_proposition(candidate)
        candidate_missing = [
            value for value in represented
            if not contains_tokens(normalized, _normalized_proposition(value))
        ]
        if not candidate_missing:
            missing = []
        elif not valid:
            missing = candidate_missing
        if (
            not candidate_missing
            and _scenario_numeric_literals(candidate) == expected_numeric
            and 40 <= len(candidate) <= max_chars
            and candidate[-1] in ".?!"
        ):
            valid.append(candidate)
    if not valid:
        shortest = min(alternatives, key=lambda item: (len(item), item))
        raise SynthesisProviderError(
            "complete objective task cannot fit the prompt projection",
            code="staged_micro_A_prompt_budget_unsatisfied",
            details={
                "terminal_content_rejection": True,
                "stage": "micro_A",
                "prompt_chars": len(shortest),
                "prompt_max_chars": max_chars,
                "missing_sha256": [_sha(value) for value in missing],
                "numeric_contract_matches": (
                    _scenario_numeric_literals(shortest) == expected_numeric
                ),
            },
        )
    return min(valid, key=lambda item: (len(item), item))


def _learner_task_coverage(
    card: Mapping[str, Any], prompt: str,
) -> list[Dict[str, Any]]:
    """Bind each structured objective atom to an exact prompt span."""
    atoms: list[tuple[str, str, str]] = [
        ("/bloom_verb", "bloom", _clean(card.get("bloom_verb"))),
        ("/action_object", "action", _clean(card.get("action_object"))),
    ]
    for field, kind in (
        ("conditions", "condition"),
        ("content_obligations", "obligation"),
        ("performance_criteria", "criterion"),
    ):
        atoms.extend(
            (f"/{field}/{index}", kind, _clean(value).rstrip(".?!"))
            for index, value in enumerate(card.get(field) or [])
            if _clean(value)
        )
    lowered = prompt.casefold()
    ledger: list[Dict[str, Any]] = []
    for pointer, kind, value in atoms:
        start = lowered.find(value.casefold())
        if start < 0:
            raise SynthesisProviderError(
                "learner-task coverage ledger cannot bind an objective atom",
                code="staged_micro_A_prompt_budget_unsatisfied",
            )
        end = start + len(value)
        ledger.append({
            "pointer": pointer,
            "kind": kind,
            "span": [start, end],
            "value_sha256": _sha(value),
            "numeric_literals": sorted(_scenario_numeric_literals(value)),
            "formula_literals": re.findall(
                r"\b[A-Za-z]\s*=\s*[A-Za-z0-9^*/+\-]+", value,
            ),
            "unit_constraints": re.findall(
                r"\b(?:consistent|inconsistent|same|different)\s+units?\b",
                value.casefold(),
            ),
        })
    return ledger


def micro_preference_eligibility(
    chunk: Mapping[str, Any], *, focus: Mapping[str, Any],
    capture: Optional[Any] = None,
) -> Dict[str, Any]:
    """Resolve the exact source dependencies required by micro Stage E.

    ``capture`` (optional; a ``DecisionCapture``-shaped object exposing
    ``log_decision``) records a ``dpo_pair_from_misconception`` event per
    Arm-A admitted candidate -- but ONLY when ``TRAINFORGE_BLOOM_WINDOWS``
    is on, like every other behavior in this initiative, so a flag-off run
    emits a byte-identical capture stream. ``None`` (the default, and every
    pre-existing caller that does not thread one) is a silent no-op either
    way, so this is purely additive telemetry.
    """
    chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
    try:
        window = build_evidence_window(
            chunk, focus, target_rung=focus.get("bloom_level"),
        )
    except ValueError as exc:
        reason = "preference_misconception_source_polarity_unresolved"
        telemetry = {
            "contract": "ed4all.micro-preference-eligibility.v1",
            "chunk_id": chunk_id,
            "eligible": False,
            "reason_code": reason,
            "candidate_count": 0,
            "candidate_ids": [],
            "rejection_counts": {},
            "evidence_window_sha256": None,
            "source_window_error": _clean(exc),
        }
        return {
            "eligible": False,
            "reason": reason,
            "candidates": [],
            "telemetry": telemetry,
        }
    blocks = list(window.get("blocks") or [])
    source_evidence = " ".join(_clean(block.get("text")) for block in blocks)
    candidates: list[Dict[str, Any]] = []
    rejection_counts = {
        "claim_missing": 0,
        "correction_missing": 0,
        "source_unbacked": 0,
        "mechanism_missing": 0,
        "id_mismatch": 0,
        "id_collision": 0,
    }
    bloom_level = _clean(
        focus.get("bloom_level") or chunk.get("bloom_level")
    )

    def source_backs(statement: str) -> bool:
        if statement.casefold() in source_evidence.casefold():
            return True

        def normalized_tokens(value: str) -> set[str]:
            tokens = set(re.findall(r"[a-z][a-z'-]{2,}", value.casefold()))
            normalized = set()
            for token in tokens:
                if token.endswith("ing") and len(token) > 5:
                    token = token[:-3]
                elif token.endswith("es") and len(token) > 4:
                    token = token[:-2]
                elif token.endswith("s") and len(token) > 3:
                    token = token[:-1]
                normalized.add(token)
            return normalized

        required = normalized_tokens(statement)
        return bool(required) and required <= normalized_tokens(source_evidence)

    for item in chunk.get("misconceptions") or []:
        if not isinstance(item, Mapping):
            continue
        # ``statement`` is a schema-legal alias for ``misconception``
        # (chunk_v4 $defs.Misconception requires one OR the other), and the
        # content-generator subagents emit it. Reading only one spelling made
        # a schema-valid misconception read as claim_missing here while
        # build_evidence_window read the OTHER spelling — one alias, two
        # readers, opposite keys.
        claim = _clean(item.get("misconception") or item.get("statement"))
        correction = _clean(item.get("correction"))
        mechanism = _clean(
            item.get("mechanism_evidence") or item.get("error_mechanism")
            or correction
        )
        if not claim:
            rejection_counts["claim_missing"] += 1
            continue
        if not correction:
            rejection_counts["correction_missing"] += 1
            continue
        if not source_backs(claim) or not source_backs(correction):
            rejection_counts["source_unbacked"] += 1
            continue
        if not mechanism:
            rejection_counts["mechanism_missing"] += 1
            continue
        # Arm A previously minted every card's id (and stamped every
        # candidate's own "bloom_level") from the single function-scoped
        # ``bloom_level`` (the objective/chunk-wide value) -- collapsing a
        # chunk that mixes cards at several rungs onto one shared id seed.
        # TRAINFORGE_BLOOM_WINDOWS on: the CARD's own ``bloom_level`` (set
        # by ``resolve_chunk_misconceptions`` / an authoring pipeline that
        # already populates the schema-legal per-item field) wins; absent
        # -> the chunk-wide value is still the fallback. Off, or the item
        # carries no rung of its own, keeps the pre-existing byte-identical
        # chunk-wide behavior.
        rung = bloom_level
        if resolve_bloom_windows_enabled():
            card_rung = _clean(item.get("bloom_level")).lower()
            if card_rung:
                rung = card_rung
        canonical_id = canonical_mc_id(claim, correction, rung)
        incoming_id = _clean(item.get("id"))
        if incoming_id and incoming_id != canonical_id:
            rejection_counts["id_mismatch"] += 1
            continue
        candidates.append({
            "id": canonical_id,
            "misconception": claim,
            "correction": correction,
            "mechanism_evidence": mechanism,
            "bloom_level": rung,
            "source_block_id": _clean(item.get("source_block_id")) or chunk_id,
            "source_role": "misconception_claim",
            "source_polarity": "incorrect_with_correction",
        })
        if capture is not None and resolve_bloom_windows_enabled():
            capture.log_decision(
                decision_type="dpo_pair_from_misconception",
                decision=(
                    f"Admitted DPO misconception candidate {canonical_id} "
                    f"for chunk {chunk_id} at rung {rung or 'unset'}"
                ),
                rationale=(
                    f"mc_id={canonical_id}, chunk_id={chunk_id}, "
                    f"rung={rung or 'unset'}, source_backs_claim=True, "
                    f"source_backs_correction=True; Arm A admits this "
                    f"candidate only once both the claim and its "
                    f"correction are lexically backed by the evidence "
                    f"window and the recomputed canonical id agrees with "
                    f"any supplied one, so this event fires once per "
                    f"admitted candidate each time Arm A evaluates this "
                    f"chunk's eligibility."
                ),
            )

    # Arm A is the DESIGNED path and Arm B below is its fallback for corpora
    # whose blocks carry no structural split. Once the chunk read seam
    # recovers authored cards into ``chunk["misconceptions"]``, BOTH arms see
    # the same card and mint the same content-addressed id — which the
    # collision check below would read as an id that fails to identify, and
    # reject the whole chunk. It is the same candidate found twice, not two
    # candidates colliding, so Arm B defers here. A repeat WITHIN Arm B still
    # collides: two distinct blocks minting one id is the case the check is
    # for.
    structured_ids = {candidate["id"] for candidate in candidates}

    claim_pattern = re.compile(
        r"(?:a\s+common\s+misconception\s+is\s+that|"
        r"students?\s+may\s+incorrectly\s+believe\s+that|"
        r"misconception:\s*[^.]*?\s+)"
        r"(?P<claim>[^.!?]+)[.!?]\s*(?P<correction>.+)",
        flags=re.IGNORECASE,
    )
    for block in blocks:
        block_id = _clean(block.get("block_id"))
        role = _clean(block.get("role")).casefold()
        polarity = _clean(block.get("polarity")).casefold()
        text = _clean(block.get("text"))
        role_enabled = (
            role in {"misconception_claim", "misconception_evidence"}
            or polarity == "incorrect"
            or "misconception" in block_id.casefold()
        )
        if not role_enabled:
            continue
        # An authored misconception card carries its own (claim, correction)
        # boundary in its markup; ``build_evidence_window`` recovers it as
        # ``misconception_pairs``.  Prefer it over the prose cue pattern: the
        # pattern requires one of three fixed openers, and authored prose has
        # no fixed opener, so it recognises a minority of real cards.  The
        # pattern stays as the fallback for corpora whose blocks carry no
        # structural split.
        structured = block.get("misconception_pairs") or []
        extracted: list[tuple[str, str, str]] = []
        if structured:
            for item in structured:
                if not isinstance(item, Mapping):
                    continue
                # The authored correction IS the mechanism evidence, matching
                # the structured arm above (``mechanism_evidence or
                # error_mechanism or correction``).
                correction = _clean(item.get("correction"))
                extracted.append((
                    _clean(item.get("claim")), correction, correction,
                ))
        else:
            match = claim_pattern.search(text)
            if match:
                extracted.append((
                    _clean(match.group("claim")),
                    _clean(match.group("correction")),
                    # No structural boundary is available on this path, so the
                    # whole block text remains the best mechanism evidence.
                    text,
                ))
        if not extracted:
            rejection_counts["claim_missing"] += 1
            continue
        for claim, correction, mechanism in extracted:
            if not claim:
                rejection_counts["claim_missing"] += 1
                continue
            if not correction:
                rejection_counts["correction_missing"] += 1
                continue
            # The structured arm above accepts any non-empty correction as
            # mechanism evidence.  This arm previously additionally required a
            # lexical token ("correct"/"not"/"instead"/...), which rejected
            # correct corrective prose that happened to phrase itself
            # differently ("non-zero" does not match \bnot\b).  One predicate
            # now governs both arms.
            if not mechanism:
                rejection_counts["mechanism_missing"] += 1
                continue
            fallback_id = canonical_mc_id(claim, correction, bloom_level)
            if fallback_id in structured_ids:
                # Already admitted by Arm A from the same authored card.
                continue
            candidates.append({
                "id": fallback_id,
                "misconception": claim,
                "correction": correction,
                "mechanism_evidence": mechanism,
                "bloom_level": bloom_level,
                "source_block_id": block_id,
                "source_role": (
                    role if role != "evidence" else "misconception_evidence"
                ),
                "source_polarity": (
                    polarity if polarity != "factual"
                    else "incorrect_with_correction"
                ),
            })

    by_id: Dict[str, Dict[str, Any]] = {}
    collisions: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate["id"]
        if candidate_id in by_id:
            collisions.add(candidate_id)
        else:
            by_id[candidate_id] = candidate
    if collisions:
        rejection_counts["id_collision"] += len(collisions)
        ordered = []
    else:
        ordered = sorted(
            by_id.values(),
            key=lambda item: (item["source_block_id"], item["id"]),
        )
    if collisions:
        reason = "preference_misconception_id_collision"
    elif ordered:
        reason = None
    elif rejection_counts["id_mismatch"]:
        reason = "preference_misconception_id_mismatch"
    elif rejection_counts["correction_missing"]:
        reason = "preference_misconception_correction_missing"
    elif rejection_counts["mechanism_missing"]:
        reason = "preference_misconception_mechanism_evidence_missing"
    elif rejection_counts["source_unbacked"]:
        reason = "preference_misconception_source_unbacked"
    else:
        reason = "preference_misconception_candidate_missing"
    telemetry = {
        "contract": "ed4all.micro-preference-eligibility.v1",
        "chunk_id": chunk_id,
        "eligible": bool(ordered),
        "reason_code": reason,
        "candidate_count": len(ordered),
        "candidate_ids": [item["id"] for item in ordered],
        "rejection_counts": rejection_counts,
        "evidence_window_sha256": _sha(_stable_json(blocks)),
    }
    return {
        "eligible": bool(ordered),
        "reason": reason,
        "candidates": ordered,
        "telemetry": telemetry,
    }


def unnecessary_generated_givens_error(
    learner_task: Any, givens: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    """Require scalar use while treating opaque given symbols as metadata."""
    task = _clean(learner_task)
    for index, item in enumerate(givens):
        symbol = _clean(item.get("symbol"))
        scalar = _clean(item.get("value"))
        symbol_used = bool(
            symbol and re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", task)
        )
        scalar_pattern = (
            rf"(?<![\d.]){re.escape(scalar)}(?!\d)(?!\.\d)"
        )
        scalar_used = bool(
            scalar and re.search(scalar_pattern, task)
        )
        if not scalar_used:
            return (
                f"generated_givens[{index}] is unnecessary: learner_task must "
                "use its scalar"
            )
        if symbol_used:
            binding = re.compile(
                rf"(?:"
                rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
                rf"\s*(?:=|:|\bis\b|\bof\b)\s*{scalar_pattern}"
                rf"|{scalar_pattern}\s*(?:[xX*×]\s*)?"
                rf"{re.escape(symbol)}(?![A-Za-z0-9_])"
                rf")",
                flags=re.IGNORECASE,
            )
            if not binding.search(task):
                return (
                    f"generated_givens[{index}] names symbol {symbol!r} but "
                    "learner_task does not bind it to the declared scalar"
                )
    return None


def assemble_claims(
    claims: Iterable[Mapping[str, Any]],
    *, objective_id: str,
    learner_task: str,
    generated_givens: Sequence[Mapping[str, Any]],
    requirement_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Deduplicate, stable-order, identify, and hash accepted atomic claims."""
    claims = list(claims)
    unique: Dict[str, Dict[str, Any]] = {}
    private_stage_calls = [
        dict(item["_stage_call"])
        for item in claims
        if isinstance(item, Mapping)
        and isinstance(item.get("_stage_call"), Mapping)
    ]
    for item in claims:
        canonical = {
            "claim": _clean(item.get("claim")),
            "evidence_quote": _clean(item.get("evidence_quote")),
            "source_block_id": _clean(item.get("source_block_id")),
            "source_role": _clean(item.get("source_role")) or "evidence",
            "source_polarity": (
                _clean(item.get("source_polarity")) or "factual"
            ),
        }
        if any(
            _equal_canonical_proposition(
                existing["claim"], canonical["claim"],
            )
            for existing in unique.values()
        ):
            continue
        identity = _sha(_stable_json(canonical))
        unique.setdefault(identity, {**canonical, "stable_id": identity[:16]})
    ordered = sorted(
        unique.values(),
        key=lambda row: (row["source_block_id"], row["stable_id"]),
    )
    if not 1 <= len(ordered) <= 3:
        raise ValueError("assembled claims must contain one to three unique claims")
    artifact = {
        "contract_version": MICRO_CONTRACT_VERSION,
        "objective_id": _clean(objective_id).lower(),
        "learner_task": _clean(learner_task),
        "generated_givens": [dict(item) for item in generated_givens],
        "claims": ordered,
        "private_stage_calls": private_stage_calls,
        "joined_claims": " ".join(item["claim"] for item in ordered),
        "_stage_c_deterministic": {
            "schema": "ed4all.micro-stage-c-deterministic.v1",
            "model_calls": 0,
            "decision_capture_events": 0,
            "telemetry": {
                "deterministic_events": 1,
                "model_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
    }
    if requirement_contract is not None:
        artifact["objective_requirement_contract"] = dict(
            requirement_contract
        )
    artifact["claims_sha256"] = _sha(_stable_json(ordered))
    artifact["artifact_sha256"] = _sha(_stable_json(artifact))
    return artifact


def immutable_artifact_error(
    artifact: Mapping[str, Any], expected_sha256: str,
) -> Optional[str]:
    candidate = dict(artifact)
    recorded = candidate.pop("artifact_sha256", None)
    actual = _sha(_stable_json(candidate))
    if recorded != expected_sha256 or actual != expected_sha256:
        return "immutable assembled artifact hash mismatch"
    return None


def numeric_closed_world_error(
    *, prompt: str, output: str, artifact: Mapping[str, Any],
) -> Optional[str]:
    _ledger, error = _deterministic_derivation_ledger(
        prompt=prompt,
        output=output,
        claims=[str(item["claim"]) for item in artifact["claims"]],
        generated_givens=list(artifact.get("generated_givens") or []),
        source_facts=[str(item["evidence_quote"]) for item in artifact["claims"]],
    )
    return error


def _prompt_numeric_error(
    prompt: str, artifact: Mapping[str, Any],
) -> Optional[str]:
    """Validate learner-visible numerics without using answer text as evidence."""
    _ledger, error = _deterministic_derivation_ledger(
        prompt="",
        output=prompt,
        claims=[str(item["claim"]) for item in artifact["claims"]],
        generated_givens=list(artifact.get("generated_givens") or []),
        source_facts=[str(item["evidence_quote"]) for item in artifact["claims"]],
    )
    return error


def _task_numeric_error(
    prompt: str, artifact: Mapping[str, Any],
) -> Optional[str]:
    """Return an actionable, jointly editable Stage-A numeric failure."""
    error = _prompt_numeric_error(prompt, artifact)
    if not error:
        return None
    literal_match = re.search(r"literal\(s\):\s*(\[[^\]]*\])", error)
    if not literal_match:
        return error
    illegal = re.findall(r"'([^']+)'", literal_match.group(1))
    authoritative = " ".join([
        *(
            _clean(item.get("claim")) + " " + _clean(item.get("evidence_quote"))
            for item in artifact.get("claims") or []
        ),
        *(
            _clean(item.get("value"))
            for item in artifact.get("generated_givens") or []
        ),
    ])
    allowed = sorted(
        set(_scenario_numeric_literals(authoritative)),
        key=lambda item: (Decimal(item), item),
    )
    return (
        f"illegal numeric literals={illegal}; affected JSON pointers="
        "['/learner_task', '/generated_givens']; allowed numeric "
        f"literals={allowed}"
    )


def _allowed_realization_numeric_inventory(
    artifact: Mapping[str, Any],
) -> list[str]:
    """Return the exact immutable numeric authority exposed to Stage D."""
    authoritative = " ".join([
        *(
            _clean(item.get("claim")) + " " + _clean(item.get("evidence_quote"))
            for item in artifact.get("claims") or []
        ),
        *(
            _clean(item.get("value"))
            for item in artifact.get("generated_givens") or []
        ),
    ])
    allowed = set(_scenario_numeric_literals(authoritative))
    task = _clean(artifact.get("learner_task"))
    allowed.update(
        _number_key(item)
        for item in re.findall(r"\^\s*(-?(?:\d+(?:\.\d+)?|\.\d+))", task)
    )
    allowed.update(
        _number_key(item.translate(_SUPERSCRIPT_TRANSLATION))
        for item in _SUPERSCRIPT_SEQUENCE_RE.findall(task)
    )
    return sorted(allowed, key=lambda item: (Decimal(item), item))


def _illegal_realization_numeric_literals(
    value: Any, *, allowed: Sequence[str],
) -> list[str]:
    allowed_set = set(allowed)
    illegal: list[str] = []
    for literal in _NUMBER_RE.findall(_clean(value)):
        key = _number_key(literal)
        if key not in allowed_set and key not in illegal:
            illegal.append(key)
    return illegal


def _realization_numeric_error(
    value: Mapping[str, Any], *, artifact: Mapping[str, Any],
    answer_field: str,
) -> Optional[str]:
    """Return pointer-scoped illegal literals for one Stage-D realization."""
    allowed = _allowed_realization_numeric_inventory(artifact)
    violations = {
        pointer: illegal
        for pointer, illegal in (
            (
                "/prompt",
                _illegal_realization_numeric_literals(
                    value.get("prompt"), allowed=allowed,
                ),
            ),
            (
                f"/{answer_field}",
                _illegal_realization_numeric_literals(
                    value.get(answer_field), allowed=allowed,
                ),
            ),
        )
        if illegal
    }
    if "/prompt" in violations:
        violations.setdefault(f"/{answer_field}", [])
    if not violations:
        return None
    pointers = list(violations)
    illegal = list(dict.fromkeys(
        item for pointer in pointers for item in violations[pointer]
    ))
    return (
        f"illegal numeric literals={illegal}; affected JSON pointers="
        f"{pointers}; allowed numeric literals={allowed}"
    )


def _claim_checklist(artifact: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Expose every immutable Stage-C claim as a Stage-D dependency."""
    return [
        {
            "index": index,
            "claim_id": _clean(item.get("stable_id")),
            "claim": _clean(item.get("claim")),
        }
        for index, item in enumerate(artifact.get("claims") or [])
    ]


def stage_d_realization_view(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """Project only validated columns required by the Stage-D generator."""
    requirement_contract = artifact.get("objective_requirement_contract")
    claims = [
        {
            "index": index,
            "claim_id": _clean(item.get("stable_id")),
            "claim": _clean(item.get("claim")),
            "claim_sha256": _sha(_clean(item.get("claim"))),
            "source_block_id": _clean(item.get("source_block_id")),
            "source_role": _clean(item.get("source_role")) or "evidence",
            "source_polarity": (
                _clean(item.get("source_polarity")) or "factual"
            ),
            "private_evidence_quote_sha256": _sha(
                _clean(item.get("evidence_quote"))
            ),
        }
        for index, item in enumerate(artifact.get("claims") or [])
    ]
    if not claims or any(
        not row["claim_id"] or not row["claim"] or not row["source_block_id"]
        for row in claims
    ):
        raise SynthesisProviderError(
            "Stage-D realization view requires validated immutable claims",
            code="staged_micro_D_realization_view_invalid",
            details={
                "terminal_content_rejection": True,
                "stage": "micro_D",
                "model_calls": 0,
            },
        )
    view = {
        "contract_version": MICRO_STAGE_D_REALIZATION_VIEW_VERSION,
        "private_artifact_sha256": _clean(artifact.get("artifact_sha256")),
        "objective_id": _clean(artifact.get("objective_id")),
        "learner_task": _clean(artifact.get("learner_task")),
        "generated_givens": [
            dict(item) for item in artifact.get("generated_givens") or []
        ],
        "claims": claims,
        **({"objective_requirement_contract": {
            "contract_version": requirement_contract.get(
                "contract_version"
            ),
            "objective_contract_sha256": requirement_contract.get(
                "objective_contract_sha256"
            ),
            "result_required": bool(
                requirement_contract.get("result_required")
            ),
            "requirements": [
                {
                    key: item.get(key)
                    for key in (
                        "requirement_id", "kind", "ordinal", "value",
                        "requirement_sha256",
                    )
                }
                for item in requirement_contract.get("requirements") or []
                if isinstance(item, Mapping)
            ],
        }} if isinstance(requirement_contract, Mapping) else {}),
        "coverage_checklist": _claim_checklist(artifact),
        "claims_sha256": _clean(artifact.get("claims_sha256")),
    }
    view["view_sha256"] = _sha(_stable_json(view))
    return view


def _obvious_unrelated_assertion(
    realization: str, *, claim: str, evidence_quote: str,
) -> Optional[str]:
    """Reject only a sentence with no lexical connection to its bound source."""
    authoritative = set(re.findall(
        r"[^\W\d_]{4,}", f"{claim} {evidence_quote}".casefold(),
        flags=re.UNICODE,
    ))
    stop = {
        "also", "because", "from", "have", "into", "that", "their", "there",
        "these", "this", "those", "when", "where", "which", "with",
    }
    authoritative -= stop
    for sentence in _sentences(realization):
        content = set(re.findall(
            r"[^\W\d_]{4,}", sentence.casefold(), flags=re.UNICODE,
        )) - stop
        if len(content) >= 5 and authoritative and not content & authoritative:
            return sentence
    return None


def assemble_claim_realizations(
    rows: Sequence[Mapping[str, Any]], *, artifact: Mapping[str, Any],
    synthesis_seed: int,
) -> Dict[str, Any]:
    """Deterministically project ordered validated claim rows into prose."""
    expected = [
        _clean(item.get("stable_id")) for item in artifact.get("claims") or []
    ]
    actual = [_clean(item.get("claim_id")) for item in rows]
    if actual != expected:
        raise SynthesisProviderError(
            "claim realizations differ from immutable Stage-C order",
            code="staged_micro_D_realization_order_invalid",
        )
    connectors = ("\n\n", " Additionally, ", " In addition, ")
    seed_material = _stable_json({
        "contract": MICRO_STAGE_D_REALIZATION_CONTRACT_VERSION,
        "artifact_sha256": artifact.get("artifact_sha256"),
        "synthesis_seed": synthesis_seed,
    })
    connector_index = int(_sha(seed_material)[:16], 16) % len(connectors)
    connector = connectors[connector_index]
    texts = [_clean(item.get("realization")) for item in rows]
    assembled = connector.join(texts)
    provenance = {
        "contract_version": MICRO_STAGE_D_REALIZATION_CONTRACT_VERSION,
        "connector_index": connector_index,
        "ordered_claim_ids": actual,
        "item_sha256": [_sha(text) for text in texts],
        "assembled_sha256": _sha(assembled),
        "private_artifact_sha256": _clean(artifact.get("artifact_sha256")),
    }
    provenance["provenance_sha256"] = _sha(_stable_json(provenance))
    return {"text": assembled, "provenance": provenance}


def assemble_objective_execution(
    value: Mapping[str, Any], *, artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble requirement-bound worked steps and exact completion spans."""
    contract = artifact.get("objective_requirement_contract")
    if not isinstance(contract, Mapping):
        raise SynthesisProviderError(
            "objective execution assembly requires its contract",
            code="staged_micro_D_objective_requirements_missing",
        )
    rows = [dict(item) for item in value.get("worked_steps") or []]
    result = value.get("result")
    if isinstance(result, Mapping):
        rows.append(dict(result))
    text = ""
    projected = []
    for index, row in enumerate(rows):
        realization = _clean(row.get("realization"))
        if index:
            text += "\n\n"
        start = len(text)
        text += realization
        projected.append({
            "step_id": f"step-{index + 1:02d}",
            "requirement_id": _clean(row.get("requirement_id")),
            "claim_ids": list(row.get("claim_ids") or []),
            "realization": realization,
            "completion_span": [start, len(text)],
        })
    return {
        "text": text,
        "steps": projected,
        "result": projected[-1] if isinstance(result, Mapping) else None,
        "assembled_sha256": _sha(text),
        "final_assembly": {
            "contract_version": (
                "ed4all.objective-execution-final-assembly.v1"
            ),
            "ordered_step_ids": [
                item["step_id"] for item in projected
            ],
            "separator": "\n\n",
            "item_sha256": [
                _sha(item["realization"]) for item in projected
            ],
            "completion_spans": [
                item["completion_span"] for item in projected
            ],
            "assembled_sha256": _sha(text),
        },
    }


def _scoped_coverage_error(
    error: Optional[str], *, artifact: Mapping[str, Any], answer_field: str,
) -> Optional[str]:
    """Attach immutable claim identity and the minimum Stage-D edit scope."""
    if not error:
        return error
    if "substantive output unit " in error:
        claim_ids = [
            _clean(item.get("stable_id"))
            for item in artifact.get("claims") or []
            if _clean(item.get("stable_id"))
        ]
        return (
            f"{error}; preserve_immutable_claim_ids={claim_ids}; "
            f"affected JSON pointers=['/{answer_field}']; "
            "repair_scope=edit only the named genuinely unsupported clause; "
            "do not delete any deterministically proven immutable claim"
        )
    if "missing atomic claim " not in error:
        return error
    match = re.search(r"missing atomic claim (\d+)", error)
    if not match:
        return error
    index = int(match.group(1))
    claims = list(artifact.get("claims") or [])
    if not 0 <= index < len(claims):
        return error
    claim_id = _clean(claims[index].get("stable_id"))
    return (
        f"{error}; missing_claim_id={claim_id!r}; required_entailment="
        f"explicitly realize immutable claim {index} in /{answer_field}; "
        f"affected JSON pointers=['/{answer_field}']"
    )


def _leakage_error(
    value: str, *, source: str, artifact: Mapping[str, Any],
    pointer: str = "/output",
) -> Optional[str]:
    """Apply the staged-V4 distinctive-span policy to micro realizations."""
    span = _first_distinctive_verbatim_span(
        value,
        source,
        canonical_contract_text=(
            str(artifact.get("learner_task") or ""),
        ),
    )
    if span is None:
        return None
    return (
        f"provider output contains distinctive source span sha256={_sha(span)}; "
        "category=private_source_overlap; "
        f"affected JSON pointers=['{pointer}']"
    )


class MicroResumeStore:
    """Fsynced per-unit hash-chain with strict terminal replay semantics."""

    def __init__(
        self, path: Path, *, fingerprint: str,
        store_identity: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = Path(path)
        self.fingerprint = str(fingerprint)
        self.store_identity = (
            dict(store_identity) if store_identity is not None else None
        )

    @staticmethod
    def _row_hash(row: Mapping[str, Any]) -> str:
        unsigned = dict(row)
        unsigned.pop("row_sha256", None)
        return _sha(_stable_json(unsigned))

    def _rows(self) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[Dict[str, Any]] = []
        previous = "0" * 64
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SynthesisProviderError(
                    f"microstate journal has torn row {line_number}",
                    code="staged_micro_resume_corrupt",
                ) from exc
            if (
                row.get("contract_fingerprint") != self.fingerprint
                or (
                    self.store_identity is not None
                    and row.get("store_identity") != self.store_identity
                )
                or row.get("previous_sha256") != previous
                or row.get("row_sha256") != self._row_hash(row)
            ):
                raise SynthesisProviderError(
                    f"microstate journal identity drift at row {line_number}",
                    code="staged_micro_resume_drift",
                )
            previous = str(row["row_sha256"])
            rows.append(dict(row))
        return rows

    def load(
        self, *, allow_failure_outcomes: bool = False,
    ) -> Dict[str, Any]:
        """Return terminal artifacts; every logical unit is unique."""
        started: Dict[str, tuple[int, int]] = {}
        open_units: set[str] = set()
        terminal: Dict[str, Any] = {}
        terminal_hashes: Dict[str, str] = {}
        failure_outcomes: list[Mapping[str, Any]] = []
        for row in self._rows():
            unit = str(row["unit"])
            if row["state"] == "started":
                if unit in open_units or unit in terminal:
                    raise SynthesisProviderError(
                        f"duplicate unresolved microstate start for {unit}",
                        code="staged_micro_resume_ambiguous",
                    )
                started[unit] = (
                    int(row["sequence"]), int(row.get("attempt", -1)),
                )
                open_units.add(unit)
                continue
            if row["state"] != "terminal" or (
                unit not in open_units and unit not in terminal
            ):
                raise SynthesisProviderError(
                    f"microstate terminal ordering invalid for {unit}",
                    code="staged_micro_resume_corrupt",
                )
            digest = str(row.get("artifact_sha256") or "")
            if unit in terminal_hashes:
                raise SynthesisProviderError(
                    f"duplicate microstate terminal identity for {unit}",
                    code="staged_micro_duplicate_identity",
                )
            if int(row.get("attempt", -1)) != started[unit][1]:
                raise SynthesisProviderError(
                    f"microstate attempt identity differs for {unit}",
                    code="staged_micro_resume_corrupt",
                )
            terminal_hashes[unit] = digest
            outcome = str(row.get("outcome") or "success")
            if outcome not in {"success", "content_rejected"}:
                raise SynthesisProviderError(
                    f"microstate terminal outcome invalid for {unit}",
                    code="staged_micro_resume_corrupt",
                )
            if outcome == "success":
                terminal[unit] = row.get("artifact")
            else:
                evidence = row.get("terminal_evidence")
                if (
                    not isinstance(evidence, Mapping)
                    or evidence.get("outcome") != outcome
                    or evidence.get("unit") != unit
                    or evidence.get("stage") != row.get("stage")
                    or evidence.get("attempt") != row.get("attempt")
                    or evidence.get("evidence_sha256")
                    != _sha(_stable_json({
                        key: value for key, value in evidence.items()
                        if key != "evidence_sha256"
                    }))
                ):
                    raise SynthesisProviderError(
                        f"microstate terminal evidence invalid for {unit}",
                        code="staged_micro_resume_corrupt",
                    )
                failure_outcomes.append(evidence)
            open_units.discard(unit)
        unresolved = sorted(open_units)
        if unresolved:
            raise SynthesisProviderError(
                "microstate has started-without-terminal intents: "
                + ",".join(unresolved),
                code="staged_micro_resume_ambiguous",
            )
        if failure_outcomes and not allow_failure_outcomes:
            first = failure_outcomes[0]
            raise SynthesisProviderError(
                "microstate contains cached terminal "
                f"{first['outcome']}: {first['reason_code']}",
                code=str(first["reason_code"]),
                details={
                    "terminal_content_rejection": (
                        first["outcome"] == "content_rejected"
                    ),
                    "cached_terminal_outcome": True,
                    "terminal_evidence_sha256": first["evidence_sha256"],
                },
            )
        return terminal

    def append(
        self, *, unit: str, stage: str, slot: Optional[int],
        attempt: int, state: str, artifact: Any = None,
        outcome: Optional[str] = None,
        terminal_evidence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if stage not in _STAGES or state not in {"started", "terminal"}:
            raise ValueError("invalid microstate stage/state")
        rows = self._rows()
        sequence = len(rows) + 1
        artifact_sha = (
            _sha(_stable_json(artifact)) if state == "terminal" else None
        )
        if state == "started":
            if outcome is not None or terminal_evidence is not None:
                raise ValueError("started microstate cannot carry an outcome")
        else:
            outcome = outcome or "success"
            if outcome not in {"success", "content_rejected"}:
                raise ValueError("invalid microstate terminal outcome")
            if outcome == "success" and terminal_evidence is not None:
                raise ValueError("successful microstate cannot carry failure evidence")
            if outcome == "content_rejected":
                if not isinstance(terminal_evidence, Mapping):
                    raise ValueError(
                        "non-accepted microstate requires terminal evidence"
                    )
                terminal_evidence = dict(terminal_evidence)
                terminal_evidence["evidence_sha256"] = _sha(_stable_json({
                    key: value for key, value in terminal_evidence.items()
                    if key != "evidence_sha256"
                }))
        row = {
            "contract_version": MICRO_CONTRACT_VERSION,
            "contract_fingerprint": self.fingerprint,
            "store_identity": self.store_identity,
            "sequence": sequence,
            "previous_sha256": (
                str(rows[-1]["row_sha256"]) if rows else "0" * 64
            ),
            "unit": str(unit),
            "stage": stage,
            "slot": slot,
            "attempt": int(attempt),
            "state": state,
            "outcome": outcome if state == "terminal" else None,
            "artifact_sha256": artifact_sha,
            "artifact": artifact if state == "terminal" else None,
            "terminal_evidence": (
                terminal_evidence if state == "terminal" else None
            ),
        }
        row["row_sha256"] = self._row_hash(row)
        parent_created = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_created = not self.path.exists()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_stable_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if parent_created or file_created:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return row


class MicroStagedSynthesisProvider(StagedSynthesisProvider):
    """Expose legacy paraphrase methods through the A–F micro contract."""

    contract_version = MICRO_CONTRACT_VERSION

    def __init__(
        self, base_provider: Any, *, synthesis_seed: int,
        completion_cap: int = MICRO_DEFAULT_COMPLETION_CAP,
    ) -> None:
        super().__init__(base_provider)
        if (
            isinstance(synthesis_seed, bool)
            or not isinstance(synthesis_seed, int)
            or not 0 <= synthesis_seed < 2**64
        ):
            raise SynthesisProviderError(
                "micro synthesis requires an explicit unsigned 64-bit seed",
                code="staged_micro_synthesis_seed_invalid",
            )
        self._synthesis_seed = synthesis_seed
        if (
            isinstance(completion_cap, bool)
            or completion_cap not in MICRO_COMPLETION_CAP_CANDIDATES
        ):
            raise SynthesisProviderError(
                "micro completion cap must be one of the bounded "
                f"qualification candidates {MICRO_COMPLETION_CAP_CANDIDATES}",
                code="staged_micro_completion_cap_invalid",
            )
        self._completion_cap = int(completion_cap)

    def _release_identity(self) -> Dict[str, Any]:
        """Return the core-owned, immutable portion of the release tuple."""
        identity = {
            "release_contract_version": MICRO_RELEASE_CONTRACT_VERSION,
            "synthesis_contract_version": MICRO_CONTRACT_VERSION,
            "synthesis_contract_sha256": micro_contract_fingerprint(),
            "synthesis_seed": self._synthesis_seed,
            "projections": {
                "instruction": MICRO_SFT_PROJECTION,
                "preference": MICRO_DPO_PROJECTION,
            },
            "provider": self._provider_name,
            "model": self._model,
            "completion_cap": self._completion_cap,
        }
        return identity

    def _decision_identity_context(self) -> Dict[str, Any]:
        return self._release_identity()

    @staticmethod
    def _micro_stage_family(stage: str) -> str:
        matches = [
            family
            for family in MICRO_STAGE_MAX_TOKENS
            if str(stage).startswith(f"micro_{family}_")
        ]
        if len(matches) != 1:
            raise SynthesisProviderError(
                f"micro synthesis stage has no token budget: {stage!r}",
                code="staged_micro_token_budget_unknown",
            )
        family = matches[0]
        budget = MICRO_STAGE_MAX_TOKENS[family]
        if not isinstance(budget, int) or not 1 <= budget <= 4096:
            raise SynthesisProviderError(
                f"micro synthesis stage token budget is invalid: {family}",
                code="staged_micro_token_budget_invalid",
            )
        return family

    def _call_stage(self, *, stage: str, **kwargs: Any) -> Any:
        """Bind every model-backed micro call to its fingerprinted stage cap."""
        family = self._micro_stage_family(stage)
        if "max_output_tokens" in kwargs:
            raise SynthesisProviderError(
                "micro stage token budget cannot be overridden per call",
                code="staged_micro_token_budget_override",
            )
        self._validation_audit.decision_capture_event_id = ""
        result = super()._call_stage(
            stage=stage,
            max_output_tokens=MICRO_STAGE_MAX_TOKENS[family],
            **kwargs,
        )
        event_id = str(
            getattr(self._validation_audit, "decision_capture_event_id", "")
            or ""
        )
        if event_id and isinstance(result.value, dict):
            result.value["_decision_capture_id"] = event_id
            result.value["_stage_call"] = {
                "stage": stage,
                "decision_capture_id": event_id,
                "finish_reason": "stop",
                "prompt_tokens": int(result.usage.get("prompt_tokens", 0)),
                "completion_tokens": int(
                    result.usage.get("completion_tokens", 0)
                ),
                "requests": int(result.requests),
                "truncated": False,
                "max_output_tokens": MICRO_STAGE_MAX_TOKENS[family],
            }
        return result

    @staticmethod
    def _projection_decision_capture_id(
        artifact: Mapping[str, Any], *, stage: str,
    ) -> str:
        event_id = _clean(artifact.get("_decision_capture_id"))
        if not event_id:
            raise SynthesisProviderError(
                f"{stage} projection lacks its actual provider capture event",
                code="staged_micro_projection_capture_missing",
                details={
                    "terminal_content_rejection": True,
                    "stage": stage,
                },
            )
        return event_id

    @staticmethod
    def _routed_claim_blocks(
        chunk: Mapping[str, Any], *, focus: Mapping[str, Any],
        learner_task: str = "",
    ) -> list[Dict[str, Any]]:
        """Return the one canonical objective-routed scalar-claim block order."""
        window = build_evidence_window(
            chunk, focus, target_rung=focus.get("bloom_level"),
        )
        eligible = [
            dict(block) for block in window["blocks"]
            if block["polarity"] in {"factual", "corrective"}
        ]
        card = objective_card(focus)
        routing_signals = card.get("content_obligations") or [
            card.get("statement"),
            card.get("bloom_verb"),
            card.get("action_object"),
            card.get("bloom_level"),
            learner_task,
        ]
        objective_terms = set(re.findall(
            r"[a-z0-9]+",
            " ".join(str(item) for item in routing_signals if item).casefold(),
        ))
        eligible.sort(key=lambda block: (
            -len(objective_terms & set(re.findall(
                r"[a-z0-9]+", _clean(block["text"]).casefold(),
            ))),
            _clean(block["block_id"]),
            _sha(_clean(block["text"])),
        ))
        return eligible

    def _resume_store(
        self, *, chunk: Mapping[str, Any], draft: Mapping[str, Any], kind: str,
    ) -> Optional[MicroResumeStore]:
        root = getattr(self._capture, "output_dir", None)
        if root is None:
            return None
        draft_identity = draft.get("_micro_manifest_identity")
        if not isinstance(draft_identity, Mapping):
            # No manifest-stamped identity: fall back to the explicitly bound
            # production generation unit.  Unbound stays a LOUD failure — a
            # defaulted identity would silently share one resume-store file
            # across units.
            bound = current_micro_generation_unit()
            if bound is None:
                raise SynthesisProviderError(
                    "micro synthesis requires manifest-bound draft identity: "
                    "the draft carried no '_micro_manifest_identity' and the "
                    "call was not wrapped in bind_micro_generation_unit(...)",
                    code="staged_micro_draft_identity_missing",
                )
            if bound.get("kind") != kind:
                raise SynthesisProviderError(
                    "micro synthesis generation-unit binding names kind "
                    f"{bound.get('kind')!r} but the call is {kind!r}",
                    code="staged_micro_draft_identity_invalid",
                )
            draft_identity = build_micro_manifest_identity(
                chunk, kind=kind, variant=bound.get("variant"),
                repetition=bound.get("repetition", MICRO_PRODUCTION_REPETITION),
                draft=draft,
            )
        expected_draft_identity = build_micro_manifest_identity(
            chunk, kind=kind, variant=draft_identity.get("variant"),
            repetition=draft_identity.get(
                "repetition", MICRO_PRODUCTION_REPETITION,
            ),
            draft=draft,
        )
        if dict(draft_identity) != expected_draft_identity:
            raise SynthesisProviderError(
                "micro synthesis draft identity differs from runtime inputs",
                code="staged_micro_draft_identity_invalid",
            )
        identity = {
            "contract": micro_contract_fingerprint(),
            "release_identity": self._release_identity(),
            "kind": kind,
            "chunk_id": _clean(chunk.get("id") or chunk.get("chunk_id")),
            "chunk_sha256": _sha(_stable_json(chunk)),
            "draft_identity": expected_draft_identity,
            "draft_sha256": _sha(_stable_json(expected_draft_identity)),
            "model": self._model,
            "pilot_run_id": getattr(self, "_pilot_run_id", None),
            "cell_id": getattr(self, "_pilot_cell_id", None),
            "manifest_sha256": getattr(
                self, "_pilot_manifest_sha256", None,
            ),
            "eligibility_sha256": getattr(
                self, "_pilot_eligibility_sha256", None,
            ),
            "execution_fingerprint": getattr(
                self, "_pilot_execution_fingerprint", None,
            ),
        }
        identity["input_fingerprint"] = _sha(_stable_json({
            key: identity[key] for key in (
                "manifest_sha256", "eligibility_sha256", "chunk_sha256",
                "draft_sha256",
            )
        }))
        fingerprint = _sha(_stable_json(identity))
        return MicroResumeStore(
            Path(root) / "micro_synthesis_state" / f"{fingerprint}.jsonl",
            fingerprint=fingerprint, store_identity=identity,
        )

    @staticmethod
    def _journal_unit(kind: str, stage: str, slot: Optional[int] = None) -> str:
        return f"{kind}:{stage}" + (f":slot-{slot}" if slot is not None else "")

    def _journaled(
        self, store: Optional[MicroResumeStore], cache: Mapping[str, Any],
        *, kind: str, stage: str, slot: Optional[int], attempt: int,
        call: Any,
    ) -> Any:
        self._check_stop(0)
        unit = self._journal_unit(kind, stage, slot)
        if unit in cache:
            return cache[unit]
        if store is not None:
            store.append(
                unit=unit, stage=stage, slot=slot, attempt=attempt, state="started",
            )
        try:
            artifact = call()
        except SynthesisProviderError as exc:
            details = getattr(exc, "details", None)
            safe_details = (
                dict(details) if isinstance(details, Mapping) else {}
            )
            if store is not None and safe_details.get(
                "terminal_content_rejection"
            ) is True:
                reason_code = str(exc.code or type(exc).__name__)
                evidence_core = {
                    "contract": "ed4all.micro-terminal-outcome.v1",
                    "outcome": "content_rejected",
                    "reason_code": reason_code,
                    "exception_class": type(exc).__name__,
                    "message_sha256": _sha(_clean(exc)),
                    "details_sha256": _sha(_stable_json(safe_details)),
                    "unit": unit,
                    "stage": stage,
                    "slot": slot,
                    "attempt": int(attempt),
                    "logical_intent": {
                        "kind": kind,
                        "unit": unit,
                        "stage": stage,
                        "slot": slot,
                        "attempt": int(attempt),
                    },
                    "external_linkage": {
                        "decision_capture_id": _clean(
                            safe_details.get("decision_capture_id")
                        ),
                        "http_attempt_ledger": "cell-bound-authority",
                        "call_intent_ledger": "cell-bound-authority",
                    },
                }
                store.append(
                    unit=unit, stage=stage, slot=slot, attempt=attempt,
                    state="terminal", outcome="content_rejected",
                    terminal_evidence=evidence_core,
                )
            raise
        # A returned call is a completed success. Seal it before observing a
        # later stop request so resume never repeats completed provider work.
        if store is not None:
            store.append(
                unit=unit, stage=stage, slot=slot, attempt=attempt,
                state="terminal", artifact=artifact, outcome="success",
            )
        self._check_stop(0)
        return artifact

    def _check_stop(self, completed: int) -> None:
        if stop_control.stop_requested():
            raise stop_control.GracefulStopRequested(
                "training_synthesis.micro", completed
            )

    def _task_design(self, chunk: Mapping[str, Any]) -> Dict[str, Any]:
        focus = self._focus(chunk)
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        window = build_evidence_window(
            chunk, focus, target_rung=focus.get("bloom_level"),
        )
        allowed_numeric_text = " ".join([
            _stable_json(objective_card(focus)),
            *(_clean(block.get("text")) for block in window["blocks"]),
        ])
        value = deterministic_stage_a_task(
            focus, synthesis_seed=self._synthesis_seed,
        )
        objective_error = self._objective_contract_error(
            value["learner_task"], focus, answer_bearing=False,
        )
        if objective_error:
            raise SynthesisProviderError(
                f"deterministic Stage A assembly violated its objective: "
                f"{objective_error}",
                code="staged_micro_A_normalization_invalid",
                chunk_id=chunk_id,
                details={
                    "terminal_content_rejection": True,
                    "stage": "micro_A",
                    "model_calls": 0,
                },
            )
        allowed = sorted(
            set(_scenario_numeric_literals(allowed_numeric_text)),
            key=lambda item: (Decimal(item), item),
        )
        actual = set(_scenario_numeric_literals(value["learner_task"]))
        if not actual <= set(allowed):
            raise SynthesisProviderError(
                "deterministic Stage A introduced unauthorized numeric material",
                code="staged_micro_A_normalization_invalid",
                chunk_id=chunk_id,
                details={
                    "terminal_content_rejection": True,
                    "stage": "micro_A",
                    "model_calls": 0,
                    "illegal_numeric_literals": sorted(actual - set(allowed)),
                },
            )
        value["_normalization_evidence"]["allowed_numeric_literals"] = allowed
        return value

    def _validate_claim(
        self, value: Dict[str, Any], *, block: Mapping[str, Any],
        private_guard: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        claim = _clean(value.get("claim"))
        quote = _clean(value.get("evidence_quote"))
        source_block_id = _clean(value.get("source_block_id"))
        if source_block_id != _clean(block.get("block_id")):
            return "source_block_id differs from assigned block"
        located = canonical_quote(
            quote,
            {"blocks": [block]},
        )
        if located is None:
            return "evidence_quote is not an exact assigned-block span"
        exact, _ = located
        value["evidence_quote"] = exact
        if private_guard is not None:
            if private_guard and (
                source_block_id != private_guard["source_block_id"]
                or exact != private_guard["evidence_quote"]
            ):
                return (
                    "claim repair mutated frozen private provenance fields; "
                    "affected JSON pointers=['/evidence_quote', "
                    "'/source_block_id']; editable JSON pointers=['/claim']"
                )
            private_guard.setdefault("source_block_id", source_block_id)
            private_guard.setdefault("evidence_quote", exact)
        private_source = " ".join(filter(None, (
            exact,
            _clean(block.get("text")),
        )))
        allowed_numeric = sorted(
            set(_scenario_numeric_literals(private_source)),
            key=lambda item: (Decimal(item), item),
        )
        illegal_numeric = _illegal_realization_numeric_literals(
            claim, allowed=allowed_numeric,
        )
        if illegal_numeric:
            return (
                f"public claim contains illegal numeric literals="
                f"{illegal_numeric}; allowed numeric literals="
                f"{allowed_numeric}; affected JSON pointers=['/claim']; "
                "preserve frozen /evidence_quote and /source_block_id"
            )
        operator_error = _relation_operator_mutation(exact, claim)
        if operator_error is not None:
            return operator_error
        if _equal_canonical_proposition(exact, claim):
            semantic_error = None
        elif _relation_preserving_support(exact, claim):
            semantic_error = None
        elif _symbolic_math_supports(exact, claim):
            semantic_error = None
        elif (formula_proof := formula_relation_proof(exact, claim)) is not None:
            derivations = list(
                getattr(self._validation_audit, "derivations", []) or []
            )
            derivations.append({
                **formula_proof,
                "premise_sha256": _sha(exact),
                "hypothesis_sha256": _sha(claim),
            })
            self._validation_audit.derivations = derivations
            semantic_error = None
        elif (
            affine_proof := affine_two_line_relation_proof(
                _clean(block.get("text")), claim,
            )
        ) is not None:
            derivations = list(
                getattr(self._validation_audit, "derivations", []) or []
            )
            derivations.append({
                **affine_proof,
                "premise_sha256": _sha(_clean(block.get("text"))),
                "hypothesis_sha256": _sha(claim),
            })
            self._validation_audit.derivations = derivations
            semantic_error = None
        else:
            score = self._score_pair(
                decision_type="micro_claim_quote",
                premise=exact,
                hypothesis=claim,
            )
            if (
                float(score.contradiction) >= _PLAN_CONTRADICTION_CEILING
                and _explicit_relation_conflict(exact, claim)
            ):
                semantic_error = "claim contradicts its exact quote"
            elif float(score.entailment) < _PLAN_ENTAILMENT_FLOOR:
                semantic_error = "claim is not entailed by its exact quote"
            else:
                semantic_error = None
        if semantic_error is not None:
            return semantic_error
        leaked = _first_distinctive_verbatim_span(claim, private_source)
        if leaked is not None:
            return (
                "public claim contains distinctive private-source span "
                f"sha256={_sha(leaked)}; category=private_source_overlap; "
                "affected JSON pointers=['/claim']; repair only /claim and "
                "preserve frozen /evidence_quote and /source_block_id"
            )
        return None

    def _claim_slot(
        self, *, chunk_id: str, slot: int, block: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run one scalar claim call; at most two attempts and never list repair."""
        user = (
            f"CONTRACT={MICRO_CONTRACT_VERSION}\nSLOT={slot}\n"
            f"ASSIGNED_BLOCK={_stable_json(dict(block))}\n"
            "Return one leakage-safe PUBLIC claim, one exact PRIVATE "
            "evidence_quote, and source_block_id. The PUBLIC claim must remain "
            "entailed by the PRIVATE quote while avoiding any distinctive "
            "50-character span from the quote or assigned block."
        )
        private_guard: Dict[str, str] = {}
        seen_claims: set[str] = set()
        last_error = "unknown"

        def validate_claim(value: Dict[str, Any]) -> Optional[str]:
            canonical = _sha(_clean(value.get("claim")).casefold())
            if canonical in seen_claims:
                return (
                    "scalar claim repair made no semantic progress; "
                    "identical canonical /claim was already rejected"
                )
            seen_claims.add(canonical)
            return self._validate_claim(
                value, block=block, private_guard=private_guard,
            )

        for attempt in (1, 2):
            self._check_stop(slot)
            try:
                result = self._call_stage(
                    stage=f"micro_B_claim_{slot}_attempt_{attempt}",
                    chunk_id=chunk_id,
                    system=_CLAIM_SYSTEM,
                    user=user,
                    required_keys=("claim", "evidence_quote", "source_block_id"),
                    validator=validate_claim,
                    response_schema=_CLAIM_SCHEMA,
                    max_stage_repairs=0,
                    max_leakage_repairs=0,
                ).value
                return result
            except SynthesisProviderError as exc:
                last_error = str(exc)
                if attempt == 2:
                    break
                user += (
                    "\nThe previous scalar claim failed validation. Repair only "
                    "the PUBLIC /claim. Preserve these frozen PRIVATE fields "
                    f"exactly: {_stable_json(private_guard)}. Return the complete "
                    "one-claim object; do not return a list."
                )
        # A slot may abstain; assembly requires at least one successful slot.
        if slot == 0:
            raise SynthesisProviderError(
                f"micro claim slot exhausted: {last_error}",
                code="staged_micro_claim_unavailable",
                chunk_id=chunk_id,
                details={"terminal_content_rejection": True, "stage": "micro_B"},
            )
        return None

    def _stage_e_faulty_step_error(
        self, candidate: Mapping[str, Any], statement: Any,
    ) -> Optional[str]:
        """Validate one selected fault against only its canonical authority."""
        candidate_evidence = " ".join(filter(None, (
            _clean(candidate.get("misconception")),
            _clean(candidate.get("correction")),
            _clean(candidate.get("mechanism_evidence")),
        )))
        faulty_step = _clean(statement)
        if (
            _equal_canonical_proposition(candidate_evidence, faulty_step)
            or _relation_preserving_support(candidate_evidence, faulty_step)
            or _symbolic_math_supports(candidate_evidence, faulty_step)
        ):
            return None
        score = self._score_pair(
            decision_type="micro_misconception_faulty_step",
            premise=candidate_evidence,
            hypothesis=faulty_step,
        )
        if (
            float(score.contradiction) >= _PLAN_CONTRADICTION_CEILING
            and _explicit_relation_conflict(candidate_evidence, faulty_step)
        ):
            return "faulty_step contradicts indexed misconception evidence"
        if float(score.entailment) < _PLAN_ENTAILMENT_FLOOR:
            return "faulty_step is not supported by indexed misconception evidence"
        return None

    def _assemble(
        self, chunk: Mapping[str, Any], *, kind: str,
        store: Optional[MicroResumeStore], cache: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        task = self._journaled(
            store, cache, kind=kind, stage="A", slot=None, attempt=1,
            call=lambda: self._task_design(chunk),
        )
        focus = task["_objective_card"]
        eligible = self._routed_claim_blocks(
            chunk, focus=focus, learner_task=_clean(task.get("learner_task")),
        )
        claims = []
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        for slot in range(min(3, len(eligible))):
            claim = self._journaled(
                store, cache, kind=kind, stage="B", slot=slot, attempt=1,
                call=lambda slot=slot: self._claim_slot(
                    chunk_id=chunk_id, slot=slot, block=eligible[slot],
                ),
            )
            if claim is not None:
                claims.append({
                    **claim,
                    "source_role": (
                        _clean(eligible[slot].get("role")) or "evidence"
                    ),
                    "source_polarity": (
                        _clean(eligible[slot].get("polarity")) or "factual"
                    ),
                })
        artifact = self._journaled(
            store, cache, kind=kind, stage="C", slot=None, attempt=1,
            call=lambda: assemble_claims(
                claims,
                objective_id=task["objective_id"],
                learner_task=task["learner_task"],
                generated_givens=task["generated_givens"],
                requirement_contract=derive_objective_requirements(
                    task["_objective_card"],
                ),
            ),
        )
        return task, artifact

    def _realize(
        self, *, chunk_id: str, artifact: Mapping[str, Any], preference: bool,
    ) -> Dict[str, Any]:
        artifact_sha = str(artifact["artifact_sha256"])
        realization_view = stage_d_realization_view(artifact)
        prompt = deterministic_stage_d_prompt(artifact)
        fields = ("prompt", "chosen") if preference else ("prompt", "completion")
        schema = _stage_d_realization_schema(
            artifact, completion_cap=self._completion_cap,
        )
        requirement_contract = artifact.get("objective_requirement_contract")
        requirement_rows = (
            list(requirement_contract.get("requirements") or [])
            if isinstance(requirement_contract, Mapping) else []
        )
        requirement_by_id = {
            _clean(item.get("requirement_id")): item
            for item in requirement_rows if isinstance(item, Mapping)
        }
        worked_requirement_ids = [
            key for key, item in requirement_by_id.items()
            if item.get("kind") != "result"
        ]
        result_requirement_ids = [
            key for key, item in requirement_by_id.items()
            if item.get("kind") == "result"
        ]
        repair_guard: Optional[Dict[str, Any]] = None
        objective_proof_ledger: Dict[str, Dict[str, Any]] = {}

        def validate(value: Dict[str, Any]) -> Optional[str]:
            nonlocal repair_guard
            allowed_numeric = _allowed_realization_numeric_inventory(artifact)
            error = immutable_artifact_error(artifact, artifact_sha)
            if error:
                return error
            expected_fields = (
                {"claim_realizations", "worked_steps", "result"}
                if requirement_rows else {"claim_realizations"}
            )
            if set(value) != expected_fields:
                if not requirement_rows:
                    return (
                        "Stage-D model envelope must contain only "
                        "claim_realizations; affected JSON pointers=['/']"
                    )
                return (
                    "Stage-D model envelope fields differ from the bound "
                    f"contract: expected={sorted(expected_fields)}; "
                    "affected JSON pointers=['/']"
                )
            rows = value.get("claim_realizations")
            expected_ids = [
                _clean(item.get("stable_id")) for item in artifact["claims"]
            ]
            if not isinstance(rows, list):
                return (
                    "claim_realizations must be an array; affected JSON "
                    "pointers=['/claim_realizations']"
                )
            actual_ids = [
                _clean(item.get("claim_id")) if isinstance(item, Mapping) else ""
                for item in rows
            ]
            if (
                len(rows) != len(expected_ids)
                or len(set(actual_ids)) != len(actual_ids)
                or actual_ids != expected_ids
            ):
                return (
                    "claim_realizations must contain exactly one known row per "
                    "immutable claim in Stage-C order; expected_claim_ids="
                    f"{expected_ids}; affected JSON pointers="
                    "['/claim_realizations']"
                )
            guarded = {
                "prompt": prompt,
                "supported_claims": [dict(item) for item in rows],
            }
            if repair_guard is not None:
                diff = _claim_repair_diff(guarded, repair_guard)
                if diff is not None:
                    return (
                        "claim realization repair mutated or reordered frozen "
                        f"state: {diff['reason']}; affected JSON pointers="
                        "['/claim_realizations']"
                    )
            source = " ".join(
                _clean(item.get("evidence_quote")) for item in artifact["claims"]
            )
            for index, (row, bound) in enumerate(zip(rows, artifact["claims"])):
                realization = _clean(row.get("realization"))
                pointer = f"/claim_realizations/{index}/realization"
                unrelated_assertion = _obvious_unrelated_assertion(
                    realization,
                    claim=_clean(bound.get("claim")),
                    evidence_quote=_clean(bound.get("evidence_quote")),
                )
                validation_error = None
                if unrelated_assertion:
                    validation_error = (
                        "contains obvious unrelated assertion sha256="
                        f"{_sha(unrelated_assertion)}"
                    )
                illegal_numeric = _illegal_realization_numeric_literals(
                    realization, allowed=allowed_numeric,
                )
                if illegal_numeric:
                    validation_error = (
                        f"contains illegal numeric literals={illegal_numeric}; "
                        f"allowed numeric literals={allowed_numeric}"
                    )
                basis = (
                    _deterministic_entailment_basis(
                        bound.get("claim"), realization,
                    )
                    or _deterministic_entailment_basis(
                        bound.get("evidence_quote"), realization,
                    )
                )
                if validation_error is None and basis is None:
                    score = self._score_pair(
                        decision_type="micro_D_claim_realization",
                        premise=_clean(bound.get("claim")),
                        hypothesis=realization,
                    )
                    if (
                        float(score.contradiction)
                        >= _PLAN_CONTRADICTION_CEILING
                        and _explicit_relation_conflict(
                            bound.get("claim"), realization,
                        )
                    ):
                        validation_error = "contradicts its bound claim"
                    elif float(score.entailment) < _PLAN_ENTAILMENT_FLOOR:
                        validation_error = "does not entail its bound claim"
                leakage = _leakage_error(
                    realization, source=source, artifact=artifact,
                    pointer=pointer,
                )
                if validation_error is None and leakage:
                    validation_error = leakage
                if validation_error is not None:
                    if repair_guard is None:
                        repair_guard = _claim_repair_guard(
                            guarded, [index], editable_pointers=[pointer],
                        )
                    return (
                        f"claim_id={expected_ids[index]!r} {validation_error}; "
                        f"affected JSON pointers=['{pointer}']; repair only "
                        "this realization text and preserve prompt, IDs/order, "
                        "and every other realization"
                    )
            claim_assembly = assemble_claim_realizations(
                rows, artifact=artifact, synthesis_seed=self._synthesis_seed,
            )
            if requirement_rows:
                worked_steps = value.get("worked_steps")
                if not isinstance(worked_steps, list):
                    return "worked_steps must be an array"
                actual_requirement_ids = [
                    _clean(item.get("requirement_id"))
                    if isinstance(item, Mapping) else ""
                    for item in worked_steps
                ]
                if actual_requirement_ids != worked_requirement_ids:
                    return (
                        "worked_steps must cover every non-result requirement "
                        "exactly once in canonical order"
                    )
                claim_by_id = {
                    _clean(item.get("stable_id")): item
                    for item in artifact["claims"]
                }
                execution_rows = list(worked_steps)
                result_row = value.get("result")
                if result_requirement_ids:
                    if (
                        not isinstance(result_row, Mapping)
                        or _clean(result_row.get("requirement_id"))
                        != result_requirement_ids[0]
                    ):
                        return "required terminal result is missing"
                    execution_rows.append(result_row)
                elif result_row is not None:
                    return "result must be null when the objective requires none"
                for index, row in enumerate(execution_rows):
                    requirement_id = _clean(row.get("requirement_id"))
                    requirement = requirement_by_id.get(requirement_id)
                    realization = _clean(row.get("realization"))
                    claim_ids = list(row.get("claim_ids") or [])
                    if (
                        requirement is None or not claim_ids
                        or any(item not in claim_by_id for item in claim_ids)
                    ):
                        return (
                            f"objective execution row {index} has unknown "
                            "requirement or claim identity"
                        )
                    requirement_text = _clean(requirement.get("value"))
                    proof = _deterministic_entailment_basis(
                        realization, requirement_text,
                    )
                    requirement_entailment = 1.0 if proof is not None else 0.0
                    requirement_contradiction = 0.0
                    if proof is None:
                        score = self._score_pair(
                            decision_type="micro_D_objective_requirement",
                            premise=realization,
                            hypothesis=requirement_text,
                        )
                        requirement_entailment = float(score.entailment)
                        requirement_contradiction = float(
                            score.contradiction
                        )
                        if (
                            float(score.entailment) < _PLAN_ENTAILMENT_FLOOR
                            or float(score.contradiction)
                            >= _PLAN_CONTRADICTION_CEILING
                        ):
                            return (
                                f"requirement_id={requirement_id!r} is not "
                                "delivered by its completion-only worked step"
                            )
                    evidence = " ".join(
                        _clean(claim_by_id[item].get("claim"))
                        for item in claim_ids
                    )
                    support = _deterministic_entailment_basis(
                        evidence, realization,
                    )
                    support_entailment = 1.0 if support is not None else 0.0
                    support_contradiction = 0.0
                    if support is None:
                        score = self._score_pair(
                            decision_type="micro_D_worked_step_support",
                            premise=evidence,
                            hypothesis=realization,
                        )
                        support_entailment = float(score.entailment)
                        support_contradiction = float(score.contradiction)
                        if (
                            float(score.entailment) < _PLAN_ENTAILMENT_FLOOR
                            or float(score.contradiction)
                            >= _PLAN_CONTRADICTION_CEILING
                        ):
                            return (
                                f"requirement_id={requirement_id!r} worked "
                                "step is not supported by its bound claims"
                            )
                    objective_proof_ledger[requirement_id] = {
                        "requirement_basis": proof or "nli",
                        "requirement_entailment": requirement_entailment,
                        "requirement_contradiction": (
                            requirement_contradiction
                        ),
                        "claim_support_basis": support or "nli",
                        "claim_support_entailment": support_entailment,
                        "claim_support_contradiction": support_contradiction,
                    }
                execution = assemble_objective_execution(
                    value, artifact=artifact,
                )
                assembled = execution["text"]
            else:
                assembled = claim_assembly["text"]
            if len(assembled) > self._completion_cap:
                return (
                    "assembled completion exceeds the candidate schema ceiling "
                    f"without truncation: length={len(assembled)}, "
                    f"cap={self._completion_cap}; affected JSON pointers="
                    "['/claim_realizations']"
                )
            return numeric_closed_world_error(
                prompt=prompt, output=assembled, artifact=artifact,
            )

        result = self._call_stage(
            stage="micro_D_dpo_chosen" if preference else "micro_D_sft",
            chunk_id=chunk_id,
            system=_REALIZE_SYSTEM,
            user=(
                f"REALIZATION_VIEW_VERSION="
                f"{MICRO_STAGE_D_REALIZATION_VIEW_VERSION}\n"
                f"REALIZATION_VIEW={_stable_json(realization_view)}\n"
                "Return claim_realizations with exactly one row per checklist "
                "claim_id in listed order. Each realization "
                "must reproduce its assigned leakage-safe PUBLIC atomic claim "
                "exactly. Do not paraphrase it again or add transitions, "
                "unrelated assertions, IDs, or PRIVATE evidence quotes.\n"
                "ALLOWED_NUMERIC_LITERALS="
                f"{_stable_json(_allowed_realization_numeric_inventory(artifact))}\n"
                "Use no other numeric literal. Return only claim_realizations; "
                "the prompt is deterministic and is not a model field. "
                "When objective_requirement_contract is present, also return "
                "worked_steps in exact requirement order and result. Each "
                "worked step must execute its requirement using only listed "
                "claim_ids; restating a claim alone is not execution. Return "
                "a terminal result exactly when result_required is true."
            ),
            required_keys=(
                ("claim_realizations", "worked_steps", "result")
                if requirement_rows else ("claim_realizations",)
            ),
            validator=validate,
            response_schema=schema,
        ).value
        assembly = assemble_claim_realizations(
            result["claim_realizations"],
            artifact=artifact,
            synthesis_seed=self._synthesis_seed,
        )
        execution = (
            assemble_objective_execution(result, artifact=artifact)
            if requirement_rows else None
        )
        result[fields[1]] = (
            execution["text"] if execution is not None else assembly["text"]
        )
        result["prompt"] = prompt
        result["_claim_realization_map"] = {
            item["claim_id"]: item["realization"]
            for item in result["claim_realizations"]
        }
        result["_assembled_realization_provenance"] = assembly["provenance"]
        if execution is not None:
            result["_objective_execution"] = execution
            result["_objective_requirement_proofs"] = dict(
                objective_proof_ledger
            )
        return result

    def _finalize_objective_execution(
        self, *, out: Dict[str, Any], artifact: Mapping[str, Any],
        realized: Mapping[str, Any], chunk: Mapping[str, Any],
        extra_stage_calls: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Build and persist the private sidecar before exposing qualification."""
        contract = artifact.get("objective_requirement_contract")
        execution = realized.get("_objective_execution")
        if not isinstance(contract, Mapping) or not isinstance(
            execution, Mapping
        ):
            raise SynthesisProviderError(
                "new staged pair lacks objective execution instrumentation",
                code="staged_micro_objective_execution_missing",
            )
        completion = str(
            out.get("completion") or out.get("chosen") or ""
        ).strip()
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        requirement_by_id = {
            _clean(item.get("requirement_id")): dict(item)
            for item in contract.get("requirements") or []
            if isinstance(item, Mapping)
        }
        validator_fingerprint = content_sha256({
            "contract": objective_requirement_contract_components(),
            "realizer": inspect.getsource(
                MicroStagedSynthesisProvider._realize
            ),
            "reconciler": inspect.getsource(
                reconcile_completion_execution
            ),
        })
        proofs = []
        records = []
        sidecar_steps = []
        proof_ledger = realized.get("_objective_requirement_proofs")
        if not isinstance(proof_ledger, Mapping):
            raise SynthesisProviderError(
                "objective execution proof ledger is missing",
                code="staged_micro_objective_execution_drift",
            )
        for step in execution.get("steps") or []:
            requirement_id = _clean(step.get("requirement_id"))
            requirement = requirement_by_id.get(requirement_id)
            if requirement is None:
                raise SynthesisProviderError(
                    "objective execution step references unknown requirement",
                    code="staged_micro_objective_execution_drift",
                )
            scores = proof_ledger.get(requirement_id)
            if (
                not isinstance(scores, Mapping)
                or float(scores.get("requirement_entailment", 0.0))
                < _PLAN_ENTAILMENT_FLOOR
                or float(scores.get("requirement_contradiction", 1.0))
                >= _PLAN_CONTRADICTION_CEILING
                or float(scores.get("claim_support_entailment", 0.0))
                < _PLAN_ENTAILMENT_FLOOR
                or float(scores.get("claim_support_contradiction", 1.0))
                >= _PLAN_CONTRADICTION_CEILING
            ):
                raise SynthesisProviderError(
                    "objective execution proof scores are incomplete",
                    code="staged_micro_objective_execution_drift",
                )
            proof = {
                "proof_type": "hybrid",
                "requirement_id": requirement_id,
                "requirement_sha256": requirement["requirement_sha256"],
                "realization_sha256": _sha(_clean(step.get("realization"))),
                "claim_ids": list(step.get("claim_ids") or []),
                "completion_span": list(step.get("completion_span") or []),
                "validator_fingerprint": validator_fingerprint,
                "scores": dict(scores),
            }
            proofs.append(proof)
            records.append({
                "requirement_id": requirement_id,
                "status": "delivered",
                "completion_spans": [
                    list(step.get("completion_span") or [])
                ],
                "completion_span_text": _clean(step.get("realization")),
                "proof_sha256": content_sha256(proof),
                "validator_fingerprint": validator_fingerprint,
            })
            sidecar_steps.append({
                "step_id": _clean(step.get("step_id")),
                "requirement_ids": [requirement_id],
                "claim_ids": list(step.get("claim_ids") or []),
                "realization": _clean(step.get("realization")),
                "completion_span": list(step.get("completion_span") or []),
            })
        calls = [
            dict(item) for item in artifact.get("private_stage_calls") or []
            if isinstance(item, Mapping)
        ]
        calls.extend(
            dict(item) for item in (
                [realized.get("_stage_call"), *extra_stage_calls]
            ) if isinstance(item, Mapping)
        )
        calls = [{
            "call_id": _clean(
                item.get("decision_capture_id") or item.get("call_id")
            ),
            "stage": _clean(item.get("stage")),
            "finish_reason": _clean(item.get("finish_reason")) or "stop",
            "prompt_tokens": int(item.get("prompt_tokens", 0)),
            "completion_tokens": int(item.get("completion_tokens", 0)),
            "truncated": bool(item.get("truncated", False)),
        } for item in calls]
        source_text = _clean(chunk.get("text"))
        private_evidence = []
        for item in artifact.get("claims") or []:
            quote = _clean(item.get("evidence_quote"))
            start = source_text.find(quote)
            private_evidence.append({
                "claim_id": _clean(item.get("stable_id")),
                "quote": quote,
                "quote_sha256": _sha(quote),
                "source_block_id": _clean(item.get("source_block_id")),
                "source_span": (
                    [start, start + len(quote)] if start >= 0 else None
                ),
            })
        sidecar = build_private_sidecar(
            requirement_contract=contract,
            release_pair=out,
            claims=[{
                "claim_id": _clean(item.get("stable_id")),
                "claim": _clean(item.get("claim")),
                "source_chunk_ids": [chunk_id],
                "source_block_id": _clean(item.get("source_block_id")),
            } for item in artifact.get("claims") or []],
            source_bindings=[{
                "source_chunk_id": chunk_id,
                "source_sha256": _sha(source_text),
            }],
            steps=sidecar_steps,
            result=execution.get("result"),
            private_evidence=private_evidence,
            proofs=proofs,
            fingerprints={
                "prompt": _sha(_clean(out.get("prompt"))),
                "schema": _schema_sha(_stage_d_realization_schema(
                    artifact, completion_cap=self._completion_cap,
                )),
                "validator": validator_fingerprint,
                "threshold": _sha(_stable_json({
                    "entailment": _PLAN_ENTAILMENT_FLOOR,
                    "contradiction": _PLAN_CONTRADICTION_CEILING,
                })),
            },
            stage_calls=calls,
            decision_capture_id=_clean(out.get("decision_capture_id")),
            final_assembly=execution.get("final_assembly"),
        )
        audit = reconcile_completion_execution(
            completion=completion,
            requirement_contract=contract,
            execution_records=records,
            sidecar=sidecar,
            release_pair=out,
            validator_fingerprint=validator_fingerprint,
        )
        root = getattr(self._capture, "output_dir", None)
        if root is not None:
            self._check_stop(0)
            write_private_sidecar(
                Path(root) / "objective_execution_sidecars", sidecar,
            )
        return {
            "audit": audit,
            "execution_records": records,
            "sidecar": sidecar,
        }

    def _verify_finalized_objective_execution(
        self, *, out: Mapping[str, Any], artifact: Mapping[str, Any],
        finalized: Mapping[str, Any],
    ) -> None:
        """Reconcile Q on fresh execution and every terminal resume."""
        sidecar = finalized.get("sidecar")
        audit = finalized.get("audit")
        records = finalized.get("execution_records")
        contract = artifact.get("objective_requirement_contract")
        if not all(isinstance(item, Mapping) for item in (
            sidecar, audit, contract,
        )) or not isinstance(records, list):
            raise SynthesisProviderError(
                "objective finalization journal is incomplete",
                code="staged_micro_objective_finalization_drift",
            )
        expected = _clean(sidecar.get("sidecar_sha256"))
        root = getattr(self._capture, "output_dir", None)
        if root is not None:
            loaded = load_private_sidecar(
                Path(root) / "objective_execution_sidecars"
                / f"{expected}.json",
                expected_sha256=expected,
            )
            if loaded != sidecar:
                raise SynthesisProviderError(
                    "persisted objective sidecar differs from Q terminal",
                    code="staged_micro_objective_finalization_drift",
                )
        reconciled = reconcile_completion_execution(
            completion=str(
                out.get("completion") or out.get("chosen") or ""
            ).strip(),
            requirement_contract=contract,
            execution_records=records,
            sidecar=sidecar,
            release_pair=out,
            validator_fingerprint=_clean(
                audit.get("validator_fingerprint")
            ),
        )
        if reconciled != audit:
            raise SynthesisProviderError(
                "objective finalization reconciliation differs from Q",
                code="staged_micro_objective_finalization_drift",
            )

    def paraphrase_instruction(
        self, draft: Dict[str, Any], chunk: Dict[str, Any],
        *, preserve_tokens: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        self._check_stop(0)
        store = self._resume_store(chunk=chunk, draft=draft, kind="instruction")
        cache = store.load() if store is not None else {}
        task, artifact = self._assemble(
            chunk, kind="instruction", store=store, cache=cache,
        )
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        realized = self._journaled(
            store, cache, kind="instruction", stage="D", slot=None, attempt=1,
            call=lambda: self._realize(
                chunk_id=chunk_id, artifact=artifact, preference=False,
            ),
        )
        out = {
            key: value for key, value in draft.items()
            if key != "_micro_manifest_identity"
        }
        out.update({
            "prompt": _clean(realized["prompt"]),
            "completion": str(realized["completion"]).strip(),
            "observed_bloom": task["bloom_level"],
            "objective_contract": dict(task["_objective_card"]),
            "provider": getattr(self._base, "_provenance_provider", self._provider_name),
            "synthesis_plan_sha256": artifact["artifact_sha256"],
            "synthesis_contract_version": MICRO_CONTRACT_VERSION,
            "synthesis_seed": self._synthesis_seed,
            "seed": self._synthesis_seed,
            "projection_contract": MICRO_SFT_PROJECTION,
            "chunk_id": chunk_id,
            "source_chunk_id": chunk_id,
            "lo_refs": list(chunk.get("learning_outcome_refs") or []),
            "bloom_level": task["bloom_level"],
            "content_type": _clean(
                chunk.get("content_type_label")
                or chunk.get("chunk_type")
                or chunk.get("content_type")
                or "explanation"
            ),
            "decision_capture_id": self._projection_decision_capture_id(
                realized, stage="micro_D_sft",
            ),
            "provenance": {
                "source_refs": [
                    _clean(item.get("source_block_id"))
                    for item in artifact["claims"]
                ],
                "source_chunk_id": chunk_id,
                "synthesis_plan_sha256": artifact["artifact_sha256"],
                "provider": getattr(
                    self._base, "_provenance_provider", self._provider_name,
                ),
                "model": self._model,
                "claim_realizations": dict(
                    realized["_claim_realization_map"]
                ),
                "claim_evidence": [
                    {
                        "claim_id": _clean(item.get("stable_id")),
                        "claim": _clean(item.get("claim")),
                        "evidence_quote_sha256": _sha(
                            _clean(item.get("evidence_quote"))
                        ),
                        "source_block_id": _clean(item.get("source_block_id")),
                    }
                    for item in artifact["claims"]
                ],
                "assembled_realization": dict(
                    realized["_assembled_realization_provenance"]
                ),
            },
            "source_row_identity": {
                "chunk_id": chunk_id,
                "chunk_sha256": _sha(_stable_json(chunk)),
            },
            "schema_version": "v1",
            "release_identity_sha256": _sha(
                _stable_json(self._release_identity())
            ),
        })
        out["provenance"]["provenance_sha256"] = _sha(
            _stable_json(out["provenance"])
        )
        finalized = self._journaled(
            store, cache, kind="instruction", stage="Q", slot=None, attempt=1,
            call=lambda: self._finalize_objective_execution(
                out=out, artifact=artifact, realized=realized, chunk=chunk,
            ),
        )
        self._verify_finalized_objective_execution(
            out=out, artifact=artifact, finalized=finalized,
        )
        out["_objective_execution_private_sidecar"] = dict(
            finalized["sidecar"]
        )
        out["_objective_execution_candidate"] = {
            "pair_objective_execution": [dict(finalized["audit"])],
            "pair_objective_execution_pass_rate": 1.0,
            "execution_records": list(finalized["execution_records"]),
            "claim_support_rate": 1.0,
            "claim_contradicted_rate": 0.0,
        }
        return out

    def paraphrase_preference(
        self, draft: Dict[str, Any], chunk: Dict[str, Any],
        *, preserve_tokens: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        self._check_stop(0)
        focus = self._focus(chunk)
        eligibility = micro_preference_eligibility(
            chunk, focus=focus, capture=self._capture,
        )
        if not eligibility["eligible"]:
            raise SynthesisProviderError(
                "micro preference input lacks a usable source-backed "
                "misconception candidate",
                code=str(eligibility["reason"]),
                chunk_id=_clean(chunk.get("id") or chunk.get("chunk_id")),
                details={
                    "terminal_content_rejection": True,
                    "stage": "micro_preference_eligibility",
                    "eligibility": eligibility["telemetry"],
                    "model_calls": 0,
                },
            )
        candidates = list(eligibility["candidates"])
        store = self._resume_store(chunk=chunk, draft=draft, kind="preference")
        cache = store.load() if store is not None else {}
        task, artifact = self._assemble(
            chunk, kind="preference", store=store, cache=cache,
        )
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        chosen = self._journaled(
            store, cache, kind="preference", stage="D", slot=None, attempt=1,
            call=lambda: self._realize(
                chunk_id=chunk_id, artifact=artifact, preference=True,
            ),
        )
        def validate_selection(value: Dict[str, Any]) -> Optional[str]:
            candidate_id = _clean(value.get("misconception_id"))
            try:
                _index, _candidate, proof = _stage_e_exact_faulty_step_proof(
                    candidates, candidate_id,
                )
            except SynthesisProviderError:
                return (
                    "misconception_id does not resolve to exactly one "
                    "canonical candidate"
                )
            if not _clean(value.get("rationale")):
                return "rationale required"
            derivations = list(
                getattr(self._validation_audit, "derivations", []) or []
            )
            if proof not in derivations:
                derivations.append(proof)
                self._validation_audit.derivations = derivations
            return None

        selection = self._journaled(
            store, cache, kind="preference", stage="E", slot=None, attempt=1,
            call=lambda: self._call_stage(
                stage="micro_E_misconception_selection",
                chunk_id=chunk_id,
                system=_MISCONCEPTION_SYSTEM,
                user=(
                    f"CANONICAL_CANDIDATES={_stable_json(candidates)}\n"
                    "Return one exact misconception_id from the enumerated "
                    "authority plus a rationale. The orchestrator derives the "
                    "exact faulty_step; do not return it or invent an "
                    "error-mechanism label."
                ),
                required_keys=("misconception_id", "rationale"),
                validator=validate_selection,
                response_schema=_stage_e_selection_schema(candidates),
            ).value,
        )
        selected_id = _clean(selection.get("misconception_id"))
        try:
            selected_index, selected_candidate, _proof = (
                _stage_e_exact_faulty_step_proof(candidates, selected_id)
            )
        except SynthesisProviderError as exc:
            raise SynthesisProviderError(
                "terminal Stage-E selection has unresolved canonical authority",
                code="staged_micro_E_authority_unresolved",
                chunk_id=chunk_id,
                details={
                    "terminal_content_rejection": True,
                    "stage": "micro_E",
                },
            ) from exc
        selection["misconception_index"] = selected_index
        selection["faulty_step"] = selected_candidate["misconception"]
        selection["error_mechanism"] = _clean(
            selected_candidate.get("misconception")
        )
        immutable = {
            "artifact_sha256": artifact["artifact_sha256"],
            "prompt_sha256": _sha(_clean(chosen["prompt"])),
            "chosen_sha256": _sha(_clean(chosen["chosen"])),
            "selection": selection,
        }
        immutable_sha = _sha(_stable_json(immutable))
        def validate_rejected(value: Dict[str, Any]) -> Optional[str]:
            text = _clean(value.get("rejected"))
            if not text or text.casefold() == _clean(chosen["chosen"]).casefold():
                return "rejected is empty or duplicates chosen"
            numeric = numeric_closed_world_error(
                prompt=_clean(chosen["prompt"]), output=text, artifact=artifact,
            )
            if numeric:
                return f"rejected has undeclared numeric material: {numeric}"
            declared = " ".join((
                _clean(selection["error_mechanism"]),
                _clean(selection["faulty_step"]),
            ))
            declared_score = self._score_pair(
                decision_type="micro_rejected_declared_fault",
                premise=text,
                hypothesis=declared,
            )
            if float(declared_score.entailment) < _PLAN_ENTAILMENT_FLOOR:
                return "rejected does not exhibit the declared fault mechanism"
            secondary_entailments = []
            for index, candidate in enumerate(candidates):
                if index == selected_index:
                    continue
                other_fault = _clean(candidate.get("misconception"))
                score = self._score_pair(
                    decision_type="micro_rejected_excluded_fault",
                    premise=text,
                    hypothesis=other_fault,
                )
                if float(score.entailment) >= _PLAN_ENTAILMENT_FLOOR:
                    return f"rejected exhibits a second fault from candidate {index}"
                secondary_entailments.append(float(score.entailment))
            source = " ".join(
                _clean(item.get("evidence_quote")) for item in artifact["claims"]
            )
            leakage = _leakage_error(text, source=source, artifact=artifact)
            if leakage:
                return leakage
            declared_entailment = float(declared_score.entailment)
            secondary_max = max(secondary_entailments, default=0.0)
            quality = {
                "judge_identity": (
                    f"{type(self._nli()).__module__}."
                    f"{type(self._nli()).__qualname__}"
                ),
                "judge_contract_version": MICRO_CONTRACT_VERSION,
                "chosen_score": 1.0,
                "rejected_score": 1.0 - declared_entailment,
                "rejected_declared_fault_score": declared_entailment,
                "excluded_fault_max_score": secondary_max,
                "chosen_minus_rejected_margin": declared_entailment,
                "one_fault_separation_margin": (
                    declared_entailment - secondary_max
                ),
                "confidence": min(
                    declared_entailment, max(0.0, 1.0 - secondary_max),
                ),
                "fault_taxonomy": _clean(
                    candidates[selected_index].get("misconception")
                ),
                "fault_severity": _clean(
                    candidates[selected_index].get("severity") or "unspecified"
                ),
                "rationale_sha256": _sha(_clean(selection.get("rationale"))),
                "chosen_authority": "deterministic_grounding_validators",
            }
            value["_preference_quality"] = quality
            return None

        rejected = self._journaled(
            store, cache, kind="preference", stage="F", slot=None, attempt=1,
            call=lambda: self._call_stage(
                stage="micro_F_one_fault_rejected",
                chunk_id=chunk_id,
                system=_REJECTED_SYSTEM,
                user=(
                    f"IMMUTABLE_METADATA_SHA256={immutable_sha}\n"
                    f"IMMUTABLE_METADATA={_stable_json(immutable)}\n"
                    f"FIXED_PROMPT={_clean(chosen['prompt'])}\n"
                    f"FIXED_CHOSEN={_clean(chosen['chosen'])}\n"
                    "Return only rejected."
                ),
                required_keys=("rejected",),
                validator=validate_rejected,
                response_schema=_REJECTED_SCHEMA,
            ).value,
        )
        recorded_quality = rejected.get("_preference_quality")
        if not isinstance(recorded_quality, Mapping) or not recorded_quality:
            raise SynthesisProviderError(
                "micro DPO terminal lacks preference-quality evidence",
                code="staged_micro_preference_quality_missing",
                chunk_id=chunk_id,
                details={"terminal_content_rejection": True, "stage": "micro_F"},
            )
        out = {
            key: value for key, value in draft.items()
            if key != "_micro_manifest_identity"
        }
        out.update({
            "prompt": _clean(chosen["prompt"]),
            "chosen": str(chosen["chosen"]).strip(),
            "rejected": _clean(rejected["rejected"]),
            "observed_bloom": task["bloom_level"],
            "objective_contract": dict(task["_objective_card"]),
            "provider": getattr(self._base, "_provenance_provider", self._provider_name),
            "synthesis_plan_sha256": artifact["artifact_sha256"],
            "synthesis_contract_version": MICRO_CONTRACT_VERSION,
            "synthesis_seed": self._synthesis_seed,
            "seed": self._synthesis_seed,
            "projection_contract": MICRO_DPO_PROJECTION,
            "chunk_id": chunk_id,
            "source_chunk_id": chunk_id,
            "lo_refs": list(chunk.get("learning_outcome_refs") or []),
            "decision_capture_id": self._projection_decision_capture_id(
                rejected, stage="micro_F_one_fault_rejected",
            ),
            "provenance": {
                "source_refs": [
                    _clean(item.get("source_block_id"))
                    for item in artifact["claims"]
                ],
                "source_chunk_id": chunk_id,
                "synthesis_plan_sha256": artifact["artifact_sha256"],
                "provider": getattr(
                    self._base, "_provenance_provider", self._provider_name,
                ),
                "model": self._model,
                "misconception_metadata_sha256": immutable_sha,
                "claim_realizations": dict(
                    chosen["_claim_realization_map"]
                ),
                "claim_evidence": [
                    {
                        "claim_id": _clean(item.get("stable_id")),
                        "claim": _clean(item.get("claim")),
                        "evidence_quote_sha256": _sha(
                            _clean(item.get("evidence_quote"))
                        ),
                        "source_block_id": _clean(item.get("source_block_id")),
                    }
                    for item in artifact["claims"]
                ],
                "assembled_realization": dict(
                    chosen["_assembled_realization_provenance"]
                ),
            },
            "source_row_identity": {
                "chunk_id": chunk_id,
                "chunk_sha256": _sha(_stable_json(chunk)),
            },
            "rejected_source": "misconception",
            "misconception_id": (
                _clean(candidates[int(selection["misconception_index"])].get("id"))
                or None
            ),
            "schema_version": "v1",
            "release_identity_sha256": _sha(
                _stable_json(self._release_identity())
            ),
            "misconception_metadata_sha256": immutable_sha,
            "preference_quality": dict(recorded_quality),
        })
        out["provenance"]["provenance_sha256"] = _sha(
            _stable_json(out["provenance"])
        )
        finalized = self._journaled(
            store, cache, kind="preference", stage="Q", slot=None, attempt=1,
            call=lambda: self._finalize_objective_execution(
                out=out, artifact=artifact, realized=chosen, chunk=chunk,
                extra_stage_calls=tuple(
                    item for item in (
                        selection.get("_stage_call"),
                        rejected.get("_stage_call"),
                    ) if isinstance(item, Mapping)
                ),
            ),
        )
        self._verify_finalized_objective_execution(
            out=out, artifact=artifact, finalized=finalized,
        )
        out["_objective_execution_private_sidecar"] = dict(
            finalized["sidecar"]
        )
        out["_objective_execution_candidate"] = {
            "pair_objective_execution": [dict(finalized["audit"])],
            "pair_objective_execution_pass_rate": 1.0,
            "execution_records": list(finalized["execution_records"]),
            "claim_support_rate": 1.0,
            "claim_contradicted_rate": 0.0,
        }
        return out


__all__ = [
    "ENV_STAGED_SYNTHESIS_MICRO_V1",
    "MICRO_CONTRACT_VERSION",
    "MICRO_COMPLETION_CAP_CANDIDATES",
    "MICRO_DEFAULT_COMPLETION_CAP",
    "MICRO_DPO_PROJECTION",
    "MICRO_RELEASE_CONTRACT_VERSION",
    "MICRO_SFT_PROJECTION",
    "MICRO_STAGE_MAX_TOKENS",
    "MICRO_STAGE_D_REALIZATION_VIEW_VERSION",
    "MICRO_STAGE_D_REALIZATION_CONTRACT_VERSION",
    "MICRO_STAGE_TOKEN_BUDGET_VERSION",
    "MicroResumeStore",
    "MicroStagedSynthesisProvider",
    "assemble_claims",
    "assemble_claim_realizations",
    "MICRO_PRODUCTION_REPETITION",
    "bind_micro_generation_unit",
    "build_micro_manifest_identity",
    "current_micro_generation_unit",
    "deterministic_stage_a_task",
    "immutable_artifact_error",
    "micro_contract_components",
    "production_variant_token",
    "micro_contract_fingerprint",
    "micro_preference_eligibility",
    "numeric_closed_world_error",
    "staged_synthesis_micro_v1_enabled",
    "stage_d_realization_view",
    "unnecessary_generated_givens_error",
    "validate_typed_givens",
    "verify_expected_input_hashes",
]
