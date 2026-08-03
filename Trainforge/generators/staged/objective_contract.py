"""Deterministic objective-execution instrumentation for staged synthesis.

This module deliberately has no provider or persistence dependency.  It builds
the private, hash-bound record needed to qualify a completion and exposes the
temporary completion-cap seam used by the bounded cap benchmark.  Release
schema projection is intentionally owned elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lib.ontology.cognitive_task import detect_cognitive_task_type
from Trainforge.generators.staged.window_contract import objective_card

OBJECTIVE_REQUIREMENT_CONTRACT_VERSION = (
    "ed4all.objective-requirement-normalization.v1"
)
PRIVATE_SIDECAR_CONTRACT_VERSION = "ed4all.objective-execution-sidecar.v1"
CANDIDATE_COMPLETION_CAPS = (600, 800, 1000, 1200, 1600)
DEFAULT_CANDIDATE_COMPLETION_CAP = 600
_DETACHED_OBJECTIVE_AUDIT_FIELDS = frozenset({
    "pair_objective_execution",
    "pair_objective_execution_pass_rate",
    "_objective_execution_candidate",
    "_objective_execution_private_sidecar",
})

# These canonical task types demand a separately identifiable terminal product.
# The classification is about action semantics, never the subject domain.
_RESULT_ACTIONS = frozenset(
    {
        "identify",
        "explain",
        "summarize",
        "classify",
        "compare",
        "apply",
        "compute",
        "analyze",
        "debug",
        "predict",
        "infer",
        "critique",
        "evaluate",
        "construct",
    }
)


class ObjectiveExecutionContractError(ValueError):
    """Fail-closed objective execution contract error."""


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_sha256(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def objective_requirement_contract_components() -> dict[str, Any]:
    """Return behavior-bearing normalization inputs for resume fingerprints."""
    return {
        "version": OBJECTIVE_REQUIREMENT_CONTRACT_VERSION,
        "result_actions": sorted(_RESULT_ACTIONS),
        "candidate_completion_caps": list(CANDIDATE_COMPLETION_CAPS),
        "default_candidate_completion_cap": DEFAULT_CANDIDATE_COMPLETION_CAP,
    }


def release_content_sha256(pair: Mapping[str, Any]) -> str:
    """Hash the release payload independently of its detached sidecar link.

    The pair carries the sidecar hash inside its objective-execution audit,
    while the sidecar binds the release payload. Excluding exactly those two
    detached audit fields avoids a circular hash without excluding any learner
    content, source identity, provider identity, or ordinary validation audit.
    """
    if not isinstance(pair, Mapping):
        raise ObjectiveExecutionContractError("release pair must be a mapping")
    return content_sha256({
        key: value for key, value in pair.items()
        if key not in _DETACHED_OBJECTIVE_AUDIT_FIELDS
    })


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _clean(raw)
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _task_type(card: Mapping[str, Any]) -> str | None:
    action_text = " ".join(
        _clean(card.get(key))
        for key in ("bloom_verb", "statement", "action_object")
    )
    return detect_cognitive_task_type(action_text)


def _requirement(
    *,
    objective_id: str,
    kind: str,
    ordinal: int,
    value: str,
    rationale: str,
) -> dict[str, Any]:
    canonical = {
        "objective_id": objective_id,
        "kind": kind,
        "ordinal": ordinal,
        "value": _clean(value),
    }
    digest = content_sha256(canonical)
    return {
        **canonical,
        "requirement_id": f"req-{digest[:24]}",
        "requirement_sha256": digest,
        "derivation_rationale": rationale,
    }


def derive_objective_requirements(
    raw_objective_card: Mapping[str, Any],
    *,
    max_requirements: int | None = None,
) -> dict[str, Any]:
    """Project an objective card to an ordered, deterministic requirement set.

    Values come only from the objective card.  The sole generated value is the
    structural result marker, whose presence is selected from canonical
    cognitive action semantics.
    """
    if not isinstance(raw_objective_card, Mapping):
        raise ObjectiveExecutionContractError("objective card must be a mapping")
    card = objective_card(raw_objective_card)
    objective_id = _clean(card.get("id")).lower()
    if not objective_id:
        raise ObjectiveExecutionContractError("objective card id is required")

    task_type = _task_type(card)
    action = _clean(card.get("bloom_verb"))
    requirements: list[dict[str, Any]] = []

    def add(kind: str, values: Sequence[str], rationale: str) -> None:
        for ordinal, value in enumerate(_unique(values)):
            requirements.append(
                _requirement(
                    objective_id=objective_id,
                    kind=kind,
                    ordinal=ordinal,
                    value=value,
                    rationale=rationale,
                )
            )

    if action:
        add(
            "action",
            [action],
            "Projected from objective_card.bloom_verb.",
        )
    add(
        "condition",
        list(card.get("conditions") or []),
        "Projected from objective_card.conditions in canonical order.",
    )
    add(
        "content_obligation",
        list(card.get("content_obligations") or []),
        "Projected from objective_card.content_obligations in canonical order.",
    )
    add(
        "performance_criterion",
        list(card.get("performance_criteria") or []),
        "Projected from objective_card.performance_criteria in canonical order.",
    )
    result_required = task_type in _RESULT_ACTIONS
    if result_required:
        add(
            "result",
            [f"terminal product of the {task_type} action"],
            (
                "Derived from canonical cognitive-task action semantics; "
                f"detect_cognitive_task_type resolved {task_type!r}."
            ),
        )

    if not requirements:
        raise ObjectiveExecutionContractError(
            "objective card produces no execution requirements"
        )
    if max_requirements is not None:
        if isinstance(max_requirements, bool) or max_requirements <= 0:
            raise ObjectiveExecutionContractError(
                "max_requirements must be a positive integer"
            )
        if len(requirements) > max_requirements:
            raise ObjectiveExecutionContractError(
                "objective requirement count exceeds configured capacity"
            )

    contract_core = {
        "contract_version": OBJECTIVE_REQUIREMENT_CONTRACT_VERSION,
        "objective_id": objective_id,
        "cognitive_task_type": task_type,
        "result_required": result_required,
        "requirements": requirements,
    }
    return {
        **contract_core,
        "objective_contract_sha256": content_sha256(contract_core),
        "normalized_objective_card": card,
    }


def validate_candidate_completion(
    completion: str,
    *,
    cap: int = DEFAULT_CANDIDATE_COMPLETION_CAP,
    finish_reason: str = "stop",
    truncated: bool = False,
) -> dict[str, Any]:
    """Validate one unmodified candidate against the temporary cap seam."""
    if cap not in CANDIDATE_COMPLETION_CAPS:
        raise ObjectiveExecutionContractError(
            f"candidate completion cap must be one of {CANDIDATE_COMPLETION_CAPS}"
        )
    text = str(completion)
    length = len(text)
    if truncated:
        raise ObjectiveExecutionContractError("candidate completion was truncated")
    if _clean(finish_reason).lower() == "length":
        raise ObjectiveExecutionContractError(
            "candidate completion has length finish reason"
        )
    if length >= cap:
        raise ObjectiveExecutionContractError(
            "candidate completion reached or exceeded its character cap"
        )
    return {
        "selected_cap": cap,
        "completion_characters": length,
        "cap_utilization": length / cap,
        "finish_reason": _clean(finish_reason).lower(),
        "truncated": False,
    }


def _normalized_sequence(
    values: Iterable[Mapping[str, Any]], name: str
) -> list[dict[str, Any]]:
    result = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ObjectiveExecutionContractError(
                f"{name} entries must be mappings"
            )
        result.append(dict(value))
    return result


def build_private_sidecar(
    *,
    requirement_contract: Mapping[str, Any],
    release_pair: Mapping[str, Any],
    claims: Iterable[Mapping[str, Any]],
    source_bindings: Iterable[Mapping[str, Any]],
    steps: Iterable[Mapping[str, Any]],
    result: Mapping[str, Any] | None,
    private_evidence: Iterable[Mapping[str, Any]],
    proofs: Iterable[Mapping[str, Any]],
    fingerprints: Mapping[str, str],
    stage_calls: Iterable[Mapping[str, Any]],
    decision_capture_id: str,
    final_assembly: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical private sidecar and append its self-excluding hash."""
    contract = dict(requirement_contract)
    expected_hash = contract.get("objective_contract_sha256")
    core = {
        key: contract[key]
        for key in (
            "contract_version",
            "objective_id",
            "cognitive_task_type",
            "result_required",
            "requirements",
        )
    }
    if expected_hash != content_sha256(core):
        raise ObjectiveExecutionContractError(
            "objective requirement contract hash mismatch"
        )
    calls = _normalized_sequence(stage_calls, "stage_calls")
    for call in calls:
        if bool(call.get("truncated")):
            raise ObjectiveExecutionContractError("stage call was truncated")
        if _clean(call.get("finish_reason")).lower() == "length":
            raise ObjectiveExecutionContractError(
                "stage call has length finish reason"
            )
        for key in ("prompt_tokens", "completion_tokens"):
            value = call.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ObjectiveExecutionContractError(
                    f"stage call {key} must be a non-negative integer"
                )
    dc_id = _clean(decision_capture_id)
    if not dc_id:
        raise ObjectiveExecutionContractError("decision_capture_id is required")

    claim_records = _normalized_sequence(claims, "claims")
    source_records = _normalized_sequence(source_bindings, "source_bindings")
    step_records = _normalized_sequence(steps, "steps")
    evidence_records = _normalized_sequence(
        private_evidence, "private_evidence"
    )
    proof_records = _normalized_sequence(proofs, "proofs")
    requirement_ids = {
        item["requirement_id"] for item in contract["requirements"]
    }
    claim_ids = {_clean(item.get("claim_id")) for item in claim_records}
    source_ids = {
        _clean(item.get("source_chunk_id")) for item in source_records
    }
    if "" in claim_ids or len(claim_ids) != len(claim_records):
        raise ObjectiveExecutionContractError(
            "claim IDs must be non-empty and unique"
        )
    if "" in source_ids or len(source_ids) != len(source_records):
        raise ObjectiveExecutionContractError(
            "source chunk IDs must be non-empty and unique"
        )
    for claim in claim_records:
        bound = set(_unique(claim.get("source_chunk_ids") or []))
        if not bound or not bound.issubset(source_ids):
            raise ObjectiveExecutionContractError(
                "every claim must resolve to a sidecar source binding"
            )
    for step in step_records:
        if not set(_unique(step.get("requirement_ids") or [])).issubset(
            requirement_ids
        ):
            raise ObjectiveExecutionContractError(
                "worked step references an unknown requirement"
            )
        if not set(_unique(step.get("claim_ids") or [])).issubset(claim_ids):
            raise ObjectiveExecutionContractError(
                "worked step references an unknown claim"
            )
    for evidence in evidence_records:
        if _clean(evidence.get("claim_id")) not in claim_ids:
            raise ObjectiveExecutionContractError(
                "private evidence references an unknown claim"
            )

    sidecar_core = {
        "contract_version": PRIVATE_SIDECAR_CONTRACT_VERSION,
        "objective_card": contract.get("normalized_objective_card"),
        "objective_contract_sha256": expected_hash,
        "requirements": contract["requirements"],
        "claims": claim_records,
        "source_bindings": source_records,
        "steps": step_records,
        "result": dict(result) if isinstance(result, Mapping) else None,
        "private_evidence": evidence_records,
        "proofs": proof_records,
        "fingerprints": dict(fingerprints),
        "stage_calls": calls,
        "decision_capture_id": dc_id,
        "final_assembly": (
            dict(final_assembly)
            if isinstance(final_assembly, Mapping) else None
        ),
        "release_content_sha256": release_content_sha256(release_pair),
    }
    return {
        **sidecar_core,
        "sidecar_sha256": content_sha256(sidecar_core),
    }


def write_private_sidecar(
    root: Path, sidecar: Mapping[str, Any],
) -> Path:
    """Persist one immutable, content-addressed private sidecar with fsync."""
    record = dict(sidecar)
    digest = record.pop("sidecar_sha256", None)
    if not isinstance(digest, str) or digest != content_sha256(record):
        raise ObjectiveExecutionContractError("private sidecar hash mismatch")
    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    path = destination_root / f"{digest}.json"
    payload = f"{_canonical_json(sidecar)}\n".encode("utf-8")
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
        )
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ObjectiveExecutionContractError(
                "existing private sidecar content differs"
            )
        return path
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(destination_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Leave an incomplete file visible so resume fails closed; do not
        # silently replace evidence whose durability is uncertain.
        raise
    return path


def load_private_sidecar(
    path: Path, *, expected_sha256: str,
) -> dict[str, Any]:
    """Load and verify an immutable sidecar before terminal resume."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObjectiveExecutionContractError(
            "private sidecar is absent or unparseable"
        ) from exc
    if not isinstance(record, dict):
        raise ObjectiveExecutionContractError(
            "private sidecar must be a JSON object"
        )
    core = dict(record)
    declared = core.pop("sidecar_sha256", None)
    actual = content_sha256(core)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256))
        or declared != expected_sha256
        or actual != expected_sha256
    ):
        raise ObjectiveExecutionContractError("private sidecar hash mismatch")
    return record


def reconcile_completion_execution(
    *,
    completion: str,
    requirement_contract: Mapping[str, Any],
    execution_records: Iterable[Mapping[str, Any]],
    sidecar: Mapping[str, Any],
    release_pair: Mapping[str, Any],
    validator_fingerprint: str,
) -> dict[str, Any]:
    """Fail closed unless completion-only coverage and sidecar binding are exact."""
    sidecar_record = dict(sidecar)
    sidecar_hash = sidecar_record.pop("sidecar_sha256", None)
    if sidecar_hash != content_sha256(sidecar_record):
        raise ObjectiveExecutionContractError("private sidecar hash mismatch")
    if sidecar_record.get("release_content_sha256") != release_content_sha256(
        release_pair
    ):
        raise ObjectiveExecutionContractError("release content hash mismatch")
    release_answer = (
        release_pair.get("completion")
        if "completion" in release_pair
        else release_pair.get("chosen")
    )
    if not isinstance(release_answer, str) or completion != release_answer:
        raise ObjectiveExecutionContractError(
            "completion must exactly match release completion or chosen"
        )
    contract_hash = requirement_contract.get("objective_contract_sha256")
    if sidecar_record.get("objective_contract_sha256") != contract_hash:
        raise ObjectiveExecutionContractError("objective contract hash mismatch")

    expected = list(requirement_contract.get("requirements") or [])
    expected_by_id = {item["requirement_id"]: item for item in expected}
    if len(expected_by_id) != len(expected):
        raise ObjectiveExecutionContractError("duplicate expected requirement id")
    records = _normalized_sequence(execution_records, "execution_records")
    if {item.get("requirement_id") for item in records} != set(expected_by_id):
        raise ObjectiveExecutionContractError(
            "execution requirements do not reconcile exactly"
        )
    if len(records) != len(expected):
        raise ObjectiveExecutionContractError("duplicate execution requirement")

    proof_hashes = {
        content_sha256(proof)
        for proof in sidecar_record.get("proofs") or []
        if isinstance(proof, Mapping)
    }
    for record in records:
        if record.get("status") != "delivered":
            raise ObjectiveExecutionContractError(
                "every objective requirement must be delivered"
            )
        if record.get("validator_fingerprint") != validator_fingerprint:
            raise ObjectiveExecutionContractError(
                "validator fingerprint mismatch"
            )
        if record.get("proof_sha256") not in proof_hashes:
            raise ObjectiveExecutionContractError("proof hash mismatch")
        spans = record.get("completion_spans")
        if not isinstance(spans, list) or not spans:
            raise ObjectiveExecutionContractError(
                "delivered requirement needs a completion span"
            )
        seen_spans: set[tuple[int, int]] = set()
        for span in spans:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(isinstance(value, bool) for value in span)
                or not all(isinstance(value, int) for value in span)
            ):
                raise ObjectiveExecutionContractError(
                    "completion span must be an integer [start,end) pair"
                )
            start, end = span
            key = (start, end)
            if start < 0 or end <= start or end > len(completion):
                raise ObjectiveExecutionContractError(
                    "completion span is out of bounds"
                )
            if key in seen_spans:
                raise ObjectiveExecutionContractError(
                    "duplicate completion span"
                )
            seen_spans.add(key)
            expected_text = record.get("completion_span_text")
            if expected_text is not None and completion[start:end] != expected_text:
                raise ObjectiveExecutionContractError(
                    "completion span text mismatch"
                )

    result_required = bool(requirement_contract.get("result_required"))
    result_ids = {
        item["requirement_id"]
        for item in expected
        if item.get("kind") == "result"
    }
    delivered_ids = {item["requirement_id"] for item in records}
    if result_required and not result_ids.issubset(delivered_ids):
        raise ObjectiveExecutionContractError("required result is missing")
    if result_required and not isinstance(sidecar_record.get("result"), Mapping):
        raise ObjectiveExecutionContractError(
            "private sidecar required result is missing"
        )
    assembly = sidecar_record.get("final_assembly")
    if isinstance(assembly, Mapping):
        sidecar_steps = sidecar_record.get("steps")
        if (
            assembly.get("contract_version")
            != "ed4all.objective-execution-final-assembly.v1"
            or not isinstance(sidecar_steps, list)
            or assembly.get("ordered_step_ids")
            != [item.get("step_id") for item in sidecar_steps]
            or assembly.get("completion_spans")
            != [item.get("completion_span") for item in sidecar_steps]
            or assembly.get("item_sha256")
            != [
                hashlib.sha256(
                    str(item.get("realization") or "").encode("utf-8")
                ).hexdigest()
                for item in sidecar_steps
            ]
        ):
            raise ObjectiveExecutionContractError(
                "final objective assembly ledger mismatch"
            )
        separator = assembly.get("separator")
        if not isinstance(separator, str):
            raise ObjectiveExecutionContractError(
                "final objective assembly separator is invalid"
            )
        reconstructed = separator.join(
            str(item.get("realization") or "") for item in sidecar_steps
        )
        if (
            reconstructed != completion
            or assembly.get("assembled_sha256")
            != hashlib.sha256(completion.encode("utf-8")).hexdigest()
        ):
            raise ObjectiveExecutionContractError(
                "final objective assembly does not exactly cover completion"
            )

    return {
        "objective_id": requirement_contract["objective_id"],
        "completion_only": True,
        "status": "delivered",
        "requirement_pass_rate": 1.0,
        "result_required": result_required,
        "result_delivered": not result_required or bool(result_ids),
        "objective_contract_sha256": contract_hash,
        "sidecar_sha256": sidecar_hash,
        "validator_fingerprint": validator_fingerprint,
    }


__all__ = [
    "CANDIDATE_COMPLETION_CAPS",
    "DEFAULT_CANDIDATE_COMPLETION_CAP",
    "OBJECTIVE_REQUIREMENT_CONTRACT_VERSION",
    "PRIVATE_SIDECAR_CONTRACT_VERSION",
    "ObjectiveExecutionContractError",
    "build_private_sidecar",
    "content_sha256",
    "derive_objective_requirements",
    "load_private_sidecar",
    "objective_requirement_contract_components",
    "reconcile_completion_execution",
    "release_content_sha256",
    "validate_candidate_completion",
    "write_private_sidecar",
]
