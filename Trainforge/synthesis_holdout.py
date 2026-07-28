"""Fail-closed, manually-reviewed evaluation holdout for synthesis."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from Trainforge.eval.manual_review import (
    ManualReviewError,
    evaluate_manual_review_gate,
)
from Trainforge.eval.expanded_suite import _source_family_closure

ENV_ENABLED = "TRAINFORGE_SYNTHESIS_HOLDOUT_EXCLUSION"
ENV_REGISTRY = "TRAINFORGE_SYNTHESIS_HOLDOUT_REGISTRY"
ENV_MANIFEST = "TRAINFORGE_SYNTHESIS_HOLDOUT_MANIFEST"
_TRUE = frozenset({"1", "true", "yes", "on"})
_SPLITS = ("checkpoint_dev", "grounding_stress", "pedagogy_misconception")


class SynthesisHoldoutError(RuntimeError):
    """The reviewed evaluation contract cannot safely govern synthesis."""


def holdout_exclusion_enabled(
    environment: Optional[Mapping[str, str]] = None,
) -> bool:
    source = os.environ if environment is None else environment
    return str(source.get(ENV_ENABLED, "")).strip().lower() in _TRUE


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SynthesisHoldoutError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SynthesisHoldoutError(f"{label} must be a JSON object")
    return value


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SynthesisHoldoutError(f"{field} must be a SHA-256 string")
    return value


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise SynthesisHoldoutError(f"{field} must be a list of non-empty strings")
    result = [item.strip() for item in value]
    if result != sorted(set(result)):
        raise SynthesisHoldoutError(f"{field} must be sorted and unique")
    return result


def _repo_root(contract_path: Path) -> Optional[Path]:
    for parent in (contract_path.parent, *contract_path.parents):
        if (parent / "Trainforge").is_dir() and (parent / "LibV2").is_dir():
            return parent
    return None


def _resolve_path(raw: Any, *, contract_path: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SynthesisHoldoutError("contract artifact path must be non-empty")
    supplied = Path(raw)
    candidates = [supplied]
    root = _repo_root(contract_path)
    if not supplied.is_absolute():
        candidates.extend([contract_path.parent / supplied])
        if root is not None:
            candidates.append(root / supplied)
    elif root is not None:
        parts = supplied.parts
        for anchor in ("LibV2", "Trainforge"):
            if anchor in parts:
                candidates.append(root.joinpath(*parts[parts.index(anchor) :]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SynthesisHoldoutError(f"contract artifact is missing: {raw}")


def _normalize_content_text(value: Any) -> str:
    plain = html.unescape(re.sub(r"<[^>]+>", " ", str(value or ""))).lower()
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def _subject_fingerprints(path: Path) -> tuple[str, str, int]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise SynthesisHoldoutError(f"invalid split subject {path}: {exc}") from exc
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("fingerprint"), str)
        or len(row["fingerprint"]) != 64
        for row in rows
    ):
        raise SynthesisHoldoutError(f"split subject lacks item fingerprints: {path}")
    split_fingerprint = _hash_json([row["fingerprint"] for row in rows])
    canonical_content = []
    for row in rows:
        expected = row.get("expected")
        expected_answer = row.get("expected_answer")
        answer = row.get("answers")
        if not answer and isinstance(expected, Mapping):
            answer = expected.get("answer")
        if not answer and isinstance(expected_answer, Mapping):
            answer = expected_answer.get("keyed_correct_results")
        objective = row.get("objective") or row.get("canonical_objective")
        if not objective:
            objective = {"id": row.get("canonical_objective_id")}
        objective_id = (
            objective.get("id") if isinstance(objective, Mapping) else None
        )
        canonical_content.append(
            {
                "prompt": _normalize_content_text(
                    row.get("question") or row.get("prompt")
                ),
                "answer": _normalize_content_text(answer),
                "objective_id": objective_id,
                "source_chunk_id": row.get("source_chunk_id"),
            }
        )
    return split_fingerprint, _hash_json(canonical_content), len(rows)


def _without_fingerprint(
    value: Mapping[str, Any], field: str,
) -> tuple[str, dict[str, Any]]:
    declared = _sha(value.get(field), field=field)
    payload = dict(value)
    payload.pop(field, None)
    return declared, payload


def _subject_source_ids(path: Path) -> set[str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {
        str(row.get("source_chunk_id") or "")
        for row in rows
        if isinstance(row, Mapping)
    }
    if "" in result or len(result) != len(rows):
        raise SynthesisHoldoutError(
            f"split subject has missing or duplicate source chunks: {path}"
        )
    return result


@dataclass(frozen=True)
class SynthesisHoldoutRegistry:
    path: Path
    manifest_path: Path
    registry_fingerprint: str
    corpus_sha256: str
    objectives_sha256: str
    exclusion_sha256: str
    forbidden_chunk_ids: frozenset[str]
    assessment_chunk_ids: frozenset[str]
    final_source_chunk_ids: frozenset[str]
    family_chunk_ids: frozenset[str]
    trust_identity: Mapping[str, str]

    @property
    def identity(self) -> dict[str, str]:
        return dict(self.trust_identity)

    def reserves(self, chunk: Mapping[str, Any]) -> bool:
        return str(chunk.get("id") or chunk.get("chunk_id") or "") in (
            self.forbidden_chunk_ids
        )


def load_synthesis_holdout_registry(
    *,
    chunks_path: Path,
    objectives_path: Path,
    chunks: Optional[Sequence[Mapping[str, Any]]] = None,
    registry_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> Optional[SynthesisHoldoutRegistry]:
    """Prove the final reviewed contract before any synthesis-owned write."""

    source = os.environ if environment is None else environment
    if not holdout_exclusion_enabled(source):
        return None
    raw_registry = str(registry_path or source.get(ENV_REGISTRY, "")).strip()
    raw_manifest = str(manifest_path or source.get(ENV_MANIFEST, "")).strip()
    if not raw_registry:
        raise SynthesisHoldoutError(
            f"{ENV_REGISTRY} is required when {ENV_ENABLED}=true"
        )
    if not raw_manifest:
        raise SynthesisHoldoutError(
            f"{ENV_MANIFEST} is required when {ENV_ENABLED}=true"
        )
    registry_file, manifest_file = Path(raw_registry), Path(raw_manifest)
    registry = _load_object(registry_file, label="synthesis holdout registry")
    manifest_raw_sha = _hash_file(manifest_file) if manifest_file.is_file() else ""
    manifest = _load_object(manifest_file, label="expanded-suite final manifest")
    manifest_fingerprint, manifest_payload = _without_fingerprint(
        manifest, "manifest_fingerprint"
    )
    if manifest_fingerprint != _hash_json(manifest_payload):
        raise SynthesisHoldoutError("expanded-suite manifest fingerprint mismatch")
    registry_fingerprint, registry_payload = _without_fingerprint(
        registry, "registry_fingerprint"
    )
    if registry_fingerprint != _hash_json(registry_payload):
        raise SynthesisHoldoutError("holdout registry fingerprint mismatch")
    if (
        manifest.get("schema_version") != "expanded-suite-final-contract-v1"
        or manifest.get("candidate") is not False
        or manifest.get("ready") is not True
        or manifest.get("ready_blockers") != []
    ):
        raise SynthesisHoldoutError("expanded-suite manifest is not final and ready")
    if registry.get("candidate") is not False or registry.get("ready") is not True:
        raise SynthesisHoldoutError("holdout exclusion registry is not final and ready")

    manifest_exclusion = manifest.get("training_input_exclusion")
    if not isinstance(manifest_exclusion, Mapping):
        raise SynthesisHoldoutError("manifest lacks training_input_exclusion")
    if _resolve_path(
        manifest_exclusion.get("path"), contract_path=manifest_file
    ).resolve() != registry_file.resolve():
        raise SynthesisHoldoutError("manifest does not bind the selected exclusion registry")
    registry_raw_sha = _hash_file(registry_file)
    declared_registry_sha = manifest_exclusion.get("sha256")
    if declared_registry_sha is not None and registry_raw_sha != _sha(
        declared_registry_sha, field="training_input_exclusion.sha256"
    ):
        raise SynthesisHoldoutError("holdout exclusion registry raw hash mismatch")

    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise SynthesisHoldoutError("manifest lacks source_hashes")
    corpus_sha = _sha(source_hashes.get("corpus_chunks"), field="corpus_chunks")
    objectives_sha = _sha(source_hashes.get("objectives"), field="objectives")
    if not chunks_path.is_file() or _hash_file(chunks_path) != corpus_sha:
        raise SynthesisHoldoutError("holdout registry corpus hash mismatch")
    if not objectives_path.is_file() or _hash_file(objectives_path) != objectives_sha:
        raise SynthesisHoldoutError("holdout registry objective hash mismatch")
    if registry.get("source_hashes") != source_hashes:
        raise SynthesisHoldoutError("manifest and exclusion source hashes disagree")

    forbidden = _string_list(registry.get("chunk_ids"), field="chunk_ids")
    assessment = _string_list(
        registry.get("assessment_chunk_ids"), field="assessment_chunk_ids"
    )
    registered_subject_ids = registry.get("subject_source_chunk_ids")
    if not isinstance(registered_subject_ids, Mapping):
        raise SynthesisHoldoutError("subject_source_chunk_ids are required")
    family = _string_list(
        registry.get("derived_source_family_chunk_ids", []),
        field="derived_source_family_chunk_ids",
    )
    exclusion_sha = _sha(registry.get("chunk_ids_sha256"), field="chunk_ids_sha256")
    if exclusion_sha != _hash_json(forbidden):
        raise SynthesisHoldoutError("holdout exclusion hash mismatch")
    if manifest_exclusion.get("chunk_ids_sha256") != exclusion_sha:
        raise SynthesisHoldoutError("manifest exclusion ID hash mismatch")
    if registry.get("registry_fingerprint") != manifest_exclusion.get(
        "registry_fingerprint"
    ):
        raise SynthesisHoldoutError("manifest exclusion fingerprint mismatch")

    rows = list(chunks) if chunks is not None else [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {
        str(row.get("id") or row.get("chunk_id") or ""): row
        for row in rows
        if isinstance(row, Mapping) and (row.get("id") or row.get("chunk_id"))
    }
    if len(by_id) != len(rows):
        raise SynthesisHoldoutError("corpus contains missing or duplicate chunk IDs")
    unknown = sorted(set(forbidden) - set(by_id))
    if unknown:
        raise SynthesisHoldoutError(
            "holdout registry contains unknown chunk IDs: " + ", ".join(unknown[:5])
        )
    actual_assessment = {
        chunk_id
        for chunk_id, row in by_id.items()
        if row.get("chunk_type") == "assessment_item"
    }
    if actual_assessment != set(assessment):
        raise SynthesisHoldoutError(
            "assessment holdout set does not exactly match the bound corpus"
        )

    subjects = manifest.get("subjects")
    split_fingerprints = registry.get("split_fingerprints")
    if not isinstance(subjects, Mapping) or not isinstance(
        split_fingerprints, Mapping
    ):
        raise SynthesisHoldoutError("subjects and split_fingerprints are required")
    identity: dict[str, str] = {
        "holdout_manifest_sha256": manifest_raw_sha,
        "holdout_manifest_fingerprint": manifest_fingerprint,
        "holdout_registry_sha256": registry_raw_sha,
        "holdout_registry_fingerprint": registry_fingerprint,
        "holdout_corpus_sha256": corpus_sha,
        "holdout_objectives_sha256": objectives_sha,
        "holdout_exclusion_sha256": exclusion_sha,
        "holdout_exclusion_chunk_count": str(len(forbidden)),
    }
    actual_subject_ids: set[str] = set()
    for split in _SPLITS:
        spec = subjects.get(split)
        if not isinstance(spec, Mapping):
            raise SynthesisHoldoutError(f"manifest lacks subject {split}")
        subject_path = _resolve_path(spec.get("path"), contract_path=manifest_file)
        split_source_ids = _subject_source_ids(subject_path)
        declared_split_ids = set(
            _string_list(
                registered_subject_ids.get(split),
                field=f"subject_source_chunk_ids.{split}",
            )
        )
        if split_source_ids != declared_split_ids:
            raise SynthesisHoldoutError(
                f"{split} subject source exclusion mismatch"
            )
        actual_subject_ids.update(split_source_ids)
        subject_sha = _sha(spec.get("sha256"), field=f"{split}.sha256")
        if _hash_file(subject_path) != subject_sha:
            raise SynthesisHoldoutError(f"{split} subject hash mismatch")
        actual_split, actual_content, count = _subject_fingerprints(subject_path)
        expected_split = _sha(
            spec.get("split_fingerprint"), field=f"{split}.split_fingerprint"
        )
        if split_fingerprints.get(split) != expected_split:
            raise SynthesisHoldoutError(
                f"{split} split_fingerprints contract mismatch"
            )
        if actual_split != expected_split:
            raise SynthesisHoldoutError(f"{split} item fingerprint mismatch")
        content_fingerprint = _sha(
            spec.get("content_fingerprint"), field=f"{split}.content_fingerprint"
        )
        if actual_content != content_fingerprint:
            raise SynthesisHoldoutError(f"{split} content fingerprint mismatch")
        registry_content_fingerprints = registry.get(
            "split_content_fingerprints"
        )
        if (
            not isinstance(registry_content_fingerprints, Mapping)
            or registry_content_fingerprints.get(split) != actual_content
        ):
            raise SynthesisHoldoutError(
                f"{split} split_content_fingerprints contract mismatch"
            )
        if spec.get("item_count") != count:
            raise SynthesisHoldoutError(f"{split} item count mismatch")

        slots = manifest.get("review_artifact_slots")
        slot = slots.get(split) if isinstance(slots, Mapping) else None
        gate_spec = slot.get("combined_gate") if isinstance(slot, Mapping) else None
        if (
            not isinstance(slot, Mapping)
            or slot.get("semantic_approval") is not True
            or slot.get("status") != "approved_combined_manual_review_gate"
            or not isinstance(gate_spec, Mapping)
        ):
            raise SynthesisHoldoutError(f"{split} is unadjudicated")
        gate_path = _resolve_path(gate_spec.get("path"), contract_path=manifest_file)
        gate_sha = _sha(gate_spec.get("sha256"), field=f"{split}.gate.sha256")
        if _hash_file(gate_path) != gate_sha:
            raise SynthesisHoldoutError(f"{split} combined gate hash mismatch")
        gate_doc = _load_object(gate_path, label=f"{split} combined gate")
        gate_fingerprint, gate_payload = _without_fingerprint(
            gate_doc, "fingerprint"
        )
        if gate_fingerprint != _hash_json(gate_payload):
            raise SynthesisHoldoutError(
                f"{split} combined gate fingerprint mismatch"
            )
        gate_result = gate_doc.get("gate", {}).get("result", {})
        if (
            gate_doc.get("gate", {}).get("completed") is not True
            or gate_result.get("subject_sha256") != subject_sha
        ):
            raise SynthesisHoldoutError(f"{split} combined gate is incomplete or stale")
        inputs = gate_doc.get("inputs")
        review_specs = inputs.get("reviews") if isinstance(inputs, Mapping) else None
        if not isinstance(review_specs, list) or len(review_specs) < 2:
            raise SynthesisHoldoutError(f"{split} gate lacks two review artifacts")
        compatible_reviews: list[Path] = []
        review_hashes: list[str] = []
        for index, review_spec in enumerate(review_specs):
            if not isinstance(review_spec, Mapping):
                raise SynthesisHoldoutError(f"{split} review contract is invalid")
            review_path = _resolve_path(
                review_spec.get("path"), contract_path=manifest_file
            )
            expected_review_sha = _sha(
                review_spec.get("sha256"), field=f"{split}.review[{index}].sha256"
            )
            if _hash_file(review_path) != expected_review_sha:
                raise SynthesisHoldoutError(f"{split} review artifact hash mismatch")
            review_hashes.append(expected_review_sha)
            review_doc = _load_object(review_path, label=f"{split} review")
            if review_doc.get("schema_version") == "manual-eval-review-v1":
                compatible_reviews.append(review_path)
        required_reviewers = inputs.get("required_reviewers")
        expected_count = inputs.get("expected_item_count")
        if len(compatible_reviews) != required_reviewers:
            raise SynthesisHoldoutError(f"{split} has missing or unadjudicated reviews")
        try:
            recomputed = evaluate_manual_review_gate(
                subject_path,
                compatible_reviews,
                expected_item_count=int(expected_count),
                required_reviewers=int(required_reviewers),
            )
        except (ManualReviewError, TypeError, ValueError) as exc:
            raise SynthesisHoldoutError(
                f"{split} manual review recomputation failed: {exc}"
            ) from exc
        if not recomputed.passed or recomputed.defects:
            raise SynthesisHoldoutError(
                f"{split} manual review gate rejected or disagreed"
            )
        if recomputed.as_dict() != gate_result:
            raise SynthesisHoldoutError(f"{split} stored gate result is stale")
        if (
            gate_result.get("passed") is not True
            or gate_result.get("defects") != []
        ):
            raise SynthesisHoldoutError(
                f"{split} combined gate did not pass cleanly"
            )
        identity.update(
            {
                f"holdout_{split}_subject_sha256": subject_sha,
                f"holdout_{split}_split_fingerprint": expected_split,
                f"holdout_{split}_content_fingerprint": content_fingerprint,
                f"holdout_{split}_gate_sha256": gate_sha,
                f"holdout_{split}_gate_fingerprint": gate_fingerprint,
                f"holdout_{split}_review_artifacts_sha256": _hash_json(
                    review_hashes
                ),
            }
        )
    family_closed_list, derived_list, family_audit = _source_family_closure(
        rows, actual_subject_ids
    )
    family_closed = set(family_closed_list)
    family_policy = registry.get("family_policy")
    if (
        not isinstance(family_policy, Mapping)
        or family_policy.get("contract") != "cf_block_id_or_contiguous_follows_v1"
        or family_policy.get("status") != "approved"
        or family_policy.get("source_references_used") is not False
        or registry.get("family_contract_audit") != family_audit
    ):
        raise SynthesisHoldoutError("family-closure policy/audit mismatch")
    declared_family_closed = set(
        _string_list(
            registry.get("family_closed_source_chunk_ids"),
            field="family_closed_source_chunk_ids",
        )
    )
    if family_closed != declared_family_closed:
        raise SynthesisHoldoutError("family-closed source exclusion mismatch")
    if set(family) != set(derived_list):
        raise SynthesisHoldoutError("derived source-family exclusion mismatch")
    required_forbidden = actual_assessment | family_closed
    if set(forbidden) != required_forbidden:
        raise SynthesisHoldoutError(
            "forbidden registry is not the exact reviewed/assessment family closure"
        )
    final_source = set(
        _string_list(
            registered_subject_ids.get("pedagogy_misconception"),
            field="subject_source_chunk_ids.pedagogy_misconception",
        )
    )
    identity["holdout_trust_chain_fingerprint"] = _hash_json(identity)
    return SynthesisHoldoutRegistry(
        path=registry_file,
        manifest_path=manifest_file,
        registry_fingerprint=str(registry.get("registry_fingerprint")),
        corpus_sha256=corpus_sha,
        objectives_sha256=objectives_sha,
        exclusion_sha256=exclusion_sha,
        forbidden_chunk_ids=frozenset(forbidden),
        assessment_chunk_ids=frozenset(assessment),
        final_source_chunk_ids=frozenset(final_source),
        family_chunk_ids=frozenset(family),
        trust_identity=identity,
    )


__all__ = [
    "ENV_ENABLED",
    "ENV_MANIFEST",
    "ENV_REGISTRY",
    "SynthesisHoldoutError",
    "SynthesisHoldoutRegistry",
    "holdout_exclusion_enabled",
    "load_synthesis_holdout_registry",
]
