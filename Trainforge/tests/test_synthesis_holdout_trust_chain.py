"""End-to-end trust-chain tests for synthesis holdout preflight.

The fixtures deliberately live entirely under ``tmp_path``.  They model the
portable final-contract shape without depending on a course slug, repository
location, or a previously generated evaluation artifact.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from Trainforge.eval.expanded_suite import _source_family_closure
from Trainforge.eval.qualification.manual_review import evaluate_manual_review_gate
from Trainforge.synthesis_holdout import (
    ENV_ENABLED,
    ENV_MANIFEST,
    ENV_REGISTRY,
    SynthesisHoldoutError,
    load_synthesis_holdout_registry,
)
from Trainforge.synthesize_training import run_synthesis

SPLITS = ("checkpoint_dev", "grounding_stress", "pedagogy_misconception")


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _seal(document: dict[str, Any], field: str) -> None:
    document.pop(field, None)
    document[field] = _canonical_sha(document)


def _normalize_content(value: Any) -> str:
    plain = html.unescape(re.sub(r"<[^>]+>", " ", str(value or ""))).lower()
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def _content_fingerprint(rows: list[dict[str, Any]]) -> str:
    content = []
    for row in rows:
        expected = row.get("expected")
        expected_answer = row.get("expected_answer")
        answer = row.get("answers")
        if not answer and isinstance(expected, dict):
            answer = expected.get("answer")
        if not answer and isinstance(expected_answer, dict):
            answer = expected_answer.get("keyed_correct_results")
        objective = row.get("objective") or row.get("canonical_objective")
        if not objective:
            objective = {"id": row.get("canonical_objective_id")}
        content.append({
            "prompt": _normalize_content(
                row.get("question") or row.get("prompt")
            ),
            "answer": _normalize_content(answer),
            "objective_id": objective.get("id"),
            "source_chunk_id": row.get("source_chunk_id"),
        })
    return _canonical_sha(content)


def _review(
    subject_path: Path, *, reviewer: str, role: str = "independent"
) -> dict[str, Any]:
    subject = json.loads(subject_path.read_text(encoding="utf-8"))
    checks = {
        "assignment_binding": True,
        "answer_key_binding": True,
        "proof_integrity": True,
        "task_answer_consistency": True,
        "split_specific_quality": True,
        "objective_alignment": True,
        "bloom_alignment": True,
        "citation_integrity": True,
        "citation_semantic_support": True,
        "split_fidelity": True,
        "uniqueness": True,
        "split_separation": True,
    }
    return {
        "schema_version": "manual-eval-review-v1",
        "review_type": "independent_item_level_manual_review",
        "reviewer": {"label": reviewer, "role": role},
        "independence_attestation": {
            "did_not_author_subject_items": True,
            "did_not_copy_prior_verdicts": True,
            "reviewed_all_items_individually": True,
        },
        "subject": {"sha256": _file_sha(subject_path), "item_count": 1},
        "method": {
            "judgment_method": "manual_item_level_semantic_judgment",
            "limitations": [],
        },
        "items": [{
            "item_id": subject["item_id"],
            "item_fingerprint": subject["fingerprint"],
            "verdict": "approve",
            "reasons": [],
            "checks": checks,
            "evidence": {
                "assignment": {
                    "source_assignment_id": "assignment",
                    "source_family_id": "family",
                },
                "answer_key": {"keyed_results_sha256": "a" * 64},
                "proof_replay": {
                    "proof_sha256": "b" * 64,
                    "replay_passed": True,
                },
                "semantic_judgment": {
                    "task_semantic_rationale": (
                        "The task has a coherent and reviewable semantic target."
                    ),
                    "task_answer_rationale": (
                        "The worked response reaches the keyed answer exactly."
                    ),
                    "split_specific_rationale": (
                        "The item satisfies its declared split behavior fully."
                    ),
                    "objective_rationale": (
                        "The question directly elicits the declared objective."
                    ),
                    "bloom_rationale": (
                        "The learner must diagnose and correct the reasoning."
                    ),
                    "source_support_rationale": (
                        "The cited source explicitly supports the correction."
                    ),
                },
                "citations": [{
                    "chunk_id": subject["source_chunk_id"],
                    "quote_sha256": "c" * 64,
                    "exact_span_replayed": True,
                    "semantic_support": True,
                    "semantic_support_rationale": (
                        "The quote directly supports the expected result."
                    ),
                }],
                "separation": {
                    "dev_overlap": False,
                    "duplicate_item": False,
                    "family_policy_satisfied": True,
                },
            },
        }],
        "aggregate": {
            "verdict_counts": {"approve": 1, "reject": 0, "escalate": 0},
            "defect_code_counts": {},
        },
    }


def _gate(
    subject_path: Path,
    review_paths: list[Path],
    *,
    passed: bool = True,
) -> dict[str, Any]:
    subject_sha = _file_sha(subject_path)
    result = {
        "passed": passed,
        "subject_sha256": subject_sha,
        "expected_item_count": 1,
        "reviewer_count": len(review_paths),
        "final_verdict_counts": {"approve": 1},
        "disagreements": [],
        "defects": [] if passed else ["review rejected"],
    }
    gate = {
        "schema_version": "manual-review-combined-gate-result-v1",
        "deterministic": True,
        "inputs": {
            "expected_item_count": 1,
            "required_reviewers": len(review_paths),
            "subject": {"path": str(subject_path), "sha256": subject_sha},
            "reviews": [
                {
                    "label": f"review-{index}",
                    "path": str(path),
                    "sha256": _file_sha(path),
                }
                for index, path in enumerate(review_paths, start=1)
            ],
        },
        "gate": {"completed": True, "error": None, "result": result},
    }
    _seal(gate, "fingerprint")
    return gate


@pytest.fixture
def contract(tmp_path: Path) -> dict[str, Any]:
    corpus = tmp_path / "upstream" / "chunks.jsonl"
    objectives = tmp_path / "upstream" / "objectives.json"
    rows = [
        {"id": "chunk-assessment", "chunk_type": "assessment_item", "text": "A"},
        {"id": "chunk-dev", "chunk_type": "content", "text": "D"},
        {"id": "chunk-ground", "chunk_type": "content", "text": "G"},
        {"id": "chunk-pedagogy", "chunk_type": "content", "text": "P"},
    ]
    _write_jsonl(corpus, rows)
    _write_json(objectives, {"objectives": [{"id": "objective-1"}]})

    subjects: dict[str, Path] = {}
    gates: dict[str, Path] = {}
    review_paths: dict[str, list[Path]] = {}
    split_fingerprints: dict[str, str] = {}
    content_fingerprints: dict[str, str] = {}
    source_ids = {
        "checkpoint_dev": "chunk-dev",
        "grounding_stress": "chunk-ground",
        "pedagogy_misconception": "chunk-pedagogy",
    }
    for split in SPLITS:
        subject_path = tmp_path / "subjects" / f"{split}.jsonl"
        item_fingerprint = _canonical_sha(
            {"split": split, "source_chunk_id": source_ids[split]}
        )
        _write_jsonl(subject_path, [{
            "item_id": f"{split}-item-1",
            "fingerprint": item_fingerprint,
            "split": split,
            "source_chunk_id": source_ids[split],
            "prompt": f"Prompt for {split}",
            "expected": f"Expected for {split}",
        }])
        subjects[split] = subject_path
        split_fingerprints[split] = _canonical_sha([item_fingerprint])
        subject_rows = [{
            "item_id": f"{split}-item-1",
            "fingerprint": item_fingerprint,
            "split": split,
            "source_chunk_id": source_ids[split],
            "prompt": f"Prompt for {split}",
            "expected": f"Expected for {split}",
        }]
        # This binds parsed ordered row content, not JSONL formatting.
        content_fingerprints[split] = _content_fingerprint(subject_rows)
        split_reviews = []
        for index in (1, 2):
            review_path = tmp_path / "reviews" / f"{split}-{index}.json"
            _write_json(
                review_path, _review(subject_path, reviewer=f"reviewer-{index}")
            )
            split_reviews.append(review_path)
        review_paths[split] = split_reviews
        gate_path = tmp_path / "gates" / f"{split}.json"
        _write_json(gate_path, _gate(subject_path, split_reviews))
        gates[split] = gate_path

    forbidden = sorted(["chunk-assessment", *source_ids.values()])
    _, derived_family_ids, family_audit = _source_family_closure(
        rows, set(source_ids.values())
    )
    exclusion = {
        "schema_version": "expanded-suite-training-exclusion-v1",
        "candidate": False,
        "ready": True,
        "ready_blockers": [],
        "assessment_chunk_ids": ["chunk-assessment"],
        "subject_source_chunk_ids": {
            split: [source_ids[split]] for split in SPLITS
        },
        "derived_source_family_chunk_ids": derived_family_ids,
        "family_closed_source_chunk_ids": sorted(source_ids.values()),
        "family_policy": {
            "contract": "cf_block_id_or_contiguous_follows_v1",
            "status": "approved",
            "source_references_used": False,
        },
        "family_contract_audit": family_audit,
        "chunk_ids": forbidden,
        "chunk_ids_sha256": _canonical_sha(forbidden),
        "source_hashes": {
            "corpus_chunks": _file_sha(corpus),
            "objectives": _file_sha(objectives),
        },
        "split_fingerprints": split_fingerprints,
        "split_content_fingerprints": content_fingerprints,
        "readiness_checks": {
            "all_subject_sources_forbidden": True,
            "all_family_closed_sources_forbidden": True,
            "assessment_surface_reserved": True,
        },
    }
    _seal(exclusion, "registry_fingerprint")
    registry = tmp_path / "contract" / "training_input_exclusion.json"
    _write_json(registry, exclusion)

    slots = {}
    subject_entries = {}
    for split in SPLITS:
        slots[split] = {
            "status": "approved_combined_manual_review_gate",
            "semantic_approval": True,
            "available_reviews": [
                {"path": str(path), "sha256": _file_sha(path)}
                for path in review_paths[split]
            ],
            "combined_gate": {
                "path": str(gates[split]),
                "sha256": _file_sha(gates[split]),
                "subject_sha256": _file_sha(subjects[split]),
                "passed": True,
                "defects": [],
                "disagreements": [],
            },
        }
        subject_entries[split] = {
            "path": str(subjects[split]),
            "sha256": _file_sha(subjects[split]),
            "item_count": 1,
            "split_fingerprint": split_fingerprints[split],
            "content_fingerprint": content_fingerprints[split],
        }
    manifest = {
        "schema_version": "expanded-suite-final-contract-v1",
        "candidate": False,
        "ready": True,
        "ready_blockers": [],
        "total_item_count": 3,
        "source_hashes": {
            "corpus_chunks": _file_sha(corpus),
            "objectives": _file_sha(objectives),
        },
        "subjects": subject_entries,
        "review_artifact_slots": slots,
        "semantic_approval_claims": {split: True for split in SPLITS},
        "training_input_exclusion": {
            "path": str(registry),
            "ready": True,
            "chunk_count": len(forbidden),
            "chunk_ids_sha256": exclusion["chunk_ids_sha256"],
            "registry_fingerprint": exclusion["registry_fingerprint"],
            "sha256": _file_sha(registry),
        },
        "readiness_checks": {
            "subject_counts_exact": True,
            "subject_hashes_exact": True,
            **{f"{split}_combined_gate_passed": True for split in SPLITS},
        },
    }
    _seal(manifest, "manifest_fingerprint")
    manifest_path = tmp_path / "contract" / "expanded_suite_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "corpus": corpus,
        "objectives": objectives,
        "registry": registry,
        "manifest": manifest_path,
        "subjects": subjects,
        "gates": gates,
        "reviews": review_paths,
    }


def _load(contract: dict[str, Any], **overrides: Any):
    arguments = {
        "chunks_path": contract["corpus"],
        "objectives_path": contract["objectives"],
        "registry_path": contract["registry"],
        "manifest_path": contract["manifest"],
        "environment": {ENV_ENABLED: "true"},
    }
    arguments.update(overrides)
    return load_synthesis_holdout_registry(**arguments)


def _mutate_json(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(path, value)


def _reseal_gate_and_manifest(
    contract: dict[str, Any], split: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    gate_path = contract["gates"][split]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    mutation(gate)
    _seal(gate, "fingerprint")
    _write_json(gate_path, gate)
    manifest = json.loads(contract["manifest"].read_text(encoding="utf-8"))
    manifest["review_artifact_slots"][split]["combined_gate"]["sha256"] = (
        _file_sha(gate_path)
    )
    _seal(manifest, "manifest_fingerprint")
    _write_json(contract["manifest"], manifest)


def _coherently_omit_forbidden(
    contract: dict[str, Any], *, split: str | None, chunk_id: str
) -> None:
    registry = json.loads(contract["registry"].read_text(encoding="utf-8"))
    registry["chunk_ids"].remove(chunk_id)
    if chunk_id in registry["family_closed_source_chunk_ids"]:
        registry["family_closed_source_chunk_ids"].remove(chunk_id)
    if split is None:
        registry["assessment_chunk_ids"].remove(chunk_id)
    else:
        registry["subject_source_chunk_ids"][split].remove(chunk_id)
    registry["chunk_ids_sha256"] = _canonical_sha(registry["chunk_ids"])
    _seal(registry, "registry_fingerprint")
    _write_json(contract["registry"], registry)
    manifest = json.loads(contract["manifest"].read_text(encoding="utf-8"))
    exclusion = manifest["training_input_exclusion"]
    exclusion["chunk_count"] = len(registry["chunk_ids"])
    exclusion["chunk_ids_sha256"] = registry["chunk_ids_sha256"]
    exclusion["registry_fingerprint"] = registry["registry_fingerprint"]
    exclusion["sha256"] = _file_sha(contract["registry"])
    _seal(manifest, "manifest_fingerprint")
    _write_json(contract["manifest"], manifest)


def _coherent_negative_reviews(
    contract: dict[str, Any], *, disagree: bool,
) -> None:
    split = "checkpoint_dev"
    selected = contract["reviews"][split][:1] if disagree else contract["reviews"][split]
    for path in selected:
        review = json.loads(path.read_text(encoding="utf-8"))
        review["items"][0]["verdict"] = "reject"
        review["items"][0]["reasons"] = [{
            "code": "SEMANTIC_DEFECT",
            "severity": "major",
            "detail": "The independent semantic review found a substantive defect.",
        }]
        review["aggregate"]["verdict_counts"] = {
            "approve": 0, "reject": 1, "escalate": 0,
        }
        review["aggregate"]["defect_code_counts"] = {"SEMANTIC_DEFECT": 1}
        _write_json(path, review)
    review_paths = contract["reviews"][split]
    recomputed = evaluate_manual_review_gate(
        contract["subjects"][split],
        review_paths,
        expected_item_count=1,
        required_reviewers=2,
    )
    gate = _gate(contract["subjects"][split], review_paths)
    gate["gate"]["result"] = recomputed.as_dict()
    _seal(gate, "fingerprint")
    _write_json(contract["gates"][split], gate)
    manifest = json.loads(contract["manifest"].read_text(encoding="utf-8"))
    slot = manifest["review_artifact_slots"][split]
    slot["available_reviews"] = [
        {"path": str(path), "sha256": _file_sha(path)}
        for path in review_paths
    ]
    slot["combined_gate"]["sha256"] = _file_sha(contract["gates"][split])
    _seal(manifest, "manifest_fingerprint")
    _write_json(contract["manifest"], manifest)


def _install_resolved_adjudication(contract: dict[str, Any]) -> None:
    split = "checkpoint_dev"
    dissent = contract["reviews"][split][1]
    review = json.loads(dissent.read_text(encoding="utf-8"))
    review["items"][0]["verdict"] = "reject"
    review["items"][0]["reasons"] = [{
        "code": "SEMANTIC_DEFECT",
        "severity": "major",
        "detail": "One independent reviewer found a substantive defect.",
    }]
    review["aggregate"]["verdict_counts"] = {
        "approve": 0, "reject": 1, "escalate": 0,
    }
    review["aggregate"]["defect_code_counts"] = {"SEMANTIC_DEFECT": 1}
    _write_json(dissent, review)
    third = contract["manifest"].parent / "adjudicator-review.json"
    _write_json(
        third,
        _review(
            contract["subjects"][split],
            reviewer="reviewer-adjudicator",
            role="adjudicator",
        ),
    )
    reviews = [*contract["reviews"][split], third]
    recomputed = evaluate_manual_review_gate(
        contract["subjects"][split],
        reviews,
        expected_item_count=1,
        required_reviewers=3,
    )
    assert recomputed.passed and recomputed.disagreements
    gate = _gate(contract["subjects"][split], reviews)
    gate["gate"]["result"] = recomputed.as_dict()
    _seal(gate, "fingerprint")
    _write_json(contract["gates"][split], gate)
    manifest = json.loads(contract["manifest"].read_text(encoding="utf-8"))
    slot = manifest["review_artifact_slots"][split]
    slot["available_reviews"] = [
        {"path": str(path), "sha256": _file_sha(path)} for path in reviews
    ]
    slot["combined_gate"]["sha256"] = _file_sha(contract["gates"][split])
    _seal(manifest, "manifest_fingerprint")
    _write_json(contract["manifest"], manifest)


def test_valid_final_contract_proves_complete_chain(contract: dict[str, Any]) -> None:
    registry = _load(contract)
    assert registry is not None
    assert registry.forbidden_chunk_ids == {
        "chunk-assessment", "chunk-dev", "chunk-ground", "chunk-pedagogy",
    }
    assert registry.reserves({"id": "chunk-ground"})
    assert not registry.reserves({"id": "unreserved"})


def test_flag_off_returns_none_without_reading_contract(
    contract: dict[str, Any],
) -> None:
    assert load_synthesis_holdout_registry(
        chunks_path=contract["corpus"],
        objectives_path=contract["objectives"],
        registry_path=contract["registry"].with_name("missing.json"),
        manifest_path=contract["manifest"].with_name("missing.json"),
        environment={ENV_ENABLED: "false"},
    ) is None


def test_flag_off_preserves_legacy_order_without_reading_malformed_inputs(
    tmp_path: Path,
) -> None:
    assert load_synthesis_holdout_registry(
        chunks_path=tmp_path / "missing-and-malformed-chunks.jsonl",
        objectives_path=tmp_path / "missing-objectives.json",
        registry_path=tmp_path / "missing-registry.json",
        manifest_path=tmp_path / "missing-manifest.json",
        environment={ENV_ENABLED: ""},
    ) is None


def test_run_synthesis_flag_off_preserves_legacy_malformed_corpus_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "course"
    chunks = corpus / "imscc_chunks" / "chunks.jsonl"
    chunks.parent.mkdir(parents=True)
    chunks.write_text("{malformed-json\n", encoding="utf-8")
    (corpus / "objectives.json").write_text("{also-malformed", encoding="utf-8")
    output = tmp_path / "legacy-output"
    monkeypatch.setenv(ENV_ENABLED, "false")
    monkeypatch.setenv(ENV_REGISTRY, str(tmp_path / "must-not-read-registry"))
    monkeypatch.setenv(ENV_MANIFEST, str(tmp_path / "must-not-read-manifest"))
    with pytest.raises(json.JSONDecodeError):
        run_synthesis(corpus, "COURSE", provider="mock", output_dir=output)
    assert output.is_dir()


def test_run_synthesis_flag_on_rejects_malformed_corpus_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "course"
    chunks = corpus / "imscc_chunks" / "chunks.jsonl"
    chunks.parent.mkdir(parents=True)
    chunks.write_text("{malformed-json\n", encoding="utf-8")
    output = tmp_path / "guarded-output"
    monkeypatch.setenv(ENV_ENABLED, "true")
    monkeypatch.setenv(ENV_REGISTRY, str(tmp_path / "registry.json"))
    monkeypatch.setenv(ENV_MANIFEST, str(tmp_path / "manifest.json"))
    with pytest.raises(json.JSONDecodeError):
        run_synthesis(corpus, "COURSE", provider="mock", output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("missing", ["manifest", "registry"])
def test_missing_contract_artifact_fails_closed(
    contract: dict[str, Any], missing: str,
) -> None:
    contract[missing].unlink()
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


@pytest.mark.parametrize(
    ("kind", "mutation"),
    [
        ("rejected", lambda gate: gate["gate"]["result"].update(passed=False)),
        (
            "disagreed",
            lambda gate: gate["gate"]["result"]["disagreements"].append("item-1"),
        ),
        (
            "defective",
            lambda gate: gate["gate"]["result"]["defects"].append("bad evidence"),
        ),
    ],
)
def test_rejected_disagreed_or_defective_gate_fails_closed(
    contract: dict[str, Any],
    kind: str,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    del kind
    _reseal_gate_and_manifest(contract, "checkpoint_dev", mutation)
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


def test_unadjudicated_review_slot_fails_closed(contract: dict[str, Any]) -> None:
    _mutate_json(
        contract["manifest"],
        lambda manifest: manifest["review_artifact_slots"][
            "grounding_stress"
        ].update(status="awaiting_adjudication", semantic_approval=False),
    )
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


@pytest.mark.parametrize("remove_index", [0, 1])
def test_missing_independent_review_fails_closed(
    contract: dict[str, Any], remove_index: int,
) -> None:
    contract["reviews"]["pedagogy_misconception"][remove_index].unlink()
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


def test_split_fingerprint_mismatch_fails_closed(
    contract: dict[str, Any],
) -> None:
    _mutate_json(
        contract["manifest"],
        lambda manifest: manifest["subjects"]["checkpoint_dev"].update(
            {"split_fingerprint": "0" * 64}
        ),
    )
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


def test_subject_bytes_mutation_fails_closed(contract: dict[str, Any]) -> None:
    with contract["subjects"]["grounding_stress"].open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


@pytest.mark.parametrize("target", ["corpus", "objectives", "registry"])
def test_every_pinned_input_mutation_fails_closed(
    contract: dict[str, Any], target: str,
) -> None:
    with contract[target].open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


def test_manifest_readiness_mutation_fails_closed(contract: dict[str, Any]) -> None:
    _mutate_json(
        contract["manifest"],
        lambda manifest: manifest.update(
            ready=False, ready_blockers=["contract changed after review"]
        ),
    )
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


def test_environment_paths_are_supported(contract: dict[str, Any]) -> None:
    registry = load_synthesis_holdout_registry(
        chunks_path=contract["corpus"],
        objectives_path=contract["objectives"],
        environment={
            ENV_ENABLED: "true",
            ENV_REGISTRY: str(contract["registry"]),
            ENV_MANIFEST: str(contract["manifest"]),
        },
    )
    assert registry is not None


@pytest.mark.parametrize(
    ("split", "chunk_id"),
    [
        ("checkpoint_dev", "chunk-dev"),
        ("grounding_stress", "chunk-ground"),
        ("pedagogy_misconception", "chunk-pedagogy"),
        (None, "chunk-assessment"),
    ],
)
def test_coherent_reseal_cannot_omit_any_required_forbidden_class(
    contract: dict[str, Any], split: str | None, chunk_id: str,
) -> None:
    _coherently_omit_forbidden(contract, split=split, chunk_id=chunk_id)
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


@pytest.mark.parametrize("target", ["manifest", "registry", "gate"])
def test_arbitrary_internal_fingerprint_is_rejected(
    contract: dict[str, Any], target: str,
) -> None:
    if target == "manifest":
        _mutate_json(
            contract["manifest"],
            lambda value: value.update(manifest_fingerprint="f" * 64),
        )
    elif target == "registry":
        _mutate_json(
            contract["registry"],
            lambda value: value.update(registry_fingerprint="f" * 64),
        )
    else:
        _mutate_json(
            contract["gates"]["checkpoint_dev"],
            lambda value: value.update(fingerprint="f" * 64),
        )
    with pytest.raises(SynthesisHoldoutError):
        _load(contract)


def test_coherently_forged_split_content_fingerprint_is_rejected(
    contract: dict[str, Any],
) -> None:
    forged = "f" * 64
    registry = json.loads(contract["registry"].read_text(encoding="utf-8"))
    registry["split_content_fingerprints"]["checkpoint_dev"] = forged
    _seal(registry, "registry_fingerprint")
    _write_json(contract["registry"], registry)
    manifest = json.loads(contract["manifest"].read_text(encoding="utf-8"))
    manifest["subjects"]["checkpoint_dev"]["content_fingerprint"] = forged
    exclusion = manifest["training_input_exclusion"]
    exclusion["registry_fingerprint"] = registry["registry_fingerprint"]
    exclusion["sha256"] = _file_sha(contract["registry"])
    _seal(manifest, "manifest_fingerprint")
    _write_json(contract["manifest"], manifest)
    with pytest.raises(SynthesisHoldoutError, match="content fingerprint"):
        _load(contract)


def test_divergent_reviews_resolved_by_adjudicator_are_supported(
    contract: dict[str, Any],
) -> None:
    _install_resolved_adjudication(contract)
    assert _load(contract) is not None


def test_stored_adjudication_evidence_mismatch_fails_closed(
    contract: dict[str, Any],
) -> None:
    _install_resolved_adjudication(contract)
    _reseal_gate_and_manifest(
        contract,
        "checkpoint_dev",
        lambda gate: gate["gate"]["result"].update(disagreements=[]),
    )
    with pytest.raises(SynthesisHoldoutError, match="stored gate result is stale"):
        _load(contract)


@pytest.mark.parametrize("disagree", [False, True])
def test_coherently_resealed_negative_reviews_fail_after_recomputation(
    contract: dict[str, Any], disagree: bool,
) -> None:
    _coherent_negative_reviews(contract, disagree=disagree)
    with pytest.raises(
        SynthesisHoldoutError,
        match="rejected or disagreed|did not pass cleanly",
    ):
        _load(contract)
