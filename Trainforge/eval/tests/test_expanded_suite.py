from __future__ import annotations

import json
from pathlib import Path

from Trainforge.eval.expanded_suite import (
    EVAL_ARMS,
    _authoring_queue,
    _merge_authored_items,
    _reviewed_ood_items,
    arm_cases,
    build_suite,
    leakage_findings,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, count: int = 3) -> dict[str, Path]:
    objectives = {
        "terminal_outcomes": [],
        "component_objectives": [
            {"id": f"CO-{i:02}", "statement": f"Do skill {i}", "bloom_level": "apply"}
            for i in range(count)
        ],
    }
    questions = []
    answers = []
    chunks = []
    for i in range(count):
        qid, oid, cid = f"q-{i}", f"CO-{i:02}", f"chunk-{i}"
        questions.append(
            {
                "question_id": qid,
                "objective_id": oid,
                "stem": f"<p>Solve problem {i}?</p>",
                "bloom_level": "apply",
                "question_type": "short_answer",
                "source_chunks": [cid],
                "correct_answer": f"answer {i}",
            }
        )
        answers.append(
            {
                "item_id": qid,
                "correct_answers": [f"answer {i}"],
                "source_chunk_ids": [cid],
                "worked_solution_steps": [{"text": f"key point {i}", "cites": [cid]}],
            }
        )
        chunks.append(
            {
                "id": cid,
                "source": {
                    "item_path": f"unit-{i}.html",
                    "section_heading": f"Unit {i}",
                    "char_span": [0, 20],
                    "source_references": [{"sourceId": f"source:{i}"}],
                },
            }
        )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in chunks), encoding="utf-8"
    )
    return {
        "assessments": _write(
            tmp_path / "assessment_items.json",
            {"assessments": [{"questions": questions}]},
        ),
        "answer_key": _write(tmp_path / "answer_key.json", {"items": answers}),
        "objectives": _write(tmp_path / "objectives.json", objectives),
        "chunks": chunks_path,
    }


def test_objective_heldout_has_priority_and_deficits_never_duplicate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    suite = build_suite(
        assessments_path=paths["assessments"],
        answer_key_path=paths["answer_key"],
        objectives_path=paths["objectives"],
        chunks_path=paths["chunks"],
        targets={
            "checkpoint_dev": 2,
            "objective_heldout": 3,
            "grounding_stress": 1,
            "pedagogy_misconception": 1,
            "out_of_domain": 1,
        },
    )
    assert len(suite["splits"]["objective_heldout"]) == 3
    assert suite["deficits"]["checkpoint_dev"] == 2
    fingerprints = [
        row["fingerprint"] for rows in suite["splits"].values() for row in rows
    ]
    assert len(fingerprints) == len(set(fingerprints))
    assert suite["ready"] is False


def test_source_and_keypoint_leakage_are_detected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, count=1)
    suite = build_suite(
        assessments_path=paths["assessments"],
        answer_key_path=paths["answer_key"],
        objectives_path=paths["objectives"],
        chunks_path=paths["chunks"],
        targets={
            "checkpoint_dev": 0,
            "objective_heldout": 1,
            "grounding_stress": 0,
            "pedagogy_misconception": 0,
            "out_of_domain": 0,
        },
    )
    findings = leakage_findings(
        suite["splits"],
        [{"prompt": "unrelated", "completion": "x", "source_chunk_id": "chunk-0"}],
    )
    assert findings[0]["layer"] == "source_id"
    findings = leakage_findings(
        suite["splits"],
        [{"prompt": "please explain key point 0", "completion": "x"}],
    )
    assert any(row["layer"] == "keypoint_containment" for row in findings)
    findings = leakage_findings(
        suite["splits"],
        [{"prompt": "semantically related but lexically separate", "completion": "x"}],
        semantic_similarity=lambda evaluation, training: (
            0.95 if "Solve problem" in evaluation and "semantically" in training else 0.0
        ),
    )
    assert any(row["layer"] == "semantic" for row in findings)


def test_arm_projection_uses_identical_item_fingerprint(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, count=1)
    suite = build_suite(
        assessments_path=paths["assessments"],
        answer_key_path=paths["answer_key"],
        objectives_path=paths["objectives"],
        chunks_path=paths["chunks"],
        targets={
            "checkpoint_dev": 0,
            "objective_heldout": 1,
            "grounding_stress": 0,
            "pedagogy_misconception": 0,
            "out_of_domain": 0,
        },
    )
    cases = arm_cases(suite)
    assert {case["arm"] for case in cases} == set(EVAL_ARMS)
    assert len({case["item_fingerprint"] for case in cases}) == 1


def test_authored_items_require_review_and_fill_only_preassigned_split() -> None:
    splits = {
        "checkpoint_dev": [],
        "objective_heldout": [],
        "grounding_stress": [],
        "pedagogy_misconception": [],
        "out_of_domain": [],
    }
    base = {
        "question_id": "authored-1",
        "question": "A sufficiently independent question?",
        "answers": ["answer"],
        "expected_keypoints": ["point one", "point two"],
        "source_refs": [{"chunk_id": "chunk-1"}],
        "fingerprint": "fp-1",
        "split": "checkpoint_dev",
        "provenance": {"review_status": "QUALIFIED_AUTOMATIC_REVIEW"},
    }
    rejected = _merge_authored_items(
        splits,
        [base, {**base, "question_id": "authored-2", "fingerprint": "fp-2",
                "split": "objective_heldout"}],
        {"checkpoint_dev": 1, "objective_heldout": 1},
    )
    assert [row["question_id"] for row in splits["checkpoint_dev"]] == ["authored-1"]
    assert rejected == [
        {"question_id": "authored-2", "reason": "invalid_authored_split"}
    ]


def test_ood_items_require_operator_review_and_verified_abstention() -> None:
    document = {
        "course_slug": "foreign-course",
        "probes": [
            {
                "probe_id": "pending",
                "question_text": "Question outside this corpus?",
                "why_unanswerable": "The corpus has no evidence for it.",
                "dry_run": {"verified": True, "top_passage_answers": False},
                "authoring": {"reviewed_by": "PENDING_REVIEW"},
            },
            {
                "probe_id": "reviewed",
                "question_text": "Another question outside this corpus?",
                "why_unanswerable": "The corpus has no evidence for this topic.",
                "dry_run": {
                    "verified": True,
                    "top_passage_answers": False,
                    "top_chunk_id": "foreign-1",
                    "engine": "lexical",
                },
                "authoring": {"reviewed_by": "operator", "status": "reviewed"},
            },
        ],
    }
    rows, rejected = _reviewed_ood_items(document, 1)
    assert [row["question_id"] for row in rows] == ["reviewed"]
    assert rows[0]["retrieval"]["expected_behavior"] == "abstain_without_fabrication"
    assert rejected == [{"question_id": "pending", "reason": "not_operator_reviewed"}]


def test_authoring_queue_uses_dev_assessment_surface_and_rejects_aggregate() -> None:
    objectives = {
        "component_objectives": [
            {
                "id": "CO-01",
                "statement": "Solve linear equations by subtracting a constant.",
                "bloom_level": "apply",
            }
        ]
    }
    chunks = [
        {
            "id": "good",
            "chunk_type": "example",
            "text": "Key Idea: Solve linear equations by subtracting a constant.",
            "learning_outcome_refs": ["CO-01"],
        },
        {
            "id": "aggregate",
            "chunk_type": "explanation",
            "text": "A course overview.",
            "learning_outcome_refs": [f"CO-{index:02}" for index in range(1, 8)],
        },
        {
            "id": "assessment",
            "chunk_type": "assessment_item",
            "text": "Solve a linear equation by subtracting a constant.",
            "learning_outcome_refs": ["CO-01"],
        },
    ]
    queue = _authoring_queue(
        chunks,
        objectives,
        {"assessment"},
        {"good", "aggregate"},
        {
            "good": "family-good",
            "aggregate": "family-aggregate",
            "assessment": "family-assessment",
        },
        {
            "checkpoint_dev": 2,
            "grounding_stress": 0,
            "pedagogy_misconception": 0,
            "out_of_domain": 0,
        },
    )
    assert [row["source_chunk_id"] for row in queue["candidates"]] == ["assessment"]
    assert queue["course_candidate_deficit"] == 1
    assert queue["candidates"][0]["canonical_objective"].startswith("Solve linear")


def test_source_reuse_policy_keeps_dev_final_disjoint_and_caps_final_at_two() -> None:
    objectives = {
        "component_objectives": [
            {"id": "CO-01", "statement": "Add rational numbers.", "bloom_level": "apply"},
            {"id": "CO-02", "statement": "Subtract rational numbers.", "bloom_level": "apply"},
        ]
    }
    chunks = [
        {
            "id": "dev",
            "chunk_type": "assessment_item",
            "text": "Add rational numbers.",
            "learning_outcome_refs": ["CO-01"],
        },
        *[
            {
                "id": f"final-{index}",
                "chunk_type": "example",
                "text": "Key Idea: Add and subtract rational numbers.",
                "learning_outcome_refs": ["CO-01", "CO-02"],
            }
            for index in range(2)
        ],
    ]
    queue = _authoring_queue(
        chunks,
        objectives,
        {"dev"},
        {"final-0", "final-1"},
        {"dev": "dev-family", "final-0": "f0", "final-1": "f1"},
        {
            "checkpoint_dev": 1,
            "grounding_stress": 2,
            "pedagogy_misconception": 2,
            "out_of_domain": 0,
        },
    )
    rows = queue["candidates"]
    by_family: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row["source_family_id"]), []).append(row)
    assert {row["source_family_id"] for row in rows if row["target_split"] == "checkpoint_dev"} == {
        "dev-family"
    }
    assert all(len(group) <= 2 for family, group in by_family.items() if family != "dev-family")
    assert all(
        len({row["canonical_objective_id"] for row in group}) == len(group)
        for family, group in by_family.items()
        if family != "dev-family"
    )
