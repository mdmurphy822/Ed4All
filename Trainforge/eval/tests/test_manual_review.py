from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from Trainforge.eval.manual_review import (
    ManualReviewError,
    evaluate_manual_review_gate,
)


CHECKS = {
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


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _subject(tmp_path: Path) -> tuple[Path, list[dict]]:
    rows = [
        {"item_id": f"item-{index}", "fingerprint": f"{index:064x}"}
        for index in range(1, 4)
    ]
    path = tmp_path / "subject.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows


def _evidence() -> dict:
    return {
        "assignment": {
            "source_assignment_id": "assignment",
            "source_family_id": "family",
        },
        "answer_key": {"keyed_results_sha256": "a" * 64},
        "proof_replay": {"proof_sha256": "b" * 64, "replay_passed": True},
        "semantic_judgment": {
            "task_semantic_rationale": "The task has a coherent and reviewable semantic target.",
            "task_answer_rationale": "The worked response reaches the keyed answer exactly.",
            "split_specific_rationale": "The item satisfies its declared split behavior fully.",
            "objective_rationale": "The question directly elicits the declared objective.",
            "bloom_rationale": "The learner must diagnose and correct the reasoning.",
            "source_support_rationale": "The cited source explicitly supports the correction.",
        },
        "citations": [{
            "chunk_id": "chunk",
            "quote_sha256": "c" * 64,
            "exact_span_replayed": True,
            "semantic_support": True,
            "semantic_support_rationale": "The quote directly supports the expected result.",
        }],
        "separation": {
            "dev_overlap": False,
            "duplicate_item": False,
            "family_policy_satisfied": True,
        },
    }


def _review(subject: Path, rows: list[dict], label: str, role: str = "independent") -> dict:
    items = [
        {
            "item_id": row["item_id"],
            "item_fingerprint": row["fingerprint"],
            "verdict": "approve",
            "reasons": [],
            "checks": dict(CHECKS),
            "evidence": _evidence(),
        }
        for row in rows
    ]
    return {
        "schema_version": "manual-eval-review-v1",
        "review_type": "independent_item_level_manual_review",
        "reviewer": {"label": label, "role": role},
        "independence_attestation": {
            "did_not_author_subject_items": True,
            "did_not_copy_prior_verdicts": True,
            "reviewed_all_items_individually": True,
        },
        "subject": {
            "sha256": _sha(subject.read_bytes()),
            "item_count": len(rows),
        },
        "method": {
            "judgment_method": "manual_item_level_semantic_judgment",
            "limitations": [],
        },
        "aggregate": {
            "verdict_counts": {
                "approve": len(rows), "reject": 0, "escalate": 0,
            },
            "defect_code_counts": {},
        },
        "items": items,
    }


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def _reject(review: dict, item_index: int = 0) -> None:
    row = review["items"][item_index]
    row["verdict"] = "reject"
    row["checks"]["objective_alignment"] = False
    row["reasons"] = [{
        "code": "OBJECTIVE_MISMATCH",
        "severity": "critical",
        "detail": "The task does not elicit the declared objective behavior.",
    }]
    verdicts = Counter(item["verdict"] for item in review["items"])
    review["aggregate"]["verdict_counts"] = {
        name: verdicts.get(name, 0) for name in ("approve", "reject", "escalate")
    }
    review["aggregate"]["defect_code_counts"] = {"OBJECTIVE_MISMATCH": 1}


def test_two_complete_independent_approvals_pass(tmp_path: Path) -> None:
    subject, rows = _subject(tmp_path)
    paths = [
        _write(tmp_path / f"review-{n}.json", _review(subject, rows, f"reviewer-{n}"))
        for n in (1, 2)
    ]
    gate = evaluate_manual_review_gate(
        subject, paths, expected_item_count=3,
    )
    assert gate.passed
    assert gate.final_verdict_counts == {"approve": 3}


@pytest.mark.parametrize("mutation", ["sha", "fingerprint", "coverage", "attestation"])
def test_binding_and_independence_fail_closed(tmp_path: Path, mutation: str) -> None:
    subject, rows = _subject(tmp_path)
    review = _review(subject, rows, "reviewer")
    if mutation == "sha":
        review["subject"]["sha256"] = "0" * 64
    elif mutation == "fingerprint":
        review["items"][0]["item_fingerprint"] = "f" * 64
    elif mutation == "coverage":
        review["items"].pop()
        review["aggregate"]["verdict_counts"]["approve"] -= 1
    else:
        review["independence_attestation"]["did_not_author_subject_items"] = False
    path = _write(tmp_path / "review.json", review)
    with pytest.raises(ManualReviewError):
        evaluate_manual_review_gate(subject, [path])


def test_approve_cannot_hide_failed_check_or_reason(tmp_path: Path) -> None:
    subject, rows = _subject(tmp_path)
    review = _review(subject, rows, "reviewer")
    review["items"][0]["checks"]["citation_semantic_support"] = False
    path = _write(tmp_path / "review.json", review)
    with pytest.raises(ManualReviewError, match="approves"):
        evaluate_manual_review_gate(subject, [path])


def test_reject_requires_defect_code_and_replayed_aggregates(tmp_path: Path) -> None:
    subject, rows = _subject(tmp_path)
    review = _review(subject, rows, "reviewer")
    review["items"][0]["verdict"] = "reject"
    path = _write(tmp_path / "review.json", review)
    with pytest.raises(ManualReviewError, match="defect codes"):
        evaluate_manual_review_gate(subject, [path])


def test_disagreement_requires_third_adjudicator(tmp_path: Path) -> None:
    subject, rows = _subject(tmp_path)
    first = _review(subject, rows, "first", "primary")
    second = _review(subject, rows, "second", "independent")
    _reject(second)
    first_path = _write(tmp_path / "first.json", first)
    second_path = _write(tmp_path / "second.json", second)
    gate = evaluate_manual_review_gate(subject, [first_path, second_path])
    assert not gate.passed
    assert "third_review_required_for_disagreement" in gate.defects

    third = _review(subject, rows, "third", "adjudicator")
    third_path = _write(tmp_path / "third.json", third)
    gate = evaluate_manual_review_gate(
        subject, [first_path, second_path, third_path],
    )
    assert gate.passed
    assert gate.disagreements == ("item-1",)


def test_majority_rejection_keeps_gate_closed(tmp_path: Path) -> None:
    subject, rows = _subject(tmp_path)
    reviews = [
        _review(subject, rows, "first", "primary"),
        _review(subject, rows, "second", "independent"),
        _review(subject, rows, "third", "adjudicator"),
    ]
    _reject(reviews[1])
    _reject(reviews[2])
    paths = [
        _write(tmp_path / f"{index}.json", review)
        for index, review in enumerate(reviews)
    ]
    gate = evaluate_manual_review_gate(subject, paths)
    assert not gate.passed
    assert gate.final_verdict_counts["reject"] == 1
    assert "non_approved_final_verdicts" in gate.defects


def test_aggregate_counts_are_recomputed_not_trusted(tmp_path: Path) -> None:
    subject, rows = _subject(tmp_path)
    review = _review(subject, rows, "reviewer")
    review["aggregate"]["verdict_counts"]["approve"] = 99
    path = _write(tmp_path / "review.json", review)
    with pytest.raises(ManualReviewError, match="aggregate"):
        evaluate_manual_review_gate(subject, [path])
