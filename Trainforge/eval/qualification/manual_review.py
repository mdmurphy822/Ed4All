"""Fail-closed enforcement for independent manual evaluation reviews.

This module validates bindings and review evidence.  It deliberately does not
make semantic judgments: objective fit, Bloom fit, misconception plausibility,
correction quality, and source support are explicit reviewer-supplied fields.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

SCHEMA_PATH = (
    Path(__file__).parents[1] / "schemas" / "manual_eval_review.schema.json"
)


class ManualReviewError(ValueError):
    """A review artifact cannot safely participate in the gate."""


@dataclass(frozen=True)
class ManualReviewGate:
    passed: bool
    subject_sha256: str
    expected_item_count: int
    reviewer_count: int
    final_verdict_counts: Mapping[str, int]
    disagreements: tuple[str, ...]
    defects: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "subject_sha256": self.subject_sha256,
            "expected_item_count": self.expected_item_count,
            "reviewer_count": self.reviewer_count,
            "final_verdict_counts": dict(self.final_verdict_counts),
            "disagreements": list(self.disagreements),
            "defects": list(self.defects),
        }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _subject_items(payload: bytes) -> dict[str, str]:
    items: dict[str, str] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManualReviewError(
                f"subject contains invalid JSON on line {line_number}"
            ) from exc
        if not isinstance(item, Mapping):
            raise ManualReviewError(
                f"subject item on line {line_number} must be an object"
            )
        item_id = str(item.get("item_id") or "")
        fingerprint = str(item.get("fingerprint") or "")
        if not item_id or len(fingerprint) != 64:
            raise ManualReviewError(
                f"subject item on line {line_number} lacks item_id/fingerprint"
            )
        if item_id in items:
            raise ManualReviewError(f"subject contains duplicate item_id {item_id!r}")
        items[item_id] = fingerprint
    if not items:
        raise ManualReviewError("subject contains no reviewable items")
    return items


def _load_review(path: Path, schema: Mapping[str, Any]) -> dict[str, Any]:
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManualReviewError(f"invalid review artifact at {path}: {exc}") from exc
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(review),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.absolute_path)) or "<root>"
        raise ManualReviewError(
            f"review artifact {path} violates schema at {location}: {first.message}"
        )
    return review


def _validate_review(
    review: Mapping[str, Any],
    *,
    path: Path,
    subject_sha256: str,
    subject_items: Mapping[str, str],
) -> dict[str, str]:
    if review["subject"]["sha256"] != subject_sha256:
        raise ManualReviewError(f"review artifact {path} subject SHA mismatch")
    if review["subject"]["item_count"] != len(subject_items):
        raise ManualReviewError(f"review artifact {path} subject count mismatch")
    rows = review["items"]
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = row["item_id"]
        if item_id in by_id:
            raise ManualReviewError(
                f"review artifact {path} repeats verdict for {item_id!r}"
            )
        by_id[item_id] = row
    if set(by_id) != set(subject_items):
        missing = sorted(set(subject_items) - set(by_id))
        extra = sorted(set(by_id) - set(subject_items))
        raise ManualReviewError(
            f"review artifact {path} item coverage mismatch; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )

    verdicts: Counter[str] = Counter()
    defect_codes: Counter[str] = Counter()
    for item_id, row in by_id.items():
        if row["item_fingerprint"] != subject_items[item_id]:
            raise ManualReviewError(
                f"review artifact {path} fingerprint mismatch for {item_id!r}"
            )
        verdict = row["verdict"]
        checks = row["checks"]
        reasons = row["reasons"]
        all_checks_pass = all(checks.values())
        if verdict == "approve" and (not all_checks_pass or reasons):
            raise ManualReviewError(
                f"review artifact {path} approves {item_id!r} despite failed "
                "checks or defect reasons"
            )
        if verdict != "approve" and not reasons:
            raise ManualReviewError(
                f"review artifact {path} must provide defect codes for "
                f"{verdict} verdict on {item_id!r}"
            )
        verdicts[verdict] += 1
        defect_codes.update(reason["code"] for reason in reasons)
    expected_verdicts = {name: verdicts.get(name, 0) for name in (
        "approve", "reject", "escalate"
    )}
    if review["aggregate"]["verdict_counts"] != expected_verdicts:
        raise ManualReviewError(
            f"review artifact {path} verdict aggregate does not replay"
        )
    if review["aggregate"]["defect_code_counts"] != dict(defect_codes):
        raise ManualReviewError(
            f"review artifact {path} defect aggregate does not replay"
        )
    return {item_id: str(row["verdict"]) for item_id, row in by_id.items()}


def evaluate_manual_review_gate(
    subject_path: Path,
    review_paths: Sequence[Path],
    *,
    expected_item_count: int | None = None,
    required_reviewers: int = 2,
) -> ManualReviewGate:
    """Validate reviews and resolve item verdicts, failing closed.

    Two agreeing reviews resolve an item.  Any disagreement requires a third
    artifact whose reviewer role is ``adjudicator``; the majority verdict then
    resolves that item.  Semantic booleans remain manual reviewer judgments.
    """

    if required_reviewers < 2:
        raise ManualReviewError("manual review gate requires at least two reviewers")
    payload = subject_path.read_bytes()
    subject_sha = _sha256(payload)
    subject_items = _subject_items(payload)
    if expected_item_count is not None and len(subject_items) != expected_item_count:
        raise ManualReviewError(
            f"subject item count mismatch: expected {expected_item_count}, "
            f"found {len(subject_items)}"
        )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    reviews = [_load_review(Path(path), schema) for path in review_paths]
    labels = [review["reviewer"]["label"] for review in reviews]
    if len(labels) != len(set(labels)):
        raise ManualReviewError("reviewer labels must be unique")
    defects: list[str] = []
    if len(reviews) < required_reviewers:
        defects.append("insufficient_reviewers")
    verdict_maps = [
        _validate_review(
            review,
            path=Path(path),
            subject_sha256=subject_sha,
            subject_items=subject_items,
        )
        for path, review in zip(review_paths, reviews)
    ]
    disagreements = tuple(
        item_id
        for item_id in subject_items
        if len({verdicts[item_id] for verdicts in verdict_maps}) > 1
    )
    if disagreements and not any(
        review["reviewer"]["role"] == "adjudicator" for review in reviews
    ):
        defects.append("third_review_required_for_disagreement")
    if disagreements and len(reviews) < 3:
        defects.append("insufficient_reviews_to_resolve_disagreement")

    final: Counter[str] = Counter()
    for item_id in subject_items:
        votes = Counter(verdicts[item_id] for verdicts in verdict_maps)
        if not votes:
            final["unresolved"] += 1
            continue
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            final["unresolved"] += 1
        else:
            final[top[0][0]] += 1
    if final.get("reject") or final.get("escalate") or final.get("unresolved"):
        defects.append("non_approved_final_verdicts")
    return ManualReviewGate(
        passed=not defects,
        subject_sha256=subject_sha,
        expected_item_count=len(subject_items),
        reviewer_count=len(reviews),
        final_verdict_counts=dict(final),
        disagreements=disagreements,
        defects=tuple(dict.fromkeys(defects)),
    )


__all__ = [
    "ManualReviewError",
    "ManualReviewGate",
    "evaluate_manual_review_gate",
]
