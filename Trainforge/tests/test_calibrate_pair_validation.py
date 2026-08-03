"""Tests for ``Trainforge.scripts.harness.calibrate_pair_validation``.

The script is operator-facing tooling that proposes warning → critical
threshold flips post corpus rebuild. It is **read-only** — these tests
also assert the no-source-mutation contract.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.scripts.harness.calibrate_pair_validation import (  # noqa: E402
    VERDICT_KEEP,
    VERDICT_MISSING,
    VERDICT_NEEDS_DATA,
    VERDICT_READY,
    main as calibrate_main,
    run_calibration,
)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _build_minimal_course(course_root: Path, n_pairs: int = 50) -> None:
    """Build a minimal LibV2 course tree.

    Healthy distribution: most pairs have entailed claims; a few are
    contradicted / unsupported (well below the W4.A 20% / 5% ceilings)
    so all the day-1 floors fire on a low-fire-rate distribution.
    """
    pairs: List[Dict[str, Any]] = []
    for i in range(n_pairs):
        # ~10% unsupported, ~3% contradicted, ~3% dart_disagreement.
        if i % 10 == 0:
            outcome = "unsupported"
        elif i % 33 == 0:
            outcome = "contradicted"
        else:
            outcome = "entailed"
        dart_check = "dart_disagreement" if i % 33 == 0 else outcome
        pairs.append({
            "chunk_id": f"chunk_{i:03d}",
            "prompt": f"prompt {i}",
            "completion": f"completion {i}",
            "per_claim_support": [
                {
                    "outcome": outcome,
                    "dart_source_check": dart_check,
                    "max_entailment": 0.85 if outcome == "entailed" else 0.30,
                    "max_contradiction": 0.10 if outcome != "contradicted" else 0.65,
                    "claim_match_cosine": 0.72 if outcome == "entailed" else 0.45,
                    "dart_max_contradiction": 0.40 if dart_check != "dart_disagreement" else 0.60,
                },
            ],
        })

    instruction_path = course_root / "training_specs" / "instruction_pairs.jsonl"
    preference_path = course_root / "training_specs" / "preference_pairs.jsonl"
    _write_jsonl(instruction_path, pairs[: n_pairs // 2])
    _write_jsonl(preference_path, pairs[n_pairs // 2 :])

    # quality_report.json — minimal per_question_type_summary.
    _write_json(course_root / "quality" / "quality_report.json", {
        "assessments": {
            "per_question_type_summary": {
                "multiple_choice": {
                    "stem_diversity": 0.80,
                    "correct_answer_diversity": 0.70,
                    "distractor_template_max_ratio": 0.20,
                    "min_stem_chars": 14,
                    "question_count": 20,
                },
                "true_false": {
                    "stem_diversity": 0.55,
                    "correct_answer_diversity": 0.45,
                    "distractor_template_max_ratio": 1.0,
                    "min_stem_chars": 12,
                    "question_count": 5,
                },
            },
        },
    })

    # eval_report.json — minimal per-type buckets.
    _write_json(course_root / "eval" / "eval_report.json", {
        "answerable_rate": {
            "per_question_type": {
                "multiple_choice": {
                    "answerable_rate": 0.85,
                    "total_questions": 20,
                    "relevant": True,
                },
                "true_false": {
                    "answerable_rate": 0.70,
                    "total_questions": 5,
                    "relevant": True,
                },
            },
        },
        "single_correct_rate": {
            "per_question_type": {
                "multiple_choice": {
                    "single_correct_rate": 0.95,
                    "total_questions": 20,
                    "relevant": True,
                },
            },
        },
    })

    # dataset_config.json — minimal.
    _write_json(course_root / "training_specs" / "dataset_config.json", {
        "statistics": {
            "synthesis": {
                "last_run": {
                    "claim_support_rejected": 4,
                    "lo_refs_rejected": 1,
                    "objective_delivery_rejected": 2,
                    "pair_validation_passed": 50,
                    "dropped_count": 7,
                    "promotion_rejection_reasons": {
                        "claim_support": 4,
                        "lo_refs": 1,
                        "objective_delivery": 2,
                    },
                },
            },
        },
    })


def test_calibrate_emits_proposal_markdown(tmp_path: Path) -> None:
    """End-to-end smoke: run script, assert markdown sections present."""
    course_root = tmp_path / "demo-course-1"
    _build_minimal_course(course_root)

    out_path = tmp_path / "proposal.md"
    rc = calibrate_main([
        "--course-code", "demo-course-1",
        "--course-root", str(course_root),
        "--out", str(out_path),
    ])
    assert rc == 0, "script must exit 0 on a healthy course"
    assert out_path.exists(), "proposal markdown must be written"
    md = out_path.read_text()
    # Per-wave section headers.
    assert "## Wave 4 W4.A" in md
    assert "## Wave 5 W5.D" in md
    assert "## Wave 6 W6.A" in md
    assert "## Wave 7 W7.D" in md
    assert "## Wave 9 TIGHT" in md
    assert "## Wave 1.7 W1.7.C" in md
    # Summary block.
    assert "## Summary" in md
    assert "READY_TO_FLIP" in md
    assert "NEEDS_MORE_DATA" in md
    # File-path hints for operator edits.
    assert "lib/validators/claim_support.py" in md
    assert "lib/validators/eval_gating.py" in md
    assert "lib/validators/pair_claim_support.py" in md


def test_calibrate_fire_rate_calc_matches_input_distribution(
    tmp_path: Path,
) -> None:
    """Per-pair fire-rate calc matches a deterministic input distribution.

    Build 100 pairs where exactly 5 carry an unsupported claim → expect
    the W4.A unsupported-rate row to report fire_rate = 0.05.
    """
    course_root = tmp_path / "course-fixed"
    pairs: List[Dict[str, Any]] = []
    for i in range(100):
        outcome = "unsupported" if i < 5 else "entailed"
        pairs.append({
            "chunk_id": f"chunk_{i:03d}",
            "per_claim_support": [
                {
                    "outcome": outcome,
                    "max_entailment": 0.80 if outcome == "entailed" else 0.20,
                    "max_contradiction": 0.10,
                },
            ],
        })
    _write_jsonl(course_root / "training_specs" / "instruction_pairs.jsonl", pairs)
    _write_jsonl(course_root / "training_specs" / "preference_pairs.jsonl", [])
    _write_json(course_root / "quality" / "quality_report.json", {})
    _write_json(course_root / "eval" / "eval_report.json", {})
    _write_json(course_root / "training_specs" / "dataset_config.json", {})

    rows_by_wave, _missing = run_calibration(
        course_root=course_root,
        course_code="course-fixed",
        target_percentile=5.0,
    )
    w4a = rows_by_wave["Wave 4 W4.A — Claim-Support Thresholds"]
    # Find the unsupported row.
    unsup_row = next(r for r in w4a if r.metric == "max_unsupported_claim_rate")
    # Each pair has exactly 1 claim → per-pair unsupported_rate is 1.0
    # for the 5 unsupported pairs and 0.0 for the rest. The W4.A
    # ceiling = 0.20 → 5 of 100 pairs (5%) exceed the ceiling.
    assert unsup_row.sample_n == 100
    assert unsup_row.fire_rate == pytest.approx(0.05, abs=1e-6)


def test_calibrate_handles_missing_eval_report(tmp_path: Path) -> None:
    """Missing eval_report → MISSING_INPUT_eval_report.json + continue."""
    course_root = tmp_path / "course-no-eval"
    _build_minimal_course(course_root)
    # Wipe eval_report.json.
    eval_path = course_root / "eval" / "eval_report.json"
    eval_path.unlink()

    out_path = tmp_path / "proposal.md"
    rc = calibrate_main([
        "--course-code", "course-no-eval",
        "--course-root", str(course_root),
        "--out", str(out_path),
    ])
    assert rc == 0, "missing eval_report MUST NOT crash the script"
    md = out_path.read_text()
    assert "MISSING_INPUT_eval_report.json" in md
    # W6 + W4 sections still produced (they don't depend on eval_report).
    assert "## Wave 6 W6.A" in md
    assert "## Wave 4 W4.A" in md
    # Every Wave 7 row should be MISSING_INPUT.
    assert "## Wave 7 W7.D" in md
    assert "**MISSING_INPUT**" in md


def test_calibrate_ci_widens_with_low_sample_size(tmp_path: Path) -> None:
    """N=3 per bucket → recommendation defaults to NEEDS_MORE_DATA."""
    course_root = tmp_path / "course-low-n"
    pairs: List[Dict[str, Any]] = []
    for i in range(3):
        pairs.append({
            "chunk_id": f"chunk_{i:03d}",
            "per_claim_support": [
                {
                    "outcome": "entailed",
                    "max_entailment": 0.75 + (i * 0.01),
                    "max_contradiction": 0.10,
                    "claim_match_cosine": 0.55 + (i * 0.01),
                },
            ],
        })
    _write_jsonl(course_root / "training_specs" / "instruction_pairs.jsonl", pairs)
    _write_jsonl(course_root / "training_specs" / "preference_pairs.jsonl", [])
    _write_json(course_root / "quality" / "quality_report.json", {})
    _write_json(course_root / "eval" / "eval_report.json", {})
    _write_json(course_root / "training_specs" / "dataset_config.json", {})

    rows_by_wave, _ = run_calibration(
        course_root=course_root,
        course_code="course-low-n",
        target_percentile=5.0,
    )
    # Sample size of 3 is below the floor of 20 → every populated row
    # must verdict NEEDS_MORE_DATA, NOT READY_TO_FLIP.
    w4a = rows_by_wave["Wave 4 W4.A — Claim-Support Thresholds"]
    populated_rows = [r for r in w4a if r.sample_n > 0]
    assert populated_rows, "fixture must produce at least one populated row"
    for row in populated_rows:
        assert row.verdict != VERDICT_READY, (
            f"{row.metric} must NOT be READY_TO_FLIP at N={row.sample_n}; "
            f"got {row.verdict}"
        )


def test_calibrate_does_not_modify_source_files(tmp_path: Path) -> None:
    """Read-only contract: validator source files must not be touched."""
    # Snapshot mtimes of every validator module the script references.
    validator_dir = PROJECT_ROOT / "lib" / "validators"
    targets = [
        validator_dir / "claim_support.py",
        validator_dir / "pair_claim_support.py",
        validator_dir / "assessment.py",
        validator_dir / "eval_gating.py",
        validator_dir / "block_objective_delivery.py",
    ]
    before = {p: p.stat().st_mtime_ns for p in targets if p.exists()}
    assert before, "fixture sanity: at least one validator module must exist"

    course_root = tmp_path / "demo-course-1"
    _build_minimal_course(course_root)

    out_path = tmp_path / "proposal.md"
    rc = calibrate_main([
        "--course-code", "demo-course-1",
        "--course-root", str(course_root),
        "--out", str(out_path),
    ])
    assert rc == 0
    after = {p: p.stat().st_mtime_ns for p in targets if p.exists()}
    for p, ts in before.items():
        assert after[p] == ts, (
            f"calibrate_pair_validation MUST NOT modify {p}; "
            f"mtime drifted from {ts} to {after[p]}"
        )


def test_calibrate_dart_disagreement_rate_calc(tmp_path: Path) -> None:
    """5%% of pairs carrying dart_disagreement → fire_rate=0.05.

    The ``dart_source_check`` / ``dart_disagreement`` / ``dart_max_contradiction``
    tokens are the legacy decision-capture keys/values the calibration reader
    consumes (``c.get("dart_source_check") or c.get("outcome")``), and
    ``dart_disagreement_rate_ceiling`` / the ``Wave 9 TIGHT — Dual-Source DART``
    label are the source-emitted metric + wave names — legacy-compat reads, so
    they are pinned verbatim here.
    """
    course_root = tmp_path / "course-a"
    pairs: List[Dict[str, Any]] = []
    # 100 pairs, 5 carry dart_disagreement on a per-claim entry.
    for i in range(100):
        is_disagreement = i < 5
        pairs.append({
            "chunk_id": f"chunk_{i:03d}",
            "per_claim_support": [
                {
                    "outcome": "entailed",
                    "dart_source_check": (
                        "dart_disagreement" if is_disagreement else "entailed"
                    ),
                    "max_entailment": 0.80,
                    "max_contradiction": 0.10,
                    "dart_max_contradiction": 0.55 if is_disagreement else 0.20,
                },
            ],
        })
    _write_jsonl(course_root / "training_specs" / "instruction_pairs.jsonl", pairs)
    _write_jsonl(course_root / "training_specs" / "preference_pairs.jsonl", [])
    _write_json(course_root / "quality" / "quality_report.json", {})
    _write_json(course_root / "eval" / "eval_report.json", {})
    _write_json(course_root / "training_specs" / "dataset_config.json", {})

    rows_by_wave, _ = run_calibration(
        course_root=course_root,
        course_code="course-a",
        target_percentile=5.0,
    )
    w9 = rows_by_wave["Wave 9 TIGHT — Dual-Source DART Pair Validation"]
    rate_row = next(
        r for r in w9 if r.metric == "dart_disagreement_rate_ceiling"
    )
    # 5 of 100 audited pairs → fire_rate at the 5% gate ceiling = 0.05.
    assert rate_row.sample_n == 100
    assert rate_row.fire_rate == pytest.approx(0.05, abs=1e-6)
    # Right at the gate threshold → KEEP_WARNING (warning is the right
    # severity; the existing gate already catches this rate).
    assert rate_row.verdict == VERDICT_KEEP, (
        f"5% disagreement rate is exactly at the W9 ceiling; "
        f"verdict should be KEEP_WARNING, got {rate_row.verdict}"
    )
