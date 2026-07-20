"""Regression net for the SFT-program S2 scaffold carve-out + S10 audit
additions on the ``synthesis_leakage`` validator and ``audit_pairs`` script.

Covers:
* S2 — ``source="assessment_item"`` rows are EXEMPT from the assessment-
  scaffolding pattern check (marker scope, never global) in BOTH the runtime
  ``SynthesisLeakageValidator`` and ``audit_pairs._check_assessment_scaffolding``;
  a non-marker row carrying the same text still fails; the verbatim-span check
  still runs on marker rows.
* S10 — the new audit dimensions (pair_format_distribution,
  rubric_citation_resolution, sympy_key_presence, unique_opening_rate,
  promotion_ladder_invariants) are present and behave.

Offline / deterministic — no network, no course slugs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.synthesis_leakage import SynthesisLeakageValidator  # noqa: E402
from Trainforge.scripts.audit_pairs import run_audit  # noqa: E402

# The canonical scaffolding fragment the gate flags (Wave 122 pattern).
_SCAFFOLD = "Question 1 (CO-07, Bloom: Understand). Question 2 (CO-07, Bloom: Apply)."


def _write(course: Path, chunks: List[dict], pairs: List[dict]) -> None:
    cp = course / "corpus" / "chunks.jsonl"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")
    pp = course / "training_specs" / "instruction_pairs.jsonl"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text("\n".join(json.dumps(r) for r in pairs) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# S2 — synthesis_leakage carve-out
# --------------------------------------------------------------------------- #

def test_leakage_exempts_assessment_item_scaffold(tmp_path: Path) -> None:
    course = tmp_path / "course"
    chunk = {"id": "c1", "text": "Prose about the topic that is entirely distinct."}
    rows = [
        {
            "chunk_id": "assessment_item:q1",
            "source": "assessment_item",
            "prompt": "Solve this problem and show your reasoning step by step.",
            "completion": f"Grade the response. {_SCAFFOLD} Full marks awarded here.",
        }
        for _ in range(5)
    ]
    _write(course, [chunk], rows)
    result = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert result.passed is True
    assert not [i for i in result.issues if i.code == "ASSESSMENT_SCAFFOLDING_ABOVE_THRESHOLD"]


def test_leakage_still_flags_unmarked_scaffold(tmp_path: Path) -> None:
    course = tmp_path / "course"
    chunk = {"id": "c1", "text": "Prose about the topic that is entirely distinct."}
    rows = [
        {
            "chunk_id": "c1",
            "prompt": "Explain the concept for a learner in plain terms.",
            "completion": f"Here is the outline. {_SCAFFOLD}",
        }
        for _ in range(5)
    ]
    _write(course, [chunk], rows)
    result = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert result.passed is False
    assert any(i.code == "ASSESSMENT_SCAFFOLDING_ABOVE_THRESHOLD" for i in result.issues)


def test_leakage_verbatim_still_runs_on_marker_rows(tmp_path: Path) -> None:
    # A marker row that copies a >=50-char span from its cited chunk is STILL
    # caught by the verbatim gate — the carve-out is scaffold-only.
    course = tmp_path / "course"
    span = "This exact fifty-plus character span is copied verbatim ok"
    chunk = {"id": "c1", "text": span + " and continues afterwards in the source."}
    rows = [
        {
            "chunk_id": "c1",
            "source": "assessment_item",
            "prompt": "Solve and show your work for this item, please.",
            "completion": f"Step 1: {span}. Therefore the answer follows.",
        }
        for _ in range(5)
    ]
    _write(course, [chunk], rows)
    result = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert any(i.code == "VERBATIM_LEAKAGE_ABOVE_THRESHOLD" for i in result.issues)


# --------------------------------------------------------------------------- #
# S10 — audit_pairs additions
# --------------------------------------------------------------------------- #

def _dim(report: Any, name: str) -> Any:
    for d in report.dimensions:
        if d.name == name:
            return d
    raise AssertionError(f"dimension {name!r} not found")


def _audit_course(tmp_path: Path, pairs: List[dict], ladder: Dict[str, Any] | None) -> Any:
    course = tmp_path / "course"
    _write(course, [{"id": "c1", "text": "grounding prose for the chunkset."}], pairs)
    if ladder is not None:
        cfg = {"statistics": {"promotion_ladder": ladder}}
        cfg_path = course / "training_specs" / "dataset_config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return run_audit(course)


def _assessment_pair(fmt: str, i: int, **over: Any) -> dict:
    base = {
        "chunk_id": f"assessment_item:q{i}",
        "source": "assessment_item",
        "prompt": f"Solve the following problem number {i}, showing each step.",
        "completion": f"Step 1: work item {i}. Therefore, the answer is {i}.",
        "pair_format": fmt,
        "template_id": f"assessment_sft.{fmt}",
        "content_type": "assessment_sft",
    }
    base.update(over)
    return base


def test_audit_scaffold_exempts_assessment_marker(tmp_path: Path) -> None:
    pairs = [
        _assessment_pair("solve_steps", i, completion=f"See {_SCAFFOLD} step {i}.")
        for i in range(4)
    ]
    report = _audit_course(tmp_path, pairs, ladder=None)
    assert _dim(report, "assessment_scaffolding").passed is True


def test_audit_new_dimensions_present_and_pass(tmp_path: Path) -> None:
    pairs = [
        _assessment_pair("solve_steps", 1,
                         verifier_results={"sympy_key_present": True, "sympy_verified": True}),
        _assessment_pair("error_diagnosis", 2),
        _assessment_pair("explain_why", 3),
        _assessment_pair("grade_rubric", 4, rubric_cites=["c1"]),
        _assessment_pair("hint_no_reveal", 5),
        _assessment_pair("verify_answer", 6),
    ]
    report = _audit_course(tmp_path, pairs, ladder=None)
    # All five S10 dimensions must exist.
    for name in ("pair_format_distribution", "rubric_citation_resolution",
                 "sympy_key_presence", "unique_opening_rate",
                 "promotion_ladder_invariants"):
        _dim(report, name)
    # rubric citation resolves against c1 -> pass.
    assert _dim(report, "rubric_citation_resolution").passed is True
    # numeric solve_steps records a key -> pass.
    assert _dim(report, "sympy_key_presence").passed is True


def test_audit_rubric_citation_unresolved_fails(tmp_path: Path) -> None:
    pairs = [_assessment_pair("grade_rubric", 1, rubric_cites=["nonexistent_chunk"])]
    report = _audit_course(tmp_path, pairs, ladder=None)
    assert _dim(report, "rubric_citation_resolution").passed is False


def test_audit_promotion_ladder_invariants(tmp_path: Path) -> None:
    ok = {"candidate_pairs_total": 10, "validated_pairs_total": 7,
          "rejected_promotion_pairs": 3,
          "promotion_rejection_reasons": {"low_support": 2, "phantom_lo": 1}}
    report = _audit_course(tmp_path, [_assessment_pair("explain_why", 1)], ladder=ok)
    assert _dim(report, "promotion_ladder_invariants").passed is True

    bad = {"candidate_pairs_total": 10, "validated_pairs_total": 7,
           "rejected_promotion_pairs": 2,  # 7 + 2 != 10
           "promotion_rejection_reasons": {"low_support": 2}}
    report2 = _audit_course(tmp_path, [_assessment_pair("explain_why", 1)], ladder=bad)
    assert _dim(report2, "promotion_ladder_invariants").passed is False
