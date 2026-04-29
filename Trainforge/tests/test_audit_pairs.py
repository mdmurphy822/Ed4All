"""Wave 122: regression tests for ``Trainforge.scripts.audit_pairs``."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.scripts.audit_pairs import (  # noqa: E402
    run_audit,
    format_report_text,
    format_report_json,
    main as audit_main,
)


def _write(p: Path, rows: List[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _clean_pair_template(chunk_id: str, i: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "prompt": f"Define this concept clearly enough to apply it to topic {i}.",
        "completion": (
            f"This concept variant {i} constrains the data graph and "
            f"applies during validation. A learner uses it to evaluate "
            f"whether a target node satisfies declared properties."
        ),
        "lo_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type": "concept",
        "seed": i,
        "decision_capture_id": "evt_test",
        "template_id": f"understand._tpl{i % 5}",
        "provider": "mock",
        "schema_version": "v1",
    }


def _clean_pref_template(chunk_id: str, i: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "prompt": f"Explain this concept clearly enough to avoid the misconception {i}.",
        "chosen": (
            f"This concept variant {i} is best framed as a constraint over "
            f"data graph instances; learners should apply it during shape "
            f"validation."
        ),
        "rejected": (
            f"This concept variant {i} is mostly theoretical; a learner can "
            f"safely ignore the formal definition for everyday RDF work."
        ),
        "lo_refs": ["TO-01"],
        "seed": i,
        "decision_capture_id": "evt_test",
        "rejected_source": "rule_synthesized",
        "source": "rule_synthesized",
        "provider": "mock",
        "schema_version": "v1",
    }


def _build_clean_course(tmp_path: Path) -> Path:
    course = tmp_path / "course"
    _write(course / "corpus" / "chunks.jsonl", [
        {"id": "c1", "text": "Some clean source text about SHACL constraints."},
    ])
    _write(course / "training_specs" / "instruction_pairs.jsonl", [
        _clean_pair_template("c1", i) for i in range(20)
    ])
    _write(course / "training_specs" / "preference_pairs.jsonl", [
        _clean_pref_template("c1", i) for i in range(15)
    ])
    return course


def test_audit_passes_on_clean_corpus(tmp_path: Path) -> None:
    course = _build_clean_course(tmp_path)
    report = run_audit(course)
    assert report.overall_passed is True
    crit_pass = [d for d in report.dimensions if d.severity == "critical" and d.passed]
    assert len(crit_pass) == 8


def test_audit_fails_on_assessment_scaffolding(tmp_path: Path) -> None:
    course = _build_clean_course(tmp_path)
    rows = [_clean_pair_template("c1", i) for i in range(19)]
    rows.append({
        **_clean_pair_template("c1", 99),
        "completion": (
            "Question 1 (CO-07, Bloom: Understand). Question 2 (CO-07, "
            "Bloom: Apply). Question 3 (CO-07, Bloom: Analyze)."
        ),
    })
    _write(course / "training_specs" / "instruction_pairs.jsonl", rows)
    report = run_audit(course)
    assert report.overall_passed is False
    failing = [d.name for d in report.critical_failures]
    assert "assessment_scaffolding" in failing


def test_audit_fails_on_verbatim_leakage(tmp_path: Path) -> None:
    course = _build_clean_course(tmp_path)
    long_chunk_text = (
        "uha:Trial and brl:ClinicalStudy are linked by owl:equivalentClass "
        "and convey the same intent across vocabularies for resolver use."
    )
    _write(course / "corpus" / "chunks.jsonl", [
        {"id": "c1", "text": long_chunk_text},
    ])
    rows = [_clean_pair_template("c1", i) for i in range(15)]
    for i in range(5):
        rows.append({
            **_clean_pair_template("c1", 100 + i),
            "completion": long_chunk_text + f" {i}",
        })
    _write(course / "training_specs" / "instruction_pairs.jsonl", rows)
    report = run_audit(course)
    assert report.overall_passed is False
    failing = [d.name for d in report.critical_failures]
    assert "verbatim_leakage" in failing


def test_audit_fails_on_legacy_scaffold(tmp_path: Path) -> None:
    course = _build_clean_course(tmp_path)
    rows = [_clean_pair_template("c1", i) for i in range(19)]
    rows.append({
        **_clean_pair_template("c1", 99),
        "completion": (
            "topic should be explained through the concrete RDF/SHACL role "
            "of x, y, z, not just by listing related labels. A learner "
            "applies it after grasping the underlying schema relationships."
        ),
    })
    _write(course / "training_specs" / "instruction_pairs.jsonl", rows)
    report = run_audit(course)
    failing = [d.name for d in report.critical_failures]
    assert "legacy_scaffold_template" in failing


def test_audit_smoke_mode_reads_smoke_files(tmp_path: Path) -> None:
    course = tmp_path / "course"
    _write(course / "corpus" / "chunks.jsonl", [{"id": "c1", "text": "x"}])
    _write(course / "training_specs" / "smoke_instruction_pairs.jsonl", [
        _clean_pair_template("c1", i) for i in range(3)
    ])
    _write(course / "training_specs" / "smoke_preference_pairs.jsonl", [
        _clean_pref_template("c1", i) for i in range(3)
    ])
    report = run_audit(course, smoke=True)
    assert report.instruction_count == 3
    assert report.preference_count == 3


def test_audit_json_format(tmp_path: Path) -> None:
    course = _build_clean_course(tmp_path)
    report = run_audit(course)
    payload = json.loads(format_report_json(report))
    assert payload["overall_passed"] is True
    assert payload["instruction_count"] == 20
    assert payload["preference_count"] == 15
    assert any(d["name"] == "assessment_scaffolding" for d in payload["dimensions"])


def test_audit_text_format_marks_pass(tmp_path: Path) -> None:
    course = _build_clean_course(tmp_path)
    report = run_audit(course)
    text = format_report_text(report)
    assert "OVERALL: PASS" in text
    assert "Critical dimensions:" in text


def test_audit_main_exit_codes(tmp_path: Path) -> None:
    course = _build_clean_course(tmp_path)
    rc = audit_main(["--course", str(course)])
    assert rc == 0

    rows = [_clean_pair_template("c1", i) for i in range(19)]
    rows.append({
        **_clean_pair_template("c1", 99),
        "completion": "Question 1 (CO-07, Bloom: Understand). Question 2 (CO-07, Bloom: Apply).",
    })
    _write(course / "training_specs" / "instruction_pairs.jsonl", rows)
    rc = audit_main(["--course", str(course)])
    assert rc == 1

    rc = audit_main(["--course", str(tmp_path / "doesnt_exist")])
    assert rc == 2


def test_audit_missing_chunks_returns_2(tmp_path: Path) -> None:
    course = tmp_path / "course"
    _write(course / "training_specs" / "instruction_pairs.jsonl", [
        _clean_pair_template("c1", 0),
    ])
    rc = audit_main(["--course", str(course)])
    assert rc == 2
