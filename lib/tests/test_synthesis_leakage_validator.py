"""Wave 121: SynthesisLeakageValidator regression tests.

The 2026-04-29 smoke audit on rdf-shacl-551-2 found 11/20 instruction
completions contained ≥50-char verbatim spans from chunk.text. Without
this gate, training would proceed on memorisation-poisoned pairs.
These tests pin the gate's fail-closed contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from lib.validators.synthesis_leakage import SynthesisLeakageValidator


def _write_corpus(course_dir: Path, chunks: List[dict]) -> Path:
    p = course_dir / "corpus" / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")
    return p


def _write_pairs(course_dir: Path, rows: List[dict]) -> Path:
    p = course_dir / "training_specs" / "instruction_pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_passes_when_no_pair_leaks(tmp_path: Path) -> None:
    """Clean corpus → leak rate 0 → gate passes."""
    course = tmp_path / "course"
    chunk_text = (
        "A SHACL shape declared with sh:NodeShape constrains the data "
        "graph. The shape can specify class targets and property "
        "constraints together when validating instances."
    )
    _write_corpus(course, [{"id": "c1", "text": chunk_text}])
    _write_pairs(course, [
        {
            "chunk_id": "c1",
            "prompt": f"Define this term for a learner: sh:NodeShape, variant {i}.",
            "completion": (
                f"sh:NodeShape is a SHACL construct {i} that describes "
                f"how validation proceeds against typed nodes in the data."
            ),
        }
        for i in range(20)
    ])
    result = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert result.passed is True
    assert not [i for i in result.issues if i.severity == "critical"]


def test_fails_critical_when_leak_rate_above_threshold(tmp_path: Path) -> None:
    """11/20 pairs leak ≥50-char spans (matches the audit signal). Gate fails."""
    course = tmp_path / "course"
    chunk_text = (
        "uha:Trial and brl:ClinicalStudy are linked by owl:equivalentClass "
        "and therefore convey the same intent across vocabularies. "
        "Validators should resolve them to the same canonical entity "
        "during inference."
    )
    _write_corpus(course, [{"id": "c1", "text": chunk_text}])
    leaky_completion = (
        "uha:Trial and brl:ClinicalStudy are linked by owl:equivalentClass "
        "and therefore convey the same intent across vocabularies."
    )
    rows = []
    for i in range(11):
        rows.append({
            "chunk_id": "c1",
            "prompt": f"Variant {i}: explain the linkage briefly.",
            "completion": leaky_completion,
        })
    for i in range(9):
        rows.append({
            "chunk_id": "c1",
            "prompt": f"Variant {i}: explain the linkage briefly.",
            "completion": (
                f"The two terms denote the same study type {i} and a "
                f"learner should resolve them to a canonical entity."
            ),
        })
    _write_pairs(course, rows)
    result = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "VERBATIM_LEAKAGE_ABOVE_THRESHOLD" in codes
    msg = " ".join(i.message for i in result.issues)
    assert "55.0%" in msg or "11/20" in msg


def test_threshold_override_via_inputs(tmp_path: Path) -> None:
    """Operator can lift the threshold to 60% (e.g. for a one-off
    experimental run); same input that fails at 5% passes at 60%."""
    course = tmp_path / "course"
    chunk_text = (
        "A SHACL NodeShape constrains the typed nodes in a target data "
        "graph. The shape can declare both class targets and property "
        "constraints in a single block."
    )
    _write_corpus(course, [{"id": "c1", "text": chunk_text}])
    leaky = (
        "A SHACL NodeShape constrains the typed nodes in a target data "
        "graph."
    )
    rows = [
        {"chunk_id": "c1", "prompt": f"v{i}", "completion": leaky}
        for i in range(2)
    ] + [
        {"chunk_id": "c1", "prompt": f"v{i}", "completion": "fresh wording about the constraint."}
        for i in range(8)
    ]
    _write_pairs(course, rows)
    strict = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert strict.passed is False
    relaxed = SynthesisLeakageValidator().validate({
        "course_dir": str(course),
        "thresholds": {"leak_rate_threshold": 0.6},
    })
    assert relaxed.passed is True


def test_missing_pair_file_fails_critical(tmp_path: Path) -> None:
    """No instruction_pairs.jsonl on disk → gate fails closed."""
    course = tmp_path / "course"
    _write_corpus(course, [{"id": "c1", "text": "any text"}])
    result = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "INSTRUCTION_PAIRS_NOT_FOUND" in codes


def test_missing_chunks_file_fails_critical(tmp_path: Path) -> None:
    """No corpus chunks.jsonl on disk → gate fails closed (can't compare)."""
    course = tmp_path / "course"
    _write_pairs(course, [{"chunk_id": "c1", "prompt": "p", "completion": "c"}])
    result = SynthesisLeakageValidator().validate({"course_dir": str(course)})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "CHUNKS_NOT_FOUND" in codes


def test_missing_inputs_fails_critical(tmp_path: Path) -> None:
    """course_dir input is required."""
    result = SynthesisLeakageValidator().validate({})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "MISSING_INPUTS" in codes
