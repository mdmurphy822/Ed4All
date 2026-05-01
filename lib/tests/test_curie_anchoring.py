"""Wave 135c — CurieAnchoringValidator regression tests.

Pins the binary per-pair CURIE anchoring sentinel that replaces
Wave 130b's mean-retention metric. Healthy Wave 135b force-injection
keeps the per-pair anchoring rate at ~1.00 by construction; a broken
injector drops the rate to whatever the natural paraphrase rate is.

Cases:

* **Positive (anchored):** every paraphrase pair contains ≥1 source
  CURIE → rate=1.0, gate passes.
* **Negative (injector regression):** 4/5 pairs anchored, 1 dropped
  → rate=0.8 < 0.95, gate fails closed.
* **Skip-deterministic:** 5 paraphrase pairs (anchored) + 5
  deterministic pairs (unanchored, but skipped) → rate=1.0, passes.
* **Threshold override:** override config to 0.5 → corpus at 0.6 passes
  (would fail at the default 0.95).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pytest

from lib.validators.curie_anchoring import CurieAnchoringValidator


# --------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------- #


def _write_corpus(course_dir: Path, chunks: Iterable[dict]) -> Path:
    p = course_dir / "corpus" / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n",
        encoding="utf-8",
    )
    return p


def _write_pairs(course_dir: Path, rows: Iterable[dict]) -> Path:
    p = course_dir / "training_specs" / "instruction_pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return p


def _chunk_text_with_curies(curies: List[str]) -> str:
    """Compose chunk text with the listed CURIEs woven into prose."""
    sentences = [
        f"The constraint {c} appears here as a typed predicate."
        for c in curies
    ]
    return (
        "Background prose without any CURIE tokens. "
        + " ".join(sentences)
        + " The vocabulary is fixed."
    )


# --------------------------------------------------------------- #
# Positive: anchored corpus → rate=1.0, passes
# --------------------------------------------------------------- #


def test_passes_when_every_pair_anchors_at_least_one_curie(
    tmp_path: Path,
) -> None:
    """5 chunks × 2 CURIEs each, every paraphrase pair contains ≥1
    source CURIE → pair_anchoring_rate=1.0, gate passes."""
    course = tmp_path / "course"
    chunks: List[dict] = []
    pairs: List[dict] = []
    for i in range(5):
        curies = [f"sh:Shape{i}A", f"rdfs:label{i}B"]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        # Two paraphrase pairs per chunk — each anchors at least one
        # source CURIE in its body.
        for j in range(2):
            anchor = curies[j % 2]
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": f"paraphrase.def_{j}",
                "prompt": (
                    f"Define for a learner: introduce the relevant "
                    f"constraint (variant {j})."
                ),
                "completion": (
                    f"In SHACL the {anchor} predicate types the node, "
                    f"and the validator interprets it as a typed shape."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []


# --------------------------------------------------------------- #
# Negative: injector regression → rate=0.8 < 0.95, fails closed
# --------------------------------------------------------------- #


def test_fails_closed_when_anchoring_rate_below_threshold(
    tmp_path: Path,
) -> None:
    """5 chunks × 2 CURIEs, 4 of 5 paraphrase pairs anchor, 1 has zero
    source CURIE in body → rate=0.8 < 0.95, gate fails."""
    course = tmp_path / "course"
    chunks: List[dict] = []
    pairs: List[dict] = []
    for i in range(5):
        curies = [f"sh:Shape{i}A", f"rdfs:label{i}B"]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        if i == 4:
            # Injector regression: pair body contains zero source
            # CURIEs.
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Briefly explain the constraint shape.",
                "completion": (
                    "It declares structural validation behaviour for "
                    "typed nodes without naming the predicate."
                ),
            })
        else:
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Define the constraint and its predicate.",
                "completion": (
                    f"The {curies[0]} shape types the node and the "
                    f"validator follows the declared constraint."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is False
    assert result.score == pytest.approx(0.8)
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert "PAIR_ANCHORING_BELOW_THRESHOLD" in critical_codes
    report = next(
        i for i in result.issues if i.code == "PAIR_ANCHORING_REPORT"
    )
    payload = json.loads(report.message)
    assert payload["total_eligible_pairs"] == 5
    assert payload["anchored_count"] == 4
    assert payload["unanchored_count"] == 1
    assert payload["pair_anchoring_rate"] == pytest.approx(0.8)


# --------------------------------------------------------------- #
# Skip-deterministic: oracle-grounded pairs are excluded from audit
# --------------------------------------------------------------- #


def test_skips_deterministic_template_ids(tmp_path: Path) -> None:
    """5 paraphrase pairs (all anchored) + 5 deterministic-prefixed
    pairs (unanchored). Only paraphrase pairs count → rate=1.0, gate
    passes despite the deterministic pairs' missing CURIEs."""
    course = tmp_path / "course"
    curies = ["sh:NodeShape", "rdfs:label"]
    chunks = [{"id": "c1", "text": _chunk_text_with_curies(curies)}]
    pairs: List[dict] = []
    # 5 clean paraphrase pairs (anchored).
    for j in range(5):
        pairs.append({
            "chunk_id": "c1",
            "template_id": "paraphrase.def_0",
            "prompt": (
                f"Define the constraint shape (variant {j})."
            ),
            "completion": (
                f"In SHACL, {curies[0]} types the node and pairs with "
                f"{curies[1]} as the human-readable label."
            ),
        })
    # 5 deterministic-prefixed pairs that DROP all CURIEs (would tank
    # the anchoring rate if counted; must be skipped).
    deterministic_prefixes = [
        "kg_metadata.entity",
        "violation_detection.shape",
        "abstention.unsupported",
        "schema_translation.translate",
        "kg_metadata.lookup",
    ]
    for prefix in deterministic_prefixes:
        pairs.append({
            "chunk_id": "c1",
            "template_id": prefix,
            "prompt": (
                "Answer the structured question without ontology "
                "tokens (deterministic generator)."
            ),
            "completion": (
                "The expected answer omits ontology-prefixed terms by "
                "construction in this generator family."
            ),
        })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []
    report = next(
        (i for i in result.issues
         if i.code == "PAIR_ANCHORING_REPORT"),
        None,
    )
    if report is not None:
        payload = json.loads(report.message)
        assert payload["skipped_deterministic"] == 5
        assert payload["total_eligible_pairs"] == 5


# --------------------------------------------------------------- #
# Threshold override: relaxed gate passes a corpus that would fail
# at the default 0.95.
# --------------------------------------------------------------- #


def test_threshold_override_via_inputs(tmp_path: Path) -> None:
    """Override min_pair_anchoring_rate to 0.5; a corpus at 0.6
    passes the relaxed gate but fails the default 0.95."""
    course = tmp_path / "course"
    chunks: List[dict] = []
    pairs: List[dict] = []
    for i in range(5):
        curies = [f"sh:Shape{i}A", f"rdfs:label{i}B"]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        # 3 of 5 chunks contribute anchored pairs (60%); the other 2
        # produce unanchored pairs.
        if i < 3:
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Describe the constraint and its label.",
                "completion": (
                    f"The {curies[0]} predicate types the node and "
                    f"binds to the {curies[1]} annotation."
                ),
            })
        else:
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Briefly summarize the validation behaviour.",
                "completion": (
                    "It refines validation of typed nodes through a "
                    "declarative predicate without naming it."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    strict = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert strict.passed is False
    assert strict.score == pytest.approx(0.6)

    relaxed = CurieAnchoringValidator().validate({
        "course_dir": str(course),
        "thresholds": {"min_pair_anchoring_rate": 0.5},
    })
    assert relaxed.passed is True
    assert relaxed.score == pytest.approx(0.6)


# --------------------------------------------------------------- #
# Missing-input fail-closed cases
# --------------------------------------------------------------- #


def test_missing_inputs_fails_critical() -> None:
    """No course_dir / training_specs_dir / instruction_pairs_path on
    inputs → gate fails closed with MISSING_INPUTS."""
    result = CurieAnchoringValidator().validate({})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "MISSING_INPUTS" in codes


def test_missing_pair_file_fails_critical(tmp_path: Path) -> None:
    """No instruction_pairs.jsonl on disk → gate fails closed."""
    course = tmp_path / "course"
    _write_corpus(course, [{"id": "c1", "text": "no curies here"}])
    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "INSTRUCTION_PAIRS_NOT_FOUND" in codes


def test_missing_chunks_file_fails_critical(tmp_path: Path) -> None:
    """No corpus chunks.jsonl on disk → gate fails closed."""
    course = tmp_path / "course"
    _write_pairs(course, [{
        "chunk_id": "c1",
        "template_id": "paraphrase.def_0",
        "prompt": "p", "completion": "c",
    }])
    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "CHUNKS_NOT_FOUND" in codes


def test_passes_when_no_auditable_pairs(tmp_path: Path) -> None:
    """Corpus with chunks that have zero CURIEs → no pairs audited
    → gate passes with NO_AUDITABLE_PAIRS info issue."""
    course = tmp_path / "course"
    _write_corpus(course, [
        {"id": "c1", "text": "Plain text without ontology-prefixed terms."},
    ])
    _write_pairs(course, [
        {
            "chunk_id": "c1",
            "template_id": "paraphrase.def_0",
            "prompt": "Summarize the chunk.",
            "completion": "It contains plain prose.",
        },
    ])
    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is True
    info_codes = [i.code for i in result.issues if i.severity == "info"]
    assert "NO_AUDITABLE_PAIRS" in info_codes
