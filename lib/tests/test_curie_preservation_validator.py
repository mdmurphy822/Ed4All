"""Wave 130b — CuriePreservationValidator regression tests.

Pins the architectural backstop to the Wave 120 preserve-retry
mechanism: the validator runs on the chunk's *full* CURIE set
(regex-extracted) and fails closed when mean retention across
paraphrase pairs falls below 0.40.

The three primary cases mirror the Wave 130b spec:

* **Positive (clean):** every paraphrase pair preserves all source
  CURIEs → mean_retention = 1.0, gate passes.
* **Negative (poisoned):** paraphrase pairs strip 0–1 of 5 source
  CURIEs → mean_retention ≈ 0.1, gate fails closed.
* **Skip-deterministic:** mixed corpus of clean paraphrase pairs +
  deterministic-prefixed pairs that drop all CURIEs; only the
  paraphrase pairs count toward the retention score → gate passes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pytest

from lib.validators.curie_preservation import (
    CURIE_REGEX,  # noqa: F401  (kept as public symbol for back-compat)
    CuriePreservationValidator,
    _extract_curies,
)


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
# Sanity: dynamic open-prefix regex matches CURIEs and rejects URLs
# --------------------------------------------------------------- #


def test_curie_regex_matches_canonical_prefixes() -> None:
    """Wave 131: open-prefix detection. The regex matches any
    ``prefix:LocalName`` where the local-name leads with a letter; URL
    schemes are filtered via EXCLUDED_PREFIXES inside _extract_curies.

    Asserts:
    1. The original 8 canonical prefixes still match
       (sh / rdfs / owl / rdf / xsd / skos / dcterms / foaf).
    2. New prefixes the corpus actually uses also match
       (prov / dcat / geo / ex / schema).
    3. URL schemes are rejected (http / mailto / urn).
    4. Digit-leading local names are rejected (10:30, 8:00, localhost:8080).
    """
    text = (
        # Canonical 8
        "We use sh:NodeShape, rdfs:label, owl:sameAs, rdf:type, "
        "xsd:string, skos:Concept, dcterms:title, foaf:Person. "
        # Wave 131 new prefixes (silently dropped pre-Wave-131)
        "Audit found prov:Activity, dcat:Dataset, geo:lat, ex:Person, "
        "schema:name in chunks. "
        # URL schemes — must be filtered
        "But http://example.org, mailto:x@y, and urn:isbn:1234 should not match. "
        # Digit-leading local names — must be regex-rejected
        "Times like 10:30 and 8:00 AM and localhost:8080 also do not count."
    )
    matches = _extract_curies(text)
    # Canonical 8 still present
    assert {
        "sh:NodeShape", "rdfs:label", "owl:sameAs", "rdf:type",
        "xsd:string", "skos:Concept", "dcterms:title", "foaf:Person",
    }.issubset(matches)
    # Wave 131 new surface forms admitted
    assert {
        "prov:Activity", "dcat:Dataset", "geo:lat", "ex:Person",
        "schema:name",
    }.issubset(matches)
    # URL schemes filtered out by EXCLUDED_PREFIXES
    assert not any(c.startswith("http:") for c in matches)
    assert not any(c.startswith("mailto:") for c in matches)
    assert not any(c.startswith("urn:") for c in matches)
    # Digit-leading local names rejected by regex shape (no prefix match)
    assert not any(c.startswith("localhost:") for c in matches)
    # 10:30 / 8:00 — neither prefix nor local-name leads with a letter,
    # so the regex never produces a tuple for them.
    assert "10:30" not in matches
    assert "8:00" not in matches


def test_pair_dynamic_prefix_detection_regresses_unprotected(
    tmp_path: Path,
) -> None:
    """Wave 131 regression pin: a chunk + pair using only `prov:Activity`
    (a non-canonical-8 prefix) must trigger the validator's gate when
    the pair drops it. Pre-Wave-131 the validator silently passed
    because the allowlist regex never extracted `prov:Activity`."""
    course = tmp_path / "course"
    # Single chunk carrying ONLY `prov:Activity` — no canonical-8 CURIE.
    chunks = [{
        "id": "c1",
        "text": (
            "Provenance metadata anchors data lineage. The constraint "
            "prov:Activity is the central record-keeping shape, with "
            "prov:Activity used as the typed predicate target."
        ),
    }]
    # 5 paraphrase pairs that ALL drop `prov:Activity` → retention=0.
    pairs = [
        {
            "chunk_id": "c1",
            "template_id": "paraphrase.def_0",
            "prompt": f"Briefly explain provenance recording (variant {j}).",
            "completion": (
                "Provenance recording captures lineage as activities "
                "and entities, with predicates relating them in the "
                "graph."
            ),
        }
        for j in range(5)
    ]
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CuriePreservationValidator().validate(
        {"course_dir": str(course)}
    )

    # Pre-Wave-131 this would have returned NO_AUDITABLE_PAIRS (the
    # source CURIE was invisible) and silently passed. Wave 131 sees
    # `prov:Activity` and fails closed because the pairs never preserve it.
    assert result.passed is False
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert "CURIE_RETENTION_BELOW_THRESHOLD" in critical_codes


# --------------------------------------------------------------- #
# Positive: clean corpus → retention=1.0, passes
# --------------------------------------------------------------- #


def test_passes_when_all_pairs_preserve_all_curies(tmp_path: Path) -> None:
    """5 chunks × 3 CURIEs each, paraphrase pairs preserve every
    source CURIE → mean_retention=1.0, gate passes."""
    course = tmp_path / "course"
    chunks = []
    pairs = []
    for i in range(5):
        curies = [f"sh:Shape{i}A", f"rdfs:label{i}B", f"owl:Class{i}C"]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        # Two paraphrase pairs per chunk, each preserving all CURIEs.
        for j in range(2):
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": f"paraphrase.def_{j}",
                "prompt": (
                    f"Define for a learner: {curies[0]}, {curies[1]}, "
                    f"{curies[2]} (variant {j})."
                ),
                "completion": (
                    f"In SHACL, {curies[0]} is paired with {curies[1]} "
                    f"and refines {curies[2]} as the typing target."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CuriePreservationValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []


# --------------------------------------------------------------- #
# Negative: poisoned corpus → retention << 0.4, fails closed
# --------------------------------------------------------------- #


def test_fails_closed_when_mean_retention_below_threshold(
    tmp_path: Path,
) -> None:
    """5 chunks × 5 CURIEs, paraphrase pairs preserve 0-1 of 5
    (mean retention ≤ 0.2) → gate fails with critical issue."""
    course = tmp_path / "course"
    chunks = []
    pairs = []
    for i in range(5):
        curies = [
            f"sh:Shape{i}A", f"rdfs:label{i}B", f"owl:Class{i}C",
            f"rdf:type{i}D", f"xsd:string{i}E",
        ]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        # One pair per chunk: drops 4-5 of 5 CURIEs (4 chunks zero,
        # 1 chunk preserves a single CURIE → 1/5 retention).
        for j in range(2):
            preserved = curies[0] if (i == 0 and j == 0) else ""
            completion_curie_part = (
                f"It interacts with {preserved} in declarative form. "
                if preserved else ""
            )
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": f"Briefly explain the constraint (variant {j}).",
                "completion": (
                    "The constraint refines validation behaviour for "
                    "typed nodes. " + completion_curie_part
                    + "Learners should resolve the canonical entity."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CuriePreservationValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is False
    assert result.score is not None and result.score <= 0.2
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert "CURIE_RETENTION_BELOW_THRESHOLD" in critical_codes
    # Aggregate report sanity: zero_retention_count >= 4 (4 chunks
    # x 2 pairs each = 8 pairs that dropped every CURIE; only the
    # very first pair on chunk c0 preserved a single CURIE).
    report_issue = next(
        i for i in result.issues if i.code == "CURIE_RETENTION_REPORT"
    )
    payload = json.loads(report_issue.message)
    assert payload["zero_retention_count"] >= 4
    assert payload["pairs_audited"] == 10
    assert payload["mean_retention"] <= 0.2


# --------------------------------------------------------------- #
# Skip-deterministic: oracle-grounded pairs are excluded from audit
# --------------------------------------------------------------- #


def test_skips_deterministic_template_ids(tmp_path: Path) -> None:
    """5 paraphrase pairs preserve all CURIEs (clean) + 5
    deterministic-prefixed pairs that drop all CURIEs. Only the
    paraphrase pairs count toward retention → gate passes despite
    the deterministic pairs' zero retention."""
    course = tmp_path / "course"
    curies = ["sh:NodeShape", "rdfs:label", "owl:sameAs"]
    chunks = [{"id": "c1", "text": _chunk_text_with_curies(curies)}]
    pairs: List[dict] = []
    # 5 clean paraphrase pairs (all CURIEs preserved).
    for j in range(5):
        pairs.append({
            "chunk_id": "c1",
            "template_id": "paraphrase.def_0",
            "prompt": (
                f"Define {curies[0]}, {curies[1]}, {curies[2]} "
                f"(variant {j})."
            ),
            "completion": (
                f"In SHACL, {curies[0]} pairs with {curies[1]} and "
                f"is connected to {curies[2]}."
            ),
        })
    # 5 deterministic-prefixed pairs that drop ALL CURIEs (would tank
    # the mean retention if counted; must be skipped).
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
                "Answer the structured question with no CURIE tokens "
                "(deterministic generator)."
            ),
            "completion": (
                "The expected answer omits ontology-prefixed terms by "
                "construction in this generator family."
            ),
        })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CuriePreservationValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []
    # The aggregate report (when emitted) should reflect that the
    # deterministic pairs were skipped, not counted.
    report = next(
        (i for i in result.issues
         if i.code == "CURIE_RETENTION_REPORT"),
        None,
    )
    if report is not None:
        payload = json.loads(report.message)
        assert payload["skipped_deterministic"] == 5
        assert payload["pairs_audited"] == 5


# --------------------------------------------------------------- #
# Threshold override + missing-input fail-closed cases
# --------------------------------------------------------------- #


def test_threshold_override_via_inputs(tmp_path: Path) -> None:
    """Operator can lift the threshold to 0.10 (e.g. emergency
    diagnostic run); the same poisoned corpus that fails at 0.40
    passes at 0.10."""
    course = tmp_path / "course"
    curies = ["sh:Shape", "rdfs:label", "owl:Class"]
    chunks = [{"id": "c1", "text": _chunk_text_with_curies(curies)}]
    # Each pair preserves one of three CURIEs → retention 0.333.
    pairs = [
        {
            "chunk_id": "c1",
            "template_id": "paraphrase.def_0",
            "prompt": "Define the constraint shape.",
            "completion": (
                f"It declares the {curies[0]} shape applicable to "
                f"typed nodes during validation."
            ),
        }
        for _ in range(5)
    ]
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    strict = CuriePreservationValidator().validate(
        {"course_dir": str(course)}
    )
    assert strict.passed is False
    relaxed = CuriePreservationValidator().validate({
        "course_dir": str(course),
        "thresholds": {"min_mean_retention": 0.10},
    })
    assert relaxed.passed is True


def test_missing_inputs_fails_critical(tmp_path: Path) -> None:
    """No course_dir / training_specs_dir / instruction_pairs_path on
    inputs → gate fails closed with MISSING_INPUTS."""
    result = CuriePreservationValidator().validate({})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "MISSING_INPUTS" in codes


def test_missing_pair_file_fails_critical(tmp_path: Path) -> None:
    """No instruction_pairs.jsonl on disk → gate fails closed."""
    course = tmp_path / "course"
    _write_corpus(course, [{"id": "c1", "text": "no curies here"}])
    result = CuriePreservationValidator().validate(
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
    result = CuriePreservationValidator().validate(
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
    result = CuriePreservationValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is True
    info_codes = [i.code for i in result.issues if i.severity == "info"]
    assert "NO_AUDITABLE_PAIRS" in info_codes
