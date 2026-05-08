"""W-D11 T11.3 — synthesis_leakage scope must NOT include evidence_quote.

The synthesis_leakage validator (`lib/validators/synthesis_leakage.py`)
scans instruction-pair ``prompt`` and ``completion`` fields for ≥50-
char verbatim spans copied from ``chunk.text``. The W-D11 T11.3 emit
introduces a NEW field — ``key_claims[].evidence_quote`` (and the
mirrored pair-side ``per_claim_support[].evidence_quote``) — whose
ENTIRE PURPOSE is to carry a verbatim substring of the cited chunk
text. If the leakage validator's scope ever expands to include audit-
trail fields, ``evidence_quote`` would be flagged as "verbatim chunk
leakage" and the gate would fail closed on every healthy run.

Today's scope (verified at file read): only ``prompt`` and
``completion`` are scanned (lines 282 + 296 of
`lib/validators/synthesis_leakage.py`). This test locks that contract
in: an instruction pair whose ``key_claims[].evidence_quote``
legitimately equals a long contiguous substring of the chunk (the
T11.2 validator's WHOLE POINT is to verify this) MUST NOT trigger
``VERBATIM_LEAKAGE_ABOVE_THRESHOLD``.

If a future commit expands the leakage scope to scan additional
fields, this test fires and forces the author to explicitly exclude
``evidence_quote`` (audit-trail field, not training-completion text).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from lib.validators.synthesis_leakage import SynthesisLeakageValidator


def _write_corpus(course_dir: Path, chunks: List[dict]) -> Path:
    p = course_dir / "corpus" / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n",
        encoding="utf-8",
    )
    return p


def _write_pairs(course_dir: Path, rows: List[dict]) -> Path:
    p = course_dir / "training_specs" / "instruction_pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return p


def test_evidence_quote_substring_not_flagged_as_leakage(tmp_path: Path) -> None:
    """``evidence_quote`` legitimately equals a long contiguous span
    of ``chunk.text`` by T11.2's contract. The leakage gate's scope
    is ``prompt`` + ``completion`` only — ``evidence_quote`` lives
    inside ``key_claims[]`` / ``per_claim_support[]`` and must NOT
    be treated as completion-side leakage."""
    course = tmp_path / "course"
    chunk_text = (
        "SHACL shapes constrain RDF graphs by attaching property "
        "constraints to typed nodes. The validator walks each shape "
        "and reports violations against the targeted data graph."
    )
    _write_corpus(course, [{"id": "c1", "text": chunk_text}])
    # The evidence_quote here is a 100-char verbatim span — exactly
    # the kind of text the leakage gate would flag if it scanned the
    # field. The prompt + completion are paraphrased and DO NOT
    # leak.
    long_quote = (
        "SHACL shapes constrain RDF graphs by attaching property "
        "constraints to typed nodes."
    )
    assert len(long_quote) >= 50  # would trigger leakage if scoped in
    assert long_quote in chunk_text  # legitimate verbatim substring
    rows = []
    for i in range(20):
        rows.append({
            "chunk_id": "c1",
            "prompt": (
                f"Variant {i}: explain how shapes restrict graph "
                f"structure for a learner audience."
            ),
            "completion": (
                f"For attempt {i}, the constraint mechanism reports "
                f"discrepancies through a validation walk. Each "
                f"violation is surfaced through the report."
            ),
            "key_claims": [
                {
                    "claim": "SHACL shapes constrain RDF graphs.",
                    "source_chunk_ids": ["c1"],
                    "evidence_quote": long_quote,
                    "evidence_char_span": [0, len(long_quote)],
                },
            ],
        })
    _write_pairs(course, rows)
    result = SynthesisLeakageValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is True, (
        f"synthesis_leakage scope leaked into key_claims[].evidence_quote: "
        f"{[i.code for i in result.issues]}"
    )
    # Belt + braces: explicitly assert no critical
    # VERBATIM_LEAKAGE_ABOVE_THRESHOLD issue surfaced.
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "VERBATIM_LEAKAGE_ABOVE_THRESHOLD" not in codes


def test_evidence_quote_in_per_claim_support_not_flagged(tmp_path: Path) -> None:
    """Pair-side ``per_claim_support[].evidence_quote`` mirrors the
    chunk-side ``key_claims[].evidence_quote`` shape. Same exclusion
    contract — the leakage gate scope is prompt + completion only."""
    course = tmp_path / "course"
    chunk_text = (
        "Owl:sameAs links two IRI nodes that denote the same entity "
        "across different vocabularies. The reasoner propagates "
        "annotations across linked entities during inference."
    )
    _write_corpus(course, [{"id": "c1", "text": chunk_text}])
    long_quote = (
        "Owl:sameAs links two IRI nodes that denote the same entity "
        "across different vocabularies."
    )
    assert len(long_quote) >= 50
    assert long_quote in chunk_text
    rows = [
        {
            "chunk_id": "c1",
            "prompt": (
                "Briefly describe how the owl:sameAs relation behaves "
                "for an inference engine."
            ),
            "completion": (
                "An identity assertion treats two names as the same "
                "thing, and downstream rules merge their annotations."
            ),
            "per_claim_support": [
                {
                    "sentence": "An identity assertion treats two names as the same thing.",
                    "entailment": 0.92,
                    "contradiction": 0.01,
                    "outcome": "entailed",
                    "evidence_quote": long_quote,
                    "evidence_char_span": [0, len(long_quote)],
                },
            ],
        }
    ]
    _write_pairs(course, rows)
    result = SynthesisLeakageValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is True
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "VERBATIM_LEAKAGE_ABOVE_THRESHOLD" not in codes


def test_actual_completion_leakage_still_caught(tmp_path: Path) -> None:
    """Sanity check — the exclusion above is scope-narrow, not a
    blanket "ignore long substrings" change. If a pair's
    ``completion`` field DOES contain a ≥50-char verbatim span from
    chunk.text, the gate must still fail-closed. Without this test,
    the previous two tests could regress to "always pass" without
    detection."""
    course = tmp_path / "course"
    chunk_text = (
        "SHACL shapes constrain RDF graphs by attaching property "
        "constraints to typed nodes. The validator walks each shape "
        "and reports violations against the targeted data graph."
    )
    _write_corpus(course, [{"id": "c1", "text": chunk_text}])
    leaky_completion = (
        "SHACL shapes constrain RDF graphs by attaching property "
        "constraints to typed nodes."
    )
    rows = []
    for i in range(11):
        rows.append({
            "chunk_id": "c1",
            "prompt": f"Variant {i}: explain.",
            "completion": leaky_completion,
        })
    for i in range(9):
        rows.append({
            "chunk_id": "c1",
            "prompt": f"Variant {i}: explain.",
            "completion": (
                f"For attempt {i}, the constraint mechanism reports "
                f"discrepancies through a validation walk. Each "
                f"violation is surfaced through the report."
            ),
        })
    _write_pairs(course, rows)
    result = SynthesisLeakageValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "VERBATIM_LEAKAGE_ABOVE_THRESHOLD" in codes
