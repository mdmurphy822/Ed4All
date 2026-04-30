"""Audit 2026-04-30 — ChunkLabelResolver tests.

The resolver maps chunk-ID literals to human-readable labels for use
in eval probes. Without it, probes echo `shacl_551_chunk_NNNNN` into
the model's context, the model echoes them back, the classifier
scores ambiguous → faithfulness collapses (the cc07cc76 bug class).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.chunk_labels import ChunkLabelResolver


def _write_chunks(tmp_path: Path, records: list) -> Path:
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def test_resolver_prefers_summary_over_text(tmp_path: Path) -> None:
    p = _write_chunks(tmp_path, [
        {"id": "chunk_001", "summary": "SHACL property shapes",
         "text": "A long body of text about property shapes that we don't want as the label."},
    ])
    resolver = ChunkLabelResolver.from_jsonl(p)
    assert resolver.label_for("chunk_001") == "SHACL property shapes"


def test_resolver_falls_back_to_first_sentence_of_text(tmp_path: Path) -> None:
    p = _write_chunks(tmp_path, [
        {"id": "chunk_002", "text": "First sentence here. Second sentence ignored."},
    ])
    resolver = ChunkLabelResolver.from_jsonl(p)
    assert resolver.label_for("chunk_002") == "First sentence here."


def test_resolver_truncates_long_labels(tmp_path: Path) -> None:
    long_summary = "x" * 200
    p = _write_chunks(tmp_path, [
        {"id": "chunk_003", "summary": long_summary},
    ])
    resolver = ChunkLabelResolver.from_jsonl(p)
    label = resolver.label_for("chunk_003")
    assert len(label) <= 80  # _LABEL_MAX_CHARS
    assert label.endswith("…")


def test_resolver_returns_fallback_for_unknown_chunk(tmp_path: Path) -> None:
    p = _write_chunks(tmp_path, [
        {"id": "chunk_001", "summary": "Known chunk"},
    ])
    resolver = ChunkLabelResolver.from_jsonl(p)
    assert resolver.label_for("chunk_999") == "an unnamed chunk"
    # Critically, it does NOT return the chunk_id itself — that would
    # defeat the entire purpose of the resolver.
    assert "chunk_999" not in resolver.label_for("chunk_999")


@pytest.mark.parametrize("value,expected", [
    # Chunk-ID forms — should match
    ("chunk_00001", True),
    ("chunk_99999", True),
    ("rdf_shacl_551_chunk_00270", True),       # corpus-prefixed (production form)
    ("test_corpus_chunk_42", True),             # arbitrary-prefix form
    # Non-chunk forms — should NOT match
    ("CO-18", False),                           # course objective
    ("TO-01", False),                           # terminal outcome
    ("concept_a", False),                       # raw concept identifier
    ("concept_alpha_beta", False),              # underscored concept name
    ("bloom:remember", False),                  # bloom level target
    ("bloom:apply", False),
    ("mc_a1b2c3d4e5f6789a", False),             # 16-hex misconception ID
    ("shape_node_42", False),                   # generic graph node
    ("", False),                                # empty string
    ("not-an-id", False),                       # arbitrary text
])
def test_is_chunk_id_recognizes_chunk_forms_only(value, expected) -> None:
    """The resolver must recognise both canonical (`chunk_NNN`) and
    corpus-prefixed (`<corpus>_chunk_NNN`) chunk-ID forms but NOT match
    any other node class — concept IDs, course outcomes, bloom-level
    targets, misconception IDs (`mc_<16hex>`), or generic graph node
    names. Misclassification in either direction breaks scrub: false
    positives swallow real concept refs into "an unnamed chunk", false
    negatives leak chunk-IDs into probes."""
    resolver = ChunkLabelResolver(labels={})
    assert resolver.is_chunk_id(value) is expected, (
        f"is_chunk_id({value!r}) returned {not expected}, expected {expected}"
    )


@pytest.mark.parametrize("value", [
    "CO-18",
    "TO-01",
    "concept_alpha",
    "bloom:remember",
    "mc_a1b2c3d4e5f6789a",
    "shape_node_42",
])
def test_scrub_passes_through_non_chunk_strings(tmp_path: Path, value) -> None:
    """Non-chunk strings (concept IDs, course outcomes, bloom targets,
    misconception IDs, generic graph nodes) must flow through ``scrub``
    unchanged. Only chunk-ID literals get replaced with their label."""
    p = _write_chunks(tmp_path, [
        {"id": "rdf_shacl_551_chunk_00270", "summary": "Validating SHACL shapes"},
    ])
    resolver = ChunkLabelResolver.from_jsonl(p)
    assert resolver.scrub(value) == value


def test_scrub_replaces_chunk_id_with_label(tmp_path: Path) -> None:
    """The positive path: a chunk-ID literal IS swapped for its label."""
    p = _write_chunks(tmp_path, [
        {"id": "rdf_shacl_551_chunk_00270", "summary": "Validating SHACL shapes"},
        {"id": "chunk_00001", "summary": "First chunk"},
    ])
    resolver = ChunkLabelResolver.from_jsonl(p)
    assert resolver.scrub("rdf_shacl_551_chunk_00270") == "Validating SHACL shapes"
    assert resolver.scrub("chunk_00001") == "First chunk"


def test_resolver_from_course_handles_missing_corpus(tmp_path: Path) -> None:
    """Empty resolver is non-fatal — eval still runs, probes just
    carry the generic fallback."""
    course = tmp_path / "course-with-no-corpus"
    course.mkdir()
    resolver = ChunkLabelResolver.from_course(course)
    assert resolver.labels == {}
    assert resolver.label_for("chunk_001") == "an unnamed chunk"
