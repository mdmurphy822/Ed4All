#!/usr/bin/env python3
"""Tests for ``Trainforge.instruction_pair_extractor`` (Wave 79).

Covered contracts:
  - assessment_item chunks with ``Q: ... A: ...`` markers yield Q -> A
    pairs with reasoning preserved.
  - exercise chunks with explicit ``Solution:`` sections yield
    task -> solution pairs.
  - example chunks with ``First, ... Then, ... Finally, ...`` worked
    structure yield reasoning chains with at least three steps.
  - explanation chunks emit lower-quality template pairs tagged
    ``derived_from=explanation_template`` with quality_score 0.6.
  - misconception entries yield BOTH distinguish + contrast pairs.
  - Empty chunk text is handled without crashing and emits zero pairs.
  - ``--min-quality 0.7`` filter drops the 0.6-quality
    explanation_template pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.instruction_pair_extractor import (  # noqa: E402
    ALL_METHODS,
    METHOD_ASSESSMENT_ITEM,
    METHOD_EXAMPLE_REASONING,
    METHOD_EXERCISE,
    METHOD_EXPLANATION_TEMPLATE,
    METHOD_MISCONCEPTION_CONTRAST,
    METHOD_MISCONCEPTION_DISTINGUISH,
    extract_from_assessment_item,
    extract_from_example,
    extract_from_exercise,
    extract_from_explanation,
    extract_from_misconceptions,
    run_extraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_chunk(**overrides):
    """Build a minimal chunk dict with sensible defaults."""
    base = {
        "id": "chunk_001",
        "chunk_type": "explanation",
        "text": "Some text here.",
        "learning_outcome_refs": ["co-01", "to-01"],
        "bloom_level": "understand",
        "difficulty": "foundational",
        "concept_tags": ["sample_topic"],
    }
    base.update(overrides)
    return base


def _write_chunks(path: Path, chunks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")


# ---------------------------------------------------------------------------
# Extractor 1: assessment_item
# ---------------------------------------------------------------------------


def test_assessment_item_q_a_marker_yields_qa_pair_with_reasoning():
    chunk = _stub_chunk(
        id="ai_001",
        chunk_type="assessment_item",
        text=(
            "Q: What is X?\n\n"
            "A: X is Y because Z."
        ),
    )
    pairs = extract_from_assessment_item(chunk)
    assert len(pairs) == 1, f"expected one pair, got {pairs}"
    p = pairs[0]
    assert p["instruction"] == "What is X?"
    assert "X is Y because Z" in p["output"]
    md = p["metadata"]
    assert md["extraction_method"] == METHOD_ASSESSMENT_ITEM
    assert md["quality_score"] == 1.0
    assert md["source_chunk_id"] == "ai_001"
    assert md["objective_ids"] == ["co-01", "to-01"]


def test_assessment_item_show_answer_marker_yields_letter_plus_reasoning():
    chunk = _stub_chunk(
        id="ai_002",
        chunk_type="assessment_item",
        text=(
            "Intro paragraph that explains the formative quiz.\n\n"
            "What is the correct order? A. order one B. order two C. order three D. order four "
            "Show answer B. The order is load-bearing — it expresses the direction.\n\n"
            "Which choice is FALSE? A. choice one B. choice two C. choice three D. choice four "
            "Show answer C. The primer emphasizes the second clause."
        ),
    )
    pairs = extract_from_assessment_item(chunk)
    assert len(pairs) == 2, f"expected two pairs, got {len(pairs)}"
    p1, p2 = pairs
    assert "What is the correct order" in p1["instruction"]
    assert p1["output"].startswith("B.")
    assert "Reasoning:" in p1["output"]
    assert p2["output"].startswith("C.")


# ---------------------------------------------------------------------------
# Extractor 2: exercise
# ---------------------------------------------------------------------------


def test_exercise_with_solution_marker_yields_task_solution_pair():
    chunk = _stub_chunk(
        id="ex_001",
        chunk_type="exercise",
        text=(
            "Write a function that reverses a string.\n\n"
            "Solution: def reverse(s): return s[::-1]"
        ),
    )
    pairs = extract_from_exercise(chunk)
    assert len(pairs) == 1
    p = pairs[0]
    assert "reverses a string" in p["instruction"]
    assert "return s[::-1]" in p["output"]
    md = p["metadata"]
    assert md["extraction_method"] == METHOD_EXERCISE
    assert md["quality_score"] == 1.0
    assert md["split_marker"] == "solution"


# ---------------------------------------------------------------------------
# Extractor 3: example -> reasoning chain
# ---------------------------------------------------------------------------


def test_example_with_first_then_finally_yields_reasoning_chain():
    chunk = _stub_chunk(
        id="ex_002",
        chunk_type="example",
        text=(
            "Consider how to solve a linear equation. "
            "First, isolate the variable on one side of the equals sign. "
            "Then, divide both sides by the coefficient of the variable. "
            "Finally, simplify the resulting fraction to obtain the answer."
        ),
        bloom_level="apply",
        summary="Solving a linear equation is a three-step procedure.",
    )
    pairs = extract_from_example(chunk)
    assert len(pairs) == 1
    p = pairs[0]
    chain = p.get("reasoning_chain")
    assert chain and len(chain) == 3
    assert any("isolate" in s.lower() for s in chain)
    assert any("divide" in s.lower() for s in chain)
    assert any("simplify" in s.lower() for s in chain)
    assert "Step 1:" in p["output"]
    assert p["metadata"]["extraction_method"] == METHOD_EXAMPLE_REASONING
    assert p["metadata"]["quality_score"] == 1.0
    assert p["metadata"]["step_count"] == 3


# ---------------------------------------------------------------------------
# Extractor 4: explanation template
# ---------------------------------------------------------------------------


def test_explanation_chunk_emits_low_quality_template_pair():
    chunk = _stub_chunk(
        id="exp_001",
        chunk_type="explanation",
        text=(
            "Triples are the fundamental building block of RDF. "
            "They consist of a subject, predicate, and object."
        ),
        concept_tags=["rdf_triple"],
    )
    pairs = extract_from_explanation(chunk)
    assert len(pairs) >= 1
    p = pairs[0]
    md = p["metadata"]
    assert md["extraction_method"] == METHOD_EXPLANATION_TEMPLATE
    assert md["quality_score"] == 0.6
    assert md["derived_from"] == "explanation_template"
    # Question is templated from concept tag.
    assert "rdf triple" in p["instruction"].lower()


# ---------------------------------------------------------------------------
# Extractor 5: misconceptions -> distinguish + contrast
# ---------------------------------------------------------------------------


def test_misconception_yields_distinguish_and_contrast_pairs():
    chunk = _stub_chunk(
        id="mis_001",
        chunk_type="explanation",
        concept_tags=["rdf_triple"],
        misconceptions=[
            {
                "misconception": "An RDF triple is like a row in a relational table.",
                "correction": (
                    "Triples are not rows; every triple is a first-class fact and "
                    "the schema is open."
                ),
            }
        ],
    )
    pairs = extract_from_misconceptions(
        chunk,
        methods=(METHOD_MISCONCEPTION_DISTINGUISH, METHOD_MISCONCEPTION_CONTRAST),
    )
    methods = [p["metadata"]["extraction_method"] for p in pairs]
    assert METHOD_MISCONCEPTION_DISTINGUISH in methods
    assert METHOD_MISCONCEPTION_CONTRAST in methods
    distinguish = next(p for p in pairs if p["metadata"]["extraction_method"] == METHOD_MISCONCEPTION_DISTINGUISH)
    contrast = next(p for p in pairs if p["metadata"]["extraction_method"] == METHOD_MISCONCEPTION_CONTRAST)
    assert "claim correct" in distinguish["instruction"].lower()
    assert distinguish["output"].startswith("No.")
    assert "Compare" in contrast["instruction"]
    assert contrast["output"].startswith("Common misunderstanding:")
    for p in pairs:
        assert p["metadata"]["quality_score"] == 0.9


# ---------------------------------------------------------------------------
# Edge case: empty chunk text
# ---------------------------------------------------------------------------


def test_empty_chunk_text_emits_no_pairs_and_does_not_crash():
    empty = _stub_chunk(text="")
    # Every extractor should be safe on empty input.
    assert extract_from_assessment_item({**empty, "chunk_type": "assessment_item"}) == []
    assert extract_from_exercise({**empty, "chunk_type": "exercise"}) == []
    assert extract_from_example({**empty, "chunk_type": "example"}) == []
    assert extract_from_explanation(empty) == []
    assert extract_from_misconceptions(empty, methods=ALL_METHODS) == []


# ---------------------------------------------------------------------------
# CLI / driver: --min-quality filter
# ---------------------------------------------------------------------------


def test_min_quality_07_filters_explanation_template_pairs(tmp_path):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks = [
        _stub_chunk(
            id="exp_a",
            chunk_type="explanation",
            text="Triples are the unit of RDF assertion. They have three slots.",
            concept_tags=["rdf"],
        ),
        _stub_chunk(
            id="ai_b",
            chunk_type="assessment_item",
            text=(
                "Q: What is RDF?\n\n"
                "A: A graph data model where each statement is a subject-predicate-object triple."
            ),
        ),
        _stub_chunk(
            id="mis_c",
            chunk_type="explanation",
            concept_tags=["rdf"],
            text="RDF is a graph data model.",
            misconceptions=[
                {
                    "misconception": "RDF is a relational database.",
                    "correction": "RDF is a graph data model — there are no fixed columns.",
                }
            ],
        ),
    ]
    _write_chunks(chunks_file, chunks)
    out_dir = tmp_path / "out"
    stats, pairs = run_extraction(
        chunks_path=chunks_file,
        output_dir=out_dir,
        methods=ALL_METHODS,
        min_quality=0.7,
        capture=None,
    )
    methods = {p["metadata"]["extraction_method"] for p in pairs}
    # 0.6-quality explanation_template pairs are filtered out.
    assert METHOD_EXPLANATION_TEMPLATE not in methods
    # Higher-quality methods survive.
    assert METHOD_ASSESSMENT_ITEM in methods
    assert METHOD_MISCONCEPTION_DISTINGUISH in methods
    assert METHOD_MISCONCEPTION_CONTRAST in methods
    # Filter counter is non-zero.
    assert stats.pairs_filtered_quality >= 1
    # Outputs were written.
    assert (out_dir / "instruction_pairs.jsonl").exists()
    assert (out_dir / "extraction_report.json").exists()


def test_run_extraction_writes_reasoning_chains_subset(tmp_path):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks = [
        _stub_chunk(
            id="ex_chain",
            chunk_type="example",
            text=(
                "Walk through how to debug a SHACL validation failure. "
                "First, identify which shape failed. "
                "Then, inspect the focus node referenced by the failure. "
                "Finally, adjust either the data or the shape so the validation passes."
            ),
            bloom_level="analyze",
            summary="Three-step SHACL debugging procedure.",
        ),
        _stub_chunk(
            id="exp_no_chain",
            chunk_type="explanation",
            text="A SHACL shape is a set of constraints.",
        ),
    ]
    _write_chunks(chunks_file, chunks)
    out_dir = tmp_path / "out"
    stats, pairs = run_extraction(
        chunks_path=chunks_file,
        output_dir=out_dir,
        capture=None,
    )
    chains_path = out_dir / "reasoning_chains.jsonl"
    assert chains_path.exists()
    chains = [json.loads(l) for l in chains_path.read_text().splitlines() if l.strip()]
    assert len(chains) == 1
    assert chains[0]["metadata"]["extraction_method"] == METHOD_EXAMPLE_REASONING
    assert stats.reasoning_chains_emitted == 1
