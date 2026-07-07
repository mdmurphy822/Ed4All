"""Synthetic-wiring tests for scripts/integration/build_kg_stage3_local.py.

Proves the Stage-3 -> compile-seeds -> re-tag -> co-occurrence-grounding
wiring WITHOUT a live 7B call: a fake provider returns a fixed concept
list, and the test asserts (a) the loaded chunks get re-tagged with the
synthesized concept slugs and (b) the resulting graph carries chunk-anchored
DomainConcept nodes whose chunk-anchored coverage clears the 0.50 floor.

The GPU/live-LLM path is NOT exercised — the provider is mocked.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List

import pytest


# Import the runner module (scripts.integration is a package).
build = importlib.import_module("scripts.integration.build_kg_stage3_local")


# Five synthetic domain concepts. Each appears verbatim in MANY chunks so
# the 2-chunk co-occurrence node-mint floor + page grouping yield grounded
# DomainConcept nodes after re-tag.
_SYNTH_CONCEPTS = [
    "linear equation",
    "quadratic function",
    "slope intercept",
    "polynomial expression",
    "system of equations",
]


class _FakeSynthesisProvider:
    """Stand-in for TextbookSynthesisProvider — no LLM, fixed output."""

    _provider = "fake"
    _model = "fake-7b"

    @staticmethod
    def batch_chapters(chapters: List[Any], batch_size: int = 10) -> List[List[Any]]:
        size = max(1, int(batch_size))
        return [chapters[i:i + size] for i in range(0, len(chapters), size)]

    def synthesize_concepts(
        self, chapter: Dict[str, Any], *, course_name: str
    ) -> Dict[str, Any]:
        # Return ALL synthetic concepts for the (single) chapter so the
        # compiled seeds cover every concept used in the chunk text.
        return {
            "chapter_id": str(chapter.get("id") or ""),
            "concepts": [
                {
                    "canonical": name,
                    "aliases": [name],
                    "definition_hint": f"a hint about {name}",
                    "chapter_ids": [str(chapter.get("id") or "")],
                }
                for name in _SYNTH_CONCEPTS
            ],
        }


def _make_chunks() -> List[Dict[str, Any]]:
    """Build chunks whose text mentions the synthetic concepts.

    Two distinct pages so page-grouped co-occurrence connects concepts that
    don't share a single chunk. Every chunk starts with only 1 generic tag
    so coverage-before (on the degenerate existing-tags graph) is low.
    """
    chunks: List[Dict[str, Any]] = []
    idx = 0
    # Page 1: linear / slope / system concepts.
    page1_texts = [
        "A linear equation can be written in slope intercept form. "
        "The slope intercept form makes the slope explicit.",
        "Graphing a linear equation uses two points. A system of equations "
        "is two or more linear equations solved together.",
        "The slope intercept form of a linear equation is y = mx + b, where "
        "the system of equations may have one solution.",
    ]
    # Page 2: quadratic / polynomial concepts.
    page2_texts = [
        "A quadratic function is a polynomial expression of degree two. "
        "Every quadratic function graphs as a parabola.",
        "A polynomial expression can be factored. The quadratic function "
        "and the polynomial expression are closely related.",
        "Solving a quadratic function uses the quadratic formula; the "
        "polynomial expression underneath is the discriminant.",
    ]
    for page_id, texts in (("page_1", page1_texts), ("page_2", page2_texts)):
        for text in texts:
            idx += 1
            chunks.append({
                "id": f"alg_chunk_{idx:05d}",
                "text": text,
                # Start with a single generic tag (degenerate, like the
                # real-course case where chunks carry ~2 distinct tags total).
                "concept_tags": ["mathematics"],
                "learning_outcome_refs": [],
                "chunk_type": "content",
                "bloom_level": "understand",
                "difficulty": "intermediate",
                "source": {
                    "module_id": "demo-course",
                    "lesson_id": page_id,
                    "item_path": f"demo-course#{page_id}",
                    "course_id": "DEMO_COURSE_1",
                },
            })
    return chunks


def _coverage(graph: Dict[str, Any], out_dir) -> float:
    from Trainforge.rag.kg_quality_report import KGQualityReporter

    reporter = KGQualityReporter(
        course_slug="demo-course-1", run_id="test", output_dir=out_dir
    )
    report = reporter.compute_metrics_only(graph)
    dims = report.get("dimensions") or {}
    return float((dims.get("coverage") or {}).get("score", 0.0))


@pytest.fixture(autouse=True)
def _graph_shaping_env(monkeypatch):
    """Apply the runner's measured-best graph-shaping env for the build."""
    for env_var, value in build._GRAPH_SHAPING_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_var, value)
    yield


def test_stage3_retags_chunks_with_synthesized_concepts(tmp_path):
    """Re-tag UNIONs the synthesized concept slugs into chunk.concept_tags."""
    chunks = _make_chunks()
    ts_path = tmp_path / "textbook_structure.json"
    ts_path.write_text(
        '{"chapters": [{"id": "ch1", "chapter_text": '
        '"linear equation quadratic function slope intercept polynomial '
        'expression system of equations"}]}',
        encoding="utf-8",
    )

    tags_before = {t for c in chunks for t in (c.get("concept_tags") or [])}

    vocab = build.run_stage3_concept_synthesis(
        chunks=chunks,
        textbook_structure_path=ts_path,
        course_name="DEMO_COURSE_1",
        course_slug="demo-course-1",
        provider_factory=_FakeSynthesisProvider,
    )

    assert vocab is not None
    assert vocab["concept_count"] == len(_SYNTH_CONCEPTS)
    assert vocab["_chunks_retagged"] > 0

    tags_after = {t for c in chunks for t in (c.get("concept_tags") or [])}
    # New domain-concept slugs must have been UNIONed in.
    assert len(tags_after) > len(tags_before)
    # At least the multi-word concept slugs land on chunks.
    assert any("linear-equation" in (c.get("concept_tags") or []) for c in chunks)
    assert any("quadratic-function" in (c.get("concept_tags") or []) for c in chunks)
    # The pre-existing generic tag is preserved (UNION semantics).
    assert all("mathematics" in (c.get("concept_tags") or []) for c in chunks)


# Objectives whose key_concepts are the SAME slugs the chunk prose discusses.
# BEFORE re-tag those slugs are ungrounded LO-key-concept orphan nodes
# (frequency=0) that drag the chunk-anchored coverage floor — faithfully
# reproducing the real-course failure shape (coverage 0.133). AFTER
# re-tag the slugs land on ≥2 chunks → mint grounded co-occurrence nodes →
# coverage clears the floor.
_OBJECTIVES = {
    "course_code": "DEMO_COURSE_1",
    "learning_outcomes": [
        {"id": "CO-01", "statement": "Solve a linear equation",
         "bloom_level": "apply",
         "key_concepts": ["linear-equation", "slope-intercept"]},
        {"id": "CO-02", "statement": "Graph a quadratic function",
         "bloom_level": "apply",
         "key_concepts": ["quadratic-function", "polynomial-expression"]},
        {"id": "CO-03", "statement": "Solve a system of equations",
         "bloom_level": "apply",
         "key_concepts": ["system-of-equations"]},
    ],
}


def test_stage3_retag_lifts_coverage_above_floor(tmp_path):
    """Coverage BEFORE re-tag is below floor; AFTER re-tag clears 0.50.

    With objectives present, the LO key_concept slugs are ungrounded orphan
    nodes BEFORE re-tag (the real-course failure shape: coverage ~0.13). The
    Stage-3 re-tag grounds them, lifting coverage above the 0.50 floor.
    """
    # --- BEFORE: build over the degenerate existing-tags chunks -----------
    chunks_before = _make_chunks()
    graph_before = build.build_kg(
        chunks=chunks_before,
        course_slug="demo-course-1",
        objectives_payload=_OBJECTIVES,
        run_id="test-before",
    )
    cov_before = (
        _coverage(graph_before, tmp_path / "q_before")
        if graph_before.get("edges") else 0.0
    )
    # Faithful reproduction of the failure: coverage must be BELOW the floor
    # before re-tag (the LO key_concepts are ungrounded orphans).
    assert cov_before < 0.5, (
        f"expected degenerate before-coverage < 0.50, got {cov_before:.3f}"
    )

    # --- AFTER: re-tag via Stage-3, then build ----------------------------
    chunks_after = _make_chunks()
    ts_path = tmp_path / "textbook_structure.json"
    ts_path.write_text(
        '{"chapters": [{"id": "ch1", "chapter_text": "algebra concepts"}]}',
        encoding="utf-8",
    )
    vocab = build.run_stage3_concept_synthesis(
        chunks=chunks_after,
        textbook_structure_path=ts_path,
        course_name="DEMO_COURSE_1",
        course_slug="demo-course-1",
        provider_factory=_FakeSynthesisProvider,
    )
    assert vocab is not None and vocab["concept_count"] > 0

    graph_after = build.build_kg(
        chunks=chunks_after,
        course_slug="demo-course-1",
        objectives_payload=_OBJECTIVES,
        run_id="test-after",
    )
    assert graph_after.get("edges"), "post-re-tag graph must carry edges"
    cov_after = _coverage(graph_after, tmp_path / "q_after")

    # The synthesized concepts must have become chunk-grounded DomainConcept
    # nodes, lifting coverage above the 0.50 floor — proving the
    # Stage-3 -> re-tag -> coverage wiring end to end.
    assert cov_after > 0.5, (
        f"coverage after re-tag {cov_after:.3f} must clear the 0.50 floor "
        f"(before={cov_before:.3f})"
    )
    assert cov_after > cov_before

    # And the diagnostic counter sees the synthesized concepts grounded.
    grounded = build._count_grounded_synthesized_concepts(graph_after, vocab)
    assert grounded >= 3, (
        f"expected ≥3 synthesized concepts grounded as nodes, got {grounded}"
    )


def test_runner_fails_loud_without_synthesis_provider(tmp_path, monkeypatch, capsys):
    """main() exits 1 when TEXTBOOK_SYNTHESIS_PROVIDER is unset."""
    monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text('{"id": "c1", "text": "x", "concept_tags": []}\n', "utf-8")
    ts_path = tmp_path / "ts.json"
    ts_path.write_text('{"chapters": []}', "utf-8")

    rc = build.main([
        "--chunks", str(chunks_path),
        "--course-slug", "x-1",
        "--textbook-structure", str(ts_path),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "TEXTBOOK_SYNTHESIS_PROVIDER" in err
