"""Tests for ``LibV2.tools.intent_router`` (Wave 78 Worker C).

The intent router classifies natural-language queries into one of six
canonical intent classes and dispatches each to the appropriate
retrieval backend. Tests cover:

* The canonical 6-query routing matrix from the design contract,
  asserting both the intent classification *and* the entity
  extraction (so we'd catch silent regressions in either layer).
* Live-archive dispatch against ``rdf-shacl-550-rdf-shacl-550``
  (skipped if the fixture isn't present in-repo).
* Edge cases: empty query, unknown slug, ambiguous query.

The live-archive assertions are loose lower-bounds (``>= 1``) rather
than exact counts because the BM25 / similarity scoring is
order-sensitive; precision is governed by the chunk-quality reviews,
not the router.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from LibV2.tools.intent_router import (
    INTENT_CLASSES,
    classify_intent,
    dispatch,
    extract_entities,
)
from lib.paths import LIBV2_PATH


LIVE_SLUG = "rdf-shacl-550-rdf-shacl-550"
LIVE_ARCHIVE = LIBV2_PATH / "courses" / LIVE_SLUG


# ---------------------------------------------------------------------- #
# Synthetic archive (for backend-agnostic structural tests)              #
# ---------------------------------------------------------------------- #


def _make_synthetic_archive(courses_root: Path, slug: str) -> Path:
    """Tiny archive with predictable filter outcomes (mirrors the
    chunk_query test fixture)."""
    root = courses_root / slug
    (root / "corpus").mkdir(parents=True)
    chunks = [
        {
            "id": f"c{i:02d}",
            "chunk_type": ct,
            "difficulty": diff,
            "bloom_level": bl,
            "text": text,
            "word_count": wc,
            "learning_outcome_refs": refs,
            "source": {"module_id": mod},
        }
        for i, (ct, diff, bl, text, wc, refs, mod) in enumerate(
            [
                ("explanation", "foundational", "remember",
                 "Intro chunk about RDF.", 100, ["co-01"], "week_01_overview"),
                ("example", "intermediate", "apply",
                 "Example with sh:minCount usage in SHACL.", 150, ["co-02"],
                 "week_03_content"),
                ("exercise", "intermediate", "apply",
                 "Exercise: write a SHACL shape.", 200, ["co-16"],
                 "week_07_application"),
                ("exercise", "advanced", "analyze",
                 "Analyze the constraint violation.", 250, ["co-17"],
                 "week_07_application"),
                ("assessment_item", "advanced", "evaluate",
                 "Question on SHACL features.", 50, ["to-04"],
                 "week_08_assessment"),
            ],
            start=1,
        )
    ]
    with (root / "corpus" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    objectives = {
        "terminal_outcomes": [{"id": "to-04"}],
        "component_objectives": [
            {"id": "co-16", "parent_terminal": "to-04"},
            {"id": "co-17", "parent_terminal": "to-04"},
        ],
    }
    (root / "objectives.json").write_text(json.dumps(objectives), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def live_archive_present() -> bool:
    return (LIVE_ARCHIVE / "corpus" / "chunks.jsonl").is_file()


# ---------------------------------------------------------------------- #
# 1. INTENT_CLASSES contract                                             #
# ---------------------------------------------------------------------- #


def test_intent_classes_are_the_canonical_six():
    assert set(INTENT_CLASSES) == {
        "objective_lookup",
        "prerequisite_query",
        "misconception_query",
        "assessment_query",
        "faceted_query",
        "concept_query",
    }


# ---------------------------------------------------------------------- #
# 2. Canonical 6-query routing matrix (classification only)              #
# ---------------------------------------------------------------------- #


def test_objective_lookup_extracts_objective_id():
    out = classify_intent("Which chunks assess to-04?")
    assert out["intent_class"] == "objective_lookup"
    assert out["extracted_entities"]["objective_ids"] == ["to-04"]
    assert out["confidence"] >= 0.9


def test_prerequisite_query_marker_match():
    out = classify_intent("What is a prerequisite for SHACL validation?")
    assert out["intent_class"] == "prerequisite_query"
    assert out["extracted_entities"]["has_prereq_marker"] is True


def test_misconception_query_plural_marker_match():
    out = classify_intent("What misconceptions exist about RDF triples?")
    assert out["intent_class"] == "misconception_query"
    assert out["extracted_entities"]["has_misconception_marker"] is True


def test_faceted_query_extracts_week_bloom_chunktype():
    out = classify_intent("Show me apply-level exercises for week 7")
    ent = out["extracted_entities"]
    assert out["intent_class"] == "faceted_query"
    assert ent["weeks"] == [7]
    # bloom verb extraction may pick up "apply"
    bloom_levels = {lvl for _v, lvl in ent["bloom_verbs"]}
    assert "apply" in bloom_levels
    assert "exercise" in ent["chunk_types"]


def test_concept_query_default_fallback():
    out = classify_intent("How does sh:minCount work?")
    assert out["intent_class"] == "concept_query"
    # Confidence is intentionally low for the open-ended fallback.
    assert out["confidence"] <= 0.6


def test_faceted_query_examples_chunk_type():
    out = classify_intent("Show me 5 worked examples of SHACL constraints")
    ent = out["extracted_entities"]
    assert out["intent_class"] == "faceted_query"
    assert "example" in ent["chunk_types"]


# ---------------------------------------------------------------------- #
# 3. Entity extraction unit tests                                         #
# ---------------------------------------------------------------------- #


def test_extract_entities_objective_id_case_insensitive():
    e = extract_entities("explain TO-04 and Co-18")
    assert "to-04" in e["objective_ids"]
    assert "co-18" in e["objective_ids"]


def test_extract_entities_multiple_weeks():
    e = extract_entities("compare week 3 to week 7")
    assert e["weeks"] == [3, 7]


def test_extract_entities_no_cues():
    e = extract_entities("RDF is a graph data model")
    assert e["objective_ids"] == []
    assert e["weeks"] == []
    assert e["has_prereq_marker"] is False
    assert e["has_misconception_marker"] is False


def test_extract_entities_residual_strips_cues():
    e = extract_entities("Show me apply-level exercises for week 7")
    # Cue words ("week 7", "exercises") removed; "apply-level" persists
    # as a leftover (it's not a chunk-type word per se).
    residual = e["residual_text"]
    assert "week" not in residual.lower() or "7" not in residual
    assert "exercises" not in residual.lower()


# ---------------------------------------------------------------------- #
# 4. Dispatch routing — synthetic archive (deterministic)                #
# ---------------------------------------------------------------------- #


def test_dispatch_objective_lookup_synthetic(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    out = dispatch(
        "Which chunks assess to-04?",
        "demo",
        top_k=10,
        courses_root=courses_root,
    )
    assert out["intent_class"] == "objective_lookup"
    # to-04 + co-16 + co-17 = 3 chunks in the synthetic fixture.
    assert len(out["results"]) == 3
    for chunk in out["results"]:
        assert (
            "to-04" in chunk.get("learning_outcome_refs", [])
            or "co-16" in chunk.get("learning_outcome_refs", [])
            or "co-17" in chunk.get("learning_outcome_refs", [])
        )


def test_dispatch_faceted_synthetic(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    out = dispatch(
        "Show me apply-level exercises for week 7",
        "demo",
        top_k=10,
        courses_root=courses_root,
    )
    assert out["intent_class"] == "faceted_query"
    # The synthetic fixture has 1 exercise+apply chunk in week 7
    # (c03; c04 is exercise+analyze, so it's filtered out).
    assert len(out["results"]) == 1
    assert out["results"][0]["chunk_type"] == "exercise"
    assert out["results"][0]["bloom_level"] == "apply"


def test_dispatch_assessment_synthetic_no_facets(tmp_path: Path):
    """Bare assessment marker (no week / bloom) -> assessment_query."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    out = dispatch(
        "give me a quiz",
        "demo",
        top_k=10,
        courses_root=courses_root,
    )
    # "quiz" is in CHUNK_TYPE_WORDS; precedence reorders to faceted.
    # That's intentional: a chunk-type cue is a structural facet.
    # Test the assessment_query path with a phrase that has the
    # marker but no chunk-type word.
    # Use a phrase that has the assessment marker ("test", "questions")
    # but no bloom verb / week / chunk-type-word; otherwise faceted
    # precedence kicks in.
    out2 = dispatch(
        "any test for me?",
        "demo",
        top_k=10,
        courses_root=courses_root,
    )
    assert out2["intent_class"] == "assessment_query"


def test_dispatch_unknown_slug_returns_empty(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    out = dispatch(
        "How does RDF work?",
        "no-such-slug",
        top_k=5,
        courses_root=courses_root,
    )
    # No exception; empty results.
    assert out["intent_class"] == "concept_query"
    assert out["results"] == []


def test_dispatch_empty_query_returns_concept_class():
    out = dispatch("", "any-slug", top_k=5)
    assert out["intent_class"] == "concept_query"


# ---------------------------------------------------------------------- #
# 5. Live-archive dispatch tests                                          #
# ---------------------------------------------------------------------- #


def test_live_objective_lookup_returns_results(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    out = dispatch("Which chunks assess to-04?", LIVE_SLUG, top_k=5)
    assert out["intent_class"] == "objective_lookup"
    assert out["entities"]["objective_ids"] == ["to-04"]
    assert len(out["results"]) >= 1
    # Each returned chunk should carry to-04 or one of its child COs.
    expected = {"to-04", "co-16", "co-17", "co-18", "co-19"}
    for chunk in out["results"]:
        refs = set(chunk.get("learning_outcome_refs") or [])
        assert refs & expected, f"chunk {chunk.get('id')} doesn't carry to-04 rollup"


def test_live_prerequisite_query_returns_concepts(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    out = dispatch(
        "What is a prerequisite for SHACL validation?",
        LIVE_SLUG,
        top_k=10,
    )
    assert out["intent_class"] == "prerequisite_query"
    # The archive has prerequisite_of edges; we should resolve at
    # least one (the router picks the best concept anchor).
    assert len(out["results"]) >= 1
    for r in out["results"]:
        assert r["relation"] == "prerequisite_of"
        assert r["concept"]
        assert r["target"]


def test_live_misconception_query_returns_corrections(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    out = dispatch(
        "What misconceptions exist about RDF triples?",
        LIVE_SLUG,
        top_k=5,
    )
    assert out["intent_class"] == "misconception_query"
    assert len(out["results"]) >= 1
    for r in out["results"]:
        assert "misconception" in r
        assert "correction" in r


def test_live_faceted_query_extracts_facets(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    out = dispatch(
        "Show me apply-level exercises for week 7",
        LIVE_SLUG,
        top_k=10,
    )
    # Routing: the entity extraction is what we validate (the
    # archive happens not to have exercise-typed chunks in week 7,
    # so the result count is data-dependent).
    assert out["intent_class"] == "faceted_query"
    assert out["entities"]["weeks"] == [7]
    bloom_levels = {lvl for _v, lvl in out["entities"]["bloom_verbs"]}
    assert "apply" in bloom_levels
    assert "exercise" in out["entities"]["chunk_types"]


def test_live_concept_query_returns_bm25_chunks(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    out = dispatch("How does sh:minCount work?", LIVE_SLUG, top_k=10)
    assert out["intent_class"] == "concept_query"
    assert len(out["results"]) >= 1
    # At least one of the top-k results should mention the term.
    has_match = any(
        "sh:mincount" in (r.get("text") or "").lower()
        for r in out["results"]
    )
    assert has_match, (
        "expected at least one BM25-ranked chunk to contain 'sh:minCount'"
    )


def test_live_faceted_examples_chunk_type(live_archive_present):
    if not live_archive_present:
        pytest.skip("rdf-shacl-550 archive not present")
    out = dispatch(
        "Show me 5 worked examples of SHACL constraints",
        LIVE_SLUG,
        top_k=5,
    )
    assert out["intent_class"] == "faceted_query"
    assert "example" in out["entities"]["chunk_types"]
    # Live archive has 25 example chunks, so ≥1 result expected.
    assert len(out["results"]) >= 1
    for chunk in out["results"]:
        assert chunk.get("chunk_type") == "example"


# ---------------------------------------------------------------------- #
# 6. Envelope shape contract                                              #
# ---------------------------------------------------------------------- #


def test_dispatch_envelope_carries_required_keys(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    out = dispatch("Which chunks assess to-04?", "demo",
                   top_k=3, courses_root=courses_root)
    for key in (
        "query", "slug", "intent_class", "confidence",
        "route", "source_path", "entities", "results",
    ):
        assert key in out, f"missing envelope key: {key}"
    # source_path mirrors route for ChatGPT-review parity.
    assert out["source_path"] == out["route"]
    # confidence in [0, 1].
    assert 0.0 <= out["confidence"] <= 1.0


def test_dispatch_no_internal_raw_query_leaks(tmp_path: Path):
    """The dispatcher passes _raw_query to backends but must strip it
    before returning the envelope. Otherwise the envelope shape leaks
    an internal helper field."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _make_synthetic_archive(courses_root, "demo")
    out = dispatch("anything", "demo", top_k=1, courses_root=courses_root)
    assert "_raw_query" not in out["entities"]
